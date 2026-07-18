#!/usr/bin/env python3
"""Acquire and verify exact authoritative PDFs for empty CourtListener rows.

This is a source-bound acquisition step, not a corpus-generation fallback.
Both named HTTPS origins must return the exact reviewed PDF bytes. The
published generation is immutable and is the only input accepted by the
Cuyahoga extractor when an exact bulk row has no authoritative text field.
"""

from __future__ import annotations

import argparse
import bz2
import csv
from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from cuyahoga_contract import (
    ContractError,
    ResolvedText,
    TEXT_FIELDS,
    normalize_lines,
    official_rod_evidence,
    sha256_text,
)
from law_generation import (
    GenerationPublisher,
    sha256_file,
    verify_generation,
)
from structured_error import StructuredArgumentParser, parse_cli_args, write_error


SOURCE_FORMAT = "calyx-cuyahoga-authoritative-document-sources-v1"
GENERATION_FORMAT = "calyx-cuyahoga-authoritative-documents-generation-v1"
MANIFEST_FILE = "authoritative_documents_manifest.json"
DOCUMENTS_FILE = "authoritative_documents.jsonl"
EXTRACTOR_NAME = "pypdf"
EXTRACTOR_VERSION = "6.14.2"
USER_AGENT = "Calyx-Cuyahoga-authoritative-document-acquirer/1"
APPROVED_SOURCE_CONTRACT_SHA256 = (
    "7bcba266d3d9b142c3803bb116f1cce07f7bf61fae9813423441af13bdd68c34"
)

# A different manifest cannot silently widen this reviewed exception set.
EXPECTED_IDENTITIES = {
    11227531: {
        "cluster_id": 10760946,
        "source_row_sha256": "9210d9dfce555d8ad5c9c525ef52d97a6f6419b2004df799ba70195ea193813f",
        "pdf_sha256": "8bae0003b4864f015adb4873bf7dc0266bbd55e7e2fb878ebe9800eb63d6ba36",
    },
    11257339: {
        "cluster_id": 10790698,
        "source_row_sha256": "755a65a915df139d50c20854e520f120ad738ca781edab5a5f040898a554ab09",
        "pdf_sha256": "85bf041443f7666d458883c72704ddfb7e2743b9652662ccba6ad0064f148311",
    },
}

OPINION_FIELDS = (
    "id",
    "type",
    "author_str",
    "per_curiam",
    "author_id",
    "cluster_id",
    "sha1",
    "download_url",
    "local_path",
    "date_created",
    "date_modified",
) + TEXT_FIELDS

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AuthoritativeDocumentError(ContractError):
    code = "authoritative_document_error"


def _error(message: str, **context):
    raise AuthoritativeDocumentError(message, **context)


def _require_pypdf():
    try:
        import pypdf
    except ImportError as error:
        _error(
            "authoritative PDF extraction dependency is unavailable",
            dependency=EXTRACTOR_NAME,
            expected_version=EXTRACTOR_VERSION,
            remediation="install pypdf==%s before acquiring or verifying authoritative PDFs"
            % EXTRACTOR_VERSION,
            error=str(error),
        )
    if pypdf.__version__ != EXTRACTOR_VERSION:
        _error(
            "installed pypdf version does not match the source contract",
            expected=EXTRACTOR_VERSION,
            actual=pypdf.__version__,
            remediation="install pypdf==%s" % EXTRACTOR_VERSION,
        )
    return pypdf


def stable_row_sha256(row: dict) -> str:
    encoded = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_json(path: str | os.PathLike[str]) -> tuple[dict, str]:
    resolved = Path(path).absolute()
    if not resolved.is_file() or resolved.is_symlink():
        _error("JSON contract is not a plain file", path=str(resolved))
    try:
        with open(resolved, "r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        _error("cannot read JSON contract", path=str(resolved), error=str(error))
    if not isinstance(value, dict):
        _error("JSON contract root must be an object", path=str(resolved))
    return value, sha256_file(resolved)


def _positive_int(value, *, field: str, opinion_id: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _error(
            "document contract field must be a positive integer",
            field=field,
            opinion_id=opinion_id,
            value=value,
        )
    return value


def _exact_https_url(value, *, role: str, opinion_id: int) -> str:
    if not isinstance(value, str):
        _error("document URL must be a string", role=role, opinion_id=opinion_id)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        _error(
            "document URL cannot be parsed",
            role=role,
            opinion_id=opinion_id,
            error=str(error),
        )
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        _error(
            "document URL violates the exact HTTPS origin contract",
            role=role,
            opinion_id=opinion_id,
            url=value,
        )
    if role == "storage":
        if parsed.hostname != "storage.courtlistener.com":
            _error(
                "storage URL host is not CourtListener storage",
                opinion_id=opinion_id,
                url=value,
            )
    elif role == "official":
        evidence = official_rod_evidence(value)
        if len(evidence) != 1 or evidence[0].district != 8:
            _error(
                "official URL is not an Eighth District Ohio ROD PDF",
                opinion_id=opinion_id,
                url=value,
            )
    else:
        _error("unknown document URL role", role=role, opinion_id=opinion_id)
    return value


def load_source_contract(
    path: str | os.PathLike[str],
    *,
    opinions_archive_sha256: str | None = None,
    snapshot_date: str | None = None,
) -> dict:
    contract, contract_sha256 = _plain_json(path)
    if contract_sha256 != APPROVED_SOURCE_CONTRACT_SHA256:
        _error(
            "authoritative document source contract is not the reviewed byte generation",
            path=str(path),
            expected=APPROVED_SOURCE_CONTRACT_SHA256,
            actual=contract_sha256,
        )
    if set(contract) != {"format", "source_binding", "extractor", "documents"}:
        _error("authoritative document source contract schema mismatch", path=str(path))
    if contract["format"] != SOURCE_FORMAT:
        _error("unsupported authoritative document source contract", path=str(path))
    binding = contract["source_binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "opinions_archive_sha256",
        "snapshot_date",
    }:
        _error("authoritative document source binding schema mismatch", path=str(path))
    try:
        date.fromisoformat(binding["snapshot_date"])
    except (TypeError, ValueError) as error:
        _error("source snapshot date is invalid", path=str(path), error=str(error))
    if not _SHA256_RE.fullmatch(binding.get("opinions_archive_sha256", "")):
        _error("source opinions archive SHA-256 is invalid", path=str(path))
    if (
        opinions_archive_sha256 is not None
        and binding["opinions_archive_sha256"] != opinions_archive_sha256
    ):
        _error(
            "authoritative document contract has the wrong opinions archive",
            expected=opinions_archive_sha256,
            actual=binding["opinions_archive_sha256"],
        )
    if snapshot_date is not None and binding["snapshot_date"] != snapshot_date:
        _error(
            "authoritative document contract has the wrong snapshot date",
            expected=snapshot_date,
            actual=binding["snapshot_date"],
        )
    extractor = contract["extractor"]
    expected_extractor = {
        "name": EXTRACTOR_NAME,
        "version": EXTRACTOR_VERSION,
        "strict": True,
        "text_method": "PageObject.extract_text-default",
        "join_pages": "two-newlines",
    }
    if extractor != expected_extractor:
        _error(
            "authoritative PDF extractor contract mismatch",
            expected=expected_extractor,
            actual=extractor,
        )
    documents = contract["documents"]
    if not isinstance(documents, list):
        _error("authoritative document list must be an array", path=str(path))
    by_id = {}
    expected_keys = {
        "opinion_id",
        "cluster_id",
        "source_row_sha256",
        "sha1",
        "local_path",
        "download_url",
        "storage_url",
        "max_pdf_bytes",
        "pdf_sha1",
        "pdf_sha256",
        "expected_pages",
        "expected_raw_chars",
        "expected_raw_text_sha256",
        "expected_normalized_chars",
        "expected_normalized_text_sha256",
    }
    for document in documents:
        if not isinstance(document, dict) or set(document) != expected_keys:
            _error("authoritative document entry schema mismatch", document=document)
        opinion_id = _positive_int(document["opinion_id"], field="opinion_id")
        if opinion_id in by_id:
            _error("duplicate authoritative opinion ID", opinion_id=opinion_id)
        cluster_id = _positive_int(
            document["cluster_id"], field="cluster_id", opinion_id=opinion_id
        )
        identity = EXPECTED_IDENTITIES.get(opinion_id)
        if identity is None or {key: document[key] for key in identity} != identity:
            _error(
                "document is outside the reviewed authoritative identity set",
                opinion_id=opinion_id,
                cluster_id=cluster_id,
            )
        for field in (
            "source_row_sha256",
            "pdf_sha256",
            "expected_raw_text_sha256",
            "expected_normalized_text_sha256",
        ):
            if not _SHA256_RE.fullmatch(document[field]):
                _error(
                    "invalid SHA-256 in document contract",
                    opinion_id=opinion_id,
                    field=field,
                )
        for field in ("sha1", "pdf_sha1"):
            if not _SHA1_RE.fullmatch(document[field]):
                _error(
                    "invalid SHA-1 in document contract",
                    opinion_id=opinion_id,
                    field=field,
                )
        for field in (
            "max_pdf_bytes",
            "expected_pages",
            "expected_raw_chars",
            "expected_normalized_chars",
        ):
            _positive_int(document[field], field=field, opinion_id=opinion_id)
        local_path = document["local_path"]
        if (
            not isinstance(local_path, str)
            or not local_path.startswith("pdf/")
            or ".." in Path(local_path).parts
            or Path(local_path).is_absolute()
        ):
            _error(
                "invalid CourtListener local_path",
                opinion_id=opinion_id,
                value=local_path,
            )
        storage_url = _exact_https_url(
            document["storage_url"], role="storage", opinion_id=opinion_id
        )
        if urlsplit(storage_url).path != "/" + local_path:
            _error(
                "storage URL path differs from CourtListener local_path",
                opinion_id=opinion_id,
                storage_url=storage_url,
                local_path=local_path,
            )
        _exact_https_url(
            document["download_url"], role="official", opinion_id=opinion_id
        )
        by_id[opinion_id] = document
    if set(by_id) != set(EXPECTED_IDENTITIES):
        _error(
            "authoritative source contract must contain the exact reviewed identity set",
            expected=sorted(EXPECTED_IDENTITIES),
            actual=sorted(by_id),
        )
    return {
        "value": contract,
        "sha256": contract_sha256,
        "by_opinion": by_id,
    }


def _open_bulk_text(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def source_rows(path: str, wanted: set[int]) -> dict[int, dict]:
    _set_csv_field_limit()
    found = {}
    with _open_bulk_text(path) as source:
        reader = csv.reader(
            source, quotechar='"', escapechar="\\", doublequote=False, strict=True
        )
        try:
            header = next(reader)
        except StopIteration:
            _error("opinions bulk CSV is empty", path=path)
        if len(header) != len(set(header)):
            _error("opinions bulk CSV has duplicate columns", path=path)
        missing = [field for field in OPINION_FIELDS if field not in header]
        if missing:
            _error("opinions bulk CSV is missing columns", path=path, missing=missing)
        indexes = {field: header.index(field) for field in OPINION_FIELDS}
        try:
            for row in reader:
                if len(row) != len(header):
                    _error(
                        "opinions bulk CSV column-count mismatch",
                        path=path,
                        physical_line=reader.line_num,
                        expected=len(header),
                        actual=len(row),
                    )
                raw_id = row[indexes["id"]]
                try:
                    opinion_id = int(raw_id)
                except ValueError:
                    _error(
                        "opinion ID is not decimal",
                        path=path,
                        physical_line=reader.line_num,
                        value=raw_id,
                    )
                if opinion_id not in wanted:
                    continue
                if opinion_id in found:
                    _error(
                        "duplicate required opinion row",
                        path=path,
                        opinion_id=opinion_id,
                    )
                found[opinion_id] = {
                    field: row[position] for field, position in indexes.items()
                }
                if set(found) == wanted:
                    break
        except csv.Error as error:
            _error(
                "opinions bulk CSV parse error",
                path=path,
                physical_line=reader.line_num,
                error=str(error),
            )
    if set(found) != wanted:
        _error(
            "required authoritative source rows are missing",
            path=path,
            missing=sorted(wanted - set(found)),
        )
    return found


def verify_source_rows(path: str, contract: dict) -> dict[int, dict]:
    resolved = Path(path).absolute()
    if not resolved.is_file() or resolved.is_symlink():
        _error("opinions archive is not a plain file", path=str(resolved))
    expected_archive = contract["value"]["source_binding"]["opinions_archive_sha256"]
    actual_archive = sha256_file(resolved)
    if actual_archive != expected_archive:
        _error(
            "opinions archive SHA-256 mismatch",
            path=str(resolved),
            expected=expected_archive,
            actual=actual_archive,
        )
    rows = source_rows(str(resolved), set(contract["by_opinion"]))
    for opinion_id, document in contract["by_opinion"].items():
        row = rows[opinion_id]
        facts = {
            "cluster_id": int(row["cluster_id"]),
            "source_row_sha256": stable_row_sha256(row),
            "sha1": row["sha1"],
            "local_path": row["local_path"],
            "download_url": row["download_url"],
        }
        expected = {key: document[key] for key in facts}
        if facts != expected:
            _error(
                "authoritative document binding differs from the physical bulk row",
                opinion_id=opinion_id,
                expected=expected,
                actual=facts,
            )
        nonempty = [field for field in TEXT_FIELDS if row[field].strip()]
        if nonempty:
            _error(
                "authoritative PDF exception is prohibited for a nonempty bulk text row",
                opinion_id=opinion_id,
                nonempty_fields=nonempty,
            )
    return rows


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del file_pointer, headers
        _error(
            "authoritative document server attempted a redirect",
            source_url=request.full_url,
            status=code,
            reason=message,
            redirect_url=new_url,
        )


def fetch_pdf(url: str, *, max_bytes: int, timeout_seconds: int) -> tuple[bytes, dict]:
    opener = build_opener(_RejectRedirects())
    request = Request(
        url,
        headers={"Accept": "application/pdf", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None)
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            content_length = response.headers.get("Content-Length")
            if status != 200:
                _error(
                    "authoritative document HTTP status is not 200",
                    url=url,
                    status=status,
                )
            if final_url != url:
                _error(
                    "authoritative document final URL changed",
                    expected=url,
                    actual=final_url,
                )
            if content_type != "application/pdf":
                _error(
                    "authoritative document content type is not application/pdf",
                    url=url,
                    content_type=content_type,
                )
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    _error(
                        "authoritative document Content-Length is invalid",
                        url=url,
                        value=content_length,
                    )
                if declared_length <= 0 or declared_length > max_bytes:
                    _error(
                        "authoritative document Content-Length violates the byte limit",
                        url=url,
                        content_length=declared_length,
                        max_bytes=max_bytes,
                    )
            chunks = []
            total = 0
            while True:
                chunk = response.read(min(1 << 16, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    _error(
                        "authoritative document body exceeds the byte limit",
                        url=url,
                        bytes=total,
                        max_bytes=max_bytes,
                    )
            value = b"".join(chunks)
            if content_length is not None and len(value) != declared_length:
                _error(
                    "authoritative document body differs from Content-Length",
                    url=url,
                    declared=declared_length,
                    actual=len(value),
                )
    except AuthoritativeDocumentError:
        raise
    except HTTPError as error:
        _error(
            "authoritative document HTTP request failed",
            url=url,
            status=error.code,
            reason=str(error),
        )
    except (URLError, OSError) as error:
        _error(
            "authoritative document network request failed",
            url=url,
            error_type=type(error).__name__,
            error=str(error),
        )
    return value, {
        "url": url,
        "content_type": "application/pdf",
        "bytes": len(value),
        "sha1": hashlib.sha1(value).hexdigest(),
        "sha256": hashlib.sha256(value).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def extract_pdf(value: bytes, expected: dict) -> dict:
    opinion_id = expected["opinion_id"]
    pypdf = _require_pypdf()
    if len(value) <= 8 or not value.startswith(b"%PDF-"):
        _error("authoritative document lacks a PDF signature", opinion_id=opinion_id)
    if b"%%EOF" not in value[-4096:]:
        _error(
            "authoritative document lacks a terminal PDF EOF marker",
            opinion_id=opinion_id,
        )
    pdf_sha1 = hashlib.sha1(value).hexdigest()
    pdf_sha256 = hashlib.sha256(value).hexdigest()
    if pdf_sha1 != expected["pdf_sha1"] or pdf_sha256 != expected["pdf_sha256"]:
        _error(
            "authoritative document binary digest mismatch",
            opinion_id=opinion_id,
            expected_sha1=expected["pdf_sha1"],
            actual_sha1=pdf_sha1,
            expected_sha256=expected["pdf_sha256"],
            actual_sha256=pdf_sha256,
        )
    try:
        reader = pypdf.PdfReader(BytesIO(value), strict=True)
        if reader.is_encrypted:
            _error("authoritative document is encrypted", opinion_id=opinion_id)
        pages = len(reader.pages)
        page_text = []
        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not isinstance(text, str):
                _error(
                    "authoritative PDF page did not produce text",
                    opinion_id=opinion_id,
                    page=page_number,
                )
            page_text.append(text)
    except AuthoritativeDocumentError:
        raise
    except Exception as error:
        _error(
            "authoritative document PDF parsing failed",
            opinion_id=opinion_id,
            error_type=type(error).__name__,
            error=str(error),
        )
    raw_text = "\n\n".join(page_text)
    normalized_text = normalize_lines(raw_text)
    facts = {
        "pages": pages,
        "raw_chars": len(raw_text),
        "raw_text_sha256": sha256_text(raw_text),
        "normalized_chars": len(normalized_text),
        "normalized_text_sha256": sha256_text(normalized_text),
    }
    expected_facts = {
        "pages": expected["expected_pages"],
        "raw_chars": expected["expected_raw_chars"],
        "raw_text_sha256": expected["expected_raw_text_sha256"],
        "normalized_chars": expected["expected_normalized_chars"],
        "normalized_text_sha256": expected["expected_normalized_text_sha256"],
    }
    if facts != expected_facts or not normalized_text:
        _error(
            "authoritative document extracted text differs from the reviewed contract",
            opinion_id=opinion_id,
            expected=expected_facts,
            actual=facts,
        )
    return {**facts, "raw_text": raw_text, "normalized_text": normalized_text}


def _pdf_member(opinion_id: int, role: str) -> str:
    suffix = {
        "courtlistener_storage": "courtlistener",
        "ohio_official": "ohio",
    }.get(role)
    if suffix is None:
        _error("unknown authoritative PDF member role", opinion_id=opinion_id, role=role)
    return "opinion-%d.%s.pdf" % (opinion_id, suffix)


def build_generation(args) -> dict:
    destination = Path(args.out).absolute()
    if os.path.lexists(destination):
        _error(
            "authoritative document generation destination already exists",
            destination=str(destination),
        )
    if not destination.parent.is_dir():
        _error(
            "authoritative document generation parent does not exist",
            parent=str(destination.parent),
        )
    contract = load_source_contract(args.source_contract)
    source_rows_verified = verify_source_rows(args.opinions, contract)
    try:
        acquired_at = datetime.fromisoformat(args.acquired_at.replace("Z", "+00:00"))
    except ValueError as error:
        _error(
            "acquired-at is not an ISO timestamp",
            value=args.acquired_at,
            error=str(error),
        )
    if acquired_at.tzinfo is None:
        _error(
            "acquired-at must include an explicit UTC offset", value=args.acquired_at
        )
    records = []
    pdfs = {}
    for opinion_id, expected in sorted(contract["by_opinion"].items()):
        observations = []
        bodies = {}
        for role, url in (
            ("courtlistener_storage", expected["storage_url"]),
            ("ohio_official", expected["download_url"]),
        ):
            body, observation = fetch_pdf(
                url,
                max_bytes=expected["max_pdf_bytes"],
                timeout_seconds=args.timeout_seconds,
            )
            observation["role"] = role
            observations.append(observation)
            bodies[role] = body
        if bodies["courtlistener_storage"] != bodies["ohio_official"]:
            _error(
                "CourtListener and Ohio authoritative PDF bytes disagree",
                opinion_id=opinion_id,
                storage_sha256=observations[0]["sha256"],
                official_sha256=observations[1]["sha256"],
            )
        extracted = extract_pdf(bodies["courtlistener_storage"], expected)
        pdf_members = {
            role: _pdf_member(opinion_id, role) for role in sorted(bodies)
        }
        for role, member in pdf_members.items():
            pdfs[member] = bodies[role]
        records.append(
            {
                "opinion_id": opinion_id,
                "cluster_id": expected["cluster_id"],
                "source_row_sha256": stable_row_sha256(
                    source_rows_verified[opinion_id]
                ),
                "courtlistener_source": {
                    "sha1": expected["sha1"],
                    "local_path": expected["local_path"],
                    "download_url": expected["download_url"],
                },
                "pdf_members": pdf_members,
                "pdf_bytes": len(bodies["courtlistener_storage"]),
                "pdf_sha1": expected["pdf_sha1"],
                "pdf_sha256": expected["pdf_sha256"],
                "observations": observations,
                "extractor": contract["value"]["extractor"],
                "pages": extracted["pages"],
                "raw_chars": extracted["raw_chars"],
                "raw_text_sha256": extracted["raw_text_sha256"],
                "raw_text": extracted["raw_text"],
                "normalized_chars": extracted["normalized_chars"],
                "normalized_text_sha256": extracted["normalized_text_sha256"],
                "normalized_text": extracted["normalized_text"],
            }
        )
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        for member, value in sorted(pdfs.items()):
            generation.write_bytes(member, value)
        documents_output = generation.open_text(DOCUMENTS_FILE)
        for record in records:
            documents_output.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        generation.publish(
            {
                "format": GENERATION_FORMAT,
                "source_of_truth": MANIFEST_FILE,
                "source_contract": {
                    "name": Path(args.source_contract).name,
                    "sha256": contract["sha256"],
                    "format": SOURCE_FORMAT,
                },
                "source_binding": contract["value"]["source_binding"],
                "source_archive": {
                    "name": Path(args.opinions).name,
                    "sha256": contract["value"]["source_binding"][
                        "opinions_archive_sha256"
                    ],
                },
                "acquired_at": args.acquired_at,
                "extractor": contract["value"]["extractor"],
                "opinion_ids": sorted(contract["by_opinion"]),
            }
        )
    return verify_authoritative_generation(args.out)


def _read_records(root: Path) -> list[dict]:
    records = []
    with open(root / DOCUMENTS_FILE, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                _error(
                    "authoritative document record is invalid JSON",
                    line=line_number,
                    error=str(error),
                )
            if not isinstance(record, dict):
                _error(
                    "authoritative document record is not an object", line=line_number
                )
            records.append(record)
    return records


def verify_authoritative_generation(
    path: str | os.PathLike[str], *, reextract_pdfs: bool = True
) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != GENERATION_FORMAT:
        _error("unsupported authoritative document generation", path=str(root))
    expected_members = {DOCUMENTS_FILE} | {
        _pdf_member(opinion_id, role)
        for opinion_id in EXPECTED_IDENTITIES
        for role in ("courtlistener_storage", "ohio_official")
    }
    if set(manifest["files"]) != expected_members:
        _error(
            "authoritative document generation member contract mismatch",
            expected=sorted(expected_members),
            actual=sorted(manifest["files"]),
        )
    if manifest.get("opinion_ids") != sorted(EXPECTED_IDENTITIES):
        _error("authoritative generation opinion identity set mismatch")
    source_contract = manifest.get("source_contract")
    if (
        not isinstance(source_contract, dict)
        or source_contract.get("format") != SOURCE_FORMAT
        or source_contract.get("sha256") != APPROVED_SOURCE_CONTRACT_SHA256
    ):
        _error(
            "authoritative generation is not bound to the reviewed source contract",
            actual=source_contract,
        )
    extractor = manifest.get("extractor")
    if extractor != {
        "name": EXTRACTOR_NAME,
        "version": EXTRACTOR_VERSION,
        "strict": True,
        "text_method": "PageObject.extract_text-default",
        "join_pages": "two-newlines",
    }:
        _error("authoritative generation extractor mismatch", actual=extractor)
    records = _read_records(root)
    by_id = {}
    for record in records:
        opinion_id = record.get("opinion_id")
        if opinion_id in by_id or opinion_id not in EXPECTED_IDENTITIES:
            _error(
                "invalid or duplicate authoritative record opinion ID",
                opinion_id=opinion_id,
            )
        identity = EXPECTED_IDENTITIES[opinion_id]
        if any(record.get(key) != value for key, value in identity.items()):
            _error("authoritative record identity mismatch", opinion_id=opinion_id)
        expected_pdf_members = {
            role: _pdf_member(opinion_id, role)
            for role in ("courtlistener_storage", "ohio_official")
        }
        if record.get("pdf_members") != expected_pdf_members:
            _error(
                "authoritative record PDF member mapping mismatch",
                opinion_id=opinion_id,
                expected=expected_pdf_members,
                actual=record.get("pdf_members"),
            )
        expected = {
            "opinion_id": opinion_id,
            "pdf_sha1": record.get("pdf_sha1"),
            "pdf_sha256": record.get("pdf_sha256"),
            "expected_pages": record.get("pages"),
            "expected_raw_chars": record.get("raw_chars"),
            "expected_raw_text_sha256": record.get("raw_text_sha256"),
            "expected_normalized_chars": record.get("normalized_chars"),
            "expected_normalized_text_sha256": record.get("normalized_text_sha256"),
        }
        raw_text = record.get("raw_text")
        normalized_text = record.get("normalized_text")
        if (
            not isinstance(raw_text, str)
            or not raw_text
            or not isinstance(normalized_text, str)
            or not normalized_text
        ):
            _error(
                "authoritative record contains invalid extracted text",
                opinion_id=opinion_id,
            )
        stored_facts = {
            "pages": record.get("pages"),
            "raw_chars": len(raw_text),
            "raw_text_sha256": sha256_text(raw_text),
            "normalized_chars": len(normalized_text),
            "normalized_text_sha256": sha256_text(normalized_text),
        }
        declared_facts = {
            key.removeprefix("expected_"): value for key, value in expected.items()
            if key.startswith("expected_")
        }
        if (
            stored_facts != declared_facts
            or normalize_lines(raw_text) != normalized_text
        ):
            _error(
                "authoritative record extracted-text facts are inconsistent",
                opinion_id=opinion_id,
                expected=declared_facts,
                actual=stored_facts,
            )
        physical_pdfs = {}
        for role, member in expected_pdf_members.items():
            with open(root / member, "rb") as source:
                pdf = source.read()
            physical_pdfs[role] = pdf
            if (
                len(pdf) != record.get("pdf_bytes")
                or hashlib.sha1(pdf).hexdigest() != record.get("pdf_sha1")
                or hashlib.sha256(pdf).hexdigest() != record.get("pdf_sha256")
            ):
                _error(
                    "authoritative record differs from physical PDF readback",
                    opinion_id=opinion_id,
                    role=role,
                )
            if reextract_pdfs:
                extracted = extract_pdf(pdf, expected)
                if (
                    extracted["raw_text"] != raw_text
                    or extracted["normalized_text"] != normalized_text
                ):
                    _error(
                        "authoritative record differs from physical PDF text readback",
                        opinion_id=opinion_id,
                        role=role,
                    )
        if physical_pdfs["courtlistener_storage"] != physical_pdfs["ohio_official"]:
            _error(
                "persisted CourtListener and Ohio PDF bytes disagree",
                opinion_id=opinion_id,
            )
        observations = record.get("observations")
        if not isinstance(observations, list) or len(observations) != 2:
            _error(
                "authoritative acquisition observations are incomplete",
                opinion_id=opinion_id,
            )
        source = record.get("courtlistener_source")
        if not isinstance(source, dict):
            _error(
                "authoritative CourtListener source provenance is missing",
                opinion_id=opinion_id,
            )
        expected_observation_urls = {
            "courtlistener_storage": "https://storage.courtlistener.com/"
            + source.get("local_path", ""),
            "ohio_official": source.get("download_url"),
        }
        observed_roles = set()
        for observation in observations:
            role = observation.get("role")
            if (
                role in observed_roles
                or role not in expected_observation_urls
                or observation.get("url") != expected_observation_urls.get(role)
                or observation.get("sha1") != record["pdf_sha1"]
                or observation.get("sha256") != record["pdf_sha256"]
                or observation.get("bytes") != record["pdf_bytes"]
                or observation.get("content_type") != "application/pdf"
            ):
                _error(
                    "authoritative acquisition observation mismatch",
                    opinion_id=opinion_id,
                )
            observed_roles.add(role)
        if observed_roles != set(expected_observation_urls):
            _error(
                "authoritative acquisition observation roles are incomplete",
                opinion_id=opinion_id,
                expected=sorted(expected_observation_urls),
                actual=sorted(observed_roles),
            )
        by_id[opinion_id] = record
    if set(by_id) != set(EXPECTED_IDENTITIES):
        _error(
            "authoritative generation records are incomplete",
            expected=sorted(EXPECTED_IDENTITIES),
            actual=sorted(by_id),
        )
    return {
        "status": "verified",
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "opinion_ids": sorted(by_id),
        "pdf_bytes": sum(record["pdf_bytes"] for record in records),
        "normalized_chars": sum(record["normalized_chars"] for record in records),
    }


def load_authoritative_generation(
    path: str | os.PathLike[str],
    *,
    opinions_archive_sha256: str,
    snapshot_date: str,
    reextract_pdfs: bool = True,
) -> dict:
    verification = verify_authoritative_generation(
        path, reextract_pdfs=reextract_pdfs
    )
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    expected_binding = {
        "opinions_archive_sha256": opinions_archive_sha256,
        "snapshot_date": snapshot_date,
    }
    if manifest.get("source_binding") != expected_binding:
        _error(
            "authoritative generation has the wrong source binding",
            expected=expected_binding,
            actual=manifest.get("source_binding"),
        )
    records = _read_records(root)
    return {
        "manifest": manifest,
        "manifest_sha256": verification["manifest_sha256"],
        "by_opinion": {record["opinion_id"]: record for record in records},
    }


def resolve_pdf_record(
    record: dict, *, row: dict, cluster_id: int, where: str
) -> ResolvedText:
    opinion_id = int(row["id"])
    actual = {
        "cluster_id": cluster_id,
        "source_row_sha256": stable_row_sha256(row),
        "sha1": row["sha1"],
        "local_path": row["local_path"],
        "download_url": row["download_url"],
    }
    expected = {
        "cluster_id": record["cluster_id"],
        "source_row_sha256": record["source_row_sha256"],
        **record["courtlistener_source"],
    }
    if actual != expected:
        _error(
            "authoritative PDF record does not bind to this bulk row",
            where=where,
            opinion_id=opinion_id,
            expected=expected,
            actual=actual,
        )
    fields = {}
    for field in TEXT_FIELDS:
        raw = row.get(field)
        if raw is not None and not isinstance(raw, str):
            _error(
                "bulk text field is not a string",
                where=where,
                opinion_id=opinion_id,
                field=field,
            )
        raw = raw or ""
        fields[field] = {
            "raw_present": row.get(field) is not None,
            "nonempty": bool(raw.strip()),
            "raw_chars": len(raw),
            "raw_sha256": sha256_text(raw) if raw else None,
        }
    nonempty = [field for field, facts in fields.items() if facts["nonempty"]]
    if nonempty:
        _error(
            "authoritative PDF record cannot replace a nonempty bulk text field",
            where=where,
            opinion_id=opinion_id,
            nonempty_fields=nonempty,
        )
    text = record["normalized_text"]
    return ResolvedText(
        text=text,
        source_field="authoritative_pdf_supplement",
        source_raw_sha256=record["raw_text_sha256"],
        normalized_sha256=record["normalized_text_sha256"],
        normalized_chars=record["normalized_chars"],
        caption_blocks=tuple(
            line.strip() for line in text.splitlines() if line.strip()
        ),
        fields=fields,
    )


def record_provenance(record: dict, manifest_sha256: str) -> dict:
    return {
        "generation_manifest_sha256": manifest_sha256,
        "opinion_id": record["opinion_id"],
        "cluster_id": record["cluster_id"],
        "source_row_sha256": record["source_row_sha256"],
        "courtlistener_source": record["courtlistener_source"],
        "pdf_members": record["pdf_members"],
        "pdf_bytes": record["pdf_bytes"],
        "pdf_sha1": record["pdf_sha1"],
        "pdf_sha256": record["pdf_sha256"],
        "observations": record["observations"],
        "extractor": record["extractor"],
        "pages": record["pages"],
        "raw_chars": record["raw_chars"],
        "raw_text_sha256": record["raw_text_sha256"],
        "normalized_chars": record["normalized_chars"],
        "normalized_text_sha256": record["normalized_text_sha256"],
    }


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="acquire, verify, and publish the exact PDF generation"
    )
    build.add_argument("--source-contract", required=True)
    build.add_argument("--opinions", required=True)
    build.add_argument("--acquired-at", required=True)
    build.add_argument("--timeout-seconds", type=int, default=60)
    build.add_argument("--out", required=True)
    verify = commands.add_parser(
        "verify", help="independently verify the physical generation"
    )
    verify.add_argument("--generation", required=True)
    args = parse_cli_args(parser)
    try:
        if args.command == "build":
            if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
                _error(
                    "timeout-seconds must be in [1, 300]", value=args.timeout_seconds
                )
            result = build_generation(args)
        else:
            result = verify_authoritative_generation(args.generation)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="authoritative_document_command_error",
            remediation=(
                "repair the named authoritative-document source or generation condition "
                "and rerun at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
