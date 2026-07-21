#!/usr/bin/env python3
"""Build a canonical-content ingest generation and lossless opinion aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from cuyahoga_contract import (
    CAP_COURTS,
    GENERIC_OHIO_APPEALS_COURT,
    TEXT_FIELDS,
    sha256_text,
)
from extract_cuyahoga import (
    MANIFEST_FILE as EXTRACT_MANIFEST,
    RAW_FILE,
    verify_extract_generation,
)
from law_generation import (
    GenerationPublisher,
    generation_member,
    sha256_file,
    verify_generation,
)
from structured_error import (
    StructuredArgumentParser,
    StructuredError,
    parse_cli_args,
    write_error,
)


LEGACY_FORMATS = {
    "calyx-cuyahoga-ingest-generation-v2",
    "calyx-cuyahoga-ingest-generation-v3",
}
FORMAT = "calyx-cuyahoga-ingest-generation-v4"
FULL_BATCH_FORMATS = {"calyx-cuyahoga-ingest-generation-v3", FORMAT}
MANIFEST_FILE = "ingest_manifest.json"
CAP_FILE = "cap-cuyahoga.jsonl"
MODERN_FILE = "ohctapp8-modern.jsonl"
FULL_FILE = "cuyahoga-canonical.jsonl"
ALIASES_FILE = "opinion_aliases.jsonl"
SAMPLE_FILE = "sample_1k.jsonl"
SAMPLE_SIZE = 1_000
LONGEST_COUNT = 10
ANCHOR_COUNT = 6
PROGRESS_EVERY = 2_000
REQUIRED_METADATA = ("source_dataset", "source_sha256", "license", "retrieval_ts")
LOCATOR_KEYS = ("source_url", "doi", "pmid", "pmcid")


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def producer_contract() -> dict:
    root = Path(__file__).resolve().parent
    members = {}
    for name in (
        "authoritative_documents.py",
        "build_ingest_jsonl.py",
        "cuyahoga_contract.py",
        "extract_cuyahoga.py",
        "law_generation.py",
        "signal_audit.py",
        "source_scan_lock.py",
    ):
        path = root / name
        if not path.is_file() or path.is_symlink():
            fail("ingest producer source is not a plain file", path=str(path))
        members[name] = sha256_file(path)
    config = {
        "format": FORMAT,
        "full_file": FULL_FILE,
        "anchor_count": ANCHOR_COUNT,
        "text_fields": list(TEXT_FIELDS),
        "canonical_identity": "exact composed UTF-8 text",
    }
    return {
        "implementation_sha256": members,
        "config": config,
        "config_sha256": sha256_text(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    }


def source_extract_binding(extract_generation: str, extract_report: dict) -> dict:
    manifest = verify_generation(
        Path(extract_generation).absolute(), EXTRACT_MANIFEST
    )
    producer = manifest.get("producer")
    return {
        "manifest_sha256": extract_report["manifest_sha256"],
        "accepted_rows": extract_report["accepted_rows"],
        "format": manifest["format"],
        "normalized_schema_version": manifest.get(
            "normalized_schema_version", manifest["format"]
        ),
        "producer_config_sha256": (
            producer.get("config_sha256") if isinstance(producer, dict) else ""
        ),
        "selector_version": manifest.get("selector_version"),
        "text_policy_version": manifest.get("text_policy_version"),
        "correction_policy_version": manifest.get("correction_policy_version"),
        "source_archive_sha256": {
            role: value["archive_sha256"]
            for role, value in manifest["source_archives"].items()
            if role in {"dockets", "clusters", "opinions"}
        },
    }


class IngestBuildError(StructuredError):
    code = "cuyahoga_ingest_generation_error"
    default_remediation = (
        "repair the named extract or ingest mismatch and rebuild the complete immutable "
        "ingest generation at a new destination"
    )


def fail(message: str, **context):
    raise IngestBuildError(message, **context)


def raw_rows(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                fail("raw extraction contains a blank line", path=str(path), line=lineno)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail("raw extraction has invalid JSON", path=str(path), line=lineno, error=str(error))
            if not isinstance(row, dict):
                fail("raw extraction row is not an object", path=str(path), line=lineno)
            yield lineno, row


def field_string(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def era_bucket(date_filed: str | None) -> str:
    if not date_filed or len(date_filed) < 4 or not date_filed[:4].isdigit():
        return "unknown"
    return "%ds" % (int(date_filed[:4]) // 10 * 10)


def cited_bucket(value, *, line: int) -> str:
    if value in (None, ""):
        return "none"
    try:
        count = int(value)
    except (TypeError, ValueError):
        fail("citation_count is not an integer", line=line, value=value)
    if count < 0:
        fail("citation_count is negative", line=line, value=value)
    if count == 0:
        return "none"
    if count < 10:
        return "some"
    return "high"


def build_text(row: dict, *, line: int) -> str:
    title = row.get("case_name_full") or row.get("case_name") or ""
    body = row.get("text")
    if not isinstance(body, str) or not body:
        fail("accepted raw row has missing/empty text", line=line, opinion_id=row.get("opinion_id"))
    text = (
        "%s\n%s | filed %s | docket %s | %s\nJudges: %s\n\n%s"
        % (
            title,
            field_string(row, "court_id"),
            field_string(row, "date_filed"),
            field_string(row, "docket_number"),
            field_string(row, "precedential_status"),
            field_string(row, "judges") or field_string(row, "author_str"),
            body,
        )
    )
    if not text:
        fail("composed ingest text is empty", line=line)
    return text


def validate_raw_row(row: dict, *, line: int) -> int:
    opinion_id = row.get("opinion_id")
    if not isinstance(opinion_id, int) or isinstance(opinion_id, bool) or opinion_id <= 0:
        fail("raw opinion_id must be a positive integer", line=line, value=opinion_id)
    selection = row.get("selection")
    if not isinstance(selection, dict) or selection.get("status") != "accepted":
        fail("raw row is not accepted by selector", line=line, opinion_id=opinion_id)
    provenance = row.get("text_provenance")
    text = row.get("text")
    if (
        not isinstance(provenance, dict)
        or not isinstance(text, str)
        or provenance.get("normalized_sha256") != sha256_text(text)
    ):
        fail("raw row text provenance mismatch", line=line, opinion_id=opinion_id)
    canonical_url = row.get("canonical_source_url")
    expected_url = "https://www.courtlistener.com/opinion/%s/%s/" % (
        row.get("cluster_id"),
        row.get("cluster_slug"),
    )
    if canonical_url != expected_url:
        fail(
            "raw row canonical source URL mismatch",
            line=line,
            opinion_id=opinion_id,
            expected=expected_url,
            actual=canonical_url,
        )
    return opinion_id


def canonical_preference(row: dict, opinion_id: int) -> tuple[int, int]:
    source_field = row.get("text_source")
    if source_field == "html_with_citations":
        source_rank = 0
    elif source_field == "authoritative_pdf_supplement":
        source_rank = 1
    elif source_field == "plain_text_no_html_with_citations":
        source_rank = 2
    else:
        fail(
            "raw row has prohibited text source for canonicalization",
            opinion_id=opinion_id,
            source_field=source_field,
        )
    return source_rank, opinion_id


def anchors(row: dict, dataset_tag: str, *, line: int) -> list[dict]:
    return [
        {"kind": "label:court", "value": field_string(row, "court_id") or "unknown"},
        {
            "kind": "label:precedential",
            "value": field_string(row, "precedential_status") or "unknown",
        },
        {"kind": "label:optype", "value": field_string(row, "opinion_type") or "unknown"},
        {"kind": "label:cited", "value": cited_bucket(row.get("citation_count"), line=line)},
        {"kind": "label:era", "value": era_bucket(row.get("date_filed"))},
        {"kind": "label:dataset", "value": dataset_tag},
    ]


def metadata(
    row: dict,
    *,
    canonical_opinion_id: int,
    ingest_text_sha256: str,
    dataset_tag: str,
    source_extract: dict,
    full_lineage: bool = True,
) -> dict[str, str]:
    carry = [
        "opinion_id",
        "cluster_id",
        "cluster_slug",
        "docket_id",
        "court_id",
        "cuyahoga_signal",
        "date_filed",
        "case_name",
        "case_name_full",
        "docket_number",
        "citation_count",
        "opinion_type",
        "author_id",
        "author_str",
        "precedential_status",
        "text_source",
        "source_snapshot_date",
        "source_acquired_at",
    ]
    if full_lineage:
        carry.extend(
            [
                "per_curiam",
                "judges",
                "disposition",
                "posture",
                "nature_of_suit",
                "syllabus",
                "cluster_source",
            ]
        )
    result = {key: field_string(row, key) for key in carry}
    text_provenance = row["text_provenance"]
    source_archive = row["source_archive"]
    courtlistener_source = row.get("courtlistener_opinion_source")
    field_facts = text_provenance.get("fields")
    if full_lineage:
        if not isinstance(courtlistener_source, dict) or set(courtlistener_source) != {
            "source_row_sha256",
            "sha1",
            "download_url",
            "local_path",
            "date_created",
            "date_modified",
        }:
            fail(
                "CourtListener opinion source provenance is malformed",
                opinion_id=row["opinion_id"],
            )
        if not isinstance(field_facts, dict) or set(field_facts) != set(TEXT_FIELDS):
            fail(
                "competing source-field provenance is malformed",
                opinion_id=row["opinion_id"],
            )
    correction = row.get("correction")
    authoritative_source = row.get("authoritative_document_source")
    if authoritative_source is not None and not isinstance(
        authoritative_source, dict
    ):
        fail(
            "authoritative document provenance is not an object",
            opinion_id=row["opinion_id"],
        )
    observations = (
        authoritative_source.get("observations", [])
        if authoritative_source is not None
        else []
    )
    observation_urls = {
        observation.get("role"): observation.get("url")
        for observation in observations
        if isinstance(observation, dict)
    }
    result.update(
        {
            "canonical_opinion_id": str(canonical_opinion_id),
            "source_dataset": dataset_tag,
            "jurisdiction_country": "United States",
            "jurisdiction_court_system": "state",
            "jurisdiction_state": "Ohio",
            "jurisdiction_county": "Cuyahoga",
            "jurisdiction_appellate_district": "Eighth",
            "source_sha256": text_provenance["source_raw_sha256"],
            "normalized_text_sha256": text_provenance["normalized_sha256"],
            "ingest_text_sha256": ingest_text_sha256,
            "source_archive_sha256": source_archive["archive_sha256"],
            "license": "public-domain (CourtListener bulk export; US court opinions)",
            "retrieval_ts": row["source_acquired_at"],
            "source_url": row["canonical_source_url"],
            "correction_id": (
                correction["correction_id"] if isinstance(correction, dict) else ""
            ),
            "authoritative_document_manifest_sha256": (
                authoritative_source["generation_manifest_sha256"]
                if authoritative_source is not None
                else ""
            ),
            "authoritative_pdf_sha256": (
                authoritative_source["pdf_sha256"]
                if authoritative_source is not None
                else ""
            ),
            "authoritative_pdf_pages": (
                str(authoritative_source["pages"])
                if authoritative_source is not None
                else ""
            ),
            "authoritative_pdf_extractor": (
                "%s@%s"
                % (
                    authoritative_source["extractor"]["name"],
                    authoritative_source["extractor"]["version"],
                )
                if authoritative_source is not None
                else ""
            ),
            "authoritative_pdf_official_url": observation_urls.get(
                "ohio_official", ""
            ),
            "authoritative_pdf_storage_url": observation_urls.get(
                "courtlistener_storage", ""
            ),
        }
    )
    if full_lineage:
        result.update(
            {
                "normalized_schema_version": source_extract["format"],
                "source_extract_manifest_sha256": source_extract["manifest_sha256"],
                "source_extractor_config_sha256": source_extract[
                    "producer_config_sha256"
                ],
                "source_archive_name": source_archive["archive_name"],
                "courtlistener_source_row_sha256": courtlistener_source[
                    "source_row_sha256"
                ],
                "courtlistener_sha1": field_string(courtlistener_source, "sha1"),
                "courtlistener_download_url": field_string(
                    courtlistener_source, "download_url"
                ),
                "courtlistener_local_path": field_string(
                    courtlistener_source, "local_path"
                ),
                "courtlistener_date_created": field_string(
                    courtlistener_source, "date_created"
                ),
                "courtlistener_date_modified": field_string(
                    courtlistener_source, "date_modified"
                ),
            }
        )
        for field in TEXT_FIELDS:
            facts = field_facts[field]
            if not isinstance(facts, dict) or set(facts) != {
                "raw_present",
                "nonempty",
                "raw_chars",
                "raw_sha256",
            }:
                fail(
                    "source-field facts have the wrong schema",
                    opinion_id=row["opinion_id"],
                    field=field,
                )
            prefix = "source_field_%s_" % field
            result[prefix + "raw_present"] = field_string(facts, "raw_present")
            result[prefix + "nonempty"] = field_string(facts, "nonempty")
            result[prefix + "raw_chars"] = field_string(facts, "raw_chars")
            result[prefix + "raw_sha256"] = field_string(facts, "raw_sha256")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in result.items()):
        fail("metadata keys and values must all be strings", opinion_id=row["opinion_id"])
    return result


def alias_record(
    row: dict,
    *,
    canonical_opinion_id: int,
    content_sha256: str,
    source_extract: dict,
) -> dict:
    return {
        "opinion_id": row["opinion_id"],
        "canonical_opinion_id": canonical_opinion_id,
        "is_canonical": row["opinion_id"] == canonical_opinion_id,
        "content_sha256": content_sha256,
        "source_normalized_sha256": row["text_provenance"]["normalized_sha256"],
        "source_raw_sha256": row["text_provenance"]["source_raw_sha256"],
        "source_field": row["text_source"],
        "cluster_id": row["cluster_id"],
        "docket_id": row["docket_id"],
        "court_id": row["court_id"],
        "source_url": row["canonical_source_url"],
        "source_lineage": {
            "normalized_schema_version": source_extract["format"],
            "source_extract": source_extract,
            "source_archive": row["source_archive"],
            "source_snapshot_date": row["source_snapshot_date"],
            "source_acquired_at": row["source_acquired_at"],
            "courtlistener_opinion_source": row["courtlistener_opinion_source"],
            "text_source": row["text_source"],
            "text_provenance": row["text_provenance"],
            "authoritative_document_source": row.get(
                "authoritative_document_source"
            ),
            "cluster_slug": row["cluster_slug"],
            "cluster_source": row["cluster_source"],
            "case_name": row["case_name"],
            "case_name_full": row["case_name_full"],
            "syllabus": row["syllabus"],
        },
    }


def validate_batch_line(value: dict, *, where: str) -> str:
    if not isinstance(value, dict):
        fail("batch line is not an object", where=where)
    text = value.get("text")
    if not isinstance(text, str) or not text:
        fail("batch line text is missing/empty", where=where)
    md = value.get("metadata")
    if not isinstance(md, dict):
        fail("batch metadata is missing", where=where)
    for key, item in md.items():
        if not isinstance(key, str) or not isinstance(item, str):
            fail("batch metadata is not BTreeMap<String,String> compatible", where=where)
    for key in REQUIRED_METADATA:
        if not md.get(key, "").strip():
            fail("batch provenance metadata is missing", where=where, key=key)
    if not any(md.get(key, "").strip() for key in LOCATOR_KEYS):
        fail("batch metadata has no source locator", where=where)
    if md.get("ingest_text_sha256") != sha256_text(text):
        fail("batch ingest_text_sha256 mismatch", where=where)
    row_anchors = value.get("anchors")
    if not isinstance(row_anchors, list) or len(row_anchors) != ANCHOR_COUNT:
        fail("batch anchor count mismatch", where=where, actual=len(row_anchors or []))
    for anchor in row_anchors:
        if (
            not isinstance(anchor, dict)
            or not isinstance(anchor.get("kind"), str)
            or not anchor["kind"]
            or not isinstance(anchor.get("value"), str)
            or not anchor["value"]
        ):
            fail("batch anchor is invalid", where=where, anchor=anchor)
    return text


def canonical_index(raw_path: Path) -> tuple[dict, dict, set]:
    canonical_keys = {}
    counts = {}
    opinion_ids = set()
    rows = 0
    for line, row in raw_rows(raw_path):
        rows += 1
        opinion_id = validate_raw_row(row, line=line)
        if opinion_id in opinion_ids:
            fail("duplicate raw opinion_id", line=line, opinion_id=opinion_id)
        opinion_ids.add(opinion_id)
        text = build_text(row, line=line)
        digest = sha256_text(text)
        counts[digest] = counts.get(digest, 0) + 1
        candidate = canonical_preference(row, opinion_id)
        current = canonical_keys.get(digest)
        if current is None or candidate < current:
            canonical_keys[digest] = candidate
        if rows % PROGRESS_EVERY == 0:
            sys.stderr.write("canonical-index: %d rows, %d content identities\n" % (rows, len(canonical_keys)))
    if not opinion_ids:
        fail("raw extraction contains zero opinions", path=str(raw_path))
    return (
        {digest: key[1] for digest, key in canonical_keys.items()},
        counts,
        opinion_ids,
    )


def write_generation_rows(
    raw_path: Path,
    canonical: dict,
    content_counts: dict,
    dataset_tag: str,
    source_extract: dict,
    cap_output,
    modern_output,
    alias_output,
) -> dict:
    exact_duplicate_text = {}
    content_rows = 0
    alias_rows = 0
    duplicate_aliases = 0
    cap_rows = 0
    modern_rows = 0
    longest = []
    for line, row in raw_rows(raw_path):
        opinion_id = validate_raw_row(row, line=line)
        text = build_text(row, line=line)
        digest = sha256_text(text)
        canonical_id = canonical[digest]
        if content_counts[digest] > 1:
            first = exact_duplicate_text.setdefault(digest, text)
            if first != text:
                fail(
                    "SHA-256 collision between unequal composed texts",
                    line=line,
                    opinion_id=opinion_id,
                    digest=digest,
                )
        alias = alias_record(
            row,
            canonical_opinion_id=canonical_id,
            content_sha256=digest,
            source_extract=source_extract,
        )
        alias_output.write(json.dumps(alias, ensure_ascii=False, sort_keys=True) + "\n")
        alias_rows += 1
        if opinion_id != canonical_id:
            duplicate_aliases += 1
            continue
        batch = {
            "text": text,
            "metadata": metadata(
                row,
                canonical_opinion_id=canonical_id,
                ingest_text_sha256=digest,
                dataset_tag=dataset_tag,
                source_extract=source_extract,
            ),
            "anchors": anchors(row, dataset_tag, line=line),
        }
        validate_batch_line(batch, where="raw line %d" % line)
        serialized = json.dumps(batch, ensure_ascii=False, sort_keys=True)
        court = row["court_id"]
        if court in CAP_COURTS:
            output = cap_output
            file_name = CAP_FILE
            row_number = cap_rows
            cap_rows += 1
        elif court == GENERIC_OHIO_APPEALS_COURT:
            output = modern_output
            file_name = MODERN_FILE
            row_number = modern_rows
            modern_rows += 1
        else:
            fail("accepted raw row has unsupported court_id", line=line, court_id=court)
        output.write(serialized + "\n")
        content_rows += 1
        longest.append((len(text), digest, file_name, row_number))
        if content_rows % PROGRESS_EVERY == 0:
            sys.stderr.write("ingest-write: %d canonical rows, %d aliases\n" % (content_rows, alias_rows))
    return {
        "source_opinion_rows": alias_rows,
        "canonical_content_rows": content_rows,
        "duplicate_aliases": duplicate_aliases,
        "cap_rows": cap_rows,
        "modern_rows": modern_rows,
        "longest": longest,
    }


def write_sample(generation: GenerationPublisher, counts: dict) -> int:
    candidates = []
    for file_name in (CAP_FILE, MODERN_FILE):
        path = generation.path(file_name)
        with open(path, "r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                row = json.loads(line)
                text = row["text"]
                digest = row["metadata"]["ingest_text_sha256"]
                rank = hashlib.sha256(("sample-v2:" + digest).encode("ascii")).hexdigest()
                candidates.append((rank, file_name, index, len(text), digest))
    longest = sorted(candidates, key=lambda item: (-item[3], item[4]))[:LONGEST_COUNT]
    wanted = {(item[1], item[2]) for item in longest}
    for item in sorted(candidates):
        if len(wanted) >= min(SAMPLE_SIZE, len(candidates)):
            break
        wanted.add((item[1], item[2]))
    output = generation.open_text(SAMPLE_FILE)
    written = 0
    for file_name in (CAP_FILE, MODERN_FILE):
        with open(generation.path(file_name), "r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                if (file_name, index) in wanted:
                    output.write(line)
                    written += 1
    if written != len(wanted):
        fail("sample write count mismatch", expected=len(wanted), actual=written)
    return written


def write_full_batch(generation: GenerationPublisher) -> int:
    output = generation.open_text(FULL_FILE)
    written = 0
    for file_name in (CAP_FILE, MODERN_FILE):
        with open(generation.path(file_name), "r", encoding="utf-8") as source:
            for line in source:
                output.write(line)
                written += 1
    return written


def verify_full_batch(root: Path) -> None:
    offset = 0
    with open(root / FULL_FILE, "rb") as combined:
        for file_name in (CAP_FILE, MODERN_FILE):
            with open(root / file_name, "rb") as source:
                while True:
                    expected = source.read(1024 * 1024)
                    if not expected:
                        break
                    actual = combined.read(len(expected))
                    if actual != expected:
                        fail(
                            "full ingest member differs from canonical partitions",
                            file=file_name,
                            offset=offset,
                            expected_bytes=len(expected),
                            actual_bytes=len(actual),
                        )
                    offset += len(expected)
        trailing = combined.read(1)
        if trailing:
            fail(
                "full ingest member has bytes after canonical partitions",
                offset=offset,
            )


def verify_ingest_generation(path: str, extract_generation: str | None = None) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    generation_format = manifest.get("format")
    if generation_format not in LEGACY_FORMATS | {FORMAT}:
        fail("unsupported ingest generation format", format=manifest.get("format"))
    if generation_format == FORMAT:
        if manifest.get("normalized_schema_version") != FORMAT:
            fail("ingest normalized schema version differs from its format")
        producer = manifest.get("producer")
        implementation = (
            producer.get("implementation_sha256")
            if isinstance(producer, dict)
            else None
        )
        config = producer.get("config") if isinstance(producer, dict) else None
        if (
            not isinstance(implementation, dict)
            or set(implementation)
            != {
                "authoritative_documents.py",
                "build_ingest_jsonl.py",
                "cuyahoga_contract.py",
                "extract_cuyahoga.py",
                "law_generation.py",
                "signal_audit.py",
                "source_scan_lock.py",
            }
            or any(not is_sha256(value) for value in implementation.values())
            or not isinstance(config, dict)
            or producer.get("config_sha256")
            != sha256_text(
                json.dumps(
                    config,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        ):
            fail("ingest producer provenance is malformed")
        if config != {
            "format": FORMAT,
            "full_file": FULL_FILE,
            "anchor_count": ANCHOR_COUNT,
            "text_fields": list(TEXT_FIELDS),
            "canonical_identity": "exact composed UTF-8 text",
        }:
            fail("ingest producer config differs from manifest contract")
        if not is_sha256(
            manifest.get("source_extract_generation", {}).get(
                "producer_config_sha256"
            )
        ):
            fail("ingest manifest lacks the source extractor config binding")
    required = {CAP_FILE, MODERN_FILE, ALIASES_FILE, SAMPLE_FILE}
    if generation_format in FULL_BATCH_FORMATS:
        required.add(FULL_FILE)
    if set(manifest["files"]) != required:
        fail("ingest generation member contract mismatch", actual=sorted(manifest["files"]))

    content = {}
    batches = {}
    canonical_ids = set()
    output_counts = {}
    for file_name in (CAP_FILE, MODERN_FILE):
        rows = 0
        with open(root / file_name, "r", encoding="utf-8") as source:
            for lineno, line in enumerate(source, 1):
                try:
                    batch = json.loads(line)
                except json.JSONDecodeError as error:
                    fail("invalid batch JSON", file=file_name, line=lineno, error=str(error))
                text = validate_batch_line(batch, where="%s line %d" % (file_name, lineno))
                digest = sha256_text(text)
                if digest in content:
                    fail("duplicate canonical content across ingest files", digest=digest)
                opinion_id = int(batch["metadata"]["opinion_id"])
                canonical_id = int(batch["metadata"]["canonical_opinion_id"])
                if opinion_id != canonical_id:
                    fail("canonical batch row has differing canonical_opinion_id", opinion_id=opinion_id)
                if opinion_id in canonical_ids:
                    fail("duplicate canonical opinion ID", opinion_id=opinion_id)
                canonical_ids.add(opinion_id)
                content[digest] = opinion_id
                batches[opinion_id] = batch
                rows += 1
        output_counts[file_name] = rows

    aliases = {}
    alias_groups = {}
    alias_rows = 0
    duplicate_aliases = 0
    with open(root / ALIASES_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            try:
                alias = json.loads(line)
            except json.JSONDecodeError as error:
                fail("invalid alias JSON", line=lineno, error=str(error))
            opinion_id = alias.get("opinion_id")
            if not isinstance(opinion_id, int) or opinion_id in aliases:
                fail("invalid/duplicate alias opinion_id", line=lineno, opinion_id=opinion_id)
            canonical_id = alias.get("canonical_opinion_id")
            digest = alias.get("content_sha256")
            if alias.get("source_field") not in {
                "html_with_citations",
                "plain_text_no_html_with_citations",
                "authoritative_pdf_supplement",
            }:
                fail("alias has prohibited source_field", line=lineno)
            if content.get(digest) != canonical_id:
                fail(
                    "alias does not resolve to physical canonical content",
                    line=lineno,
                    opinion_id=opinion_id,
                    canonical_id=canonical_id,
                    digest=digest,
                )
            if alias.get("is_canonical") != (opinion_id == canonical_id):
                fail("alias is_canonical flag mismatch", line=lineno, opinion_id=opinion_id)
            aliases[opinion_id] = alias
            alias_groups.setdefault(digest, []).append(opinion_id)
            alias_rows += 1
            duplicate_aliases += opinion_id != canonical_id

    if set(content.values()) != canonical_ids:
        fail("canonical content and canonical ID sets differ")
    if set(alias_groups) != set(content):
        fail(
            "physical canonical content and alias digest sets differ",
            content_only=sorted(set(content) - set(alias_groups))[:20],
            alias_only=sorted(set(alias_groups) - set(content))[:20],
        )
    alias_targets = {alias["canonical_opinion_id"] for alias in aliases.values()}
    if alias_targets != canonical_ids:
        fail(
            "physical canonical IDs and alias target sets differ",
            content_only=sorted(canonical_ids - alias_targets)[:20],
            alias_only=sorted(alias_targets - canonical_ids)[:20],
        )
    for digest, opinion_ids in alias_groups.items():
        expected_canonical = min(
            opinion_ids,
            key=lambda opinion_id: canonical_preference(
                {"text_source": aliases[opinion_id].get("source_field")},
                opinion_id,
            ),
        )
        if content[digest] != expected_canonical:
            fail(
                "canonical opinion does not use preferred source then minimum ID",
                digest=digest,
                declared=content[digest],
                expected=expected_canonical,
            )
    counts = manifest["row_counts"]
    actual = {
        "source_opinion_rows": alias_rows,
        "canonical_content_rows": len(content),
        "duplicate_aliases": duplicate_aliases,
        "cap_rows": output_counts[CAP_FILE],
        "modern_rows": output_counts[MODERN_FILE],
    }
    for key, value in actual.items():
        if counts.get(key) != value:
            fail("manifest row count mismatch", key=key, declared=counts.get(key), actual=value)
    if generation_format in FULL_BATCH_FORMATS:
        if manifest.get("canonicalization", {}).get("full_batch_file") != FULL_FILE:
            fail("ingest generation does not name its full batch member")
        if counts.get("full_rows") != len(content):
            fail(
                "manifest full row count mismatch",
                declared=counts.get("full_rows"),
                actual=len(content),
            )
        verify_full_batch(root)

    if extract_generation is not None:
        extract_report = verify_extract_generation(extract_generation)
        if generation_format == FORMAT:
            expected_source_extract = source_extract_binding(
                extract_generation, extract_report
            )
            if manifest.get("source_extract_generation") != expected_source_extract:
                fail("ingest manifest differs from the physical extract binding")
        raw_path = generation_member(extract_generation, EXTRACT_MANIFEST, RAW_FILE)
        source_ids = set()
        for line, row in raw_rows(raw_path):
            opinion_id = validate_raw_row(row, line=line)
            source_ids.add(opinion_id)
            text = build_text(row, line=line)
            alias = aliases.get(opinion_id)
            if alias is None or alias["content_sha256"] != sha256_text(text):
                fail("physical raw row does not match alias relation", opinion_id=opinion_id)
            if alias["source_normalized_sha256"] != row["text_provenance"]["normalized_sha256"]:
                fail("alias normalized source digest mismatch", opinion_id=opinion_id)
            if alias["source_raw_sha256"] != row["text_provenance"]["source_raw_sha256"]:
                fail("alias raw source digest mismatch", opinion_id=opinion_id)
            if generation_format == FORMAT:
                expected_alias = alias_record(
                    row,
                    canonical_opinion_id=alias["canonical_opinion_id"],
                    content_sha256=sha256_text(text),
                    source_extract=manifest["source_extract_generation"],
                )
                if alias != expected_alias:
                    fail(
                        "opinion alias loses source lineage",
                        opinion_id=opinion_id,
                    )
            if opinion_id == alias["canonical_opinion_id"]:
                batch = batches.get(opinion_id)
                if batch is None or batch["text"] != text:
                    fail("canonical physical ingest text differs from raw composition", opinion_id=opinion_id)
                md = batch["metadata"]
                expected_metadata = metadata(
                    row,
                    canonical_opinion_id=opinion_id,
                    ingest_text_sha256=sha256_text(text),
                    dataset_tag=manifest["dataset_tag"],
                    source_extract=manifest["source_extract_generation"],
                    full_lineage=generation_format == FORMAT,
                )
                if md != expected_metadata:
                    fail(
                        "canonical ingest metadata differs from raw provenance",
                        opinion_id=opinion_id,
                    )
        if source_ids != set(aliases):
            fail(
                "raw source and alias opinion sets differ",
                raw_only=sorted(source_ids - set(aliases))[:20],
                alias_only=sorted(set(aliases) - source_ids)[:20],
            )
    sample_digests = set()
    with open(root / SAMPLE_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            try:
                batch = json.loads(line)
            except json.JSONDecodeError as error:
                fail("invalid sample JSON", line=lineno, error=str(error))
            text = validate_batch_line(batch, where="%s line %d" % (SAMPLE_FILE, lineno))
            digest = sha256_text(text)
            if digest not in content or digest in sample_digests:
                fail("sample row is missing from canonical content or duplicated", line=lineno)
            sample_digests.add(digest)
    if len(sample_digests) != counts.get("sample_rows"):
        fail(
            "sample count differs from physical sample rows",
            declared=counts.get("sample_rows"),
            actual=len(sample_digests),
        )
    longest_digests = {
        item["content_sha256"] for item in manifest.get("longest_content", [])
    }
    if not longest_digests.issubset(sample_digests):
        fail("sample does not contain every declared longest content row")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "canonical_content_rows": len(content),
        "source_opinion_rows": alias_rows,
        "duplicate_aliases": duplicate_aliases,
        "status": "verified",
    }


def build(args) -> dict:
    producer = producer_contract()
    extract_report = verify_extract_generation(args.extract_generation)
    source_extract = source_extract_binding(args.extract_generation, extract_report)
    raw_path = generation_member(args.extract_generation, EXTRACT_MANIFEST, RAW_FILE)
    canonical, content_counts, source_ids = canonical_index(raw_path)
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        cap_output = generation.open_text(CAP_FILE)
        modern_output = generation.open_text(MODERN_FILE)
        alias_output = generation.open_text(ALIASES_FILE)
        counts = write_generation_rows(
            raw_path,
            canonical,
            content_counts,
            args.dataset_tag,
            source_extract,
            cap_output,
            modern_output,
            alias_output,
        )
        if counts["source_opinion_rows"] != len(source_ids):
            fail(
                "source row count changed between ingest scans",
                first=len(source_ids),
                second=counts["source_opinion_rows"],
            )
        for handle in (cap_output, modern_output, alias_output):
            handle.flush()
        full_rows = write_full_batch(generation)
        if full_rows != counts["canonical_content_rows"]:
            fail(
                "full ingest member row count mismatch",
                expected=counts["canonical_content_rows"],
                actual=full_rows,
            )
        sample_rows = write_sample(generation, counts)
        longest = counts.pop("longest")
        counts["full_rows"] = full_rows
        counts["sample_rows"] = sample_rows
        if producer_contract() != producer:
            fail("ingest producer bytes changed before publication")
        generation.publish(
            {
                "format": FORMAT,
                "normalized_schema_version": FORMAT,
                "producer": producer,
                "dataset_tag": args.dataset_tag,
                "source_extract_generation": source_extract,
                "canonicalization": {
                    "identity": "exact composed UTF-8 text",
                    "canonical_choice": (
                        "html_with_citations, then authoritative PDF supplement, "
                        "then plain text; minimum CourtListener opinion_id within rank"
                    ),
                    "alias_file": ALIASES_FILE,
                    "full_batch_file": FULL_FILE,
                    "collision_check": "byte-for-byte comparison for every repeated SHA-256",
                },
                "row_counts": counts,
                "longest_content": [
                    {
                        "chars": item[0],
                        "content_sha256": item[1],
                        "file": item[2],
                        "row_index": item[3],
                    }
                    for item in sorted(longest, key=lambda item: (-item[0], item[1]))[:LONGEST_COUNT]
                ],
                "source_of_truth": MANIFEST_FILE,
            }
        )
    result = verify_ingest_generation(args.out, args.extract_generation)
    if producer_contract() != producer:
        fail("ingest producer bytes changed before final readback")
    return result


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("build")
    command.add_argument("--extract-generation", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--dataset-tag", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--generation", required=True)
    verify.add_argument("--extract-generation")
    args = parse_cli_args(parser)
    try:
        if args.command == "build":
            report = build(args)
        else:
            report = verify_ingest_generation(args.generation, args.extract_generation)
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_ingest_generation_error",
            remediation=(
                "repair the named extract or ingest mismatch and rebuild the complete "
                "immutable ingest generation at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
