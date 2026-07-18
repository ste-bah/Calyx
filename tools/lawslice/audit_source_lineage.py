#!/usr/bin/env python3
"""Audit source→extract→ingest→physical-Base lineage without trusting returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from authoritative_documents import stable_row_sha256
from build_ingest_jsonl import (
    ALIASES_FILE,
    FULL_FILE,
    MANIFEST_FILE as INGEST_MANIFEST,
    alias_record,
    build_text,
    metadata,
    source_extract_binding,
    verify_ingest_generation,
)
from cuyahoga_contract import TEXT_FIELDS, sha256_text
from extract_cuyahoga import (
    MANIFEST_FILE as EXTRACT_MANIFEST,
    OPINION_FIELDS,
    RAW_FILE,
    REJECTIONS_FILE,
    csv_rows,
    verify_extract_generation,
)
from law_generation import generation_member, sha256_file, verify_generation
from structured_error import (
    StructuredArgumentParser,
    StructuredError,
    parse_cli_args,
    write_error,
)


KNOWN_ACCEPTED_OPINION = 4636687
KNOWN_REJECTED_OPINION = 4678958


class LineageAuditError(StructuredError):
    code = "source_lineage_audit_failed"


def fail(message: str, *, remediation: str, **context):
    raise LineageAuditError(message, remediation=remediation, **context)


def plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).absolute()
    if not path.is_file() or path.is_symlink():
        fail(
            "%s is not a plain file" % label,
            remediation="provide the sealed generation member or DB-native readback",
            path=str(path),
        )
    return path


def json_lines(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                fail(
                    "JSONL contains a blank row",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(
                    "JSONL row is invalid",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                    error=str(error),
                )
            if not isinstance(row, dict):
                fail(
                    "JSONL row is not an object",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                )
            yield line_number, row


def positive_id(value, *, where: str) -> int:
    if isinstance(value, bool):
        parsed = 0
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
    if parsed <= 0 or str(parsed) != str(value):
        fail(
            "opinion identity is not a canonical positive integer",
            remediation="rebuild from the source-bound extraction generation",
            where=where,
            value=value,
        )
    return parsed


def load_aliases(path: Path) -> dict[int, dict]:
    rows = {}
    for line_number, row in json_lines(path):
        opinion_id = positive_id(row.get("opinion_id"), where="alias:%d" % line_number)
        if opinion_id in rows:
            fail(
                "opinion alias is duplicated",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
            )
        rows[opinion_id] = row
    return rows


def load_batches(path: Path) -> dict[int, dict]:
    rows = {}
    for line_number, row in json_lines(path):
        text = row.get("text")
        metadata_map = row.get("metadata")
        if not isinstance(text, str) or not isinstance(metadata_map, dict):
            fail(
                "full ingest row lacks text or metadata",
                remediation="rebuild the canonical ingest generation",
                line=line_number,
            )
        opinion_id = positive_id(
            metadata_map.get("opinion_id"), where="full-ingest:%d" % line_number
        )
        if opinion_id in rows:
            fail(
                "full ingest opinion is duplicated",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
            )
        rows[opinion_id] = {
            "metadata": metadata_map,
            "text_sha256": sha256_text(text),
        }
    return rows


def load_base(path: Path) -> tuple[dict[int, dict], set[int]]:
    try:
        with open(path, "r", encoding="utf-8") as source:
            values = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        fail(
            "cannot read the physical Base CF readback",
            remediation="repeat an unbounded DB-native Base readback",
            path=str(path),
            error=str(error),
        )
    if not isinstance(values, list):
        fail(
            "physical Base CF readback is not an array",
            remediation="repeat an unbounded DB-native Base readback",
            path=str(path),
        )
    rows = {}
    panels = set()
    for index, row in enumerate(values, 1):
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            fail(
                "physical Base row lacks metadata",
                remediation="rebuild and promote the canonical vault",
                row=index,
            )
        opinion_id = positive_id(
            row["metadata"].get("opinion_id"), where="base:%d" % index
        )
        if opinion_id in rows:
            fail(
                "physical Base duplicates an opinion",
                remediation="rebuild and promote the canonical vault",
                opinion_id=opinion_id,
            )
        panel = row.get("panel_version")
        if not isinstance(panel, int) or isinstance(panel, bool) or panel <= 0:
            fail(
                "physical Base row has an invalid panel version",
                remediation="rebuild and promote the canonical vault",
                row=index,
                panel_version=panel,
            )
        rows[opinion_id] = {
            "cx_id": row.get("cx_id"),
            "metadata": row["metadata"],
        }
        panels.add(panel)
    return rows, panels


def selected_source_digest(row: dict) -> str:
    source_field = row["text_source"]
    if source_field == "authoritative_pdf_supplement":
        authoritative = row.get("authoritative_document_source")
        actual = (
            authoritative.get("raw_text_sha256")
            if isinstance(authoritative, dict)
            else None
        )
    else:
        selected_field = (
            "plain_text"
            if source_field == "plain_text_no_html_with_citations"
            else source_field
        )
        actual = row["text_provenance"]["fields"][selected_field]["raw_sha256"]
    declared = row["text_provenance"]["source_raw_sha256"]
    if actual != declared:
        fail(
            "selected source digest is not linked to its declared source bytes",
            remediation="rebuild from the bound bulk archive or PDF generation",
            opinion_id=row.get("opinion_id"),
            source_field=source_field,
            expected=actual,
            actual=declared,
        )
    return declared


def source_binding_sha256(values: dict[int, dict]) -> str:
    ordered = [values[opinion_id] for opinion_id in sorted(values)]
    return sha256_text(
        json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def expected_source_binding(opinion_id: int, expected: dict) -> dict:
    source_field = expected["text_source"]
    authoritative = expected["authoritative_document_source"]
    return {
        "opinion_id": opinion_id,
        "courtlistener_opinion_source": expected["courtlistener_opinion_source"],
        "field_facts": expected["field_facts"],
        "bulk_selected_raw_sha256": (
            None
            if source_field == "authoritative_pdf_supplement"
            else expected["source_raw_sha256"]
        ),
        "legacy_projected_source_row_sha256": (
            authoritative.get("legacy_projected_source_row_sha256")
            if isinstance(authoritative, dict)
            else None
        ),
    }


def extract_source_expectations(path: Path) -> dict[int, dict]:
    values = {}
    for line_number, row in json_lines(path):
        opinion_id = positive_id(
            row.get("opinion_id"), where="extract-source:%d" % line_number
        )
        if opinion_id in values:
            fail(
                "extract source identity is duplicated",
                remediation="rebuild the extraction generation",
                opinion_id=opinion_id,
            )
        values[opinion_id] = {
            "courtlistener_opinion_source": row["courtlistener_opinion_source"],
            "field_facts": row["text_provenance"]["fields"],
            "source_raw_sha256": row["text_provenance"]["source_raw_sha256"],
            "text_source": row["text_source"],
            "authoritative_document_source": row.get(
                "authoritative_document_source"
            ),
        }
    return values


def physical_source_audit(
    path: Path,
    *,
    expected_archive_sha256: str,
    expected_rows: dict[int, dict],
) -> dict:
    before = path.stat()
    first_archive_sha256 = sha256_file(path)
    if first_archive_sha256 != expected_archive_sha256:
        fail(
            "physical opinions archive differs from the extract binding",
            remediation="restore the exact source archive and rebuild from zero",
            expected=expected_archive_sha256,
            actual=first_archive_sha256,
        )
    found = set()
    scanned = 0
    selected_raw_rehashed = 0
    source_columns = None
    physical_bindings = {}
    for line, row in csv_rows(
        str(path), "search_opinion", OPINION_FIELDS, include_all=True
    ):
        scanned += 1
        if scanned % 500_000 == 0:
            print(
                "physical-source-audit: %d rows; %d/%d accepted identities"
                % (scanned, len(found), len(expected_rows)),
                file=sys.stderr,
                flush=True,
            )
        opinion_id = positive_id(row["id"], where="physical-source:%d" % line)
        expected = expected_rows.get(opinion_id)
        if expected is None:
            continue
        if opinion_id in found:
            fail(
                "physical opinions archive duplicates an accepted opinion",
                remediation="quarantine the source archive and reacquire it",
                opinion_id=opinion_id,
            )
        found.add(opinion_id)
        columns = sorted(row)
        if source_columns is None:
            source_columns = columns
        elif columns != source_columns:
            fail(
                "physical opinions archive changed column schema within the scan",
                remediation="quarantine the source archive and reacquire it",
                opinion_id=opinion_id,
            )
        row_digest = stable_row_sha256(row)
        declared_row_digest = expected["courtlistener_opinion_source"][
            "source_row_sha256"
        ]
        if row_digest != declared_row_digest:
            fail(
                "complete physical CourtListener row digest differs",
                remediation="quarantine the extract generation and rebuild from source",
                opinion_id=opinion_id,
                expected=row_digest,
                actual=declared_row_digest,
            )
        actual_source = {
            key: row[key] or None
            for key in (
                "sha1",
                "download_url",
                "local_path",
                "date_created",
                "date_modified",
            )
        }
        declared_source = {
            key: expected["courtlistener_opinion_source"][key]
            for key in actual_source
        }
        if actual_source != declared_source:
            fail(
                "physical CourtListener source fields differ",
                remediation="quarantine the extract generation and rebuild from source",
                opinion_id=opinion_id,
            )
        actual_field_facts = {
            field: {
                "raw_present": True,
                "nonempty": bool(row[field].strip()),
                "raw_chars": len(row[field]),
                "raw_sha256": sha256_text(row[field]) if row[field] else None,
            }
            for field in TEXT_FIELDS
        }
        if actual_field_facts != expected["field_facts"]:
            fail(
                "physical competing source-field facts differ",
                remediation="quarantine the extract generation and rebuild from source",
                opinion_id=opinion_id,
            )
        source_field = expected["text_source"]
        physical_binding = {
            "opinion_id": opinion_id,
            "courtlistener_opinion_source": {
                "source_row_sha256": row_digest,
                **actual_source,
            },
            "field_facts": actual_field_facts,
            "bulk_selected_raw_sha256": None,
            "legacy_projected_source_row_sha256": None,
        }
        if source_field == "authoritative_pdf_supplement":
            if any(row[field].strip() for field in TEXT_FIELDS):
                fail(
                    "PDF supplement replaced a nonempty physical bulk field",
                    remediation="quarantine the extract generation and review the exception",
                    opinion_id=opinion_id,
                )
            authoritative = expected["authoritative_document_source"]
            projected = {field: row[field] for field in OPINION_FIELDS}
            if (
                not isinstance(authoritative, dict)
                or authoritative.get("legacy_projected_source_row_sha256")
                != stable_row_sha256(projected)
                or authoritative.get("legacy_source_row_digest_fields")
                != list(OPINION_FIELDS)
            ):
                fail(
                    "PDF supplement legacy projection does not bind to source",
                    remediation="quarantine the extract generation and review the exception",
                    opinion_id=opinion_id,
                )
            physical_binding["legacy_projected_source_row_sha256"] = (
                stable_row_sha256(projected)
            )
        else:
            selected_field = (
                "plain_text"
                if source_field == "plain_text_no_html_with_citations"
                else source_field
            )
            actual_raw_sha256 = sha256_text(row[selected_field])
            if actual_raw_sha256 != expected["source_raw_sha256"]:
                fail(
                    "selected physical raw-field digest differs",
                    remediation="quarantine the extract generation and rebuild from source",
                    opinion_id=opinion_id,
                    source_field=selected_field,
                    expected=actual_raw_sha256,
                    actual=expected["source_raw_sha256"],
                )
            physical_binding["bulk_selected_raw_sha256"] = actual_raw_sha256
            selected_raw_rehashed += 1
        physical_bindings[opinion_id] = physical_binding
    if found != set(expected_rows):
        fail(
            "accepted opinions are missing from the physical archive",
            remediation="restore the exact source archive and rebuild from zero",
            missing=sorted(set(expected_rows) - found)[:20],
        )
    expected_bindings = {
        opinion_id: expected_source_binding(opinion_id, expected)
        for opinion_id, expected in expected_rows.items()
    }
    physical_binding_sha256 = source_binding_sha256(physical_bindings)
    expected_binding_sha256 = source_binding_sha256(expected_bindings)
    if physical_binding_sha256 != expected_binding_sha256:
        fail(
            "physical source aggregate differs from the extract aggregate",
            remediation="quarantine the extract generation and rebuild from source",
            physical=physical_binding_sha256,
            expected=expected_binding_sha256,
        )
    after = path.stat()
    second_archive_sha256 = sha256_file(path)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_after != identity_before
        or second_archive_sha256 != first_archive_sha256
    ):
        fail(
            "physical opinions archive changed during the independent audit",
            remediation="stop the mutator, restore the exact archive, and repeat the audit",
            identity_before=identity_before,
            identity_after=identity_after,
            digest_before=first_archive_sha256,
            digest_after=second_archive_sha256,
        )
    return {
        "status": "verified",
        "archive_sha256": second_archive_sha256,
        "rows_scanned": scanned,
        "accepted_rows_rehashed": len(found),
        "source_columns": source_columns,
        "source_column_count": len(source_columns or []),
        "source_row_digest_scope": "all physical CSV columns",
        "selected_raw_fields_rehashed": selected_raw_rehashed,
        "pdf_supplement_rows": len(found) - selected_raw_rehashed,
        "accepted_source_binding_sha256": physical_binding_sha256,
    }


def source_only_audit(args) -> dict:
    extract_report = verify_extract_generation(args.extract_generation)
    extract_rows = plain_file(
        generation_member(args.extract_generation, EXTRACT_MANIFEST, RAW_FILE),
        label="extract rows",
    )
    archive = plain_file(args.opinions_archive, label="physical opinions archive")
    expectations = extract_source_expectations(extract_rows)
    if len(expectations) != extract_report["accepted_rows"]:
        fail(
            "extract report and source identity count differ",
            remediation="rebuild the extraction generation",
            report=extract_report["accepted_rows"],
            actual=len(expectations),
        )
    extract_manifest = verify_generation(
        Path(args.extract_generation).absolute(), EXTRACT_MANIFEST
    )
    expected_archive = extract_manifest["source_archives"]["opinions"][
        "archive_sha256"
    ]
    return {
        "status": "verified",
        "mode": "physical-source-only",
        "audit_implementation_sha256": sha256_file(Path(__file__).absolute()),
        "extract_manifest_sha256": extract_report["manifest_sha256"],
        "expected_source_binding_sha256": source_binding_sha256(
            {
                opinion_id: expected_source_binding(opinion_id, expected)
                for opinion_id, expected in expectations.items()
            }
        ),
        "physical_source": physical_source_audit(
            archive,
            expected_archive_sha256=expected_archive,
            expected_rows=expectations,
        ),
        "sources": {
            "extract_rows": {
                "path": str(extract_rows),
                "bytes": extract_rows.stat().st_size,
                "sha256": sha256_file(extract_rows),
            },
            "opinions_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": expected_archive,
            },
        },
    }


def read_physical_source_report(
    path: Path,
    *,
    extract_manifest_sha256: str,
    expected_archive_sha256: str,
    expected_rows: dict[int, dict],
    archive_path: Path,
) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as source:
            report = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        fail(
            "cannot read the physical source audit report",
            remediation="repeat the exhaustive physical source audit",
            path=str(path),
            error=str(error),
        )
    if not isinstance(report, dict):
        fail(
            "physical source audit report is not an object",
            remediation="repeat the exhaustive physical source audit",
            path=str(path),
        )
    expected_binding = source_binding_sha256(
        {
            opinion_id: expected_source_binding(opinion_id, expected)
            for opinion_id, expected in expected_rows.items()
        }
    )
    physical = report.get("physical_source")
    if (
        report.get("status") != "verified"
        or report.get("mode") != "physical-source-only"
        or report.get("audit_implementation_sha256")
        != sha256_file(Path(__file__).absolute())
        or report.get("extract_manifest_sha256") != extract_manifest_sha256
        or report.get("expected_source_binding_sha256") != expected_binding
        or not isinstance(physical, dict)
        or physical.get("status") != "verified"
        or physical.get("archive_sha256") != expected_archive_sha256
        or physical.get("accepted_rows_rehashed") != len(expected_rows)
        or physical.get("accepted_source_binding_sha256") != expected_binding
        or physical.get("source_row_digest_scope") != "all physical CSV columns"
    ):
        fail(
            "physical source audit report differs from the current contract",
            remediation="repeat the exhaustive physical source audit with current code",
            path=str(path),
        )
    current_archive_sha256 = sha256_file(archive_path)
    if current_archive_sha256 != expected_archive_sha256:
        fail(
            "physical source archive changed after its exhaustive audit",
            remediation="restore the exact archive and repeat the exhaustive audit",
            expected=expected_archive_sha256,
            actual=current_archive_sha256,
        )
    return physical


def audit(args) -> dict:
    extract_report = verify_extract_generation(args.extract_generation)
    ingest_report = verify_ingest_generation(
        args.ingest_generation, args.extract_generation
    )
    extract_manifest = verify_generation(
        Path(args.extract_generation).absolute(), EXTRACT_MANIFEST
    )
    ingest_manifest = verify_generation(
        Path(args.ingest_generation).absolute(), INGEST_MANIFEST
    )
    source_extract = source_extract_binding(args.extract_generation, extract_report)
    if ingest_manifest.get("source_extract_generation") != source_extract:
        fail(
            "ingest manifest is not bound to the physical extract generation",
            remediation="rebuild the canonical ingest generation",
        )

    paths = {
        "extract_rows": plain_file(
            generation_member(args.extract_generation, EXTRACT_MANIFEST, RAW_FILE),
            label="extract rows",
        ),
        "rejections": plain_file(
            generation_member(
                args.extract_generation, EXTRACT_MANIFEST, REJECTIONS_FILE
            ),
            label="selection rejections",
        ),
        "ingest_aliases": plain_file(
            generation_member(args.ingest_generation, INGEST_MANIFEST, ALIASES_FILE),
            label="ingest aliases",
        ),
        "full_ingest": plain_file(
            generation_member(args.ingest_generation, INGEST_MANIFEST, FULL_FILE),
            label="full ingest",
        ),
        "base_readback": plain_file(args.base_readback, label="Base CF readback"),
        "opinions_archive": plain_file(
            args.opinions_archive, label="physical opinions archive"
        ),
        "regression_manifest": plain_file(
            args.regression_manifest, label="real regression manifest"
        ),
        "regression_records": plain_file(
            args.regression_records, label="real regression records"
        ),
    }
    if args.physical_source_report:
        paths["physical_source_report"] = plain_file(
            args.physical_source_report, label="physical source audit report"
        )
    with open(paths["regression_manifest"], "r", encoding="utf-8") as source:
        regression_manifest = json.load(source)
    with open(paths["regression_records"], "r", encoding="utf-8") as source:
        regression_records = json.load(source)
    expected_regression = regression_manifest.get("files", {}).get(
        paths["regression_records"].name
    )
    if (
        not isinstance(expected_regression, dict)
        or expected_regression.get("bytes") != paths["regression_records"].stat().st_size
        or expected_regression.get("sha256")
        != sha256_file(paths["regression_records"])
        or regression_manifest.get("source_archive", {}).get("sha256")
        != source_extract["source_archive_sha256"]["opinions"]
        or regression_records.get("source_archive", {}).get("sha256")
        != source_extract["source_archive_sha256"]["opinions"]
    ):
        fail(
            "real regression capture is not bound to the opinion source archive",
            remediation="recapture deterministic rows from the exact bound archive",
        )
    regression_cases = {
        row.get("opinion_id"): row for row in regression_records.get("cases", [])
    }
    known_regression = regression_cases.get(KNOWN_ACCEPTED_OPINION)
    if not isinstance(known_regression, dict):
        fail(
            "known accepted row is absent from the real regression capture",
            remediation="recapture the deterministic accepted specimen",
            opinion_id=KNOWN_ACCEPTED_OPINION,
        )
    aliases = load_aliases(paths["ingest_aliases"])
    batches = load_batches(paths["full_ingest"])
    base, panel_versions = load_base(paths["base_readback"])
    if set(batches) != set(base):
        fail(
            "full ingest and physical Base identity sets differ",
            remediation="rebuild and promote the canonical vault",
            ingest_only=sorted(set(batches) - set(base))[:20],
            base_only=sorted(set(base) - set(batches))[:20],
        )
    for opinion_id, batch in batches.items():
        if base[opinion_id]["metadata"] != batch["metadata"]:
            fail(
                "physical Base metadata differs from full ingest bytes",
                remediation="rebuild and promote the canonical vault",
                opinion_id=opinion_id,
            )

    source_ids = set()
    source_expectations = {}
    counts = {
        "source_rows": 0,
        "canonical_rows": 0,
        "duplicate_aliases": 0,
        "source_digest_links": 0,
        "normalized_digest_links": 0,
        "composed_digest_links": 0,
        "alias_lineage_exact": 0,
        "case_name_full_nonempty": 0,
        "case_name_full_alias_exact": 0,
        "case_name_full_base_exact": 0,
    }
    known = None
    for line_number, row in json_lines(paths["extract_rows"]):
        opinion_id = positive_id(
            row.get("opinion_id"), where="extract:%d" % line_number
        )
        if opinion_id in source_ids:
            fail(
                "extract opinion is duplicated",
                remediation="rebuild the extraction generation",
                opinion_id=opinion_id,
            )
        source_ids.add(opinion_id)
        source_expectations[opinion_id] = {
            "courtlistener_opinion_source": row["courtlistener_opinion_source"],
            "field_facts": row["text_provenance"]["fields"],
            "source_raw_sha256": row["text_provenance"]["source_raw_sha256"],
            "text_source": row["text_source"],
            "authoritative_document_source": row.get(
                "authoritative_document_source"
            ),
        }
        alias = aliases.get(opinion_id)
        if alias is None:
            fail(
                "extract opinion has no ingest alias",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
            )
        canonical_id = positive_id(
            alias.get("canonical_opinion_id"), where="alias:%d" % opinion_id
        )
        batch = batches.get(canonical_id)
        if batch is None:
            fail(
                "opinion alias has no canonical constellation",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
                canonical_opinion_id=canonical_id,
            )
        source_digest = selected_source_digest(row)
        counts["source_digest_links"] += 1
        normalized_digest = sha256_text(row["text"])
        if normalized_digest != row["text_provenance"]["normalized_sha256"]:
            fail(
                "normalized text digest differs",
                remediation="rebuild the extraction generation",
                opinion_id=opinion_id,
            )
        counts["normalized_digest_links"] += 1
        composed_digest = sha256_text(build_text(row, line=line_number))
        if (
            alias.get("content_sha256") != composed_digest
            or batch["text_sha256"] != composed_digest
        ):
            fail(
                "composed ingest digest differs",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
            )
        counts["composed_digest_links"] += 1
        expected_alias = alias_record(
            row,
            canonical_opinion_id=canonical_id,
            content_sha256=composed_digest,
            source_extract=source_extract,
        )
        if alias != expected_alias:
            fail(
                "ingest alias differs from exact source lineage",
                remediation="rebuild the canonical ingest generation",
                opinion_id=opinion_id,
            )
        counts["alias_lineage_exact"] += 1
        if row.get("case_name_full"):
            counts["case_name_full_nonempty"] += 1
            if alias["source_lineage"]["case_name_full"] != row["case_name_full"]:
                fail(
                    "case_name_full was lost from source alias lineage",
                    remediation="rebuild the canonical ingest generation",
                    opinion_id=opinion_id,
                )
            counts["case_name_full_alias_exact"] += 1
            if batch["metadata"].get("case_name_full") != row["case_name_full"]:
                fail(
                    "case_name_full differs at the canonical constellation",
                    remediation="do not canonicalize records with differing legal fields",
                    opinion_id=opinion_id,
                    canonical_opinion_id=canonical_id,
                )
            counts["case_name_full_base_exact"] += 1
        if opinion_id == canonical_id:
            expected_metadata = metadata(
                row,
                canonical_opinion_id=canonical_id,
                ingest_text_sha256=composed_digest,
                dataset_tag=ingest_manifest["dataset_tag"],
                source_extract=source_extract,
            )
            if batch["metadata"] != expected_metadata:
                fail(
                    "canonical metadata differs from source lineage",
                    remediation="rebuild the canonical ingest generation",
                    opinion_id=opinion_id,
                )
            counts["canonical_rows"] += 1
        else:
            counts["duplicate_aliases"] += 1
        counts["source_rows"] += 1
        if opinion_id == KNOWN_ACCEPTED_OPINION:
            if (
                source_digest != known_regression.get("full_source_raw_sha256")
                or normalized_digest
                != known_regression.get("full_normalized_sha256")
            ):
                fail(
                    "known accepted row differs from the source-bound real capture",
                    remediation="quarantine the generation and inspect source drift",
                    opinion_id=opinion_id,
                )
            known = {
                "opinion_id": opinion_id,
                "canonical_opinion_id": canonical_id,
                "cx_id": base[canonical_id]["cx_id"],
                "cluster_slug": row["cluster_slug"],
                "source_field": row["text_source"],
                "source_sha256": source_digest,
                "source_row_sha256": row["courtlistener_opinion_source"][
                    "source_row_sha256"
                ],
                "normalized_text_sha256": normalized_digest,
                "ingest_text_sha256": composed_digest,
                "source_snapshot_date": row["source_snapshot_date"],
                "retrieval_ts": batch["metadata"]["retrieval_ts"],
                "source_url": row["canonical_source_url"],
            }

    if source_ids != set(aliases):
        fail(
            "extract and alias identity sets differ",
            remediation="rebuild the canonical ingest generation",
            extract_only=sorted(source_ids - set(aliases))[:20],
            alias_only=sorted(set(aliases) - source_ids)[:20],
        )
    if known is None:
        fail(
            "known accepted lineage specimen is absent",
            remediation="inspect the selection partitions before promotion",
            opinion_id=KNOWN_ACCEPTED_OPINION,
        )

    expected_opinions_archive = source_extract["source_archive_sha256"][
        "opinions"
    ]
    if args.physical_source_report:
        physical_source = read_physical_source_report(
            paths["physical_source_report"],
            extract_manifest_sha256=source_extract["manifest_sha256"],
            expected_archive_sha256=expected_opinions_archive,
            expected_rows=source_expectations,
            archive_path=paths["opinions_archive"],
        )
    else:
        physical_source = physical_source_audit(
            paths["opinions_archive"],
            expected_archive_sha256=expected_opinions_archive,
            expected_rows=source_expectations,
        )

    rejected = None
    for _, row in json_lines(paths["rejections"]):
        if row.get("opinion_id") == KNOWN_REJECTED_OPINION:
            selection = row.get("selection")
            rejected = {
                "opinion_id": KNOWN_REJECTED_OPINION,
                "selection": selection,
                "present_in_extract": KNOWN_REJECTED_OPINION in source_ids,
                "present_in_aliases": KNOWN_REJECTED_OPINION in aliases,
                "present_in_base": KNOWN_REJECTED_OPINION in base,
            }
            break
    if (
        rejected is None
        or not isinstance(rejected["selection"], dict)
        or rejected["selection"].get("status") != "rejected"
        or rejected["selection"].get("reason") != "explicit_non_eighth_district"
        or any(
            rejected[key]
            for key in ("present_in_extract", "present_in_aliases", "present_in_base")
        )
    ):
        fail(
            "known false-positive opinion is not a typed rejection",
            remediation="rebuild from direct issuing-court evidence",
            opinion_id=KNOWN_REJECTED_OPINION,
            observed=rejected,
        )

    return {
        "status": "verified",
        "extract": extract_report,
        "ingest": ingest_report,
        "counts": counts,
        "panel_versions": sorted(panel_versions),
        "source_binding": {
            "normalized_schema_version": extract_manifest.get(
                "normalized_schema_version"
            ),
            "extract_manifest_sha256": source_extract["manifest_sha256"],
            "extractor_config_sha256": source_extract["producer_config_sha256"],
            "archive_sha256": source_extract["source_archive_sha256"],
        },
        "known_accepted": known,
        "known_rejected": rejected,
        "physical_source": physical_source,
        "regression_binding": {
            "opinion_id": KNOWN_ACCEPTED_OPINION,
            "source_capture_manifest_sha256": regression_manifest.get(
                "source_capture_manifest_sha256"
            ),
            "source_archive_sha256": regression_manifest["source_archive"]["sha256"],
            "historical_projected_source_row_sha256": known_regression[
                "source_row_sha256"
            ],
        },
        "sources": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": (
                    physical_source["archive_sha256"]
                    if key == "opinions_archive"
                    else sha256_file(path)
                ),
            }
            for key, path in paths.items()
        },
    }


def parser() -> argparse.ArgumentParser:
    root = StructuredArgumentParser(description=__doc__)
    root.add_argument("--source-only", action="store_true")
    root.add_argument("--extract-generation", required=True)
    root.add_argument("--ingest-generation")
    root.add_argument("--base-readback")
    root.add_argument("--opinions-archive", required=True)
    root.add_argument("--physical-source-report")
    root.add_argument("--regression-manifest")
    root.add_argument("--regression-records")
    return root


def main() -> int:
    try:
        args = parse_cli_args(parser())
        if args.source_only:
            if any(
                value
                for value in (
                    args.ingest_generation,
                    args.base_readback,
                    args.physical_source_report,
                    args.regression_manifest,
                    args.regression_records,
                )
            ):
                fail(
                    "source-only audit received downstream arguments",
                    remediation="pass only extract-generation and opinions-archive",
                )
            result = source_only_audit(args)
        else:
            missing = [
                name
                for name in (
                    "ingest_generation",
                    "base_readback",
                    "regression_manifest",
                    "regression_records",
                )
                if not getattr(args, name)
            ]
            if missing:
                fail(
                    "full lineage audit lacks required arguments",
                    remediation="provide the complete ingest/Base/regression evidence set",
                    missing=missing,
                )
            result = audit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except LineageAuditError as error:
        write_error(
            error,
            code="source_lineage_audit_failed",
            remediation="repair the reported lineage authority before promotion",
            include_traceback=False,
        )
        return 1
    except Exception as error:
        write_error(
            error,
            code="source_lineage_audit_unhandled",
            remediation="inspect the traceback and add a typed fail-closed audit path",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
