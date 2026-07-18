#!/usr/bin/env python3
"""Build one immutable, provenance-complete Cuyahoga CourtListener generation.

This command deliberately replaces the former independently published pass1,
pass2, and pass3 files. A build scans the three bound bulk archives, classifies
generic Ohio appellate opinions from direct issuing-court evidence, validates
authoritative text, applies source-bound corrections, independently verifies
every staged artifact, and exposes the entire directory with one rename.
"""

from __future__ import annotations

import argparse
import bz2
import csv
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from authoritative_documents import (
    load_authoritative_generation,
    record_provenance as authoritative_record_provenance,
    resolve_pdf_record,
    stable_row_sha256,
)
from cuyahoga_contract import (
    AuthoritativeTextError,
    CAP_COURTS,
    ContractError,
    CORRECTION_POLICY_VERSION,
    CorrectionError,
    EvidenceConflictError,
    GENERIC_OHIO_APPEALS_COURT,
    ResolvedText,
    SELECTOR_VERSION,
    TEXT_FIELDS,
    TEXT_POLICY_VERSION,
    apply_correction,
    classify,
    correction_manifest_sha256,
    direct_evidence,
    load_corrections,
    resolve_text,
    sha256_text,
)
from law_generation import (
    GenerationPublisher,
    sha256_file,
    verify_generation,
)
from signal_audit import (
    ERROR_CODE as EXTERNAL_SIGNAL_CODE,
    ExternalSignal,
    install_signal_audit,
    update_signal_progress,
)
from source_scan_lock import SourceScanLock, SourceScanLockError, make_source_identity
from structured_error import (
    StructuredArgumentParser,
    StructuredError,
    parse_cli_args,
    write_error,
)


LEGACY_FORMAT = "calyx-cuyahoga-extract-generation-v3"
PREVIOUS_FORMAT = "calyx-cuyahoga-extract-generation-v4"
FORMAT = "calyx-cuyahoga-extract-generation-v5"
SOURCE_ROW_FORMATS = frozenset({PREVIOUS_FORMAT, FORMAT})
MANIFEST_FILE = "extract_manifest.json"
RAW_FILE = "opinions_cuyahoga.raw.jsonl"
IDMAP_FILE = "idmap.csv"
DOCKETS_FILE = "dockets_map.jsonl"
CLUSTERS_FILE = "clusters_map.jsonl"
REJECTIONS_FILE = "selection_rejections.jsonl"
CONFLICTS_FILE = "selection_conflicts.jsonl"
COUNTS_FILE = "selection_counts.json"
CANDIDATE_CLUSTER_ACCOUNTING_FILE = "candidate_cluster_accounting.jsonl"
CANDIDATE_SPOOL_FORMAT = "calyx-cuyahoga-resolved-candidate-spool-v3"
LEGACY_CANDIDATE_SPOOL_FORMAT = "calyx-cuyahoga-resolved-candidate-spool-v2"
PROGRESS_EVERY = 500_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def producer_contract() -> dict:
    """Fingerprint the exact producer code and its interpretation contract."""
    root = Path(__file__).resolve().parent
    members = {}
    for name in (
        "authoritative_documents.py",
        "cuyahoga_contract.py",
        "extract_cuyahoga.py",
        "law_generation.py",
        "signal_audit.py",
        "source_scan_lock.py",
    ):
        path = root / name
        if not path.is_file() or path.is_symlink():
            fail("extractor producer source is not a plain file", path=str(path))
        members[name] = sha256_file(path)
    config = {
        "format": FORMAT,
        "selector_version": SELECTOR_VERSION,
        "text_policy_version": TEXT_POLICY_VERSION,
        "correction_policy_version": CORRECTION_POLICY_VERSION,
        "candidate_spool_format": CANDIDATE_SPOOL_FORMAT,
        "text_fields": list(TEXT_FIELDS),
        "source_row_digest": "all physical CSV columns as a canonical JSON object",
        "authoritative_pdf_v1_row_digest": (
            "explicitly labeled legacy projection over OPINION_FIELDS"
        ),
        "candidate_cluster_accounting": (
            "one row per candidate cluster with explicit opinion partitions or no_opinion"
        ),
    }
    return {
        "implementation_sha256": members,
        "config": config,
        "config_sha256": sha256_text(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    }


class ExtractionError(StructuredError):
    code = "cuyahoga_extraction_error"
    default_remediation = (
        "inspect the recorded physical source or generation mismatch, then rebuild "
        "the complete bound generation at a new destination"
    )


def fail(message: str, **context):
    raise ExtractionError(message, **context)


def authoritative_error_identity(record: dict | None) -> dict | None:
    """Return the causal identity of an authoritative-text error.

    ``context.where`` is an observation location, not part of the source
    decision. Archive scanning and sealed-spool replay intentionally observe
    the same row at different paths, so comparing that label would turn a
    successful independent replay into a false source-drift failure.
    """
    if record is None:
        return None
    if not isinstance(record, dict) or frozenset(record) not in {
        frozenset({"code", "message", "context"}),
        frozenset({"code", "message", "remediation", "context"}),
    }:
        fail("authoritative text error record schema mismatch", record=record)
    context = record["context"]
    if not isinstance(context, dict):
        fail("authoritative text error context is not an object", record=record)
    return {
        "code": record["code"],
        "message": record["message"],
        "remediation": record.get(
            "remediation", AuthoritativeTextError.default_remediation
        ),
        "context": {key: value for key, value in context.items() if key != "where"},
    }


def flush_and_sync(handle) -> None:
    """Make a completed spool visible to a separate physical file read."""
    handle.flush()
    os.fsync(handle.fileno())


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


class Progress:
    def __init__(self, label: str):
        self.label = label
        self.rows = 0
        self.started = time.monotonic()
        update_signal_progress(label, 0)

    def tick(self) -> None:
        self.rows += 1
        if self.rows % PROGRESS_EVERY == 0:
            update_signal_progress(self.label, self.rows)
            elapsed = time.monotonic() - self.started
            rate = self.rows / elapsed if elapsed else 0.0
            sys.stderr.write(
                "[%s] %d rows, %.0f rows/s, %.1f min\n"
                % (self.label, self.rows, rate, elapsed / 60.0)
            )
            sys.stderr.flush()

    def done(self) -> None:
        update_signal_progress(self.label, self.rows)
        elapsed = time.monotonic() - self.started
        rate = self.rows / elapsed if elapsed else 0.0
        sys.stderr.write(
            "[%s] DONE %d rows, %.0f rows/s, %.1f min\n"
            % (self.label, self.rows, rate, elapsed / 60.0)
        )
        sys.stderr.flush()


def open_text(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def csv_rows(
    path: str,
    table: str,
    required: tuple[str, ...] | list[str],
    *,
    include_all: bool = False,
):
    # csv_rows is also imported by real-source capture/audit tools. Keep the
    # large-field invariant at the parser boundary instead of relying on a CLI
    # entry point to initialize process-global csv state.
    _set_csv_field_limit()
    with open_text(path) as source:
        reader = csv.reader(
            source,
            quotechar='"',
            escapechar="\\",
            doublequote=False,
            strict=True,
        )
        try:
            header = next(reader)
        except StopIteration:
            fail("bulk CSV is empty", path=path, table=table)
        if len(header) != len(set(header)):
            fail("bulk CSV has duplicate columns", path=path, table=table)
        missing = [field for field in required if field not in header]
        if missing:
            fail(
                "bulk CSV is missing required columns",
                path=path,
                table=table,
                missing=missing,
            )
        selected = header if include_all else required
        index = {field: header.index(field) for field in selected}
        while True:
            try:
                row = next(reader)
            except StopIteration:
                return
            except csv.Error as error:
                fail(
                    "bulk CSV parse error",
                    path=path,
                    table=table,
                    physical_line=reader.line_num,
                    error=str(error),
                )
            if len(row) != len(header):
                fail(
                    "bulk CSV column-count mismatch",
                    path=path,
                    table=table,
                    physical_line=reader.line_num,
                    expected=len(header),
                    actual=len(row),
                )
            yield (
                reader.line_num,
                {field: row[position] for field, position in index.items()},
            )


def positive_int(value: str, *, field: str, path: str, line: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail(
            "required ID is not decimal", path=path, line=line, field=field, value=value
        )
    if parsed <= 0:
        fail(
            "required ID is not positive",
            path=path,
            line=line,
            field=field,
            value=value,
        )
    return parsed


def optional_int(value: str, *, field: str, path: str, line: int) -> int | None:
    if value == "":
        return None
    return positive_int(value, field=field, path=path, line=line)


def nonnegative_int(value: str, *, field: str, path: str, line: int) -> int:
    if value == "":
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail("field is not decimal", path=path, line=line, field=field, value=value)
    if parsed < 0:
        fail("field is negative", path=path, line=line, field=field, value=value)
    return parsed


_HASH_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def prepare_bulk_sources(args) -> dict:
    """Read the small checksum manifest and derive the pre-I/O lock identity."""
    try:
        date.fromisoformat(args.snapshot_date)
    except ValueError as error:
        fail(
            "snapshot date must be ISO YYYY-MM-DD",
            value=args.snapshot_date,
            error=str(error),
        )
    try:
        acquired = datetime.fromisoformat(args.acquired_at.replace("Z", "+00:00"))
    except ValueError as error:
        fail(
            "acquired-at must be an ISO timestamp",
            value=args.acquired_at,
            error=str(error),
        )
    if acquired.tzinfo is None:
        fail("acquired-at must include an explicit UTC offset", value=args.acquired_at)
    manifest_path = Path(args.bulk_manifest).absolute()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("bulk SHA-256 manifest is not a plain file", path=str(manifest_path))
    entries = {}
    with open(manifest_path, "r", encoding="utf-8") as source:
        for lineno, raw in enumerate(source, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            match = _HASH_LINE.fullmatch(line)
            if not match:
                fail(
                    "invalid SHA-256 manifest line",
                    path=str(manifest_path),
                    line=lineno,
                )
            digest, name = match.groups()
            if name in entries:
                fail(
                    "duplicate SHA-256 manifest member",
                    path=str(manifest_path),
                    member=name,
                )
            entries[name] = digest
    if not entries:
        fail("bulk SHA-256 manifest is empty", path=str(manifest_path))

    roles = {
        "dockets": Path(args.dockets).absolute(),
        "clusters": Path(args.clusters).absolute(),
        "opinions": Path(args.opinions).absolute(),
    }
    identity_sources = {}
    expected_by_role = {}
    for role, path in roles.items():
        expected = entries.get(path.name)
        if expected is None:
            fail(
                "bulk input basename is absent from SHA-256 manifest",
                role=role,
                path=str(path),
                manifest=str(manifest_path),
            )
        expected_by_role[role] = expected
        identity_sources[role] = {
            "path": str(Path(os.path.realpath(path))),
            "sha256": expected,
        }
    identity = make_source_identity(
        identity_sources,
        selector_version=SELECTOR_VERSION,
        text_policy_version=TEXT_POLICY_VERSION,
    )
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "roles": roles,
        "physical_sources": {
            **{role: str(path) for role, path in roles.items()},
            "manifest": str(manifest_path),
        },
        "expected_by_role": expected_by_role,
        "identity": identity,
    }


def load_and_verify_bulk_sources(args, prepared: dict | None = None) -> dict:
    prepared = prepared or prepare_bulk_sources(args)
    manifest_path = prepared["manifest_path"]
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != prepared["manifest_sha256"]:
        fail(
            "bulk SHA-256 manifest changed after source lock identity was derived",
            path=str(manifest_path),
            expected=prepared["manifest_sha256"],
            actual=manifest_sha256,
        )
    roles = prepared["roles"]
    provenance = {}
    for role, path in roles.items():
        if not path.is_file() or path.is_symlink():
            fail("bulk input is not a plain file", role=role, path=str(path))
        resolved = str(path.resolve(strict=True))
        identity_path = prepared["identity"]["sources"][role]["path"]
        if resolved != identity_path:
            fail(
                "bulk input path changed after source lock identity was derived",
                role=role,
                path=str(path),
                expected_resolved_path=identity_path,
                actual_resolved_path=resolved,
            )
        expected = prepared["expected_by_role"][role]
        sys.stderr.write("source-hash: verifying %s\n" % path)
        actual = sha256_file(path)
        if actual != expected:
            fail(
                "bulk input SHA-256 mismatch",
                role=role,
                path=str(path),
                expected=expected,
                actual=actual,
            )
        provenance[role] = {
            "archive_name": path.name,
            "archive_sha256": actual,
            "snapshot_date": args.snapshot_date,
            "acquired_at": args.acquired_at,
        }
    provenance["manifest"] = {
        "name": manifest_path.name,
        "sha256": manifest_sha256,
    }
    return provenance


DOCKET_FIELDS = (
    "id",
    "court_id",
    "docket_number",
    "case_name",
    "appeal_from_str",
    "assigned_to_str",
    "date_filed",
)


def scan_dockets(path: str, output) -> tuple[dict, dict]:
    dockets = {}
    counts = {}
    progress = Progress("dockets")
    for line, row in csv_rows(path, "search_docket", DOCKET_FIELDS):
        progress.tick()
        court = row["court_id"]
        if court not in CAP_COURTS and court != GENERIC_OHIO_APPEALS_COURT:
            continue
        docket_id = positive_int(row["id"], field="id", path=path, line=line)
        if docket_id in dockets:
            fail(
                "duplicate candidate docket ID",
                path=path,
                line=line,
                docket_id=docket_id,
            )
        record = {
            "docket_id": docket_id,
            "court_id": court,
            "docket_number": row["docket_number"],
            "case_name": row["case_name"],
            "appeal_from_str": row["appeal_from_str"],
            "assigned_to_str": row["assigned_to_str"],
            "date_filed": row["date_filed"],
            "source_field_sha256": {
                field: sha256_text(row[field])
                for field in DOCKET_FIELDS
                if field != "id"
            },
        }
        dockets[docket_id] = record
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        counts[court] = counts.get(court, 0) + 1
    progress.done()
    if not dockets:
        fail("no candidate dockets found", path=path)
    return dockets, {
        "rows_scanned": progress.rows,
        "candidate_rows": len(dockets),
        "by_court": counts,
    }


CLUSTER_FIELDS = (
    "id",
    "docket_id",
    "slug",
    "case_name",
    "case_name_full",
    "date_filed",
    "judges",
    "disposition",
    "posture",
    "nature_of_suit",
    "syllabus",
    "precedential_status",
    "citation_count",
    "source",
)
_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def scan_clusters(
    path: str, dockets: dict, corrections: dict, output
) -> tuple[dict, dict, set]:
    clusters = {}
    counts = {}
    applied = set()
    progress = Progress("clusters")
    for line, row in csv_rows(path, "search_opinioncluster", CLUSTER_FIELDS):
        progress.tick()
        if not row["docket_id"]:
            continue
        docket_id = positive_int(
            row["docket_id"], field="docket_id", path=path, line=line
        )
        docket = dockets.get(docket_id)
        if docket is None:
            continue
        cluster_id = positive_int(row["id"], field="id", path=path, line=line)
        if cluster_id in clusters:
            fail(
                "duplicate candidate cluster ID",
                path=path,
                line=line,
                cluster_id=cluster_id,
            )
        slug = row["slug"]
        if not _SLUG_RE.fullmatch(slug):
            fail(
                "candidate cluster has invalid canonical slug",
                path=path,
                line=line,
                cluster_id=cluster_id,
                slug=slug,
            )
        record = {
            "cluster_id": cluster_id,
            "docket_id": docket_id,
            "cluster_slug": slug,
            "court_id": docket["court_id"],
            "docket_number": docket["docket_number"],
            "appeal_from_str": docket["appeal_from_str"],
            "case_name": row["case_name"] or docket["case_name"],
            "case_name_full": row["case_name_full"],
            "date_filed": row["date_filed"],
            "judges": row["judges"],
            "disposition": row["disposition"],
            "posture": row["posture"],
            "nature_of_suit": row["nature_of_suit"],
            "syllabus": row["syllabus"],
            "precedential_status": row["precedential_status"],
            "citation_count": nonnegative_int(
                row["citation_count"],
                field="citation_count",
                path=path,
                line=line,
            ),
            "source": row["source"],
            "source_field_sha256": {
                field: sha256_text(row[field])
                for field in CLUSTER_FIELDS
                if field not in {"id", "docket_id"}
            },
        }
        correction = corrections["by_docket"].get(docket_id)
        correction_record = apply_correction(
            record, correction, where="%s line %d" % (path, line)
        )
        if correction_record is not None:
            record["correction"] = correction_record
            applied.add(correction["correction_id"])
        else:
            record["correction"] = None
        clusters[cluster_id] = record
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        court = record["court_id"]
        counts[court] = counts.get(court, 0) + 1
    progress.done()
    if not clusters:
        fail("no candidate clusters found", path=path)
    expected = {
        entry["correction_id"] for entry in corrections["manifest"]["corrections"]
    }
    if applied != expected:
        raise CorrectionError(
            "not every correction matched exactly one source cluster",
            expected=sorted(expected),
            applied=sorted(applied),
            missing=sorted(expected - applied),
        )
    return (
        clusters,
        {
            "rows_scanned": progress.rows,
            "candidate_rows": len(clusters),
            "by_court": counts,
            "corrections_applied": len(applied),
        },
        applied,
    )


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
OPINION_META_FIELDS = tuple(
    field for field in OPINION_FIELDS if field not in TEXT_FIELDS
)
SPOOL_OPINION_FIELDS = OPINION_META_FIELDS + ("source_row_sha256",)
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def opinion_identity(row: dict, *, path: str, line: int) -> tuple[int, int]:
    return (
        positive_int(row["id"], field="id", path=path, line=line),
        positive_int(row["cluster_id"], field="cluster_id", path=path, line=line),
    )


def resolve_authoritative_text(
    row: dict,
    *,
    cluster_id: int,
    where: str,
    authoritative_documents: dict,
):
    """Resolve bulk text, or the one exact source-bound PDF exception.

    A PDF is never considered when a normal source field succeeds. If bulk
    resolution fails, only a record keyed to the exact opinion, cluster, the
    immutable v1 projected-row digest, and source fields can be used. The v4
    output separately binds the complete physical CSV row.
    """
    try:
        return resolve_text(row, where=where), None, None
    except AuthoritativeTextError as source_error:
        try:
            opinion_id = int(row["id"])
        except (KeyError, TypeError, ValueError) as error:
            fail(
                "cannot identify bulk row while resolving authoritative text",
                where=where,
                error=str(error),
            )
        record = authoritative_documents["by_opinion"].get(opinion_id)
        if record is None:
            raise
        legacy_projected_row = {field: row[field] for field in OPINION_FIELDS}
        resolved = resolve_pdf_record(
            record,
            row=legacy_projected_row,
            cluster_id=cluster_id,
            where=where,
        )
        provenance = authoritative_record_provenance(
            record, authoritative_documents["manifest_sha256"]
        )
        provenance["legacy_projected_source_row_sha256"] = provenance.pop(
            "source_row_sha256"
        )
        provenance["legacy_source_row_digest_fields"] = list(OPINION_FIELDS)
        return (
            resolved,
            source_error.record(),
            provenance,
        )


def evidence_scan(
    path: str, clusters: dict, authoritative_documents: dict, spool_output
) -> tuple[dict, dict]:
    states = {}
    by_cluster = {}
    by_docket = {}
    progress = Progress("opinions/evidence")
    candidate_rows = 0
    resolved_rows = 0
    text_errors = 0
    supplemented_ids = set()
    for line, row in csv_rows(
        path, "search_opinion", OPINION_FIELDS, include_all=True
    ):
        progress.tick()
        if not row["cluster_id"]:
            continue
        opinion_id, cluster_id = opinion_identity(row, path=path, line=line)
        cluster = clusters.get(cluster_id)
        if cluster is None:
            continue
        candidate_rows += 1
        if opinion_id in states:
            fail(
                "duplicate candidate opinion ID",
                path=path,
                line=line,
                opinion_id=opinion_id,
            )
        resolved = None
        text_error = None
        bulk_text_error = None
        authoritative_document_source = None
        try:
            resolved, bulk_text_error, authoritative_document_source = (
                resolve_authoritative_text(
                    row,
                    cluster_id=cluster_id,
                    where="%s line %d opinion %d" % (path, line, opinion_id),
                    authoritative_documents=authoritative_documents,
                )
            )
            if bulk_text_error is not None:
                text_errors += 1
                supplemented_ids.add(opinion_id)
        except AuthoritativeTextError as error:
            text_error = error.record()
            text_errors += 1
        if resolved is not None:
            resolved_rows += 1
        spool_output.write(
            json.dumps(
                {
                    "format": CANDIDATE_SPOOL_FORMAT,
                    "source_row": line,
                    "row": {
                        **{field: row[field] for field in OPINION_META_FIELDS},
                        "source_row_sha256": stable_row_sha256(row),
                    },
                    "resolved": resolved_text_to_spool(resolved),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        own = []
        if resolved is not None:
            own = direct_evidence(row["download_url"], resolved)
        else:
            # URL provenance remains classifiable when content itself is broken.
            from cuyahoga_contract import official_rod_evidence

            own = official_rod_evidence(row["download_url"])
        state = {
            "opinion_id": opinion_id,
            "cluster_id": cluster_id,
            "docket_id": cluster["docket_id"],
            "own": own,
            "text_error": text_error,
            "bulk_text_error": bulk_text_error,
            "authoritative_document_source": authoritative_document_source,
            "normalized_sha256": resolved.normalized_sha256 if resolved else None,
            "source_raw_sha256": resolved.source_raw_sha256 if resolved else None,
        }
        states[opinion_id] = state
        for evidence in own:
            by_cluster.setdefault(cluster_id, []).append((opinion_id, evidence))
            by_docket.setdefault(cluster["docket_id"], []).append(
                (opinion_id, evidence)
            )
    progress.done()
    if not states:
        fail("no candidate opinions found", path=path)
    expected_supplements = set(authoritative_documents["by_opinion"])
    if supplemented_ids != expected_supplements:
        fail(
            "authoritative PDF generation was not consumed by the exact expected rows",
            expected=sorted(expected_supplements),
            actual=sorted(supplemented_ids),
            unused=sorted(expected_supplements - supplemented_ids),
        )

    status_counts = {}
    reason_counts = {}
    conflicts = 0
    for opinion_id, state in states.items():
        sibling_items = []
        seen = set()
        for source_id, evidence in by_cluster.get(
            state["cluster_id"], []
        ) + by_docket.get(state["docket_id"], []):
            if source_id == opinion_id:
                continue
            key = (
                source_id,
                evidence.kind,
                evidence.district,
                evidence.evidence_sha256,
            )
            if key in seen:
                continue
            seen.add(key)
            sibling_items.append((source_id, evidence))
        sibling = [evidence for _, evidence in sibling_items]
        cluster = clusters[state["cluster_id"]]
        try:
            decision = classify(
                court_id=cluster["court_id"],
                own_evidence=state["own"],
                sibling_evidence=sibling,
                where="opinion %d" % opinion_id,
            )
        except EvidenceConflictError as error:
            conflicts += 1
            decision = {
                "status": "conflict",
                "reason": error.code,
                "district": None,
                "evidence": [asdict(item) for item in state["own"] + sibling],
                "selector_version": SELECTOR_VERSION,
                "error": error.record(),
            }
        decision["sibling_opinion_ids"] = sorted({item[0] for item in sibling_items})
        state["decision"] = decision
        status_counts[decision["status"]] = status_counts.get(decision["status"], 0) + 1
        reason_counts[decision["reason"]] = reason_counts.get(decision["reason"], 0) + 1
    accepted_content_errors = [
        {
            "opinion_id": opinion_id,
            "cluster_id": state["cluster_id"],
            "docket_id": state["docket_id"],
            "selection": state["decision"],
            "text_error": state["text_error"],
        }
        for opinion_id, state in sorted(states.items())
        if state["decision"]["status"] == "accepted" and state["text_error"] is not None
    ]
    if accepted_content_errors:
        raise AuthoritativeTextError(
            "accepted opinions have no usable authoritative text; generation is not publishable",
            count=len(accepted_content_errors),
            opinions=accepted_content_errors,
            rows_scanned=progress.rows,
            candidate_rows=candidate_rows,
            text_errors_observed=text_errors,
            typed_conflicts=conflicts,
            status_counts=status_counts,
            reason_counts=reason_counts,
        )
    return states, {
        "rows_scanned": progress.rows,
        "candidate_rows": candidate_rows,
        "resolved_rows": resolved_rows,
        "source_scan_passes": 1,
        "text_resolution_passes": 1,
        "candidate_spool_format": CANDIDATE_SPOOL_FORMAT,
        "text_errors_observed": text_errors,
        "authoritative_pdf_supplements": len(supplemented_ids),
        "authoritative_pdf_opinion_ids": sorted(supplemented_ids),
        "typed_conflicts": conflicts,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
    }


def hashed_spool_lines(path: str, *, expected_sha256: str, expected_bytes: int):
    """Yield decoded physical lines and verify the exact replayed byte stream."""
    if (
        not isinstance(expected_sha256, str)
        or not _SHA256_RE.fullmatch(expected_sha256)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes <= 0
    ):
        fail(
            "candidate opinion spool replay expectation is invalid",
            path=path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
    digest = hashlib.sha256()
    bytes_read = 0
    with open(path, "rb") as source:
        for spool_line, raw_bytes in enumerate(source, 1):
            digest.update(raw_bytes)
            bytes_read += len(raw_bytes)
            try:
                raw = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                fail(
                    "candidate opinion spool is not UTF-8",
                    path=path,
                    spool_line=spool_line,
                    error=str(error),
                )
            yield spool_line, raw
    actual_sha256 = digest.hexdigest()
    if bytes_read != expected_bytes or actual_sha256 != expected_sha256:
        fail(
            "candidate opinion spool changed during physical replay",
            path=path,
            expected_bytes=expected_bytes,
            actual_bytes=bytes_read,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
        )


def spooled_opinion_rows(path: str, *, expected_sha256: str, expected_bytes: int):
    """Read the sealed candidate subset produced by the sole archive scan."""
    for spool_line, raw in hashed_spool_lines(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    ):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            fail(
                "candidate opinion spool contains invalid JSON",
                path=path,
                spool_line=spool_line,
                error=str(error),
            )
        if not isinstance(payload, dict) or set(payload) != {
            "format",
            "source_row",
            "row",
            "resolved",
        }:
            fail(
                "candidate opinion spool envelope mismatch",
                path=path,
                spool_line=spool_line,
            )
        if payload["format"] != CANDIDATE_SPOOL_FORMAT:
            fail(
                "candidate opinion spool format mismatch",
                path=path,
                spool_line=spool_line,
                expected=CANDIDATE_SPOOL_FORMAT,
                actual=payload["format"],
            )
        source_row = payload["source_row"]
        row = payload["row"]
        if not isinstance(source_row, int) or source_row < 1:
            fail(
                "candidate opinion spool has invalid source row",
                path=path,
                spool_line=spool_line,
                value=source_row,
            )
        if not isinstance(row, dict) or set(row) != set(SPOOL_OPINION_FIELDS):
            fail(
                "candidate opinion spool row schema mismatch",
                path=path,
                spool_line=spool_line,
                expected=sorted(SPOOL_OPINION_FIELDS),
                actual=sorted(row) if isinstance(row, dict) else type(row).__name__,
            )
        if any(not isinstance(row[field], str) for field in SPOOL_OPINION_FIELDS):
            fail(
                "candidate opinion spool contains a non-string source field",
                path=path,
                spool_line=spool_line,
            )
        resolved = resolved_text_from_spool(
            payload["resolved"], path=path, spool_line=spool_line
        )
        yield source_row, row, resolved


def resolved_text_to_spool(resolved: ResolvedText | None):
    if resolved is None:
        return None
    payload = asdict(resolved)
    payload["caption_blocks"] = list(resolved.caption_blocks)
    return payload


def resolved_text_from_spool(value, *, path: str, spool_line: int):
    if value is None:
        return None
    expected_keys = {
        "text",
        "source_field",
        "source_raw_sha256",
        "normalized_sha256",
        "normalized_chars",
        "caption_blocks",
        "fields",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        fail(
            "candidate spool resolved-text schema mismatch",
            path=path,
            spool_line=spool_line,
        )
    text = value["text"]
    source_field = value["source_field"]
    source_raw_sha256 = value["source_raw_sha256"]
    normalized_sha256 = value["normalized_sha256"]
    normalized_chars = value["normalized_chars"]
    caption_blocks = value["caption_blocks"]
    fields = value["fields"]
    if (
        not isinstance(text, str)
        or not text
        or source_field
        not in {
            "html_with_citations",
            "plain_text_no_html_with_citations",
            "authoritative_pdf_supplement",
        }
        or not isinstance(source_raw_sha256, str)
        or not _SHA256_RE.fullmatch(source_raw_sha256)
        or not isinstance(normalized_sha256, str)
        or not _SHA256_RE.fullmatch(normalized_sha256)
        or not isinstance(normalized_chars, int)
        or isinstance(normalized_chars, bool)
        or normalized_chars <= 0
        or not isinstance(caption_blocks, list)
        or any(not isinstance(item, str) for item in caption_blocks)
        or not isinstance(fields, dict)
        or set(fields) != set(TEXT_FIELDS)
    ):
        fail(
            "candidate spool resolved-text value contract failed",
            path=path,
            spool_line=spool_line,
        )
    if normalized_chars != len(text) or normalized_sha256 != sha256_text(text):
        fail(
            "candidate spool resolved text differs from its physical text hash",
            path=path,
            spool_line=spool_line,
            declared_chars=normalized_chars,
            actual_chars=len(text),
            declared_sha256=normalized_sha256,
            actual_sha256=sha256_text(text),
        )
    for field in TEXT_FIELDS:
        facts = fields[field]
        if not isinstance(facts, dict) or set(facts) != {
            "raw_present",
            "nonempty",
            "raw_chars",
            "raw_sha256",
        }:
            fail(
                "candidate spool text-field provenance schema mismatch",
                path=path,
                spool_line=spool_line,
                field=field,
            )
        raw_chars = facts["raw_chars"]
        raw_sha256 = facts["raw_sha256"]
        if (
            not isinstance(facts["raw_present"], bool)
            or not isinstance(facts["nonempty"], bool)
            or not isinstance(raw_chars, int)
            or isinstance(raw_chars, bool)
            or raw_chars < 0
            or (
                not facts["raw_present"]
                and (raw_chars != 0 or facts["nonempty"] or raw_sha256 is not None)
            )
            or (raw_chars == 0 and (facts["nonempty"] or raw_sha256 is not None))
            or (
                raw_chars > 0
                and (
                    not isinstance(raw_sha256, str)
                    or not _SHA256_RE.fullmatch(raw_sha256)
                )
            )
        ):
            fail(
                "candidate spool text-field provenance value mismatch",
                path=path,
                spool_line=spool_line,
                field=field,
            )
    source_text_field = {
        "html_with_citations": "html_with_citations",
        "plain_text_no_html_with_citations": "plain_text",
    }.get(source_field)
    if (
        source_text_field is not None
        and fields[source_text_field]["raw_sha256"] != source_raw_sha256
    ):
        fail(
            "candidate spool source hash differs from text-field provenance",
            path=path,
            spool_line=spool_line,
            source_field=source_field,
        )
    return ResolvedText(
        text=text,
        source_field=source_field,
        source_raw_sha256=source_raw_sha256,
        normalized_sha256=normalized_sha256,
        normalized_chars=normalized_chars,
        caption_blocks=tuple(caption_blocks),
        fields=fields,
    )


def source_opinion_provenance(row: dict) -> dict:
    if not _SHA256_RE.fullmatch(row["source_row_sha256"]):
        fail(
            "CourtListener opinion source row digest is malformed",
            value=row["source_row_sha256"],
        )
    if row["sha1"] and not _SHA1_RE.fullmatch(row["sha1"]):
        fail("CourtListener opinion sha1 is malformed", value=row["sha1"])
    return {
        "source_row_sha256": row["source_row_sha256"],
        "sha1": row["sha1"] or None,
        "download_url": row["download_url"] or None,
        "local_path": row["local_path"] or None,
        "date_created": row["date_created"] or None,
        "date_modified": row["date_modified"] or None,
    }


def write_opinions(
    spool_path: str,
    spool_sha256: str,
    spool_bytes: int,
    clusters: dict,
    states: dict,
    source_provenance: dict,
    snapshot_date: str,
    acquired_at: str,
    raw_output,
    idmap_output,
    rejection_output,
    conflict_output,
    candidate_cluster_accounting_output,
    corrections: dict,
) -> dict:
    progress = Progress("opinions/output-spool")
    accepted_ids = set()
    rejected_ids = set()
    conflict_ids = set()
    correction_opinion_partitions = {}
    cluster_opinion_partitions = {
        cluster_id: {"accepted": [], "rejected": [], "conflict": []}
        for cluster_id in clusters
    }
    by_court = {}
    text_sources = {}
    resolved_spool_rows = 0
    idmap_writer = csv.writer(idmap_output, lineterminator="\n")
    idmap_writer.writerow(["opinion_id", "cluster_id", "docket_id", "court_id"])

    for line, row, resolved in spooled_opinion_rows(
        spool_path,
        expected_sha256=spool_sha256,
        expected_bytes=spool_bytes,
    ):
        progress.tick()
        if resolved is not None:
            resolved_spool_rows += 1
        opinion_id, cluster_id = opinion_identity(row, path=spool_path, line=line)
        cluster = clusters.get(cluster_id)
        if cluster is None:
            fail(
                "candidate opinion spool references an unknown cluster",
                path=spool_path,
                line=line,
                cluster_id=cluster_id,
            )
        state = states.pop(opinion_id, None)
        if state is None:
            fail(
                "sealed candidate spool differs from evidence state",
                path=spool_path,
                line=line,
                opinion_id=opinion_id,
            )
        decision = state["decision"]
        audit = {
            "opinion_id": opinion_id,
            "cluster_id": cluster_id,
            "docket_id": cluster["docket_id"],
            "court_id": cluster["court_id"],
            "selection": decision,
            "text_error": state["text_error"],
            "bulk_text_error": state["bulk_text_error"],
            "supporting_lower_court_origin": cluster["appeal_from_str"] or None,
            "correction": cluster["correction"],
        }
        partition = decision["status"]
        cluster_opinion_partitions[cluster_id][partition].append(opinion_id)
        if cluster["correction"] is not None:
            correction_id = cluster["correction"]["correction_id"]
            correction_opinion_partitions.setdefault(
                correction_id,
                {"accepted": [], "rejected": [], "conflict": []},
            )[partition].append(opinion_id)
        if decision["status"] == "conflict":
            conflict_output.write(
                json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
            )
            conflict_ids.add(opinion_id)
            continue
        if decision["status"] == "rejected":
            rejection_output.write(
                json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
            )
            rejected_ids.add(opinion_id)
            continue
        if resolved is None:
            fail(
                "accepted opinion has no resolved text in the sealed candidate spool",
                opinion_id=opinion_id,
            )
        if (
            resolved.normalized_sha256 != state["normalized_sha256"]
            or resolved.source_raw_sha256 != state["source_raw_sha256"]
        ):
            fail(
                "sealed candidate spool text differs from evidence state",
                opinion_id=opinion_id,
                first_normalized_sha256=state["normalized_sha256"],
                second_normalized_sha256=resolved.normalized_sha256,
            )
        author_id = optional_int(
            row["author_id"], field="author_id", path=spool_path, line=line
        )
        if row["per_curiam"] not in {"", "t", "f"}:
            fail(
                "per_curiam is not a PostgreSQL boolean",
                path=spool_path,
                line=line,
                opinion_id=opinion_id,
                value=row["per_curiam"],
            )
        record = {
            "opinion_id": opinion_id,
            "cluster_id": cluster_id,
            "cluster_slug": cluster["cluster_slug"],
            "docket_id": cluster["docket_id"],
            "court_id": cluster["court_id"],
            "cuyahoga_signal": decision["reason"],
            "selection": decision,
            "opinion_type": row["type"],
            "author_id": author_id,
            "author_str": row["author_str"],
            "per_curiam": row["per_curiam"] == "t",
            "date_filed": cluster["date_filed"],
            "case_name": cluster["case_name"],
            "case_name_full": cluster["case_name_full"],
            "docket_number": cluster["docket_number"],
            "judges": cluster["judges"],
            "disposition": cluster["disposition"],
            "posture": cluster["posture"],
            "nature_of_suit": cluster["nature_of_suit"],
            "syllabus": cluster["syllabus"],
            "precedential_status": cluster["precedential_status"],
            "citation_count": cluster["citation_count"],
            "cluster_source": cluster["source"],
            "correction": cluster["correction"],
            "text": resolved.text,
            "text_source": resolved.source_field,
            "text_chars": resolved.normalized_chars,
            "text_provenance": resolved.provenance(),
            "authoritative_document_source": state["authoritative_document_source"],
            "courtlistener_opinion_source": source_opinion_provenance(row),
            "source_archive": source_provenance["opinions"],
            "source_snapshot_date": snapshot_date,
            "source_acquired_at": acquired_at,
            "canonical_source_url": (
                "https://www.courtlistener.com/opinion/%d/%s/"
                % (cluster_id, cluster["cluster_slug"])
            ),
        }
        raw_output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        idmap_writer.writerow(
            [opinion_id, cluster_id, cluster["docket_id"], cluster["court_id"]]
        )
        accepted_ids.add(opinion_id)
        court = cluster["court_id"]
        by_court[court] = by_court.get(court, 0) + 1
        text_sources[resolved.source_field] = (
            text_sources.get(resolved.source_field, 0) + 1
        )
    progress.done()
    if states:
        fail(
            "opinion IDs from evidence scan were absent from output scan",
            count=len(states),
            sample=sorted(states)[:20],
        )
    expected_correction_opinion_partitions = {}
    for entry in corrections["manifest"]["corrections"]:
        partitions = {"accepted": [], "rejected": [], "conflict": []}
        for expected in entry["expected_opinions"]:
            partitions[expected["partition"]].append(expected["opinion_id"])
        expected_correction_opinion_partitions[entry["correction_id"]] = {
            key: sorted(value) for key, value in partitions.items()
        }
    actual_correction_opinion_partitions = {
        correction_id: {
            key: sorted(value) for key, value in partitions.items()
        }
        for correction_id, partitions in correction_opinion_partitions.items()
    }
    if actual_correction_opinion_partitions != expected_correction_opinion_partitions:
        raise CorrectionError(
            "corrected opinions do not exactly match correction manifest partitions",
            expected=expected_correction_opinion_partitions,
            actual=actual_correction_opinion_partitions,
        )
    if not accepted_ids:
        fail("selector accepted zero opinions")
    if (
        accepted_ids & rejected_ids
        or accepted_ids & conflict_ids
        or rejected_ids & conflict_ids
    ):
        fail("selection output partitions overlap")
    cluster_partition_status_counts = {}
    clusters_with_opinions = 0
    for cluster_id, cluster in clusters.items():
        partitions = {
            key: sorted(value)
            for key, value in cluster_opinion_partitions[cluster_id].items()
        }
        nonempty = [key for key, value in partitions.items() if value]
        if not nonempty:
            partition_status = "no_opinion"
        elif len(nonempty) == 1:
            partition_status = nonempty[0]
            clusters_with_opinions += 1
        else:
            partition_status = "mixed"
            clusters_with_opinions += 1
        opinion_count = sum(len(value) for value in partitions.values())
        correction = cluster["correction"]
        candidate_cluster_accounting_output.write(
            json.dumps(
                {
                    "cluster_id": cluster_id,
                    "docket_id": cluster["docket_id"],
                    "court_id": cluster["court_id"],
                    "correction_id": (
                        correction["correction_id"] if correction is not None else None
                    ),
                    "opinion_ids_by_partition": partitions,
                    "opinion_count": opinion_count,
                    "partition_status": partition_status,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        cluster_partition_status_counts[partition_status] = (
            cluster_partition_status_counts.get(partition_status, 0) + 1
        )
    actual_correction_opinions = {
        correction_id: sorted(
            opinion_id
            for values in partitions.values()
            for opinion_id in values
        )
        for correction_id, partitions in actual_correction_opinion_partitions.items()
    }
    return {
        "spool_rows_read": progress.rows,
        "spool_resolved_rows": resolved_spool_rows,
        "accepted": len(accepted_ids),
        "rejected": len(rejected_ids),
        "conflicts": len(conflict_ids),
        "by_court": by_court,
        "text_sources": text_sources,
        "correction_opinions": actual_correction_opinions,
        "correction_opinion_partitions": actual_correction_opinion_partitions,
        "candidate_cluster_accounting": {
            "total": len(clusters),
            "with_opinions": clusters_with_opinions,
            "without_opinions": len(clusters) - clusters_with_opinions,
            "by_partition_status": cluster_partition_status_counts,
        },
    }


def validate_source_provenance(manifest: dict) -> None:
    snapshot = manifest.get("source_snapshot_date")
    acquired_at = manifest.get("source_acquired_at")
    try:
        date.fromisoformat(snapshot)
    except (TypeError, ValueError) as error:
        fail("extraction source snapshot date is invalid", value=snapshot, error=str(error))
    try:
        acquired = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        fail("extraction source acquisition time is invalid", value=acquired_at, error=str(error))
    if acquired.tzinfo is None:
        fail("extraction source acquisition time lacks an offset", value=acquired_at)
    source_archives = manifest.get("source_archives")
    if not isinstance(source_archives, dict):
        fail("extraction source archive map is absent")
    for role in ("dockets", "clusters", "opinions"):
        archive = source_archives.get(role)
        if (
            not isinstance(archive, dict)
            or set(archive)
            != {"archive_name", "archive_sha256", "snapshot_date", "acquired_at"}
            or not isinstance(archive.get("archive_name"), str)
            or not archive["archive_name"]
            or not isinstance(archive.get("archive_sha256"), str)
            or not _SHA256_RE.fullmatch(archive["archive_sha256"])
            or archive.get("snapshot_date") != snapshot
            or archive.get("acquired_at") != acquired_at
        ):
            fail(
                "extraction source archives have mixed or malformed provenance",
                role=role,
            )
    checksum_manifest = source_archives.get("manifest")
    if (
        not isinstance(checksum_manifest, dict)
        or set(checksum_manifest) != {"name", "sha256"}
        or not isinstance(checksum_manifest.get("name"), str)
        or not checksum_manifest["name"]
        or not isinstance(checksum_manifest.get("sha256"), str)
        or not _SHA256_RE.fullmatch(checksum_manifest["sha256"])
    ):
        fail("extraction source checksum-manifest provenance is malformed")


def verify_extract_generation(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    generation_format = manifest.get("format")
    if generation_format not in {LEGACY_FORMAT, PREVIOUS_FORMAT, FORMAT}:
        fail("unsupported extraction generation format", format=manifest.get("format"))
    if generation_format in SOURCE_ROW_FORMATS:
        if manifest.get("normalized_schema_version") != generation_format:
            fail("extraction normalized schema version differs from its format")
        producer = manifest.get("producer")
        if not isinstance(producer, dict):
            fail("extraction manifest is missing producer provenance")
        implementation = producer.get("implementation_sha256")
        config = producer.get("config")
        if (
            not isinstance(implementation, dict)
            or set(implementation)
            != {
                "authoritative_documents.py",
                "cuyahoga_contract.py",
                "extract_cuyahoga.py",
                "law_generation.py",
                "signal_audit.py",
                "source_scan_lock.py",
            }
            or any(
                not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
                for value in implementation.values()
            )
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
            fail("extraction producer provenance is malformed")
        expected_config = {
            "format": generation_format,
            "selector_version": manifest.get("selector_version"),
            "text_policy_version": manifest.get("text_policy_version"),
            "correction_policy_version": manifest.get("correction_policy_version"),
            "candidate_spool_format": CANDIDATE_SPOOL_FORMAT,
            "text_fields": list(TEXT_FIELDS),
            "source_row_digest": (
                "all physical CSV columns as a canonical JSON object"
            ),
            "authoritative_pdf_v1_row_digest": (
                "explicitly labeled legacy projection over OPINION_FIELDS"
            ),
        }
        if generation_format == FORMAT:
            expected_config["candidate_cluster_accounting"] = (
                "one row per candidate cluster with explicit opinion partitions or no_opinion"
            )
        if config != expected_config:
            fail("extraction producer config differs from manifest contract")
    validate_source_provenance(manifest)
    required = {
        RAW_FILE,
        IDMAP_FILE,
        DOCKETS_FILE,
        CLUSTERS_FILE,
        REJECTIONS_FILE,
        CONFLICTS_FILE,
        COUNTS_FILE,
    }
    if generation_format == FORMAT:
        required.add(CANDIDATE_CLUSTER_ACCOUNTING_FILE)
    if set(manifest["files"]) != required:
        fail(
            "extraction generation member contract mismatch",
            expected=sorted(required),
            actual=sorted(manifest["files"]),
        )
    raw_ids = set()
    text_sources = {}
    authoritative_opinion_ids = set()
    correction_opinions = {}
    correction_opinion_partitions = {}
    cluster_opinion_partitions = {
        "accepted": {},
        "rejected": {},
        "conflict": {},
    }
    opinion_archive = manifest.get("source_archives", {}).get("opinions")
    if not isinstance(opinion_archive, dict):
        fail("extraction manifest is missing opinion archive provenance")
    authoritative_manifest = manifest.get("authoritative_documents")
    if not isinstance(authoritative_manifest, dict) or not isinstance(
        authoritative_manifest.get("generation"), str
    ):
        fail("extraction manifest is missing authoritative document provenance")
    authoritative_documents = load_authoritative_generation(
        authoritative_manifest["generation"],
        opinions_archive_sha256=opinion_archive.get("archive_sha256"),
        snapshot_date=manifest.get("source_snapshot_date"),
        # This derived-generation audit re-reads every authoritative member,
        # digest, record, and provenance binding. Exact PDF text re-extraction
        # remains mandatory in the authoritative-document build/verify path.
        reextract_pdfs=False,
    )
    if (
        authoritative_manifest.get("manifest_sha256")
        != authoritative_documents["manifest_sha256"]
        or authoritative_manifest.get("format")
        != authoritative_documents["manifest"].get("format")
        or authoritative_manifest.get("opinion_ids")
        != sorted(authoritative_documents["by_opinion"])
    ):
        fail(
            "physical authoritative document generation differs from extraction manifest"
        )
    correction_ids = set(
        manifest.get("correction_manifest", {}).get("corrections_applied", [])
    )

    def record_cluster_partition(
        partition: str, row: dict, opinion_id: int, *, file: str, line: int
    ) -> None:
        cluster_id = row.get("cluster_id")
        if (
            not isinstance(cluster_id, int)
            or isinstance(cluster_id, bool)
            or cluster_id <= 0
        ):
            fail(
                "opinion partition has invalid cluster_id",
                file=file,
                line=line,
                opinion_id=opinion_id,
                cluster_id=cluster_id,
            )
        cluster_opinion_partitions[partition].setdefault(cluster_id, []).append(
            opinion_id
        )

    def record_partition_correction(
        partition: str,
        correction: object,
        opinion_id: int,
        *,
        file: str,
        line: int,
    ) -> None:
        if correction is None:
            return
        correction_id = (
            correction.get("correction_id")
            if isinstance(correction, dict)
            else None
        )
        if correction_id not in correction_ids:
            fail(
                "opinion partition has undeclared correction",
                file=file,
                line=line,
                opinion_id=opinion_id,
                correction_id=correction_id,
            )
        if generation_format == FORMAT or partition == "accepted":
            correction_opinions.setdefault(correction_id, []).append(opinion_id)
        if generation_format == FORMAT:
            correction_opinion_partitions.setdefault(
                correction_id,
                {"accepted": [], "rejected": [], "conflict": []},
            )[partition].append(opinion_id)

    with open(root / RAW_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail("raw output contains invalid JSON", line=lineno, error=str(error))
            opinion_id = row.get("opinion_id")
            if not isinstance(opinion_id, int) or opinion_id in raw_ids:
                fail(
                    "raw output has invalid/duplicate opinion_id",
                    line=lineno,
                    value=opinion_id,
                )
            raw_ids.add(opinion_id)
            record_cluster_partition(
                "accepted", row, opinion_id, file=RAW_FILE, line=lineno
            )
            text = row.get("text")
            provenance = row.get("text_provenance")
            if not isinstance(text, str) or not text:
                fail(
                    "raw accepted row has empty text",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            if (
                not isinstance(provenance, dict)
                or provenance.get("normalized_sha256") != sha256_text(text)
                or provenance.get("normalized_chars") != len(text)
            ):
                fail(
                    "raw text provenance readback mismatch",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            selection = row.get("selection")
            if not isinstance(selection, dict) or selection.get("status") != "accepted":
                fail(
                    "raw row is not explicitly accepted",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            source_field = row.get("text_source")
            if source_field not in {
                "html_with_citations",
                "plain_text_no_html_with_citations",
                "authoritative_pdf_supplement",
            }:
                fail(
                    "raw row has prohibited text source",
                    line=lineno,
                    value=source_field,
                )
            fields = provenance.get("fields")
            if not isinstance(fields, dict) or set(fields) != set(TEXT_FIELDS):
                fail(
                    "competing-field provenance schema mismatch",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            for field in TEXT_FIELDS:
                facts = fields[field]
                if not isinstance(facts, dict) or set(facts) != {
                    "raw_present",
                    "nonempty",
                    "raw_chars",
                    "raw_sha256",
                }:
                    fail(
                        "competing-field provenance value schema mismatch",
                        line=lineno,
                        opinion_id=opinion_id,
                        field=field,
                    )
                raw_chars = facts["raw_chars"]
                raw_sha256 = facts["raw_sha256"]
                if (
                    not isinstance(facts["raw_present"], bool)
                    or not isinstance(facts["nonempty"], bool)
                    or not isinstance(raw_chars, int)
                    or isinstance(raw_chars, bool)
                    or raw_chars < 0
                    or (
                        not facts["raw_present"]
                        and (raw_chars != 0 or facts["nonempty"] or raw_sha256 is not None)
                    )
                    or (
                        raw_chars == 0
                        and (facts["nonempty"] or raw_sha256 is not None)
                    )
                    or (
                        raw_chars > 0
                        and (
                            not isinstance(raw_sha256, str)
                            or not _SHA256_RE.fullmatch(raw_sha256)
                        )
                    )
                ):
                    fail(
                        "competing-field provenance values are inconsistent",
                        line=lineno,
                        opinion_id=opinion_id,
                        field=field,
                    )
            authoritative_source = row.get("authoritative_document_source")
            if source_field == "authoritative_pdf_supplement":
                if any(facts.get("nonempty") for facts in fields.values()):
                    fail(
                        "PDF supplement replaced a nonempty bulk text field",
                        line=lineno,
                        opinion_id=opinion_id,
                    )
                authoritative_manifest = manifest.get("authoritative_documents")
                if (
                    not isinstance(authoritative_source, dict)
                    or not isinstance(authoritative_manifest, dict)
                    or authoritative_source.get("generation_manifest_sha256")
                    != authoritative_manifest.get("manifest_sha256")
                    or authoritative_source.get("opinion_id") != opinion_id
                    or authoritative_source.get("cluster_id") != row.get("cluster_id")
                    or authoritative_source.get("raw_text_sha256")
                    != provenance.get("source_raw_sha256")
                    or authoritative_source.get("normalized_text_sha256")
                    != provenance.get("normalized_sha256")
                    or authoritative_source.get("normalized_chars") != len(text)
                ):
                    fail(
                        "PDF supplement provenance does not bind to the physical raw row",
                        line=lineno,
                        opinion_id=opinion_id,
                    )
                if generation_format in SOURCE_ROW_FORMATS and (
                    "source_row_sha256" in authoritative_source
                    or not _SHA256_RE.fullmatch(
                        authoritative_source.get(
                            "legacy_projected_source_row_sha256", ""
                        )
                    )
                    or authoritative_source.get("legacy_source_row_digest_fields")
                    != list(OPINION_FIELDS)
                ):
                    fail(
                        "PDF supplement legacy row-digest scope is ambiguous",
                        line=lineno,
                        opinion_id=opinion_id,
                    )
                authoritative_opinion_ids.add(opinion_id)
            else:
                if authoritative_source is not None:
                    fail(
                        "normal bulk text row has prohibited PDF provenance",
                        line=lineno,
                        opinion_id=opinion_id,
                    )
                field_name = (
                    "plain_text"
                    if source_field == "plain_text_no_html_with_citations"
                    else source_field
                )
                if fields.get(field_name, {}).get("raw_sha256") != provenance.get(
                    "source_raw_sha256"
                ):
                    fail(
                        "selected raw text digest is not linked to competing-field provenance",
                        line=lineno,
                        opinion_id=opinion_id,
                        field=field_name,
                    )
            if row.get("source_archive") != opinion_archive:
                fail(
                    "raw archive provenance differs from manifest",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            if row.get("source_snapshot_date") != manifest.get(
                "source_snapshot_date"
            ) or row.get("source_acquired_at") != manifest.get("source_acquired_at"):
                fail(
                    "raw source times differ from manifest",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            source_record = row.get("courtlistener_opinion_source")
            expected_source_record = {
                "sha1",
                "download_url",
                "local_path",
                "date_created",
                "date_modified",
            }
            if generation_format in SOURCE_ROW_FORMATS:
                expected_source_record.add("source_row_sha256")
            if (
                not isinstance(source_record, dict)
                or set(source_record) != expected_source_record
            ):
                fail(
                    "raw CourtListener source schema mismatch",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            if generation_format in SOURCE_ROW_FORMATS and not _SHA256_RE.fullmatch(
                source_record["source_row_sha256"]
            ):
                fail(
                    "raw CourtListener source row digest is malformed",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            expected_url = "https://www.courtlistener.com/opinion/%d/%s/" % (
                row.get("cluster_id"),
                row.get("cluster_slug"),
            )
            if row.get("canonical_source_url") != expected_url:
                fail(
                    "raw canonical cluster URL mismatch",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            correction = row.get("correction")
            record_partition_correction(
                "accepted",
                correction,
                opinion_id,
                file=RAW_FILE,
                line=lineno,
            )
            text_sources[source_field] = text_sources.get(source_field, 0) + 1
    idmap_ids = set()
    with open(root / IDMAP_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["opinion_id", "cluster_id", "docket_id", "court_id"]:
            fail("idmap header mismatch", header=reader.fieldnames)
        for row in reader:
            opinion_id = int(row["opinion_id"])
            if opinion_id in idmap_ids:
                fail("idmap duplicate opinion_id", opinion_id=opinion_id)
            idmap_ids.add(opinion_id)
    if raw_ids != idmap_ids:
        fail(
            "raw and idmap opinion sets differ",
            raw_only=sorted(raw_ids - idmap_ids)[:20],
            idmap_only=sorted(idmap_ids - raw_ids)[:20],
        )
    rejected_ids = set()
    with open(root / REJECTIONS_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            row = json.loads(line)
            opinion_id = row.get("opinion_id")
            decision = row.get("selection")
            if (
                not isinstance(opinion_id, int)
                or opinion_id in rejected_ids
                or not isinstance(decision, dict)
                or decision.get("status") != "rejected"
            ):
                fail(
                    "invalid selection rejection row",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            if (
                decision.get("reason") == "explicit_non_eighth_district"
                and decision.get("district") == 8
            ):
                fail(
                    "non-Eighth rejection claims district 8",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            rejected_ids.add(opinion_id)
            record_cluster_partition(
                "rejected", row, opinion_id, file=REJECTIONS_FILE, line=lineno
            )
            record_partition_correction(
                "rejected",
                row.get("correction"),
                opinion_id,
                file=REJECTIONS_FILE,
                line=lineno,
            )
    conflict_ids = set()
    with open(root / CONFLICTS_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            row = json.loads(line)
            opinion_id = row.get("opinion_id")
            decision = row.get("selection")
            if (
                not isinstance(opinion_id, int)
                or opinion_id in conflict_ids
                or not isinstance(decision, dict)
                or decision.get("status") != "conflict"
                or decision.get("reason") != "issuing_court_evidence_conflict"
            ):
                fail(
                    "invalid typed selection conflict row",
                    line=lineno,
                    opinion_id=opinion_id,
                )
            conflict_ids.add(opinion_id)
            record_cluster_partition(
                "conflict", row, opinion_id, file=CONFLICTS_FILE, line=lineno
            )
            record_partition_correction(
                "conflict",
                row.get("correction"),
                opinion_id,
                file=CONFLICTS_FILE,
                line=lineno,
            )
    if raw_ids & rejected_ids or raw_ids & conflict_ids or rejected_ids & conflict_ids:
        fail("physical selection partitions overlap")
    with open(root / COUNTS_FILE, "r", encoding="utf-8") as source:
        counts = json.load(source)
    if counts != manifest.get("row_counts"):
        fail("physical counts JSON differs from extraction manifest")
    if counts["output"]["accepted"] != len(raw_ids):
        fail(
            "accepted count differs from physical raw rows",
            declared=counts["output"]["accepted"],
            actual=len(raw_ids),
        )
    if counts["output"]["text_sources"] != text_sources:
        fail(
            "text source counts differ from physical raw rows",
            declared=counts["output"]["text_sources"],
            actual=text_sources,
        )
    declared_authoritative_ids = set(
        manifest.get("authoritative_documents", {}).get("opinion_ids", [])
    )
    if authoritative_opinion_ids != declared_authoritative_ids:
        fail(
            "physical PDF supplement rows differ from the extraction manifest",
            declared=sorted(declared_authoritative_ids),
            actual=sorted(authoritative_opinion_ids),
        )
    if counts["output"]["rejected"] != len(rejected_ids):
        fail("rejected count differs from physical rejection rows")
    if counts["output"]["conflicts"] != len(conflict_ids):
        fail("conflict count differs from physical conflict rows")
    if counts["evidence_scan"]["candidate_rows"] != (
        len(raw_ids) + len(rejected_ids) + len(conflict_ids)
    ):
        fail("candidate count differs from physical decision partitions")
    evidence_counts = counts["evidence_scan"]
    output_counts = counts["output"]
    if evidence_counts.get("source_scan_passes") != 1:
        fail(
            "extraction did not declare exactly one opinion archive scan",
            actual=evidence_counts.get("source_scan_passes"),
        )
    if evidence_counts.get("text_resolution_passes") != 1:
        fail(
            "extraction did not declare exactly one text-resolution pass",
            actual=evidence_counts.get("text_resolution_passes"),
        )
    expected_spool_format = (
        CANDIDATE_SPOOL_FORMAT
        if generation_format in SOURCE_ROW_FORMATS
        else LEGACY_CANDIDATE_SPOOL_FORMAT
    )
    if evidence_counts.get("candidate_spool_format") != expected_spool_format:
        fail(
            "extraction candidate spool format provenance is missing",
            expected=expected_spool_format,
            actual=evidence_counts.get("candidate_spool_format"),
        )
    spool_bytes = evidence_counts.get("candidate_spool_bytes")
    spool_sha256 = evidence_counts.get("candidate_spool_sha256")
    if (
        not isinstance(spool_bytes, int)
        or spool_bytes <= 0
        or not isinstance(spool_sha256, str)
        or not _SHA256_RE.fullmatch(spool_sha256)
    ):
        fail(
            "candidate spool provenance is malformed",
            bytes=spool_bytes,
            sha256=spool_sha256,
        )
    if output_counts.get("spool_rows_read") != evidence_counts["candidate_rows"]:
        fail(
            "sealed candidate spool count differs from the evidence scan",
            evidence_candidates=evidence_counts["candidate_rows"],
            spool_rows_read=output_counts.get("spool_rows_read"),
        )
    if output_counts.get("spool_resolved_rows") != evidence_counts["resolved_rows"]:
        fail(
            "sealed spool resolved-text count differs from the source scan",
            source_resolved=evidence_counts["resolved_rows"],
            spool_resolved=output_counts.get("spool_resolved_rows"),
        )
    physical_corrections = {
        key: sorted(value) for key, value in correction_opinions.items()
    }
    if counts["output"]["correction_opinions"] != physical_corrections:
        fail("correction opinion counts differ from physical decision partitions")
    if generation_format == FORMAT:
        physical_correction_partitions = {
            correction_id: {
                key: sorted(value) for key, value in partitions.items()
            }
            for correction_id, partitions in correction_opinion_partitions.items()
        }
        if (
            output_counts.get("correction_opinion_partitions")
            != physical_correction_partitions
        ):
            fail(
                "correction partition accounting differs from physical decision rows"
            )

    with open(root / DOCKETS_FILE, "r", encoding="utf-8") as source:
        physical_dockets = sum(1 for line in source if line.strip())
    if physical_dockets != counts["dockets"]["candidate_rows"]:
        fail(
            "join map count differs from physical rows",
            file=DOCKETS_FILE,
            declared=counts["dockets"]["candidate_rows"],
            actual=physical_dockets,
        )

    physical_clusters = {}
    with open(root / CLUSTERS_FILE, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(
                    "cluster map contains invalid JSON",
                    line=lineno,
                    error=str(error),
                )
            cluster_id = row.get("cluster_id")
            if (
                not isinstance(cluster_id, int)
                or isinstance(cluster_id, bool)
                or cluster_id <= 0
                or cluster_id in physical_clusters
            ):
                fail(
                    "cluster map has invalid or duplicate cluster_id",
                    line=lineno,
                    cluster_id=cluster_id,
                )
            correction = row.get("correction")
            correction_id = (
                correction.get("correction_id")
                if isinstance(correction, dict)
                else None
            )
            if correction is not None and correction_id not in correction_ids:
                fail(
                    "cluster map has undeclared correction",
                    line=lineno,
                    cluster_id=cluster_id,
                    correction_id=correction_id,
                )
            physical_clusters[cluster_id] = {
                "docket_id": row.get("docket_id"),
                "court_id": row.get("court_id"),
                "correction_id": correction_id,
            }
    if len(physical_clusters) != counts["clusters"]["candidate_rows"]:
        fail(
            "join map count differs from physical rows",
            file=CLUSTERS_FILE,
            declared=counts["clusters"]["candidate_rows"],
            actual=len(physical_clusters),
        )
    opinion_cluster_ids = {
        cluster_id
        for partitions in cluster_opinion_partitions.values()
        for cluster_id in partitions
    }
    unknown_opinion_clusters = opinion_cluster_ids - set(physical_clusters)
    if unknown_opinion_clusters:
        fail(
            "physical opinion partitions reference unknown candidate clusters",
            sample=sorted(unknown_opinion_clusters)[:20],
        )

    if generation_format == FORMAT:
        expected_accounting = {}
        status_counts = {}
        clusters_with_opinions = 0
        for cluster_id, cluster in physical_clusters.items():
            partitions = {
                partition: sorted(
                    cluster_opinion_partitions[partition].get(cluster_id, [])
                )
                for partition in ("accepted", "rejected", "conflict")
            }
            nonempty = [key for key, value in partitions.items() if value]
            if not nonempty:
                partition_status = "no_opinion"
            elif len(nonempty) == 1:
                partition_status = nonempty[0]
                clusters_with_opinions += 1
            else:
                partition_status = "mixed"
                clusters_with_opinions += 1
            status_counts[partition_status] = status_counts.get(partition_status, 0) + 1
            expected_accounting[cluster_id] = {
                "cluster_id": cluster_id,
                "docket_id": cluster["docket_id"],
                "court_id": cluster["court_id"],
                "correction_id": cluster["correction_id"],
                "opinion_ids_by_partition": partitions,
                "opinion_count": sum(len(value) for value in partitions.values()),
                "partition_status": partition_status,
            }
        observed_accounting_ids = set()
        with open(
            root / CANDIDATE_CLUSTER_ACCOUNTING_FILE, "r", encoding="utf-8"
        ) as source:
            for lineno, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    fail(
                        "candidate cluster accounting contains invalid JSON",
                        line=lineno,
                        error=str(error),
                    )
                cluster_id = row.get("cluster_id")
                if cluster_id in observed_accounting_ids:
                    fail(
                        "candidate cluster accounting repeats cluster_id",
                        line=lineno,
                        cluster_id=cluster_id,
                    )
                expected = expected_accounting.get(cluster_id)
                if expected is None or row != expected:
                    fail(
                        "candidate cluster accounting differs from physical maps and partitions",
                        line=lineno,
                        cluster_id=cluster_id,
                        expected=expected,
                        actual=row,
                    )
                observed_accounting_ids.add(cluster_id)
        if observed_accounting_ids != set(physical_clusters):
            fail(
                "candidate cluster accounting does not cover every physical candidate cluster",
                missing=sorted(set(physical_clusters) - observed_accounting_ids)[:20],
                extra=sorted(observed_accounting_ids - set(physical_clusters))[:20],
            )
        expected_counts = {
            "total": len(physical_clusters),
            "with_opinions": clusters_with_opinions,
            "without_opinions": len(physical_clusters) - clusters_with_opinions,
            "by_partition_status": status_counts,
        }
        if output_counts.get("candidate_cluster_accounting") != expected_counts:
            fail(
                "candidate cluster accounting counts differ from physical rows",
                expected=expected_counts,
                actual=output_counts.get("candidate_cluster_accounting"),
            )
    result = {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "accepted_rows": len(raw_ids),
        "rejected_rows": len(rejected_ids),
        "conflict_rows": len(conflict_ids),
        "text_sources": text_sources,
        "status": "verified",
    }
    if generation_format == FORMAT:
        result["candidate_cluster_accounting"] = output_counts[
            "candidate_cluster_accounting"
        ]
        result["correction_opinion_partitions"] = output_counts[
            "correction_opinion_partitions"
        ]
    return result


def _populate_generation(
    args,
    prepared: dict,
    generation: GenerationPublisher,
    producer: dict,
) -> None:
    source_provenance = load_and_verify_bulk_sources(args, prepared)
    authoritative_documents = load_authoritative_generation(
        args.authoritative_documents_generation,
        opinions_archive_sha256=source_provenance["opinions"]["archive_sha256"],
        snapshot_date=args.snapshot_date,
    )
    corrections = load_corrections(
        args.corrections,
        archive_sha256=source_provenance["opinions"]["archive_sha256"],
        snapshot_date=args.snapshot_date,
    )
    correction_binding = corrections["manifest"]["source_binding"]
    for role in ("dockets", "clusters", "opinions"):
        expected = correction_binding["%s_archive_sha256" % role]
        actual = source_provenance[role]["archive_sha256"]
        if expected != actual:
            raise CorrectionError(
                "correction manifest archive binding mismatch",
                role=role,
                expected=expected,
                actual=actual,
            )

    dockets_output = generation.open_text(DOCKETS_FILE)
    dockets, docket_counts = scan_dockets(args.dockets, dockets_output)
    dockets_output.flush()

    clusters_output = generation.open_text(CLUSTERS_FILE)
    clusters, cluster_counts, applied = scan_clusters(
        args.clusters, dockets, corrections, clusters_output
    )
    clusters_output.flush()

    spool_path = generation.path("_candidate_opinions.spool.jsonl")
    with open(spool_path, "x", encoding="utf-8", newline="\n") as spool_output:
        states, evidence_counts = evidence_scan(
            args.opinions,
            clusters,
            authoritative_documents,
            spool_output,
        )
        flush_and_sync(spool_output)
    evidence_counts["candidate_spool_bytes"] = spool_path.stat().st_size
    evidence_counts["candidate_spool_sha256"] = sha256_file(spool_path)
    raw_output = generation.open_text(RAW_FILE)
    idmap_output = generation.open_text(IDMAP_FILE, newline="")
    rejection_output = generation.open_text(REJECTIONS_FILE)
    conflict_output = generation.open_text(CONFLICTS_FILE)
    candidate_cluster_accounting_output = generation.open_text(
        CANDIDATE_CLUSTER_ACCOUNTING_FILE
    )
    output_counts = write_opinions(
        str(spool_path),
        evidence_counts["candidate_spool_sha256"],
        evidence_counts["candidate_spool_bytes"],
        clusters,
        states,
        source_provenance,
        args.snapshot_date,
        args.acquired_at,
        raw_output,
        idmap_output,
        rejection_output,
        conflict_output,
        candidate_cluster_accounting_output,
        corrections,
    )
    if output_counts["spool_rows_read"] != evidence_counts["candidate_rows"]:
        fail(
            "candidate opinion spool row count changed before output",
            expected=evidence_counts["candidate_rows"],
            actual=output_counts["spool_rows_read"],
        )
    if output_counts["spool_resolved_rows"] != evidence_counts["resolved_rows"]:
        fail(
            "candidate spool resolved-text count changed before output",
            expected=evidence_counts["resolved_rows"],
            actual=output_counts["spool_resolved_rows"],
        )
    spool_path.unlink()
    counts = {
        "dockets": docket_counts,
        "clusters": cluster_counts,
        "evidence_scan": evidence_counts,
        "output": output_counts,
    }
    generation.write_json(COUNTS_FILE, counts)
    final_correction_sha256 = correction_manifest_sha256(args.corrections)
    if final_correction_sha256 != corrections["source_sha256"]:
        raise CorrectionError(
            "correction manifest changed after its exact bytes were parsed",
            parsed_sha256=corrections["source_sha256"],
            final_sha256=final_correction_sha256,
        )
    if producer_contract() != producer:
        fail("extractor producer bytes changed during the bound source scan")
    generation.publish(
        {
            "format": FORMAT,
            "normalized_schema_version": FORMAT,
            "producer": producer,
            "selector_version": SELECTOR_VERSION,
            "text_policy_version": TEXT_POLICY_VERSION,
            "correction_policy_version": CORRECTION_POLICY_VERSION,
            "source_snapshot_date": args.snapshot_date,
            "source_acquired_at": args.acquired_at,
            "source_archives": source_provenance,
            "authoritative_documents": {
                "generation": str(
                    Path(args.authoritative_documents_generation).absolute()
                ),
                "manifest_sha256": authoritative_documents["manifest_sha256"],
                "format": authoritative_documents["manifest"]["format"],
                "opinion_ids": sorted(authoritative_documents["by_opinion"]),
            },
            "correction_manifest": {
                "name": Path(args.corrections).name,
                "sha256": corrections["source_sha256"],
                "corrections_applied": sorted(applied),
            },
            "row_counts": counts,
            "source_of_truth": MANIFEST_FILE,
        }
    )


def build(args) -> dict:
    producer = producer_contract()
    prepared = prepare_bulk_sources(args)
    with SourceScanLock(
        prepared["identity"],
        physical_sources=prepared["physical_sources"],
        destination=args.out,
        repo_root=Path(__file__).resolve().parents[2],
    ) as source_lock:
        sys.stderr.write(
            "source-lock: acquired physical_key=%s physical_path=%s "
            "semantic_key=%s semantic_path=%s\n"
            % (
                source_lock.physical_key,
                source_lock.physical_lock_path,
                source_lock.key,
                source_lock.semantic_lock_path,
            )
        )
        sys.stderr.flush()
        source_lock.assert_sources_unchanged()
        with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
            _populate_generation(args, prepared, generation, producer)
        source_lock.assert_sources_unchanged()
        # Keep the kernel lock held through a separate read of the final
        # generation, not merely through the publish return value.
        result = verify_extract_generation(args.out)
        if producer_contract() != producer:
            fail("extractor producer bytes changed before final readback")
        source_lock.assert_sources_unchanged()
        return result


def main() -> None:
    _set_csv_field_limit()
    parser = StructuredArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser(
        "build", help="build and atomically publish one generation"
    )
    command.add_argument("--dockets", required=True)
    command.add_argument("--clusters", required=True)
    command.add_argument("--opinions", required=True)
    command.add_argument("--bulk-manifest", required=True)
    command.add_argument("--authoritative-documents-generation", required=True)
    command.add_argument("--snapshot-date", required=True)
    command.add_argument("--acquired-at", required=True)
    command.add_argument(
        "--corrections",
        default=str(Path(__file__).with_name("cuyahoga_corrections.v2.json")),
    )
    command.add_argument("--out", required=True)
    verify = subcommands.add_parser(
        "verify", help="independently read and verify a generation"
    )
    verify.add_argument("--generation", required=True)
    args = parse_cli_args(parser)
    signal_audit = install_signal_audit()
    signal_audit.update_progress(args.command, 0)
    try:
        if args.command == "build":
            result = build(args)
        else:
            result = verify_extract_generation(args.generation)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    except BaseException as error:
        external = isinstance(error, ExternalSignal)
        write_error(
            error,
            code=EXTERNAL_SIGNAL_CODE if external else "cuyahoga_extraction_error",
            remediation=(
                "identify and correct the recorded sender_pid/sender_uid before rerunning "
                "the complete bound generation at a new destination"
                if external
                else "inspect the recorded physical source or generation mismatch, then "
                "rebuild the complete bound generation at a new destination"
            ),
            context=error.record if external else None,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
