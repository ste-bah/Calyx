#!/usr/bin/env python3
"""Audit reviewed Cuyahoga case/docket corrections against compressed source bytes."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from cuyahoga_contract import CorrectionError, apply_correction, load_corrections
from structured_error import StructuredArgumentParser, StructuredError, write_error


class AuditError(StructuredError):
    code = "correction_source_audit_failed"


class AuditArgumentParser(StructuredArgumentParser):
    def error(self, message: str) -> None:
        fail(
            message,
            remediation="provide all immutable archive, correction, and generation paths",
        )


def fail(message: str, *, remediation: str, **context):
    raise AuditError(message, remediation=remediation, **context)


def plain_file(value: str, *, label: str) -> Path:
    path = Path(value).absolute()
    if not path.is_file() or path.is_symlink():
        fail(
            "%s is not a plain file" % label,
            remediation="provide the exact immutable source-of-truth file",
            path=str(path),
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def source_rows(
    path: Path,
    *,
    expected_sha256: str,
    target_ids: set[int],
    role: str,
    bzip2: str,
) -> dict:
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        fail(
            "%s archive digest differs from the correction binding" % role,
            remediation="restore the immutable 2026-06-30 archive or commission a new correction generation",
            path=str(path),
            expected_sha256=expected_sha256,
            actual_sha256=observed_sha256,
        )
    process = subprocess.Popen(
        [bzip2, "-dc", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        fail(
            "could not open the bzip2 source stream",
            remediation="install a working bzip2 runtime and retry",
            role=role,
        )
    rows: dict[int, dict] = {}
    scanned = 0
    try:
        with io.TextIOWrapper(process.stdout, encoding="utf-8", newline="") as stream:
            set_csv_field_limit()
            reader = csv.DictReader(
                stream,
                quotechar='"',
                escapechar="\\",
                doublequote=False,
                strict=True,
            )
            if reader.fieldnames is None or "id" not in reader.fieldnames:
                fail(
                    "%s archive CSV header has no id field" % role,
                    remediation="use the exact CourtListener bulk archive",
                    header=reader.fieldnames,
                )
            for record_number, row in enumerate(reader, 2):
                scanned += 1
                try:
                    row_id = int(row["id"])
                except (TypeError, ValueError):
                    fail(
                        "%s archive contains a non-integer row id" % role,
                        remediation="replace the corrupt source archive",
                        record_number=record_number,
                        value=row.get("id"),
                    )
                if row_id not in target_ids:
                    continue
                if row_id in rows:
                    fail(
                        "%s archive repeats a reviewed source id" % role,
                        remediation="quarantine the source snapshot and investigate duplicate identities",
                        row_id=row_id,
                        record_number=record_number,
                    )
                rows[row_id] = {
                    "logical_record_number": record_number,
                    "physical_line": reader.line_num,
                    "values": row,
                    "field_sha256": {
                        key: sha256_text(value) for key, value in row.items()
                    },
                }
    except BaseException:
        process.kill()
        process.wait()
        raise
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        fail(
            "%s archive decompression failed" % role,
            remediation="verify the compressed archive and bzip2 runtime",
            return_code=return_code,
            stderr=stderr[-4000:],
        )
    missing = target_ids - set(rows)
    if missing:
        fail(
            "%s archive is missing reviewed source rows" % role,
            remediation="do not apply the correction to a different source snapshot",
            missing=sorted(missing),
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "records_scanned": scanned,
        "selected": rows,
    }


def audit(args: argparse.Namespace) -> dict:
    corrections_path = plain_file(args.corrections, label="corrections")
    dockets_path = plain_file(args.dockets_archive, label="dockets archive")
    clusters_path = plain_file(args.clusters_archive, label="clusters archive")
    protected_manifest = plain_file(
        args.protected_generation_manifest, label="protected generation manifest"
    )
    bzip2 = shutil.which(args.bzip2)
    if bzip2 is None:
        fail(
            "bzip2 runtime is unavailable",
            remediation="install bzip2 and rerun the source-byte audit",
            executable=args.bzip2,
        )
    loaded = load_corrections(
        str(corrections_path),
        archive_sha256=args.opinions_archive_sha256,
        snapshot_date=args.snapshot_date,
    )
    correction_source_sha256 = loaded["source_sha256"]
    binding = loaded["manifest"]["source_binding"]
    target_dockets = set(loaded["by_docket"])
    target_clusters = {
        correction["cluster_id"] for correction in loaded["by_docket"].values()
    }
    protected_before = sha256_file(protected_manifest)
    with ThreadPoolExecutor(max_workers=2) as executor:
        docket_future = executor.submit(
            source_rows,
            dockets_path,
            expected_sha256=binding["dockets_archive_sha256"],
            target_ids=target_dockets,
            role="dockets",
            bzip2=bzip2,
        )
        cluster_future = executor.submit(
            source_rows,
            clusters_path,
            expected_sha256=binding["clusters_archive_sha256"],
            target_ids=target_clusters,
            role="clusters",
            bzip2=bzip2,
        )
        dockets = docket_future.result()
        clusters = cluster_future.result()

    reviewed = []
    mismatch_failures = []
    for docket_id, correction in sorted(loaded["by_docket"].items()):
        cluster_id = correction["cluster_id"]
        docket_source = dockets["selected"][docket_id]
        cluster_source = clusters["selected"][cluster_id]
        raw_record = {
            "cluster_id": cluster_id,
            "case_name": cluster_source["values"]["case_name"],
            "docket_number": docket_source["values"]["docket_number"],
        }
        applied_record = dict(raw_record)
        application = apply_correction(
            applied_record,
            correction,
            where="direct compressed source rows",
        )
        expected_opinions = correction["expected_opinions"]
        reviewed.append(
            {
                "correction_id": correction["correction_id"],
                "docket_id": docket_id,
                "cluster_id": cluster_id,
                "opinion_partitions": expected_opinions,
                "docket_source_record": docket_source["logical_record_number"],
                "docket_source_physical_line": docket_source["physical_line"],
                "cluster_source_record": cluster_source["logical_record_number"],
                "cluster_source_physical_line": cluster_source["physical_line"],
                "raw": raw_record,
                "raw_field_sha256": {
                    "case_name": cluster_source["field_sha256"]["case_name"],
                    "docket_number": docket_source["field_sha256"]["docket_number"],
                },
                "corrected": applied_record,
                "application": application,
            }
        )
        already_corrected = dict(applied_record)
        try:
            apply_correction(
                already_corrected,
                correction,
                where="real already-corrected source mutation",
            )
        except CorrectionError as error:
            mismatch_failures.append(error.record())
        else:
            fail(
                "an already-corrected source bypassed the source-value guard",
                remediation="restore exact source-value validation before publishing",
                correction_id=correction["correction_id"],
            )
    protected_after = sha256_file(protected_manifest)
    if protected_before != protected_after:
        fail(
            "the protected generation manifest changed during a read-only audit",
            remediation="quarantine the generation and investigate the unexpected mutation",
            before=protected_before,
            after=protected_after,
        )
    correction_after = sha256_file(corrections_path)
    if correction_after != correction_source_sha256:
        fail(
            "the correction manifest changed after its exact bytes were parsed",
            remediation="rerun against one immutable correction manifest byte object",
            parsed_sha256=correction_source_sha256,
            final_sha256=correction_after,
        )
    return {
        "status": "verified",
        "policy": loaded["manifest"]["policy"],
        "snapshot_date": args.snapshot_date,
        "correction_manifest": {
            "path": str(corrections_path),
            "sha256": correction_source_sha256,
            "sha256_after": correction_after,
            "byte_identical": True,
            "count": len(reviewed),
        },
        "archives": {
            "dockets": {
                key: value for key, value in dockets.items() if key != "selected"
            },
            "clusters": {
                key: value for key, value in clusters.items() if key != "selected"
            },
            "opinions_sha256_binding": args.opinions_archive_sha256,
        },
        "reviewed_corrections": reviewed,
        "fail_closed_real_mutations": {
            "attempted": len(reviewed),
            "refused": len(mismatch_failures),
            "errors": mismatch_failures,
        },
        "protected_generation_manifest": {
            "path": str(protected_manifest),
            "sha256_before": protected_before,
            "sha256_after": protected_after,
            "byte_identical": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    root = AuditArgumentParser(description=__doc__)
    root.add_argument("--dockets-archive", required=True)
    root.add_argument("--clusters-archive", required=True)
    root.add_argument("--corrections", required=True)
    root.add_argument("--protected-generation-manifest", required=True)
    root.add_argument("--opinions-archive-sha256", required=True)
    root.add_argument("--snapshot-date", required=True)
    root.add_argument("--bzip2", default="bzip2")
    return root


def main() -> int:
    try:
        result = audit(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except SystemExit as error:
        if error.code == 0:
            return 0
        raise
    except (AuditError, CorrectionError) as error:
        write_error(
            error,
            code="correction_source_audit_failed",
            remediation=(
                "do not publish; restore the exact reviewed source or correction contract"
            ),
            include_traceback=False,
        )
        return 1
    except Exception as error:
        write_error(
            error,
            code="correction_source_audit_unhandled",
            remediation="inspect the traceback and add a typed fail-closed audit path",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
