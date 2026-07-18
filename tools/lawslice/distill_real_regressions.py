#!/usr/bin/env python3
"""Distill bounded, exact regression evidence from captured CourtListener rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from capture_real_opinions import (
    MANIFEST_FILE as CAPTURE_MANIFEST,
    ROWS_FILE,
)
from cuyahoga_contract import (
    ContractError,
    TEXT_FIELDS,
    classify,
    direct_evidence,
    resolve_text,
    sha256_text,
)
from law_generation import (
    GenerationPublisher,
    generation_member,
    sha256_file,
    verify_generation,
)
from structured_error import StructuredArgumentParser, parse_cli_args, write_error


FORMAT = "calyx-courtlistener-real-regressions-v1"
MANIFEST_FILE = "regression_manifest.json"
CASES_FILE = "opinion_regressions.json"
EXCERPT_CHARS = 12_000


def stable_row_sha256(row: dict) -> str:
    encoded = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_current_metadata(path: str, clusters_path: str) -> tuple[dict, dict]:
    result = {}
    with open(path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            opinion_id = int(row["opinion_id"])
            if opinion_id in result:
                raise RuntimeError("duplicate production opinion_id %d" % opinion_id)
            result[opinion_id] = {
                key: row.get(key)
                for key in (
                    "opinion_id",
                    "cluster_id",
                    "docket_id",
                    "court_id",
                    "case_name",
                    "docket_number",
                )
            }
    clusters = {}
    with open(clusters_path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            cluster_id = int(row["cluster_id"])
            if cluster_id in clusters:
                raise RuntimeError("duplicate cluster_id %d" % cluster_id)
            clusters[cluster_id] = row
    return result, clusters


def verify_plain_file(path: str, expected_sha256: str, *, role: str) -> str:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("%s is not a plain file: %s" % (role, source))
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise RuntimeError(
            "%s SHA-256 mismatch: expected=%s actual=%s"
            % (role, expected_sha256, actual)
        )
    return actual


def distill(args):
    capture = verify_generation(args.capture_generation, CAPTURE_MANIFEST)
    rows_path = generation_member(args.capture_generation, CAPTURE_MANIFEST, ROWS_FILE)
    current_raw_sha256 = verify_plain_file(
        args.current_raw, args.current_raw_sha256, role="current raw extraction"
    )
    clusters_map_sha256 = verify_plain_file(
        args.clusters_map, args.clusters_map_sha256, role="current cluster map"
    )
    metadata, clusters = load_current_metadata(args.current_raw, args.clusters_map)
    cases = []
    with open(rows_path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            row = json.loads(line)
            opinion_id = int(row["id"])
            current = metadata.get(opinion_id)
            if current is None:
                cluster_id = int(row["cluster_id"])
                cluster = clusters.get(cluster_id)
                if cluster is None:
                    raise RuntimeError(
                        "captured opinion %d cluster %d absent from physical source map"
                        % (opinion_id, cluster_id)
                    )
                current = {
                    "opinion_id": opinion_id,
                    "cluster_id": cluster_id,
                    "docket_id": cluster["docket_id"],
                    "court_id": cluster["court_id"],
                    "case_name": cluster["case_name"],
                    "docket_number": cluster["docket_number"],
                }
            field_state = {
                name: {
                    "nonempty": bool(row.get(name, "").strip()),
                    "chars": len(row.get(name, "")),
                    "sha256": sha256_text(row[name]) if row.get(name) else None,
                }
                for name in TEXT_FIELDS
            }
            case = {
                **current,
                "courtlistener_source": {
                    key: row.get(key) or None
                    for key in (
                        "sha1",
                        "download_url",
                        "local_path",
                        "date_created",
                        "date_modified",
                    )
                },
                "source_row_sha256": stable_row_sha256(row),
                "text_fields": field_state,
            }
            try:
                resolved = resolve_text(row, where="captured opinion %d" % opinion_id)
            except ContractError as error:
                case["authoritative_text_error"] = error.record()
                case["authoritative_excerpt"] = {
                    "html_with_citations": row.get("html_with_citations", ""),
                    "plain_text": row.get("plain_text", ""),
                }
                case["expected_decision"] = None
            else:
                excerpt_values = {name: "" for name in TEXT_FIELDS}
                for candidate in ("html_with_citations", "plain_text"):
                    if row.get(candidate):
                        excerpt_values[candidate] = row[candidate][:EXCERPT_CHARS]
                excerpt = resolve_text(
                    excerpt_values, where="excerpt opinion %d" % opinion_id
                )
                evidence = direct_evidence(row.get("download_url"), excerpt)
                decision = classify(
                    court_id=current["court_id"],
                    own_evidence=evidence,
                    sibling_evidence=[],
                    where="excerpt opinion %d" % opinion_id,
                )
                case.update(
                    {
                        "authoritative_text_error": None,
                        "authoritative_excerpt": {
                            name: excerpt_values[name]
                            for name in ("html_with_citations", "plain_text")
                            if excerpt_values[name]
                        },
                        "excerpt_source_field": excerpt.source_field,
                        "excerpt_sha256": excerpt.source_raw_sha256,
                        "full_source_field": resolved.source_field,
                        "full_source_raw_sha256": resolved.source_raw_sha256,
                        "full_normalized_sha256": resolved.normalized_sha256,
                        "full_normalized_chars": resolved.normalized_chars,
                        "expected_evidence": [item.record() for item in evidence],
                        "expected_decision": decision,
                    }
                )
            cases.append(case)
    cases.sort(key=lambda item: item["opinion_id"])
    value = {
        "format": FORMAT,
        "source_archive": capture["source_archive"],
        "source_capture_manifest_sha256": sha256_file(
            Path(args.capture_generation) / CAPTURE_MANIFEST
        ),
        "source_current_raw": {
            "name": Path(args.current_raw).name,
            "sha256": current_raw_sha256,
        },
        "source_cluster_map": {
            "name": Path(args.clusters_map).name,
            "sha256": clusters_map_sha256,
        },
        "excerpt_chars": EXCERPT_CHARS,
        "cases": cases,
    }
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        generation.write_json(CASES_FILE, value)
        generation.publish(
            {
                "format": FORMAT,
                "source_archive": capture["source_archive"],
                "source_capture_manifest_sha256": value[
                    "source_capture_manifest_sha256"
                ],
                "source_current_raw": value["source_current_raw"],
                "source_cluster_map": value["source_cluster_map"],
                "case_count": len(cases),
                "opinion_ids": [item["opinion_id"] for item in cases],
                "source_of_truth": MANIFEST_FILE,
            }
        )
    verify_generation(args.out, MANIFEST_FILE)
    return {
        "generation": str(Path(args.out).absolute()),
        "manifest_sha256": sha256_file(Path(args.out) / MANIFEST_FILE),
        "case_count": len(cases),
    }


def main():
    parser = StructuredArgumentParser(description=__doc__)
    parser.add_argument("--capture-generation", required=True)
    parser.add_argument("--current-raw", required=True)
    parser.add_argument("--current-raw-sha256", required=True)
    parser.add_argument("--clusters-map", required=True)
    parser.add_argument("--clusters-map-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parse_cli_args(parser)
    try:
        print(json.dumps(distill(args), sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_real_regression_distill_error",
            remediation=(
                "repair the bound capture or current-source mismatch and distill at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
