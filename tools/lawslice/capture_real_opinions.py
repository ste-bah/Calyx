#!/usr/bin/env python3
"""Capture exact CourtListener bulk rows for reproducible regression evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from extract_cuyahoga import OPINION_FIELDS, csv_rows, positive_int
from law_generation import GenerationPublisher, sha256_file, verify_generation
from structured_error import StructuredArgumentParser, parse_cli_args, write_error


FORMAT = "calyx-courtlistener-real-opinion-regressions-v1"
MANIFEST_FILE = "capture_manifest.json"
ROWS_FILE = "opinion_rows.jsonl"


def capture(args):
    wanted = set(args.opinion_id)
    if not wanted or any(value <= 0 for value in wanted):
        raise RuntimeError("--opinion-id values must be positive and nonempty")
    source_path = Path(args.opinions).absolute()
    if not source_path.is_file() or source_path.is_symlink():
        raise RuntimeError("opinions archive is not a plain file: %s" % source_path)
    actual_archive_sha256 = sha256_file(source_path)
    if actual_archive_sha256 != args.archive_sha256:
        raise RuntimeError(
            "opinions archive SHA-256 mismatch: expected=%s actual=%s"
            % (args.archive_sha256, actual_archive_sha256)
        )
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        output = generation.open_text(ROWS_FILE)
        found = set()
        scanned = 0
        for line, row in csv_rows(args.opinions, "search_opinion", OPINION_FIELDS):
            scanned += 1
            opinion_id = positive_int(
                row["id"], field="id", path=args.opinions, line=line
            )
            if opinion_id not in wanted:
                continue
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            found.add(opinion_id)
            if found == wanted:
                break
        if found != wanted:
            raise RuntimeError(
                "source scan did not find IDs: %r" % sorted(wanted - found)
            )
        generation.publish(
            {
                "format": FORMAT,
                "source_archive": {
                    "name": source_path.name,
                    "sha256": actual_archive_sha256,
                    "snapshot_date": args.snapshot_date,
                },
                "requested_opinion_ids": sorted(wanted),
                "captured_opinion_ids": sorted(found),
                "source_rows_scanned_until_complete": scanned,
                "source_of_truth": MANIFEST_FILE,
            }
        )
    manifest = verify_generation(args.out, MANIFEST_FILE)
    return {
        "generation": str(Path(args.out).absolute()),
        "manifest_sha256": sha256_file(Path(args.out) / MANIFEST_FILE),
        "captured": len(found),
        "rows_scanned": manifest["source_rows_scanned_until_complete"],
    }


def main():
    parser = StructuredArgumentParser(description=__doc__)
    parser.add_argument("--opinions", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--opinion-id", type=int, action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parse_cli_args(parser)
    try:
        print(json.dumps(capture(args), sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_real_opinion_capture_error",
            remediation=(
                "repair the bound opinion archive or requested identity set and capture at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
