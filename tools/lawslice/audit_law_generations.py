#!/usr/bin/env python3
"""Cross-generation physical readback for the Cuyahoga law pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from build_ingest_jsonl import (
    MANIFEST_FILE as INGEST_MANIFEST,
    verify_ingest_generation,
)
from build_judge_tables import (
    MANIFEST_FILE as JUDGE_MANIFEST,
    verify_judge_generation,
)
from extract_citations import (
    MANIFEST_FILE as CITATION_MANIFEST,
    verify_citation_generation,
)
from extract_cuyahoga import (
    MANIFEST_FILE as EXTRACT_MANIFEST,
    verify_extract_generation,
)
from law_generation import GenerationPublisher, sha256_file, verify_generation
from structured_error import StructuredArgumentParser, parse_cli_args, write_error


FORMAT = "calyx-cuyahoga-cross-generation-audit-v2"
MANIFEST_FILE = "audit_manifest.json"
REPORT_FILE = "provenance_audit.json"


def source_manifest(path: str, name: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, name)
    return {
        "directory": str(root),
        "manifest": name,
        "manifest_sha256": sha256_file(root / name),
        "format": manifest.get("format"),
        "source_of_truth": manifest.get("source_of_truth"),
    }


def judge_source_manifest(path: str, extract_directory: str) -> dict:
    root = Path(path).absolute()
    value = source_manifest(str(root), JUDGE_MANIFEST)
    manifest = verify_generation(root, JUDGE_MANIFEST)
    binding = manifest.get("source_extract_generation")
    if not isinstance(binding, dict):
        raise RuntimeError("judge generation is missing its extract binding")
    expected_extract = Path(extract_directory).absolute()
    if binding.get("directory") != str(expected_extract):
        raise RuntimeError(
            "judge generation is bound to a different extract generation: %r"
            % binding.get("directory")
        )
    value["source_extract_generation"] = binding
    return value


def verify_audit_generation(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        raise RuntimeError(
            "unsupported audit generation format: %r" % manifest.get("format")
        )
    if set(manifest["files"]) != {REPORT_FILE}:
        raise RuntimeError("audit generation member contract mismatch")
    with open(root / REPORT_FILE, "r", encoding="utf-8") as source:
        report = json.load(source)
    if report.get("status") != "verified":
        raise RuntimeError("physical audit report does not say verified")
    if report.get("generations") != manifest.get("generations"):
        raise RuntimeError("physical audit report differs from audit manifest")
    recorded = report["generations"]
    if set(recorded) != {"extract", "ingest", "citations", "judges"}:
        raise RuntimeError("physical audit report generation set is invalid")
    extract_directory = recorded["extract"].get("directory")
    ingest_directory = recorded["ingest"].get("directory")
    citation_directory = recorded["citations"].get("directory")
    judge_directory = recorded["judges"].get("directory")
    actual = {
        "extract": {
            **source_manifest(extract_directory, EXTRACT_MANIFEST),
            "readback": verify_extract_generation(extract_directory),
        },
        "ingest": {
            **source_manifest(ingest_directory, INGEST_MANIFEST),
            "readback": verify_ingest_generation(ingest_directory, extract_directory),
        },
        "citations": {
            **source_manifest(citation_directory, CITATION_MANIFEST),
            "readback": verify_citation_generation(
                citation_directory, ingest_directory
            ),
        },
        "judges": {
            **judge_source_manifest(judge_directory, extract_directory),
            "readback": verify_judge_generation(judge_directory),
        },
    }
    if actual != recorded:
        raise RuntimeError("source generation state differs from sealed audit report")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "status": "verified",
        "generations": report["generations"],
    }


def build(args) -> dict:
    extract = verify_extract_generation(args.extract_generation)
    ingest = verify_ingest_generation(args.ingest_generation, args.extract_generation)
    citations = verify_citation_generation(
        args.citation_generation, args.ingest_generation
    )
    judges = verify_judge_generation(args.judge_generation)
    generations = {
        "extract": {
            **source_manifest(args.extract_generation, EXTRACT_MANIFEST),
            "readback": extract,
        },
        "ingest": {
            **source_manifest(args.ingest_generation, INGEST_MANIFEST),
            "readback": ingest,
        },
        "citations": {
            **source_manifest(args.citation_generation, CITATION_MANIFEST),
            "readback": citations,
        },
        "judges": {
            **judge_source_manifest(args.judge_generation, args.extract_generation),
            "readback": judges,
        },
    }
    report = {
        "format": FORMAT,
        "status": "verified",
        "generations": generations,
        "proof": {
            "extract": "physical manifest/members, decisions, IDs, text digests, source provenance, corrections",
            "ingest": "physical canonical content, all source aliases, raw-to-alias digest equality",
            "citations": "physical source edges resolved through the physical alias relation",
            "judges": "physical opinion-author mapping independently recomputed from the extract and bound people/position archives",
        },
    }
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        generation.write_json(REPORT_FILE, report)
        generation.publish(
            {
                "format": FORMAT,
                "generations": generations,
                "source_of_truth": MANIFEST_FILE,
            }
        )
    return verify_audit_generation(args.out)


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("build")
    command.add_argument("--extract-generation", required=True)
    command.add_argument("--ingest-generation", required=True)
    command.add_argument("--citation-generation", required=True)
    command.add_argument("--judge-generation", required=True)
    command.add_argument("--out", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--generation", required=True)
    args = parse_cli_args(parser)
    try:
        report = (
            build(args)
            if args.command == "build"
            else verify_audit_generation(args.generation)
        )
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_cross_generation_audit_error",
            remediation=(
                "repair the named immutable generation mismatch and rerun the complete audit"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
