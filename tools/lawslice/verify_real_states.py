#!/usr/bin/env python3
"""Publish before/action/after state proofs from sealed real source regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback

from build_ingest_jsonl import canonical_preference
from cuyahoga_contract import (
    ContractError,
    EvidenceConflictError,
    TEXT_FIELDS,
    apply_correction,
    classify,
    direct_evidence,
    load_corrections,
    official_rod_evidence,
    resolve_text,
)
from law_generation import (
    GenerationError,
    GenerationPublisher,
    sha256_file,
    verify_generation,
)
from structured_error import StructuredArgumentParser, parse_cli_args, write_error


FORMAT = "calyx-cuyahoga-real-state-verification-v1"
MANIFEST_FILE = "state_verification_manifest.json"
EVENTS_FILE = "state_transitions.jsonl"
REPORT_FILE = "state_verification_report.json"
OPINIONS_SHA256 = "65db4902a4ae42c48ef9958f52600dedca5401c194d3611d0b2672cfc6dd4d9c"


def real_cases(path: Path) -> dict[int, dict]:
    verify_generation(path, "regression_manifest.json")
    with open(path / "opinion_regressions.json", "r", encoding="utf-8") as source:
        value = json.load(source)
    return {row["opinion_id"]: row for row in value["cases"]}


def fields(case: dict) -> dict:
    value = {name: "" for name in TEXT_FIELDS}
    value.update(case["authoritative_excerpt"])
    return value


def evidence(case: dict):
    resolved = resolve_text(fields(case), where="opinion %d" % case["opinion_id"])
    return resolved, direct_evidence(
        case["courtlistener_source"]["download_url"], resolved
    )


def append(events: list, name: str, before: dict, action: dict, after: dict) -> None:
    events.append(
        {"name": name, "before": before, "action": action, "after": after}
    )


def build_events(cases: dict[int, dict], corrections_path: str) -> tuple[list[dict], str]:
    events = []

    for opinion_id, name in (
        (4636687, "happy_direct_eighth"),
        (3713167, "edge_body_only_cuyahoga"),
        (4678958, "edge_explicit_tenth"),
        (8621357, "edge_exact_cap_allowlist"),
        (6227032, "edge_maximum_length"),
    ):
        case = cases[opinion_id]
        resolved, own = evidence(case)
        decision = classify(
            court_id=case["court_id"],
            own_evidence=own,
            sibling_evidence=[],
            where=name,
        )
        if decision != case["expected_decision"]:
            raise RuntimeError("%s decision differs from sealed real evidence" % name)
        append(
            events,
            name,
            {
                "opinion_id": opinion_id,
                "court_id": case["court_id"],
                "download_url": case["courtlistener_source"]["download_url"],
                "excerpt_contains_cuyahoga": "cuyahoga" in resolved.text.lower(),
                "recorded_full_chars": case["full_normalized_chars"],
            },
            {
                "selector": "classify",
                "evidence": [item.record() for item in own],
            },
            {
                "status": decision["status"],
                "reason": decision["reason"],
                "district": decision["district"],
                "selected_text_field": resolved.source_field,
                "excerpt_normalized_sha256": resolved.normalized_sha256,
            },
        )

    cap_case = cases[8621357]
    _, tenth_direct = evidence(cases[4678958])
    try:
        classify(
            court_id=cap_case["court_id"],
            own_evidence=tenth_direct,
            sibling_evidence=[],
            where="real CAP identity / Tenth evidence conflict",
        )
    except EvidenceConflictError as error:
        cap_conflict_after = {
            "status": "conflict",
            **error.record(),
            "districts": error.context["districts"],
            "published": False,
        }
    else:
        raise RuntimeError("exact CAP identity ignored conflicting direct evidence")
    append(
        events,
        "edge_exact_cap_conflicting_direct_evidence",
        {
            "cap_opinion_id": cap_case["opinion_id"],
            "court_id": cap_case["court_id"],
            "direct_evidence_opinion_id": 4678958,
            "direct_evidence_districts": sorted(
                {item.district for item in tenth_direct}
            ),
        },
        {"operation": "classify exact CAP ID with real Tenth direct evidence"},
        cap_conflict_after,
    )

    real_rod_url = cases[4636687]["courtlistener_source"]["download_url"]
    invalid_rod_urls = {
        "district_zero": real_rod_url.replace("/pdf/8/", "/pdf/0/"),
        "explicit_port": real_rod_url.replace(
            "www.supremecourt.ohio.gov", "www.supremecourt.ohio.gov:443"
        ),
    }
    invalid_results = {
        name: [item.record() for item in official_rod_evidence(value)]
        for name, value in invalid_rod_urls.items()
    }
    if any(invalid_results.values()):
        raise RuntimeError("invalid official URL shape produced direct evidence")
    append(
        events,
        "edge_invalid_official_url_shapes",
        {"real_url": real_rod_url, "mutations": invalid_rod_urls},
        {"operation": "parse strict official ROD district URL"},
        {"status": "rejected", "evidence": invalid_results},
    )

    for opinion_id in (11227531, 11257339):
        case = cases[opinion_id]
        before = {
            "opinion_id": opinion_id,
            "download_url": case["courtlistener_source"]["download_url"],
            "text_fields": {
                name: facts["nonempty"] for name, facts in case["text_fields"].items()
            },
        }
        try:
            resolve_text(fields(case), where="empty opinion %d" % opinion_id)
        except ContractError as error:
            after = {
                "status": "error",
                **error.record(),
                "published": False,
            }
        else:
            raise RuntimeError("empty real opinion unexpectedly resolved")
        append(
            events,
            "edge_empty_authoritative_%d" % opinion_id,
            before,
            {"operation": "resolve_text"},
            after,
        )

    _, eighth = evidence(cases[4636687])
    _, tenth = evidence(cases[4678958])
    try:
        classify(
            court_id="ohioctapp",
            own_evidence=eighth + tenth,
            sibling_evidence=[],
            where="real evidence conflict",
        )
    except EvidenceConflictError as error:
        conflict_after = {
            "status": "conflict",
            **error.record(),
            "districts": error.context["districts"],
            "published": False,
        }
    else:
        raise RuntimeError("real Eighth/Tenth conflict was not rejected")
    append(
        events,
        "edge_conflicting_direct_evidence",
        {
            "eighth_opinion_id": 4636687,
            "tenth_opinion_id": 4678958,
            "districts": sorted({item.district for item in eighth + tenth}),
        },
        {"operation": "classify combined exact source evidence"},
        conflict_after,
    )

    malformed_case = cases[11173418]
    malformed_fields = fields(malformed_case)
    malformed_fields["html_with_citations"] += "<script>unterminated"
    try:
        resolve_text(malformed_fields, where="malformed real HTML mutation")
    except ContractError as error:
        malformed_after = {
            "status": "error",
            **error.record(),
            "plain_text_used": False,
        }
    else:
        raise RuntimeError("malformed preferred HTML used plain text")
    append(
        events,
        "edge_malformed_preferred_html",
        {
            "opinion_id": 11173418,
            "preferred_html_nonempty": True,
            "plain_text_nonempty": True,
            "captured_source_row_sha256": malformed_case["source_row_sha256"],
        },
        {"operation": "append invalid unclosed script to captured HTML excerpt"},
        malformed_after,
    )

    plain = cases[11171895]
    html = cases[11173418]
    if plain["full_normalized_sha256"] != html["full_normalized_sha256"]:
        raise RuntimeError("sealed duplicate source texts no longer match")
    preferences = {
        case["opinion_id"]: canonical_preference(
            {"text_source": case["full_source_field"]}, case["opinion_id"]
        )
        for case in (plain, html)
    }
    canonical_id = min(preferences, key=preferences.get)
    if canonical_id != 11173418:
        raise RuntimeError("preferred HTML duplicate was not canonical")
    append(
        events,
        "happy_duplicate_alias_relation",
        {
            "source_opinion_ids": sorted(preferences),
            "full_normalized_sha256": plain["full_normalized_sha256"],
            "source_fields": {
                str(case["opinion_id"]): case["full_source_field"]
                for case in (plain, html)
            },
        },
        {"operation": "canonical_preference"},
        {
            "canonical_opinion_id": canonical_id,
            "alias_opinion_ids": sorted(preferences),
            "source_identities_preserved": 2,
        },
    )

    corrections = load_corrections(
        corrections_path,
        archive_sha256=OPINIONS_SHA256,
        snapshot_date="2026-06-30",
    )
    correction_source_sha256 = corrections["source_sha256"]
    for docket_id, correction in sorted(corrections["by_docket"].items()):
        record = {
            "cluster_id": correction["cluster_id"],
            "case_name": correction["changes"]["case_name"]["from"],
            "docket_number": correction["changes"]["docket_number"]["from"],
        }
        before = dict(record)
        audit = apply_correction(
            record, correction, where="physical correction %d" % docket_id
        )
        append(
            events,
            "happy_exact_correction_%d" % docket_id,
            before,
            {"operation": "apply_correction", "audit": audit},
            record,
        )

    correction_value = json.loads(Path(corrections_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "generation"
        publication_before = {"destination_exists": destination.exists()}
        with GenerationPublisher(destination, "manifest.json") as publisher:
            publisher.write_json("corrections.json", correction_value)
            publisher.publish(
                {
                    "format": "real-publication-state-test-v1",
                    "source_of_truth": "manifest.json",
                }
            )
        readback = verify_generation(destination, "manifest.json")
        append(
            events,
            "happy_durable_publication",
            publication_before,
            {"operation": "publish and independently verify"},
            {
                "destination_exists": destination.is_dir(),
                "files": readback["files"],
            },
        )
        before_collision = {
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "member_sha256": sha256_file(destination / "corrections.json"),
        }
        try:
            GenerationPublisher(destination, "manifest.json")
        except GenerationError as error:
            collision_message = str(error)
            collision_error_type = type(error).__name__
            expected_prefix = (
                "generation destination already exists; refusing overwrite: "
            )
            if not collision_message.startswith(expected_prefix):
                raise RuntimeError(
                    "existing destination failed for an unexpected reason: %s"
                    % collision_message
                ) from error
        else:
            raise RuntimeError("existing generation was accepted")
        after_collision = {
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "member_sha256": sha256_file(destination / "corrections.json"),
            "error_type": collision_error_type,
            "error": "generation destination already exists; refusing overwrite",
            "destination_is_existing_directory": destination.is_dir(),
        }
        if before_collision["member_sha256"] != after_collision["member_sha256"]:
            raise RuntimeError("existing generation changed after collision")
        append(
            events,
            "edge_existing_destination",
            before_collision,
            {"operation": "construct publisher at existing destination"},
            after_collision,
        )
    correction_after = sha256_file(corrections_path)
    if correction_after != correction_source_sha256:
        raise RuntimeError(
            "correction manifest changed after its exact bytes were parsed: "
            f"parsed={correction_source_sha256} final={correction_after}"
        )
    return events, correction_source_sha256


def verify_output(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        raise RuntimeError("wrong state-verification format")
    with open(root / EVENTS_FILE, "r", encoding="utf-8") as source:
        events = [json.loads(line) for line in source]
    with open(root / REPORT_FILE, "r", encoding="utf-8") as source:
        report = json.load(source)
    if len(events) != report["event_count"] or len(events) != manifest["event_count"]:
        raise RuntimeError("physical state event count mismatch")
    if {item["name"] for item in events} != set(report["events"]):
        raise RuntimeError("physical state event names mismatch")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "event_count": len(events),
        "events": [item["name"] for item in events],
        "status": "verified",
    }


def build(args) -> dict:
    cases = real_cases(Path(args.real_regressions))
    events, correction_source_sha256 = build_events(cases, args.corrections)
    report = {
        "format": FORMAT,
        "status": "verified",
        "event_count": len(events),
        "events": [item["name"] for item in events],
        "real_regression_manifest_sha256": sha256_file(
            Path(args.real_regressions) / "regression_manifest.json"
        ),
        "correction_manifest_sha256": correction_source_sha256,
    }
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        output = generation.open_text(EVENTS_FILE)
        for event in events:
            output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        generation.write_json(REPORT_FILE, report)
        generation.publish(
            {
                **report,
                "source_of_truth": MANIFEST_FILE,
            }
        )
    return verify_output(args.out)


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("build")
    command.add_argument(
        "--real-regressions",
        default=str(Path(__file__).with_name("real_regressions")),
    )
    command.add_argument(
        "--corrections",
        default=str(Path(__file__).with_name("cuyahoga_corrections.v2.json")),
    )
    command.add_argument("--out", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--generation", required=True)
    args = parse_cli_args(parser)
    try:
        report = build(args) if args.command == "build" else verify_output(args.generation)
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_real_state_verification_error",
            remediation=(
                "repair the sealed real-state evidence mismatch and rerun at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
