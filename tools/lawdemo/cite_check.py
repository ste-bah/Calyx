#!/usr/bin/env python3
"""Run a live, fail-closed Cuyahoga brief citation and proposition check."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from cite_check_sources import (
    exhaustive_walk,
    fail,
    load_citation_graph,
    load_extract_index,
    load_fixture,
    load_ingest_index,
    load_vault_aliases,
    normalized,
    plain_file,
    verify_overlay_report,
)


LAWSLICE_DIR = Path(__file__).resolve().parent.parent / "lawslice"
if str(LAWSLICE_DIR) not in sys.path:
    sys.path.insert(0, str(LAWSLICE_DIR))

from law_generation import (  # noqa: E402
    GenerationPublisher,
    sha256_file,
    verify_generation,
)


FORMAT = "calyx-cuyahoga-citecheck-generation-v1"
MANIFEST_FILE = "cite_check_manifest.json"
REPORT_FILE = "cite_check_report.json"
ROWS_FILE = "cite_check_rows.jsonl"
RUNS_FILE = "search_runs.jsonl"
WALKS_FILE = "citation_walks.jsonl"


class CiteCheckError(RuntimeError):
    pass


def cite_fail(message: str, **context) -> None:
    if context:
        message = "%s | %s" % (message, json.dumps(context, sort_keys=True))
    raise CiteCheckError(message)


def executable(path: str) -> Path:
    binary = plain_file(path, label="calyx binary")
    if os.name == "posix" and not os.access(binary, os.X_OK):
        cite_fail("calyx binary is not executable", path=str(binary))
    return binary


def run_json(command: list[str], env: dict[str, str], *, label: str) -> tuple[dict | list, dict]:
    started = time.monotonic_ns()
    try:
        process = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=env,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        cite_fail("live Calyx command timed out", label=label, timeout_seconds=180, command=command)
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    record = {
        "label": label,
        "command": command,
        "exit_code": process.returncode,
        "elapsed_ms": elapsed_ms,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    if process.returncode != 0:
        cite_fail(
            "live Calyx command failed",
            label=label,
            exit_code=process.returncode,
            stderr=process.stderr[-4000:],
        )
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        cite_fail("live Calyx stdout is not JSON", label=label, error=str(error))
    return value, record


def search(
    binary: Path,
    vault: str,
    resident_addr: str,
    query: str,
    k: int,
    env: dict[str, str],
    *,
    label: str,
    target_pointer_fragment: str | None = None,
) -> tuple[list[dict], dict, dict]:
    command = [
        str(binary),
        "search",
        vault,
        query,
        "--k",
        str(k),
        "--fusion",
        "weighted-rrf",
        "--guard",
        "off",
        "--fresh",
        "--explain",
    ]
    if target_pointer_fragment is not None:
        command.extend(
            [
                "--filter",
                json.dumps(
                    {
                        "metadata": [
                            {"input_pointer_contains": target_pointer_fragment}
                        ]
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    command.extend(["--resident-addr", resident_addr])
    value, record = run_json(command, env, label=label)
    if not isinstance(value, dict) or not isinstance(value.get("hits"), list):
        cite_fail("live search did not return an explain envelope", label=label)
    slots = value.get("slots")
    if not isinstance(slots, dict):
        cite_fail("live search omitted serving slot roster", label=label)
    resident_gpu = slots.get("resident_gpu", [])
    local_cpu = slots.get("local_cpu", [])
    retired = slots.get("retired_excluded", [])
    if len(resident_gpu) < 10:
        cite_fail(
            "live search served fewer than ten resident GPU text lenses",
            label=label,
            active=len(resident_gpu),
        )
    if local_cpu != []:
        cite_fail("live search used a CPU content-lens fallback", label=label, active=local_cpu)
    for excluded in ("parked_excluded", "unregistered_excluded"):
        if slots.get(excluded) != []:
            cite_fail("live search excluded a frozen panel slot", label=label, category=excluded)
    active_slots = {row.get("slot") for row in resident_gpu if isinstance(row, dict)}
    retired_slots = {row.get("slot") for row in retired if isinstance(row, dict)}
    if len(active_slots) != len(resident_gpu) or active_slots.intersection(retired_slots):
        cite_fail(
            "live search active and retired slot identities are invalid",
            label=label,
            active_slots=sorted(active_slots),
            retired_slots=sorted(retired_slots),
        )
    ranks = []
    for expected_rank, hit in enumerate(value["hits"], 1):
        if not isinstance(hit, dict) or hit.get("rank") != expected_rank:
            cite_fail("live search ranks are malformed or discontinuous", label=label)
        cx_id = hit.get("cx_id")
        score = hit.get("score")
        freshness = hit.get("freshness")
        provenance = hit.get("provenance")
        if (
            not isinstance(cx_id, str)
            or not isinstance(score, (int, float))
            or not isinstance(freshness, dict)
            or freshness.get("stale_by") != 0
            or not isinstance(provenance, dict)
        ):
            cite_fail("live search hit lacks fresh physical provenance", label=label, rank=expected_rank)
        ranks.append(hit)
    record["parsed_hit_count"] = len(ranks)
    record["serving_slots"] = slots
    return ranks, slots, record


def find_hit(hits: list[dict], cx_id: str) -> dict | None:
    return next((hit for hit in hits if hit.get("cx_id") == cx_id), None)


def paired_support_score(
    identity_hit: dict,
    proposition_hit: dict,
    slots: dict,
    *,
    citation_id: str,
) -> tuple[float, list[dict]]:
    active_slots = {
        row["slot"]
        for row in slots["resident_gpu"]
        if isinstance(row, dict) and isinstance(row.get("slot"), int)
    }

    def lens_map(hit: dict, label: str) -> dict[int, dict]:
        per_lens = hit.get("per_lens")
        if not isinstance(per_lens, list):
            cite_fail("target-filtered hit omitted per-lens evidence", citation_id=citation_id, label=label)
        mapped = {}
        for row in per_lens:
            slot = row.get("slot") if isinstance(row, dict) else None
            raw = row.get("raw") if isinstance(row, dict) else None
            weight = row.get("weight") if isinstance(row, dict) else None
            if (
                not isinstance(slot, int)
                or isinstance(slot, bool)
                or not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
                or not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
                or slot in mapped
            ):
                cite_fail("target-filtered per-lens evidence is invalid", citation_id=citation_id, label=label)
            mapped[slot] = {"raw": float(raw), "weight": float(weight)}
        if set(mapped) != active_slots:
            cite_fail(
                "target-filtered per-lens evidence differs from active GPU roster",
                citation_id=citation_id,
                label=label,
                expected=sorted(active_slots),
                actual=sorted(mapped),
            )
        return mapped

    identity = lens_map(identity_hit, "identity")
    proposition = lens_map(proposition_hit, "proposition")
    evidence = []
    weighted_gain = 0.0
    total_weight = 0.0
    for slot in sorted(active_slots):
        if identity[slot]["weight"] != proposition[slot]["weight"]:
            cite_fail("target-filtered panel weight changed between paired queries", citation_id=citation_id, slot=slot)
        weight = identity[slot]["weight"]
        gain = proposition[slot]["raw"] - identity[slot]["raw"]
        weighted_gain += gain * weight
        total_weight += weight
        evidence.append(
            {
                "slot": slot,
                "weight": weight,
                "identity_raw": identity[slot]["raw"],
                "proposition_raw": proposition[slot]["raw"],
                "raw_gain": gain,
            }
        )
    return weighted_gain / total_weight, evidence


def support_outliers(rows: list[dict]) -> dict:
    if len(rows) != 10:
        cite_fail("support calibration requires exactly ten physically present citations")
    scored = [(row["citation_id"], float(row["support_score"])) for row in rows]
    if not all(math.isfinite(score) for _, score in scored):
        cite_fail("support calibration contains a non-finite paired score")
    flagged = [row for row in rows if float(row["support_score"]) < 0.0]
    if len(flagged) != 1:
        cite_fail(
            "live paired proposition scores did not yield exactly one negative-gain outlier",
            threshold=0.0,
            scores=scored,
            flagged=[row["citation_id"] for row in flagged],
        )
    return {
        "method": "target-filtered paired panel-weighted mean raw gain over ten active GPU lenses",
        "low_support_threshold": 0.0,
        "scores": [
            {"citation_id": citation_id, "paired_raw_gain": score}
            for citation_id, score in scored
        ],
        "flagged_citation_id": flagged[0]["citation_id"],
    }


def lifecycle_readback(binary: Path, vault: str, collection: str, env: dict[str, str]) -> tuple[dict, dict]:
    value, record = run_json(
        [str(binary), "graph-collection-generations", vault, "--collection", collection],
        env,
        label="citation-overlay-lifecycle",
    )
    if not isinstance(value, dict) or value.get("status") != "ok":
        cite_fail("citation overlay lifecycle readback is not successful")
    return value, record


def accepted_overlay(lifecycle: dict, overlay: dict) -> None:
    generation_id = overlay["value"].get("graph_generation")
    accepted = []
    for row in lifecycle.get("generations", []):
        state = row.get("state", {}) if isinstance(row, dict) else {}
        if state.get("generation") == generation_id and str(state.get("status", "")).lower() == "accepted":
            accepted.append(row)
    if len(accepted) != 1:
        cite_fail(
            "physical graph lifecycle lacks exactly one accepted overlay generation",
            graph_generation=generation_id,
            matches=len(accepted),
        )
    if lifecycle.get("vault_id") != overlay["value"].get("vault_id"):
        cite_fail("overlay report and live lifecycle refer to different vault IDs")


def build(args) -> dict:
    binary = executable(args.calyx)
    calyx_home = Path(args.calyx_home).absolute()
    if not calyx_home.is_dir() or calyx_home.is_symlink():
        cite_fail("CALYX_HOME is not a plain directory", path=str(calyx_home))
    env = dict(os.environ)
    env["CALYX_HOME"] = str(calyx_home)
    fixture = load_fixture(args.fixture, args.brief)
    extract = load_extract_index(args.extract_generation)
    ingest = load_ingest_index(args.ingest_generation, args.extract_generation)
    vault_aliases = load_vault_aliases(
        args.vault_alias_generation, args.ingest_generation, args.cx_list
    )
    citation_graph = load_citation_graph(
        args.citation_generation,
        args.ingest_generation,
        vault_aliases["aliases"],
    )
    overlay = verify_overlay_report(
        args.overlay_report, citation_graph, vault_aliases
    )
    lifecycle, lifecycle_run = lifecycle_readback(
        binary, args.vault, overlay["value"]["collection"], env
    )
    accepted_overlay(lifecycle, overlay)
    runs = [lifecycle_run]
    rows = []
    walks = []
    roster_fingerprint = None
    canonical_limit = ingest["verification"]["canonical_content_rows"]
    for fixture_row in fixture["value"]["citations"]:
        citation_id = fixture_row["citation_id"]
        identity = (
            normalized(fixture_row["case_name"]),
            normalized(fixture_row["docket_number"]),
        )
        matches = extract["by_identity"].get(identity, [])
        target = fixture_row.get("target_opinion_id")
        if matches:
            if target is None or int(target) not in matches:
                cite_fail(
                    "physically present fixture cite lacks a matching target opinion",
                    citation_id=citation_id,
                    matches=matches,
                    target=target,
                )
            target = int(target)
            target_alias = vault_aliases["aliases"].get(target)
            if target_alias is None:
                cite_fail("fixture target has no physical vault alias", citation_id=citation_id)
            expected_cx = target_alias["cx_id"]
            target_pointer_fragment = target_alias["input_pointer_fragment"]
        else:
            if target is not None:
                cite_fail("fixture target claims an opinion absent from the ingest index", citation_id=citation_id)
            expected_cx = None
            target_pointer_fragment = None

        known_query = "%s | docket %s" % (
            fixture_row["case_name"],
            fixture_row["docket_number"],
        )
        known_hits, slots, known_run = search(
            binary,
            args.vault,
            args.resident_addr,
            known_query,
            args.known_item_k,
            env,
            label="%s-known-item" % citation_id,
        )
        runs.append(known_run)
        current_roster = json.dumps(slots, sort_keys=True, separators=(",", ":"))
        if roster_fingerprint is None:
            roster_fingerprint = current_roster
        elif current_roster != roster_fingerprint:
            cite_fail("live serving roster changed within cite-check run", citation_id=citation_id)
        known_hit = find_hit(known_hits, expected_cx) if expected_cx is not None else None
        if expected_cx is not None and known_hit is None:
            cite_fail(
                "real canonical citation failed live known-item retrieval",
                citation_id=citation_id,
                expected_cx=expected_cx,
                k=args.known_item_k,
            )
        row = {
            "citation_id": citation_id,
            "citation_text": fixture_row["citation_text"],
            "case_name": fixture_row["case_name"],
            "docket_number": fixture_row["docket_number"],
            "proposition": fixture_row["proposition"],
            "reviewed_class": fixture_row["reviewed_class"],
            "exact_ingest_match_count": len(matches),
            "target_opinion_id": target,
            "expected_cx_id": expected_cx,
            "corpus_verdict": "FOUND" if matches else "NOT_IN_CORPUS",
            "known_item_rank": known_hit["rank"] if known_hit else None,
            "known_item_score": known_hit["score"] if known_hit else None,
            "known_item_latency_ms": known_run["elapsed_ms"],
            "nearest_returned_cx_id": known_hits[0]["cx_id"] if known_hits else None,
        }
        if expected_cx is not None:
            identity_hits, identity_slots, identity_run = search(
                binary,
                args.vault,
                args.resident_addr,
                known_query,
                args.support_k,
                env,
                label="%s-identity-baseline" % citation_id,
                target_pointer_fragment=target_pointer_fragment,
            )
            runs.append(identity_run)
            if json.dumps(identity_slots, sort_keys=True, separators=(",", ":")) != roster_fingerprint:
                cite_fail("live serving roster changed during identity baseline", citation_id=citation_id)
            if len(identity_hits) != 1 or identity_hits[0]["cx_id"] != expected_cx:
                cite_fail(
                    "target-filtered identity baseline did not return exactly the expected opinion",
                    citation_id=citation_id,
                    expected_cx=expected_cx,
                )
            support_hits, support_slots, support_run = search(
                binary,
                args.vault,
                args.resident_addr,
                fixture_row["proposition"],
                args.support_k,
                env,
                label="%s-proposition" % citation_id,
                target_pointer_fragment=target_pointer_fragment,
            )
            runs.append(support_run)
            if json.dumps(support_slots, sort_keys=True, separators=(",", ":")) != roster_fingerprint:
                cite_fail("live serving roster changed during proposition check", citation_id=citation_id)
            if len(support_hits) != 1 or support_hits[0]["cx_id"] != expected_cx:
                cite_fail(
                    "target-filtered proposition check did not return exactly the expected opinion",
                    citation_id=citation_id,
                    expected_cx=expected_cx,
                )
            support_hit = support_hits[0]
            support_score, paired_lenses = paired_support_score(
                identity_hits[0],
                support_hit,
                support_slots,
                citation_id=citation_id,
            )
            row.update(
                {
                    "support_rank": support_hit["rank"],
                    "support_score": support_score,
                    "support_per_lens": paired_lenses,
                    "identity_latency_ms": identity_run["elapsed_ms"],
                    "support_latency_ms": support_run["elapsed_ms"],
                }
            )
            walk = exhaustive_walk(
                target_alias["canonical_opinion_id"], citation_graph, canonical_limit
            )
            walk["citation_id"] = citation_id
            walks.append(walk)
            row["citation_walk"] = {
                key: value for key, value in walk.items() if key != "frontier_exits"
            }
        else:
            row.update(
                {
                    "support_rank": None,
                    "support_score": None,
                    "support_per_lens": [],
                    "identity_latency_ms": None,
                    "support_latency_ms": None,
                    "support_verdict": "NOT_APPLICABLE",
                    "citation_walk": None,
                }
            )
        rows.append(row)

    real_rows = [row for row in rows if row["corpus_verdict"] == "FOUND"]
    calibration = support_outliers(real_rows)
    for row in real_rows:
        row["support_verdict"] = (
            "LOW_SUPPORT"
            if row["citation_id"] == calibration["flagged_citation_id"]
            else "SUPPORTED"
        )
    for row in rows:
        expected = {
            "supported": ("FOUND", "SUPPORTED"),
            "fabricated": ("NOT_IN_CORPUS", "NOT_APPLICABLE"),
            "low_support": ("FOUND", "LOW_SUPPORT"),
        }[row["reviewed_class"]]
        actual = (row["corpus_verdict"], row["support_verdict"])
        row["reviewed_agreement"] = actual == expected
        if not row["reviewed_agreement"]:
            cite_fail("computed cite-check result differs from independent review", citation_id=row["citation_id"], expected=expected, actual=actual)

    counts = {
        "citations": len(rows),
        "found": sum(row["corpus_verdict"] == "FOUND" for row in rows),
        "not_in_corpus": sum(row["corpus_verdict"] == "NOT_IN_CORPUS" for row in rows),
        "low_support": sum(row["support_verdict"] == "LOW_SUPPORT" for row in rows),
        "reviewed_agreement": sum(row["reviewed_agreement"] for row in rows),
    }
    if counts != {"citations": 12, "found": 10, "not_in_corpus": 2, "low_support": 1, "reviewed_agreement": 12}:
        cite_fail("cite-check acceptance accounting differs", actual=counts)
    report = {
        "status": "verified",
        "scope": "Cuyahoga County Eighth District opinions only",
        "vault": args.vault,
        "vault_id": lifecycle["vault_id"],
        "source_of_truth": {
            "existence": "physical Aster Base CF cx-list joined to canonical ingest metadata",
            "retrieval": "live fresh Calyx search through the frozen resident panel",
            "citation_overlay": "accepted physical Aster Graph CF lifecycle plus exhaustive verified sidecar traversal",
            "report": "immutable generation members re-read by SHA-256",
        },
        "counts": counts,
        "support_calibration": calibration,
        "semantic_caveat": "LOW_SUPPORT is a live panel outlier signal, not a legal-treatment judgment.",
        "rows": rows,
    }
    cx_list_path = plain_file(args.cx_list, label="physical cx-list")
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        row_output = generation.open_text(ROWS_FILE)
        for row in rows:
            row_output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        run_output = generation.open_text(RUNS_FILE)
        for run in runs:
            run_output.write(json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n")
        walk_output = generation.open_text(WALKS_FILE)
        for walk in walks:
            walk_output.write(json.dumps(walk, sort_keys=True, separators=(",", ":")) + "\n")
        generation.write_json(REPORT_FILE, report)
        generation.publish(
            {
                "format": FORMAT,
                "vault": args.vault,
                "vault_id": lifecycle["vault_id"],
                "resident_addr": args.resident_addr,
                "known_item_k": args.known_item_k,
                "support_k": args.support_k,
                "sources": {
                    "calyx_binary": str(binary),
                    "calyx_binary_sha256": sha256_file(binary),
                    "fixture": str(fixture["path"]),
                    "fixture_sha256": fixture["sha256"],
                    "brief": str(fixture["brief_path"]),
                    "brief_sha256": fixture["brief_sha256"],
                    "extract_generation": str(Path(args.extract_generation).absolute()),
                    "extract_manifest_sha256": extract["verification"]["manifest_sha256"],
                    "ingest_generation": str(Path(args.ingest_generation).absolute()),
                    "ingest_manifest_sha256": ingest["verification"]["manifest_sha256"],
                    "vault_alias_generation": str(Path(args.vault_alias_generation).absolute()),
                    "vault_alias_manifest_sha256": vault_aliases["verification"]["manifest_sha256"],
                    "physical_cx_list": str(cx_list_path),
                    "physical_cx_list_sha256": sha256_file(cx_list_path),
                    "citation_generation": str(Path(args.citation_generation).absolute()),
                    "citation_manifest_sha256": citation_graph["verification"]["manifest_sha256"],
                    "overlay_report": str(overlay["path"]),
                    "overlay_report_sha256": overlay["sha256"],
                },
                "counts": counts,
                "proof": {
                    "all_live_search_hits_fresh": True,
                    "all_live_search_hits_provenanced": True,
                    "serving_roster_stable": True,
                    "all_real_citations_target_conditioned_across_full_panel": True,
                    "citation_walks_complete": True,
                    "physical_overlay_generation_accepted": True,
                },
            }
        )
    return verify(args.out)


def verify(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        cite_fail("cite-check generation format is unsupported")
    if set(manifest["files"]) != {REPORT_FILE, ROWS_FILE, RUNS_FILE, WALKS_FILE}:
        cite_fail("cite-check generation member contract differs")
    report = json.loads((root / REPORT_FILE).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / ROWS_FILE).read_text(encoding="utf-8").splitlines()]
    walks = [json.loads(line) for line in (root / WALKS_FILE).read_text(encoding="utf-8").splitlines()]
    runs = [json.loads(line) for line in (root / RUNS_FILE).read_text(encoding="utf-8").splitlines()]
    if report.get("status") != "verified" or report.get("rows") != rows:
        cite_fail("cite-check report differs from physical row readback")
    if manifest.get("counts") != report.get("counts") or report["counts"]["citations"] != len(rows):
        cite_fail("cite-check manifest/report/row accounting differs")
    if len(walks) != report["counts"]["found"] or len(runs) != 1 + 12 + (2 * report["counts"]["found"]):
        cite_fail("cite-check physical command or walk accounting differs")
    if not all(walk.get("walk_complete") is True for walk in walks):
        cite_fail("cite-check contains an incomplete citation walk")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "report_sha256": sha256_file(root / REPORT_FILE),
        "rows_sha256": sha256_file(root / ROWS_FILE),
        "runs_sha256": sha256_file(root / RUNS_FILE),
        "walks_sha256": sha256_file(root / WALKS_FILE),
        **report["counts"],
        "status": "verified",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--calyx", required=True)
    build_parser.add_argument("--calyx-home", required=True)
    build_parser.add_argument("--vault", required=True)
    build_parser.add_argument("--resident-addr", required=True)
    build_parser.add_argument("--fixture", required=True)
    build_parser.add_argument("--brief", required=True)
    build_parser.add_argument("--extract-generation", required=True)
    build_parser.add_argument("--ingest-generation", required=True)
    build_parser.add_argument("--vault-alias-generation", required=True)
    build_parser.add_argument("--cx-list", required=True)
    build_parser.add_argument("--citation-generation", required=True)
    build_parser.add_argument("--overlay-report", required=True)
    build_parser.add_argument("--known-item-k", type=int, default=50)
    build_parser.add_argument("--support-k", type=int, default=200)
    build_parser.add_argument("--out", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--generation", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            if args.known_item_k < 10 or args.support_k < args.known_item_k:
                cite_fail("search bounds require support-k >= known-item-k >= 10")
            result = build(args)
        else:
            result = verify(args.generation)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(
            "cite-check: ERROR type=%s message=%s" % (type(error).__name__, error),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
