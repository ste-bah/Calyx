#!/usr/bin/env python3
"""Physical source loading and graph traversal for the Cuyahoga cite checker."""

from __future__ import annotations

from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


LAWDEMO_DIR = Path(__file__).resolve().parent
LAWSLICE_DIR = LAWDEMO_DIR.parent / "lawslice"
if str(LAWSLICE_DIR) not in sys.path:
    sys.path.insert(0, str(LAWSLICE_DIR))

from build_ingest_jsonl import (  # noqa: E402
    CAP_FILE,
    MANIFEST_FILE as INGEST_MANIFEST,
    MODERN_FILE,
    verify_ingest_generation,
)
from extract_citations import (  # noqa: E402
    CITATIONS_FILE,
    FRONTIER_EDGES_FILE,
    MANIFEST_FILE as CITATION_MANIFEST,
    verify_citation_generation,
)
from extract_cuyahoga import (  # noqa: E402
    MANIFEST_FILE as EXTRACT_MANIFEST,
    RAW_FILE as EXTRACT_RAW_FILE,
    verify_extract_generation,
)
from law_generation import generation_member, sha256_file  # noqa: E402
from materialize_vault_aliases import (  # noqa: E402
    ALIASES_FILE as VAULT_ALIASES_FILE,
    MANIFEST_FILE as VAULT_ALIAS_MANIFEST,
    verify_vault_alias_generation,
)


FIXTURE_FORMAT = "calyx-cuyahoga-citecheck-fixture-v1"
REVIEWED_CLASSES = {"supported", "fabricated", "low_support"}
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
INPUT_POINTER = re.compile(r"^calyx-vault://inputs/([0-9a-f]{64})\.bin$")


class CiteCheckSourceError(RuntimeError):
    pass


def fail(message: str, **context) -> None:
    if context:
        message = "%s | %s" % (message, json.dumps(context, sort_keys=True))
    raise CiteCheckSourceError(message)


def plain_file(path: str, *, label: str) -> Path:
    resolved = Path(path).absolute()
    if not resolved.is_file() or resolved.is_symlink():
        fail("input is not a plain file", label=label, path=str(resolved))
    return resolved


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def positive_int(value, *, field: str, where: str) -> int:
    if isinstance(value, bool):
        fail("positive integer is boolean", field=field, where=where)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail("positive integer is invalid", field=field, where=where, value=value)
    if parsed <= 0 or str(parsed) != str(value):
        fail("positive integer is not canonical", field=field, where=where, value=value)
    return parsed


def load_fixture(path: str, brief_path: str) -> dict:
    fixture_path = plain_file(path, label="fixture")
    brief = plain_file(brief_path, label="brief")
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail("fixture is invalid JSON", path=str(fixture_path), error=str(error))
    if not isinstance(fixture, dict) or fixture.get("format") != FIXTURE_FORMAT:
        fail("fixture format is unsupported", path=str(fixture_path))
    citations = fixture.get("citations")
    if not isinstance(citations, list) or len(citations) != 12:
        fail("fixture must contain exactly 12 citations", actual=len(citations or []))
    brief_text = brief.read_text(encoding="utf-8")
    expected_brief_hash = fixture.get("brief_sha256")
    actual_brief_hash = hashlib.sha256(brief_text.encode("utf-8")).hexdigest()
    if expected_brief_hash != actual_brief_hash:
        fail(
            "fixture brief digest differs from physical brief",
            expected=expected_brief_hash,
            actual=actual_brief_hash,
        )
    identifiers = set()
    reviewed_counts = {key: 0 for key in REVIEWED_CLASSES}
    for index, row in enumerate(citations, 1):
        where = "citation:%d" % index
        if not isinstance(row, dict):
            fail("fixture citation is not an object", where=where)
        for field in (
            "citation_id",
            "citation_text",
            "case_name",
            "docket_number",
            "proposition",
            "reviewed_class",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                fail("fixture citation field is empty or not text", where=where, field=field)
        citation_id = row["citation_id"]
        if citation_id in identifiers:
            fail("fixture duplicates citation_id", citation_id=citation_id)
        identifiers.add(citation_id)
        reviewed = row["reviewed_class"]
        if reviewed not in REVIEWED_CLASSES:
            fail("fixture reviewed class is invalid", where=where, reviewed_class=reviewed)
        reviewed_counts[reviewed] += 1
        target = row.get("target_opinion_id")
        if target is not None:
            positive_int(target, field="target_opinion_id", where=where)
        if row["citation_text"] not in brief_text or row["proposition"] not in brief_text:
            fail("fixture citation or proposition is absent from physical brief", where=where)
    expected_counts = {"supported": 9, "fabricated": 2, "low_support": 1}
    if reviewed_counts != expected_counts:
        fail("fixture reviewed class accounting is invalid", actual=reviewed_counts)
    return {
        "path": fixture_path,
        "sha256": sha256_file(fixture_path),
        "brief_path": brief,
        "brief_sha256": actual_brief_hash,
        "value": fixture,
    }


def _jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                fail("JSONL contains a blank line", path=str(path), line=line_number)
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                fail("JSONL contains invalid JSON", path=str(path), line=line_number, error=str(error))
            if not isinstance(value, dict):
                fail("JSONL row is not an object", path=str(path), line=line_number)
            yield line_number, value


def load_ingest_index(path: str, extract_generation: str) -> dict:
    verification = verify_ingest_generation(path, extract_generation)
    by_id = {}
    by_identity = {}
    for member in (CAP_FILE, MODERN_FILE):
        batch_path = generation_member(path, INGEST_MANIFEST, member)
        for line_number, batch in _jsonl(batch_path):
            metadata = batch.get("metadata")
            if not isinstance(metadata, dict):
                fail("ingest row metadata is absent", file=member, line=line_number)
            opinion_id = positive_int(
                metadata.get("opinion_id"), field="opinion_id", where="%s:%d" % (member, line_number)
            )
            if opinion_id in by_id:
                fail("canonical ingest opinion is duplicated", opinion_id=opinion_id)
            case_name = metadata.get("case_name_full") or metadata.get("case_name")
            docket = metadata.get("docket_number")
            if not isinstance(case_name, str) or not case_name.strip() or not isinstance(docket, str):
                fail("canonical ingest identity fields are invalid", opinion_id=opinion_id)
            identity = (normalized(case_name), normalized(docket))
            by_id[opinion_id] = metadata
            by_identity.setdefault(identity, []).append(opinion_id)
    if len(by_id) != verification["canonical_content_rows"]:
        fail("ingest index count differs from verified generation")
    for ids in by_identity.values():
        ids.sort()
    return {"verification": verification, "by_id": by_id, "by_identity": by_identity}


def load_extract_index(path: str) -> dict:
    verification = verify_extract_generation(path)
    raw_path = generation_member(path, EXTRACT_MANIFEST, EXTRACT_RAW_FILE)
    opinion_ids = set()
    by_identity = {}
    for line_number, row in _jsonl(raw_path):
        opinion_id = positive_int(
            row.get("opinion_id"),
            field="opinion_id",
            where="extract:%d" % line_number,
        )
        if opinion_id in opinion_ids:
            fail("extract opinion is duplicated", opinion_id=opinion_id)
        case_name = row.get("case_name_full") or row.get("case_name")
        docket = row.get("docket_number")
        if not isinstance(case_name, str) or not case_name.strip() or not isinstance(docket, str):
            fail("extract identity fields are invalid", opinion_id=opinion_id)
        identity = (normalized(case_name), normalized(docket))
        opinion_ids.add(opinion_id)
        by_identity.setdefault(identity, []).append(opinion_id)
    if len(opinion_ids) != verification["accepted_rows"]:
        fail("extract identity count differs from verified generation")
    for ids in by_identity.values():
        ids.sort()
    return {
        "verification": verification,
        "opinion_ids": opinion_ids,
        "by_identity": by_identity,
    }


def load_vault_aliases(path: str, ingest_generation: str, cx_list: str) -> dict:
    verification = verify_vault_alias_generation(path, ingest_generation, cx_list)
    aliases_path = generation_member(path, VAULT_ALIAS_MANIFEST, VAULT_ALIASES_FILE)
    aliases = {}
    canonical_by_cx = {}
    with open(aliases_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for line_number, row in enumerate(reader, 2):
            opinion_id = positive_int(row.get("opinion_id"), field="opinion_id", where="alias:%d" % line_number)
            canonical = positive_int(
                row.get("canonical_opinion_id"),
                field="canonical_opinion_id",
                where="alias:%d" % line_number,
            )
            cx_id = row.get("cx_id")
            if opinion_id in aliases or not isinstance(cx_id, str) or not HEX_32.fullmatch(cx_id):
                fail("vault alias identity is invalid or duplicated", line=line_number)
            aliases[opinion_id] = {"canonical_opinion_id": canonical, "cx_id": cx_id}
            if opinion_id == canonical:
                canonical_by_cx[cx_id] = canonical
    if len(aliases) != verification["source_opinion_rows"]:
        fail("vault alias count differs from verified physical relation")
    cx_list_path = plain_file(cx_list, label="physical cx-list")
    try:
        physical_rows = json.loads(cx_list_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail("physical cx-list cannot be read for target filters", error=str(error))
    if not isinstance(physical_rows, list):
        fail("physical cx-list is not an array for target filters")
    pointer_fragment_by_cx = {}
    for row_number, row in enumerate(physical_rows, 1):
        if not isinstance(row, dict):
            fail("physical cx-list target-filter row is invalid", row=row_number)
        cx_id = row.get("cx_id")
        input_ref = row.get("input_ref")
        pointer = input_ref.get("pointer") if isinstance(input_ref, dict) else None
        match = INPUT_POINTER.fullmatch(pointer) if isinstance(pointer, str) else None
        if (
            not isinstance(cx_id, str)
            or not HEX_32.fullmatch(cx_id)
            or match is None
            or cx_id in pointer_fragment_by_cx
        ):
            fail("physical cx-list target pointer is invalid or duplicated", row=row_number)
        pointer_fragment_by_cx[cx_id] = match.group(1)
    if set(pointer_fragment_by_cx) != set(canonical_by_cx):
        fail("physical target-pointer set differs from canonical vault aliases")
    for alias in aliases.values():
        alias["input_pointer_fragment"] = pointer_fragment_by_cx[alias["cx_id"]]
    return {
        "verification": verification,
        "aliases": aliases,
        "canonical_by_cx": canonical_by_cx,
    }


def load_citation_graph(path: str, ingest_generation: str, aliases: dict) -> dict:
    verification = verify_citation_generation(path, ingest_generation)
    adjacency = {}
    edge_rows = 0
    invalid_depth = 0
    duplicate_pair = 0
    materialized_edges = 0
    seen_pairs = set()
    materialized_nodes = set()
    citation_path = generation_member(path, CITATION_MANIFEST, CITATIONS_FILE)
    with open(citation_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for line_number, row in enumerate(reader, 2):
            citing = positive_int(row["citing_opinion_id"], field="citing_opinion_id", where="edge:%d" % line_number)
            cited = positive_int(row["cited_opinion_id"], field="cited_opinion_id", where="edge:%d" % line_number)
            depth = int(row["depth"])
            src = aliases[citing]["canonical_opinion_id"]
            dst = aliases[cited]["canonical_opinion_id"]
            edge_rows += 1
            if depth <= 0:
                invalid_depth += 1
                continue
            pair = (aliases[citing]["cx_id"], aliases[cited]["cx_id"])
            if pair in seen_pairs:
                duplicate_pair += 1
                continue
            seen_pairs.add(pair)
            materialized_nodes.update(pair)
            adjacency.setdefault(src, []).append((dst, citing, cited, depth))
            materialized_edges += 1
    frontier = {}
    frontier_path = generation_member(path, CITATION_MANIFEST, FRONTIER_EDGES_FILE)
    with open(frontier_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for line_number, row in enumerate(reader, 2):
            citing = positive_int(row["citing_opinion_id"], field="citing_opinion_id", where="frontier:%d" % line_number)
            canonical = aliases[citing]["canonical_opinion_id"]
            frontier.setdefault(canonical, []).append(
                {
                    "citation_id": int(row["citation_id"]),
                    "citing_opinion_id": citing,
                    "cited_external_opinion_id": int(row["cited_opinion_id"]),
                    "source_depth": int(row["depth"]),
                }
            )
    for edges in adjacency.values():
        edges.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    for exits in frontier.values():
        exits.sort(key=lambda item: (item["cited_external_opinion_id"], item["citation_id"]))
    if edge_rows != verification["in_slice_edges"]:
        fail("citation graph edge count differs from generation verification")
    return {
        "verification": verification,
        "adjacency": adjacency,
        "frontier": frontier,
        "overlay_expected": {
            "total_rows": edge_rows,
            "edges_built": materialized_edges,
            "skipped_total": invalid_depth + duplicate_pair,
            "skipped_unresolved_citing": 0,
            "skipped_unresolved_cited": 0,
            "skipped_unresolved_both": 0,
            "skipped_invalid_depth": invalid_depth,
            "skipped_duplicate_pair": duplicate_pair,
            "idmap_entries": len(aliases),
        },
        "overlay_node_count": len(materialized_nodes),
    }


def verify_overlay_report(path: str, citation_graph: dict, vault_aliases: dict) -> dict:
    report_path = plain_file(path, label="citation overlay report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail("citation overlay report is invalid JSON", path=str(report_path), error=str(error))
    if not isinstance(report, dict) or report.get("status") != "ok":
        fail("citation overlay report is not successful")
    skip = report.get("skip_report")
    readback = report.get("readback")
    if not isinstance(skip, dict) or not isinstance(readback, dict):
        fail("citation overlay report lacks skip/readback evidence")
    expected_skip = citation_graph["overlay_expected"]
    actual_skip = {key: skip.get(key) for key in expected_skip}
    if actual_skip != expected_skip:
        fail(
            "citation overlay skip accounting differs from independently reconstructed inputs",
            expected=expected_skip,
            actual=actual_skip,
        )
    if expected_skip["idmap_entries"] != vault_aliases["verification"]["source_opinion_rows"]:
        fail("citation graph alias count differs from verified physical aliases")
    edge_count = skip.get("edges_built")
    required_equal = {
        "edge_rows_written": edge_count,
        "physical_edge_out_keys": edge_count,
        "csr_edges": edge_count,
        "assoc_graph_edges": edge_count,
    }
    for field, expected in required_equal.items():
        if readback.get(field) != expected:
            fail("citation overlay physical readback count differs", field=field, expected=expected)
    node_count = citation_graph["overlay_node_count"]
    for field in ("node_rows_written", "physical_node_keys", "csr_nodes", "assoc_graph_nodes"):
        if readback.get(field) != node_count:
            fail("citation overlay physical node count differs", field=field, expected=node_count)
    if readback.get("all_node_values_read_back") is not True or readback.get("all_edge_values_read_back") is not True:
        fail("citation overlay did not read every Graph CF value back")
    return {"path": report_path, "sha256": sha256_file(report_path), "value": report}


def exhaustive_walk(start: int, graph: dict, canonical_limit: int) -> dict:
    queue = deque([(start, 0)])
    seen = {start}
    hops = {0: 1}
    frontier_exits = []
    internal_edges = 0
    max_hop = 0
    while queue:
        node, hop = queue.popleft()
        max_hop = max(max_hop, hop)
        for exit_row in graph["frontier"].get(node, []):
            frontier_exits.append({"at_canonical_opinion_id": node, "graph_hop": hop, **exit_row})
        for dst, _citing, _cited, _depth in graph["adjacency"].get(node, []):
            internal_edges += 1
            if dst not in seen:
                seen.add(dst)
                queue.append((dst, hop + 1))
                hops[hop + 1] = hops.get(hop + 1, 0) + 1
                if len(seen) > canonical_limit:
                    fail("citation walk exceeded canonical corpus cardinality", start=start)
    frontier_exits.sort(
        key=lambda row: (
            row["graph_hop"],
            row["at_canonical_opinion_id"],
            row["cited_external_opinion_id"],
            row["citation_id"],
        )
    )
    return {
        "start_canonical_opinion_id": start,
        "reachable_canonical_opinion_count": len(seen),
        "internal_edges_examined": internal_edges,
        "maximum_graph_hop": max_hop,
        "nodes_by_hop": {str(key): value for key, value in sorted(hops.items())},
        "frontier_exit_count": len(frontier_exits),
        "frontier_exits": frontier_exits,
        "walk_complete": True,
    }
