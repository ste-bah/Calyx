#!/usr/bin/env python3
"""Audit opinion aliases and citation provenance from a raw Graph-CF readback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from structured_error import (
    StructuredArgumentParser,
    StructuredError,
    parse_cli_args,
    write_error,
)


ALIAS_COLLECTION = "legal-opinion-aliases-v1"
CITATION_COLLECTION = "legal-citations-alias-v2"


class AuditError(StructuredError):
    code = "CALYX_OPINION_ALIAS_PHYSICAL_AUDIT_FAILED"
    default_remediation = (
        "quarantine the alias and citation collections and rebuild them from sealed source bytes"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_rank(path: str, line_number: int) -> tuple[int, ...]:
    numbers = tuple(int(value) for value in re.findall(r"\d+", Path(path).name))
    return (*numbers, line_number)


def decode_graph_key(raw: bytes) -> tuple[str, int, bytes]:
    if len(raw) < 4 or raw[0] != ord("g"):
        raise AuditError("Graph-CF row has an invalid key prefix")
    collection_len = int.from_bytes(raw[1:3], "big")
    boundary = 3 + collection_len
    if len(raw) <= boundary:
        raise AuditError("Graph-CF row has a truncated collection prefix")
    collection = raw[3:boundary].decode("utf-8")
    return collection, raw[boundary], raw[boundary + 1 :]


def decode_edge_key(payload: bytes) -> tuple[str, str, str]:
    if len(payload) < 34:
        raise AuditError("Graph-CF edge key is truncated")
    src = payload[:16].hex()
    type_len = int.from_bytes(payload[16:18], "big")
    type_end = 18 + type_len
    if type_len == 0 or len(payload) != type_end + 16:
        raise AuditError("Graph-CF edge key has an invalid edge-type length")
    edge_type = payload[18:type_end].decode("utf-8")
    return src, edge_type, payload[type_end:].hex()


def load_latest_rows(path: Path) -> dict[str, dict[tuple[Any, ...], bytes]]:
    wanted = {ALIAS_COLLECTION, CITATION_COLLECTION}
    latest: dict[str, dict[tuple[Any, ...], tuple[tuple[int, ...], bytes]]] = {
        name: {} for name in wanted
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8 or fields[0] != "CF" or fields[4] != "KEY":
                raise AuditError(f"raw Graph-CF line {line_number} has an invalid envelope")
            key = bytes.fromhex(fields[5])
            collection, kind, payload = decode_graph_key(key)
            if collection not in wanted or kind not in (0, 1):
                continue
            if kind == 0:
                if len(payload) != 16:
                    raise AuditError(f"raw Graph-CF line {line_number} has a bad node key")
                identity: tuple[Any, ...] = ("node", payload.hex())
            else:
                identity = ("edge", *decode_edge_key(payload))
            value = bytes.fromhex(fields[7])
            rank = file_rank(fields[3], line_number)
            previous = latest[collection].get(identity)
            if previous is None or rank > previous[0]:
                latest[collection][identity] = (rank, value)
    return {
        collection: {identity: value for identity, (_rank, value) in rows.items()}
        for collection, rows in latest.items()
    }


def load_alias_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "opinion_id",
        "cx_id",
        "canonical_opinion_id",
        "content_sha256",
        "is_canonical",
        "source_url",
    }
    if not rows or set(rows[0]) != required:
        raise AuditError("alias CSV schema is not the exact six-column contract")
    aliases: dict[str, dict[str, str]] = {}
    for row in rows:
        opinion_id = row["opinion_id"]
        if opinion_id in aliases:
            raise AuditError(f"alias CSV duplicates opinion {opinion_id}")
        aliases[opinion_id] = row
    return aliases


def load_citations(path: Path) -> Counter[tuple[str, str, int, str]]:
    expected: Counter[tuple[str, str, int, str]] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"citing_opinion_id", "cited_opinion_id", "depth"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AuditError("citation CSV lacks the required endpoint/depth columns")
        for line_number, row in enumerate(reader, 2):
            source_row_id = row.get("id") or f"row{line_number}"
            expected[
                (
                    row["citing_opinion_id"],
                    row["cited_opinion_id"],
                    int(row["depth"]),
                    source_row_id,
                )
            ] += 1
    return expected


def load_ingest_alias_sets(
    path: Path, aliases: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    pairs = [row for row in aliases.values() if row["is_canonical"] == "false"]
    wanted = {
        opinion_id
        for row in pairs
        for opinion_id in (row["opinion_id"], row["canonical_opinion_id"])
    }
    source: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(
                    f"ingest alias row {line_number} is invalid JSON: {error}"
                ) from error
            opinion_id = str(row.get("opinion_id", ""))
            if opinion_id in wanted:
                source[opinion_id] = row
    if set(source) != wanted:
        raise AuditError("ingest alias bytes do not contain every duplicate/canonical identity")
    result = []
    for pair in pairs:
        alias = source[pair["opinion_id"]]
        canonical = source[pair["canonical_opinion_id"]]
        if (
            str(alias.get("canonical_opinion_id")) != pair["canonical_opinion_id"]
            or str(alias.get("content_sha256")) != pair["content_sha256"]
            or alias.get("content_sha256") != canonical.get("content_sha256")
        ):
            raise AuditError(f"ingest alias set differs for opinion {pair['opinion_id']}")
        result.append(
            {
                "opinion_id": pair["opinion_id"],
                "canonical_opinion_id": pair["canonical_opinion_id"],
                "canonical_cx_id": pair["cx_id"],
                "content_sha256": pair["content_sha256"],
                "alias_cluster_id": alias.get("cluster_id"),
                "canonical_cluster_id": canonical.get("cluster_id"),
                "alias_docket_id": alias.get("docket_id"),
                "canonical_docket_id": canonical.get("docket_id"),
                "same_cluster": alias.get("cluster_id") == canonical.get("cluster_id"),
                "same_docket": alias.get("docket_id") == canonical.get("docket_id"),
            }
        )
    return result


def json_value(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} is not a JSON object")
    return value


def audit_aliases(
    rows: dict[tuple[Any, ...], bytes], aliases: dict[str, dict[str, str]]
) -> dict[str, str]:
    nodes = {
        identity[1]: json_value(value, f"alias node {identity[1]}")
        for identity, value in rows.items()
        if identity[0] == "node"
    }
    edges = [
        (identity, json_value(value, f"alias edge {identity[1]}->{identity[3]}"))
        for identity, value in rows.items()
        if identity[0] == "edge" and identity[2] == "aliases_to"
    ]
    if len(edges) != len(aliases):
        raise AuditError(f"physical aliases_to edges={len(edges)} expected={len(aliases)}")
    physical: dict[str, str] = {}
    for identity, edge in edges:
        _kind, src, _edge_type, dst = identity
        opinion_id = str(edge.get("opinion_id", ""))
        expected = aliases.get(opinion_id)
        if expected is None or opinion_id in physical:
            raise AuditError(f"physical alias opinion {opinion_id!r} is absent or duplicated")
        if dst != expected["cx_id"] or str(edge.get("canonical_cx_id")) != expected["cx_id"]:
            raise AuditError(f"physical alias opinion {opinion_id} targets the wrong CxId")
        for field in ("canonical_opinion_id", "content_sha256", "source_url"):
            if str(edge.get(field)) != expected[field]:
                raise AuditError(f"physical alias opinion {opinion_id} differs at {field}")
        source_metadata = nodes.get(src, {}).get("metadata", {})
        target_metadata = nodes.get(dst, {}).get("metadata", {})
        if source_metadata.get("opinion_id") != opinion_id:
            raise AuditError(f"physical source alias node differs for opinion {opinion_id}")
        if target_metadata.get("canonical_cx_id") != dst:
            raise AuditError(f"physical canonical target node differs at {dst}")
        physical[opinion_id] = dst
    return physical


def audit_citations(
    rows: dict[tuple[Any, ...], bytes],
    alias_map: dict[str, str],
    expected: Counter[tuple[str, str, int, str]],
) -> tuple[Counter[tuple[str, str, int, str]], int]:
    observed: Counter[tuple[str, str, int, str]] = Counter()
    physical_edges = 0
    for identity, raw in rows.items():
        if identity[0] != "edge" or identity[2] != "cites":
            continue
        physical_edges += 1
        _kind, src, _edge_type, dst = identity
        edge = json_value(raw, f"citation edge {src}->{dst}")
        sources = edge.get("source_citations")
        if not isinstance(sources, list) or edge.get("source_citation_count") != len(sources):
            raise AuditError(f"citation edge {src}->{dst} lacks exact source provenance")
        for source in sources:
            citing = str(source.get("citing_opinion_id", ""))
            cited = str(source.get("cited_opinion_id", ""))
            if alias_map.get(citing) != src or alias_map.get(cited) != dst:
                raise AuditError(f"citation source {citing}->{cited} resolves to wrong physical Cx")
            observed[(citing, cited, int(source["depth"]), str(source["source_row_id"]))] += 1
    if observed != expected:
        missing = sum((expected - observed).values())
        extra = sum((observed - expected).values())
        raise AuditError(f"physical citation provenance differs: missing={missing}, extra={extra}")
    return observed, physical_edges


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AuditError(f"report destination already exists: {path}")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.read_bytes() != data:
        raise AuditError("report byte readback mismatch")


def run(args: argparse.Namespace) -> None:
    graph_rows = load_latest_rows(args.graph_readback)
    aliases = load_alias_csv(args.aliases)
    alias_sets = load_ingest_alias_sets(args.ingest_aliases, aliases)
    expected_citations = load_citations(args.citations)
    alias_map = audit_aliases(graph_rows[ALIAS_COLLECTION], aliases)
    observed, edge_count = audit_citations(
        graph_rows[CITATION_COLLECTION], alias_map, expected_citations
    )
    noncanonical = {
        opinion_id for opinion_id, row in aliases.items() if row["is_canonical"] == "false"
    }
    affected = sum(
        count
        for (citing, cited, _depth, _source), count in observed.items()
        if citing in noncanonical or cited in noncanonical
    )
    report = {
        "status": "verified",
        "source_of_truth": "raw physical Graph-CF readback with latest SST row selected by sequence",
        "inputs": {
            "graph_readback_sha256": sha256_file(args.graph_readback),
            "aliases_sha256": sha256_file(args.aliases),
            "ingest_aliases_sha256": sha256_file(args.ingest_aliases),
            "citations_sha256": sha256_file(args.citations),
        },
        "alias_readback": {
            "source_opinion_aliases": len(alias_map),
            "canonical_constellations": len(set(alias_map.values())),
            "noncanonical_aliases": len(noncanonical),
            "all_nodes_edges_metadata_exact": True,
            "same_cluster_alias_sets": sum(row["same_cluster"] for row in alias_sets),
            "cross_cluster_alias_sets": sum(not row["same_cluster"] for row in alias_sets),
            "exact_noncanonical_alias_sets": alias_sets,
        },
        "citation_readback": {
            "physical_canonical_edges": edge_count,
            "source_citation_rows": sum(observed.values()),
            "coalesced_source_rows": sum(observed.values()) - edge_count,
            "affected_noncanonical_alias_rows": affected,
            "all_source_rows_exact": True,
            "all_endpoints_resolve_to_physical_cx": True,
        },
    }
    write_new(args.out, report)
    print(json.dumps(report, sort_keys=True))


def main() -> int:
    parser = StructuredArgumentParser()
    parser.add_argument("--graph-readback", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--ingest-aliases", type=Path, required=True)
    parser.add_argument("--citations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parse_cli_args(parser)
    try:
        run(args)
        return 0
    except (AuditError, OSError, ValueError, KeyError) as error:
        write_error(
            error,
            code="CALYX_OPINION_ALIAS_PHYSICAL_AUDIT_FAILED",
            remediation=(
                "quarantine the alias and citation collections and rebuild them from sealed source bytes"
            ),
            include_traceback=False,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
