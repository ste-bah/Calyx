#!/usr/bin/env python3
"""Build a citation generation that resolves every source opinion alias."""

from __future__ import annotations

import argparse
import bz2
import csv
import json
from pathlib import Path
import re
import sys
import traceback

from build_ingest_jsonl import (
    ALIASES_FILE,
    MANIFEST_FILE as INGEST_MANIFEST,
    verify_ingest_generation,
)
from extract_cuyahoga import verify_extract_generation
from law_generation import (
    GenerationPublisher,
    generation_member,
    sha256_file,
    verify_generation,
)
from structured_error import (
    StructuredArgumentParser,
    StructuredError,
    parse_cli_args,
    write_error,
)


FORMAT = "calyx-cuyahoga-citations-generation-v3"
MANIFEST_FILE = "citation_manifest.json"
CITATIONS_FILE = "citations_cuyahoga.csv"
FRONTIER_EDGES_FILE = "citation_frontier_edges.csv"
FRONTIER_FILE = "frontier_top500.csv"
PARENTHETICALS_FILE = "parentheticals_cuyahoga.csv"
STATS_FILE = "citation_stats.json"
FRONTIER_LIMIT = 500
TOP_LIMIT = 20
PROGRESS_EVERY = 5_000_000


class CitationBuildError(StructuredError):
    code = "cuyahoga_citation_generation_error"
    default_remediation = (
        "repair the named citation or ingest source mismatch and rebuild at a new destination"
    )


def fail(message: str, **context):
    raise CitationBuildError(message, **context)


def open_text(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def csv_rows(path: str, required: tuple[str, ...]):
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
            fail("source CSV is empty", path=path)
        missing = [name for name in required if name not in header]
        if missing:
            fail("source CSV is missing columns", path=path, missing=missing)
        index = {name: header.index(name) for name in required}
        while True:
            try:
                row = next(reader)
            except StopIteration:
                return
            except csv.Error as error:
                fail("source CSV parse error", path=path, line=reader.line_num, error=str(error))
            if len(row) != len(header):
                fail(
                    "source CSV column-count mismatch",
                    path=path,
                    line=reader.line_num,
                    expected=len(header),
                    actual=len(row),
                )
            yield reader.line_num, {name: row[position] for name, position in index.items()}


def integer(value: str, *, path: str, line: int, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        fail("source field is not an integer", path=path, line=line, field=field, value=value)


def positive_integer(value: str, *, path: str, line: int, field: str) -> int:
    parsed = integer(value, path=path, line=line, field=field)
    if parsed <= 0:
        fail(
            "source field is not a positive integer",
            path=path,
            line=line,
            field=field,
            value=value,
        )
    return parsed


_HASH_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")


def verify_bulk_sources(manifest_path: str, paths: dict[str, str]) -> dict:
    manifest = Path(manifest_path).absolute()
    if not manifest.is_file() or manifest.is_symlink():
        fail("bulk manifest is not a plain file", path=str(manifest))
    declared = {}
    with open(manifest, "r", encoding="utf-8") as source:
        for lineno, raw in enumerate(source, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            match = _HASH_LINE.fullmatch(line)
            if not match:
                fail("invalid bulk manifest line", path=manifest_path, line=lineno)
            digest, name = match.groups()
            if name in declared:
                fail("duplicate bulk manifest member", path=manifest_path, name=name)
            declared[name] = digest
    result = {"manifest_sha256": sha256_file(manifest), "members": {}}
    for role, source_path in paths.items():
        path = Path(source_path).absolute()
        if not path.is_file() or path.is_symlink():
            fail("bulk source is not a plain file", role=role, path=str(path))
        expected = declared.get(path.name)
        if expected is None:
            fail("bulk source absent from manifest", role=role, path=str(path))
        sys.stderr.write("source-hash: verifying %s\n" % path)
        actual = sha256_file(path)
        if actual != expected:
            fail("bulk source SHA-256 mismatch", role=role, expected=expected, actual=actual)
        result["members"][role] = {"name": path.name, "sha256": actual}
    return result


def load_aliases(ingest_generation: str) -> dict[int, int]:
    path = generation_member(ingest_generation, INGEST_MANIFEST, ALIASES_FILE)
    aliases = {}
    with open(path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail("invalid opinion alias JSON", path=str(path), line=lineno, error=str(error))
            opinion_id = row.get("opinion_id")
            canonical_id = row.get("canonical_opinion_id")
            if (
                not isinstance(opinion_id, int)
                or isinstance(opinion_id, bool)
                or opinion_id <= 0
                or not isinstance(canonical_id, int)
                or isinstance(canonical_id, bool)
                or canonical_id <= 0
                or opinion_id in aliases
            ):
                fail("invalid/duplicate opinion alias", path=str(path), line=lineno)
            aliases[opinion_id] = canonical_id
    if not aliases:
        fail("opinion alias relation is empty", path=str(path))
    missing_targets = set(aliases.values()) - set(aliases)
    if missing_targets:
        fail(
            "canonical alias targets are absent from the source identity relation",
            path=str(path),
            missing=sorted(missing_targets)[:20],
        )
    return aliases


def stream_citations(path: str, aliases: dict, output, frontier_output) -> dict:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "citing_opinion_id",
            "cited_opinion_id",
            "citing_canonical_opinion_id",
            "cited_canonical_opinion_id",
            "depth",
        ]
    )
    frontier_writer = csv.writer(frontier_output, lineterminator="\n")
    frontier_writer.writerow(
        [
            "citation_id",
            "citing_opinion_id",
            "citing_canonical_opinion_id",
            "cited_opinion_id",
            "depth",
        ]
    )
    rows = 0
    touching = 0
    in_slice = 0
    incoming_only = 0
    frontier = {}
    source_cited = {}
    canonical_cited = {}
    depth_histogram = {}
    for line, row in csv_rows(
        path, ("id", "depth", "cited_opinion_id", "citing_opinion_id")
    ):
        rows += 1
        citation_id = positive_integer(row["id"], path=path, line=line, field="id")
        cited = positive_integer(
            row["cited_opinion_id"], path=path, line=line, field="cited_opinion_id"
        )
        citing = positive_integer(
            row["citing_opinion_id"], path=path, line=line, field="citing_opinion_id"
        )
        depth = integer(row["depth"], path=path, line=line, field="depth")
        if depth < 0:
            fail("citation depth is negative", path=path, line=line, depth=depth)
        citing_in = citing in aliases
        cited_in = cited in aliases
        if not citing_in and not cited_in:
            if rows % PROGRESS_EVERY == 0:
                sys.stderr.write("citation-map: %d rows, %d touching\n" % (rows, touching))
            continue
        touching += 1
        if citing_in and cited_in:
            citing_canonical = aliases[citing]
            cited_canonical = aliases[cited]
            writer.writerow([citing, cited, citing_canonical, cited_canonical, depth])
            in_slice += 1
            depth_histogram[depth] = depth_histogram.get(depth, 0) + 1
            source_entry = source_cited.setdefault(cited, [0, 0])
            source_entry[0] += 1
            source_entry[1] += depth
            canonical_entry = canonical_cited.setdefault(cited_canonical, [0, 0])
            canonical_entry[0] += 1
            canonical_entry[1] += depth
        elif citing_in:
            entry = frontier.setdefault(cited, [0, 0])
            entry[0] += 1
            entry[1] += depth
            frontier_writer.writerow([citation_id, citing, aliases[citing], cited, depth])
        else:
            incoming_only += 1
        if rows % PROGRESS_EVERY == 0:
            sys.stderr.write("citation-map: %d rows, %d touching, %d in-slice\n" % (rows, touching, in_slice))
    return {
        "rows_scanned": rows,
        "edges_touching_slice": touching,
        "in_slice_edges": in_slice,
        "incoming_only_edges": incoming_only,
        "frontier": frontier,
        "source_cited": source_cited,
        "canonical_cited": canonical_cited,
        "depth_histogram": depth_histogram,
    }


def write_frontier(frontier: dict, output) -> int:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["cited_opinion_id", "citing_edge_count", "total_depth"])
    ranked = sorted(frontier.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
    for opinion_id, (count, total_depth) in ranked[:FRONTIER_LIMIT]:
        writer.writerow([opinion_id, count, total_depth])
    return min(len(ranked), FRONTIER_LIMIT)


def stream_parentheticals(path: str, aliases: dict, output) -> dict:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "id",
            "text",
            "score",
            "described_opinion_id",
            "describing_opinion_id",
            "described_canonical_opinion_id",
            "describing_canonical_opinion_id",
            "group_id",
        ]
    )
    rows = 0
    kept = 0
    for line, row in csv_rows(
        path,
        (
            "id",
            "text",
            "score",
            "described_opinion_id",
            "describing_opinion_id",
            "group_id",
        ),
    ):
        rows += 1
        positive_integer(row["id"], path=path, line=line, field="id")
        described = positive_integer(
            row["described_opinion_id"],
            path=path,
            line=line,
            field="described_opinion_id",
        )
        describing = positive_integer(
            row["describing_opinion_id"],
            path=path,
            line=line,
            field="describing_opinion_id",
        )
        if described in aliases or describing in aliases:
            writer.writerow(
                [
                    row["id"],
                    row["text"],
                    row["score"],
                    described,
                    describing,
                    aliases.get(described, ""),
                    aliases.get(describing, ""),
                    row["group_id"],
                ]
            )
            kept += 1
        if rows % 1_000_000 == 0:
            sys.stderr.write("parentheticals: %d rows, %d kept\n" % (rows, kept))
    return {"rows_scanned": rows, "kept": kept}


def load_case_names(extract_generation: str) -> dict[int, str]:
    from extract_cuyahoga import MANIFEST_FILE as EXTRACT_MANIFEST, RAW_FILE

    path = generation_member(extract_generation, EXTRACT_MANIFEST, RAW_FILE)
    names = {}
    with open(path, "r", encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            row = json.loads(line)
            opinion_id = row.get("opinion_id")
            if not isinstance(opinion_id, int) or opinion_id in names:
                fail("invalid/duplicate raw opinion ID while loading names", line=lineno)
            names[opinion_id] = row.get("case_name_full") or row.get("case_name") or ""
    return names


def verify_citation_generation(path: str, ingest_generation: str) -> dict:
    root = Path(path).absolute()
    verify_ingest_generation(ingest_generation)
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        fail("unsupported citation generation format", format=manifest.get("format"))
    required = {
        CITATIONS_FILE,
        FRONTIER_EDGES_FILE,
        FRONTIER_FILE,
        PARENTHETICALS_FILE,
        STATS_FILE,
    }
    if set(manifest["files"]) != required:
        fail("citation generation member contract mismatch", actual=sorted(manifest["files"]))
    aliases = load_aliases(ingest_generation)
    edges = 0
    source_cited = {}
    canonical_cited = {}
    depth_histogram = {}
    with open(root / CITATIONS_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_header = [
            "citing_opinion_id",
            "cited_opinion_id",
            "citing_canonical_opinion_id",
            "cited_canonical_opinion_id",
            "depth",
        ]
        if reader.fieldnames != expected_header:
            fail("citation output header mismatch", actual=reader.fieldnames)
        for line, row in enumerate(reader, 2):
            citing = positive_integer(
                row["citing_opinion_id"], path=str(root / CITATIONS_FILE), line=line, field="citing_opinion_id"
            )
            cited = positive_integer(
                row["cited_opinion_id"], path=str(root / CITATIONS_FILE), line=line, field="cited_opinion_id"
            )
            citing_canonical = positive_integer(
                row["citing_canonical_opinion_id"],
                path=str(root / CITATIONS_FILE),
                line=line,
                field="citing_canonical_opinion_id",
            )
            cited_canonical = positive_integer(
                row["cited_canonical_opinion_id"],
                path=str(root / CITATIONS_FILE),
                line=line,
                field="cited_canonical_opinion_id",
            )
            depth = integer(
                row["depth"], path=str(root / CITATIONS_FILE), line=line, field="depth"
            )
            if depth < 0:
                fail("physical citation depth is negative", line=line, depth=depth)
            if aliases.get(citing) != citing_canonical:
                fail("citing alias readback mismatch", line=line, opinion_id=citing)
            if aliases.get(cited) != cited_canonical:
                fail("cited alias readback mismatch", line=line, opinion_id=cited)
            source_entry = source_cited.setdefault(cited, [0, 0])
            source_entry[0] += 1
            source_entry[1] += depth
            canonical_entry = canonical_cited.setdefault(cited_canonical, [0, 0])
            canonical_entry[0] += 1
            canonical_entry[1] += depth
            depth_histogram[depth] = depth_histogram.get(depth, 0) + 1
            edges += 1
    with open(root / STATS_FILE, "r", encoding="utf-8") as source:
        stats = json.load(source)
    if stats.get("in_slice_edge_count") != edges:
        fail(
            "citation edge count differs from physical CSV",
            declared=stats.get("in_slice_edge_count"),
            actual=edges,
        )
    if manifest["row_counts"]["in_slice_edges"] != edges:
        fail("citation manifest edge count differs from physical CSV")
    expected_source_top = sorted(
        source_cited.items(), key=lambda item: (-item[1][0], -item[1][1], item[0])
    )[:TOP_LIMIT]
    physical_source_top = [
        {
            "opinion_id": opinion_id,
            "canonical_opinion_id": aliases[opinion_id],
            "in_slice_citing_edges": values[0],
            "total_depth": values[1],
        }
        for opinion_id, values in expected_source_top
    ]
    declared_source_top = [
        {key: row.get(key) for key in physical_source_top[0]}
        for row in stats.get("top20_source_opinions_cited", [])
    ] if physical_source_top else stats.get("top20_source_opinions_cited", [])
    if declared_source_top != physical_source_top:
        fail("source citation ranking differs from physical citation rows")
    expected_canonical_top = sorted(
        canonical_cited.items(), key=lambda item: (-item[1][0], -item[1][1], item[0])
    )[:TOP_LIMIT]
    physical_canonical_top = [
        {
            "canonical_opinion_id": opinion_id,
            "in_slice_citing_edges": values[0],
            "total_depth": values[1],
        }
        for opinion_id, values in expected_canonical_top
    ]
    declared_canonical_top = [
        {key: row.get(key) for key in physical_canonical_top[0]}
        for row in stats.get("top20_canonical_contents_cited", [])
    ] if physical_canonical_top else stats.get("top20_canonical_contents_cited", [])
    if declared_canonical_top != physical_canonical_top:
        fail("canonical citation ranking differs from physical citation rows")
    if stats.get("depth_histogram_in_slice") != {
        str(key): value for key, value in sorted(depth_histogram.items())
    }:
        fail("citation depth histogram differs from physical citation rows")
    frontier_edges = 0
    frontier = {}
    frontier_citation_ids = set()
    with open(root / FRONTIER_EDGES_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_header = [
            "citation_id",
            "citing_opinion_id",
            "citing_canonical_opinion_id",
            "cited_opinion_id",
            "depth",
        ]
        if reader.fieldnames != expected_header:
            fail("frontier edge header mismatch", actual=reader.fieldnames)
        for line, row in enumerate(reader, 2):
            citation_id = positive_integer(
                row["citation_id"],
                path=str(root / FRONTIER_EDGES_FILE),
                line=line,
                field="citation_id",
            )
            citing = positive_integer(
                row["citing_opinion_id"],
                path=str(root / FRONTIER_EDGES_FILE),
                line=line,
                field="citing_opinion_id",
            )
            citing_canonical = positive_integer(
                row["citing_canonical_opinion_id"],
                path=str(root / FRONTIER_EDGES_FILE),
                line=line,
                field="citing_canonical_opinion_id",
            )
            cited = positive_integer(
                row["cited_opinion_id"],
                path=str(root / FRONTIER_EDGES_FILE),
                line=line,
                field="cited_opinion_id",
            )
            depth = integer(
                row["depth"],
                path=str(root / FRONTIER_EDGES_FILE),
                line=line,
                field="depth",
            )
            if citation_id in frontier_citation_ids or depth < 0:
                fail("frontier edge has duplicate citation ID or negative depth", line=line)
            if aliases.get(citing) != citing_canonical or cited in aliases:
                fail(
                    "frontier edge does not cross from the slice to an external opinion",
                    line=line,
                    citing_opinion_id=citing,
                    cited_opinion_id=cited,
                )
            frontier_citation_ids.add(citation_id)
            entry = frontier.setdefault(cited, [0, 0])
            entry[0] += 1
            entry[1] += depth
            frontier_edges += 1

    frontier_rows = 0
    previous_rank = None
    frontier_ids = set()
    physical_frontier = []
    with open(root / FRONTIER_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != [
            "cited_opinion_id",
            "citing_edge_count",
            "total_depth",
        ]:
            fail("frontier output header mismatch", actual=reader.fieldnames)
        for line, row in enumerate(reader, 2):
            cited = positive_integer(
                row["cited_opinion_id"], path=str(root / FRONTIER_FILE), line=line, field="cited_opinion_id"
            )
            edge_count = positive_integer(
                row["citing_edge_count"], path=str(root / FRONTIER_FILE), line=line, field="citing_edge_count"
            )
            total_depth = integer(
                row["total_depth"], path=str(root / FRONTIER_FILE), line=line, field="total_depth"
            )
            if total_depth < 0 or cited in frontier_ids:
                fail("frontier row has invalid depth or duplicate opinion", line=line, opinion_id=cited)
            frontier_ids.add(cited)
            rank = (-edge_count, -total_depth, cited)
            if previous_rank is not None and rank < previous_rank:
                fail("frontier output is not deterministically sorted", line=line)
            previous_rank = rank
            physical_frontier.append((cited, edge_count, total_depth))
            frontier_rows += 1
    if frontier_rows > FRONTIER_LIMIT:
        fail("frontier output exceeds its declared limit", actual=frontier_rows)
    expected_frontier = [
        (opinion_id, values[0], values[1])
        for opinion_id, values in sorted(
            frontier.items(), key=lambda item: (-item[1][0], -item[1][1], item[0])
        )[:FRONTIER_LIMIT]
    ]
    if physical_frontier != expected_frontier:
        fail("frontier top-500 summary differs from physical frontier edges")
    parenthetical_rows = 0
    with open(root / PARENTHETICALS_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_header = [
            "id",
            "text",
            "score",
            "described_opinion_id",
            "describing_opinion_id",
            "described_canonical_opinion_id",
            "describing_canonical_opinion_id",
            "group_id",
        ]
        if reader.fieldnames != expected_header:
            fail("parenthetical output header mismatch", actual=reader.fieldnames)
        for line, row in enumerate(reader, 2):
            positive_integer(row["id"], path=str(root / PARENTHETICALS_FILE), line=line, field="id")
            described = positive_integer(
                row["described_opinion_id"], path=str(root / PARENTHETICALS_FILE), line=line, field="described_opinion_id"
            )
            describing = positive_integer(
                row["describing_opinion_id"], path=str(root / PARENTHETICALS_FILE), line=line, field="describing_opinion_id"
            )
            if described not in aliases and describing not in aliases:
                fail("parenthetical output does not touch the slice", line=line)
            expected_described = str(aliases[described]) if described in aliases else ""
            expected_describing = str(aliases[describing]) if describing in aliases else ""
            if (
                row["described_canonical_opinion_id"] != expected_described
                or row["describing_canonical_opinion_id"] != expected_describing
            ):
                fail("parenthetical alias readback mismatch", line=line)
            parenthetical_rows += 1
    physical_counts = {
        "in_slice_edges": edges,
        "frontier_edges": frontier_edges,
        "frontier_rows": frontier_rows,
        "parentheticals_kept": parenthetical_rows,
    }
    if manifest.get("row_counts") != physical_counts:
        fail(
            "citation manifest row counts differ from physical members",
            declared=manifest.get("row_counts"),
            actual=physical_counts,
        )
    if (
        stats.get("frontier_edge_count") != frontier_edges
        or stats.get("frontier_cited_opinion_count") != len(frontier)
        or stats.get("frontier_top500_rows") != frontier_rows
        or stats.get("parentheticals_kept") != parenthetical_rows
        or stats.get("source_opinion_ids") != len(aliases)
        or stats.get("canonical_content_ids") != len(set(aliases.values()))
    ):
        fail("citation stats differ from physical generation state")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "source_opinion_aliases": len(aliases),
        "in_slice_edges": edges,
        "frontier_edges": frontier_edges,
        "frontier_rows": frontier_rows,
        "parenthetical_rows": parenthetical_rows,
        "status": "verified",
    }


def build(args) -> dict:
    verify_extract_generation(args.extract_generation)
    ingest_report = verify_ingest_generation(args.ingest_generation, args.extract_generation)
    aliases = load_aliases(args.ingest_generation)
    names = load_case_names(args.extract_generation)
    if set(aliases) != set(names):
        fail(
            "ingest aliases and extracted opinion identities differ",
            alias_only=sorted(set(aliases) - set(names))[:20],
            extract_only=sorted(set(names) - set(aliases))[:20],
        )
    source_provenance = verify_bulk_sources(
        args.bulk_manifest,
        {"citation_map": args.citation_map, "parentheticals": args.parentheticals},
    )
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        citation_output = generation.open_text(CITATIONS_FILE, newline="")
        frontier_edges_output = generation.open_text(FRONTIER_EDGES_FILE, newline="")
        citation = stream_citations(
            args.citation_map, aliases, citation_output, frontier_edges_output
        )
        frontier_output = generation.open_text(FRONTIER_FILE, newline="")
        frontier_rows = write_frontier(citation["frontier"], frontier_output)
        parenthetical_output = generation.open_text(PARENTHETICALS_FILE, newline="")
        parentheticals = stream_parentheticals(
            args.parentheticals, aliases, parenthetical_output
        )
        top_source = sorted(
            citation["source_cited"].items(),
            key=lambda item: (-item[1][0], -item[1][1], item[0]),
        )[:TOP_LIMIT]
        top_canonical = sorted(
            citation["canonical_cited"].items(),
            key=lambda item: (-item[1][0], -item[1][1], item[0]),
        )[:TOP_LIMIT]
        touching = citation["edges_touching_slice"]
        stats = {
            "source_opinion_ids": len(aliases),
            "canonical_content_ids": len(set(aliases.values())),
            "citation_map_rows_scanned": citation["rows_scanned"],
            "in_slice_edge_count": citation["in_slice_edges"],
            "edges_touching_slice": touching,
            "in_slice_fraction_of_touching": (
                round(citation["in_slice_edges"] / touching, 6) if touching else None
            ),
            "incoming_only_edges": citation["incoming_only_edges"],
            "frontier_edge_count": touching
            - citation["in_slice_edges"]
            - citation["incoming_only_edges"],
            "frontier_cited_opinion_count": len(citation["frontier"]),
            "frontier_top500_rows": frontier_rows,
            "depth_histogram_in_slice": {
                str(key): value for key, value in sorted(citation["depth_histogram"].items())
            },
            "top20_source_opinions_cited": [
                {
                    "opinion_id": opinion_id,
                    "canonical_opinion_id": aliases[opinion_id],
                    "in_slice_citing_edges": values[0],
                    "total_depth": values[1],
                    "case_name": names.get(opinion_id, ""),
                }
                for opinion_id, values in top_source
            ],
            "top20_canonical_contents_cited": [
                {
                    "canonical_opinion_id": opinion_id,
                    "in_slice_citing_edges": values[0],
                    "total_depth": values[1],
                    "case_name": names.get(opinion_id, ""),
                }
                for opinion_id, values in top_canonical
            ],
            "parentheticals_rows_scanned": parentheticals["rows_scanned"],
            "parentheticals_kept": parentheticals["kept"],
        }
        generation.write_json(STATS_FILE, stats)
        row_counts = {
            "in_slice_edges": citation["in_slice_edges"],
            "frontier_edges": touching
            - citation["in_slice_edges"]
            - citation["incoming_only_edges"],
            "frontier_rows": frontier_rows,
            "parentheticals_kept": parentheticals["kept"],
        }
        generation.publish(
            {
                "format": FORMAT,
                "source_ingest_manifest_sha256": ingest_report["manifest_sha256"],
                "source_archives": source_provenance,
                "alias_contract": {
                    "file": ALIASES_FILE,
                    "source_opinion_ids": len(aliases),
                    "canonical_content_ids": len(set(aliases.values())),
                    "edge_columns": [
                        "citing_opinion_id",
                        "cited_opinion_id",
                        "citing_canonical_opinion_id",
                        "cited_canonical_opinion_id",
                    ],
                },
                "row_counts": row_counts,
                "source_of_truth": MANIFEST_FILE,
            }
        )
    return verify_citation_generation(args.out, args.ingest_generation)


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("build")
    command.add_argument("--extract-generation", required=True)
    command.add_argument("--ingest-generation", required=True)
    command.add_argument("--citation-map", required=True)
    command.add_argument("--parentheticals", required=True)
    command.add_argument("--bulk-manifest", required=True)
    command.add_argument("--out", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--generation", required=True)
    verify.add_argument("--ingest-generation", required=True)
    args = parse_cli_args(parser)
    try:
        report = (
            build(args)
            if args.command == "build"
            else verify_citation_generation(args.generation, args.ingest_generation)
        )
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_citation_generation_error",
            remediation=(
                "repair the named citation or ingest source mismatch and rebuild at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
