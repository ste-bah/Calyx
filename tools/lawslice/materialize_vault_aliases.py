#!/usr/bin/env python3
"""Bind every source opinion alias to a physical Calyx Base-CF constellation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

from build_ingest_jsonl import (
    ALIASES_FILE as INGEST_ALIASES_FILE,
    CAP_FILE,
    MANIFEST_FILE as INGEST_MANIFEST,
    MODERN_FILE,
    verify_ingest_generation,
)
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


FORMAT = "calyx-cuyahoga-vault-alias-generation-v1"
MANIFEST_FILE = "vault_alias_manifest.json"
ALIASES_FILE = "opinion_cx_aliases.csv"
CSV_HEADER = (
    "opinion_id",
    "cx_id",
    "canonical_opinion_id",
    "content_sha256",
    "is_canonical",
    "source_url",
)
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class VaultAliasError(StructuredError):
    code = "cuyahoga_vault_alias_generation_error"
    default_remediation = (
        "repair the named ingest or physical Base mismatch and rebuild the alias relation"
    )


def fail(message: str, **context):
    raise VaultAliasError(message, **context)


def positive_int(value, *, field: str, where: str) -> int:
    if isinstance(value, bool):
        fail("positive integer field is boolean", field=field, where=where)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail("positive integer field is invalid", field=field, where=where, value=value)
    if parsed <= 0 or str(parsed) != str(value):
        fail("positive integer field is not canonical", field=field, where=where, value=value)
    return parsed


def json_lines(path: Path):
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


def load_ingest(ingest_generation: str) -> dict:
    verification = verify_ingest_generation(ingest_generation)
    batches = {}
    for member in (CAP_FILE, MODERN_FILE):
        path = generation_member(ingest_generation, INGEST_MANIFEST, member)
        for line_number, batch in json_lines(path):
            metadata = batch.get("metadata")
            if not isinstance(metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            ):
                fail("ingest metadata is not a string map", file=member, line=line_number)
            opinion_id = positive_int(
                metadata.get("opinion_id"), field="opinion_id", where="%s:%d" % (member, line_number)
            )
            canonical_id = positive_int(
                metadata.get("canonical_opinion_id"),
                field="canonical_opinion_id",
                where="%s:%d" % (member, line_number),
            )
            if opinion_id != canonical_id or opinion_id in batches:
                fail(
                    "canonical ingest identity is invalid or duplicated",
                    file=member,
                    line=line_number,
                    opinion_id=opinion_id,
                    canonical_opinion_id=canonical_id,
                )
            digest = metadata.get("ingest_text_sha256", "")
            if not HEX_64.fullmatch(digest):
                fail("canonical ingest digest is invalid", opinion_id=opinion_id, digest=digest)
            batches[opinion_id] = {
                "metadata": metadata,
                "content_sha256": digest,
            }

    aliases = {}
    alias_path = generation_member(ingest_generation, INGEST_MANIFEST, INGEST_ALIASES_FILE)
    for line_number, alias in json_lines(alias_path):
        opinion_id = positive_int(
            alias.get("opinion_id"), field="opinion_id", where="alias:%d" % line_number
        )
        canonical_id = positive_int(
            alias.get("canonical_opinion_id"),
            field="canonical_opinion_id",
            where="alias:%d" % line_number,
        )
        if opinion_id in aliases:
            fail("duplicate ingest opinion alias", line=line_number, opinion_id=opinion_id)
        canonical = batches.get(canonical_id)
        if canonical is None:
            fail("opinion alias has no canonical ingest row", line=line_number, opinion_id=opinion_id)
        digest = alias.get("content_sha256")
        source_url = alias.get("source_url")
        if digest != canonical["content_sha256"]:
            fail("opinion alias digest differs from canonical ingest row", opinion_id=opinion_id)
        if not isinstance(source_url, str) or not source_url.startswith("https://www.courtlistener.com/opinion/"):
            fail("opinion alias source URL is invalid", opinion_id=opinion_id, source_url=source_url)
        if alias.get("is_canonical") is not (opinion_id == canonical_id):
            fail("opinion alias canonical flag is invalid", opinion_id=opinion_id)
        aliases[opinion_id] = {
            "canonical_opinion_id": canonical_id,
            "content_sha256": digest,
            "is_canonical": opinion_id == canonical_id,
            "source_url": source_url,
        }

    if set(batches) != {row["canonical_opinion_id"] for row in aliases.values()}:
        fail("canonical ingest and opinion alias identity sets differ")
    if verification["canonical_content_rows"] != len(batches):
        fail("ingest verification canonical count differs from physical batches")
    if verification["source_opinion_rows"] != len(aliases):
        fail("ingest verification alias count differs from physical aliases")
    return {"verification": verification, "batches": batches, "aliases": aliases}


def load_physical_base(cx_list_path: str, batches: dict) -> dict:
    path = Path(cx_list_path).absolute()
    try:
        with open(path, "r", encoding="utf-8") as source:
            rows = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        fail("cannot read physical cx-list JSON", path=str(path), error=str(error))
    if not isinstance(rows, list) or not rows:
        fail("physical cx-list must be a nonempty JSON array", path=str(path))
    by_opinion = {}
    seen_cx = set()
    panel_versions = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            fail("physical cx-list row is not an object", row=index)
        cx_id = row.get("cx_id")
        if not isinstance(cx_id, str) or not HEX_32.fullmatch(cx_id):
            fail("physical cx-list row has invalid or absent cx_id", row=index, cx_id=cx_id)
        if cx_id in seen_cx:
            fail("physical cx-list duplicates cx_id", row=index, cx_id=cx_id)
        seen_cx.add(cx_id)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            fail("physical Base metadata is not a string map", row=index, cx_id=cx_id)
        opinion_id = positive_int(
            metadata.get("opinion_id"), field="opinion_id", where="cx-list:%d" % index
        )
        canonical_id = positive_int(
            metadata.get("canonical_opinion_id"),
            field="canonical_opinion_id",
            where="cx-list:%d" % index,
        )
        if opinion_id != canonical_id:
            fail("physical Base row is not a canonical opinion", row=index, opinion_id=opinion_id)
        expected = batches.get(opinion_id)
        if expected is None:
            fail("physical Base opinion is absent from ingest generation", row=index, opinion_id=opinion_id)
        if metadata != expected["metadata"]:
            differing = sorted(
                key
                for key in set(metadata) | set(expected["metadata"])
                if metadata.get(key) != expected["metadata"].get(key)
            )
            fail(
                "physical Base metadata differs from canonical ingest bytes",
                row=index,
                opinion_id=opinion_id,
                differing_keys=differing[:20],
            )
        input_ref = row.get("input_ref")
        if not isinstance(input_ref, dict) or not isinstance(input_ref.get("pointer"), str) or not input_ref["pointer"]:
            fail("physical Base row has no retained input pointer", row=index, opinion_id=opinion_id)
        panel_version = row.get("panel_version")
        if not isinstance(panel_version, int) or isinstance(panel_version, bool) or panel_version <= 0:
            fail("physical Base row has invalid panel version", row=index, panel_version=panel_version)
        panel_versions.add(panel_version)
        if opinion_id in by_opinion:
            fail("physical Base duplicates canonical opinion", opinion_id=opinion_id)
        by_opinion[opinion_id] = cx_id
    if set(by_opinion) != set(batches):
        fail(
            "physical Base and canonical ingest identity sets differ",
            base_only=sorted(set(by_opinion) - set(batches))[:20],
            ingest_only=sorted(set(batches) - set(by_opinion))[:20],
        )
    if len(panel_versions) != 1:
        fail("physical Base rows span multiple panel versions", panel_versions=sorted(panel_versions))
    return {
        "path": path,
        "by_opinion": by_opinion,
        "panel_version": next(iter(panel_versions)),
        "sha256": sha256_file(path),
    }


def expected_rows(ingest: dict, physical: dict) -> list[tuple[str, ...]]:
    rows = []
    for opinion_id, alias in sorted(ingest["aliases"].items()):
        canonical_id = alias["canonical_opinion_id"]
        rows.append(
            (
                str(opinion_id),
                physical["by_opinion"][canonical_id],
                str(canonical_id),
                alias["content_sha256"],
                "true" if alias["is_canonical"] else "false",
                alias["source_url"],
            )
        )
    return rows


def verify_vault_alias_generation(
    path: str, ingest_generation: str | None = None, cx_list: str | None = None
) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    if manifest.get("format") != FORMAT:
        fail("unsupported vault alias generation format", format=manifest.get("format"))
    if set(manifest["files"]) != {ALIASES_FILE}:
        fail("vault alias generation member contract mismatch", files=sorted(manifest["files"]))
    rows = []
    opinion_ids = set()
    canonical_cx = {}
    with open(root / ALIASES_FILE, "r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = tuple(next(reader, []))
        if header != CSV_HEADER:
            fail("vault alias CSV header mismatch", expected=CSV_HEADER, actual=header)
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(CSV_HEADER):
                fail("vault alias CSV row width mismatch", line=line_number, fields=len(row))
            opinion_id, cx_id, canonical_id, digest, is_canonical, source_url = row
            opinion = positive_int(opinion_id, field="opinion_id", where="csv:%d" % line_number)
            canonical = positive_int(
                canonical_id, field="canonical_opinion_id", where="csv:%d" % line_number
            )
            if opinion in opinion_ids or not HEX_32.fullmatch(cx_id) or not HEX_64.fullmatch(digest):
                fail("vault alias CSV identity is invalid or duplicated", line=line_number)
            opinion_ids.add(opinion)
            expected_flag = "true" if opinion == canonical else "false"
            if is_canonical != expected_flag or not source_url.startswith("https://www.courtlistener.com/opinion/"):
                fail("vault alias CSV provenance is invalid", line=line_number)
            if opinion == canonical:
                canonical_cx[canonical] = cx_id
            rows.append(tuple(row))
    if not rows:
        fail("vault alias CSV has no data rows")
    for row in rows:
        canonical = int(row[2])
        if canonical_cx.get(canonical) != row[1]:
            fail("vault alias resolves to a different cx than its canonical opinion", opinion_id=row[0])
    accounting = manifest.get("accounting")
    if not isinstance(accounting, dict) or accounting != {
        "source_opinion_rows": len(rows),
        "canonical_content_rows": len(canonical_cx),
        "duplicate_aliases": len(rows) - len(canonical_cx),
        "physical_base_rows": len(canonical_cx),
    }:
        fail("vault alias manifest accounting differs from physical CSV", actual=accounting)

    result = {
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "aliases_sha256": sha256_file(root / ALIASES_FILE),
        **accounting,
        "panel_version": manifest.get("panel_version"),
    }
    if ingest_generation is not None or cx_list is not None:
        if ingest_generation is None or cx_list is None:
            fail("verification against sources requires both ingest generation and cx-list")
        ingest = load_ingest(ingest_generation)
        physical = load_physical_base(cx_list, ingest["batches"])
        if rows != expected_rows(ingest, physical):
            fail("vault alias CSV differs from independently reconstructed source relation")
        if manifest.get("panel_version") != physical["panel_version"]:
            fail("vault alias manifest panel version differs from physical Base rows")
        sources = manifest.get("sources")
        expected_sources = {
            "ingest_generation": str(Path(ingest_generation).absolute()),
            "ingest_manifest_sha256": ingest["verification"]["manifest_sha256"],
            "cx_list": str(physical["path"]),
            "cx_list_sha256": physical["sha256"],
        }
        if sources != expected_sources:
            fail("vault alias manifest source contract differs from physical inputs")
    return result


def build(args) -> dict:
    ingest = load_ingest(args.ingest_generation)
    physical = load_physical_base(args.cx_list, ingest["batches"])
    rows = expected_rows(ingest, physical)
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        output = generation.open_text(ALIASES_FILE, newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
        generation.publish(
            {
                "format": FORMAT,
                "sources": {
                    "ingest_generation": str(Path(args.ingest_generation).absolute()),
                    "ingest_manifest_sha256": ingest["verification"]["manifest_sha256"],
                    "cx_list": str(physical["path"]),
                    "cx_list_sha256": physical["sha256"],
                },
                "panel_version": physical["panel_version"],
                "accounting": {
                    "source_opinion_rows": len(rows),
                    "canonical_content_rows": len(physical["by_opinion"]),
                    "duplicate_aliases": len(rows) - len(physical["by_opinion"]),
                    "physical_base_rows": len(physical["by_opinion"]),
                },
                "proof": {
                    "source_of_truth": "calyx readback cx-list physical Base CF rows",
                    "canonical_metadata_exact_match": True,
                    "retained_input_pointer_required": True,
                    "all_source_aliases_expanded": True,
                },
            }
        )
    return verify_vault_alias_generation(args.out, args.ingest_generation, args.cx_list)


def parser() -> argparse.ArgumentParser:
    root = StructuredArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--ingest-generation", required=True)
    build_parser.add_argument("--cx-list", required=True)
    build_parser.add_argument("--out", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--generation", required=True)
    verify_parser.add_argument("--ingest-generation")
    verify_parser.add_argument("--cx-list")
    return root


def main() -> int:
    args = parse_cli_args(parser())
    try:
        result = (
            build(args)
            if args.command == "build"
            else verify_vault_alias_generation(
                args.generation, args.ingest_generation, args.cx_list
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        write_error(
            error,
            code="cuyahoga_vault_alias_generation_error",
            remediation=(
                "repair the named ingest or physical Base mismatch and rebuild the alias relation"
            ),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
