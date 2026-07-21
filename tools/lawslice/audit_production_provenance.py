#!/usr/bin/env python3
"""Audit canonical law provenance against physical Base and Anchors readbacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import traceback

from structured_error import StructuredArgumentParser, StructuredError, write_error


URL_PREFIX = "https://www.courtlistener.com/opinion/"
HEX_32 = re.compile(r"^[0-9a-f]{32}$")
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ANCHOR_LABELS = {"court", "precedential", "optype", "cited", "era", "dataset"}
REVERSED_SOURCE_IDS = {2748659, 2753578, 10771254, 11101864}
REJECTED_FALSE_POSITIVE_ID = 4678958
KNOWN_CANONICAL_ID = 4636687


class AuditError(StructuredError):
    code = "canonical_provenance_audit_failed"


class AuditArgumentParser(StructuredArgumentParser):
    def error(self, message: str) -> None:
        fail(
            message,
            remediation="provide every required immutable generation and physical readback path",
        )


def fail(message: str, *, remediation: str, **context):
    raise AuditError(message, remediation=remediation, **context)


def plain_file(value: str, *, label: str) -> Path:
    path = Path(value).absolute()
    if not path.is_file() or path.is_symlink():
        fail(
            "%s is not a plain file" % label,
            remediation="provide the exact immutable or DB-native readback file",
            path=str(path),
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                fail(
                    "JSONL contains a blank row",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                fail(
                    "JSONL row is invalid",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                    error=str(error),
                )
            if not isinstance(value, dict):
                fail(
                    "JSONL row is not an object",
                    remediation="rebuild and republish the immutable generation",
                    path=str(path),
                    line=line_number,
                )
            yield line_number, value


def positive_int(value, *, field: str, where: str) -> int:
    if isinstance(value, bool):
        parsed = 0
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 0
    if parsed <= 0 or str(parsed) != str(value):
        fail(
            "identity field is not a canonical positive integer",
            remediation="repair the source-bound generation before promotion",
            field=field,
            where=where,
            value=value,
        )
    return parsed


def canonical_url(cluster_id, slug, url, *, where: str) -> str:
    cluster = positive_int(cluster_id, field="cluster_id", where=where)
    if not isinstance(slug, str) or not SLUG.fullmatch(slug):
        fail(
            "cluster slug is not canonical",
            remediation="carry the authoritative cluster slug from CourtListener",
            where=where,
            slug=slug,
        )
    expected = "%s%d/%s/" % (URL_PREFIX, cluster, slug)
    if url != expected:
        fail(
            "source URL is not the exact cluster-plus-slug locator",
            remediation="rebuild from the corrected canonical ingest generation",
            where=where,
            expected=expected,
            actual=url,
        )
    return expected


def load_full_ingest(path: Path) -> dict[int, dict]:
    rows = {}
    for line_number, row in json_lines(path):
        metadata = row.get("metadata")
        anchors = row.get("anchors")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            fail(
                "full-ingest metadata is not a string map",
                remediation="rebuild the full immutable ingest batch",
                line=line_number,
            )
        opinion_id = positive_int(
            metadata.get("opinion_id"),
            field="opinion_id",
            where="full-ingest:%d" % line_number,
        )
        canonical_id = positive_int(
            metadata.get("canonical_opinion_id"),
            field="canonical_opinion_id",
            where="full-ingest:%d" % line_number,
        )
        if opinion_id != canonical_id or opinion_id in rows:
            fail(
                "full-ingest canonical identity is duplicated or inconsistent",
                remediation="rebuild the canonical ingest generation",
                line=line_number,
                opinion_id=opinion_id,
                canonical_opinion_id=canonical_id,
            )
        canonical_url(
            metadata.get("cluster_id"),
            metadata.get("cluster_slug"),
            metadata.get("source_url"),
            where="full-ingest:%d" % line_number,
        )
        for field in ("source_sha256", "normalized_text_sha256", "ingest_text_sha256"):
            if not HEX_64.fullmatch(metadata.get(field, "")):
                fail(
                    "full-ingest source digest is invalid",
                    remediation="rebuild the canonical ingest generation",
                    line=line_number,
                    field=field,
                    value=metadata.get(field),
                )
        if not isinstance(anchors, list):
            fail(
                "full-ingest anchors are absent",
                remediation="rebuild the anchored ingest batch",
                line=line_number,
            )
        expected_anchors = {}
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != {"kind", "value"}:
                fail(
                    "full-ingest anchor schema is invalid",
                    remediation="rebuild the anchored ingest batch",
                    line=line_number,
                )
            kind = anchor["kind"]
            value = anchor["value"]
            if not isinstance(kind, str) or not kind.startswith("label:") or not isinstance(value, str):
                fail(
                    "full-ingest anchor is not an enum label",
                    remediation="rebuild the anchored ingest batch",
                    line=line_number,
                    anchor=anchor,
                )
            label = kind.removeprefix("label:")
            if label in expected_anchors:
                fail(
                    "full-ingest anchor label is duplicated",
                    remediation="rebuild the anchored ingest batch",
                    line=line_number,
                    label=label,
                )
            expected_anchors[label] = value
        if set(expected_anchors) != ANCHOR_LABELS:
            fail(
                "full-ingest anchor panel differs",
                remediation="rebuild the anchored ingest batch",
                line=line_number,
                actual=sorted(expected_anchors),
            )
        rows[opinion_id] = {"metadata": metadata, "anchors": expected_anchors}
    return rows


def load_extract(path: Path) -> dict[int, dict]:
    rows = {}
    for line_number, row in json_lines(path):
        opinion_id = positive_int(
            row.get("opinion_id"), field="opinion_id", where="extract:%d" % line_number
        )
        if opinion_id in rows:
            fail(
                "extract opinion identity is duplicated",
                remediation="rebuild the canonical extract generation",
                opinion_id=opinion_id,
            )
        url = canonical_url(
            row.get("cluster_id"),
            row.get("cluster_slug"),
            row.get("canonical_source_url"),
            where="extract:%d" % line_number,
        )
        rows[opinion_id] = {
            "url": url,
            "cluster_id": row["cluster_id"],
            "cluster_slug": row["cluster_slug"],
            "case_name": row.get("case_name"),
            "docket_number": row.get("docket_number"),
            "correction": row.get("correction"),
        }
    return rows


def load_ingest_aliases(path: Path, extract: dict[int, dict]) -> dict[int, dict]:
    rows = {}
    for line_number, row in json_lines(path):
        opinion_id = positive_int(
            row.get("opinion_id"), field="opinion_id", where="alias:%d" % line_number
        )
        canonical_id = positive_int(
            row.get("canonical_opinion_id"),
            field="canonical_opinion_id",
            where="alias:%d" % line_number,
        )
        source = extract.get(opinion_id)
        if opinion_id in rows or source is None or row.get("source_url") != source["url"]:
            fail(
                "ingest alias differs from the accepted extract locator",
                remediation="rebuild the ingest alias relation",
                line=line_number,
                opinion_id=opinion_id,
            )
        if not HEX_64.fullmatch(row.get("content_sha256", "")):
            fail(
                "ingest alias content digest is invalid",
                remediation="rebuild the ingest alias relation",
                opinion_id=opinion_id,
            )
        rows[opinion_id] = {
            "canonical_opinion_id": canonical_id,
            "content_sha256": row["content_sha256"],
            "is_canonical": row.get("is_canonical"),
            "source_url": row["source_url"],
        }
    if set(rows) != set(extract):
        fail(
            "extract and ingest alias identity sets differ",
            remediation="rebuild the ingest generation from the accepted extract",
            extract_only=sorted(set(extract) - set(rows))[:20],
            alias_only=sorted(set(rows) - set(extract))[:20],
        )
    return rows


def load_base(path: Path, expected: dict[int, dict]) -> tuple[dict[int, str], dict[str, dict], set[int]]:
    try:
        with open(path, "r", encoding="utf-8") as source:
            values = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        fail(
            "cannot read Base CF JSON",
            remediation="repeat an unbounded DB-native Base readback",
            path=str(path),
            error=str(error),
        )
    if not isinstance(values, list):
        fail(
            "Base CF readback is not an array",
            remediation="repeat an unbounded DB-native Base readback",
            path=str(path),
        )
    by_opinion = {}
    by_cx = {}
    panel_versions = set()
    for index, row in enumerate(values, 1):
        if not isinstance(row, dict):
            fail(
                "Base CF row is not an object",
                remediation="repeat an unbounded DB-native Base readback",
                row=index,
            )
        cx_id = row.get("cx_id")
        metadata = row.get("metadata")
        if not isinstance(cx_id, str) or not HEX_32.fullmatch(cx_id) or cx_id in by_cx:
            fail(
                "Base CF constellation identity is invalid or duplicated",
                remediation="rebuild the production vault from the canonical batch",
                row=index,
                cx_id=cx_id,
            )
        if not isinstance(metadata, dict):
            fail(
                "Base CF metadata is absent",
                remediation="rebuild the production vault from the canonical batch",
                row=index,
                cx_id=cx_id,
            )
        opinion_id = positive_int(
            metadata.get("opinion_id"), field="opinion_id", where="base:%d" % index
        )
        wanted = expected.get(opinion_id)
        if wanted is None or metadata != wanted["metadata"]:
            differing = (
                sorted(
                    key
                    for key in set(metadata) | set(wanted["metadata"])
                    if metadata.get(key) != wanted["metadata"].get(key)
                )[:20]
                if wanted is not None
                else []
            )
            fail(
                "Base CF metadata differs from full-ingest bytes",
                remediation="rebuild and promote a fresh canonical vault",
                row=index,
                opinion_id=opinion_id,
                differing_keys=differing,
            )
        if opinion_id in by_opinion:
            fail(
                "Base CF duplicates a canonical opinion",
                remediation="rebuild and promote a fresh canonical vault",
                opinion_id=opinion_id,
            )
        panel_version = row.get("panel_version")
        if not isinstance(panel_version, int) or isinstance(panel_version, bool) or panel_version <= 0:
            fail(
                "Base CF panel version is invalid",
                remediation="rebuild and promote a fresh canonical vault",
                row=index,
                panel_version=panel_version,
            )
        by_opinion[opinion_id] = cx_id
        by_cx[cx_id] = wanted["anchors"]
        panel_versions.add(panel_version)
    if set(by_opinion) != set(expected):
        fail(
            "Base CF and full-ingest identity sets differ",
            remediation="rebuild and promote a fresh canonical vault",
            base_only=sorted(set(by_opinion) - set(expected))[:20],
            ingest_only=sorted(set(expected) - set(by_opinion))[:20],
        )
    return by_opinion, by_cx, panel_versions


def load_vault_aliases(
    path: Path, aliases: dict[int, dict], base: dict[int, str]
) -> dict:
    seen = set()
    noncanonical = 0
    with open(path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_header = [
            "opinion_id",
            "cx_id",
            "canonical_opinion_id",
            "content_sha256",
            "is_canonical",
            "source_url",
        ]
        if reader.fieldnames != expected_header:
            fail(
                "vault alias header differs",
                remediation="rebuild the physical alias generation",
                actual=reader.fieldnames,
            )
        for line_number, row in enumerate(reader, 2):
            opinion_id = positive_int(
                row.get("opinion_id"), field="opinion_id", where="vault-alias:%d" % line_number
            )
            canonical_id = positive_int(
                row.get("canonical_opinion_id"),
                field="canonical_opinion_id",
                where="vault-alias:%d" % line_number,
            )
            wanted = aliases.get(opinion_id)
            expected = {
                "opinion_id": str(opinion_id),
                "cx_id": base.get(canonical_id),
                "canonical_opinion_id": str(canonical_id),
                "content_sha256": wanted.get("content_sha256") if wanted else None,
                "is_canonical": "true" if opinion_id == canonical_id else "false",
                "source_url": wanted.get("source_url") if wanted else None,
            }
            if opinion_id in seen or row != expected:
                fail(
                    "vault alias row differs from the source and Base relation",
                    remediation="rebuild the physical alias generation",
                    line=line_number,
                    opinion_id=opinion_id,
                    expected=expected,
                    actual=row,
                )
            seen.add(opinion_id)
            noncanonical += opinion_id != canonical_id
    if seen != set(aliases):
        fail(
            "vault alias identity set differs",
            remediation="rebuild the physical alias generation",
            missing=sorted(set(aliases) - seen)[:20],
            extra=sorted(seen - set(aliases))[:20],
        )
    return {"rows": len(seen), "noncanonical_aliases": noncanonical}


class Cursor:
    def __init__(self, value: bytes):
        self.value = value
        self.offset = 0

    def take(self, count: int) -> bytes:
        end = self.offset + count
        if count < 0 or end > len(self.value):
            fail(
                "anchor encoding is truncated",
                remediation="rebuild the Anchors CF from the canonical batch",
                offset=self.offset,
                requested=count,
                length=len(self.value),
            )
        result = self.value[self.offset:end]
        self.offset = end
        return result

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "big")

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "big")

    def string32(self) -> str:
        try:
            return self.take(self.u32()).decode("utf-8")
        except UnicodeDecodeError as error:
            fail(
                "anchor string is not UTF-8",
                remediation="rebuild the Anchors CF from the canonical batch",
                error=str(error),
            )

    def finish(self):
        if self.offset != len(self.value):
            fail(
                "anchor encoding has trailing bytes",
                remediation="rebuild the Anchors CF from the canonical batch",
                consumed=self.offset,
                length=len(self.value),
            )


def decode_anchor_key(value: bytes) -> tuple[str, str]:
    cursor = Cursor(value)
    cx_id = cursor.take(16).hex()
    if cursor.u16() != 7:
        fail(
            "anchor key is not a label key",
            remediation="rebuild the Anchors CF from the canonical batch",
            cx_id=cx_id,
        )
    length = cursor.u64()
    try:
        label = cursor.take(length).decode("utf-8")
    except UnicodeDecodeError as error:
        fail(
            "anchor key label is not UTF-8",
            remediation="rebuild the Anchors CF from the canonical batch",
            cx_id=cx_id,
            error=str(error),
        )
    cursor.finish()
    return cx_id, label


def decode_anchor_value(value: bytes) -> dict:
    cursor = Cursor(value)
    if cursor.u16() != 3:
        fail(
            "anchor value kind is not Label",
            remediation="rebuild the Anchors CF from the canonical batch",
        )
    label = cursor.string32()
    if cursor.u8() != 1:
        fail(
            "anchor payload is not Enum",
            remediation="rebuild the Anchors CF from the canonical batch",
            label=label,
        )
    enum = cursor.string32()
    source = cursor.string32()
    observed_at = cursor.u64()
    confidence = struct.unpack(">f", cursor.take(4))[0]
    cursor.finish()
    if not source or observed_at <= 0 or not math.isfinite(confidence) or not 0 < confidence <= 1:
        fail(
            "anchor grounding fields are invalid",
            remediation="rebuild the Anchors CF from the canonical batch",
            label=label,
            source=source,
            observed_at=observed_at,
            confidence=confidence,
        )
    return {
        "label": label,
        "enum": enum,
        "source": source,
        "observed_at": observed_at,
        "confidence": confidence,
    }


def audit_anchors(path: Path, expected: dict[str, dict]) -> dict:
    physical_rows = 0
    values_by_key = {}
    copies_by_key = {}
    files = set()
    sources = set()
    observed_at = set()
    confidences = set()
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8 or fields[0::2] != ["CF", "FILE", "KEY", "VALUE"] or fields[1] != "anchors":
                fail(
                    "Anchors CF readback row schema differs",
                    remediation="repeat the raw DB-native Anchors CF readback",
                    line=line_number,
                )
            try:
                key_bytes = bytes.fromhex(fields[5])
                value_bytes = bytes.fromhex(fields[7])
            except ValueError as error:
                fail(
                    "Anchors CF readback contains invalid hex",
                    remediation="repeat the raw DB-native Anchors CF readback",
                    line=line_number,
                    error=str(error),
                )
            physical_rows += 1
            files.add(fields[3])
            key = fields[5]
            prior = values_by_key.get(key)
            if prior is not None and prior != fields[7]:
                fail(
                    "physical Anchor versions disagree",
                    remediation="repair or rebuild the Anchors CF before promotion",
                    line=line_number,
                    key=key,
                )
            values_by_key[key] = fields[7]
            copies_by_key[key] = copies_by_key.get(key, 0) + 1
            cx_id, key_label = decode_anchor_key(key_bytes)
            decoded = decode_anchor_value(value_bytes)
            wanted = expected.get(cx_id)
            if (
                wanted is None
                or decoded["label"] != key_label
                or wanted.get(key_label) != decoded["enum"]
            ):
                fail(
                    "physical Anchor differs from its constellation batch",
                    remediation="rebuild the Anchors CF from the canonical batch",
                    line=line_number,
                    cx_id=cx_id,
                    label=key_label,
                    expected=wanted.get(key_label) if wanted else None,
                    actual=decoded,
                )
            sources.add(decoded["source"])
            observed_at.add(decoded["observed_at"])
            confidences.add(decoded["confidence"])
    labels_by_cx = {}
    for key in values_by_key:
        cx_id, label = decode_anchor_key(bytes.fromhex(key))
        labels_by_cx.setdefault(cx_id, set()).add(label)
    if set(labels_by_cx) != set(expected) or any(
        labels != ANCHOR_LABELS for labels in labels_by_cx.values()
    ):
        fail(
            "logical Anchor key set differs from the canonical constellation panel",
            remediation="rebuild the Anchors CF from the canonical batch",
            cx_count=len(labels_by_cx),
        )
    ordered_files = sorted(files)
    ordered_observed_at = sorted(observed_at)
    return {
        "physical_rows": physical_rows,
        "logical_rows": len(values_by_key),
        "constellations": len(labels_by_cx),
        "anchors_per_constellation": sorted({len(value) for value in labels_by_cx.values()}),
        "physical_versions_per_key": sorted(set(copies_by_key.values())),
        "physical_files": {
            "count": len(ordered_files),
            "first": ordered_files[0] if ordered_files else None,
            "last": ordered_files[-1] if ordered_files else None,
        },
        "sources": sorted(sources),
        "observed_at": {
            "unique": len(ordered_observed_at),
            "minimum": ordered_observed_at[0] if ordered_observed_at else None,
            "maximum": ordered_observed_at[-1] if ordered_observed_at else None,
        },
        "confidences": sorted(confidences),
    }


def special_rows(
    extract: dict[int, dict], aliases: dict[int, dict], base: dict[int, str], full: dict[int, dict]
) -> dict:
    if REJECTED_FALSE_POSITIVE_ID in extract or REJECTED_FALSE_POSITIVE_ID in aliases or REJECTED_FALSE_POSITIVE_ID in base:
        fail(
            "known rejected false positive survives in canonical state",
            remediation="rebuild from the corrected Cuyahoga selector",
            opinion_id=REJECTED_FALSE_POSITIVE_ID,
        )
    known = full.get(KNOWN_CANONICAL_ID)
    if known is None:
        fail(
            "known canonical URL specimen is absent",
            remediation="rebuild from the accepted Cuyahoga extract",
            opinion_id=KNOWN_CANONICAL_ID,
        )
    corrections = []
    for opinion_id in sorted(REVERSED_SOURCE_IDS):
        source = extract.get(opinion_id)
        stored = full.get(opinion_id)
        correction = source.get("correction") if source else None
        if (
            source is None
            or stored is None
            or not isinstance(correction, dict)
            or correction.get("corrected")
            != {
                "case_name": stored["metadata"].get("case_name"),
                "docket_number": stored["metadata"].get("docket_number"),
            }
            or stored["metadata"].get("correction_id") != correction.get("correction_id")
        ):
            fail(
                "reviewed reversed-field correction is absent or inconsistent",
                remediation="rebuild using the reviewed source-bound correction manifest",
                opinion_id=opinion_id,
            )
        corrections.append(
            {
                "opinion_id": opinion_id,
                "cx_id": base[opinion_id],
                "case_name": stored["metadata"]["case_name"],
                "docket_number": stored["metadata"]["docket_number"],
                "correction_id": stored["metadata"]["correction_id"],
                "source_url": stored["metadata"]["source_url"],
            }
        )
    return {
        "known_canonical": {
            "opinion_id": KNOWN_CANONICAL_ID,
            "cx_id": base[KNOWN_CANONICAL_ID],
            "source_url": known["metadata"]["source_url"],
        },
        "reviewed_corrections": corrections,
        "rejected_false_positive": {
            "opinion_id": REJECTED_FALSE_POSITIVE_ID,
            "present_in_extract": False,
            "present_in_aliases": False,
            "present_in_base": False,
        },
    }


def audit(args) -> dict:
    paths = {
        "base_readback": plain_file(args.base_readback, label="Base CF readback"),
        "anchors_readback": plain_file(args.anchors_readback, label="Anchors CF readback"),
        "full_ingest": plain_file(args.full_ingest, label="full ingest batch"),
        "extract_rows": plain_file(args.extract_rows, label="extract rows"),
        "ingest_aliases": plain_file(args.ingest_aliases, label="ingest aliases"),
        "vault_aliases": plain_file(args.vault_aliases, label="vault aliases"),
    }
    full = load_full_ingest(paths["full_ingest"])
    extract = load_extract(paths["extract_rows"])
    aliases = load_ingest_aliases(paths["ingest_aliases"], extract)
    base, anchors_by_cx, panel_versions = load_base(paths["base_readback"], full)
    if {row["canonical_opinion_id"] for row in aliases.values()} != set(base):
        fail(
            "source aliases do not resolve to the complete Base identity set",
            remediation="rebuild the physical alias generation",
        )
    alias_report = load_vault_aliases(paths["vault_aliases"], aliases, base)
    anchor_report = audit_anchors(paths["anchors_readback"], anchors_by_cx)
    specials = special_rows(extract, aliases, base, full)
    return {
        "status": "verified",
        "counts": {
            "accepted_source_opinions": len(extract),
            "ingest_alias_rows": len(aliases),
            "canonical_constellations": len(base),
            "vault_alias_rows": alias_report["rows"],
            "duplicate_aliases": alias_report["noncanonical_aliases"],
        },
        "url_contract": {
            "format": URL_PREFIX + "{cluster_id}/{cluster_slug}/",
            "extract_rows_valid": len(extract),
            "ingest_alias_rows_valid": len(aliases),
            "base_rows_valid": len(base),
            "vault_alias_rows_valid": alias_report["rows"],
            "invalid": 0,
        },
        "base": {
            "metadata_exact_full_ingest": len(base),
            "panel_versions": sorted(panel_versions),
            "source_sha256_rows": sum(
                bool(HEX_64.fullmatch(row["metadata"].get("source_sha256", "")))
                for row in full.values()
            ),
        },
        "anchors": anchor_report,
        "specials": specials,
        "sources": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
    }


def parser() -> argparse.ArgumentParser:
    root = AuditArgumentParser(description=__doc__)
    root.add_argument("--base-readback", required=True)
    root.add_argument("--anchors-readback", required=True)
    root.add_argument("--full-ingest", required=True)
    root.add_argument("--extract-rows", required=True)
    root.add_argument("--ingest-aliases", required=True)
    root.add_argument("--vault-aliases", required=True)
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
    except AuditError as error:
        write_error(
            error,
            code="canonical_provenance_audit_failed",
            remediation="repair the reported provenance authority before promotion",
            include_traceback=False,
        )
        return 1
    except Exception as error:
        write_error(
            error,
            code="canonical_provenance_audit_unhandled",
            remediation="inspect the traceback and add a typed fail-closed audit path",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
