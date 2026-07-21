#!/usr/bin/env python3
"""Fail-closed Cuyahoga selection, text, correction, and provenance contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from structured_error import StructuredError


SELECTOR_VERSION = "cuyahoga-direct-evidence-v2"
TEXT_POLICY_VERSION = "courtlistener-html-with-citations-v2"
CORRECTION_POLICY_VERSION = "exact-source-bound-corrections-v2"
EXPECTED_CORRECTION_IDENTITIES = {
    2593342: (2748659, ((2748659, "accepted"),)),
    2607299: (2753578, ((2753578, "accepted"),)),
    7532664: (4521320, ((4298573, "rejected"),)),
    69498717: (10304666, ((10771254, "accepted"),)),
    70827619: (10635277, ((11101864, "accepted"),)),
}
CORRECTION_PARTITIONS = frozenset({"accepted", "rejected", "conflict"})

CAP_COURTS = frozenset(
    {
        "ohctapp8",
        "ohctapp8cuyahog",
        "ohctcomplcuyaho",
        "ohcirctcuyahoga",
        "ohjuvctcuyahoga",
        "ohmunictcuyahog",
        "ohprobctcuyahog",
        "ohctinsolvcuyah",
    }
)
GENERIC_OHIO_APPEALS_COURT = "ohioctapp"
TEXT_FIELDS = (
    "html_with_citations",
    "plain_text",
    "html_columbia",
    "html_lawbox",
    "html_anon_2020",
    "xml_harvard",
    "html",
)


class ContractError(StructuredError):
    code = "contract_error"
    default_remediation = (
        "repair the named source or selection contract and rebuild at a new destination"
    )


class AuthoritativeTextError(ContractError):
    code = "authoritative_text_error"
    default_remediation = (
        "provide nonempty authoritative text from the bound source or a reviewed exact supplement"
    )


class EvidenceConflictError(ContractError):
    code = "issuing_court_evidence_conflict"
    default_remediation = (
        "quarantine the conflicting issuing-court evidence and resolve it before publication"
    )


class CorrectionError(ContractError):
    code = "correction_contract_error"
    default_remediation = (
        "restore the exact reviewed source and correction contract before publication"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_TRAIL_RE = re.compile(r"[ \t]+\n")
_BLANK_RE = re.compile(r"\n{3,}")


def normalize_lines(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _TRAIL_RE.sub("\n", value)
    value = _BLANK_RE.sub("\n\n", value)
    return value.strip()


class _OpinionHTMLParser(HTMLParser):
    BLOCKS = frozenset(
        {
            "address",
            "article",
            "blockquote",
            "br",
            "center",
            "div",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "opinion",
            "p",
            "paragraph",
            "section",
            "table",
            "td",
            "th",
            "tr",
        }
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.blocks: list[str] = []
        self._block_parts: list[str] = []
        self._ignored_depth = 0
        self._tag_count = 0

    def _boundary(self) -> None:
        text = _SPACE_RE.sub(" ", " ".join(self._block_parts)).strip()
        if text and (not self.blocks or self.blocks[-1] != text):
            self.blocks.append(text)
        self._block_parts.clear()
        if self.parts and self.parts[-1] != "\n":
            self.parts.append("\n")

    def handle_starttag(self, tag, attrs):
        del attrs
        lowered = tag.lower()
        self._tag_count += 1
        if lowered in {"script", "style"}:
            self._ignored_depth += 1
            return
        if not self._ignored_depth and lowered in self.BLOCKS:
            self._boundary()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.lower() in {"script", "style"}:
            self._ignored_depth -= 1

    def handle_endtag(self, tag):
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            if self._ignored_depth == 0:
                raise AuthoritativeTextError("unmatched closing HTML tag", tag=lowered)
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and lowered in self.BLOCKS:
            self._boundary()

    def handle_data(self, data):
        if self._ignored_depth or not data:
            return
        self.parts.append(data)
        self._block_parts.append(data)

    def finish(self) -> tuple[str, list[str]]:
        self.close()
        if self._ignored_depth:
            raise AuthoritativeTextError(
                "unclosed script/style element in html_with_citations"
            )
        self._boundary()
        # CourtListener frequently stores the entire opinion inside one <pre>
        # node. Newlines are structural boundaries for preformatted content;
        # retaining that node as one block would let body headings contaminate
        # the caption. Expand every DOM block into its nonempty visual lines.
        expanded = []
        for block in self.blocks:
            for line in block.splitlines():
                normalized = _SPACE_RE.sub(" ", line).strip()
                if normalized and (not expanded or expanded[-1] != normalized):
                    expanded.append(normalized)
        return normalize_lines("".join(self.parts)), expanded


@dataclass(frozen=True)
class ResolvedText:
    text: str
    source_field: str
    source_raw_sha256: str
    normalized_sha256: str
    normalized_chars: int
    caption_blocks: tuple[str, ...]
    fields: dict

    def provenance(self) -> dict:
        value = asdict(self)
        value.pop("text")
        return value


def resolve_text(values: dict, *, where: str) -> ResolvedText:
    fields = {}
    for name in TEXT_FIELDS:
        raw = values.get(name)
        if raw is not None and not isinstance(raw, str):
            raise AuthoritativeTextError(
                "CourtListener text field is not a string", where=where, field=name
            )
        raw = raw or ""
        fields[name] = {
            "raw_present": values.get(name) is not None,
            "nonempty": bool(raw.strip()),
            "raw_chars": len(raw),
            "raw_sha256": sha256_text(raw) if raw else None,
        }

    preferred = values.get("html_with_citations") or ""
    if preferred and not preferred.strip():
        raise AuthoritativeTextError(
            "html_with_citations is present but contains only whitespace",
            where=where,
        )
    if preferred:
        if "\x00" in preferred:
            raise AuthoritativeTextError("NUL byte in html_with_citations", where=where)
        parser = _OpinionHTMLParser()
        try:
            parser.feed(preferred)
            normalized, blocks = parser.finish()
        except ContractError:
            raise
        except Exception as error:
            raise AuthoritativeTextError(
                "cannot structurally parse html_with_citations",
                where=where,
                error_type=type(error).__name__,
                error=str(error),
            ) from error
        if not normalized:
            raise AuthoritativeTextError(
                "nonempty html_with_citations produced empty authoritative text",
                where=where,
            )
        source_field = "html_with_citations"
        raw = preferred
    else:
        plain = values.get("plain_text") or ""
        if plain and not plain.strip():
            raise AuthoritativeTextError(
                "plain_text is present but contains only whitespace",
                where=where,
            )
        if not plain:
            raise AuthoritativeTextError(
                "html_with_citations and plain_text are both empty",
                where=where,
                available_fields={k: v["nonempty"] for k, v in fields.items()},
            )
        if "\x00" in plain:
            raise AuthoritativeTextError("NUL byte in plain_text", where=where)
        normalized = normalize_lines(plain)
        if not normalized:
            raise AuthoritativeTextError(
                "nonempty plain_text normalized to empty text", where=where
            )
        blocks = []
        source_field = "plain_text_no_html_with_citations"
        raw = plain
    return ResolvedText(
        text=normalized,
        source_field=source_field,
        source_raw_sha256=sha256_text(raw),
        normalized_sha256=sha256_text(normalized),
        normalized_chars=len(normalized),
        caption_blocks=tuple(blocks),
        fields=fields,
    )


@dataclass(frozen=True)
class DirectEvidence:
    kind: str
    district: int
    evidence_sha256: str
    detail: str

    def record(self) -> dict:
        return asdict(self)


_ROD_PATHS = (
    re.compile(r"^/rod/docs/pdf/(?P<district>(?:[1-9]|1[0-2]))/.+\.pdf$", re.I),
    re.compile(r"^/rod/newpdf/(?P<district>(?:[1-9]|1[0-2]))/.+\.pdf$", re.I),
)
_ROD_HOSTS = frozenset({"www.supremecourt.ohio.gov", "www.sconet.state.oh.us"})


def official_rod_evidence(download_url: str | None) -> list[DirectEvidence]:
    if not download_url or not download_url.strip():
        return []
    value = download_url.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return []
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname not in _ROD_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        return []
    for pattern in _ROD_PATHS:
        match = pattern.fullmatch(parsed.path)
        if match:
            district = int(match.group("district"))
            return [
                DirectEvidence(
                    kind="official_rod_district_path",
                    district=district,
                    evidence_sha256=sha256_text(value),
                    detail=value,
                )
            ]
    return []


_PARAGRAPH_START = re.compile(
    r"^(?:\{?\s*¶\s*\d+\s*\}?|\[\s*(?:\*?P|¶)?\s*\d+\s*\])",
    re.I,
)
_DISTRICT_WORDS = {
    "FIRST": 1,
    "SECOND": 2,
    "THIRD": 3,
    "FOURTH": 4,
    "FIFTH": 5,
    "SIXTH": 6,
    "SEVENTH": 7,
    "EIGHTH": 8,
    "NINTH": 9,
    "TENTH": 10,
    "ELEVENTH": 11,
    "TWELFTH": 12,
}
_DISTRICT_RE = re.compile(
    r"\b("
    + "|".join(_DISTRICT_WORDS)
    + r"|(?:[1-9]|1[0-2])(?:ST|ND|RD|TH))\s+APPELLATE\s+DISTRICT\b",
    re.I,
)
_COURT_RE = re.compile(r"\bCOURT\s+OF\s+APPEALS(?:\s+OF\s+OHIO)?\b", re.I)
_CUYAHOGA_COUNTY_RE = re.compile(r"\bCOUNTY\s+OF\s+CUYAHOGA\b", re.I)


def _caption_region(blocks: tuple[str, ...] | list[str]) -> list[str]:
    region = []
    for block in blocks:
        normalized = _SPACE_RE.sub(" ", block).strip()
        if not normalized:
            continue
        if _PARAGRAPH_START.search(normalized):
            break
        region.append(normalized)
        if len(region) >= 80:
            break
    return region


def caption_evidence(blocks: tuple[str, ...] | list[str]) -> list[DirectEvidence]:
    region = _caption_region(blocks)
    evidence: list[DirectEvidence] = []
    seen: set[tuple[str, int, str]] = set()
    for index, block in enumerate(region):
        window = "\n".join(region[max(0, index - 2) : min(len(region), index + 3)])
        if not _COURT_RE.search(window):
            continue
        for match in _DISTRICT_RE.finditer(block):
            token = match.group(1).upper()
            district = _DISTRICT_WORDS.get(token)
            if district is None:
                district = int(re.match(r"\d+", token).group())
            detail = window
            key = ("html_caption_district", district, sha256_text(detail))
            if key not in seen:
                seen.add(key)
                evidence.append(
                    DirectEvidence(
                        kind=key[0],
                        district=district,
                        evidence_sha256=key[2],
                        detail=detail,
                    )
                )
    joined = "\n".join(region)
    if _COURT_RE.search(joined) and _CUYAHOGA_COUNTY_RE.search(joined):
        detail = joined
        key = ("html_caption_court_county", 8, sha256_text(detail))
        if key not in seen:
            evidence.append(
                DirectEvidence(
                    kind=key[0], district=8, evidence_sha256=key[2], detail=detail
                )
            )
    return evidence


def direct_evidence(
    download_url: str | None, resolved: ResolvedText
) -> list[DirectEvidence]:
    evidence = official_rod_evidence(download_url)
    if resolved.source_field in {
        "html_with_citations",
        "authoritative_pdf_supplement",
    }:
        evidence.extend(caption_evidence(resolved.caption_blocks))
    return evidence


def evidence_districts(evidence: list[DirectEvidence]) -> set[int]:
    return {item.district for item in evidence}


def classify(
    *,
    court_id: str,
    own_evidence: list[DirectEvidence],
    sibling_evidence: list[DirectEvidence],
    where: str,
) -> dict:
    own_districts = evidence_districts(own_evidence)
    sibling_districts = evidence_districts(sibling_evidence)
    if court_id in CAP_COURTS:
        asserted_districts = own_districts | sibling_districts
        if asserted_districts - {8}:
            raise EvidenceConflictError(
                "exact Cuyahoga court ID conflicts with direct issuing-district evidence",
                where=where,
                court_id=court_id,
                districts=sorted(asserted_districts | {8}),
                evidence=[item.record() for item in own_evidence + sibling_evidence],
            )
        return {
            "status": "accepted",
            "reason": "exact_cap_cuyahoga_court_allowlist",
            "district": 8,
            "evidence": [item.record() for item in own_evidence + sibling_evidence],
            "selector_version": SELECTOR_VERSION,
        }
    if court_id != GENERIC_OHIO_APPEALS_COURT:
        return {
            "status": "rejected",
            "reason": "court_not_in_cuyahoga_candidate_set",
            "district": None,
            "evidence": [],
            "selector_version": SELECTOR_VERSION,
        }

    if len(own_districts) > 1:
        raise EvidenceConflictError(
            "direct fields assert multiple issuing districts",
            where=where,
            districts=sorted(own_districts),
            evidence=[item.record() for item in own_evidence],
        )
    if len(sibling_districts) > 1:
        raise EvidenceConflictError(
            "siblings do not unanimously assert one issuing district",
            where=where,
            districts=sorted(sibling_districts),
            evidence=[item.record() for item in sibling_evidence],
        )
    if own_districts and sibling_districts and own_districts != sibling_districts:
        raise EvidenceConflictError(
            "direct and sibling issuing-district evidence disagree",
            where=where,
            own_districts=sorted(own_districts),
            sibling_districts=sorted(sibling_districts),
            evidence=[item.record() for item in own_evidence + sibling_evidence],
        )
    selected = own_evidence if own_evidence else sibling_evidence
    districts = evidence_districts(selected)
    if not districts:
        return {
            "status": "rejected",
            "reason": "unclassified_insufficient_direct_evidence",
            "district": None,
            "evidence": [],
            "selector_version": SELECTOR_VERSION,
        }
    district = next(iter(districts))
    if district != 8:
        return {
            "status": "rejected",
            "reason": "explicit_non_eighth_district",
            "district": district,
            "evidence": [item.record() for item in selected],
            "selector_version": SELECTOR_VERSION,
        }
    return {
        "status": "accepted",
        "reason": (
            "direct_eighth_district_evidence"
            if own_evidence
            else "unanimous_sibling_eighth_district_evidence"
        ),
        "district": 8,
        "evidence": [item.record() for item in selected],
        "selector_version": SELECTOR_VERSION,
    }


def load_corrections(path: str, *, archive_sha256: str, snapshot_date: str) -> dict:
    try:
        source_bytes = Path(path).read_bytes()
        manifest = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorrectionError(
            "cannot read correction manifest", path=path, error=str(error)
        )
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if manifest.get("format") != "calyx-cuyahoga-corrections-v2":
        raise CorrectionError("unsupported correction manifest format", path=path)
    binding = manifest.get("source_binding")
    expected = {
        "snapshot_date": snapshot_date,
        "opinions_archive_sha256": archive_sha256,
    }
    if not isinstance(binding, dict) or any(
        binding.get(k) != v for k, v in expected.items()
    ):
        raise CorrectionError(
            "correction manifest is not bound to this source archive",
            path=path,
            expected=expected,
            actual=binding,
        )
    corrections = manifest.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise CorrectionError("correction manifest contains no corrections", path=path)
    by_docket = {}
    correction_ids = set()
    for entry in corrections:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("docket_id"), int)
            or isinstance(entry.get("docket_id"), bool)
            or entry["docket_id"] <= 0
        ):
            raise CorrectionError("invalid correction entry", entry=entry)
        docket_id = entry["docket_id"]
        correction_id = entry.get("correction_id")
        if (
            not isinstance(correction_id, str)
            or not correction_id
            or correction_id in correction_ids
        ):
            raise CorrectionError(
                "correction_id must be a unique nonempty string",
                correction_id=correction_id,
            )
        correction_ids.add(correction_id)
        if docket_id in by_docket:
            raise CorrectionError("duplicate correction docket_id", docket_id=docket_id)
        expected_opinions = entry.get("expected_opinions")
        if (
            not isinstance(expected_opinions, list)
            or not expected_opinions
            or any(
                not isinstance(value, dict)
                or set(value) != {"opinion_id", "partition"}
                or not isinstance(value.get("opinion_id"), int)
                or isinstance(value.get("opinion_id"), bool)
                or value["opinion_id"] <= 0
                or value.get("partition") not in CORRECTION_PARTITIONS
                for value in expected_opinions
            )
            or len({value["opinion_id"] for value in expected_opinions})
            != len(expected_opinions)
        ):
            raise CorrectionError(
                "correction expected_opinions must bind unique positive IDs to partitions",
                docket_id=docket_id,
            )
        changes = entry.get("changes")
        if not isinstance(changes, dict) or set(changes) != {
            "case_name",
            "docket_number",
        }:
            raise CorrectionError(
                "correction must contain exactly the audited field pair",
                docket_id=docket_id,
                fields=sorted(changes) if isinstance(changes, dict) else None,
            )
        for field, change in changes.items():
            if not isinstance(change, dict):
                raise CorrectionError(
                    "correction field change is not an object",
                    docket_id=docket_id,
                    field=field,
                )
            for side in ("from", "to"):
                value = change.get(side)
                digest = change.get("%s_sha256" % side)
                if not isinstance(value, str) or digest != sha256_text(value):
                    raise CorrectionError(
                        "correction field digest does not match its declared value",
                        docket_id=docket_id,
                        field=field,
                        side=side,
                    )
        by_docket[docket_id] = entry
    physical_identities = {
        docket_id: (
            entry.get("cluster_id"),
            tuple(
                (value["opinion_id"], value["partition"])
                for value in entry["expected_opinions"]
            ),
        )
        for docket_id, entry in by_docket.items()
    }
    if physical_identities != EXPECTED_CORRECTION_IDENTITIES:
        raise CorrectionError(
            "correction manifest does not contain exactly the five audited identities",
            expected=EXPECTED_CORRECTION_IDENTITIES,
            actual=physical_identities,
        )
    return {
        "manifest": manifest,
        "by_docket": by_docket,
        "source_sha256": source_sha256,
    }


def apply_correction(
    record: dict, correction: dict | None, *, where: str
) -> dict | None:
    if correction is None:
        return None
    expected_cluster = correction.get("cluster_id")
    if expected_cluster is not None and record.get("cluster_id") != expected_cluster:
        raise CorrectionError(
            "correction cluster binding mismatch",
            where=where,
            expected=expected_cluster,
            actual=record.get("cluster_id"),
        )
    changes = correction.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise CorrectionError("correction has no field changes", where=where)
    original = {}
    corrected = {}
    for field, change in changes.items():
        if not isinstance(change, dict) or "from" not in change or "to" not in change:
            raise CorrectionError(
                "invalid correction field change", where=where, field=field
            )
        actual = record.get(field)
        source_hash = sha256_text(actual) if isinstance(actual, str) else None
        if actual != change["from"] or source_hash != change.get("from_sha256"):
            raise CorrectionError(
                "source value does not exactly match correction manifest",
                where=where,
                field=field,
                expected_value=change["from"],
                actual_value=actual,
                expected_sha256=change.get("from_sha256"),
                actual_sha256=source_hash,
            )
        original[field] = actual
        record[field] = change["to"]
        corrected[field] = change["to"]
    return {
        "correction_id": correction.get("correction_id"),
        "policy_version": CORRECTION_POLICY_VERSION,
        "original": original,
        "corrected": corrected,
        "evidence": correction.get("evidence"),
    }


def correction_manifest_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
