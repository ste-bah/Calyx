#!/usr/bin/env python3
"""Build and independently verify the Cuyahoga opinion-author generation.

The extract generation, people table, positions table, and bulk SHA-256
manifest are the only inputs.  No loose-file input or output is supported.
An upstream author foreign key is retained unless explicit author-name
evidence contradicts it.  Every fallback requires one unique real person;
ambiguous and incomplete source data remain typed UNRESOLVED results.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import traceback

from extract_cuyahoga import (
    MANIFEST_FILE as EXTRACT_MANIFEST,
    RAW_FILE,
    csv_rows,
    verify_extract_generation,
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


LEGACY_FORMAT = "calyx-cuyahoga-judges-generation-v2"
LEGACY_RESOLUTION_POLICY = "source-conflict-aware-initials-unique-v2"
FORMAT = "calyx-cuyahoga-judges-generation-v3"
RESOLUTION_POLICY = "signed-byline-tenure-aware-initials-unique-v3"
MANIFEST_FILE = "judge_manifest.json"
JUDGES_FILE = "judges_cuyahoga.jsonl"
MAPPING_FILE = "opinion_judges.csv"
COVERAGE_FILE = "judge_coverage.json"
RESOLVE_50_THRESHOLD = 50

CAP_CUYAHOGA_COURTS = frozenset(
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
MODERN_COURT = "ohioctapp"
POSITION_COURTS = CAP_CUYAHOGA_COURTS | {MODERN_COURT}
EIGHTH_DISTRICT_GROUP = frozenset({MODERN_COURT, "ohctapp8", "ohctapp8cuyahog"})

PEOPLE_FIELDS = (
    "id",
    "name_first",
    "name_middle",
    "name_last",
    "name_suffix",
    "is_alias_of_id",
)
POSITION_FIELDS = (
    "id",
    "court_id",
    "person_id",
    "date_start",
    "date_termination",
    "position_type",
    "job_title",
    "organization_name",
)
_HASH_LINE = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
_PER_CURIAM = re.compile(r"\bper\s+curiam\b", re.IGNORECASE)
_TITLE = re.compile(
    r"(?:,?\s*)\b(?:chief|presiding|administrative|visiting|retired|hon(?:orable)?|"
    r"judge|justice|p\.?\s*j\.?|a\.?\s*j\.?|c\.?\s*j\.?|j\.?\s*j\.?|j\.?)\b\.?,?",
    re.IGNORECASE,
)
_SEPARATOR = re.compile(r"\s*(?:;|\n|\r|\band\b|&)\s*", re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
_PAGE_ELEMENT = re.compile(
    r"<page_number\b[^>]*>.*?</page_number\s*>", re.IGNORECASE | re.DOTALL
)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_SIGNED_BYLINE = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z .'-]+?)\s*,\s*"
    r"(?:chief\s+|presiding\s+|administrative\s+|visiting\s+|retired\s+)?"
    r"(?:judge|justice|p\.?\s*j\.?|a\.?\s*j\.?|c\.?\s*j\.?|j\.?)\s*[.:]?\s*$",
    re.IGNORECASE,
)


class JudgeBuildError(StructuredError):
    """The judge generation could not be built or physically verified."""

    code = "cuyahoga_judge_generation_error"
    default_remediation = (
        "repair the named judge source or generation mismatch and rebuild at a new destination"
    )


def fail(message: str, **context) -> None:
    raise JudgeBuildError(message, **context)


def positive_int(value, *, field: str, where: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        fail("required ID is not decimal", field=field, value=value, where=where)
    if parsed <= 0:
        fail("required ID is not positive", field=field, value=value, where=where)
    return parsed


def optional_int(value, *, field: str, where: str) -> int | None:
    if value in (None, ""):
        return None
    return positive_int(value, field=field, where=where)


def normalized_word(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def court_group(court_id: str) -> str:
    return "eighth-district" if court_id in EIGHTH_DISTRICT_GROUP else court_id


def iso_date(value: str, *, field: str, where: str) -> str:
    if not value:
        return ""
    try:
        date.fromisoformat(value)
    except ValueError as error:
        fail(
            "date is not ISO YYYY-MM-DD",
            field=field,
            value=value,
            where=where,
            error=str(error),
        )
    return value


def date_compatible(position: dict, filed: str) -> bool:
    if not filed:
        return True
    return (not position["date_start"] or position["date_start"] <= filed) and (
        not position["date_termination"] or filed <= position["date_termination"]
    )


def load_bulk_manifest(path: str) -> tuple[Path, dict[str, str]]:
    manifest = Path(path).absolute()
    if not manifest.is_file() or manifest.is_symlink():
        fail("bulk SHA-256 manifest is not a plain file", path=str(manifest))
    entries: dict[str, str] = {}
    with open(manifest, "r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            match = _HASH_LINE.fullmatch(line)
            if not match:
                fail(
                    "invalid bulk SHA-256 manifest line",
                    path=str(manifest),
                    line=line_number,
                )
            digest, name = match.groups()
            if name in entries:
                fail(
                    "duplicate bulk SHA-256 manifest member",
                    path=str(manifest),
                    member=name,
                )
            entries[name] = digest
    if not entries:
        fail("bulk SHA-256 manifest is empty", path=str(manifest))
    return manifest, entries


def verify_sources(
    people_path: str, positions_path: str, bulk_manifest: str, snapshot: str
) -> dict:
    try:
        date.fromisoformat(snapshot)
    except ValueError as error:
        fail("source snapshot date is invalid", value=snapshot, error=str(error))
    manifest_path, entries = load_bulk_manifest(bulk_manifest)
    archives = {}
    for role, raw_path in (("people", people_path), ("positions", positions_path)):
        path = Path(raw_path).absolute()
        if not path.is_file() or path.is_symlink():
            fail("judge source archive is not a plain file", role=role, path=str(path))
        expected = entries.get(path.name)
        if expected is None:
            fail(
                "judge source archive is absent from bulk manifest",
                role=role,
                path=str(path),
            )
        sys.stderr.write("source-hash: verifying %s\n" % path)
        actual = sha256_file(path)
        if actual != expected:
            fail(
                "judge source archive SHA-256 mismatch",
                role=role,
                expected=expected,
                actual=actual,
            )
        archives[role] = {
            "path": str(path),
            "archive_name": path.name,
            "archive_sha256": actual,
            "snapshot_date": snapshot,
        }
    return {
        "archives": archives,
        "bulk_manifest": {
            "path": str(manifest_path),
            "name": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
    }


def load_people(path: str) -> tuple[dict[int, dict], dict[int, int]]:
    people: dict[int, dict] = {}
    for line, row in csv_rows(path, "people", PEOPLE_FIELDS):
        where = "%s physical line %d" % (path, line)
        person_id = positive_int(row["id"], field="id", where=where)
        if person_id in people:
            fail("duplicate person ID", person_id=person_id, where=where)
        person = {
            "person_id": person_id,
            "name_first": row["name_first"].strip(),
            "name_middle": row["name_middle"].strip(),
            "name_last": row["name_last"].strip(),
            "name_suffix": row["name_suffix"].strip(),
            "is_alias_of_id": optional_int(
                row["is_alias_of_id"], field="is_alias_of_id", where=where
            ),
        }
        if not person["name_last"]:
            fail("person has no surname", person_id=person_id, where=where)
        people[person_id] = person
    if not people:
        fail("people source contains zero rows", path=path)

    canonical: dict[int, int] = {}
    for person_id in people:
        seen = set()
        current = person_id
        while people[current]["is_alias_of_id"] is not None:
            if current in seen:
                fail(
                    "people alias cycle",
                    person_id=person_id,
                    cycle=sorted(seen | {current}),
                )
            seen.add(current)
            target = people[current]["is_alias_of_id"]
            if target not in people:
                fail("people alias target is missing", person_id=current, target=target)
            current = target
        canonical[person_id] = current
    return people, canonical


def load_positions(
    path: str, canonical: dict[int, int]
) -> tuple[dict[int, list[dict]], dict]:
    positions: dict[int, list[dict]] = defaultdict(list)
    total = 0
    relevant = 0
    invalid_date_ranges = []
    for line, row in csv_rows(path, "positions", POSITION_FIELDS):
        total += 1
        court_id = row["court_id"].strip()
        if court_id not in POSITION_COURTS:
            continue
        relevant += 1
        where = "%s physical line %d" % (path, line)
        position_id = positive_int(row["id"], field="id", where=where)
        source_person_id = positive_int(
            row["person_id"], field="person_id", where=where
        )
        if source_person_id not in canonical:
            fail(
                "position references missing person",
                position_id=position_id,
                person_id=source_person_id,
            )
        person_id = canonical[source_person_id]
        position = {
            "position_id": position_id,
            "court_id": court_id,
            "date_start": iso_date(
                row["date_start"].strip(), field="date_start", where=where
            ),
            "date_termination": iso_date(
                row["date_termination"].strip(), field="date_termination", where=where
            ),
            "position_type": row["position_type"].strip(),
            "job_title": row["job_title"].strip(),
            "organization_name": row["organization_name"].strip(),
            "source_person_id": source_person_id,
        }
        if (
            position["date_start"]
            and position["date_termination"]
            and position["date_start"] > position["date_termination"]
        ):
            # A reversed upstream tenure interval has no defensible temporal
            # meaning. Preserve the exact physical row identity and values in
            # coverage, but never admit it to matching or bench analytics.
            invalid_date_ranges.append(
                {
                    "position_id": position_id,
                    "person_id": person_id,
                    "source_person_id": source_person_id,
                    "court_id": court_id,
                    "date_start": position["date_start"],
                    "date_termination": position["date_termination"],
                    "reason": "date_start_after_date_termination",
                }
            )
            continue
        positions[person_id].append(position)
    if not positions:
        fail("positions source has no Cuyahoga/Ohio appellate positions", path=path)
    for values in positions.values():
        values.sort(
            key=lambda item: (item["court_id"], item["date_start"], item["position_id"])
        )
    return dict(positions), {
        "rows_total": total,
        "rows_relevant": relevant,
        "rows_usable": relevant - len(invalid_date_ranges),
        "invalid_date_ranges": invalid_date_ranges,
        "people_relevant": len(positions),
    }


def name_queries(raw: str) -> list[dict]:
    """Return conservative individual-name queries; multiple parts mean a panel."""
    if not raw or _PER_CURIAM.search(raw):
        return []
    # The bulk `judges` field contains real markup pollution. Page-number
    # element text is not a name; formatting tags preserve their contents. If
    # anything tag-shaped remains, refuse to turn malformed markup into a name.
    cleaned = _PAGE_ELEMENT.sub(" ", raw)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    if "<" in cleaned or ">" in cleaned:
        return []
    cleaned = _TITLE.sub(" ", cleaned)
    parts = []
    for major in _SEPARATOR.split(cleaned):
        # After titles are removed, a remaining comma separates people.  A
        # trailing comma from "Name, J." produces no extra part.
        parts.extend(piece for piece in major.split(",") if piece.strip())
    queries = []
    for part in parts:
        tokens = [normalized_word(token) for token in _TOKEN.findall(part)]
        tokens = [token for token in tokens if token and token not in _SUFFIXES]
        if not tokens:
            continue
        queries.append(
            {"raw": part.strip(), "surname": tokens[-1], "given": tokens[:-1]}
        )
    return queries


def person_matches_query(person: dict, query: dict) -> bool:
    if normalized_word(person["name_last"]) != query["surname"]:
        return False
    given = query["given"]
    if not given:
        return True
    actual = [
        normalized_word(token)
        for value in (person["name_first"], person["name_middle"])
        for token in _TOKEN.findall(value)
        if normalized_word(token)
    ]
    if len(given) > len(actual):
        return False
    for expected, value in zip(given, actual, strict=False):
        if len(expected) == 1 or len(value) == 1:
            if expected[0] != value[0]:
                return False
        elif expected != value:
            return False
    return True


def person_conflicts_with_query(person: dict, query: dict) -> bool:
    if normalized_word(person["name_last"]) != query["surname"]:
        return True
    return bool(query["given"]) and not person_matches_query(person, query)


def compatible_positions(
    person_id: int, court_id: str, filed: str, positions: dict
) -> list[dict]:
    group = court_group(court_id)
    return [
        position
        for position in positions.get(person_id, [])
        if court_group(position["court_id"]) == group
        and date_compatible(position, filed)
    ]


def bench_relation(person_id: int, court_id: str, filed: str, positions: dict) -> str:
    compatible = compatible_positions(person_id, court_id, filed, positions)
    if any(position["court_id"] in CAP_CUYAHOGA_COURTS for position in compatible):
        return "compatible_cuyahoga_position"
    if any(position["court_id"] == MODERN_COURT for position in compatible):
        return "compatible_combined_ohioctapp_position"
    if positions.get(person_id):
        return "relevant_position_outside_date_window"
    return "no_relevant_position"


def read_raw(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                fail(
                    "extract raw output contains a blank line",
                    path=str(path),
                    line=line_number,
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(
                    "extract raw output contains invalid JSON",
                    path=str(path),
                    line=line_number,
                    error=str(error),
                )
            yield line_number, row


def collect_source_author_ids(
    raw_path: Path, canonical: dict[int, int], people: dict
) -> set[int]:
    author_ids = set()
    seen = set()
    for line, row in read_raw(raw_path):
        opinion_id = positive_int(
            row.get("opinion_id"), field="opinion_id", where="raw line %d" % line
        )
        if opinion_id in seen:
            fail(
                "extract raw output contains duplicate opinion ID",
                opinion_id=opinion_id,
            )
        seen.add(opinion_id)
        source_id = optional_int(
            row.get("author_id"), field="author_id", where="raw line %d" % line
        )
        if source_id is not None and source_id in people:
            author_ids.add(canonical[source_id])
    if not seen:
        fail("extract generation contains zero accepted opinions", path=str(raw_path))
    return author_ids


def resolution_query(row: dict) -> tuple[dict | None, str | None, str | None]:
    author = str(row.get("author_str") or "").strip()
    judges = str(row.get("judges") or "").strip()
    author_queries = name_queries(author)
    if len(author_queries) == 1:
        return author_queries[0], "author_str", None
    if len(author_queries) > 1:
        return None, None, "UNRESOLVED_multiple_authors"
    judge_queries = name_queries(judges)
    if len(judge_queries) == 1:
        return judge_queries[0], "judges", None
    if len(judge_queries) > 1:
        return None, None, "UNRESOLVED_panel_no_author"
    return None, None, "UNRESOLVED_no_author_info"


def signed_leading_byline_query(row: dict) -> dict | None:
    """Return a specific name only when the opinion opens with a signed byline."""
    text = str(row.get("text") or "")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = _SIGNED_BYLINE.fullmatch(first)
    if match is None:
        return None
    queries = name_queries(match.group("name"))
    if len(queries) != 1 or not queries[0]["given"]:
        return None
    return queries[0]


def resolve_row(
    row: dict,
    people: dict,
    canonical: dict,
    positions: dict,
    eligible: set[int],
    resolution_policy: str = RESOLUTION_POLICY,
) -> tuple[int | None, str, str]:
    court_id = str(row.get("court_id") or "")
    if court_id not in POSITION_COURTS:
        fail(
            "accepted extract row has unexpected court",
            opinion_id=row.get("opinion_id"),
            court_id=court_id,
        )
    filed = iso_date(
        str(row.get("date_filed") or ""),
        field="date_filed",
        where="opinion %s" % row.get("opinion_id"),
    )
    if row.get("per_curiam") is True or _PER_CURIAM.search(
        str(row.get("author_str") or "")
    ):
        return None, "UNRESOLVED_per_curiam", "unresolved"

    query, query_source, query_error = resolution_query(row)
    source_id = optional_int(
        row.get("author_id"),
        field="author_id",
        where="opinion %s" % row.get("opinion_id"),
    )
    if source_id is not None:
        if source_id not in people:
            return None, "UNRESOLVED_author_id_missing_person", "unresolved"
        person_id = canonical[source_id]
        if query is not None and person_conflicts_with_query(people[person_id], query):
            return None, "UNRESOLVED_author_id_name_conflict", "unresolved"
        if resolution_policy == LEGACY_RESOLUTION_POLICY:
            return (
                person_id,
                "author_id",
                bench_relation(person_id, court_id, filed, positions),
            )
        if resolution_policy != RESOLUTION_POLICY:
            fail("unsupported judge resolution policy", policy=resolution_policy)
        signed_byline = signed_leading_byline_query(row)
        if signed_byline is not None and person_conflicts_with_query(
            people[person_id], signed_byline
        ):
            return None, "UNRESOLVED_author_id_name_conflict", "unresolved"
        relation = bench_relation(person_id, court_id, filed, positions)
        if relation in {
            "relevant_position_outside_date_window",
            "no_relevant_position",
        }:
            specific_queries = [
                candidate
                for candidate in (query, signed_byline)
                if candidate is not None and candidate["given"]
            ]
            if not any(
                person_matches_query(people[person_id], candidate)
                for candidate in specific_queries
            ):
                return (
                    None,
                    "UNRESOLVED_author_id_unverified_tenure",
                    "unresolved",
                )
        return (
            person_id,
            "author_id",
            relation,
        )

    if query_error is not None:
        return None, query_error, "unresolved"
    if query is None or query_source is None:
        return None, "UNRESOLVED_no_author_info", "unresolved"

    name_candidates = {
        person_id
        for person_id in eligible
        if person_matches_query(people[person_id], query)
    }
    if query["given"]:
        candidates = name_candidates
        method = "%s_%s_unique" % (
            query_source,
            "initials" if any(len(token) == 1 for token in query["given"]) else "name",
        )
    else:
        candidates = {
            person_id
            for person_id in name_candidates
            if compatible_positions(person_id, court_id, filed, positions)
        }
        method = "%s_surname_unique_position" % query_source
    if len(candidates) == 1:
        person_id = next(iter(candidates))
        return person_id, method, bench_relation(person_id, court_id, filed, positions)
    if len(candidates) > 1:
        return None, "UNRESOLVED_ambiguous_identity", "unresolved"
    return None, "UNRESOLVED_name_no_match", "unresolved"


def era_of(value: str) -> str:
    return (
        "%ds" % (int(value[:4]) // 10 * 10)
        if len(value) >= 4 and value[:4].isdigit()
        else "unknown"
    )


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def compute_artifacts(
    raw_path: Path,
    people_path: str,
    positions_path: str,
    resolution_policy: str = RESOLUTION_POLICY,
) -> tuple[dict[str, bytes], dict]:
    people, canonical = load_people(people_path)
    positions, position_counts = load_positions(positions_path, canonical)
    source_author_ids = collect_source_author_ids(raw_path, canonical, people)
    eligible = set(positions) | source_author_ids

    mapping_stream = io.StringIO(newline="")
    writer = csv.writer(mapping_stream, lineterminator="\n")
    writer.writerow(["opinion_id", "person_id", "method"])
    method_counts = Counter()
    relation_counts = Counter()
    era_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_judge = Counter()
    per_judge_relations: dict[int, Counter] = defaultdict(Counter)
    defects: dict[str, list[int]] = defaultdict(list)
    conflict_counts = Counter()
    conflict_samples: dict[int, list[int]] = defaultdict(list)
    opinion_ids = set()

    for line, row in read_raw(raw_path):
        opinion_id = positive_int(
            row.get("opinion_id"), field="opinion_id", where="raw line %d" % line
        )
        if opinion_id in opinion_ids:
            fail(
                "extract raw output contains duplicate opinion ID",
                opinion_id=opinion_id,
            )
        opinion_ids.add(opinion_id)
        person_id, method, relation = resolve_row(
            row,
            people,
            canonical,
            positions,
            eligible,
            resolution_policy,
        )
        writer.writerow(
            [opinion_id, person_id if person_id is not None else "UNRESOLVED", method]
        )
        method_counts[method] += 1
        relation_counts[relation] += 1
        era = era_of(str(row.get("date_filed") or ""))
        era_stats[era][1] += 1
        if person_id is not None:
            era_stats[era][0] += 1
            per_judge[person_id] += 1
            per_judge_relations[person_id][relation] += 1
        if method in {
            "UNRESOLVED_author_id_name_conflict",
            "UNRESOLVED_author_id_missing_person",
            "UNRESOLVED_author_id_unverified_tenure",
        } or relation in {
            "relevant_position_outside_date_window",
            "no_relevant_position",
        }:
            key = method if method.startswith("UNRESOLVED") else relation
            if len(defects[key]) < 100:
                defects[key].append(opinion_id)
        if method == "UNRESOLVED_author_id_name_conflict":
            source_author_id = positive_int(
                row.get("author_id"),
                field="author_id",
                where="opinion %d conflict" % opinion_id,
            )
            conflict_counts[source_author_id] += 1
            if len(conflict_samples[source_author_id]) < 20:
                conflict_samples[source_author_id].append(opinion_id)

    if not opinion_ids:
        fail("judge mapping produced zero opinions")

    judges_stream = io.StringIO(newline="")
    for person_id, count in sorted(per_judge.items()):
        person = people[person_id]
        alias_ids = sorted(
            source_id
            for source_id, target in canonical.items()
            if target == person_id and source_id != person_id
        )
        record = {
            "person_id": person_id,
            "alias_person_ids": alias_ids,
            "name_first": person["name_first"],
            "name_middle": person["name_middle"],
            "name_last": person["name_last"],
            "name_suffix": person["name_suffix"],
            "resolved_opinions": count,
            "bench_relation_counts": dict(
                sorted(per_judge_relations[person_id].items())
            ),
            "courts": positions.get(person_id, []),
        }
        judges_stream.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )

    resolved = sum(per_judge.values())
    judges_50 = [
        {
            "person_id": person_id,
            "name_first": people[person_id]["name_first"],
            "name_middle": people[person_id]["name_middle"],
            "name_last": people[person_id]["name_last"],
            "resolved_opinions": count,
            "bench_relation_counts": dict(
                sorted(per_judge_relations[person_id].items())
            ),
        }
        for person_id, count in sorted(
            per_judge.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= RESOLVE_50_THRESHOLD
    ]
    coverage = {
        "opinions_total": len(opinion_ids),
        "resolved_total": resolved,
        "unresolved_total": len(opinion_ids) - resolved,
        "resolved_fraction": round(resolved / len(opinion_ids), 9),
        "method_counts": dict(sorted(method_counts.items())),
        "bench_relation_counts": dict(sorted(relation_counts.items())),
        "coverage_by_era": {
            era: {
                "resolved": values[0],
                "total": values[1],
                "fraction": round(values[0] / values[1], 9),
            }
            for era, values in sorted(era_stats.items())
        },
        "source_people_rows": len(people),
        "source_position_counts": position_counts,
        "eligible_people": len(eligible),
        "judges_in_table": len(per_judge),
        "judges_with_50plus_resolved": judges_50,
        "per_judge_resolved_counts": {
            str(person_id): count
            for person_id, count in sorted(
                per_judge.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "source_quality_samples": {
            key: values for key, values in sorted(defects.items())
        },
        "author_id_name_conflicts": [
            {
                "source_author_id": source_author_id,
                "canonical_person_id": canonical[source_author_id],
                "name_first": people[canonical[source_author_id]]["name_first"],
                "name_middle": people[canonical[source_author_id]]["name_middle"],
                "name_last": people[canonical[source_author_id]]["name_last"],
                "count": count,
                "sample_opinion_ids": conflict_samples[source_author_id],
            }
            for source_author_id, count in sorted(
                conflict_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "opinion_id_set_sha256": hashlib.sha256(
            "".join("%d\n" % value for value in sorted(opinion_ids)).encode("ascii")
        ).hexdigest(),
    }
    artifacts = {
        MAPPING_FILE: mapping_stream.getvalue().encode("utf-8"),
        JUDGES_FILE: judges_stream.getvalue().encode("utf-8"),
        COVERAGE_FILE: json_bytes(coverage),
    }
    return artifacts, coverage


def source_extract(path: str) -> tuple[Path, dict, dict]:
    root = Path(path).absolute()
    report = verify_extract_generation(str(root))
    manifest = verify_generation(root, EXTRACT_MANIFEST)
    return root, report, manifest


def verify_judge_generation(path: str) -> dict:
    root = Path(path).absolute()
    manifest = verify_generation(root, MANIFEST_FILE)
    format_policy = (manifest.get("format"), manifest.get("resolution_policy"))
    if format_policy not in {
        (LEGACY_FORMAT, LEGACY_RESOLUTION_POLICY),
        (FORMAT, RESOLUTION_POLICY),
    }:
        fail(
            "unsupported judge generation contract",
            format=manifest.get("format"),
            policy=manifest.get("resolution_policy"),
        )
    if set(manifest["files"]) != {JUDGES_FILE, MAPPING_FILE, COVERAGE_FILE}:
        fail(
            "judge generation member contract mismatch",
            actual=sorted(manifest["files"]),
        )
    extract_source = manifest.get("source_extract_generation")
    if not isinstance(extract_source, dict) or not isinstance(
        extract_source.get("directory"), str
    ):
        fail("judge manifest has no physical extract generation binding")
    extract_root, extract_report, extract_manifest = source_extract(
        extract_source["directory"]
    )
    if extract_source != {
        "directory": str(extract_root),
        "manifest_sha256": extract_report["manifest_sha256"],
        "accepted_rows": extract_report["accepted_rows"],
        "format": extract_manifest["format"],
    }:
        fail("physical extract generation differs from judge manifest")
    sources = manifest.get("source_tables")
    if not isinstance(sources, dict):
        fail("judge manifest has no source table binding")
    actual_sources = verify_sources(
        sources.get("archives", {}).get("people", {}).get("path", ""),
        sources.get("archives", {}).get("positions", {}).get("path", ""),
        sources.get("bulk_manifest", {}).get("path", ""),
        extract_manifest["source_snapshot_date"],
    )
    if actual_sources != sources:
        fail("physical judge source tables differ from judge manifest")
    raw_path = generation_member(extract_root, EXTRACT_MANIFEST, RAW_FILE)
    artifacts, coverage = compute_artifacts(
        raw_path,
        actual_sources["archives"]["people"]["path"],
        actual_sources["archives"]["positions"]["path"],
        manifest["resolution_policy"],
    )
    for name, expected in artifacts.items():
        actual = (root / name).read_bytes()
        if actual != expected:
            fail(
                "judge member differs from independent source recomputation",
                member=name,
                expected_sha256=hashlib.sha256(expected).hexdigest(),
                actual_sha256=hashlib.sha256(actual).hexdigest(),
            )
    if manifest.get("coverage") != coverage:
        fail("judge manifest coverage differs from independent source recomputation")
    return {
        "generation": str(root),
        "manifest_sha256": sha256_file(root / MANIFEST_FILE),
        "opinions_total": coverage["opinions_total"],
        "resolved_total": coverage["resolved_total"],
        "unresolved_total": coverage["unresolved_total"],
        "judges_in_table": coverage["judges_in_table"],
        "status": "verified",
    }


def build(args) -> dict:
    extract_root, extract_report, extract_manifest = source_extract(
        args.extract_generation
    )
    sources = verify_sources(
        args.people,
        args.positions,
        args.bulk_manifest,
        extract_manifest["source_snapshot_date"],
    )
    raw_path = generation_member(extract_root, EXTRACT_MANIFEST, RAW_FILE)
    artifacts, coverage = compute_artifacts(raw_path, args.people, args.positions)
    with GenerationPublisher(args.out, MANIFEST_FILE) as generation:
        for name, value in artifacts.items():
            generation.write_bytes(name, value)
        generation.publish(
            {
                "format": FORMAT,
                "resolution_policy": RESOLUTION_POLICY,
                "source_extract_generation": {
                    "directory": str(extract_root),
                    "manifest_sha256": extract_report["manifest_sha256"],
                    "accepted_rows": extract_report["accepted_rows"],
                    "format": extract_manifest["format"],
                },
                "source_tables": sources,
                "coverage": coverage,
                "source_of_truth": MANIFEST_FILE,
            }
        )
    return verify_judge_generation(args.out)


def main() -> None:
    parser = StructuredArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser(
        "build", help="build one immutable judge generation"
    )
    build_command.add_argument("--extract-generation", required=True)
    build_command.add_argument("--people", required=True)
    build_command.add_argument("--positions", required=True)
    build_command.add_argument("--bulk-manifest", required=True)
    build_command.add_argument("--out", required=True)
    verify_command = commands.add_parser(
        "verify", help="independently recompute and verify a generation"
    )
    verify_command.add_argument("--generation", required=True)
    args = parse_cli_args(parser)
    try:
        report = (
            build(args)
            if args.command == "build"
            else verify_judge_generation(args.generation)
        )
        print(json.dumps(report, sort_keys=True))
    except BaseException as error:
        write_error(
            error,
            code="cuyahoga_judge_generation_error",
            remediation=(
                "repair the named judge source or generation mismatch and rebuild at a new destination"
            ),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
