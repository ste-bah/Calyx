from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


LAW_DIR = Path(__file__).resolve().parents[1]
if str(LAW_DIR) not in sys.path:
    sys.path.insert(0, str(LAW_DIR))

from build_ingest_jsonl import canonical_preference  # noqa: E402
from extract_cuyahoga import (  # noqa: E402
    ExtractionError,
    authoritative_error_identity,
    flush_and_sync,
    hashed_spool_lines,
    resolved_text_from_spool,
    resolved_text_to_spool,
)
from cuyahoga_contract import (  # noqa: E402
    AuthoritativeTextError,
    CAP_COURTS,
    EvidenceConflictError,
    TEXT_FIELDS,
    apply_correction,
    classify,
    direct_evidence,
    load_corrections,
    official_rod_evidence,
    resolve_text,
)
from law_generation import (  # noqa: E402
    GenerationError,
    GenerationPublisher,
    verify_generation,
)


OPINIONS_SHA256 = "65db4902a4ae42c48ef9958f52600dedca5401c194d3611d0b2672cfc6dd4d9c"
CORRECTIONS = LAW_DIR / "cuyahoga_corrections.v1.json"
REGRESSION_DIR = LAW_DIR / "real_regressions"
REGRESSION_MANIFEST = "regression_manifest.json"
REGRESSION_CASES = "opinion_regressions.json"


def load_real_cases() -> dict[int, dict]:
    manifest = verify_generation(REGRESSION_DIR, REGRESSION_MANIFEST)
    if manifest["source_archive"]["sha256"] != OPINIONS_SHA256:
        raise AssertionError("real regression generation has the wrong source archive")
    value = json.loads((REGRESSION_DIR / REGRESSION_CASES).read_text(encoding="utf-8"))
    if value["format"] != "calyx-courtlistener-real-regressions-v1":
        raise AssertionError("real regression format changed")
    cases = {row["opinion_id"]: row for row in value["cases"]}
    if set(cases) != set(manifest["opinion_ids"]):
        raise AssertionError("real regression IDs differ from their physical manifest")
    return cases


REAL_CASES = load_real_cases()


class CandidateSpoolDurabilityTests(unittest.TestCase):
    def test_real_regression_row_is_readable_after_flush_and_fsync(self):
        payload = json.dumps(REAL_CASES[4636687], sort_keys=True) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate-spool.jsonl"
            with open(path, "x", encoding="utf-8", newline="\n") as output:
                output.write(payload)
                flush_and_sync(output)
                with open(path, "r", encoding="utf-8") as physical:
                    self.assertEqual(physical.read(), payload)

    def test_largest_captured_real_text_roundtrips_through_strict_resolved_spool(self):
        resolved = max(
            (
                resolve_case(case)[0]
                for case in REAL_CASES.values()
                if case["authoritative_text_error"] is None
            ),
            key=lambda item: len(item.text),
        )
        self.assertGreater(len(resolved.text), 10_000)
        payload = json.loads(json.dumps(resolved_text_to_spool(resolved)))
        readback = resolved_text_from_spool(
            payload, path="real-regression-spool", spool_line=1
        )
        self.assertEqual(readback, resolved)

    def test_empty_resolution_roundtrips_without_synthesizing_text(self):
        self.assertIsNone(resolved_text_to_spool(None))
        self.assertIsNone(
            resolved_text_from_spool(
                None, path="real-empty-row-spool", spool_line=1
            )
        )

    def test_mutated_real_resolution_hash_is_rejected(self):
        resolved, _ = resolve_case(REAL_CASES[4636687])
        payload = resolved_text_to_spool(resolved)
        payload["normalized_sha256"] = "0" * 64
        with self.assertRaises(ExtractionError):
            resolved_text_from_spool(
                payload, path="mutated-real-resolution", spool_line=1
            )

    def test_physical_replay_hashes_exact_real_regression_bytes(self):
        real_row = REAL_CASES[4636687]
        payload = (json.dumps(real_row, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate-spool.jsonl"
            with open(path, "xb") as output:
                output.write(payload)
                flush_and_sync(output)
            expected_sha256 = hashlib.sha256(payload).hexdigest()
            rows = list(
                hashed_spool_lines(
                    str(path),
                    expected_sha256=expected_sha256,
                    expected_bytes=len(payload),
                )
            )
            self.assertEqual(rows, [(1, payload.decode("utf-8"))])
            with self.assertRaises(ExtractionError):
                list(
                    hashed_spool_lines(
                        str(path),
                        expected_sha256="0" * 64,
                        expected_bytes=len(payload),
                    )
                )
            physical = path.read_bytes()
            self.assertEqual(physical, payload)
            self.assertEqual(
                hashlib.sha256(physical).hexdigest(),
                expected_sha256,
            )


def excerpt_fields(case: dict) -> dict:
    fields = {name: "" for name in TEXT_FIELDS}
    fields.update(case["authoritative_excerpt"])
    return fields


def resolve_case(case: dict):
    resolved = resolve_text(
        excerpt_fields(case), where="CourtListener opinion %d" % case["opinion_id"]
    )
    evidence = direct_evidence(case["courtlistener_source"]["download_url"], resolved)
    return resolved, evidence


class RealRegressionGenerationTests(unittest.TestCase):
    def test_every_real_excerpt_reproduces_recorded_text_evidence_and_decision(self):
        for opinion_id, case in sorted(REAL_CASES.items()):
            with self.subTest(opinion_id=opinion_id):
                if case["authoritative_text_error"] is not None:
                    with self.assertRaises(AuthoritativeTextError):
                        resolve_text(
                            excerpt_fields(case),
                            where="CourtListener opinion %d" % opinion_id,
                        )
                    continue
                resolved, evidence = resolve_case(case)
                self.assertEqual(resolved.source_field, case["excerpt_source_field"])
                self.assertEqual(resolved.source_raw_sha256, case["excerpt_sha256"])
                self.assertEqual(
                    [item.record() for item in evidence],
                    case["expected_evidence"],
                )
                decision = classify(
                    court_id=case["court_id"],
                    own_evidence=evidence,
                    sibling_evidence=[],
                    where="CourtListener opinion %d" % opinion_id,
                )
                self.assertEqual(decision, case["expected_decision"])

    def test_real_tenth_district_record_is_explicit_non_eighth(self):
        case = REAL_CASES[4678958]
        _, evidence = resolve_case(case)
        self.assertEqual({item.district for item in evidence}, {10})
        self.assertEqual(
            case["expected_decision"]["reason"], "explicit_non_eighth_district"
        )

    def test_real_body_only_cuyahoga_mentions_do_not_become_caption_evidence(self):
        # These same-cluster opinions were selected by the former first-2,000
        # character regex despite having no direct issuing-court caption/URL.
        for opinion_id in (3713167, 3713168):
            case = REAL_CASES[opinion_id]
            _, evidence = resolve_case(case)
            self.assertEqual(evidence, [])
            self.assertEqual(
                case["expected_decision"]["reason"],
                "unclassified_insufficient_direct_evidence",
            )

    def test_real_preformatted_caption_stops_before_body_district_references(self):
        # Cirino's body mentions Cuyahoga, but its structural caption and URL
        # both say Tenth District. The captured 12k-character excerpt includes
        # body content beyond the first numbered paragraph.
        case = REAL_CASES[4678958]
        resolved, evidence = resolve_case(case)
        self.assertGreater(len(resolved.text), 10_000)
        self.assertEqual({item.district for item in evidence}, {10})

    def test_real_eighth_and_tenth_evidence_conflict_is_typed(self):
        _, eighth = resolve_case(REAL_CASES[4636687])
        _, tenth = resolve_case(REAL_CASES[4678958])
        with self.assertRaises(EvidenceConflictError) as raised:
            classify(
                court_id="ohioctapp",
                own_evidence=eighth + tenth,
                sibling_evidence=[],
                where="real cross-source conflict boundary",
            )
        self.assertEqual(raised.exception.code, "issuing_court_evidence_conflict")
        self.assertEqual(raised.exception.context["districts"], [8, 10])

    def test_real_eighth_evidence_can_classify_an_evidence_less_sibling(self):
        _, eighth = resolve_case(REAL_CASES[4636687])
        accepted = classify(
            court_id="ohioctapp",
            own_evidence=[],
            sibling_evidence=eighth,
            where="unanimous sibling boundary using captured evidence",
        )
        self.assertEqual(
            accepted["reason"], "unanimous_sibling_eighth_district_evidence"
        )
        rejected = classify(
            court_id="ohioctapp",
            own_evidence=[],
            sibling_evidence=[],
            where="captured evidence-less boundary",
        )
        self.assertEqual(
            rejected["reason"], "unclassified_insufficient_direct_evidence"
        )

    def test_exact_cap_allowlist_accepts_real_cap_row_without_widening(self):
        cap_case = REAL_CASES[8621357]
        _, evidence = resolve_case(cap_case)
        self.assertEqual(evidence, [])
        self.assertEqual(
            cap_case["expected_decision"]["reason"],
            "exact_cap_cuyahoga_court_allowlist",
        )
        self.assertIn(cap_case["court_id"], CAP_COURTS)
        near_match = classify(
            court_id=cap_case["court_id"] + "a",
            own_evidence=[],
            sibling_evidence=[],
            where="invalid court-id boundary derived from captured CAP row",
        )
        self.assertEqual(near_match["status"], "rejected")

    def test_exact_cap_identity_conflicting_with_real_tenth_evidence_aborts(self):
        cap_case = REAL_CASES[8621357]
        _, tenth = resolve_case(REAL_CASES[4678958])
        with self.assertRaises(EvidenceConflictError) as raised:
            classify(
                court_id=cap_case["court_id"],
                own_evidence=tenth,
                sibling_evidence=[],
                where="real CAP identity / Tenth evidence conflict",
            )
        self.assertEqual(raised.exception.context["districts"], [8, 10])

    def test_mutated_real_rod_urls_fail_strict_host_path_and_query_contract(self):
        real_url = REAL_CASES[4636687]["courtlistener_source"]["download_url"]
        mutations = (
            real_url.replace("www.supremecourt.ohio.gov", "example.com"),
            real_url.replace("/rod/docs/pdf/", "/other/pdf/"),
            real_url.replace("/pdf/8/", "/pdf/0/"),
            real_url.replace(
                "www.supremecourt.ohio.gov", "www.supremecourt.ohio.gov:443"
            ),
            real_url + "?changed=1",
        )
        for value in mutations:
            with self.subTest(value=value):
                self.assertEqual(official_rod_evidence(value), [])


class AuthoritativeTextContractTests(unittest.TestCase):
    def test_real_empty_rows_replay_the_same_causal_error_across_locations(self):
        for opinion_id in (11227531, 11257339):
            fields = excerpt_fields(REAL_CASES[opinion_id])
            records = []
            for where in (
                "bound CourtListener archive evidence scan",
                "sealed candidate spool replay",
            ):
                with self.assertRaises(AuthoritativeTextError) as raised:
                    resolve_text(fields, where=where)
                records.append(raised.exception.record())
            self.assertNotEqual(records[0], records[1])
            self.assertEqual(
                authoritative_error_identity(records[0]),
                authoritative_error_identity(records[1]),
            )

    def test_causal_error_mutation_is_not_hidden_by_location_normalization(self):
        fields = excerpt_fields(REAL_CASES[11227531])
        with self.assertRaises(AuthoritativeTextError) as empty:
            resolve_text(fields, where="archive")
        fields["plain_text"] = " "
        with self.assertRaises(AuthoritativeTextError) as whitespace:
            resolve_text(fields, where="spool")
        self.assertNotEqual(
            authoritative_error_identity(empty.exception.record()),
            authoritative_error_identity(whitespace.exception.record()),
        )

    def test_real_duplicate_prefers_html_over_available_plain(self):
        case = REAL_CASES[11173418]
        self.assertTrue(case["text_fields"]["html_with_citations"]["nonempty"])
        self.assertTrue(case["text_fields"]["plain_text"]["nonempty"])
        resolved, _ = resolve_case(case)
        self.assertEqual(resolved.source_field, "html_with_citations")

    def test_real_plain_only_row_is_explicitly_labeled(self):
        case = REAL_CASES[11171895]
        self.assertFalse(case["text_fields"]["html_with_citations"]["nonempty"])
        self.assertTrue(case["text_fields"]["plain_text"]["nonempty"])
        resolved, _ = resolve_case(case)
        self.assertEqual(resolved.source_field, "plain_text_no_html_with_citations")

    def test_both_real_empty_rows_are_hard_errors(self):
        for opinion_id in (11227531, 11257339):
            case = REAL_CASES[opinion_id]
            self.assertFalse(case["text_fields"]["html_with_citations"]["nonempty"])
            self.assertFalse(case["text_fields"]["plain_text"]["nonempty"])
            with self.assertRaises(AuthoritativeTextError):
                resolve_text(
                    excerpt_fields(case),
                    where="CourtListener empty opinion %d" % opinion_id,
                )

    def test_malformed_mutation_of_real_preferred_html_never_uses_plain(self):
        case = REAL_CASES[11173418]
        fields = excerpt_fields(case)
        self.assertTrue(fields["plain_text"])
        fields["html_with_citations"] += "<script>unterminated"
        with self.assertRaises(AuthoritativeTextError):
            resolve_text(fields, where="malformed mutation of opinion 11173418")

    def test_real_maximum_length_row_provenance_is_preserved(self):
        case = REAL_CASES[6227032]
        self.assertEqual(case["full_normalized_chars"], 201_515)
        self.assertEqual(len(case["full_normalized_sha256"]), 64)
        resolved, evidence = resolve_case(case)
        self.assertEqual(resolved.source_field, "html_with_citations")
        self.assertEqual({item.district for item in evidence}, {8})


class AliasCanonicalizationTests(unittest.TestCase):
    def test_real_duplicate_chooses_preferred_html_source_and_preserves_both_ids(self):
        plain = REAL_CASES[11171895]
        html = REAL_CASES[11173418]
        self.assertEqual(
            plain["full_normalized_sha256"], html["full_normalized_sha256"]
        )
        candidates = {
            plain["opinion_id"]: canonical_preference(
                {"text_source": plain["full_source_field"]}, plain["opinion_id"]
            ),
            html["opinion_id"]: canonical_preference(
                {"text_source": html["full_source_field"]}, html["opinion_id"]
            ),
        }
        canonical_id = min(candidates, key=candidates.get)
        self.assertEqual(canonical_id, 11173418)
        self.assertEqual(set(candidates), {11171895, 11173418})


class CorrectionContractTests(unittest.TestCase):
    def setUp(self):
        self.loaded = load_corrections(
            str(CORRECTIONS),
            archive_sha256=OPINIONS_SHA256,
            snapshot_date="2026-06-30",
        )

    def test_all_four_physically_observed_reversals_apply_exactly(self):
        expected = {
            2593342: ("100980", "State v. Scott"),
            2607299: ("100885", "State v. Bullitt"),
            69498717: ("113703", "Pepper Pike v. R.E.S."),
            70827619: ("114476", "Roll v. Gertburg Licata Co., LPA"),
        }
        for docket_id, (docket_number, case_name) in expected.items():
            correction = self.loaded["by_docket"][docket_id]
            record = {
                "cluster_id": correction["cluster_id"],
                "case_name": docket_number,
                "docket_number": case_name,
            }
            audit = apply_correction(
                record, correction, where="CourtListener docket %d" % docket_id
            )
            self.assertEqual(record["case_name"], case_name)
            self.assertEqual(record["docket_number"], docket_number)
            self.assertEqual(audit["original"]["case_name"], docket_number)
            self.assertEqual(audit["corrected"]["case_name"], case_name)

    def test_correction_source_mismatch_aborts(self):
        correction = self.loaded["by_docket"][2593342]
        record = {
            "cluster_id": correction["cluster_id"],
            "case_name": "State v. Scott",
            "docket_number": "100980",
        }
        with self.assertRaises(Exception):
            apply_correction(record, correction, where="already-mutated source")

    def test_wrong_archive_binding_aborts(self):
        with self.assertRaises(Exception):
            load_corrections(
                str(CORRECTIONS),
                archive_sha256="0" * 64,
                snapshot_date="2026-06-30",
            )


class GenerationPublicationTests(unittest.TestCase):
    def test_publish_readback_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation"
            correction_value = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
            with GenerationPublisher(destination, "manifest.json") as publisher:
                publisher.write_json("corrections.json", correction_value)
                publisher.publish(
                    {
                        "format": "real-correction-publication-test-v1",
                        "source_of_truth": "manifest.json",
                    }
                )
            manifest = verify_generation(destination, "manifest.json")
            self.assertEqual(manifest["format"], "real-correction-publication-test-v1")
            (destination / "corrections.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(GenerationError):
                verify_generation(destination, "manifest.json")

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "generation"
            destination.mkdir()
            marker = destination / "physical-state.txt"
            marker.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(GenerationError):
                GenerationPublisher(destination, "manifest.json")
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_failed_build_leaves_no_visible_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generation"
            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                with GenerationPublisher(destination, "manifest.json") as publisher:
                    publisher.write_json(
                        "corrections.json",
                        json.loads(CORRECTIONS.read_text(encoding="utf-8")),
                    )
                    raise RuntimeError("intentional failure")
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".generation.staging.*")), [])


if __name__ == "__main__":
    unittest.main()
