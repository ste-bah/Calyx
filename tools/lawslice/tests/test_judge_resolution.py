from __future__ import annotations

from pathlib import Path
import sys
import unittest


LAW_DIR = Path(__file__).resolve().parents[1]
if str(LAW_DIR) not in sys.path:
    sys.path.insert(0, str(LAW_DIR))

from build_judge_tables import name_queries, resolve_row  # noqa: E402


# Exact identities and names from the physically hashed 2026-06-30
# people-db archive (SHA-256 95ad6df10205bda27cde649d285c381c93dadef9e16a47d
# 8076ffe46b569778a).  These are captured real regressions, not invented judges.
PEOPLE = {
    8091: {
        "person_id": 8091,
        "name_first": "Eileen",
        "name_middle": "A.",
        "name_last": "Gallagher",
        "name_suffix": "",
        "is_alias_of_id": None,
    },
    8092: {
        "person_id": 8092,
        "name_first": "Sean",
        "name_middle": "C.",
        "name_last": "Gallagher",
        "name_suffix": "",
        "is_alias_of_id": None,
    },
    8117: {
        "person_id": 8117,
        "name_first": "Christine",
        "name_middle": "T.",
        "name_last": "McMonagle",
        "name_suffix": "",
        "is_alias_of_id": None,
    },
    8115: {
        "person_id": 8115,
        "name_first": "Matthew",
        "name_middle": "Walden",
        "name_last": "McFarland",
        "name_suffix": "",
        "is_alias_of_id": None,
    },
    8055: {
        "person_id": 8055,
        "name_first": "Mary",
        "name_middle": "J.",
        "name_last": "Boyle",
        "name_suffix": "",
        "is_alias_of_id": None,
    },
}
CANONICAL = {person_id: person_id for person_id in PEOPLE}
POSITIONS = {
    8091: [
        {
            "position_id": 33050,
            "court_id": "ohioctapp",
            "date_start": "2011-01-01",
            "date_termination": "",
        }
    ],
    8092: [
        {
            "position_id": 33051,
            "court_id": "ohioctapp",
            "date_start": "2003-01-01",
            "date_termination": "",
        }
    ],
    8117: [
        {
            "position_id": 33084,
            "court_id": "ohioctapp",
            "date_start": "2005-01-03",
            "date_termination": "2011-01-02",
        }
    ],
}
ELIGIBLE = set(PEOPLE)


def opinion(**changes):
    row = {
        "opinion_id": 3742922,
        "court_id": "ohioctapp",
        "date_filed": "2002-01-10",
        "author_id": None,
        "author_str": "",
        "judges": "",
        "per_curiam": False,
    }
    row.update(changes)
    return row


class RealJudgeResolutionRegressions(unittest.TestCase):
    def resolve(self, row):
        return resolve_row(row, PEOPLE, CANONICAL, POSITIONS, ELIGIBLE)

    def test_real_mcmonagle_fk_conflict_is_quarantined(self):
        result = self.resolve(
            opinion(author_id=8117, author_str="TIMOTHY E. McMONAGLE, P.J.")
        )
        self.assertEqual(
            result,
            (None, "UNRESOLVED_author_id_name_conflict", "unresolved"),
        )

    def test_same_real_fk_with_matching_signed_name_resolves(self):
        person_id, method, relation = self.resolve(
            opinion(
                author_id=8117,
                author_str="CHRISTINE T. McMONAGLE, J.",
                date_filed="2008-01-10",
            )
        )
        self.assertEqual((person_id, method), (8117, "author_id"))
        self.assertEqual(relation, "compatible_combined_ohioctapp_position")

    def test_real_gallagher_initials_resolve_only_exact_unique_people(self):
        self.assertEqual(
            self.resolve(opinion(opinion_id=9901937, author_str="E.A. Gallagher"))[:2],
            (8091, "author_str_initials_unique"),
        )
        self.assertEqual(
            self.resolve(opinion(author_str="S. Gallagher"))[:2],
            (8092, "author_str_initials_unique"),
        )
        self.assertEqual(
            self.resolve(opinion(author_str="E.T. Gallagher"))[:2],
            (None, "UNRESOLVED_name_no_match"),
        )

    def test_real_multi_judge_string_never_becomes_an_author_guess(self):
        result = self.resolve(
            opinion(judges="Stevens, Doyle, Hunsicker, Ninth, Eighth")
        )
        self.assertEqual(result, (None, "UNRESOLVED_panel_no_author", "unresolved"))

    def test_per_curiam_overrides_even_a_present_fk(self):
        result = self.resolve(opinion(author_id=8092, per_curiam=True))
        self.assertEqual(result, (None, "UNRESOLVED_per_curiam", "unresolved"))

    def test_real_outside_roster_fk_is_retained_but_not_called_bench(self):
        result = self.resolve(
            opinion(
                opinion_id=4195391,
                date_filed="2017-01-01",
                author_id=8115,
                author_str="McFarland",
            )
        )
        self.assertEqual(result, (8115, "author_id", "no_relevant_position"))

    def test_title_parser_does_not_turn_one_real_author_into_a_panel(self):
        self.assertEqual(
            name_queries("COLLEEN CONWAY COONEY, J."),
            [
                {
                    "raw": "COLLEEN CONWAY COONEY",
                    "surname": "cooney",
                    "given": ["colleen", "conway"],
                }
            ],
        )

    def test_real_page_number_and_bold_markup_cannot_pollute_the_name(self):
        self.assertEqual(
            name_queries("SEAN C. GALLAGHER, P.J.:<page_number>Page 3</page_number>"),
            [
                {
                    "raw": "SEAN C. GALLAGHER :",
                    "surname": "gallagher",
                    "given": ["sean", "c"],
                }
            ],
        )

    def test_real_full_middle_name_matches_source_middle_initial(self):
        self.assertEqual(
            self.resolve(
                opinion(
                    author_id=8055,
                    judges="MARY JANE BOYLE, JUDGE.",
                    date_filed="2010-01-01",
                )
            )[:2],
            (8055, "author_id"),
        )
        self.assertEqual(
            name_queries("<bold>SEAN C. GALLAGHER, Judge</bold>."),
            [
                {
                    "raw": "SEAN C. GALLAGHER  .",
                    "surname": "gallagher",
                    "given": ["sean", "c"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
