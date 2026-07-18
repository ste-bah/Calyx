use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MetadataMatch {
    pub(crate) case_name: bool,
    pub(crate) docket: bool,
}

impl MetadataMatch {
    pub(crate) fn any(self) -> bool {
        self.case_name || self.docket
    }

    pub(crate) fn specificity(self) -> u8 {
        match (self.case_name, self.docket) {
            (true, true) => 3,
            (false, true) => 2,
            (true, false) => 1,
            (false, false) => 0,
        }
    }
}

pub(crate) fn metadata_match(query: &str, attributes: &BTreeMap<String, String>) -> MetadataMatch {
    let query_terms = normalized_terms(query).into_iter().collect::<BTreeSet<_>>();
    let case_name = attributes
        .get("case_name")
        .map(|value| {
            let terms = normalized_terms(value)
                .into_iter()
                .filter(|term| informative_case_term(term))
                .collect::<BTreeSet<_>>();
            terms.len() >= 2 && terms.is_subset(&query_terms)
        })
        .unwrap_or(false);
    let docket = attributes
        .get("docket_number")
        .map(|value| {
            let terms = normalized_terms(value)
                .into_iter()
                .filter(|term| term.len() >= 4 && term.bytes().any(|byte| byte.is_ascii_digit()))
                .collect::<BTreeSet<_>>();
            !terms.is_empty() && terms.is_subset(&query_terms)
        })
        .unwrap_or(false);
    MetadataMatch { case_name, docket }
}

fn normalized_terms(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn informative_case_term(term: &str) -> bool {
    term.len() >= 2 && !matches!(term, "in" | "re" | "no" | "nos" | "vs")
}
