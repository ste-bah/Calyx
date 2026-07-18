use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::kernel_generation::KernelJurisdictionContract;
use crate::error::{CliError, CliResult};

const LEGAL_DATASET_PREFIX: &str = "courtlistener-";
const FIELDS: [(&str, &str); 5] = [
    ("jurisdiction_country", "country"),
    ("jurisdiction_court_system", "court_system"),
    ("jurisdiction_state", "state"),
    ("jurisdiction_county", "county"),
    ("jurisdiction_appellate_district", "appellate_district"),
];
const US_STATES: [&str; 51] = [
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "district of columbia",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
];
const FEDERAL_SIGNALS: [&str; 6] = [
    "federal",
    "federal circuit",
    "united states court",
    "united states supreme court",
    "u.s. supreme court",
    "supreme court of the united states",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScopeConflict {
    pub kind: &'static str,
    pub detected: String,
}

pub(crate) fn derive_jurisdiction(
    rows: &[BTreeMap<String, String>],
) -> CliResult<Option<KernelJurisdictionContract>> {
    let legal_rows = rows
        .iter()
        .filter(|row| {
            row.get("source_dataset")
                .is_some_and(|value| value.starts_with(LEGAL_DATASET_PREFIX))
                || row.get("court_id").is_some_and(|value| !value.is_empty())
                || row
                    .get("source_url")
                    .is_some_and(|value| value.contains("courtlistener.com/"))
        })
        .count();
    if legal_rows == 0 {
        return Ok(None);
    }
    if legal_rows != rows.len() {
        return Err(CliError::runtime(format!(
            "kernel jurisdiction contract found mixed legal/non-legal rows: legal={legal_rows} total={}",
            rows.len()
        )));
    }
    if rows.iter().any(|row| {
        !row.get("source_dataset")
            .is_some_and(|value| value.starts_with(LEGAL_DATASET_PREFIX))
    }) {
        return Err(CliError::runtime(
            "kernel jurisdiction contract requires a CourtListener source_dataset on every legal row",
        ));
    }
    let mut values = BTreeMap::new();
    for (metadata_key, contract_key) in FIELDS {
        let distinct = rows
            .iter()
            .map(|row| row.get(metadata_key).map(String::as_str).unwrap_or(""))
            .collect::<BTreeSet<_>>();
        if distinct.len() != 1 || distinct.contains("") {
            return Err(CliError::runtime(format!(
                "kernel jurisdiction metadata {metadata_key} must be present and identical on every legal row; observed={distinct:?}"
            )));
        }
        values.insert(
            contract_key,
            distinct.into_iter().next().unwrap().to_string(),
        );
    }
    let mut digest = Sha256::new();
    digest.update(b"calyx-kernel-jurisdiction-v1");
    for (key, value) in &values {
        digest.update(key.as_bytes());
        digest.update([0]);
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    digest.update(rows.len().to_le_bytes());
    Ok(Some(KernelJurisdictionContract {
        schema_version: 1,
        country: values.remove("country").unwrap(),
        court_system: values.remove("court_system").unwrap(),
        state: values.remove("state").unwrap(),
        county: values.remove("county").unwrap(),
        appellate_district: values.remove("appellate_district").unwrap(),
        source_rows: rows.len(),
        metadata_contract_sha256: super::kernel_generation::hex32(&digest.finalize().into()),
    }))
}

pub(crate) fn explicit_scope_conflict(
    query: &str,
    contract: &KernelJurisdictionContract,
) -> Option<ScopeConflict> {
    let normalized = query.to_ascii_lowercase();
    let expected_state = contract.state.to_ascii_lowercase();
    if let Some(state) = US_STATES
        .iter()
        .find(|state| phrase_present(&normalized, state) && **state != expected_state)
    {
        return Some(ScopeConflict {
            kind: "state",
            detected: title_case(state),
        });
    }
    if !contract.court_system.eq_ignore_ascii_case("federal")
        && let Some(signal) = FEDERAL_SIGNALS
            .iter()
            .find(|signal| phrase_present(&normalized, signal))
    {
        return Some(ScopeConflict {
            kind: "court_system",
            detected: signal.to_string(),
        });
    }
    None
}

fn phrase_present(text: &str, phrase: &str) -> bool {
    text.match_indices(phrase).any(|(start, found)| {
        let end = start + found.len();
        let left = text[..start]
            .chars()
            .next_back()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric());
        let right = text[end..]
            .chars()
            .next()
            .is_none_or(|ch| !ch.is_ascii_alphanumeric());
        left && right
    })
}

fn title_case(value: &str) -> String {
    value
        .split_ascii_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|first| first.to_ascii_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}
