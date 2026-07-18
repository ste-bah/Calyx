use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct BitsOut {
    #[serde(default)]
    pub schema_version: u32,
    pub anchor: String,
    pub panel_sufficiency: f64,
    pub n: usize,
    pub dpi_ceiling: f64,
    pub per_slot: Vec<SlotBitsOut>,
    #[serde(default)]
    pub population_n: usize,
    #[serde(default)]
    pub outcome_classes: BTreeMap<String, usize>,
    #[serde(default)]
    pub population_outcome_classes: BTreeMap<String, usize>,
    #[serde(default)]
    pub population_outcome_entropy_bits: f64,
    #[serde(default)]
    pub sample_cx_ids: Vec<String>,
    #[serde(default)]
    pub panel_bits: f64,
    #[serde(default)]
    pub panel_ci: [f64; 2],
    #[serde(default)]
    pub sufficiency_passed: bool,
    #[serde(default)]
    pub pairwise_redundancy: Vec<PairRedundancyOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<BitsExplainOut>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct SlotBitsOut {
    pub slot: u16,
    pub name: String,
    #[serde(default)]
    pub n: usize,
    pub bits: f64,
    pub ci: [f64; 2],
    pub estimator: String,
    #[serde(default)]
    pub representation: String,
    #[serde(default)]
    pub trust: String,
    pub state: String,
    pub low_signal: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct PairRedundancyOut {
    pub left_slot: u16,
    pub right_slot: u16,
    pub nmi: f64,
    pub mi_bits: f64,
    pub n: usize,
    pub estimator: String,
    #[serde(default)]
    pub representation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct BitsExplainOut {
    pub positive_anchor_count: usize,
    pub comparison_count: usize,
    pub persisted_cf: String,
    pub persisted_key_hex: String,
    #[serde(default)]
    pub outcome_mode: String,
    #[serde(default)]
    pub sample_policy: String,
    #[serde(default)]
    pub strict_cuda_required: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct KernelOut {
    pub kernel_size: usize,
    pub recall: f32,
    pub total_cx: usize,
    pub kernel_cx_ids: Vec<String>,
    pub grounding_gaps: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct AbundanceOut {
    pub n: usize,
    pub pairs: u64,
    pub materialized: usize,
    pub n_eff: f64,
    pub dpi_ceiling: f64,
    pub panel_size: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct GuardProfileOut {
    pub domain: String,
    pub tau: f32,
    pub far: f32,
    pub frr: f32,
    pub calibration_corpus_size: usize,
    pub blocked_injection_rate: f32,
    pub per_slot_tau: Vec<SlotTauOut>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct SlotTauOut {
    pub slot: u16,
    pub tau: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct GuardCheckOut {
    pub verdict: &'static str,
    pub tau: f32,
    pub distance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct ProposeLensOut {
    pub name: String,
    pub rationale: String,
    pub predicted_bits_gain: f64,
    pub runtime_hint: String,
    pub candidate: serde_json::Value,
}

pub(super) fn assay_key(anchor: &str) -> Vec<u8> {
    row_key("bits", anchor)
}

pub(super) fn kernel_key(anchor: Option<&str>) -> Vec<u8> {
    row_key("kernel", anchor.unwrap_or("all"))
}

pub(super) fn guard_profile_key(domain: &str) -> Vec<u8> {
    row_key("profile", domain)
}

pub(super) fn default_guard_key() -> Vec<u8> {
    row_key("profile", "default")
}

fn row_key(prefix: &str, subject: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(prefix.len() + 1 + subject.len());
    key.extend_from_slice(prefix.as_bytes());
    key.push(0);
    key.extend_from_slice(subject.as_bytes());
    key
}

pub(super) fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(nibble(byte >> 4));
        out.push(nibble(byte & 0x0f));
    }
    out
}

fn nibble(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        10..=15 => char::from(b'a' + value - 10),
        _ => unreachable!("nibble out of range"),
    }
}
