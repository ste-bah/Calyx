//! Wire types for the resident-service protocol. Moved verbatim from
//! calyx-cli so client and server share one definition; field order is the
//! bincode wire format — do not reorder.

use std::net::SocketAddr;
use std::path::PathBuf;

use calyx_core::{AbsentReason, Modality, Placement, SlotVector};
use serde::{Deserialize, Serialize};

pub const READY_SCHEMA: &str = "calyx-panel-resident-readiness-v1";
pub const MEASURE_SCHEMA: &str = "calyx-panel-resident-measure-v1";
pub const MEASURE_BATCH_SCHEMA: &str = "calyx-panel-resident-measure-batch-v1";
/// v2 (#1002): measure_batch responses stream as one header frame, one frame
/// per measured input row, and one end frame — never a single response frame
/// carrying every multi-vector payload at once.
pub const RESIDENT_BINARY_PROTOCOL_VERSION: u16 = 2;

#[derive(Debug)]
pub enum ClientMeasureInput {
    Utf8(String),
    Hex(String),
}

#[derive(Deserialize)]
pub struct ResidentRequest {
    pub op: String,
    pub modality: Option<Modality>,
    pub input: Option<String>,
    pub input_hex: Option<String>,
    pub inputs_hex: Option<Vec<String>>,
    pub runtime_batch_limit: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ReadyResponse {
    pub schema: String,
    pub ready: bool,
    pub residency_scope: &'static str,
    pub process_id: u32,
    pub bind: SocketAddr,
    pub uptime_ms: u128,
    pub source_of_truth: String,
    pub home: PathBuf,
    pub template_selector: String,
    pub template_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ready_out: Option<PathBuf>,
    pub max_resident_vram_mib: u64,
    pub declared_template_vram_mib: u64,
    pub resident_overhead_multiplier: f32,
    pub estimated_resident_vram_mib: u64,
    pub max_load_secs: u64,
    pub load_parallelism: usize,
    pub load_ms: u128,
    pub probe_ms: u128,
    pub slot_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub slot_scope: Vec<u16>,
    pub content_lens_count: usize,
    pub registry_lens_count: usize,
    pub warmed_lens_count: usize,
    pub gpu_content_lens_count: usize,
    pub cpu_content_lens_count: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MeasureResponse {
    pub schema: String,
    pub ready: bool,
    pub process_id: u32,
    pub template_source: String,
    pub modality: Modality,
    pub input_len: usize,
    pub elapsed_ms: u128,
    pub measured_slot_count: usize,
    pub absent_slot_count: usize,
    pub slots: Vec<ResidentSlotMeasure>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MeasureBatchResponse {
    pub schema: String,
    pub ready: bool,
    pub process_id: u32,
    pub template_source: String,
    pub modality: Modality,
    pub input_count: usize,
    pub elapsed_ms: u128,
    pub runtime_batch_limit: Option<usize>,
    pub rows: Vec<ResidentMeasuredInput>,
}

#[derive(Debug)]
pub struct MeasureBatchAtResponse {
    pub response: MeasureBatchResponse,
    pub request_bytes: usize,
    pub response_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct MeasureBatchSummaryResponse {
    pub schema: String,
    pub ready: bool,
    pub process_id: u32,
    pub template_source: String,
    pub modality: Modality,
    pub input_count: usize,
    pub elapsed_ms: u128,
    pub runtime_batch_limit: Option<usize>,
    pub row_count: usize,
    pub measured_slot_counts: Vec<usize>,
    pub absent_slot_counts: Vec<usize>,
    pub response_rows_sha256: String,
    pub request_bytes: usize,
    pub response_bytes: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResidentMeasureBatchBinaryRequest {
    pub protocol_version: u16,
    pub modality: Modality,
    pub inputs: Vec<Vec<u8>>,
    pub runtime_batch_limit: Option<usize>,
}

/// One length-prefixed bincode frame of the streamed measure_batch response.
#[derive(Debug, Deserialize, Serialize)]
pub enum ResidentMeasureBatchStreamFrame {
    Header(ResidentMeasureBatchStreamHeader),
    Row(Box<ResidentMeasuredInput>),
    End(ResidentMeasureBatchStreamEnd),
    Err {
        code: String,
        message: String,
        remediation: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResidentMeasureBatchStreamHeader {
    pub protocol_version: u16,
    pub schema: String,
    pub ready: bool,
    pub process_id: u32,
    pub template_source: String,
    pub modality: Modality,
    pub input_count: usize,
    pub runtime_batch_limit: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResidentMeasureBatchStreamEnd {
    pub row_count: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResidentMeasuredInput {
    pub input_index: usize,
    pub input_len: usize,
    pub measured_slot_count: usize,
    pub absent_slot_count: usize,
    pub slots: Vec<ResidentSlotMeasure>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResidentSlotMeasure {
    pub slot: u16,
    pub key: String,
    pub lens_id: String,
    pub modality: Modality,
    pub placement: Placement,
    pub measured: bool,
    pub vector: Option<SlotVector>,
    pub absent_reason: Option<AbsentReason>,
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn hex_decode(raw: &str) -> Result<Vec<u8>, String> {
    if !raw.len().is_multiple_of(2) {
        return Err(format!(
            "input_hex length {} is odd; expected complete bytes",
            raw.len()
        ));
    }
    let mut bytes = Vec::with_capacity(raw.len() / 2);
    let raw = raw.as_bytes();
    let mut idx = 0;
    while idx < raw.len() {
        let hi = hex_nibble(raw[idx]).ok_or_else(|| invalid_hex(idx, raw[idx]))?;
        let lo = hex_nibble(raw[idx + 1]).ok_or_else(|| invalid_hex(idx + 1, raw[idx + 1]))?;
        bytes.push((hi << 4) | lo);
        idx += 2;
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn invalid_hex(index: usize, byte: u8) -> String {
    format!("input_hex contains non-hex byte 0x{byte:02x} at character index {index}")
}
