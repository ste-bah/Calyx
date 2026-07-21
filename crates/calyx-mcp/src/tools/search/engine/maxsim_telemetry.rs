use calyx_search::SearchTraceEvent;
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct McpMaxSimCudaTelemetry {
    slot: u16,
    backend: String,
    source_placement: String,
    staging_placement: String,
    compute_placement: String,
    strict: bool,
    filtered: bool,
    source_rows: usize,
    source_tokens: usize,
    source_bytes: u64,
    candidate_rows_requested: Option<usize>,
    candidate_rows_resolved: Option<usize>,
    resident_cache_hit: Option<bool>,
    rows_scanned: usize,
    tokens_scanned: usize,
    bytes_scanned: u64,
    rows_decoded: usize,
    tokens_decoded: usize,
    bytes_decoded: u64,
    physical_bytes_read: u64,
    rows_uploaded: usize,
    tokens_uploaded: usize,
    vector_bytes_uploaded: u64,
    h2d_bytes: u64,
    slot_elapsed_ms: u128,
    slot_elapsed_us: u128,
    cuda_elapsed_us: u128,
}

pub(super) fn from_events(events: &[SearchTraceEvent]) -> Option<McpMaxSimCudaTelemetry> {
    let event = events.iter().rev().find(|event| {
        event.phase == "search_slot.done"
            && event
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("maxsim_cuda_backend="))
    })?;
    let detail = event.detail.as_deref()?;
    Some(McpMaxSimCudaTelemetry {
        slot: event.slot?.get(),
        backend: detail_value(detail, "maxsim_cuda_backend")?.to_string(),
        source_placement: detail_value(detail, "maxsim_source_placement")?.to_string(),
        staging_placement: detail_value(detail, "maxsim_staging_placement")?.to_string(),
        compute_placement: detail_value(detail, "maxsim_compute_placement")?.to_string(),
        strict: detail_parse(detail, "maxsim_cuda_strict")?,
        filtered: detail_parse(detail, "maxsim_cuda_filtered")?,
        source_rows: detail_parse(detail, "maxsim_source_rows")?,
        source_tokens: detail_parse(detail, "maxsim_source_tokens")?,
        source_bytes: detail_parse(detail, "maxsim_source_bytes")?,
        candidate_rows_requested: detail_optional(detail, "maxsim_candidate_rows_requested")?,
        candidate_rows_resolved: detail_optional(detail, "maxsim_candidate_rows_resolved")?,
        resident_cache_hit: detail_optional(detail, "maxsim_resident_cache_hit")?,
        rows_scanned: detail_parse(detail, "maxsim_rows_scanned")?,
        tokens_scanned: detail_parse(detail, "maxsim_tokens_scanned")?,
        bytes_scanned: detail_parse(detail, "maxsim_bytes_scanned")?,
        rows_decoded: detail_parse(detail, "maxsim_rows_decoded")?,
        tokens_decoded: detail_parse(detail, "maxsim_tokens_decoded")?,
        bytes_decoded: detail_parse(detail, "maxsim_bytes_decoded")?,
        physical_bytes_read: detail_parse(detail, "maxsim_physical_bytes_read")?,
        rows_uploaded: detail_parse(detail, "maxsim_rows_uploaded")?,
        tokens_uploaded: detail_parse(detail, "maxsim_tokens_uploaded")?,
        vector_bytes_uploaded: detail_parse(detail, "maxsim_vector_bytes_uploaded")?,
        h2d_bytes: detail_parse(detail, "maxsim_h2d_bytes")?,
        slot_elapsed_ms: detail_parse(detail, "slot_elapsed_ms")?,
        slot_elapsed_us: detail_parse(detail, "maxsim_slot_elapsed_us")?,
        cuda_elapsed_us: detail_parse(detail, "maxsim_cuda_elapsed_us")?,
    })
}

fn detail_value<'a>(detail: &'a str, field: &str) -> Option<&'a str> {
    detail.split_whitespace().find_map(|entry| {
        let (name, value) = entry.split_once('=')?;
        (name == field).then_some(value)
    })
}

fn detail_parse<T: std::str::FromStr>(detail: &str, field: &str) -> Option<T> {
    detail_value(detail, field)?.parse().ok()
}

fn detail_optional<T: std::str::FromStr>(detail: &str, field: &str) -> Option<Option<T>> {
    let value = detail_value(detail, field)?;
    if value == "none" {
        Some(None)
    } else {
        Some(Some(value.parse().ok()?))
    }
}
