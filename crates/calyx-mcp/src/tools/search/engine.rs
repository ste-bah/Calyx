use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{AnchorKind, CalyxError, Constellation, CxId, VaultStore};
use calyx_search::{
    FusionChoice, GuardChoice, PersistedSearchGeneration, SearchBudget, SearchError,
    SearchRerankReport, SearchSlotCache, SearchSlotCacheDiagnostic, SearchTraceEvent,
};
use calyx_sextant::{DroppedGuardHit, Hit, RerankerClient};
use serde::Serialize;

use crate::server::{ToolError, ToolResult};

use super::output::KernelAnswerOut;
use super::runtime_cache;
use super::{NeighborsRequest, SearchRequest};
use crate::tools::vault::store::{ResolvedVault, home_dir, resolve_vault_info, vault_salt};

#[path = "engine/maxsim_telemetry.rs"]
mod maxsim_telemetry;
use maxsim_telemetry::McpMaxSimCudaTelemetry;

pub(super) struct SearchOutcome {
    pub(super) hits: Vec<Hit>,
    pub(super) docs: BTreeMap<CxId, Constellation>,
    pub(super) dropped_guard_hits: Vec<DroppedGuardHit>,
    pub(super) execution: McpSearchExecution,
}

#[derive(Serialize)]
pub(super) struct McpSearchExecution {
    executor: &'static str,
    generation: Option<PersistedSearchGeneration>,
    request_index_builds: usize,
    measured_slots: usize,
    persisted_slot_searches: usize,
    persisted_slot_cache_hits: usize,
    persisted_slot_latency_ms: BTreeMap<u16, u128>,
    persisted_slot_compute_placement: BTreeMap<u16, String>,
    slot_cache_enabled: bool,
    vault_runtime_cache_hit: bool,
    panel_runtime_cache_hit: bool,
    vault_open_ms: u128,
    panel_validation_ms: u128,
    vault_open_and_panel_validation_ms: u128,
    pin_budget_required_bytes: Option<u64>,
    pin_budget_projected_process_bytes: Option<u64>,
    pin_budget_configured_bytes: Option<u64>,
    multi_stage1_candidate_count: Option<usize>,
    maxsim_cuda: Option<McpMaxSimCudaTelemetry>,
    model_measure_ms: u128,
    search_pipeline_ms: u128,
    persisted_slot_phase_ms: Option<u128>,
    hydration_ms: Option<u128>,
    rerank: Option<SearchRerankReport>,
    provenance_ms: Option<u128>,
    provenance_ledger_view: &'static str,
    provenance_ledger_view_reused: bool,
    provenance_ledger_head_height: Option<u64>,
    provenance_ledger_rows_read: Option<usize>,
    provenance_ledger_physical_reopens: usize,
    elapsed_ms: u128,
    cache_before: SearchSlotCacheDiagnostic,
    cache_after: SearchSlotCacheDiagnostic,
    capabilities: McpSearchCapabilities,
}

#[derive(Serialize)]
struct McpSearchCapabilities {
    cuda_feature: bool,
    forge_cuda: bool,
    registry_candle_cuda: bool,
    search_cuda: bool,
    sextant_cuvs: bool,
}

#[derive(Serialize)]
pub(super) struct NeighborOut {
    cx_id: String,
    score: f32,
    slot: u16,
}

pub(super) fn search(request: &SearchRequest) -> ToolResult<SearchOutcome> {
    ensure_compiled_capability_contract()?;
    let started = Instant::now();
    let resolved = resolve_requested_vault(&request.vault)?;
    let vault_open_started = Instant::now();
    let (vault, ledger_view, vault_runtime_cache_hit, vault_generation) =
        runtime_cache::cached_vault(&resolved)?;
    let vault_open_ms = vault_open_started.elapsed().as_millis();
    let panel_validation_started = Instant::now();
    let (state, panel_runtime_cache_hit) = runtime_cache::cached_panel_state(&resolved.path)?;
    runtime_cache::ensure_vault_generation_unchanged(&resolved, &vault_generation)?;
    let panel_validation_ms = panel_validation_started.elapsed().as_millis();
    let vault_open_and_panel_validation_ms = started.elapsed().as_millis();
    let model_measure_started = Instant::now();
    let query_vectors = map_search(calyx_search::measure_query_vectors(&state, &request.query))?;
    let model_measure_ms = model_measure_started.elapsed().as_millis();
    let filter_json = request
        .filter
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| ToolError::invalid_params(format!("encode search filter: {error}")))?;
    let mut events = Vec::new();
    let mut trace = |event: SearchTraceEvent| events.push(event);
    let search_started = Instant::now();
    // Filtered recall may widen to the complete candidate set. Keep those
    // request-specific hit lists out of the process cache.
    let cache_enabled = filter_json.is_none();
    let mut run = |cache| {
        let outcome = if request.rerank {
            calyx_search::search_outcome_with_query_vectors_freshness_cached_ledger_view_exact_metadata(
                &vault,
                &resolved.path,
                &query_vectors,
                &request.query,
                request.k,
                request.fusion.to_choice(),
                request.guard.to_choice(),
                None,
                Some(u64::from(state.panel.version)),
                filter_json.as_deref(),
                request.explain,
                request.freshness,
                SearchBudget::disabled(),
                cache,
                Some(&ledger_view),
                Some(&mut trace),
            )
        } else {
            calyx_search::search_outcome_with_query_vectors_freshness_cached_ledger_view(
                &vault,
                &resolved.path,
                &query_vectors,
                request.k,
                request.fusion.to_choice(),
                request.guard.to_choice(),
                None,
                Some(u64::from(state.panel.version)),
                filter_json.as_deref(),
                request.explain,
                request.freshness,
                SearchBudget::disabled(),
                cache,
                Some(&ledger_view),
                Some(&mut trace),
            )
        };
        map_search(outcome)
    };
    let (mut outcome, cache_before, cache_after) = if cache_enabled {
        let mut cache = lock_slot_cache()?;
        let before = cache.diagnostics();
        let outcome = run(Some(&mut cache))?;
        let after = cache.diagnostics();
        (outcome, before, after)
    } else {
        let diagnostic = lock_slot_cache()?.diagnostics();
        let outcome = run(None)?;
        (outcome, diagnostic.clone(), diagnostic)
    };
    let search_pipeline_ms = search_started.elapsed().as_millis();
    let rerank = if request.rerank {
        let client = configured_reranker()?;
        Some(map_search(calyx_search::rerank_search_outcome(
            &resolved.path,
            &request.query,
            request.k,
            &client,
            &mut outcome,
        ))?)
    } else {
        None
    };
    let docs = if request.include_all_docs {
        map_search(calyx_search::load_docs(&vault))?
    } else {
        outcome.docs
    };
    runtime_cache::ensure_vault_generation_unchanged(&resolved, &vault_generation)?;
    let execution = McpSearchExecution {
        executor: "calyx-search/persisted",
        generation: outcome.generation,
        request_index_builds: 0,
        measured_slots: query_vectors.len(),
        persisted_slot_searches: count_phase(&events, "search_slot.done"),
        persisted_slot_cache_hits: count_phase(&events, "search_slot.cache_hit"),
        persisted_slot_latency_ms: slot_latencies(&events),
        persisted_slot_compute_placement: slot_compute_placements(&events),
        slot_cache_enabled: cache_enabled,
        vault_runtime_cache_hit,
        panel_runtime_cache_hit,
        vault_open_ms,
        panel_validation_ms,
        vault_open_and_panel_validation_ms,
        pin_budget_required_bytes: phase_detail_u64(
            &events,
            "indexes.pin_budget.done",
            "required_bytes",
        ),
        pin_budget_projected_process_bytes: phase_detail_u64(
            &events,
            "indexes.pin_budget.done",
            "projected_process_bytes",
        ),
        pin_budget_configured_bytes: phase_detail_u64(
            &events,
            "indexes.pin_budget.done",
            "configured_bytes",
        ),
        multi_stage1_candidate_count: phase_count(&events, "search_slots.multi_candidates"),
        maxsim_cuda: maxsim_telemetry::from_events(&events),
        model_measure_ms,
        search_pipeline_ms,
        persisted_slot_phase_ms: phase_duration_ms(
            &events,
            "search_slots.start",
            "search_slots.done",
        ),
        hydration_ms: phase_duration_ms(&events, "hit_docs.hydrate.start", "hit_docs.hydrate.done"),
        rerank,
        provenance_ms: phase_duration_ms(
            &events,
            "provenance.attach.start",
            "provenance.attach.done",
        ),
        provenance_ledger_view: "aster-ledger-cf-store-snapshot",
        provenance_ledger_view_reused: vault_runtime_cache_hit,
        provenance_ledger_head_height: vault_generation.ledger_head_height,
        provenance_ledger_rows_read: phase_count(&events, "provenance.ledger_snapshot_read.done"),
        provenance_ledger_physical_reopens: count_phase(
            &events,
            "provenance.ledger_point_read.tier",
        ),
        elapsed_ms: started.elapsed().as_millis(),
        cache_before,
        cache_after,
        capabilities: capabilities(),
    };
    Ok(SearchOutcome {
        hits: outcome.hits,
        docs,
        dropped_guard_hits: outcome.dropped_guard_hits,
        execution,
    })
}

fn configured_reranker() -> ToolResult<RerankerClient> {
    let endpoint = std::env::var("CALYX_SEARCH_RERANKER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8089".to_string());
    let timeout_ms = std::env::var("CALYX_SEARCH_RERANKER_TIMEOUT_MS")
        .ok()
        .map(|value| {
            value.parse::<u64>().map_err(|error| {
                ToolError::invalid_params(format!(
                    "CALYX_SEARCH_RERANKER_TIMEOUT_MS must be an unsigned integer: {error}"
                ))
            })
        })
        .transpose()?
        .unwrap_or(3_000);
    Ok(RerankerClient::new(
        endpoint,
        Duration::from_millis(timeout_ms),
    ))
}

fn phase_duration_ms(
    events: &[SearchTraceEvent],
    start_phase: &'static str,
    end_phase: &'static str,
) -> Option<u128> {
    let started = events
        .iter()
        .find(|event| event.phase == start_phase)?
        .elapsed_ms;
    let ended = events
        .iter()
        .rev()
        .find(|event| event.phase == end_phase)?
        .elapsed_ms;
    ended.checked_sub(started)
}

fn phase_count(events: &[SearchTraceEvent], phase: &str) -> Option<usize> {
    events
        .iter()
        .rev()
        .find(|event| event.phase == phase)?
        .count
}

fn phase_detail_u64(events: &[SearchTraceEvent], phase: &str, field: &str) -> Option<u64> {
    let detail = events
        .iter()
        .rev()
        .find(|event| event.phase == phase)?
        .detail
        .as_deref()?;
    detail.split_whitespace().find_map(|entry| {
        let (name, value) = entry.split_once('=')?;
        (name == field).then(|| value.parse().ok()).flatten()
    })
}

fn slot_latencies(events: &[SearchTraceEvent]) -> BTreeMap<u16, u128> {
    events
        .iter()
        .filter(|event| event.phase == "search_slot.done")
        .filter_map(|event| {
            let slot = event.slot?.get();
            let elapsed = event
                .detail
                .as_deref()?
                .split_whitespace()
                .find_map(|entry| {
                    let (name, value) = entry.split_once('=')?;
                    (name == "slot_elapsed_ms")
                        .then(|| value.parse::<u128>().ok())
                        .flatten()
                })?;
            Some((slot, elapsed))
        })
        .collect()
}

fn slot_compute_placements(events: &[SearchTraceEvent]) -> BTreeMap<u16, String> {
    events
        .iter()
        .filter(|event| event.phase == "search_slot.done")
        .filter_map(|event| {
            let slot = event.slot?.get();
            let placement = event
                .detail
                .as_deref()?
                .split_whitespace()
                .find_map(|entry| {
                    let (name, value) = entry.split_once('=')?;
                    matches!(name, "slot_compute_placement" | "maxsim_compute_placement")
                        .then(|| value.to_string())
                })?;
            Some((slot, placement))
        })
        .collect()
}

pub(super) fn neighbors(request: &NeighborsRequest) -> ToolResult<Vec<NeighborOut>> {
    ensure_compiled_capability_contract()?;
    let resolved = resolve_requested_vault(&request.vault)?;
    let vault = open_vault(&resolved)?;
    let snapshot = vault.snapshot();
    let seed = vault.get(request.cx_id, snapshot)?;
    let indexes = map_search(calyx_search::PersistedSearchIndexes::open(&resolved.path))?;
    indexes
        .ensure_fresh_at_snapshot(snapshot, vault.derived_content_seq().min(snapshot))
        .map_err(map_search_error)?;
    indexes.ensure_search_bounded().map_err(map_search_error)?;
    let generation = map_search(indexes.generation())?;
    let mut out = Vec::new();
    for persisted in generation.slots {
        if request.slot.is_some_and(|wanted| wanted != persisted.slot) {
            continue;
        }
        let Some(vector) = seed.slots.get(&persisted.slot) else {
            continue;
        };
        let hits = map_search(calyx_search::search_outcome_with_query_vectors(
            &vault,
            &resolved.path,
            &[(persisted.slot, vector.clone())],
            request.k,
            FusionChoice::SingleLensSlot(persisted.slot),
            GuardChoice::Off,
            None,
            false,
            None,
        ))?;
        out.extend(hits.hits.into_iter().map(|hit| NeighborOut {
            cx_id: hit.cx_id.to_string(),
            score: hit.score,
            slot: persisted.slot.get(),
        }));
    }
    if out.is_empty() && request.slot.is_some() {
        return Err(no_indexable_stored_vectors().into());
    }
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.slot.cmp(&b.slot))
            .then_with(|| a.cx_id.cmp(&b.cx_id))
    });
    out.truncate(request.k);
    Ok(out)
}

pub(super) fn kernel_report(
    docs: &BTreeMap<CxId, Constellation>,
    hits: &[Hit],
    anchor: Option<&AnchorKind>,
) -> ToolResult<KernelAnswerOut> {
    let grounded = docs
        .values()
        .filter(|cx| has_grounding(cx, anchor))
        .map(|cx| cx.cx_id)
        .collect::<Vec<_>>();
    if grounded.is_empty() {
        return Err(CalyxError::kernel_ungrounded("kernel-answer has no grounded anchors").into());
    }
    let mut kernel_ids = hits
        .iter()
        .map(|hit| hit.cx_id)
        .filter(|cx_id| grounded.contains(cx_id))
        .take(5)
        .collect::<Vec<_>>();
    if kernel_ids.is_empty() {
        kernel_ids.extend(grounded.iter().copied().take(5));
    }
    let gap_count = docs.len().saturating_sub(grounded.len());
    let gaps = (gap_count > 0)
        .then(|| format!("grounding_gaps:{gap_count}"))
        .into_iter()
        .collect();
    Ok(KernelAnswerOut {
        answer: format!(
            "grounded kernel answer over {} anchored constellations",
            grounded.len()
        ),
        kernel_cx_ids: kernel_ids.into_iter().map(|id| id.to_string()).collect(),
        recall: grounded.len() as f32 / docs.len().max(1) as f32,
        gaps,
    })
}

pub(super) fn resolve_requested_vault(vault: &str) -> ToolResult<ResolvedVault> {
    resolve_vault_info(&home_dir()?, vault)
}

pub(super) fn open_vault(resolved: &ResolvedVault) -> ToolResult<AsterVault> {
    Ok(AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        VaultOptions::default(),
    )?)
}

fn has_grounding(cx: &Constellation, requested: Option<&AnchorKind>) -> bool {
    cx.anchors
        .iter()
        .any(|anchor| requested.is_none_or(|kind| &anchor.kind == kind))
}

fn slot_cache() -> &'static Mutex<SearchSlotCache> {
    static CACHE: OnceLock<Mutex<SearchSlotCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(SearchSlotCache::new()))
}

fn lock_slot_cache() -> ToolResult<std::sync::MutexGuard<'static, SearchSlotCache>> {
    slot_cache().lock().map_err(|_| {
        CalyxError::stale_derived("MCP persisted search slot cache lock is poisoned").into()
    })
}

fn count_phase(events: &[SearchTraceEvent], phase: &str) -> usize {
    events.iter().filter(|event| event.phase == phase).count()
}

fn capabilities() -> McpSearchCapabilities {
    McpSearchCapabilities {
        cuda_feature: cfg!(feature = "cuda"),
        forge_cuda: calyx_forge::CUDA_COMPILED,
        registry_candle_cuda: calyx_registry::CANDLE_CUDA_COMPILED,
        search_cuda: calyx_search::CUDA_COMPILED,
        sextant_cuvs: calyx_sextant::CUVS_COMPILED,
    }
}

fn ensure_compiled_capability_contract() -> ToolResult<()> {
    let capabilities = capabilities();
    if capabilities.cuda_feature
        && !(capabilities.forge_cuda
            && capabilities.registry_candle_cuda
            && capabilities.search_cuda
            && capabilities.sextant_cuvs)
    {
        return Err(CalyxError::stale_derived(
            "calyx-mcp cuda feature is enabled without the complete Forge/Registry/Sextant CUDA capability set",
        )
        .into());
    }
    Ok(())
}

fn map_search<T>(result: Result<T, SearchError>) -> ToolResult<T> {
    result.map_err(map_search_error)
}

fn map_search_error(error: SearchError) -> ToolError {
    match error {
        SearchError::Calyx(error) => error.into(),
        SearchError::Usage(message) => ToolError::invalid_params(message),
        SearchError::Io(message) => {
            CalyxError::stale_derived(format!("persistent search I/O failure: {message}")).into()
        }
    }
}

fn no_indexable_stored_vectors() -> CalyxError {
    CalyxError::stale_derived(
        "search has no indexable stored slot vectors matching the requested persisted generation",
    )
}

#[cfg(test)]
pub(super) fn reset_slot_cache_for_tests() {
    if let Ok(mut cache) = slot_cache().lock() {
        *cache = SearchSlotCache::new();
    }
}
