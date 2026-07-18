//! CLI glue for generic search. Grounded kernel answering and reproduction
//! live in sibling modules so every durable state transition has one owner.

#[cfg(test)]
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use calyx_aster::vault::AsterVault;
#[cfg(test)]
use calyx_core::{AnchorKind, Constellation, CxId};
use calyx_core::{CalyxError, Input, Modality, SlotId, SlotVector};
use calyx_registry::{VaultPanelState, load_vault_panel_state, require_vault_registry_contracts};
use calyx_search::{
    FusionChoice, GuardChoice, RERANK_CANDIDATE_FLOOR, SearchBudget, SearchFreshness,
    SearchTraceEvent, rerank_search_outcome, search_outcome_with_freshness,
    search_outcome_with_query_vectors_freshness,
    search_outcome_with_query_vectors_freshness_cached_ledger_view_exact_metadata,
};
#[cfg(test)]
use calyx_sextant::Hit;
use calyx_sextant::RerankerClient;

use super::super::Subcommand;
use super::super::vault::{ResolvedVault, home_dir, resolve_vault_info, vault_salt};
use super::output;
use super::parse::{SearchArgs, SearchFreshnessArg, SearchFusionArg, SearchGuardArg};
use super::roster::{
    SearchTextRoster, measure_local_cpu_query_vectors, query_vectors_from_resident_row,
    require_resident_template_binding,
};
use super::{base_read_cfs, latest_read_vault_options_for_cfs, panel_read_cfs};
use crate::error::CliResult;
use crate::output::print_json;
use crate::panel_commands::measure_resident_batch_at;
use crate::path_identity::vault_template_source;

pub(super) fn run(command: Subcommand) -> CliResult {
    match command {
        Subcommand::Search(args) => search_command(args),
        Subcommand::KernelAnswer(args) => super::kernel_answer::run(args),
        _ => unreachable!("non-search command routed to search module"),
    }
}

fn search_command(args: SearchArgs) -> CliResult {
    let resolved = resolve_cli_vault(&args.vault)?;
    require_vault_registry_contracts(&resolved.path)?;
    let state = load_vault_panel_state(&resolved.path)?;
    // The serving roster is derived from THIS vault's frozen panel. It is
    // emitted only after the search succeeds so fail-closed errors retain the
    // CLI's single-JSON stderr contract.
    let roster = SearchTextRoster::derive(&state);
    let fusion = fusion_choice(args.fusion);
    let guard = guard_choice(args.guard);
    let freshness = freshness_choice(args.freshness);
    let mut outcome = match args.resident_addr {
        Some(addr) => {
            let query_vectors = measure_search_query_vectors_via_resident(
                &state,
                &roster,
                &resolved,
                &args.query,
                addr,
            )?;
            let vault = open_vault(&resolved, search_read_cfs(&state, guard))?;
            let mut trace_sink = emit_search_trace;
            if args.rerank {
                search_outcome_with_query_vectors_freshness_cached_ledger_view_exact_metadata(
                    &vault,
                    &resolved.path,
                    &query_vectors,
                    &args.query,
                    args.k,
                    fusion,
                    guard,
                    None,
                    Some(u64::from(state.panel.version)),
                    args.filter.as_deref(),
                    args.explain,
                    freshness,
                    SearchBudget::disabled(),
                    None,
                    None,
                    Some(&mut trace_sink),
                )?
            } else {
                search_outcome_with_query_vectors_freshness(
                    &vault,
                    &resolved.path,
                    &query_vectors,
                    args.k,
                    fusion,
                    guard,
                    Some(u64::from(state.panel.version)),
                    args.filter.as_deref(),
                    args.explain,
                    freshness,
                    SearchBudget::disabled(),
                    Some(&mut trace_sink),
                )?
            }
        }
        None => {
            require_resident_for_gpu_text_search(&roster)?;
            let vault = open_vault(&resolved, search_read_cfs(&state, guard))?;
            let candidate_k = if args.rerank {
                args.k.max(RERANK_CANDIDATE_FLOOR)
            } else {
                args.k
            };
            search_outcome_with_freshness(
                &vault,
                &state,
                &resolved.path,
                &args.query,
                candidate_k,
                fusion,
                guard,
                args.filter.as_deref(),
                args.explain,
                freshness,
            )?
        }
    };
    let rerank = if args.rerank {
        let reranker = configured_reranker()?;
        Some(rerank_search_outcome(
            &resolved.path,
            &args.query,
            args.k,
            &reranker,
            &mut outcome,
        )?)
    } else {
        None
    };
    if let Some(generation) = &outcome.generation {
        eprintln!(
            "CALYX_SEARCH_GENERATION base_seq={} manifest_sha256={} slots={}",
            generation.base_seq,
            generation.manifest_sha256,
            generation.slots.len()
        );
    }
    let hits = output::render_hits(
        &outcome.hits,
        args.explain,
        args.provenance,
        outcome.guard_tau,
    );
    roster.emit_runtime_line();
    if args.explain {
        return print_json(&output::SearchExplainOut {
            slots: roster.to_out(),
            rerank,
            hits,
        });
    }
    print_json(&hits)
}

fn configured_reranker() -> CliResult<RerankerClient> {
    let endpoint = std::env::var("CALYX_SEARCH_RERANKER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8089".to_string());
    let timeout_ms = std::env::var("CALYX_SEARCH_RERANKER_TIMEOUT_MS")
        .ok()
        .map(|value| {
            value.parse::<u64>().map_err(|error| CalyxError {
                code: "CALYX_SEARCH_RERANKER_CONFIG_INVALID",
                message: format!(
                    "CALYX_SEARCH_RERANKER_TIMEOUT_MS must be an unsigned integer: {error}"
                ),
                remediation: "set CALYX_SEARCH_RERANKER_TIMEOUT_MS to a positive millisecond timeout",
            })
        })
        .transpose()?
        .unwrap_or(3_000);
    Ok(RerankerClient::new(
        endpoint,
        Duration::from_millis(timeout_ms),
    ))
}

fn emit_search_trace(event: SearchTraceEvent) {
    let slot = event
        .slot
        .map(|slot| slot.get().to_string())
        .unwrap_or_else(|| "-".to_string());
    let count = event
        .count
        .map(|count| count.to_string())
        .unwrap_or_else(|| "-".to_string());
    let detail = event.detail.as_deref().unwrap_or("-");
    eprintln!(
        "CALYX_SEARCH_RUNTIME phase={} slot={} elapsed_ms={} count={} detail={}",
        event.phase, slot, event.elapsed_ms, count, detail
    );
}

pub(super) fn measure_search_query_vectors_via_resident(
    state: &VaultPanelState,
    roster: &SearchTextRoster<'_>,
    resolved: &ResolvedVault,
    query: &str,
    addr: SocketAddr,
) -> CliResult<Vec<(SlotId, SlotVector)>> {
    // Active CPU-placed slots are measured by the search process on this host,
    // using the same frozen-contract path as ingest, and are never demanded
    // from the GPU-only resident (#1490 deadlocked mixed panels before this).
    let local_vectors = measure_local_cpu_query_vectors(state, &roster.local_cpu, query)?;
    if roster.resident_gpu.is_empty() {
        eprintln!(
            "CALYX_SEARCH_RUNTIME phase=search_resident_service_skipped addr={addr} reason=no_active_gpu_text_slots local_cpu_slots={}",
            local_vectors.len()
        );
        return Ok(local_vectors);
    }
    let started = Instant::now();
    let input = Input::new(Modality::Text, query.as_bytes().to_vec());
    eprintln!(
        "CALYX_SEARCH_RUNTIME phase=search_resident_service_start addr={} vault={} inputs=1 demanded_gpu_slots={} local_cpu_slots={}",
        addr,
        resolved.path.display(),
        roster.resident_gpu.len(),
        local_vectors.len()
    );
    let resident = measure_resident_batch_at(addr, Modality::Text, &[input], None).map_err(
        |error| {
            CalyxError {
                code: "CALYX_PANEL_RESIDENT_UNAVAILABLE",
                message: format!(
                    "resident search measurement failed addr={addr} code={} message={}",
                    error.code(),
                    error.message()
                ),
                remediation: "start `calyx panel resident serve --vault <vault-path>` for the matching active vault, then retry search with --resident-addr",
            }
        },
    )?;
    let request_bytes = resident.request_bytes;
    let response_bytes = resident.response_bytes;
    let response = resident.response;
    if !response.ready {
        return Err(CalyxError::lens_unreachable(format!(
            "resident service {addr} returned ready=false for search measurement"
        ))
        .into());
    }
    let expected_template_source = vault_template_source(&resolved.path)?;
    require_resident_template_binding(&response.template_source, &expected_template_source, addr)?;
    if response.modality != Modality::Text || response.input_count != 1 || response.rows.len() != 1
    {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "resident search response mismatch: modality {:?} input_count {} rows {}, expected Text/1/1",
            response.modality,
            response.input_count,
            response.rows.len()
        ))
        .into());
    }
    let row = &response.rows[0];
    if row.input_index != 0 {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "resident search response row index {}, expected 0",
            row.input_index
        ))
        .into());
    }

    let mut out = query_vectors_from_resident_row(state, &roster.resident_gpu, row, addr)?;
    let resident_slot_count = out.len();
    out.extend(local_vectors);
    out.sort_by_key(|(slot_id, _)| *slot_id);
    eprintln!(
        "CALYX_SEARCH_RUNTIME phase=search_resident_service_ok addr={} process_id={} template_source={} inputs=1 slots={} local_cpu_slots={} elapsed_ms={} resident_elapsed_ms={} protocol=binary request_bytes={} response_bytes={}",
        addr,
        response.process_id,
        response.template_source,
        resident_slot_count,
        out.len() - resident_slot_count,
        started.elapsed().as_millis(),
        response.elapsed_ms,
        request_bytes,
        response_bytes
    );
    Ok(out)
}

pub(super) fn require_resident_for_gpu_text_search(roster: &SearchTextRoster<'_>) -> CliResult {
    let gpu_slots = roster
        .resident_gpu
        .iter()
        .map(|slot| {
            format!(
                "slot={} key={} lens={} placement={:?}",
                slot.slot_id.get(),
                slot.slot_key.key(),
                slot.lens_id,
                slot.resource.placement
            )
        })
        .collect::<Vec<_>>();
    if gpu_slots.is_empty() {
        return Ok(());
    }
    Err(CalyxError {
        code: "CALYX_SEARCH_RESIDENT_REQUIRED",
        message: format!(
            "search refuses cold local query measurement for {} active GPU text lens(es): {}",
            gpu_slots.len(),
            gpu_slots.join(", ")
        ),
        remediation: "start `calyx panel resident serve --vault <vault-path>` and rerun search with --resident-addr",
    }
    .into())
}

fn fusion_choice(arg: SearchFusionArg) -> FusionChoice {
    match arg {
        SearchFusionArg::Rrf => FusionChoice::Rrf,
        SearchFusionArg::WeightedRrf => FusionChoice::WeightedRrf,
        SearchFusionArg::SingleLens => FusionChoice::SingleLens,
        SearchFusionArg::KernelFirst => FusionChoice::KernelFirst,
        SearchFusionArg::Pipeline => FusionChoice::Pipeline,
    }
}

fn guard_choice(arg: SearchGuardArg) -> GuardChoice {
    match arg {
        SearchGuardArg::Off => GuardChoice::Off,
        SearchGuardArg::InRegion => GuardChoice::InRegion,
    }
}

fn freshness_choice(arg: SearchFreshnessArg) -> SearchFreshness {
    match arg {
        SearchFreshnessArg::Fresh => SearchFreshness::Fresh,
        SearchFreshnessArg::StaleOk => SearchFreshness::StaleOk,
    }
}

#[cfg(test)]
pub(super) fn kernel_report_from_docs(
    docs: &BTreeMap<CxId, Constellation>,
    hits: &[Hit],
    anchor: Option<&AnchorKind>,
) -> CliResult<output::KernelAnswerOut> {
    let grounded = docs
        .values()
        .filter(|cx| has_grounding(cx, anchor))
        .map(|cx| cx.cx_id)
        .collect::<Vec<_>>();
    if grounded.is_empty() {
        return Err(CalyxError::kernel_ungrounded("kernel-answer has no grounded anchors").into());
    }
    let kernel_ids = hits
        .iter()
        .map(|hit| hit.cx_id)
        .filter(|cx_id| grounded.contains(cx_id))
        .take(5)
        .collect::<Vec<_>>();
    if kernel_ids.is_empty() {
        return Err(CalyxError::kernel_ungrounded(
            "kernel-answer search returned no grounded hits",
        )
        .into());
    }
    let gap_count = docs.len().saturating_sub(grounded.len());
    let gaps = (gap_count > 0)
        .then(|| format!("grounding_gaps:{gap_count}"))
        .into_iter()
        .collect();
    Ok(output::KernelAnswerOut {
        answer: format!(
            "grounded kernel answer over {} anchored constellations",
            grounded.len()
        ),
        kernel_cx_ids: kernel_ids.into_iter().map(|id| id.to_string()).collect(),
        recall: grounded.len() as f32 / docs.len().max(1) as f32,
        gaps,
    })
}

#[cfg(test)]
fn has_grounding(cx: &Constellation, anchor: Option<&AnchorKind>) -> bool {
    cx.anchors
        .iter()
        .any(|item| anchor.is_none_or(|kind| &item.kind == kind))
}

pub(super) fn resolve_cli_vault(vault: &str) -> CliResult<ResolvedVault> {
    resolve_vault_info(&home_dir()?, vault)
}

fn open_vault(
    resolved: &ResolvedVault,
    selected_cfs: Option<Vec<calyx_aster::cf::ColumnFamily>>,
) -> CliResult<AsterVault> {
    Ok(AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        latest_read_vault_options_for_cfs(selected_cfs),
    )?)
}

fn search_read_cfs(
    state: &VaultPanelState,
    guard: GuardChoice,
) -> Option<Vec<calyx_aster::cf::ColumnFamily>> {
    match guard {
        GuardChoice::Off => Some(base_read_cfs()),
        GuardChoice::InRegion => {
            // Profile-backed guarding (#1094) reads the calibrated Ward
            // profile from the Guard CF; reading an unselected CF silently
            // returns None, which would masquerade as a missing profile.
            let mut cfs = panel_read_cfs(&state.panel)?;
            cfs.push(calyx_aster::cf::ColumnFamily::Guard);
            cfs.sort();
            cfs.dedup();
            Some(cfs)
        }
    }
}
