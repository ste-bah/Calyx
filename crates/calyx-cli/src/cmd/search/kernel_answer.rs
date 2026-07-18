//! Issue #1480 grounded answer execution against a sealed kernel generation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::SocketAddr;
use std::path::Path;

use calyx_aster::ledger_view::AsterLedgerCfStore;
use calyx_aster::plain_graph::PhysicalPlainGraph;
use calyx_aster::vault::AsterVault;
use calyx_core::{CalyxError, CxId, SlotId, SlotVector};
use calyx_ledger::{EntryKind, LedgerCfStore, SubjectId, decode};
use calyx_lodestar::{
    ASTER_ASSOC_METADATA_KEY, AsterAssocMetadata, AsterAssocNodeProps,
    AsterPanelKernelAnswerRequest, FsKernelStore, KernelAnswerRecordContext,
    PANEL_ASTER_ASSOC_COLLECTION, PANEL_RRF_K, PanelFusionHit, PanelVectors, RecallPassMode,
    derive_panel_kernel_answer, kernel_health, load_panel_kernel_index,
    panel_kernel_answer_with_aster_ledger, panel_kernel_search, rank_panel_candidate_refs,
    read_kernel_artifact,
};
use calyx_registry::{VaultPanelState, load_vault_panel_state, require_vault_registry_contracts};

use super::super::ingest::parse_anchor_kind;
use super::super::kernel_generation::{
    decode_sha256, hex32, load_current_generation, physical_graph_contract, sha256_bytes,
};
use super::super::vault::{ResolvedVault, vault_salt};
use super::engine::{
    measure_search_query_vectors_via_resident, require_resident_for_gpu_text_search,
    resolve_cli_vault,
};
use super::kernel_source_support::{CALYX_KERNEL_SOURCE_LOW_SUPPORT, evaluate_path_source_support};
use super::output;
use super::parse::KernelAnswerArgs;
use super::roster::{SearchTextRoster, measure_local_cpu_query_vectors};
use crate::durable_write::write_bytes_atomic;
use crate::error::{CliError, CliResult};
use crate::output::print_json;

pub(super) fn run(args: KernelAnswerArgs) -> CliResult {
    let anchor = args.anchor.as_deref().map(parse_anchor_kind).transpose()?;
    let resolved = resolve_cli_vault(&args.vault)?;
    require_vault_registry_contracts(&resolved.path)?;
    let state = load_vault_panel_state(&resolved.path)?;
    let roster = SearchTextRoster::derive(&state);
    let current = load_current_generation(&resolved.path)?;
    let graph_contract = &current.manifest.graph;
    let admission = &current.manifest.admission;
    if graph_contract.panel_version != u64::from(state.panel.version) {
        return Err(CliError::runtime(format!(
            "CURRENT kernel panel version {} differs from active panel {}",
            graph_contract.panel_version, state.panel.version
        )));
    }
    let plain = PhysicalPlainGraph::open_latest(&resolved.path, PANEL_ASTER_ASSOC_COLLECTION)?;
    let metadata_bytes = plain
        .get_metadata(ASTER_ASSOC_METADATA_KEY)?
        .ok_or_else(|| CliError::runtime("physical graph metadata disappeared"))?;
    let metadata: AsterAssocMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| CliError::runtime(format!("decode physical graph metadata: {error}")))?;
    let node_rows = plain.node_props()?;
    let csr_bytes = plain
        .read_csr_bytes()?
        .ok_or_else(|| CliError::runtime("physical graph CSR disappeared"))?;
    let graph = plain.assoc_graph()?;
    let (physical_hash, node_props_sha256) =
        physical_graph_contract(&metadata_bytes, &node_rows, &csr_bytes);
    if metadata.embedding_slot.is_some()
        || metadata.embedding_slots != graph_contract.embedding_slots
        || metadata.fusion.as_deref() != Some(graph_contract.fusion.as_str())
        || metadata.rrf_k != Some(graph_contract.rrf_k)
        || metadata.panel_version != Some(graph_contract.panel_version)
        || metadata.graph_source_seq != Some(graph_contract.source_seq)
        || metadata.knn != Some(graph_contract.knn)
        || metadata.edge_score_threshold != Some(graph_contract.edge_score_threshold)
        || graph.node_count() != graph_contract.nodes
        || graph.edge_count() != graph_contract.edges
        || sha256_bytes(&metadata_bytes) != graph_contract.metadata_sha256
        || node_props_sha256 != graph_contract.node_props_sha256
        || sha256_bytes(&csr_bytes) != graph_contract.csr_sha256
        || hex32(&physical_hash) != graph_contract.physical_contract_sha256
    {
        return Err(CliError::runtime(
            "physical association graph differs from the CURRENT kernel generation contract",
        ));
    }
    let store = FsKernelStore::new(&resolved.path);
    let kernel = read_kernel_artifact(current.manifest.kernel_id, &store)?;
    let kernel_index = load_panel_kernel_index(current.manifest.kernel_id, &store)?;
    let health = kernel_health(current.manifest.kernel_id, &store)?;
    if health.recall.pass_mode != RecallPassMode::Passed
        || health.recall.ratio < health.recall.min_recall_ratio
        || health.grounded_fraction != 1.0
    {
        return Err(CalyxError::kernel_ungrounded(format!(
            "CURRENT kernel failed answer gate: recall_mode={:?} ratio={} min={} grounded_fraction={}",
            health.recall.pass_mode,
            health.recall.ratio,
            health.recall.min_recall_ratio,
            health.grounded_fraction
        ))
        .into());
    }
    if let Some(jurisdiction) = &current.manifest.jurisdiction
        && let Some(conflict) =
            super::super::kernel_scope::explicit_scope_conflict(&args.query, jurisdiction)
    {
        roster.emit_runtime_line();
        return print_json(&output::GroundedKernelScopeRefusalOut {
            status: "refused",
            code: "CALYX_KERNEL_QUERY_OUT_OF_SCOPE",
            reason: "explicit_query_jurisdiction_conflicts_with_kernel_scope",
            detected_kind: conflict.kind,
            detected_scope: conflict.detected,
            allowed_court_system: jurisdiction.court_system.clone(),
            allowed_state: jurisdiction.state.clone(),
            allowed_county: jurisdiction.county.clone(),
            kernel_id: current.manifest.kernel_id.to_string(),
            kernel_manifest_sha256: current.manifest_sha256,
        });
    }
    let query_vectors =
        measure_kernel_query_vectors(&state, &roster, &resolved, &args.query, args.resident_addr)?;
    let query_panel = query_panel_vectors(&query_vectors, &graph_contract.embedding_slots)?;
    let parsed_props = parse_graph_props(&node_rows, &graph_contract.embedding_slots)?;
    let nearest = nearest_graph_node(&parsed_props, &query_panel, &graph_contract.embedding_slots)?;
    let query_cx = nearest.cx_id;
    if nearest.score < admission.threshold {
        roster.emit_runtime_line();
        return print_json(&output::GroundedKernelRefusalOut {
            status: "refused",
            code: "CALYX_KERNEL_QUERY_OUT_OF_SCOPE",
            reason: "nearest_constellation_panel_score_below_calibrated_in_domain_lower_tail",
            nearest_cx_id: query_cx.to_string(),
            nearest_score: nearest.score,
            nearest_lanes: nearest.lanes,
            admission_threshold: admission.threshold,
            kernel_id: kernel.kernel_id.to_string(),
            kernel_manifest_sha256: current.manifest_sha256,
            embedding_slots: graph_contract
                .embedding_slots
                .iter()
                .map(|slot| slot.get())
                .collect(),
            fusion: "rrf",
            rrf_k: graph_contract.rrf_k,
        });
    }
    if args.citation_target.is_some() {
        return super::kernel_citation_answer::run(
            super::kernel_citation_answer::CitationAnswerRequest {
                args: &args,
                resolved: &resolved,
                roster: &roster,
                kernel_id: kernel.kernel_id,
                kernel_manifest_sha256: &current.manifest_sha256,
                embedding_slots: &graph_contract.embedding_slots,
                fusion: &graph_contract.fusion,
                rrf_k: graph_contract.rrf_k,
                nearest: &nearest,
                admission_threshold: admission.threshold,
            },
        );
    }
    let kernel_members = kernel.members.iter().copied().collect::<BTreeSet<_>>();
    let anchored_kernel_nodes = parsed_props
        .iter()
        .filter(|(id, props)| {
            kernel_members.contains(id)
                && props
                    .anchors
                    .iter()
                    .any(|stored| anchor.as_ref().is_none_or(|requested| stored == requested))
        })
        .map(|(id, _)| *id)
        .collect::<Vec<_>>();
    if anchored_kernel_nodes.is_empty() {
        return Err(CalyxError::kernel_ungrounded(
            "CURRENT kernel has no members matching the requested physical graph anchors",
        )
        .into());
    }
    let ranked_kernel_nodes =
        panel_kernel_search(&kernel_index, &query_panel, kernel_index.rows().len())?
            .into_iter()
            .map(|hit| hit.cx_id)
            .collect::<Vec<_>>();
    let derivation = derive_panel_kernel_answer(
        kernel.kernel_id,
        &ranked_kernel_nodes,
        &graph,
        query_cx,
        &anchored_kernel_nodes,
        args.max_hops,
    )?;
    let source_support = evaluate_path_source_support(&resolved, &args.query, &derivation)?;
    if source_support.verdict != "supported" {
        roster.emit_runtime_line();
        return print_json(&output::GroundedKernelSourceRefusalOut {
            status: "refused",
            code: CALYX_KERNEL_SOURCE_LOW_SUPPORT,
            reason: "retained_constellation_sources_do_not_support_query_proposition",
            nearest_cx_id: query_cx.to_string(),
            nearest_score: nearest.score,
            nearest_lanes: nearest.lanes,
            admission_threshold: admission.threshold,
            kernel_id: kernel.kernel_id.to_string(),
            kernel_manifest_sha256: current.manifest_sha256,
            embedding_slots: graph_contract
                .embedding_slots
                .iter()
                .map(|slot| slot.get())
                .collect(),
            fusion: "rrf",
            rrf_k: graph_contract.rrf_k,
            source_support,
        });
    }
    let query_sha256 = sha256_bytes(args.query.as_bytes());
    let query_hash = decode_sha256(&query_sha256, "query_input_sha256")?;
    let manifest_hash = decode_sha256(&current.manifest_sha256, "kernel_manifest_sha256")?;
    let answer_id = kernel_answer_id(
        resolved.vault_id.to_string().as_bytes(),
        &query_hash,
        &manifest_hash,
        args.anchor.as_deref(),
        args.max_hops,
    );
    let query_pointer = retain_kernel_query(&resolved.path, &query_sha256, args.query.as_bytes())?;
    let resident_addr = args
        .resident_addr
        .map(|addr| addr.to_string())
        .unwrap_or_else(|| "local-cpu".to_string());
    let context = KernelAnswerRecordContext {
        answer_id: answer_id.to_vec(),
        query_input_sha256: query_hash,
        query_input_pointer: query_pointer.clone(),
        kernel_manifest_sha256: manifest_hash,
        embedding_slots: graph_contract.embedding_slots.clone(),
        fusion: graph_contract.fusion.clone(),
        rrf_k: graph_contract.rrf_k,
        nearest_score: nearest.score,
        nearest_lanes: nearest.lanes.clone(),
        admission_threshold: admission.threshold,
        resident_addr,
        anchor: args.anchor.clone(),
        max_hops: args.max_hops,
        source_support: source_support.clone(),
    };
    let writable_vault = AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        Default::default(),
    )?;
    let answer = panel_kernel_answer_with_aster_ledger(AsterPanelKernelAnswerRequest {
        kernel_id: kernel.kernel_id,
        ranked_kernel_nodes: &ranked_kernel_nodes,
        graph: &graph,
        query_cx,
        anchored_kernel_nodes: &anchored_kernel_nodes,
        max_hops: args.max_hops,
        context: &context,
        vault: &writable_vault,
        vault_dir: &resolved.path,
    })?;
    let complete_ref = answer
        .provenance
        .last()
        .ok_or_else(|| CliError::runtime("kernel answer returned no complete ledger ref"))?;
    verify_complete_answer_readback(&resolved.path, complete_ref, &context, query_cx)?;
    let ledger_refs = answer
        .provenance
        .iter()
        .map(ledger_ref_out)
        .collect::<Vec<_>>();
    let hops = answer
        .hops
        .iter()
        .map(|hop| output::GroundedKernelAnswerHopOut {
            edge_kind: "association",
            from_cx_id: hop.from.to_string(),
            to_cx_id: hop.to.to_string(),
            edge_weight: hop.edge_weight,
            hop_index: hop.hop_index,
            hop_score: hop.hop_score,
            ledger_ref: ledger_ref_out(&hop.ledger_ref),
        })
        .collect();
    roster.emit_runtime_line();
    print_json(&output::GroundedKernelAnswerOut {
        status: "grounded",
        traversal_mode: "association",
        answer_id: hex32(&answer_id),
        query_cx_id: query_cx.to_string(),
        kernel_id: kernel.kernel_id.to_string(),
        kernel_manifest_sha256: current.manifest_sha256,
        embedding_slots: graph_contract
            .embedding_slots
            .iter()
            .map(|slot| slot.get())
            .collect(),
        fusion: "rrf",
        rrf_k: graph_contract.rrf_k,
        nearest_score: nearest.score,
        nearest_lanes: nearest.lanes,
        admission_threshold: admission.threshold,
        anchor_kernel_node_id: answer.anchor_kernel_node.to_string(),
        hops,
        total_score: answer.total_score,
        ledger_refs,
        retained_query_pointer: query_pointer,
        source_support,
        physical_readback: "verified",
    })
}

pub(super) fn measure_kernel_query_vectors(
    state: &VaultPanelState,
    roster: &SearchTextRoster<'_>,
    resolved: &ResolvedVault,
    query: &str,
    resident_addr: Option<SocketAddr>,
) -> CliResult<Vec<(SlotId, SlotVector)>> {
    match resident_addr {
        Some(addr) => {
            measure_search_query_vectors_via_resident(state, roster, resolved, query, addr)
        }
        None => {
            require_resident_for_gpu_text_search(roster)?;
            measure_local_cpu_query_vectors(state, &roster.local_cpu, query)
        }
    }
}

pub(super) fn parse_graph_props(
    rows: &[(CxId, Vec<u8>)],
    embedding_slots: &[SlotId],
) -> CliResult<Vec<(CxId, AsterAssocNodeProps)>> {
    let mut parsed = Vec::with_capacity(rows.len());
    for (id, bytes) in rows {
        let props: AsterAssocNodeProps = serde_json::from_slice(bytes).map_err(|error| {
            CliError::runtime(format!("decode physical graph node {id} props: {error}"))
        })?;
        if props.embedding.is_some() {
            return Err(CliError::runtime(format!(
                "physical graph node {id} contains a forbidden legacy single embedding"
            )));
        }
        if props
            .embeddings
            .keys()
            .copied()
            .ne(embedding_slots.iter().copied())
        {
            return Err(CliError::runtime(format!(
                "physical graph node {id} slots {:?} differ from contract {:?}",
                props
                    .embeddings
                    .keys()
                    .map(|slot| slot.get())
                    .collect::<Vec<_>>(),
                embedding_slots
                    .iter()
                    .map(|slot| slot.get())
                    .collect::<Vec<_>>()
            )));
        }
        parsed.push((*id, props));
    }
    if parsed.is_empty() {
        return Err(CliError::runtime("physical graph has no node props"));
    }
    Ok(parsed)
}

pub(super) fn nearest_graph_node(
    props: &[(CxId, AsterAssocNodeProps)],
    query: &PanelVectors,
    embedding_slots: &[SlotId],
) -> CliResult<PanelFusionHit> {
    let candidates = props
        .iter()
        .map(|(id, props)| (*id, &props.embeddings))
        .collect::<BTreeMap<_, _>>();
    rank_panel_candidate_refs(query, &candidates, embedding_slots, PANEL_RRF_K)?
        .into_iter()
        .next()
        .ok_or_else(|| CliError::runtime("physical graph has no nearest constellation"))
}

pub(super) fn query_panel_vectors(
    vectors: &[(SlotId, SlotVector)],
    embedding_slots: &[SlotId],
) -> CliResult<PanelVectors> {
    let panel = vectors
        .iter()
        .filter(|(slot, _)| embedding_slots.contains(slot))
        .map(|(slot, vector)| {
            let dense = vector.as_dense().map(ToOwned::to_owned).ok_or_else(|| {
                CliError::runtime(format!(
                    "kernel graph slot {slot} measured a non-dense vector"
                ))
            })?;
            Ok((*slot, dense))
        })
        .collect::<CliResult<PanelVectors>>()?;
    if panel.keys().copied().ne(embedding_slots.iter().copied()) {
        return Err(CliError::runtime(format!(
            "kernel query measured slots {:?}, expected {:?}",
            panel.keys().map(|slot| slot.get()).collect::<Vec<_>>(),
            embedding_slots
                .iter()
                .map(|slot| slot.get())
                .collect::<Vec<_>>()
        )));
    }
    Ok(panel)
}

fn kernel_answer_id(
    vault_id: &[u8],
    query_sha256: &[u8; 32],
    manifest_sha256: &[u8; 32],
    anchor: Option<&str>,
    max_hops: usize,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"calyx-kernel-answer-v2-panel");
    hasher.update(&(vault_id.len() as u64).to_be_bytes());
    hasher.update(vault_id);
    hasher.update(query_sha256);
    hasher.update(manifest_sha256);
    hasher.update(anchor.unwrap_or("any").as_bytes());
    hasher.update(&(max_hops as u64).to_be_bytes());
    *hasher.finalize().as_bytes()
}

pub(super) fn retain_kernel_query(vault: &Path, sha256: &str, bytes: &[u8]) -> CliResult<String> {
    let relative = format!("queries/{sha256}.txt");
    let path = vault.join("inputs").join(&relative);
    if path.exists() {
        let existing = fs::read(&path).map_err(|error| {
            CliError::io(format!(
                "read retained kernel query {}: {error}",
                path.display()
            ))
        })?;
        if existing != bytes {
            return Err(CliError::runtime(format!(
                "retained query path {} contains bytes that do not match its digest",
                path.display()
            )));
        }
    } else {
        write_bytes_atomic(&path, bytes, "retained kernel query")?;
    }
    let readback = fs::read(&path).map_err(|error| {
        CliError::io(format!(
            "read back retained kernel query {}: {error}",
            path.display()
        ))
    })?;
    if readback != bytes || sha256_bytes(&readback) != sha256 {
        return Err(CliError::runtime(format!(
            "retained kernel query physical readback mismatch at {}",
            path.display()
        )));
    }
    Ok(format!("calyx-vault://inputs/{relative}"))
}

fn verify_complete_answer_readback(
    vault: &Path,
    reference: &calyx_core::LedgerRef,
    context: &KernelAnswerRecordContext,
    query_cx: CxId,
) -> CliResult {
    let store = AsterLedgerCfStore::open(vault)?;
    let row = store
        .read_seq(reference.seq)?
        .ok_or_else(|| CliError::runtime("complete kernel Answer row is physically absent"))?;
    let entry = decode(&row.bytes)?;
    let payload: serde_json::Value = serde_json::from_slice(&entry.payload).map_err(|error| {
        CliError::runtime(format!("decode complete kernel Answer payload: {error}"))
    })?;
    let answer_id: &[u8; 32] = context
        .answer_id
        .as_slice()
        .try_into()
        .map_err(|_| CliError::runtime("kernel answer id is not exactly 32 bytes"))?;
    let expected_answer_id = hex32(answer_id);
    let expected_query_id = query_cx.to_string();
    if entry.kind != EntryKind::Answer
        || entry.entry_hash != reference.hash
        || !matches!(&entry.subject, SubjectId::Query(id) if id == &context.answer_id)
        || payload.get("type").and_then(|value| value.as_str()) != Some("kernel_answer_v3")
        || payload.get("answer_id").and_then(|value| value.as_str())
            != Some(expected_answer_id.as_str())
        || payload.get("query_id").and_then(|value| value.as_str())
            != Some(expected_query_id.as_str())
        || payload.get("source_support")
            != Some(
                &serde_json::to_value(&context.source_support).map_err(|error| {
                    CliError::runtime(format!("encode expected source support: {error}"))
                })?,
            )
    {
        return Err(CliError::runtime(
            "complete kernel Answer physical readback differs from the issued answer",
        ));
    }
    Ok(())
}

fn ledger_ref_out(reference: &calyx_core::LedgerRef) -> output::GroundedKernelLedgerRefOut {
    output::GroundedKernelLedgerRefOut {
        seq: reference.seq,
        hash: hex32(&reference.hash),
    }
}
