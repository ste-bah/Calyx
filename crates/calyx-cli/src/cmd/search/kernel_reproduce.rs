//! Re-derive a recorded kernel answer from its retained query and immutable generation.

use std::collections::BTreeSet;
use std::fs;
use std::net::SocketAddr;
use std::path::{Component, Path};

use calyx_aster::plain_graph::PhysicalPlainGraph;
use calyx_core::{CalyxError, CxId, SlotId};
use calyx_lodestar::{
    ASTER_ASSOC_METADATA_KEY, FsKernelStore, KernelAnswerRecordContext, KernelAnswerSourceSupport,
    PANEL_ASTER_ASSOC_COLLECTION, PanelFusionLane, RecallPassMode, derive_panel_kernel_answer,
    kernel_answer_derivation_hash, kernel_answer_derivation_hash_v2_legacy, kernel_health,
    load_panel_kernel_index, panel_kernel_search, read_kernel_artifact,
};
use calyx_registry::{load_vault_panel_state, require_vault_registry_contracts};

use super::super::ingest::parse_anchor_kind;
use super::super::kernel_generation::{
    decode_sha256, hex32, load_generation_by_sha256, physical_graph_contract, sha256_bytes,
};
use super::super::vault::ResolvedVault;
use super::kernel_answer::{
    measure_kernel_query_vectors, nearest_graph_node, parse_graph_props, query_panel_vectors,
};
use super::kernel_source_support::evaluate_path_source_support;
use super::roster::SearchTextRoster;
use crate::error::{CliError, CliResult};

pub(crate) fn rederive_kernel_answer_hash(
    resolved: &ResolvedVault,
    answer_id: &[u8],
    payload: &serde_json::Value,
    resident_override: Option<SocketAddr>,
) -> CliResult<[u8; 32]> {
    let payload_type = payload.get("type").and_then(|value| value.as_str());
    if !matches!(payload_type, Some("kernel_answer_v2" | "kernel_answer_v3")) {
        return Err(CliError::runtime(
            "kernel reproduce received a non-kernel Answer payload",
        ));
    }
    let recorded_answer_id =
        decode_sha256(required_string(payload, "answer_id")?, "kernel answer_id")?;
    if recorded_answer_id.as_slice() != answer_id {
        return Err(CliError::runtime(
            "kernel Answer payload id differs from its ledger subject",
        ));
    }
    let query_hash = decode_sha256(
        required_string(payload, "query_input_sha256")?,
        "query_input_sha256",
    )?;
    let manifest_sha256 = required_string(payload, "kernel_manifest_sha256")?;
    let manifest_hash = decode_sha256(manifest_sha256, "kernel_manifest_sha256")?;
    let loaded = load_generation_by_sha256(&resolved.path, manifest_sha256)?;
    let graph_contract = &loaded.manifest.graph;
    let admission = &loaded.manifest.admission;
    let embedding_slots = required_slots(payload, "embedding_slots")?;
    if embedding_slots != graph_contract.embedding_slots
        || required_string(payload, "fusion")? != graph_contract.fusion
        || required_u64(payload, "rrf_k")? != u64::from(graph_contract.rrf_k)
    {
        return Err(CliError::runtime(
            "kernel Answer panel fusion contract differs from its immutable manifest",
        ));
    }
    let query_pointer = format!("calyx-vault://inputs/queries/{}.txt", hex32(&query_hash));
    let query_path = retained_kernel_query_path(&resolved.path, &query_pointer)?;
    let query = fs::read(&query_path).map_err(|error| {
        CliError::io(format!(
            "read retained kernel query {}: {error}",
            query_path.display()
        ))
    })?;
    if sha256_bytes(&query) != hex32(&query_hash) {
        return Err(CliError::runtime(format!(
            "retained kernel query {} fails its recorded SHA-256",
            query_path.display()
        )));
    }
    let query_text = std::str::from_utf8(&query).map_err(|error| {
        CliError::runtime(format!("retained kernel query is not UTF-8: {error}"))
    })?;
    if let Some(jurisdiction) = &loaded.manifest.jurisdiction
        && let Some(conflict) =
            super::super::kernel_scope::explicit_scope_conflict(query_text, jurisdiction)
    {
        return Err(CalyxError::reproduce_drift_exceeded(format!(
            "retained kernel query now conflicts with immutable scope: {}={}",
            conflict.kind, conflict.detected
        ))
        .into());
    }
    let saved_resident = required_string(payload, "resident_addr")?;
    let resident_addr = match resident_override {
        Some(addr) => Some(addr),
        None if saved_resident == "local-cpu" => None,
        None => Some(super::parse::parse_resident_addr(saved_resident)?),
    };
    require_vault_registry_contracts(&resolved.path)?;
    let state = load_vault_panel_state(&resolved.path)?;
    let roster = SearchTextRoster::derive(&state);
    let vectors =
        measure_kernel_query_vectors(&state, &roster, resolved, query_text, resident_addr)?;
    let query_panel = query_panel_vectors(&vectors, &embedding_slots)?;
    let plain = PhysicalPlainGraph::open_latest(&resolved.path, PANEL_ASTER_ASSOC_COLLECTION)?;
    let metadata_bytes = plain
        .get_metadata(ASTER_ASSOC_METADATA_KEY)?
        .ok_or_else(|| CliError::runtime("physical graph metadata disappeared"))?;
    let node_rows = plain.node_props()?;
    let csr_bytes = plain
        .read_csr_bytes()?
        .ok_or_else(|| CliError::runtime("physical graph CSR disappeared"))?;
    let graph = plain.assoc_graph()?;
    let (physical_hash, node_props_hash) =
        physical_graph_contract(&metadata_bytes, &node_rows, &csr_bytes);
    if graph.node_count() != graph_contract.nodes
        || graph.edge_count() != graph_contract.edges
        || sha256_bytes(&metadata_bytes) != graph_contract.metadata_sha256
        || node_props_hash != graph_contract.node_props_sha256
        || sha256_bytes(&csr_bytes) != graph_contract.csr_sha256
        || hex32(&physical_hash) != graph_contract.physical_contract_sha256
    {
        return Err(CliError::runtime(
            "kernel reproduce physical graph differs from the immutable answer manifest",
        ));
    }
    let store = FsKernelStore::new(&resolved.path);
    let kernel = read_kernel_artifact(loaded.manifest.kernel_id, &store)?;
    let index = load_panel_kernel_index(loaded.manifest.kernel_id, &store)?;
    let health = kernel_health(loaded.manifest.kernel_id, &store)?;
    if health.recall.pass_mode != RecallPassMode::Passed || health.grounded_fraction != 1.0 {
        return Err(CalyxError::kernel_ungrounded(
            "kernel reproduce refuses an ungrounded or below-gate immutable kernel",
        )
        .into());
    }
    let parsed_props = parse_graph_props(&node_rows, &embedding_slots)?;
    let nearest = nearest_graph_node(&parsed_props, &query_panel, &embedding_slots)?;
    let query_cx = nearest.cx_id;
    let recorded_query_cx = required_string(payload, "query_id")?
        .parse::<CxId>()
        .map_err(|error| CliError::runtime(format!("parse kernel Answer query_id: {error}")))?;
    let recorded_score = required_f32(payload, "nearest_score")?;
    let recorded_lanes: Vec<PanelFusionLane> = serde_json::from_value(
        payload
            .get("nearest_lanes")
            .cloned()
            .ok_or_else(|| CliError::runtime("kernel Answer nearest_lanes is missing"))?,
    )
    .map_err(|error| CliError::runtime(format!("decode kernel Answer nearest_lanes: {error}")))?;
    let admission_threshold = required_f32(payload, "admission_threshold")?;
    if query_cx != recorded_query_cx
        || nearest.score.to_bits() != recorded_score.to_bits()
        || nearest.lanes != recorded_lanes
        || admission_threshold.to_bits() != admission.threshold.to_bits()
        || nearest.score < admission_threshold
    {
        return Err(CalyxError::reproduce_drift_exceeded(format!(
            "kernel query admission drift: target {}->{query_cx} score {}->{} threshold {}->{}",
            recorded_query_cx,
            recorded_score,
            nearest.score,
            admission_threshold,
            admission.threshold
        ))
        .into());
    }
    let anchor_raw = payload
        .get("anchor")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let anchor = anchor_raw.as_deref().map(parse_anchor_kind).transpose()?;
    let max_hops: usize = required_u64(payload, "max_hops")?
        .try_into()
        .map_err(|_| CliError::runtime("kernel Answer max_hops exceeds usize"))?;
    let kernel_members = kernel.members.iter().copied().collect::<BTreeSet<_>>();
    let anchored = parsed_props
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
    let ranked_kernel_nodes = panel_kernel_search(&index, &query_panel, index.rows().len())?
        .into_iter()
        .map(|hit| hit.cx_id)
        .collect::<Vec<_>>();
    let derivation = derive_panel_kernel_answer(
        kernel.kernel_id,
        &ranked_kernel_nodes,
        &graph,
        query_cx,
        &anchored,
        max_hops,
    )?;
    let source_support = if payload_type == Some("kernel_answer_v3") {
        let recorded: KernelAnswerSourceSupport = serde_json::from_value(
            payload
                .get("source_support")
                .cloned()
                .ok_or_else(|| CliError::runtime("kernel Answer source_support is missing"))?,
        )
        .map_err(|error| {
            CliError::runtime(format!("decode kernel Answer source_support: {error}"))
        })?;
        let reproduced = evaluate_path_source_support(resolved, query_text, &derivation)?;
        if reproduced != recorded || reproduced.verdict != "supported" {
            return Err(CalyxError::reproduce_drift_exceeded(
                "kernel Answer retained source-support proof differs from its ledger payload",
            )
            .into());
        }
        reproduced
    } else {
        legacy_unverified_source_support()
    };
    let context = KernelAnswerRecordContext {
        answer_id: answer_id.to_vec(),
        query_input_sha256: query_hash,
        query_input_pointer: query_pointer,
        kernel_manifest_sha256: manifest_hash,
        embedding_slots: embedding_slots.clone(),
        fusion: graph_contract.fusion.clone(),
        rrf_k: graph_contract.rrf_k,
        nearest_score: nearest.score,
        nearest_lanes: nearest.lanes,
        admission_threshold,
        resident_addr: saved_resident.to_string(),
        anchor: anchor_raw,
        max_hops,
        source_support,
    };
    if payload_type == Some("kernel_answer_v2") {
        kernel_answer_derivation_hash_v2_legacy(&derivation, &context).map_err(Into::into)
    } else {
        kernel_answer_derivation_hash(&derivation, &context).map_err(Into::into)
    }
}

fn legacy_unverified_source_support() -> KernelAnswerSourceSupport {
    KernelAnswerSourceSupport {
        schema_version: 0,
        method: "legacy_v2_unverified".to_string(),
        verdict: "unverified".to_string(),
        query_terms: Vec::new(),
        matched_terms: Vec::new(),
        missing_terms: Vec::new(),
        matched_term_pairs: Vec::new(),
        matched_weight: 0,
        total_weight: 0,
        weighted_coverage_bps: 0,
        minimum_weighted_coverage_bps: 0,
        sources: Vec::new(),
    }
}

fn retained_kernel_query_path(vault: &Path, pointer: &str) -> CliResult<std::path::PathBuf> {
    let relative = pointer
        .strip_prefix("calyx-vault://inputs/")
        .ok_or_else(|| {
            CliError::runtime("kernel Answer query pointer has an unsupported scheme")
        })?;
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(CliError::runtime(
            "kernel Answer query pointer is not a strict relative input path",
        ));
    }
    Ok(vault.join("inputs").join(path))
}

fn required_string<'a>(payload: &'a serde_json::Value, field: &str) -> CliResult<&'a str> {
    payload
        .get(field)
        .and_then(|value| value.as_str())
        .ok_or_else(|| CliError::runtime(format!("kernel Answer payload field {field} is missing")))
}

fn required_u64(payload: &serde_json::Value, field: &str) -> CliResult<u64> {
    payload
        .get(field)
        .and_then(|value| value.as_u64())
        .ok_or_else(|| CliError::runtime(format!("kernel Answer payload field {field} is not u64")))
}

fn required_slots(payload: &serde_json::Value, field: &str) -> CliResult<Vec<SlotId>> {
    let values = payload
        .get(field)
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            CliError::runtime(format!(
                "kernel Answer payload field {field} is not an array"
            ))
        })?;
    let slots = values
        .iter()
        .map(|value| {
            let raw = value.as_u64().ok_or_else(|| {
                CliError::runtime(format!(
                    "kernel Answer payload field {field} contains a non-u64"
                ))
            })?;
            let raw = u16::try_from(raw).map_err(|_| {
                CliError::runtime(format!("kernel Answer payload field {field} exceeds u16"))
            })?;
            Ok(SlotId::new(raw))
        })
        .collect::<CliResult<Vec<_>>>()?;
    if slots.len() < 2 || !slots.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(CliError::runtime(format!(
            "kernel Answer payload field {field} is not a strict ordered panel"
        )));
    }
    Ok(slots)
}

fn required_f32(payload: &serde_json::Value, field: &str) -> CliResult<f32> {
    let value = payload
        .get(field)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| {
            CliError::runtime(format!(
                "kernel Answer payload field {field} is not numeric"
            ))
        })? as f32;
    if !value.is_finite() {
        return Err(CliError::runtime(format!(
            "kernel Answer payload field {field} is non-finite"
        )));
    }
    Ok(value)
}
