use std::collections::BTreeMap;
use std::path::Path;

use calyx_aster::cf::ColumnFamily;
use calyx_aster::plain_graph::{
    GraphCollectionGenerationState, GraphCollectionGenerationStatus, GraphCollectionLifecycle,
    PhysicalGraphCollectionLifecycle, PhysicalPlainGraph, PlainGraph,
};
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{AnchorKind, CalyxError, Clock, CxId, VaultStore};
use calyx_lodestar::{AsterAssocNodeProps, encode_assoc_node_props};
use serde::Serialize;
use serde_json::{Value, json};

use super::{
    COURT_EDGE, Counts, Draft, MaterializeArgs, OPERATOR_EDGE, SCHEMA, SummaryNode, TARGET_NODE,
    source_contract, write_report,
};
use crate::cmd::citation_overlay::write::{
    build_csr_projection, read_back_edges, read_back_nodes, sha256_hex,
};
use crate::cmd::vault::{ResolvedVault, resolve_vault_info, vault_salt};
use crate::error::{CliError, CliResult};

const GRAPH_ROWS_PER_BATCH: usize = 10_000;

#[derive(Debug, Serialize)]
pub(super) struct Report {
    status: &'static str,
    source_of_truth: &'static str,
    vault: String,
    vault_id: String,
    alias_vault: String,
    collection: String,
    graph_generation: String,
    schema: &'static str,
    source_inputs: Vec<Value>,
    counts: Counts,
    coverage: Vec<super::CoverageRow>,
    readback: Readback,
    doctrine: &'static str,
}

#[derive(Debug, Serialize)]
struct Readback {
    physical_nodes: usize,
    physical_edge_out: usize,
    csr_nodes: usize,
    csr_edges: usize,
    assoc_nodes: usize,
    assoc_edges: usize,
    csr_bytes: usize,
    csr_sha256: String,
    all_node_values_read_back: bool,
    all_edge_values_read_back: bool,
    lifecycle_accepted: bool,
    graph_wal_batches: usize,
}

pub(super) fn write_to_calyx(
    home: &Path,
    args: &MaterializeArgs,
    draft: Draft,
) -> CliResult<Report> {
    let resolved = resolve_vault_info(home, &args.vault)?;
    let vault = AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        VaultOptions {
            restore_mvcc_rows: false,
            ..VaultOptions::default()
        },
    )?;
    let graph = PlainGraph::new(&vault, &args.collection)?;
    let generation = format!("materialize-summary-attribution-{}", ulid::Ulid::new());
    let lifecycle = GraphCollectionLifecycle::new(&vault)?;
    lifecycle.put_state(
        &GraphCollectionGenerationState::new(
            args.collection.clone(),
            generation.clone(),
            GraphCollectionGenerationStatus::Writing,
            "materialize-summary-attribution",
        )
        .with_reason("typed summary attribution physical write started")
        .with_detail("schema", SCHEMA),
    )?;
    let mut node_values = BTreeMap::<CxId, Vec<u8>>::new();
    let mut edge_values = Vec::<(CxId, String, CxId, Vec<u8>)>::new();
    let mut batch =
        Vec::with_capacity(draft.targets.len() + draft.summary_nodes.len() + 2 * draft.edges.len());
    for target in draft.targets.values() {
        let node = SummaryNode {
            id: target.id,
            node_type: TARGET_NODE,
            metadata: BTreeMap::from([
                ("physical_cx_id".to_string(), target.id.to_string()),
                (
                    "physical_input_sha256".to_string(),
                    target.input_sha256.clone(),
                ),
                (
                    "physical_input_pointer".to_string(),
                    target.input_pointer.clone(),
                ),
                (
                    "canonical_opinion_id".to_string(),
                    target.canonical_opinion_id.clone(),
                ),
                (
                    "opinion_ids".to_string(),
                    target
                        .opinion_ids
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("|"),
                ),
            ]),
        };
        write_node(&graph, &node, &mut node_values, &mut batch)?;
    }
    for node in draft.summary_nodes.values() {
        write_node(&graph, node, &mut node_values, &mut batch)?;
    }
    for edge in draft.edges.values() {
        let value = serde_json::to_vec(&edge.value)
            .map_err(|error| CliError::runtime(format!("serialize summary edge: {error}")))?;
        let out = graph.edge_out_key(edge.src, edge.edge_type, edge.dst)?;
        let incoming = graph.edge_in_key(edge.dst, edge.edge_type, edge.src)?;
        batch.push((ColumnFamily::Graph, out.clone(), value.clone()));
        batch.push((ColumnFamily::Graph, incoming, out));
        edge_values.push((edge.src, edge.edge_type.to_string(), edge.dst, value));
    }
    let graph_wal_batches = batch.len().div_ceil(GRAPH_ROWS_PER_BATCH);
    for rows in batch.chunks(GRAPH_ROWS_PER_BATCH) {
        vault.write_cf_batch(rows.to_vec())?;
    }
    let inputs = vec![
        source_contract("requested_cx_set", &args.cx_set)?,
        source_contract("opinion_aliases", &args.aliases)?,
        source_contract("court_parentheticals", &args.parentheticals)?,
        source_contract("operator_summaries", &args.operator_summaries)?,
    ];
    let metadata = serde_json::to_vec(&json!({
        "schema": SCHEMA,
        "collection": args.collection,
        "target_vault": args.vault,
        "alias_authority_vault": args.alias_vault,
        "edge_types": [COURT_EDGE, OPERATOR_EDGE],
        "source_inputs": inputs,
        "counts": draft.counts,
        "doctrine": "court-authored and operator-authored text remain separately typed",
    }))
    .map_err(|error| CliError::runtime(format!("serialize summary metadata: {error}")))?;
    graph.put_metadata("summary_attribution", &metadata)?;
    let projection = build_csr_projection(
        &args.collection,
        vault.snapshot(),
        &node_values,
        &edge_values,
    )?;
    graph.write_csr_projection(projection)?;
    vault.flush()?;
    drop(graph);
    drop(vault);

    let physical = PhysicalPlainGraph::open_latest_unchecked(&resolved.path, &args.collection)?;
    read_back_nodes(&physical, &node_values)?;
    let physical_edges = read_back_edges(&physical, &edge_values)?;
    let raw = physical.read_csr_bytes()?.ok_or_else(|| {
        contract_error(
            "CALYX_SUMMARY_CSR_MISSING",
            "summary attribution CSR bytes are missing",
            "rebuild and physically reopen the summary attribution generation",
        )
    })?;
    let csr = physical.read_csr()?.ok_or_else(|| {
        contract_error(
            "CALYX_SUMMARY_CSR_MISSING",
            "summary attribution CSR failed to decode",
            "rebuild and physically reopen the summary attribution generation",
        )
    })?;
    let assoc = physical.assoc_graph()?;
    if physical_edges != draft.edges.len()
        || csr.nodes.len() != node_values.len()
        || csr.edges.len() != draft.edges.len()
        || assoc.node_count() != node_values.len()
        || assoc.edge_count() != draft.edges.len()
    {
        return Err(contract_error(
            "CALYX_SUMMARY_READBACK_MISMATCH",
            format!(
                "expected nodes={} edges={}; physical edges={physical_edges}, csr nodes={} edges={}, assoc nodes={} edges={}",
                node_values.len(),
                draft.edges.len(),
                csr.nodes.len(),
                csr.edges.len(),
                assoc.node_count(),
                assoc.edge_count()
            ),
            "do not accept until Graph CF, CSR, and association counts agree",
        ));
    }
    let readback = Readback {
        physical_nodes: node_values.len(),
        physical_edge_out: physical_edges,
        csr_nodes: csr.nodes.len(),
        csr_edges: csr.edges.len(),
        assoc_nodes: assoc.node_count(),
        assoc_edges: assoc.edge_count(),
        csr_bytes: raw.len(),
        csr_sha256: sha256_hex(&raw),
        all_node_values_read_back: true,
        all_edge_values_read_back: true,
        lifecycle_accepted: false,
        graph_wal_batches,
    };
    let mut report = Report {
        status: "complete",
        source_of_truth: "physical Base rows plus exact Graph CF node/edge/CSR/lifecycle bytes",
        vault: resolved.name.clone(),
        vault_id: resolved.vault_id.to_string(),
        alias_vault: args.alias_vault.clone(),
        collection: args.collection.clone(),
        graph_generation: generation.clone(),
        schema: SCHEMA,
        source_inputs: inputs,
        counts: draft.counts,
        coverage: draft.coverage,
        readback,
        doctrine: "court-authored, operator-authored, and missing remain distinct; no constellation lens is flattened or changed",
    };
    accept_generation(&resolved, &args.collection, &generation, &report)?;
    report.readback.lifecycle_accepted = true;
    write_report(&args.report, &report, "summary attribution report")?;
    Ok(report)
}

fn write_node<C: Clock>(
    graph: &PlainGraph<'_, C>,
    node: &SummaryNode,
    node_values: &mut BTreeMap<CxId, Vec<u8>>,
    batch: &mut Vec<(ColumnFamily, Vec<u8>, Vec<u8>)>,
) -> CliResult {
    let mut metadata = node.metadata.clone();
    metadata.insert("node_type".to_string(), node.node_type.to_string());
    metadata.insert("schema".to_string(), SCHEMA.to_string());
    let value = encode_assoc_node_props(&AsterAssocNodeProps {
        anchors: vec![AnchorKind::Label(format!(
            "legal_summary_attribution:{}",
            node.node_type
        ))],
        metadata,
        ..Default::default()
    })?;
    if node_values.insert(node.id, value.clone()).is_some() {
        return Err(contract_error(
            "CALYX_SUMMARY_NODE_COLLISION",
            format!("node {} collides with another typed node", node.id),
            "repair source identity before writing summary attribution",
        ));
    }
    batch.push((ColumnFamily::Graph, graph.node_key(node.id), value));
    Ok(())
}

fn accept_generation(
    resolved: &ResolvedVault,
    collection: &str,
    generation: &str,
    report: &Report,
) -> CliResult {
    let vault = AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        VaultOptions {
            restore_mvcc_rows: false,
            ..VaultOptions::default()
        },
    )?;
    let lifecycle = GraphCollectionLifecycle::new(&vault)?;
    lifecycle.put_state(
        &GraphCollectionGenerationState::new(
            collection.to_string(),
            generation.to_string(),
            GraphCollectionGenerationStatus::Accepted,
            "materialize-summary-attribution",
        )
        .with_reason("summary attribution Graph CF and CSR physical readback passed")
        .with_detail("schema", SCHEMA)
        .with_detail("nodes", report.readback.physical_nodes.to_string())
        .with_detail("edges", report.readback.physical_edge_out.to_string())
        .with_detail(
            "court_targets",
            report.counts.court_authored_targets.to_string(),
        )
        .with_detail(
            "operator_targets",
            report.counts.operator_authored_targets.to_string(),
        )
        .with_detail("missing_targets", report.counts.missing_targets.to_string())
        .with_detail("csr_sha256", report.readback.csr_sha256.clone()),
    )?;
    vault.flush()?;
    drop(vault);
    let lifecycle = PhysicalGraphCollectionLifecycle::open_latest(&resolved.path)?;
    if !lifecycle.list_states()?.into_iter().any(|row| {
        row.state.collection == collection
            && row.state.generation == generation
            && row.state.status == GraphCollectionGenerationStatus::Accepted
    }) {
        return Err(CliError::runtime(
            "accepted summary attribution lifecycle row is absent after physical reopen",
        ));
    }
    Ok(())
}

fn contract_error(
    code: &'static str,
    message: impl Into<String>,
    remediation: &'static str,
) -> CliError {
    CliError::from(CalyxError {
        code,
        message: message.into(),
        remediation,
    })
}
