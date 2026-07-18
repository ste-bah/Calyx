use std::collections::{BTreeMap, BTreeSet};

use calyx_aster::plain_graph::{
    PhysicalPlainGraph, PlainGraphCsr, PlainGraphCsrEdge, plain_graph_edge_raw_weight,
    plain_graph_normalized_edge_weight,
};
use calyx_core::CxId;

use super::super::{EDGE_TYPE, contract_error};
use crate::error::{CliError, CliResult};

pub(super) fn build_csr(
    collection: &str,
    snapshot: u64,
    node_values: &BTreeMap<CxId, Vec<u8>>,
    edge_values: &[(CxId, CxId, Vec<u8>)],
) -> CliResult<PlainGraphCsr> {
    let nodes = node_values.keys().copied().collect::<Vec<_>>();
    let node_index = nodes
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<BTreeMap<_, _>>();
    let mut max_raw_weight = 0.0_f32;
    let mut drafts = Vec::with_capacity(edge_values.len());
    let mut association_edges = BTreeSet::new();
    for (src, dst, value) in edge_values {
        let src_index = *node_index
            .get(src)
            .ok_or_else(|| CliError::runtime("alias CSR source node is absent"))?;
        if !node_index.contains_key(dst) {
            return Err(CliError::runtime("alias CSR target node is absent"));
        }
        let raw = plain_graph_edge_raw_weight(value)?;
        max_raw_weight = max_raw_weight.max(raw);
        drafts.push((src_index, *dst, raw));
        association_edges.insert((*src, *dst));
    }
    let mut by_source = vec![Vec::<PlainGraphCsrEdge>::new(); nodes.len()];
    for (src_index, dst, raw) in drafts {
        by_source[src_index].push(PlainGraphCsrEdge {
            dst,
            edge_type: EDGE_TYPE.to_string(),
            weight: plain_graph_normalized_edge_weight(raw, max_raw_weight)?,
        });
    }
    let mut offsets = Vec::with_capacity(nodes.len() + 1);
    let mut edges = Vec::with_capacity(edge_values.len());
    offsets.push(0);
    for mut list in by_source {
        list.sort_by_key(|edge| edge.dst);
        edges.extend(list);
        offsets.push(edges.len());
    }
    Ok(PlainGraphCsr {
        collection: collection.to_string(),
        source_snapshot: snapshot,
        nodes,
        offsets,
        edges,
        association_edge_count: association_edges.len(),
    })
}

pub(super) fn verify_nodes(
    physical: &PhysicalPlainGraph,
    expected: &BTreeMap<CxId, Vec<u8>>,
) -> CliResult {
    let actual = physical
        .node_props()?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    if actual != *expected {
        return Err(contract_error(
            "CALYX_OPINION_ALIAS_NODE_READBACK_MISMATCH",
            format!(
                "physical node map has {} rows; expected {}",
                actual.len(),
                expected.len()
            ),
            "quarantine the collection and inspect Graph CF node persistence",
        ));
    }
    Ok(())
}

pub(super) fn verify_edges(
    physical: &PhysicalPlainGraph,
    expected: &[(CxId, CxId, Vec<u8>)],
) -> CliResult {
    let expected = expected
        .iter()
        .map(|(src, dst, value)| ((*src, EDGE_TYPE.to_string(), *dst), value.clone()))
        .collect::<BTreeMap<_, _>>();
    let actual = physical
        .edge_out_props()?
        .into_iter()
        .map(|edge| ((edge.src, edge.edge_type, edge.dst), edge.value))
        .collect::<BTreeMap<_, _>>();
    if actual != expected {
        return Err(contract_error(
            "CALYX_OPINION_ALIAS_EDGE_READBACK_MISMATCH",
            format!(
                "physical edge map has {} rows; expected {}",
                actual.len(),
                expected.len()
            ),
            "quarantine the collection and inspect Graph CF edge/reverse persistence",
        ));
    }
    Ok(())
}
