use std::collections::{BTreeMap, HashMap, HashSet};

use calyx_core::{Constellation, CxId, SlotId};
use calyx_lodestar::LodestarError;
use serde::Serialize;

const EXAMPLE_MISSING_LIMIT: usize = 5;
const COVERED_SCAN_BATCH_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum CandidateSelectionMode {
    Covered,
    BasePrefix,
}

impl CandidateSelectionMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Covered => "covered",
            Self::BasePrefix => "base-prefix",
        }
    }
}

/// Exact per-slot coverage accounting. The counters are additive by
/// construction: `dense + non_dense + absent + tombstoned + missing ==
/// candidate_rows`. `missing_rows` counts only candidates whose base row does
/// not list the slot — a base-listed slot with no physical row fails the scan
/// closed instead of being counted here (issue #1096).
#[derive(Clone, Debug, Serialize)]
pub(super) struct DenseSlotCoverage {
    pub slot_id: u16,
    pub candidate_rows: usize,
    pub dense_rows: usize,
    pub missing_rows: usize,
    pub non_dense_rows: usize,
    pub absent_rows: usize,
    pub tombstoned_rows: usize,
    pub example_missing_cx_ids: Vec<String>,
    pub read_stats: crate::provenance_read::ProvenanceReadStats,
}

pub(super) struct DenseSlotCoverageScan {
    pub constellations_in_vault: usize,
    pub scanned_candidates: Vec<Constellation>,
    pub slot_maps: BTreeMap<SlotId, HashMap<CxId, Vec<f32>>>,
    pub coverage: Vec<DenseSlotCoverage>,
    pub base_page_index_live_entries: usize,
    pub candidate_scan_rows: usize,
    pub candidate_scan_complete: bool,
}

pub(super) struct DenseSlotPreflight {
    pub constellations_in_vault: usize,
    pub candidates: Vec<Constellation>,
    pub slot_maps: BTreeMap<SlotId, HashMap<CxId, Vec<f32>>>,
    pub coverage: Vec<DenseSlotCoverage>,
    pub base_page_index_live_entries: usize,
    pub candidate_scan_rows: usize,
    pub candidate_scan_complete: bool,
    pub selected_candidate_rows: usize,
    pub selected_candidate_cx_ids: Vec<String>,
}

mod scan;
pub(super) use scan::scan_dense_slot_coverage;
/// Materialize only complete constellations: a selected node must have a
/// physical dense vector in every contracted panel slot. This is an
/// intersection, never the coverage set of one preferred lens.
pub(super) fn materialize_panel_preflight(
    scan: DenseSlotCoverageScan,
    slots: &[SlotId],
    limit: usize,
) -> Result<DenseSlotPreflight, String> {
    if slots.len() < 2 {
        return Err("CALYX_WEAVE_LOOM_PANEL_TOO_SMALL: need at least two dense slots".to_string());
    }
    if scan.slot_maps.keys().copied().ne(slots.iter().copied()) {
        return Err(format!(
            "CALYX_WEAVE_LOOM_PANEL_NOT_SCANNED: scanned slots {:?} differ from contracted slots {:?}",
            scan.slot_maps
                .keys()
                .map(|slot| slot.get())
                .collect::<Vec<_>>(),
            slots.iter().map(|slot| slot.get()).collect::<Vec<_>>()
        ));
    }
    let target = if limit == 0 { usize::MAX } else { limit };
    let selected_ids = scan
        .scanned_candidates
        .iter()
        .filter(|cx| {
            slots.iter().all(|slot| {
                scan.slot_maps
                    .get(slot)
                    .is_some_and(|rows| rows.contains_key(&cx.cx_id))
            })
        })
        .take(target)
        .map(|cx| cx.cx_id)
        .collect::<Vec<_>>();
    if selected_ids.len() < 2 {
        return Err(format!(
            "CALYX_WEAVE_LOOM_COMPLETE_PANEL_EMPTY: only {} constellations have all {} contracted dense slots; coverage={}",
            selected_ids.len(),
            slots.len(),
            coverage_summary(&scan.coverage)
        ));
    }
    let selected_set = selected_ids.iter().copied().collect::<HashSet<_>>();
    let candidates = scan
        .scanned_candidates
        .into_iter()
        .filter(|cx| selected_set.contains(&cx.cx_id))
        .collect::<Vec<_>>();
    let slot_maps = scan
        .slot_maps
        .into_iter()
        .map(|(slot, rows)| {
            let rows = rows
                .into_iter()
                .filter(|(id, _)| selected_set.contains(id))
                .collect();
            (slot, rows)
        })
        .collect();
    Ok(DenseSlotPreflight {
        constellations_in_vault: scan.constellations_in_vault,
        selected_candidate_rows: candidates.len(),
        candidate_scan_rows: scan.candidate_scan_rows,
        candidate_scan_complete: scan.candidate_scan_complete,
        candidates,
        slot_maps,
        coverage: scan.coverage,
        base_page_index_live_entries: scan.base_page_index_live_entries,
        selected_candidate_cx_ids: selected_ids.iter().map(ToString::to_string).collect(),
    })
}

pub(super) fn coverage_summary(coverage: &[DenseSlotCoverage]) -> String {
    coverage
        .iter()
        .map(|row| {
            format!(
                "slot {} dense={}/{} missing={} non_dense={} absent={} tombstoned={} examples={:?}",
                row.slot_id,
                row.dense_rows,
                row.candidate_rows,
                row.missing_rows,
                row.non_dense_rows,
                row.absent_rows,
                row.tombstoned_rows,
                row.example_missing_cx_ids
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub(super) fn invalid_params(detail: impl Into<String>) -> crate::error::CliError {
    LodestarError::KernelInvalidParams {
        detail: detail.into(),
    }
    .into()
}
