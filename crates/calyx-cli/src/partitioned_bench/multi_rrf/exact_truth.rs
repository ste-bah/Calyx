use std::collections::BTreeMap;

use calyx_core::SlotId;
use calyx_sextant::index::CuvsChunkedExactReport;
use serde_json::{Value, json};

use super::report;
use super::{OpenSlot, fuse, fused_hit_ids, row_for_metric, slot_id, to_index_hits};
use crate::error::CliResult;
use crate::partitioned_bench::brute_force::{ExactTruth, exact_topk_vecfile};

struct SlotBatch {
    slot: u16,
    ranked: Vec<Vec<(u64, f32)>>,
    execution: CuvsChunkedExactReport,
}

pub(super) struct ExactFallback {
    slots: BTreeMap<SlotId, SlotBatch>,
}

impl ExactFallback {
    pub(super) fn build(slots: &[OpenSlot], query_count: usize, depth: usize) -> CliResult<Self> {
        let mut batches = BTreeMap::new();
        for slot in slots {
            let queries = (0..query_count)
                .map(|query_idx| {
                    row_for_metric(
                        &slot.queries,
                        slot.query_row(query_idx),
                        slot.distance_metric,
                    )
                })
                .collect::<Vec<_>>();
            let ExactTruth { ranked, execution } =
                exact_topk_vecfile(&slot.corpus, &queries, depth, slot.distance_metric)?;
            batches.insert(
                slot_id(slot.spec.slot),
                SlotBatch {
                    slot: slot.spec.slot,
                    ranked,
                    execution,
                },
            );
        }
        Ok(Self { slots: batches })
    }

    pub(super) fn fused_for_query(
        &self,
        slots: &[OpenSlot],
        query_idx: usize,
        k: usize,
    ) -> (Vec<u64>, Vec<Value>) {
        let mut exact_per_slot = BTreeMap::new();
        let mut exact_slot_rows = Vec::with_capacity(slots.len());
        for slot in slots {
            let slot_id = slot_id(slot.spec.slot);
            let exact = &self.slots[&slot_id].ranked[query_idx];
            exact_slot_rows.push(json!({
                "slot": slot.spec.slot,
                "source": "cuda_cuvs_chunked_slot_corpus",
                "exact_top_k": exact.iter().take(k).map(|(id, _)| *id).collect::<Vec<_>>(),
            }));
            exact_per_slot.insert(slot_id, to_index_hits(exact.clone()));
        }
        let exact_fused = fuse(&exact_per_slot, k);
        (fused_hit_ids(&exact_fused, k), exact_slot_rows)
    }

    pub(super) fn source(&self, query_count: usize, truth_depth: usize) -> Value {
        let executions = self
            .slots
            .values()
            .map(|batch| {
                json!({
                    "slot": batch.slot,
                    "execution": batch.execution,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "mode": "cuda_cuvs_chunked_slot_corpora",
            "metric_class": report::METRIC_CLASS,
            "metric_scope": report::METRIC_SCOPE,
            "truth_reference_class": report::TRUTH_REFERENCE_CLASS,
            "valid_real_outcome": false,
            "grounded_phase_exit_eligible": false,
            "scale_suitable": true,
            "query_count": query_count,
            "truth_depth": truth_depth,
            "slot_executions": executions,
        })
    }
}
