use super::{
    OrphanGcTarget, OrphanIndexRepair, OrphanReconciler, OrphanRepairResult, OrphanReport,
    orphan_error,
};
use calyx_core::Result;
use std::sync::atomic::Ordering;

const MAX_REPAIR_CX_PER_CHUNK: usize = 128;
const MAX_REPAIR_SLOT_ROWS_PER_CHUNK: usize = 4_096;

impl OrphanReconciler {
    pub fn repair<T>(&self, target: &T, report: &OrphanReport) -> Result<OrphanRepairResult>
    where
        T: OrphanGcTarget + ?Sized,
    {
        validate_report_order(report)?;
        let mut remaining_budget = self.max_repairs_per_run;
        let mut repaired_index = 0;
        let mut degraded_base = 0;
        let mut entry_cursor = 0;
        let mut id_cursor = 0;
        let mut pending_repair: Option<OrphanIndexRepair> = None;

        let index_result = (|| -> Result<()> {
            while remaining_budget > 0 && id_cursor < report.orphan_index.len() {
                let mut repairs = Vec::new();
                let mut slot_rows: usize = 0;
                if let Some(repair) = pending_repair.take() {
                    slot_rows = repair.slots.len();
                    repairs.push(repair);
                }
                while id_cursor < report.orphan_index.len()
                    && repairs.len() < remaining_budget.min(MAX_REPAIR_CX_PER_CHUNK)
                {
                    let cx_id = report.orphan_index[id_cursor];
                    while entry_cursor < report.orphan_index_entries.len()
                        && report.orphan_index_entries[entry_cursor].cx_id < cx_id
                    {
                        #[cfg(test)]
                        super::record_report_entry_visit();
                        entry_cursor += 1;
                    }
                    let start = entry_cursor;
                    while entry_cursor < report.orphan_index_entries.len()
                        && report.orphan_index_entries[entry_cursor].cx_id == cx_id
                    {
                        #[cfg(test)]
                        super::record_report_entry_visit();
                        entry_cursor += 1;
                    }
                    let group_len = entry_cursor - start;
                    let repair = OrphanIndexRepair {
                        cx_id,
                        slots: report.orphan_index_entries[start..entry_cursor]
                            .iter()
                            .map(|entry| entry.slot)
                            .collect(),
                    };
                    id_cursor += 1;
                    if !repairs.is_empty()
                        && slot_rows.saturating_add(group_len) > MAX_REPAIR_SLOT_ROWS_PER_CHUNK
                    {
                        pending_repair = Some(repair);
                        break;
                    }
                    repairs.push(repair);
                    slot_rows = slot_rows.saturating_add(group_len);
                }

                let outcomes = target.purge_orphan_indexes(&repairs)?;
                if outcomes.len() != repairs.len() {
                    return Err(orphan_error(format!(
                        "orphan target returned {} index outcomes for {} repairs",
                        outcomes.len(),
                        repairs.len()
                    )));
                }
                for (repair, outcome) in repairs.iter().zip(outcomes) {
                    if outcome.cx_id != repair.cx_id {
                        return Err(orphan_error(format!(
                            "orphan index outcome order mismatch: expected {}, got {}",
                            repair.cx_id, outcome.cx_id
                        )));
                    }
                    if outcome.purged_rows > 0 {
                        repaired_index += 1;
                        remaining_budget -= 1;
                    }
                }
            }
            Ok(())
        })();
        let finish_result = target.finish_orphan_index_repairs();
        match (index_result, finish_result) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) | (Ok(()), Err(error)) => return Err(error),
            (Err(repair_error), Err(finish_error)) => {
                return Err(orphan_error(format!(
                    "orphan index repair failed ({repair_error}); post-repair compaction also failed ({finish_error})"
                )));
            }
        }

        let mut base_cursor = 0;
        while remaining_budget > 0 && base_cursor < report.orphan_base.len() {
            let end = (base_cursor + remaining_budget.min(MAX_REPAIR_CX_PER_CHUNK))
                .min(report.orphan_base.len());
            let ids = &report.orphan_base[base_cursor..end];
            let outcomes = target.flag_orphan_bases(ids)?;
            if outcomes.len() != ids.len() {
                return Err(orphan_error(format!(
                    "orphan target returned {} Base outcomes for {} repairs",
                    outcomes.len(),
                    ids.len()
                )));
            }
            for (expected, outcome) in ids.iter().zip(outcomes) {
                if outcome.cx_id != *expected {
                    return Err(orphan_error(format!(
                        "orphan Base outcome order mismatch: expected {expected}, got {}",
                        outcome.cx_id
                    )));
                }
                if outcome.degraded {
                    degraded_base += 1;
                    remaining_budget -= 1;
                }
            }
            base_cursor = end;
        }

        let repaired = repaired_index + degraded_base;
        let repairs_total = self
            .orphan_repairs_total
            .fetch_add(repaired as u64, Ordering::Relaxed)
            + repaired as u64;
        let remaining_inconsistencies = report.inconsistencies.saturating_sub(repaired);
        Ok(OrphanRepairResult {
            orphan_index_repaired: repaired_index,
            orphan_base_degraded: degraded_base,
            repairs_total,
            remaining_inconsistencies,
            rate_limited: remaining_inconsistencies > 0,
        })
    }
}

fn validate_report_order(report: &OrphanReport) -> Result<()> {
    if report.orphan_index.windows(2).any(|ids| ids[0] >= ids[1]) {
        return Err(orphan_error(
            "orphan_index must be strictly sorted for linear repair",
        ));
    }
    if report
        .orphan_index_entries
        .windows(2)
        .any(|entries| entries[0] > entries[1])
    {
        return Err(orphan_error(
            "orphan_index_entries must be sorted for linear repair",
        ));
    }
    Ok(())
}
