use super::*;

#[derive(Debug)]
pub(super) struct BatchPhysicalBaseState {
    pub(super) visible: BTreeSet<CxId>,
    pub(super) tombstoned: BTreeSet<CxId>,
}

pub(super) fn physical_batch_base_state(
    vault_path: &std::path::Path,
    cx_ids: &BTreeSet<CxId>,
) -> CliResult<BatchPhysicalBaseState> {
    let keys = cx_ids
        .iter()
        .map(|cx_id| base_key(*cx_id))
        .collect::<Vec<_>>();
    // The Base page index is a checked physical source of truth, not an
    // optional cache. A missing manifest must follow the same rebuild path as
    // a stale manifest; silently substituting a full CF scan both repeats the
    // expensive I/O on every ingest and lets ingest report complete while
    // `fsv vault-health` correctly reports the vault unready (#1675).
    let mut visible = BTreeSet::new();
    let mut tombstoned = BTreeSet::new();
    let stats = visit_indexed_batch_base_rows(vault_path, &keys, |key, value| {
        let key_bytes: [u8; 16] = key.try_into().map_err(|_| {
            CliError::runtime(format!(
                "physical Base CF selection returned a {}-byte key; expected 16",
                key.len()
            ))
        })?;
        let cx_id = CxId::from_bytes(key_bytes);
        let Some(value) = value else {
            return Ok(());
        };
        if calyx_aster::mvcc::is_tombstone_value(&value) {
            tombstoned.insert(cx_id);
            return Ok(());
        }
        let decoded = decode_constellation_base(&value)?;
        if decoded.cx_id != cx_id {
            return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
                "physical Base CF key {} decoded as mismatched cx {}",
                cx_id, decoded.cx_id
            ))
            .into());
        }
        visible.insert(cx_id);
        Ok(())
    })?;
    ingest_runtime_log(format_args!(
        "phase=batch_physical_base_index_visit unique_keys={} touched_pages={} source_files={} live_rows={} missing_rows={}",
        stats.unique_keys,
        stats.touched_pages,
        stats.source_files,
        stats.live_rows,
        stats.missing_rows
    ));
    Ok(BatchPhysicalBaseState {
        visible,
        tombstoned,
    })
}

pub(super) fn visit_indexed_batch_base_rows(
    vault_path: &std::path::Path,
    keys: &[Vec<u8>],
    mut visitor: impl FnMut(&[u8], Option<Vec<u8>>) -> CliResult<()>,
) -> CliResult<calyx_aster::base_page_index::SelectedBaseRowsVisit> {
    match calyx_aster::base_page_index::visit_indexed_base_rows_for_keys(
        vault_path,
        keys,
        &mut visitor,
    ) {
        Ok(stats) => Ok(stats),
        Err(error)
            if matches!(
                error.code(),
                "CALYX_BASE_PAGE_INDEX_MISSING" | "CALYX_BASE_PAGE_INDEX_STALE"
            ) =>
        {
            let reason = if error.code() == "CALYX_BASE_PAGE_INDEX_MISSING" {
                "missing"
            } else {
                "stale"
            };
            ingest_runtime_log(format_args!(
                "phase=batch_base_page_index_rebuild_start reason={reason} vault={} key_count={}",
                vault_path.display(),
                keys.len()
            ));
            let manifest = calyx_aster::base_page_index::build_base_page_index(
                vault_path,
                calyx_aster::base_page_index::DEFAULT_BASE_PAGE_INDEX_PAGE_SIZE,
                |_| Ok(()),
            )?;
            ingest_runtime_log(format_args!(
                "phase=batch_base_page_index_rebuild_ok reason={reason} vault={} live_entries={} pages={} ledger_head={}",
                vault_path.display(),
                manifest.live_entries,
                manifest.pages.len(),
                manifest.ledger_head_height
            ));
            calyx_aster::base_page_index::visit_indexed_base_rows_for_keys(
                vault_path, keys, visitor,
            )
        }
        Err(error) => Err(error),
    }
}

pub(super) fn reject_tombstoned_batch_ids(state: &BatchPhysicalBaseState) -> CliResult<()> {
    if state.tombstoned.is_empty() {
        return Ok(());
    }
    let ids = sample_ids(&state.tombstoned);
    Err(calyx_core::CalyxError {
        code: "CALYX_INGEST_TOMBSTONED_CX",
        message: format!(
            "batch contains {} Cx IDs whose physical Base CF rows are MVCC tombstones; refusing to resurrect erased data (sample_cx_ids={ids})",
            state.tombstoned.len()
        ),
        remediation:
            "remove tombstoned inputs from the batch or ingest intentionally new content with different bytes",
    }
    .into())
}

pub(super) fn reconcile_summary_with_physical_base(
    summary: &mut BatchIngestSummary,
    before: &BatchPhysicalBaseState,
    after: &BatchPhysicalBaseState,
) -> CliResult<()> {
    let materialized = after
        .visible
        .difference(&before.visible)
        .copied()
        .collect::<BTreeSet<_>>();
    let disappeared = before
        .visible
        .difference(&after.visible)
        .copied()
        .collect::<BTreeSet<_>>();
    let missing_after = summary
        .batch_cx_ids
        .difference(&after.visible)
        .copied()
        .collect::<BTreeSet<_>>();
    if !disappeared.is_empty() || !missing_after.is_empty() || !after.tombstoned.is_empty() {
        ingest_runtime_log(format_args!(
            "phase=batch_physical_base_readback_error distinct_cx={} disappeared={} missing_after={} tombstoned_after={} runtime_new_count={} runtime_already_count={}",
            summary.batch_cx_ids.len(),
            disappeared.len(),
            missing_after.len(),
            after.tombstoned.len(),
            summary.runtime_new_count,
            summary.runtime_already_count
        ));
        return Err(calyx_core::CalyxError {
            code: "CALYX_INGEST_BASE_READBACK_MISMATCH",
            message: format!(
                "batch physical Base CF reconciliation failed after flush: distinct_cx={}, disappeared={}, missing_after={}, tombstoned_after={}, disappeared_sample={}, missing_after_sample={}, tombstoned_after_sample={}",
                summary.batch_cx_ids.len(),
                disappeared.len(),
                missing_after.len(),
                after.tombstoned.len(),
                sample_ids(&disappeared),
                sample_ids(&missing_after),
                sample_ids(&after.tombstoned)
            ),
            remediation:
                "inspect the named Cx IDs with `calyx readback cx-list --vault <vault> --cx-id <cx> --include-slots` and rerun ingest only after Base CF state is consistent",
        }
        .into());
    }
    if materialized.len() > summary.row_count {
        return Err(calyx_core::CalyxError {
            code: "CALYX_INGEST_BASE_READBACK_MISMATCH",
            message: format!(
                "batch materialized {} physical Base CF rows for only {} input rows (sample_cx_ids={})",
                materialized.len(),
                summary.row_count,
                sample_ids(&materialized)
            ),
            remediation:
                "inspect concurrent writers and the Base CF readback before trusting the ingest summary",
        }
        .into());
    }
    summary.distinct_cx_count = summary.batch_cx_ids.len();
    summary.batch_base_visible_before = before.visible.len();
    summary.batch_base_visible_after = after.visible.len();
    summary.batch_base_materialized_count = materialized.len();
    summary.batch_base_tombstoned_before = before.tombstoned.len();
    summary.batch_base_tombstoned_after = after.tombstoned.len();
    summary.new_count = materialized.len();
    summary.already_count = summary.row_count - summary.new_count;
    summary.physical_reconciled = true;
    ingest_runtime_log(format_args!(
        "phase=batch_physical_base_readback_ok row_count={} distinct_cx={} new_count={} already_count={} runtime_new_count={} runtime_already_count={} visible_before={} visible_after={} tombstoned_before={} tombstoned_after={}",
        summary.row_count,
        summary.distinct_cx_count,
        summary.new_count,
        summary.already_count,
        summary.runtime_new_count,
        summary.runtime_already_count,
        summary.batch_base_visible_before,
        summary.batch_base_visible_after,
        summary.batch_base_tombstoned_before,
        summary.batch_base_tombstoned_after
    ));
    Ok(())
}

fn sample_ids(ids: &BTreeSet<CxId>) -> String {
    ids.iter()
        .take(8)
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}
