use super::batch_physical::{BatchPhysicalBaseState, visit_indexed_batch_base_rows};
use super::batch_support::{
    BatchOrderRow, IdentityFields, append_idempotent_batch_ledger, append_missing_batch_anchors,
    append_oracle_events, current_anchor_kinds, existing_replay_incoming, identity_mismatch_reason,
    verify_existing_batch_replay_identity,
};
use super::*;

pub(crate) struct BatchExistingPreflight {
    batch_ids: BTreeSet<CxId>,
    expected: BTreeMap<CxId, Vec<ExistingPlainReplayRow>>,
    materialized: BTreeSet<CxId>,
    distinct_existing: usize,
    exact_pointer_skipped: usize,
    pointer_backfills: BTreeMap<CxId, InputRef>,
    physical_base: BatchPhysicalBaseState,
}

impl BatchExistingPreflight {
    pub(crate) fn batch_ids(&self) -> &BTreeSet<CxId> {
        &self.batch_ids
    }

    pub(super) fn physical_base(&self) -> &BatchPhysicalBaseState {
        &self.physical_base
    }

    fn is_materialized(&self, cx_id: CxId) -> bool {
        self.materialized.contains(&cx_id)
    }
}

pub(crate) fn preflight_batch_existing_identity(
    vault: &AsterVault,
    state: &VaultPanelState,
    vault_path: &std::path::Path,
    path: &std::path::Path,
    validated_row_count: usize,
) -> CliResult<BatchExistingPreflight> {
    use std::io::BufRead;

    let started = std::time::Instant::now();
    ingest_runtime_log(format_args!(
        "phase=batch_existing_identity_preflight_start rows={validated_row_count}"
    ));
    let file = std::fs::File::open(path)
        .map_err(|err| CliError::io(format!("open batch {}: {err}", path.display())))?;
    let reader = std::io::BufReader::new(file);
    let snapshot = vault.snapshot();
    let mut expected = BTreeMap::<CxId, Vec<ExistingPlainReplayRow>>::new();
    for (index, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|err| CliError::io(format!("read batch line {}: {err}", index + 1)))?;
        let Some((text, mut metadata, anchors, oracle)) = parse_batch_line(index, &line)? else {
            continue;
        };
        if let Some(event) = &oracle {
            event.apply_metadata(&mut metadata)?;
        }
        let input = retention::retained_text_input(vault_path, &text)?;
        let row = ExistingPlainReplayRow {
            cx_id: vault.cx_id_for_input(&input.bytes, state.panel.version),
            panel_version: state.panel.version,
            input_ref: InputRef {
                hash: input_hash(&input.bytes),
                pointer: input.pointer,
                redacted: false,
            },
            modality: input.modality,
            metadata,
            anchors,
        };
        expected.entry(row.cx_id).or_default().push(row);
    }
    let batch_ids = expected.keys().copied().collect::<BTreeSet<_>>();
    let keys = batch_ids
        .iter()
        .map(|cx_id| base_key(*cx_id))
        .collect::<Vec<_>>();
    let mut materialized = BTreeSet::new();
    let mut distinct_existing = 0_usize;
    let mut checked_existing = 0_usize;
    let mut not_existing_or_incomplete = 0_usize;
    let mut exact_pointer_skipped = 0_usize;
    let mut pointer_backfills = BTreeMap::new();
    let mut visible = BTreeSet::new();
    let mut tombstoned = BTreeSet::new();
    let mut required_anchor_rows = BTreeMap::<Vec<u8>, Anchor>::new();
    let base_stats = visit_indexed_batch_base_rows(vault_path, &keys, |key, value| {
        let key_bytes: [u8; 16] = key.try_into().map_err(|_| {
            CliError::runtime(format!(
                "batch identity preflight received a {}-byte Base key; expected 16",
                key.len()
            ))
        })?;
        let cx_id = CxId::from_bytes(key_bytes);
        let rows = expected.get(&cx_id).ok_or_else(|| {
            CliError::runtime(format!(
                "batch identity preflight received unrequested Base cx_id {cx_id}"
            ))
        })?;
        let Some(value) = value else {
            not_existing_or_incomplete += rows.len();
            return Ok(());
        };
        if calyx_aster::mvcc::is_tombstone_value(&value) {
            tombstoned.insert(cx_id);
            not_existing_or_incomplete += rows.len();
            return Ok(());
        }
        let existing = decode_constellation_base(&value)?;
        if existing.cx_id != cx_id {
            return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
                "batch identity preflight Base key {cx_id} decoded as {}",
                existing.cx_id
            ))
            .into());
        }
        visible.insert(cx_id);
        distinct_existing += 1;
        let expected_input_ref = &rows[0].input_ref;
        if existing.input_ref == *expected_input_ref {
            exact_pointer_skipped += 1;
        } else if retention::input_ref_matches_or_backfillable(
            &existing.input_ref,
            expected_input_ref,
        ) {
            pointer_backfills.insert(cx_id, expected_input_ref.clone());
        }
        let mut complete = true;
        for row in rows {
            if verify_existing_base_replay_value(&existing, row, true)? {
                checked_existing += 1;
            } else {
                not_existing_or_incomplete += 1;
                complete = false;
            }
        }
        if complete {
            for row in rows {
                for anchor in &row.anchors {
                    let anchor_key = anchor_key(cx_id, &anchor.kind);
                    match required_anchor_rows.entry(anchor_key) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(anchor.clone());
                        }
                        std::collections::btree_map::Entry::Occupied(entry)
                            if entry.get().kind != anchor.kind
                                || entry.get().value != anchor.value =>
                        {
                            return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
                                "batch repeats cx {cx_id} with conflicting {:?} anchor value",
                                anchor.kind
                            ))
                            .into());
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {}
                    }
                }
            }
            materialized.insert(cx_id);
        }
        Ok(())
    })?;
    let required_anchor_count = required_anchor_rows.len();
    if required_anchor_count > 0 {
        let mut found = BTreeSet::new();
        vault.scan_cf_pages_at_renewing_latest(
            snapshot,
            ColumnFamily::Anchors,
            4096,
            |page| -> CliResult<()> {
                for (key, value) in page {
                    let Some(expected_anchor) = required_anchor_rows.get(&key) else {
                        continue;
                    };
                    let indexed = encode::decode_anchor(&value)?;
                    if indexed.kind != expected_anchor.kind
                        || indexed.value != expected_anchor.value
                    {
                        let cx_id = key
                            .get(..16)
                            .and_then(|bytes| <[u8; 16]>::try_from(bytes).ok())
                            .map(CxId::from_bytes)
                            .map(|id| id.to_string())
                            .unwrap_or_else(|| "<malformed-key>".to_string());
                        return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
                            "batch identity preflight Anchors CF value mismatch for cx {cx_id} key_len={}",
                            key.len()
                        ))
                        .into());
                    }
                    found.insert(key);
                }
                Ok(())
            },
        )?;
        if found.len() != required_anchor_count {
            let missing = required_anchor_rows
                .keys()
                .find(|key| !found.contains(*key))
                .expect("anchor count mismatch has a missing key");
            let cx_id = missing
                .get(..16)
                .and_then(|bytes| <[u8; 16]>::try_from(bytes).ok())
                .map(CxId::from_bytes)
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<malformed-key>".to_string());
            return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
                "batch identity preflight Base CF anchor for cx {cx_id} is missing from Anchors CF"
            ))
            .into());
        }
    }
    ingest_runtime_log(format_args!(
        "phase=batch_existing_identity_preflight_ok rows={} distinct_ids={} existing_checked={} not_existing_or_incomplete={} base_pages={} base_source_files={} anchors_required={} exact_pointer_skipped={} pointer_backfills={} elapsed_ms={}",
        validated_row_count,
        batch_ids.len(),
        checked_existing,
        not_existing_or_incomplete,
        base_stats.touched_pages,
        base_stats.source_files,
        required_anchor_count,
        exact_pointer_skipped,
        pointer_backfills.len(),
        started.elapsed().as_millis()
    ));
    Ok(BatchExistingPreflight {
        batch_ids,
        expected,
        materialized,
        distinct_existing,
        exact_pointer_skipped,
        pointer_backfills,
        physical_base: BatchPhysicalBaseState {
            visible,
            tombstoned,
        },
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchInputPointerBackfillStats {
    pub(crate) distinct_existing: usize,
    pub(crate) exact_pointer_skipped: usize,
    pub(crate) backfill_attempted: usize,
    pub(crate) changed: usize,
}

pub(crate) fn backfill_batch_existing_input_pointers(
    vault: &AsterVault,
    preflight: &BatchExistingPreflight,
) -> CliResult<BatchInputPointerBackfillStats> {
    let mut stats = BatchInputPointerBackfillStats {
        distinct_existing: preflight.distinct_existing,
        exact_pointer_skipped: preflight.exact_pointer_skipped,
        backfill_attempted: preflight.pointer_backfills.len(),
        changed: 0,
    };
    for (cx_id, expected) in &preflight.pointer_backfills {
        if retention::apply_existing_input_pointer(vault, *cx_id, expected)? {
            stats.changed += 1;
        }
    }
    ingest_runtime_log(format_args!(
        "phase=batch_retained_input_pointer_outcomes distinct_existing={} exact_pointer_skipped={} backfill_attempted={} changed={}",
        stats.distinct_existing,
        stats.exact_pointer_skipped,
        stats.backfill_attempted,
        stats.changed,
    ));
    Ok(stats)
}

#[derive(Clone)]
pub(crate) struct ExistingPlainReplayRow {
    cx_id: CxId,
    panel_version: u32,
    input_ref: InputRef,
    modality: Modality,
    metadata: BTreeMap<String, String>,
    anchors: Vec<Anchor>,
}

pub(crate) struct ExistingBatchReplayRow {
    pub(crate) cx_id: CxId,
    pub(crate) input_ref: InputRef,
    pub(crate) modality: Modality,
    pub(crate) metadata: BTreeMap<String, String>,
    pub(crate) anchors: Vec<Anchor>,
    pub(crate) oracle: Option<OracleEvent>,
}

pub(crate) fn existing_plain_batch_replay_rows(
    vault: &AsterVault,
    state: &VaultPanelState,
    rows: &[BatchRow],
    preflight: &BatchExistingPreflight,
) -> CliResult<Option<Vec<ExistingPlainReplayRow>>> {
    let mut out = Vec::with_capacity(rows.len());
    for (text, metadata, anchors, oracle) in rows {
        let cx_id = vault.cx_id_for_input(text.as_bytes(), state.panel.version);
        if !preflight.is_materialized(cx_id) {
            ingest_runtime_log(format_args!(
                "phase=batch_existing_replay_base_only_preflight_mixed rows={} missing_or_incomplete_cx={} measurement_required=true slot_decode_skipped=true",
                rows.len(),
                cx_id
            ));
            return Ok(None);
        }
        let mut metadata = metadata.clone();
        if let Some(event) = oracle {
            event.apply_metadata(&mut metadata)?;
        }
        let row = preflight
            .expected
            .get(&cx_id)
            .and_then(|candidates| {
                candidates.iter().find(|candidate| {
                    candidate.metadata == metadata
                        && candidate.anchors.len() == anchors.len()
                        && candidate.anchors.iter().all(|expected| {
                            anchors.iter().any(|actual| {
                                actual.kind == expected.kind && actual.value == expected.value
                            })
                        })
                })
            })
            .ok_or_else(|| {
                CliError::runtime(format!(
                    "batch replay row for preflighted cx {cx_id} no longer matches its preflight metadata/anchors"
                ))
            })?;
        out.push(row.clone());
    }
    Ok(Some(out))
}

pub(crate) fn flush_plain_existing_batch_replay(
    vault: &AsterVault,
    vault_path: &std::path::Path,
    rows: Vec<ExistingPlainReplayRow>,
    summary: &mut BatchIngestSummary,
    output: IngestOutput,
) -> CliResult<()> {
    for sub in rows.chunks(EXISTING_REPLAY_CHUNK) {
        let ids = sub.iter().map(|row| row.cx_id).collect::<Vec<_>>();
        let ledger_seq = append_cli_batch_ledger(
            vault,
            EntryKind::Ingest,
            &ids,
            "cli-idempotent-ingest-batch",
        )?;
        vault.flush()?;
        calyx_aster::base_page_index::advance_base_page_index_head_if_base_unchanged(vault_path)?;
        for row in sub {
            let report = IngestReport {
                cx_id: row.cx_id.to_string(),
                new: false,
                ledger_seq,
            };
            summary.record(row.cx_id, &report);
            if output == IngestOutput::Rows {
                print_json(&report)?;
            }
        }
    }
    Ok(())
}

fn verify_existing_base_replay_value(
    existing: &Constellation,
    row: &ExistingPlainReplayRow,
    allow_pointerless: bool,
) -> CliResult<bool> {
    let input_ref_matches = existing.input_ref == row.input_ref
        || (allow_pointerless
            && retention::input_ref_matches_or_backfillable(&existing.input_ref, &row.input_ref));
    if existing.panel_version != row.panel_version
        || !input_ref_matches
        || existing.modality != row.modality
        || existing.metadata != row.metadata
    {
        return Err(CliError::usage(format!(
            "idempotent batch replay for cx {} changed stored non-anchor identity: {}",
            row.cx_id,
            identity_mismatch_reason(
                IdentityFields {
                    panel_version: existing.panel_version,
                    input_ref: &existing.input_ref,
                    modality: existing.modality,
                    metadata: &existing.metadata,
                },
                IdentityFields {
                    panel_version: row.panel_version,
                    input_ref: &row.input_ref,
                    modality: row.modality,
                    metadata: &row.metadata,
                },
            )
        )));
    }
    if !incoming_anchors_already_materialized(row.cx_id, &row.anchors, existing)? {
        return Ok(false);
    }
    Ok(true)
}

fn incoming_anchors_already_materialized(
    cx_id: CxId,
    incoming_anchors: &[Anchor],
    existing_base: &Constellation,
) -> CliResult<bool> {
    if incoming_anchors.is_empty() {
        return Ok(true);
    }
    let mut incoming = existing_base.clone();
    incoming.anchors = incoming_anchors.to_vec();
    if let AnchorConflictResult::Conflicting {
        anchor_type,
        reason,
    } = check_anchor_conflict(&incoming, existing_base)
    {
        return Err(calyx_core::CalyxError::aster_corrupt_shard(format!(
            "idempotent batch replay for cx {cx_id} has conflicting {anchor_type:?} anchor: {reason:?}"
        ))
        .into());
    }
    for anchor in incoming_anchors {
        if !existing_base
            .anchors
            .iter()
            .any(|existing| existing.kind == anchor.kind)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn existing_batch_replay_rows(
    vault: &AsterVault,
    state: &VaultPanelState,
    vault_path: &std::path::Path,
    rows: &[BatchRow],
) -> CliResult<Option<Vec<ExistingBatchReplayRow>>> {
    let mut out = Vec::with_capacity(rows.len());
    let mut all_exist = true;
    let mut checked_existing = 0_usize;
    for (text, metadata, anchors, oracle) in rows {
        let input = retention::retained_text_input(vault_path, text)?;
        let cx_id = vault.cx_id_for_input(&input.bytes, state.panel.version);
        if !base_exists(vault, cx_id)? {
            all_exist = false;
            continue;
        }
        let input_ref = InputRef {
            hash: input_hash(&input.bytes),
            pointer: input.pointer,
            redacted: false,
        };
        let mut metadata = metadata.clone();
        if let Some(event) = oracle {
            event.apply_metadata(&mut metadata)?;
        }
        let row = ExistingBatchReplayRow {
            cx_id,
            input_ref,
            modality: input.modality,
            metadata,
            anchors: anchors.clone(),
            oracle: oracle.clone(),
        };
        verify_existing_batch_replay_identity(vault, state, &row)?;
        checked_existing += 1;
        out.push(row);
    }
    if all_exist {
        Ok(Some(out))
    } else {
        ingest_runtime_log(format_args!(
            "phase=batch_existing_replay_preflight_mixed rows={} existing_checked={} measurement_required=true",
            rows.len(),
            checked_existing
        ));
        Ok(None)
    }
}

pub(crate) fn flush_existing_batch_replay(
    vault: &AsterVault,
    state: &VaultPanelState,
    rows: Vec<ExistingBatchReplayRow>,
    summary: &mut BatchIngestSummary,
    output: IngestOutput,
) -> CliResult<()> {
    for sub in rows.chunks(EXISTING_REPLAY_CHUNK) {
        let mut order = Vec::with_capacity(sub.len());
        let mut known_anchor_kinds = BTreeMap::<CxId, BTreeSet<AnchorKind>>::new();
        for row in sub {
            let existing = verify_existing_batch_replay_identity(vault, state, row)?;
            let known = match known_anchor_kinds.entry(row.cx_id) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(current_anchor_kinds(vault, row.cx_id, true)?)
                }
            };
            let mut marker_kinds = Vec::new();
            for anchor in &row.anchors {
                if known.insert(anchor.kind.clone()) {
                    marker_kinds.push(anchor.kind.clone());
                }
            }
            let incoming = existing_replay_incoming(&existing, row);
            let expected_readback = if marker_kinds.is_empty() {
                existing
            } else {
                append_missing_batch_anchors(vault, &existing, &incoming, &marker_kinds)?
            };
            order.push(BatchOrderRow {
                cx_id: row.cx_id,
                expected_readback,
                new: false,
                marker_kinds,
                oracle: row.oracle.clone(),
            });
        }
        vault.flush()?;
        let snapshot = vault.snapshot();
        for row in &order {
            verify_base_readback(
                vault,
                snapshot,
                &row.expected_readback,
                row.cx_id,
                &row.marker_kinds,
            )?;
        }
        append_oracle_events(vault, &order)?;
        let idempotent_ledger_seq = append_idempotent_batch_ledger(vault, &order)?;
        for row in order {
            let cx_id = row.cx_id;
            let ledger_seq = idempotent_ledger_seq.ok_or_else(|| {
                CliError::usage("missing idempotent batch ledger seq for replay row")
            })?;
            for kind in row.marker_kinds {
                append_anchor_marker_ledger(vault, cx_id, &kind)?;
            }
            let report = IngestReport {
                cx_id: cx_id.to_string(),
                new: false,
                ledger_seq,
            };
            summary.record(cx_id, &report);
            if output == IngestOutput::Rows {
                print_json(&report)?;
            }
        }
        vault.flush()?;
    }
    Ok(())
}
