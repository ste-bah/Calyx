use calyx_aster::cf::ColumnFamily;
use calyx_aster::dedup::{AnchorConflictResult, check_anchor_conflict};
use calyx_core::{Anchor, AnchorKind, AnchorValue, Constellation, CxId, VaultStore};
use serde_json::Value;

use super::super::error::{AnchorConflictError, EngineError};
use super::super::{EngineResult, VaultHandle};

const ANCHOR_COMPACTION_MARKER: &[u8] = b"cx_anchor_compaction_v1";
const ANCHOR_CONFLICT_REMEDIATION: &str =
    "send only one compatible anchor value per kind for duplicate cx.put";

/// Preserves Leapable's structured anchor-conflict contract without restoring
/// a Base preflight read on successful puts. The diagnostic read occurs only
/// after Aster has rejected the atomic put and is never used as a fallback.
pub(super) fn classify_put_error(
    handle: &VaultHandle,
    cx_id: CxId,
    incoming_anchors: &[Anchor],
    error: calyx_core::CalyxError,
) -> EngineError {
    if error.code != "CALYX_ASTER_CORRUPT_SHARD" || incoming_anchors.is_empty() {
        return EngineError::Calyx(error);
    }
    let existing = match handle.vault.get(cx_id, handle.vault.snapshot()) {
        Ok(existing) => existing,
        Err(_) => return EngineError::Calyx(error),
    };
    anchor_conflict_error(incoming_anchors, &existing)
        .map(|conflict| EngineError::AnchorConflict(Box::new(conflict)))
        .unwrap_or(EngineError::Calyx(error))
}

/// Identifies the first real anchor conflict after an atomic batch rejection.
/// Stored rows and earlier same-id inputs are inspected only on the error path.
pub(super) fn classify_batch_put_error(
    handle: &VaultHandle,
    constellations: &[Constellation],
    error: calyx_core::CalyxError,
) -> EngineError {
    if error.code != "CALYX_ASTER_CORRUPT_SHARD" {
        return EngineError::Calyx(error);
    }
    let snapshot = handle.vault.snapshot();
    let mut earlier = std::collections::BTreeMap::new();
    for constellation in constellations {
        if !constellation.anchors.is_empty() {
            if let Some(existing) = earlier.get(&constellation.cx_id) {
                if let Some(conflict) = anchor_conflict_error(&constellation.anchors, existing) {
                    return EngineError::AnchorConflict(Box::new(conflict));
                }
            } else if let Ok(existing) = handle.vault.get(constellation.cx_id, snapshot)
                && let Some(conflict) = anchor_conflict_error(&constellation.anchors, &existing)
            {
                return EngineError::AnchorConflict(Box::new(conflict));
            }
        }
        earlier.insert(constellation.cx_id, constellation.clone());
    }
    EngineError::Calyx(error)
}

pub(super) fn repair_duplicate_anchor_bloat(handle: &VaultHandle) -> EngineResult<()> {
    let snapshot = handle.vault.snapshot();
    if handle
        .vault
        .read_cf_at(snapshot, ColumnFamily::Leapable, ANCHOR_COMPACTION_MARKER)?
        .is_some()
    {
        return Ok(());
    }
    let report = handle.vault.compact_duplicate_anchors()?;
    for conflict in &report.conflicts {
        eprintln!(
            "calyx-leapable: CALYX_LEAPABLE_ANCHOR_COMPACTION_CONFLICT: vault_ref={} cx_id={} anchor_kind={:?} existing_value={:?} incoming_value={:?}",
            handle.vault_ref.as_str(),
            conflict.cx_id,
            conflict.anchor_kind,
            conflict.existing_value,
            conflict.incoming_value
        );
    }
    if report.compacted > 0 || !report.conflicts.is_empty() {
        eprintln!(
            "calyx-leapable: CALYX_LEAPABLE_ANCHOR_COMPACTION: vault_ref={} scanned={} compacted={} removed_duplicates={} conflicts={}",
            handle.vault_ref.as_str(),
            report.scanned,
            report.compacted,
            report.removed_duplicates,
            report.conflicts.len()
        );
    }
    handle.vault.write_cf_batch([(
        ColumnFamily::Leapable,
        ANCHOR_COMPACTION_MARKER.to_vec(),
        anchor_compaction_marker_value(&report),
    )])?;
    Ok(())
}

fn anchor_conflict_error(
    incoming: &[Anchor],
    existing: &Constellation,
) -> Option<AnchorConflictError> {
    let mut candidate = existing.clone();
    candidate.anchors = incoming.to_vec();
    let AnchorConflictResult::Conflicting {
        anchor_type,
        reason,
    } = check_anchor_conflict(&candidate, existing)
    else {
        return None;
    };
    let (incoming_anchor, existing_anchor) =
        first_conflicting_pair(incoming, &existing.anchors, &anchor_type)?;
    Some(AnchorConflictError {
        message: format!(
            "duplicate cx.put anchor conflict for kind {:?}: existing {:?}, incoming {:?}",
            anchor_type, existing_anchor.value, incoming_anchor.value
        ),
        anchor_kind: anchor_kind_value(&anchor_type),
        conflict_reason: format!("{reason:?}"),
        existing_value: anchor_value(&existing_anchor.value),
        incoming_value: anchor_value(&incoming_anchor.value),
        remediation: ANCHOR_CONFLICT_REMEDIATION,
    })
}

fn first_conflicting_pair<'a>(
    incoming: &'a [Anchor],
    existing: &'a [Anchor],
    kind: &AnchorKind,
) -> Option<(&'a Anchor, &'a Anchor)> {
    for incoming_anchor in incoming.iter().filter(|anchor| &anchor.kind == kind) {
        for existing_anchor in existing.iter().filter(|anchor| &anchor.kind == kind) {
            if incoming_anchor.value != existing_anchor.value {
                return Some((incoming_anchor, existing_anchor));
            }
        }
    }
    None
}

fn anchor_kind_value(kind: &AnchorKind) -> Value {
    serde_json::to_value(kind).unwrap_or_else(|_| Value::String(format!("{kind:?}")))
}

fn anchor_value(value: &AnchorValue) -> Value {
    serde_json::to_value(value).unwrap_or_else(|_| Value::String(format!("{value:?}")))
}

fn anchor_compaction_marker_value(report: &calyx_aster::vault::AnchorCompactionReport) -> Vec<u8> {
    format!(
        "scanned={} compacted={} removed_duplicates={} conflicts={}",
        report.scanned,
        report.compacted,
        report.removed_duplicates,
        report.conflicts.len()
    )
    .into_bytes()
}
