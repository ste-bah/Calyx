use std::collections::BTreeSet;

use super::encode::{self, WriteRow};
use crate::cf::{ColumnFamily, anchor_key, base_key};
use crate::dedup::{AnchorConflictResult, check_anchor_conflict};
use crate::recurrence::FREQUENCY_SCALAR;
use calyx_core::{Anchor, CalyxError, Constellation, CxId, LedgerRef, Result};

pub(super) fn merge_duplicate_anchors(
    existing: &mut Constellation,
    incoming: &Constellation,
) -> Result<Vec<Anchor>> {
    let identity_differences = anchor_merge_identity_differences(existing, incoming);
    if !identity_differences.is_empty() {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "CxId collision or non-idempotent duplicate constellation; differing identity fields: {}",
            identity_differences.join(", ")
        )));
    }
    if let AnchorConflictResult::Conflicting {
        anchor_type,
        reason,
    } = check_anchor_conflict(incoming, existing)
    {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "CxId duplicate has conflicting {anchor_type:?} anchor: {reason:?}"
        )));
    }

    let mut existing_kinds = existing
        .anchors
        .iter()
        .map(|anchor| anchor.kind.clone())
        .collect::<BTreeSet<_>>();
    let mut added = Vec::new();
    for anchor in &incoming.anchors {
        if existing_kinds.insert(anchor.kind.clone()) {
            existing.anchors.push(anchor.clone());
            added.push(anchor.clone());
        }
    }
    if !added.is_empty() {
        existing.flags.ungrounded = existing.anchors.is_empty();
        existing.validate_schema()?;
    }
    Ok(added)
}

/// Merges anchors for a repeated content observation while retaining the
/// authoritative first-write metadata, scalars, pointer, and derived state.
/// The fields that prove the content identity itself still fail closed.
pub(super) fn merge_observation_anchors(
    existing: &mut Constellation,
    incoming: &Constellation,
) -> Result<Vec<Anchor>> {
    let mut differences = Vec::new();
    if existing.cx_id != incoming.cx_id {
        differences.push("cx_id");
    }
    if existing.vault_id != incoming.vault_id {
        differences.push("vault_id");
    }
    if existing.panel_version != incoming.panel_version {
        differences.push("panel_version");
    }
    if existing.input_ref.hash != incoming.input_ref.hash {
        differences.push("input_hash");
    }
    if existing.input_ref.redacted != incoming.input_ref.redacted {
        differences.push("input_redaction");
    }
    if existing.modality != incoming.modality {
        differences.push("modality");
    }
    if existing.slots != incoming.slots {
        differences.push("slots");
    }
    if !differences.is_empty() {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "content-addressed observation collision; differing identity fields: {}",
            differences.join(", ")
        )));
    }
    let mut proposal = existing.clone();
    proposal.anchors = incoming.anchors.clone();
    proposal.flags.ungrounded = proposal.anchors.is_empty();
    merge_duplicate_anchors(existing, &proposal)
}

pub(super) fn stage_anchor_merge_rows(
    id: CxId,
    merged: &Constellation,
    added: &[Anchor],
) -> Result<Vec<WriteRow>> {
    let mut rows = Vec::with_capacity(1 + added.len());
    rows.push(WriteRow {
        cf: ColumnFamily::Base,
        key: base_key(id),
        value: encode::encode_constellation_base(merged)?,
    });
    for anchor in added {
        rows.push(WriteRow {
            cf: ColumnFamily::Anchors,
            key: anchor_key(id, &anchor.kind),
            value: encode::encode_anchor(anchor)?,
        });
    }
    Ok(rows)
}

fn normalized_anchor_identity(cx: &Constellation) -> Constellation {
    let mut normalized = cx.clone();
    normalized.anchors.clear();
    normalized.created_at = 0;
    normalized.flags.ungrounded = false;
    normalized.provenance = LedgerRef {
        seq: 0,
        hash: [0; 32],
    };
    // Recurrence is authoritative derived state written after duplicate
    // ingestion. A later duplicate must compare against the original caller
    // identity without erasing or requiring this system-owned counter.
    normalized.scalars.remove(FREQUENCY_SCALAR);
    normalized
}

fn anchor_merge_identity_differences(left: &Constellation, right: &Constellation) -> Vec<String> {
    let left = normalized_anchor_identity(left);
    let right = normalized_anchor_identity(right);
    let mut differences = Vec::new();
    if left.cx_id != right.cx_id {
        differences.push("cx_id".to_string());
    }
    if left.vault_id != right.vault_id {
        differences.push("vault_id".to_string());
    }
    if left.panel_version != right.panel_version {
        differences.push("panel_version".to_string());
    }
    if left.input_ref != right.input_ref {
        differences.push("input_ref".to_string());
    }
    if left.modality != right.modality {
        differences.push("modality".to_string());
    }
    if left.slots != right.slots {
        differences.push("slots".to_string());
    }
    if left.scalars != right.scalars {
        differences.push("scalars".to_string());
    }
    if left.metadata != right.metadata {
        differences.push("metadata".to_string());
    }
    if left.flags != right.flags {
        differences.push("flags".to_string());
    }
    differences
}
