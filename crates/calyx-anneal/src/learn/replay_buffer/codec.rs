use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use serde::{Deserialize, Serialize};

use super::*;

pub(super) struct LoadedReplayState {
    pub(super) heap: BinaryHeap<Reverse<ReplayEntry>>,
    pub(super) generation: u64,
    pub(super) delta_seq: u64,
    pub(super) live_keys: Vec<Vec<u8>>,
    pub(super) legacy: bool,
}

pub(super) fn load_replay_state(
    rows: &[(Vec<u8>, Vec<u8>)],
    capacity: usize,
) -> Result<LoadedReplayState> {
    if rows.is_empty() {
        return Ok(LoadedReplayState {
            heap: BinaryHeap::new(),
            generation: 0,
            delta_seq: 0,
            live_keys: Vec::new(),
            legacy: false,
        });
    }
    if rows.len() == 1 && rows[0].0.as_slice() == LEGACY_SNAPSHOT_KEY {
        let snapshot = decode_replay_snapshot(&rows[0].1)?;
        return Ok(LoadedReplayState {
            heap: strict_heap_from_snapshot(snapshot, capacity)?,
            generation: 0,
            delta_seq: 0,
            live_keys: vec![LEGACY_SNAPSHOT_KEY.to_vec()],
            legacy: true,
        });
    }

    let by_key = rows
        .iter()
        .map(|(key, value)| (key.as_slice(), value.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let head_bytes = by_key
        .get(HEAD_KEY)
        .ok_or_else(|| invalid_row("anneal_replay v3 rows are missing head/v3"))?;
    let head: ReplayHeadRow = decode_row(head_bytes, "head")?;
    if head.tag != HEAD_TAG || head.generation == 0 {
        return Err(invalid_row(
            "anneal_replay head has invalid tag or generation",
        ));
    }
    if head.capacity != capacity {
        return Err(invalid_row(format!(
            "anneal_replay capacity mismatch: persisted {}, configured {capacity}",
            head.capacity
        )));
    }
    let checkpoint_key = replay_checkpoint_key(head.generation);
    let checkpoint_bytes = by_key.get(checkpoint_key.as_slice()).ok_or_else(|| {
        invalid_row(format!(
            "anneal_replay generation {} is missing its checkpoint",
            head.generation
        ))
    })?;
    let checkpoint: ReplayCheckpointRow = decode_row(checkpoint_bytes, "checkpoint")?;
    if checkpoint.tag != CHECKPOINT_TAG || checkpoint.generation != head.generation {
        return Err(invalid_row(
            "anneal_replay checkpoint metadata does not match head",
        ));
    }
    let mut heap = strict_heap_from_snapshot(checkpoint.snapshot, capacity)?;
    let mut expected_keys = vec![HEAD_KEY.to_vec(), checkpoint_key];
    for seq in 1..=head.delta_seq {
        let key = replay_delta_key(head.generation, seq);
        let bytes = by_key.get(key.as_slice()).ok_or_else(|| {
            invalid_row(format!(
                "anneal_replay generation {} is missing delta {seq}",
                head.generation
            ))
        })?;
        let delta: ReplayDeltaRow = decode_row(bytes, "delta")?;
        if delta.tag != DELTA_TAG || delta.generation != head.generation || delta.seq != seq {
            return Err(invalid_row(format!(
                "anneal_replay delta {seq} metadata does not match its key/head"
            )));
        }
        apply_recovered_delta(&mut heap, capacity, delta.operation)?;
        expected_keys.push(key);
    }
    expected_keys.sort();
    let mut actual_keys = rows.iter().map(|(key, _)| key.clone()).collect::<Vec<_>>();
    actual_keys.sort();
    if actual_keys != expected_keys {
        return Err(invalid_row(format!(
            "anneal_replay contains unexpected or stale live rows: expected {}, found {}",
            expected_keys.len(),
            actual_keys.len()
        )));
    }
    Ok(LoadedReplayState {
        heap,
        generation: head.generation,
        delta_seq: head.delta_seq,
        live_keys: expected_keys,
        legacy: false,
    })
}

fn strict_heap_from_snapshot(
    snapshot: ReplaySnapshot,
    capacity: usize,
) -> Result<BinaryHeap<Reverse<ReplayEntry>>> {
    validate_capacity(snapshot.capacity)?;
    if snapshot.capacity != capacity {
        return Err(invalid_row(format!(
            "anneal_replay snapshot capacity mismatch: persisted {}, configured {capacity}",
            snapshot.capacity
        )));
    }
    if snapshot.entries.len() > capacity {
        return Err(invalid_row(format!(
            "anneal_replay snapshot has {} entries above capacity {capacity}",
            snapshot.entries.len()
        )));
    }
    let mut heap = BinaryHeap::with_capacity(snapshot.entries.len());
    for entry in snapshot.entries {
        validate_entry(&entry)?;
        heap.push(Reverse(entry));
    }
    Ok(heap)
}

fn apply_recovered_delta(
    heap: &mut BinaryHeap<Reverse<ReplayEntry>>,
    capacity: usize,
    operation: ReplayDeltaOperation,
) -> Result<()> {
    match operation {
        ReplayDeltaOperation::Add { entry } => {
            validate_entry(&entry)?;
            if heap_admission(heap, capacity, &entry)? != ReplayAdmission::Add {
                return Err(invalid_row(
                    "anneal_replay add delta violates heap admission",
                ));
            }
            heap.push(Reverse(entry));
        }
        ReplayDeltaOperation::ReplaceMin { evicted, entry } => {
            validate_entry(&evicted)?;
            validate_entry(&entry)?;
            if heap_admission(heap, capacity, &entry)? != ReplayAdmission::ReplaceMin
                || heap.peek().map(|row| &row.0) != Some(&evicted)
            {
                return Err(invalid_row(
                    "anneal_replay replace-min delta does not match recovered heap minimum",
                ));
            }
            heap.pop();
            heap.push(Reverse(entry));
        }
    }
    Ok(())
}

pub fn replay_snapshot_key() -> Vec<u8> {
    LEGACY_SNAPSHOT_KEY.to_vec()
}

pub fn encode_replay_snapshot(snapshot: &ReplaySnapshot) -> Result<Vec<u8>> {
    validate_capacity(snapshot.capacity)?;
    if snapshot.entries.len() > snapshot.capacity {
        return Err(invalid_row("anneal_replay snapshot exceeds its capacity"));
    }
    for entry in &snapshot.entries {
        validate_entry(entry)?;
    }
    encode_row(
        &LegacySnapshotRow {
            tag: LEGACY_SNAPSHOT_TAG.to_string(),
            snapshot: snapshot.clone(),
        },
        "legacy snapshot",
    )
}

pub fn decode_replay_snapshot(bytes: &[u8]) -> Result<ReplaySnapshot> {
    let row: LegacySnapshotRow = decode_row(bytes, "legacy snapshot")?;
    if row.tag != LEGACY_SNAPSHOT_TAG {
        return Err(invalid_row("anneal_replay snapshot has invalid tag"));
    }
    validate_capacity(row.snapshot.capacity)?;
    if row.snapshot.entries.len() > row.snapshot.capacity {
        return Err(invalid_row("anneal_replay snapshot exceeds its capacity"));
    }
    for entry in &row.snapshot.entries {
        validate_entry(entry)?;
    }
    Ok(row.snapshot)
}

/// Reconstructs the logical replay snapshot from a consistent set of live CF
/// rows. Supports legacy v2 and checkpoint/delta v3 without mutating storage.
pub fn decode_replay_rows(rows: &[(Vec<u8>, Vec<u8>)]) -> Result<Option<ReplaySnapshot>> {
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() == 1 && rows[0].0.as_slice() == LEGACY_SNAPSHOT_KEY {
        return decode_replay_snapshot(&rows[0].1).map(Some);
    }
    let head_bytes = rows
        .iter()
        .find(|(key, _)| key.as_slice() == HEAD_KEY)
        .map(|(_, value)| value)
        .ok_or_else(|| invalid_row("anneal_replay v3 rows are missing head/v3"))?;
    let head: ReplayHeadRow = decode_row(head_bytes, "head")?;
    if head.tag != HEAD_TAG {
        return Err(invalid_row("anneal_replay head has invalid tag"));
    }
    let loaded = load_replay_state(rows, head.capacity)?;
    Ok(Some(ReplaySnapshot {
        capacity: head.capacity,
        entries: entries_by_priority(&loaded.heap),
    }))
}

pub(super) fn replay_checkpoint_key(generation: u64) -> Vec<u8> {
    format!("checkpoint/v3/{generation:020}").into_bytes()
}

pub(super) fn replay_delta_key(generation: u64, seq: u64) -> Vec<u8> {
    format!("delta/v3/{generation:020}/{seq:020}").into_bytes()
}

pub(super) fn encode_head(generation: u64, delta_seq: u64, capacity: usize) -> Result<Vec<u8>> {
    encode_row(
        &ReplayHeadRow {
            tag: HEAD_TAG.to_string(),
            generation,
            delta_seq,
            capacity,
        },
        "head",
    )
}

pub(super) fn encode_row<T: Serialize>(row: &T, kind: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(row, &mut bytes)
        .map_err(|error| invalid_row(format!("encode anneal_replay {kind}: {error}")))?;
    Ok(bytes)
}

fn decode_row<T>(bytes: &[u8], kind: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    ciborium::de::from_reader(bytes)
        .map_err(|error| invalid_row(format!("decode anneal_replay {kind}: {error}")))
}

pub(super) fn entries_by_priority(heap: &BinaryHeap<Reverse<ReplayEntry>>) -> Vec<ReplayEntry> {
    let mut entries = heap.iter().map(|entry| entry.0.clone()).collect::<Vec<_>>();
    entries.sort_by(|left, right| right.cmp(left));
    entries
}
