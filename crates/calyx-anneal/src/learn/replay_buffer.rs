use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use calyx_core::{CalyxError, Clock, CxId, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::{MistakeEntry, MistakeLog, MistakeRef, MistakeStorage};
use crate::LogicalTime;
pub use codec::{
    decode_replay_rows, decode_replay_snapshot, encode_replay_snapshot, replay_snapshot_key,
};
use codec::{
    encode_head, encode_row, entries_by_priority, load_replay_state, replay_checkpoint_key,
    replay_delta_key,
};
use errors::{cf_unavailable, invalid_row};
pub use storage::{AsterReplayStorage, ReplayStorage, ReplayWrite};

mod codec;
mod errors;
mod storage;

pub const DEFAULT_REPLAY_CAPACITY: usize = 4096;
pub const DEFAULT_REPLAY_CHECKPOINT_INTERVAL: u64 = 256;
pub const CALYX_ANNEAL_INVALID_CAPACITY: &str = "CALYX_ANNEAL_INVALID_CAPACITY";
pub const CALYX_ANNEAL_REPLAY_INVALID_ROW: &str = "CALYX_ANNEAL_REPLAY_INVALID_ROW";

const LEGACY_SNAPSHOT_TAG: &str = "anneal_replay_snapshot_v2";
const LEGACY_SNAPSHOT_KEY: &[u8] = b"snapshot/v1";
const HEAD_TAG: &str = "anneal_replay_head_v3";
const CHECKPOINT_TAG: &str = "anneal_replay_checkpoint_v3";
const DELTA_TAG: &str = "anneal_replay_delta_v3";
const HEAD_KEY: &[u8] = b"head/v3";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayEntry {
    pub cx_id: CxId,
    pub target: f64,
    pub surprise: f64,
    pub mistake_ref: MistakeRef,
    pub added_ts: LogicalTime,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySnapshot {
    pub capacity: usize,
    pub entries: Vec<ReplayEntry>,
}

#[derive(Serialize, Deserialize)]
struct LegacySnapshotRow {
    tag: String,
    snapshot: ReplaySnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayHeadRow {
    tag: String,
    generation: u64,
    delta_seq: u64,
    capacity: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayCheckpointRow {
    tag: String,
    generation: u64,
    snapshot: ReplaySnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayDeltaRow {
    tag: String,
    generation: u64,
    seq: u64,
    operation: ReplayDeltaOperation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ReplayDeltaOperation {
    Add {
        entry: ReplayEntry,
    },
    ReplaceMin {
        evicted: ReplayEntry,
        entry: ReplayEntry,
    },
}

pub struct ReplayBuffer<S> {
    heap: BinaryHeap<Reverse<ReplayEntry>>,
    capacity: usize,
    clock: Arc<dyn Clock>,
    storage: S,
    generation: u64,
    delta_seq: u64,
    checkpoint_interval: u64,
    live_keys: Vec<Vec<u8>>,
}

impl<S> ReplayBuffer<S>
where
    S: ReplayStorage,
{
    pub fn open(storage: S, capacity: usize, clock: Arc<dyn Clock>) -> Result<Self> {
        Self::open_with_checkpoint_interval(
            storage,
            capacity,
            DEFAULT_REPLAY_CHECKPOINT_INTERVAL,
            clock,
        )
    }

    pub fn open_with_checkpoint_interval(
        storage: S,
        capacity: usize,
        checkpoint_interval: u64,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        validate_capacity(capacity)?;
        if checkpoint_interval == 0 {
            return Err(invalid_row("replay checkpoint interval must be > 0"));
        }
        let rows = storage.scan_rows()?;
        let loaded = load_replay_state(&rows, capacity)?;
        let mut buffer = Self {
            heap: loaded.heap,
            capacity,
            clock,
            storage,
            generation: loaded.generation,
            delta_seq: loaded.delta_seq,
            checkpoint_interval,
            live_keys: loaded.live_keys,
        };
        if loaded.legacy {
            buffer.persist_checkpoint()?;
        }
        Ok(buffer)
    }

    pub fn open_default(storage: S, clock: Arc<dyn Clock>) -> Result<Self> {
        Self::open(storage, DEFAULT_REPLAY_CAPACITY, clock)
    }

    pub fn push(&mut self, entry: ReplayEntry) -> Result<bool> {
        validate_entry(&entry)?;
        let admission = heap_admission(&self.heap, self.capacity, &entry)?;
        if admission == ReplayAdmission::Reject {
            return Ok(false);
        }

        if self.generation == 0 || self.delta_seq + 1 >= self.checkpoint_interval {
            let mut next_heap = self.heap.clone();
            apply_admission(&mut next_heap, admission, entry);
            self.persist_heap_checkpoint(&next_heap)?;
            self.heap = next_heap;
            return Ok(true);
        }

        let next_seq = self
            .delta_seq
            .checked_add(1)
            .ok_or_else(|| invalid_row("replay delta sequence overflow"))?;
        let operation = match admission {
            ReplayAdmission::Reject => unreachable!("rejected admission returned above"),
            ReplayAdmission::Add => ReplayDeltaOperation::Add {
                entry: entry.clone(),
            },
            ReplayAdmission::ReplaceMin => ReplayDeltaOperation::ReplaceMin {
                evicted: self
                    .heap
                    .peek()
                    .map(|row| row.0.clone())
                    .ok_or_else(|| invalid_row("replace-min admission has no minimum entry"))?,
                entry: entry.clone(),
            },
        };
        let delta_key = replay_delta_key(self.generation, next_seq);
        self.storage.commit(&[
            ReplayWrite::Put {
                key: delta_key.clone(),
                value: encode_row(
                    &ReplayDeltaRow {
                        tag: DELTA_TAG.to_string(),
                        generation: self.generation,
                        seq: next_seq,
                        operation,
                    },
                    "delta",
                )?,
            },
            ReplayWrite::Put {
                key: HEAD_KEY.to_vec(),
                value: encode_head(self.generation, next_seq, self.capacity)?,
            },
        ])?;
        apply_admission(&mut self.heap, admission, entry);
        self.delta_seq = next_seq;
        self.live_keys.push(delta_key);
        Ok(true)
    }

    pub fn sample_batch(&self, n: usize, seed: u64) -> Vec<ReplayEntry> {
        let mut candidates = self.entries_by_priority();
        if n >= candidates.len() {
            return candidates;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sampled = Vec::with_capacity(n);
        while sampled.len() < n && !candidates.is_empty() {
            let total: f64 = candidates.iter().map(|entry| entry.surprise).sum();
            let index = if total > 0.0 {
                weighted_index(&candidates, rng.random_range(0.0..total))
            } else {
                0
            };
            sampled.push(candidates.remove(index));
        }
        sampled
    }

    /// Seeds the heap in memory and publishes one atomic checkpoint. This is
    /// one durable commit regardless of how many recent mistakes are admitted.
    pub fn seed_from_log<M>(&mut self, log: &MistakeLog<M>, n: usize) -> Result<usize>
    where
        M: MistakeStorage,
    {
        let mut next_heap = self.heap.clone();
        let mut accepted = 0;
        for row in log.readback_recent(n)? {
            let entry = ReplayEntry::from_mistake(row.seq, &row.entry)?;
            if push_into_heap(&mut next_heap, self.capacity, entry)? {
                accepted += 1;
            }
        }
        if accepted != 0 {
            self.persist_heap_checkpoint(&next_heap)?;
            self.heap = next_heap;
        }
        Ok(accepted)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn entries_by_priority(&self) -> Vec<ReplayEntry> {
        entries_by_priority(&self.heap)
    }

    pub fn top_surprises(&self, n: usize) -> Vec<f64> {
        self.entries_by_priority()
            .into_iter()
            .take(n)
            .map(|entry| entry.surprise)
            .collect()
    }

    pub fn snapshot(&self) -> ReplaySnapshot {
        ReplaySnapshot {
            capacity: self.capacity,
            entries: self.entries_by_priority(),
        }
    }

    pub fn entry(
        &self,
        cx_id: CxId,
        target: f64,
        surprise: f64,
        mistake_ref: MistakeRef,
    ) -> Result<ReplayEntry> {
        ReplayEntry::new(cx_id, target, surprise, mistake_ref, self.clock.now())
    }

    fn persist_checkpoint(&mut self) -> Result<()> {
        let heap = self.heap.clone();
        self.persist_heap_checkpoint(&heap)
    }

    fn persist_heap_checkpoint(&mut self, heap: &BinaryHeap<Reverse<ReplayEntry>>) -> Result<()> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_row("replay checkpoint generation overflow"))?;
        let checkpoint_key = replay_checkpoint_key(generation);
        let snapshot = ReplaySnapshot {
            capacity: self.capacity,
            entries: entries_by_priority(heap),
        };
        let mut writes = Vec::with_capacity(self.live_keys.len() + 2);
        for key in &self.live_keys {
            if key.as_slice() != HEAD_KEY {
                writes.push(ReplayWrite::Delete { key: key.clone() });
            }
        }
        writes.push(ReplayWrite::Put {
            key: checkpoint_key.clone(),
            value: encode_row(
                &ReplayCheckpointRow {
                    tag: CHECKPOINT_TAG.to_string(),
                    generation,
                    snapshot,
                },
                "checkpoint",
            )?,
        });
        writes.push(ReplayWrite::Put {
            key: HEAD_KEY.to_vec(),
            value: encode_head(generation, 0, self.capacity)?,
        });
        self.storage.commit(&writes)?;
        self.generation = generation;
        self.delta_seq = 0;
        self.live_keys = vec![HEAD_KEY.to_vec(), checkpoint_key];
        Ok(())
    }
}

impl ReplayEntry {
    pub fn new(
        cx_id: CxId,
        target: f64,
        surprise: f64,
        mistake_ref: MistakeRef,
        added_ts: LogicalTime,
    ) -> Result<Self> {
        let entry = Self {
            cx_id,
            target,
            surprise,
            mistake_ref,
            added_ts,
        };
        validate_entry(&entry)?;
        Ok(entry)
    }

    pub fn from_mistake(seq: u64, entry: &MistakeEntry) -> Result<Self> {
        Self::new(
            entry.cx_id,
            entry.observed,
            entry.surprise,
            MistakeRef {
                seq,
                surprise: entry.surprise,
            },
            entry.ts,
        )
    }
}

impl PartialEq for ReplayEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cx_id == other.cx_id
            && self.target.to_bits() == other.target.to_bits()
            && self.surprise.to_bits() == other.surprise.to_bits()
            && self.mistake_ref.seq == other.mistake_ref.seq
            && self.mistake_ref.surprise.to_bits() == other.mistake_ref.surprise.to_bits()
            && self.added_ts == other.added_ts
    }
}

impl Eq for ReplayEntry {}

impl PartialOrd for ReplayEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReplayEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.surprise
            .total_cmp(&other.surprise)
            .then_with(|| other.added_ts.cmp(&self.added_ts))
            .then_with(|| other.mistake_ref.seq.cmp(&self.mistake_ref.seq))
            .then_with(|| self.cx_id.as_bytes().cmp(other.cx_id.as_bytes()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayAdmission {
    Reject,
    Add,
    ReplaceMin,
}

fn push_into_heap(
    heap: &mut BinaryHeap<Reverse<ReplayEntry>>,
    capacity: usize,
    entry: ReplayEntry,
) -> Result<bool> {
    let admission = heap_admission(heap, capacity, &entry)?;
    if admission == ReplayAdmission::Reject {
        return Ok(false);
    }
    apply_admission(heap, admission, entry);
    Ok(true)
}

fn heap_admission(
    heap: &BinaryHeap<Reverse<ReplayEntry>>,
    capacity: usize,
    entry: &ReplayEntry,
) -> Result<ReplayAdmission> {
    validate_capacity(capacity)?;
    if heap.len() < capacity {
        return Ok(ReplayAdmission::Add);
    }
    let Some(min_entry) = heap.peek().map(|entry| &entry.0) else {
        return Ok(ReplayAdmission::Reject);
    };
    Ok(if entry.cmp(min_entry) == Ordering::Greater {
        ReplayAdmission::ReplaceMin
    } else {
        ReplayAdmission::Reject
    })
}

fn apply_admission(
    heap: &mut BinaryHeap<Reverse<ReplayEntry>>,
    admission: ReplayAdmission,
    entry: ReplayEntry,
) {
    match admission {
        ReplayAdmission::Reject => {}
        ReplayAdmission::Add => heap.push(Reverse(entry)),
        ReplayAdmission::ReplaceMin => {
            heap.pop();
            heap.push(Reverse(entry));
        }
    }
}

fn weighted_index(entries: &[ReplayEntry], draw: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, entry) in entries.iter().enumerate() {
        cumulative += entry.surprise;
        if draw < cumulative {
            return index;
        }
    }
    entries.len().saturating_sub(1)
}

fn validate_capacity(capacity: usize) -> Result<()> {
    if capacity == 0 {
        return Err(CalyxError {
            code: CALYX_ANNEAL_INVALID_CAPACITY,
            message: "replay buffer capacity must be > 0".to_string(),
            remediation: "configure a positive anneal replay capacity",
        });
    }
    Ok(())
}

fn validate_entry(entry: &ReplayEntry) -> Result<()> {
    if !entry.target.is_finite() {
        return Err(invalid_row("replay target must be finite"));
    }
    if !entry.surprise.is_finite() || entry.surprise < 0.0 {
        return Err(invalid_row("replay surprise must be finite and >= 0"));
    }
    if !entry.mistake_ref.surprise.is_finite() || entry.mistake_ref.surprise < 0.0 {
        return Err(invalid_row(
            "replay mistake_ref surprise must be finite and >= 0",
        ));
    }
    if entry.mistake_ref.seq == 0 {
        return Err(invalid_row("replay mistake_ref seq must be > 0"));
    }
    if entry.surprise.to_bits() != entry.mistake_ref.surprise.to_bits() {
        return Err(invalid_row("replay surprise must match mistake_ref"));
    }
    Ok(())
}

#[cfg(test)]
mod fsv_tests;
