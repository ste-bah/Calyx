use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use calyx_anneal::{
    AsterMistakeStorage, AsterReplayStorage, CALYX_ANNEAL_INVALID_CAPACITY,
    CALYX_ANNEAL_REPLAY_INVALID_ROW, MistakeLog, MistakeRef, ReplayBuffer, ReplayEntry,
    ReplayStorage, ReplayWrite, decode_replay_snapshot, encode_replay_snapshot,
    replay_snapshot_key,
};
use calyx_aster::cf::ColumnFamily;
use calyx_aster::vault::AsterVault;
use calyx_core::{AnchorKind, Clock, CxId, FixedClock, Result};
use proptest::prelude::*;

// calyx-shared-module: path=fsv_support/mod.rs alias=__calyx_shared_fsv_support_mod_rs local=fsv_support visibility=private

use crate::__calyx_shared_fsv_support_mod_rs as fsv_support;
use fsv_support::vault_id;

#[test]
fn capacity_evicts_lowest_surprise() {
    let mut buffer = memory_buffer(2, 200);

    assert!(buffer.push(entry(1, 0.8, 1)).unwrap());
    assert!(buffer.push(entry(2, 0.3, 2)).unwrap());
    assert!(!buffer.push(entry(3, 0.1, 3)).unwrap());

    assert_eq!(buffer.len(), 2);
    assert_eq!(buffer.top_surprises(5), vec![0.8, 0.3]);
    let entries = buffer.entries_by_priority();
    assert_eq!(entries[0].cx_id, cx(1));
    assert_eq!(entries[1].cx_id, cx(2));
}

#[test]
fn sample_batch_is_seeded_and_read_only() {
    let mut buffer = memory_buffer(5, 201);
    for (idx, surprise) in [0.1, 0.2, 0.3, 0.4, 0.5].into_iter().enumerate() {
        buffer
            .push(entry((idx + 1) as u8, surprise, (idx + 1) as u64))
            .unwrap();
    }

    let first = buffer.sample_batch(2, 42);
    let second = buffer.sample_batch(2, 42);
    let third = buffer.sample_batch(2, 7);

    assert_eq!(first, second);
    assert_ne!(first, third);
    assert_eq!(buffer.len(), 5);
}

#[test]
fn capacity_one_replaces_only_on_higher_surprise() {
    let mut buffer = memory_buffer(1, 202);

    assert!(buffer.push(entry(1, 0.2, 1)).unwrap());
    assert!(!buffer.push(entry(2, 0.2, 2)).unwrap());
    assert_eq!(buffer.entries_by_priority()[0].mistake_ref.seq, 1);
    assert!(buffer.push(entry(3, 0.9, 3)).unwrap());
    assert_eq!(buffer.entries_by_priority()[0].mistake_ref.seq, 3);
}

#[test]
fn empty_and_overwide_sampling_edges_are_safe() {
    let mut buffer = memory_buffer(3, 203);

    assert!(buffer.sample_batch(2, 42).is_empty());
    buffer.push(entry(1, 0.6, 1)).unwrap();
    buffer.push(entry(2, 0.4, 2)).unwrap();

    let all = buffer.sample_batch(10, 42);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].surprise, 0.6);
    assert_eq!(all[1].surprise, 0.4);
}

#[test]
fn zero_capacity_fails_closed() {
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(204));
    let err = ReplayBuffer::open(MemoryReplayStorage::default(), 0, clock)
        .err()
        .unwrap();

    assert_eq!(err.code, CALYX_ANNEAL_INVALID_CAPACITY);
}

#[test]
fn seed_from_log_replays_recent_mistakes_without_log_feedback() {
    let vault = AsterVault::with_clock(vault_id(), b"issue407-replay-seed", FixedClock::new(205));
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(205));
    let log = MistakeLog::open(AsterMistakeStorage::new(&vault), 10, clock.clone()).unwrap();
    log.append(cx(1), 0.9, 0.1, AnchorKind::Reward).unwrap();
    log.append(cx(2), 0.7, 0.3, AnchorKind::Reward).unwrap();
    log.append(cx(3), 0.5, 0.5, AnchorKind::Reward).unwrap();
    let log_rows_before = log.readback_recent(10).unwrap().len();
    let storage = MemoryReplayStorage::default();
    let mut buffer = ReplayBuffer::open(storage.clone(), 2, clock).unwrap();

    let accepted = buffer.seed_from_log(&log, 3).unwrap();

    assert_eq!(accepted, 2);
    assert_eq!(
        storage.commit_count(),
        1,
        "bulk seed must publish one checkpoint"
    );
    assert_eq!(buffer.top_surprises(5), vec![0.8, 0.39999999999999997]);
    assert_eq!(
        buffer
            .entries_by_priority()
            .into_iter()
            .map(|entry| entry.target)
            .collect::<Vec<_>>(),
        vec![0.1, 0.3]
    );
    assert_eq!(log.readback_recent(10).unwrap().len(), log_rows_before);
}

#[test]
fn legacy_snapshot_without_target_fails_closed() {
    #[derive(serde::Serialize)]
    struct LegacyEntry {
        cx_id: CxId,
        surprise: f64,
        mistake_ref: MistakeRef,
        added_ts: u64,
    }

    #[derive(serde::Serialize)]
    struct LegacySnapshot {
        capacity: usize,
        entries: Vec<LegacyEntry>,
    }

    #[derive(serde::Serialize)]
    struct LegacyRow {
        tag: String,
        snapshot: LegacySnapshot,
    }

    let mut bytes = Vec::new();
    ciborium::ser::into_writer(
        &LegacyRow {
            tag: "anneal_replay_snapshot_v1".to_string(),
            snapshot: LegacySnapshot {
                capacity: 1,
                entries: vec![LegacyEntry {
                    cx_id: cx(1),
                    surprise: 0.5,
                    mistake_ref: MistakeRef {
                        seq: 1,
                        surprise: 0.5,
                    },
                    added_ts: 1001,
                }],
            },
        },
        &mut bytes,
    )
    .unwrap();

    let error = decode_replay_snapshot(&bytes).unwrap_err();
    assert_eq!(error.code, CALYX_ANNEAL_REPLAY_INVALID_ROW);
}

#[test]
fn aster_storage_writes_cbor_snapshot_under_anneal_replay_cf() {
    let vault = AsterVault::with_clock(vault_id(), b"issue407-replay", FixedClock::new(206));
    let storage = AsterReplayStorage::new(&vault);
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(206));
    let mut buffer = ReplayBuffer::open(storage, 2, clock).unwrap();

    buffer.push(entry(9, 0.75, 1)).unwrap();

    let rows = vault
        .scan_cf_at(vault.latest_seq(), ColumnFamily::AnnealReplay)
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().any(|(key, _)| key == b"head/v3"));
    assert!(
        rows.iter()
            .any(|(key, _)| key.starts_with(b"checkpoint/v3/"))
    );
    let snapshot = ReplayBuffer::open(
        AsterReplayStorage::new(&vault),
        2,
        Arc::new(FixedClock::new(207)),
    )
    .unwrap()
    .snapshot();
    assert_eq!(snapshot.capacity, 2);
    assert_eq!(snapshot.entries.len(), 1);
    assert_eq!(snapshot.entries[0].cx_id, cx(9));
    assert_eq!(snapshot.entries[0].target, 0.75);
    assert_eq!(snapshot.entries[0].surprise, 0.75);
}

#[test]
fn admitted_pushes_append_deltas_rejections_write_nothing_and_checkpoint_reclaims() {
    let storage = MemoryReplayStorage::default();
    let mut buffer = ReplayBuffer::open_with_checkpoint_interval(
        storage.clone(),
        2,
        2,
        Arc::new(FixedClock::new(208)),
    )
    .unwrap();

    assert!(buffer.push(entry(1, 0.5, 1)).unwrap());
    assert_eq!(storage.commit_count(), 1);
    assert_eq!(storage.row_keys().len(), 2);
    assert!(buffer.push(entry(2, 0.6, 2)).unwrap());
    assert_eq!(storage.commit_count(), 2);
    assert_eq!(storage.row_keys().len(), 3);
    assert!(!buffer.push(entry(3, 0.1, 3)).unwrap());
    assert_eq!(storage.commit_count(), 2, "rejected pushes must not write");
    assert!(buffer.push(entry(4, 0.9, 4)).unwrap());
    assert_eq!(storage.commit_count(), 3);
    let keys = storage.row_keys();
    assert_eq!(
        keys.len(),
        2,
        "checkpoint must reclaim prior live generation"
    );
    assert!(keys.iter().any(|key| key == b"head/v3"));
    assert!(keys.iter().any(|key| key.starts_with(b"checkpoint/v3/")));

    let reopened = ReplayBuffer::open(storage, 2, Arc::new(FixedClock::new(209))).unwrap();
    assert_eq!(reopened.top_surprises(4), vec![0.9, 0.6]);
}

#[test]
fn legacy_v2_snapshot_migrates_atomically_and_missing_delta_fails_closed() {
    let legacy = MemoryReplayStorage::default();
    legacy
        .commit(&[ReplayWrite::Put {
            key: replay_snapshot_key(),
            value: encode_replay_snapshot(&calyx_anneal::ReplaySnapshot {
                capacity: 2,
                entries: vec![entry(7, 0.7, 7)],
            })
            .unwrap(),
        }])
        .unwrap();
    let migrated = ReplayBuffer::open(legacy.clone(), 2, Arc::new(FixedClock::new(210))).unwrap();
    assert_eq!(migrated.top_surprises(2), vec![0.7]);
    assert!(!legacy.row_keys().contains(&replay_snapshot_key()));
    assert_eq!(legacy.row_keys().len(), 2);

    let corrupt = MemoryReplayStorage::default();
    let mut buffer = ReplayBuffer::open_with_checkpoint_interval(
        corrupt.clone(),
        3,
        10,
        Arc::new(FixedClock::new(211)),
    )
    .unwrap();
    buffer.push(entry(1, 0.4, 1)).unwrap();
    buffer.push(entry(2, 0.5, 2)).unwrap();
    let delta = corrupt
        .row_keys()
        .into_iter()
        .find(|key| key.starts_with(b"delta/v3/"))
        .unwrap();
    corrupt
        .commit(&[ReplayWrite::Delete { key: delta }])
        .unwrap();
    let error = ReplayBuffer::open(corrupt, 3, Arc::new(FixedClock::new(212)))
        .err()
        .unwrap();
    assert_eq!(error.code, CALYX_ANNEAL_REPLAY_INVALID_ROW);
    assert!(error.message.contains("missing delta"));
}

proptest! {
    #![proptest_config(calyx_testkit::integration_proptest_config(256))]

    #[test]
    fn len_never_exceeds_capacity(
        surprises in proptest::collection::vec(0.0f64..1.0, 0..80),
        capacity in 1usize..20,
    ) {
        let mut buffer = memory_buffer(capacity, 207);
        for (idx, surprise) in surprises.into_iter().enumerate() {
            buffer.push(entry((idx % 255) as u8, surprise, idx as u64 + 1)).unwrap();
            prop_assert!(buffer.len() <= capacity);
        }
    }
}

#[derive(Clone, Default)]
struct MemoryReplayStorage {
    rows: Arc<Mutex<BTreeMap<Vec<u8>, Vec<u8>>>>,
    commits: Arc<Mutex<Vec<Vec<ReplayWrite>>>>,
}

impl MemoryReplayStorage {
    fn commit_count(&self) -> usize {
        self.commits.lock().unwrap().len()
    }

    fn row_keys(&self) -> Vec<Vec<u8>> {
        self.rows.lock().unwrap().keys().cloned().collect()
    }
}

impl ReplayStorage for MemoryReplayStorage {
    fn scan_rows(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        Ok(self
            .rows
            .lock()
            .unwrap()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect())
    }

    fn commit(&self, writes: &[ReplayWrite]) -> Result<()> {
        let mut rows = self.rows.lock().unwrap();
        for write in writes {
            match write {
                ReplayWrite::Put { key, value } => {
                    rows.insert(key.clone(), value.clone());
                }
                ReplayWrite::Delete { key } => {
                    rows.remove(key);
                }
            }
        }
        self.commits.lock().unwrap().push(writes.to_vec());
        Ok(())
    }
}

fn memory_buffer(capacity: usize, ts: u64) -> ReplayBuffer<MemoryReplayStorage> {
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(ts));
    ReplayBuffer::open(MemoryReplayStorage::default(), capacity, clock).unwrap()
}

fn entry(byte: u8, surprise: f64, seq: u64) -> ReplayEntry {
    ReplayEntry::new(
        cx(byte),
        surprise,
        surprise,
        MistakeRef { seq, surprise },
        1000 + seq,
    )
    .unwrap()
}

fn cx(byte: u8) -> CxId {
    CxId::from_bytes([byte; 16])
}
