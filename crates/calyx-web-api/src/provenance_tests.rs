use super::*;
use crate::metrics::probe_ledger;
use crate::provenance::provenance_body_from_sources;
use calyx_core::CalyxError;
use calyx_ledger::{ActorId, EntryKind, LedgerRow, SubjectId, encode};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingLedgerStore {
    rows: Vec<LedgerRow>,
    scans: AtomicUsize,
}

impl CountingLedgerStore {
    fn one_valid_row() -> Self {
        let entry = LedgerEntry::new(
            0,
            [0; 32],
            EntryKind::Ingest,
            SubjectId::Query(b"synthetic-source".to_vec()),
            br#"{"input":"known"}"#.to_vec(),
            ActorId::Service("web-perf-fsv".to_string()),
            1,
        );
        Self {
            rows: vec![LedgerRow {
                seq: 0,
                bytes: encode(&entry),
            }],
            scans: AtomicUsize::new(0),
        }
    }

    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            scans: AtomicUsize::new(0),
        }
    }

    fn corrupt() -> Self {
        Self {
            rows: vec![LedgerRow {
                seq: 0,
                bytes: vec![0xff, 0x00],
            }],
            scans: AtomicUsize::new(0),
        }
    }

    fn valid_rows(count: usize) -> Self {
        let mut previous = [0; 32];
        let rows = (0..count)
            .map(|seq| {
                let entry = LedgerEntry::new(
                    seq as u64,
                    previous,
                    EntryKind::Ingest,
                    SubjectId::Query(format!("source-{seq}").into_bytes()),
                    br#"{"input":"known"}"#.to_vec(),
                    ActorId::Service("web-perf-fsv".to_string()),
                    seq as u64 + 1,
                );
                previous = entry.entry_hash;
                LedgerRow {
                    seq: seq as u64,
                    bytes: encode(&entry),
                }
            })
            .collect();
        Self {
            rows,
            scans: AtomicUsize::new(0),
        }
    }
}

impl LedgerCfStore for CountingLedgerStore {
    fn scan(&self) -> calyx_core::Result<Vec<LedgerRow>> {
        self.scans.fetch_add(1, Ordering::Relaxed);
        Ok(self.rows.clone())
    }

    fn put_new(&mut self, seq: u64, _bytes: &[u8]) -> calyx_core::Result<()> {
        Err(CalyxError::ledger_append_only_violation(format!(
            "counting read store rejected append at {seq}"
        )))
    }
}

#[test]
fn provenance_computation_acquires_ledger_rows_once() {
    let store = CountingLedgerStore::one_valid_row();
    println!("PROVENANCE_SNAPSHOT_BEFORE rows=1 scans=0 answer=known-missing-answer");
    let body = provenance_body_from_sources(
        &store,
        &QuarantineSet::default(),
        "known-missing-answer".to_string(),
    )
    .expect("provenance computation");

    assert_eq!(store.scans.load(Ordering::Relaxed), 1);
    assert_eq!(body["found"], false);
    assert_eq!(body["trusted"], false);
    assert_eq!(body["chain"]["result"], "intact");
    assert_eq!(body["chain"]["count"], 1);
    println!(
        "PROVENANCE_SNAPSHOT_AFTER rows=1 scans={} found={} chain={}",
        store.scans.load(Ordering::Relaxed),
        body["found"],
        body["chain"]["result"]
    );
}

#[test]
fn metrics_computation_acquires_ledger_rows_once() {
    let store = CountingLedgerStore::one_valid_row();
    println!("METRICS_SNAPSHOT_BEFORE rows=1 scans=0");
    assert_eq!(probe_ledger(&store), (1, 1, 1, -1));
    assert_eq!(store.scans.load(Ordering::Relaxed), 1);
    println!("METRICS_SNAPSHOT_AFTER rows=1 scans=1 intact=1");
}

#[test]
fn provenance_snapshot_edges_cover_empty_and_corrupt_rows() {
    let empty = CountingLedgerStore::empty();
    println!("PROVENANCE_EDGE_EMPTY_BEFORE rows=0 scans=0");
    let body = provenance_body_from_sources(&empty, &QuarantineSet::default(), "none".to_string())
        .expect("empty ledger has a valid empty chain");
    assert_eq!(empty.scans.load(Ordering::Relaxed), 1);
    assert_eq!(body["chain"]["count"], 0);
    println!("PROVENANCE_EDGE_EMPTY_AFTER rows=0 scans=1 chain_count=0");

    let corrupt = CountingLedgerStore::corrupt();
    println!("PROVENANCE_EDGE_CORRUPT_BEFORE rows=1 scans=0 bytes=ff00");
    assert!(
        provenance_body_from_sources(&corrupt, &QuarantineSet::default(), "none".to_string(),)
            .is_err(),
        "corrupt row must fail closed"
    );
    assert_eq!(corrupt.scans.load(Ordering::Relaxed), 1);
    println!("PROVENANCE_EDGE_CORRUPT_AFTER rows=1 scans=1 result=error");
}

#[test]
#[ignore = "manual provenance snapshot scaling FSV"]
fn provenance_snapshot_scaling_fsv() {
    for row_count in [1, 1_000, 100_000] {
        let store = CountingLedgerStore::valid_rows(row_count);
        let ledger_bytes = store.rows.iter().map(|row| row.bytes.len()).sum::<usize>();
        let started = Instant::now();
        let body = provenance_body_from_sources(
            &store,
            &QuarantineSet::default(),
            "missing-answer".to_string(),
        )
        .expect("scaled provenance");
        assert_eq!(store.scans.load(Ordering::Relaxed), 1);
        assert_eq!(body["chain"]["count"], row_count as u64);
        println!(
            "PROVENANCE_SCALE rows={} ledger_bytes={} scans=1 chain_count={} elapsed_ms={}",
            row_count,
            ledger_bytes,
            body["chain"]["count"],
            started.elapsed().as_millis()
        );
    }
}
