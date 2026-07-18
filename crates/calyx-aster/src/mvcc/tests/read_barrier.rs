use super::*;
use crate::cf::{ColumnFamily, KeyRange};

#[test]
fn read_barrier_blocks_point_batch_and_scan_reads_until_removed() {
    let clock = FixedClock::new(100);
    let store = VersionedCfStore::default();
    let blocked = base_key(cx(0x10));
    let outside = base_key(cx(0x20));
    let seq = store
        .commit_batch([
            (ColumnFamily::Base, blocked.clone(), b"blocked".to_vec()),
            (ColumnFamily::Base, outside.clone(), b"outside".to_vec()),
        ])
        .unwrap();
    let snapshot = Snapshot::new(
        seq,
        Freshness::FreshDerived,
        ReaderLease::new(1, seq, 100, 1_000),
    );
    let range = KeyRange {
        start: blocked.clone(),
        end: Some(base_key(cx(0x11))),
    };

    store.install_read_barrier(ReadBarrier::base_corrupt("shard_10", range));

    let error = store
        .read_at(snapshot, ColumnFamily::Base, &blocked, &clock)
        .expect_err("blocked point read fails closed");
    assert_eq!(error.code, CALYX_ASTER_BASE_CORRUPT);
    assert_eq!(
        store
            .read_at(snapshot, ColumnFamily::Base, &outside, &clock)
            .unwrap(),
        Some(b"outside".to_vec())
    );
    assert_eq!(
        store
            .read_batch(
                snapshot,
                &[CfRead::new(ColumnFamily::Base, blocked.clone())],
                &clock,
            )
            .expect_err("blocked batch fails")
            .code,
        CALYX_ASTER_BASE_CORRUPT
    );
    assert_eq!(
        store
            .scan_cf_at(snapshot, ColumnFamily::Base, &clock)
            .expect_err("blocked scan fails")
            .code,
        CALYX_ASTER_BASE_CORRUPT
    );

    assert!(store.remove_read_barrier("shard_10"));
    assert_eq!(
        store
            .read_at(snapshot, ColumnFamily::Base, &blocked, &clock)
            .unwrap(),
        Some(b"blocked".to_vec())
    );
}

#[test]
fn read_batch_fails_closed_for_first_middle_or_last_blocked_key() {
    for blocked_index in 0..3 {
        let clock = FixedClock::new(100);
        let store = VersionedCfStore::default();
        let keys = [b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        store
            .commit_batch(keys.iter().map(|key| {
                (
                    ColumnFamily::Base,
                    key.clone(),
                    [b"value-".as_slice(), key].concat(),
                )
            }))
            .unwrap();
        let snapshot = store.pin_snapshot(Freshness::FreshDerived, &clock, 1_000);
        let blocked = keys[blocked_index].clone();
        let mut end = blocked.clone();
        end.push(0);
        store.install_read_barrier(ReadBarrier::base_corrupt(
            format!("blocked-{blocked_index}"),
            KeyRange {
                start: blocked,
                end: Some(end),
            },
        ));
        let reads = keys
            .iter()
            .map(|key| CfRead::new(ColumnFamily::Base, key.clone()))
            .collect::<Vec<_>>();
        let phases_before = store.batch_read_phase_counts();
        println!("MVCC_BARRIER_BEFORE blocked_index={blocked_index} phases={phases_before:?}");
        let error = store.read_batch(snapshot, &reads, &clock).unwrap_err();
        let phases_after = store.batch_read_phase_counts();
        assert_eq!(error.code, CALYX_ASTER_BASE_CORRUPT);
        assert_eq!(phases_after.0 - phases_before.0, 1);
        assert_eq!(
            phases_after.1 - phases_before.1,
            0,
            "blocked batch returns before reading any row"
        );
        println!(
            "MVCC_BARRIER_AFTER blocked_index={blocked_index} code={} row_phase_delta=0",
            error.code
        );
    }
}
