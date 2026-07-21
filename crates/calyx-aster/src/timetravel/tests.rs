use super::*;
use crate::cf::ColumnFamily;
use crate::timetravel::time_index::encode_key;
use crate::vault::VaultOptions;
use calyx_core::{
    Clock, Constellation, CxFlags, CxId, InputRef, LedgerRef, Modality, SlotId, SlotVector, Ts,
    VaultId, VaultStore,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// A clock the test advances between commits so each group-commit is stamped
/// with a known wall-clock millisecond.
struct StepClock(AtomicU64);

impl StepClock {
    fn new(start: Ts) -> Self {
        Self(AtomicU64::new(start))
    }
    fn set(&self, t: Ts) {
        self.0.store(t, Ordering::SeqCst);
    }
}

impl Clock for StepClock {
    fn now(&self) -> Ts {
        self.0.load(Ordering::SeqCst)
    }
}

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
}

fn vault_at(t: Ts) -> AsterVault<StepClock> {
    AsterVault::with_clock(vault_id(), b"timetravel", StepClock::new(t))
}

/// Builds a one-slot constellation whose dense vector encodes `tag`, so a
/// time-travel read can be byte-compared against the value at ingest time.
fn constellation(vault: &AsterVault<StepClock>, input: &[u8], tag: f32) -> Constellation {
    let cx_id = vault.cx_id_for_input(input, 1);
    let mut input_hash = [0_u8; 32];
    input_hash[..input.len().min(32)].copy_from_slice(&input[..input.len().min(32)]);
    let mut slots = BTreeMap::new();
    slots.insert(
        SlotId::new(0),
        SlotVector::Dense {
            dim: 2,
            data: vec![tag, tag + 1.0],
        },
    );
    Constellation {
        cx_id,
        vault_id: vault_id(),
        panel_version: 1,
        created_at: 10,
        input_ref: InputRef {
            hash: input_hash,
            pointer: None,
            redacted: false,
        },
        modality: Modality::Text,
        slots,
        scalars: BTreeMap::new(),
        metadata: BTreeMap::new(),
        anchors: Vec::new(),
        provenance: LedgerRef {
            seq: 1,
            hash: [7; 32],
        },
        flags: CxFlags {
            ungrounded: true,
            ..CxFlags::default()
        },
    }
}

fn ingest(vault: &AsterVault<StepClock>, input: &[u8], tag: f32, at: Ts) -> CxId {
    vault.clock_set(at);
    let cx = constellation(vault, input, tag);
    let id = cx.cx_id;
    vault.put(cx).expect("ingest");
    id
}

impl AsterVault<StepClock> {
    fn clock_set(&self, t: Ts) {
        self.clock_ref().set(t);
    }
}

#[test]
fn as_of_excludes_writes_after_the_timestamp() {
    let vault = vault_at(0);
    let c1 = ingest(&vault, b"c1", 1.0, 1000);
    let c2 = ingest(&vault, b"c2", 2.0, 2000);

    // as_of(1500): C1 present, C2 absent (it had not been written yet).
    let snap = vault.as_of(1500).expect("as_of 1500");
    assert!(snap.get_cx(c1).is_ok(), "C1 must exist at t=1500");
    let missing = snap.get_cx(c2).unwrap_err();
    assert!(
        missing.code == "CALYX_STALE_DERIVED" || missing.code == "CALYX_CX_NOT_FOUND",
        "C2 must be reported missing at t=1500, got {}",
        missing.code
    );
}

#[test]
fn as_of_at_later_time_includes_both() {
    let vault = vault_at(0);
    let c1 = ingest(&vault, b"c1", 1.0, 1000);
    let c2 = ingest(&vault, b"c2", 2.0, 2000);
    let snap = vault.as_of(2000).expect("as_of 2000");
    assert!(snap.get_cx(c1).is_ok());
    assert!(snap.get_cx(c2).is_ok());
}

#[test]
fn as_of_returns_pre_mutation_bytes() {
    // Constellations are content-addressed and immutable (re-putting the same
    // CxId with different bytes is correctly rejected as a collision), so
    // time-travel over an in-place mutation is proven on a mutable KV row.
    let vault = vault_at(0);
    vault.clock_set(1000);
    vault
        .write_cf(ColumnFamily::Graph, b"k".to_vec(), b"v1".to_vec())
        .expect("write v1");
    vault.clock_set(2000);
    vault
        .write_cf(ColumnFamily::Graph, b"k".to_vec(), b"v2".to_vec())
        .expect("write v2");

    // as_of(1500) reads the pre-mutation byte value; as_of(2000) reads the new.
    let past = vault.as_of(1500).expect("as_of 1500");
    assert_eq!(
        past.read_cf(ColumnFamily::Graph, b"k").unwrap(),
        Some(b"v1".to_vec()),
        "time-travel must see pre-mutation bytes"
    );
    let now = vault.as_of(2000).expect("as_of 2000");
    assert_eq!(
        now.read_cf(ColumnFamily::Graph, b"k").unwrap(),
        Some(b"v2".to_vec())
    );
}

#[test]
fn deterministic_prefix_property_each_t_sees_exactly_k() {
    // The proptest property, run deterministically: after k monotonic ingests,
    // as_of(t_k) sees exactly the first k constellations.
    let vault = vault_at(0);
    let mut ids = Vec::new();
    for k in 1..=12u64 {
        let input = format!("cx{k}");
        ids.push(ingest(&vault, input.as_bytes(), k as f32, k * 1000));
    }
    for k in 1..=12u64 {
        let snap = vault.as_of(k * 1000).expect("as_of");
        for (i, id) in ids.iter().enumerate() {
            let present = snap.get_cx(*id).is_ok();
            let expected = (i as u64) < k; // first k present
            assert_eq!(
                present,
                expected,
                "at t={}, cx#{} present={present} expected={expected}",
                k * 1000,
                i + 1
            );
        }
    }
}

#[test]
fn as_of_before_any_write_is_no_data() {
    let vault = vault_at(0);
    ingest(&vault, b"c1", 1.0, 1000);
    let err = vault.as_of(0).unwrap_err();
    assert_eq!(err.code, "CALYX_TIMETRAVEL_NO_DATA");
}

#[test]
fn single_write_boundary() {
    let vault = vault_at(0);
    let c1 = ingest(&vault, b"only", 1.0, 500);
    assert_eq!(
        vault.as_of(499).unwrap_err().code,
        "CALYX_TIMETRAVEL_NO_DATA"
    );
    let snap = vault.as_of(500).expect("as_of 500");
    assert!(snap.get_cx(c1).is_ok());
}

#[test]
fn absolute_horizon_fails_closed_before_inclusive_boundary() {
    let vault = vault_at(0);
    vault
        .set_retention_horizon(RetentionHorizon::absolute(5000))
        .expect("set horizon");
    let c1 = ingest(&vault, b"horizon-boundary", 1.0, 5000);

    let before = vault.as_of(4999).unwrap_err();
    assert_eq!(before.code, CALYX_TIMETRAVEL_BEFORE_HORIZON);
    assert!(before.message.contains("requested_millis=4999"));
    assert!(before.message.contains("horizon_millis=5000"));

    let at_boundary = vault.as_of(5000).expect("inclusive horizon boundary");
    assert!(at_boundary.get_cx(c1).is_ok());
}

#[test]
fn none_horizon_preserves_no_data_error() {
    let vault = vault_at(0);
    ingest(&vault, b"no-data", 1.0, 500);
    vault
        .set_retention_horizon(RetentionHorizon::none())
        .expect("default none is valid");
    let err = vault.as_of(0).unwrap_err();
    assert_eq!(err.code, "CALYX_TIMETRAVEL_NO_DATA");
}

#[test]
fn rolling_zero_horizon_rejects_anything_before_now() {
    let vault = vault_at(10_000);
    vault
        .set_retention_horizon(RetentionHorizon::rolling(Duration::ZERO))
        .expect("set rolling zero");
    let c1 = ingest(&vault, b"rolling-zero", 1.0, 10_000);

    let before = vault.as_of(9_999).unwrap_err();
    assert_eq!(before.code, CALYX_TIMETRAVEL_BEFORE_HORIZON);
    let at_now = vault.as_of(10_000).expect("as_of now");
    assert!(at_now.get_cx(c1).is_ok());
}

#[test]
fn rolling_horizon_uses_current_clock_and_saturating_math() {
    let vault = vault_at(10_000);
    vault
        .set_retention_horizon(RetentionHorizon::rolling(Duration::from_secs(1)))
        .expect("set rolling horizon");
    let c1 = ingest(&vault, b"rolling", 1.0, 9_000);
    vault.clock_set(10_000);

    let before = vault.as_of(8_999).unwrap_err();
    assert_eq!(before.code, CALYX_TIMETRAVEL_BEFORE_HORIZON);
    assert!(before.message.contains("horizon_millis=9000"));
    let at_horizon = vault.as_of(9_000).expect("rolling inclusive horizon");
    assert!(at_horizon.get_cx(c1).is_ok());
}

#[test]
fn dropped_snapshot_releases_its_pin() {
    let vault = vault_at(0);
    ingest(&vault, b"c1", 1.0, 1000);
    let snap = vault.as_of(1000).expect("as_of");
    let lease = snap.lease_id_for_test();
    drop(snap);
    // After drop the lease is already gone: releasing it again is a no-op. If
    // drop had leaked the pin, this would return true.
    assert!(
        !vault.release_reader(lease),
        "drop must have released the lease pin"
    );
}

#[test]
fn corrupt_time_index_key_fails_closed() {
    let vault = vault_at(0);
    ingest(&vault, b"c1", 1.0, 1000);
    let before = time_index::read_all(&vault).expect("read valid index");
    // Raw TimeIndex mutation is forbidden at the commit boundary. This keeps
    // malformed logical rows out of the source of truth rather than relying on
    // every optimized reader to rediscover corruption later.
    let error = vault
        .write_cf(
            ColumnFamily::TimeIndex,
            vec![0xff; 15],
            time_index::SENTINEL.to_vec(),
        )
        .expect_err("reserved TimeIndex mutation must fail closed");
    assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
    assert_eq!(time_index::read_all(&vault).unwrap(), before);
    assert!(vault.as_of(u64::MAX).is_ok());
}

#[test]
fn time_index_has_one_entry_per_committed_seq() {
    let vault = vault_at(0);
    ingest(&vault, b"c1", 1.0, 1000);
    ingest(&vault, b"c2", 2.0, 2000);
    let entries = read_all(&vault).expect("read time index");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].millis, 1000);
    assert_eq!(entries[1].millis, 2000);
    // Each entry's seqno resolves to a real snapshot.
    assert_eq!(encode_key(entries[0].millis, entries[0].seqno).len(), 16);
}

#[test]
#[ignore = "manual full-state verification for issue #1533"]
fn issue1533_manual_fsv_reads_bounded_predecessor_after_cold_reopen() {
    let root = std::env::var_os("CALYX_ISSUE1533_FSV_ROOT")
        .map(std::path::PathBuf::from)
        .expect("set CALYX_ISSUE1533_FSV_ROOT to a fresh path");
    assert!(!root.exists(), "FSV root must be fresh: {}", root.display());

    let vault = AsterVault::new_durable_with_clock(
        &root,
        vault_id(),
        b"issue1533-bounded-predecessor-fsv".to_vec(),
        VaultOptions::default(),
        StepClock::new(0),
    )
    .expect("open durable FSV vault");
    let before = serde_json::json!({
        "latest_seq": vault.latest_seq(),
        "time_index_rows": read_all(&vault).expect("read empty index").len(),
    });

    let before_first_error = vault.as_of(0).expect_err("empty index has no predecessor");
    let edge_empty_after = serde_json::json!({
        "error_code": before_first_error.code,
        "latest_seq": vault.latest_seq(),
        "time_index_rows": read_all(&vault).expect("empty index remains readable").len(),
    });

    let c1 = ingest(&vault, b"issue1533-c1", 1.0, 1_000);
    let before_boundary_error = vault
        .as_of(999)
        .expect_err("one millisecond before first commit has no predecessor");
    let exact = vault
        .as_of(1_000)
        .expect("exact boundary resolves first row");
    assert!(exact.get_cx(c1).is_ok());
    let edge_boundary_after = serde_json::json!({
        "before_error_code": before_boundary_error.code,
        "exact_seq": exact.seqno(),
        "exact_contains_c1": exact.get_cx(c1).is_ok(),
        "time_index_rows": read_all(&vault).expect("read one row").len(),
    });
    drop(exact);

    let c2 = ingest(&vault, b"issue1533-c2", 2.0, 2_000);
    let middle = vault.as_of(1_500).expect("floor resolves first row");
    assert!(middle.get_cx(c1).is_ok());
    assert!(middle.get_cx(c2).is_err());
    let committed_time_index = read_all(&vault)
        .expect("read committed index")
        .into_iter()
        .map(|entry| serde_json::json!({ "millis": entry.millis, "seqno": entry.seqno }))
        .collect::<Vec<_>>();
    let happy_after = serde_json::json!({
        "resolved_seq": middle.seqno(),
        "contains_c1": middle.get_cx(c1).is_ok(),
        "contains_c2": middle.get_cx(c2).is_ok(),
        "time_index": committed_time_index,
    });
    drop(middle);

    let max = vault
        .as_of(u64::MAX)
        .expect("maximum timestamp uses a non-overflowing predecessor target");
    assert!(max.get_cx(c1).is_ok());
    assert!(max.get_cx(c2).is_ok());
    let edge_max_after = serde_json::json!({
        "resolved_seq": max.seqno(),
        "contains_c1": max.get_cx(c1).is_ok(),
        "contains_c2": max.get_cx(c2).is_ok(),
    });
    drop(max);
    vault
        .flush()
        .expect("flush time-index rows to physical SSTs");
    drop(vault);

    let reopened = AsterVault::open_with_clock(
        &root,
        vault_id(),
        b"issue1533-bounded-predecessor-fsv".to_vec(),
        VaultOptions::default(),
        StepClock::new(2_001),
    )
    .expect("cold reopen durable FSV vault");
    let reopened_snapshot = reopened
        .as_of(1_500)
        .expect("cold reopened predecessor resolves first row");
    assert!(reopened_snapshot.get_cx(c1).is_ok());
    assert!(reopened_snapshot.get_cx(c2).is_err());
    let physical_time_index_files = std::fs::read_dir(root.join("cf").join("time_index"))
        .expect("read physical TimeIndex directory")
        .map(|entry| {
            entry
                .expect("physical TimeIndex entry")
                .path()
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert!(!physical_time_index_files.is_empty());

    let reopened_time_index = read_all(&reopened)
        .expect("read cold reopened index")
        .into_iter()
        .map(|entry| serde_json::json!({ "millis": entry.millis, "seqno": entry.seqno }))
        .collect::<Vec<_>>();
    let report = serde_json::json!({
        "issue": 1533,
        "source_of_truth": root.display().to_string(),
        "before": before,
        "edge_empty_after": edge_empty_after,
        "edge_boundary_after": edge_boundary_after,
        "happy_after": happy_after,
        "edge_max_after": edge_max_after,
        "cold_reopen_after": {
            "resolved_seq": reopened_snapshot.seqno(),
            "contains_c1": reopened_snapshot.get_cx(c1).is_ok(),
            "contains_c2": reopened_snapshot.get_cx(c2).is_ok(),
            "time_index": reopened_time_index,
        },
        "physical_time_index_files": physical_time_index_files,
    });
    let report_path = root.join("issue1533-fsv.json");
    std::fs::write(
        &report_path,
        serde_json::to_vec_pretty(&report).expect("encode FSV report"),
    )
    .expect("persist FSV report");
    let readback: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path).expect("read FSV report from source of truth"),
    )
    .expect("decode FSV report readback");
    assert_eq!(readback, report);
    eprintln!("ISSUE1533_FSV={report}");
}
