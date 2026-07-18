use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use calyx_aster::cf::ColumnFamily;
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{Clock, CxId, FixedClock, VaultId};
use serde_json::json;

use super::*;

#[test]
#[ignore = "manual full-state verification for issue #1534"]
fn issue1534_manual_fsv_reads_replay_rows_after_cold_reopen() {
    let root = std::env::var_os("CALYX_ISSUE1534_FSV_ROOT")
        .map(PathBuf::from)
        .expect("set CALYX_ISSUE1534_FSV_ROOT to a fresh path");
    assert!(!root.exists(), "FSV root must be fresh: {}", root.display());
    let vault_id: VaultId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
    let salt = b"issue1534-replay-delta-fsv".to_vec();
    let clock: Arc<dyn Clock> = Arc::new(FixedClock::new(1_534_000));
    let vault = AsterVault::new_durable_with_clock(
        &root,
        vault_id,
        salt.clone(),
        VaultOptions::default(),
        FixedClock::new(1_534_000),
    )
    .expect("open durable replay FSV vault");
    let mut buffer =
        ReplayBuffer::open_with_checkpoint_interval(AsterReplayStorage::new(&vault), 2, 2, clock)
            .unwrap();
    let before = live_rows(&vault);
    assert!(before.is_empty());

    let empty_sample = buffer.sample_batch(1, 7);
    let after_empty = live_rows(&vault);
    assert!(empty_sample.is_empty() && after_empty.is_empty());

    let invalid = buffer
        .push(ReplayEntry {
            cx_id: CxId::from_bytes([0xee; 16]),
            target: 0.0,
            surprise: f64::NAN,
            mistake_ref: MistakeRef {
                seq: 1,
                surprise: f64::NAN,
            },
            added_ts: 1_534_001,
        })
        .expect_err("NaN entry fails before storage");
    let after_invalid = live_rows(&vault);
    assert!(after_invalid.is_empty());

    buffer.push(entry(1, 0.5, 1)).unwrap();
    let checkpoint_rows = live_rows(&vault);
    assert_eq!(checkpoint_rows.len(), 2);
    let checkpoint_value = checkpoint_rows
        .iter()
        .find(|(key, _)| key.starts_with(b"checkpoint/v3/"))
        .map(|(_, value)| value.clone())
        .unwrap();
    buffer.push(entry(2, 0.6, 2)).unwrap();
    let delta_rows = live_rows(&vault);
    assert_eq!(delta_rows.len(), 3);
    assert!(
        delta_rows
            .iter()
            .any(|(key, _)| key.starts_with(b"delta/v3/"))
    );
    assert_eq!(
        delta_rows
            .iter()
            .find(|(key, _)| key.starts_with(b"checkpoint/v3/"))
            .map(|(_, value)| value),
        Some(&checkpoint_value),
        "ordinary admission must not rewrite the checkpoint"
    );

    let rejected = buffer.push(entry(3, 0.1, 3)).unwrap();
    let after_rejected = live_rows(&vault);
    assert!(!rejected);
    assert_eq!(after_rejected, delta_rows);

    buffer.push(entry(4, 0.9, 4)).unwrap();
    let after_checkpoint = live_rows(&vault);
    assert_eq!(after_checkpoint.len(), 2);
    assert!(
        !after_checkpoint
            .iter()
            .any(|(key, _)| key.starts_with(b"delta/v3/"))
    );
    let expected = buffer.snapshot();
    vault.flush().expect("flush replay rows to physical SSTs");
    drop(buffer);
    drop(vault);

    let reopened = AsterVault::open_with_clock(
        &root,
        vault_id,
        salt,
        VaultOptions::default(),
        FixedClock::new(1_534_100),
    )
    .expect("cold reopen physical vault");
    let reopened_buffer = ReplayBuffer::open(
        AsterReplayStorage::new(&reopened),
        2,
        Arc::new(FixedClock::new(1_534_100)),
    )
    .expect("recover checkpoint/delta state");
    assert_eq!(reopened_buffer.snapshot(), expected);
    let reopened_rows = live_rows(&reopened);
    let report = json!({
        "issue": 1534,
        "source_of_truth": root.display().to_string(),
        "before": row_report(&before),
        "edge_empty_after": row_report(&after_empty),
        "edge_invalid_after": {"error_code": invalid.code, "rows": row_report(&after_invalid)},
        "happy_checkpoint_after": row_report(&checkpoint_rows),
        "happy_delta_after": row_report(&delta_rows),
        "edge_rejected_after": {"accepted": rejected, "rows": row_report(&after_rejected)},
        "periodic_checkpoint_after": row_report(&after_checkpoint),
        "cold_reopen_rows": row_report(&reopened_rows),
        "cold_reopen_snapshot": reopened_buffer.snapshot(),
        "sst_files": physical_files(&root.join("cf").join("anneal_replay")),
        "wal_files": physical_files(&root.join("wal")),
    });
    let artifact = root.join("issue1534-fsv.json");
    fs::write(&artifact, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    let readback: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifact).unwrap()).unwrap();
    assert_eq!(readback["cold_reopen_rows"].as_array().unwrap().len(), 2);
    println!(
        "ISSUE1534_FSV={}",
        serde_json::to_string(&readback).unwrap()
    );
}

fn entry(byte: u8, surprise: f64, seq: u64) -> ReplayEntry {
    ReplayEntry::new(
        CxId::from_bytes([byte; 16]),
        surprise,
        surprise,
        MistakeRef { seq, surprise },
        1_534_000 + seq,
    )
    .unwrap()
}

fn live_rows<C: Clock>(vault: &AsterVault<C>) -> Vec<(Vec<u8>, Vec<u8>)> {
    vault
        .scan_cf_at(vault.latest_seq(), ColumnFamily::AnnealReplay)
        .unwrap()
}

fn row_report(rows: &[(Vec<u8>, Vec<u8>)]) -> Vec<serde_json::Value> {
    rows.iter()
        .map(|(key, value)| {
            json!({
                "key": String::from_utf8_lossy(key),
                "value_len": value.len(),
                "value_hash": blake3::hash(value).to_hex().to_string(),
            })
        })
        .collect()
}

fn physical_files(dir: &std::path::Path) -> Vec<String> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let mut files = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path().display().to_string())
        .collect::<Vec<_>>();
    files.sort();
    files
}
