use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use calyx_core::{CxId, SlotId, VaultId};
use calyx_ledger::decode as decode_ledger;
use serde_json::json;

use super::{OrphanReconciler, VaultOrphanGcTarget, orphan_io_counts, reset_orphan_io_counts};
use crate::cf::{ColumnFamily, slot_key};
use crate::vault::{AsterVault, VaultOptions};

const SETUP_ROWS_PER_COMMIT: usize = 10_000;
const SETUP_ROWS_PER_FLUSH: usize = 50_000;

#[test]
#[ignore = "manual durable 1K/100K/1M #1548 scale FSV; requires CALYX_FSV_ROOT"]
fn issue1548_durable_linear_grouping_scale_and_cold_readback() {
    let root = PathBuf::from(
        std::env::var("CALYX_FSV_ROOT")
            .expect("CALYX_FSV_ROOT must name the durable evidence directory"),
    );
    fs::create_dir_all(&root).expect("create issue1548 FSV root");
    let mut scenarios = Vec::new();

    for (total_rows, repair_limit) in [(1_000_usize, 1_usize), (100_000, 100), (1_000_000, 1_000)] {
        let vault_dir = root.join(format!("issue1548-{total_rows}-row-vault"));
        fs::remove_dir_all(&vault_dir).ok();
        let salt = format!("issue1548-durable-scale-{total_rows}").into_bytes();
        let vault = AsterVault::new_durable(
            &vault_dir,
            issue1548_vault_id(),
            salt.clone(),
            VaultOptions::default(),
        )
        .expect("create durable issue1548 vault");
        let slot = SlotId::new(0);

        let setup_started = Instant::now();
        for start in (0..total_rows).step_by(SETUP_ROWS_PER_COMMIT) {
            let end = (start + SETUP_ROWS_PER_COMMIT).min(total_rows);
            vault
                .write_cf_batch((start..end).map(|index| {
                    (
                        ColumnFamily::slot(slot),
                        slot_key(scale_cx(index)),
                        (index as u64).to_be_bytes().to_vec(),
                    )
                }))
                .expect("write durable orphan setup rows");
            if end % SETUP_ROWS_PER_FLUSH == 0 || end == total_rows {
                vault.flush().expect("flush durable orphan setup rows");
            }
        }
        let setup_ms = millis(setup_started.elapsed());
        let before_seq = vault.latest_seq();
        let before_rows = vault
            .scan_cf_at(before_seq, ColumnFamily::slot(slot))
            .expect("scan physical Slot rows before repair")
            .len();
        assert_eq!(before_rows, total_rows);
        eprintln!(
            "ISSUE1548_SCALE before total_rows={total_rows} repair_limit={repair_limit} slot_rows={before_rows} seq={before_seq}"
        );

        let target = VaultOrphanGcTarget::new(&vault, [slot]).without_auto_compaction();
        let reconciler = OrphanReconciler::new(Duration::ZERO, repair_limit);
        let scan_started = Instant::now();
        let report = reconciler
            .scan(&target)
            .expect("scan durable orphan report");
        let scan_ms = millis(scan_started.elapsed());
        assert_eq!(report.orphan_index.len(), total_rows);
        assert_eq!(report.orphan_index_entries.len(), total_rows);
        assert_eq!(report.inconsistencies, total_rows);

        reset_orphan_io_counts();
        let repair_started = Instant::now();
        let repaired = reconciler
            .repair(&target, &report)
            .expect("repair bounded durable orphan prefix");
        let repair_ms = millis(repair_started.elapsed());
        let counts = orphan_io_counts();
        assert_eq!(repaired.orphan_index_repaired, repair_limit);
        assert_eq!(
            repaired.remaining_inconsistencies,
            total_rows - repair_limit
        );
        assert_eq!(counts.report_entry_visits, repair_limit);
        assert_eq!(counts.ledger_entries, repair_limit);
        assert_eq!(counts.group_commits, repair_limit.div_ceil(128));
        assert!(counts.compaction_calls.is_empty());
        assert!(
            vault
                .read_cf_at(
                    vault.latest_seq(),
                    ColumnFamily::slot(slot),
                    &slot_key(scale_cx(0)),
                )
                .expect("read first repaired Slot row")
                .is_none()
        );
        assert!(
            vault
                .read_cf_at(
                    vault.latest_seq(),
                    ColumnFamily::slot(slot),
                    &slot_key(scale_cx(repair_limit)),
                )
                .expect("read first retained Slot row")
                .is_some()
        );
        vault.flush().expect("flush durable orphan repairs");
        drop(target);
        drop(vault);

        let reopened = AsterVault::open(
            &vault_dir,
            issue1548_vault_id(),
            salt,
            VaultOptions::default(),
        )
        .expect("cold-open durable issue1548 vault");
        let cold_slot_rows = reopened
            .scan_cf_at(reopened.latest_seq(), ColumnFamily::slot(slot))
            .expect("scan Slot CF after cold reopen")
            .len();
        let cold_ledger = reopened
            .scan_cf_at(reopened.latest_seq(), ColumnFamily::Ledger)
            .expect("scan Ledger CF after cold reopen");
        let cold_ledger_rows = cold_ledger.len();
        let cold_orphan_audits = cold_ledger
            .iter()
            .filter_map(|(_, bytes)| decode_ledger(bytes).ok())
            .filter(|entry| {
                serde_json::from_slice::<serde_json::Value>(&entry.payload)
                    .ok()
                    .and_then(|payload| payload["event"].as_str().map(str::to_owned))
                    .as_deref()
                    == Some("orphan_index_purged")
            })
            .count();
        assert_eq!(cold_slot_rows, total_rows - repair_limit);
        assert_eq!(cold_orphan_audits, repair_limit);
        let physical_files = physical_files(&vault_dir);
        assert!(physical_files.iter().any(|file| {
            file["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(".sst"))
        }));
        eprintln!(
            "ISSUE1548_SCALE after total_rows={total_rows} repaired={repair_limit} cold_slot_rows={cold_slot_rows} cold_ledger_rows={cold_ledger_rows} cold_orphan_audits={cold_orphan_audits} report_visits={} commits={} scan_ms={scan_ms} repair_ms={repair_ms}",
            counts.report_entry_visits, counts.group_commits,
        );
        scenarios.push(json!({
            "total_rows": total_rows,
            "repair_limit": repair_limit,
            "before_slot_rows": before_rows,
            "cold_reopen_slot_rows": cold_slot_rows,
            "cold_reopen_ledger_rows": cold_ledger_rows,
            "cold_reopen_orphan_audits": cold_orphan_audits,
            "setup_ms": setup_ms,
            "scan_ms": scan_ms,
            "repair_ms": repair_ms,
            "report_entry_visits": counts.report_entry_visits,
            "point_reads": counts.point_reads,
            "group_commits": counts.group_commits,
            "ledger_commits": counts.ledger_commits,
            "flushes": counts.flushes,
            "committed_rows": counts.committed_rows,
            "committed_bytes": counts.committed_bytes,
            "max_chunk_rows": counts.max_chunk_rows,
            "max_chunk_bytes": counts.max_chunk_bytes,
            "physical_files": physical_files,
        }));
    }

    let evidence = json!({
        "issue": 1548,
        "source_of_truth": root.display().to_string(),
        "scenarios": scenarios,
    });
    let evidence_path = root.join("issue1548-scaling-readback.json");
    fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).expect("serialize issue1548 evidence"),
    )
    .expect("write issue1548 evidence");
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&evidence_path).expect("reread issue1548 evidence"))
            .expect("decode issue1548 evidence");
    assert_eq!(persisted["scenarios"].as_array().map(Vec::len), Some(3));
    assert_eq!(persisted["scenarios"][2]["total_rows"], 1_000_000);
    assert_eq!(persisted["scenarios"][2]["cold_reopen_slot_rows"], 999_000);
    eprintln!(
        "ISSUE1548_SCALE_SOURCE_OF_TRUTH path={} scenarios=3 max_rows=1000000",
        evidence_path.display()
    );
}

fn scale_cx(index: usize) -> CxId {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&(index as u64).to_be_bytes());
    CxId::from_bytes(bytes)
}

fn issue1548_vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        .parse()
        .expect("valid issue1548 vault id")
}

fn millis(elapsed: Duration) -> u128 {
    elapsed.as_millis()
}

fn physical_files(root: &Path) -> Vec<serde_json::Value> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("read issue1548 physical directory") {
            let path = entry.expect("read issue1548 physical entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(json!({
                    "path": path.display().to_string(),
                    "bytes": fs::metadata(&path).expect("read issue1548 file metadata").len(),
                }));
            }
        }
    }
    files.sort_by(|left, right| left["path"].as_str().cmp(&right["path"].as_str()));
    files
}
