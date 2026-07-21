use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use calyx_core::{CxFlags, InputRef, LedgerRef, Modality, SlotId, SlotVector, SystemClock};
use serde_json::json;

use super::*;

fn issue1547_vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        .parse()
        .expect("valid vault id")
}

fn synthetic_constellation(
    vault: &AsterVault<SystemClock>,
    scenario: &str,
    index: usize,
) -> Constellation {
    let input = format!("issue1547:{scenario}:{index}").into_bytes();
    let mut slots = BTreeMap::new();
    slots.insert(
        SlotId::new(0),
        SlotVector::Dense {
            dim: 2,
            data: vec![index as f32, 1.0],
        },
    );
    Constellation {
        cx_id: vault.cx_id_for_input(&input, 1_547),
        vault_id: issue1547_vault_id(),
        panel_version: 1_547,
        created_at: index as u64,
        input_ref: InputRef {
            hash: *blake3::hash(&input).as_bytes(),
            pointer: Some(format!("synthetic://issue1547/{scenario}/{index}")),
            redacted: false,
        },
        modality: Modality::Text,
        slots,
        scalars: BTreeMap::new(),
        metadata: BTreeMap::new(),
        anchors: Vec::new(),
        provenance: LedgerRef {
            seq: 0,
            hash: [0; 32],
        },
        flags: CxFlags::default(),
    }
}

#[test]
#[ignore = "manual durable-SST #1547 scaling FSV; requires CALYX_FSV_ROOT"]
fn issue1547_durable_sst_scaling_has_one_base_read_per_unique_id() {
    let root = PathBuf::from(
        std::env::var("CALYX_FSV_ROOT")
            .expect("CALYX_FSV_ROOT must name the durable evidence directory"),
    );
    let vault_dir = root.join("issue1547-scaling-vault");
    fs::remove_dir_all(&vault_dir).ok();
    fs::create_dir_all(&root).expect("create issue1547 FSV root");
    let vault = AsterVault::new_durable(
        &vault_dir,
        issue1547_vault_id(),
        b"issue1547-durable-scaling".to_vec(),
        VaultOptions::default(),
    )
    .expect("create durable issue1547 vault");

    let mut evidence = Vec::new();
    let mut expected_base_rows = 0;
    for size in [1_usize, 100, 1_024] {
        let all_new = (0..size)
            .map(|index| synthetic_constellation(&vault, &format!("all-new-{size}"), index))
            .collect::<Vec<_>>();
        let before_rows = vault
            .scan_cf_at(vault.latest_seq(), ColumnFamily::Base)
            .expect("scan Base before all-new")
            .len();
        batch_ingest::reset_batch_read_counts();
        let outcomes = vault
            .put_batch_with_outcomes(all_new.clone())
            .expect("put all-new batch");
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.disposition == PutDisposition::Inserted)
        );
        assert_eq!(batch_ingest::batch_read_counts(), (size, 1));
        expected_base_rows += size;
        let after_rows = vault
            .scan_cf_at(vault.latest_seq(), ColumnFamily::Base)
            .expect("scan Base after all-new")
            .len();
        assert_eq!(after_rows, before_rows + size);
        vault.flush().expect("flush all-new rows to SST");
        evidence.push(json!({
            "size": size,
            "scenario": "all_new",
            "before_base_rows": before_rows,
            "after_base_rows": after_rows,
            "base_lookups": size,
            "snapshot_pins": 1,
        }));

        batch_ingest::reset_batch_read_counts();
        let seq_before_existing = vault.latest_seq();
        let existing = vault
            .put_batch_with_outcomes(all_new.clone())
            .expect("put all-existing batch");
        assert!(
            existing
                .iter()
                .all(|outcome| outcome.disposition == PutDisposition::ExistingIdentical)
        );
        assert_eq!(batch_ingest::batch_read_counts(), (size, 1));
        assert_eq!(vault.latest_seq(), seq_before_existing);
        evidence.push(json!({
            "size": size,
            "scenario": "all_existing",
            "before_seq": seq_before_existing,
            "after_seq": vault.latest_seq(),
            "base_lookups": size,
            "snapshot_pins": 1,
        }));

        let existing_count = size / 2;
        let mut mixed = all_new[..existing_count].to_vec();
        mixed.extend(
            (existing_count..size)
                .map(|index| synthetic_constellation(&vault, &format!("mixed-new-{size}"), index)),
        );
        batch_ingest::reset_batch_read_counts();
        let mixed_outcomes = vault
            .put_batch_with_outcomes(mixed)
            .expect("put mixed batch");
        assert_eq!(batch_ingest::batch_read_counts(), (size, 1));
        assert!(
            mixed_outcomes[..existing_count]
                .iter()
                .all(|outcome| { outcome.disposition == PutDisposition::ExistingIdentical })
        );
        assert!(
            mixed_outcomes[existing_count..]
                .iter()
                .all(|outcome| { outcome.disposition == PutDisposition::Inserted })
        );
        expected_base_rows += size - existing_count;
        evidence.push(json!({
            "size": size,
            "scenario": "mixed",
            "existing": existing_count,
            "inserted": size - existing_count,
            "base_lookups": size,
            "snapshot_pins": 1,
        }));

        let repeated = synthetic_constellation(&vault, &format!("duplicates-{size}"), 0);
        batch_ingest::reset_batch_read_counts();
        let duplicate_outcomes = vault
            .put_batch_with_outcomes(std::iter::repeat_n(repeated, size))
            .expect("put high-duplication batch");
        assert_eq!(batch_ingest::batch_read_counts(), (1, 1));
        assert_eq!(duplicate_outcomes[0].disposition, PutDisposition::Inserted);
        assert!(
            duplicate_outcomes[1..].iter().all(|outcome| matches!(
                outcome.disposition,
                PutDisposition::InBatchDuplicate { .. }
            ))
        );
        expected_base_rows += 1;
        evidence.push(json!({
            "size": size,
            "scenario": "high_within_batch_duplication",
            "unique_ids": 1,
            "base_lookups": 1,
            "snapshot_pins": 1,
        }));
        vault.flush().expect("flush scenario rows to SST");
    }

    assert_eq!(expected_base_rows, 1_691);
    drop(vault);
    let reopened = AsterVault::open(
        &vault_dir,
        issue1547_vault_id(),
        b"issue1547-durable-scaling".to_vec(),
        VaultOptions::default(),
    )
    .expect("cold-open issue1547 vault");
    let persisted_rows = reopened
        .scan_cf_at(reopened.latest_seq(), ColumnFamily::Base)
        .expect("scan Base after cold reopen")
        .len();
    assert_eq!(persisted_rows, expected_base_rows);
    let base_files = fs::read_dir(vault_dir.join("cf").join("base"))
        .expect("read physical Base CF directory")
        .map(|entry| {
            let path = entry.expect("Base CF entry").path();
            json!({
                "path": path.display().to_string(),
                "bytes": fs::metadata(&path).expect("Base CF metadata").len(),
            })
        })
        .collect::<Vec<_>>();
    assert!(!base_files.is_empty());

    let readback = json!({
        "issue": 1547,
        "source_of_truth": vault_dir.display().to_string(),
        "expected_base_rows": expected_base_rows,
        "cold_reopen_base_rows": persisted_rows,
        "base_cf_files": base_files,
        "scenarios": evidence,
    });
    let evidence_path = root.join("issue1547-scaling-readback.json");
    fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&readback).expect("serialize issue1547 evidence"),
    )
    .expect("write issue1547 evidence");
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&evidence_path).expect("reread issue1547 evidence"))
            .expect("decode issue1547 evidence");
    assert_eq!(persisted["cold_reopen_base_rows"], expected_base_rows);
    eprintln!(
        "ISSUE1547_SCALING source_of_truth={} cold_reopen_base_rows={} scenarios={} max_batch=1024",
        vault_dir.display(),
        persisted_rows,
        persisted["scenarios"].as_array().map(Vec::len).unwrap_or(0),
    );
}
