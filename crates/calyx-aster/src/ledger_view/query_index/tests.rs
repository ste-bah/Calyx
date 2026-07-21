use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use calyx_core::{CxId, VaultId};
use calyx_ledger::{ActorId, EntryKind, LedgerCfStore, SubjectId};

use super::*;
use crate::ledger_view::{AsterLedgerCfStore, visit_ledger_reverse};
use crate::vault::{AsterVault, VaultOptions};

#[test]
fn reverse_query_stops_at_logical_limit_on_one_stable_physical_snapshot() {
    let root = test_vault_dir("issue1532-reverse-ledger");
    let vault = AsterVault::new_durable(
        &root,
        vault_id(),
        b"issue1532-reverse-ledger-salt",
        VaultOptions::default(),
    )
    .expect("open durable vault");
    for seq in 0..300u64 {
        vault
            .append_ledger_entry(
                EntryKind::Admin,
                SubjectId::Query(format!("unrelated-{seq}").into_bytes()),
                format!("payload-{seq}").into_bytes(),
                ActorId::System,
            )
            .expect("append physical ledger row");
    }
    vault.flush().expect("flush ledger SST rows");

    let mut visited = Vec::new();
    let stats = visit_ledger_reverse(&root, 32, |entry| {
        visited.push(entry.seq);
        Ok(visited.len() == 17)
    })
    .expect("bounded reverse query");

    assert_eq!(stats.snapshot_height, 300);
    assert_eq!(stats.batches_read, 1);
    assert_eq!(stats.rows_visited, 17);
    assert_eq!(visited, (283..300).rev().collect::<Vec<_>>());
    let physical = AsterLedgerCfStore::open(&root)
        .expect("open source-of-truth ledger")
        .scan()
        .expect("read source-of-truth rows");
    assert_eq!(physical.len(), 300);
    assert_eq!(physical.last().map(|row| row.seq), Some(299));

    fs::remove_dir_all(root).ok();
}

#[test]
fn query_index_reuses_generation_and_extends_only_the_new_tail() {
    let root = test_vault_dir("issue1532-ledger-query-index");
    let vault = AsterVault::new_durable(
        &root,
        vault_id(),
        b"issue1532-ledger-query-index-salt",
        VaultOptions::default(),
    )
    .expect("open durable vault");
    let target = CxId::from_bytes([0x53; 16]);
    for seq in 0..300u64 {
        let (subject, payload) = if seq == 173 {
            (
                SubjectId::Cx(target),
                serde_json::to_vec(&serde_json::json!({"cx_id": target.to_string()})).unwrap(),
            )
        } else {
            (
                SubjectId::Query(format!("unrelated-{seq}").into_bytes()),
                b"malformed unrelated payload is not JSON".to_vec(),
            )
        };
        vault
            .append_ledger_entry(EntryKind::Admin, subject, payload, ActorId::System)
            .expect("append indexed ledger row");
    }

    let first = LedgerQuerySnapshot::open(&root).expect("build query index");
    assert_eq!(first.height(), 300);
    assert_eq!(first.open_stats().rows_indexed, 300);
    assert_eq!(first.open_stats().rows_reused, 0);
    assert!(first.open_stats().index_rebuilt);
    let matches = first.entries_for_cx(target).expect("targeted CX query");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].seq, 173);

    let reused = LedgerQuerySnapshot::open(&root).expect("reuse exact query generation");
    assert!(!reused.open_stats().index_rebuilt);
    assert_eq!(reused.open_stats().rows_indexed, 0);
    assert_eq!(reused.open_stats().rows_reused, 300);

    vault
        .append_ledger_entry(
            EntryKind::Ingest,
            SubjectId::Cx(target),
            serde_json::to_vec(&serde_json::json!({"cx_id": target.to_string()})).unwrap(),
            ActorId::System,
        )
        .expect("append one tail row");
    let extended = LedgerQuerySnapshot::open(&root).expect("extend query index");
    assert!(extended.open_stats().index_rebuilt);
    assert_eq!(extended.open_stats().rows_indexed, 1);
    assert_eq!(extended.open_stats().rows_reused, 300);
    assert_eq!(extended.entries_for_cx(target).unwrap().len(), 2);
    assert_eq!(
        fs::read_dir(root.join("ledger_query_index"))
            .unwrap()
            .count(),
        1,
        "publishing a new generation must reclaim the old side index"
    );

    let mut visited = Vec::new();
    let stats = extended
        .visit_kind_reverse(EntryKind::Ingest, 16, |entry| {
            visited.push(entry.seq);
            Ok(false)
        })
        .expect("visit indexed kind");
    assert_eq!(visited, vec![300]);
    assert_eq!(stats.matching_rows_visited, 1);
    assert!(stats.physical_rows_read <= 2);

    fs::remove_dir_all(root).ok();
}

#[test]
fn empty_subject_byte_query_returns_empty_against_real_ledger_rows() {
    let root = test_vault_dir("issue1567-empty-subject-bytes");
    let vault = AsterVault::new_durable(
        &root,
        vault_id(),
        b"issue1567-empty-subject-bytes-salt",
        VaultOptions::default(),
    )
    .expect("open durable vault");
    let before = AsterLedgerCfStore::open(&root)
        .expect("open physical ledger before append")
        .scan()
        .expect("scan physical ledger before append");
    println!("ISSUE1567_BEFORE rows={} seqs=[]", before.len());
    vault
        .append_ledger_entry(
            EntryKind::Admin,
            SubjectId::Query(b"present-answer".to_vec()),
            b"present-payload".to_vec(),
            ActorId::System,
        )
        .expect("append durable ledger row");
    vault.flush().expect("flush durable ledger row");

    let snapshot = LedgerQuerySnapshot::open(&root).expect("build query index");
    let missing = snapshot
        .entries_for_subject_bytes(b"missing-answer")
        .expect("empty subject-byte query should be valid");
    println!(
        "ISSUE1567_QUERY index_height={} missing_subject_rows={}",
        snapshot.height(),
        missing.len()
    );
    assert!(
        missing.is_empty(),
        "absent subject-byte query must return an empty posting list"
    );

    let physical = AsterLedgerCfStore::open(&root)
        .expect("open physical ledger source of truth")
        .scan()
        .expect("scan physical ledger source of truth");
    println!(
        "ISSUE1567_AFTER rows={} seqs={:?}",
        physical.len(),
        physical.iter().map(|row| row.seq).collect::<Vec<_>>()
    );
    assert_eq!(physical.len(), 1);
    assert_eq!(physical[0].seq, 0);

    fs::remove_dir_all(root).ok();
}

#[test]
fn query_index_corruption_fails_closed_without_rebuilding() {
    let root = test_vault_dir("issue1532-ledger-query-corrupt");
    let vault = AsterVault::new_durable(
        &root,
        vault_id(),
        b"issue1532-ledger-query-corrupt-salt",
        VaultOptions::default(),
    )
    .expect("open durable vault");
    vault
        .append_ledger_entry(
            EntryKind::Admin,
            SubjectId::Query(b"answer".to_vec()),
            b"payload".to_vec(),
            ActorId::System,
        )
        .expect("append ledger row");
    LedgerQuerySnapshot::open(&root).expect("build query index");
    let index_path = fs::read_dir(root.join("ledger_query_index"))
        .expect("read index directory")
        .map(|entry| entry.expect("index entry").path())
        .find(|path| path.extension().is_some_and(|extension| extension == "idx"))
        .expect("index generation exists");
    let mut bytes = fs::read(&index_path).expect("read index generation");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x80;
    fs::write(&index_path, bytes).expect("corrupt index generation");

    let error = LedgerQuerySnapshot::open(&root).expect_err("corrupt index must fail closed");
    assert_eq!(error.code, "CALYX_LEDGER_CORRUPT");
    assert!(error.message.contains("checksum mismatch"));

    fs::remove_dir_all(root).ok();
}

fn test_vault_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("calyx-aster-{name}-{}-{nanos}", std::process::id()))
}

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
}
