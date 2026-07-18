use super::*;

#[test]
fn durable_memtable_oversize_rejects_before_wal_append() {
    let fsv_root = calyx_fsv::fsv_root("CALYX_FSV_ROOT");
    let dir = fsv_root.as_ref().map_or_else(
        || test_dir("memtable-oversize-preflight"),
        |root| {
            let dir = root.join("memtable-oversize-preflight").join("vault");
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("create fsv vault");
            dir
        },
    );
    let options = VaultOptions {
        memtable_byte_cap: 16,
        ..VaultOptions::default()
    };
    let vault =
        AsterVault::new_durable(&dir, vault_id(), b"salt".to_vec(), options).expect("open durable");

    let before_wal_bytes = wal_bytes(&dir);
    let before_seq = vault.snapshot();
    let error = vault
        .write_cf(ColumnFamily::Base, b"k".to_vec(), vec![0xAA; 64])
        .expect_err("oversize row rejects before WAL");

    assert_eq!(error.code, "CALYX_BACKPRESSURE");
    assert_eq!(vault.snapshot(), before_seq);
    assert_eq!(wal_bytes(&dir), before_wal_bytes);
    drop(vault);

    let reopened = AsterVault::open(&dir, vault_id(), b"salt".to_vec(), VaultOptions::default())
        .expect("cold open after rejected write");

    assert_eq!(reopened.snapshot(), before_seq);
    assert_eq!(
        reopened
            .read_cf_at(reopened.snapshot(), ColumnFamily::Base, b"k")
            .unwrap(),
        None
    );
    if let Some(root) = fsv_root {
        let readback = serde_json::json!({
            "error_code": error.code,
            "snapshot_before": before_seq,
            "cold_open_snapshot": reopened.snapshot(),
            "wal_bytes_before": before_wal_bytes,
            "wal_bytes_after": wal_bytes(&dir),
            "rejected_key_visible": false,
        });
        fs::write(
            root.join("memtable-oversize-preflight-readback.json"),
            serde_json::to_vec_pretty(&readback).unwrap(),
        )
        .unwrap();
    } else {
        cleanup(dir);
    }
}

#[test]
#[ignore = "manual FSV for PH35 ledger group-commit WAL rows"]
fn ph35_ledger_group_commit_manual_fsv() {
    let root = fsv_root().join("group-commit-hook");
    reset_dir(&root);
    let vault_dir = root.join("vault");
    let vault = AsterVault::new_durable(
        &vault_dir,
        vault_id(),
        b"salt".to_vec(),
        VaultOptions::default(),
    )
    .expect("open durable");
    let cx = sample_constellation(&AsterVault::with_clock(
        vault_id(),
        b"salt".to_vec(),
        FixedClock::new(123),
    ));
    let id = cx.cx_id;

    let before = vault
        .read_cf_at(0, ColumnFamily::Ledger, &ledger_key(0))
        .expect("read before");
    vault.put(cx).expect("durable put");
    vault.flush().expect("flush durable");

    let wal_path = vault_dir.join("wal/00000000000000000000.wal");
    let wal_bytes = fs::read(&wal_path).expect("read wal");
    let replay = crate::wal::replay_dir(vault_dir.join("wal")).expect("replay wal");
    let wal_rows = encode::decode_write_batch(&replay.records[0].payload).expect("decode batch");
    let ledger_index = row_index(&wal_rows, ColumnFamily::Ledger);
    let base_index = row_index(&wal_rows, ColumnFamily::Base);
    let ledger_entry = decode_ledger(&wal_rows[ledger_index].value).expect("decode ledger entry");
    let after = vault
        .read_cf_at(vault.snapshot(), ColumnFamily::Ledger, &ledger_key(0))
        .expect("read after")
        .expect("ledger row");
    let got = vault.get(id, vault.snapshot()).expect("get stored");
    let payload_json: serde_json::Value =
        serde_json::from_slice(&ledger_entry.payload).expect("payload json");

    let readback = serde_json::json!({
        "before_ledger_row_present": before.is_some(),
        "after_ledger_row_present": true,
        "ledger_cf_matches_wal_row": after == wal_rows[ledger_index].value,
        "same_wal_record": true,
        "wal_record_seq": replay.records[0].seq,
        "wal_row_count": wal_rows.len(),
        "ledger_row_index": ledger_index,
        "base_row_index": base_index,
        "ledger_before_base": ledger_index < base_index,
        "ledger_key_hex": hex(&wal_rows[ledger_index].key),
        "base_key_hex": hex(&wal_rows[base_index].key),
        "entry": {
            "seq": ledger_entry.seq,
            "prev_hash": hex(&ledger_entry.prev_hash),
            "kind": ledger_entry.kind.as_str(),
            "subject_is_cx": matches!(ledger_entry.subject, SubjectId::Cx(value) if value == id),
            "entry_hash": hex(&ledger_entry.entry_hash),
            "payload": payload_json,
        },
        "stored_constellation_provenance": {
            "seq": got.provenance.seq,
            "hash": hex(&got.provenance.hash),
        },
        "wal_file": wal_path,
        "wal_bytes": wal_bytes.len(),
        "wal_prefix_hex": hex(&wal_bytes[..wal_bytes.len().min(256)]),
    });
    let readback_path = root.join("ledger-group-commit-readback.json");
    fs::write(
        &readback_path,
        serde_json::to_vec_pretty(&readback).unwrap(),
    )
    .unwrap();

    println!("PH35_GROUP_COMMIT_FSV_ROOT={}", root.display());
    println!("PH35_GROUP_COMMIT_READBACK={}", readback_path.display());
    println!("{}", serde_json::to_string_pretty(&readback).unwrap());

    assert_eq!(before, None);
    assert!(ledger_index < base_index);
    assert_eq!(ledger_entry.seq, 0);
    assert_eq!(ledger_entry.prev_hash, [0; 32]);
    assert_eq!(ledger_entry.kind, EntryKind::Ingest);
    assert_eq!(got.provenance.seq, ledger_entry.seq);
    assert_eq!(got.provenance.hash, ledger_entry.entry_hash);
    assert_eq!(after, wal_rows[ledger_index].value);
}
