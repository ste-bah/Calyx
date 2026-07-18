use super::*;

#[test]
fn selected_snapshot_read_hydrates_only_requested_slots_and_fails_closed() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);
    let id = cx.cx_id;
    vault.put(cx.clone()).expect("put");

    let snapshot = vault.snapshot_handle(vault.snapshot());
    let selected = vault
        .get_selected_slots_at_snapshot(id, snapshot.snapshot(), [SlotId::new(0)])
        .expect("selected slot read");

    assert_eq!(selected.cx_id, id);
    assert_eq!(selected.metadata, cx.metadata);
    assert_eq!(selected.slots.len(), 1);
    assert_eq!(
        selected.slots.get(&SlotId::new(0)),
        cx.slots.get(&SlotId::new(0))
    );
    assert!(
        !selected.slots.contains_key(&SlotId::new(1)),
        "selected-slot read must not hydrate unrelated slots"
    );

    let error = vault
        .get_selected_slots_at_snapshot(id, snapshot.snapshot(), [SlotId::new(999)])
        .expect_err("missing selected slot must fail closed");
    assert_eq!(error.code, "CALYX_STALE_DERIVED");
}

#[test]
fn duplicate_put_is_idempotent_noop() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);

    vault.put(cx.clone()).expect("first put");
    let seq_after_first = vault.snapshot();
    vault.put(cx).expect("duplicate put");

    assert_eq!(vault.snapshot(), seq_after_first);
}

#[test]
fn same_cxid_with_different_bytes_fails_closed() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);
    let mut changed = cx.clone();
    // `created_at` is deliberately excluded from constellation identity
    // (`normalized_anchor_identity` zeroes it) so a re-put with a newer
    // timestamp stays idempotent. Mutate an identity-bearing content field (the
    // input-reference bytes) to exercise the same-cxid/different-bytes invariant.
    changed.input_ref.hash[0] ^= 0xFF;

    vault.put(cx).expect("first put");
    let error = vault.put(changed).expect_err("collision rejected");

    assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
}

#[test]
fn anchor_writes_anchor_cf_and_updates_get() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);
    let id = cx.cx_id;
    let anchor = Anchor {
        kind: AnchorKind::Reward,
        value: AnchorValue::Number(1.0),
        source: "unit-test".to_string(),
        observed_at: 124,
        confidence: 1.0,
    };

    vault.put(cx).expect("put");
    vault.anchor(id, anchor.clone()).expect("anchor");
    let got = vault.get(id, vault.snapshot()).expect("get anchored");
    let anchor_bytes = vault
        .read_cf_at(
            vault.snapshot(),
            ColumnFamily::Anchors,
            &anchor_key(id, &AnchorKind::Reward),
        )
        .expect("read anchor cf")
        .expect("anchor row");

    assert_eq!(got.anchors.as_slice(), std::slice::from_ref(&anchor));
    assert!(!got.flags.ungrounded);
    assert_eq!(encode::decode_anchor(&anchor_bytes).unwrap(), anchor);
}

#[test]
fn duplicate_put_after_anchor_preserves_anchor_noop() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);
    let id = cx.cx_id;
    let anchor = Anchor {
        kind: AnchorKind::Reward,
        value: AnchorValue::Number(1.0),
        source: "unit-test".to_string(),
        observed_at: 124,
        confidence: 1.0,
    };

    vault.put(cx.clone()).expect("put");
    vault.anchor(id, anchor.clone()).expect("anchor");
    let seq_after_anchor = vault.snapshot();
    vault.put(cx).expect("duplicate put after anchor");
    let got = vault.get(id, vault.snapshot()).expect("get anchored");

    assert_eq!(vault.snapshot(), seq_after_anchor);
    assert_eq!(got.anchors.as_slice(), std::slice::from_ref(&anchor));
}

#[test]
fn duplicate_put_with_conflicting_anchor_fails_closed() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let mut cx = sample_constellation(&vault);
    let id = cx.cx_id;
    cx.anchors = vec![Anchor {
        kind: AnchorKind::SpeakerMatch,
        value: AnchorValue::Text("speaker-a".to_string()),
        source: "unit-test".to_string(),
        observed_at: 124,
        confidence: 1.0,
    }];
    let mut changed = cx.clone();
    changed.anchors[0].value = AnchorValue::Text("speaker-b".to_string());

    vault.put(cx.clone()).expect("first put");
    let error = vault
        .put(changed)
        .expect_err("same-CxId anchor conflict must fail closed");
    let got = vault.get(id, vault.snapshot()).expect("get original");

    assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
    assert_eq!(got.anchors, cx.anchors);
}

#[test]
fn binary_codecs_roundtrip_known_offsets_and_fail_closed() {
    let vault = AsterVault::with_clock(vault_id(), b"salt".to_vec(), FixedClock::new(123));
    let cx = sample_constellation(&vault);
    let header = encode::encode_header(&cx);

    assert_eq!(&header[0..16], cx.cx_id.as_bytes());
    assert_eq!(&header[32..36], &7_u32.to_be_bytes());
    assert_eq!(header.len(), encode::HEADER_LEN);
    assert_eq!(encode::decode_header(&header).unwrap().cx_id, cx.cx_id);

    let base = encode::encode_constellation_base(&cx).expect("encode base");
    let decoded = encode::decode_constellation_base(&base).expect("decode base");
    assert_eq!(decoded.cx_id, cx.cx_id);
    assert_eq!(decoded.input_ref, cx.input_ref);
    assert!(encode::decode_header(&header[..encode::HEADER_LEN - 1]).is_err());

    for vector in cx.slots.values() {
        let bytes = encode::encode_slot_vector(vector).expect("encode slot");
        assert_eq!(encode::decode_slot_vector(&bytes).unwrap(), *vector);
    }
    let anchor = Anchor {
        kind: AnchorKind::Label("axis".to_string()),
        value: AnchorValue::Text("grounded".to_string()),
        source: "unit-test".to_string(),
        observed_at: 125,
        confidence: 0.5,
    };
    let bytes = encode::encode_anchor(&anchor).expect("encode anchor");
    assert_eq!(encode::decode_anchor(&bytes).unwrap(), anchor);
    assert!(encode::decode_anchor(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn anchor_vector_decode_rejects_non_finite_values() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&6_u16.to_be_bytes());
    bytes.push(5);
    bytes.extend_from_slice(&2_u32.to_be_bytes());
    bytes.extend_from_slice(&f32::NAN.to_bits().to_be_bytes());
    bytes.extend_from_slice(&1.0_f32.to_bits().to_be_bytes());
    bytes.extend_from_slice(&0_u32.to_be_bytes());
    bytes.extend_from_slice(&0_u64.to_be_bytes());
    bytes.extend_from_slice(&1.0_f32.to_bits().to_be_bytes());

    let err = encode::decode_anchor(&bytes).expect_err("nan vector must fail closed");
    assert!(err.to_string().contains("non-finite"));
}

#[test]
fn durable_vault_writes_wal_sst_manifest_and_cold_opens() {
    let dir = test_dir("durable");
    let vault =
        AsterVault::new_durable(&dir, vault_id(), b"salt".to_vec(), VaultOptions::default())
            .expect("open durable");
    let cx = sample_constellation(&AsterVault::with_clock(
        vault_id(),
        b"salt".to_vec(),
        FixedClock::new(123),
    ));
    let id = cx.cx_id;

    vault.put(cx.clone()).expect("durable put");
    vault.flush().expect("flush durable");

    let wal = dir.join("wal/00000000000000000000.wal");
    let wal_bytes = fs::read(&wal).expect("read wal");
    assert_eq!(&wal_bytes[0..4], b"CXW1");
    let replay = crate::wal::replay_dir(dir.join("wal")).expect("replay wal");
    let wal_rows = encode::decode_write_batch(&replay.records[0].payload).expect("decode batch");
    let ledger_index = wal_rows
        .iter()
        .position(|row| row.cf == ColumnFamily::Ledger)
        .expect("ledger row in WAL batch");
    let base_index = wal_rows
        .iter()
        .position(|row| row.cf == ColumnFamily::Base)
        .expect("base row in WAL batch");
    assert!(ledger_index < base_index);
    let ledger_entry = decode_ledger(&wal_rows[ledger_index].value).expect("decode ledger entry");
    assert_eq!(wal_rows[ledger_index].key, ledger_key(0));
    assert_eq!(ledger_entry.seq, 0);
    assert_eq!(ledger_entry.prev_hash, [0; 32]);
    assert_eq!(ledger_entry.kind, EntryKind::Ingest);
    assert_eq!(ledger_entry.subject, SubjectId::Cx(id));

    assert!(dir.join("CURRENT").exists());
    assert_eq!(sst_count(dir.join("cf/base")), 2);

    let reopened = AsterVault::open(&dir, vault_id(), b"salt".to_vec(), VaultOptions::default())
        .expect("cold open");
    let stored_ledger = reopened
        .read_cf_at(reopened.snapshot(), ColumnFamily::Ledger, &ledger_key(0))
        .expect("read ledger cf")
        .expect("ledger row");
    let mut expected = cx;
    expected.provenance = LedgerRef {
        seq: ledger_entry.seq,
        hash: ledger_entry.entry_hash,
    };
    assert_eq!(reopened.snapshot(), 1);
    assert_eq!(stored_ledger, wal_rows[ledger_index].value);
    assert_eq!(reopened.get(id, reopened.snapshot()).unwrap(), expected);
    cleanup(dir);
}
