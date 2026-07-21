use super::*;
use crate::cf::{ColumnFamily, slot_key};
use crate::mvcc::Freshness;
use calyx_core::{CxFlags, InputRef, LedgerRef, Modality, SlotId, SlotVector, VaultStore};
use std::collections::BTreeMap;
use std::fs;
use ulid::Ulid;

#[test]
fn pinned_slot_batch_reads_real_durable_rows_in_request_order() {
    let root = std::env::temp_dir().join(format!(
        "calyx-issue1799-slot-batch-{}-{}",
        std::process::id(),
        Ulid::new()
    ));
    let vault_id = VaultId::from_ulid(Ulid::new());
    let salt = b"issue1799-real-durable-slot-batch".to_vec();
    let vault = AsterVault::new_durable(&root, vault_id, salt.clone(), VaultOptions::default())
        .expect("create durable vault");
    let slot = SlotId::new(11);
    let first = row(&vault, vault_id, slot, b"first", vec![vec![1.0, 0.0]]);
    let second = row(
        &vault,
        vault_id,
        slot,
        b"second",
        vec![vec![0.0, 1.0], vec![0.5, 0.5]],
    );
    let ids = vault.put_batch(vec![first, second]).expect("put real rows");
    vault.flush().expect("flush durable rows");
    drop(vault);

    // Reopen through the real latest-only router/SST path, not the writer's
    // in-memory table, then read in deliberately reversed request order.
    let vault = AsterVault::open(&root, vault_id, salt, VaultOptions::default())
        .expect("reopen durable vault");
    let missing = CxId::from_bytes([0xff; 16]);
    let snapshot = vault.pin_reader(Freshness::FreshDerived, 60_000);
    let requested = [ids[1], ids[0], missing];
    let before = vault
        .scan_cf_snapshot(snapshot, ColumnFamily::slot(slot))
        .expect("physical before scan");
    let values = vault
        .read_slot_cf_batch_snapshot(snapshot, slot, &requested)
        .expect("bounded point read");
    let after = vault
        .scan_cf_snapshot(snapshot, ColumnFamily::slot(slot))
        .expect("physical after scan");
    assert!(vault.release_reader(snapshot.lease().id()));

    assert_eq!(before, after, "read must not mutate the physical slot CF");
    assert_eq!(values.len(), 3);
    let second = encode::decode_slot_vector(values[0].as_deref().expect("second row")).unwrap();
    let first = encode::decode_slot_vector(values[1].as_deref().expect("first row")).unwrap();
    assert!(values[2].is_none(), "missing key stays explicit");
    assert!(matches!(second, SlotVector::Multi { ref tokens, .. } if tokens.len() == 2));
    assert!(matches!(first, SlotVector::Multi { ref tokens, .. } if tokens.len() == 1));
    assert!(
        vault
            .read_cf_at(
                vault.snapshot(),
                ColumnFamily::slot(slot),
                &slot_key(ids[0])
            )
            .unwrap()
            .is_some()
    );
    eprintln!(
        "ISSUE1799_SLOT_BATCH before_rows={} after_rows={} request_order={:?} missing_explicit=true",
        before.len(),
        after.len(),
        requested
    );
    fs::remove_dir_all(root).ok();
}

fn row(
    vault: &AsterVault,
    vault_id: VaultId,
    slot: SlotId,
    input: &[u8],
    tokens: Vec<Vec<f32>>,
) -> Constellation {
    let cx_id = vault.cx_id_for_input(input, 1);
    let mut slots = BTreeMap::new();
    slots.insert(
        slot,
        SlotVector::Multi {
            token_dim: 2,
            tokens,
        },
    );
    Constellation {
        cx_id,
        vault_id,
        panel_version: 1,
        created_at: 1,
        input_ref: InputRef {
            hash: [input[0]; 32],
            pointer: None,
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
