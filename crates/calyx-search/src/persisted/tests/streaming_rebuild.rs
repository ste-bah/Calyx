use calyx_aster::cf::{ColumnFamily, base_key};
use calyx_aster::vault::encode;
use calyx_core::{
    Asymmetry, LensId, Modality, Panel, QuantPolicy, Slot, SlotId, SlotKey, SlotShape, SlotState,
    SlotVector, SparseEntry, VaultId,
};
use calyx_registry::{Registry, VaultPanelState, persist_vault_panel_state};
use serde_json::json;
use ulid::Ulid;

use super::*;

#[path = "streaming_rebuild_large.rs"]
mod streaming_rebuild_large;

#[test]
fn streaming_rebuild_reads_physical_slot_cfs_and_validates_sidecars() {
    let root = scratch("streaming-rebuild");
    let vault_id = VaultId::from_ulid(Ulid::from_bytes([0x45; 16]));
    let salt = b"streaming-search-rebuild".to_vec();
    let vault = AsterVault::new_durable(&root, vault_id, salt, VaultOptions::default())
        .expect("open durable vault");
    let mut first = constellation(cx(51), vec![1.0, 0.0]);
    first.vault_id = vault_id;
    first.slots.insert(SlotId::new(1), sparse(16, &[(3, 1.0)]));
    first
        .slots
        .insert(SlotId::new(2), multi(2, &[&[1.0, 0.0], &[0.5, 0.5]]));
    let mut second = constellation(cx(52), vec![0.0, 1.0]);
    second.vault_id = vault_id;
    second.slots.insert(SlotId::new(1), sparse(16, &[(7, 2.0)]));
    second
        .slots
        .insert(SlotId::new(2), multi(2, &[&[0.0, 1.0]]));
    let ids = vault
        .put_batch(vec![first, second])
        .expect("write durable constellations");
    vault.flush().expect("flush real slot rows");
    drop(vault);
    let vault = AsterVault::open(
        &root,
        vault_id,
        b"streaming-search-rebuild".to_vec(),
        VaultOptions::default(),
    )
    .expect("reopen through durable router state");
    let before = physical_counts(&vault);
    let mut events = Vec::new();

    rebuild_for_vault_with_progress(&root, &vault, |event| {
        events.push((
            event.phase.to_string(),
            event.slot.map(|slot| slot.get()),
            event.rows,
            event.detail,
        ));
    })
    .expect("streaming rebuild");

    let indexes = PersistedSearchIndexes::open(&root).expect("open indexes");
    let after = physical_counts(&vault);
    let manifest_path = root.join("idx/search/manifest.json");
    let manifest_bytes = fs::read(&manifest_path).expect("read manifest");
    let manifest_json: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest json");
    let entries = indexes
        .manifest
        .slots
        .iter()
        .map(|entry| (entry.slot, entry.kind.clone(), entry.len))
        .collect::<Vec<_>>();
    let dense_hits = indexes
        .search(SlotId::new(0), &dense(vec![1.0, 0.0]), 1)
        .expect("dense search");
    let multi_hits = indexes
        .search(SlotId::new(2), &multi(2, &[&[1.0, 0.0]]), 2)
        .expect("multi search");
    let sidecars = indexes
        .manifest
        .slots
        .iter()
        .filter_map(|entry| {
            entry.index_rel.as_ref().map(|rel| {
                let bytes = fs::read(root.join(rel)).expect("read sidecar");
                json!({
                    "slot": entry.slot,
                    "kind": entry.kind,
                    "rel": rel,
                    "exists": root.join(rel).is_file(),
                    "bytes": bytes.len(),
                    "sha256": sha256_hex(&bytes),
                    "manifest_sha256": entry.sha256,
                })
            })
        })
        .collect::<Vec<_>>();
    let multi_entry = indexes
        .manifest
        .slots
        .iter()
        .find(|entry| entry.slot == 2)
        .expect("multi manifest entry");
    let segment_manifest_path = root.join(multi_entry.index_rel.as_ref().unwrap());
    let segment_manifest_bytes = fs::read(&segment_manifest_path).expect("read segment manifest");
    let segment_manifest: serde_json::Value =
        serde_json::from_slice(&segment_manifest_bytes).expect("segment manifest JSON");
    let binary_segments = segment_manifest["segments"]
        .as_array()
        .expect("segment refs")
        .iter()
        .map(|segment| {
            let rel = segment["index_rel"].as_str().expect("segment path");
            let bytes = fs::read(root.join(rel)).expect("read binary segment");
            let actual_sha256 = sha256_hex(&bytes);
            assert_eq!(actual_sha256, segment["sha256"]);
            assert!(bytes.len() <= 64 * 1024 * 1024);
            json!({
                "rel": rel,
                "bytes": bytes.len(),
                "sha256": actual_sha256,
                "rows": segment["row_count"],
                "tokens": segment["token_count"],
            })
        })
        .collect::<Vec<_>>();
    let phases = events
        .iter()
        .map(|event| event.0.clone())
        .collect::<Vec<_>>();

    assert_eq!(before, after);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0], (0, "flat_dense".to_string(), 2));
    assert_eq!(entries[1], (1, "sparse_inverted".to_string(), 2));
    assert_eq!(entries[2].0, 2);
    assert_eq!(dense_hits[0].cx_id, ids[0]);
    assert_eq!(multi_hits[0].cx_id, ids[0]);
    assert_eq!(multi_entry.kind, "multi_maxsim_segments");
    assert_eq!(segment_manifest["row_count"], 2);
    assert_eq!(binary_segments.len(), 1);
    assert!(manifest_path.is_file());
    assert_eq!(
        manifest_json["diskann_build_backend"],
        if calyx_sextant::CUVS_COMPILED {
            "cuvs-cagra"
        } else {
            "cpu-vamana"
        }
    );
    assert_eq!(
        manifest_json["diskann_build_backend_source"],
        if calyx_sextant::CUVS_COMPILED {
            "compiled-cuvs-default"
        } else {
            "compiled-cpu-default"
        }
    );
    assert_eq!(
        manifest_json["sextant_cuvs_compiled"],
        calyx_sextant::CUVS_COMPILED
    );
    assert!(phases.contains(&"base_scan_page".to_string()));
    assert!(events.iter().any(|event| {
        event.0 == "load_docs_ok"
            && event.3.as_deref() == Some("slot_memberships=6 retained_base_slot_payloads=0")
    }));
    assert!(phases.contains(&"slot_plan_ok".to_string()));
    assert!(phases.contains(&"slot_build_start".to_string()));
    assert!(phases.contains(&"slot_point_read_page".to_string()));
    assert!(phases.contains(&"multi_segment_write_ok".to_string()));
    assert!(phases.contains(&"manifest_validate_ok".to_string()));
    assert!(events.iter().any(|event| {
        event.0 == "multi_segment_write_ok"
            && event.3.as_deref().is_some_and(|detail| {
                detail.contains("bytes=") && detail.contains("sha256=") && detail.contains("path=")
            })
    }));
    println!(
        "STREAMING_REBUILD_FSV {}",
        json!({
            "source_of_truth": "durable Aster Base/Slot CF rows plus idx/search manifest and sidecar bytes",
            "before": before,
            "after": after,
            "manifest_path": manifest_path,
            "manifest_sha256": sha256_hex(&manifest_bytes),
            "diskann_build_backend": manifest_json["diskann_build_backend"],
            "diskann_build_backend_source": manifest_json["diskann_build_backend_source"],
            "sextant_cuvs_compiled": manifest_json["sextant_cuvs_compiled"],
            "manifest_slots": manifest_json["slots"],
            "sidecars": sidecars,
            "multi_segment_manifest": segment_manifest_path,
            "multi_segment_manifest_sha256": sha256_hex(&segment_manifest_bytes),
            "binary_segments": binary_segments,
            "dense_hit": dense_hits[0].cx_id.to_string(),
            "multi_hit": multi_hits[0].cx_id.to_string(),
            "events": events,
        })
    );
    cleanup(root);
}

#[test]
fn missing_slot_cf_row_fails_before_manifest_swap() {
    let root = scratch("missing-slot-streaming");
    let vault_id = VaultId::from_ulid(Ulid::from_bytes([0x46; 16]));
    let salt = b"streaming-search-missing-slot".to_vec();
    let vault = AsterVault::new_durable(&root, vault_id, salt, VaultOptions::default())
        .expect("open durable vault");
    let mut broken = constellation(cx(61), vec![1.0, 0.0]);
    broken.vault_id = vault_id;
    let base = encode::encode_constellation_base(&broken).expect("encode base");
    vault
        .write_cf(ColumnFamily::Base, base_key(broken.cx_id), base)
        .expect("write base without slot row");
    let before = physical_counts(&vault);

    let err = rebuild_for_vault(&root, &vault).unwrap_err();

    let after = physical_counts(&vault);
    let manifest_path = root.join("idx/search/manifest.json");
    assert_eq!(err.code(), "CALYX_ASTER_CORRUPT_SHARD");
    assert!(err.message().contains("slot CF row missing"));
    assert_eq!(before, after);
    assert!(!manifest_path.exists());
    println!(
        "STREAMING_REBUILD_MISSING_SLOT_FSV {}",
        json!({
            "source_of_truth": "durable Aster Base/Slot CF rows and absent idx/search manifest",
            "before": before,
            "after": after,
            "manifest_exists_after": manifest_path.exists(),
            "error_code": err.code(),
            "error_message": err.message(),
        })
    );
    cleanup(root);
}

#[test]
fn panel_aware_rebuild_omits_parked_physical_slots() {
    let root = scratch("parked-slot-rebuild");
    let vault_id = VaultId::from_ulid(Ulid::from_bytes([0x48; 16]));
    let salt = b"parked-slot-search-rebuild".to_vec();
    let panel = panel_with_parked_multi_slot();
    let registry = Registry::new();
    let vault = AsterVault::new_durable(
        &root,
        vault_id,
        salt,
        VaultOptions {
            panel: Some(panel.clone()),
            ..VaultOptions::default()
        },
    )
    .expect("open durable vault");
    persist_vault_panel_state(&root, &panel, &registry).expect("persist panel state");
    let state = VaultPanelState {
        panel,
        registry,
        registry_snapshot: None,
    };
    let mut first = constellation(cx(81), vec![1.0, 0.0]);
    first.vault_id = vault_id;
    first
        .slots
        .insert(SlotId::new(2), multi(2, &[&[1.0, 0.0], &[0.5, 0.5]]));
    let mut second = constellation(cx(82), vec![0.0, 1.0]);
    second.vault_id = vault_id;
    second
        .slots
        .insert(SlotId::new(2), multi(2, &[&[0.0, 1.0]]));
    let ids = vault
        .put_batch(vec![first, second])
        .expect("write durable constellations");
    vault.flush().expect("flush vault");
    let physical_before = physical_counts(&vault);
    let mut phases = Vec::new();

    rebuild_for_vault_with_panel_state_progress(&root, &vault, &state, |event| {
        phases.push((event.phase.to_string(), event.slot.map(|slot| slot.get())));
    })
    .expect("panel-aware rebuild");

    let indexes = PersistedSearchIndexes::open(&root).expect("open indexes");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("idx/search/manifest.json")).unwrap())
            .expect("manifest json");
    let dense_hits = indexes
        .search(SlotId::new(0), &dense(vec![1.0, 0.0]), 1)
        .expect("active dense search");
    let parked_error = indexes
        .search(SlotId::new(2), &multi(2, &[&[1.0, 0.0]]), 1)
        .expect_err("parked slot must not have a manifest entry");

    assert_eq!(physical_before["slot_2_rows"], 2);
    assert_eq!(manifest["slots"].as_array().unwrap().len(), 1);
    assert_eq!(manifest["slots"][0]["slot"], 0);
    assert_eq!(dense_hits[0].cx_id, ids[0]);
    assert!(
        !phases
            .iter()
            .any(|(phase, slot)| phase == "slot_build_start" && *slot == Some(2)),
        "parked slot must not be planned: {phases:?}"
    );
    assert!(
        parked_error
            .message()
            .contains("no index for active slot 2")
    );
    println!(
        "PANEL_AWARE_REBUILD_SKIPS_PARKED_SLOT_FSV {}",
        json!({
            "source_of_truth": "panel state marks slot 2 parked while Aster slot_2 CF rows physically exist",
            "physical_before": physical_before,
            "manifest_slots": manifest["slots"],
            "phases": phases,
            "dense_hit": dense_hits[0].cx_id.to_string(),
            "parked_error": {
                "code": parked_error.code(),
                "message": parked_error.message(),
            },
        })
    );
    cleanup(root);
}

fn physical_counts(vault: &AsterVault) -> serde_json::Value {
    json!({
        "base_rows": vault.scan_cf_at(vault.latest_seq(), ColumnFamily::Base).unwrap().len(),
        "slot_0_rows": vault.scan_cf_at(vault.latest_seq(), ColumnFamily::slot(SlotId::new(0))).unwrap().len(),
        "slot_1_rows": vault.scan_cf_at(vault.latest_seq(), ColumnFamily::slot(SlotId::new(1))).unwrap().len(),
        "slot_2_rows": vault.scan_cf_at(vault.latest_seq(), ColumnFamily::slot(SlotId::new(2))).unwrap().len(),
    })
}

fn panel_with_parked_multi_slot() -> Panel {
    Panel {
        version: 1,
        slots: vec![
            panel_slot(
                SlotId::new(0),
                "active-dense",
                SlotShape::Dense(2),
                SlotState::Active,
            ),
            panel_slot(
                SlotId::new(2),
                "parked-multi",
                SlotShape::Multi { token_dim: 2 },
                SlotState::Parked,
            ),
        ],
        created_at: 1,
        kernel_ref: None,
        guard_ref: None,
    }
}

fn panel_slot(slot_id: SlotId, key: &str, shape: SlotShape, state: SlotState) -> Slot {
    Slot {
        slot_id,
        slot_key: SlotKey::new(slot_id, key),
        lens_id: LensId::from_bytes([slot_id.get() as u8; 16]),
        shape,
        modality: Modality::Text,
        asymmetry: Asymmetry::None,
        quant: QuantPolicy::None,
        resource: Default::default(),
        axis: Some(key.to_string()),
        retrieval_only: false,
        excluded_from_dedup: false,
        bits_about: Default::default(),
        state,
        added_at_panel_version: 1,
    }
}

fn sparse(dim: u32, entries: &[(u32, f32)]) -> SlotVector {
    SlotVector::Sparse {
        dim,
        entries: entries
            .iter()
            .map(|(idx, val)| SparseEntry {
                idx: *idx,
                val: *val,
            })
            .collect(),
    }
}

fn multi(token_dim: u32, rows: &[&[f32]]) -> SlotVector {
    SlotVector::Multi {
        token_dim,
        tokens: rows.iter().map(|row| row.to_vec()).collect(),
    }
}

fn cleanup(root: std::path::PathBuf) {
    if calyx_fsv::fsv_root("CALYX_FSV_ROOT").is_none() {
        fs::remove_dir_all(root).ok();
    }
}
