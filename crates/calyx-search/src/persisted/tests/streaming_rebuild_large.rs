use calyx_aster::cf::{ColumnFamily, base_key};
use calyx_aster::vault::encode;
use calyx_core::{SlotId, VaultId};
use serde_json::json;
use std::path::Path;
use ulid::Ulid;

use super::*;

#[test]
#[ignore = "manual FSV allocates one minimally oversized 64 MiB multi row"]
fn oversized_multi_row_fails_before_decode_and_preserves_active_manifest() {
    const TOKEN_DIM: u32 = 1;
    const TOKEN_COUNT: u32 = 16_777_200;
    const PAYLOAD_BYTES: usize = TOKEN_DIM as usize * TOKEN_COUNT as usize * 4;
    let root = scratch("oversized-multi-row");
    let vault_id = VaultId::from_ulid(Ulid::from_bytes([0x49; 16]));
    let salt = b"streaming-search-oversized-multi".to_vec();
    let vault = AsterVault::new_durable(
        &root,
        vault_id,
        salt,
        VaultOptions {
            memtable_byte_cap: 80 * 1024 * 1024,
            ..VaultOptions::default()
        },
    )
    .expect("open durable vault");
    let mut initial = constellation(cx(91), vec![1.0, 0.0]);
    initial.vault_id = vault_id;
    initial
        .slots
        .insert(SlotId::new(2), multi(2, &[&[1.0, 0.0]]));
    vault.put(initial).expect("write initial constellation");
    rebuild_for_vault(&root, &vault).expect("publish initial manifest");
    let manifest_path = root.join("idx/search/manifest.json");
    let manifest_before = fs::read(&manifest_path).expect("read initial manifest");
    let files_before = multi_artifact_names(&root);

    let mut oversized_base = constellation(cx(92), vec![0.0, 1.0]);
    oversized_base.vault_id = vault_id;
    oversized_base
        .slots
        .insert(SlotId::new(2), multi(2, &[&[0.0, 1.0]]));
    let cx_id = oversized_base.cx_id;
    vault
        .write_cf(
            ColumnFamily::Base,
            base_key(cx_id),
            encode::encode_constellation_base(&oversized_base).unwrap(),
        )
        .expect("write oversized row base");
    vault
        .write_cf(
            ColumnFamily::slot(SlotId::new(0)),
            base_key(cx_id),
            encode::encode_slot_vector(oversized_base.slots.get(&SlotId::new(0)).unwrap()).unwrap(),
        )
        .expect("write dense slot row");
    let mut encoded = Vec::with_capacity(9 + PAYLOAD_BYTES);
    encoded.push(3);
    encoded.extend_from_slice(&TOKEN_DIM.to_be_bytes());
    encoded.extend_from_slice(&TOKEN_COUNT.to_be_bytes());
    encoded.resize(9 + PAYLOAD_BYTES, 0);
    let encoded_len = encoded.len();
    vault
        .write_cf(ColumnFamily::slot(SlotId::new(2)), base_key(cx_id), encoded)
        .expect("write physically valid oversized multi row");
    let physical_before = vault
        .read_cf_at(
            vault.latest_seq(),
            ColumnFamily::slot(SlotId::new(2)),
            &base_key(cx_id),
        )
        .expect("read oversized row before rebuild")
        .expect("oversized row exists before rebuild")
        .len();

    let error = rebuild_for_vault(&root, &vault).expect_err("oversized row must fail rebuild");

    let physical_after = vault
        .read_cf_at(
            vault.latest_seq(),
            ColumnFamily::slot(SlotId::new(2)),
            &base_key(cx_id),
        )
        .expect("read oversized row after rebuild")
        .expect("oversized row remains source truth")
        .len();
    let manifest_after = fs::read(&manifest_path).expect("reread active manifest");
    let files_after = multi_artifact_names(&root);
    let marker = marker::read_rebuild_required_marker(&root)
        .expect("read rebuild marker")
        .expect("failed rebuild retains marker");

    assert_eq!(error.code(), "CALYX_SEARCH_MULTI_SIDECAR_UNBOUNDED");
    assert!(error.message().contains(&cx_id.to_string()));
    assert!(error.message().contains("before decode"));
    assert_eq!(physical_before, encoded_len);
    assert_eq!(physical_after, encoded_len);
    assert_eq!(manifest_after, manifest_before);
    assert_eq!(files_after, files_before);
    println!(
        "OVERSIZED_MULTI_ROW_FSV {}",
        json!({
            "source_of_truth": "durable Aster slot_2 CF row plus byte-identical active search manifest",
            "before": {
                "physical_row_bytes": physical_before,
                "manifest_sha256": sha256_hex(&manifest_before),
                "multi_artifacts": files_before,
            },
            "action": {
                "token_dim": TOKEN_DIM,
                "token_count": TOKEN_COUNT,
                "error_code": error.code(),
                "error_message": error.message(),
            },
            "after": {
                "physical_row_bytes": physical_after,
                "manifest_sha256": sha256_hex(&manifest_after),
                "manifest_unchanged": manifest_after == manifest_before,
                "multi_artifacts": files_after,
                "rebuild_marker": marker,
            }
        })
    );
    cleanup(root);
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn cuda_rebuild_reads_cagra_backend_from_physical_diskann_manifest() {
    const ROWS: usize = 32_769;
    let root = scratch("cuda-cagra-rebuild");
    let vault_id = VaultId::from_ulid(Ulid::from_bytes([0x47; 16]));
    let salt = b"streaming-search-cuda-cagra".to_vec();
    let vault = AsterVault::new_durable(&root, vault_id, salt, VaultOptions::default())
        .expect("open durable vault");
    let mut rows = Vec::with_capacity(ROWS);
    for idx in 0..ROWS {
        let mut cx = constellation(cx_from_u32(idx as u32), unit_circle(idx));
        cx.vault_id = vault_id;
        rows.push(cx);
    }
    let ids = vault.put_batch(rows).expect("write cagra constellations");
    let before = physical_counts(&vault);
    let mut phases = Vec::new();

    rebuild_for_vault_with_progress(&root, &vault, |event| {
        phases.push(event.phase.to_string());
    })
    .expect("cuda cagra rebuild");

    let after = physical_counts(&vault);
    let indexes = PersistedSearchIndexes::open(&root).expect("open indexes");
    let manifest_path = root.join("idx/search/manifest.json");
    let manifest_bytes = fs::read(&manifest_path).expect("read manifest");
    let manifest_json: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).expect("manifest json");
    let slot = indexes
        .manifest
        .slots
        .iter()
        .find(|entry| entry.slot == 0)
        .expect("slot 0 manifest entry");
    let graph_rel = slot.graph_rel.as_ref().expect("diskann graph rel");
    let id_map_rel = slot.id_map_rel.as_ref().expect("diskann id map rel");
    let graph_bytes = fs::read(root.join(graph_rel)).expect("read diskann graph");
    let id_map_bytes = fs::read(root.join(id_map_rel)).expect("read id map");
    let hits = indexes
        .search(SlotId::new(0), &dense(unit_circle(0)), 1)
        .expect("search cagra-built diskann");

    assert_eq!(before, after);
    assert_eq!(after["base_rows"], ROWS);
    assert_eq!(after["slot_0_rows"], ROWS);
    assert_eq!(slot.kind, "diskann");
    assert_eq!(slot.len, ROWS);
    assert_eq!(manifest_json["diskann_build_backend"], "cuvs-cagra");
    assert_eq!(
        manifest_json["diskann_build_backend_source"],
        "compiled-cuvs-default"
    );
    assert_eq!(manifest_json["sextant_cuvs_compiled"], true);
    assert_eq!(hits.len(), 1);
    assert!(phases.contains(&"diskann_cuvs_cagra_start".to_string()));
    assert!(phases.contains(&"diskann_cuvs_cagra_ok".to_string()));
    println!(
        "CUDA_CAGRA_REBUILD_FSV {}",
        json!({
            "source_of_truth": "durable Aster Base/Slot CF rows plus physical idx/search manifest, DiskANN graph, and id-map bytes",
            "row_count": ROWS,
            "before": before,
            "after": after,
            "manifest_path": manifest_path,
            "manifest_sha256": sha256_hex(&manifest_bytes),
            "diskann_build_backend": manifest_json["diskann_build_backend"],
            "diskann_build_backend_source": manifest_json["diskann_build_backend_source"],
            "sextant_cuvs_compiled": manifest_json["sextant_cuvs_compiled"],
            "slot_kind": slot.kind,
            "slot_len": slot.len,
            "graph_rel": graph_rel,
            "graph_bytes": graph_bytes.len(),
            "graph_sha256": sha256_hex(&graph_bytes),
            "id_map_rel": id_map_rel,
            "id_map_bytes": id_map_bytes.len(),
            "id_map_sha256": sha256_hex(&id_map_bytes),
            "top_hit": hits[0].cx_id.to_string(),
            "expected_top_hit": ids[0].to_string(),
            "phases": phases,
        })
    );
    cleanup(root);
}

fn multi_artifact_names(root: &Path) -> Vec<String> {
    let mut names = fs::read_dir(root.join("idx/search"))
        .expect("read search artifact directory")
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.contains(".multi."))
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cx_from_u32(value: u32) -> calyx_core::CxId {
    let mut bytes = [0; 16];
    bytes[0..4].copy_from_slice(&value.to_be_bytes());
    calyx_core::CxId::from_bytes(bytes)
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn unit_circle(idx: usize) -> Vec<f32> {
    let angle = idx as f32 * 0.0001;
    vec![angle.cos(), angle.sin()]
}
