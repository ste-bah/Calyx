use super::*;

#[test]
fn durable_blob_fsv_writes_readback_artifacts() {
    let fsv_root = calyx_fsv::fsv_root("CALYX_FSV_ROOT");
    let dir = fsv_root
        .as_ref()
        .map(|root| root.join("blob-vault"))
        .unwrap_or_else(|| temp_dir("blob-fsv"));
    fs::remove_dir_all(&dir).ok();

    let vault = AsterVault::new_durable(
        &dir,
        vault_id(),
        b"blob-fsv-salt".to_vec(),
        VaultOptions::default(),
    )
    .unwrap();
    let layer = BlobLayer::new(&vault);
    let col = blob_collection();
    create_collection(&vault, col.clone()).unwrap();

    let data = synthetic(2 * 1024 * 1024); // 2 MiB -> 8 chunks
    let expected_hash = *blake3::hash(&data).as_bytes();
    let put = layer.blob_put_content_addressed(&col, &data).unwrap();
    let id = put.blob_id;
    assert_eq!(put.manifest.content_hash, expected_hash);

    vault.flush().unwrap();
    drop(vault);

    let reopened = AsterVault::open(
        &dir,
        vault_id(),
        b"blob-fsv-salt".to_vec(),
        VaultOptions::default(),
    )
    .unwrap();
    let reopened_layer = BlobLayer::new(&reopened);

    let read = reopened_layer.blob_read(&col, id).unwrap().unwrap();
    let manifest = read.manifest;
    assert_eq!(manifest.chunk_count, 8);
    assert_eq!(manifest.total_bytes, data.len() as u64);
    assert_eq!(manifest.content_hash, expected_hash);
    // Byte-exact round-trip across a cold reopen (the `cmp` equivalent).
    let roundtrip = read.data;
    assert_eq!(roundtrip, data);

    let cf_files = physical_files(&dir.join("cf").join("blob"));
    assert!(!cf_files.is_empty(), "cf/blob must hold on-disk shards");

    let ck = chunk_key(&col, id, 0);
    let mk = manifest_key(&col, id);
    let readback = serde_json::json!({
        "issue": 1549,
        "layer": "blob",
        "source_of_truth": dir.display().to_string(),
        "cf": ColumnFamily::Blob.name(),
        "chunk_key_hex": hex_bytes(&ck),
        "chunk_disc": format!("{:#04x}", ck[0]),
        "chunk_kind": format!("{:#04x}", ck[1]),
        "manifest_key_hex": hex_bytes(&mk),
        "manifest_kind": format!("{:#04x}", mk[1]),
        "manifest_chunk_count": manifest.chunk_count,
        "manifest_total_bytes": manifest.total_bytes,
        "manifest_created_at_ms": manifest.created_at_ms,
        "content_hash_hex": hex_bytes(&manifest.content_hash),
        "roundtrip_byte_exact": roundtrip == data,
        "put_result_matches_persisted_manifest": put.manifest == manifest,
        "put_result_seq": put.seq,
        "blob_cf_files": cf_files,
    });
    assert_eq!(readback["roundtrip_byte_exact"], serde_json::json!(true));

    if let Some(root) = fsv_root {
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("issue1549-blob-readback.json"),
            serde_json::to_vec_pretty(&readback).unwrap(),
        )
        .unwrap();
        println!("issue1549_blob_fsv_root={}", root.display());
        println!("{}", serde_json::to_string_pretty(&readback).unwrap());
    } else {
        fs::remove_dir_all(dir).ok();
    }
}
