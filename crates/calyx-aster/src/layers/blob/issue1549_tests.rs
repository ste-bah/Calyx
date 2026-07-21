use std::fs;
use std::path::PathBuf;

use calyx_core::VaultId;
use serde_json::json;

use super::*;
use crate::collection::{
    DedupPolicy, RetentionPolicy, TemporalPolicy, TenantId, TxnPolicy, create_collection,
};
use crate::vault::VaultOptions;

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV"
        .parse()
        .expect("valid vault id")
}

fn collection() -> Collection {
    Collection {
        name: "issue1549-scaling".to_string(),
        mode: CollectionMode::Blob,
        schema: None,
        panel: None,
        indexes: Vec::new(),
        dedup: DedupPolicy::Off,
        temporal: TemporalPolicy::default(),
        retention: RetentionPolicy::Forever,
        txn_policy: TxnPolicy::default(),
        tenant: TenantId::default(),
    }
}

fn synthetic(size: usize) -> Vec<u8> {
    let pattern = (0..251)
        .map(|index| ((index * 31 + 17) % 251) as u8)
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(size);
    while size - data.len() >= pattern.len() {
        data.extend_from_slice(&pattern);
    }
    data.extend_from_slice(&pattern[..size - data.len()]);
    data
}

#[test]
#[ignore = "manual 1-GiB durable #1549 scaling FSV; requires CALYX_FSV_ROOT"]
fn issue1549_durable_scaling_counts_hashes_reads_and_physical_rows() {
    let root = PathBuf::from(
        std::env::var("CALYX_FSV_ROOT")
            .expect("CALYX_FSV_ROOT must name the durable evidence directory"),
    );
    let vault_dir = root.join("issue1549-scaling-vault");
    fs::remove_dir_all(&vault_dir).ok();
    fs::create_dir_all(&root).expect("create issue1549 FSV root");
    let vault = AsterVault::new_durable(
        &vault_dir,
        vault_id(),
        b"issue1549-durable-scaling".to_vec(),
        VaultOptions::default(),
    )
    .expect("create durable issue1549 vault");
    let col = collection();
    create_collection(&vault, col.clone()).expect("create Blob collection");
    let layer = BlobLayer::new(&vault);
    let sizes = [
        0,
        1,
        1_024,
        BLOB_CHUNK_SIZE - 1,
        BLOB_CHUNK_SIZE,
        BLOB_CHUNK_SIZE + 1,
        1 << 20,
        100 << 20,
        MAX_BLOB_BYTES,
    ];
    let mut results = Vec::new();
    let mut expected = Vec::new();

    for size in sizes {
        let data = synthetic(size);
        let expected_hash = *blake3::hash(&data).as_bytes();
        eprintln!(
            "ISSUE1549_SCALE before size={} latest_seq={} manifest_present=false",
            size,
            vault.latest_seq()
        );

        reset_blob_io_counts();
        let put = layer
            .blob_put_content_addressed(&col, &data)
            .expect("content-addressed put");
        let put_counts = blob_io_counts();
        assert_eq!(put_counts.hash_calls, 1);
        assert_eq!(put_counts.hash_bytes, size);
        assert_eq!(put_counts.snapshot_pins, 0);
        assert_eq!(put_counts.manifest_reads, 0);
        assert_eq!(put_counts.manifest_decodes, 0);
        assert_eq!(put_counts.chunk_reads, 0);
        assert_eq!(put_counts.chunk_rows_written, chunk_count_for_size(size));
        assert_eq!(
            put_counts.chunk_group_commits,
            chunk_count_for_size(size).div_ceil(chunks_per_group())
        );
        assert_eq!(put.manifest.content_hash, expected_hash);
        assert_eq!(put.manifest.total_bytes, size as u64);
        assert_eq!(
            put.manifest.chunk_count as usize,
            if size == 0 {
                0
            } else {
                size.div_ceil(BLOB_CHUNK_SIZE)
            }
        );

        reset_blob_io_counts();
        let manifest = layer
            .blob_manifest(&col, put.blob_id)
            .expect("manifest-only read")
            .expect("manifest exists");
        let manifest_counts = blob_io_counts();
        assert_eq!(manifest, put.manifest);
        assert_eq!(manifest_counts.hash_calls, 0);
        assert_eq!(manifest_counts.hash_bytes, 0);
        assert_eq!(manifest_counts.snapshot_pins, 1);
        assert_eq!(manifest_counts.manifest_reads, 1);
        assert_eq!(manifest_counts.manifest_decodes, 1);
        assert_eq!(manifest_counts.chunk_reads, 0);

        reset_blob_io_counts();
        let read = layer
            .blob_read(&col, put.blob_id)
            .expect("combined data read")
            .expect("blob exists");
        let read_counts = blob_io_counts();
        assert_eq!(read.manifest, put.manifest);
        assert_eq!(read.data, data);
        assert_eq!(read_counts.hash_calls, 1);
        assert_eq!(read_counts.hash_bytes, size);
        assert_eq!(read_counts.snapshot_pins, 1);
        assert_eq!(read_counts.manifest_reads, 1);
        assert_eq!(read_counts.manifest_decodes, 1);
        assert_eq!(read_counts.chunk_reads, put.manifest.chunk_count as usize);

        let persisted_bytes = vault
            .read_cf_at(
                vault.latest_seq(),
                ColumnFamily::Blob,
                &manifest_key(&col, put.blob_id),
            )
            .expect("read physical manifest row")
            .expect("physical manifest row exists");
        assert_eq!(decode_manifest(&persisted_bytes).unwrap(), put.manifest);
        results.push(json!({
            "bytes": size,
            "blob_id_hex": hex_bytes(put.blob_id.as_bytes()),
            "content_hash_hex": hex_bytes(&expected_hash),
            "chunk_count": put.manifest.chunk_count,
            "put": {
                "hash_calls": put_counts.hash_calls,
                "hash_bytes": put_counts.hash_bytes,
                "snapshot_pins": put_counts.snapshot_pins,
                "manifest_reads": put_counts.manifest_reads,
                "chunk_group_commits": put_counts.chunk_group_commits,
                "chunk_rows_written": put_counts.chunk_rows_written,
            },
            "manifest_only": {
                "hash_calls": manifest_counts.hash_calls,
                "snapshot_pins": manifest_counts.snapshot_pins,
                "manifest_reads": manifest_counts.manifest_reads,
                "manifest_decodes": manifest_counts.manifest_decodes,
                "chunk_reads": manifest_counts.chunk_reads,
            },
            "combined_read": {
                "hash_calls": read_counts.hash_calls,
                "hash_bytes": read_counts.hash_bytes,
                "snapshot_pins": read_counts.snapshot_pins,
                "manifest_reads": read_counts.manifest_reads,
                "manifest_decodes": read_counts.manifest_decodes,
                "chunk_reads": read_counts.chunk_reads,
            },
            "byte_exact": true,
        }));
        expected.push((put.blob_id, put.manifest));
        eprintln!(
            "ISSUE1549_SCALE after size={} chunks={} put_hash_bytes={} put_manifest_reads=0 read_hash_bytes={} read_manifest_reads=1 read_chunk_reads={} byte_exact=true",
            size,
            put.manifest.chunk_count,
            put_counts.hash_bytes,
            read_counts.hash_bytes,
            read_counts.chunk_reads,
        );
    }

    vault.flush().expect("flush issue1549 scaling vault");
    drop(vault);
    let reopened = AsterVault::open(
        &vault_dir,
        vault_id(),
        b"issue1549-durable-scaling".to_vec(),
        VaultOptions::default(),
    )
    .expect("cold-open issue1549 scaling vault");
    let reopened_layer = BlobLayer::new(&reopened);
    for (blob_id, manifest) in &expected {
        assert_eq!(
            reopened_layer
                .blob_manifest(&col, *blob_id)
                .expect("cold manifest read"),
            Some(*manifest)
        );
    }
    let physical_files = fs::read_dir(vault_dir.join("cf").join("blob"))
        .expect("read physical Blob CF")
        .map(|entry| {
            let path = entry.expect("Blob CF entry").path();
            json!({
                "path": path.display().to_string(),
                "bytes": fs::metadata(&path).expect("Blob CF metadata").len(),
            })
        })
        .collect::<Vec<_>>();
    assert!(!physical_files.is_empty());

    let evidence = json!({
        "issue": 1549,
        "source_of_truth": vault_dir.display().to_string(),
        "cold_reopen_manifest_count": expected.len(),
        "physical_blob_cf_files": physical_files,
        "results": results,
    });
    let path = root.join("issue1549-scaling-readback.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&evidence).expect("serialize issue1549 evidence"),
    )
    .expect("write issue1549 evidence");
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("reread issue1549 evidence"))
            .expect("decode issue1549 evidence");
    assert_eq!(persisted["cold_reopen_manifest_count"], sizes.len());
    assert_eq!(
        persisted["results"].as_array().map(Vec::len),
        Some(sizes.len())
    );
    eprintln!(
        "ISSUE1549_SCALE_SOURCE_OF_TRUTH path={} manifests={} physical_files={}",
        path.display(),
        sizes.len(),
        persisted["physical_blob_cf_files"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0),
    );
}

fn chunk_count_for_size(size: usize) -> usize {
    if size == 0 {
        0
    } else {
        size.div_ceil(BLOB_CHUNK_SIZE)
    }
}

fn chunks_per_group() -> usize {
    BLOB_CHUNK_GROUP_VALUE_BYTES / BLOB_CHUNK_SIZE
}
