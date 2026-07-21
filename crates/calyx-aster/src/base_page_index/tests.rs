use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use calyx_ledger::LedgerHeadAnchor;

use super::*;
use crate::manifest::{ImmutableRef, ManifestStore, VaultManifest};
use crate::sst::{SstReader, write_sst};
use crate::vault::encode::{WriteRow, encode_write_batch};
use crate::wal::{Wal, WalOptions};

#[test]
fn index_pages_preserve_latest_sorted_live_rows_and_skip_tombstones() {
    let root = temp_root("latest-live");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"old-a".as_slice()), (b"b", b"old-b")],
    )
    .unwrap();
    let tombstone = crate::mvcc::tombstone_value();
    write_sst(
        base.join("00000000000000000002.sst"),
        [(b"a".as_slice(), b"new-a".as_slice()), (b"b", &tombstone)],
    )
    .unwrap();

    let mut progress = Vec::new();
    let manifest = build_base_page_index(&root, 1, |event| {
        progress.push(event);
        Ok(())
    })
    .unwrap();
    let read_manifest = read_base_page_index_manifest(&root).unwrap();
    let rows = read_indexed_base_rows(&root, 10).unwrap();

    assert_eq!(manifest.total_entries, 2);
    assert_eq!(manifest.live_entries, 1);
    assert_eq!(manifest.tombstone_entries, 1);
    assert_eq!(read_manifest.pages.len(), 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows.get(b"a".as_slice()).unwrap(), b"new-a");
    assert!(matches!(
        progress.last(),
        Some(BasePageIndexBuildProgress::Complete {
            live_entries: 1,
            ..
        })
    ));
    cleanup(root);
}

#[test]
fn missing_index_fails_closed_for_bounded_read() {
    let root = temp_root("missing");
    let error = read_indexed_base_rows(&root, 1).unwrap_err();

    assert_eq!(error.code, MISSING_CODE);
    assert!(error.message.contains("manifest is missing"));
    cleanup(root);
}

#[test]
fn stale_ledger_head_fails_closed() {
    let root = temp_root("stale-head");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"value".as_slice())],
    )
    .unwrap();
    build_base_page_index(&root, 4, |_| Ok(())).unwrap();
    fs::create_dir_all(root.join("ledger_head")).unwrap();
    let anchor = LedgerHeadAnchor::new(1, [1_u8; 32]).unwrap();
    fs::write(
        root.join("ledger_head").join("current.json"),
        serde_json::to_vec(&anchor).unwrap(),
    )
    .unwrap();

    let error = read_indexed_base_rows(&root, 1).unwrap_err();

    assert_eq!(error.code, STALE_CODE);
    assert!(error.message.contains("current head"));
    cleanup(root);
}

#[test]
fn page_sha_mismatch_fails_closed() {
    let root = temp_root("page-sha");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"value".as_slice())],
    )
    .unwrap();
    let manifest = build_base_page_index(&root, 4, |_| Ok(())).unwrap();
    fs::write(
        root.join(BASE_PAGE_INDEX_DIR).join(&manifest.pages[0].path),
        b"{\"entries\":[]}",
    )
    .unwrap();

    let error = read_indexed_base_rows(&root, 1).unwrap_err();

    assert_eq!(error.code, CORRUPT_CODE);
    assert!(error.message.contains("sha256 mismatch"));
    cleanup(root);
}

#[test]
fn indexed_key_read_returns_requested_rows_and_tombstones() {
    let root = temp_root("keyed-read");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    let tombstone = crate::mvcc::tombstone_value();
    write_sst(
        base.join("00000000000000000001.sst"),
        [
            (b"a".as_slice(), b"live".as_slice()),
            (b"b".as_slice(), tombstone.as_slice()),
        ],
    )
    .unwrap();
    build_base_page_index(&root, 1, |_| Ok(())).unwrap();

    let rows = read_indexed_base_rows_for_keys(
        &root,
        &[b"a".to_vec(), b"b".to_vec(), b"missing".to_vec()],
    )
    .unwrap();

    assert_eq!(rows.get(b"a".as_slice()).unwrap(), &Some(b"live".to_vec()));
    assert_eq!(rows.get(b"b".as_slice()).unwrap(), &Some(tombstone));
    assert_eq!(rows.get(b"missing".as_slice()).unwrap(), &None);
    cleanup(root);
}

#[test]
fn selected_key_visitor_reads_each_touched_page_once_and_deduplicates_keys() {
    let root = temp_root("selected-key-visitor");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [
            (b"a".as_slice(), b"value-a".as_slice()),
            (b"b".as_slice(), b"value-b".as_slice()),
            (b"c".as_slice(), b"value-c".as_slice()),
            (b"d".as_slice(), b"value-d".as_slice()),
        ],
    )
    .unwrap();
    build_base_page_index(&root, 2, |_| Ok(())).unwrap();
    let mut seen = Vec::new();

    let stats = visit_indexed_base_rows_for_keys(
        &root,
        &[
            b"d".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"b".to_vec(),
            b"z".to_vec(),
        ],
        |key, value| -> std::result::Result<(), calyx_core::CalyxError> {
            seen.push((key.to_vec(), value));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(stats.unique_keys, 4);
    assert_eq!(stats.touched_pages, 2);
    assert_eq!(stats.source_files, 1);
    assert_eq!(stats.live_rows, 3);
    assert_eq!(stats.missing_rows, 1);
    seen.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(
        seen,
        vec![
            (b"a".to_vec(), Some(b"value-a".to_vec())),
            (b"b".to_vec(), Some(b"value-b".to_vec())),
            (b"d".to_vec(), Some(b"value-d".to_vec())),
            (b"z".to_vec(), None),
        ]
    );
    cleanup(root);
}

#[test]
fn selected_key_visitor_empty_request_reads_no_pages() {
    let root = temp_root("selected-key-empty");
    build_base_page_index(&root, 2, |_| Ok(())).unwrap();

    let stats = visit_indexed_base_rows_for_keys(
        &root,
        &[],
        |_, _| -> std::result::Result<(), calyx_core::CalyxError> {
            panic!("empty request must not invoke visitor")
        },
    )
    .unwrap();

    assert_eq!(stats, SelectedBaseRowsVisit::default());
    cleanup(root);
}

#[test]
fn selected_wal_rows_use_exact_offsets_and_open_one_source_file() {
    let root = temp_root("selected-wal-offsets");
    let mut wal = Wal::open(root.join("wal"), WalOptions::default()).unwrap();
    let large_unrelated_slot = vec![7_u8; 2 * 1024 * 1024];
    let payload = encode_write_batch(&[
        WriteRow {
            cf: ColumnFamily::Base,
            key: b"a".to_vec(),
            value: b"base-a".to_vec(),
        },
        WriteRow {
            cf: ColumnFamily::slot_raw(calyx_core::SlotId::new(3)),
            key: b"a".to_vec(),
            value: large_unrelated_slot,
        },
        WriteRow {
            cf: ColumnFamily::Base,
            key: b"b".to_vec(),
            value: b"base-b".to_vec(),
        },
    ])
    .unwrap();
    wal.append(&payload).unwrap();
    drop(wal);
    let manifest = build_base_page_index(&root, 8, |_| Ok(())).unwrap();
    let page = read_page(&root, &manifest.pages[0]).unwrap();
    assert!(page.entries.iter().all(|entry| matches!(
        entry.source,
        BasePageIndexSource::Wal {
            row_offset: Some(_),
            ..
        }
    )));
    let mut seen = BTreeMap::new();

    let stats = visit_indexed_base_rows_for_keys(
        &root,
        &[b"b".to_vec(), b"a".to_vec()],
        |key, value| -> std::result::Result<(), CalyxError> {
            seen.insert(key.to_vec(), value.unwrap());
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(stats.source_files, 1);
    assert_eq!(stats.live_rows, 2);
    assert_eq!(seen[b"a".as_slice()], b"base-a");
    assert_eq!(seen[b"b".as_slice()], b"base-b");
    cleanup(root);
}

#[test]
fn checkpointed_wal_payload_is_skipped_and_sst_remains_the_source() {
    let root = temp_root("checkpointed-wal-skip");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"base-a".as_slice())],
    )
    .unwrap();
    let mut wal = Wal::open(root.join("wal"), WalOptions::default()).unwrap();
    let ack = wal
        .append(
            &encode_write_batch(&[
                WriteRow {
                    cf: ColumnFamily::Base,
                    key: b"a".to_vec(),
                    value: b"base-a".to_vec(),
                },
                WriteRow {
                    cf: ColumnFamily::slot_raw(calyx_core::SlotId::new(3)),
                    key: b"a".to_vec(),
                    value: vec![9_u8; 2 * 1024 * 1024],
                },
            ])
            .unwrap(),
        )
        .unwrap();
    drop(wal);
    let store = ManifestStore::open(&root);
    let mut durable = store.load_current().unwrap();
    durable.manifest_seq += 1;
    durable.durable_seq = ack.seq;
    store.write_current(&durable).unwrap();

    let manifest = build_base_page_index(&root, 8, |_| Ok(())).unwrap();
    let page = read_page(&root, &manifest.pages[0]).unwrap();

    assert_eq!(manifest.wal_records, 0);
    assert!(matches!(
        page.entries[0].source,
        BasePageIndexSource::Sst {
            record_offset: Some(_),
            ..
        }
    ));
    assert_eq!(
        read_indexed_base_rows_for_keys(&root, &[b"a".to_vec()]).unwrap()[b"a".as_slice()],
        Some(b"base-a".to_vec())
    );
    cleanup(root);
}

#[test]
fn generation_without_exact_source_offsets_fails_stale() {
    let root = temp_root("legacy-source-offset");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"value".as_slice())],
    )
    .unwrap();
    let mut manifest = build_base_page_index(&root, 4, |_| Ok(())).unwrap();
    let page_path = root.join(BASE_PAGE_INDEX_DIR).join(&manifest.pages[0].path);
    let mut page: BasePageIndexPage =
        serde_json::from_slice(&fs::read(&page_path).unwrap()).unwrap();
    let BasePageIndexSource::Sst { record_offset, .. } = &mut page.entries[0].source else {
        panic!("fixture must use an SST source")
    };
    *record_offset = None;
    let page_bytes = serde_json::to_vec(&page).unwrap();
    fs::write(&page_path, &page_bytes).unwrap();
    manifest.pages[0].sha256_hex = sha256_hex(&page_bytes);
    fs::write(
        root.join(BASE_PAGE_INDEX_DIR)
            .join(BASE_PAGE_INDEX_MANIFEST),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();

    let error = read_indexed_base_rows_for_keys(&root, &[b"a".to_vec()]).unwrap_err();

    assert_eq!(error.code, STALE_CODE);
    assert!(error.message.contains("exact record offset"));
    cleanup(root);
}

#[test]
fn row_page_visitor_stops_after_first_verified_page() {
    let root = temp_root("page-visitor-stop");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [
            (b"a".as_slice(), b"value-a".as_slice()),
            (b"b".as_slice(), b"value-b".as_slice()),
            (b"c".as_slice(), b"value-c".as_slice()),
        ],
    )
    .unwrap();
    build_base_page_index(&root, 1, |_| Ok(())).unwrap();

    let mut seen = Vec::new();
    let live_rows_read = visit_indexed_base_row_pages(
        &root,
        |offset, rows| -> std::result::Result<bool, calyx_core::CalyxError> {
            let keys = rows
                .iter()
                .map(|(key, _)| String::from_utf8(key.clone()).unwrap())
                .collect::<Vec<_>>();
            seen.push((offset, keys));
            Ok(false)
        },
    )
    .unwrap();

    assert_eq!(live_rows_read, 1);
    assert_eq!(seen, vec![(0, vec!["a".to_string()])]);
    cleanup(root);
}

#[test]
fn advancing_index_head_preserves_pages_when_base_files_unchanged() {
    let root = temp_root("advance-head");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"value".as_slice())],
    )
    .unwrap();
    build_base_page_index(&root, 4, |_| Ok(())).unwrap();
    fs::create_dir_all(root.join("ledger_head")).unwrap();
    let anchor = LedgerHeadAnchor::new(7, [7_u8; 32]).unwrap();
    fs::write(
        root.join("ledger_head").join("current.json"),
        serde_json::to_vec(&anchor).unwrap(),
    )
    .unwrap();

    assert!(advance_base_page_index_head_if_base_unchanged(&root).unwrap());
    let manifest = read_base_page_index_manifest(&root).unwrap();

    assert_eq!(manifest.ledger_head_height, 7);
    assert_eq!(manifest.base_sst_files, 1);
    assert_eq!(read_indexed_base_rows(&root, 1).unwrap().len(), 1);
    cleanup(root);
}

#[test]
fn advancing_index_head_refuses_base_file_count_drift() {
    let root = temp_root("advance-head-drift");
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"value".as_slice())],
    )
    .unwrap();
    build_base_page_index(&root, 4, |_| Ok(())).unwrap();
    write_sst(
        base.join("00000000000000000002.sst"),
        [(b"b".as_slice(), b"new".as_slice())],
    )
    .unwrap();
    fs::create_dir_all(root.join("ledger_head")).unwrap();
    let anchor = LedgerHeadAnchor::new(8, [8_u8; 32]).unwrap();
    fs::write(
        root.join("ledger_head").join("current.json"),
        serde_json::to_vec(&anchor).unwrap(),
    )
    .unwrap();

    let error = advance_base_page_index_head_if_base_unchanged(&root).unwrap_err();

    assert_eq!(error.code, STALE_CODE);
    assert!(error.message.contains("current vault has 2"));
    cleanup(root);
}

#[test]
fn interruption_after_page_write_keeps_previous_commit_point_readable() {
    let (root, rows, snapshot, previous) = publication_fixture("interrupt-page");
    let error = write_index_with_hook(
        &root,
        1,
        rows,
        snapshot,
        |event| {
            if matches!(event, BasePageIndexBuildProgress::PageWritten { .. }) {
                return Err(CalyxError::disk_pressure(
                    "test interruption after durable page write",
                ));
            }
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap_err();

    assert_eq!(error.code, "CALYX_DISK_PRESSURE");
    assert_eq!(read_base_page_index_manifest(&root).unwrap(), previous);
    assert_eq!(
        read_indexed_base_rows(&root, 1).unwrap()[b"a".as_slice()],
        b"old"
    );
    cleanup(root);
}

#[test]
fn interruption_before_commit_point_keeps_previous_generation_readable() {
    for boundary in [
        PublicationBoundary::GenerationManifestSynced,
        PublicationBoundary::GenerationPublished,
    ] {
        let (root, rows, snapshot, previous) =
            publication_fixture(&format!("interrupt-{boundary:?}"));
        let error = write_index_with_hook(
            &root,
            1,
            rows,
            snapshot,
            |_| Ok(()),
            |observed| {
                if observed == boundary {
                    return Err(CalyxError::disk_pressure(format!(
                        "test interruption at {boundary:?}"
                    )));
                }
                Ok(())
            },
        )
        .unwrap_err();

        assert_eq!(error.code, "CALYX_DISK_PRESSURE");
        assert_eq!(read_base_page_index_manifest(&root).unwrap(), previous);
        assert_eq!(
            read_indexed_base_rows(&root, 1).unwrap()[b"a".as_slice()],
            b"old"
        );
        cleanup(root);
    }
}

#[test]
fn interruption_after_commit_point_exposes_only_complete_new_generation() {
    let (root, rows, snapshot, previous) = publication_fixture("interrupt-after-commit");
    let error = write_index_with_hook(
        &root,
        1,
        rows,
        snapshot,
        |_| Ok(()),
        |boundary| {
            if boundary == PublicationBoundary::CommitPointPublished {
                return Err(CalyxError::disk_pressure(
                    "test interruption after commit-point publication",
                ));
            }
            Ok(())
        },
    )
    .unwrap_err();

    assert_eq!(error.code, "CALYX_DISK_PRESSURE");
    let published = read_base_page_index_manifest(&root).unwrap();
    assert_ne!(published.generation, previous.generation);
    assert_eq!(published.version, INDEX_VERSION);
    assert_eq!(
        read_indexed_base_rows(&root, 1).unwrap()[b"a".as_slice()],
        b"new"
    );
    cleanup(root);
}

fn publication_fixture(
    name: &str,
) -> (
    PathBuf,
    BTreeMap<Vec<u8>, IndexedValue>,
    BuildSnapshot,
    BasePageIndexManifest,
) {
    let root = temp_root(name);
    let base = root.join("cf").join(ColumnFamily::Base.name());
    fs::create_dir_all(&base).unwrap();
    write_sst(
        base.join("00000000000000000001.sst"),
        [(b"a".as_slice(), b"old".as_slice())],
    )
    .unwrap();
    let previous = build_base_page_index(&root, 1, |_| Ok(())).unwrap();
    let new_sst = base.join("00000000000000000002.sst");
    write_sst(&new_sst, [(b"a".as_slice(), b"new".as_slice())]).unwrap();
    let order = sst_order_key(&new_sst).unwrap().unwrap();
    let record_offset = SstReader::open(&new_sst)
        .unwrap()
        .iter_with_offsets()
        .unwrap()[0]
        .0;
    let rows = BTreeMap::from([(
        b"a".to_vec(),
        IndexedValue {
            value_sha256_hex: sha256_hex(b"new"),
            tombstoned: false,
            source: BasePageIndexSource::Sst {
                path: relative_path(&root, &new_sst),
                order_epoch: order.epoch,
                order_seq: order.seq,
                order_class_rank: order.class_rank,
                order_index: order.index,
                record_offset: Some(record_offset),
            },
        },
    )]);
    let snapshot = BuildSnapshot {
        ledger_head_height: 0,
        ledger_head_tip_hash_hex: hex_bytes(&[0_u8; 32]),
        base_sst_files: 2,
        wal_records: 0,
    };
    (root, rows, snapshot, previous)
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("calyx-base-page-index-{name}-{nanos}"));
    let panel_path = root.join("panel").join("panel.json");
    fs::create_dir_all(panel_path.parent().unwrap()).unwrap();
    let panel_bytes = b"{}";
    fs::write(&panel_path, panel_bytes).unwrap();
    let panel_ref = ImmutableRef::from_bytes("panel/panel.json", panel_bytes).unwrap();
    let manifest = VaultManifest::new(1, 0, panel_ref, Vec::new()).unwrap();
    ManifestStore::open(&root).write_current(&manifest).unwrap();
    root
}

fn cleanup(path: PathBuf) {
    fs::remove_dir_all(path).ok();
}
