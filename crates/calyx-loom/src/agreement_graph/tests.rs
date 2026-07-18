use super::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

#[cfg(not(feature = "cuda"))]
#[test]
fn strict_weave_requires_cuda_feature_without_fallback() {
    let mut store = LoomStore::new(8);
    let slots = BTreeMap::from([
        (SlotId::new(1), vec![1.0, 0.0]),
        (SlotId::new(2), vec![0.0, 1.0]),
    ]);
    let error = store
        .weave_cuda_strict(CxId::from_bytes([61; 16]), &slots)
        .unwrap_err();
    assert_eq!(error.code, crate::error::CALYX_LOOM_FORGE_UNAVAILABLE);
    assert_eq!(store.xterm_count(), 0);
}

#[test]
fn xterms_roundtrip_through_aster_cf() {
    let dir = test_dir("xterm");
    let mut router = CfRouter::open(&dir, 1024).unwrap();
    let mut store = LoomStore::new(8);
    let slots = BTreeMap::from([
        (SlotId::new(1), vec![1.0, 0.0]),
        (SlotId::new(2), vec![0.0, 1.0]),
    ]);
    store.weave(CxId::from_bytes([1; 16]), &slots).unwrap();

    assert_eq!(store.persist_xterms_to_aster(&mut router).unwrap(), 1);
    drop(router);
    let reopened = CfRouter::open(&dir, 1024).unwrap();
    let loaded = LoomStore::load_xterms_from_aster(&reopened, 8).unwrap();

    assert_eq!(loaded.xterm_count(), 1);
    assert_eq!(loaded.agreement_graph().unwrap()[0].n, 1);
    cleanup(dir);
}

#[test]
fn agreement_graph_rejects_non_finite_xterm_rows() {
    let mut store = LoomStore::new(8);
    store.xterm_cf.insert(
        CrossTermKey {
            cx_id: CxId::from_bytes([9; 16]),
            a: SlotId::new(1),
            b: SlotId::new(2),
            kind: CrossTermKind::Agreement,
        },
        XtermRow {
            key: CrossTermKey {
                cx_id: CxId::from_bytes([9; 16]),
                a: SlotId::new(1),
                b: SlotId::new(2),
                kind: CrossTermKind::Agreement,
            },
            value: CrossTermValue::Scalar(f32::NAN),
            tag: SignalProvenanceTag::Derived,
        },
    );
    let err = store
        .agreement_graph()
        .expect_err("NaN xterm must fail closed");
    assert_eq!(err.code, crate::error::CALYX_LOOM_NON_FINITE_VECTOR);
}

#[test]
fn xterm_kv_rows_match_router_persist_encoding() {
    let dir = test_dir("xterm-kv");
    let mut router = CfRouter::open(&dir, 1024).unwrap();
    let mut store = LoomStore::new(8);
    let slots = BTreeMap::from([
        (SlotId::new(1), vec![1.0, 0.0]),
        (SlotId::new(2), vec![0.0, 1.0]),
        (SlotId::new(3), vec![0.5, 0.5]),
    ]);
    store.weave(CxId::from_bytes([7; 16]), &slots).unwrap();

    // The same three rows, written through the explicit kv-row path used by
    // the corpus weave-loom command (vault.write_cf_batch), must produce a CF
    // that load_xterms_from_aster reads back identically to the in-memory store.
    let kv = store.xterm_kv_rows().unwrap();
    assert_eq!(kv.len(), store.xterm_count());
    for (key, value) in &kv {
        router.put(ColumnFamily::XTerm, key, value).unwrap();
    }
    router.flush_cf(ColumnFamily::XTerm).unwrap();
    drop(router);

    let reopened = CfRouter::open(&dir, 1024).unwrap();
    let loaded = LoomStore::load_xterms_from_aster(&reopened, 8).unwrap();
    assert_eq!(loaded.xterm_count(), store.xterm_count());
    assert_eq!(loaded.xterm_rows(), store.xterm_rows());
    cleanup(dir);
}

fn test_dir(name: &str) -> PathBuf {
    let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("calyx-loom-{name}-{}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: PathBuf) {
    fs::remove_dir_all(dir).unwrap();
}
