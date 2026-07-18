#![cfg(feature = "cuda")]

use super::*;
use crate::error::{
    CALYX_LOOM_DIM_MISMATCH, CALYX_LOOM_NON_FINITE_VECTOR, CALYX_LOOM_ZERO_NORM_VECTOR,
};
use crate::materialization::MaterializationEntry;

const ROWS: usize = 12;
const DIM: usize = 64;

fn panel() -> BTreeMap<SlotId, Vec<f32>> {
    (0..ROWS)
        .map(|row| {
            let values = (0..DIM)
                .map(|col| {
                    let stripe = (col % 11) as f32 - 5.0;
                    (row + 1) as f32 * 0.17 + stripe * 0.053
                })
                .collect();
            (SlotId::new(row as u16 + 1), values)
        })
        .collect()
}

fn assert_rows_close(actual: &[XtermRow], expected: &[XtermRow]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.key, expected.key);
        assert_eq!(actual.tag, expected.tag);
        match (&actual.value, &expected.value) {
            (CrossTermValue::Scalar(actual), CrossTermValue::Scalar(expected)) => {
                assert!((actual - expected).abs() <= 2.0e-4);
            }
            (CrossTermValue::Vector(actual), CrossTermValue::Vector(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (&actual, &expected) in actual.iter().zip(expected) {
                    assert!((actual - expected).abs() <= 1.0e-6);
                }
            }
            _ => panic!("cross-term value kinds differ"),
        }
    }
}

#[test]
fn strict_cuda_weave_matches_cpu_with_constant_transfer_and_reuse() {
    let slots = panel();
    let mut cpu = LoomStore::new(128);
    let mut cuda = LoomStore::new(128);
    let first_cx = CxId::from_bytes([21; 16]);
    let second_cx = CxId::from_bytes([22; 16]);

    assert_eq!(cpu.weave_cpu(first_cx, &slots).unwrap(), 66);
    assert_eq!(cuda.weave_cuda_strict(first_cx, &slots).unwrap(), 66);
    assert_rows_close(&cuda.xterm_rows(), &cpu.xterm_rows());
    let first = cuda.last_cuda_stats().unwrap();
    assert_eq!(first.row_count, ROWS);
    assert_eq!(first.agreement_pairs, 66);
    assert_eq!(first.kernel_launches, 3);
    assert_eq!(first.gemm_calls, 1);
    assert_eq!(first.host_to_device_copies, 3);
    assert_eq!(first.device_to_host_copies, 2);
    assert!(first.host_to_device_copies < first.agreement_pairs);

    assert_eq!(cuda.weave_cuda_strict(second_cx, &slots).unwrap(), 66);
    assert!(cuda.last_cuda_stats().unwrap().workspace_reused);
    println!(
        "LOOM_CUDA_WEAVE rows={ROWS} dim={DIM} pairs=66 launches=3 h2d_copies=3 d2h_copies=2 reused=true"
    );
}

#[test]
fn strict_cuda_materialization_batches_agreement_delta_interaction_and_readback() {
    let slots = panel();
    let cx = CxId::from_bytes([31; 16]);
    let one = SlotId::new(1);
    let two = SlotId::new(2);
    let three = SlotId::new(3);
    let plan = MaterializationPlan {
        entries: vec![
            eager(two, one, CrossTermKind::Agreement),
            eager(one, two, CrossTermKind::Delta),
            eager(one, three, CrossTermKind::Interaction),
            eager(one, two, CrossTermKind::Concat),
        ],
    };
    let mut cpu = LoomStore::new(16);
    let mut cuda = LoomStore::new(16);

    assert_eq!(cpu.materialize_plan_cpu(cx, &slots, &plan).unwrap(), 4);
    assert_eq!(
        cuda.materialize_plan_cuda_strict(cx, &slots, &plan)
            .unwrap(),
        4
    );
    assert_rows_close(&cuda.xterm_rows(), &cpu.xterm_rows());
    let stats = cuda.last_cuda_stats().unwrap();
    assert_eq!(stats.agreement_pairs, 1);
    assert_eq!(stats.vector_terms, 2);
    assert_eq!(stats.kernel_launches, 4);
    assert_eq!(stats.host_to_device_copies, 6);
    assert_eq!(stats.device_to_host_copies, 3);

    let encoded = cuda.xterm_kv_rows().unwrap();
    assert_eq!(encoded.len(), 4);
    for (_, value) in encoded {
        let row: XtermRow = serde_json::from_slice(&value).unwrap();
        assert!(matches!(row.tag, SignalProvenanceTag::Derived));
    }
    println!(
        "LOOM_CUDA_MATERIALIZE agreements=1 vectors=2 concat=1 launches=4 h2d_copies=6 persisted_readback=4"
    );
}

#[test]
fn strict_cuda_edges_fail_closed_with_loom_codes() {
    let cx = CxId::from_bytes([41; 16]);
    let mut store = LoomStore::new(8);
    let zero = BTreeMap::from([
        (SlotId::new(1), vec![0.0; 4]),
        (SlotId::new(2), vec![1.0; 4]),
    ]);
    let error = store.weave_cuda_strict(cx, &zero).unwrap_err();
    assert_eq!(error.code, CALYX_LOOM_ZERO_NORM_VECTOR);

    let drift = BTreeMap::from([
        (SlotId::new(1), vec![1.0; 4]),
        (SlotId::new(2), vec![1.0; 5]),
    ]);
    let error = store.weave_cuda_strict(cx, &drift).unwrap_err();
    assert_eq!(error.code, CALYX_LOOM_DIM_MISMATCH);

    let nonfinite = BTreeMap::from([
        (SlotId::new(1), vec![1.0; 4]),
        (SlotId::new(2), vec![1.0, f32::NAN, 1.0, 1.0]),
    ]);
    let error = store.weave_cuda_strict(cx, &nonfinite).unwrap_err();
    assert_eq!(error.code, CALYX_LOOM_NON_FINITE_VECTOR);
    println!("LOOM_CUDA_EDGES zero_norm dimension_drift nonfinite fail_closed");
}

#[test]
fn production_weave_uses_cuda_when_strict_environment_is_enabled() {
    if !loom_cuda_strict_requested() {
        return;
    }
    let mut store = LoomStore::new(128);
    assert_eq!(
        store.weave(CxId::from_bytes([51; 16]), &panel()).unwrap(),
        66
    );
    assert_eq!(store.last_cuda_stats().unwrap().gemm_calls, 1);
    println!(
        "LOOM_CUDA_PRODUCTION_ROUTE env={} backend=forge_cuda",
        crate::LOOM_CUDA_STRICT_ENV
    );
}

fn eager(a: SlotId, b: SlotId, kind: CrossTermKind) -> MaterializationEntry {
    MaterializationEntry {
        a,
        b,
        kind,
        action: MaterializationAction::EagerStore,
    }
}
