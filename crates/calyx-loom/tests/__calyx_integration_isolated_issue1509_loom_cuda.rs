//! Process-isolated release benchmark and persisted readback for issue #1509.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use calyx_aster::cf::CfRouter;
use calyx_core::{CxId, SlotId};
use calyx_loom::agreement_graph::XtermRow;
use calyx_loom::{CrossTermValue, LOOM_CUDA_STRICT_ENV, LoomStore, SignalProvenanceTag};
use serde_json::json;

#[test]
#[ignore = "release-only GPU-host utilization/speedup benchmark; writes issue #1509 FSV"]
fn issue1509_loom_cuda_production_benchmark_and_persisted_readback() {
    let rows = env_usize("CALYX_LOOM_ISSUE1509_ROWS", 256);
    let dim = env_usize("CALYX_LOOM_ISSUE1509_DIM", 4_096);
    let gpu_iters = env_usize("CALYX_LOOM_ISSUE1509_GPU_ITERS", 3);
    assert!(rows >= 10, "acceptance requires a real >=10-lens panel");
    assert!(dim > 0);
    assert!(gpu_iters > 1, "two iterations prove workspace reuse");
    let root = PathBuf::from(
        std::env::var_os("CALYX_LOOM_ISSUE1509_FSV_DIR")
            .expect("CALYX_LOOM_ISSUE1509_FSV_DIR must name an absolute evidence directory"),
    );
    assert!(root.is_absolute());
    std::fs::create_dir_all(&root).unwrap();

    let warm = panel(12, 64);
    let mut warm_store = LoomStore::new(128);
    warm_store
        .weave_cuda_strict(CxId::from_bytes([71; 16]), &warm)
        .unwrap();

    let slots = panel(rows, dim);
    let previous = std::env::var_os(LOOM_CUDA_STRICT_ENV);
    unsafe { std::env::remove_var(LOOM_CUDA_STRICT_ENV) };
    let mut cpu = LoomStore::new(rows * rows);
    let started = Instant::now();
    let cpu_pairs = cpu.weave(CxId::from_bytes([72; 16]), &slots).unwrap();
    let cpu_ms = started.elapsed().as_secs_f64() * 1_000.0;
    assert!(cpu.last_cuda_stats().is_none());

    unsafe { std::env::set_var(LOOM_CUDA_STRICT_ENV, "1") };
    let mut gpu_iteration_ms = Vec::with_capacity(gpu_iters);
    let mut gpu = LoomStore::new(rows * rows);
    for _iteration in 0..gpu_iters {
        let mut candidate = LoomStore::new(rows * rows);
        let started = Instant::now();
        let gpu_pairs = candidate.weave(CxId::from_bytes([72; 16]), &slots).unwrap();
        gpu_iteration_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(gpu_pairs, cpu_pairs);
        gpu = candidate;
    }
    restore_env(previous);
    let gpu_ms = gpu_iteration_ms.iter().sum::<f64>() / gpu_iteration_ms.len() as f64;
    let speedup = cpu_ms / gpu_ms;
    assert!(speedup.is_finite() && speedup > 0.0);
    let stats = gpu.last_cuda_stats().unwrap().clone();
    assert_eq!(stats.row_count, rows);
    assert_eq!(stats.agreement_pairs, cpu_pairs);
    assert_eq!(stats.host_to_device_copies, 3);
    assert_eq!(stats.device_to_host_copies, 2);
    assert_eq!(stats.kernel_launches, 3);
    assert_eq!(stats.gemm_calls, 1);
    assert!(stats.workspace_reused);
    assert!(stats.host_to_device_copies < stats.agreement_pairs);
    assert_rows_close(&gpu.xterm_rows(), &cpu.xterm_rows());

    let readback_slots = panel(12, 64);
    let mut readback_store = LoomStore::new(128);
    readback_store
        .weave_cuda_strict(CxId::from_bytes([74; 16]), &readback_slots)
        .unwrap();
    let vault = root.join("xterm-vault");
    assert!(
        !vault.exists(),
        "use a fresh FSV directory for persisted readback"
    );
    let mut router = CfRouter::open(&vault, 1024).unwrap();
    let persisted = readback_store.persist_xterms_to_aster(&mut router).unwrap();
    drop(router);
    let reopened = CfRouter::open(&vault, 1024).unwrap();
    let loaded = LoomStore::load_xterms_from_aster(&reopened, 16).unwrap();
    assert_eq!(loaded.xterm_count(), readback_store.xterm_count());
    assert_eq!(loaded.xterm_rows(), readback_store.xterm_rows());

    let artifact = json!({
        "artifact_kind": "issue1509.loom-production-cuda-benchmark.v1",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "panel": {"lens_count": rows, "dimension": dim, "pair_count": cpu_pairs},
        "timing": {
            "cpu_ms": cpu_ms,
            "gpu_mean_ms": gpu_ms,
            "gpu_iteration_ms": gpu_iteration_ms,
            "speedup": speedup,
        },
        "production_route": {
            "environment": LOOM_CUDA_STRICT_ENV,
            "value": "1",
            "standard_weave_observed_cuda_stats": true,
            "stats": stats,
        },
        "persisted_readback": {
            "lens_count": 12,
            "dimension": 64,
            "rows_written": persisted,
            "rows_loaded": loaded.xterm_count(),
            "encoding": "LoomStore::persist_xterms_to_aster / load_xterms_from_aster",
            "exact_row_equality": true,
        },
        "minimum_sufficient_corpus": {
            "lens_floor": 10,
            "why": "The panel exercises all-pairs ordering and makes launch/copy counts comparable to pair_count.",
            "why_larger_is_configurable": "Rows, dimension, and iterations are environment-controlled for utilization sampling without burdening the normal gate."
        }
    });
    let path = root.join("issue1509-loom-cuda-benchmark.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
    let restored: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(restored["persisted_readback"]["rows_loaded"], persisted);
    println!(
        "ISSUE1509_LOOM_BENCH rows={rows} dim={dim} pairs={cpu_pairs} cpu_ms={cpu_ms:.3} gpu_ms={gpu_ms:.3} speedup={speedup:.3} persisted={persisted}"
    );
}

fn panel(rows: usize, dim: usize) -> BTreeMap<SlotId, Vec<f32>> {
    assert!(rows <= u16::MAX as usize);
    (0..rows)
        .map(|row| {
            let values = (0..dim)
                .map(|col| {
                    let stripe = (col % 31) as f32 - 15.0;
                    (row + 1) as f32 * 0.0031 + stripe * 0.017
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
        assert_eq!(actual.tag, SignalProvenanceTag::Derived);
        match (&actual.value, &expected.value) {
            (CrossTermValue::Scalar(actual), CrossTermValue::Scalar(expected)) => {
                assert!((actual - expected).abs() <= 2.0e-4);
            }
            _ => panic!("weave persisted a non-scalar agreement"),
        }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("benchmark variable must be usize"))
        .unwrap_or(default)
}

fn restore_env(previous: Option<std::ffi::OsString>) {
    if let Some(value) = previous {
        unsafe { std::env::set_var(LOOM_CUDA_STRICT_ENV, value) };
    } else {
        unsafe { std::env::remove_var(LOOM_CUDA_STRICT_ENV) };
    }
}
