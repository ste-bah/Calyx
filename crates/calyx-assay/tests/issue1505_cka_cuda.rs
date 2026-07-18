#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use calyx_assay::ensemble_redundancy_from_lenses_cuda_strict;
#[cfg(feature = "cuda")]
use calyx_assay::{EnsembleLensInput, EnsembleRedundancyEvidence, ensemble_redundancy_from_lenses};
#[cfg(feature = "cuda")]
use calyx_core::CalyxError;
use calyx_core::SlotId;
#[cfg(feature = "cuda")]
use serde_json::json;

#[cfg(feature = "cuda")]
#[test]
fn issue1505_cka_cuda_matches_cpu_and_writes_fsv() {
    let parity_lenses = cka_lenses(256, 10);
    let cpu = timed(|| ensemble_redundancy_from_lenses(&parity_lenses, 10).unwrap());
    let gpu = timed(|| ensemble_redundancy_from_lenses_cuda_strict(&parity_lenses, 10).unwrap());
    assert_evidence_close("sampled_10_lenses", &cpu.value, &gpu.value, 2e-4);

    let previous_strict = std::env::var_os("CALYX_ASSAY_CUDA_STRICT");
    unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", "1") };
    let env_strict = ensemble_redundancy_from_lenses(&parity_lenses, 10).unwrap();
    restore_strict_env(previous_strict);
    assert_evidence_close("strict env CKA route", &gpu.value, &env_strict, 0.0);

    let max_lenses = cka_lenses(10_000, 10);
    let max_cpu = timed(|| ensemble_redundancy_from_lenses(&max_lenses, 10).unwrap());
    let max_gpu = timed(|| ensemble_redundancy_from_lenses_cuda_strict(&max_lenses, 10).unwrap());
    assert_eq!(max_gpu.value.method.tuple_count, 160_000);
    assert_evidence_close("max_tuple_10_lenses", &max_cpu.value, &max_gpu.value, 2e-4);

    let edges = edge_case_readbacks(&parity_lenses);
    let artifact = json!({
        "artifact_kind": "issue1505.assay-linear-cka-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1505_FSV_DIR/issue1505-cka-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_isolated_issue1505_cka_cuda issue1505_cka_cuda -- --nocapture",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "happy_path": {
            "sampled_10_lenses": evidence_readback(cpu, gpu),
            "strict_env_route": evidence_json(&env_strict),
            "max_tuple_10_lenses": evidence_readback(max_cpu, max_gpu),
        },
        "edge_cases": edges,
    });
    let restored = write_fsv_artifact(artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1505.assay-linear-cka-cuda-fsv.v1"
    );
    assert_eq!(
        restored["happy_path"]["max_tuple_10_lenses"]["gpu"]["method"]["tuple_count"],
        160_000
    );
    assert_eq!(
        restored["edge_cases"].as_array().unwrap().len(),
        3,
        "issue1505 CKA FSV records three edge cases"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1505_cka_cuda_strict_errors_without_cuda_feature() {
    let lenses = cka_lenses(64, 2);
    let err = ensemble_redundancy_from_lenses_cuda_strict(&lenses, 10).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}

fn cka_lenses(row_count: usize, lens_count: usize) -> Vec<calyx_assay::EnsembleLensInput> {
    (0..lens_count)
        .map(|lens| {
            let dim = 2 + lens % 3;
            let mut rows = Vec::with_capacity(row_count);
            for row in 0..row_count {
                let mut values = Vec::with_capacity(dim);
                for col in 0..dim {
                    values.push(cka_value(row, col, lens));
                }
                rows.push(values);
            }
            calyx_assay::EnsembleLensInput::new(
                format!("lens_{lens:02}"),
                SlotId::new(10_000 + lens as u16),
                rows,
            )
        })
        .collect()
}

fn cka_value(row: usize, col: usize, lens: usize) -> f32 {
    let t = row as f32 * 0.0037;
    let c = col as f32 + 1.0;
    let base0 = (t * 1.3 + c * 0.17).sin() + 0.2 * (t * 0.7 + c).cos();
    match lens {
        0 => base0 + c * 0.01,
        1 => {
            let source = (t * 1.3 + c * 0.17).sin() + 0.2 * (t * 0.7 + c).cos();
            source * if col.is_multiple_of(2) { 1.6 } else { -0.9 } + 0.75
        }
        _ => {
            let freq = 0.8 + lens as f32 * 0.11;
            (t * freq + c * 0.31).sin() * (0.35 + lens as f32 * 0.015)
                + (t * (1.7 + lens as f32 * 0.05) - c).cos() * 0.25
                + (row % (lens + 3)) as f32 * 0.0007
        }
    }
}

#[cfg(feature = "cuda")]
fn edge_case_readbacks(lenses: &[EnsembleLensInput]) -> Vec<serde_json::Value> {
    let empty = ensemble_redundancy_from_lenses_cuda_strict(&[], 10).unwrap_err();

    let mut nonfinite = lenses.to_vec();
    nonfinite[2].vectors[17][1] = f32::NAN;
    let nonfinite_err = ensemble_redundancy_from_lenses_cuda_strict(&nonfinite, 10).unwrap_err();

    let mut constant = lenses.to_vec();
    constant[3].vectors = vec![vec![1.0, 1.0]; lenses[3].vectors.len()];
    let constant_err = ensemble_redundancy_from_lenses_cuda_strict(&constant, 10).unwrap_err();

    vec![
        edge("empty_lenses", json!({"lens_count": 0}), empty),
        edge(
            "nonfinite_lens_value",
            json!({"lens": "lens_02", "row": 17, "col": 1}),
            nonfinite_err,
        ),
        edge(
            "zero_centered_energy",
            json!({"lens": "lens_03", "constant_rows": constant[3].vectors.len()}),
            constant_err,
        ),
    ]
}

#[cfg(feature = "cuda")]
fn edge(name: &'static str, before: serde_json::Value, err: CalyxError) -> serde_json::Value {
    json!({
        "name": name,
        "before": before,
        "after": {
            "code": err.code,
            "message": err.message,
        }
    })
}

#[cfg(feature = "cuda")]
struct Timed<T> {
    value: T,
    elapsed_ms: f64,
}

#[cfg(feature = "cuda")]
fn timed<T>(f: impl FnOnce() -> T) -> Timed<T> {
    let started = Instant::now();
    let value = f();
    Timed {
        value,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(feature = "cuda")]
fn evidence_readback(
    cpu: Timed<EnsembleRedundancyEvidence>,
    gpu: Timed<EnsembleRedundancyEvidence>,
) -> serde_json::Value {
    json!({
        "cpu_ms": cpu.elapsed_ms,
        "gpu_ms": gpu.elapsed_ms,
        "speedup": speedup(cpu.elapsed_ms, gpu.elapsed_ms),
        "cpu": evidence_json(&cpu.value),
        "gpu": evidence_json(&gpu.value),
    })
}

#[cfg(feature = "cuda")]
fn evidence_json(evidence: &EnsembleRedundancyEvidence) -> serde_json::Value {
    json!({
        "method": evidence.method,
        "pair_count": evidence.pairs.len(),
        "pairs": evidence.pairs,
    })
}

#[cfg(feature = "cuda")]
fn assert_evidence_close(
    name: &str,
    left: &EnsembleRedundancyEvidence,
    right: &EnsembleRedundancyEvidence,
    tolerance: f32,
) {
    assert_eq!(left.method, right.method, "{name} method mismatch");
    assert_eq!(left.pairs.len(), right.pairs.len(), "{name} pair count");
    for (idx, (left, right)) in left.pairs.iter().zip(&right.pairs).enumerate() {
        assert_eq!(left.a, right.a, "{name} pair {idx} a");
        assert_eq!(left.b, right.b, "{name} pair {idx} b");
        assert_eq!(left.slot_a, right.slot_a, "{name} pair {idx} slot_a");
        assert_eq!(left.slot_b, right.slot_b, "{name} pair {idx} slot_b");
        assert_close(&format!("{name} pair {idx} nmi"), left.nmi, right.nmi, 0.0);
        assert_close(
            &format!("{name} pair {idx} raw"),
            left.linear_cka.raw_signed_point,
            right.linear_cka.raw_signed_point,
            tolerance,
        );
        assert_close(
            &format!("{name} pair {idx} point"),
            left.linear_cka.redundancy_point,
            right.linear_cka.redundancy_point,
            tolerance,
        );
        assert_close(
            &format!("{name} pair {idx} se"),
            left.linear_cka.mc_standard_error,
            right.linear_cka.mc_standard_error,
            tolerance,
        );
        assert_close(
            &format!("{name} pair {idx} gate"),
            left.linear_cka.mc_gate_upper_estimate,
            right.linear_cka.mc_gate_upper_estimate,
            tolerance,
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_close(name: &str, left: f32, right: f32, tolerance: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch: left={left} right={right} diff={diff} tolerance={tolerance}"
    );
}

#[cfg(feature = "cuda")]
fn speedup(cpu_ms: f64, gpu_ms: f64) -> f64 {
    if gpu_ms > 0.0 {
        cpu_ms / gpu_ms
    } else {
        f64::INFINITY
    }
}

#[cfg(feature = "cuda")]
fn restore_strict_env(previous: Option<std::ffi::OsString>) {
    match previous {
        Some(value) => unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", value) },
        None => unsafe { std::env::remove_var("CALYX_ASSAY_CUDA_STRICT") },
    }
}

#[cfg(feature = "cuda")]
fn write_fsv_artifact(value: serde_json::Value) -> serde_json::Value {
    let root = std::env::var_os("CALYX_ASSAY_ISSUE1505_FSV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/issue1505-fsv"));
    std::fs::create_dir_all(&root).expect("create issue1505 FSV dir");
    let path = root.join("issue1505-cka-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1505 CKA FSV");
    std::fs::write(&path, bytes).expect("write issue1505 CKA FSV");
    let readback = std::fs::read(&path).expect("read issue1505 CKA FSV");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1505 CKA FSV");
    println!(
        "ISSUE1505_CKA_FSV_READBACK path={} bytes={} blake3={}",
        path.display(),
        readback.len(),
        blake3::hash(&readback).to_hex()
    );
    println!(
        "ISSUE1505_CKA_FSV_DATA {}",
        String::from_utf8_lossy(&readback)
    );
    restored
}
