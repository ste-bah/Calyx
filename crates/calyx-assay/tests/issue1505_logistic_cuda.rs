#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use calyx_assay::logistic_probe_mi_cuda_strict;
#[cfg(feature = "cuda")]
use calyx_assay::{
    LogisticProbeReport, logistic_probe_mi_multiseed, logistic_probe_mi_multiseed_calibrated,
    logistic_probe_mi_multiseed_calibrated_cuda_strict, logistic_probe_mi_multiseed_cuda_strict,
};
#[cfg(feature = "cuda")]
use calyx_core::CalyxError;
#[cfg(feature = "cuda")]
use serde_json::json;

#[cfg(feature = "cuda")]
#[test]
fn issue1505_logistic_cuda_matches_cpu_and_writes_fsv() {
    let (samples, labels, groups) = logistic_fixture(96, 6);

    let cpu = timed(|| logistic_probe_mi_multiseed(&samples, &labels, Some(&groups)).unwrap());
    let gpu = timed(|| {
        logistic_probe_mi_multiseed_cuda_strict(&samples, &labels, Some(&groups)).unwrap()
    });
    assert_report_close("multiseed logistic", &cpu.value, &gpu.value, 1e-5);

    let cpu_cal =
        timed(|| logistic_probe_mi_multiseed_calibrated(&samples, &labels, Some(&groups)).unwrap());
    let gpu_cal = timed(|| {
        logistic_probe_mi_multiseed_calibrated_cuda_strict(&samples, &labels, Some(&groups))
            .unwrap()
    });
    assert_report_close("calibrated logistic", &cpu_cal.value, &gpu_cal.value, 1e-5);
    assert_power_close(
        cpu_cal
            .value
            .estimate
            .power_calibration
            .as_ref()
            .expect("CPU calibration"),
        gpu_cal
            .value
            .estimate
            .power_calibration
            .as_ref()
            .expect("CUDA calibration"),
        1e-5,
    );

    let previous_strict = std::env::var_os("CALYX_ASSAY_CUDA_STRICT");
    unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", "1") };
    let env_strict = logistic_probe_mi_multiseed(&samples, &labels, Some(&groups)).unwrap();
    restore_strict_env(previous_strict);
    assert_report_close("strict env route", &gpu.value, &env_strict, 0.0);

    let edges = edge_case_readbacks(&samples, &labels, &groups);
    let artifact = json!({
        "artifact_kind": "issue1505.assay-logistic-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1505_FSV_DIR/issue1505-logistic-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_isolated_issue1505_logistic_cuda issue1505_logistic_cuda -- --nocapture",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "happy_path": {
            "multiseed": logistic_readback(cpu, gpu),
            "calibrated": logistic_readback(cpu_cal, gpu_cal),
            "strict_env_route": report_json(&env_strict),
        },
        "edge_cases": edges,
    });
    let restored = write_fsv_artifact(artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1505.assay-logistic-cuda-fsv.v1"
    );
    assert_eq!(
        restored["edge_cases"].as_array().unwrap().len(),
        4,
        "issue1505 logistic FSV records four edge cases"
    );
    assert_eq!(
        restored["happy_path"]["multiseed"]["gpu"]["estimate"]["bits"],
        restored["happy_path"]["strict_env_route"]["estimate"]["bits"]
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1505_logistic_cuda_strict_errors_without_cuda_feature() {
    let (samples, labels, _) = logistic_fixture(64, 4);
    let err = logistic_probe_mi_cuda_strict(&samples, &labels).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}

fn logistic_fixture(n: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<bool>, Vec<String>) {
    let mut samples = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut groups = Vec::with_capacity(n);
    for row in 0..n {
        let label = row % 2 == 1;
        let sign = if label { 1.0 } else { -1.0 };
        let mut sample = Vec::with_capacity(dim);
        for col in 0..dim {
            let value = match col {
                0 => sign * 3.0,
                1 => (row as f32 + 1.0) * 0.01,
                2 => sign * 0.75 + (row % 5) as f32 * 0.01,
                _ => ((row + col) as f32 * 0.037).sin() * 0.05,
            };
            sample.push(value);
        }
        samples.push(sample);
        labels.push(label);
        groups.push(format!("row-{row}"));
    }
    (samples, labels, groups)
}

#[cfg(feature = "cuda")]
fn edge_case_readbacks(
    samples: &[Vec<f32>],
    labels: &[bool],
    groups: &[String],
) -> Vec<serde_json::Value> {
    let empty = logistic_probe_mi_cuda_strict(&[], &[]).unwrap_err();

    let mut nonfinite = samples.to_vec();
    nonfinite[7][1] = f32::NAN;
    let nonfinite_err =
        logistic_probe_mi_multiseed_cuda_strict(&nonfinite, labels, Some(groups)).unwrap_err();

    let degenerate_labels = vec![false; labels.len()];
    let degenerate_err = logistic_probe_mi_multiseed_calibrated_cuda_strict(
        samples,
        &degenerate_labels,
        Some(groups),
    )
    .unwrap_err();

    let (oversized, oversized_labels, oversized_groups) = logistic_fixture(labels.len(), 1025);
    let oversized_err = logistic_probe_mi_multiseed_cuda_strict(
        &oversized,
        &oversized_labels,
        Some(&oversized_groups),
    )
    .unwrap_err();

    vec![
        edge("empty_inputs", json!({"rows": 0, "labels": 0}), empty),
        edge(
            "nonfinite_sample",
            json!({"rows": samples.len(), "nan_row": 7, "nan_col": 1}),
            nonfinite_err,
        ),
        edge(
            "degenerate_calibration_labels",
            json!({"rows": labels.len(), "positive_labels": 0}),
            degenerate_err,
        ),
        edge(
            "dimension_over_cuda_limit",
            json!({"rows": oversized.len(), "dim": 1025}),
            oversized_err,
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
fn logistic_readback(
    cpu: Timed<LogisticProbeReport>,
    gpu: Timed<LogisticProbeReport>,
) -> serde_json::Value {
    json!({
        "cpu_ms": cpu.elapsed_ms,
        "gpu_ms": gpu.elapsed_ms,
        "speedup": speedup(cpu.elapsed_ms, gpu.elapsed_ms),
        "cpu": report_json(&cpu.value),
        "gpu": report_json(&gpu.value),
    })
}

#[cfg(feature = "cuda")]
fn report_json(report: &LogisticProbeReport) -> serde_json::Value {
    json!({
        "estimate": report.estimate,
        "accuracy": report.accuracy,
        "selected_field": report.selected_field,
    })
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
fn assert_report_close(
    name: &str,
    left: &LogisticProbeReport,
    right: &LogisticProbeReport,
    tolerance: f32,
) {
    assert_close(
        &format!("{name} bits"),
        left.estimate.bits,
        right.estimate.bits,
        tolerance,
    );
    assert_close(
        &format!("{name} ci_low"),
        left.estimate.ci_low,
        right.estimate.ci_low,
        tolerance,
    );
    assert_close(
        &format!("{name} ci_high"),
        left.estimate.ci_high,
        right.estimate.ci_high,
        tolerance,
    );
    assert_close(
        &format!("{name} accuracy"),
        left.accuracy,
        right.accuracy,
        tolerance,
    );
    assert_eq!(left.selected_field, right.selected_field);
    assert_eq!(left.estimate.n_samples, right.estimate.n_samples);
    assert_eq!(left.estimate.estimator, right.estimate.estimator);
    assert_eq!(left.estimate.trust, right.estimate.trust);
    let left_reliability = left
        .estimate
        .reliability
        .as_ref()
        .expect("left reliability");
    let right_reliability = right
        .estimate
        .reliability
        .as_ref()
        .expect("right reliability");
    assert_eq!(left_reliability.seed_count, right_reliability.seed_count);
    assert_eq!(left_reliability.unresolved, right_reliability.unresolved);
    assert_close(
        &format!("{name} seed_sigma"),
        left_reliability.seed_sigma,
        right_reliability.seed_sigma,
        tolerance,
    );
}

#[cfg(feature = "cuda")]
fn assert_power_close(
    left: &calyx_assay::PowerCalibration,
    right: &calyx_assay::PowerCalibration,
    tolerance: f32,
) {
    assert_eq!(left.status, right.status);
    assert_eq!(left.n_samples, right.n_samples);
    assert_eq!(left.n_features, right.n_features);
    assert_eq!(left.planted_column, right.planted_column);
    assert_close(
        "power planted_bits",
        left.planted_bits,
        right.planted_bits,
        tolerance,
    );
    assert_close(
        "power recovered_bits",
        left.recovered_bits,
        right.recovered_bits,
        tolerance,
    );
    assert_close(
        "power recovery_ratio",
        left.recovery_ratio,
        right.recovery_ratio,
        tolerance,
    );
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
    let path = root.join("issue1505-logistic-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1505 logistic FSV");
    std::fs::write(&path, bytes).expect("write issue1505 logistic FSV");
    let readback = std::fs::read(&path).expect("read issue1505 logistic FSV");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1505 logistic FSV");
    println!(
        "ISSUE1505_LOGISTIC_FSV_READBACK path={} bytes={} blake3={}",
        path.display(),
        readback.len(),
        blake3::hash(&readback).to_hex()
    );
    println!(
        "ISSUE1505_LOGISTIC_FSV_DATA {}",
        String::from_utf8_lossy(&readback)
    );
    restored
}
