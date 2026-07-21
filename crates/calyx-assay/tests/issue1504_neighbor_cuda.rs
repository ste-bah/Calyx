#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use calyx_assay::ksg_mi_continuous_cuda_strict;
#[cfg(feature = "cuda")]
use calyx_assay::{
    CcmConfig, TotalCorrelationConfig, TransferEntropyConfig, convergent_cross_mapping,
    convergent_cross_mapping_cuda_strict, interaction_information_with_config,
    interaction_information_with_config_cuda_strict, ksg_mi_continuous, ksg_mi_continuous_discrete,
    ksg_mi_continuous_discrete_cuda_strict, total_correlation_with_config,
    total_correlation_with_config_cuda_strict, transfer_entropy_with_config,
    transfer_entropy_with_config_cuda_strict,
};
#[cfg(feature = "cuda")]
use calyx_core::{CalyxError, FixedClock};
#[cfg(feature = "cuda")]
use serde_json::json;

#[cfg(feature = "cuda")]
#[test]
fn issue1504_strict_cuda_estimators_match_cpu_and_write_fsv() {
    let clock = FixedClock::new(1_786_150_004);
    let (x, y) = paired_vectors(72);
    let (mixed_x, labels) = mixed_vectors(72);
    let slots = panel_slots(180);
    let (stream_a, stream_b) = recurrence_streams(110);
    let ccm_config = CcmConfig::new(3, 1, vec![20, 80, 150], 0.01, 0.01);
    let tc_config = TotalCorrelationConfig {
        k: 3,
        bootstrap_resamples: 5,
        bootstrap_seed: 0x1504,
    };
    let te_config = TransferEntropyConfig {
        window_size: 2,
        k: 3,
        bootstrap_resamples: 5,
        bootstrap_seed: 0x2504,
    };

    let ksg_cpu = timed(|| ksg_mi_continuous(&x, &y, 3).unwrap());
    let ksg_gpu = timed(|| ksg_mi_continuous_cuda_strict(&x, &y, 3).unwrap());
    assert_close("ksg bits", ksg_cpu.value.bits, ksg_gpu.value.bits, 1e-5);
    assert_eq!(ksg_cpu.value.n_samples, ksg_gpu.value.n_samples);

    let mixed_cpu = timed(|| ksg_mi_continuous_discrete(&mixed_x, &labels, 3).unwrap());
    let mixed_gpu = timed(|| ksg_mi_continuous_discrete_cuda_strict(&mixed_x, &labels, 3).unwrap());
    assert_close(
        "mixed KSG bits",
        mixed_cpu.value.bits,
        mixed_gpu.value.bits,
        1e-5,
    );

    let ccm_cpu =
        timed(|| convergent_cross_mapping("x", &slots[0], "y", &slots[1], &ccm_config).unwrap());
    let ccm_gpu = timed(|| {
        convergent_cross_mapping_cuda_strict("x", &slots[0], "y", &slots[1], &ccm_config).unwrap()
    });
    assert_close(
        "ccm x_to_y final rho",
        ccm_cpu.value.x_manifold_to_y.final_rho,
        ccm_gpu.value.x_manifold_to_y.final_rho,
        1e-5,
    );
    assert_close(
        "ccm y_to_x final rho",
        ccm_cpu.value.y_manifold_to_x.final_rho,
        ccm_gpu.value.y_manifold_to_x.final_rho,
        1e-5,
    );

    let te_cpu = timed(|| {
        transfer_entropy_with_config(&stream_a, &stream_b, 2, &clock, &te_config).unwrap()
    });
    let te_gpu = timed(|| {
        transfer_entropy_with_config_cuda_strict(&stream_a, &stream_b, 2, &clock, &te_config)
            .unwrap()
    });
    assert_close(
        "TE a_to_b",
        te_cpu.value.t_a_to_b,
        te_gpu.value.t_a_to_b,
        1e-5,
    );
    assert_close(
        "TE b_to_a",
        te_cpu.value.t_b_to_a,
        te_gpu.value.t_b_to_a,
        1e-5,
    );

    let tc_cpu = timed(|| total_correlation_with_config(&slots, &clock, &tc_config).unwrap());
    let tc_gpu =
        timed(|| total_correlation_with_config_cuda_strict(&slots, &clock, &tc_config).unwrap());
    assert_close("TC", tc_cpu.value.tc, tc_gpu.value.tc, 1e-5);
    assert_close(
        "TC joint entropy",
        tc_cpu.value.joint_entropy,
        tc_gpu.value.joint_entropy,
        1e-5,
    );

    let ii_cpu = timed(|| {
        interaction_information_with_config(&slots[0], &slots[1], &slots[2], &clock, &tc_config)
            .unwrap()
    });
    let ii_gpu = timed(|| {
        interaction_information_with_config_cuda_strict(
            &slots[0], &slots[1], &slots[2], &clock, &tc_config,
        )
        .unwrap()
    });
    assert_close("II", ii_cpu.value.ii, ii_gpu.value.ii, 1e-5);

    let edges = edge_case_readbacks(&x, &y, &slots, &clock, &tc_config);
    let artifact = json!({
        "artifact_kind": "issue1504.assay-neighbor-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1504_FSV_DIR/issue1504-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_suite_issue1502_1504_cuda issue1504_neighbor_cuda -- --nocapture",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "happy_path": {
            "ksg": estimator_readback(ksg_cpu, ksg_gpu),
            "mixed_ksg": estimator_readback(mixed_cpu, mixed_gpu),
            "ccm": {
                "cpu_ms": ccm_cpu.elapsed_ms,
                "gpu_ms": ccm_gpu.elapsed_ms,
                "speedup": speedup(ccm_cpu.elapsed_ms, ccm_gpu.elapsed_ms),
                "cpu_x_to_y_final_rho": ccm_cpu.value.x_manifold_to_y.final_rho,
                "gpu_x_to_y_final_rho": ccm_gpu.value.x_manifold_to_y.final_rho,
                "cpu_y_to_x_final_rho": ccm_cpu.value.y_manifold_to_x.final_rho,
                "gpu_y_to_x_final_rho": ccm_gpu.value.y_manifold_to_x.final_rho,
                "library_sizes": ccm_gpu.value.x_manifold_to_y.library_skills.iter().map(|skill| skill.library_size).collect::<Vec<_>>(),
            },
            "transfer_entropy": {
                "cpu_ms": te_cpu.elapsed_ms,
                "gpu_ms": te_gpu.elapsed_ms,
                "speedup": speedup(te_cpu.elapsed_ms, te_gpu.elapsed_ms),
                "cpu_a_to_b": te_cpu.value.t_a_to_b,
                "gpu_a_to_b": te_gpu.value.t_a_to_b,
                "cpu_b_to_a": te_cpu.value.t_b_to_a,
                "gpu_b_to_a": te_gpu.value.t_b_to_a,
                "n_samples": te_gpu.value.n_samples,
            },
            "total_correlation": {
                "cpu_ms": tc_cpu.elapsed_ms,
                "gpu_ms": tc_gpu.elapsed_ms,
                "speedup": speedup(tc_cpu.elapsed_ms, tc_gpu.elapsed_ms),
                "cpu_tc": tc_cpu.value.tc,
                "gpu_tc": tc_gpu.value.tc,
                "cpu_joint_entropy": tc_cpu.value.joint_entropy,
                "gpu_joint_entropy": tc_gpu.value.joint_entropy,
                "n_samples": tc_gpu.value.n_samples,
                "slot_count": tc_gpu.value.slot_count,
            },
            "interaction_information": {
                "cpu_ms": ii_cpu.elapsed_ms,
                "gpu_ms": ii_gpu.elapsed_ms,
                "speedup": speedup(ii_cpu.elapsed_ms, ii_gpu.elapsed_ms),
                "cpu_ii": ii_cpu.value.ii,
                "gpu_ii": ii_gpu.value.ii,
                "n_samples": ii_gpu.value.n_samples,
            }
        },
        "edge_cases": edges,
    });
    let restored = write_fsv_artifact(artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1504.assay-neighbor-cuda-fsv.v1"
    );
    assert!(restored["happy_path"]["ksg"]["gpu_bits"].as_f64().unwrap() >= 0.0);
    assert_eq!(
        restored["edge_cases"].as_array().unwrap().len(),
        3,
        "three edge cases are required for issue1504 FSV"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1504_strict_cuda_errors_without_cuda_feature() {
    let (x, y) = paired_vectors(72);
    let err = ksg_mi_continuous_cuda_strict(&x, &y, 3).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}

fn paired_vectors(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.071 + 0.13;
        x.push(vec![t.sin() + 0.11 * (3.0 * t).cos(), (1.7 * t).cos()]);
        y.push(vec![(t + 0.35).sin() + 0.05 * (5.0 * t).cos()]);
    }
    (x, y)
}

#[cfg(feature = "cuda")]
fn mixed_vectors(n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut x = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let class_width = n / 3;
    for i in 0..n {
        let label = (i / class_width).min(2);
        let local = (i % class_width) as f32;
        let base = label as f32 * 4.0;
        x.push(vec![
            base + local * 0.037 + (local * 0.11).sin() * 0.01,
            base * 0.5 + local * 0.019 + (local * 0.07).cos() * 0.01,
        ]);
        labels.push(label);
    }
    (x, labels)
}

#[cfg(feature = "cuda")]
fn panel_slots(n: usize) -> Vec<Vec<f32>> {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.05;
        a.push(t.sin() + 0.02 * (11.0 * t).cos());
        b.push((t + 0.4).sin() * 0.7 + 0.15 * (2.3 * t).cos());
        c.push((1.4 * t).cos() * 0.4 + 0.12 * t.sin());
    }
    vec![a, b, c]
}

#[cfg(feature = "cuda")]
type RecurrenceFixture = Vec<(u64, f32)>;

#[cfg(feature = "cuda")]
fn recurrence_streams(n: usize) -> (RecurrenceFixture, RecurrenceFixture) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.06;
        let av = t.sin() + 0.1 * (3.1 * t).cos();
        let bv = (t - 0.12).sin() * 0.8 + 0.2 * av + 0.03 * (7.0 * t).cos();
        a.push((i as u64, av));
        b.push((i as u64, bv));
    }
    (a, b)
}

#[cfg(feature = "cuda")]
fn edge_case_readbacks(
    x: &[Vec<f32>],
    y: &[Vec<f32>],
    slots: &[Vec<f32>],
    clock: &FixedClock,
    tc_config: &TotalCorrelationConfig,
) -> Vec<serde_json::Value> {
    let empty = ksg_mi_continuous_cuda_strict(&[], &[], 3).unwrap_err();
    let k_too_large = ksg_mi_continuous_cuda_strict(x, y, 33).unwrap_err();
    let mut nonfinite_slots = slots.to_vec();
    nonfinite_slots[1][7] = f32::NAN;
    let nonfinite =
        total_correlation_with_config_cuda_strict(&nonfinite_slots, clock, tc_config).unwrap_err();
    vec![
        edge(
            "empty_ksg",
            json!({"x_rows": 0, "y_rows": 0, "k": 3}),
            empty,
        ),
        edge(
            "k_over_cuda_limit",
            json!({"x_rows": x.len(), "y_rows": y.len(), "k": 33}),
            k_too_large,
        ),
        edge(
            "nonfinite_total_correlation",
            json!({"slot_count": nonfinite_slots.len(), "nan_slot": 1, "nan_index": 7}),
            nonfinite,
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
fn estimator_readback(
    cpu: Timed<calyx_assay::MiEstimate>,
    gpu: Timed<calyx_assay::MiEstimate>,
) -> serde_json::Value {
    json!({
        "cpu_ms": cpu.elapsed_ms,
        "gpu_ms": gpu.elapsed_ms,
        "speedup": speedup(cpu.elapsed_ms, gpu.elapsed_ms),
        "cpu_bits": cpu.value.bits,
        "gpu_bits": gpu.value.bits,
        "cpu_ci": [cpu.value.ci_low, cpu.value.ci_high],
        "gpu_ci": [gpu.value.ci_low, gpu.value.ci_high],
        "n_samples": gpu.value.n_samples,
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
fn assert_close(name: &str, left: f32, right: f32, tolerance: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch: left={left} right={right} diff={diff} tolerance={tolerance}"
    );
}

#[cfg(feature = "cuda")]
fn write_fsv_artifact(value: serde_json::Value) -> serde_json::Value {
    let root = std::env::var_os("CALYX_ASSAY_ISSUE1504_FSV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/issue1504-fsv"));
    std::fs::create_dir_all(&root).expect("create issue1504 FSV dir");
    let path = root.join("issue1504-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1504 FSV artifact");
    std::fs::write(&path, bytes).expect("write issue1504 FSV artifact");
    let readback = std::fs::read(&path).expect("read issue1504 FSV artifact");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1504 FSV artifact");
    println!(
        "ISSUE1504_FSV_READBACK path={} bytes={} blake3={}",
        path.display(),
        readback.len(),
        blake3::hash(&readback).to_hex()
    );
    restored
}
