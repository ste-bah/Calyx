#[cfg(feature = "cuda")]
use std::path::PathBuf;

#[cfg(not(feature = "cuda"))]
use calyx_assay::granger_causality_cuda_strict;
use calyx_assay::partial_correlation_controlling_cuda_strict;
#[cfg(feature = "cuda")]
use calyx_assay::{
    PartialNetworkReport, PartialNetworkSeries, PartialReport, PcSeries, PcStableReport,
    conditional_mutual_information_gaussian_with_alpha_cuda_strict, granger_causality_lags,
    granger_causality_lags_cuda_strict, granger_causality_sweep_lags,
    granger_causality_sweep_lags_cuda_strict, partial_correlation_controlling,
    partial_correlation_network, partial_correlation_network_cuda_strict, pc_stable_gaussian,
    pc_stable_gaussian_cuda_strict,
};
#[cfg(feature = "cuda")]
use calyx_core::CalyxError;
#[cfg(feature = "cuda")]
use serde_json::json;

#[cfg(feature = "cuda")]
#[test]
fn issue1506_linalg_cuda_matches_cpu_and_writes_fsv() {
    let (x, y, z1, z2) = partial_fixture(128);
    let controls = [&z1[..], &z2[..]];
    let partial_cpu = partial_correlation_controlling(&x, &y, &controls).unwrap();
    let partial_gpu = partial_correlation_controlling_cuda_strict(&x, &y, &controls).unwrap();
    assert_partial_close("partial", partial_cpu, partial_gpu, 2e-4);

    let cmi_gpu =
        conditional_mutual_information_gaussian_with_alpha_cuda_strict(&x, &y, &controls, 0.05)
            .unwrap();
    assert!(cmi_gpu.cmi_bits.is_finite());
    assert_eq!(cmi_gpu.n_controls, 2);

    let network_series = network_fixture(120);
    let network_cpu = partial_correlation_network(&network_series, 0.01, 0.12).unwrap();
    let network_gpu = partial_correlation_network_cuda_strict(&network_series, 0.01, 0.12).unwrap();
    assert_network_same_edges(&network_cpu, &network_gpu);

    let pc_series = pc_fixture(140);
    let pc_cpu = pc_stable_gaussian(&pc_series, 0.01, 1).unwrap();
    let pc_gpu = pc_stable_gaussian_cuda_strict(&pc_series, 0.01, 1).unwrap();
    assert_pc_same_edges(&pc_cpu, &pc_gpu);

    let (gx, gy) = granger_fixture(128);
    let granger_cpu = granger_causality_lags(&gx, &gy, 1).unwrap();
    let granger_gpu = granger_causality_lags_cuda_strict(&gx, &gy, 1).unwrap();
    assert_close(
        "granger rss_r",
        granger_cpu.rss_restricted,
        granger_gpu.rss_restricted,
        2e-4,
    );
    assert_close(
        "granger rss_u",
        granger_cpu.rss_unrestricted,
        granger_gpu.rss_unrestricted,
        2e-4,
    );
    assert_close(
        "granger f",
        granger_cpu.f_statistic,
        granger_gpu.f_statistic,
        5e-3,
    );

    let sweep_cpu = granger_causality_sweep_lags(&gx, &gy, &[1, 2, 4]).unwrap();
    let sweep_gpu = granger_causality_sweep_lags_cuda_strict(&gx, &gy, &[1, 2, 4]).unwrap();
    assert_eq!(sweep_cpu.lags, sweep_gpu.lags);
    assert_close(
        "sweep f",
        sweep_cpu.f_statistic,
        sweep_gpu.f_statistic,
        5e-3,
    );

    let previous_strict = std::env::var_os("CALYX_ASSAY_CUDA_STRICT");
    unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", "1") };
    let routed_partial = partial_correlation_controlling(&x, &y, &controls).unwrap();
    let routed_granger = granger_causality_lags(&gx, &gy, 1).unwrap();
    restore_strict_env(previous_strict);
    assert_partial_close("strict env partial route", partial_gpu, routed_partial, 0.0);
    assert_close(
        "strict env granger route",
        granger_gpu.rss_unrestricted,
        routed_granger.rss_unrestricted,
        0.0,
    );

    let edges = edge_case_readbacks(&x, &y, &z1, &z2, &network_series, &pc_series, &gx);
    let artifact = json!({
        "artifact_kind": "issue1506.assay-linear-algebra-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1506_FSV_DIR/issue1506-linalg-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_isolated_issue1506_linalg_cuda issue1506_linalg_cuda -- --nocapture",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "happy_path": {
            "partial": {"cpu": partial_cpu, "gpu": partial_gpu, "cmi_gpu": cmi_gpu},
            "partial_network": {"cpu": network_summary(&network_cpu), "gpu": network_summary(&network_gpu)},
            "pc_stable": {"cpu": pc_summary(&pc_cpu), "gpu": pc_summary(&pc_gpu)},
            "granger_lag1": {"cpu": granger_cpu, "gpu": granger_gpu},
            "granger_sweep": {"cpu": sweep_cpu, "gpu": sweep_gpu},
            "strict_env": {"partial": routed_partial, "granger": routed_granger},
        },
        "edge_cases": edges,
    });
    let restored = write_fsv_artifact(artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1506.assay-linear-algebra-cuda-fsv.v1"
    );
    assert_eq!(restored["happy_path"]["partial"]["gpu"]["n_controls"], 2);
    assert_eq!(
        restored["happy_path"]["granger_lag1"]["gpu"]["lags"],
        json!(1)
    );
    assert!(
        restored["edge_cases"].as_array().unwrap().len() >= 5,
        "issue1506 FSV must persist at least five edge readbacks"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1506_linalg_cuda_strict_errors_without_cuda_feature() {
    let x = [1.0_f32, 2.0, 3.0, 4.0];
    let y = [1.0_f32, 1.5, 2.5, 4.0];
    let z = [0.0_f32, 1.0, 0.0, 1.0];
    let err = partial_correlation_controlling_cuda_strict(&x, &y, &[&z]).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    let err = granger_causality_cuda_strict(&[1.0; 8], &[2.0; 8]).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}

#[cfg(feature = "cuda")]
fn partial_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z1 = Vec::with_capacity(n);
    let mut z2 = Vec::with_capacity(n);
    for row in 0..n {
        let t = row as f32 * 0.041;
        let a = t.sin() + 0.2 * (t * 1.7).cos();
        let b = (t * 0.63 + 0.4).cos();
        let driver = (t * 2.1 + 0.1).sin();
        z1.push(a);
        z2.push(b);
        x.push(0.7 * a + 0.25 * b + 0.35 * driver);
        y.push(0.65 * driver + 0.8 * a - 0.2 * b + 0.06 * (t * 3.7).cos());
    }
    (x, y, z1, z2)
}

#[cfg(feature = "cuda")]
fn network_fixture(n: usize) -> Vec<PartialNetworkSeries<'static>> {
    let (x, y, z1, z2) = partial_fixture(n);
    let extra = (0..n)
        .map(|row| {
            let t = row as f32 * 0.052;
            (t + 0.9).cos() * 0.4 + (row % 7) as f32 * 0.01
        })
        .collect::<Vec<_>>();
    vec![
        owned_series("x", x),
        owned_series("y", y),
        owned_series("z1", z1),
        owned_series("z2", z2),
        owned_series("extra", extra),
    ]
}

#[cfg(feature = "cuda")]
fn pc_fixture(n: usize) -> Vec<PcSeries<'static>> {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for row in 0..n {
        let t = row as f32 * 0.037;
        let av = t.sin() + 0.1 * (t * 2.3).cos();
        let bv = 0.8 * av + 0.15 * (t * 1.1 + 0.4).cos();
        let cv = 0.7 * bv + 0.12 * (t * 1.7).sin();
        let dv = (splitmix(30_000 + row as u64) - 0.5) as f32;
        a.push(av);
        b.push(bv);
        c.push(cv);
        d.push(dv);
    }
    vec![
        owned_pc_series("a", a),
        owned_pc_series("b", b),
        owned_pc_series("c", c),
        owned_pc_series("d", d),
    ]
}

#[cfg(feature = "cuda")]
fn granger_fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (idx, value) in x.iter_mut().enumerate() {
        *value = (splitmix(idx as u64) - 0.5) as f32;
    }
    for idx in 2..n {
        let noise = (splitmix(4000 + idx as u64) - 0.5) * 0.08;
        y[idx] = 0.5 * y[idx - 1] + 1.1 * x[idx - 1] + 0.35 * x[idx - 2] + noise as f32;
    }
    (x, y)
}

#[cfg(feature = "cuda")]
fn edge_case_readbacks(
    x: &[f32],
    y: &[f32],
    z1: &[f32],
    z2: &[f32],
    network_series: &[PartialNetworkSeries<'_>],
    pc_series: &[PcSeries<'_>],
    gx: &[f32],
) -> Vec<serde_json::Value> {
    let constant = vec![1.0_f32; x.len()];
    let partial_constant =
        partial_correlation_controlling_cuda_strict(x, y, &[z1, &constant]).unwrap_err();

    let mut nonfinite = x.to_vec();
    nonfinite[9] = f32::NAN;
    let partial_nonfinite =
        partial_correlation_controlling_cuda_strict(&nonfinite, y, &[z1, z2]).unwrap_err();

    let mut duplicate_network = network_series.to_vec();
    duplicate_network[1].name = duplicate_network[0].name;
    let network_duplicate =
        partial_correlation_network_cuda_strict(&duplicate_network, 0.01, 0.12).unwrap_err();

    let mut duplicate_pc = pc_series.to_vec();
    duplicate_pc[2].name = duplicate_pc[1].name;
    let pc_duplicate = pc_stable_gaussian_cuda_strict(&duplicate_pc, 0.01, 1).unwrap_err();

    let constant_y = vec![4.0_f32; gx.len()];
    let granger_rank = granger_causality_lags_cuda_strict(gx, &constant_y, 1).unwrap_err();

    let invalid_sweep =
        granger_causality_sweep_lags_cuda_strict(gx, &constant_y, &[0, 33]).unwrap_err();

    vec![
        edge(
            "partial_constant_control",
            json!({"n": x.len(), "constant_control": true}),
            partial_constant,
        ),
        edge(
            "partial_nonfinite_input",
            json!({"n": x.len(), "nan_row": 9}),
            partial_nonfinite,
        ),
        edge(
            "network_duplicate_name",
            json!({"variables": duplicate_network.len(), "duplicate": duplicate_network[0].name}),
            network_duplicate,
        ),
        edge(
            "pc_duplicate_name",
            json!({"variables": duplicate_pc.len(), "duplicate": duplicate_pc[1].name}),
            pc_duplicate,
        ),
        edge(
            "granger_rank_deficient",
            json!({"n": gx.len(), "target": "constant_y"}),
            granger_rank,
        ),
        edge(
            "granger_invalid_sweep",
            json!({"lags": [0, 33], "target": "constant_y"}),
            invalid_sweep,
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
            "remediation": err.remediation,
        }
    })
}

#[cfg(feature = "cuda")]
fn network_summary(report: &PartialNetworkReport) -> serde_json::Value {
    json!({
        "retained": report.retained_edges,
        "pruned": report.pruned_edges,
        "variables": report.variables,
    })
}

#[cfg(feature = "cuda")]
fn pc_summary(report: &PcStableReport) -> serde_json::Value {
    json!({
        "retained": report.retained_edges,
        "removed": report.removed_edges,
        "variables": report.variables,
    })
}

#[cfg(feature = "cuda")]
fn assert_partial_close(name: &str, left: PartialReport, right: PartialReport, tolerance: f32) {
    assert_close(
        &format!("{name} partial_r"),
        left.partial_r,
        right.partial_r,
        tolerance,
    );
    assert_close(
        &format!("{name} zero_order_r"),
        left.zero_order_r,
        right.zero_order_r,
        tolerance,
    );
    assert_close(&format!("{name} p"), left.p_value, right.p_value, 2e-3);
    assert_eq!(left.n_controls, right.n_controls);
    assert_eq!(left.n_samples, right.n_samples);
}

#[cfg(feature = "cuda")]
fn assert_network_same_edges(left: &PartialNetworkReport, right: &PartialNetworkReport) {
    assert_eq!(left.variables, right.variables);
    assert_eq!(left.retained_edges.len(), right.retained_edges.len());
    assert_eq!(left.pruned_edges.len(), right.pruned_edges.len());
    for (left, right) in left.retained_edges.iter().zip(&right.retained_edges) {
        assert_eq!(left.left, right.left);
        assert_eq!(left.right, right.right);
        assert_close("network partial", left.partial_r, right.partial_r, 2e-4);
    }
}

#[cfg(feature = "cuda")]
fn assert_pc_same_edges(left: &PcStableReport, right: &PcStableReport) {
    assert_eq!(left.variables, right.variables);
    assert_eq!(left.retained_edges, right.retained_edges);
    assert_eq!(left.removed_edges.len(), right.removed_edges.len());
    for (left, right) in left.removed_edges.iter().zip(&right.removed_edges) {
        assert_eq!(left.left, right.left);
        assert_eq!(left.right, right.right);
        assert_eq!(left.conditioning_set, right.conditioning_set);
        assert_eq!(left.depth, right.depth);
    }
}

#[cfg(feature = "cuda")]
fn assert_close(name: &str, left: f32, right: f32, tolerance: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch left={left} right={right} diff={diff} tolerance={tolerance}"
    );
}

#[cfg(feature = "cuda")]
fn owned_series(name: &'static str, values: Vec<f32>) -> PartialNetworkSeries<'static> {
    PartialNetworkSeries {
        name,
        values: Box::leak(values.into_boxed_slice()),
    }
}

#[cfg(feature = "cuda")]
fn owned_pc_series(name: &'static str, values: Vec<f32>) -> PcSeries<'static> {
    PcSeries {
        name,
        values: Box::leak(values.into_boxed_slice()),
    }
}

#[cfg(feature = "cuda")]
fn splitmix(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1_u64 << 53) as f64)
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
    let root = std::env::var_os("CALYX_ASSAY_ISSUE1506_FSV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/issue1506-fsv"));
    std::fs::create_dir_all(&root).expect("create issue1506 FSV dir");
    let path = root.join("issue1506-linalg-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1506 FSV");
    std::fs::write(&path, bytes).expect("write issue1506 FSV");
    let readback = std::fs::read(&path).expect("read issue1506 FSV");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1506 FSV");
    println!(
        "ISSUE1506_LINALG_FSV_READBACK path={} bytes={} blake3={}",
        path.display(),
        readback.len(),
        blake3::hash(&readback).to_hex()
    );
    println!(
        "ISSUE1506_LINALG_FSV_DATA {}",
        String::from_utf8_lossy(&readback)
    );
    restored
}
