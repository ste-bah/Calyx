#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use calyx_assay::{
    CATEGORICAL_CUDA_MIN_SAMPLES, COPULA_CUDA_MIN_SAMPLES, DEPENDENCE_CUDA_AUTO_ENV,
    MIC_CUDA_MIN_SAMPLES, NMI_CUDA_MIN_SAMPLES, RANK_CUDA_MIN_SAMPLES, categorical_association,
    categorical_association_cuda_strict_with_stats, empirical_copula_tail_dependence_with_q,
    empirical_copula_tail_dependence_with_q_cuda_strict,
    empirical_copula_tail_dependence_with_q_cuda_strict_with_stats, kendall_tau_b, mic,
    mic_with_alpha_cuda_strict_with_stats, partitioned_histogram_nmi,
    partitioned_histogram_nmi_cuda_strict_with_stats, rank_correlations_cuda_strict, spearman_rho,
};
#[cfg(not(feature = "cuda"))]
use calyx_assay::{
    categorical_association_cuda_strict, empirical_copula_tail_dependence_cuda_strict,
    mic_cuda_strict, partitioned_histogram_nmi_cuda_strict, rank_correlations_cuda_strict,
};
#[cfg(feature = "cuda")]
use serde_json::{Value, json};

#[cfg(feature = "cuda")]
#[test]
fn issue1508_dependence_cuda_matches_cpu_and_writes_fsv() {
    let (nmi_x, nmi_y) = nmi_fixture(128);
    let nmi_cpu = partitioned_histogram_nmi(&nmi_x, &nmi_y, 16).unwrap();
    let (nmi_gpu, nmi_stats) =
        partitioned_histogram_nmi_cuda_strict_with_stats(&nmi_x, &nmi_y, 16).unwrap();
    assert_eq!(nmi_cpu, nmi_gpu);
    assert_stats("histogram", &nmi_stats, 1);

    let (category_x, category_y) = categorical_fixture(192);
    let category_cpu = categorical_association(&category_x, &category_y).unwrap();
    let (category_gpu, category_stats) =
        categorical_association_cuda_strict_with_stats(&category_x, &category_y).unwrap();
    assert_eq!(category_cpu, category_gpu);
    assert_stats("categorical", &category_stats, 1);

    let (rank_x, rank_y) = rank_fixture(96);
    let spearman_cpu = spearman_rho(&rank_x, &rank_y).unwrap();
    let kendall_cpu = kendall_tau_b(&rank_x, &rank_y).unwrap();
    let rank_gpu = rank_correlations_cuda_strict(&rank_x, &rank_y).unwrap();
    assert_eq!(spearman_cpu, rank_gpu.spearman);
    assert_eq!(kendall_cpu, rank_gpu.kendall);
    assert_stats("rank", &rank_gpu.stats, 2);

    let (copula_x, copula_y) = copula_fixture(80);
    let copula_cpu = empirical_copula_tail_dependence_with_q(&copula_x, &copula_y, 0.1).unwrap();
    let (copula_gpu, copula_stats) =
        empirical_copula_tail_dependence_with_q_cuda_strict_with_stats(&copula_x, &copula_y, 0.1)
            .unwrap();
    assert_copula_close(&copula_cpu, &copula_gpu, 1e-12);
    assert_stats("copula", &copula_stats, 2);

    let (mic_x, mic_y) = mic_fixture(41);
    let mic_cpu = mic(&mic_x, &mic_y).unwrap();
    let (mic_gpu, mic_stats) = mic_with_alpha_cuda_strict_with_stats(&mic_x, &mic_y, 0.6).unwrap();
    assert_eq!(mic_cpu.best_nx, mic_gpu.best_nx);
    assert_eq!(mic_cpu.best_ny, mic_gpu.best_ny);
    assert_close(mic_cpu.mic as f64, mic_gpu.mic as f64, 1e-6);
    assert_stats("mic", &mic_stats, 1);

    let edge_cases = edge_case_readbacks();
    let artifact = json!({
        "artifact_kind": "issue1508.assay-dependence-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1508_FSV_DIR/issue1508-dependence-fsv-readback.json",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "minimum_sufficient_corpus": {
            "histogram_samples": nmi_x.len(),
            "categorical_samples": category_x.len(),
            "rank_samples": rank_x.len(),
            "copula_samples": copula_x.len(),
            "mic_samples": mic_x.len(),
            "why_smaller_insufficient": "The corpus needs multiple histogram bins, sparse category codes, rank ties, unique copula margins and both MIC orientations.",
            "why_larger_wasteful": "Larger inputs repeat the same kernel ABI; the ignored workload matrix owns break-even characterization."
        },
        "parity": {
            "histogram_nmi": {"cpu": nmi_cpu, "gpu": nmi_gpu, "stats": nmi_stats},
            "categorical": {"cpu": category_cpu, "gpu": category_gpu, "stats": category_stats},
            "rank": {"spearman_cpu": spearman_cpu, "kendall_cpu": kendall_cpu, "gpu": rank_gpu},
            "copula": {"cpu": copula_cpu, "gpu": copula_gpu, "stats": copula_stats},
            "mic": {"cpu": mic_cpu, "gpu": mic_gpu, "stats": mic_stats}
        },
        "edge_cases": edge_cases,
    });
    let restored = write_fsv_artifact("issue1508-dependence-fsv-readback.json", artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1508.assay-dependence-cuda-fsv.v1"
    );
    assert_eq!(
        restored["parity"]["rank"]["gpu"]["stats"]["kernel_launches"],
        2
    );
    println!(
        "ISSUE1508_DEPENDENCE_FSV histogram_launches={} categorical_launches={} rank_launches={} copula_launches={} mic_launches={}",
        nmi_stats.kernel_launches,
        category_stats.kernel_launches,
        rank_gpu.stats.kernel_launches,
        copula_stats.kernel_launches,
        mic_stats.kernel_launches
    );
}

#[cfg(feature = "cuda")]
#[test]
fn issue1508_dependence_boundaries_fail_closed() {
    let edges = edge_case_readbacks();
    assert!(edges.as_array().unwrap().len() >= 5);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU-host workload matrix; writes a benchmark FSV artifact"]
fn issue1508_dependence_cuda_workload_matrix() {
    let previous = std::env::var_os(DEPENDENCE_CUDA_AUTO_ENV);
    unsafe { std::env::set_var(DEPENDENCE_CUDA_AUTO_ENV, "0") };
    let (warm_x, warm_y) = nmi_fixture(64);
    partitioned_histogram_nmi_cuda_strict_with_stats(&warm_x, &warm_y, 8).unwrap();
    let mut rows = Vec::new();
    for &n in &[256usize, 1_024, 4_096, 16_384] {
        let (x, y) = nmi_fixture(n);
        rows.push(bench_row(
            "histogram_nmi",
            n,
            NMI_CUDA_MIN_SAMPLES,
            || partitioned_histogram_nmi(&x, &y, 32).unwrap(),
            || partitioned_histogram_nmi_cuda_strict_with_stats(&x, &y, 32).unwrap(),
        ));
    }
    for &n in &[256usize, 1_024, 4_096, 16_384, 65_536, 262_144] {
        let (x, y) = categorical_fixture(n);
        rows.push(bench_row(
            "categorical",
            n,
            CATEGORICAL_CUDA_MIN_SAMPLES,
            || categorical_association(&x, &y).unwrap(),
            || categorical_association_cuda_strict_with_stats(&x, &y).unwrap(),
        ));
    }
    for &n in &[128usize, 512, 1_024, 2_048, 4_096, 8_192] {
        let (x, y) = rank_fixture(n);
        rows.push(bench_row(
            "rank_pair",
            n,
            RANK_CUDA_MIN_SAMPLES,
            || {
                (
                    spearman_rho(&x, &y).unwrap(),
                    kendall_tau_b(&x, &y).unwrap(),
                )
            },
            || {
                let report = rank_correlations_cuda_strict(&x, &y).unwrap();
                let stats = report.stats.clone();
                (report, stats)
            },
        ));
        let (x, y) = copula_fixture(n);
        rows.push(bench_row(
            "copula",
            n,
            COPULA_CUDA_MIN_SAMPLES,
            || empirical_copula_tail_dependence_with_q(&x, &y, 0.1).unwrap(),
            || empirical_copula_tail_dependence_with_q_cuda_strict_with_stats(&x, &y, 0.1).unwrap(),
        ));
    }
    for &n in &[32usize, 64, 128, 256, 512] {
        let (x, y) = mic_fixture(n);
        rows.push(bench_row(
            "mic",
            n,
            MIC_CUDA_MIN_SAMPLES,
            || mic(&x, &y).unwrap(),
            || mic_with_alpha_cuda_strict_with_stats(&x, &y, 0.6).unwrap(),
        ));
    }
    restore_env(DEPENDENCE_CUDA_AUTO_ENV, previous);
    let artifact = json!({
        "artifact_kind": "issue1508.assay-dependence-cuda-benchmark.v1",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "rows": rows,
        "routing_rule": "CUDA only at or above a measured break-even; strict entry points bypass thresholds",
        "production_thresholds": {
            "histogram_nmi": NMI_CUDA_MIN_SAMPLES,
            "categorical": CATEGORICAL_CUDA_MIN_SAMPLES,
            "rank_pair": RANK_CUDA_MIN_SAMPLES,
            "copula": COPULA_CUDA_MIN_SAMPLES,
            "mic": "strict_only",
        },
    });
    let restored = write_fsv_artifact("issue1508-dependence-benchmark.json", artifact);
    assert!(restored["rows"].as_array().unwrap().len() >= 20);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1508_cuda_strict_errors_without_cuda_feature() {
    let x = (0..64).map(|index| index as f32).collect::<Vec<_>>();
    assert_eq!(
        partitioned_histogram_nmi_cuda_strict(&x, &x, 8)
            .unwrap_err()
            .code,
        "CALYX_FORGE_DEVICE_UNAVAILABLE"
    );
    let labels = (0..64).map(|index| (index % 2) as u32).collect::<Vec<_>>();
    assert_eq!(
        categorical_association_cuda_strict(&labels, &labels)
            .unwrap_err()
            .code,
        "CALYX_FORGE_DEVICE_UNAVAILABLE"
    );
    assert_eq!(
        rank_correlations_cuda_strict(&x, &x).unwrap_err().code,
        "CALYX_FORGE_DEVICE_UNAVAILABLE"
    );
    let xd = (0..64).map(|index| index as f64).collect::<Vec<_>>();
    assert_eq!(
        empirical_copula_tail_dependence_cuda_strict(&xd, &xd)
            .unwrap_err()
            .code,
        "CALYX_FORGE_DEVICE_UNAVAILABLE"
    );
    assert_eq!(
        mic_cuda_strict(&x, &x).unwrap_err().code,
        "CALYX_FORGE_DEVICE_UNAVAILABLE"
    );
}

#[cfg(feature = "cuda")]
fn edge_case_readbacks() -> Value {
    let y = (0..64).map(|index| index as f32).collect::<Vec<_>>();
    let constant = vec![1.0; 64];
    let nmi = partitioned_histogram_nmi_cuda_strict_with_stats(&constant, &y, 8)
        .unwrap_err()
        .code;
    let sparse_x = (0..64)
        .map(|index| if index % 3 == 0 { 7 } else { 90_001 })
        .collect::<Vec<_>>();
    let sparse_y = (0..64)
        .map(|index| if index % 2 == 0 { 42 } else { u32::MAX })
        .collect::<Vec<_>>();
    let sparse_cpu = categorical_association(&sparse_x, &sparse_y).unwrap();
    let sparse_gpu = categorical_association_cuda_strict_with_stats(&sparse_x, &sparse_y)
        .unwrap()
        .0;
    assert_eq!(sparse_cpu, sparse_gpu);
    let mut nonfinite = y.clone();
    nonfinite[9] = f32::NAN;
    let rank = rank_correlations_cuda_strict(&nonfinite, &y)
        .unwrap_err()
        .code;
    let mut duplicate = (0..32).map(|index| index as f64).collect::<Vec<_>>();
    duplicate[11] = duplicate[10];
    let copula = empirical_copula_tail_dependence_with_q_cuda_strict(&duplicate, &duplicate, 0.1)
        .unwrap_err()
        .code;
    let mic = mic_with_alpha_cuda_strict_with_stats(&constant, &y, 0.6)
        .unwrap_err()
        .code;
    json!([
        {"case": "constant_histogram_margin", "error_code": nmi},
        {"case": "sparse_category_codes", "n_rows": sparse_gpu.n_rows, "n_cols": sparse_gpu.n_cols},
        {"case": "nonfinite_rank", "error_code": rank},
        {"case": "duplicate_copula_margin", "error_code": copula},
        {"case": "constant_mic_margin", "error_code": mic}
    ])
}

#[cfg(feature = "cuda")]
fn assert_stats(operation: &str, stats: &calyx_assay::DependenceCudaStats, launches: usize) {
    assert_eq!(stats.operation, operation);
    assert_eq!(stats.kernel_launches, launches);
    assert!(stats.work_items > 0);
    assert!(stats.host_to_device_bytes > 0);
    assert!(stats.device_to_host_bytes > 0);
    assert!(stats.peak_device_bytes >= stats.host_to_device_bytes);
}

#[cfg(feature = "cuda")]
fn assert_copula_close(
    cpu: &calyx_assay::CopulaTailReport,
    gpu: &calyx_assay::CopulaTailReport,
    tolerance: f64,
) {
    assert_eq!(cpu.estimator, gpu.estimator);
    assert_eq!(cpu.n_samples, gpu.n_samples);
    assert_eq!(cpu.lower_tail_count, gpu.lower_tail_count);
    assert_eq!(cpu.upper_tail_count, gpu.upper_tail_count);
    for (left, right) in [
        (cpu.blomqvist_beta, gpu.blomqvist_beta),
        (cpu.hoeffding_d_cvm, gpu.hoeffding_d_cvm),
        (cpu.gini_gamma, gpu.gini_gamma),
        (cpu.lower_tail_lambda, gpu.lower_tail_lambda),
        (cpu.upper_tail_lambda, gpu.upper_tail_lambda),
    ] {
        assert_close(left, right, tolerance);
    }
}

#[cfg(feature = "cuda")]
fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "left={left} right={right} tolerance={tolerance}"
    );
}

#[cfg(feature = "cuda")]
fn bench_row<Cpu, Gpu, C, G>(
    operation: &str,
    n: usize,
    production_threshold: usize,
    cpu: Cpu,
    gpu: Gpu,
) -> Value
where
    Cpu: FnOnce() -> C,
    Gpu: FnOnce() -> (G, calyx_assay::DependenceCudaStats),
{
    let start = Instant::now();
    std::hint::black_box(cpu());
    let cpu_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    let (_, stats) = std::hint::black_box(gpu());
    let gpu_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let measured_winner = if gpu_ms < cpu_ms { "cuda" } else { "cpu" };
    let production_route = if n >= production_threshold {
        "cuda"
    } else {
        "cpu"
    };
    let row = json!({
        "operation": operation,
        "n_samples": n,
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "speedup": cpu_ms / gpu_ms,
        "measured_winner": measured_winner,
        "production_route": production_route,
        "production_threshold_samples": if production_threshold == usize::MAX {
            Value::String("strict_only".to_owned())
        } else {
            json!(production_threshold)
        },
        "stats": stats,
    });
    println!("ISSUE1508_BENCH {row}");
    row
}

#[cfg(feature = "cuda")]
fn nmi_fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..n).map(|index| (index % 37) as f32).collect();
    let y = (0..n)
        .map(|index| ((index * 11 + index / 7) % 53) as f32)
        .collect();
    (x, y)
}

#[cfg(feature = "cuda")]
fn categorical_fixture(n: usize) -> (Vec<u32>, Vec<u32>) {
    let x = (0..n)
        .map(|index| [7, 90_001, u32::MAX][index % 3])
        .collect();
    let y = (0..n)
        .map(|index| [42, 9_999][(index * 5 + index / 11) % 2])
        .collect();
    (x, y)
}

#[cfg(feature = "cuda")]
fn rank_fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x = (0..n).map(|index| (index / 3) as f32).collect();
    let y = (0..n)
        .map(|index| ((index * 17 + index / 5) % n.max(1)) as f32)
        .collect();
    (x, y)
}

#[cfg(feature = "cuda")]
fn copula_fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let multiplier = if n.is_multiple_of(2) { n - 1 } else { 2 };
    let x = (0..n).map(|index| index as f64).collect();
    let y = (0..n)
        .map(|index| ((index * multiplier + 7) % n) as f64)
        .collect();
    (x, y)
}

#[cfg(feature = "cuda")]
fn mic_fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let center = (n.saturating_sub(1) as f32) * 0.5;
    let x = (0..n)
        .map(|index| index as f32 - center)
        .collect::<Vec<_>>();
    let y = x.iter().map(|value| value * value).collect();
    (x, y)
}

#[cfg(feature = "cuda")]
fn write_fsv_artifact(name: &str, value: Value) -> Value {
    let root = std::env::var_os("CALYX_ASSAY_ISSUE1508_FSV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("calyx-issue1508-fsv"));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(name);
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[cfg(feature = "cuda")]
fn restore_env(name: &str, previous: Option<std::ffi::OsString>) {
    if let Some(value) = previous {
        unsafe { std::env::set_var(name, value) };
    } else {
        unsafe { std::env::remove_var(name) };
    }
}
