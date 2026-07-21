#[cfg(feature = "cuda")]
use super::super::*;

#[cfg(feature = "cuda")]
pub(in super::super) fn benchmark_readback() -> serde_json::Value {
    let before = nvidia_smi_snapshot("before");
    let (period_times, period_values) = periodic_fixture(160);
    let period_config = PeriodogramConfig::default();
    let started = std::time::Instant::now();
    let period_cpu = lomb_scargle_with_config(&period_times, &period_values, &period_config)
        .expect("CPU default periodogram benchmark");
    let period_cpu_us = started.elapsed().as_micros();
    let started = std::time::Instant::now();
    let period_gpu =
        lomb_scargle_with_config_cuda_strict(&period_times, &period_values, &period_config)
            .expect("CUDA default periodogram benchmark");
    let period_gpu_us = started.elapsed().as_micros();

    let (ccf_x, ccf_y) = ccf_fixture(256, 5);
    let started = std::time::Instant::now();
    let ccf_cpu = cross_correlation_profile(&ccf_x, &ccf_y, 32).expect("CPU CCF benchmark");
    let ccf_cpu_us = started.elapsed().as_micros();
    let started = std::time::Instant::now();
    let ccf_gpu =
        cross_correlation_profile_cuda_strict(&ccf_x, &ccf_y, 32).expect("CUDA CCF benchmark");
    let ccf_gpu_us = started.elapsed().as_micros();

    let (alpha, beta) = hawkes_benchmark_fixture(60);
    let hawkes_processes = [
        HawkesEventSeries {
            name: "alpha",
            event_times: &alpha,
        },
        HawkesEventSeries {
            name: "beta",
            event_times: &beta,
        },
    ];
    let hawkes_config = HawkesConfig::new(610.0, 2.0, 40, 0.05);
    let started = std::time::Instant::now();
    let hawkes_cpu =
        exponential_hawkes_em(&hawkes_processes, &hawkes_config).expect("CPU Hawkes benchmark");
    let hawkes_cpu_us = started.elapsed().as_micros();
    let started = std::time::Instant::now();
    let hawkes_gpu = exponential_hawkes_em_cuda_strict(&hawkes_processes, &hawkes_config)
        .expect("CUDA Hawkes benchmark");
    let hawkes_gpu_us = started.elapsed().as_micros();
    let after = nvidia_smi_snapshot("after");

    json!({
        "source_of_truth": "same JSON readback; benchmark rows are persisted and re-read with the FSV artifact",
        "gpu_probe_before": before,
        "gpu_probe_after": after,
        "periodogram_default_grid": {
            "samples": period_times.len(),
            "frequencies": period_gpu.frequencies.len(),
            "fap_permutations": period_config.fap_permutations,
            "cpu_us": period_cpu_us,
            "gpu_us": period_gpu_us,
            "speedup_cpu_over_gpu": speedup(period_cpu_us, period_gpu_us),
            "estimated_transfer_bytes": period_transfer_bytes(&period_gpu, period_config.fap_permutations),
            "cpu_dominant_period": period_cpu.dominant().map(|peak| peak.period),
            "gpu_dominant_period": period_gpu.dominant().map(|peak| peak.period)
        },
        "cross_correlation_default_lag_grid": {
            "samples": ccf_x.len(),
            "max_lag": 32,
            "points": ccf_gpu.points.len(),
            "cpu_us": ccf_cpu_us,
            "gpu_us": ccf_gpu_us,
            "speedup_cpu_over_gpu": speedup(ccf_cpu_us, ccf_gpu_us),
            "estimated_transfer_bytes": ccf_transfer_bytes(ccf_x.len(), ccf_gpu.points.len()),
            "cpu_peak_lag": ccf_cpu.peak_lag,
            "gpu_peak_lag": ccf_gpu.peak_lag
        },
        "hawkes_realistic_event_count": {
            "processes": 2,
            "alpha_events": alpha.len(),
            "beta_events": beta.len(),
            "total_events": alpha.len() + beta.len(),
            "iterations": hawkes_config.iterations,
            "cpu_us": hawkes_cpu_us,
            "gpu_us": hawkes_gpu_us,
            "speedup_cpu_over_gpu": speedup(hawkes_cpu_us, hawkes_gpu_us),
            "estimated_transfer_bytes": hawkes_transfer_bytes(alpha.len() + beta.len(), 2),
            "cpu_spectral_radius": hawkes_cpu.spectral_radius,
            "gpu_spectral_radius": hawkes_gpu.spectral_radius
        }
    })
}

#[cfg(feature = "cuda")]
fn nvidia_smi_snapshot(label: &str) -> serde_json::Value {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .expect("run nvidia-smi for issue1507 benchmark readback");
    assert!(
        output.status.success(),
        "nvidia-smi failed for issue1507 benchmark readback: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("nvidia-smi emitted UTF-8");
    let rows: Vec<&str> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "nvidia-smi query must return exactly one GPU row: {stdout:?}"
    );
    let fields: Vec<&str> = rows[0].split(',').map(str::trim).collect();
    assert_eq!(
        fields.len(),
        5,
        "nvidia-smi GPU row must have timestamp,name,utilization,memory,power: {:?}",
        rows[0]
    );
    let utilization_gpu_percent = fields[2]
        .parse::<u32>()
        .expect("parse nvidia-smi utilization.gpu");
    let memory_used_mib = fields[3]
        .parse::<u32>()
        .expect("parse nvidia-smi memory.used");
    let power_draw_watts = fields[4]
        .parse::<f64>()
        .expect("parse nvidia-smi power.draw");
    json!({
        "label": label,
        "timestamp": fields[0],
        "name": fields[1],
        "utilization_gpu_percent": utilization_gpu_percent,
        "memory_used_mib": memory_used_mib,
        "power_draw_watts": power_draw_watts,
        "raw": rows[0]
    })
}

#[cfg(feature = "cuda")]
fn speedup(cpu_us: u128, gpu_us: u128) -> f64 {
    assert!(cpu_us > 0, "CPU benchmark duration must be non-zero");
    assert!(gpu_us > 0, "GPU benchmark duration must be non-zero");
    cpu_us as f64 / gpu_us as f64
}

#[cfg(feature = "cuda")]
fn period_transfer_bytes(report: &PeriodicityReport, fap_permutations: usize) -> usize {
    let n = report.n_samples;
    let frequencies = report.frequencies.len();
    let host_to_device =
        (n * std::mem::size_of::<f64>() * 2) + (frequencies * std::mem::size_of::<f64>());
    let permutation_indices = n * fap_permutations * std::mem::size_of::<i32>();
    let device_to_host = (frequencies * std::mem::size_of::<f64>())
        + (fap_permutations * std::mem::size_of::<f64>());
    host_to_device + permutation_indices + device_to_host
}

#[cfg(feature = "cuda")]
fn ccf_transfer_bytes(samples: usize, points: usize) -> usize {
    let host_to_device = samples * std::mem::size_of::<f32>() * 2;
    let device_to_host = points * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());
    host_to_device + device_to_host
}

#[cfg(feature = "cuda")]
fn hawkes_transfer_bytes(total_events: usize, processes: usize) -> usize {
    let host_to_device = (total_events * std::mem::size_of::<f64>())
        + ((processes + 1) * std::mem::size_of::<i32>())
        + (total_events * std::mem::size_of::<i32>());
    let resident_device_state = (processes * std::mem::size_of::<f64>())
        + (total_events * processes * std::mem::size_of::<f64>())
        + (processes * std::mem::size_of::<f64>())
        + (processes * processes * std::mem::size_of::<f64>())
        + (processes * std::mem::size_of::<f64>())
        + (processes * processes * std::mem::size_of::<f64>())
        + std::mem::size_of::<f64>()
        + std::mem::size_of::<i32>();
    let device_to_host = (processes * std::mem::size_of::<f64>())
        + (processes * processes * std::mem::size_of::<f64>())
        + std::mem::size_of::<f64>();
    host_to_device + resident_device_state + device_to_host
}

#[cfg(feature = "cuda")]
fn hawkes_benchmark_fixture(clusters: usize) -> (Vec<f32>, Vec<f32>) {
    let mut alpha = Vec::with_capacity(clusters * 3);
    let mut beta = Vec::with_capacity(clusters * 2);
    for cluster in 0..clusters {
        let base = 5.0 + cluster as f32 * 10.0;
        alpha.push(base);
        alpha.push(base + 0.3);
        alpha.push(base + 1.4);
        beta.push(base + 0.9);
        beta.push(base + 1.2);
    }
    (alpha, beta)
}
