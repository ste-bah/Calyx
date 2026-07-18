#[cfg(feature = "cuda")]
use super::super::*;

#[cfg(feature = "cuda")]
pub(in super::super) fn edge_case_readbacks(
    period_times: &[f64],
    period_values: &[f64],
    ccf_x: &[f32],
    ccf_y: &[f32],
) -> Vec<serde_json::Value> {
    let bad_fap = lomb_scargle_with_config_cuda_strict(
        period_times,
        period_values,
        &PeriodogramConfig {
            fap_permutations: 0,
            ..PeriodogramConfig::default()
        },
    )
    .unwrap_err();
    assert_eq!(bad_fap.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");

    let mut unordered_times = period_times.to_vec();
    unordered_times[8] = unordered_times[7];
    let unordered_acf = autocorrelation_cuda_strict(&unordered_times, period_values).unwrap_err();
    assert_eq!(unordered_acf.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");

    let constant_ccf =
        cross_correlation_profile_cuda_strict(&vec![4.0; ccf_x.len()], ccf_y, 1).unwrap_err();
    assert_eq!(constant_ccf.code, "CALYX_ASSAY_DEGENERATE_INPUT");

    let beta = [1.0_f32, 2.0, 3.0];
    let duplicate = exponential_hawkes_em_cuda_strict(
        &[
            HawkesEventSeries {
                name: "alpha",
                event_times: &beta,
            },
            HawkesEventSeries {
                name: "alpha",
                event_times: &beta,
            },
        ],
        &HawkesConfig::new(10.0, 2.0, 5, 0.1),
    )
    .unwrap_err();
    assert_eq!(duplicate.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");

    let outside = [1.0_f32, 12.0];
    let outside_err = exponential_hawkes_em_cuda_strict(
        &[HawkesEventSeries {
            name: "alpha",
            event_times: &outside,
        }],
        &HawkesConfig::new(10.0, 2.0, 5, 0.1),
    )
    .unwrap_err();
    assert_eq!(outside_err.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");

    vec![
        json!({
            "case": "periodogram_zero_fap",
            "before": {"fap_permutations": 0, "n": period_times.len()},
            "after": {"code": bad_fap.code, "message": bad_fap.message}
        }),
        json!({
            "case": "acf_non_increasing_time",
            "before": {"times[8]": "set equal to times[7]"},
            "after": {"code": unordered_acf.code, "message": unordered_acf.message}
        }),
        json!({
            "case": "ccf_constant_series",
            "before": {"x": "all 4.0", "y_len": ccf_y.len()},
            "after": {"code": constant_ccf.code, "message": constant_ccf.message}
        }),
        json!({
            "case": "hawkes_duplicate_name",
            "before": {"names": ["alpha", "alpha"]},
            "after": {"code": duplicate.code, "message": duplicate.message}
        }),
        json!({
            "case": "hawkes_event_outside_observation",
            "before": {"event_times": outside, "observation_end": 10.0},
            "after": {"code": outside_err.code, "message": outside_err.message}
        }),
    ]
}

#[cfg(feature = "cuda")]
pub(in super::super) fn periodic_fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut t = 0.0;
    let mut times = Vec::with_capacity(n);
    for idx in 0..n {
        t += 0.7 + 0.6 * splitmix(idx as u64 + 17);
        times.push(t);
    }
    let values = times
        .iter()
        .map(|time| {
            (2.0 * std::f64::consts::PI * time / 7.0).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * time / 13.0).cos()
        })
        .collect();
    (times, values)
}

#[cfg(feature = "cuda")]
pub(in super::super) fn acf_fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
    let times: Vec<f64> = (0..n).map(|idx| idx as f64).collect();
    let values = times
        .iter()
        .map(|time| (2.0 * std::f64::consts::PI * time / 8.0).sin())
        .collect();
    (times, values)
}

#[cfg(feature = "cuda")]
pub(in super::super) fn ccf_fixture(n: usize, lag: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for t in 0..n {
        x.push((splitmix(t as u64 * 17 + 11) - 0.5) as f32);
    }
    for t in 0..n {
        let value = if t >= lag {
            x[t - lag] + 0.001 * (splitmix(10_000 + t as u64) - 0.5) as f32
        } else {
            (splitmix(20_000 + t as u64) - 0.5) as f32
        };
        y.push(value);
    }
    (x, y)
}

#[cfg(feature = "cuda")]
pub(in super::super) fn hawkes_fixture() -> (Vec<f32>, Vec<f32>) {
    let mut alpha = Vec::new();
    let mut beta = Vec::new();
    for base in (5..100).step_by(10) {
        let base = base as f32;
        alpha.push(base);
        alpha.push(base + 0.3);
        alpha.push(base + 1.4);
        beta.push(base + 0.9);
        beta.push(base + 1.2);
    }
    (alpha, beta)
}

#[cfg(feature = "cuda")]
pub(in super::super) fn assert_periodicity_close(
    left: &PeriodicityReport,
    right: &PeriodicityReport,
) {
    assert_eq!(left.frequencies.len(), right.frequencies.len());
    assert_eq!(left.powers.len(), right.powers.len());
    for (idx, (&l, &r)) in left.powers.iter().zip(&right.powers).enumerate() {
        assert_close_f64(&format!("power[{idx}]"), l, r, 2e-9);
    }
    assert_eq!(left.peaks.len(), right.peaks.len());
    for (idx, (l, r)) in left.peaks.iter().zip(&right.peaks).enumerate() {
        assert_close_f64(
            &format!("peak{idx}.frequency"),
            l.frequency,
            r.frequency,
            0.0,
        );
        assert_close_f64(&format!("peak{idx}.power"), l.power, r.power, 2e-9);
        assert_close_f64(
            &format!("peak{idx}.fap"),
            l.false_alarm_probability,
            r.false_alarm_probability,
            0.0,
        );
    }
}

#[cfg(feature = "cuda")]
pub(in super::super) fn assert_acf_close(
    left: &AutocorrelationReport,
    right: &AutocorrelationReport,
) {
    assert_eq!(left.lags, right.lags);
    assert_eq!(left.pair_counts, right.pair_counts);
    for (idx, (&l, &r)) in left
        .coefficients
        .iter()
        .zip(&right.coefficients)
        .enumerate()
    {
        assert_close_f64(&format!("acf[{idx}]"), l, r, 2e-9);
    }
    assert_eq!(left.dominant_period, right.dominant_period);
}

#[cfg(feature = "cuda")]
pub(in super::super) fn assert_ccf_close(
    left: &CrossCorrelationReport,
    right: &CrossCorrelationReport,
) {
    assert_eq!(left.points.len(), right.points.len());
    assert_eq!(left.peak_lag, right.peak_lag);
    for (idx, (l, r)) in left.points.iter().zip(&right.points).enumerate() {
        assert_eq!(l.lag, r.lag);
        assert_eq!(l.n_pairs, r.n_pairs);
        assert_close_f32(&format!("ccf[{idx}].r"), l.correlation, r.correlation, 2e-5);
        assert_close_f32(&format!("ccf[{idx}].p"), l.p_value, r.p_value, 2e-4);
    }
}

#[cfg(feature = "cuda")]
pub(in super::super) fn assert_hawkes_close(left: &HawkesReport, right: &HawkesReport) {
    assert_eq!(left.processes, right.processes);
    assert_eq!(left.event_counts, right.event_counts);
    assert_close_f32(
        "hawkes spectral",
        left.spectral_radius,
        right.spectral_radius,
        5e-4,
    );
    for (row_idx, (left_row, right_row)) in left
        .branching_matrix
        .iter()
        .zip(&right.branching_matrix)
        .enumerate()
    {
        for (col_idx, (&l, &r)) in left_row.iter().zip(right_row).enumerate() {
            assert_close_f32(&format!("hawkes[{row_idx}][{col_idx}]"), l, r, 5e-4);
        }
    }
    for (source, target) in [
        ("alpha", "alpha"),
        ("beta", "beta"),
        ("alpha", "beta"),
        ("beta", "alpha"),
    ] {
        assert!(has_edge(right, source, target), "{right:?}");
    }
}

#[cfg(feature = "cuda")]
pub(in super::super) fn acf_summary(report: &AutocorrelationReport) -> serde_json::Value {
    json!({
        "n_samples": report.n_samples,
        "slot_width": report.slot_width,
        "dominant_period": report.dominant_period,
        "first_lags": report.lags.iter().take(6).copied().collect::<Vec<_>>(),
        "first_coefficients": report.coefficients.iter().take(6).copied().collect::<Vec<_>>(),
        "first_pair_counts": report.pair_counts.iter().take(6).copied().collect::<Vec<_>>()
    })
}

#[cfg(feature = "cuda")]
pub(in super::super) fn ccf_summary(report: &CrossCorrelationReport) -> serde_json::Value {
    json!({
        "peak_lag": report.peak_lag,
        "peak_correlation": report.peak_correlation,
        "points": report.points.iter().map(|point| {
            json!({
                "lag": point.lag,
                "correlation": point.correlation,
                "p_value": point.p_value,
                "n_pairs": point.n_pairs
            })
        }).collect::<Vec<_>>()
    })
}

#[cfg(feature = "cuda")]
pub(in super::super) fn hawkes_summary(report: &HawkesReport) -> serde_json::Value {
    json!({
        "estimator": report.estimator,
        "spectral_radius": report.spectral_radius,
        "stability": report.stability,
        "baseline_rates": report.baseline_rates,
        "branching_matrix": report.branching_matrix,
        "retained_edges": report.retained_edges
    })
}

#[cfg(feature = "cuda")]
fn has_edge(report: &HawkesReport, source: &str, target: &str) -> bool {
    report
        .retained_edges
        .iter()
        .any(|edge| edge.source == source && edge.target == target)
}

#[cfg(feature = "cuda")]
fn assert_close_f64(name: &str, left: f64, right: f64, tolerance: f64) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch left={left} right={right} diff={diff} tolerance={tolerance}"
    );
}

#[cfg(feature = "cuda")]
fn assert_close_f32(name: &str, left: f32, right: f32, tolerance: f32) {
    let diff = (left - right).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch left={left} right={right} diff={diff} tolerance={tolerance}"
    );
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
pub(in super::super) fn restore_strict_env(previous: Option<std::ffi::OsString>) {
    match previous {
        Some(value) => unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", value) },
        None => unsafe { std::env::remove_var("CALYX_ASSAY_CUDA_STRICT") },
    }
}

#[cfg(feature = "cuda")]
pub(in super::super) fn write_fsv_artifact(value: serde_json::Value) -> serde_json::Value {
    let root = std::env::var_os("CALYX_ASSAY_ISSUE1507_FSV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/issue1507-fsv"));
    std::fs::create_dir_all(&root).expect("create issue1507 FSV dir");
    let root = std::fs::canonicalize(&root).expect("canonicalize issue1507 FSV dir");
    let path = root.join("issue1507-periodicity-hawkes-fsv-readback.json");
    let mut value = value;
    value["source_of_truth"] = json!(path.display().to_string());
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1507 FSV");
    std::fs::write(&path, bytes).expect("write issue1507 FSV");
    let readback = std::fs::read(&path).expect("read issue1507 FSV");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1507 FSV");
    println!(
        "ISSUE1507_PERIODICITY_HAWKES_FSV_READBACK path={} bytes={} blake3={}",
        path.display(),
        readback.len(),
        blake3::hash(&readback).to_hex()
    );
    println!(
        "ISSUE1507_PERIODICITY_HAWKES_FSV_DATA {}",
        String::from_utf8_lossy(&readback)
    );
    restored
}
