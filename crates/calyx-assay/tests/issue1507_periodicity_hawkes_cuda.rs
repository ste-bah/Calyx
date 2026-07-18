#[cfg(feature = "cuda")]
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use calyx_assay::{
    AutocorrelationReport, CrossCorrelationReport, HawkesConfig, HawkesEventSeries, HawkesReport,
    PeriodicityReport, PeriodogramConfig, autocorrelation, autocorrelation_cuda_strict,
    cross_correlation_profile, cross_correlation_profile_cuda_strict, exponential_hawkes_em,
    exponential_hawkes_em_cuda_strict, lomb_scargle_with_config,
    lomb_scargle_with_config_cuda_strict,
};
#[cfg(not(feature = "cuda"))]
use calyx_assay::{
    autocorrelation_cuda_strict, cross_correlation_profile_cuda_strict,
    exponential_hawkes_em_cuda_strict, lomb_scargle_cuda_strict,
};
#[cfg(feature = "cuda")]
use serde_json::json;
#[path = "issue1507_periodicity_hawkes_cuda/support/mod.rs"]
mod support;
#[cfg(feature = "cuda")]
use support::*;

#[cfg(feature = "cuda")]
#[test]
fn issue1507_periodicity_ccf_hawkes_cuda_matches_cpu_and_writes_fsv() {
    let (period_times, period_values) = periodic_fixture(96);
    let period_config = PeriodogramConfig {
        oversample: 4.0,
        min_frequency: Some(0.05),
        max_frequency: Some(0.25),
        fap_permutations: 16,
        seed: 1507,
        max_peaks: 3,
    };
    let period_cpu =
        lomb_scargle_with_config(&period_times, &period_values, &period_config).unwrap();
    let period_gpu =
        lomb_scargle_with_config_cuda_strict(&period_times, &period_values, &period_config)
            .unwrap();
    assert_periodicity_close(&period_cpu, &period_gpu);
    let dominant = period_gpu.dominant().expect("planted CUDA peak");
    assert!(
        (dominant.period - 7.0).abs() <= 0.35,
        "CUDA GLS period must recover planted period 7.0: {dominant:?}"
    );

    let (acf_times, acf_values) = acf_fixture(96);
    let acf_cpu = autocorrelation(&acf_times, &acf_values).unwrap();
    let acf_gpu = autocorrelation_cuda_strict(&acf_times, &acf_values).unwrap();
    assert_acf_close(&acf_cpu, &acf_gpu);

    let (ccf_x, ccf_y) = ccf_fixture(96, 3);
    let ccf_cpu = cross_correlation_profile(&ccf_x, &ccf_y, 8).unwrap();
    let ccf_gpu = cross_correlation_profile_cuda_strict(&ccf_x, &ccf_y, 8).unwrap();
    assert_ccf_close(&ccf_cpu, &ccf_gpu);
    assert_eq!(ccf_gpu.peak_lag, 3);

    let (alpha_events, beta_events) = hawkes_fixture();
    let hawkes_config = HawkesConfig::new(110.0, 2.0, 80, 0.15);
    let hawkes_processes = [
        HawkesEventSeries {
            name: "alpha",
            event_times: &alpha_events,
        },
        HawkesEventSeries {
            name: "beta",
            event_times: &beta_events,
        },
    ];
    let hawkes_cpu = exponential_hawkes_em(&hawkes_processes, &hawkes_config).unwrap();
    let hawkes_gpu = exponential_hawkes_em_cuda_strict(&hawkes_processes, &hawkes_config).unwrap();
    assert_hawkes_close(&hawkes_cpu, &hawkes_gpu);

    let previous_strict = std::env::var_os("CALYX_ASSAY_CUDA_STRICT");
    unsafe { std::env::set_var("CALYX_ASSAY_CUDA_STRICT", "1") };
    let routed_period =
        lomb_scargle_with_config(&period_times, &period_values, &period_config).unwrap();
    let routed_acf = autocorrelation(&acf_times, &acf_values).unwrap();
    let routed_ccf = cross_correlation_profile(&ccf_x, &ccf_y, 8).unwrap();
    let routed_hawkes = exponential_hawkes_em(&hawkes_processes, &hawkes_config).unwrap();
    restore_strict_env(previous_strict);
    assert_periodicity_close(&period_gpu, &routed_period);
    assert_acf_close(&acf_gpu, &routed_acf);
    assert_ccf_close(&ccf_gpu, &routed_ccf);
    assert_hawkes_close(&hawkes_gpu, &routed_hawkes);

    let edges = edge_case_readbacks(&period_times, &period_values, &ccf_x, &ccf_y);
    let artifact = json!({
        "artifact_kind": "issue1507.assay-periodicity-ccf-hawkes-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1507_FSV_DIR/issue1507-periodicity-hawkes-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_isolated_issue1507_periodicity_hawkes_cuda issue1507_periodicity_hawkes_cuda -- --nocapture",
        "device": calyx_forge::query_device_info(&calyx_forge::init_cuda(0, false).unwrap()),
        "benchmarks": benchmark_readback(),
        "minimum_sufficient_corpus": {
            "periodicity_samples": period_times.len(),
            "periodicity_frequencies": period_gpu.frequencies.len(),
            "fap_permutations": period_config.fap_permutations,
            "acf_samples": acf_times.len(),
            "ccf_samples": ccf_x.len(),
            "ccf_max_lag": 8,
            "hawkes_processes": 2,
            "hawkes_total_events": alpha_events.len() + beta_events.len(),
            "why_smaller_insufficient": "The proof needs off-peak frequencies/lags, a non-zero permutation denominator, both CCF signs, and four Hawkes self/mutual branches.",
            "why_larger_wasteful": "Larger corpora exercise the same CUDA grids, EM resident iteration, write, and readback paths without adding a distinct outcome."
        },
        "happy_path": {
            "periodicity": {"cpu": period_cpu, "gpu": period_gpu, "routed": routed_period},
            "autocorrelation": {"cpu": acf_summary(&acf_cpu), "gpu": acf_summary(&acf_gpu), "routed": acf_summary(&routed_acf)},
            "cross_correlation": {"cpu": ccf_summary(&ccf_cpu), "gpu": ccf_summary(&ccf_gpu), "routed": ccf_summary(&routed_ccf)},
            "hawkes": {"cpu": hawkes_summary(&hawkes_cpu), "gpu": hawkes_summary(&hawkes_gpu), "routed": hawkes_summary(&routed_hawkes)}
        },
        "edge_cases": edges,
    });
    let restored = write_fsv_artifact(artifact);
    assert_eq!(
        restored["artifact_kind"],
        "issue1507.assay-periodicity-ccf-hawkes-cuda-fsv.v1"
    );
    assert_eq!(
        restored["happy_path"]["cross_correlation"]["gpu"]["peak_lag"],
        3
    );
    assert!(
        restored["happy_path"]["hawkes"]["gpu"]["spectral_radius"]
            .as_f64()
            .unwrap()
            > 0.60
    );
    assert!(
        restored["edge_cases"].as_array().unwrap().len() >= 5,
        "issue1507 FSV must persist edge readbacks"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn issue1507_cuda_strict_errors_without_cuda_feature() {
    let times: Vec<f64> = (0..8).map(|idx| idx as f64).collect();
    let values: Vec<f64> = times.iter().map(|time| time.sin()).collect();
    let err = lomb_scargle_cuda_strict(&times, &values).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    let err = autocorrelation_cuda_strict(&times, &values).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    let err =
        cross_correlation_profile_cuda_strict(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 0).unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    let events = [1.0_f32, 2.0];
    let config = calyx_assay::HawkesConfig::new(3.0, 2.0, 2, 0.1);
    let err = exponential_hawkes_em_cuda_strict(
        &[calyx_assay::HawkesEventSeries {
            name: "alpha",
            event_times: &events,
        }],
        &config,
    )
    .unwrap_err();
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
}
