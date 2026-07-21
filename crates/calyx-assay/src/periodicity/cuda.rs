use super::*;

#[cfg(feature = "cuda")]
pub(super) fn lomb_scargle_with_config_cuda_strict_impl(
    times: &[f64],
    values: &[f64],
    config: &PeriodogramConfig,
) -> Result<PeriodicityReport> {
    let stats = validate_series(times, values)?;
    if config.fap_permutations == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "fap_permutations must be >= 1; the FAP is mandatory, not optional",
        ));
    }
    let frequencies = frequency_grid(times, stats.span, config)?;
    let centered: Vec<f64> = values.iter().map(|value| value - stats.mean).collect();
    let permutations = crate::cuda_strict::deterministic_permutations(
        times.len(),
        config.fap_permutations,
        config.seed,
    )?;
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("periodogram", err))?;
    let batch = calyx_forge::periodogram_batch_host(
        backend.context(),
        times,
        &centered,
        stats.variance,
        &frequencies,
        Some(&permutations),
    )
    .map_err(|err| crate::cuda_strict::forge_to_calyx("periodogram", err))?;
    let mut peaks = ranked_peaks(&frequencies, &batch.powers, config.max_peaks);
    assign_permutation_fap_from_maxes(&mut peaks, &batch.permutation_max_powers, config)?;
    Ok(PeriodicityReport {
        frequencies,
        powers: batch.powers,
        peaks,
        n_samples: times.len(),
        time_span: stats.span,
        trust: TrustTag::Provisional,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn lomb_scargle_with_config_cuda_strict_impl(
    _times: &[f64],
    _values: &[f64],
    _config: &PeriodogramConfig,
) -> Result<PeriodicityReport> {
    Err(crate::cuda_strict::cuda_unavailable("periodogram"))
}

#[cfg(feature = "cuda")]
pub(super) fn autocorrelation_cuda_strict_impl(
    times: &[f64],
    values: &[f64],
) -> Result<AutocorrelationReport> {
    let stats = validate_series(times, values)?;
    if times.len() > MAX_ACF_SAMPLES {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "autocorrelation input has {} samples (max {MAX_ACF_SAMPLES}); bin first",
            times.len()
        )));
    }
    let slot_width = median_spacing(times);
    let max_lag = stats.span / 2.0;
    let slot_count = (max_lag / slot_width).floor() as usize;
    if slot_count == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "autocorrelation span too short for a single lag slot",
        ));
    }
    let centered: Vec<f64> = values.iter().map(|value| value - stats.mean).collect();
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("autocorrelation", err))?;
    let sums = calyx_forge::autocorrelation_sums_host(
        backend.context(),
        times,
        &centered,
        stats.variance,
        slot_width,
        max_lag,
        slot_count,
    )
    .map_err(|err| crate::cuda_strict::forge_to_calyx("autocorrelation", err))?;
    let mut lags = Vec::new();
    let mut coefficients = Vec::new();
    let mut pair_counts = Vec::new();
    for slot in 1..=slot_count {
        if sums.counts[slot] > 0 {
            lags.push(slot as f64 * slot_width);
            coefficients.push((sums.sums[slot] / sums.counts[slot] as f64) / stats.variance);
            pair_counts.push(sums.counts[slot]);
        }
    }
    let dominant_period = positive_local_max(&lags, &coefficients);
    Ok(AutocorrelationReport {
        lags,
        coefficients,
        pair_counts,
        slot_width,
        dominant_period,
        n_samples: times.len(),
        trust: TrustTag::Provisional,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn autocorrelation_cuda_strict_impl(
    _times: &[f64],
    _values: &[f64],
) -> Result<AutocorrelationReport> {
    Err(crate::cuda_strict::cuda_unavailable("autocorrelation"))
}

#[cfg(feature = "cuda")]
fn assign_permutation_fap_from_maxes(
    peaks: &mut [PeriodogramPeak],
    max_powers: &[f64],
    config: &PeriodogramConfig,
) -> Result<()> {
    if config.fap_permutations == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "fap_permutations must be >= 1; the FAP is mandatory, not optional",
        ));
    }
    if peaks.is_empty() {
        return Ok(());
    }
    if max_powers.len() != config.fap_permutations {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "periodogram CUDA returned {} permutation maxima for {} configured permutations",
            max_powers.len(),
            config.fap_permutations
        )));
    }
    for (idx, &max_power) in max_powers.iter().enumerate() {
        if !max_power.is_finite() {
            return Err(CalyxError::forge_numerical_invariant(format!(
                "periodogram CUDA permutation max[{idx}] is non-finite: {max_power}"
            )));
        }
    }
    for peak in peaks.iter_mut() {
        let exceed = max_powers
            .iter()
            .filter(|&&max_power| max_power >= peak.power)
            .count();
        peak.false_alarm_probability = (exceed + 1) as f64 / (config.fap_permutations + 1) as f64;
    }
    Ok(())
}
