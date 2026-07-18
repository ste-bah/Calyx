use super::*;

#[cfg(feature = "cuda")]
pub(super) fn granger_causality_lags_cuda_strict_impl(
    x: &[f32],
    y: &[f32],
    lags: usize,
) -> Result<GrangerReport> {
    validate_granger_lags_request(x, y, lags)?;
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("Granger", err))?;
    let batch = calyx_forge::granger_lag_summaries_host(backend.context(), x, y, &[lags])
        .map_err(|err| crate::cuda_strict::forge_linear_algebra_to_calyx("Granger", err))?;
    let summary = batch.summaries.first().ok_or_else(|| {
        CalyxError::forge_numerical_invariant("Granger CUDA returned no lag summary")
    })?;
    granger_report_from_cuda_summary(x, y, summary)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn granger_causality_lags_cuda_strict_impl(
    _x: &[f32],
    _y: &[f32],
    _lags: usize,
) -> Result<GrangerReport> {
    Err(crate::cuda_strict::cuda_unavailable("Granger"))
}

#[cfg(feature = "cuda")]
pub(super) fn granger_causality_sweep_lags_cuda_strict_impl(
    x: &[f32],
    y: &[f32],
    lags: &[usize],
) -> Result<GrangerReport> {
    if lags.is_empty() {
        return Err(CalyxError::assay_insufficient_samples(
            "Granger sweep requires a non-empty lag set",
        ));
    }
    validate_granger_pair_for_cuda(x, y)?;
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("Granger", err))?;
    let batch = calyx_forge::granger_lag_summaries_host(backend.context(), x, y, lags)
        .map_err(|err| crate::cuda_strict::forge_linear_algebra_to_calyx("Granger", err))?;
    let mut best: Option<GrangerReport> = None;
    let mut last_err: Option<CalyxError> = None;
    for summary in &batch.summaries {
        match granger_report_from_cuda_summary(x, y, summary) {
            Ok(report) => {
                let take = match &best {
                    None => true,
                    Some(b) => {
                        report.p_value < b.p_value
                            || (report.p_value == b.p_value && report.f_statistic > b.f_statistic)
                    }
                };
                if take {
                    best = Some(report);
                }
            }
            Err(err) => last_err = Some(err),
        }
    }
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            CalyxError::assay_insufficient_samples("Granger sweep: no admissible lag")
        })
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn granger_causality_sweep_lags_cuda_strict_impl(
    _x: &[f32],
    _y: &[f32],
    _lags: &[usize],
) -> Result<GrangerReport> {
    Err(crate::cuda_strict::cuda_unavailable("Granger"))
}

#[cfg(feature = "cuda")]
fn validate_granger_lags_request(x: &[f32], y: &[f32], lags: usize) -> Result<()> {
    if lags == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "Granger causality requires lags ≥ 1",
        ));
    }
    validate_granger_pair_for_cuda(x, y)?;
    let n = x.len();
    if n < 3 * lags + 2 {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "Granger causality with {lags} lags requires at least {} samples (3p+2); got {n}",
            3 * lags + 2
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_granger_pair_for_cuda(x: &[f32], y: &[f32]) -> Result<()> {
    if x.len() != y.len() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "Granger causality requires paired series: x={} y={}",
            x.len(),
            y.len()
        )));
    }
    let _ = to_finite_f64("x", x)?;
    let _ = to_finite_f64("y", y)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn granger_report_from_cuda_summary(
    _x: &[f32],
    y: &[f32],
    summary: &calyx_forge::CudaGrangerLagSummary,
) -> Result<GrangerReport> {
    match summary.status {
        calyx_forge::CUDA_GRANGER_STATUS_OK => {}
        calyx_forge::CUDA_GRANGER_STATUS_INVALID_LAG => {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "Granger CUDA lag {} is inadmissible for sample count {} or exceeds the strict CUDA lag limit",
                summary.lag,
                y.len()
            )));
        }
        calyx_forge::CUDA_GRANGER_STATUS_NONFINITE => {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "Granger CUDA lag {} produced non-finite linear algebra state",
                summary.lag
            )));
        }
        calyx_forge::CUDA_GRANGER_STATUS_RANK_DEFICIENT => {
            return Err(CalyxError::assay_degenerate_input(format!(
                "Granger causality undefined at lag {}: design matrix is rank-deficient (collinear/constant regressors)",
                summary.lag
            )));
        }
        other => {
            return Err(CalyxError::forge_numerical_invariant(format!(
                "Granger CUDA returned unknown status {other} for lag {}",
                summary.lag
            )));
        }
    }
    let p = summary.lag;
    let t = summary.n_used;
    if t == 0 || summary.df_den == 0 {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "Granger CUDA returned empty degrees of freedom for lag {p}: n_used={} df_den={}",
            summary.n_used, summary.df_den
        )));
    }
    let response = &y[p..];
    let mean_y = response.iter().map(|&v| v as f64).sum::<f64>() / t as f64;
    let tss = response
        .iter()
        .map(|&v| {
            let delta = v as f64 - mean_y;
            delta * delta
        })
        .sum::<f64>();
    if summary.rss_unrestricted <= tss * MIN_RSS_FRACTION {
        return Err(CalyxError::assay_degenerate_input(
            "Granger causality undefined: unrestricted model fits Y perfectly (RSS_u ≈ 0)",
        ));
    }

    let df_num = p;
    let df_den = summary.df_den;
    let numerator = ((summary.rss_restricted - summary.rss_unrestricted).max(0.0)) / df_num as f64;
    let denominator = summary.rss_unrestricted / df_den as f64;
    let f_statistic = numerator / denominator;
    let p_value = f_upper_tail_p(f_statistic, df_num as f64, df_den as f64)?;
    Ok(GrangerReport {
        f_statistic: f_statistic as f32,
        p_value: p_value as f32,
        lags: p,
        df_num,
        df_den,
        rss_restricted: summary.rss_restricted as f32,
        rss_unrestricted: summary.rss_unrestricted as f32,
        n_used: t,
    })
}
