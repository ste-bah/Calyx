use super::*;

#[cfg(feature = "cuda")]
pub(super) fn empirical_copula_cuda_strict_with_stats_impl(
    x: &[f64],
    y: &[f64],
    tail_q: f64,
) -> Result<(CopulaTailReport, DependenceCudaStats)> {
    let context = crate::dependence_dispatch::dependence_cuda_context("empirical copula")?;
    let device = calyx_forge::copula_terms_host(context, x, y, tail_q)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("empirical copula", err))?;
    if device
        .x_tie_sizes
        .iter()
        .chain(&device.y_tie_sizes)
        .any(|&size| size > 1)
    {
        return Err(CalyxError::assay_degenerate_input(
            "empirical copula requires continuous margins; CUDA rank metadata found a tie",
        ));
    }
    let n = x.len() as f64;
    let c_mid = device.c_mid_count as f64 / n;
    let report = CopulaTailReport {
        estimator: "empirical_rank_copula_tail_dependence".to_string(),
        n_samples: x.len(),
        tail_q,
        blomqvist_beta: (4.0 * c_mid - 1.0).clamp(-1.0, 1.0),
        hoeffding_d_cvm: 30.0 * device.hoeffding_terms.iter().sum::<f64>() / n,
        gini_gamma: (4.0 * device.gini_terms.iter().sum::<f64>() / n - 2.0).clamp(-1.0, 1.0),
        lower_tail_lambda: ((device.lower_tail_count as f64 / n) / tail_q).clamp(0.0, 1.0),
        upper_tail_lambda: ((device.upper_tail_count as f64 / n) / tail_q).clamp(0.0, 1.0),
        lower_tail_count: device.lower_tail_count as usize,
        upper_tail_count: device.upper_tail_count as usize,
    };
    Ok((report, device.stats.into()))
}

#[cfg(not(feature = "cuda"))]
pub(super) fn empirical_copula_cuda_strict_with_stats_impl(
    _x: &[f64],
    _y: &[f64],
    _tail_q: f64,
) -> Result<(CopulaTailReport, DependenceCudaStats)> {
    Err(crate::cuda_strict::cuda_unavailable("empirical copula"))
}
