use super::*;

#[cfg(feature = "cuda")]
pub(super) fn rank_correlations_cuda_strict_with_stats_impl(
    x: &[f64],
    y: &[f64],
) -> Result<RankCorrelationCudaReport> {
    let context = crate::dependence_dispatch::dependence_cuda_context("rank correlation")?;
    let device = calyx_forge::rank_pair_host(context, x, y)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("rank correlation", err))?;
    let spearman = spearman_from_ranks(&device.x_ranks, &device.y_ranks)?;
    let x_ties = device
        .x_tie_sizes
        .iter()
        .copied()
        .filter(|&size| size > 0)
        .map(|size| size as usize)
        .collect::<Vec<_>>();
    let y_ties = device
        .y_tie_sizes
        .iter()
        .copied()
        .filter(|&size| size > 0)
        .map(|size| size as usize)
        .collect::<Vec<_>>();
    let kendall = kendall_from_counts(
        x.len(),
        &x_ties,
        &y_ties,
        device.concordant,
        device.discordant,
    )?;
    Ok(RankCorrelationCudaReport {
        spearman,
        kendall,
        stats: device.stats.into(),
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn rank_correlations_cuda_strict_with_stats_impl(
    _x: &[f64],
    _y: &[f64],
) -> Result<RankCorrelationCudaReport> {
    Err(crate::cuda_strict::cuda_unavailable("rank correlation"))
}
