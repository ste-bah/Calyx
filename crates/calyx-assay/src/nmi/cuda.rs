use super::*;

#[cfg(feature = "cuda")]
pub(super) fn partitioned_histogram_nmi_cuda_strict_with_stats_impl(
    x: &[f32],
    y: &[f32],
    bins: usize,
) -> Result<(NmiReport, DependenceCudaStats)> {
    let context = crate::dependence_dispatch::dependence_cuda_context("histogram NMI")?;
    let counts = calyx_forge::histogram_counts_host(context, x, y, bins)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("histogram NMI", err))?;
    let report = nmi_from_counts(
        &counts.x_counts,
        &counts.y_counts,
        &counts.joint_counts,
        bins,
        x.len(),
    );
    Ok((report, counts.stats.into()))
}

#[cfg(not(feature = "cuda"))]
pub(super) fn partitioned_histogram_nmi_cuda_strict_with_stats_impl(
    _x: &[f32],
    _y: &[f32],
    _bins: usize,
) -> Result<(NmiReport, DependenceCudaStats)> {
    Err(crate::cuda_strict::cuda_unavailable("histogram NMI"))
}
