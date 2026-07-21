use super::*;

#[cfg(feature = "cuda")]
pub(super) fn mic_with_alpha_cuda_strict_with_stats_impl(
    x: &[f64],
    y: &[f64],
    b_budget: usize,
) -> Result<(MicReport, DependenceCudaStats)> {
    let context = crate::dependence_dispatch::dependence_cuda_context("MIC")?;
    let device = calyx_forge::mic_pair_host(context, x, y, b_budget)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("MIC", err))?;
    let (mic, best_nx, best_ny) = if device.primary_x.score >= device.primary_y.score {
        (
            device.primary_x.score,
            device.primary_x.primary_bins,
            device.primary_x.secondary_bins,
        )
    } else {
        (
            device.primary_y.score,
            device.primary_y.secondary_bins,
            device.primary_y.primary_bins,
        )
    };
    let report = MicReport {
        mic: mic.clamp(0.0, 1.0) as f32,
        best_nx,
        best_ny,
        b_budget,
        n_samples: x.len(),
    };
    Ok((report, device.stats.into()))
}

#[cfg(not(feature = "cuda"))]
pub(super) fn mic_with_alpha_cuda_strict_with_stats_impl(
    _x: &[f64],
    _y: &[f64],
    _b_budget: usize,
) -> Result<(MicReport, DependenceCudaStats)> {
    Err(crate::cuda_strict::cuda_unavailable("MIC"))
}
