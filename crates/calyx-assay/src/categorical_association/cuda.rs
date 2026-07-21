use super::*;

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(super) fn categorical_association_cuda_strict_with_stats_impl(
    x_dense: &[u32],
    y_dense: &[u32],
    rows: usize,
    cols: usize,
    n_samples: usize,
) -> Result<(CategoricalReport, DependenceCudaStats)> {
    let context = crate::dependence_dispatch::dependence_cuda_context("categorical association")?;
    let counts = calyx_forge::categorical_counts_host(context, x_dense, y_dense, rows, cols)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("categorical association", err))?;
    let report = categorical_report_from_table(&counts.table, rows, cols, n_samples)?;
    Ok((report, counts.stats.into()))
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "cuda"))]
pub(super) fn categorical_association_cuda_strict_with_stats_impl(
    _x_dense: &[u32],
    _y_dense: &[u32],
    _rows: usize,
    _cols: usize,
    _n_samples: usize,
) -> Result<(CategoricalReport, DependenceCudaStats)> {
    Err(crate::cuda_strict::cuda_unavailable(
        "categorical association",
    ))
}
