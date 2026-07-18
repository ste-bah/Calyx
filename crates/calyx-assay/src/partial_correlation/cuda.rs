use super::*;

#[cfg(feature = "cuda")]
pub(super) fn pearson_cuda_strict_impl(x: &[f32], y: &[f32]) -> Result<PearsonReport> {
    validate_pearson_inputs_for_cuda(x, y)?;
    let n = x.len();
    let columns = variable_major_columns(&[x, y]);
    let matrix = correlation_precision_cuda(&columns, n, 2, "Pearson")?;
    let r = matrix.corr[1];
    let (t_statistic, p_value, ci_low, ci_high) = correlation_inference(r, n, 0)?;
    Ok(PearsonReport {
        r: r as f32,
        t_statistic: t_statistic as f32,
        p_value: p_value as f32,
        ci_low: ci_low as f32,
        ci_high: ci_high as f32,
        n_samples: n,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn pearson_cuda_strict_impl(_x: &[f32], _y: &[f32]) -> Result<PearsonReport> {
    Err(crate::cuda_strict::cuda_unavailable("Pearson"))
}

#[cfg(feature = "cuda")]
pub(super) fn partial_correlation_cuda_strict_impl(
    x: &[f32],
    y: &[f32],
    z: &[f32],
) -> Result<PartialReport> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "partial correlation requires equal-length x/y/z: x={} y={} z={}",
            x.len(),
            y.len(),
            z.len()
        )));
    }
    let n = x.len();
    if n < MIN_PEARSON_SAMPLES + 1 {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "first-order partial correlation requires at least {} samples; got {n}",
            MIN_PEARSON_SAMPLES + 1
        )));
    }
    let _ = to_finite_f64("partial correlation", "x", x)?;
    let _ = to_finite_f64("partial correlation", "y", y)?;
    let _ = to_finite_f64("partial correlation", "z", z)?;
    let columns = variable_major_columns(&[x, y, z]);
    let matrix = correlation_precision_cuda(&columns, n, 3, "partial correlation")?;
    partial_report_from_precision(
        matrix.corr[1],
        matrix.precision[1],
        matrix.precision[0],
        matrix.precision[4],
        n,
        1,
    )
}

#[cfg(not(feature = "cuda"))]
pub(super) fn partial_correlation_cuda_strict_impl(
    _x: &[f32],
    _y: &[f32],
    _z: &[f32],
) -> Result<PartialReport> {
    Err(crate::cuda_strict::cuda_unavailable("partial correlation"))
}

#[cfg(feature = "cuda")]
pub(super) fn partial_correlation_controlling_cuda_strict_impl(
    x: &[f32],
    y: &[f32],
    controls: &[&[f32]],
) -> Result<PartialReport> {
    if controls.is_empty() {
        return Err(CalyxError::assay_insufficient_samples(
            "partial correlation requires at least one control column; use `pearson` for zero-order",
        ));
    }
    let k = controls.len();
    let n = x.len();
    if y.len() != n || controls.iter().any(|c| c.len() != n) {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "partial correlation requires all columns length {n}: y={}, controls={:?}",
            y.len(),
            controls.iter().map(|c| c.len()).collect::<Vec<_>>()
        )));
    }
    if n < k + MIN_PEARSON_SAMPLES {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "partial correlation controlling for {k} confounders requires at least {} samples; got {n}",
            k + MIN_PEARSON_SAMPLES
        )));
    }
    let _ = to_finite_f64("partial correlation", "x", x)?;
    let _ = to_finite_f64("partial correlation", "y", y)?;
    for (i, c) in controls.iter().enumerate() {
        let _ = to_finite_f64("partial correlation", &format!("control[{i}]"), c)?;
    }
    let mut slices = Vec::with_capacity(2 + controls.len());
    slices.push(x);
    slices.push(y);
    slices.extend_from_slice(controls);
    let d = slices.len();
    let columns = variable_major_columns(&slices);
    let matrix = correlation_precision_cuda(&columns, n, d, "partial correlation")?;
    partial_report_from_precision(
        matrix.corr[1],
        matrix.precision[1],
        matrix.precision[0],
        matrix.precision[d + 1],
        n,
        k,
    )
}

#[cfg(not(feature = "cuda"))]
pub(super) fn partial_correlation_controlling_cuda_strict_impl(
    _x: &[f32],
    _y: &[f32],
    _controls: &[&[f32]],
) -> Result<PartialReport> {
    Err(crate::cuda_strict::cuda_unavailable("partial correlation"))
}

#[cfg(feature = "cuda")]
pub(crate) fn correlation_precision_cuda(
    columns: &[f32],
    n: usize,
    d: usize,
    op: &str,
) -> Result<calyx_forge::CudaCorrelationPrecision> {
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx(op, err))?;
    calyx_forge::correlation_precision_host(backend.context(), columns, n, d)
        .map_err(|err| crate::cuda_strict::forge_linear_algebra_to_calyx(op, err))
}

#[cfg(feature = "cuda")]
pub(crate) fn variable_major_columns(slices: &[&[f32]]) -> Vec<f32> {
    let n = slices
        .first()
        .map(|values| values.len())
        .unwrap_or_default();
    let mut out = Vec::with_capacity(n * slices.len());
    for values in slices {
        out.extend_from_slice(values);
    }
    out
}

#[cfg(feature = "cuda")]
fn validate_pearson_inputs_for_cuda(x: &[f32], y: &[f32]) -> Result<()> {
    if x.len() != y.len() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "Pearson requires paired samples: x={} y={}",
            x.len(),
            y.len()
        )));
    }
    let n = x.len();
    if n < MIN_PEARSON_SAMPLES {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "Pearson requires at least {MIN_PEARSON_SAMPLES} paired samples; got {n}"
        )));
    }
    let _ = to_finite_f64("Pearson", "x", x)?;
    let _ = to_finite_f64("Pearson", "y", y)?;
    Ok(())
}
