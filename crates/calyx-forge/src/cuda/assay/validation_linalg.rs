use super::*;

pub(super) fn validate_correlation_columns(columns: &[f32], n: usize, d: usize) -> Result<()> {
    if n < 2 || !(2..=MAX_LINALG_VARIABLES).contains(&d) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![2, 2, MAX_LINALG_VARIABLES],
            got: vec![n, d],
            remediation: format!(
                "correlation precision requires n >= 2 and 2 <= variables <= {MAX_LINALG_VARIABLES}"
            ),
        });
    }
    let expected = n
        .checked_mul(d)
        .ok_or_else(|| shape_overflow("correlation column buffer shape overflow"))?;
    if columns.len() != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![expected],
            got: vec![columns.len()],
            remediation:
                "correlation precision input must be variable-major columns with length n*d"
                    .to_string(),
        });
    }
    for (idx, value) in columns.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                "correlation_precision_host",
                format!("correlation input contains non-finite value at flat index {idx}: {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_matrix_readback(op: &'static str, values: &[f64]) -> Result<()> {
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                op,
                format!(
                    "{op} device readback contains non-finite value at flat index {idx}: {value}"
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_granger_batch_inputs(x: &[f32], y: &[f32], lags: &[usize]) -> Result<()> {
    validate_pair_f32("granger_lag_summaries_host", x, y, 1)?;
    if lags.is_empty() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: "Granger CUDA batch requires at least one lag".to_string(),
        });
    }
    for (idx, &lag) in lags.iter().enumerate() {
        if lag > i32::MAX as usize {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![i32::MAX as usize],
                got: vec![lag],
                remediation: format!("Granger lag at index {idx} exceeds CUDA i32 argument range"),
            });
        }
    }
    Ok(())
}
