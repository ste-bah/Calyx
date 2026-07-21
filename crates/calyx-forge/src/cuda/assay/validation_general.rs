use super::*;

pub(super) fn validate_flat_matrix(
    op: &'static str,
    values: &[f32],
    rows: usize,
    dim: usize,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1],
            got: vec![rows, dim],
            remediation: format!("{op} requires non-empty rows and non-zero dimension"),
        });
    }
    let expected = rows
        .checked_mul(dim)
        .ok_or_else(|| shape_overflow("assay flat matrix shape overflow"))?;
    if values.len() != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![expected],
            got: vec![values.len()],
            remediation: format!("{op} flat input length must equal rows*dim"),
        });
    }
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                op,
                format!("{op} contains non-finite value at flat index {idx}: {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_vector_f32(op: &'static str, values: &[f32], len: usize) -> Result<()> {
    if values.len() != len {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![len],
            got: vec![values.len()],
            remediation: format!("{op} length must equal sample count"),
        });
    }
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                op,
                format!("{op} contains non-finite value at index {idx}: {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_labels(labels: &[i32], n: usize) -> Result<()> {
    if labels.len() != n {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![n],
            got: vec![labels.len()],
            remediation: "mixed KSG labels length must equal sample count".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_binary_labels(labels: &[i32], n: usize) -> Result<()> {
    validate_labels(labels, n)?;
    for (idx, label) in labels.iter().copied().enumerate() {
        if label != 0 && label != 1 {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![0, 1],
                got: vec![label.max(0) as usize],
                remediation: format!("logistic label at row {idx} must be 0 or 1"),
            });
        }
    }
    Ok(())
}
