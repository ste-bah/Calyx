use super::*;

pub(super) fn validate_offsets(
    op: &'static str,
    name: &'static str,
    n: usize,
    offsets: &[i32],
    index_len: usize,
    indices: &[i32],
) -> Result<()> {
    if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(index_len as i32) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![0, index_len],
            got: vec![
                offsets.first().copied().unwrap_or_default().max(0) as usize,
                offsets.last().copied().unwrap_or_default().max(0) as usize,
            ],
            remediation: format!("{op} {name} offsets must start at 0 and end at index length"),
        });
    }
    for window in offsets.windows(2) {
        if window[0] < 0 || window[1] < window[0] {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![0],
                got: vec![window[0].max(0) as usize, window[1].max(0) as usize],
                remediation: format!("{op} {name} offsets must be non-negative and monotonic"),
            });
        }
    }
    for (idx, row) in indices.iter().copied().enumerate() {
        if row < 0 || row as usize >= n {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![0, n.saturating_sub(1)],
                got: vec![row.max(0) as usize],
                remediation: format!("{op} {name} index {idx} is outside sample range"),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_neighbor_k(op: &'static str, n: usize, k: usize) -> Result<()> {
    if n < 2 || k == 0 || k >= n || k > 32 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, n.saturating_sub(1).min(32)],
            got: vec![k],
            remediation: format!("{op} requires 0 < k < sample count and k <= 32"),
        });
    }
    Ok(())
}

pub(super) fn read_f32(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    op: &'static str,
) -> Result<Vec<f32>> {
    let host = ctx
        .inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} readback failed: {err}")))?;
    for (idx, value) in host.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                op,
                format!("{op} readback contains non-finite value at index {idx}: {value}"),
            ));
        }
    }
    Ok(host)
}

pub(super) fn read_usize_counts(
    ctx: &CudaContext,
    values: &CudaSlice<i32>,
    op: &'static str,
) -> Result<Vec<usize>> {
    let host = ctx
        .inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} readback failed: {err}")))?;
    let mut counts = Vec::with_capacity(host.len());
    for (idx, value) in host.into_iter().enumerate() {
        if value < 0 {
            return Err(numerical(
                op,
                format!("{op} readback contains negative count at index {idx}: {value}"),
            ));
        }
        counts.push(value as usize);
    }
    Ok(counts)
}

pub(super) fn validate_pair_f32(
    op: &'static str,
    x: &[f32],
    y: &[f32],
    min_n: usize,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![x.len()],
            got: vec![y.len()],
            remediation: format!("{op} requires paired equal-length samples"),
        });
    }
    if x.len() < min_n {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![min_n],
            got: vec![x.len()],
            remediation: format!("{op} has too few samples"),
        });
    }
    for (idx, (&left, &right)) in x.iter().zip(y.iter()).enumerate() {
        if !(left.is_finite() && right.is_finite()) {
            return Err(numerical(
                op,
                format!("non-finite paired input at row {idx}: x={left} y={right}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_pair_f64(
    op: &'static str,
    x: &[f64],
    y: &[f64],
    min_n: usize,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![x.len()],
            got: vec![y.len()],
            remediation: format!("{op} requires paired equal-length samples"),
        });
    }
    if x.len() < min_n {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![min_n],
            got: vec![x.len()],
            remediation: format!("{op} has too few samples"),
        });
    }
    for (idx, (&left, &right)) in x.iter().zip(y.iter()).enumerate() {
        if !(left.is_finite() && right.is_finite()) {
            return Err(numerical(
                op,
                format!("non-finite paired input at row {idx}: x={left} y={right}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_mmd_inputs(
    pooled: &[f64],
    n_a: usize,
    n_b: usize,
    dim: usize,
    bandwidth: f64,
) -> Result<()> {
    if n_a == 0 || n_b == 0 || dim == 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1, 1],
            got: vec![n_a, n_b, dim],
            remediation: "MMD requires non-empty sides and non-zero dimension".to_string(),
        });
    }
    let n = n_a
        .checked_add(n_b)
        .ok_or_else(|| shape_overflow("MMD sample count overflow"))?;
    let expected = n
        .checked_mul(dim)
        .ok_or_else(|| shape_overflow("MMD pooled shape overflow"))?;
    if pooled.len() != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![expected],
            got: vec![pooled.len()],
            remediation: "MMD pooled input length must equal (n_a+n_b)*dim".to_string(),
        });
    }
    validate_bandwidth("MMD bandwidth", bandwidth)?;
    for (idx, value) in pooled.iter().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                "gaussian_mmd_host",
                format!("non-finite pooled input at flat index {idx}: {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_mmd_change_inputs(
    samples: &[f64],
    n: usize,
    dim: usize,
    min_window: usize,
    bandwidth: f64,
) -> Result<()> {
    if n == 0 || dim == 0 || min_window < 2 || n < min_window.saturating_mul(2) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1, 2, min_window.saturating_mul(2)],
            got: vec![n, dim, min_window, n],
            remediation: "MMD change-point requires n >= 2*min_window, min_window >= 2, and non-zero dimension".to_string(),
        });
    }
    let expected = n
        .checked_mul(dim)
        .ok_or_else(|| shape_overflow("MMD change-point shape overflow"))?;
    if samples.len() != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![expected],
            got: vec![samples.len()],
            remediation: "MMD change-point flat input length must equal n*dim".to_string(),
        });
    }
    validate_bandwidth("MMD change-point bandwidth", bandwidth)?;
    for (idx, value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                "mmd_change_point_host",
                format!("non-finite sample at flat index {idx}: {value}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_bandwidth(name: &str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(ForgeError::ShapeMismatch {
        expected: vec![1],
        got: vec![0],
        remediation: format!("{name} must be finite and positive"),
    })
}

pub(super) fn validate_permutations(permutations: Option<&[i32]>, n: usize) -> Result<usize> {
    let Some(permutations) = permutations else {
        return Ok(0);
    };
    if n == 0 || !permutations.len().is_multiple_of(n) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![n],
            got: vec![permutations.len()],
            remediation: "assay permutation buffer length must be permutations*n".to_string(),
        });
    }
    let count = permutations.len() / n;
    for (row, chunk) in permutations.chunks(n).enumerate() {
        let mut seen = vec![false; n];
        for (col, &value) in chunk.iter().enumerate() {
            if value < 0 || value as usize >= n {
                return Err(ForgeError::ShapeMismatch {
                    expected: vec![n - 1],
                    got: vec![value.max(0) as usize],
                    remediation: format!(
                        "assay permutation row {row} col {col} has out-of-range index"
                    ),
                });
            }
            let idx = value as usize;
            if seen[idx] {
                return Err(ForgeError::ShapeMismatch {
                    expected: vec![n],
                    got: vec![idx],
                    remediation: format!(
                        "assay permutation row {row} contains duplicate index {idx}"
                    ),
                });
            }
            seen[idx] = true;
        }
    }
    Ok(count)
}
