use super::*;

pub(super) fn validate_split_buffers(
    op: &'static str,
    n: usize,
    train_offsets: &[i32],
    train_indices: &[i32],
    test_offsets: &[i32],
    test_indices: &[i32],
) -> Result<usize> {
    if train_offsets.len() < 2 || train_offsets.len() != test_offsets.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![2, train_offsets.len()],
            got: vec![train_offsets.len(), test_offsets.len()],
            remediation: format!("{op} split offsets must have matching fit_count+1 length"),
        });
    }
    let fit_count = train_offsets.len() - 1;
    validate_offsets(
        op,
        "train",
        n,
        train_offsets,
        train_indices.len(),
        train_indices,
    )?;
    validate_offsets(
        op,
        "test",
        n,
        test_offsets,
        test_indices.len(),
        test_indices,
    )?;
    for fit in 0..fit_count {
        if train_offsets[fit + 1] <= train_offsets[fit]
            || test_offsets[fit + 1] <= test_offsets[fit]
        {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![1, 1],
                got: vec![
                    (train_offsets[fit + 1] - train_offsets[fit]).max(0) as usize,
                    (test_offsets[fit + 1] - test_offsets[fit]).max(0) as usize,
                ],
                remediation: format!("{op} fit {fit} requires non-empty train and test splits"),
            });
        }
    }
    Ok(fit_count)
}

pub(super) fn validate_linear_cka_inputs(
    values: &[f32],
    lens_offsets: &[i32],
    dimensions: &[i32],
    row_count: usize,
    tuples: &[i32],
) -> Result<(usize, usize, usize)> {
    if row_count < 4 || dimensions.len() < 2 || lens_offsets.len() != dimensions.len() + 1 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![4, 2, dimensions.len() + 1],
            got: vec![row_count, dimensions.len(), lens_offsets.len()],
            remediation:
                "linear CKA requires row_count >= 4, at least two lenses, and lens_count+1 offsets"
                    .to_string(),
        });
    }
    if tuples.is_empty() || !tuples.len().is_multiple_of(4) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![4],
            got: vec![tuples.len()],
            remediation: "linear CKA tuple buffer length must be tuple_count*4".to_string(),
        });
    }
    if lens_offsets.first().copied() != Some(0)
        || lens_offsets.last().copied() != Some(to_i32(values.len(), "linear CKA values len")?)
    {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![0, values.len()],
            got: vec![
                lens_offsets.first().copied().unwrap_or_default().max(0) as usize,
                lens_offsets.last().copied().unwrap_or_default().max(0) as usize,
            ],
            remediation: "linear CKA lens offsets must start at 0 and end at values length"
                .to_string(),
        });
    }
    for lens in 0..dimensions.len() {
        let dim = dimensions[lens];
        let start = lens_offsets[lens];
        let end = lens_offsets[lens + 1];
        if dim <= 0 || start < 0 || end <= start {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![1],
                got: vec![
                    dim.max(0) as usize,
                    start.max(0) as usize,
                    end.max(0) as usize,
                ],
                remediation: format!("linear CKA lens {lens} has invalid dimension or offsets"),
            });
        }
        let expected_len = row_count
            .checked_mul(dim as usize)
            .ok_or_else(|| shape_overflow("linear CKA lens shape overflow"))?;
        if (end - start) as usize != expected_len {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![expected_len],
                got: vec![(end - start).max(0) as usize],
                remediation: format!("linear CKA lens {lens} offset span must equal rows*dim"),
            });
        }
    }
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(numerical(
                "linear_cka_pair_estimates_host",
                format!("linear CKA value at flat index {idx} is non-finite: {value}"),
            ));
        }
    }
    for (tuple_idx, tuple) in tuples.chunks_exact(4).enumerate() {
        let valid = tuple[0] >= 0
            && tuple[1] >= 0
            && tuple[2] >= 0
            && tuple[3] >= 0
            && tuple[0] < tuple[1]
            && tuple[1] < tuple[2]
            && tuple[2] < tuple[3]
            && (tuple[3] as usize) < row_count;
        if !valid {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![row_count],
                got: tuple.iter().map(|value| (*value).max(0) as usize).collect(),
                remediation: format!(
                    "linear CKA tuple {tuple_idx} must contain four sorted distinct row indices"
                ),
            });
        }
    }
    let lens_count = dimensions.len();
    let tuple_count = tuples.len() / 4;
    let pair_count = lens_count
        .checked_mul(lens_count.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| shape_overflow("linear CKA pair count overflow"))?;
    Ok((lens_count, tuple_count, pair_count))
}

pub(super) fn validate_linear_cka_outputs(
    raw_signed_point: &[f32],
    redundancy_point: &[f32],
    mc_standard_error: &[f32],
    mc_gate_upper_estimate: &[f32],
) -> Result<()> {
    let len = raw_signed_point.len();
    if redundancy_point.len() != len
        || mc_standard_error.len() != len
        || mc_gate_upper_estimate.len() != len
    {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![len],
            got: vec![
                redundancy_point.len(),
                mc_standard_error.len(),
                mc_gate_upper_estimate.len(),
            ],
            remediation: "linear CKA output buffers must have matching pair count".to_string(),
        });
    }
    for idx in 0..len {
        let raw = raw_signed_point[idx];
        let point = redundancy_point[idx];
        let se = mc_standard_error[idx];
        let gate = mc_gate_upper_estimate[idx];
        let valid = raw.is_finite()
            && (-1.0..=1.0).contains(&raw)
            && point.is_finite()
            && (0.0..=1.0).contains(&point)
            && (point - raw.max(0.0)).abs() <= 1.0e-5
            && se.is_finite()
            && se >= 0.0
            && gate.is_finite()
            && (point..=1.0).contains(&gate);
        if !valid {
            return Err(numerical(
                "linear CKA outputs",
                format!(
                    "invalid linear CKA pair {idx}: raw={raw} point={point} se={se} gate={gate}"
                ),
            ));
        }
    }
    Ok(())
}
