use super::*;

pub(super) fn flatten_logistic_samples(samples: &[Vec<f32>], dim: usize) -> Result<Vec<f32>> {
    let capacity = samples
        .len()
        .checked_mul(dim)
        .ok_or_else(|| CalyxError::forge_vram_budget("logistic CUDA flat sample overflow"))?;
    let mut flat = Vec::with_capacity(capacity);
    for (row_idx, row) in samples.iter().enumerate() {
        if row.len() != dim {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "logistic CUDA row {row_idx} has dim {}, expected {dim}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(CalyxError::forge_numerical_invariant(format!(
                    "logistic CUDA sample row {row_idx} col {col_idx} is non-finite: {value}"
                )));
            }
            flat.push(value);
        }
    }
    Ok(flat)
}

type LogisticCudaSplitBuffers = (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>);

pub(super) fn split_buffers_for_cuda(
    splits: &[GroupSplit],
    n_samples: usize,
) -> Result<LogisticCudaSplitBuffers> {
    let mut train_offsets = vec![0];
    let mut train_indices = Vec::new();
    let mut test_offsets = vec![0];
    let mut test_indices = Vec::new();
    for (fit, split) in splits.iter().enumerate() {
        push_split_for_cuda(
            "train",
            fit,
            &split.train,
            n_samples,
            &mut train_offsets,
            &mut train_indices,
        )?;
        push_split_for_cuda(
            "test",
            fit,
            &split.test,
            n_samples,
            &mut test_offsets,
            &mut test_indices,
        )?;
    }
    Ok((train_offsets, train_indices, test_offsets, test_indices))
}

fn push_split_for_cuda(
    name: &'static str,
    fit: usize,
    rows: &[usize],
    n_samples: usize,
    offsets: &mut Vec<i32>,
    indices: &mut Vec<i32>,
) -> Result<()> {
    if rows.is_empty() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "logistic CUDA {name} split {fit} is empty"
        )));
    }
    for &row in rows {
        if row >= n_samples {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "logistic CUDA {name} split {fit} contains row {row}, n_samples={n_samples}"
            )));
        }
        indices.push(usize_to_i32_for_cuda(row, "logistic row index")?);
    }
    offsets.push(usize_to_i32_for_cuda(
        indices.len(),
        "logistic split offset",
    )?);
    Ok(())
}

fn usize_to_i32_for_cuda(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        CalyxError::assay_insufficient_samples(format!(
            "{name} exceeds CUDA i32 index range: {value}"
        ))
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LogisticCudaInputs<'a> {
    pub(super) samples: &'a [f32],
    pub(super) labels: &'a [i32],
    pub(super) rows: usize,
    pub(super) dim: usize,
    pub(super) train_offsets: &'a [i32],
    pub(super) train_indices: &'a [i32],
    pub(super) test_offsets: &'a [i32],
    pub(super) test_indices: &'a [i32],
}

#[cfg(feature = "cuda")]
pub(super) fn logistic_summaries_cuda_strict_impl(
    input: LogisticCudaInputs<'_>,
) -> Result<calyx_forge::CudaLogisticSummaries> {
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("logistic probe", err))?;
    calyx_forge::logistic_summaries_host(
        backend.context(),
        calyx_forge::CudaLogisticDataset {
            samples: input.samples,
            labels: input.labels,
            rows: input.rows,
            dim: input.dim,
        },
        calyx_forge::CudaLogisticSplits {
            train_offsets: input.train_offsets,
            train_indices: input.train_indices,
            test_offsets: input.test_offsets,
            test_indices: input.test_indices,
        },
        calyx_forge::CudaLogisticConfig {
            steps: LOGISTIC_STEPS,
            learning_rate: LOGISTIC_LR,
            l2_penalty: LOGISTIC_L2,
        },
    )
    .map_err(|err| crate::cuda_strict::forge_to_calyx("logistic probe", err))
}

#[cfg(not(feature = "cuda"))]
pub(super) fn logistic_summaries_cuda_strict_impl(
    input: LogisticCudaInputs<'_>,
) -> Result<UnavailableCudaLogisticSummaries> {
    Err(cuda_unavailable(&format!(
        "logistic probe (rows={}, dim={}, sample_values={}, labels={}, train_offsets={}, train_indices={}, test_offsets={}, test_indices={})",
        input.rows,
        input.dim,
        input.samples.len(),
        input.labels.len(),
        input.train_offsets.len(),
        input.train_indices.len(),
        input.test_offsets.len(),
        input.test_indices.len()
    )))
}

#[cfg(not(feature = "cuda"))]
pub(super) struct UnavailableCudaLogisticSummaries {
    pub(super) bits: Vec<f32>,
    pub(super) accuracy: Vec<f32>,
}
