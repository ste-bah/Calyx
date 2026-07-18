use calyx_core::{CalyxError, Result, SlotVector, SparseEntry};

use super::super::batch::TokenBatch;
use super::{CustomOutput, output_tensor, positive_usize};
use crate::frozen::NormPolicy;
use crate::runtime::onnx::PoolingPolicy;
use crate::runtime::onnx::device_postprocess::{
    device_tensor, forge_error, normalize_on_device, pooling_to_cuda, tensor_data_ptr, tensor_shape,
};

pub(in crate::runtime::onnx::custom) fn vectors_from_device_output(
    outputs: &ort::session::SessionOutputs<'_>,
    batch: &TokenBatch,
    output: CustomOutput,
    ctx: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let tensor = device_tensor(output_tensor(outputs)?, "custom ONNX")?;
    let shape = tensor_shape(&tensor, "custom ONNX")?;
    let ptr = tensor_data_ptr(&tensor, &shape, "custom ONNX")?;
    match output {
        CustomOutput::Dense {
            dim,
            pooling,
            norm_policy,
        } => dense_device_output_batch(&shape, ptr, batch, pooling, dim, norm_policy, ctx),
        CustomOutput::Sparse { dim } => sparse_device_output_batch(&shape, ptr, batch, dim, ctx),
    }
}

#[allow(clippy::too_many_arguments)]
fn dense_device_output_batch(
    shape: &[i64],
    values_ptr: u64,
    batch: &TokenBatch,
    policy: PoolingPolicy,
    dim: u32,
    norm_policy: NormPolicy,
    ctx: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let dim_usize = dim as usize;
    let normalize = normalize_on_device(norm_policy);
    let rows = match shape {
        [actual_batch, actual_dim]
            if positive_usize(*actual_batch) == Some(batch.batch)
                && positive_usize(*actual_dim) == Some(dim_usize) =>
        {
            calyx_forge::cuda::dense_2d_from_external_f32(
                ctx,
                values_ptr,
                batch.batch,
                dim_usize,
                normalize,
            )
            .map_err(forge_error)?
        }
        [actual_batch, seq, actual_dim]
            if positive_usize(*actual_batch) == Some(batch.batch)
                && positive_usize(*seq) == Some(batch.seq)
                && positive_usize(*actual_dim) == Some(dim_usize) =>
        {
            calyx_forge::cuda::dense_tokens_from_external_f32(
                ctx,
                calyx_forge::cuda::CudaDenseTokenPostprocess {
                    values_ptr,
                    mask: &batch.mask,
                    batch: batch.batch,
                    seq: batch.seq,
                    dim: dim_usize,
                    pooling: pooling_to_cuda(policy),
                    normalize,
                },
            )
            .map_err(forge_error)?
        }
        _ => {
            return Err(CalyxError::lens_dim_mismatch(format!(
                "custom ONNX device output shape {shape:?} is incompatible with batch={} seq={} dim={dim_usize}",
                batch.batch, batch.seq
            )));
        }
    };
    rows.chunks_exact(dim_usize)
        .map(|row| {
            Ok(SlotVector::Dense {
                dim,
                data: row.to_vec(),
            })
        })
        .collect()
}

fn sparse_device_output_batch(
    shape: &[i64],
    values_ptr: u64,
    batch: &TokenBatch,
    dim: u32,
    ctx: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let dim_usize = dim as usize;
    match shape {
        [actual_batch, actual_dim]
            if positive_usize(*actual_batch) == Some(batch.batch)
                && positive_usize(*actual_dim) == Some(dim_usize) => {}
        _ => {
            return Err(CalyxError::lens_dim_mismatch(format!(
                "custom ONNX sparse device output shape {shape:?} must be [batch={}, dim={dim_usize}]",
                batch.batch
            )));
        }
    }
    let rows = calyx_forge::cuda::sparse_positive_from_external_f32(
        ctx,
        values_ptr,
        batch.batch,
        dim_usize,
    )
    .map_err(forge_error)?;
    rows.rows
        .into_iter()
        .map(|row| {
            Ok(SlotVector::Sparse {
                dim,
                entries: row
                    .into_iter()
                    .map(|(idx, val)| SparseEntry { idx, val })
                    .collect(),
            })
        })
        .collect()
}
