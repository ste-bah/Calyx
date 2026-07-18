use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{
    CudaPostprocessPooling, FLAG_EMPTY_MASK, FLAG_INVALID_INDEX, FLAG_NONFINITE, FLAG_ZERO_NORM,
    POSTPROCESS_THREADS, REMEDIATION,
};
use crate::cuda::distance::distance_module;
use crate::cuda::validate::DeviceRange;
use crate::{CudaContext, ForgeError, Result};

pub(super) fn launch_copy_dense(
    ctx: &CudaContext,
    values_ptr: u64,
    batch: usize,
    dim: usize,
    out: &mut CudaSlice<f32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let total = checked_product(&[batch, dim], "dense copy total")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(total)?, 1, 1),
        block_dim: (POSTPROCESS_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = postprocess_function(ctx, "copy_dense_external_f32")?;
    let stream = ctx.inner().default_stream();
    let batch_i32 = to_i32(batch, "dense batch")?;
    let dim_i32 = to_i32(dim, "dense dim")?;
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_ptr)
            .arg(&batch_i32)
            .arg(&dim_i32)
            .arg(out)
            .arg(flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("dense copy kernel launch failed: {err}")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_pool_tokens(
    ctx: &CudaContext,
    values_ptr: u64,
    mask: &CudaSlice<i64>,
    batch: usize,
    seq: usize,
    dim: usize,
    pooling: CudaPostprocessPooling,
    out: &mut CudaSlice<f32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (to_u32(batch, "dense token batch")?, 1, 1),
        block_dim: (POSTPROCESS_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = postprocess_function(ctx, "pool_tokens_external_f32")?;
    let stream = ctx.inner().default_stream();
    let batch_i32 = to_i32(batch, "dense token batch")?;
    let seq_i32 = to_i32(seq, "dense token seq")?;
    let dim_i32 = to_i32(dim, "dense token dim")?;
    let policy_i32 = pooling.kernel_code();
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_ptr)
            .arg(mask)
            .arg(&batch_i32)
            .arg(&seq_i32)
            .arg(&dim_i32)
            .arg(&policy_i32)
            .arg(out)
            .arg(flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("token pooling kernel launch failed: {err}")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_sparse_positive(
    ctx: &CudaContext,
    values_ptr: u64,
    batch: usize,
    dim: usize,
    indices: &mut CudaSlice<u32>,
    values: &mut CudaSlice<f32>,
    counts: &mut CudaSlice<i32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(dim)?, to_u32(batch, "sparse compact batch")?, 1),
        block_dim: (POSTPROCESS_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = postprocess_function(ctx, "sparse_positive_external_f32")?;
    let stream = ctx.inner().default_stream();
    let batch_i32 = to_i32(batch, "sparse compact batch")?;
    let dim_i32 = to_i32(dim, "sparse compact dim")?;
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_ptr)
            .arg(&batch_i32)
            .arg(&dim_i32)
            .arg(indices)
            .arg(values)
            .arg(counts)
            .arg(flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(ctx, format!("sparse compact kernel launch failed: {err}"))
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_colbert_compact(
    ctx: &CudaContext,
    values_ptr: u64,
    mask: &CudaSlice<i64>,
    batch: usize,
    seq: usize,
    dim: usize,
    normalize: bool,
    values: &mut CudaSlice<f32>,
    counts: &mut CudaSlice<i32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (
            to_u32(seq, "ColBERT compact seq")?,
            to_u32(batch, "ColBERT compact batch")?,
            1,
        ),
        block_dim: (POSTPROCESS_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = postprocess_function(ctx, "colbert_compact_external_f32")?;
    let stream = ctx.inner().default_stream();
    let batch_i32 = to_i32(batch, "ColBERT compact batch")?;
    let seq_i32 = to_i32(seq, "ColBERT compact seq")?;
    let dim_i32 = to_i32(dim, "ColBERT compact dim")?;
    let normalize_i32 = i32::from(normalize);
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_ptr)
            .arg(mask)
            .arg(&batch_i32)
            .arg(&seq_i32)
            .arg(&dim_i32)
            .arg(&normalize_i32)
            .arg(values)
            .arg(counts)
            .arg(flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(ctx, format!("ColBERT compact kernel launch failed: {err}"))
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_bgem3_sparse_compact(
    ctx: &CudaContext,
    values_ptr: u64,
    token_ids: &CudaSlice<i64>,
    mask: &CudaSlice<i64>,
    batch: usize,
    seq: usize,
    vocab_dim: usize,
    indices: &mut CudaSlice<u32>,
    values: &mut CudaSlice<f32>,
    counts: &mut CudaSlice<i32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (to_u32(batch, "BGE-M3 sparse batch")?, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = postprocess_function(ctx, "bgem3_sparse_compact_external_f32")?;
    let batch_i32 = to_i32(batch, "BGE-M3 sparse batch")?;
    let seq_i32 = to_i32(seq, "BGE-M3 sparse seq")?;
    let vocab_i32 = to_i32(vocab_dim, "BGE-M3 sparse vocab")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_ptr)
            .arg(token_ids)
            .arg(mask)
            .arg(&batch_i32)
            .arg(&seq_i32)
            .arg(&vocab_i32)
            .arg(indices)
            .arg(values)
            .arg(counts)
            .arg(flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(
            ctx,
            format!("BGE-M3 sparse compact kernel launch failed: {err}"),
        )
    })?;
    Ok(())
}

fn postprocess_function(
    ctx: &CudaContext,
    name: &'static str,
) -> Result<std::sync::Arc<cudarc::driver::CudaFunction>> {
    let module = distance_module(ctx)?;
    ctx.cached_function(&module, postprocess_cache_key(name), name)
        .map_err(|err| device_unavailable(ctx, format!("load {name} failed: {err}")))
}

fn postprocess_cache_key(name: &'static str) -> &'static str {
    match name {
        "copy_dense_external_f32" => "distance.copy_dense_external_f32",
        "pool_tokens_external_f32" => "distance.pool_tokens_external_f32",
        "sparse_positive_external_f32" => "distance.sparse_positive_external_f32",
        "colbert_compact_external_f32" => "distance.colbert_compact_external_f32",
        "bgem3_sparse_compact_external_f32" => "distance.bgem3_sparse_compact_external_f32",
        _ => name,
    }
}

pub(super) fn upload_mask(
    ctx: &CudaContext,
    mask: &[i64],
    op: &'static str,
) -> Result<CudaSlice<i64>> {
    ctx.inner()
        .default_stream()
        .clone_htod(mask)
        .map_err(|err| device_unavailable(ctx, format!("{op} upload failed: {err}")))
}

pub(super) fn upload_i64(
    ctx: &CudaContext,
    values: &[i64],
    op: &'static str,
) -> Result<CudaSlice<i64>> {
    ctx.inner()
        .default_stream()
        .clone_htod(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} upload failed: {err}")))
}

pub(super) fn alloc_f32(
    ctx: &CudaContext,
    len: usize,
    label: &'static str,
) -> Result<CudaSlice<f32>> {
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{label} allocation failed: {err}")))
}

pub(super) fn alloc_i32(
    ctx: &CudaContext,
    len: usize,
    label: &'static str,
) -> Result<CudaSlice<i32>> {
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{label} allocation failed: {err}")))
}

pub(super) fn alloc_u32(
    ctx: &CudaContext,
    len: usize,
    label: &'static str,
) -> Result<CudaSlice<u32>> {
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{label} allocation failed: {err}")))
}

pub(super) fn read_flags(ctx: &CudaContext, flags: &CudaSlice<u32>) -> Result<u32> {
    let host = ctx
        .inner()
        .default_stream()
        .clone_dtoh(flags)
        .map_err(|err| device_unavailable(ctx, format!("postprocess flags read failed: {err}")))?;
    Ok(host.first().copied().unwrap_or(0))
}

pub(super) fn read_counts(
    ctx: &CudaContext,
    op: &'static str,
    counts: &CudaSlice<i32>,
    max_count: usize,
) -> Result<Vec<usize>> {
    let host = ctx
        .inner()
        .default_stream()
        .clone_dtoh(counts)
        .map_err(|err| device_unavailable(ctx, format!("{op} read failed: {err}")))?;
    host.into_iter()
        .map(|value| {
            let count = usize::try_from(value).map_err(|_| ForgeError::NumericalInvariant {
                op: op.to_string(),
                detail: format!("negative compact count {value}"),
                remediation: REMEDIATION.to_string(),
            })?;
            if count > max_count {
                return Err(ForgeError::NumericalInvariant {
                    op: op.to_string(),
                    detail: format!("compact count {count} exceeds row limit {max_count}"),
                    remediation: REMEDIATION.to_string(),
                });
            }
            Ok(count)
        })
        .collect()
}

pub(super) fn compact_ranges(counts: &[usize], row_stride: usize) -> Result<Vec<DeviceRange>> {
    compact_value_ranges(counts, row_stride, 1)
}

pub(super) fn compact_value_ranges(
    counts: &[usize],
    row_stride: usize,
    value_width: usize,
) -> Result<Vec<DeviceRange>> {
    counts
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .map(|(row, count)| {
            Ok(DeviceRange {
                offset: checked_product(&[row, row_stride], "compact range offset")?,
                len: checked_product(&[count, value_width], "compact range length")?,
            })
        })
        .collect()
}

pub(super) fn check_mask(
    mask: &[i64],
    batch: usize,
    seq: usize,
    label: &'static str,
) -> Result<()> {
    let expected = checked_product(&[batch, seq], label)?;
    if mask.len() != expected {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![expected],
            got: vec![mask.len()],
            remediation: REMEDIATION.to_string(),
        });
    }
    Ok(())
}

pub(super) fn decode_flags(op: &'static str, flags: u32) -> Result<()> {
    if flags == 0 {
        return Ok(());
    }
    let detail = if flags & FLAG_NONFINITE != 0 {
        "postprocess output contained NaN or Inf"
    } else if flags & FLAG_EMPTY_MASK != 0 {
        "attention mask selected no tokens"
    } else if flags & FLAG_INVALID_INDEX != 0 {
        "postprocess saw an invalid token/vocabulary index or exceeded compact capacity"
    } else if flags & FLAG_ZERO_NORM != 0 {
        "postprocess cannot normalize a zero-norm token vector"
    } else {
        "postprocess kernel reported an unknown invariant failure"
    };
    Err(ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail: detail.to_string(),
        remediation: REMEDIATION.to_string(),
    })
}

pub(super) fn checked_product(values: &[usize], label: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| ForgeError::ShapeMismatch {
                expected: vec![usize::MAX],
                got: values.to_vec(),
                remediation: format!("{label} overflowed usize; lower batch, sequence, or dim"),
            })
    })
}

fn grid_blocks(len: usize) -> Result<u32> {
    let blocks = len.div_ceil(POSTPROCESS_THREADS as usize).max(1);
    to_u32(blocks, "postprocess grid blocks")
}

fn to_u32(value: usize, label: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![value],
        remediation: format!("{label} exceeds CUDA grid limit"),
    })
}

fn to_i32(value: usize, label: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![i32::MAX as usize],
        got: vec![value],
        remediation: format!("{label} exceeds CUDA kernel i32 limit"),
    })
}

pub(super) fn device_unavailable(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}:{}", ctx.device_idx(), ctx.name()),
        detail,
        remediation: REMEDIATION.to_string(),
    }
}
