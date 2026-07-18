use crate::cuda::distance::normalize_rows_gpu;
use crate::cuda::validate::{check_device_f32, check_finite_ranges, read_checked_device_f32};
use crate::{CudaContext, Result};

mod support;

use support::{
    alloc_f32, alloc_i32, alloc_u32, check_mask, checked_product, compact_ranges,
    compact_value_ranges, decode_flags, device_unavailable, launch_bgem3_sparse_compact,
    launch_colbert_compact, launch_copy_dense, launch_pool_tokens, launch_sparse_positive,
    read_counts, read_flags, upload_i64, upload_mask,
};

const POSTPROCESS_THREADS: u32 = 256;
const FLAG_NONFINITE: u32 = 1;
const FLAG_EMPTY_MASK: u32 = 1 << 1;
const FLAG_INVALID_INDEX: u32 = 1 << 2;
const FLAG_ZERO_NORM: u32 = 1 << 3;
const REMEDIATION: &str =
    "verify the model output shape, attention mask, and CUDA output binding before retrying";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaPostprocessPooling {
    Mean,
    Cls,
    LastToken,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDenseTokenPostprocess<'a> {
    pub values_ptr: u64,
    pub mask: &'a [i64],
    pub batch: usize,
    pub seq: usize,
    pub dim: usize,
    pub pooling: CudaPostprocessPooling,
    pub normalize: bool,
}

impl CudaPostprocessPooling {
    fn kernel_code(self) -> i32 {
        match self {
            Self::Mean => 0,
            Self::Cls => 1,
            Self::LastToken => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaSparseRows {
    pub rows: Vec<Vec<(u32, f32)>>,
    pub input_floats: usize,
    pub host_value_floats: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaMultiRows {
    pub rows: Vec<Vec<Vec<f32>>>,
    pub input_floats: usize,
    pub host_value_floats: usize,
}

pub fn dense_2d_from_external_f32(
    ctx: &CudaContext,
    values_ptr: u64,
    batch: usize,
    dim: usize,
    normalize: bool,
) -> Result<Vec<f32>> {
    let total = checked_product(&[batch, dim], "dense 2d total")?;
    let mut out = alloc_f32(ctx, total, "dense postprocess output")?;
    let mut flags = alloc_u32(ctx, 1, "dense postprocess flags")?;
    if total > 0 {
        launch_copy_dense(ctx, values_ptr, batch, dim, &mut out, &mut flags)?;
        decode_flags("cuda_postprocess_dense_copy", read_flags(ctx, &flags)?)?;
        if normalize {
            normalize_rows_gpu(ctx, &mut out, batch, dim)?;
        } else {
            check_device_f32(ctx, "cuda_postprocess_dense_copy", &out, false, REMEDIATION)?;
        }
    }
    read_checked_device_f32(
        ctx,
        "cuda_postprocess_dense_final",
        &out,
        false,
        REMEDIATION,
    )
}

pub fn dense_tokens_from_external_f32(
    ctx: &CudaContext,
    input: CudaDenseTokenPostprocess<'_>,
) -> Result<Vec<f32>> {
    let CudaDenseTokenPostprocess {
        values_ptr,
        mask,
        batch,
        seq,
        dim,
        pooling,
        normalize,
    } = input;
    check_mask(mask, batch, seq, "dense token postprocess mask")?;
    let total = checked_product(&[batch, dim], "dense token output total")?;
    let mut out = alloc_f32(ctx, total, "dense token postprocess output")?;
    let mut flags = alloc_u32(ctx, 1, "dense token postprocess flags")?;
    if total > 0 {
        let mask_dev = upload_mask(ctx, mask, "dense token postprocess mask")?;
        launch_pool_tokens(
            ctx, values_ptr, &mask_dev, batch, seq, dim, pooling, &mut out, &mut flags,
        )?;
        decode_flags("cuda_postprocess_pool_tokens", read_flags(ctx, &flags)?)?;
        if normalize {
            normalize_rows_gpu(ctx, &mut out, batch, dim)?;
        } else {
            check_device_f32(
                ctx,
                "cuda_postprocess_pool_tokens",
                &out,
                false,
                REMEDIATION,
            )?;
        }
    }
    read_checked_device_f32(
        ctx,
        "cuda_postprocess_dense_tokens_final",
        &out,
        false,
        REMEDIATION,
    )
}

pub fn sparse_positive_from_external_f32(
    ctx: &CudaContext,
    values_ptr: u64,
    batch: usize,
    dim: usize,
) -> Result<CudaSparseRows> {
    let input_floats = checked_product(&[batch, dim], "sparse input total")?;
    let mut values = alloc_f32(ctx, input_floats, "sparse compact values")?;
    let mut indices = alloc_u32(ctx, input_floats, "sparse compact indices")?;
    let mut counts = alloc_i32(ctx, batch, "sparse compact counts")?;
    let mut flags = alloc_u32(ctx, 1, "sparse compact flags")?;
    if input_floats > 0 {
        launch_sparse_positive(
            ctx,
            values_ptr,
            batch,
            dim,
            &mut indices,
            &mut values,
            &mut counts,
            &mut flags,
        )?;
    }
    decode_flags("cuda_postprocess_sparse_positive", read_flags(ctx, &flags)?)?;
    let counts = read_counts(ctx, "cuda_postprocess_sparse_counts", &counts, dim)?;
    let ranges = compact_ranges(&counts, dim)?;
    check_finite_ranges(
        ctx,
        "cuda_postprocess_sparse_values",
        &values,
        &ranges,
        REMEDIATION,
    )?;
    let stream = ctx.inner().default_stream();
    let mut rows = Vec::with_capacity(batch);
    let mut host_value_floats = 0usize;
    for (row, count) in counts.iter().copied().enumerate() {
        let base = row * dim;
        let idx = if count == 0 {
            Vec::new()
        } else {
            stream
                .clone_dtoh(&indices.slice(base..base + count))
                .map_err(|err| {
                    device_unavailable(ctx, format!("sparse indices read failed: {err}"))
                })?
        };
        let vals = if count == 0 {
            Vec::new()
        } else {
            stream
                .clone_dtoh(&values.slice(base..base + count))
                .map_err(|err| {
                    device_unavailable(ctx, format!("sparse values read failed: {err}"))
                })?
        };
        host_value_floats += vals.len();
        rows.push(idx.into_iter().zip(vals).collect());
    }
    Ok(CudaSparseRows {
        rows,
        input_floats,
        host_value_floats,
    })
}

pub fn colbert_tokens_from_external_f32(
    ctx: &CudaContext,
    values_ptr: u64,
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Result<CudaMultiRows> {
    colbert_tokens_from_external_f32_inner(ctx, values_ptr, mask, batch, seq, dim, false)
}

/// Compacts BGE-M3 ColBERT rows and L2-normalizes each retained token on the
/// GPU before copying only compact vectors to the host.
pub fn bgem3_colbert_tokens_from_external_f32(
    ctx: &CudaContext,
    values_ptr: u64,
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Result<CudaMultiRows> {
    colbert_tokens_from_external_f32_inner(ctx, values_ptr, mask, batch, seq, dim, true)
}

fn colbert_tokens_from_external_f32_inner(
    ctx: &CudaContext,
    values_ptr: u64,
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
    normalize: bool,
) -> Result<CudaMultiRows> {
    check_mask(mask, batch, seq, "ColBERT postprocess mask")?;
    let input_floats = checked_product(&[batch, seq, dim], "ColBERT input total")?;
    let mut values = alloc_f32(ctx, input_floats, "ColBERT compact values")?;
    let mut counts = alloc_i32(ctx, batch, "ColBERT compact counts")?;
    let mut flags = alloc_u32(ctx, 1, "ColBERT compact flags")?;
    if input_floats > 0 {
        let mask_dev = upload_mask(ctx, mask, "ColBERT postprocess mask")?;
        launch_colbert_compact(
            ctx,
            values_ptr,
            &mask_dev,
            batch,
            seq,
            dim,
            normalize,
            &mut values,
            &mut counts,
            &mut flags,
        )?;
    }
    decode_flags("cuda_postprocess_colbert_compact", read_flags(ctx, &flags)?)?;
    let counts = read_counts(ctx, "cuda_postprocess_colbert_counts", &counts, seq)?;
    let ranges = compact_value_ranges(&counts, seq * dim, dim)?;
    check_finite_ranges(
        ctx,
        "cuda_postprocess_colbert_values",
        &values,
        &ranges,
        REMEDIATION,
    )?;
    let stream = ctx.inner().default_stream();
    let mut rows = Vec::with_capacity(batch);
    let mut host_value_floats = 0usize;
    for (row, count) in counts.iter().copied().enumerate() {
        let base = row * seq * dim;
        let len = count * dim;
        let flat = if len == 0 {
            Vec::new()
        } else {
            stream
                .clone_dtoh(&values.slice(base..base + len))
                .map_err(|err| {
                    device_unavailable(ctx, format!("ColBERT values read failed: {err}"))
                })?
        };
        host_value_floats += flat.len();
        rows.push(flat.chunks_exact(dim).map(|chunk| chunk.to_vec()).collect());
    }
    Ok(CudaMultiRows {
        rows,
        input_floats,
        host_value_floats,
    })
}

/// Converts BGE-M3's `[batch, seq, 1]` lexical weights into sorted sparse
/// vocabulary entries without copying the full token-weight tensor to host.
/// Duplicate token ids are max-reduced exactly like the reference
/// FlagEmbedding/FastEmbed implementation.
pub fn bgem3_sparse_from_external_f32(
    ctx: &CudaContext,
    values_ptr: u64,
    token_ids: &[i64],
    mask: &[i64],
    batch: usize,
    seq: usize,
    vocab_dim: usize,
) -> Result<CudaSparseRows> {
    check_mask(token_ids, batch, seq, "BGE-M3 sparse token ids")?;
    check_mask(mask, batch, seq, "BGE-M3 sparse mask")?;
    let capacity = checked_product(&[batch, seq], "BGE-M3 sparse compact capacity")?;
    let mut values = alloc_f32(ctx, capacity, "BGE-M3 sparse compact values")?;
    let mut indices = alloc_u32(ctx, capacity, "BGE-M3 sparse compact indices")?;
    let mut counts = alloc_i32(ctx, batch, "BGE-M3 sparse compact counts")?;
    let mut flags = alloc_u32(ctx, 1, "BGE-M3 sparse compact flags")?;
    if capacity > 0 {
        let ids_dev = upload_i64(ctx, token_ids, "BGE-M3 sparse token ids")?;
        let mask_dev = upload_mask(ctx, mask, "BGE-M3 sparse mask")?;
        launch_bgem3_sparse_compact(
            ctx,
            values_ptr,
            &ids_dev,
            &mask_dev,
            batch,
            seq,
            vocab_dim,
            &mut indices,
            &mut values,
            &mut counts,
            &mut flags,
        )?;
    }
    decode_flags("cuda_postprocess_bgem3_sparse", read_flags(ctx, &flags)?)?;
    let counts = read_counts(ctx, "cuda_postprocess_bgem3_sparse_counts", &counts, seq)?;
    let ranges = compact_ranges(&counts, seq)?;
    check_finite_ranges(
        ctx,
        "cuda_postprocess_bgem3_sparse_values",
        &values,
        &ranges,
        REMEDIATION,
    )?;
    let stream = ctx.inner().default_stream();
    let mut rows = Vec::with_capacity(batch);
    let mut host_value_floats = 0usize;
    for (row, count) in counts.iter().copied().enumerate() {
        let base = row * seq;
        let idx = if count == 0 {
            Vec::new()
        } else {
            stream
                .clone_dtoh(&indices.slice(base..base + count))
                .map_err(|err| {
                    device_unavailable(ctx, format!("BGE-M3 sparse indices read failed: {err}"))
                })?
        };
        let vals = if count == 0 {
            Vec::new()
        } else {
            stream
                .clone_dtoh(&values.slice(base..base + count))
                .map_err(|err| {
                    device_unavailable(ctx, format!("BGE-M3 sparse values read failed: {err}"))
                })?
        };
        host_value_floats += vals.len();
        rows.push(idx.into_iter().zip(vals).collect());
    }
    Ok(CudaSparseRows {
        rows,
        input_floats: capacity,
        host_value_floats,
    })
}
