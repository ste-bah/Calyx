#![allow(clippy::too_many_arguments)]

use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::cuda::kernels::{QUANT_CUBIN, QUANT_PTX};
use crate::{CudaContext, ForgeError, Result};

const BLOCK: u32 = 256;

pub(super) fn rotate_fwht(
    ctx: &CudaContext,
    input: &CudaSlice<f32>,
    diagonal: &CudaSlice<f32>,
    input_dim: usize,
    rot_width: usize,
    rows: usize,
    output: &mut CudaSlice<f32>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    rotate_fwht_with_stride(
        ctx, input, diagonal, 0, input_dim, rot_width, rows, output, bad,
    )
}

pub(super) fn rotate_fwht_rows(
    ctx: &CudaContext,
    input: &CudaSlice<f32>,
    diagonal: &CudaSlice<f32>,
    input_dim: usize,
    rot_width: usize,
    rows: usize,
    output: &mut CudaSlice<f32>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    rotate_fwht_with_stride(
        ctx, input, diagonal, rot_width, input_dim, rot_width, rows, output, bad,
    )
}

#[allow(clippy::too_many_arguments)]
fn rotate_fwht_with_stride(
    ctx: &CudaContext,
    input: &CudaSlice<f32>,
    diagonal: &CudaSlice<f32>,
    diagonal_stride: usize,
    input_dim: usize,
    rot_width: usize,
    rows: usize,
    output: &mut CudaSlice<f32>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let diagonal_stride = as_i32(diagonal_stride, "diagonal stride")?;
    let input_dim = as_i32(input_dim, "input dim")?;
    let rot_width_i32 = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let shared = u32::try_from(rot_width.checked_mul(4).ok_or_else(shape_overflow)?)
        .map_err(|_| shape("CUDA TurboQuant rotation shared-memory request exceeds u32"))?;
    let config = LaunchConfig {
        grid_dim: (as_u32(rows, "row count")?, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: shared,
    };
    let function = function(ctx, "quant.tq_rotate_fwht", "tq_rotate_fwht_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(input)
            .arg(diagonal)
            .arg(&diagonal_stride)
            .arg(&input_dim)
            .arg(&rot_width_i32)
            .arg(&rows_i32)
            .arg(output)
            .arg(bad)
            .launch(config)
    }
    .map_err(|error| device(ctx, format!("TurboQuant FWHT launch failed: {error}")))?;
    sync(ctx, "TurboQuant FWHT")
}

pub(super) fn quantize_rows(
    ctx: &CudaContext,
    rotated: &CudaSlice<f32>,
    thresholds: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    rot_width: usize,
    rows: usize,
    level: i32,
    scales: &mut CudaSlice<f32>,
    codes: &mut CudaSlice<u8>,
    decoded: &mut CudaSlice<f32>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let rot_width = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let config = row_blocks(rows)?;
    let function = function(ctx, "quant.tq_quantize_rows", "tq_quantize_rows_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(rotated)
            .arg(thresholds)
            .arg(centroids)
            .arg(&rot_width)
            .arg(&rows_i32)
            .arg(&level)
            .arg(scales)
            .arg(codes)
            .arg(decoded)
            .arg(bad)
            .launch(config)
    }
    .map_err(|error| device(ctx, format!("TurboQuant scalar launch failed: {error}")))?;
    sync(ctx, "TurboQuant scalar quantize")
}

pub(super) fn pack_scalar(
    ctx: &CudaContext,
    codes: &CudaSlice<u8>,
    rot_width: usize,
    rows: usize,
    level: i32,
    encoded_stride: usize,
    encoded: &mut CudaSlice<u8>,
) -> Result<()> {
    let rot_width = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let encoded_stride = as_i32(encoded_stride, "encoded stride")?;
    let config = flat_threads(rows)?;
    let function = function(ctx, "quant.tq_pack_scalar", "tq_pack_scalar_v4")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(codes)
            .arg(&rot_width)
            .arg(&rows_i32)
            .arg(&level)
            .arg(&encoded_stride)
            .arg(encoded)
            .launch(config)
    }
    .map_err(|error| device(ctx, format!("TurboQuant scalar pack failed: {error}")))?;
    sync(ctx, "TurboQuant scalar pack")
}

pub(super) fn residual_rows(
    ctx: &CudaContext,
    rotated: &CudaSlice<f32>,
    decoded: &CudaSlice<f32>,
    rot_width: usize,
    rows: usize,
    residual: &mut CudaSlice<f32>,
    residual_norms: &mut CudaSlice<f32>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let rot_width = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let function = function(ctx, "quant.tq_residual_rows", "tq_residual_rows_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(rotated)
            .arg(decoded)
            .arg(&rot_width)
            .arg(&rows_i32)
            .arg(residual)
            .arg(residual_norms)
            .arg(bad)
            .launch(row_blocks(rows)?)
    }
    .map_err(|error| device(ctx, format!("TurboQuant residual launch failed: {error}")))?;
    sync(ctx, "TurboQuant residual")
}

pub(super) fn pack_qjl(
    ctx: &CudaContext,
    rotated: &CudaSlice<f32>,
    residual_norms: &CudaSlice<f32>,
    seed: &CudaSlice<u8>,
    rot_width: usize,
    rows: usize,
    scalar_len: usize,
    encoded_stride: usize,
    signs: &mut CudaSlice<u8>,
    encoded: &mut CudaSlice<u8>,
) -> Result<()> {
    pack_qjl_with_stride(
        ctx,
        rotated,
        residual_norms,
        seed,
        0,
        rot_width,
        rows,
        scalar_len,
        encoded_stride,
        signs,
        encoded,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pack_qjl_rows(
    ctx: &CudaContext,
    rotated: &CudaSlice<f32>,
    residual_norms: &CudaSlice<f32>,
    seed: &CudaSlice<u8>,
    rot_width: usize,
    rows: usize,
    scalar_len: usize,
    encoded_stride: usize,
    signs: &mut CudaSlice<u8>,
    encoded: &mut CudaSlice<u8>,
) -> Result<()> {
    pack_qjl_with_stride(
        ctx,
        rotated,
        residual_norms,
        seed,
        32,
        rot_width,
        rows,
        scalar_len,
        encoded_stride,
        signs,
        encoded,
    )
}

#[allow(clippy::too_many_arguments)]
fn pack_qjl_with_stride(
    ctx: &CudaContext,
    rotated: &CudaSlice<f32>,
    residual_norms: &CudaSlice<f32>,
    seed: &CudaSlice<u8>,
    seed_stride: usize,
    rot_width: usize,
    rows: usize,
    scalar_len: usize,
    encoded_stride: usize,
    signs: &mut CudaSlice<u8>,
    encoded: &mut CudaSlice<u8>,
) -> Result<()> {
    let seed_stride = as_i32(seed_stride, "seed stride")?;
    let rot_width = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let scalar_len = as_i32(scalar_len, "scalar bytes")?;
    let encoded_stride = as_i32(encoded_stride, "encoded stride")?;
    let function = function(ctx, "quant.tq_pack_qjl", "tq_pack_qjl_v2")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(rotated)
            .arg(residual_norms)
            .arg(seed)
            .arg(&seed_stride)
            .arg(&rot_width)
            .arg(&rows_i32)
            .arg(&scalar_len)
            .arg(&encoded_stride)
            .arg(signs)
            .arg(encoded)
            .launch(flat_threads(rows)?)
    }
    .map_err(|error| device(ctx, format!("TurboQuant QJL pack failed: {error}")))?;
    sync(ctx, "TurboQuant QJL pack")
}

pub(super) fn decode(
    ctx: &CudaContext,
    codes: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    diagonal: &CudaSlice<f32>,
    dim: usize,
    rot_width: usize,
    rows: usize,
    level: i32,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    let dim = as_i32(dim, "dimension")?;
    let rot_width_i32 = as_i32(rot_width, "rotation width")?;
    let rows_i32 = as_i32(rows, "row count")?;
    let shared = u32::try_from(rot_width.checked_mul(4).ok_or_else(shape_overflow)?)
        .map_err(|_| shape("CUDA TurboQuant decode shared-memory request exceeds u32"))?;
    let config = LaunchConfig {
        grid_dim: (as_u32(rows, "row count")?, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: shared,
    };
    let function = function(ctx, "quant.tq_decode", "tq_decode_inverse_fwht_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(codes)
            .arg(scales)
            .arg(centroids)
            .arg(diagonal)
            .arg(&dim)
            .arg(&rot_width_i32)
            .arg(&rows_i32)
            .arg(&level)
            .arg(output)
            .launch(config)
    }
    .map_err(|error| device(ctx, format!("TurboQuant decode failed: {error}")))?;
    sync(ctx, "TurboQuant decode")
}

pub(super) fn score(
    ctx: &CudaContext,
    query_codes: &CudaSlice<u8>,
    query_signs: &CudaSlice<u8>,
    query_scales: &CudaSlice<f32>,
    query_norms: &CudaSlice<f32>,
    candidate_codes: &CudaSlice<u8>,
    candidate_signs: &CudaSlice<u8>,
    candidate_scales: &CudaSlice<f32>,
    candidate_norms: &CudaSlice<f32>,
    centroids: &CudaSlice<f32>,
    rot_width: usize,
    candidates: usize,
    level: i32,
    scores: &mut CudaSlice<f32>,
) -> Result<()> {
    let rot_width = as_i32(rot_width, "rotation width")?;
    let candidates_i32 = as_i32(candidates, "candidate count")?;
    let function = function(ctx, "quant.tq_score", "tq_score_prepared_v4")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(query_codes)
            .arg(query_signs)
            .arg(query_scales)
            .arg(query_norms)
            .arg(candidate_codes)
            .arg(candidate_signs)
            .arg(candidate_scales)
            .arg(candidate_norms)
            .arg(centroids)
            .arg(&rot_width)
            .arg(&candidates_i32)
            .arg(&level)
            .arg(scores)
            .launch(flat_threads(candidates)?)
    }
    .map_err(|error| device(ctx, format!("TurboQuant score launch failed: {error}")))?;
    sync(ctx, "TurboQuant score")
}

fn function(
    ctx: &CudaContext,
    cache_key: &'static str,
    name: &'static str,
) -> Result<Arc<cudarc::driver::CudaFunction>> {
    let module = quant_module(ctx)?;
    ctx.cached_function(&module, cache_key, name)
        .map_err(|error| device(ctx, format!("load {name} failed: {error}")))
}

fn quant_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.quant_module_cache().get() {
        return Ok(module.clone());
    }
    let module =
        match ctx
            .inner()
            .load_module(Ptx::from_binary(QUANT_CUBIN.to_vec()))
        {
            Ok(module) => module,
            Err(cubin_error) => {
                let ptx = str::from_utf8(QUANT_PTX)
                    .map_err(|error| device(ctx, format!("quant PTX is not UTF-8: {error}")))?;
                ctx.inner().load_module(Ptx::from_src(ptx)).map_err(|error| {
                device(
                    ctx,
                    format!(
                        "quant CUBIN load failed: {cubin_error}; PTX fallback failed: {error}"
                    ),
                )
            })?
            }
        };
    let _ = ctx.quant_module_cache().set(module.clone());
    Ok(module)
}

fn row_blocks(rows: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (as_u32(rows, "row count")?, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn flat_threads(items: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (as_u32(items.div_ceil(BLOCK as usize), "grid blocks")?, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn sync(ctx: &CudaContext, operation: &str) -> Result<()> {
    ctx.inner()
        .default_stream()
        .synchronize()
        .map_err(|error| device(ctx, format!("{operation} synchronization failed: {error}")))
}

fn as_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| shape(format!("{name} exceeds i32")))
}

fn as_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| shape(format!("{name} exceeds u32")))
}

fn shape_overflow() -> ForgeError {
    shape("CUDA TurboQuant allocation shape overflow")
}

fn shape(detail: impl Into<String>) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![1],
        got: vec![0],
        remediation: detail.into(),
    }
}

fn device(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation: "Use the embedded sm_120 quant kernels on an available CUDA device"
            .to_string(),
    }
}
