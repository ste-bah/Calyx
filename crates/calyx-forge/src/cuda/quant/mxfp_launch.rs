#![allow(clippy::too_many_arguments)]

use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::cuda::kernels::{MXFP_QUANT_CUBIN, MXFP_QUANT_PTX};
use crate::{CudaContext, ForgeError, Result};

const BLOCK: u32 = 256;

pub(super) fn encode(
    ctx: &CudaContext,
    input: &CudaSlice<f32>,
    dim: usize,
    rows: usize,
    level: i32,
    encoded: &mut CudaSlice<u8>,
    bad: &mut CudaSlice<i32>,
) -> Result<()> {
    let (dim_i32, rows_i32) = dims(dim, rows)?;
    let (key, name) = match level {
        4 => ("mxfp.encode4", "mq_mxfp4_encode_f32"),
        8 => ("mxfp.encode8", "mq_mxfp8_encode_f32"),
        _ => return Err(shape("MXFP CUDA encode level must be 4 or 8")),
    };
    let function = function(ctx, key, name)?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(input)
            .arg(&dim_i32)
            .arg(&rows_i32)
            .arg(encoded)
            .arg(bad)
            .launch(row_blocks(rows)?)
    }
    .map_err(|error| device(ctx, format!("MXFP{level} encode launch failed: {error}")))?;
    sync(ctx, "MXFP encode")
}

pub(super) fn decode(
    ctx: &CudaContext,
    encoded: &CudaSlice<u8>,
    dim: usize,
    rows: usize,
    level: i32,
    output: &mut CudaSlice<f32>,
) -> Result<()> {
    let (dim_i32, rows_i32) = dims(dim, rows)?;
    let count = dim.checked_mul(rows).ok_or_else(shape_overflow)?;
    let function = function(ctx, "mxfp.decode", "mq_mxfp_decode_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(encoded)
            .arg(&dim_i32)
            .arg(&rows_i32)
            .arg(&level)
            .arg(output)
            .launch(flat_threads(count)?)
    }
    .map_err(|error| device(ctx, format!("MXFP decode launch failed: {error}")))?;
    sync(ctx, "MXFP decode")
}

pub(super) fn score(
    ctx: &CudaContext,
    query: &CudaSlice<u8>,
    query_level: i32,
    candidates: &CudaSlice<u8>,
    candidate_level: i32,
    dim: usize,
    rows: usize,
    scores: &mut CudaSlice<f32>,
) -> Result<()> {
    let (dim_i32, rows_i32) = dims(dim, rows)?;
    if !matches!(query_level, 4 | 8) || !matches!(candidate_level, 4 | 8) {
        return Err(shape("MXFP CUDA score levels must be 4 or 8"));
    }
    let function = function(ctx, "mxfp.score", "mq_mxfp_score")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(query)
            .arg(&query_level)
            .arg(candidates)
            .arg(&candidate_level)
            .arg(&dim_i32)
            .arg(&rows_i32)
            .arg(scores)
            .launch(flat_threads(rows)?)
    }
    .map_err(|error| device(ctx, format!("MXFP score launch failed: {error}")))?;
    sync(ctx, "MXFP score")
}

fn function(
    ctx: &CudaContext,
    cache_key: &'static str,
    name: &'static str,
) -> Result<Arc<cudarc::driver::CudaFunction>> {
    let module = module(ctx)?;
    ctx.cached_function(&module, cache_key, name)
        .map_err(|error| device(ctx, format!("load {name} failed: {error}")))
}

fn module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.mxfp_quant_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(MXFP_QUANT_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_error) => {
            let ptx = str::from_utf8(MXFP_QUANT_PTX)
                .map_err(|error| device(ctx, format!("MXFP quant PTX is not UTF-8: {error}")))?;
            ctx.inner().load_module(Ptx::from_src(ptx)).map_err(|error| {
                device(
                    ctx,
                    format!(
                        "MXFP quant CUBIN load failed: {cubin_error}; PTX fallback failed: {error}"
                    ),
                )
            })?
        }
    };
    let _ = ctx.mxfp_quant_module_cache().set(module.clone());
    Ok(module)
}

fn dims(dim: usize, rows: usize) -> Result<(i32, i32)> {
    let count = dim.checked_mul(rows).ok_or_else(shape_overflow)?;
    if count > i32::MAX as usize {
        return Err(shape("MXFP CUDA element count exceeds i32 indexing"));
    }
    Ok((as_i32(dim, "dimension")?, as_i32(rows, "row count")?))
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
    shape("MXFP CUDA allocation shape overflow")
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
        remediation: "Use the embedded sm_120 MXFP quant kernels on available CUDA".to_string(),
    }
}
