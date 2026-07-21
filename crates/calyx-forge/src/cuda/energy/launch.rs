use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::buffers::{EnergyBuffers, EnergyShape, device};
use crate::cuda::kernels::{ENERGY_CUBIN, ENERGY_PTX};
use crate::{CudaContext, Result};

const THREADS: u32 = 256;

pub(super) fn run_descent(
    ctx: &CudaContext,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
    beta: f32,
    max_steps: usize,
    eps: f32,
) -> Result<usize> {
    let module = energy_module(ctx)?;
    let functions = EnergyFunctions::load(ctx, &module)?;
    let require_nonzero = i32::from(beta > 0.0);
    launch_member_norms(ctx, &functions, buffers, shape, require_nonzero)?;
    let mut launches = 1_usize;
    if beta > 0.0 {
        launch_query_norm(ctx, &functions, buffers, shape)?;
        launch_cosine(ctx, &functions, buffers, shape, beta)?;
        launches += 2;
    }
    launch_softmax(ctx, &functions, buffers, shape, beta, 0, eps)?;
    launches += 1;
    for step in 1..=max_steps {
        launch_centroid_partials(ctx, &functions, buffers, shape)?;
        launch_centroid_finalize(ctx, &functions, buffers, shape)?;
        launch_normalize(ctx, &functions, buffers, shape)?;
        launches += 3;
        if beta > 0.0 {
            launch_query_norm(ctx, &functions, buffers, shape)?;
            launch_cosine(ctx, &functions, buffers, shape, beta)?;
            launches += 2;
        }
        launch_softmax(ctx, &functions, buffers, shape, beta, step, eps)?;
        launches += 1;
    }
    ctx.inner()
        .default_stream()
        .synchronize()
        .map_err(|error| {
            device(
                ctx,
                format!("CUDA energy descent synchronization failed: {error}"),
            )
        })?;
    Ok(launches)
}

struct EnergyFunctions {
    member_norms: Arc<CudaFunction>,
    query_norm: Arc<CudaFunction>,
    cosine: Arc<CudaFunction>,
    softmax: Arc<CudaFunction>,
    centroid_partials: Arc<CudaFunction>,
    centroid_finalize: Arc<CudaFunction>,
    normalize: Arc<CudaFunction>,
}

impl EnergyFunctions {
    fn load(ctx: &CudaContext, module: &Arc<CudaModule>) -> Result<Self> {
        Ok(Self {
            member_norms: function(ctx, module, "energy_member_inv_norms_f32")?,
            query_norm: function(ctx, module, "energy_query_inv_norm_f32")?,
            cosine: function(ctx, module, "energy_cosine_scaled_f32")?,
            softmax: function(ctx, module, "energy_softmax_state_f32")?,
            centroid_partials: function(ctx, module, "energy_centroid_partials_f32")?,
            centroid_finalize: function(ctx, module, "energy_centroid_finalize_f32")?,
            normalize: function(ctx, module, "energy_normalize_query_f32")?,
        })
    }
}

fn launch_member_norms(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
    require_nonzero: i32,
) -> Result<()> {
    let members = to_i32(shape.members, "members")?;
    let dim = to_i32(shape.dim, "dim")?;
    let cfg = config(to_u32(shape.members, "members")?, THREADS);
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.member_norms.as_ref());
    unsafe {
        launch
            .arg(&buffers.members)
            .arg(&members)
            .arg(&dim)
            .arg(&require_nonzero)
            .arg(&mut buffers.member_inverse_norms)
            .arg(&mut buffers.status)
            .launch(cfg)
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA energy member norm launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn launch_query_norm(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
) -> Result<()> {
    let dim = to_i32(shape.dim, "dim")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.query_norm.as_ref());
    unsafe {
        launch
            .arg(&buffers.query)
            .arg(&dim)
            .arg(&mut buffers.query_inverse_norm)
            .arg(&buffers.control)
            .arg(&mut buffers.status)
            .launch(config(1, THREADS))
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA energy query norm launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn launch_cosine(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
    beta: f32,
) -> Result<()> {
    let members = to_i32(shape.members, "members")?;
    let dim = to_i32(shape.dim, "dim")?;
    let blocks = to_u32(shape.members.div_ceil(THREADS as usize), "cosine blocks")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.cosine.as_ref());
    unsafe {
        launch
            .arg(&buffers.query)
            .arg(&buffers.members)
            .arg(&buffers.query_inverse_norm)
            .arg(&buffers.member_inverse_norms)
            .arg(&members)
            .arg(&dim)
            .arg(&beta)
            .arg(&mut buffers.scores)
            .arg(&buffers.control)
            .arg(&mut buffers.status)
            .launch(config(blocks, THREADS))
    }
    .map_err(|error| device(ctx, format!("CUDA energy cosine launch failed: {error}")))?;
    Ok(())
}

fn launch_softmax(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
    beta: f32,
    step: usize,
    eps: f32,
) -> Result<()> {
    let members = to_i32(shape.members, "members")?;
    let step = to_i32(step, "step")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.softmax.as_ref());
    unsafe {
        launch
            .arg(&buffers.scores)
            .arg(&members)
            .arg(&beta)
            .arg(&step)
            .arg(&eps)
            .arg(&mut buffers.weights)
            .arg(&mut buffers.metadata)
            .arg(&mut buffers.control)
            .arg(&mut buffers.status)
            .launch(config(1, THREADS))
    }
    .map_err(|error| device(ctx, format!("CUDA energy softmax launch failed: {error}")))?;
    Ok(())
}

fn launch_centroid_partials(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
) -> Result<()> {
    let members = to_i32(shape.members, "members")?;
    let dim = to_i32(shape.dim, "dim")?;
    let cfg = LaunchConfig {
        grid_dim: (
            to_u32(shape.dim.div_ceil(32), "centroid dimension tiles")?,
            to_u32(shape.centroid_tiles, "centroid member tiles")?,
            1,
        ),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.centroid_partials.as_ref());
    unsafe {
        launch
            .arg(&buffers.members)
            .arg(&buffers.weights)
            .arg(&members)
            .arg(&dim)
            .arg(&mut buffers.centroid_partials)
            .arg(&buffers.control)
            .arg(&buffers.status)
            .launch(cfg)
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA energy centroid partial launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn launch_centroid_finalize(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
) -> Result<()> {
    let tiles = to_i32(shape.centroid_tiles, "centroid tiles")?;
    let dim = to_i32(shape.dim, "dim")?;
    let blocks = to_u32(shape.dim.div_ceil(THREADS as usize), "centroid blocks")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.centroid_finalize.as_ref());
    unsafe {
        launch
            .arg(&buffers.centroid_partials)
            .arg(&tiles)
            .arg(&dim)
            .arg(&mut buffers.query)
            .arg(&buffers.control)
            .arg(&mut buffers.status)
            .launch(config(blocks, THREADS))
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA energy centroid finalize launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn launch_normalize(
    ctx: &CudaContext,
    functions: &EnergyFunctions,
    buffers: &mut EnergyBuffers,
    shape: &EnergyShape,
) -> Result<()> {
    let dim = to_i32(shape.dim, "dim")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(functions.normalize.as_ref());
    unsafe {
        launch
            .arg(&mut buffers.query)
            .arg(&dim)
            .arg(&buffers.control)
            .arg(&mut buffers.status)
            .launch(config(1, THREADS))
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA energy normalization launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn energy_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.energy_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(ENERGY_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_error) => {
            let source = str::from_utf8(ENERGY_PTX)
                .map_err(|error| device(ctx, format!("CUDA energy PTX is not UTF-8: {error}")))?;
            ctx.inner()
                .load_module(Ptx::from_src(source))
                .map_err(|ptx_error| {
                    device(
                        ctx,
                        format!(
                            "CUDA energy CUBIN load failed: {cubin_error}; PTX fallback failed: {ptx_error}"
                        ),
                    )
                })?
        }
    };
    let _ = ctx.energy_module_cache().set(module.clone());
    Ok(module)
}

fn function(
    ctx: &CudaContext,
    module: &Arc<CudaModule>,
    name: &'static str,
) -> Result<Arc<CudaFunction>> {
    ctx.cached_function(module, name, name).map_err(|error| {
        device(
            ctx,
            format!("CUDA energy function {name} load failed: {error}"),
        )
    })
}

const fn config(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn to_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| device_placeholder(format!("CUDA energy {label} exceeds i32: {value}")))
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| device_placeholder(format!("CUDA energy {label} exceeds u32: {value}")))
}

fn device_placeholder(detail: String) -> crate::ForgeError {
    crate::ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: Vec::new(),
        remediation: detail,
    }
}
