use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::buffers::{SkillBuffers, SkillShape, device};
use crate::cuda::kernels::{SKILL_CUBIN, SKILL_PTX};
use crate::{CudaContext, Result};

const THREADS: u32 = 256;

pub(super) fn run(ctx: &CudaContext, shape: &SkillShape, buffers: &mut SkillBuffers) -> Result<()> {
    let module = skill_module(ctx)?;
    let pairwise = function(ctx, &module, "skill_pairwise_fused_cosine_f64")?;
    let core = function(ctx, &module, "skill_core_distance_sort_f64")?;
    let prim = function(ctx, &module, "skill_prim_mst_f64")?;
    launch_pairwise(ctx, &pairwise, shape, buffers)?;
    launch_core(ctx, &core, shape, buffers)?;
    launch_prim(ctx, &prim, shape, buffers)?;
    ctx.inner().default_stream().synchronize().map_err(|error| {
        device(
            ctx,
            format!("CUDA skill pipeline synchronization failed: {error}"),
        )
    })
}

fn launch_pairwise(
    ctx: &CudaContext,
    function: &CudaFunction,
    shape: &SkillShape,
    buffers: &mut SkillBuffers,
) -> Result<()> {
    let points = to_i32(shape.points, "points")?;
    let slots = to_i32(shape.slots, "slots")?;
    let cells = shape.points * shape.points;
    let blocks = to_u32(cells.div_ceil(THREADS as usize), "distance blocks")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(&buffers.values)
            .arg(&buffers.offsets)
            .arg(&buffers.dims)
            .arg(&points)
            .arg(&slots)
            .arg(&mut buffers.distances)
            .arg(&mut buffers.status)
            .launch(config(blocks))
    }
    .map_err(|error| device(ctx, format!("CUDA skill pairwise launch failed: {error}")))?;
    Ok(())
}

fn launch_core(
    ctx: &CudaContext,
    function: &CudaFunction,
    shape: &SkillShape,
    buffers: &mut SkillBuffers,
) -> Result<()> {
    let points = to_i32(shape.points, "points")?;
    let sort_length = to_i32(shape.sort_length, "sort length")?;
    let neighbor_rank = to_i32(shape.min_samples.min(shape.points - 1), "neighbor rank")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(&buffers.distances)
            .arg(&points)
            .arg(&sort_length)
            .arg(&neighbor_rank)
            .arg(&mut buffers.core)
            .arg(&buffers.status)
            .launch(config(to_u32(shape.points, "core rows")?))
    }
    .map_err(|error| {
        device(
            ctx,
            format!("CUDA skill core-distance launch failed: {error}"),
        )
    })?;
    Ok(())
}

fn launch_prim(
    ctx: &CudaContext,
    function: &CudaFunction,
    shape: &SkillShape,
    buffers: &mut SkillBuffers,
) -> Result<()> {
    let points = to_i32(shape.points, "points")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(&buffers.distances)
            .arg(&buffers.core)
            .arg(&points)
            .arg(&mut buffers.edge_sources)
            .arg(&mut buffers.edge_destinations)
            .arg(&mut buffers.edge_weights)
            .arg(&mut buffers.status)
            .launch(config(1))
    }
    .map_err(|error| device(ctx, format!("CUDA skill Prim launch failed: {error}")))?;
    Ok(())
}

fn skill_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.skill_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(SKILL_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_error) => {
            let source = str::from_utf8(SKILL_PTX)
                .map_err(|error| device(ctx, format!("CUDA skill PTX is not UTF-8: {error}")))?;
            ctx.inner()
                .load_module(Ptx::from_src(source))
                .map_err(|ptx_error| {
                    device(
                        ctx,
                        format!(
                            "CUDA skill CUBIN load failed: {cubin_error}; PTX fallback failed: {ptx_error}"
                        ),
                    )
                })?
        }
    };
    let _ = ctx.skill_module_cache().set(module.clone());
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
            format!("CUDA skill function {name} load failed: {error}"),
        )
    })
}

const fn config(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn to_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| crate::ForgeError::ShapeMismatch {
        expected: vec![i32::MAX as usize],
        got: vec![value],
        remediation: format!("CUDA skill {label} exceeds i32"),
    })
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| crate::ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![value],
        remediation: format!("CUDA skill {label} exceeds u32"),
    })
}
