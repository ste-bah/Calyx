use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::cuda::kernels::{OLAP_CUBIN, OLAP_PTX};
use crate::{CudaContext, ForgeError, Result};

const THREADS: u32 = 256;
const MAX_REDUCE_BLOCKS: usize = 256;
const MAX_GROUP_BLOCKS: usize = 4096;
const REMEDIATION: &str =
    "check the embedded OLAP CUDA module, pinned-memory headroom, and CUDA device availability";

pub(super) struct ReduceBuffers {
    pub counts: CudaSlice<u64>,
    pub sums: CudaSlice<f64>,
    pub mins: CudaSlice<u32>,
    pub maxs: CudaSlice<u32>,
}

pub(super) struct GroupBuffers {
    pub slots: CudaSlice<u64>,
    pub counts: CudaSlice<u64>,
    pub sums: CudaSlice<f64>,
    pub mins: CudaSlice<u32>,
    pub maxs: CudaSlice<u32>,
    pub unique_count: CudaSlice<u32>,
    pub capacity: usize,
}

pub(super) fn reduce_blocks(rows: usize) -> usize {
    rows.div_ceil(THREADS as usize).clamp(1, MAX_REDUCE_BLOCKS)
}

pub(super) fn reduce(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    rows: usize,
    output: &mut ReduceBuffers,
    status: &mut CudaSlice<u32>,
) -> Result<()> {
    let rows = as_u32(rows, "OLAP reduction rows")?;
    let function = function(ctx, "olap.reduce", "olap_reduce_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(&function);
    unsafe {
        launch
            .arg(values)
            .arg(&rows)
            .arg(&mut output.counts)
            .arg(&mut output.sums)
            .arg(&mut output.mins)
            .arg(&mut output.maxs)
            .arg(status)
            .launch(LaunchConfig {
                grid_dim: (reduce_blocks(rows as usize) as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|error| device(ctx, format!("OLAP reduction launch failed: {error}")))?;
    Ok(())
}

pub(super) fn group_init(ctx: &CudaContext, groups: &mut GroupBuffers) -> Result<()> {
    let capacity = as_u32(groups.capacity, "OLAP dictionary capacity")?;
    let function = function(ctx, "olap.group_init", "olap_group_init")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(&function);
    unsafe {
        launch
            .arg(&mut groups.slots)
            .arg(&mut groups.counts)
            .arg(&mut groups.sums)
            .arg(&mut groups.mins)
            .arg(&mut groups.maxs)
            .arg(&capacity)
            .launch(flat_config(groups.capacity, MAX_GROUP_BLOCKS)?)
    }
    .map_err(|error| device(ctx, format!("OLAP dictionary init failed: {error}")))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn group_reduce(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    keys: &CudaSlice<f32>,
    rows: usize,
    max_groups: usize,
    groups: &mut GroupBuffers,
    status: &mut CudaSlice<u32>,
) -> Result<()> {
    let rows = as_u32(rows, "OLAP grouped rows")?;
    let max_groups = as_u32(max_groups, "OLAP group cap")?;
    let capacity = as_u32(groups.capacity, "OLAP dictionary capacity")?;
    let function = function(ctx, "olap.group_reduce", "olap_group_reduce_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(&function);
    unsafe {
        launch
            .arg(values)
            .arg(keys)
            .arg(&rows)
            .arg(&max_groups)
            .arg(&mut groups.slots)
            .arg(&mut groups.counts)
            .arg(&mut groups.sums)
            .arg(&mut groups.mins)
            .arg(&mut groups.maxs)
            .arg(&capacity)
            .arg(&mut groups.unique_count)
            .arg(status)
            .launch(flat_config(rows as usize, MAX_GROUP_BLOCKS)?)
    }
    .map_err(|error| device(ctx, format!("OLAP group reduction launch failed: {error}")))?;
    Ok(())
}

pub(super) fn transpose(
    ctx: &CudaContext,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    columns: usize,
) -> Result<()> {
    let rows = as_u32(rows, "OLAP transpose rows")?;
    let columns = as_u32(columns, "OLAP transpose columns")?;
    let function = function(ctx, "olap.transpose", "olap_transpose_f32")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(&function);
    unsafe {
        launch
            .arg(input)
            .arg(output)
            .arg(&rows)
            .arg(&columns)
            .launch(LaunchConfig {
                grid_dim: (columns.div_ceil(32), rows.div_ceil(32), 1),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            })
    }
    .map_err(|error| device(ctx, format!("OLAP transpose launch failed: {error}")))?;
    Ok(())
}

pub(super) fn synchronize(ctx: &CudaContext, operation: &str) -> Result<()> {
    ctx.inner()
        .default_stream()
        .synchronize()
        .map_err(|error| device(ctx, format!("{operation} synchronization failed: {error}")))
}

fn flat_config(items: usize, max_blocks: usize) -> Result<LaunchConfig> {
    let blocks = items.div_ceil(THREADS as usize).clamp(1, max_blocks);
    Ok(LaunchConfig {
        grid_dim: (as_u32(blocks, "OLAP grid blocks")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn function(
    ctx: &CudaContext,
    cache_key: &'static str,
    name: &'static str,
) -> Result<Arc<CudaFunction>> {
    let module = module(ctx)?;
    ctx.cached_function(&module, cache_key, name)
        .map_err(|error| device(ctx, format!("{name} load failed: {error}")))
}

fn module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.olap_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(OLAP_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_error) => {
            let ptx = str::from_utf8(OLAP_PTX)
                .map_err(|error| device(ctx, format!("OLAP PTX is not UTF-8: {error}")))?;
            ctx.inner()
                .load_module(Ptx::from_src(ptx))
                .map_err(|ptx_error| {
                    device(
                        ctx,
                        format!(
                            "OLAP CUBIN load failed: {cubin_error}; PTX fallback failed: {ptx_error}"
                        ),
                    )
                })?
        }
    };
    let _ = ctx.olap_module_cache().set(module.clone());
    Ok(module)
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![value],
        remediation: format!("{label} must fit CUDA u32 indexing"),
    })
}

pub(super) fn device(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation: REMEDIATION.to_string(),
    }
}
