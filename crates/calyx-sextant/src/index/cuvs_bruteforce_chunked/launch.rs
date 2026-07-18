use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use super::{CuvsDistanceMetric, cuda_error, invalid, to_i32};

pub(super) const BOUNDARY_REPAIR_QUERY_BATCH: usize = 16;
const BOUNDARY_REPAIR_THREADS: usize = 256;
const BOUNDARY_REPAIR_ROWS_PER_BLOCK: usize = 256;
const STAGING_THREADS: usize = 256;

pub(super) fn load(
    module: &Arc<CudaModule>,
    name: &'static str,
    stage: &'static str,
) -> Result<CudaFunction> {
    module.load_function(name).map_err(cuda_error(stage))
}

pub(super) fn convert_i8(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    source: &CudaSlice<i8>,
    rows: usize,
    dim: usize,
    metric: CuvsDistanceMetric,
    destination: &mut CudaSlice<f32>,
) -> Result<()> {
    let work_items = match metric {
        CuvsDistanceMetric::Cosine => rows,
        CuvsDistanceMetric::SquaredL2 => rows
            .checked_mul(dim)
            .ok_or_else(|| invalid("i8 staging shape overflow"))?,
    };
    let rows = to_i32(rows, "chunk rows")?;
    let dim = to_i32(dim, "dimension")?;
    let normalize = i32::from(metric == CuvsDistanceMetric::Cosine);
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(source)
            .arg(&rows)
            .arg(&dim)
            .arg(&normalize)
            .arg(destination)
            .launch(linear_config(work_items)?)
    }
    .map_err(cuda_error("i8 staging launch"))?;
    stream.synchronize().map_err(cuda_error("i8 staging sync"))
}

pub(super) fn generate_synthetic(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    seed: u64,
    start: u64,
    rows: usize,
    dim: usize,
    destination: &mut CudaSlice<f32>,
) -> Result<()> {
    let rows_i32 = to_i32(rows, "chunk rows")?;
    let dim = to_i32(dim, "dimension")?;
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(&seed)
            .arg(&start)
            .arg(&rows_i32)
            .arg(&dim)
            .arg(destination)
            .launch(linear_config(rows)?)
    }
    .map_err(cuda_error("synthetic staging launch"))?;
    stream
        .synchronize()
        .map_err(cuda_error("synthetic staging sync"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn merge(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    chunk_ids: &mut CudaSlice<i64>,
    chunk_distances: &mut CudaSlice<f32>,
    global_ids: &mut CudaSlice<i64>,
    global_distances: &mut CudaSlice<f32>,
    query_count: usize,
    candidate_stride: usize,
    chunk_k: usize,
    global_k: usize,
    old_count: usize,
    output_count: usize,
    base: u64,
) -> Result<()> {
    let grid = grid(query_count)?;
    let query_count = to_i32(query_count, "query count")?;
    let candidate_stride = to_i32(candidate_stride, "candidate stride")?;
    let chunk_k = to_i32(chunk_k, "chunk k")?;
    let global_k = to_i32(global_k, "global k")?;
    let old_count = to_i32(old_count, "old count")?;
    let output_count = to_i32(output_count, "output count")?;
    let base = i64::try_from(base).map_err(|_| invalid("chunk base exceeds i64"))?;
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(chunk_ids)
            .arg(chunk_distances)
            .arg(global_ids)
            .arg(global_distances)
            .arg(&query_count)
            .arg(&candidate_stride)
            .arg(&chunk_k)
            .arg(&global_k)
            .arg(&old_count)
            .arg(&output_count)
            .arg(&base)
            .launch(config(grid))
    }
    .map_err(cuda_error("merge launch"))?;
    stream.synchronize().map_err(cuda_error("merge sync"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn repair_zero_queries(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    queries: &CudaSlice<f32>,
    dim: usize,
    query_count: usize,
    chunk_k: usize,
    ids: &mut CudaSlice<i64>,
    distances: &mut CudaSlice<f32>,
) -> Result<()> {
    let grid = grid(query_count)?;
    let dim = to_i32(dim, "dimension")?;
    let query_count = to_i32(query_count, "query count")?;
    let chunk_k = to_i32(chunk_k, "chunk k")?;
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(queries)
            .arg(&dim)
            .arg(&query_count)
            .arg(&chunk_k)
            .arg(ids)
            .arg(distances)
            .launch(config(grid))
    }
    .map_err(cuda_error("zero-query repair launch"))?;
    stream
        .synchronize()
        .map_err(cuda_error("zero-query repair sync"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn repair_boundary_ties(
    stream: &Arc<CudaStream>,
    compute_function: &CudaFunction,
    repair_function: &CudaFunction,
    corpus: &CudaSlice<f32>,
    rows: usize,
    dim: usize,
    queries: &CudaSlice<f32>,
    query_count: usize,
    candidate_k: usize,
    output_k: usize,
    metric: CuvsDistanceMetric,
    ids: &mut CudaSlice<i64>,
    distances: &mut CudaSlice<f32>,
    repair_distances: &mut CudaSlice<f32>,
) -> Result<usize> {
    let rows_i32 = to_i32(rows, "chunk rows")?;
    let dim = to_i32(dim, "dimension")?;
    let query_count_i32 = to_i32(query_count, "query count")?;
    let candidate_k = to_i32(candidate_k, "candidate k")?;
    let output_k = to_i32(output_k, "output k")?;
    let metric = match metric {
        CuvsDistanceMetric::Cosine => 0i32,
        CuvsDistanceMetric::SquaredL2 => 1i32,
    };
    let mut launches = 0;
    for query_start in (0..query_count).step_by(BOUNDARY_REPAIR_QUERY_BATCH) {
        let batch_count = BOUNDARY_REPAIR_QUERY_BATCH.min(query_count - query_start);
        let query_start = to_i32(query_start, "boundary query start")?;
        let batch_count_i32 = to_i32(batch_count, "boundary query batch")?;
        {
            let mut launch = stream.launch_builder(compute_function);
            unsafe {
                launch
                    .arg(corpus)
                    .arg(&rows_i32)
                    .arg(&dim)
                    .arg(queries)
                    .arg(&query_count_i32)
                    .arg(&query_start)
                    .arg(&batch_count_i32)
                    .arg(&candidate_k)
                    .arg(&output_k)
                    .arg(&metric)
                    .arg(&*distances)
                    .arg(&mut *repair_distances)
                    .launch(boundary_compute_config(rows, batch_count)?)
            }
            .map_err(cuda_error("boundary-tie distance launch"))?;
        }
        {
            let mut launch = stream.launch_builder(repair_function);
            unsafe {
                launch
                    .arg(&rows_i32)
                    .arg(&query_count_i32)
                    .arg(&query_start)
                    .arg(&batch_count_i32)
                    .arg(&candidate_k)
                    .arg(&output_k)
                    .arg(&mut *ids)
                    .arg(&mut *distances)
                    .arg(&*repair_distances)
                    .launch(config(grid(batch_count)?))
            }
            .map_err(cuda_error("boundary-tie repair launch"))?;
        }
        launches += 1;
    }
    stream
        .synchronize()
        .map_err(cuda_error("boundary-tie repair sync"))?;
    Ok(launches)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exact_cosine_zero_rows(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    corpus: &CudaSlice<f32>,
    rows: usize,
    dim: usize,
    queries: &CudaSlice<f32>,
    query_count: usize,
    chunk_k: usize,
    ids: &mut CudaSlice<i64>,
    distances: &mut CudaSlice<f32>,
) -> Result<()> {
    let grid = grid(query_count)?;
    let rows = to_i32(rows, "chunk rows")?;
    let dim = to_i32(dim, "dimension")?;
    let query_count = to_i32(query_count, "query count")?;
    let chunk_k = to_i32(chunk_k, "chunk k")?;
    let mut launch = stream.launch_builder(function);
    unsafe {
        launch
            .arg(corpus)
            .arg(&rows)
            .arg(&dim)
            .arg(queries)
            .arg(&query_count)
            .arg(&chunk_k)
            .arg(ids)
            .arg(distances)
            .launch(config(grid))
    }
    .map_err(cuda_error("zero-row exact launch"))?;
    stream
        .synchronize()
        .map_err(cuda_error("zero-row exact sync"))
}

fn grid(query_count: usize) -> Result<u32> {
    u32::try_from(query_count).map_err(|_| invalid("query grid exceeds u32"))
}

fn config(grid: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn boundary_compute_config(rows: usize, query_count: usize) -> Result<LaunchConfig> {
    let row_blocks = rows.div_ceil(BOUNDARY_REPAIR_ROWS_PER_BLOCK);
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(row_blocks).map_err(|_| invalid("row grid exceeds u32"))?,
            u32::try_from(query_count).map_err(|_| invalid("query grid exceeds u32"))?,
            1,
        ),
        block_dim: (BOUNDARY_REPAIR_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn linear_config(work_items: usize) -> Result<LaunchConfig> {
    let blocks = work_items.div_ceil(STAGING_THREADS);
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(blocks).map_err(|_| invalid("staging grid exceeds u32"))?,
            1,
            1,
        ),
        block_dim: (STAGING_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}
