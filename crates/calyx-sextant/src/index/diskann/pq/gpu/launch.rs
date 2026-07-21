use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use super::cuda::{cuda_error, to_i32};
use crate::index::diskann::pq::invalid;

const THREADS: usize = 256;
const AXIS_TILE: usize = 8;

pub(super) struct Kernels {
    assign: CudaFunction,
    accumulate: CudaFunction,
    finalize: CudaFunction,
}

impl Kernels {
    pub(super) fn load(module: &Arc<CudaModule>) -> Result<Self> {
        Ok(Self {
            assign: module
                .load_function("diskann_pq_assign_nearest")
                .map_err(cuda_error("load assignment kernel"))?,
            accumulate: module
                .load_function("diskann_pq_accumulate_tiled")
                .map_err(cuda_error("load accumulation kernel"))?,
            finalize: module
                .load_function("diskann_pq_finalize_centroids")
                .map_err(cuda_error("load centroid kernel"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn assign(
        &self,
        stream: &Arc<CudaStream>,
        rows: &CudaSlice<f32>,
        row_count: usize,
        dim: usize,
        subvectors: usize,
        centroids: usize,
        subdim: usize,
        codebook: &CudaSlice<f32>,
        labels: &mut CudaSlice<u8>,
    ) -> Result<()> {
        let work_items = row_count
            .checked_mul(subvectors)
            .ok_or_else(|| invalid("CUDA PQ assignment shape overflow"))?;
        let row_count = to_i32(row_count, "row count")?;
        let dim = to_i32(dim, "dimension")?;
        let subvectors = to_i32(subvectors, "subvectors")?;
        let centroids = to_i32(centroids, "centroids")?;
        let subdim = to_i32(subdim, "subdimension")?;
        let mut launch = stream.launch_builder(&self.assign);
        unsafe {
            launch
                .arg(rows)
                .arg(&row_count)
                .arg(&dim)
                .arg(&subvectors)
                .arg(&centroids)
                .arg(&subdim)
                .arg(codebook)
                .arg(labels)
                .launch(linear_config(work_items)?)
        }
        .map_err(cuda_error("launch assignment kernel"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate(
        &self,
        stream: &Arc<CudaStream>,
        rows: &CudaSlice<f32>,
        labels: &CudaSlice<u8>,
        row_count: usize,
        dim: usize,
        subvectors: usize,
        centroids: usize,
        subdim: usize,
        sums: &mut CudaSlice<f32>,
        counts: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let axis_tiles = subdim.div_ceil(AXIS_TILE);
        let grid_x = row_count.div_ceil(THREADS);
        let grid_y = subvectors
            .checked_mul(axis_tiles)
            .ok_or_else(|| invalid("CUDA PQ accumulation grid overflow"))?;
        let shared_bytes = centroids
            .checked_mul(AXIS_TILE * size_of::<f32>() + size_of::<u32>())
            .ok_or_else(|| invalid("CUDA PQ shared-memory shape overflow"))?;
        let config = LaunchConfig {
            grid_dim: (
                to_u32(grid_x, "accumulation grid x")?,
                to_u32(grid_y, "accumulation grid y")?,
                1,
            ),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: to_u32(shared_bytes, "accumulation shared bytes")?,
        };
        let row_count = to_i32(row_count, "row count")?;
        let dim = to_i32(dim, "dimension")?;
        let subvectors = to_i32(subvectors, "subvectors")?;
        let centroids = to_i32(centroids, "centroids")?;
        let subdim = to_i32(subdim, "subdimension")?;
        let axis_tiles = to_i32(axis_tiles, "axis tiles")?;
        let mut launch = stream.launch_builder(&self.accumulate);
        unsafe {
            launch
                .arg(rows)
                .arg(labels)
                .arg(&row_count)
                .arg(&dim)
                .arg(&subvectors)
                .arg(&centroids)
                .arg(&subdim)
                .arg(&axis_tiles)
                .arg(sums)
                .arg(counts)
                .launch(config)
        }
        .map_err(cuda_error("launch accumulation kernel"))?;
        Ok(())
    }

    pub(super) fn finalize(
        &self,
        stream: &Arc<CudaStream>,
        sums: &CudaSlice<f32>,
        counts: &CudaSlice<u32>,
        centroids: usize,
        subdim: usize,
        codebook: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let cells = codebook.len();
        let cells_u64 =
            u64::try_from(cells).map_err(|_| invalid("CUDA PQ codebook cells exceed u64"))?;
        let centroids = to_i32(centroids, "centroids")?;
        let subdim = to_i32(subdim, "subdimension")?;
        let mut launch = stream.launch_builder(&self.finalize);
        unsafe {
            launch
                .arg(sums)
                .arg(counts)
                .arg(&centroids)
                .arg(&subdim)
                .arg(&cells_u64)
                .arg(codebook)
                .launch(linear_config(cells)?)
        }
        .map_err(cuda_error("launch centroid kernel"))?;
        Ok(())
    }
}

fn linear_config(work_items: usize) -> Result<LaunchConfig> {
    let grid = work_items.div_ceil(THREADS);
    Ok(LaunchConfig {
        grid_dim: (to_u32(grid, "linear grid")?, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn to_u32(value: usize, name: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("CUDA PQ {name} exceeds u32")))
}
