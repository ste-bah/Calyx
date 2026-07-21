use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use super::{cuda_error, invalid};

const CUBIN: &[u8] = include_bytes!(env!("SEXTANT_CHUNKED_EXACT_MERGE_CUBIN_PATH"));
const THREADS: usize = 256;

pub(super) struct ExactKernels {
    distances: CudaFunction,
    topk: CudaFunction,
}

impl ExactKernels {
    pub(super) fn load(context: &Arc<CudaContext>) -> Result<Self> {
        let module = context
            .load_module(Ptx::from_binary(CUBIN.to_vec()))
            .map_err(cuda_error("exact CUBIN load"))?;
        Ok(Self {
            distances: module
                .load_function("exact_cosine_resident_distances")
                .map_err(cuda_error("exact distance kernel load"))?,
            topk: module
                .load_function("exact_cosine_resident_topk")
                .map_err(cuda_error("exact top-k kernel load"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        &self,
        stream: &Arc<CudaStream>,
        corpus: &CudaSlice<f32>,
        queries: &CudaSlice<f32>,
        allowed: &CudaSlice<u32>,
        row_distances: &mut CudaSlice<f32>,
        output_ids: &mut CudaSlice<i64>,
        output_distances: &mut CudaSlice<f32>,
        rows: usize,
        dim: usize,
        query_count: usize,
        output_k: usize,
    ) -> Result<()> {
        let rows_i32 = to_i32(rows, "rows")?;
        let dim_i32 = to_i32(dim, "dimension")?;
        let query_count_i32 = to_i32(query_count, "query count")?;
        let output_k_i32 = to_i32(output_k, "output k")?;
        let row_stride = i64::from(dim_i32);
        let column_stride = 1_i64;
        let metric = 0_i32;
        let mut distance_launch = stream.launch_builder(&self.distances);
        unsafe {
            distance_launch
                .arg(corpus)
                .arg(&rows_i32)
                .arg(&dim_i32)
                .arg(&row_stride)
                .arg(&column_stride)
                .arg(queries)
                .arg(&query_count_i32)
                .arg(allowed)
                .arg(&metric)
                .arg(&mut *row_distances)
                .launch(distance_config(rows, query_count)?)
        }
        .map_err(cuda_error("resident exact distance launch"))?;

        let mut topk_launch = stream.launch_builder(&self.topk);
        unsafe {
            topk_launch
                .arg(&*row_distances)
                .arg(&rows_i32)
                .arg(&query_count_i32)
                .arg(&output_k_i32)
                .arg(output_ids)
                .arg(output_distances)
                .launch(LaunchConfig {
                    grid_dim: (to_u32(query_count, "query count")?, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map(|_| ())
        .map_err(cuda_error("resident exact top-k launch"))
    }
}

fn distance_config(rows: usize, query_count: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (
            to_u32(rows.div_ceil(THREADS), "row blocks")?,
            to_u32(query_count, "query count")?,
            1,
        ),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn to_i32(value: usize, label: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("resident exact {label} exceeds i32")))
}

fn to_u32(value: usize, label: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("resident exact {label} exceeds u32")))
}
