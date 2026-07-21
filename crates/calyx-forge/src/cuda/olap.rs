use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::{CudaContext, Result, init_cuda};

mod launch;
mod scan;
mod transpose;

#[cfg(test)]
mod tests;

/// Host rows at or above this measured crossover use CUDA OLAP reduction.
pub const OLAP_CUDA_MIN_ROWS: usize = 65_536;
/// Matrix element count at or above this crossover uses tiled CUDA transpose.
pub const TRANSPOSE_CUDA_MIN_ELEMENTS: usize = 262_144;
/// Each input or output page-locked staging allocation is bounded to 16 MiB.
pub const CUDA_OLAP_PINNED_CHUNK_BYTES: usize = 16 * 1024 * 1024;

pub(crate) const VALUE_NONFINITE_OP: &str = "olap.value_finite";
pub(crate) const GROUP_NONFINITE_OP: &str = "olap.group_finite";
pub(crate) const GROUP_CAP_OP: &str = "olap.group_cap";

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaOlapAggregate {
    pub count: u64,
    pub sum: f64,
    pub min: f32,
    pub max: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaOlapGroup {
    pub key_bits: u32,
    pub aggregate: CudaOlapAggregate,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaOlapScan {
    pub aggregate: CudaOlapAggregate,
    pub groups: Vec<CudaOlapGroup>,
    pub stats: CudaOlapStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaOlapStats {
    pub rows: u64,
    pub columns: u64,
    pub chunks: u64,
    pub dictionary_capacity: u64,
    pub kernel_launches: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub peak_pinned_staging_bytes: u64,
    pub peak_device_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct CudaOlapContext {
    ctx: CudaContext,
    transpose_buffers: Arc<Mutex<Option<transpose::TransposeBuffers>>>,
}

impl CudaOlapContext {
    pub fn new(device_idx: u32) -> Result<Self> {
        init_cuda(device_idx, true).map(Self::with_context)
    }

    pub fn with_context(ctx: CudaContext) -> Self {
        Self {
            ctx,
            transpose_buffers: Arc::new(Mutex::new(None)),
        }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }
}

/// Deterministic acceptance bound for a CUDA `f64` sum against row-order CPU summation.
///
/// Count/min/max and group-key bits are exact. CUDA sums may associate finite
/// `f32` inputs differently, so the bound is `max(1e-12, 8*eps*n^2*max_abs)`.
/// Average tolerance is this value divided by `count`.
pub fn olap_sum_tolerance(count: u64, min: f32, max: f32) -> f64 {
    let n = count as f64;
    let max_abs = f64::from(min).abs().max(f64::from(max).abs());
    (8.0 * f64::EPSILON * n * n * max_abs).max(1.0e-12)
}
