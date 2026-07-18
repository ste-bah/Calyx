use serde::{Deserialize, Serialize};

use crate::{CudaContext, Result, init_cuda};

mod buffers;
mod launch;

#[cfg(test)]
mod fsv;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

/// Member-matrix element count at or above this measured crossover uses CUDA.
pub const ENERGY_CUDA_MIN_ELEMENTS: usize = 262_144;
/// Page-locked staging stays bounded while the admitted matrix becomes resident.
pub const CUDA_ENERGY_PINNED_CHUNK_BYTES: usize = 64 * 1024 * 1024;
/// One descent may reserve at most eight GiB of device memory.
pub const CUDA_ENERGY_MAX_DEVICE_BYTES: usize = 8 * 1024 * 1024 * 1024;

pub(crate) const CUDA_ENERGY_VRAM_RESERVE_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaEnergyStats {
    pub members: u64,
    pub dim: u64,
    pub max_steps: u64,
    pub kernel_launches: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub peak_pinned_staging_bytes: u64,
    pub peak_device_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaEnergyDescent {
    pub vector: Vec<f32>,
    pub steps_taken: usize,
    pub converged: bool,
    pub final_energy: f32,
    pub stats: CudaEnergyStats,
}

#[derive(Clone, Debug)]
pub struct CudaEnergyContext {
    ctx: CudaContext,
}

impl CudaEnergyContext {
    pub fn new(device_idx: u32) -> Result<Self> {
        init_cuda(device_idx, true).map(Self::with_context)
    }

    pub fn with_context(ctx: CudaContext) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Runs all descent iterations with one resident member matrix.
    ///
    /// Region values are validated on device. Oversized regions fail admission
    /// before allocation and never fall back to a host scan.
    pub fn descend(
        &self,
        initial: &[f32],
        members: &[&[f32]],
        beta: f32,
        max_steps: usize,
        eps: f32,
    ) -> Result<CudaEnergyDescent> {
        let shape = buffers::validate_and_admit(&self.ctx, initial, members, beta, max_steps, eps)?;
        let mut resident = buffers::EnergyBuffers::allocate(&self.ctx, initial, members, &shape)?;
        let kernel_launches =
            launch::run_descent(&self.ctx, &mut resident, &shape, beta, max_steps, eps)?;
        buffers::read_result(&self.ctx, resident, shape, max_steps, kernel_launches)
    }
}
