use serde::{Deserialize, Serialize};

use crate::{CudaContext, Result, init_cuda};

mod buffers;
#[cfg(test)]
mod fsv;
mod launch;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

/// Point count at or above this crossover uses the strict CUDA provider.
pub const SKILL_CUDA_MIN_POINTS: usize = 256;
pub const CUDA_SKILL_MAX_POINTS: usize = 2_048;
pub const CUDA_SKILL_MAX_DEVICE_BYTES: usize = 1024 * 1024 * 1024;
pub(crate) const CUDA_SKILL_VRAM_RESERVE_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaSkillSlot {
    pub dim: usize,
    pub point_indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaSkillEdge {
    pub source: usize,
    pub destination: usize,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaSkillStats {
    pub points: u64,
    pub slots: u64,
    pub feature_values: u64,
    pub pairwise_values: u64,
    pub kernel_launches: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub peak_device_bytes: u64,
    pub full_distance_readback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaSkillMst {
    pub edges: Vec<CudaSkillEdge>,
    pub distances: Option<Vec<f64>>,
    pub stats: CudaSkillStats,
}

#[derive(Clone, Debug)]
pub struct CudaSkillContext {
    ctx: CudaContext,
}

impl CudaSkillContext {
    pub fn new(device_idx: u32) -> Result<Self> {
        init_cuda(device_idx, true).map(Self::with_context)
    }

    pub fn with_context(ctx: CudaContext) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Computes fused distances, core distances, and the deterministic MST on device.
    pub fn minimum_spanning_tree(
        &self,
        point_count: usize,
        slots: &[CudaSkillSlot],
        min_samples: usize,
    ) -> Result<CudaSkillMst> {
        self.run(point_count, slots, min_samples, false)
    }

    /// Acceptance-only variant that also reads back the full distance matrix.
    pub fn minimum_spanning_tree_with_distances(
        &self,
        point_count: usize,
        slots: &[CudaSkillSlot],
        min_samples: usize,
    ) -> Result<CudaSkillMst> {
        self.run(point_count, slots, min_samples, true)
    }

    fn run(
        &self,
        point_count: usize,
        slots: &[CudaSkillSlot],
        min_samples: usize,
        read_distances: bool,
    ) -> Result<CudaSkillMst> {
        let (shape, host) =
            buffers::validate_and_flatten(&self.ctx, point_count, slots, min_samples)?;
        let mut device = buffers::SkillBuffers::allocate(&self.ctx, &shape, &host)?;
        launch::run(&self.ctx, &shape, &mut device)?;
        buffers::read_result(&self.ctx, shape, device, read_distances)
    }
}
