use cudarc::driver::CudaSlice;

use super::CudaQuantContext;
use crate::cuda::topk::topk_gpu;
use crate::{ForgeError, Result};

pub struct CudaQuantScores {
    pub(super) quant: CudaQuantContext,
    pub(super) scores: CudaSlice<f32>,
    pub(super) len: usize,
}

impl CudaQuantScores {
    pub(super) fn new(quant: CudaQuantContext, scores: CudaSlice<f32>, len: usize) -> Self {
        Self { quant, scores, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn read(&self) -> Result<Vec<f32>> {
        let values = self
            .quant
            .context()
            .inner()
            .default_stream()
            .clone_dtoh(&self.scores)
            .map_err(|error| device(&self.quant, format!("score readback failed: {error}")))?;
        self.quant
            .counters()
            .add_d2h(values.len() * size_of::<f32>());
        Ok(values)
    }

    pub fn topk(&self, k: usize) -> Result<Vec<(usize, f32)>> {
        let result = topk_gpu(self.quant.context(), &self.scores, k, self.len)?;
        if k != 0 && self.len != 0 {
            let compact_rows = self
                .len
                .div_ceil(crate::CUDA_EXACT_TOPK_MAX_K)
                .saturating_mul(k.min(self.len));
            let counters = self.quant.counters();
            counters.add_launches(1);
            counters.add_compact_topk_rows(compact_rows);
            counters.add_d2h(compact_rows.saturating_mul(size_of::<i32>() + size_of::<f32>()));
        }
        Ok(result)
    }
}

impl std::fmt::Debug for CudaQuantScores {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaQuantScores")
            .field("len", &self.len)
            .finish()
    }
}

fn device(quant: &CudaQuantContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", quant.context().device_idx()),
        detail,
        remediation: "Keep scores resident or use an available CUDA device".to_string(),
    }
}
