//! Shared routing and telemetry contracts for bulk dependence estimators.

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
static DEPENDENCE_CUDA_CONTEXT: std::sync::OnceLock<calyx_forge::CudaContext> =
    std::sync::OnceLock::new();

/// Set to `0`, `false`, `no`, or `off` to force the documented CPU route for
/// profiling. Strict CUDA entry points and `CALYX_ASSAY_CUDA_STRICT` ignore it.
pub const DEPENDENCE_CUDA_AUTO_ENV: &str = "CALYX_ASSAY_DEPENDENCE_CUDA_AUTO";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependenceCudaStats {
    pub operation: String,
    pub n_samples: usize,
    pub work_items: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub peak_device_bytes: usize,
    pub kernel_launches: usize,
}

#[cfg(feature = "cuda")]
impl From<calyx_forge::CudaDependenceStats> for DependenceCudaStats {
    fn from(value: calyx_forge::CudaDependenceStats) -> Self {
        Self {
            operation: value.operation,
            n_samples: value.n_samples,
            work_items: value.work_items,
            host_to_device_bytes: value.host_to_device_bytes,
            device_to_host_bytes: value.device_to_host_bytes,
            peak_device_bytes: value.peak_device_bytes,
            kernel_launches: value.kernel_launches,
        }
    }
}

pub(crate) fn auto_cuda_at(n_samples: usize, threshold: usize) -> bool {
    if !cfg!(feature = "cuda") || n_samples < threshold {
        return false;
    }
    std::env::var(DEPENDENCE_CUDA_AUTO_ENV)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
pub(crate) fn dependence_cuda_context(
    operation: &str,
) -> calyx_core::Result<&'static calyx_forge::CudaContext> {
    if let Some(context) = DEPENDENCE_CUDA_CONTEXT.get() {
        return Ok(context);
    }
    let context = calyx_forge::init_cuda(0, false)
        .map_err(|err| crate::cuda_strict::forge_to_calyx(operation, err))?;
    let _ = DEPENDENCE_CUDA_CONTEXT.set(context);
    DEPENDENCE_CUDA_CONTEXT.get().ok_or_else(|| {
        calyx_core::CalyxError::forge_device_unavailable(format!(
            "{operation} CUDA context cache did not retain the initialized device"
        ))
    })
}
