use calyx_core::Result;

#[cfg(not(sextant_cuvs))]
use super::invalid;
use super::{BuildOutput, DiskAnnPqBuildExecution, DiskAnnPqBuildParams};

#[cfg(sextant_cuvs)]
mod cuda;
#[cfg(sextant_cuvs)]
mod launch;

#[cfg(sextant_cuvs)]
pub(super) fn build(
    rows: &[(u32, Vec<f32>)],
    params: DiskAnnPqBuildParams,
    requested: DiskAnnPqBuildExecution,
) -> Result<BuildOutput> {
    cuda::build(rows, params, requested)
}

#[cfg(not(sextant_cuvs))]
pub(super) fn build(
    rows: &[(u32, Vec<f32>)],
    _params: DiskAnnPqBuildParams,
    requested: DiskAnnPqBuildExecution,
) -> Result<BuildOutput> {
    Err(invalid(format!(
        "strict CUDA PQ execution ({}) was required for {} rows; refusing silent CPU training: {}",
        requested.as_str(),
        rows.len(),
        crate::cuvs_unavailable_reason("DiskANN PQ training and encoding")
    )))
}
