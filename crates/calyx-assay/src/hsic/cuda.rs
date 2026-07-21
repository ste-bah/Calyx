#[cfg(feature = "cuda")]
use super::core::{resolve_bandwidth, to_finite_f64};
use super::*;

#[cfg(feature = "cuda")]
pub(super) fn hsic_cuda_core(
    x: &[f32],
    y: &[f32],
    config: HsicConfig,
    permutations: Option<&[i32]>,
) -> Result<(StrictHsicCore, f64, f64)> {
    if x.len() != y.len() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "HSIC requires paired samples: x={} y={}",
            x.len(),
            y.len()
        )));
    }
    let xd = to_finite_f64("x", x)?;
    let yd = to_finite_f64("y", y)?;
    let sigma_x = resolve_bandwidth("x", &xd, config.bandwidth_x)?;
    let sigma_y = resolve_bandwidth("y", &yd, config.bandwidth_y)?;
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("HSIC", err))?;
    let core = calyx_forge::hsic_1d_host(backend.context(), x, y, sigma_x, sigma_y, permutations)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("HSIC", err))?;
    Ok((StrictHsicCore::from(core), sigma_x, sigma_y))
}

#[cfg(not(feature = "cuda"))]
pub(super) fn hsic_cuda_core(
    _x: &[f32],
    _y: &[f32],
    _config: HsicConfig,
    _permutations: Option<&[i32]>,
) -> Result<(StrictHsicCore, f64, f64)> {
    Err(cuda_unavailable("HSIC"))
}

pub(super) struct StrictHsicCore {
    pub(super) hsic_biased: f32,
    pub(super) hsic_unbiased: f32,
    pub(super) tr_kc_lc: f64,
    pub(super) off_diag_sum_k: f64,
    pub(super) off_diag_sum_l: f64,
    pub(super) sum_sq_centered_offdiag: f64,
    pub(super) n_samples: usize,
    pub(super) ge_count: Option<usize>,
}

#[cfg(feature = "cuda")]
impl From<calyx_forge::CudaHsicResult> for StrictHsicCore {
    fn from(value: calyx_forge::CudaHsicResult) -> Self {
        Self {
            hsic_biased: value.hsic_biased,
            hsic_unbiased: value.hsic_unbiased,
            tr_kc_lc: value.tr_kc_lc,
            off_diag_sum_k: value.off_diag_sum_k,
            off_diag_sum_l: value.off_diag_sum_l,
            sum_sq_centered_offdiag: value.sum_sq_centered_offdiag,
            n_samples: value.n_samples,
            ge_count: value.ge_count,
        }
    }
}
