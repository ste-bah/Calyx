//! HSIC — the Hilbert–Schmidt Independence Criterion, a kernel independence test
//! that is **0 iff X and Y are independent** in the RKHS sense (#55). Unlike the
//! k-NN mutual-information estimators (KSG), whose bias and variance degrade in
//! high dimension, HSIC stays stable — it is the market screen's robust,
//! non-parametric independence test, complementing the linear/rank/dCor family.
//!
//! Construction (Gretton, Bousquet, Smola, Schölkopf 2005; Gretton et al. 2008):
//! - Gaussian RBF Gram matrices `K_ij = exp(−‖x_i−x_j‖²/(2σ²))`, `L` likewise;
//!   `σ` defaults to the **median-pairwise-distance heuristic**
//!   `σ = √(median{‖x_i−x_j‖² : i<j}/2)` per variable.
//! - **Biased** estimator `HSIC_b = n⁻²·tr(K_c L_c)` where `K_c = HKH` is the
//!   double-centred Gram matrix (`H = I − 11ᵀ/n`).
//! - **Unbiased** estimator (Song et al. 2012, `n ≥ 4`) from the diagonal-zeroed
//!   `K̃, L̃`:
//!   `HSIC_u = [tr(K̃L̃) + (1ᵀK̃1)(1ᵀL̃1)/((n−1)(n−2)) − 2/(n−2)·1ᵀK̃L̃1] / (n(n−3))`.
//!
//! Significance has two paths:
//! - **Closed-form gamma approximation** (Gretton 2008): under H₀ the statistic
//!   `T = n·HSIC_b` is Gamma-distributed with moments matched to the null;
//!   `p = 1 − F_Γ(T; α, β)` with `α = mean²/var`, `β = var·n/mean`. This is the
//!   "stable, no-permutation" test the issue asks for; it is asymptotic and needs
//!   a reasonable `n` (variance is only defined for `n ≥ 6`).
//! - **Seeded permutation test** ([`hsic_permutation_test`]) — exact-valid at any
//!   `n ≥ 4` via the add-one estimator `(1+#ge)/(1+P)`, for small-sample rigor.
//!
//! Fails closed on length mismatch, non-finite input, too few samples, or a
//! constant (zero median distance ⇒ undefined bandwidth) column — never `NaN`.

use calyx_core::{CalyxError, Result};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "cuda"))]
use crate::cuda_strict::cuda_unavailable;
use crate::cuda_strict::{deterministic_permutations, strict_cuda_requested};
use crate::special_fn::gammq;

mod core;
mod cuda;

use self::core::HsicCore;
use self::cuda::hsic_cuda_core;

/// Minimum samples for the biased/unbiased HSIC point estimates (unbiased needs
/// `n(n−3)` and `(n−1)(n−2)` denominators, i.e. `n ≥ 4`).
pub const MIN_HSIC_SAMPLES: usize = 4;
/// Minimum samples for the closed-form gamma test (null variance carries the
/// factor `(n−4)(n−5)`, so it is positive only for `n ≥ 6`).
pub const MIN_HSIC_GAMMA_SAMPLES: usize = 6;
/// Default permutation count for the permutation independence test.
pub const DEFAULT_HSIC_PERMUTATIONS: usize = 999;
/// Default deterministic seed for the permutation null.
pub const DEFAULT_HSIC_SEED: u64 = 0x0451_C0DE_0DE5_EED5;

/// Kernel-bandwidth configuration. `None` selects the median-distance heuristic
/// for that variable; `Some(σ)` pins a fixed bandwidth (e.g. for reproducible
/// regression checks).
#[derive(Clone, Copy, Debug, Default)]
pub struct HsicConfig {
    pub bandwidth_x: Option<f64>,
    pub bandwidth_y: Option<f64>,
}

/// Configuration for the permutation independence test.
#[derive(Clone, Copy, Debug)]
pub struct HsicPermConfig {
    pub kernel: HsicConfig,
    pub permutations: usize,
    pub seed: u64,
}

impl Default for HsicPermConfig {
    fn default() -> Self {
        Self {
            kernel: HsicConfig::default(),
            permutations: DEFAULT_HSIC_PERMUTATIONS,
            seed: DEFAULT_HSIC_SEED,
        }
    }
}

/// HSIC point estimates (no p-value).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsicEstimators {
    pub hsic_biased: f32,
    pub hsic_unbiased: f32,
    pub bandwidth_x: f32,
    pub bandwidth_y: f32,
    pub n_samples: usize,
}

/// HSIC with the closed-form gamma-approximation independence p-value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsicReport {
    pub hsic_biased: f32,
    pub hsic_unbiased: f32,
    /// Test statistic `T = n·HSIC_b` fed to the gamma null.
    pub test_statistic: f32,
    pub p_value: f32,
    pub gamma_shape: f32,
    pub gamma_scale: f32,
    pub bandwidth_x: f32,
    pub bandwidth_y: f32,
    pub n_samples: usize,
}

/// HSIC with a seeded permutation independence p-value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsicTest {
    pub hsic_biased: f32,
    pub p_value: f32,
    pub permutations: usize,
    pub ge_count: usize,
    pub seed: u64,
    pub n_samples: usize,
}

/// HSIC biased/unbiased point estimates with the default (median-heuristic)
/// kernel.
pub fn hsic_estimators(x: &[f32], y: &[f32]) -> Result<HsicEstimators> {
    hsic_estimators_with_config(x, y, HsicConfig::default())
}

/// HSIC point estimates with an explicit kernel configuration.
pub fn hsic_estimators_with_config(
    x: &[f32],
    y: &[f32],
    config: HsicConfig,
) -> Result<HsicEstimators> {
    if strict_cuda_requested() {
        return hsic_estimators_with_config_cuda_strict(x, y, config);
    }
    let core = HsicCore::build(x, y, config)?;
    Ok(HsicEstimators {
        hsic_biased: core.hsic_biased as f32,
        hsic_unbiased: core.hsic_unbiased as f32,
        bandwidth_x: core.sigma_x as f32,
        bandwidth_y: core.sigma_y as f32,
        n_samples: core.n,
    })
}

/// HSIC with the closed-form gamma-approximation p-value (median-heuristic kernel).
pub fn hsic(x: &[f32], y: &[f32]) -> Result<HsicReport> {
    hsic_with_config(x, y, HsicConfig::default())
}

/// HSIC gamma test with an explicit kernel configuration.
pub fn hsic_with_config(x: &[f32], y: &[f32], config: HsicConfig) -> Result<HsicReport> {
    if strict_cuda_requested() {
        return hsic_with_config_cuda_strict(x, y, config);
    }
    let core = HsicCore::build(x, y, config)?;
    let n = core.n;
    if n < MIN_HSIC_GAMMA_SAMPLES {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "HSIC gamma test requires at least {MIN_HSIC_GAMMA_SAMPLES} samples (null variance needs n≥6); got {n}. Use hsic_permutation_test for small n"
        )));
    }
    let nf = n as f64;

    // Statistic T = n·HSIC_b = tr(K_c L_c)/n.
    let test_statistic = core.tr_kc_lc / nf;

    // Null mean from the diagonal-zeroed RAW Gram sums.
    let mu_x = core.off_diag_sum_k / (nf * (nf - 1.0));
    let mu_y = core.off_diag_sum_l / (nf * (nf - 1.0));
    let mean = (1.0 + mu_x * mu_y - mu_x - mu_y) / nf;

    // Null variance: 2(n-4)(n-5)/(n(n-1)(n-2)(n-3)) · 1/(n(n-1)) · Σ_{i≠j} (Kc_ij Lc_ij)².
    let var_prefactor = 2.0 * (nf - 4.0) * (nf - 5.0)
        / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0))
        / (nf * (nf - 1.0));
    let var = var_prefactor * core.sum_sq_centered_offdiag;

    if mean.is_nan() || mean <= 0.0 || var.is_nan() || var <= 0.0 {
        return Err(CalyxError::assay_degenerate_input(
            "HSIC gamma test undefined: non-positive null moment (degenerate kernel structure)",
        ));
    }
    // Gamma(shape α, scale β): mean α·β = n·mean = E[T]; β carries the extra ·n.
    let shape = mean * mean / var;
    let scale = var * nf / mean;
    // p = 1 − F_Γ(T; α, β) = Q(α, T/β) (regularised upper incomplete gamma).
    let p_value = gammq(shape, test_statistic / scale)?;

    Ok(HsicReport {
        hsic_biased: core.hsic_biased as f32,
        hsic_unbiased: core.hsic_unbiased as f32,
        test_statistic: test_statistic as f32,
        p_value: p_value as f32,
        gamma_shape: shape as f32,
        gamma_scale: scale as f32,
        bandwidth_x: core.sigma_x as f32,
        bandwidth_y: core.sigma_y as f32,
        n_samples: n,
    })
}

/// HSIC independence test via a seeded permutation null — exact-valid at any
/// `n ≥ 4`. The null shuffles Y's sample order (equivalently re-indexes the
/// centred `L_c`, since centering commutes with permutation) and counts how often
/// the permuted `HSIC_b` reaches the observed value; the p-value is the add-one
/// estimator `(1 + #ge)/(1 + P)`.
pub fn hsic_permutation_test(x: &[f32], y: &[f32], config: HsicPermConfig) -> Result<HsicTest> {
    if strict_cuda_requested() {
        return hsic_permutation_test_cuda_strict(x, y, config);
    }
    if config.permutations == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "HSIC permutation test requires permutations > 0",
        ));
    }
    let core = HsicCore::build(x, y, config.kernel)?;
    let n = core.n;
    let observed = core.tr_kc_lc; // ∝ HSIC_b; permutation-invariant denominators
    let tol = 1e-12 * observed.abs().max(1.0);
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut ge_count = 0usize;
    for _ in 0..config.permutations {
        perm.shuffle(&mut rng);
        let mut acc = 0.0f64;
        for i in 0..n {
            let ci = i * n;
            let pi = perm[i] * n;
            for (j, &pj) in perm.iter().enumerate() {
                acc += core.kc[ci + j] * core.lc[pi + pj];
            }
        }
        if acc >= observed - tol {
            ge_count += 1;
        }
    }
    let p_value = (1.0 + ge_count as f64) / (1.0 + config.permutations as f64);
    Ok(HsicTest {
        hsic_biased: core.hsic_biased as f32,
        p_value: p_value as f32,
        permutations: config.permutations,
        ge_count,
        seed: config.seed,
        n_samples: n,
    })
}

pub fn hsic_estimators_cuda_strict(x: &[f32], y: &[f32]) -> Result<HsicEstimators> {
    hsic_estimators_with_config_cuda_strict(x, y, HsicConfig::default())
}

pub fn hsic_estimators_with_config_cuda_strict(
    x: &[f32],
    y: &[f32],
    config: HsicConfig,
) -> Result<HsicEstimators> {
    let (core, sigma_x, sigma_y) = hsic_cuda_core(x, y, config, None)?;
    Ok(HsicEstimators {
        hsic_biased: core.hsic_biased,
        hsic_unbiased: core.hsic_unbiased,
        bandwidth_x: sigma_x as f32,
        bandwidth_y: sigma_y as f32,
        n_samples: core.n_samples,
    })
}

pub fn hsic_cuda_strict(x: &[f32], y: &[f32]) -> Result<HsicReport> {
    hsic_with_config_cuda_strict(x, y, HsicConfig::default())
}

pub fn hsic_with_config_cuda_strict(
    x: &[f32],
    y: &[f32],
    config: HsicConfig,
) -> Result<HsicReport> {
    let (core, sigma_x, sigma_y) = hsic_cuda_core(x, y, config, None)?;
    let n = core.n_samples;
    if n < MIN_HSIC_GAMMA_SAMPLES {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "HSIC gamma test requires at least {MIN_HSIC_GAMMA_SAMPLES} samples (null variance needs n≥6); got {n}. Use hsic_permutation_test for small n"
        )));
    }
    let nf = n as f64;
    let test_statistic = core.tr_kc_lc / nf;
    let mu_x = core.off_diag_sum_k / (nf * (nf - 1.0));
    let mu_y = core.off_diag_sum_l / (nf * (nf - 1.0));
    let mean = (1.0 + mu_x * mu_y - mu_x - mu_y) / nf;
    let var_prefactor = 2.0 * (nf - 4.0) * (nf - 5.0)
        / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0))
        / (nf * (nf - 1.0));
    let var = var_prefactor * core.sum_sq_centered_offdiag;
    if mean.is_nan() || mean <= 0.0 || var.is_nan() || var <= 0.0 {
        return Err(CalyxError::assay_degenerate_input(
            "HSIC gamma test undefined: non-positive null moment (degenerate kernel structure)",
        ));
    }
    let shape = mean * mean / var;
    let scale = var * nf / mean;
    let p_value = gammq(shape, test_statistic / scale)?;
    Ok(HsicReport {
        hsic_biased: core.hsic_biased,
        hsic_unbiased: core.hsic_unbiased,
        test_statistic: test_statistic as f32,
        p_value: p_value as f32,
        gamma_shape: shape as f32,
        gamma_scale: scale as f32,
        bandwidth_x: sigma_x as f32,
        bandwidth_y: sigma_y as f32,
        n_samples: n,
    })
}

pub fn hsic_permutation_test_cuda_strict(
    x: &[f32],
    y: &[f32],
    config: HsicPermConfig,
) -> Result<HsicTest> {
    if config.permutations == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "HSIC permutation test requires permutations > 0",
        ));
    }
    if x.len() != y.len() {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "HSIC requires paired samples: x={} y={}",
            x.len(),
            y.len()
        )));
    }
    let permutations = deterministic_permutations(x.len(), config.permutations, config.seed)?;
    let (core, _, _) = hsic_cuda_core(x, y, config.kernel, Some(&permutations))?;
    let ge_count = core.ge_count.ok_or_else(|| {
        CalyxError::forge_numerical_invariant("HSIC CUDA did not return ge_count")
    })?;
    let p_value = (1.0 + ge_count as f64) / (1.0 + config.permutations as f64);
    Ok(HsicTest {
        hsic_biased: core.hsic_biased,
        p_value: p_value as f32,
        permutations: config.permutations,
        ge_count,
        seed: config.seed,
        n_samples: core.n_samples,
    })
}

#[cfg(test)]
#[path = "hsic_tests.rs"]
mod tests;
