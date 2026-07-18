use super::*;

pub(super) struct HsicCore {
    pub(super) n: usize,
    pub(super) sigma_x: f64,
    pub(super) sigma_y: f64,
    pub(super) hsic_biased: f64,
    pub(super) hsic_unbiased: f64,
    /// Centred Gram matrices `K_c = HKH`, `L_c = HLH` (row-major n×n).
    pub(super) kc: Vec<f64>,
    pub(super) lc: Vec<f64>,
    /// `tr(K_c L_c) = Σ_ij Kc_ij Lc_ij` (so HSIC_b = this / n²).
    pub(super) tr_kc_lc: f64,
    /// `Σ_{i≠j} K_ij` and `Σ_{i≠j} L_ij` on the RAW Gram matrices.
    pub(super) off_diag_sum_k: f64,
    pub(super) off_diag_sum_l: f64,
    /// `Σ_{i≠j} (Kc_ij Lc_ij)²`.
    pub(super) sum_sq_centered_offdiag: f64,
}

impl HsicCore {
    pub(super) fn build(x: &[f32], y: &[f32], config: HsicConfig) -> Result<Self> {
        if x.len() != y.len() {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "HSIC requires paired samples: x={} y={}",
                x.len(),
                y.len()
            )));
        }
        let n = x.len();
        if n < MIN_HSIC_SAMPLES {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "HSIC requires at least {MIN_HSIC_SAMPLES} paired samples; got {n}"
            )));
        }
        let xd = to_finite_f64("x", x)?;
        let yd = to_finite_f64("y", y)?;
        let sigma_x = resolve_bandwidth("x", &xd, config.bandwidth_x)?;
        let sigma_y = resolve_bandwidth("y", &yd, config.bandwidth_y)?;

        let k = gaussian_gram(&xd, sigma_x);
        let l = gaussian_gram(&yd, sigma_y);
        let off_diag_sum_k = off_diagonal_sum(&k, n);
        let off_diag_sum_l = off_diagonal_sum(&l, n);

        let kc = double_center(&k, n);
        let lc = double_center(&l, n);

        // tr(K_c L_c) and Σ_{i≠j}(Kc·Lc)².
        let mut tr_kc_lc = 0.0f64;
        let mut sum_sq_centered_offdiag = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let prod = kc[i * n + j] * lc[i * n + j];
                tr_kc_lc += prod;
                if i != j {
                    sum_sq_centered_offdiag += prod * prod;
                }
            }
        }
        let nf = n as f64;
        let hsic_biased = (tr_kc_lc / (nf * nf)).max(0.0);

        // Unbiased estimator from diagonal-zeroed raw Grams.
        let hsic_unbiased = unbiased_hsic(&k, &l, n);

        Ok(Self {
            n,
            sigma_x,
            sigma_y,
            hsic_biased,
            hsic_unbiased,
            kc,
            lc,
            tr_kc_lc,
            off_diag_sum_k,
            off_diag_sum_l,
            sum_sq_centered_offdiag,
        })
    }
}

/// Unbiased HSIC (Song et al. 2012) from raw Gram matrices; diagonals treated as
/// zero. `n ≥ 4` is guaranteed by the caller.
fn unbiased_hsic(k: &[f64], l: &[f64], n: usize) -> f64 {
    let nf = n as f64;
    let mut tr = 0.0f64; // Σ_{i≠j} K̃_ij L̃_ij
    let mut sum_k = 0.0f64; // 1ᵀK̃1
    let mut sum_l = 0.0f64; // 1ᵀL̃1
    // Row sums of K̃ and L̃ for 1ᵀK̃L̃1 = Σ_k rowK̃_k · rowL̃_k.
    let mut row_k = vec![0.0f64; n];
    let mut row_l = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let kij = k[i * n + j];
            let lij = l[i * n + j];
            tr += kij * lij;
            sum_k += kij;
            sum_l += lij;
            row_k[i] += kij;
            row_l[i] += lij;
        }
    }
    let one_kl_one: f64 = (0..n).map(|i| row_k[i] * row_l[i]).sum();
    (tr + sum_k * sum_l / ((nf - 1.0) * (nf - 2.0)) - 2.0 / (nf - 2.0) * one_kl_one)
        / (nf * (nf - 3.0))
}

/// Gaussian RBF Gram matrix (row-major n×n) with bandwidth `sigma`.
fn gaussian_gram(v: &[f64], sigma: f64) -> Vec<f64> {
    let n = v.len();
    let denom = 2.0 * sigma * sigma;
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        g[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let d = v[i] - v[j];
            let val = (-(d * d) / denom).exp();
            g[i * n + j] = val;
            g[j * n + i] = val;
        }
    }
    g
}

/// Row-major double-centred matrix `K_c = HKH` via row/col/grand means (O(n²)).
fn double_center(k: &[f64], n: usize) -> Vec<f64> {
    let nf = n as f64;
    let mut row = vec![0.0f64; n];
    for (i, r) in row.iter_mut().enumerate() {
        let mut s = 0.0;
        for j in 0..n {
            s += k[i * n + j];
        }
        *r = s / nf;
    }
    let grand = row.iter().sum::<f64>() / nf;
    let mut kc = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            // K symmetric ⇒ column mean_j == row mean_j.
            kc[i * n + j] = k[i * n + j] - row[i] - row[j] + grand;
        }
    }
    kc
}

fn off_diagonal_sum(g: &[f64], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s += g[i * n + j];
            }
        }
    }
    s
}

/// Resolve the RBF bandwidth: a caller-pinned value, else the median-distance
/// heuristic `σ = √(median{(x_i−x_j)² : i<j, distinct}/2)`. Fails closed when the
/// series is constant (all distances zero ⇒ undefined bandwidth).
pub(super) fn resolve_bandwidth(name: &str, v: &[f64], pinned: Option<f64>) -> Result<f64> {
    if let Some(s) = pinned {
        if !(s.is_finite() && s > 0.0) {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "HSIC {name} bandwidth must be finite and positive, got {s}"
            )));
        }
        return Ok(s);
    }
    let n = v.len();
    let mut sq = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = v[i] - v[j];
            if d != 0.0 {
                sq.push(d * d);
            }
        }
    }
    if sq.is_empty() {
        return Err(CalyxError::assay_degenerate_input(format!(
            "HSIC undefined: {name} is constant (zero median distance ⇒ undefined bandwidth)"
        )));
    }
    let med = median(&mut sq);
    Ok((0.5 * med).sqrt())
}

/// Median of a slice (mutates via sort). Non-empty guaranteed by the caller.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite-validated"));
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        0.5 * (v[m / 2 - 1] + v[m / 2])
    }
}

pub(super) fn to_finite_f64(name: &str, values: &[f32]) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(values.len());
    for (idx, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "HSIC {name}[{idx}] is not finite ({v})"
            )));
        }
        out.push(v as f64);
    }
    Ok(out)
}
