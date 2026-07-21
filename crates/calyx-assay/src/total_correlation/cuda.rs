use super::*;

#[cfg(feature = "cuda")]
use crate::ksg::ksg_mi_continuous_point_cuda_with_context;

#[cfg(feature = "cuda")]
pub(super) fn total_correlation_with_config_cuda_strict_impl(
    slots: &SlotVectors,
    clock: &dyn Clock,
    config: &TotalCorrelationConfig,
) -> Result<TCResult> {
    validate_config(config)?;
    let n_samples = validate_panel(slots)?;
    let slot_count = slots.len();
    if below_tc_quorum(n_samples, slot_count) {
        return Ok(provisional_tc(slot_count, n_samples, clock));
    }
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("total correlation", err))?;
    let estimate = estimate_total_correlation_cuda(backend.context(), slots, config.k)?;
    let ci_95 = if slot_count <= 1 {
        (0.0, 0.0)
    } else {
        bootstrap_tc_ci_cuda(
            backend.context(),
            slots,
            estimate.tc,
            config,
            seed_for_slots(slots, config),
        )?
    };
    Ok(TCResult {
        tc: estimate.tc,
        n_eff: estimate.n_eff,
        ci_95,
        n_samples,
        slot_count,
        sum_marginal_entropy: estimate.sum_marginal_entropy,
        joint_entropy: estimate.joint_entropy,
        provisional: false,
        error_code: None,
        trust: TrustTag::Provisional,
        computed_at: clock.now(),
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn total_correlation_with_config_cuda_strict_impl(
    _slots: &SlotVectors,
    _clock: &dyn Clock,
    _config: &TotalCorrelationConfig,
) -> Result<TCResult> {
    Err(crate::cuda_strict::cuda_unavailable("total correlation"))
}

#[cfg(feature = "cuda")]
pub(super) fn interaction_information_with_config_cuda_strict_impl(
    slot_a: &[f32],
    slot_b: &[f32],
    slot_c: &[f32],
    clock: &dyn Clock,
    config: &TotalCorrelationConfig,
) -> Result<IIResult> {
    validate_config(config)?;
    let n_samples = validate_triple(slot_a, slot_b, slot_c)?;
    if n_samples < MIN_QUORUM_TC_PER_SLOT * 3 || n_samples < MIN_ASSAY_SAMPLES {
        return Ok(provisional_ii(n_samples, clock));
    }
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("interaction information", err))?;
    let point =
        estimate_interaction_information_cuda(backend.context(), slot_a, slot_b, slot_c, config.k)?;
    let ci_95 = bootstrap_ii_ci_cuda(
        backend.context(),
        slot_a,
        slot_b,
        slot_c,
        point,
        config,
        seed_for_triple(slot_a, slot_b, slot_c, config),
    )?;
    Ok(IIResult {
        ii: point,
        sign: ii_sign(ci_95),
        ci_95,
        n_samples,
        provisional: false,
        error_code: None,
        trust: TrustTag::Provisional,
        computed_at: clock.now(),
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn interaction_information_with_config_cuda_strict_impl(
    _slot_a: &[f32],
    _slot_b: &[f32],
    _slot_c: &[f32],
    _clock: &dyn Clock,
    _config: &TotalCorrelationConfig,
) -> Result<IIResult> {
    Err(crate::cuda_strict::cuda_unavailable(
        "interaction information",
    ))
}

#[cfg(feature = "cuda")]
fn estimate_total_correlation_cuda(
    ctx: &calyx_forge::CudaContext,
    slots: &SlotVectors,
    k: usize,
) -> Result<TCEstimate> {
    if slots.len() <= 1 {
        let joint_entropy = slots
            .first()
            .map(|slot| entropy_bits_ksg_cuda(ctx, &one_dim(slot), k))
            .transpose()?
            .unwrap_or(0.0);
        return Ok(TCEstimate {
            tc: 0.0,
            n_eff: slots.len() as f32,
            sum_marginal_entropy: joint_entropy,
            joint_entropy,
        });
    }
    let mut sum_marginal_entropy = 0.0;
    for slot in slots {
        sum_marginal_entropy += entropy_bits_ksg_cuda(ctx, &one_dim(slot), k)?;
    }
    let joint = joint_matrix(slots);
    let joint_entropy = entropy_bits_ksg_cuda(ctx, &joint, k)?;
    let tc = (sum_marginal_entropy - joint_entropy).max(0.0);
    Ok(TCEstimate {
        tc,
        n_eff: n_eff_from_tc(slots.len(), tc, sum_marginal_entropy),
        sum_marginal_entropy,
        joint_entropy,
    })
}

#[cfg(feature = "cuda")]
fn estimate_interaction_information_cuda(
    ctx: &calyx_forge::CudaContext,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    k: usize,
) -> Result<f32> {
    let a = one_dim(a);
    let b = one_dim(b);
    let c = one_dim(c);
    let bc: Vec<_> = b
        .iter()
        .zip(&c)
        .map(|(left, right)| vec![left[0], right[0]])
        .collect();
    let i_ab = ksg_mi_continuous_point_cuda_with_context(ctx, &a, &b, k)?;
    let i_a_bc = ksg_mi_continuous_point_cuda_with_context(ctx, &a, &bc, k)?;
    let i_ac = ksg_mi_continuous_point_cuda_with_context(ctx, &a, &c, k)?;
    let conditional = (i_a_bc - i_ac).max(0.0);
    Ok(i_ab - conditional)
}

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
fn bootstrap_tc_ci_cuda(
    ctx: &calyx_forge::CudaContext,
    slots: &SlotVectors,
    point: f32,
    config: &TotalCorrelationConfig,
    seed: u64,
) -> Result<(f32, f32)> {
    let m = m_out_of_n_size(slots[0].len(), config.k, MIN_ASSAY_SAMPLES, "TC")?;
    let columns = slots.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut estimates = Vec::with_capacity(config.bootstrap_resamples);
    for _ in 0..config.bootstrap_resamples {
        let sampled = sample_paired_values_without_replacement(&columns, m, &mut rng)?;
        estimates.push(estimate_total_correlation_cuda(ctx, &sampled, config.k)?.tc);
    }
    Ok(percentile_ci(estimates, point))
}

#[cfg(feature = "cuda")]
fn bootstrap_ii_ci_cuda(
    ctx: &calyx_forge::CudaContext,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    point: f32,
    config: &TotalCorrelationConfig,
    seed: u64,
) -> Result<(f32, f32)> {
    let m = m_out_of_n_size(a.len(), config.k, MIN_ASSAY_SAMPLES, "II")?;
    let columns = [a, b, c];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut estimates = Vec::with_capacity(config.bootstrap_resamples);
    for _ in 0..config.bootstrap_resamples {
        let sampled = sample_paired_values_without_replacement(&columns, m, &mut rng)?;
        estimates.push(estimate_interaction_information_cuda(
            ctx,
            &sampled[0],
            &sampled[1],
            &sampled[2],
            config.k,
        )?);
    }
    Ok(percentile_ci(estimates, point))
}

#[cfg(feature = "cuda")]
fn flatten_matrix(values: &[Vec<f32>]) -> Result<Vec<f32>> {
    let dim = values.first().map_or(0, Vec::len);
    let len = values
        .len()
        .checked_mul(dim)
        .ok_or_else(|| CalyxError::forge_vram_budget("KSG entropy flat matrix length overflow"))?;
    let mut flat = Vec::with_capacity(len);
    for row in values {
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

#[cfg(feature = "cuda")]
pub(super) fn entropy_bits_ksg_cuda(
    ctx: &calyx_forge::CudaContext,
    samples: &[Vec<f32>],
    k: usize,
) -> Result<f32> {
    let dim = validate_rectangular_finite("entropy samples", samples)?;
    let n = samples.len();
    if n < MIN_ASSAY_SAMPLES || k == 0 || k >= n {
        return Err(insufficient(format!(
            "KSG entropy requires at least {MIN_ASSAY_SAMPLES} samples and 0 < k < n; got n={n}, k={k}"
        )));
    }
    let flat = flatten_matrix(samples)?;
    let radii = calyx_forge::entropy_radii_host(ctx, &flat, n, dim, k)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("KSG entropy", err))?;
    entropy_bits_from_radii(n, dim, k, &radii)
}

#[cfg(feature = "cuda")]
fn entropy_bits_from_radii(n: usize, dim: usize, k: usize, radii: &[f32]) -> Result<f32> {
    if radii.len() != n {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "KSG entropy CUDA radius readback length mismatch: n={n} radii={}",
            radii.len()
        )));
    }
    let mut log_radius_sum = 0.0;
    for (idx, radius) in radii.iter().copied().enumerate() {
        if radius == 0.0 {
            return Err(CalyxError::assay_degenerate_input(format!(
                "KSG entropy CUDA kth radius is zero for sample {idx}: k={k}"
            )));
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(CalyxError::forge_numerical_invariant(format!(
                "KSG entropy CUDA radius is invalid at sample {idx}: {radius}"
            )));
        }
        log_radius_sum += radius.ln() as f64;
    }
    let mean_log_radius = log_radius_sum / n as f64;
    let dim = dim as f64;
    let h_nats =
        digamma(n as f64) - digamma(k as f64) + dim * (std::f64::consts::LN_2 + mean_log_radius);
    let bits = (h_nats / std::f64::consts::LN_2) as f32;
    if !bits.is_finite() {
        return Err(CalyxError::forge_numerical_invariant(
            "KSG entropy CUDA produced non-finite entropy bits",
        ));
    }
    Ok(bits)
}
