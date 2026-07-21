use super::*;

pub(super) fn estimate_cuda_strict(
    x: &[Vec<f32>],
    labels: &[usize],
    k: usize,
    trust: TrustTag,
) -> Result<MiEstimate> {
    #[cfg(feature = "cuda")]
    {
        let class_counts = validate_classes(labels, k)?;
        validate_radius_defined(x, labels, k)?;
        let backend = calyx_forge::CudaBackend::new()
            .map_err(|err| crate::cuda_strict::forge_to_calyx("mixed KSG", err))?;
        let raw_bits =
            raw_bits_from_validated_samples_cuda(backend.context(), x, labels, k, &class_counts)?;
        let ci = subsample_ci_cuda(
            backend.context(),
            x,
            labels,
            raw_bits,
            k,
            MIXED_KSG_CI_CONFIG,
        )?;
        finalize_estimate(raw_bits, ci, x.len(), trust)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x, labels, k, trust);
        Err(crate::cuda_strict::cuda_unavailable("mixed KSG"))
    }
}

#[cfg(feature = "cuda")]
fn raw_bits_from_validated_samples_cuda(
    ctx: &calyx_forge::CudaContext,
    x: &[Vec<f32>],
    labels: &[usize],
    k: usize,
    class_counts: &BTreeMap<usize, usize>,
) -> Result<f64> {
    let dim = x.first().map_or(0, Vec::len);
    let flat = flatten_matrix(x)?;
    let labels_i32 = labels_to_i32(labels)?;
    let counts = calyx_forge::mixed_ksg_counts_host(ctx, &flat, &labels_i32, x.len(), dim, k)
        .map_err(|err| crate::cuda_strict::forge_to_calyx("mixed KSG", err))?;
    if counts.same_class_counts.len() != x.len() || counts.full_counts.len() != x.len() {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "mixed KSG CUDA count readback length mismatch: n={} same={} full={}",
            x.len(),
            counts.same_class_counts.len(),
            counts.full_counts.len()
        )));
    }
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        let same_class_count = counts.same_class_counts[i];
        let full_count = counts.full_counts[i];
        if same_class_count == 0 || full_count == 0 {
            return Err(CalyxError::forge_numerical_invariant(format!(
                "mixed KSG CUDA returned zero count at row {i}: same={same_class_count} full={full_count}"
            )));
        }
        let class_count = class_counts[&labels[i]];
        total += digamma(n as f64) + digamma(same_class_count as f64)
            - digamma(class_count as f64)
            - digamma(full_count as f64);
    }
    let bits = total / n as f64 / std::f64::consts::LN_2;
    if !bits.is_finite() {
        return Err(CalyxError::forge_numerical_invariant(
            "mixed KSG CUDA produced non-finite raw bits",
        ));
    }
    Ok(bits)
}

#[cfg(feature = "cuda")]
fn labels_to_i32(labels: &[usize]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(labels.len());
    for (idx, label) in labels.iter().copied().enumerate() {
        let label = i32::try_from(label).map_err(|_| {
            CalyxError::assay_insufficient_samples(format!(
                "mixed KSG label at row {idx} exceeds CUDA i32 label range: {label}"
            ))
        })?;
        out.push(label);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn flatten_matrix(values: &[Vec<f32>]) -> Result<Vec<f32>> {
    let dim = values.first().map_or(0, Vec::len);
    let len = values
        .len()
        .checked_mul(dim)
        .ok_or_else(|| CalyxError::forge_vram_budget("mixed KSG flat matrix length overflow"))?;
    let mut flat = Vec::with_capacity(len);
    for row in values {
        flat.extend_from_slice(row);
    }
    Ok(flat)
}

#[cfg(feature = "cuda")]
fn subsample_ci_cuda(
    ctx: &calyx_forge::CudaContext,
    x: &[Vec<f32>],
    labels: &[usize],
    raw_point_estimate: f64,
    k: usize,
    config: BootstrapConfig,
) -> Result<MixedKsgCi> {
    if config.resamples == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "mixed continuous-discrete KSG no-replacement CI requires at least one resample",
        ));
    }
    if !raw_point_estimate.is_finite() {
        return Err(CalyxError::assay_low_signal(
            "mixed continuous-discrete KSG full-sample estimate is non-finite",
        ));
    }
    let full_counts = validate_classes(labels, k)?;
    let plan = subsample_plan(x.len(), k, &full_counts)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut roots = Vec::with_capacity(config.resamples);
    let mut rejected_draws = 0usize;
    let root_scale = finite_population_root_scale(x.len(), plan.m)?;
    for _ in 0..config.resamples {
        let (indices, rejected) =
            sample_indices_with_rejections(labels, plan.m, k, MAX_DRAW_ATTEMPTS, &mut rng)?;
        rejected_draws = rejected_draws.saturating_add(rejected);
        let sampled_x = indices
            .iter()
            .map(|index| x[*index].clone())
            .collect::<Vec<_>>();
        let sampled_labels = indices
            .iter()
            .map(|index| labels[*index])
            .collect::<Vec<_>>();
        let sampled_counts = validate_classes(&sampled_labels, k)?;
        validate_radius_defined(&sampled_x, &sampled_labels, k)?;
        let sampled_raw = raw_bits_from_validated_samples_cuda(
            ctx,
            &sampled_x,
            &sampled_labels,
            k,
            &sampled_counts,
        )?;
        let root = root_scale * (sampled_raw - raw_point_estimate);
        if !sampled_raw.is_finite() || !root.is_finite() {
            return Err(CalyxError::assay_low_signal(
                "mixed continuous-discrete KSG produced a non-finite CUDA subsampling root",
            ));
        }
        roots.push(root);
    }
    Ok(MixedKsgCi {
        interval: ci_from_roots(roots, raw_point_estimate, x.len())?,
        plan,
        rejected_draws,
    })
}
