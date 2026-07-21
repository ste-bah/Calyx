use super::*;

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct CcmSeries<'a> {
    name: &'a str,
    values: &'a [f32],
}

#[cfg(feature = "cuda")]
pub(super) fn convergent_cross_mapping_cuda_strict_impl(
    x_name: &str,
    x: &[f32],
    y_name: &str,
    y: &[f32],
    config: &CcmConfig,
) -> Result<CcmReport> {
    let effective_points = validate_ccm_inputs(x_name, x, y_name, y, config)?;
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("CCM", err))?;
    let x_series = CcmSeries {
        name: x_name,
        values: x,
    };
    let y_series = CcmSeries {
        name: y_name,
        values: y,
    };
    let x_to_y = cross_map_direction_cuda(backend.context(), x_series, y_series, config)?;
    let y_to_x = cross_map_direction_cuda(backend.context(), y_series, x_series, config)?;
    let verdict = ccm_verdict(
        &x_to_y,
        &y_to_x,
        config.min_convergence_delta,
        config.min_skill_gap,
    );

    Ok(CcmReport {
        estimator: "convergent_cross_mapping_simplex_cuda_strict".to_string(),
        x_name: x_name.to_string(),
        y_name: y_name.to_string(),
        embedding_dim: config.embedding_dim,
        tau: config.tau,
        neighbor_count: config.embedding_dim + 1,
        n_samples: x.len(),
        effective_points,
        min_convergence_delta: config.min_convergence_delta,
        min_skill_gap: config.min_skill_gap,
        x_manifold_to_y: x_to_y,
        y_manifold_to_x: y_to_x,
        verdict,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn convergent_cross_mapping_cuda_strict_impl(
    _x_name: &str,
    _x: &[f32],
    _y_name: &str,
    _y: &[f32],
    _config: &CcmConfig,
) -> Result<CcmReport> {
    Err(crate::cuda_strict::cuda_unavailable("CCM"))
}

#[cfg(feature = "cuda")]
fn cross_map_direction_cuda(
    ctx: &calyx_forge::CudaContext,
    manifold: CcmSeries<'_>,
    target: CcmSeries<'_>,
    config: &CcmConfig,
) -> Result<CcmDirectionReport> {
    let start = (config.embedding_dim - 1) * config.tau;
    let embedding = delay_embedding(manifold.values, config.embedding_dim, config.tau);
    let aligned_target: Vec<f32> = target.values[start..].to_vec();
    let flat = flatten_embedding(&embedding)?;
    let predictions = calyx_forge::ccm_simplex_predictions_host(
        ctx,
        &flat,
        &aligned_target,
        embedding.len(),
        config.embedding_dim,
        config.embedding_dim + 1,
        &config.library_sizes,
    )
    .map_err(|err| crate::cuda_strict::forge_to_calyx("CCM", err))?;
    if predictions.library_predictions.len() != config.library_sizes.len() {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "CCM CUDA returned {} library prediction groups for {} requested sizes",
            predictions.library_predictions.len(),
            config.library_sizes.len()
        )));
    }
    let mut library_skills = Vec::with_capacity(config.library_sizes.len());
    for (&library_size, predicted) in config
        .library_sizes
        .iter()
        .zip(predictions.library_predictions.iter())
    {
        if predicted.len() != library_size {
            return Err(CalyxError::forge_numerical_invariant(format!(
                "CCM CUDA prediction length mismatch for library_size={library_size}: got {}",
                predicted.len()
            )));
        }
        library_skills.push(CcmLibrarySkill {
            library_size,
            rho: pearson_r(predicted, &aligned_target[..library_size])?,
        });
    }
    let first = library_skills[0].rho;
    let final_rho = library_skills[library_skills.len() - 1].rho;
    Ok(CcmDirectionReport {
        manifold: manifold.name.to_string(),
        target: target.name.to_string(),
        library_skills,
        final_rho,
        convergence_delta: final_rho - first,
    })
}
