use super::*;

pub fn dcor_1d_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    permutations: Option<&[i32]>,
) -> Result<CudaDcorResult> {
    validate_pair_f32("dcor_1d_host", x, y, 4)?;
    let n = x.len();
    let perm_count = validate_permutations(permutations, n)?;
    let matrix_len = checked_square(n, "dCor matrix length")?;
    let stat_blocks = reduction_blocks(matrix_len);
    ensure_device_room(
        ctx,
        "dcor_1d_host",
        checked_sum_bytes(&[
            bytes::<f32>(2 * n, "dCor inputs")?,
            bytes::<f64>(2 * matrix_len, "dCor centered matrices")?,
            bytes::<f64>(2 * n, "dCor row sums")?,
            bytes::<f64>(3 * stat_blocks, "dCor reduction partials")?,
            bytes::<i32>(perm_count * n, "dCor permutations")?,
            bytes::<f64>(perm_count, "dCor permutation outputs")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("dCor x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("dCor y upload failed: {err}")))?;
    let (a, row_a) = centered_abs_1d(ctx, &x_dev, n, "dcor_x")?;
    let (b, row_b) = centered_abs_1d(ctx, &y_dev, n, "dcor_y")?;
    drop(row_a);
    drop(row_b);

    let (dcov_sum, vx_sum, vy_sum) = dcor_stats(ctx, &a, &b, n)?;
    let denom_n = (n as f64) * (n as f64);
    let dcov2 = (dcov_sum / denom_n).max(0.0);
    let dvar_x = (vx_sum / denom_n).max(0.0);
    let dvar_y = (vy_sum / denom_n).max(0.0);
    let denom = (dvar_x * dvar_y).sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        return Err(numerical(
            "dcor_1d_host",
            "dCor undefined on zero distance-variance input".to_string(),
        ));
    }
    let dcor = (dcov2 / denom).clamp(0.0, 1.0).sqrt();
    let ge_count = if let Some(perms) = permutations {
        let perm_stats = dcor_permutation_stats(ctx, &a, &b, n, perms, perm_count)?;
        let tolerance = 1e-12 * dcov2.abs().max(1.0);
        Some(count_ge(ctx, &perm_stats, perm_count, dcov2, tolerance)?)
    } else {
        None
    };

    Ok(CudaDcorResult {
        dcor: dcor as f32,
        dcov2: dcov2 as f32,
        dvar_x: dvar_x as f32,
        dvar_y: dvar_y as f32,
        n_samples: n,
        ge_count,
    })
}

pub fn hsic_1d_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    sigma_x: f64,
    sigma_y: f64,
    permutations: Option<&[i32]>,
) -> Result<CudaHsicResult> {
    validate_pair_f32("hsic_1d_host", x, y, 4)?;
    validate_bandwidth("hsic sigma_x", sigma_x)?;
    validate_bandwidth("hsic sigma_y", sigma_y)?;
    let n = x.len();
    let perm_count = validate_permutations(permutations, n)?;
    let matrix_len = checked_square(n, "HSIC matrix length")?;
    let stat_blocks = reduction_blocks(matrix_len);
    ensure_device_room(
        ctx,
        "hsic_1d_host",
        checked_sum_bytes(&[
            bytes::<f32>(2 * n, "HSIC inputs")?,
            bytes::<f64>(2 * matrix_len, "HSIC centered Gram matrices")?,
            bytes::<f64>(2 * n, "HSIC row sums")?,
            bytes::<f64>(3 * stat_blocks, "HSIC reduction partials")?,
            bytes::<i32>(perm_count * n, "HSIC permutations")?,
            bytes::<f64>(perm_count, "HSIC permutation outputs")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("HSIC x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("HSIC y upload failed: {err}")))?;
    let (mut kc, row_k) = rbf_matrix_1d(ctx, &x_dev, n, sigma_x, "hsic_x")?;
    let (mut lc, row_l) = rbf_matrix_1d(ctx, &y_dev, n, sigma_y, "hsic_y")?;
    let (raw_trace_with_diag, _, _) = dcor_stats(ctx, &kc, &lc, n)?;
    let tr_raw_diag_zero = raw_trace_with_diag - n as f64;
    let total_k = reduce_sum(ctx, &row_k, n, "hsic_total_k")?;
    let total_l = reduce_sum(ctx, &row_l, n, "hsic_total_l")?;
    center_matrix(ctx, &mut kc, &row_k, total_k, n, "hsic_x")?;
    center_matrix(ctx, &mut lc, &row_l, total_l, n, "hsic_y")?;

    let stats = hsic_stats(ctx, &kc, &lc, &row_k, &row_l, n)?;
    let nf = n as f64;
    let hsic_biased = (stats.tr_kc_lc / (nf * nf)).max(0.0);
    let hsic_unbiased = (tr_raw_diag_zero
        + stats.off_diag_sum_k * stats.off_diag_sum_l / ((nf - 1.0) * (nf - 2.0))
        - 2.0 / (nf - 2.0) * stats.one_kl_one)
        / (nf * (nf - 3.0));
    let ge_count = if let Some(perms) = permutations {
        let perm_stats = hsic_permutation_stats(ctx, &kc, &lc, n, perms, perm_count)?;
        let tolerance = 1e-12 * stats.tr_kc_lc.abs().max(1.0);
        Some(count_ge(
            ctx,
            &perm_stats,
            perm_count,
            stats.tr_kc_lc,
            tolerance,
        )?)
    } else {
        None
    };

    Ok(CudaHsicResult {
        hsic_biased: hsic_biased as f32,
        hsic_unbiased: hsic_unbiased as f32,
        tr_kc_lc: stats.tr_kc_lc,
        off_diag_sum_k: stats.off_diag_sum_k,
        off_diag_sum_l: stats.off_diag_sum_l,
        sum_sq_centered_offdiag: stats.sum_sq_centered_offdiag,
        n_samples: n,
        ge_count,
    })
}
