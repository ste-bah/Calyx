use super::*;

pub(super) struct HsicDeviceStats {
    pub(super) tr_kc_lc: f64,
    pub(super) sum_sq_centered_offdiag: f64,
    pub(super) off_diag_sum_k: f64,
    pub(super) off_diag_sum_l: f64,
    pub(super) one_kl_one: f64,
}

pub(super) fn centered_abs_1d(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    n: usize,
    op: &'static str,
) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
    let (mut matrix, row_sums) = abs_matrix_1d(ctx, values, n, op)?;
    let total = reduce_sum(ctx, &row_sums, n, op)?;
    center_matrix(ctx, &mut matrix, &row_sums, total, n, op)?;
    Ok((matrix, row_sums))
}

pub(super) fn abs_matrix_1d(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    n: usize,
    op: &'static str,
) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
    let n_i32 = to_i32(n, "assay n")?;
    let len = checked_square(n, "assay matrix length")?;
    let stream = ctx.inner().default_stream();
    let mut matrix: CudaSlice<f64> = stream
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{op} matrix allocation failed: {err}")))?;
    let mut row_sums: CudaSlice<f64> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("{op} row-sum allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, op)?;
    let func = assay_function(
        ctx,
        "assay.pairwise_abs_1d_f32",
        "assay_pairwise_abs_1d_f32",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "assay rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(values)
            .arg(&n_i32)
            .arg(&mut matrix)
            .arg(&mut row_sums)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("{op} pairwise abs launch failed: {err}")))?;
    sync_and_decode(ctx, op, &flags)?;
    Ok((matrix, row_sums))
}

pub(super) fn rbf_matrix_1d(
    ctx: &CudaContext,
    values: &CudaSlice<f32>,
    n: usize,
    sigma: f64,
    op: &'static str,
) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
    let n_i32 = to_i32(n, "assay n")?;
    let len = checked_square(n, "assay matrix length")?;
    let stream = ctx.inner().default_stream();
    let mut matrix: CudaSlice<f64> = stream
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{op} matrix allocation failed: {err}")))?;
    let mut row_sums: CudaSlice<f64> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("{op} row-sum allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, op)?;
    let func = assay_function(
        ctx,
        "assay.pairwise_rbf_1d_f32",
        "assay_pairwise_rbf_1d_f32",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "assay rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(values)
            .arg(&n_i32)
            .arg(&sigma)
            .arg(&mut matrix)
            .arg(&mut row_sums)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("{op} RBF launch failed: {err}")))?;
    sync_and_decode(ctx, op, &flags)?;
    Ok((matrix, row_sums))
}

pub(super) fn rbf_matrix_f64(
    ctx: &CudaContext,
    values: &CudaSlice<f64>,
    n: usize,
    dim: usize,
    sigma: f64,
    op: &'static str,
) -> Result<(CudaSlice<f64>, CudaSlice<f64>)> {
    let n_i32 = to_i32(n, "assay n")?;
    let dim_i32 = to_i32(dim, "assay dim")?;
    let len = checked_square(n, "assay matrix length")?;
    let stream = ctx.inner().default_stream();
    let mut matrix: CudaSlice<f64> = stream
        .alloc_zeros(len)
        .map_err(|err| device_unavailable(ctx, format!("{op} matrix allocation failed: {err}")))?;
    let mut row_sums: CudaSlice<f64> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("{op} row-sum allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, op)?;
    let func = assay_function(ctx, "assay.pairwise_rbf_f64", "assay_pairwise_rbf_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "assay rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(values)
            .arg(&n_i32)
            .arg(&dim_i32)
            .arg(&sigma)
            .arg(&mut matrix)
            .arg(&mut row_sums)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("{op} RBF f64 launch failed: {err}")))?;
    sync_and_decode(ctx, op, &flags)?;
    Ok((matrix, row_sums))
}

pub(super) fn center_matrix(
    ctx: &CudaContext,
    matrix: &mut CudaSlice<f64>,
    row_sums: &CudaSlice<f64>,
    total: f64,
    n: usize,
    op: &'static str,
) -> Result<()> {
    let n_i32 = to_i32(n, "assay n")?;
    let len = checked_square(n, "assay matrix length")?;
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, op)?;
    let func = assay_function(
        ctx,
        "assay.center_symmetric_f64",
        "assay_center_symmetric_f64",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(len)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(matrix)
            .arg(row_sums)
            .arg(&total)
            .arg(&n_i32)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("{op} centering launch failed: {err}")))?;
    sync_and_decode(ctx, op, &flags)
}

pub(super) fn dcor_stats(
    ctx: &CudaContext,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    n: usize,
) -> Result<(f64, f64, f64)> {
    let n_i32 = to_i32(n, "dCor n")?;
    let matrix_len = checked_square(n, "dCor matrix length")?;
    let blocks = reduction_blocks(matrix_len);
    let stream = ctx.inner().default_stream();
    let mut partial_dcov: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("dCor dcov partial allocation failed: {err}"))
    })?;
    let mut partial_vx: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("dCor vx partial allocation failed: {err}"))
    })?;
    let mut partial_vy: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("dCor vy partial allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "dcor_stats")?;
    let func = assay_function(ctx, "assay.dcor_stats_f64", "assay_dcor_stats_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(blocks, "dCor reduction blocks")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(a)
            .arg(b)
            .arg(&n_i32)
            .arg(&mut partial_dcov)
            .arg(&mut partial_vx)
            .arg(&mut partial_vy)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("dCor stats launch failed: {err}")))?;
    sync_and_decode(ctx, "dcor_stats", &flags)?;
    Ok((
        reduce_sum(ctx, &partial_dcov, blocks, "dcor_dcov")?,
        reduce_sum(ctx, &partial_vx, blocks, "dcor_vx")?,
        reduce_sum(ctx, &partial_vy, blocks, "dcor_vy")?,
    ))
}

pub(super) fn hsic_stats(
    ctx: &CudaContext,
    kc: &CudaSlice<f64>,
    lc: &CudaSlice<f64>,
    row_k: &CudaSlice<f64>,
    row_l: &CudaSlice<f64>,
    n: usize,
) -> Result<HsicDeviceStats> {
    let n_i32 = to_i32(n, "HSIC n")?;
    let matrix_len = checked_square(n, "HSIC matrix length")?;
    let blocks = reduction_blocks(matrix_len);
    let stream = ctx.inner().default_stream();
    let mut partial_tr: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("HSIC tr partial allocation failed: {err}"))
    })?;
    let mut partial_sq: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("HSIC sq partial allocation failed: {err}"))
    })?;
    let mut partial_one: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(
            ctx,
            format!("HSIC oneKLone partial allocation failed: {err}"),
        )
    })?;
    let mut flags = alloc_flags(ctx, "hsic_stats")?;
    let func = assay_function(ctx, "assay.hsic_stats_f64", "assay_hsic_stats_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(blocks, "HSIC reduction blocks")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(kc)
            .arg(lc)
            .arg(row_k)
            .arg(row_l)
            .arg(&n_i32)
            .arg(&mut partial_tr)
            .arg(&mut partial_sq)
            .arg(&mut partial_one)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("HSIC stats launch failed: {err}")))?;
    sync_and_decode(ctx, "hsic_stats", &flags)?;

    let total_k = reduce_sum(ctx, row_k, n, "hsic_row_k")?;
    let total_l = reduce_sum(ctx, row_l, n, "hsic_row_l")?;
    let off_diag_sum_k = total_k - n as f64;
    let off_diag_sum_l = total_l - n as f64;
    Ok(HsicDeviceStats {
        tr_kc_lc: reduce_sum(ctx, &partial_tr, blocks, "hsic_tr")?,
        sum_sq_centered_offdiag: reduce_sum(ctx, &partial_sq, blocks, "hsic_sq")?,
        off_diag_sum_k,
        off_diag_sum_l,
        one_kl_one: reduce_sum(ctx, &partial_one, blocks, "hsic_one")?,
    })
}

pub(super) fn dcor_permutation_stats(
    ctx: &CudaContext,
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    n: usize,
    permutations: &[i32],
    perm_count: usize,
) -> Result<CudaSlice<f64>> {
    let n_i32 = to_i32(n, "dCor n")?;
    let perm_i32 = to_i32(perm_count, "dCor permutations")?;
    let stream = ctx.inner().default_stream();
    let perms_dev = stream.clone_htod(permutations).map_err(|err| {
        device_unavailable(ctx, format!("dCor permutations upload failed: {err}"))
    })?;
    let mut out: CudaSlice<f64> = stream.alloc_zeros(perm_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("dCor permutation output allocation failed: {err}"),
        )
    })?;
    let mut flags = alloc_flags(ctx, "dcor_permutations")?;
    let func = assay_function(
        ctx,
        "assay.dcor_permutations_f64",
        "assay_dcor_permutations_f64",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(perm_count, "dCor permutation grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(a)
            .arg(b)
            .arg(&perms_dev)
            .arg(&n_i32)
            .arg(&perm_i32)
            .arg(&mut out)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("dCor permutation launch failed: {err}")))?;
    sync_and_decode(ctx, "dcor_permutations", &flags)?;
    Ok(out)
}

pub(super) fn hsic_permutation_stats(
    ctx: &CudaContext,
    kc: &CudaSlice<f64>,
    lc: &CudaSlice<f64>,
    n: usize,
    permutations: &[i32],
    perm_count: usize,
) -> Result<CudaSlice<f64>> {
    let n_i32 = to_i32(n, "HSIC n")?;
    let perm_i32 = to_i32(perm_count, "HSIC permutations")?;
    let stream = ctx.inner().default_stream();
    let perms_dev = stream.clone_htod(permutations).map_err(|err| {
        device_unavailable(ctx, format!("HSIC permutations upload failed: {err}"))
    })?;
    let mut out: CudaSlice<f64> = stream.alloc_zeros(perm_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("HSIC permutation output allocation failed: {err}"),
        )
    })?;
    let mut flags = alloc_flags(ctx, "hsic_permutations")?;
    let func = assay_function(
        ctx,
        "assay.hsic_permutations_f64",
        "assay_hsic_permutations_f64",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(perm_count, "HSIC permutation grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(kc)
            .arg(lc)
            .arg(&perms_dev)
            .arg(&n_i32)
            .arg(&perm_i32)
            .arg(&mut out)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("HSIC permutation launch failed: {err}")))?;
    sync_and_decode(ctx, "hsic_permutations", &flags)?;
    Ok(out)
}
