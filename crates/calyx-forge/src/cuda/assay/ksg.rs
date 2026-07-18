use super::*;

pub fn ksg_continuous_counts_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    n: usize,
    dim_x: usize,
    dim_y: usize,
    k: usize,
) -> Result<CudaKsgContinuousCounts> {
    validate_flat_matrix("KSG x", x, n, dim_x)?;
    validate_flat_matrix("KSG y", y, n, dim_y)?;
    validate_neighbor_k("KSG", n, k)?;
    ensure_device_room(
        ctx,
        "ksg_continuous_counts_host",
        checked_sum_bytes(&[
            bytes::<f32>(x.len(), "KSG x")?,
            bytes::<f32>(y.len(), "KSG y")?,
            bytes::<f32>(n, "KSG radii")?,
            bytes::<i32>(2 * n, "KSG marginal counts")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("KSG x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("KSG y upload failed: {err}")))?;
    let mut radii: CudaSlice<f32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("KSG radii allocation failed: {err}")))?;
    let mut nx: CudaSlice<i32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("KSG nx allocation failed: {err}")))?;
    let mut ny: CudaSlice<i32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("KSG ny allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "ksg_continuous_counts")?;
    let func = assay_function(
        ctx,
        "assay.ksg_continuous_counts_f32",
        "assay_ksg_continuous_counts_f32",
    )?;
    let n_i32 = to_i32(n, "KSG sample count")?;
    let dim_x_i32 = to_i32(dim_x, "KSG x dimension")?;
    let dim_y_i32 = to_i32(dim_y, "KSG y dimension")?;
    let k_i32 = to_i32(k, "KSG k")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "KSG grid rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&y_dev)
            .arg(&n_i32)
            .arg(&dim_x_i32)
            .arg(&dim_y_i32)
            .arg(&k_i32)
            .arg(&mut radii)
            .arg(&mut nx)
            .arg(&mut ny)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("KSG continuous launch failed: {err}")))?;
    sync_and_decode(ctx, "ksg_continuous_counts", &flags)?;
    Ok(CudaKsgContinuousCounts {
        radii: read_f32(ctx, &radii, "KSG radii")?,
        nx: read_usize_counts(ctx, &nx, "KSG nx")?,
        ny: read_usize_counts(ctx, &ny, "KSG ny")?,
    })
}

pub fn entropy_radii_host(
    ctx: &CudaContext,
    values: &[f32],
    n: usize,
    dim: usize,
    k: usize,
) -> Result<Vec<f32>> {
    validate_flat_matrix("KSG entropy values", values, n, dim)?;
    validate_neighbor_k("KSG entropy", n, k)?;
    ensure_device_room(
        ctx,
        "entropy_radii_host",
        checked_sum_bytes(&[
            bytes::<f32>(values.len(), "entropy samples")?,
            bytes::<f32>(n, "entropy radii")?,
        ])?,
    )?;
    let stream = ctx.inner().default_stream();
    let values_dev = stream
        .clone_htod(values)
        .map_err(|err| device_unavailable(ctx, format!("entropy values upload failed: {err}")))?;
    let mut radii: CudaSlice<f32> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("entropy radii allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "entropy_radii")?;
    let func = assay_function(ctx, "assay.entropy_radii_f32", "assay_entropy_radii_f32")?;
    let n_i32 = to_i32(n, "entropy sample count")?;
    let dim_i32 = to_i32(dim, "entropy dimension")?;
    let k_i32 = to_i32(k, "entropy k")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "entropy grid rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&values_dev)
            .arg(&n_i32)
            .arg(&dim_i32)
            .arg(&k_i32)
            .arg(&mut radii)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("entropy radii launch failed: {err}")))?;
    sync_and_decode(ctx, "entropy_radii", &flags)?;
    read_f32(ctx, &radii, "entropy radii")
}

pub fn mixed_ksg_counts_host(
    ctx: &CudaContext,
    x: &[f32],
    labels: &[i32],
    n: usize,
    dim: usize,
    k: usize,
) -> Result<CudaMixedKsgCounts> {
    validate_flat_matrix("mixed KSG x", x, n, dim)?;
    validate_labels(labels, n)?;
    validate_neighbor_k("mixed KSG", n, k)?;
    ensure_device_room(
        ctx,
        "mixed_ksg_counts_host",
        checked_sum_bytes(&[
            bytes::<f32>(x.len(), "mixed KSG x")?,
            bytes::<i32>(labels.len(), "mixed KSG labels")?,
            bytes::<f32>(n, "mixed KSG radii")?,
            bytes::<i32>(2 * n, "mixed KSG counts")?,
        ])?,
    )?;
    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("mixed KSG x upload failed: {err}")))?;
    let labels_dev = stream
        .clone_htod(labels)
        .map_err(|err| device_unavailable(ctx, format!("mixed KSG labels upload failed: {err}")))?;
    let mut radii: CudaSlice<f32> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("mixed KSG radii allocation failed: {err}"))
    })?;
    let mut same: CudaSlice<i32> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(
            ctx,
            format!("mixed KSG same count allocation failed: {err}"),
        )
    })?;
    let mut full: CudaSlice<i32> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(
            ctx,
            format!("mixed KSG full count allocation failed: {err}"),
        )
    })?;
    let mut flags = alloc_flags(ctx, "mixed_ksg_counts")?;
    let func = assay_function(
        ctx,
        "assay.mixed_ksg_counts_f32",
        "assay_mixed_ksg_counts_f32",
    )?;
    let n_i32 = to_i32(n, "mixed KSG sample count")?;
    let dim_i32 = to_i32(dim, "mixed KSG dimension")?;
    let k_i32 = to_i32(k, "mixed KSG k")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(n, "mixed KSG grid rows")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&labels_dev)
            .arg(&n_i32)
            .arg(&dim_i32)
            .arg(&k_i32)
            .arg(&mut radii)
            .arg(&mut same)
            .arg(&mut full)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("mixed KSG launch failed: {err}")))?;
    sync_and_decode(ctx, "mixed_ksg_counts", &flags)?;
    Ok(CudaMixedKsgCounts {
        radii: read_f32(ctx, &radii, "mixed KSG radii")?,
        same_class_counts: read_usize_counts(ctx, &same, "mixed KSG same counts")?,
        full_counts: read_usize_counts(ctx, &full, "mixed KSG full counts")?,
    })
}
