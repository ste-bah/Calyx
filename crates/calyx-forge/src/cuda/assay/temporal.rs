use super::*;

pub fn periodogram_batch_host(
    ctx: &CudaContext,
    times: &[f64],
    centered: &[f64],
    variance: f64,
    frequencies: &[f64],
    permutations: Option<&[i32]>,
) -> Result<CudaPeriodogramBatch> {
    let perm_count =
        validate_periodogram_inputs(times, centered, variance, frequencies, permutations)?;
    let perm_cells = perm_count
        .checked_mul(frequencies.len())
        .ok_or_else(|| shape_overflow("periodogram permutation grid overflow"))?;
    if perm_cells > MAX_GLS_PERMUTATION_CELLS {
        return Err(ForgeError::VramBudget {
            detail: format!(
                "periodogram_batch_host permutation grid has {perm_cells} cells; bounded strict CUDA maximum is {MAX_GLS_PERMUTATION_CELLS}"
            ),
            remediation: VRAM_REMEDIATION.to_string(),
        });
    }
    ensure_device_room(
        ctx,
        "periodogram_batch_host",
        checked_sum_bytes(&[
            bytes::<f64>(times.len(), "GLS times")?,
            bytes::<f64>(centered.len(), "GLS centered values")?,
            bytes::<f64>(frequencies.len(), "GLS frequencies")?,
            bytes::<f64>(frequencies.len(), "GLS observed powers")?,
            bytes::<i32>(
                permutations.map_or(0, |values| values.len()),
                "GLS permutations",
            )?,
            bytes::<f64>(perm_cells, "GLS permutation powers")?,
            bytes::<f64>(perm_count, "GLS permutation max powers")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let times_dev = stream
        .clone_htod(times)
        .map_err(|err| device_unavailable(ctx, format!("GLS times upload failed: {err}")))?;
    let centered_dev = stream.clone_htod(centered).map_err(|err| {
        device_unavailable(ctx, format!("GLS centered values upload failed: {err}"))
    })?;
    let frequencies_dev = stream
        .clone_htod(frequencies)
        .map_err(|err| device_unavailable(ctx, format!("GLS frequencies upload failed: {err}")))?;
    let mut powers: CudaSlice<f64> = stream
        .alloc_zeros(frequencies.len())
        .map_err(|err| device_unavailable(ctx, format!("GLS powers allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "periodogram_powers")?;
    let func = assay_function(ctx, "assay.gls_powers_f64", "assay_gls_powers_f64")?;
    let n_i32 = to_i32(times.len(), "GLS sample count")?;
    let freq_i32 = to_i32(frequencies.len(), "GLS frequency count")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(frequencies.len(), "GLS frequency grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&times_dev)
            .arg(&centered_dev)
            .arg(&frequencies_dev)
            .arg(&n_i32)
            .arg(&freq_i32)
            .arg(&variance)
            .arg(&mut powers)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("GLS powers launch failed: {err}")))?;
    sync_and_decode(ctx, "periodogram_powers", &flags)?;

    let permutation_max_powers = if let Some(permutations) = permutations {
        let permutations_dev = stream.clone_htod(permutations).map_err(|err| {
            device_unavailable(ctx, format!("GLS permutations upload failed: {err}"))
        })?;
        let mut permutation_powers: CudaSlice<f64> =
            stream.alloc_zeros(perm_cells).map_err(|err| {
                device_unavailable(
                    ctx,
                    format!("GLS permutation powers allocation failed: {err}"),
                )
            })?;
        let mut max_powers: CudaSlice<f64> = stream.alloc_zeros(perm_count).map_err(|err| {
            device_unavailable(ctx, format!("GLS permutation max allocation failed: {err}"))
        })?;
        let mut perm_flags = alloc_flags(ctx, "periodogram_permutation_powers")?;
        let perm_func = assay_function(
            ctx,
            "assay.gls_permutation_powers_f64",
            "assay_gls_permutation_powers_f64",
        )?;
        let perm_count_i32 = to_i32(perm_count, "GLS permutation count")?;
        let perm_cfg = LaunchConfig {
            grid_dim: (
                to_u32(frequencies.len(), "GLS permutation frequency grid")?,
                to_u32(perm_count, "GLS permutation grid")?,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut perm_launch = stream.launch_builder(perm_func.as_ref());
        unsafe {
            perm_launch
                .arg(&times_dev)
                .arg(&centered_dev)
                .arg(&frequencies_dev)
                .arg(&permutations_dev)
                .arg(&n_i32)
                .arg(&freq_i32)
                .arg(&perm_count_i32)
                .arg(&variance)
                .arg(&mut permutation_powers)
                .arg(&mut perm_flags)
                .launch(perm_cfg)
        }
        .map_err(|err| {
            device_unavailable(ctx, format!("GLS permutation powers launch failed: {err}"))
        })?;
        sync_and_decode(ctx, "periodogram_permutation_powers", &perm_flags)?;

        let mut max_flags = alloc_flags(ctx, "periodogram_permutation_max")?;
        let max_func = assay_function(ctx, "assay.row_max_f64", "assay_row_max_f64")?;
        let max_cfg = LaunchConfig {
            grid_dim: (to_u32(perm_count, "GLS permutation max grid")?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut max_launch = stream.launch_builder(max_func.as_ref());
        unsafe {
            max_launch
                .arg(&permutation_powers)
                .arg(&freq_i32)
                .arg(&perm_count_i32)
                .arg(&mut max_powers)
                .arg(&mut max_flags)
                .launch(max_cfg)
        }
        .map_err(|err| {
            device_unavailable(ctx, format!("GLS permutation max launch failed: {err}"))
        })?;
        sync_and_decode(ctx, "periodogram_permutation_max", &max_flags)?;
        let max_powers = read_device_f64(ctx, "GLS permutation max readback", &max_powers)?;
        validate_matrix_readback("GLS permutation max readback", &max_powers)?;
        max_powers
    } else {
        Vec::new()
    };

    let powers = read_device_f64(ctx, "GLS powers readback", &powers)?;
    validate_matrix_readback("GLS powers readback", &powers)?;
    Ok(CudaPeriodogramBatch {
        powers,
        permutation_max_powers,
    })
}

pub fn autocorrelation_sums_host(
    ctx: &CudaContext,
    times: &[f64],
    centered: &[f64],
    variance: f64,
    slot_width: f64,
    max_lag: f64,
    slot_count: usize,
) -> Result<CudaAutocorrelationSums> {
    validate_autocorrelation_inputs(times, centered, variance, slot_width, max_lag, slot_count)?;
    let slot_len = slot_count
        .checked_add(1)
        .ok_or_else(|| shape_overflow("ACF slot count overflow"))?;
    ensure_device_room(
        ctx,
        "autocorrelation_sums_host",
        checked_sum_bytes(&[
            bytes::<f64>(times.len(), "ACF times")?,
            bytes::<f64>(centered.len(), "ACF centered values")?,
            bytes::<f64>(slot_len, "ACF slot sums")?,
            bytes::<i32>(slot_len, "ACF slot counts")?,
        ])?,
    )?;
    let stream = ctx.inner().default_stream();
    let times_dev = stream
        .clone_htod(times)
        .map_err(|err| device_unavailable(ctx, format!("ACF times upload failed: {err}")))?;
    let centered_dev = stream.clone_htod(centered).map_err(|err| {
        device_unavailable(ctx, format!("ACF centered values upload failed: {err}"))
    })?;
    let mut sums: CudaSlice<f64> = stream
        .alloc_zeros(slot_len)
        .map_err(|err| device_unavailable(ctx, format!("ACF sums allocation failed: {err}")))?;
    let mut counts: CudaSlice<i32> = stream
        .alloc_zeros(slot_len)
        .map_err(|err| device_unavailable(ctx, format!("ACF counts allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "autocorrelation_sums")?;
    let func = assay_function(ctx, "assay.acf_slotted_f64", "assay_acf_slotted_f64")?;
    let n_i32 = to_i32(times.len(), "ACF sample count")?;
    let slot_count_i32 = to_i32(slot_count, "ACF slot count")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(times.len(), "ACF row grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&times_dev)
            .arg(&centered_dev)
            .arg(&n_i32)
            .arg(&variance)
            .arg(&slot_width)
            .arg(&max_lag)
            .arg(&slot_count_i32)
            .arg(&mut sums)
            .arg(&mut counts)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("ACF slotted launch failed: {err}")))?;
    sync_and_decode(ctx, "autocorrelation_sums", &flags)?;
    let sums = read_device_f64(ctx, "ACF sums readback", &sums)?;
    validate_matrix_readback("ACF sums readback", &sums)?;
    let counts = read_usize_counts(ctx, &counts, "ACF counts")?;
    Ok(CudaAutocorrelationSums { sums, counts })
}

pub fn cross_correlation_batch_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    max_lag: usize,
    min_pairs: usize,
) -> Result<CudaCrossCorrelationBatch> {
    validate_cross_correlation_inputs(x, y, max_lag, min_pairs)?;
    let point_count = max_lag
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| shape_overflow("CCF point count overflow"))?;
    ensure_device_room(
        ctx,
        "cross_correlation_batch_host",
        checked_sum_bytes(&[
            bytes::<f32>(x.len(), "CCF x")?,
            bytes::<f32>(y.len(), "CCF y")?,
            bytes::<f32>(point_count, "CCF correlations")?,
            bytes::<i32>(point_count, "CCF pair counts")?,
        ])?,
    )?;
    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("CCF x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("CCF y upload failed: {err}")))?;
    let mut correlations: CudaSlice<f32> = stream.alloc_zeros(point_count).map_err(|err| {
        device_unavailable(ctx, format!("CCF correlations allocation failed: {err}"))
    })?;
    let mut n_pairs: CudaSlice<i32> = stream.alloc_zeros(point_count).map_err(|err| {
        device_unavailable(ctx, format!("CCF pair counts allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "cross_correlation_batch")?;
    let func = assay_function(
        ctx,
        "assay.cross_correlation_f32",
        "assay_cross_correlation_f32",
    )?;
    let n_i32 = to_i32(x.len(), "CCF sample count")?;
    let max_lag_i32 = to_i32(max_lag, "CCF max lag")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(point_count, "CCF lag grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&y_dev)
            .arg(&n_i32)
            .arg(&max_lag_i32)
            .arg(&mut correlations)
            .arg(&mut n_pairs)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("CCF lag batch launch failed: {err}")))?;
    sync_and_decode(ctx, "cross_correlation_batch", &flags)?;
    Ok(CudaCrossCorrelationBatch {
        correlations: read_f32(ctx, &correlations, "CCF correlations")?,
        n_pairs: read_usize_counts(ctx, &n_pairs, "CCF pair counts")?,
    })
}
