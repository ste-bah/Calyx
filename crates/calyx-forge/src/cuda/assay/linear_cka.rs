use super::*;

pub fn linear_cka_pair_estimates_host(
    ctx: &CudaContext,
    values: &[f32],
    lens_offsets: &[i32],
    dimensions: &[i32],
    row_count: usize,
    tuples: &[i32],
    exact: bool,
) -> Result<CudaLinearCkaPairEstimates> {
    let (lens_count, tuple_count, pair_count) =
        validate_linear_cka_inputs(values, lens_offsets, dimensions, row_count, tuples)?;
    ensure_device_room(
        ctx,
        "linear_cka_pair_estimates_host",
        checked_sum_bytes(&[
            bytes::<f32>(values.len(), "linear CKA values")?,
            bytes::<i32>(lens_offsets.len(), "linear CKA lens offsets")?,
            bytes::<i32>(dimensions.len(), "linear CKA dimensions")?,
            bytes::<i32>(tuples.len(), "linear CKA tuples")?,
            bytes::<f64>(lens_count, "linear CKA inverse energy")?,
            bytes::<f64>(lens_count * tuple_count * 3, "linear CKA sketches")?,
            bytes::<f32>(4 * pair_count, "linear CKA pair estimates")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let values_dev = stream.clone_htod(values).map_err(|err| {
        device_unavailable(ctx, format!("linear CKA values upload failed: {err}"))
    })?;
    let lens_offsets_dev = stream.clone_htod(lens_offsets).map_err(|err| {
        device_unavailable(ctx, format!("linear CKA lens offsets upload failed: {err}"))
    })?;
    let dimensions_dev = stream.clone_htod(dimensions).map_err(|err| {
        device_unavailable(ctx, format!("linear CKA dimensions upload failed: {err}"))
    })?;
    let tuples_dev = stream.clone_htod(tuples).map_err(|err| {
        device_unavailable(ctx, format!("linear CKA tuples upload failed: {err}"))
    })?;
    let mut inverse_energy: CudaSlice<f64> = stream.alloc_zeros(lens_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("linear CKA inverse energy allocation failed: {err}"),
        )
    })?;
    let mut sketch: CudaSlice<f64> =
        stream
            .alloc_zeros(lens_count * tuple_count * 3)
            .map_err(|err| {
                device_unavailable(ctx, format!("linear CKA sketch allocation failed: {err}"))
            })?;
    let mut raw_signed: CudaSlice<f32> = stream.alloc_zeros(pair_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("linear CKA raw output allocation failed: {err}"),
        )
    })?;
    let mut redundancy: CudaSlice<f32> = stream.alloc_zeros(pair_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("linear CKA redundancy output allocation failed: {err}"),
        )
    })?;
    let mut standard_error: CudaSlice<f32> = stream.alloc_zeros(pair_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("linear CKA SE output allocation failed: {err}"),
        )
    })?;
    let mut gate_upper: CudaSlice<f32> = stream.alloc_zeros(pair_count).map_err(|err| {
        device_unavailable(
            ctx,
            format!("linear CKA gate output allocation failed: {err}"),
        )
    })?;
    let mut flags = alloc_flags(ctx, "linear_cka_pair_estimates")?;
    let lens_count_i32 = to_i32(lens_count, "linear CKA lens count")?;
    let row_count_i32 = to_i32(row_count, "linear CKA row count")?;
    let tuple_count_i32 = to_i32(tuple_count, "linear CKA tuple count")?;

    let energy_func = assay_function(
        ctx,
        "assay.linear_cka_energy_f32",
        "assay_linear_cka_energy_f32",
    )?;
    let energy_cfg = LaunchConfig {
        grid_dim: (to_u32(lens_count, "linear CKA energy grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut energy_launch = stream.launch_builder(energy_func.as_ref());
    unsafe {
        energy_launch
            .arg(&values_dev)
            .arg(&lens_offsets_dev)
            .arg(&dimensions_dev)
            .arg(&lens_count_i32)
            .arg(&row_count_i32)
            .arg(&mut inverse_energy)
            .arg(&mut flags)
            .launch(energy_cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("linear CKA energy launch failed: {err}")))?;
    sync_and_decode(ctx, "linear_cka_energy", &flags)?;

    let sketch_func = assay_function(
        ctx,
        "assay.linear_cka_sketch_f32",
        "assay_linear_cka_sketch_f32",
    )?;
    let sketch_blocks = lens_count
        .checked_mul(tuple_count)
        .ok_or_else(|| shape_overflow("linear CKA sketch grid overflow"))?;
    let sketch_cfg = LaunchConfig {
        grid_dim: (to_u32(sketch_blocks, "linear CKA sketch grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut sketch_launch = stream.launch_builder(sketch_func.as_ref());
    unsafe {
        sketch_launch
            .arg(&values_dev)
            .arg(&lens_offsets_dev)
            .arg(&dimensions_dev)
            .arg(&tuples_dev)
            .arg(&inverse_energy)
            .arg(&lens_count_i32)
            .arg(&row_count_i32)
            .arg(&tuple_count_i32)
            .arg(&mut sketch)
            .arg(&mut flags)
            .launch(sketch_cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("linear CKA sketch launch failed: {err}")))?;
    sync_and_decode(ctx, "linear_cka_sketch", &flags)?;

    let pair_func = assay_function(
        ctx,
        "assay.linear_cka_pairs_f32",
        "assay_linear_cka_pairs_f32",
    )?;
    let exact_i32 = if exact { 1_i32 } else { 0_i32 };
    let pair_cfg = LaunchConfig {
        grid_dim: (to_u32(pair_count, "linear CKA pair grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut pair_launch = stream.launch_builder(pair_func.as_ref());
    unsafe {
        pair_launch
            .arg(&sketch)
            .arg(&lens_count_i32)
            .arg(&tuple_count_i32)
            .arg(&exact_i32)
            .arg(&mut raw_signed)
            .arg(&mut redundancy)
            .arg(&mut standard_error)
            .arg(&mut gate_upper)
            .arg(&mut flags)
            .launch(pair_cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("linear CKA pair launch failed: {err}")))?;
    sync_and_decode(ctx, "linear_cka_pairs", &flags)?;

    let raw_signed_point = read_f32(ctx, &raw_signed, "linear CKA raw signed point")?;
    let redundancy_point = read_f32(ctx, &redundancy, "linear CKA redundancy point")?;
    let mc_standard_error = read_f32(ctx, &standard_error, "linear CKA standard error")?;
    let mc_gate_upper_estimate = read_f32(ctx, &gate_upper, "linear CKA gate upper")?;
    validate_linear_cka_outputs(
        &raw_signed_point,
        &redundancy_point,
        &mc_standard_error,
        &mc_gate_upper_estimate,
    )?;
    Ok(CudaLinearCkaPairEstimates {
        raw_signed_point,
        redundancy_point,
        mc_standard_error,
        mc_gate_upper_estimate,
    })
}
