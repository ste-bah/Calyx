use super::*;

pub fn ccm_simplex_predictions_host(
    ctx: &CudaContext,
    embedding: &[f32],
    target: &[f32],
    n: usize,
    dim: usize,
    neighbor_count: usize,
    library_sizes: &[usize],
) -> Result<CudaCcmPredictions> {
    validate_flat_matrix("CCM embedding", embedding, n, dim)?;
    validate_vector_f32("CCM target", target, n)?;
    validate_neighbor_k("CCM", n, neighbor_count)?;
    if library_sizes.is_empty() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: "CCM requires at least one library size".to_string(),
        });
    }
    let mut prev = 0usize;
    let mut total = 0usize;
    let mut offsets = Vec::with_capacity(library_sizes.len() + 1);
    offsets.push(0i32);
    for &library_size in library_sizes {
        if library_size <= neighbor_count || library_size > n || (prev != 0 && library_size <= prev)
        {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![neighbor_count + 1, n],
                got: vec![library_size],
                remediation:
                    "CCM library sizes must be increasing and neighbor_count < library_size <= n"
                        .to_string(),
            });
        }
        total = total
            .checked_add(library_size)
            .ok_or_else(|| shape_overflow("CCM prediction task count overflow"))?;
        offsets.push(to_i32(total, "CCM prediction offset")?);
        prev = library_size;
    }
    ensure_device_room(
        ctx,
        "ccm_simplex_predictions_host",
        checked_sum_bytes(&[
            bytes::<f32>(embedding.len(), "CCM embedding")?,
            bytes::<f32>(target.len(), "CCM target")?,
            bytes::<i32>(offsets.len(), "CCM library offsets")?,
            bytes::<f32>(total, "CCM predictions")?,
        ])?,
    )?;
    let stream = ctx.inner().default_stream();
    let embedding_dev = stream
        .clone_htod(embedding)
        .map_err(|err| device_unavailable(ctx, format!("CCM embedding upload failed: {err}")))?;
    let target_dev = stream
        .clone_htod(target)
        .map_err(|err| device_unavailable(ctx, format!("CCM target upload failed: {err}")))?;
    let offsets_dev = stream
        .clone_htod(&offsets)
        .map_err(|err| device_unavailable(ctx, format!("CCM offsets upload failed: {err}")))?;
    let mut predictions: CudaSlice<f32> = stream.alloc_zeros(total).map_err(|err| {
        device_unavailable(ctx, format!("CCM predictions allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "ccm_simplex_predict")?;
    let func = assay_function(
        ctx,
        "assay.ccm_simplex_predict_f32",
        "assay_ccm_simplex_predict_f32",
    )?;
    let library_count_i32 = to_i32(library_sizes.len(), "CCM library count")?;
    let dim_i32 = to_i32(dim, "CCM embedding dimension")?;
    let neighbor_i32 = to_i32(neighbor_count, "CCM neighbor count")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(total)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&embedding_dev)
            .arg(&target_dev)
            .arg(&offsets_dev)
            .arg(&library_count_i32)
            .arg(&dim_i32)
            .arg(&neighbor_i32)
            .arg(&mut predictions)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(ctx, format!("CCM simplex prediction launch failed: {err}"))
    })?;
    sync_and_decode(ctx, "ccm_simplex_predict", &flags)?;
    let flat = read_f32(ctx, &predictions, "CCM predictions")?;
    let mut library_predictions = Vec::with_capacity(library_sizes.len());
    for window in offsets.windows(2) {
        let start = window[0] as usize;
        let end = window[1] as usize;
        library_predictions.push(flat[start..end].to_vec());
    }
    Ok(CudaCcmPredictions {
        library_predictions,
    })
}
