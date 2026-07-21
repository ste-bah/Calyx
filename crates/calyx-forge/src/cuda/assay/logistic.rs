use super::*;

#[derive(Clone, Copy, Debug)]
pub struct CudaLogisticDataset<'a> {
    pub samples: &'a [f32],
    pub labels: &'a [i32],
    pub rows: usize,
    pub dim: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaLogisticSplits<'a> {
    pub train_offsets: &'a [i32],
    pub train_indices: &'a [i32],
    pub test_offsets: &'a [i32],
    pub test_indices: &'a [i32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaLogisticConfig {
    pub steps: usize,
    pub learning_rate: f32,
    pub l2_penalty: f32,
}

pub fn logistic_summaries_host(
    ctx: &CudaContext,
    dataset: CudaLogisticDataset<'_>,
    splits: CudaLogisticSplits<'_>,
    config: CudaLogisticConfig,
) -> Result<CudaLogisticSummaries> {
    let CudaLogisticDataset {
        samples,
        labels,
        rows: n,
        dim,
    } = dataset;
    let CudaLogisticSplits {
        train_offsets,
        train_indices,
        test_offsets,
        test_indices,
    } = splits;
    let CudaLogisticConfig {
        steps,
        learning_rate: lr,
        l2_penalty: l2,
    } = config;
    validate_flat_matrix("logistic samples", samples, n, dim)?;
    validate_binary_labels(labels, n)?;
    let fit_count = validate_split_buffers(
        "logistic",
        n,
        train_offsets,
        train_indices,
        test_offsets,
        test_indices,
    )?;
    if dim > 1024 || steps == 0 || !lr.is_finite() || lr <= 0.0 || !l2.is_finite() || l2 < 0.0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1024, 1],
            got: vec![dim, steps],
            remediation: "logistic CUDA requires 0 < dim <= 1024, steps > 0, lr > 0, and l2 >= 0"
                .to_string(),
        });
    }
    ensure_device_room(
        ctx,
        "logistic_summaries_host",
        checked_sum_bytes(&[
            bytes::<f32>(samples.len(), "logistic samples")?,
            bytes::<i32>(labels.len(), "logistic labels")?,
            bytes::<i32>(train_offsets.len(), "logistic train offsets")?,
            bytes::<i32>(train_indices.len(), "logistic train indices")?,
            bytes::<i32>(test_offsets.len(), "logistic test offsets")?,
            bytes::<i32>(test_indices.len(), "logistic test indices")?,
            bytes::<f32>(2 * fit_count, "logistic summaries")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let samples_dev = stream
        .clone_htod(samples)
        .map_err(|err| device_unavailable(ctx, format!("logistic samples upload failed: {err}")))?;
    let labels_dev = stream
        .clone_htod(labels)
        .map_err(|err| device_unavailable(ctx, format!("logistic labels upload failed: {err}")))?;
    let train_offsets_dev = stream.clone_htod(train_offsets).map_err(|err| {
        device_unavailable(ctx, format!("logistic train offsets upload failed: {err}"))
    })?;
    let train_indices_dev = stream.clone_htod(train_indices).map_err(|err| {
        device_unavailable(ctx, format!("logistic train indices upload failed: {err}"))
    })?;
    let test_offsets_dev = stream.clone_htod(test_offsets).map_err(|err| {
        device_unavailable(ctx, format!("logistic test offsets upload failed: {err}"))
    })?;
    let test_indices_dev = stream.clone_htod(test_indices).map_err(|err| {
        device_unavailable(ctx, format!("logistic test indices upload failed: {err}"))
    })?;
    let mut bits: CudaSlice<f32> = stream.alloc_zeros(fit_count).map_err(|err| {
        device_unavailable(ctx, format!("logistic bits allocation failed: {err}"))
    })?;
    let mut accuracy: CudaSlice<f32> = stream.alloc_zeros(fit_count).map_err(|err| {
        device_unavailable(ctx, format!("logistic accuracy allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "logistic_summaries")?;
    let func = assay_function(
        ctx,
        "assay.logistic_summaries_f32",
        "assay_logistic_summaries_f32",
    )?;
    let fit_count_i32 = to_i32(fit_count, "logistic fit count")?;
    let n_i32 = to_i32(n, "logistic sample count")?;
    let dim_i32 = to_i32(dim, "logistic dimension")?;
    let steps_i32 = to_i32(steps, "logistic steps")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(fit_count, "logistic fit grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&samples_dev)
            .arg(&labels_dev)
            .arg(&train_offsets_dev)
            .arg(&train_indices_dev)
            .arg(&test_offsets_dev)
            .arg(&test_indices_dev)
            .arg(&fit_count_i32)
            .arg(&n_i32)
            .arg(&dim_i32)
            .arg(&steps_i32)
            .arg(&lr)
            .arg(&l2)
            .arg(&mut bits)
            .arg(&mut accuracy)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("logistic summaries launch failed: {err}")))?;
    sync_and_decode(ctx, "logistic_summaries", &flags)?;
    let bits = read_f32(ctx, &bits, "logistic bits")?;
    let accuracy = read_f32(ctx, &accuracy, "logistic accuracy")?;
    for (idx, value) in accuracy.iter().copied().enumerate() {
        if !(0.0..=1.0).contains(&value) {
            return Err(numerical(
                "logistic accuracy",
                format!("logistic accuracy readback out of range at fit {idx}: {value}"),
            ));
        }
    }
    Ok(CudaLogisticSummaries { bits, accuracy })
}
