use super::*;

const MAX_GRANGER_LAGS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GrangerWorkspace {
    row_stride: usize,
    matrix_cells: usize,
    vector_cells: usize,
    bytes: usize,
}

fn granger_workspace(n: usize, lags: &[usize]) -> Result<GrangerWorkspace> {
    let max_admissible_lag = lags
        .iter()
        .copied()
        .filter(|&lag| lag > 0 && lag <= MAX_GRANGER_LAGS && n >= 3 * lag + 2)
        .max()
        .unwrap_or(0);
    let row_stride = max_admissible_lag
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| shape_overflow("Granger workspace row stride overflow"))?;
    let matrix_cells = checked_square(row_stride, "Granger workspace matrix stride")?
        .checked_mul(lags.len())
        .ok_or_else(|| shape_overflow("Granger workspace matrix length overflow"))?;
    let vector_cells = row_stride
        .checked_mul(lags.len())
        .ok_or_else(|| shape_overflow("Granger workspace vector length overflow"))?;
    let matrix_bytes = bytes::<f64>(matrix_cells, "Granger workspace matrix")?;
    let vector_bytes = bytes::<f64>(vector_cells, "Granger workspace vector")?;
    let bytes = checked_sum_bytes(&[matrix_bytes, matrix_bytes, vector_bytes, vector_bytes])?;
    Ok(GrangerWorkspace {
        row_stride,
        matrix_cells,
        vector_cells,
        bytes,
    })
}

pub fn correlation_precision_host(
    ctx: &CudaContext,
    columns: &[f32],
    n: usize,
    d: usize,
) -> Result<CudaCorrelationPrecision> {
    validate_correlation_columns(columns, n, d)?;
    let matrix_len = checked_square(d, "correlation matrix length")?;
    ensure_device_room(
        ctx,
        "correlation_precision_host",
        checked_sum_bytes(&[
            bytes::<f32>(columns.len(), "correlation columns")?,
            bytes::<f64>(matrix_len, "correlation matrix")?,
            bytes::<f64>(matrix_len, "precision matrix")?,
            bytes::<f64>(2 * matrix_len, "correlation inversion scratch")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let columns_dev = stream.clone_htod(columns).map_err(|err| {
        device_unavailable(ctx, format!("correlation column upload failed: {err}"))
    })?;
    let mut corr: CudaSlice<f64> = stream.alloc_zeros(matrix_len).map_err(|err| {
        device_unavailable(ctx, format!("correlation matrix allocation failed: {err}"))
    })?;
    let mut precision: CudaSlice<f64> = stream.alloc_zeros(matrix_len).map_err(|err| {
        device_unavailable(ctx, format!("precision matrix allocation failed: {err}"))
    })?;
    let mut scratch: CudaSlice<f64> = stream.alloc_zeros(2 * matrix_len).map_err(|err| {
        device_unavailable(ctx, format!("inversion scratch allocation failed: {err}"))
    })?;
    let mut corr_flags = alloc_flags(ctx, "correlation_matrix")?;
    let corr_func = assay_function(ctx, "assay.corr_matrix_f32", "assay_corr_matrix_f32")?;
    let n_i32 = to_i32(n, "correlation sample count")?;
    let d_i32 = to_i32(d, "correlation variable count")?;
    let corr_cfg = LaunchConfig {
        grid_dim: (to_u32(matrix_len, "correlation pair grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut corr_launch = stream.launch_builder(corr_func.as_ref());
    unsafe {
        corr_launch
            .arg(&columns_dev)
            .arg(&n_i32)
            .arg(&d_i32)
            .arg(&mut corr)
            .arg(&mut corr_flags)
            .launch(corr_cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("correlation matrix launch failed: {err}")))?;
    sync_and_decode(ctx, "correlation_matrix", &corr_flags)?;

    let mut invert_flags = alloc_flags(ctx, "correlation_precision_invert")?;
    let invert_func = assay_function(
        ctx,
        "assay.invert_symmetric_f64",
        "assay_invert_symmetric_f64",
    )?;
    let invert_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut invert_launch = stream.launch_builder(invert_func.as_ref());
    unsafe {
        invert_launch
            .arg(&corr)
            .arg(&d_i32)
            .arg(&mut scratch)
            .arg(&mut precision)
            .arg(&mut invert_flags)
            .launch(invert_cfg)
    }
    .map_err(|err| {
        device_unavailable(
            ctx,
            format!("correlation precision inversion launch failed: {err}"),
        )
    })?;
    sync_and_decode(ctx, "correlation_precision_invert", &invert_flags)?;

    let corr = read_device_f64(ctx, "correlation matrix readback", &corr)?;
    let precision = read_device_f64(ctx, "precision matrix readback", &precision)?;
    validate_matrix_readback("correlation matrix", &corr)?;
    validate_matrix_readback("precision matrix", &precision)?;
    Ok(CudaCorrelationPrecision {
        corr,
        precision,
        n_samples: n,
        n_variables: d,
    })
}

pub fn granger_lag_summaries_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    lags: &[usize],
) -> Result<CudaGrangerLagBatch> {
    validate_granger_batch_inputs(x, y, lags)?;
    let workspace = granger_workspace(x.len(), lags)?;
    ensure_device_room(
        ctx,
        "granger_lag_summaries_host",
        checked_sum_bytes(&[
            bytes::<f32>(x.len(), "Granger x")?,
            bytes::<f32>(y.len(), "Granger y")?,
            bytes::<i32>(lags.len(), "Granger lags")?,
            bytes::<f64>(2 * lags.len(), "Granger RSS outputs")?,
            bytes::<i32>(3 * lags.len(), "Granger integer outputs")?,
            workspace.bytes,
            bytes::<u32>(1, "Granger flags")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("Granger x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("Granger y upload failed: {err}")))?;
    let lag_values = lags
        .iter()
        .map(|&lag| to_i32(lag, "Granger lag"))
        .collect::<Result<Vec<_>>>()?;
    let lags_dev = stream
        .clone_htod(&lag_values)
        .map_err(|err| device_unavailable(ctx, format!("Granger lags upload failed: {err}")))?;
    let mut rss_r: CudaSlice<f64> = stream.alloc_zeros(lags.len()).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Granger restricted RSS allocation failed: {err}"),
        )
    })?;
    let mut rss_u: CudaSlice<f64> = stream.alloc_zeros(lags.len()).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Granger unrestricted RSS allocation failed: {err}"),
        )
    })?;
    let mut n_used_dev: CudaSlice<i32> = stream.alloc_zeros(lags.len()).map_err(|err| {
        device_unavailable(ctx, format!("Granger n_used allocation failed: {err}"))
    })?;
    let mut df_den_dev: CudaSlice<i32> = stream.alloc_zeros(lags.len()).map_err(|err| {
        device_unavailable(ctx, format!("Granger df_den allocation failed: {err}"))
    })?;
    let mut status_dev: CudaSlice<i32> = stream.alloc_zeros(lags.len()).map_err(|err| {
        device_unavailable(ctx, format!("Granger status allocation failed: {err}"))
    })?;
    let mut ar_workspace: CudaSlice<f64> =
        stream.alloc_zeros(workspace.matrix_cells).map_err(|err| {
            device_unavailable(
                ctx,
                format!("Granger restricted matrix workspace allocation failed: {err}"),
            )
        })?;
    let mut au_workspace: CudaSlice<f64> =
        stream.alloc_zeros(workspace.matrix_cells).map_err(|err| {
            device_unavailable(
                ctx,
                format!("Granger unrestricted matrix workspace allocation failed: {err}"),
            )
        })?;
    let mut br_workspace: CudaSlice<f64> =
        stream.alloc_zeros(workspace.vector_cells).map_err(|err| {
            device_unavailable(
                ctx,
                format!("Granger restricted vector workspace allocation failed: {err}"),
            )
        })?;
    let mut bu_workspace: CudaSlice<f64> =
        stream.alloc_zeros(workspace.vector_cells).map_err(|err| {
            device_unavailable(
                ctx,
                format!("Granger unrestricted vector workspace allocation failed: {err}"),
            )
        })?;
    let mut flags = alloc_flags(ctx, "granger_lag_summaries")?;
    let func = assay_function(
        ctx,
        "assay.granger_lag_summaries_f32",
        "assay_granger_lag_summaries_f32",
    )?;
    let lag_count_i32 = to_i32(lags.len(), "Granger lag count")?;
    let n_i32 = to_i32(x.len(), "Granger sample count")?;
    let workspace_stride_i32 = to_i32(workspace.row_stride, "Granger workspace row stride")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(lags.len(), "Granger lag grid")?, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&y_dev)
            .arg(&lags_dev)
            .arg(&lag_count_i32)
            .arg(&n_i32)
            .arg(&workspace_stride_i32)
            .arg(&mut ar_workspace)
            .arg(&mut au_workspace)
            .arg(&mut br_workspace)
            .arg(&mut bu_workspace)
            .arg(&mut rss_r)
            .arg(&mut rss_u)
            .arg(&mut n_used_dev)
            .arg(&mut df_den_dev)
            .arg(&mut status_dev)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("Granger lag summary launch failed: {err}")))?;
    sync_and_decode(ctx, "granger_lag_summaries", &flags)?;

    let rss_restricted = read_device_f64(ctx, "Granger restricted RSS readback", &rss_r)?;
    let rss_unrestricted = read_device_f64(ctx, "Granger unrestricted RSS readback", &rss_u)?;
    let n_used = read_device_i32(ctx, "Granger n_used readback", &n_used_dev)?;
    let df_den = read_device_i32(ctx, "Granger df_den readback", &df_den_dev)?;
    let status = read_device_i32(ctx, "Granger status readback", &status_dev)?;
    let mut summaries = Vec::with_capacity(lags.len());
    for idx in 0..lags.len() {
        if status[idx] == CUDA_GRANGER_STATUS_OK
            && !(rss_restricted[idx].is_finite()
                && rss_unrestricted[idx].is_finite()
                && rss_restricted[idx] >= 0.0
                && rss_unrestricted[idx] >= 0.0
                && n_used[idx] > 0
                && df_den[idx] > 0)
        {
            return Err(numerical(
                "granger_lag_summaries_host",
                format!(
                    "invalid Granger readback at lag index {idx}: lag={} rss_r={} rss_u={} n_used={} df_den={} status={}",
                    lags[idx],
                    rss_restricted[idx],
                    rss_unrestricted[idx],
                    n_used[idx],
                    df_den[idx],
                    status[idx]
                ),
            ));
        }
        summaries.push(CudaGrangerLagSummary {
            lag: lags[idx],
            rss_restricted: rss_restricted[idx],
            rss_unrestricted: rss_unrestricted[idx],
            n_used: n_used[idx].max(0) as usize,
            df_den: df_den[idx].max(0) as usize,
            status: status[idx],
        });
    }
    Ok(CudaGrangerLagBatch {
        summaries,
        workspace_row_stride: workspace.row_stride,
        workspace_bytes: workspace.bytes,
    })
}
