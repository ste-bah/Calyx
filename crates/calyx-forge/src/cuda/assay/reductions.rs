use super::*;

pub(super) fn count_ge(
    ctx: &CudaContext,
    values: &CudaSlice<f64>,
    len: usize,
    observed: f64,
    tolerance: f64,
) -> Result<usize> {
    if len == 0 {
        return Ok(0);
    }
    let len_i32 = to_i32(len, "count length")?;
    let stream = ctx.inner().default_stream();
    let mut count: CudaSlice<u32> = stream
        .alloc_zeros(1)
        .map_err(|err| device_unavailable(ctx, format!("count allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "count_ge")?;
    let func = assay_function(ctx, "assay.count_ge_f64", "assay_count_ge_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(len)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(values)
            .arg(&len_i32)
            .arg(&observed)
            .arg(&tolerance)
            .arg(&mut count)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("count_ge launch failed: {err}")))?;
    sync_and_decode(ctx, "count_ge", &flags)?;
    let host = stream
        .clone_dtoh(&count)
        .map_err(|err| device_unavailable(ctx, format!("count_ge readback failed: {err}")))?;
    Ok(host.first().copied().unwrap_or_default() as usize)
}

pub(super) fn reduce_sum(
    ctx: &CudaContext,
    values: &CudaSlice<f64>,
    len: usize,
    op: &'static str,
) -> Result<f64> {
    if len == 0 {
        return Ok(0.0);
    }
    let mut current_len = len;
    let mut current = reduce_once(ctx, values, current_len, op)?;
    current_len = current.len();
    while current_len > 1 {
        let next = reduce_once(ctx, &current, current_len, op)?;
        current = next;
        current_len = current.len();
    }
    let values = read_device_f64(ctx, op, &current)?;
    Ok(values[0])
}

pub(super) fn reduce_once(
    ctx: &CudaContext,
    values: &CudaSlice<f64>,
    len: usize,
    op: &'static str,
) -> Result<CudaSlice<f64>> {
    let blocks = reduction_blocks(len);
    let len_i32 = to_i32(len, "reduction length")?;
    let stream = ctx.inner().default_stream();
    let mut partials: CudaSlice<f64> = stream.alloc_zeros(blocks).map_err(|err| {
        device_unavailable(ctx, format!("{op} reduction allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, op)?;
    let func = assay_function(ctx, "assay.reduce_sum_f64", "assay_reduce_sum_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(blocks, "reduction blocks")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(values)
            .arg(&len_i32)
            .arg(&mut partials)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("{op} reduction launch failed: {err}")))?;
    sync_and_decode(ctx, op, &flags)?;
    Ok(partials)
}

pub(super) fn read_device_f64(
    ctx: &CudaContext,
    op: &'static str,
    values: &CudaSlice<f64>,
) -> Result<Vec<f64>> {
    ctx.inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} device readback failed: {err}")))
}

pub(super) fn read_device_i32(
    ctx: &CudaContext,
    op: &'static str,
    values: &CudaSlice<i32>,
) -> Result<Vec<i32>> {
    ctx.inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} device readback failed: {err}")))
}
