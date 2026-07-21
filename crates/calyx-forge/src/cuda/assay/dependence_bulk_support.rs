use std::mem::size_of;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_rank(
    ctx: &CudaContext,
    x: &CudaSlice<f64>,
    y: &CudaSlice<f64>,
    n: usize,
    x_ranks: &mut CudaSlice<f64>,
    y_ranks: &mut CudaSlice<f64>,
    x_ties: &mut CudaSlice<u32>,
    y_ties: &mut CudaSlice<u32>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let n_i32 = to_i32(n, "rank sample count")?;
    let func = assay_function(ctx, "assay.rank_ties_f64", "assay_rank_ties_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(n)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(x)
            .arg(y)
            .arg(&n_i32)
            .arg(x_ranks)
            .arg(y_ranks)
            .arg(x_ties)
            .arg(y_ties)
            .arg(&mut *flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("rank launch failed: {err}")))?;
    sync_and_decode(ctx, "rank_pair", flags)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_kendall(
    ctx: &CudaContext,
    x: &CudaSlice<f64>,
    y: &CudaSlice<f64>,
    n: usize,
    concordant: &mut CudaSlice<u64>,
    discordant: &mut CudaSlice<u64>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let n_i32 = to_i32(n, "Kendall sample count")?;
    let func = assay_function(ctx, "assay.kendall_counts_f64", "assay_kendall_counts_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(n)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(x)
            .arg(y)
            .arg(&n_i32)
            .arg(concordant)
            .arg(discordant)
            .arg(&mut *flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("Kendall launch failed: {err}")))?;
    sync_and_decode(ctx, "kendall_counts", flags)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_copula(
    ctx: &CudaContext,
    x_ranks: &CudaSlice<f64>,
    y_ranks: &CudaSlice<f64>,
    n: usize,
    tail_q: f64,
    c_mid: &mut CudaSlice<u64>,
    lower: &mut CudaSlice<u64>,
    upper: &mut CudaSlice<u64>,
    hoeffding: &mut CudaSlice<f64>,
    gini: &mut CudaSlice<f64>,
    flags: &mut CudaSlice<u32>,
) -> Result<()> {
    let n_i32 = to_i32(n, "copula sample count")?;
    let func = assay_function(ctx, "assay.copula_terms_f64", "assay_copula_terms_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(n)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(x_ranks)
            .arg(y_ranks)
            .arg(&n_i32)
            .arg(&tail_q)
            .arg(c_mid)
            .arg(lower)
            .arg(upper)
            .arg(hoeffding)
            .arg(gini)
            .arg(&mut *flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("copula launch failed: {err}")))?;
    sync_and_decode(ctx, "copula_terms", flags)
}

pub(super) fn rank_peak_bytes(n: usize, kendall: bool) -> Result<usize> {
    checked_sum_bytes(&[
        bytes::<f64>(n * 4, "rank inputs and outputs")?,
        bytes::<u32>(n * 2, "rank ties")?,
        bytes::<u64>(if kendall { 2 } else { 0 }, "Kendall counts")?,
        size_of::<u32>(),
    ])
}

pub(super) fn upload_f64(ctx: &CudaContext, values: &[f64], op: &str) -> Result<CudaSlice<f64>> {
    ctx.inner()
        .default_stream()
        .clone_htod(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} upload failed: {err}")))
}

pub(super) fn read_u64(ctx: &CudaContext, values: &CudaSlice<u64>, op: &str) -> Result<Vec<u64>> {
    ctx.inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} readback failed: {err}")))
}

pub(super) fn read_u32(ctx: &CudaContext, values: &CudaSlice<u32>, op: &str) -> Result<Vec<u32>> {
    ctx.inner()
        .default_stream()
        .clone_dtoh(values)
        .map_err(|err| device_unavailable(ctx, format!("{op} readback failed: {err}")))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dependence_stats(
    operation: &str,
    n_samples: usize,
    work_items: usize,
    host_to_device_bytes: usize,
    device_to_host_bytes: usize,
    peak_device_bytes: usize,
    kernel_launches: usize,
) -> CudaDependenceStats {
    CudaDependenceStats {
        operation: operation.to_string(),
        n_samples,
        work_items,
        host_to_device_bytes,
        device_to_host_bytes,
        peak_device_bytes,
        kernel_launches,
    }
}
