use super::*;

struct HawkesEventBuffers<'a> {
    events: &'a CudaSlice<f64>,
    offsets: &'a CudaSlice<i32>,
    event_process: &'a CudaSlice<i32>,
}

struct HawkesEmBuffers<'a> {
    offsets: &'a CudaSlice<i32>,
    kernel_sums: &'a CudaSlice<f64>,
    baseline: &'a CudaSlice<f64>,
    branching: &'a CudaSlice<f64>,
}

pub fn hawkes_em_host(
    ctx: &CudaContext,
    events: &[f64],
    offsets: &[i32],
    observation_end: f64,
    decay: f64,
    iterations: usize,
) -> Result<CudaHawkesFit> {
    validate_hawkes_cuda_inputs(events, offsets, observation_end, decay, iterations)?;
    let d = offsets.len() - 1;
    let total_events = events.len();
    let matrix_len = checked_square(d, "Hawkes branching matrix length")?;
    let kernel_len = total_events
        .checked_mul(d)
        .ok_or_else(|| shape_overflow("Hawkes kernel-sum shape overflow"))?;
    ensure_device_room(
        ctx,
        "hawkes_em_host",
        checked_sum_bytes(&[
            bytes::<f64>(events.len(), "Hawkes events")?,
            bytes::<i32>(offsets.len(), "Hawkes offsets")?,
            bytes::<i32>(events.len(), "Hawkes event process map")?,
            bytes::<f64>(d, "Hawkes exposures")?,
            bytes::<f64>(kernel_len, "Hawkes kernel sums")?,
            bytes::<f64>(4 * d, "Hawkes baseline/current/next/background")?,
            bytes::<f64>(4 * matrix_len, "Hawkes branching/current/next/triggered")?,
            bytes::<f64>(1, "Hawkes spectral radius")?,
        ])?,
    )?;

    let stream = ctx.inner().default_stream();
    let event_process = hawkes_event_process_map(offsets)?;
    let mut baseline = Vec::with_capacity(d);
    for source in 0..d {
        let count = (offsets[source + 1] - offsets[source]) as f64;
        baseline.push(0.5 * count / observation_end);
    }
    let branching = vec![0.05_f64; matrix_len];
    let events_dev = stream
        .clone_htod(events)
        .map_err(|err| device_unavailable(ctx, format!("Hawkes events upload failed: {err}")))?;
    let offsets_dev = stream
        .clone_htod(offsets)
        .map_err(|err| device_unavailable(ctx, format!("Hawkes offsets upload failed: {err}")))?;
    let event_process_dev = stream.clone_htod(&event_process).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes event-process map upload failed: {err}"),
        )
    })?;
    let mut baseline_dev = stream
        .clone_htod(&baseline)
        .map_err(|err| device_unavailable(ctx, format!("Hawkes baseline upload failed: {err}")))?;
    let mut branching_dev = stream
        .clone_htod(&branching)
        .map_err(|err| device_unavailable(ctx, format!("Hawkes branching upload failed: {err}")))?;
    let mut exposures: CudaSlice<f64> = stream.alloc_zeros(d).map_err(|err| {
        device_unavailable(ctx, format!("Hawkes exposures allocation failed: {err}"))
    })?;
    let mut kernel_sums: CudaSlice<f64> = stream.alloc_zeros(kernel_len).map_err(|err| {
        device_unavailable(ctx, format!("Hawkes kernel sums allocation failed: {err}"))
    })?;
    let mut next_baseline: CudaSlice<f64> = stream.alloc_zeros(d).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes next baseline allocation failed: {err}"),
        )
    })?;
    let mut next_branching: CudaSlice<f64> = stream.alloc_zeros(matrix_len).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes next branching allocation failed: {err}"),
        )
    })?;
    let mut background_counts: CudaSlice<f64> = stream.alloc_zeros(d).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes background counts allocation failed: {err}"),
        )
    })?;
    let mut triggered_counts: CudaSlice<f64> = stream.alloc_zeros(matrix_len).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes triggered counts allocation failed: {err}"),
        )
    })?;

    launch_hawkes_exposures(
        ctx,
        &events_dev,
        &offsets_dev,
        d,
        observation_end,
        decay,
        &mut exposures,
    )?;
    launch_hawkes_kernel_sums(
        ctx,
        HawkesEventBuffers {
            events: &events_dev,
            offsets: &offsets_dev,
            event_process: &event_process_dev,
        },
        d,
        total_events,
        decay,
        &mut kernel_sums,
    )?;

    for iteration in 0..iterations {
        let em_buffers = HawkesEmBuffers {
            offsets: &offsets_dev,
            kernel_sums: &kernel_sums,
            baseline: &baseline_dev,
            branching: &branching_dev,
        };
        launch_hawkes_background(ctx, iteration, &em_buffers, d, &mut background_counts)?;
        launch_hawkes_triggered(ctx, iteration, &em_buffers, d, &mut triggered_counts)?;
        launch_hawkes_update(
            ctx,
            iteration,
            &background_counts,
            &triggered_counts,
            &exposures,
            d,
            observation_end,
            &mut next_baseline,
            &mut next_branching,
        )?;
        std::mem::swap(&mut baseline_dev, &mut next_baseline);
        std::mem::swap(&mut branching_dev, &mut next_branching);
    }

    let mut spectral: CudaSlice<f64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes spectral radius allocation failed: {err}"),
        )
    })?;
    launch_hawkes_spectral_radius(ctx, &branching_dev, d, &mut spectral)?;
    let baseline = read_device_f64(ctx, "Hawkes baseline readback", &baseline_dev)?;
    let branching = read_device_f64(ctx, "Hawkes branching readback", &branching_dev)?;
    let spectral = read_device_f64(ctx, "Hawkes spectral readback", &spectral)?;
    validate_matrix_readback("Hawkes baseline readback", &baseline)?;
    validate_matrix_readback("Hawkes branching readback", &branching)?;
    validate_matrix_readback("Hawkes spectral readback", &spectral)?;
    Ok(CudaHawkesFit {
        baseline_rates: baseline.into_iter().map(|value| value as f32).collect(),
        branching_matrix: branching.into_iter().map(|value| value as f32).collect(),
        spectral_radius: spectral[0] as f32,
    })
}

pub(super) fn launch_hawkes_exposures(
    ctx: &CudaContext,
    events: &CudaSlice<f64>,
    offsets: &CudaSlice<i32>,
    d: usize,
    observation_end: f64,
    decay: f64,
    exposures: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_exposures")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_exposures_f64",
        "assay_hawkes_exposures_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(d, "Hawkes exposure grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(events)
            .arg(offsets)
            .arg(&d_i32)
            .arg(&observation_end)
            .arg(&decay)
            .arg(exposures)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("Hawkes exposures launch failed: {err}")))?;
    sync_and_decode(ctx, "hawkes_exposures", &flags)
}

fn launch_hawkes_kernel_sums(
    ctx: &CudaContext,
    buffers: HawkesEventBuffers<'_>,
    d: usize,
    total_events: usize,
    decay: f64,
    kernel_sums: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_kernel_sums")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_kernel_sums_f64",
        "assay_hawkes_kernel_sums_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let total_i32 = to_i32(total_events, "Hawkes total event count")?;
    let cfg = LaunchConfig {
        grid_dim: (
            to_u32(total_events, "Hawkes event grid")?,
            to_u32(d, "Hawkes source grid")?,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(buffers.events)
            .arg(buffers.offsets)
            .arg(buffers.event_process)
            .arg(&d_i32)
            .arg(&total_i32)
            .arg(&decay)
            .arg(kernel_sums)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("Hawkes kernel-sum launch failed: {err}")))?;
    sync_and_decode(ctx, "hawkes_kernel_sums", &flags)
}

fn launch_hawkes_background(
    ctx: &CudaContext,
    iteration: usize,
    buffers: &HawkesEmBuffers<'_>,
    d: usize,
    background_counts: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_em_background")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_em_background_f64",
        "assay_hawkes_em_background_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(d, "Hawkes background grid")?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(buffers.offsets)
            .arg(buffers.kernel_sums)
            .arg(buffers.baseline)
            .arg(buffers.branching)
            .arg(&d_i32)
            .arg(background_counts)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes EM background launch failed at iteration {iteration}: {err}"),
        )
    })?;
    sync_and_decode(ctx, "hawkes_em_background", &flags)
}

fn launch_hawkes_triggered(
    ctx: &CudaContext,
    iteration: usize,
    buffers: &HawkesEmBuffers<'_>,
    d: usize,
    triggered_counts: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_em_triggered")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_em_triggered_f64",
        "assay_hawkes_em_triggered_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let cfg = LaunchConfig {
        grid_dim: (
            to_u32(d, "Hawkes triggered source grid")?,
            to_u32(d, "Hawkes triggered target grid")?,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(buffers.offsets)
            .arg(buffers.kernel_sums)
            .arg(buffers.baseline)
            .arg(buffers.branching)
            .arg(&d_i32)
            .arg(triggered_counts)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes EM triggered launch failed at iteration {iteration}: {err}"),
        )
    })?;
    sync_and_decode(ctx, "hawkes_em_triggered", &flags)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_hawkes_update(
    ctx: &CudaContext,
    iteration: usize,
    background_counts: &CudaSlice<f64>,
    triggered_counts: &CudaSlice<f64>,
    exposures: &CudaSlice<f64>,
    d: usize,
    observation_end: f64,
    next_baseline: &mut CudaSlice<f64>,
    next_branching: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_em_update")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_em_update_f64",
        "assay_hawkes_em_update_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let cfg = LaunchConfig {
        grid_dim: (
            to_u32(d, "Hawkes update source grid")?,
            to_u32(d, "Hawkes update target grid")?,
            1,
        ),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(background_counts)
            .arg(triggered_counts)
            .arg(exposures)
            .arg(&d_i32)
            .arg(&observation_end)
            .arg(next_baseline)
            .arg(next_branching)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(
            ctx,
            format!("Hawkes EM update launch failed at iteration {iteration}: {err}"),
        )
    })?;
    sync_and_decode(ctx, "hawkes_em_update", &flags)
}

pub(super) fn launch_hawkes_spectral_radius(
    ctx: &CudaContext,
    branching: &CudaSlice<f64>,
    d: usize,
    spectral: &mut CudaSlice<f64>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut flags = alloc_flags(ctx, "hawkes_spectral_radius")?;
    let func = assay_function(
        ctx,
        "assay.hawkes_spectral_radius_f64",
        "assay_hawkes_spectral_radius_f64",
    )?;
    let d_i32 = to_i32(d, "Hawkes process count")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(branching)
            .arg(&d_i32)
            .arg(spectral)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| {
        device_unavailable(ctx, format!("Hawkes spectral radius launch failed: {err}"))
    })?;
    sync_and_decode(ctx, "hawkes_spectral_radius", &flags)
}
