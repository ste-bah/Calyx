use std::mem::size_of;

use super::dependence_bulk_support::*;
use super::*;

pub fn histogram_counts_host(
    ctx: &CudaContext,
    x: &[f32],
    y: &[f32],
    bins: usize,
) -> Result<CudaHistogramCounts> {
    validate_pair_f32("histogram_counts_host", x, y, 1)?;
    if bins < 2 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![2],
            got: vec![bins],
            remediation: "histogram_counts_host requires at least two bins".to_string(),
        });
    }
    let cells = bins
        .checked_mul(bins)
        .ok_or_else(|| shape_overflow("histogram cell count overflow"))?;
    let x_min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let x_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let y_min = y.iter().copied().fold(f32::INFINITY, f32::min);
    let y_max = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let x_width = (x_max - x_min).max(f32::EPSILON);
    let y_width = (y_max - y_min).max(f32::EPSILON);
    let peak = checked_sum_bytes(&[
        bytes::<f32>(x.len() + y.len(), "histogram inputs")?,
        bytes::<u64>(bins * 2 + cells, "histogram counts")?,
        bytes::<u32>(1, "histogram flags")?,
    ])?;
    ensure_device_room(ctx, "histogram_counts_host", peak)?;

    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x)
        .map_err(|err| device_unavailable(ctx, format!("histogram x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y)
        .map_err(|err| device_unavailable(ctx, format!("histogram y upload failed: {err}")))?;
    let mut x_counts: CudaSlice<u64> = stream
        .alloc_zeros(bins)
        .map_err(|err| device_unavailable(ctx, format!("histogram x allocation failed: {err}")))?;
    let mut y_counts: CudaSlice<u64> = stream
        .alloc_zeros(bins)
        .map_err(|err| device_unavailable(ctx, format!("histogram y allocation failed: {err}")))?;
    let mut joint_counts: CudaSlice<u64> = stream.alloc_zeros(cells).map_err(|err| {
        device_unavailable(ctx, format!("histogram joint allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "histogram_counts")?;
    let n_i32 = to_i32(x.len(), "histogram sample count")?;
    let bins_i32 = to_i32(bins, "histogram bin count")?;
    let func = assay_function(ctx, "assay.histogram_pair_f32", "assay_histogram_pair_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(x.len())?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&y_dev)
            .arg(&n_i32)
            .arg(&bins_i32)
            .arg(&x_min)
            .arg(&x_width)
            .arg(&y_min)
            .arg(&y_width)
            .arg(&mut x_counts)
            .arg(&mut y_counts)
            .arg(&mut joint_counts)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("histogram launch failed: {err}")))?;
    sync_and_decode(ctx, "histogram_counts", &flags)?;

    let x_counts = read_u64(ctx, &x_counts, "histogram x")?;
    let y_counts = read_u64(ctx, &y_counts, "histogram y")?;
    let joint_counts = read_u64(ctx, &joint_counts, "histogram joint")?;
    let stats = dependence_stats(
        "histogram",
        x.len(),
        cells,
        bytes::<f32>(x.len() + y.len(), "histogram upload")?,
        bytes::<u64>(bins * 2 + cells, "histogram readback")? + size_of::<u32>(),
        peak,
        1,
    );
    Ok(CudaHistogramCounts {
        x_counts,
        y_counts,
        joint_counts,
        stats,
    })
}

pub fn categorical_counts_host(
    ctx: &CudaContext,
    x_dense: &[u32],
    y_dense: &[u32],
    rows: usize,
    cols: usize,
) -> Result<CudaCategoricalCounts> {
    if x_dense.len() != y_dense.len() || x_dense.is_empty() || rows < 2 || cols < 2 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![x_dense.len(), 2, 2],
            got: vec![y_dense.len(), rows, cols],
            remediation: "categorical_counts_host requires paired non-empty dense labels and a table of at least 2x2".to_string(),
        });
    }
    let cells = rows
        .checked_mul(cols)
        .ok_or_else(|| shape_overflow("contingency cell count overflow"))?;
    let peak = checked_sum_bytes(&[
        bytes::<u32>(x_dense.len() + y_dense.len(), "contingency inputs")?,
        bytes::<u64>(cells, "contingency table")?,
        size_of::<u32>(),
    ])?;
    ensure_device_room(ctx, "categorical_counts_host", peak)?;
    let stream = ctx.inner().default_stream();
    let x_dev = stream
        .clone_htod(x_dense)
        .map_err(|err| device_unavailable(ctx, format!("contingency x upload failed: {err}")))?;
    let y_dev = stream
        .clone_htod(y_dense)
        .map_err(|err| device_unavailable(ctx, format!("contingency y upload failed: {err}")))?;
    let mut table: CudaSlice<u64> = stream.alloc_zeros(cells).map_err(|err| {
        device_unavailable(ctx, format!("contingency table allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "categorical_counts")?;
    let n_i32 = to_i32(x_dense.len(), "contingency sample count")?;
    let rows_i32 = to_i32(rows, "contingency row count")?;
    let cols_i32 = to_i32(cols, "contingency column count")?;
    let func = assay_function(ctx, "assay.contingency_u32", "assay_contingency_u32")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_blocks(x_dense.len())?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&x_dev)
            .arg(&y_dev)
            .arg(&n_i32)
            .arg(&rows_i32)
            .arg(&cols_i32)
            .arg(&mut table)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("contingency launch failed: {err}")))?;
    sync_and_decode(ctx, "categorical_counts", &flags)?;
    let table = read_u64(ctx, &table, "contingency table")?;
    let upload = bytes::<u32>(x_dense.len() + y_dense.len(), "contingency upload")?;
    let readback = bytes::<u64>(cells, "contingency readback")? + size_of::<u32>();
    Ok(CudaCategoricalCounts {
        table,
        stats: dependence_stats(
            "categorical",
            x_dense.len(),
            cells,
            upload,
            readback,
            peak,
            1,
        ),
    })
}

pub fn rank_pair_host(ctx: &CudaContext, x: &[f64], y: &[f64]) -> Result<CudaRankPair> {
    validate_pair_f64("rank_pair_host", x, y, 1)?;
    let n = x.len();
    let peak = rank_peak_bytes(n, true)?;
    ensure_device_room(ctx, "rank_pair_host", peak)?;
    let stream = ctx.inner().default_stream();
    let x_dev = upload_f64(ctx, x, "rank x")?;
    let y_dev = upload_f64(ctx, y, "rank y")?;
    let mut x_ranks: CudaSlice<f64> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("rank x output allocation failed: {err}"))
    })?;
    let mut y_ranks: CudaSlice<f64> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("rank y output allocation failed: {err}"))
    })?;
    let mut x_ties: CudaSlice<u32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("rank x tie allocation failed: {err}")))?;
    let mut y_ties: CudaSlice<u32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("rank y tie allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "rank_pair")?;
    launch_rank(
        ctx,
        &x_dev,
        &y_dev,
        n,
        &mut x_ranks,
        &mut y_ranks,
        &mut x_ties,
        &mut y_ties,
        &mut flags,
    )?;
    let mut concordant: CudaSlice<u64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(ctx, format!("Kendall concordant allocation failed: {err}"))
    })?;
    let mut discordant: CudaSlice<u64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(ctx, format!("Kendall discordant allocation failed: {err}"))
    })?;
    launch_kendall(
        ctx,
        &x_dev,
        &y_dev,
        n,
        &mut concordant,
        &mut discordant,
        &mut flags,
    )?;
    let x_ranks = read_device_f64(ctx, "rank x", &x_ranks)?;
    let y_ranks = read_device_f64(ctx, "rank y", &y_ranks)?;
    let x_tie_sizes = read_u32(ctx, &x_ties, "rank x ties")?;
    let y_tie_sizes = read_u32(ctx, &y_ties, "rank y ties")?;
    let concordant = read_u64(ctx, &concordant, "Kendall concordant")?[0];
    let discordant = read_u64(ctx, &discordant, "Kendall discordant")?[0];
    Ok(CudaRankPair {
        x_ranks,
        y_ranks,
        x_tie_sizes,
        y_tie_sizes,
        concordant,
        discordant,
        stats: dependence_stats(
            "rank",
            n,
            n.saturating_mul(n),
            bytes::<f64>(n * 2, "rank upload")?,
            bytes::<f64>(n * 2, "rank readback")?
                + bytes::<u32>(n * 2, "tie readback")?
                + bytes::<u64>(2, "Kendall readback")?
                + 2 * size_of::<u32>(),
            peak,
            2,
        ),
    })
}

pub fn copula_terms_host(
    ctx: &CudaContext,
    x: &[f64],
    y: &[f64],
    tail_q: f64,
) -> Result<CudaCopulaTerms> {
    validate_pair_f64("copula_terms_host", x, y, 1)?;
    if !(tail_q.is_finite() && tail_q > 0.0 && tail_q < 0.5) {
        return Err(numerical(
            "copula_terms_host",
            format!("tail_q must be finite in (0, 0.5); got {tail_q}"),
        ));
    }
    let n = x.len();
    let peak = rank_peak_bytes(n, false)? + bytes::<f64>(n * 2, "copula terms")? + 24;
    ensure_device_room(ctx, "copula_terms_host", peak)?;
    let stream = ctx.inner().default_stream();
    let x_dev = upload_f64(ctx, x, "copula x")?;
    let y_dev = upload_f64(ctx, y, "copula y")?;
    let mut x_ranks: CudaSlice<f64> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("copula x rank allocation failed: {err}"))
    })?;
    let mut y_ranks: CudaSlice<f64> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("copula y rank allocation failed: {err}"))
    })?;
    let mut x_ties: CudaSlice<u32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("copula x tie allocation failed: {err}")))?;
    let mut y_ties: CudaSlice<u32> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("copula y tie allocation failed: {err}")))?;
    let mut flags = alloc_flags(ctx, "copula_rank")?;
    launch_rank(
        ctx,
        &x_dev,
        &y_dev,
        n,
        &mut x_ranks,
        &mut y_ranks,
        &mut x_ties,
        &mut y_ties,
        &mut flags,
    )?;
    let mut c_mid: CudaSlice<u64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(ctx, format!("copula midpoint allocation failed: {err}"))
    })?;
    let mut lower: CudaSlice<u64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(ctx, format!("copula lower-tail allocation failed: {err}"))
    })?;
    let mut upper: CudaSlice<u64> = stream.alloc_zeros(1).map_err(|err| {
        device_unavailable(ctx, format!("copula upper-tail allocation failed: {err}"))
    })?;
    let mut hoeffding: CudaSlice<f64> = stream.alloc_zeros(n).map_err(|err| {
        device_unavailable(ctx, format!("copula Hoeffding allocation failed: {err}"))
    })?;
    let mut gini: CudaSlice<f64> = stream
        .alloc_zeros(n)
        .map_err(|err| device_unavailable(ctx, format!("copula Gini allocation failed: {err}")))?;
    launch_copula(
        ctx,
        &x_ranks,
        &y_ranks,
        n,
        tail_q,
        &mut c_mid,
        &mut lower,
        &mut upper,
        &mut hoeffding,
        &mut gini,
        &mut flags,
    )?;
    let x_tie_sizes = read_u32(ctx, &x_ties, "copula x ties")?;
    let y_tie_sizes = read_u32(ctx, &y_ties, "copula y ties")?;
    let counts = [
        read_u64(ctx, &c_mid, "copula midpoint")?[0],
        read_u64(ctx, &lower, "copula lower tail")?[0],
        read_u64(ctx, &upper, "copula upper tail")?[0],
    ];
    let hoeffding_terms = read_device_f64(ctx, "copula Hoeffding terms", &hoeffding)?;
    let gini_terms = read_device_f64(ctx, "copula Gini terms", &gini)?;
    Ok(CudaCopulaTerms {
        x_tie_sizes,
        y_tie_sizes,
        c_mid_count: counts[0],
        lower_tail_count: counts[1],
        upper_tail_count: counts[2],
        hoeffding_terms,
        gini_terms,
        stats: dependence_stats(
            "copula",
            n,
            n.saturating_mul(n),
            bytes::<f64>(n * 2, "copula upload")?,
            bytes::<u32>(n * 2, "copula tie readback")?
                + bytes::<f64>(n * 2, "copula term readback")?
                + bytes::<u64>(3, "copula count readback")?
                + 2 * size_of::<u32>(),
            peak,
            2,
        ),
    })
}
