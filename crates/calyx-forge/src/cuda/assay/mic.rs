use std::mem::size_of;

use super::*;

pub fn mic_pair_host(
    ctx: &CudaContext,
    x: &[f64],
    y: &[f64],
    b_budget: usize,
) -> Result<CudaMicPair> {
    validate_pair_f64("mic_pair_host", x, y, 4)?;
    if b_budget < 4 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![4],
            got: vec![b_budget],
            remediation: "mic_pair_host requires a grid-cell budget of at least four".to_string(),
        });
    }
    if distinct_count(x) < 2 || distinct_count(y) < 2 {
        return Err(numerical(
            "mic_pair_host",
            "MIC requires at least two distinct values per margin".to_string(),
        ));
    }
    let mut packed = PackedMic::default();
    append_orientation(x, y, b_budget, &mut packed)?;
    let split = packed.candidate_count();
    append_orientation(y, x, b_budget, &mut packed)?;
    let candidate_count = packed.candidate_count();
    if split == 0 || split == candidate_count {
        return Err(numerical(
            "mic_pair_host",
            "MIC preprocessing produced no valid candidates for an orientation".to_string(),
        ));
    }
    let scratch_cells = candidate_count
        .checked_mul(2)
        .and_then(|value| value.checked_mul(x.len() + 1))
        .ok_or_else(|| shape_overflow("MIC rolling scratch size overflow"))?;
    let (weighted_log2, entropy_terms, integer_log2) = mic_log_tables(x.len());
    let upload_bytes =
        packed.upload_bytes()? + bytes::<f64>(weighted_log2.len() * 3, "MIC numerical tables")?;
    let output_bytes = checked_sum_bytes(&[
        bytes::<f64>(candidate_count, "MIC scores")?,
        bytes::<i32>(candidate_count * 2, "MIC winning bins")?,
        size_of::<u32>(),
    ])?;
    let peak = checked_sum_bytes(&[
        upload_bytes,
        bytes::<f64>(scratch_cells, "MIC rolling scratch")?,
        output_bytes,
    ])?;
    ensure_device_room(ctx, "mic_pair_host", peak)?;

    let stream = ctx.inner().default_stream();
    let cumulative = stream.clone_htod(&packed.cumulative).map_err(|err| {
        device_unavailable(ctx, format!("MIC cumulative-count upload failed: {err}"))
    })?;
    let cumulative_offsets = stream
        .clone_htod(&packed.cumulative_offsets)
        .map_err(|err| {
            device_unavailable(ctx, format!("MIC cumulative-offset upload failed: {err}"))
        })?;
    let cuts = stream
        .clone_htod(&packed.cuts)
        .map_err(|err| device_unavailable(ctx, format!("MIC cut upload failed: {err}")))?;
    let cut_offsets = stream
        .clone_htod(&packed.cut_offsets)
        .map_err(|err| device_unavailable(ctx, format!("MIC cut-offset upload failed: {err}")))?;
    let secondary_bins = stream.clone_htod(&packed.secondary_bins).map_err(|err| {
        device_unavailable(ctx, format!("MIC secondary-bin upload failed: {err}"))
    })?;
    let max_primary = stream
        .clone_htod(&packed.max_primary)
        .map_err(|err| device_unavailable(ctx, format!("MIC primary-bin upload failed: {err}")))?;
    let weighted_log2 = stream.clone_htod(&weighted_log2).map_err(|err| {
        device_unavailable(ctx, format!("MIC weighted-log table upload failed: {err}"))
    })?;
    let entropy_terms = stream.clone_htod(&entropy_terms).map_err(|err| {
        device_unavailable(ctx, format!("MIC entropy table upload failed: {err}"))
    })?;
    let integer_log2 = stream.clone_htod(&integer_log2).map_err(|err| {
        device_unavailable(ctx, format!("MIC integer-log table upload failed: {err}"))
    })?;
    let mut scratch: CudaSlice<f64> = stream.alloc_zeros(scratch_cells).map_err(|err| {
        device_unavailable(ctx, format!("MIC rolling scratch allocation failed: {err}"))
    })?;
    let mut scores: CudaSlice<f64> = stream
        .alloc_zeros(candidate_count)
        .map_err(|err| device_unavailable(ctx, format!("MIC score allocation failed: {err}")))?;
    let mut primary_out: CudaSlice<i32> = stream.alloc_zeros(candidate_count).map_err(|err| {
        device_unavailable(ctx, format!("MIC primary-bin allocation failed: {err}"))
    })?;
    let mut secondary_out: CudaSlice<i32> = stream.alloc_zeros(candidate_count).map_err(|err| {
        device_unavailable(ctx, format!("MIC secondary-bin allocation failed: {err}"))
    })?;
    let mut flags = alloc_flags(ctx, "mic_candidates")?;
    let n_i32 = to_i32(x.len(), "MIC sample count")?;
    let candidate_i32 = to_i32(candidate_count, "MIC candidate count")?;
    let func = assay_function(ctx, "assay.mic_candidates_f64", "assay_mic_candidates_f64")?;
    let cfg = LaunchConfig {
        grid_dim: (to_u32(candidate_count, "MIC candidate grid")?, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(func.as_ref());
    unsafe {
        launch
            .arg(&cumulative)
            .arg(&cumulative_offsets)
            .arg(&cuts)
            .arg(&cut_offsets)
            .arg(&secondary_bins)
            .arg(&max_primary)
            .arg(&weighted_log2)
            .arg(&entropy_terms)
            .arg(&integer_log2)
            .arg(&n_i32)
            .arg(&candidate_i32)
            .arg(&mut scratch)
            .arg(&mut scores)
            .arg(&mut primary_out)
            .arg(&mut secondary_out)
            .arg(&mut flags)
            .launch(cfg)
    }
    .map_err(|err| device_unavailable(ctx, format!("MIC candidate launch failed: {err}")))?;
    sync_and_decode(ctx, "mic_candidates", &flags)?;
    let scores = read_device_f64(ctx, "MIC score", &scores)?;
    let primary_out = read_device_i32(ctx, "MIC primary bins", &primary_out)?;
    let secondary_out = read_device_i32(ctx, "MIC secondary bins", &secondary_out)?;
    let primary_x = select_scan(&scores, &primary_out, &secondary_out, 0, split)?;
    let primary_y = select_scan(
        &scores,
        &primary_out,
        &secondary_out,
        split,
        candidate_count,
    )?;
    Ok(CudaMicPair {
        primary_x,
        primary_y,
        stats: CudaDependenceStats {
            operation: "mic".to_string(),
            n_samples: x.len(),
            work_items: candidate_count
                .saturating_mul(x.len())
                .saturating_mul(x.len()),
            host_to_device_bytes: upload_bytes,
            device_to_host_bytes: output_bytes,
            peak_device_bytes: peak,
            kernel_launches: 1,
        },
    })
}

fn mic_log_tables(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut weighted_log2 = vec![0.0; n + 1];
    let mut entropy_terms = vec![0.0; n + 1];
    let mut integer_log2 = vec![0.0; n + 1];
    let nf = n as f64;
    for value in 1..=n {
        let value_f64 = value as f64;
        let log = value_f64.log2();
        integer_log2[value] = log;
        weighted_log2[value] = value_f64 * log;
        let probability = value_f64 / nf;
        entropy_terms[value] = probability * probability.log2();
    }
    (weighted_log2, entropy_terms, integer_log2)
}

#[derive(Default)]
struct PackedMic {
    cumulative: Vec<u32>,
    cumulative_offsets: Vec<u64>,
    cuts: Vec<u32>,
    cut_offsets: Vec<u64>,
    secondary_bins: Vec<i32>,
    max_primary: Vec<i32>,
}

impl PackedMic {
    fn candidate_count(&self) -> usize {
        self.secondary_bins.len()
    }

    fn upload_bytes(&self) -> Result<usize> {
        checked_sum_bytes(&[
            bytes::<u32>(self.cumulative.len() + self.cuts.len(), "MIC u32 inputs")?,
            bytes::<u64>(
                self.cumulative_offsets.len() + self.cut_offsets.len(),
                "MIC offsets",
            )?,
            bytes::<i32>(
                self.secondary_bins.len() + self.max_primary.len(),
                "MIC dimensions",
            )?,
        ])
    }
}

fn append_orientation(
    primary: &[f64],
    secondary: &[f64],
    b_budget: usize,
    packed: &mut PackedMic,
) -> Result<()> {
    let n = primary.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        primary[a]
            .partial_cmp(&primary[b])
            .expect("finite-validated")
    });
    let primary_sorted: Vec<f64> = order.iter().map(|&index| primary[index]).collect();
    let n_primary_groups = 1
        + (1..n)
            .filter(|&index| primary_sorted[index] != primary_sorted[index - 1])
            .count();
    let cut_offset = u64::try_from(packed.cuts.len())
        .map_err(|_| shape_overflow("MIC cut offset exceeds u64"))?;
    packed.cuts.extend((0..=n).map(|index| {
        u32::from(index >= 1 && index < n && primary_sorted[index] != primary_sorted[index - 1])
    }));
    for requested_secondary in 2..=b_budget / 2 {
        let bins = equipartition(secondary, requested_secondary);
        let actual_secondary = bins.iter().copied().max().unwrap_or(0) + 1;
        let max_primary = (b_budget / requested_secondary).min(n_primary_groups);
        if actual_secondary < 2 || max_primary < 2 {
            continue;
        }
        let cumulative_offset = u64::try_from(packed.cumulative.len())
            .map_err(|_| shape_overflow("MIC cumulative offset exceeds u64"))?;
        packed.cumulative_offsets.push(cumulative_offset);
        packed.cut_offsets.push(cut_offset);
        packed
            .secondary_bins
            .push(to_i32(actual_secondary, "MIC actual secondary bins")?);
        packed
            .max_primary
            .push(to_i32(max_primary, "MIC maximum primary bins")?);
        let start = packed.cumulative.len();
        packed
            .cumulative
            .resize(start + (n + 1) * actual_secondary, 0);
        for (rank, &original) in order.iter().enumerate() {
            let previous = start + rank * actual_secondary;
            let next = previous + actual_secondary;
            for category in 0..actual_secondary {
                packed.cumulative[next + category] = packed.cumulative[previous + category];
            }
            packed.cumulative[next + bins[original]] += 1;
        }
    }
    Ok(())
}

fn equipartition(values: &[f64], k: usize) -> Vec<usize> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).expect("finite-validated"));
    let mut boundaries = Vec::new();
    for bin in 1..k {
        let ideal = ((bin as f64) * (n as f64) / (k as f64)).round() as usize;
        if let Some(position) = snap_to_value_boundary(&order, values, ideal)
            && boundaries.last() != Some(&position)
        {
            boundaries.push(position);
        }
    }
    let mut out = vec![0; n];
    let mut current = 0;
    let mut boundary = 0;
    for (rank, &original) in order.iter().enumerate() {
        while boundary < boundaries.len() && rank == boundaries[boundary] {
            current += 1;
            boundary += 1;
        }
        out[original] = current;
    }
    out
}

fn snap_to_value_boundary(order: &[usize], values: &[f64], ideal: usize) -> Option<usize> {
    let n = order.len();
    let valid = |position: usize| {
        position >= 1 && position < n && values[order[position]] != values[order[position - 1]]
    };
    if valid(ideal) {
        return Some(ideal);
    }
    for delta in 1..n {
        let up = ideal + delta;
        if up < n && valid(up) {
            return Some(up);
        }
        if ideal >= delta && valid(ideal - delta) {
            return Some(ideal - delta);
        }
    }
    None
}

fn distinct_count(values: &[f64]) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite-validated"));
    sorted.dedup();
    sorted.len()
}

fn select_scan(
    scores: &[f64],
    primary: &[i32],
    secondary: &[i32],
    start: usize,
    end: usize,
) -> Result<CudaMicScan> {
    let mut best = CudaMicScan {
        score: 0.0,
        primary_bins: 2,
        secondary_bins: 2,
    };
    for index in start..end {
        if scores[index] > best.score {
            best = CudaMicScan {
                score: scores[index],
                primary_bins: usize::try_from(primary[index])
                    .map_err(|_| numerical("mic_pair_host", "negative primary bins".to_string()))?,
                secondary_bins: usize::try_from(secondary[index]).map_err(|_| {
                    numerical("mic_pair_host", "negative secondary bins".to_string())
                })?,
            };
        }
    }
    Ok(best)
}
