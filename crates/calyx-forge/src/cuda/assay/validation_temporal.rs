use super::*;

pub(super) fn validate_periodogram_inputs(
    times: &[f64],
    centered: &[f64],
    variance: f64,
    frequencies: &[f64],
    permutations: Option<&[i32]>,
) -> Result<usize> {
    validate_pair_f64("periodogram_batch_host", times, centered, 1)?;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(numerical(
            "periodogram_batch_host",
            format!("GLS variance must be finite and positive; got {variance}"),
        ));
    }
    if frequencies.is_empty() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: "GLS CUDA requires at least one frequency".to_string(),
        });
    }
    for (idx, &frequency) in frequencies.iter().enumerate() {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(numerical(
                "periodogram_batch_host",
                format!("GLS frequency[{idx}] must be finite and positive; got {frequency}"),
            ));
        }
    }
    validate_permutations(permutations, times.len())
}

pub(super) fn validate_autocorrelation_inputs(
    times: &[f64],
    centered: &[f64],
    variance: f64,
    slot_width: f64,
    max_lag: f64,
    slot_count: usize,
) -> Result<()> {
    validate_pair_f64("autocorrelation_sums_host", times, centered, 1)?;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(numerical(
            "autocorrelation_sums_host",
            format!("ACF variance must be finite and positive; got {variance}"),
        ));
    }
    if !slot_width.is_finite() || slot_width <= 0.0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: format!("ACF slot_width must be finite and positive; got {slot_width}"),
        });
    }
    if !max_lag.is_finite() || max_lag <= 0.0 || slot_count == 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1],
            got: vec![usize::from(max_lag > 0.0), slot_count],
            remediation: format!(
                "ACF max_lag and slot_count must be positive; max_lag={max_lag} slot_count={slot_count}"
            ),
        });
    }
    Ok(())
}

pub(super) fn validate_cross_correlation_inputs(
    x: &[f32],
    y: &[f32],
    max_lag: usize,
    min_pairs: usize,
) -> Result<()> {
    validate_pair_f32("cross_correlation_batch_host", x, y, min_pairs)?;
    if max_lag > x.len().saturating_sub(min_pairs) {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![x.len().saturating_sub(min_pairs)],
            got: vec![max_lag],
            remediation: format!(
                "CCF max_lag {max_lag} leaves fewer than {min_pairs} paired samples at boundary for n={}",
                x.len()
            ),
        });
    }
    Ok(())
}

pub(super) fn validate_hawkes_cuda_inputs(
    events: &[f64],
    offsets: &[i32],
    observation_end: f64,
    decay: f64,
    iterations: usize,
) -> Result<()> {
    if offsets.len() < 2 || offsets.len() - 1 > MAX_HAWKES_PROCESSES {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, MAX_HAWKES_PROCESSES],
            got: vec![offsets.len().saturating_sub(1)],
            remediation: format!("Hawkes CUDA requires 1..={MAX_HAWKES_PROCESSES} event processes"),
        });
    }
    if !observation_end.is_finite() || observation_end <= 0.0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: format!(
                "Hawkes CUDA observation_end must be finite and positive; got {observation_end}"
            ),
        });
    }
    if !decay.is_finite() || decay <= 0.0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1],
            got: vec![0],
            remediation: format!("Hawkes CUDA decay must be finite and positive; got {decay}"),
        });
    }
    if iterations == 0 || iterations > 1_000 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![1, 1_000],
            got: vec![iterations],
            remediation: "Hawkes CUDA iterations must be in 1..=1000".to_string(),
        });
    }
    if offsets[0] != 0 || offsets[offsets.len() - 1] < 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![0, events.len()],
            got: vec![
                offsets[0].max(0) as usize,
                offsets[offsets.len() - 1].max(0) as usize,
            ],
            remediation: "Hawkes CUDA offsets must start at 0 and end at events.len()".to_string(),
        });
    }
    if offsets[offsets.len() - 1] as usize != events.len() {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![events.len()],
            got: vec![offsets[offsets.len() - 1].max(0) as usize],
            remediation: "Hawkes CUDA final offset must equal flattened event count".to_string(),
        });
    }
    for source in 0..offsets.len() - 1 {
        let start = offsets[source];
        let end = offsets[source + 1];
        if start < 0 || end <= start || end - start < 2 {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![2],
                got: vec![(end - start).max(0) as usize],
                remediation: format!(
                    "Hawkes CUDA process {source} requires at least two strictly increasing events"
                ),
            });
        }
        let start = start as usize;
        let end = end as usize;
        for idx in start..end {
            let event = events[idx];
            if !event.is_finite() || event < 0.0 || event >= observation_end {
                return Err(numerical(
                    "hawkes_em_host",
                    format!(
                        "Hawkes event[{idx}] must be finite in [0, observation_end); got {event}"
                    ),
                ));
            }
            if idx > start && events[idx] <= events[idx - 1] {
                return Err(ForgeError::ShapeMismatch {
                    expected: vec![idx],
                    got: vec![idx - 1],
                    remediation: format!(
                        "Hawkes CUDA events must be strictly increasing within process {source}; index {idx}"
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn hawkes_event_process_map(offsets: &[i32]) -> Result<Vec<i32>> {
    let total = offsets[offsets.len() - 1] as usize;
    let mut out = vec![0_i32; total];
    for source in 0..offsets.len() - 1 {
        let start = offsets[source] as usize;
        let end = offsets[source + 1] as usize;
        let source_i32 = to_i32(source, "Hawkes process index")?;
        for slot in &mut out[start..end] {
            *slot = source_i32;
        }
    }
    Ok(out)
}
