use super::*;

/// Bins event timestamps into a uniform count series `(bin_centres, counts)`
/// so point-process recurrence streams can feed [`lomb_scargle`]. Zero-count
/// bins are real observations of "no events" and are included.
pub fn bin_event_counts(event_times: &[f64], bin_width: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    if event_times.is_empty() {
        return Err(CalyxError::assay_insufficient_samples(
            "bin_event_counts requires at least one event time",
        ));
    }
    if !bin_width.is_finite() || bin_width <= 0.0 {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "bin_event_counts bin_width must be finite and positive, got {bin_width}"
        )));
    }
    for (index, pair) in event_times.windows(2).enumerate() {
        if !pair[0].is_finite() || pair[0] > pair[1] {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "event time at index {index} is non-finite or decreasing"
            )));
        }
    }
    let last = *event_times.last().expect("non-empty checked above");
    if !last.is_finite() {
        return Err(CalyxError::assay_insufficient_samples(
            "final event time is non-finite",
        ));
    }
    let first = event_times[0];
    let bin_count = ((last - first) / bin_width).floor() as usize + 1;
    if bin_count > MAX_FREQUENCY_GRID {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "bin_event_counts would produce {bin_count} bins (max {MAX_FREQUENCY_GRID}); \
             widen bin_width"
        )));
    }
    let mut counts = vec![0.0_f64; bin_count];
    for &event in event_times {
        let bin = (((event - first) / bin_width).floor() as usize).min(bin_count - 1);
        counts[bin] += 1.0;
    }
    let centres = (0..bin_count)
        .map(|bin| first + (bin as f64 + 0.5) * bin_width)
        .collect();
    Ok((centres, counts))
}
