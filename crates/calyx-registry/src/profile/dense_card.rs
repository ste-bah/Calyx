use calyx_core::{CalyxError, LensId, Result};

use crate::spec::LensHealth;

use super::dense_matrix::{DenseObservationMatrix, PairwiseDistanceMatrix};
use super::{
    CapabilityCard, CapabilitySignalKind, CostMetrics, CoverageMetrics, MetricSource,
    ProfileExecutionStats, ProfileOptions, SeparationMetrics, SpreadMetrics,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Observation {
    pub(crate) data: Vec<f32>,
    pub(crate) label: Option<String>,
}

#[derive(Clone, Debug)]
pub struct DenseProfileRequest<'a> {
    pub lens_id: LensId,
    pub probe_count: usize,
    pub vectors: &'a [Vec<f32>],
    pub labels: &'a [Option<String>],
    pub cost: CostMetrics,
    pub signal: Option<f32>,
    pub signal_kind: CapabilitySignalKind,
    pub health: LensHealth,
    pub execution: ProfileExecutionStats,
}

pub(super) struct DenseCapabilityRequest {
    pub(super) lens_id: LensId,
    pub(super) probe_count: usize,
    pub(super) observations: Vec<Observation>,
    pub(super) cost: CostMetrics,
    pub(super) signal: Option<f32>,
    pub(super) signal_kind: CapabilitySignalKind,
    pub(super) health: LensHealth,
    pub(super) options: ProfileOptions,
    pub(super) execution: super::ProfileExecutionStats,
}

pub fn profile_dense_vectors(request: DenseProfileRequest<'_>) -> Result<CapabilityCard> {
    let observations = request
        .vectors
        .iter()
        .enumerate()
        .map(|(idx, data)| Observation {
            data: data.clone(),
            label: request.labels.get(idx).cloned().unwrap_or(None),
        })
        .collect();
    dense_capability_card(DenseCapabilityRequest {
        lens_id: request.lens_id,
        probe_count: request.probe_count,
        observations,
        cost: request.cost,
        signal: request.signal,
        signal_kind: request.signal_kind,
        health: request.health,
        options: ProfileOptions::default(),
        execution: request.execution,
    })
}

pub(super) fn dense_capability_card(request: DenseCapabilityRequest) -> Result<CapabilityCard> {
    let DenseCapabilityRequest {
        lens_id,
        probe_count,
        observations,
        cost,
        signal,
        signal_kind,
        health,
        options,
        mut execution,
    } = request;
    if probe_count == 0 || observations.is_empty() {
        return Err(CalyxError::assay_insufficient_samples(
            "profile requires at least one measurable vector",
        ));
    }
    let matrix = DenseObservationMatrix::from_observations(observations)?;
    let distances = matrix.pairwise_distances()?;
    execution = merge_execution(execution, distances.stats());
    let spread = spread_metrics(&matrix, &distances);
    let separation = separation_metrics(&matrix, &distances);
    let measured = matrix.rows();
    let coverage = CoverageMetrics {
        requested: probe_count,
        measured,
        failed: probe_count.saturating_sub(measured),
        rate: measured as f32 / probe_count as f32,
    };
    let signal = signal.map(|bits| if bits.is_finite() { bits.max(0.0) } else { 0.0 });
    let signal_source = if signal.is_some() {
        MetricSource::AssayStore
    } else {
        MetricSource::AssayPending
    };
    let proxy_differentiation = separation.score;
    let proxy_signal = clamp01(
        coverage.rate
            * spread.normalized_participation_ratio
            * proxy_differentiation.clamp(0.0, 1.0),
    );
    Ok(CapabilityCard {
        lens_id,
        probe_count,
        signal,
        signal_source,
        signal_kind,
        signal_reliability: None,
        proxy_signal,
        differentiation: signal,
        differentiation_source: signal_source,
        proxy_differentiation,
        spread,
        separation,
        cost,
        coverage,
        health,
        low_spread: spread.normalized_participation_ratio < options.low_spread_threshold
            || spread.mean_pairwise_distance < options.low_distance_threshold,
        execution,
    })
}

fn spread_metrics(
    matrix: &DenseObservationMatrix,
    distances: &PairwiseDistanceMatrix,
) -> SpreadMetrics {
    let dim = matrix.dim();
    let mean = mean_vector(matrix);
    let mut variances = vec![0.0_f32; dim];
    for row in 0..matrix.rows() {
        for (idx, value) in matrix.row(row).iter().enumerate() {
            let delta = *value - mean[idx];
            variances[idx] += delta * delta;
        }
    }
    let inv_n = 1.0 / matrix.rows() as f32;
    for value in &mut variances {
        *value *= inv_n;
    }

    let total_variance: f32 = variances.iter().sum();
    let variance_square_sum: f32 = variances.iter().map(|value| value * value).sum();
    let max_variance = variances.iter().copied().fold(0.0_f32, f32::max);
    let participation_ratio = if variance_square_sum <= f32::EPSILON {
        0.0
    } else {
        (total_variance * total_variance) / variance_square_sum
    };
    let stable_rank = if max_variance <= f32::EPSILON {
        0.0
    } else {
        total_variance / max_variance
    };
    let mean_pairwise_distance = distances.mean_upper_triangle();

    SpreadMetrics {
        participation_ratio,
        normalized_participation_ratio: participation_ratio / dim as f32,
        stable_rank,
        total_variance,
        mean_pairwise_distance,
    }
}

fn separation_metrics(
    matrix: &DenseObservationMatrix,
    distances: &PairwiseDistanceMatrix,
) -> SeparationMetrics {
    let mean_pairwise_distance = distances.mean_upper_triangle();
    let groups = matrix.label_groups();
    let used_labels = groups.len() >= 2;
    let silhouette = if used_labels {
        silhouette_score(matrix, distances, &groups)
    } else {
        0.0
    };
    let score = if used_labels {
        silhouette
    } else {
        mean_pairwise_distance
    };

    SeparationMetrics {
        score,
        silhouette,
        mean_pairwise_distance,
        labeled_groups: groups.len(),
        used_labels,
    }
}

fn mean_vector(matrix: &DenseObservationMatrix) -> Vec<f32> {
    let mut mean = vec![0.0_f32; matrix.dim()];
    for row in 0..matrix.rows() {
        for (dst, src) in mean.iter_mut().zip(matrix.row(row)) {
            *dst += *src;
        }
    }
    let inv_n = 1.0 / matrix.rows() as f32;
    for value in &mut mean {
        *value *= inv_n;
    }
    mean
}

fn silhouette_score(
    matrix: &DenseObservationMatrix,
    distances: &PairwiseDistanceMatrix,
    groups: &std::collections::BTreeMap<String, Vec<usize>>,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for (idx, label) in matrix.labels().iter().enumerate() {
        let Some(label) = label else {
            continue;
        };
        let Some(same) = groups.get(label) else {
            continue;
        };
        let a = mean_distance_to_group(idx, distances, same, true);
        let mut b = f32::INFINITY;
        for (other_label, group) in groups {
            if other_label == label {
                continue;
            }
            b = b.min(mean_distance_to_group(idx, distances, group, false));
        }
        let denom = a.max(b);
        let score = if denom <= f32::EPSILON {
            0.0
        } else {
            (b - a) / denom
        };
        sum += score;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn mean_distance_to_group(
    idx: usize,
    distances: &PairwiseDistanceMatrix,
    group: &[usize],
    skip_self: bool,
) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for &other in group {
        if skip_self && other == idx {
            continue;
        }
        sum += distances.get(idx, other);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn merge_execution(
    mut base: ProfileExecutionStats,
    pairwise: ProfileExecutionStats,
) -> ProfileExecutionStats {
    base.resident_matrices = pairwise.resident_matrices;
    base.pairwise_distance_matrices = pairwise.pairwise_distance_matrices;
    base.pairwise_distance_values = pairwise.pairwise_distance_values;
    base.pairwise_distance_backend = pairwise.pairwise_distance_backend;
    base.pairwise_tile_rows = pairwise.pairwise_tile_rows;
    base.pairwise_tiles = pairwise.pairwise_tiles;
    base.measured_rows = pairwise.measured_rows;
    base.vector_dim = pairwise.vector_dim;
    base
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}
