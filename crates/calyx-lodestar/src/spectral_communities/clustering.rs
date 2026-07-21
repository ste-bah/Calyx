use calyx_core::CxId;
use calyx_mincut::EigenPair;
use calyx_paths::AssocGraph;

use crate::{LodestarError, Result};

pub(super) struct KmeansResult {
    pub(super) assignments: Vec<u8>,
    pub(super) iterations: usize,
    pub(super) inertia: f32,
}

pub(super) fn deterministic_spectral_clusters(
    graph: &AssocGraph,
    eigenmaps: &[EigenPair],
    cluster_count: usize,
    max_iter: usize,
) -> Result<KmeansResult> {
    let points = spectral_embedding(eigenmaps, cluster_count, graph.node_count())?;
    deterministic_kmeans(graph, &points, cluster_count, max_iter)
}

fn spectral_embedding(
    eigenmaps: &[EigenPair],
    dimensions: usize,
    node_count: usize,
) -> Result<Vec<Vec<f32>>> {
    if eigenmaps.len() < dimensions {
        return invalid_params(format!(
            "eigen_k produced {} eigenvectors but {} communities require at least {}",
            eigenmaps.len(),
            dimensions,
            dimensions
        ));
    }
    let mut rows = vec![Vec::with_capacity(dimensions); node_count];
    for pair in eigenmaps.iter().take(dimensions) {
        if pair.eigenvector.len() != node_count {
            return invalid_params("spectral eigenvector length must match graph node count");
        }
        for (row, value) in rows.iter_mut().zip(&pair.eigenvector) {
            if !value.is_finite() {
                return invalid_params("spectral embedding contains a non-finite coordinate");
            }
            row.push(*value);
        }
    }
    for row in &mut rows {
        let norm = squared_norm(row).sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return invalid_params("spectral embedding row has no finite positive L2 norm");
        }
        for value in row {
            *value /= norm;
        }
    }
    Ok(rows)
}

fn deterministic_kmeans(
    graph: &AssocGraph,
    points: &[Vec<f32>],
    cluster_count: usize,
    max_iter: usize,
) -> Result<KmeansResult> {
    if points.len() != graph.node_count() || points.is_empty() {
        return invalid_params("spectral point count must match the non-empty graph");
    }
    let dimensions = points[0].len();
    if dimensions == 0 || points.iter().any(|point| point.len() != dimensions) {
        return invalid_params("spectral embedding dimensions are inconsistent");
    }

    let seed_indices = farthest_first_seeds(graph, points, cluster_count)?;
    let mut centroids = seed_indices
        .iter()
        .map(|index| points[*index].clone())
        .collect::<Vec<_>>();
    let mut assignments = vec![u8::MAX; points.len()];

    for iteration in 1..=max_iter {
        let next = assign_points(points, &centroids);
        if next == assignments {
            let inertia = clustering_inertia(points, &assignments, &centroids);
            let canonical = canonicalize_assignments(graph, &assignments, cluster_count)?;
            return Ok(KmeansResult {
                inertia,
                assignments: canonical,
                iterations: iteration - 1,
            });
        }
        assignments = next;
        centroids = recompute_centroids(points, &assignments, cluster_count, dimensions)?;
    }
    invalid_params(format!(
        "deterministic spectral clustering did not converge after {max_iter} iterations"
    ))
}

fn farthest_first_seeds(
    graph: &AssocGraph,
    points: &[Vec<f32>],
    cluster_count: usize,
) -> Result<Vec<usize>> {
    let mut seeds = Vec::with_capacity(cluster_count);
    let first = (0..points.len())
        .max_by(|left, right| {
            squared_norm(&points[*left])
                .total_cmp(&squared_norm(&points[*right]))
                .then_with(|| node_id_cmp(graph, *right, *left))
        })
        .ok_or_else(|| LodestarError::KernelInvalidParams {
            detail: "spectral embedding contains no seed candidates".to_string(),
        })?;
    seeds.push(first);
    while seeds.len() < cluster_count {
        let next = (0..points.len())
            .filter(|index| !seeds.contains(index))
            .max_by(|left, right| {
                min_seed_distance(&points[*left], points, &seeds)
                    .total_cmp(&min_seed_distance(&points[*right], points, &seeds))
                    .then_with(|| node_id_cmp(graph, *right, *left))
            })
            .ok_or_else(|| LodestarError::KernelInvalidParams {
                detail: format!(
                    "community_count {cluster_count} exceeds distinct graph seed candidates"
                ),
            })?;
        seeds.push(next);
    }
    Ok(seeds)
}

fn node_id_cmp(graph: &AssocGraph, left: usize, right: usize) -> std::cmp::Ordering {
    graph
        .node_id(left)
        .expect("spectral node id")
        .as_bytes()
        .cmp(graph.node_id(right).expect("spectral node id").as_bytes())
}

fn min_seed_distance(point: &[f32], points: &[Vec<f32>], seeds: &[usize]) -> f32 {
    seeds
        .iter()
        .map(|index| squared_distance(point, &points[*index]))
        .fold(f32::INFINITY, f32::min)
}

fn assign_points(points: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<u8> {
    points
        .iter()
        .map(|point| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(left_index, left), (right_index, right)| {
                    squared_distance(point, left)
                        .total_cmp(&squared_distance(point, right))
                        .then_with(|| left_index.cmp(right_index))
                })
                .map(|(index, _)| index as u8)
                .expect("spectral clustering has at least two centroids")
        })
        .collect()
}

fn recompute_centroids(
    points: &[Vec<f32>],
    assignments: &[u8],
    cluster_count: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut sums = vec![vec![0.0_f32; dimensions]; cluster_count];
    let mut counts = vec![0_usize; cluster_count];
    for (point, assignment) in points.iter().zip(assignments) {
        let cluster = usize::from(*assignment);
        counts[cluster] += 1;
        for (sum, value) in sums[cluster].iter_mut().zip(point) {
            *sum += *value;
        }
    }
    if let Some(empty) = counts.iter().position(|count| *count == 0) {
        return invalid_params(format!(
            "deterministic spectral clustering produced empty community {empty}"
        ));
    }
    for (centroid, count) in sums.iter_mut().zip(counts) {
        for value in centroid {
            *value /= count as f32;
        }
    }
    Ok(sums)
}

fn canonicalize_assignments(
    graph: &AssocGraph,
    assignments: &[u8],
    cluster_count: usize,
) -> Result<Vec<u8>> {
    let mut minima = vec![None::<CxId>; cluster_count];
    for (index, assignment) in assignments.iter().enumerate() {
        let id = graph.node_id(index).ok_or_else(|| LodestarError::Graph {
            code: "CALYX_GRAPH_UNKNOWN_NODE",
            message: format!("graph node index {index} is missing"),
        })?;
        let slot = &mut minima[usize::from(*assignment)];
        if slot
            .as_ref()
            .is_none_or(|current| id.as_bytes() < current.as_bytes())
        {
            *slot = Some(id);
        }
    }
    if minima.iter().any(Option::is_none) {
        return invalid_params("spectral community canonicalization found an empty community");
    }
    let mut order = (0..cluster_count).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        minima[*left]
            .expect("community minimum")
            .as_bytes()
            .cmp(minima[*right].expect("community minimum").as_bytes())
    });
    let mut remap = vec![0_u8; cluster_count];
    for (canonical, original) in order.into_iter().enumerate() {
        remap[original] = canonical as u8;
    }
    Ok(assignments
        .iter()
        .map(|assignment| remap[usize::from(*assignment)])
        .collect())
}

fn clustering_inertia(points: &[Vec<f32>], assignments: &[u8], centroids: &[Vec<f32>]) -> f32 {
    points
        .iter()
        .zip(assignments)
        .map(|(point, assignment)| squared_distance(point, &centroids[usize::from(*assignment)]))
        .sum()
}

fn squared_norm(point: &[f32]) -> f32 {
    point.iter().map(|value| value * value).sum()
}

fn squared_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn invalid_params<T>(detail: impl Into<String>) -> Result<T> {
    Err(LodestarError::KernelInvalidParams {
        detail: detail.into(),
    })
}
