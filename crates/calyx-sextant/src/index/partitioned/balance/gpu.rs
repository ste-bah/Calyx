use std::path::Path;

use calyx_core::Result;

use crate::error::{CALYX_INDEX_CORRUPT, CALYX_INDEX_INVALID_PARAMS, sextant_error};
use crate::index::SpannCentroidIndex;
use crate::index::partitioned::gpu::PartitionGpu;

use super::super::assignment::{AssignmentRegion, read_ids};
use super::super::{IDX_MIX, PartitionDistanceMetric, VectorSource};
use super::sample_rows;

const MIN_GPU_RECLUSTER_DEPTH: usize = 8;

#[allow(clippy::too_many_arguments)]
pub(in crate::index::partitioned) fn balance_region_files_gpu(
    root: &Path,
    initial: &SpannCentroidIndex,
    regions: &[AssignmentRegion],
    source: &dyn VectorSource,
    seed: u64,
    cap: usize,
    _distance_metric: PartitionDistanceMetric,
    gpu: &mut PartitionGpu,
) -> Result<Vec<Vec<f32>>> {
    let mut balanced = Vec::new();
    let mut oversized = Vec::new();
    for region in regions {
        let members = read_ids(&root.join(&region.ids_rel))?;
        if members.len() != region.count {
            return Err(sextant_error(
                CALYX_INDEX_CORRUPT,
                format!(
                    "provisional region {} ids count {} != assignment count {}",
                    region.id,
                    members.len(),
                    region.count
                ),
            ));
        }
        if members.is_empty() {
            continue;
        }
        if members.len() <= cap {
            let centroid = initial.centroids().get(region.id as usize).ok_or_else(|| {
                sextant_error(
                    CALYX_INDEX_CORRUPT,
                    format!("missing initial centroid {}", region.id),
                )
            })?;
            balanced.push(centroid.clone());
            continue;
        }
        oversized.extend(members);
    }
    if !oversized.is_empty() {
        let depth_limit = gpu_recluster_depth_limit(oversized.len(), cap);
        balanced.extend(split_oversized_gpu(
            &oversized,
            source,
            seed,
            cap,
            0,
            depth_limit,
            gpu,
        )?);
    }
    Ok(balanced)
}

fn split_oversized_gpu(
    members: &[u64],
    source: &dyn VectorSource,
    seed: u64,
    cap: usize,
    depth: usize,
    depth_limit: usize,
    gpu: &mut PartitionGpu,
) -> Result<Vec<Vec<f32>>> {
    if members.len() <= cap.saturating_mul(4) {
        return chunk_centroids_by_cap_gpu(members, source, seed, cap, gpu);
    }
    if depth >= depth_limit {
        return Err(strict_balance_error(format!(
            "{} rows remained above cap {cap} at geometry-derived depth limit {depth_limit}",
            members.len(),
        )));
    }
    let sample = sample_rows(members, source);
    let minimum = members.len().div_ceil(cap).max(2);
    let sub_count = minimum
        .saturating_mul(3)
        .div_ceil(2)
        .min(sample.len().max(1));
    let sub = gpu.fit_centroids(
        &sample,
        sub_count,
        seed ^ (depth as u64).wrapping_mul(IDX_MIX),
    )?;
    let assignments = gpu.route_members(&sub, source, members)?;
    if assignments.len() != members.len() {
        return Err(sextant_error(
            CALYX_INDEX_CORRUPT,
            "CUDA balance assignment count changed",
        ));
    }
    let mut buckets: Vec<Vec<u64>> = vec![Vec::new(); sub.centroid_count()];
    for (&member, &region) in members.iter().zip(&assignments) {
        let bucket = buckets.get_mut(region as usize).ok_or_else(|| {
            sextant_error(
                CALYX_INDEX_CORRUPT,
                format!("CUDA balance returned out-of-range region {region}"),
            )
        })?;
        bucket.push(member);
    }
    let largest = buckets.iter().map(Vec::len).max().unwrap_or(0);
    if largest >= members.len() {
        return Err(strict_balance_error(format!(
            "cuVS split made no progress for {} rows and {sub_count} subclusters",
            members.len()
        )));
    }
    let mut output = Vec::new();
    let mut next_round = Vec::new();
    for (sub_index, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        if bucket.len() <= cap {
            output.push(sub.centroids()[sub_index].clone());
        } else {
            next_round.extend(bucket);
        }
    }
    if !next_round.is_empty() {
        output.extend(split_oversized_gpu(
            &next_round,
            source,
            seed,
            cap,
            depth + 1,
            depth_limit,
            gpu,
        )?);
    }
    Ok(output)
}

fn chunk_centroids_by_cap_gpu(
    members: &[u64],
    source: &dyn VectorSource,
    seed: u64,
    cap: usize,
    gpu: &mut PartitionGpu,
) -> Result<Vec<Vec<f32>>> {
    let mut centroids = Vec::with_capacity(members.len().div_ceil(cap.max(1)));
    for (chunk_index, chunk) in members.chunks(cap.max(1)).enumerate() {
        let sample = sample_rows(chunk, source);
        let fitted = gpu.fit_centroids(
            &sample,
            1,
            seed ^ (chunk_index as u64).wrapping_mul(IDX_MIX),
        )?;
        let centroid = fitted.centroids().first().ok_or_else(|| {
            sextant_error(
                CALYX_INDEX_CORRUPT,
                "CUDA terminal balance produced no centroid",
            )
        })?;
        centroids.push(centroid.clone());
    }
    Ok(centroids)
}

fn gpu_recluster_depth_limit(rows: usize, cap: usize) -> usize {
    let ratio = rows.div_ceil(cap.max(1)).max(1);
    let binary_depth = usize::BITS as usize - (ratio - 1).leading_zeros() as usize;
    MIN_GPU_RECLUSTER_DEPTH.saturating_add(binary_depth.saturating_mul(2))
}

fn strict_balance_error(detail: impl std::fmt::Display) -> calyx_core::CalyxError {
    sextant_error(
        CALYX_INDEX_INVALID_PARAMS,
        format!("strict CUDA balance refused CPU fallback: {detail}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_depth_budget_scales_with_balance_geometry() {
        assert_eq!(gpu_recluster_depth_limit(8_192, 8_192), 8);
        assert_eq!(gpu_recluster_depth_limit(62_388_775, 8_192), 34);
        assert!(gpu_recluster_depth_limit(100_000_000, 8_192) > 8);
    }
}
