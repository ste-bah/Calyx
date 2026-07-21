use std::path::Path;

use calyx_core::Result;

use crate::error::{CALYX_INDEX_CORRUPT, CALYX_INDEX_INVALID_PARAMS, sextant_error};
use crate::index::SpannCentroidIndex;

use super::{
    AssignmentBuffer, AssignmentRegion, AssignmentSink, BoundedAssignmentConfig,
    assignment_ids_rel, choose_bounded_regions,
};
use crate::index::partitioned::gpu::PartitionGpu;
use crate::index::partitioned::{ClosureAssignmentStats, VectorSource};

const CAPACITY_PROBE_FACTOR: usize = 4;

pub(in crate::index::partitioned) fn stream_assign_to_ids_gpu(
    root: &Path,
    sink: AssignmentSink,
    centroids: &SpannCentroidIndex,
    source: &dyn VectorSource,
    gpu: &mut PartitionGpu,
) -> Result<Vec<AssignmentRegion>> {
    let region_count = centroids.centroid_count();
    let mut counts = vec![0_usize; region_count];
    let mut output = AssignmentBuffer::new(root, sink, region_count)?;
    gpu.route_all(centroids, source, 1, |start, take, ids, distances| {
        for row in 0..take {
            let region =
                validate_candidate(ids[row], distances[row], region_count, start + row as u64)?;
            counts[region] += 1;
            output.push(start + row as u64, region)?;
        }
        Ok(())
    })?;
    output.finish()?;
    Ok(counts
        .into_iter()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .map(|(region, count)| AssignmentRegion {
            id: region as u32,
            count,
            ids_rel: assignment_ids_rel(sink, region as u32),
        })
        .collect())
}

pub(in crate::index::partitioned) fn stream_assign_to_ids_bounded_gpu(
    root: &Path,
    sink: AssignmentSink,
    centroids: &SpannCentroidIndex,
    source: &dyn VectorSource,
    config: BoundedAssignmentConfig,
    primary_cap: usize,
    gpu: &mut PartitionGpu,
) -> Result<(Vec<AssignmentRegion>, ClosureAssignmentStats)> {
    validate_config(
        centroids.centroid_count(),
        source.len(),
        config,
        primary_cap,
    )?;
    let region_count = centroids.centroid_count();
    let requested_probe = config.routing_probe.min(region_count);
    let capacity_probe = requested_probe
        .saturating_mul(CAPACITY_PROBE_FACTOR)
        .min(region_count);
    let total_capacity = (config.cap as u128) * (region_count as u128);
    let mut primary_counts = vec![0_usize; region_count];
    let mut stored_counts = vec![0_usize; region_count];
    let mut duplicate_budget =
        usize::try_from(total_capacity - source.len() as u128).unwrap_or(usize::MAX);
    let mut stats = ClosureAssignmentStats::default();
    let mut output = AssignmentBuffer::new(root, sink, region_count)?;
    gpu.route_all(
        centroids,
        source,
        capacity_probe,
        |start, take, ids, distances| {
            let mut candidates = Vec::with_capacity(capacity_probe);
            let mut selected = Vec::with_capacity(config.max_replication);
            for row in 0..take {
                let offset = row * capacity_probe;
                candidates.clear();
                for (&id, &distance) in ids[offset..offset + requested_probe]
                    .iter()
                    .zip(&distances[offset..offset + requested_probe])
                {
                    let region =
                        validate_candidate(id, distance, region_count, start + row as u64)?;
                    candidates.push((region, distance));
                }
                if candidates.windows(2).any(|pair| {
                    pair[0]
                        .1
                        .total_cmp(&pair[1].1)
                        .then_with(|| pair[0].0.cmp(&pair[1].0))
                        .is_gt()
                }) {
                    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                }
                let row_id = start + row as u64;
                let boundary_anchor_distance = candidates.first().map(|(_, distance)| *distance);
                let mut assigned = choose_bounded_regions(
                    &primary_counts,
                    &stored_counts,
                    primary_cap,
                    config.cap,
                    &candidates,
                    boundary_anchor_distance,
                    config.boundary_epsilon,
                    config.max_replication,
                    duplicate_budget,
                    config.apply_rng_rule.then_some((centroids, config.rng_factor)),
                    &mut stats,
                    &mut selected,
                );
                if !assigned && capacity_probe > requested_probe {
                    for (&id, &distance) in ids[offset + requested_probe..offset + capacity_probe]
                        .iter()
                        .zip(&distances[offset + requested_probe..offset + capacity_probe])
                    {
                        let region =
                            validate_candidate(id, distance, region_count, row_id)?;
                        candidates.push((region, distance));
                    }
                    if candidates.windows(2).any(|pair| {
                        pair[0]
                            .1
                            .total_cmp(&pair[1].1)
                            .then_with(|| pair[0].0.cmp(&pair[1].0))
                            .is_gt()
                    }) {
                        candidates
                            .sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                    }
                    assigned = choose_bounded_regions(
                        &primary_counts,
                        &stored_counts,
                        primary_cap,
                        config.cap,
                        &candidates,
                        candidates.first().map(|(_, distance)| *distance),
                        config.boundary_epsilon,
                        config.max_replication,
                        duplicate_budget,
                        config
                            .apply_rng_rule
                            .then_some((centroids, config.rng_factor)),
                        &mut stats,
                        &mut selected,
                    );
                }
                if !assigned {
                    return Err(sextant_error(
                        CALYX_INDEX_INVALID_PARAMS,
                        format!(
                            "bounded CUDA assignment exhausted the top {capacity_probe} regions for row {row_id}; increase regions or cap"
                        ),
                    ));
                }
                for (position, &region) in selected.iter().enumerate() {
                    if position == 0 {
                        primary_counts[region] += 1;
                    } else {
                        duplicate_budget = duplicate_budget.saturating_sub(1);
                    }
                    stored_counts[region] += 1;
                    output.push(row_id, region)?;
                }
            }
            Ok(())
        },
    )?;
    output.finish()?;
    let regions = stored_counts
        .into_iter()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .map(|(region, count)| AssignmentRegion {
            id: region as u32,
            count,
            ids_rel: assignment_ids_rel(sink, region as u32),
        })
        .collect();
    Ok((regions, stats))
}

fn validate_config(
    region_count: usize,
    row_count: u64,
    config: BoundedAssignmentConfig,
    primary_cap: usize,
) -> Result<()> {
    if primary_cap == 0
        || config.cap == 0
        || config.routing_probe == 0
        || config.max_replication == 0
        || !config.boundary_epsilon.is_finite()
        || config.boundary_epsilon < 0.0
        || !config.rng_factor.is_finite()
        || config.rng_factor <= 0.0
    {
        return Err(sextant_error(
            CALYX_INDEX_INVALID_PARAMS,
            "bounded CUDA assignment received an invalid config",
        ));
    }
    let primary_capacity = (primary_cap as u128) * (region_count as u128);
    let storage_capacity = (config.cap as u128) * (region_count as u128);
    if primary_capacity < row_count as u128 || storage_capacity < row_count as u128 {
        return Err(sextant_error(
            CALYX_INDEX_INVALID_PARAMS,
            format!(
                "bounded CUDA assignment capacity primary={primary_capacity} storage={storage_capacity} < n_cx {row_count}"
            ),
        ));
    }
    Ok(())
}

fn validate_candidate(region: i64, distance: f32, count: usize, row: u64) -> Result<usize> {
    let raw_region = region;
    let region = usize::try_from(region).map_err(|_| {
        sextant_error(
            CALYX_INDEX_CORRUPT,
            format!("CUDA assignment returned invalid region {raw_region} for row {row}"),
        )
    })?;
    if region >= count || !distance.is_finite() {
        Err(sextant_error(
            CALYX_INDEX_CORRUPT,
            format!(
                "CUDA assignment returned region {region}/{count} distance {distance} for row {row}"
            ),
        ))
    } else {
        Ok(region)
    }
}
