use std::path::Path;

use calyx_core::{CxId, Result, SlotId};

use crate::index::{DiskAnnBuildBackend, DiskAnnBuildParams, DiskAnnSearch, DiskAnnSearchParams};

use super::PartitionDistanceMetric;

pub(super) fn effective_region_build_parallelism(
    requested: usize,
    region_count: usize,
) -> Result<usize> {
    if requested == 0 {
        return Err(crate::error::sextant_error(
            crate::error::CALYX_INDEX_INVALID_PARAMS,
            "region_build_parallelism must be > 0",
        ));
    }
    Ok(requested.min(region_count.max(1)).max(1))
}

pub(super) fn build_partitioned_graph(
    graph_path: &Path,
    rows: &[(CxId, Vec<f32>)],
    build_params: DiskAnnBuildParams,
    search_params: DiskAnnSearchParams,
    backend: DiskAnnBuildBackend,
    distance_metric: PartitionDistanceMetric,
) -> Result<()> {
    match distance_metric {
        PartitionDistanceMetric::UnitL2 => {
            DiskAnnSearch::build_without_default_raw_sidecar_with_backend(
                SlotId::new(0),
                graph_path,
                rows,
                build_params,
                None,
                search_params,
                backend,
            )?;
        }
        PartitionDistanceMetric::RawL2 => {
            DiskAnnSearch::build_raw_l2_without_default_raw_sidecar_with_backend(
                SlotId::new(0),
                graph_path,
                rows,
                build_params,
                None,
                search_params,
                backend,
            )?;
        }
    }
    Ok(())
}
