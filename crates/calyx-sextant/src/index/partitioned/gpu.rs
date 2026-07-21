#[cfg(sextant_cuvs)]
mod cuda;
#[cfg(sextant_cuvs)]
mod cuvs;

use calyx_core::Result;

use crate::index::DiskAnnBuildBackend;
#[cfg(not(sextant_cuvs))]
use crate::index::SpannCentroidIndex;

#[cfg(not(sextant_cuvs))]
use super::PartitionBuildDiagnostics;
use super::VectorSource;

#[cfg(sextant_cuvs)]
pub(super) use cuda::PartitionGpu;

pub(super) fn for_backend(
    backend: DiskAnnBuildBackend,
    source: &dyn VectorSource,
    chunk_rows: usize,
    sample_rows: usize,
    initial_centroids: usize,
) -> Result<Option<PartitionGpu>> {
    match backend {
        DiskAnnBuildBackend::CpuVamana => Ok(None),
        DiskAnnBuildBackend::CuvsCagra => {
            PartitionGpu::new(source, chunk_rows, sample_rows, initial_centroids).map(Some)
        }
    }
}

/// GPU k-means is most efficient as one large fit. When the caller supplies a
/// hard balance cap, target a mean bucket size of at most five-eighths of that
/// cap. The extra headroom keeps skewed one-pass clusters from saturating and
/// displacing primary assignments. Bounded final assignment absorbs residual
/// skew without multiplying regions;
/// recursive GPU balancing remains the sample-limited backstop.
pub(super) fn initial_cluster_count(
    backend: DiskAnnBuildBackend,
    row_count: u64,
    requested: usize,
    balance_cap: Option<usize>,
    sample_rows: usize,
) -> usize {
    let target = match (backend, balance_cap) {
        (DiskAnnBuildBackend::CuvsCagra, Some(cap)) if cap > 0 => {
            let minimum = (row_count as u128).div_ceil(cap as u128);
            let with_headroom = minimum.saturating_mul(8).div_ceil(5);
            usize::try_from(with_headroom).unwrap_or(usize::MAX)
        }
        _ => requested,
    };
    requested.max(target).min(sample_rows.max(1))
}

pub(super) fn has_balance_headroom(regions: usize, rows: u64, cap: usize) -> bool {
    (regions as u128) * (cap as u128) >= rows as u128
}

/// Shrink closure radius when a one-pass GPU build has too little sample
/// support to estimate each high-dimensional centroid reliably. Exact 100M
/// balanced-primary assignment replays calibrated the closure radius at 96%
/// of one sample per dimension, leaving headroom below the frozen replication
/// ceiling. The requested epsilon remains an upper bound once that calibrated
/// support is reached.
pub(super) fn supported_boundary_epsilon(
    requested: f32,
    sample_rows: usize,
    centroids: usize,
    dim: usize,
) -> f32 {
    if requested == 0.0 || sample_rows == 0 || centroids == 0 || dim == 0 {
        return requested;
    }
    let support = sample_rows as f64 / centroids as f64;
    const SUPPORTED_DIMENSION_FRACTION: f64 = 0.96;
    let supported_dimensions = dim as f64 * SUPPORTED_DIMENSION_FRACTION;
    let support_ratio = (support / supported_dimensions).min(1.0);
    requested * support_ratio.sqrt() as f32
}

#[cfg(not(sextant_cuvs))]
pub(super) struct PartitionGpu;

#[cfg(not(sextant_cuvs))]
impl PartitionGpu {
    fn new(
        _source: &dyn VectorSource,
        _chunk_rows: usize,
        _sample_rows: usize,
        _initial_centroids: usize,
    ) -> Result<Self> {
        Err(crate::error::sextant_error(
            crate::error::CALYX_INDEX_IO,
            format!(
                "strict CUDA partition execution was required by cuvs-cagra; refusing CPU centroid training and whole-corpus scans on {}",
                std::env::consts::OS
            ),
        ))
    }

    pub(super) fn fit_centroids(
        &mut self,
        _rows: &[(u32, Vec<f32>)],
        _clusters: usize,
        _seed: u64,
    ) -> Result<SpannCentroidIndex> {
        unreachable!("PartitionGpu::new fails without sextant_cuvs")
    }

    pub(super) fn route_all<F>(
        &mut self,
        _centroids: &SpannCentroidIndex,
        _source: &dyn VectorSource,
        _probe: usize,
        _sink: F,
    ) -> Result<()>
    where
        F: FnMut(u64, usize, &[i64], &[f32]) -> Result<()>,
    {
        unreachable!("PartitionGpu::new fails without sextant_cuvs")
    }

    pub(super) fn route_members(
        &mut self,
        _centroids: &SpannCentroidIndex,
        _source: &dyn VectorSource,
        _members: &[u64],
    ) -> Result<Vec<u32>> {
        unreachable!("PartitionGpu::new fails without sextant_cuvs")
    }

    pub(super) fn diagnostics_mut(&mut self) -> &mut PartitionBuildDiagnostics {
        unreachable!("PartitionGpu::new fails without sextant_cuvs")
    }
}

#[cfg(test)]
mod tests {
    use super::supported_boundary_epsilon;

    #[test]
    fn closure_radius_only_shrinks_for_sample_starved_centroids() {
        assert_eq!(supported_boundary_epsilon(0.3, 200_000, 1_024, 100), 0.3);
        assert_eq!(supported_boundary_epsilon(0.3, 200_000, 1_832, 100), 0.3);
        let large = supported_boundary_epsilon(0.3, 200_000, 18_312, 100);
        assert!((large - 0.101_188_87).abs() < 0.000_001, "{large}");
        let headroom = supported_boundary_epsilon(0.3, 200_000, 19_533, 100);
        assert!((headroom - 0.097_975_2).abs() < 0.000_001, "{headroom}");
        assert_eq!(supported_boundary_epsilon(0.0, 1, 10_000, 100), 0.0);
    }
}
