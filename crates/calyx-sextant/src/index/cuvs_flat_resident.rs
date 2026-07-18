//! Bounded, reusable cuVS exact index for small persisted dense slots.

use calyx_core::Result;
use serde::Serialize;

use crate::error::{CALYX_INDEX_DIM_MISMATCH, CALYX_INDEX_INVALID_PARAMS, sextant_error};

pub const CUVS_RESIDENT_FLAT_MAX_BATCH: usize = 1024;
pub const CUVS_RESIDENT_FLAT_MAX_K: usize = 1024;

#[derive(Clone, Debug, Default, Serialize)]
pub struct CuvsResidentFlatDiagnostics {
    pub backend: &'static str,
    pub rows: usize,
    pub dim: usize,
    pub resident_bytes: u64,
    pub batches: u64,
    pub queries: u64,
    pub cuvs_kernel_launches: u64,
    pub exact_filtered_kernel_launches: u64,
    pub query_uploads: u64,
    pub filter_uploads: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub final_readback_pairs: u64,
    pub intermediate_readback_pairs: u64,
}

pub struct CuvsResidentFlatIndex {
    #[cfg(sextant_cuvs)]
    inner: cuda::ResidentFlat,
}

impl std::fmt::Debug for CuvsResidentFlatIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CuvsResidentFlatIndex")
            .field("diagnostics", &self.diagnostics())
            .finish()
    }
}

impl CuvsResidentFlatIndex {
    pub fn new(rows: usize, dim: usize, values: &[f32]) -> Result<Self> {
        validate_dataset(rows, dim, values)?;
        #[cfg(sextant_cuvs)]
        {
            Ok(Self {
                inner: cuda::ResidentFlat::new(rows, dim, values)?,
            })
        }
        #[cfg(not(sextant_cuvs))]
        {
            let _ = values;
            Err(unavailable())
        }
    }

    pub fn search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        k: usize,
        allowed_ids: Option<&[u32]>,
    ) -> Result<Vec<Vec<(u32, f32)>>> {
        if query_count == 0
            || query_count > CUVS_RESIDENT_FLAT_MAX_BATCH
            || k == 0
            || k > CUVS_RESIDENT_FLAT_MAX_K
            || queries.is_empty()
            || !queries.len().is_multiple_of(query_count)
        {
            return Err(sextant_error(
                CALYX_INDEX_INVALID_PARAMS,
                format!(
                    "resident flat search requires 0<queries<={CUVS_RESIDENT_FLAT_MAX_BATCH} and 0<k<={CUVS_RESIDENT_FLAT_MAX_K}"
                ),
            ));
        }
        if queries.iter().any(|value| !value.is_finite()) {
            return Err(sextant_error(
                CALYX_INDEX_INVALID_PARAMS,
                "resident flat query contains a non-finite value",
            ));
        }
        #[cfg(sextant_cuvs)]
        {
            self.inner.search(queries, query_count, k, allowed_ids)
        }
        #[cfg(not(sextant_cuvs))]
        {
            let _ = allowed_ids;
            Err(unavailable())
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        #[cfg(sextant_cuvs)]
        {
            self.inner.resident_bytes()
        }
        #[cfg(not(sextant_cuvs))]
        {
            0
        }
    }

    pub fn diagnostics(&self) -> CuvsResidentFlatDiagnostics {
        #[cfg(sextant_cuvs)]
        {
            self.inner.diagnostics()
        }
        #[cfg(not(sextant_cuvs))]
        {
            CuvsResidentFlatDiagnostics {
                backend: "unavailable",
                ..CuvsResidentFlatDiagnostics::default()
            }
        }
    }
}

fn validate_dataset(rows: usize, dim: usize, values: &[f32]) -> Result<()> {
    if rows == 0 || dim == 0 || values.len() != rows.saturating_mul(dim) {
        return Err(sextant_error(
            CALYX_INDEX_DIM_MISMATCH,
            "resident flat dataset shape mismatch",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(sextant_error(
            CALYX_INDEX_INVALID_PARAMS,
            "resident flat dataset contains a non-finite value",
        ));
    }
    Ok(())
}

#[cfg(not(sextant_cuvs))]
fn unavailable() -> calyx_core::CalyxError {
    sextant_error(
        crate::error::CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE,
        crate::cuvs_unavailable_reason("resident flat dense serving"),
    )
}

#[cfg(sextant_cuvs)]
#[path = "cuvs_flat_resident/cuda.rs"]
mod cuda;

#[cfg(all(test, sextant_cuvs))]
mod gpu_tests {
    use super::*;

    #[test]
    fn resident_flat_batches_and_filters_with_zero_vector_parity() {
        let values = [
            0.0_f32, 0.0, // zero row
            1.0, 0.0, // exact x
            1.0, 0.0, // deterministic tie
            0.0, 1.0, // y
        ];
        let mut index = CuvsResidentFlatIndex::new(4, 2, &values).expect("resident index");
        let batch = index
            .search(&[1.0, 0.0, 0.0, 0.0], 2, 3, None)
            .expect("batch search");
        assert_eq!(batch[0], vec![(1, 0.0), (2, 0.0), (0, 1.0)]);
        assert_eq!(batch[1], vec![(0, 1.0), (1, 1.0), (2, 1.0)]);

        let filtered = index
            .search(&[1.0, 0.0], 1, 2, Some(&[0, 3]))
            .expect("filtered search");
        assert_eq!(filtered[0], vec![(0, 1.0), (3, 1.0)]);
        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.intermediate_readback_pairs, 0);
        assert_eq!(diagnostics.final_readback_pairs, 8);
        assert!(diagnostics.exact_filtered_kernel_launches >= 2);
    }
}
