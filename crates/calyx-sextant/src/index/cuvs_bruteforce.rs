//! Compatibility wrapper over the resident chunked cuVS exact-kNN path.

use calyx_core::Result;

use super::cuvs_bruteforce_chunked::{
    CuvsChunkedExactRequest, CuvsDistanceMetric, cuvs_chunked_bruteforce_topk,
};
use crate::error::{CALYX_INDEX_INVALID_PARAMS, sextant_error};

#[derive(Clone, Debug)]
pub struct CuvsBruteForceTopK {
    pub query_count: usize,
    pub k: usize,
    pub neighbors: Vec<i64>,
    pub distances: Vec<f32>,
}

impl CuvsBruteForceTopK {
    pub fn row(&self, query_idx: usize) -> (&[i64], &[f32]) {
        let start = query_idx * self.k;
        let end = start + self.k;
        (&self.neighbors[start..end], &self.distances[start..end])
    }
}

pub fn cuvs_bruteforce_topk(
    dataset: &mut [f32],
    rows: usize,
    dim: usize,
    queries: &mut [f32],
    query_count: usize,
    k: usize,
) -> Result<CuvsBruteForceTopK> {
    let expected_dataset = rows.checked_mul(dim).ok_or_else(invalid_dataset_shape)?;
    if dataset.len() != expected_dataset {
        return Err(invalid_dataset_shape());
    }
    let result = cuvs_chunked_bruteforce_topk(
        CuvsChunkedExactRequest {
            corpus_rows: rows as u64,
            dim,
            queries,
            query_count,
            k,
            chunk_rows: rows,
            metric: CuvsDistanceMetric::SquaredL2,
        },
        |start, take, out| {
            let start = start as usize * dim;
            let end = start + take * dim;
            out.copy_from_slice(&dataset[start..end]);
            Ok(())
        },
    )?;
    Ok(CuvsBruteForceTopK {
        query_count,
        k,
        neighbors: result.neighbors.into_iter().map(|id| id as i64).collect(),
        distances: result.distances,
    })
}

fn invalid_dataset_shape() -> calyx_core::CalyxError {
    sextant_error(
        CALYX_INDEX_INVALID_PARAMS,
        "cuVS brute-force dataset buffer does not match rows*dim",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_dataset_shape_returns_error_instead_of_panicking() {
        let error = cuvs_bruteforce_topk(&mut [0.0], 2, 2, &mut [0.0, 0.0], 1, 1)
            .expect_err("short dataset must fail before chunk loading");

        assert_eq!(error.code, CALYX_INDEX_INVALID_PARAMS);
    }
}
