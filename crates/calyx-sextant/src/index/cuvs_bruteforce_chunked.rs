//! Bounded exact kNN that keeps queries and global top-k resident on CUDA.

use calyx_core::Result;
use serde::Serialize;

use crate::error::{CALYX_INDEX_INVALID_PARAMS, sextant_error};

pub const CUVS_CHUNKED_EXACT_MAX_K: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CuvsDistanceMetric {
    Cosine,
    SquaredL2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CuvsCorpusStaging {
    F32Host,
    I8DeviceConvert,
    SyntheticDeviceGenerate,
}

#[derive(Clone, Debug, Serialize)]
pub struct CuvsChunkedExactReport {
    pub backend: &'static str,
    pub metric: CuvsDistanceMetric,
    pub corpus_rows: u64,
    pub query_count: usize,
    pub dim: usize,
    pub k: usize,
    pub chunk_rows: usize,
    pub chunks: usize,
    pub distance_kernel_launches: usize,
    pub cuvs_searches: usize,
    pub cosine_zero_corpus_chunks: usize,
    pub zero_query_count: usize,
    pub zero_query_repair_launches: usize,
    pub boundary_tie_guard_launches: usize,
    pub merge_kernel_launches: usize,
    pub query_uploads: usize,
    pub corpus_uploads: usize,
    pub h2d_transfers: usize,
    pub d2h_transfers: usize,
    pub intermediate_readback_pairs: usize,
    pub final_readback_pairs: usize,
    pub host_merge: bool,
    pub pinned_staging: bool,
    pub query_bytes_uploaded: usize,
    pub corpus_bytes_uploaded: u64,
    pub final_bytes_read: usize,
    pub peak_device_staging_bytes: usize,
    pub peak_pinned_staging_bytes: usize,
    pub corpus_staging: CuvsCorpusStaging,
    pub staging_kernel_launches: usize,
    pub device_generated_values: u64,
    pub elapsed_us: u128,
}

#[derive(Clone, Debug)]
pub struct CuvsChunkedExactTopK {
    pub query_count: usize,
    pub k: usize,
    pub neighbors: Vec<u64>,
    pub distances: Vec<f32>,
    pub report: CuvsChunkedExactReport,
}

impl CuvsChunkedExactTopK {
    pub fn row(&self, query_idx: usize) -> (&[u64], &[f32]) {
        let start = query_idx * self.k;
        let end = start + self.k;
        (&self.neighbors[start..end], &self.distances[start..end])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CuvsChunkedExactRequest<'a> {
    pub corpus_rows: u64,
    pub dim: usize,
    pub queries: &'a [f32],
    pub query_count: usize,
    pub k: usize,
    pub chunk_rows: usize,
    pub metric: CuvsDistanceMetric,
}

pub fn cuvs_chunked_bruteforce_topk<F>(
    request: CuvsChunkedExactRequest<'_>,
    mut load_chunk: F,
) -> Result<CuvsChunkedExactTopK>
where
    F: FnMut(u64, usize, &mut [f32]) -> Result<()>,
{
    validate(request)?;
    #[cfg(sextant_cuvs)]
    {
        cuda::run(
            request.corpus_rows,
            request.dim,
            request.queries,
            request.query_count,
            request.k,
            request.chunk_rows,
            request.metric,
            cuda::ChunkSource::F32(&mut load_chunk),
        )
    }
    #[cfg(not(sextant_cuvs))]
    {
        let _ = (request.metric, &mut load_chunk);
        Err(crate::error::sextant_error(
            crate::error::CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE,
            crate::cuvs_unavailable_reason("chunked exact ground-truth generation"),
        ))
    }
}

pub fn cuvs_chunked_bruteforce_topk_i8<F>(
    request: CuvsChunkedExactRequest<'_>,
    mut load_chunk: F,
) -> Result<CuvsChunkedExactTopK>
where
    F: FnMut(u64, usize, &mut [i8]) -> Result<()>,
{
    validate(request)?;
    #[cfg(sextant_cuvs)]
    {
        cuda::run(
            request.corpus_rows,
            request.dim,
            request.queries,
            request.query_count,
            request.k,
            request.chunk_rows,
            request.metric,
            cuda::ChunkSource::I8(&mut load_chunk),
        )
    }
    #[cfg(not(sextant_cuvs))]
    {
        let _ = (request.metric, &mut load_chunk);
        unavailable()
    }
}

pub fn cuvs_chunked_bruteforce_topk_synthetic(
    seed: u64,
    request: CuvsChunkedExactRequest<'_>,
) -> Result<CuvsChunkedExactTopK> {
    validate(request)?;
    #[cfg(sextant_cuvs)]
    {
        cuda::run(
            request.corpus_rows,
            request.dim,
            request.queries,
            request.query_count,
            request.k,
            request.chunk_rows,
            request.metric,
            cuda::ChunkSource::Synthetic(seed),
        )
    }
    #[cfg(not(sextant_cuvs))]
    {
        let _ = (seed, request.metric);
        unavailable()
    }
}

fn validate(request: CuvsChunkedExactRequest<'_>) -> Result<()> {
    let expected_queries = request
        .query_count
        .checked_mul(request.dim)
        .ok_or_else(invalid_shape)?;
    if request.corpus_rows == 0
        || request.corpus_rows > i64::MAX as u64
        || request.dim == 0
        || request.query_count == 0
        || request.k == 0
        || request.k as u64 > request.corpus_rows
        || request.k > CUVS_CHUNKED_EXACT_MAX_K
        || request.chunk_rows == 0
        || request.queries.len() != expected_queries
    {
        return Err(invalid_shape());
    }
    if request.queries.iter().any(|value| !value.is_finite()) {
        return Err(sextant_error(
            CALYX_INDEX_INVALID_PARAMS,
            "chunked cuVS exact queries contain non-finite values",
        ));
    }
    Ok(())
}

fn invalid_shape() -> calyx_core::CalyxError {
    sextant_error(
        CALYX_INDEX_INVALID_PARAMS,
        format!(
            "invalid chunked cuVS exact shape; require rows>0, dim>0, queries>0, 0<k<=rows and k<={CUVS_CHUNKED_EXACT_MAX_K}, chunk_rows>0"
        ),
    )
}

#[cfg(not(sextant_cuvs))]
fn unavailable() -> Result<CuvsChunkedExactTopK> {
    Err(crate::error::sextant_error(
        crate::error::CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE,
        crate::cuvs_unavailable_reason("chunked exact ground-truth generation"),
    ))
}

#[cfg(sextant_cuvs)]
#[path = "cuvs_bruteforce_chunked/cuda.rs"]
mod cuda;

#[cfg(test)]
#[path = "cuvs_bruteforce_chunked/tests.rs"]
mod tests;
