use std::collections::HashSet;

use calyx_sextant::index::{
    CuvsChunkedExactReport, CuvsChunkedExactRequest, CuvsDistanceMetric, DenseVectorFile,
    PartitionDistanceMetric, cuvs_chunked_bruteforce_topk, cuvs_chunked_bruteforce_topk_i8,
    cuvs_chunked_bruteforce_topk_synthetic,
};
use rayon::prelude::*;

use crate::error::{CliError, CliResult};

pub(super) const DEFAULT_CUDA_TRUTH_CHUNK_ROWS: usize = 100_000;

pub(super) struct ExactTruth {
    pub(super) ranked: Vec<Vec<(u64, f32)>>,
    pub(super) execution: CuvsChunkedExactReport,
}

impl ExactTruth {
    pub(super) fn sets(&self) -> Vec<HashSet<u64>> {
        self.ranked
            .iter()
            .map(|row| row.iter().map(|(id, _)| *id).collect())
            .collect()
    }
}

pub(super) fn exact_topk_vecfile(
    corpus: &DenseVectorFile,
    queries: &[Vec<f32>],
    k: usize,
    distance_metric: PartitionDistanceMetric,
) -> CliResult<ExactTruth> {
    exact_topk_vecfile_chunked(
        corpus,
        queries,
        k,
        distance_metric,
        DEFAULT_CUDA_TRUTH_CHUNK_ROWS,
    )
}

pub(super) fn exact_topk_vecfile_chunked(
    corpus: &DenseVectorFile,
    queries: &[Vec<f32>],
    k: usize,
    distance_metric: PartitionDistanceMetric,
    chunk_rows: usize,
) -> CliResult<ExactTruth> {
    let dim = corpus.dim();
    let flat_queries = flatten_queries(queries, dim)?;
    let result = match corpus {
        DenseVectorFile::Fbin(_) => cuvs_chunked_bruteforce_topk(
            CuvsChunkedExactRequest {
                corpus_rows: corpus.count(),
                dim,
                queries: &flat_queries,
                query_count: queries.len(),
                k,
                chunk_rows,
                metric: cuvs_metric(distance_metric),
            },
            |start, take, out| {
                out.par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(offset, destination)| {
                        copy_row_for_metric(
                            corpus,
                            start + offset as u64,
                            distance_metric,
                            destination,
                        );
                    });
                debug_assert_eq!(out.len(), take * dim);
                Ok(())
            },
        ),
        DenseVectorFile::I8Bin(file) => cuvs_chunked_bruteforce_topk_i8(
            CuvsChunkedExactRequest {
                corpus_rows: corpus.count(),
                dim,
                queries: &flat_queries,
                query_count: queries.len(),
                k,
                chunk_rows,
                metric: cuvs_metric(distance_metric),
            },
            |start, take, out| {
                out.copy_from_slice(file.rows_i8(start, take));
                Ok(())
            },
        ),
    }
    .map_err(CliError::Calyx)?;
    Ok(from_cuvs(result))
}

pub(super) fn exact_topk_synthetic(
    seed: u64,
    corpus_rows: u64,
    dim: usize,
    queries: &[Vec<f32>],
    k: usize,
    distance_metric: PartitionDistanceMetric,
) -> CliResult<ExactTruth> {
    let flat_queries = flatten_queries(queries, dim)?;
    let result = cuvs_chunked_bruteforce_topk_synthetic(
        seed,
        CuvsChunkedExactRequest {
            corpus_rows,
            dim,
            queries: &flat_queries,
            query_count: queries.len(),
            k,
            chunk_rows: DEFAULT_CUDA_TRUTH_CHUNK_ROWS,
            metric: cuvs_metric(distance_metric),
        },
    )
    .map_err(CliError::Calyx)?;
    Ok(from_cuvs(result))
}

fn from_cuvs(result: calyx_sextant::index::CuvsChunkedExactTopK) -> ExactTruth {
    let ranked = (0..result.query_count)
        .map(|query| {
            let (ids, distances) = result.row(query);
            ids.iter().copied().zip(distances.iter().copied()).collect()
        })
        .collect();
    ExactTruth {
        ranked,
        execution: result.report,
    }
}

fn flatten_queries(queries: &[Vec<f32>], dim: usize) -> CliResult<Vec<f32>> {
    if queries.is_empty() || queries.iter().any(|query| query.len() != dim) {
        return Err(CliError::usage(
            "exact ground-truth queries must be non-empty and match corpus dimension",
        ));
    }
    let mut flat = Vec::with_capacity(queries.len() * dim);
    for query in queries {
        flat.extend_from_slice(query);
    }
    Ok(flat)
}

fn cuvs_metric(metric: PartitionDistanceMetric) -> CuvsDistanceMetric {
    match metric {
        PartitionDistanceMetric::UnitL2 => CuvsDistanceMetric::Cosine,
        PartitionDistanceMetric::RawL2 => CuvsDistanceMetric::SquaredL2,
    }
}

fn copy_row_for_metric(
    corpus: &DenseVectorFile,
    idx: u64,
    metric: PartitionDistanceMetric,
    destination: &mut [f32],
) {
    match metric {
        PartitionDistanceMetric::UnitL2 => corpus.copy_row_f32(idx, destination),
        PartitionDistanceMetric::RawL2 => corpus.copy_row_f32_raw(idx, destination),
    }
}

#[cfg(test)]
mod tests {
    fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
        calyx_sextant::index::cosine_distance(left, right)
    }

    #[test]
    fn cosine_reference_handles_unnormalized_vectors() {
        let query = [10.0, 0.0];
        let same_direction = [100.0, 0.0];
        let l2_closer_but_worse_angle = [9.0, 1.0];

        assert!(
            cosine_distance(&query, &same_direction)
                < cosine_distance(&query, &l2_closer_but_worse_angle)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "GPU 100M-row CUDA scale gate"]
    fn synthetic_cuda_exact_runs_strict_100m_with_bounded_staging() {
        const CORPUS_ROWS: u64 = 100_000_000;
        const DIM: usize = 512;
        const QUERY_ROW: u64 = 7_919;
        const K: usize = 10;
        let query = calyx_sextant::index::gen_row(42, QUERY_ROW, DIM);

        let exact = super::exact_topk_synthetic(
            42,
            CORPUS_ROWS,
            DIM,
            &[query],
            K,
            calyx_sextant::index::PartitionDistanceMetric::UnitL2,
        )
        .expect("100M synthetic CUDA exact truth");

        assert!(exact.ranked[0].iter().any(|(row, _)| *row == QUERY_ROW));
        assert_eq!(exact.execution.corpus_rows, CORPUS_ROWS);
        assert_eq!(exact.execution.query_uploads, 1);
        assert_eq!(exact.execution.d2h_transfers, 2);
        assert_eq!(exact.execution.intermediate_readback_pairs, 0);
        assert!(!exact.execution.host_merge);
        assert_eq!(
            exact.execution.corpus_staging,
            calyx_sextant::index::CuvsCorpusStaging::SyntheticDeviceGenerate
        );
        assert_eq!(exact.execution.corpus_uploads, 0);
        assert_eq!(
            exact.execution.device_generated_values,
            CORPUS_ROWS * DIM as u64
        );
        assert!(exact.execution.peak_device_staging_bytes <= 256 * 1024 * 1024);
        println!(
            "{}",
            serde_json::to_string_pretty(&exact.execution).expect("serialize execution report")
        );
    }
}
