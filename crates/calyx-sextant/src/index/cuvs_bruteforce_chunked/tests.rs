use std::cell::Cell;

use super::*;
use crate::error::CALYX_INDEX_INVALID_PARAMS;
#[cfg(not(sextant_cuvs))]
use crate::error::CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE;

fn request(
    corpus_rows: u64,
    dim: usize,
    queries: &[f32],
    query_count: usize,
    k: usize,
    chunk_rows: usize,
    metric: CuvsDistanceMetric,
) -> CuvsChunkedExactRequest<'_> {
    CuvsChunkedExactRequest {
        corpus_rows,
        dim,
        queries,
        query_count,
        k,
        chunk_rows,
        metric,
    }
}

#[test]
fn invalid_query_shape_is_rejected_before_loading_corpus() {
    let loaded = Cell::new(false);
    let error = cuvs_chunked_bruteforce_topk(
        request(2, 3, &[1.0, 2.0], 1, 1, 1, CuvsDistanceMetric::SquaredL2),
        |_, _, _| {
            loaded.set(true);
            Ok(())
        },
    )
    .expect_err("mismatched query shape must fail");

    assert_eq!(error.code, CALYX_INDEX_INVALID_PARAMS);
    assert!(!loaded.get());
}

#[test]
fn non_finite_queries_are_rejected_before_loading_corpus() {
    let loaded = Cell::new(false);
    let error = cuvs_chunked_bruteforce_topk(
        request(2, 2, &[f32::NAN, 0.0], 1, 1, 1, CuvsDistanceMetric::Cosine),
        |_, _, _| {
            loaded.set(true);
            Ok(())
        },
    )
    .expect_err("non-finite query must fail");

    assert_eq!(error.code, CALYX_INDEX_INVALID_PARAMS);
    assert!(!loaded.get());
}

#[test]
fn oversized_k_is_rejected() {
    let error = cuvs_chunked_bruteforce_topk(
        request(
            CUVS_CHUNKED_EXACT_MAX_K as u64 + 1,
            1,
            &[0.0],
            1,
            CUVS_CHUNKED_EXACT_MAX_K + 1,
            16,
            CuvsDistanceMetric::SquaredL2,
        ),
        |_, _, _| Ok(()),
    )
    .expect_err("k above the deterministic merge bound must fail");

    assert_eq!(error.code, CALYX_INDEX_INVALID_PARAMS);
}

#[cfg(not(sextant_cuvs))]
#[test]
fn unsupported_build_fails_closed_without_loading_corpus() {
    let loaded = Cell::new(false);
    let error = cuvs_chunked_bruteforce_topk(
        request(2, 2, &[1.0, 0.0], 1, 1, 1, CuvsDistanceMetric::Cosine),
        |_, _, _| {
            loaded.set(true);
            Ok(())
        },
    )
    .expect_err("unsupported builds must not fall back to CPU");

    assert_eq!(error.code, CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE);
    assert!(!loaded.get());
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_matches_squared_l2_reference_across_chunks_and_ties() {
    let corpus = vec![
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        -1.0, 0.0, 0.0, // 2
        1.0, 0.0, 0.0, // 3 duplicate
        0.0, 1.0, 0.0, // 4
        0.0, -1.0, 0.0, // 5
        2.0, 0.0, 0.0, // 6
    ];
    let queries = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    assert_cuda_matches_reference(&corpus, &queries, 2, 3, 7, 3, CuvsDistanceMetric::SquaredL2);
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_i8_device_staging_matches_both_cpu_metrics() {
    let corpus_i8 = vec![
        0, 0, 0, // 0
        1, 0, 0, // 1
        -1, 0, 0, // 2
        1, 0, 0, // 3 duplicate
        0, 2, 0, // 4
        0, -2, 0, // 5
        3, 1, 0, // 6
    ];
    let corpus = corpus_i8
        .iter()
        .map(|value| f32::from(*value))
        .collect::<Vec<_>>();
    let queries = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    for metric in [CuvsDistanceMetric::SquaredL2, CuvsDistanceMetric::Cosine] {
        let result = cuvs_chunked_bruteforce_topk_i8(
            request(7, 3, &queries, 2, 4, 4, metric),
            |start, take, out| {
                let start = start as usize * 3;
                out.copy_from_slice(&corpus_i8[start..start + take * 3]);
                Ok(())
            },
        )
        .expect("i8 CUDA exact search");
        let expected = cpu_reference(&corpus, &queries, 2, 3, 4, metric);

        for (query, expected) in expected.iter().enumerate() {
            assert_eq!(result.row(query).0, expected, "query {query} {metric:?}");
        }
        assert_eq!(
            result.report.corpus_staging,
            CuvsCorpusStaging::I8DeviceConvert
        );
        assert_eq!(result.report.staging_kernel_launches, 2);
        assert_eq!(result.report.corpus_bytes_uploaded, 24);
    }
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_device_generated_synthetic_rows_match_cpu_reference() {
    const SEED: u64 = 609;
    const ROWS: usize = 73;
    const DIM: usize = 17;
    let query_rows = [0, 31, 72];
    let queries = query_rows
        .iter()
        .flat_map(|row| crate::index::gen_row(SEED, *row, DIM))
        .collect::<Vec<_>>();
    let corpus = (0..ROWS)
        .flat_map(|row| crate::index::gen_row(SEED, row as u64, DIM))
        .collect::<Vec<_>>();

    for metric in [CuvsDistanceMetric::Cosine, CuvsDistanceMetric::SquaredL2] {
        let result = cuvs_chunked_bruteforce_topk_synthetic(
            SEED,
            request(ROWS as u64, DIM, &queries, query_rows.len(), 7, 19, metric),
        )
        .expect("device-generated synthetic CUDA exact search");
        let expected = cpu_reference(&corpus, &queries, query_rows.len(), DIM, 7, metric);

        for (query, expected) in expected.iter().enumerate() {
            assert_eq!(
                result.row(query).0,
                expected,
                "synthetic {metric:?} query {query}"
            );
        }
        assert_eq!(result.report.metric, metric);
        assert_eq!(
            result.report.corpus_staging,
            CuvsCorpusStaging::SyntheticDeviceGenerate
        );
        assert_eq!(result.report.corpus_uploads, 0);
        assert_eq!(result.report.h2d_transfers, 1);
        assert_eq!(result.report.staging_kernel_launches, 4);
        assert_eq!(result.report.device_generated_values, (ROWS * DIM) as u64);
    }
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_keeps_lowest_ids_when_ties_cross_chunk_topk_boundary() {
    let corpus = [1.0, 0.0].repeat(8);
    let queries = vec![1.0, 0.0];

    for metric in [CuvsDistanceMetric::SquaredL2, CuvsDistanceMetric::Cosine] {
        let result = assert_cuda_matches_reference(&corpus, &queries, 1, 2, 3, 5, metric);
        assert_eq!(result.report.boundary_tie_guard_launches, 1);
    }
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_repairs_boundary_ties_across_query_batches() {
    let corpus = [1.0, 0.0].repeat(8);
    let queries = [1.0, 0.0].repeat(17);

    let result = assert_cuda_matches_reference(
        &corpus,
        &queries,
        17,
        2,
        3,
        5,
        CuvsDistanceMetric::SquaredL2,
    );

    assert_eq!(result.report.boundary_tie_guard_launches, 2);
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_matches_cosine_reference_for_zero_vectors_and_ties() {
    let corpus = vec![
        1.0, 0.0, 0.0, // 0
        2.0, 0.0, 0.0, // 1 same direction
        0.0, 1.0, 0.0, // 2
        0.0, -1.0, 0.0, // 3
        -1.0, 0.0, 0.0, // 4
        0.0, 0.0, 0.0, // 5 zero
        1.0, 1.0, 0.0, // 6
    ];
    let queries = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    assert_cuda_matches_reference(&corpus, &queries, 2, 3, 7, 3, CuvsDistanceMetric::Cosine);
}

#[cfg(sextant_cuvs)]
#[test]
fn cuda_rejects_non_finite_corpus_values() {
    let error = cuvs_chunked_bruteforce_topk(
        request(2, 2, &[1.0, 0.0], 1, 1, 1, CuvsDistanceMetric::SquaredL2),
        |start, _, out| {
            out.copy_from_slice(if start == 0 {
                &[0.0, 0.0]
            } else {
                &[f32::INFINITY, 0.0]
            });
            Ok(())
        },
    )
    .expect_err("non-finite corpus values must fail");

    assert_eq!(error.code, CALYX_INDEX_INVALID_PARAMS);
}

#[cfg(sextant_cuvs)]
fn assert_cuda_matches_reference(
    corpus: &[f32],
    queries: &[f32],
    query_count: usize,
    dim: usize,
    k: usize,
    chunk_rows: usize,
    metric: CuvsDistanceMetric,
) -> CuvsChunkedExactTopK {
    let corpus_rows = corpus.len() / dim;
    let result = cuvs_chunked_bruteforce_topk(
        request(
            corpus_rows as u64,
            dim,
            queries,
            query_count,
            k,
            chunk_rows,
            metric,
        ),
        |start, take, out| {
            let start = start as usize * dim;
            out.copy_from_slice(&corpus[start..start + take * dim]);
            Ok(())
        },
    )
    .expect("CUDA exact search");
    let expected = cpu_reference(corpus, queries, query_count, dim, k, metric);

    for (query, expected) in expected.iter().enumerate() {
        assert_eq!(result.row(query).0, expected, "query {query}");
    }
    assert_eq!(result.report.chunks, corpus_rows.div_ceil(chunk_rows));
    assert_eq!(result.report.query_uploads, 1);
    assert_eq!(result.report.intermediate_readback_pairs, 0);
    assert_eq!(result.report.final_readback_pairs, query_count * k);
    assert!(!result.report.host_merge);
    assert!(result.report.pinned_staging);
    if metric == CuvsDistanceMetric::Cosine {
        let zero_queries = queries
            .chunks_exact(dim)
            .filter(|row| row.iter().all(|value| *value == 0.0))
            .count();
        let zero_chunks = corpus
            .chunks(dim * chunk_rows)
            .filter(|chunk| {
                chunk
                    .chunks_exact(dim)
                    .any(|row| row.iter().all(|value| *value == 0.0))
            })
            .count();
        assert_eq!(result.report.zero_query_count, zero_queries);
        assert_eq!(result.report.cosine_zero_corpus_chunks, zero_chunks);
        assert_eq!(
            result.report.zero_query_repair_launches,
            usize::from(zero_queries > 0) * (result.report.chunks - zero_chunks)
        );
    }
    result
}

#[cfg(sextant_cuvs)]
fn cpu_reference(
    corpus: &[f32],
    queries: &[f32],
    query_count: usize,
    dim: usize,
    k: usize,
    metric: CuvsDistanceMetric,
) -> Vec<Vec<u64>> {
    (0..query_count)
        .map(|query| {
            let query = &queries[query * dim..(query + 1) * dim];
            let mut ranked = corpus
                .chunks_exact(dim)
                .enumerate()
                .map(|(id, row)| (id as u64, distance(query, row, metric)))
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
            ranked.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

#[cfg(sextant_cuvs)]
fn distance(left: &[f32], right: &[f32], metric: CuvsDistanceMetric) -> f32 {
    match metric {
        CuvsDistanceMetric::Cosine => crate::index::cosine_distance(left, right),
        CuvsDistanceMetric::SquaredL2 => crate::index::l2_sq(left, right),
    }
}
