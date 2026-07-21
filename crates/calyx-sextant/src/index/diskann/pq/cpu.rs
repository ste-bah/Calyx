use std::time::Instant;

use calyx_core::Result;

use super::{
    BuildOutput, DISKANN_PQ_SMALL_CORPUS_ROWS, DiskAnnPqBuildDiagnostics, DiskAnnPqBuildExecution,
    DiskAnnPqBuildParams, initial_codebook, l2_sq,
};

pub(super) fn build(
    rows: &[(u32, Vec<f32>)],
    params: DiskAnnPqBuildParams,
    requested: DiskAnnPqBuildExecution,
) -> Result<BuildOutput> {
    let total_started = Instant::now();
    let dim = rows[0].1.len();
    let subdim = dim / params.subvectors;
    let centroids = params.centroids.min(rows.len());
    let training_started = Instant::now();
    let mut codebook = initial_codebook(rows, params.subvectors, centroids);
    for subvector in 0..params.subvectors {
        train_subspace(
            rows,
            subvector,
            subdim,
            centroids,
            params.iterations,
            &mut codebook,
        );
    }
    let training_us = training_started.elapsed().as_micros();
    let encoding_started = Instant::now();
    let codes = encode(rows, params.subvectors, centroids, subdim, &codebook);
    let encoding_us = encoding_started.elapsed().as_micros();
    Ok(BuildOutput {
        codebook,
        codes,
        diagnostics: DiskAnnPqBuildDiagnostics {
            backend: "cpu-reference-v1".to_string(),
            requested_execution: requested.as_str().to_string(),
            strict_gpu_required: false,
            small_corpus_cpu_max_rows: DISKANN_PQ_SMALL_CORPUS_ROWS,
            row_count: rows.len(),
            dim,
            subvectors: params.subvectors,
            centroids,
            iterations: params.iterations,
            pinned_staging: false,
            resident_corpus: false,
            chunk_rows: rows.len(),
            chunks_per_pass: 1,
            subspace_upload_reuse: false,
            cagra_device_reuse: false,
            cagra_device_reuse_reason: "CPU reference path has no device corpus".to_string(),
            corpus_uploads: 0,
            h2d_transfers: 0,
            d2h_transfers: 0,
            corpus_bytes_uploaded: 0,
            codebook_bytes_uploaded: 0,
            codebook_bytes_read: 0,
            codes_bytes_read: 0,
            assignment_kernel_launches: 0,
            accumulation_kernel_launches: 0,
            centroid_kernel_launches: 0,
            memset_operations: 0,
            peak_device_bytes: 0,
            peak_pinned_host_bytes: 0,
            staging_us: 0,
            training_us,
            encoding_us,
            total_us: total_started.elapsed().as_micros(),
        },
    })
}

fn train_subspace(
    rows: &[(u32, Vec<f32>)],
    subvector: usize,
    subdim: usize,
    centroids: usize,
    iterations: usize,
    codebook: &mut [f32],
) {
    let codebook_start = subvector * centroids * subdim;
    let mut sums = vec![0.0; centroids * subdim];
    let mut counts = vec![0_usize; centroids];
    for _ in 0..iterations {
        sums.fill(0.0);
        counts.fill(0);
        for (_, vector) in rows {
            let offset = subvector * subdim;
            let values = &vector[offset..offset + subdim];
            let nearest = nearest(
                values,
                &codebook[codebook_start..codebook_start + centroids * subdim],
                centroids,
                subdim,
            );
            counts[nearest] += 1;
            let sum_at = nearest * subdim;
            for (destination, source) in sums[sum_at..sum_at + subdim].iter_mut().zip(values) {
                *destination += *source;
            }
        }
        for (centroid, count) in counts.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            let destination = codebook_start + centroid * subdim;
            let sum_at = centroid * subdim;
            for axis in 0..subdim {
                codebook[destination + axis] = sums[sum_at + axis] / count as f32;
            }
        }
    }
}

fn encode(
    rows: &[(u32, Vec<f32>)],
    subvectors: usize,
    centroids: usize,
    subdim: usize,
    codebook: &[f32],
) -> Vec<u8> {
    let mut codes = vec![0; rows.len() * subvectors];
    for (row, (_, vector)) in rows.iter().enumerate() {
        for subvector in 0..subvectors {
            let vector_at = subvector * subdim;
            let codebook_at = subvector * centroids * subdim;
            codes[row * subvectors + subvector] = nearest(
                &vector[vector_at..vector_at + subdim],
                &codebook[codebook_at..codebook_at + centroids * subdim],
                centroids,
                subdim,
            ) as u8;
        }
    }
    codes
}

fn nearest(values: &[f32], codebook: &[f32], centroids: usize, subdim: usize) -> usize {
    let mut best = 0;
    let mut best_distance = f32::INFINITY;
    for centroid in 0..centroids {
        let at = centroid * subdim;
        let distance = l2_sq(values, &codebook[at..at + subdim]);
        if distance.total_cmp(&best_distance).is_lt() {
            best_distance = distance;
            best = centroid;
        }
    }
    best
}
