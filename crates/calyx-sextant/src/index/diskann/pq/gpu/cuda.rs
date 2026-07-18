use std::sync::Arc;
use std::time::Instant;

use calyx_core::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;

use super::super::{
    BuildOutput, DISKANN_PQ_SMALL_CORPUS_ROWS, DiskAnnPqBuildDiagnostics, DiskAnnPqBuildExecution,
    DiskAnnPqBuildParams, initial_codebook, invalid,
};
use crate::error::{CALYX_INDEX_IO, sextant_error};

use super::launch;

const PQ_CUBIN: &[u8] = include_bytes!(env!("SEXTANT_DISKANN_PQ_CUBIN_PATH"));
const DEFAULT_RESIDENT_MIB: usize = 512;
const DEFAULT_CHUNK_MIB: usize = 128;

#[derive(Default)]
struct Counters {
    corpus_uploads: usize,
    d2h_transfers: usize,
    assignment_launches: usize,
    accumulation_launches: usize,
    centroid_launches: usize,
    staging_us: u128,
}

pub(super) fn build(
    rows: &[(u32, Vec<f32>)],
    params: DiskAnnPqBuildParams,
    requested: DiskAnnPqBuildExecution,
) -> Result<BuildOutput> {
    let total_started = Instant::now();
    let dim = rows[0].1.len();
    let subdim = dim / params.subvectors;
    let centroids = params.centroids.min(rows.len());
    let corpus_bytes = rows
        .len()
        .checked_mul(dim)
        .and_then(|cells| cells.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid("CUDA PQ corpus byte size overflow"))?;
    let resident_limit = mib_bytes(
        "CALYX_DISKANN_PQ_GPU_RESIDENT_MIB",
        DEFAULT_RESIDENT_MIB,
        true,
    )?;
    let chunk_limit = mib_bytes("CALYX_DISKANN_PQ_GPU_CHUNK_MIB", DEFAULT_CHUNK_MIB, false)?;
    let resident_corpus = corpus_bytes <= resident_limit;
    let row_bytes = dim * size_of::<f32>();
    let chunk_rows = if resident_corpus {
        rows.len()
    } else {
        rows.len().min((chunk_limit / row_bytes).max(1))
    };
    let chunks_per_pass = rows.len().div_ceil(chunk_rows);
    let max_values = chunk_rows
        .checked_mul(dim)
        .ok_or_else(|| invalid("CUDA PQ chunk shape overflow"))?;
    let max_labels = chunk_rows
        .checked_mul(params.subvectors)
        .ok_or_else(|| invalid("CUDA PQ label shape overflow"))?;

    let context = CudaContext::new(0).map_err(cuda_error("context init"))?;
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_binary(PQ_CUBIN.to_vec()))
        .map_err(cuda_error("CUBIN load"))?;
    let kernels = launch::Kernels::load(&module)?;
    let mut corpus_host = pinned_zeros(&context, max_values, "corpus pinned allocation")?;
    let mut corpus_device = alloc_device(&stream, max_values, "corpus device allocation")?;
    let mut labels_device = alloc_device(&stream, max_labels, "label allocation")?;
    let initial = initial_codebook(rows, params.subvectors, centroids);
    let mut codebook_host = pinned_zeros(&context, initial.len(), "codebook pinned allocation")?;
    codebook_host
        .as_mut_slice()
        .map_err(cuda_error("codebook pinned access"))?
        .copy_from_slice(&initial);
    let mut codebook_device = stream
        .clone_htod(&codebook_host)
        .map_err(cuda_error("codebook upload"))?;
    stream
        .synchronize()
        .map_err(cuda_error("codebook upload sync"))?;
    let mut sums_device = alloc_device(&stream, initial.len(), "sum allocation")?;
    let mut counts_device =
        alloc_device(&stream, params.subvectors * centroids, "count allocation")?;
    let mut counters = Counters::default();

    if resident_corpus {
        stage_rows(
            rows,
            0,
            rows.len(),
            dim,
            &mut corpus_host,
            &stream,
            &mut corpus_device,
            &mut counters,
        )?;
    }

    let training_started = Instant::now();
    for _ in 0..params.iterations {
        stream
            .memset_zeros(&mut sums_device)
            .map_err(cuda_error("zero sums"))?;
        stream
            .memset_zeros(&mut counts_device)
            .map_err(cuda_error("zero counts"))?;
        if resident_corpus {
            train_chunk(
                &kernels,
                &stream,
                &corpus_device,
                rows.len(),
                dim,
                params.subvectors,
                centroids,
                subdim,
                &mut codebook_device,
                &mut labels_device,
                &mut sums_device,
                &mut counts_device,
                &mut counters,
            )?;
        } else {
            for start in (0..rows.len()).step_by(chunk_rows) {
                let take = chunk_rows.min(rows.len() - start);
                stage_rows(
                    rows,
                    start,
                    take,
                    dim,
                    &mut corpus_host,
                    &stream,
                    &mut corpus_device,
                    &mut counters,
                )?;
                train_chunk(
                    &kernels,
                    &stream,
                    &corpus_device,
                    take,
                    dim,
                    params.subvectors,
                    centroids,
                    subdim,
                    &mut codebook_device,
                    &mut labels_device,
                    &mut sums_device,
                    &mut counts_device,
                    &mut counters,
                )?;
            }
        }
        kernels.finalize(
            &stream,
            &sums_device,
            &counts_device,
            centroids,
            subdim,
            &mut codebook_device,
        )?;
        counters.centroid_launches += 1;
        stream
            .synchronize()
            .map_err(cuda_error("centroid update sync"))?;
    }
    let codebook = stream
        .clone_dtoh(&codebook_device)
        .map_err(cuda_error("codebook readback"))?;
    counters.d2h_transfers += 1;
    let training_us = training_started.elapsed().as_micros();
    if codebook.iter().any(|value| !value.is_finite()) {
        return Err(invalid("CUDA PQ produced a non-finite codebook"));
    }

    let encoding_started = Instant::now();
    let mut codes = Vec::with_capacity(rows.len() * params.subvectors);
    if resident_corpus {
        encode_chunk(
            &kernels,
            &stream,
            &corpus_device,
            rows.len(),
            dim,
            params.subvectors,
            centroids,
            subdim,
            &codebook_device,
            &mut labels_device,
            &mut counters,
        )?;
        let labels = stream
            .clone_dtoh(&labels_device)
            .map_err(cuda_error("code readback"))?;
        counters.d2h_transfers += 1;
        codes.extend_from_slice(&labels[..rows.len() * params.subvectors]);
    } else {
        for start in (0..rows.len()).step_by(chunk_rows) {
            let take = chunk_rows.min(rows.len() - start);
            stage_rows(
                rows,
                start,
                take,
                dim,
                &mut corpus_host,
                &stream,
                &mut corpus_device,
                &mut counters,
            )?;
            encode_chunk(
                &kernels,
                &stream,
                &corpus_device,
                take,
                dim,
                params.subvectors,
                centroids,
                subdim,
                &codebook_device,
                &mut labels_device,
                &mut counters,
            )?;
            let labels = stream
                .clone_dtoh(&labels_device)
                .map_err(cuda_error("code chunk readback"))?;
            counters.d2h_transfers += 1;
            codes.extend_from_slice(&labels[..take * params.subvectors]);
        }
    }
    let encoding_us = encoding_started.elapsed().as_micros();
    if codes.iter().any(|code| *code as usize >= centroids) {
        return Err(invalid("CUDA PQ produced an out-of-range code"));
    }

    let staged_corpus_bytes = max_values * size_of::<f32>();
    let codebook_bytes = initial.len() * size_of::<f32>();
    let label_bytes = max_labels * size_of::<u8>();
    let sum_bytes = codebook_bytes;
    let count_bytes = params.subvectors * centroids * size_of::<u32>();
    Ok(BuildOutput {
        codebook,
        codes,
        diagnostics: DiskAnnPqBuildDiagnostics {
            backend: "cuda-lloyd-tiled-v1".to_string(),
            requested_execution: requested.as_str().to_string(),
            strict_gpu_required: true,
            small_corpus_cpu_max_rows: DISKANN_PQ_SMALL_CORPUS_ROWS,
            row_count: rows.len(),
            dim,
            subvectors: params.subvectors,
            centroids,
            iterations: params.iterations,
            pinned_staging: true,
            resident_corpus,
            chunk_rows,
            chunks_per_pass,
            subspace_upload_reuse: true,
            cagra_device_reuse: false,
            cagra_device_reuse_reason: "the cuVS CAGRA C API releases its internal dataset allocation before returning; PQ therefore uses a bounded pinned lifecycle boundary".to_string(),
            corpus_uploads: counters.corpus_uploads,
            h2d_transfers: counters.corpus_uploads + 1,
            d2h_transfers: counters.d2h_transfers,
            corpus_bytes_uploaded: (staged_corpus_bytes as u64)
                .saturating_mul(counters.corpus_uploads as u64),
            codebook_bytes_uploaded: codebook_bytes,
            codebook_bytes_read: codebook_bytes,
            codes_bytes_read: label_bytes * chunks_per_pass,
            assignment_kernel_launches: counters.assignment_launches,
            accumulation_kernel_launches: counters.accumulation_launches,
            centroid_kernel_launches: counters.centroid_launches,
            memset_operations: params.iterations * 2,
            peak_device_bytes: staged_corpus_bytes
                + label_bytes
                + codebook_bytes
                + sum_bytes
                + count_bytes,
            peak_pinned_host_bytes: staged_corpus_bytes + codebook_bytes,
            staging_us: counters.staging_us,
            training_us,
            encoding_us,
            total_us: total_started.elapsed().as_micros(),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn train_chunk(
    kernels: &launch::Kernels,
    stream: &Arc<CudaStream>,
    rows: &CudaSlice<f32>,
    row_count: usize,
    dim: usize,
    subvectors: usize,
    centroids: usize,
    subdim: usize,
    codebook: &mut CudaSlice<f32>,
    labels: &mut CudaSlice<u8>,
    sums: &mut CudaSlice<f32>,
    counts: &mut CudaSlice<u32>,
    counters: &mut Counters,
) -> Result<()> {
    kernels.assign(
        stream, rows, row_count, dim, subvectors, centroids, subdim, codebook, labels,
    )?;
    counters.assignment_launches += 1;
    kernels.accumulate(
        stream, rows, labels, row_count, dim, subvectors, centroids, subdim, sums, counts,
    )?;
    counters.accumulation_launches += 1;
    stream
        .synchronize()
        .map_err(cuda_error("training chunk sync"))
}

#[allow(clippy::too_many_arguments)]
fn encode_chunk(
    kernels: &launch::Kernels,
    stream: &Arc<CudaStream>,
    rows: &CudaSlice<f32>,
    row_count: usize,
    dim: usize,
    subvectors: usize,
    centroids: usize,
    subdim: usize,
    codebook: &CudaSlice<f32>,
    labels: &mut CudaSlice<u8>,
    counters: &mut Counters,
) -> Result<()> {
    kernels.assign(
        stream, rows, row_count, dim, subvectors, centroids, subdim, codebook, labels,
    )?;
    counters.assignment_launches += 1;
    stream
        .synchronize()
        .map_err(cuda_error("encoding chunk sync"))
}

#[allow(clippy::too_many_arguments)]
fn stage_rows(
    rows: &[(u32, Vec<f32>)],
    start: usize,
    take: usize,
    dim: usize,
    host: &mut PinnedHostSlice<f32>,
    stream: &Arc<CudaStream>,
    device: &mut CudaSlice<f32>,
    counters: &mut Counters,
) -> Result<()> {
    let staging_started = Instant::now();
    let host_slice = host
        .as_mut_slice()
        .map_err(cuda_error("corpus pinned access"))?;
    for (destination, (_, source)) in host_slice[..take * dim]
        .chunks_exact_mut(dim)
        .zip(&rows[start..start + take])
    {
        destination.copy_from_slice(source);
    }
    counters.staging_us += staging_started.elapsed().as_micros();
    stream
        .memcpy_htod(&*host, device)
        .map_err(cuda_error("corpus upload"))?;
    stream
        .synchronize()
        .map_err(cuda_error("corpus upload sync"))?;
    counters.corpus_uploads += 1;
    Ok(())
}

fn mib_bytes(name: &'static str, default: usize, allow_zero: bool) -> Result<usize> {
    let mib = match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| invalid(format!("{name} must be an integer MiB value")))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(invalid(format!("cannot read {name}: {error}"))),
    };
    if (!allow_zero && mib == 0) || mib > 65_536 {
        return Err(invalid(format!(
            "{name}={mib} outside {}..=65536 MiB",
            usize::from(!allow_zero)
        )));
    }
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| invalid(format!("{name} byte size overflow")))
}

fn pinned_zeros<T>(
    context: &Arc<CudaContext>,
    len: usize,
    stage: &'static str,
) -> Result<PinnedHostSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    let mut pinned = unsafe { context.alloc_pinned::<T>(len) }.map_err(cuda_error(stage))?;
    let pointer = pinned.as_mut_ptr().map_err(cuda_error(stage))?;
    unsafe { pointer.write_bytes(0, len) };
    Ok(pinned)
}

fn alloc_device<T>(
    stream: &Arc<CudaStream>,
    len: usize,
    stage: &'static str,
) -> Result<CudaSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    stream.alloc_zeros(len).map_err(cuda_error(stage))
}

pub(super) fn to_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("CUDA PQ {name} exceeds i32")))
}

pub(super) fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| sextant_error(CALYX_INDEX_IO, format!("diskann CUDA PQ {stage}: {error}"))
}
