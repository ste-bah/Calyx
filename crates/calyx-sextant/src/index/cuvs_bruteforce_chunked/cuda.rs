use std::sync::Arc;
use std::time::Instant;

use calyx_core::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, PinnedHostSlice, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use super::{CuvsChunkedExactReport, CuvsChunkedExactTopK, CuvsCorpusStaging, CuvsDistanceMetric};
use crate::error::{CALYX_INDEX_INVALID_PARAMS, CALYX_INDEX_IO, sextant_error};

mod cuvs;
mod launch;
mod output;

const MERGE_CUBIN: &[u8] = include_bytes!(env!("SEXTANT_CHUNKED_EXACT_MERGE_CUBIN_PATH"));

type F32ChunkLoader<'a> = dyn FnMut(u64, usize, &mut [f32]) -> Result<()> + 'a;
type I8ChunkLoader<'a> = dyn FnMut(u64, usize, &mut [i8]) -> Result<()> + 'a;

pub(super) enum ChunkSource<'a> {
    F32(&'a mut F32ChunkLoader<'a>),
    I8(&'a mut I8ChunkLoader<'a>),
    Synthetic(u64),
}

impl ChunkSource<'_> {
    fn staging(&self) -> CuvsCorpusStaging {
        match self {
            Self::F32(_) => CuvsCorpusStaging::F32Host,
            Self::I8(_) => CuvsCorpusStaging::I8DeviceConvert,
            Self::Synthetic(_) => CuvsCorpusStaging::SyntheticDeviceGenerate,
        }
    }
}

enum CorpusBuffers {
    F32(PinnedHostSlice<f32>),
    I8 {
        host: PinnedHostSlice<i8>,
        device: CudaSlice<i8>,
    },
    Synthetic,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    corpus_rows: u64,
    dim: usize,
    queries: &[f32],
    query_count: usize,
    k: usize,
    chunk_rows: usize,
    metric: CuvsDistanceMetric,
    mut source: ChunkSource<'_>,
) -> Result<CuvsChunkedExactTopK> {
    let started = Instant::now();
    let cuda = CudaContext::new(0).map_err(cuda_error("context init"))?;
    let stream = cuda.default_stream();
    let resources = cuvs::Resources::new()?;
    let merge_module = cuda
        .load_module(Ptx::from_binary(MERGE_CUBIN.to_vec()))
        .map_err(cuda_error("merge CUBIN load"))?;
    let merge = launch::load(&merge_module, "merge_chunked_exact_topk", "merge load")?;
    let repair_zero_queries = launch::load(
        &merge_module,
        "repair_zero_cosine_queries",
        "zero-query repair load",
    )?;
    let exact_cosine_zero_rows = launch::load(
        &merge_module,
        "exact_cosine_chunk_with_zero_rows",
        "zero-row exact load",
    )?;
    let compute_boundary_distances = launch::load(
        &merge_module,
        "compute_chunked_exact_repair_distances",
        "boundary distance load",
    )?;
    let repair_boundary_ties = launch::load(
        &merge_module,
        "repair_chunked_exact_boundary_ties",
        "boundary repair load",
    )?;
    let convert_i8 = launch::load(&merge_module, "convert_i8_chunk_to_f32", "i8 staging load")?;
    let generate_synthetic = launch::load(
        &merge_module,
        "generate_synthetic_chunk",
        "synthetic staging load",
    )?;
    let zero_query_count = if metric == CuvsDistanceMetric::Cosine {
        queries
            .chunks_exact(dim)
            .filter(|query| zero_norm(query))
            .count()
    } else {
        0
    };

    let mut query_host = pinned_zeros(&cuda, queries.len(), "query")?;
    query_host
        .as_mut_slice()
        .map_err(cuda_error("query pinned access"))?
        .copy_from_slice(queries);
    let query_dev = stream
        .clone_htod(&query_host)
        .map_err(cuda_error("query upload"))?;

    let max_chunk_rows = chunk_rows.min(corpus_rows as usize);
    let max_chunk_values = max_chunk_rows
        .checked_mul(dim)
        .ok_or_else(|| invalid("chunk staging shape overflow"))?;
    let corpus_staging = source.staging();
    let mut corpus_buffers = match corpus_staging {
        CuvsCorpusStaging::F32Host => {
            CorpusBuffers::F32(pinned_zeros(&cuda, max_chunk_values, "f32 corpus")?)
        }
        CuvsCorpusStaging::I8DeviceConvert => CorpusBuffers::I8 {
            host: pinned_zeros(&cuda, max_chunk_values, "i8 corpus")?,
            device: alloc_device(&stream, max_chunk_values, "i8 corpus")?,
        },
        CuvsCorpusStaging::SyntheticDeviceGenerate => CorpusBuffers::Synthetic,
    };
    let mut corpus_dev = alloc_device(&stream, max_chunk_values, "corpus")?;
    let pair_count = query_count
        .checked_mul(k)
        .ok_or_else(|| invalid("top-k shape overflow"))?;
    let max_candidate_k = k.saturating_add(1).min(max_chunk_rows);
    let candidate_pair_count = query_count
        .checked_mul(max_candidate_k)
        .ok_or_else(|| invalid("candidate shape overflow"))?;
    let mut chunk_ids = alloc_device::<i64>(&stream, candidate_pair_count, "chunk ids")?;
    let mut chunk_distances =
        alloc_device::<f32>(&stream, candidate_pair_count, "chunk distances")?;
    let mut global_ids = alloc_device::<i64>(&stream, pair_count, "global ids")?;
    let mut global_distances = alloc_device::<f32>(&stream, pair_count, "global distances")?;
    let boundary_repair_values = max_chunk_rows
        .checked_mul(query_count.min(launch::BOUNDARY_REPAIR_QUERY_BATCH))
        .ok_or_else(|| invalid("boundary repair shape overflow"))?;
    let mut boundary_distances =
        alloc_device::<f32>(&stream, boundary_repair_values, "boundary distances")?;

    let mut chunks = 0usize;
    let mut cuvs_searches = 0usize;
    let mut cosine_zero_corpus_chunks = 0usize;
    let mut zero_query_repair_launches = 0usize;
    let mut boundary_tie_guard_launches = 0usize;
    let mut start = 0u64;
    while start < corpus_rows {
        let take = usize::try_from((corpus_rows - start).min(max_chunk_rows as u64))
            .map_err(|_| invalid("chunk row count exceeds usize"))?;
        let cosine_zero_corpus = stage_chunk(
            &mut source,
            &mut corpus_buffers,
            &stream,
            &convert_i8,
            &generate_synthetic,
            &mut corpus_dev,
            start,
            take,
            dim,
            metric,
        )?;

        let chunk_k = k.min(take);
        let candidate_k = k.saturating_add(1).min(take);
        if cosine_zero_corpus {
            launch::exact_cosine_zero_rows(
                &stream,
                &exact_cosine_zero_rows,
                &corpus_dev,
                take,
                dim,
                &query_dev,
                query_count,
                candidate_k,
                &mut chunk_ids,
                &mut chunk_distances,
            )?;
            cosine_zero_corpus_chunks += 1;
        } else {
            cuvs::search_chunk(
                &resources,
                &stream,
                &corpus_dev,
                take,
                dim,
                &query_dev,
                query_count,
                candidate_k,
                metric,
                &mut chunk_ids,
                &mut chunk_distances,
            )?;
            cuvs_searches += 1;
            if zero_query_count > 0 {
                launch::repair_zero_queries(
                    &stream,
                    &repair_zero_queries,
                    &query_dev,
                    dim,
                    query_count,
                    candidate_k,
                    &mut chunk_ids,
                    &mut chunk_distances,
                )?;
                zero_query_repair_launches += 1;
            }
            if candidate_k > chunk_k {
                boundary_tie_guard_launches += launch::repair_boundary_ties(
                    &stream,
                    &compute_boundary_distances,
                    &repair_boundary_ties,
                    &corpus_dev,
                    take,
                    dim,
                    &query_dev,
                    query_count,
                    candidate_k,
                    chunk_k,
                    metric,
                    &mut chunk_ids,
                    &mut chunk_distances,
                    &mut boundary_distances,
                )?;
            }
        }
        launch::merge(
            &stream,
            &merge,
            &mut chunk_ids,
            &mut chunk_distances,
            &mut global_ids,
            &mut global_distances,
            query_count,
            candidate_k,
            chunk_k,
            k,
            k.min(start as usize),
            k.min(start as usize + take),
            start,
        )?;
        chunks += 1;
        start += take as u64;
    }

    let ids = stream
        .clone_dtoh(&global_ids)
        .map_err(cuda_error("final id readback"))?;
    let distances = stream
        .clone_dtoh(&global_distances)
        .map_err(cuda_error("final distance readback"))?;
    let neighbors = output::validate(&ids, &distances, corpus_rows, query_count, k)?;
    let query_bytes = size_of_val(queries);
    let corpus_device_bytes = max_chunk_values * size_of::<f32>();
    let corpus_transfer_bytes = match corpus_staging {
        CuvsCorpusStaging::F32Host => corpus_device_bytes,
        CuvsCorpusStaging::I8DeviceConvert => max_chunk_values * size_of::<i8>(),
        CuvsCorpusStaging::SyntheticDeviceGenerate => 0,
    };
    let corpus_uploads = usize::from(corpus_transfer_bytes > 0) * chunks;
    let staging_kernel_launches =
        usize::from(corpus_staging != CuvsCorpusStaging::F32Host) * chunks;
    let staging_device_bytes = match corpus_staging {
        CuvsCorpusStaging::I8DeviceConvert => max_chunk_values * size_of::<i8>(),
        _ => 0,
    };
    let pair_size = size_of::<i64>() + size_of::<f32>();
    let final_pair_bytes = pair_count * pair_size;
    let candidate_pair_bytes = candidate_pair_count * pair_size;
    let boundary_repair_bytes = boundary_repair_values * size_of::<f32>();
    Ok(CuvsChunkedExactTopK {
        query_count,
        k,
        neighbors,
        distances,
        report: CuvsChunkedExactReport {
            backend: "cuvs-bruteforce-chunked-device-merge-v5",
            metric,
            corpus_rows,
            query_count,
            dim,
            k,
            chunk_rows: max_chunk_rows,
            chunks,
            distance_kernel_launches: cuvs_searches
                + cosine_zero_corpus_chunks
                + boundary_tie_guard_launches,
            cuvs_searches,
            cosine_zero_corpus_chunks,
            zero_query_count,
            zero_query_repair_launches,
            boundary_tie_guard_launches,
            merge_kernel_launches: chunks,
            query_uploads: 1,
            corpus_uploads,
            h2d_transfers: corpus_uploads + 1,
            d2h_transfers: 2,
            intermediate_readback_pairs: 0,
            final_readback_pairs: pair_count,
            host_merge: false,
            pinned_staging: true,
            query_bytes_uploaded: query_bytes,
            corpus_bytes_uploaded: (corpus_transfer_bytes as u64) * chunks as u64,
            final_bytes_read: final_pair_bytes,
            peak_device_staging_bytes: query_bytes
                + corpus_device_bytes
                + staging_device_bytes
                + candidate_pair_bytes
                + final_pair_bytes
                + boundary_repair_bytes,
            peak_pinned_staging_bytes: query_bytes + corpus_transfer_bytes,
            corpus_staging,
            staging_kernel_launches,
            device_generated_values: if corpus_staging == CuvsCorpusStaging::SyntheticDeviceGenerate
            {
                corpus_rows.saturating_mul(dim as u64)
            } else {
                0
            },
            elapsed_us: started.elapsed().as_micros(),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn stage_chunk(
    source: &mut ChunkSource<'_>,
    buffers: &mut CorpusBuffers,
    stream: &Arc<CudaStream>,
    convert_i8: &CudaFunction,
    generate_synthetic: &CudaFunction,
    corpus: &mut CudaSlice<f32>,
    start: u64,
    take: usize,
    dim: usize,
    metric: CuvsDistanceMetric,
) -> Result<bool> {
    let used_values = take * dim;
    match (source, buffers) {
        (ChunkSource::F32(load), CorpusBuffers::F32(host)) => {
            let slice = host
                .as_mut_slice()
                .map_err(cuda_error("f32 corpus pinned access"))?;
            load(start, take, &mut slice[..used_values])?;
            if slice[..used_values].iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "chunked cuVS exact corpus contains non-finite values",
                ));
            }
            let contains_zero = metric == CuvsDistanceMetric::Cosine
                && slice[..used_values].chunks_exact(dim).any(zero_norm);
            stream
                .memcpy_htod(&*host, corpus)
                .map_err(cuda_error("f32 corpus upload"))?;
            stream
                .synchronize()
                .map_err(cuda_error("f32 corpus upload sync"))?;
            Ok(contains_zero)
        }
        (ChunkSource::I8(load), CorpusBuffers::I8 { host, device }) => {
            let slice = host
                .as_mut_slice()
                .map_err(cuda_error("i8 corpus pinned access"))?;
            load(start, take, &mut slice[..used_values])?;
            let contains_zero = metric == CuvsDistanceMetric::Cosine
                && slice[..used_values]
                    .chunks_exact(dim)
                    .any(|row| row.iter().all(|value| *value == 0));
            stream
                .memcpy_htod(&*host, &mut *device)
                .map_err(cuda_error("i8 corpus upload"))?;
            launch::convert_i8(stream, convert_i8, device, take, dim, metric, corpus)?;
            Ok(contains_zero)
        }
        (ChunkSource::Synthetic(seed), CorpusBuffers::Synthetic) => {
            launch::generate_synthetic(
                stream,
                generate_synthetic,
                *seed,
                start,
                take,
                dim,
                corpus,
            )?;
            Ok(false)
        }
        _ => Err(invalid("chunked CUDA corpus staging state mismatch")),
    }
}

fn pinned_zeros<T>(
    context: &Arc<CudaContext>,
    len: usize,
    name: &'static str,
) -> Result<PinnedHostSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    let mut pinned = unsafe { context.alloc_pinned::<T>(len) }.map_err(cuda_error(name))?;
    let pointer = pinned.as_mut_ptr().map_err(cuda_error(name))?;
    unsafe { pointer.write_bytes(0, len) };
    Ok(pinned)
}

fn alloc_device<T>(stream: &Arc<CudaStream>, len: usize, name: &'static str) -> Result<CudaSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    stream.alloc_zeros(len).map_err(cuda_error(name))
}

fn to_i32(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("{name} exceeds i32")))
}

fn zero_norm(vector: &[f32]) -> bool {
    vector.iter().map(|value| value * value).sum::<f32>() == 0.0
}

fn invalid(detail: impl std::fmt::Display) -> calyx_core::CalyxError {
    sextant_error(CALYX_INDEX_INVALID_PARAMS, detail.to_string())
}

fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| {
        sextant_error(
            CALYX_INDEX_IO,
            format!("chunked cuVS exact {stage}: {error}"),
        )
    }
}
