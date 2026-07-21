//! CUDA BM25 scoring for persisted sparse indexes.

use calyx_core::Result;
use serde::Serialize;

#[cfg(sextant_cuvs)]
use crate::error::CALYX_INDEX_IO;
use crate::error::{CALYX_INDEX_INVALID_PARAMS, sextant_error};

pub const SPARSE_BM25_CUDA_MAX_K: usize = 1024;

#[path = "sparse_cuda/resident.rs"]
mod resident;
pub use resident::SparseBm25CudaIndex;

#[derive(Clone, Copy, Debug)]
pub struct SparseBm25CudaRequest<'a> {
    pub total_docs: usize,
    pub avg_doc_len: f32,
    pub k: usize,
    pub term_offsets: &'a [u32],
    pub posting_doc_ordinals: &'a [u32],
    pub posting_tfs: &'a [f32],
    pub doc_lengths: &'a [f32],
    pub query_term_ordinals: &'a [u32],
    pub query_weights: &'a [f32],
    pub candidate_mask: Option<&'a [u8]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SparseBm25CudaReport {
    pub backend: &'static str,
    pub total_docs: usize,
    pub term_count: usize,
    pub posting_count: usize,
    pub query_terms: usize,
    pub k: usize,
    pub hits: usize,
    pub score_kernel_launches: usize,
    pub topk_kernel_launches: usize,
    pub h2d_bytes: usize,
    pub d2h_bytes: usize,
    pub final_readback_pairs: usize,
    pub candidate_mask_uploaded: bool,
    pub peak_device_bytes: usize,
    pub elapsed_us: u128,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SparseBm25CudaTopK {
    pub doc_ordinals: Vec<u32>,
    pub scores: Vec<f32>,
    pub report: SparseBm25CudaReport,
}

pub fn sparse_bm25_cuda_topk(request: SparseBm25CudaRequest<'_>) -> Result<SparseBm25CudaTopK> {
    validate_request(&request)?;
    if request.k == 0 || request.query_term_ordinals.is_empty() || request.total_docs == 0 {
        return Ok(SparseBm25CudaTopK {
            doc_ordinals: Vec::new(),
            scores: Vec::new(),
            report: empty_report(&request, 0),
        });
    }
    imp::run(request)
}

fn validate_request(request: &SparseBm25CudaRequest<'_>) -> Result<()> {
    if request.k > SPARSE_BM25_CUDA_MAX_K {
        return Err(invalid(format!(
            "sparse CUDA BM25 k {} exceeds max {SPARSE_BM25_CUDA_MAX_K}",
            request.k
        )));
    }
    if request.doc_lengths.len() != request.total_docs {
        return Err(invalid(format!(
            "sparse CUDA BM25 doc length count {} != total_docs {}",
            request.doc_lengths.len(),
            request.total_docs
        )));
    }
    if request.term_offsets.is_empty() || request.term_offsets[0] != 0 {
        return Err(invalid("sparse CUDA BM25 term offsets must start at zero"));
    }
    if request.posting_doc_ordinals.len() != request.posting_tfs.len() {
        return Err(invalid(format!(
            "sparse CUDA BM25 posting docs {} != tfs {}",
            request.posting_doc_ordinals.len(),
            request.posting_tfs.len()
        )));
    }
    let Some(&last_offset) = request.term_offsets.last() else {
        return Err(invalid("sparse CUDA BM25 missing term offsets"));
    };
    if last_offset as usize != request.posting_doc_ordinals.len() {
        return Err(invalid(format!(
            "sparse CUDA BM25 final offset {last_offset} != posting count {}",
            request.posting_doc_ordinals.len()
        )));
    }
    for window in request.term_offsets.windows(2) {
        if window[1] < window[0] {
            return Err(invalid("sparse CUDA BM25 term offsets are not monotonic"));
        }
    }
    for &doc in request.posting_doc_ordinals {
        if doc as usize >= request.total_docs {
            return Err(invalid(format!(
                "sparse CUDA BM25 posting doc ordinal {doc} outside total_docs {}",
                request.total_docs
            )));
        }
    }
    if request.query_term_ordinals.len() != request.query_weights.len() {
        return Err(invalid(format!(
            "sparse CUDA BM25 query terms {} != weights {}",
            request.query_term_ordinals.len(),
            request.query_weights.len()
        )));
    }
    let term_count = request.term_offsets.len() - 1;
    for &term in request.query_term_ordinals {
        if term as usize >= term_count {
            return Err(invalid(format!(
                "sparse CUDA BM25 query term ordinal {term} outside term_count {term_count}"
            )));
        }
    }
    if let Some(mask) = request.candidate_mask
        && mask.len() != request.total_docs
    {
        return Err(invalid(format!(
            "sparse CUDA BM25 candidate mask len {} != total_docs {}",
            mask.len(),
            request.total_docs
        )));
    }
    if !request.avg_doc_len.is_finite() || request.avg_doc_len < 0.0 {
        return Err(invalid(
            "sparse CUDA BM25 avg_doc_len must be finite and non-negative",
        ));
    }
    Ok(())
}

fn empty_report(request: &SparseBm25CudaRequest<'_>, elapsed_us: u128) -> SparseBm25CudaReport {
    SparseBm25CudaReport {
        backend: "cuda-sparse-bm25-v1",
        total_docs: request.total_docs,
        term_count: request.term_offsets.len().saturating_sub(1),
        posting_count: request.posting_doc_ordinals.len(),
        query_terms: request.query_term_ordinals.len(),
        k: request.k,
        hits: 0,
        score_kernel_launches: 0,
        topk_kernel_launches: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        final_readback_pairs: 0,
        candidate_mask_uploaded: request.candidate_mask.is_some(),
        peak_device_bytes: 0,
        elapsed_us,
    }
}

fn invalid(detail: impl std::fmt::Display) -> calyx_core::CalyxError {
    sextant_error(CALYX_INDEX_INVALID_PARAMS, detail.to_string())
}

#[cfg(sextant_cuvs)]
fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| sextant_error(CALYX_INDEX_IO, format!("sparse CUDA BM25 {stage}: {error}"))
}

#[cfg(sextant_cuvs)]
mod imp {
    use std::sync::Arc;
    use std::time::Instant;

    use calyx_core::Result;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
        ValidAsZeroBits,
    };
    use cudarc::nvrtc::Ptx;

    use super::{
        SparseBm25CudaReport, SparseBm25CudaRequest, SparseBm25CudaTopK, cuda_error, invalid,
    };

    const CUBIN: &[u8] = include_bytes!(env!("SEXTANT_SPARSE_BM25_CUBIN_PATH"));
    const THREADS: usize = 256;

    pub(super) fn run(request: SparseBm25CudaRequest<'_>) -> Result<SparseBm25CudaTopK> {
        let started = Instant::now();
        let cuda = CudaContext::new(0).map_err(cuda_error("context init"))?;
        let stream = cuda.default_stream();
        let module = cuda
            .load_module(Ptx::from_binary(CUBIN.to_vec()))
            .map_err(cuda_error("CUBIN load"))?;
        let score_fn = load(&module, "sparse_bm25_score_docs", "score load")?;
        let topk_fn = load(&module, "sparse_bm25_topk", "topk load")?;

        let term_offsets = stream
            .clone_htod(request.term_offsets)
            .map_err(cuda_error("term offset upload"))?;
        let posting_doc_ordinals = stream
            .clone_htod(request.posting_doc_ordinals)
            .map_err(cuda_error("posting doc upload"))?;
        let posting_tfs = stream
            .clone_htod(request.posting_tfs)
            .map_err(cuda_error("posting tf upload"))?;
        let doc_lengths = stream
            .clone_htod(request.doc_lengths)
            .map_err(cuda_error("doc length upload"))?;
        let mask_storage;
        let candidate_mask = match request.candidate_mask {
            Some(mask) => stream
                .clone_htod(mask)
                .map_err(cuda_error("candidate mask upload"))?,
            None => {
                mask_storage = vec![1_u8; request.total_docs];
                stream
                    .clone_htod(&mask_storage)
                    .map_err(cuda_error("candidate mask upload"))?
            }
        };
        let query_term_ordinals = stream
            .clone_htod(request.query_term_ordinals)
            .map_err(cuda_error("query term upload"))?;
        let query_weights = stream
            .clone_htod(request.query_weights)
            .map_err(cuda_error("query weight upload"))?;
        let mut scores = alloc_device::<f32>(&stream, request.total_docs, "scores")?;
        let mut out_doc_ordinals = alloc_device::<u32>(&stream, request.k, "top doc ordinals")?;
        let mut out_scores = alloc_device::<f32>(&stream, request.k, "top scores")?;
        let mut out_count = alloc_device::<u32>(&stream, 1, "top count")?;

        launch_score(
            &stream,
            &score_fn,
            &term_offsets,
            &posting_doc_ordinals,
            &posting_tfs,
            &doc_lengths,
            &candidate_mask,
            &query_term_ordinals,
            &query_weights,
            request.total_docs,
            request.query_term_ordinals.len(),
            request.avg_doc_len,
            &mut scores,
        )?;
        launch_topk(
            &stream,
            &topk_fn,
            &scores,
            request.total_docs,
            request.k,
            &mut out_doc_ordinals,
            &mut out_scores,
            &mut out_count,
        )?;

        let count = stream
            .clone_dtoh(&out_count)
            .map_err(cuda_error("top count readback"))?
            .into_iter()
            .next()
            .unwrap_or(0) as usize;
        let mut doc_ordinals = stream
            .clone_dtoh(&out_doc_ordinals)
            .map_err(cuda_error("top ordinal readback"))?;
        let mut scores = stream
            .clone_dtoh(&out_scores)
            .map_err(cuda_error("top score readback"))?;
        doc_ordinals.truncate(count);
        scores.truncate(count);

        let h2d_bytes = size_of_val(request.term_offsets)
            + size_of_val(request.posting_doc_ordinals)
            + size_of_val(request.posting_tfs)
            + size_of_val(request.doc_lengths)
            + request.total_docs * size_of::<u8>()
            + size_of_val(request.query_term_ordinals)
            + size_of_val(request.query_weights);
        let d2h_bytes = size_of::<u32>() + request.k * (size_of::<u32>() + size_of::<f32>());
        let peak_device_bytes = h2d_bytes
            + request.total_docs * size_of::<f32>()
            + request.k * (size_of::<u32>() + size_of::<f32>())
            + size_of::<u32>();
        Ok(SparseBm25CudaTopK {
            doc_ordinals,
            scores,
            report: SparseBm25CudaReport {
                backend: "cuda-sparse-bm25-v1",
                total_docs: request.total_docs,
                term_count: request.term_offsets.len() - 1,
                posting_count: request.posting_doc_ordinals.len(),
                query_terms: request.query_term_ordinals.len(),
                k: request.k,
                hits: count,
                score_kernel_launches: 1,
                topk_kernel_launches: 1,
                h2d_bytes,
                d2h_bytes,
                final_readback_pairs: count,
                candidate_mask_uploaded: request.candidate_mask.is_some(),
                peak_device_bytes,
                elapsed_us: started.elapsed().as_micros(),
            },
        })
    }

    fn load(
        module: &Arc<CudaModule>,
        name: &'static str,
        stage: &'static str,
    ) -> Result<CudaFunction> {
        module.load_function(name).map_err(cuda_error(stage))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_score(
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        term_offsets: &CudaSlice<u32>,
        posting_doc_ordinals: &CudaSlice<u32>,
        posting_tfs: &CudaSlice<f32>,
        doc_lengths: &CudaSlice<f32>,
        candidate_mask: &CudaSlice<u8>,
        query_term_ordinals: &CudaSlice<u32>,
        query_weights: &CudaSlice<f32>,
        total_docs: usize,
        query_terms: usize,
        avg_doc_len: f32,
        scores: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let total_docs_i32 = to_i32(total_docs, "total docs")?;
        let query_terms_i32 = to_i32(query_terms, "query terms")?;
        let k1 = 1.2_f32;
        let b = 0.75_f32;
        let mut launch = stream.launch_builder(function);
        unsafe {
            launch
                .arg(term_offsets)
                .arg(posting_doc_ordinals)
                .arg(posting_tfs)
                .arg(doc_lengths)
                .arg(candidate_mask)
                .arg(query_term_ordinals)
                .arg(query_weights)
                .arg(&total_docs_i32)
                .arg(&query_terms_i32)
                .arg(&avg_doc_len)
                .arg(&k1)
                .arg(&b)
                .arg(scores)
                .launch(linear_config(total_docs)?)
        }
        .map_err(cuda_error("score launch"))?;
        stream.synchronize().map_err(cuda_error("score sync"))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_topk(
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        scores: &CudaSlice<f32>,
        total_docs: usize,
        k: usize,
        out_doc_ordinals: &mut CudaSlice<u32>,
        out_scores: &mut CudaSlice<f32>,
        out_count: &mut CudaSlice<u32>,
    ) -> Result<()> {
        let total_docs_i32 = to_i32(total_docs, "total docs")?;
        let k_i32 = to_i32(k, "k")?;
        let mut launch = stream.launch_builder(function);
        unsafe {
            launch
                .arg(scores)
                .arg(&total_docs_i32)
                .arg(&k_i32)
                .arg(out_doc_ordinals)
                .arg(out_scores)
                .arg(out_count)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(cuda_error("topk launch"))?;
        stream.synchronize().map_err(cuda_error("topk sync"))
    }

    fn alloc_device<T>(
        stream: &Arc<CudaStream>,
        len: usize,
        name: &'static str,
    ) -> Result<CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
    {
        stream.alloc_zeros(len).map_err(cuda_error(name))
    }

    fn to_i32(value: usize, name: &'static str) -> Result<i32> {
        i32::try_from(value).map_err(|_| invalid(format!("sparse CUDA BM25 {name} exceeds i32")))
    }

    fn linear_config(work_items: usize) -> Result<LaunchConfig> {
        let blocks = work_items.div_ceil(THREADS);
        Ok(LaunchConfig {
            grid_dim: (
                u32::try_from(blocks).map_err(|_| invalid("sparse CUDA BM25 grid exceeds u32"))?,
                1,
                1,
            ),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
}

#[cfg(not(sextant_cuvs))]
mod imp {
    use calyx_core::Result;

    use super::{SparseBm25CudaRequest, SparseBm25CudaTopK};
    use crate::{cuvs_unavailable_reason, error::CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE};

    pub(super) fn run(_request: SparseBm25CudaRequest<'_>) -> Result<SparseBm25CudaTopK> {
        Err(crate::error::sextant_error(
            CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE,
            cuvs_unavailable_reason("sparse CUDA BM25 search"),
        ))
    }
}
