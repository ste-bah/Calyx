use calyx_core::Result;

use super::{SPARSE_BM25_CUDA_MAX_K, SparseBm25CudaReport, SparseBm25CudaTopK, invalid};

pub struct SparseBm25CudaIndex {
    total_docs: usize,
    term_count: usize,
    #[cfg(sextant_cuvs)]
    inner: imp::ResidentSparse,
}

impl std::fmt::Debug for SparseBm25CudaIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SparseBm25CudaIndex")
            .field("total_docs", &self.total_docs)
            .field("term_count", &self.term_count)
            .finish_non_exhaustive()
    }
}

impl SparseBm25CudaIndex {
    pub fn new(
        total_docs: usize,
        term_offsets: &[u32],
        posting_doc_ordinals: &[u32],
        posting_tfs: &[f32],
        doc_lengths: &[f32],
    ) -> Result<Self> {
        validate_static(
            total_docs,
            term_offsets,
            posting_doc_ordinals,
            posting_tfs,
            doc_lengths,
        )?;
        #[cfg(sextant_cuvs)]
        {
            Ok(Self {
                total_docs,
                term_count: term_offsets.len() - 1,
                inner: imp::ResidentSparse::new(
                    total_docs,
                    term_offsets,
                    posting_doc_ordinals,
                    posting_tfs,
                    doc_lengths,
                )?,
            })
        }
        #[cfg(not(sextant_cuvs))]
        {
            Err(unavailable())
        }
    }

    pub fn search(
        &mut self,
        avg_doc_len: f32,
        k: usize,
        query_term_ordinals: &[u32],
        query_weights: &[f32],
        candidate_mask: Option<&[u8]>,
    ) -> Result<SparseBm25CudaTopK> {
        validate_dynamic(
            self.total_docs,
            self.term_count,
            avg_doc_len,
            k,
            query_term_ordinals,
            query_weights,
            candidate_mask,
        )?;
        if k == 0 || query_term_ordinals.is_empty() || self.total_docs == 0 {
            return Ok(SparseBm25CudaTopK {
                doc_ordinals: Vec::new(),
                scores: Vec::new(),
                report: SparseBm25CudaReport {
                    backend: "cuda-sparse-bm25-resident-v2",
                    total_docs: self.total_docs,
                    term_count: self.term_count,
                    posting_count: 0,
                    query_terms: query_term_ordinals.len(),
                    k,
                    hits: 0,
                    score_kernel_launches: 0,
                    topk_kernel_launches: 0,
                    h2d_bytes: 0,
                    d2h_bytes: 0,
                    final_readback_pairs: 0,
                    candidate_mask_uploaded: candidate_mask.is_some(),
                    peak_device_bytes: 0,
                    elapsed_us: 0,
                },
            });
        }
        #[cfg(sextant_cuvs)]
        {
            self.inner.search(
                avg_doc_len,
                k,
                query_term_ordinals,
                query_weights,
                candidate_mask,
            )
        }
        #[cfg(not(sextant_cuvs))]
        {
            Err(unavailable())
        }
    }
}

fn validate_static(
    total_docs: usize,
    term_offsets: &[u32],
    posting_doc_ordinals: &[u32],
    posting_tfs: &[f32],
    doc_lengths: &[f32],
) -> Result<()> {
    if doc_lengths.len() != total_docs || term_offsets.first().copied() != Some(0) {
        return Err(invalid("resident sparse CUDA static shape is invalid"));
    }
    if posting_doc_ordinals.len() != posting_tfs.len()
        || term_offsets.last().copied().map(|value| value as usize)
            != Some(posting_doc_ordinals.len())
        || term_offsets.windows(2).any(|window| window[1] < window[0])
        || posting_doc_ordinals
            .iter()
            .any(|doc| *doc as usize >= total_docs)
        || posting_tfs
            .iter()
            .chain(doc_lengths)
            .any(|value| !value.is_finite())
    {
        return Err(invalid("resident sparse CUDA static corpus is invalid"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_dynamic(
    total_docs: usize,
    term_count: usize,
    avg_doc_len: f32,
    k: usize,
    query_term_ordinals: &[u32],
    query_weights: &[f32],
    candidate_mask: Option<&[u8]>,
) -> Result<()> {
    if k > SPARSE_BM25_CUDA_MAX_K
        || query_term_ordinals.len() != query_weights.len()
        || query_term_ordinals
            .iter()
            .any(|term| *term as usize >= term_count)
        || query_weights.iter().any(|value| !value.is_finite())
        || !avg_doc_len.is_finite()
        || avg_doc_len < 0.0
        || candidate_mask.is_some_and(|mask| mask.len() != total_docs)
    {
        return Err(invalid("resident sparse CUDA query shape is invalid"));
    }
    Ok(())
}

#[cfg(not(sextant_cuvs))]
fn unavailable() -> calyx_core::CalyxError {
    crate::error::sextant_error(
        crate::error::CALYX_SEXTANT_GPU_PARITY_UNAVAILABLE,
        crate::cuvs_unavailable_reason("resident sparse CUDA serving"),
    )
}

#[cfg(sextant_cuvs)]
#[path = "resident/imp.rs"]
mod imp;
