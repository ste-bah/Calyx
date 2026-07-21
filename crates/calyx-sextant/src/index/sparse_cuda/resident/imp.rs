use std::sync::Arc;
use std::time::Instant;

use calyx_core::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use super::super::{SparseBm25CudaReport, SparseBm25CudaTopK, cuda_error, invalid};

const CUBIN: &[u8] = include_bytes!(env!("SEXTANT_SPARSE_BM25_CUBIN_PATH"));
const THREADS: usize = 256;

pub(super) struct ResidentSparse {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    score_fn: CudaFunction,
    topk_fn: CudaFunction,
    term_offsets: CudaSlice<u32>,
    posting_doc_ordinals: CudaSlice<u32>,
    posting_tfs: CudaSlice<f32>,
    doc_lengths: CudaSlice<f32>,
    all_mask: CudaSlice<u8>,
    total_docs: usize,
    term_count: usize,
    posting_count: usize,
    resident_bytes: usize,
    query_terms: Option<CudaSlice<u32>>,
    query_weights: Option<CudaSlice<f32>>,
    candidate_mask: Option<CudaSlice<u8>>,
    scores: Option<CudaSlice<f32>>,
    out_doc_ordinals: Option<CudaSlice<u32>>,
    out_scores: Option<CudaSlice<f32>>,
    out_count: Option<CudaSlice<u32>>,
}

impl ResidentSparse {
    pub(super) fn new(
        total_docs: usize,
        term_offsets: &[u32],
        posting_doc_ordinals: &[u32],
        posting_tfs: &[f32],
        doc_lengths: &[f32],
    ) -> Result<Self> {
        let context = CudaContext::new(0).map_err(cuda_error("resident context init"))?;
        let stream = context
            .new_stream()
            .map_err(cuda_error("resident stream init"))?;
        let module = context
            .load_module(Ptx::from_binary(CUBIN.to_vec()))
            .map_err(cuda_error("resident CUBIN load"))?;
        let static_bytes = size_of_val(term_offsets)
            + size_of_val(posting_doc_ordinals)
            + size_of_val(posting_tfs)
            + size_of_val(doc_lengths)
            + total_docs;
        Ok(Self {
            _context: context,
            score_fn: module
                .load_function("sparse_bm25_score_docs")
                .map_err(cuda_error("resident score load"))?,
            topk_fn: module
                .load_function("sparse_bm25_topk")
                .map_err(cuda_error("resident top-k load"))?,
            term_offsets: upload(&stream, term_offsets, "resident term offsets")?,
            posting_doc_ordinals: upload(&stream, posting_doc_ordinals, "resident posting docs")?,
            posting_tfs: upload(&stream, posting_tfs, "resident posting tfs")?,
            doc_lengths: upload(&stream, doc_lengths, "resident doc lengths")?,
            all_mask: upload(&stream, &vec![1_u8; total_docs], "resident all mask")?,
            stream,
            total_docs,
            term_count: term_offsets.len() - 1,
            posting_count: posting_doc_ordinals.len(),
            resident_bytes: static_bytes,
            query_terms: None,
            query_weights: None,
            candidate_mask: None,
            scores: None,
            out_doc_ordinals: None,
            out_scores: None,
            out_count: None,
        })
    }

    pub(super) fn search(
        &mut self,
        avg_doc_len: f32,
        k: usize,
        query_term_ordinals: &[u32],
        query_weights: &[f32],
        candidate_mask: Option<&[u8]>,
    ) -> Result<SparseBm25CudaTopK> {
        let started = Instant::now();
        self.ensure_buffers(query_term_ordinals.len(), k, candidate_mask.is_some())?;
        self.stream
            .memcpy_htod(
                query_term_ordinals,
                self.query_terms.as_mut().expect("query terms"),
            )
            .map_err(cuda_error("resident query term upload"))?;
        self.stream
            .memcpy_htod(
                query_weights,
                self.query_weights.as_mut().expect("query weights"),
            )
            .map_err(cuda_error("resident query weight upload"))?;
        if let Some(mask) = candidate_mask {
            self.stream
                .memcpy_htod(mask, self.candidate_mask.as_mut().expect("candidate mask"))
                .map_err(cuda_error("resident candidate upload"))?;
        }
        self.launch_score(
            avg_doc_len,
            query_term_ordinals.len(),
            candidate_mask.is_some(),
        )?;
        self.launch_topk(k)?;
        let count = self
            .stream
            .clone_dtoh(self.out_count.as_ref().expect("out count"))
            .map_err(cuda_error("resident count readback"))?[0] as usize;
        let mut doc_ordinals = self
            .stream
            .clone_dtoh(self.out_doc_ordinals.as_ref().expect("out docs"))
            .map_err(cuda_error("resident doc readback"))?;
        let mut scores = self
            .stream
            .clone_dtoh(self.out_scores.as_ref().expect("out scores"))
            .map_err(cuda_error("resident score readback"))?;
        doc_ordinals.truncate(count);
        scores.truncate(count);
        let dynamic_h2d = size_of_val(query_term_ordinals)
            + size_of_val(query_weights)
            + candidate_mask.map_or(0, <[u8]>::len);
        let output_bytes = k * (size_of::<u32>() + size_of::<f32>()) + size_of::<u32>();
        Ok(SparseBm25CudaTopK {
            doc_ordinals,
            scores,
            report: SparseBm25CudaReport {
                backend: "cuda-sparse-bm25-resident-v2",
                total_docs: self.total_docs,
                term_count: self.term_count,
                posting_count: self.posting_count,
                query_terms: query_term_ordinals.len(),
                k,
                hits: count,
                score_kernel_launches: 1,
                topk_kernel_launches: 1,
                h2d_bytes: dynamic_h2d,
                d2h_bytes: output_bytes,
                final_readback_pairs: count,
                candidate_mask_uploaded: candidate_mask.is_some(),
                peak_device_bytes: self.resident_bytes
                    + dynamic_h2d
                    + self.total_docs * size_of::<f32>()
                    + output_bytes,
                elapsed_us: started.elapsed().as_micros(),
            },
        })
    }

    fn ensure_buffers(&mut self, terms: usize, k: usize, filtered: bool) -> Result<()> {
        ensure(
            &self.stream,
            &mut self.query_terms,
            terms,
            "resident query terms",
        )?;
        ensure(
            &self.stream,
            &mut self.query_weights,
            terms,
            "resident query weights",
        )?;
        if filtered {
            ensure(
                &self.stream,
                &mut self.candidate_mask,
                self.total_docs,
                "resident candidate mask",
            )?;
        }
        ensure(
            &self.stream,
            &mut self.scores,
            self.total_docs,
            "resident scores",
        )?;
        ensure(
            &self.stream,
            &mut self.out_doc_ordinals,
            k,
            "resident out docs",
        )?;
        ensure(&self.stream, &mut self.out_scores, k, "resident out scores")?;
        ensure(&self.stream, &mut self.out_count, 1, "resident out count")
    }

    fn launch_score(&mut self, avg_doc_len: f32, query_terms: usize, filtered: bool) -> Result<()> {
        let docs = to_i32(self.total_docs, "docs")?;
        let query_terms = to_i32(query_terms, "query terms")?;
        let k1 = 1.2_f32;
        let b = 0.75_f32;
        let candidate_mask = if filtered {
            self.candidate_mask.as_ref().expect("candidate mask")
        } else {
            &self.all_mask
        };
        let mut launch = self.stream.launch_builder(&self.score_fn);
        unsafe {
            launch
                .arg(&self.term_offsets)
                .arg(&self.posting_doc_ordinals)
                .arg(&self.posting_tfs)
                .arg(&self.doc_lengths)
                .arg(candidate_mask)
                .arg(self.query_terms.as_ref().expect("query terms"))
                .arg(self.query_weights.as_ref().expect("query weights"))
                .arg(&docs)
                .arg(&query_terms)
                .arg(&avg_doc_len)
                .arg(&k1)
                .arg(&b)
                .arg(self.scores.as_mut().expect("scores"))
                .launch(linear_config(self.total_docs)?)
        }
        .map(|_| ())
        .map_err(cuda_error("resident score launch"))
    }

    fn launch_topk(&mut self, k: usize) -> Result<()> {
        let docs = to_i32(self.total_docs, "docs")?;
        let k = to_i32(k, "k")?;
        let mut launch = self.stream.launch_builder(&self.topk_fn);
        unsafe {
            launch
                .arg(self.scores.as_ref().expect("scores"))
                .arg(&docs)
                .arg(&k)
                .arg(self.out_doc_ordinals.as_mut().expect("out docs"))
                .arg(self.out_scores.as_mut().expect("out scores"))
                .arg(self.out_count.as_mut().expect("out count"))
                .launch(single_thread_config())
        }
        .map(|_| ())
        .map_err(cuda_error("resident top-k launch"))
    }
}

fn upload<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    values: &[T],
    stage: &'static str,
) -> Result<CudaSlice<T>> {
    stream.clone_htod(values).map_err(cuda_error(stage))
}

fn ensure<T: cudarc::driver::DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    target: &mut Option<CudaSlice<T>>,
    len: usize,
    stage: &'static str,
) -> Result<()> {
    if target.as_ref().is_none_or(|buffer| buffer.len() < len) {
        *target = Some(stream.alloc_zeros(len).map_err(cuda_error(stage))?);
    }
    Ok(())
}

fn linear_config(items: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(items.div_ceil(THREADS))
                .map_err(|_| invalid("resident sparse CUDA grid exceeds u32"))?,
            1,
            1,
        ),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn single_thread_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn to_i32(value: usize, label: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("resident sparse CUDA {label} exceeds i32")))
}
