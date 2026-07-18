use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use calyx_core::{CxId, SlotId, SparseEntry};
#[cfg(feature = "cuda")]
use calyx_sextant::index::SparseBm25CudaIndex;
use calyx_sextant::index::{SPARSE_BM25_CUDA_MAX_K, SparseBm25CudaReport};
use serde::Serialize;

use super::{SparseIndex, SparsePosting, pinned, stale};
use crate::error::CliResult;
use crate::persisted::SearchIndexEntry;

const DEFAULT_SPARSE_CUDA_MIN_POSTINGS: usize = 4096;

type SparseCsrCache = Mutex<BTreeMap<(String, u16, String), Arc<SparseGpuCsr>>>;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct SparseGpuCsr {
    cx_by_ord: Vec<CxId>,
    doc_lengths: Vec<f32>,
    term_ord_by_idx: BTreeMap<u32, u32>,
    term_offsets: Vec<u32>,
    posting_doc_ordinals: Vec<u32>,
    posting_tfs: Vec<f32>,
    cached_bytes: usize,
    #[cfg(feature = "cuda")]
    resident: OnceLock<Mutex<SparseBm25CudaIndex>>,
}

#[derive(Serialize)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct SparseCudaPersistedReport<'a> {
    backend: &'a str,
    slot: u16,
    strict: bool,
    total_docs: usize,
    term_count: usize,
    indexed_postings: usize,
    matched_query_terms: usize,
    matched_query_postings: usize,
    csr_cached_bytes: usize,
    gpu: &'a SparseBm25CudaReport,
}

pub(super) fn score_cuda_topk(
    vault_dir: &Path,
    entry: &SearchIndexEntry,
    slot: SlotId,
    index: &SparseIndex,
    query: &[SparseEntry],
    k: usize,
    candidates: Option<&BTreeSet<CxId>>,
) -> CliResult<Option<Vec<(CxId, f32)>>> {
    let strict = sparse_cuda_strict();
    if sparse_cuda_disabled() {
        if strict {
            return Err(stale(
                "persistent sparse CUDA strict mode requested but CALYX_SEARCH_SPARSE_CUDA=0 disabled it",
            ));
        }
        return Ok(None);
    }
    let csr = sparse_csr(vault_dir, entry, slot, index)?;
    let mut query_term_ordinals = Vec::with_capacity(query.len());
    let mut query_weights = Vec::with_capacity(query.len());
    let mut matched_query_postings = 0usize;
    for query_entry in query {
        let Some(&term_ord) = csr.term_ord_by_idx.get(&query_entry.idx) else {
            continue;
        };
        let start = csr.term_offsets[term_ord as usize] as usize;
        let end = csr.term_offsets[term_ord as usize + 1] as usize;
        matched_query_postings += end - start;
        query_term_ordinals.push(term_ord);
        query_weights.push(query_entry.val);
    }
    if query_term_ordinals.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let min_postings = sparse_cuda_min_postings();
    if !strict && matched_query_postings < min_postings {
        return Ok(None);
    }
    if k > SPARSE_BM25_CUDA_MAX_K {
        return Err(stale(format!(
            "persistent sparse CUDA workload matched {matched_query_postings} postings but k {k} exceeds max {SPARSE_BM25_CUDA_MAX_K}; lower k or set CALYX_SEARCH_SPARSE_CUDA=0 to force explicit CPU fallback"
        )));
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = candidates;
        let _ = query_weights;
        let _ = csr;
        Err(stale(format!(
            "persistent sparse CUDA workload matched {matched_query_postings} postings but calyx-search was built without --features cuda; rebuild with CUDA support or set CALYX_SEARCH_SPARSE_CUDA=0 to force explicit CPU fallback"
        )))
    }

    #[cfg(feature = "cuda")]
    {
        let candidate_mask = candidates.map(|allowed| {
            let mut mask = vec![0_u8; csr.cx_by_ord.len()];
            for (ord, cx_id) in csr.cx_by_ord.iter().enumerate() {
                if allowed.contains(cx_id) {
                    mask[ord] = 1;
                }
            }
            mask
        });
        let result = resident_index(&csr)?
            .lock()
            .map_err(|_| stale("persistent sparse CUDA resident lock poisoned"))?
            .search(
                index.avg_doc_len,
                k,
                &query_term_ordinals,
                &query_weights,
                candidate_mask.as_deref(),
            )
            .map_err(|err| {
            stale(format!(
                "persistent sparse CUDA search failed for slot {slot}: {}; rebuild with CUDA available or unset CALYX_SEARCH_SPARSE_CUDA_STRICT",
                err
            ))
            })?;
        write_sparse_cuda_report(
            slot,
            strict,
            &csr,
            query_term_ordinals.len(),
            matched_query_postings,
            &result.report,
        )?;
        let mut scored = Vec::with_capacity(result.doc_ordinals.len());
        for (ord, score) in result.doc_ordinals.iter().zip(result.scores.iter()) {
            let cx_id = *csr.cx_by_ord.get(*ord as usize).ok_or_else(|| {
                stale("sparse CUDA BM25 returned an out-of-range document ordinal")
            })?;
            scored.push((cx_id, *score));
        }
        Ok(Some(super::top_k(scored, k)))
    }
}

fn sparse_csr(
    vault_dir: &Path,
    entry: &SearchIndexEntry,
    slot: SlotId,
    index: &SparseIndex,
) -> CliResult<Arc<SparseGpuCsr>> {
    let key = (
        pinned::canonical_vault_dir(vault_dir)?,
        slot.get(),
        entry.require_sha256(slot)?.to_string(),
    );
    {
        let cache = csr_cache().lock().expect("sparse CSR cache poisoned");
        if let Some(csr) = cache.get(&key) {
            return Ok(Arc::clone(csr));
        }
    }

    let csr = Arc::new(build_sparse_csr(index)?);
    let mut cache = csr_cache().lock().expect("sparse CSR cache poisoned");
    cache.retain(|candidate, _| {
        candidate.0 != key.0 || candidate.1 != key.1 || candidate.2 == key.2
    });
    Ok(Arc::clone(cache.entry(key).or_insert(csr)))
}

fn csr_cache() -> &'static SparseCsrCache {
    static CACHE: OnceLock<SparseCsrCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn build_sparse_csr(index: &SparseIndex) -> CliResult<SparseGpuCsr> {
    let mut cx_by_ord = index.rows.iter().map(|row| row.cx_id).collect::<Vec<_>>();
    cx_by_ord.sort();
    let mut ord_by_cx = BTreeMap::new();
    for (ord, cx_id) in cx_by_ord.iter().enumerate() {
        let ord = u32::try_from(ord)
            .map_err(|_| stale("persistent sparse CUDA document ordinal exceeds u32"))?;
        ord_by_cx.insert(*cx_id, ord);
    }
    let mut doc_lengths = Vec::with_capacity(cx_by_ord.len());
    for cx_id in &cx_by_ord {
        doc_lengths.push(*index.doc_lengths.get(cx_id).ok_or_else(|| {
            stale(format!(
                "persistent sparse CUDA missing doc length for {cx_id}; rebuild the vault search indexes"
            ))
        })?);
    }

    let mut term_ord_by_idx = BTreeMap::new();
    let mut term_offsets = Vec::with_capacity(index.postings.len() + 1);
    let mut posting_doc_ordinals = Vec::new();
    let mut posting_tfs = Vec::new();
    term_offsets.push(0);
    for (term_ord, (term_idx, postings)) in index.postings.iter().enumerate() {
        let term_ord = u32::try_from(term_ord)
            .map_err(|_| stale("persistent sparse CUDA term ordinal exceeds u32"))?;
        term_ord_by_idx.insert(*term_idx, term_ord);
        append_postings(
            &ord_by_cx,
            postings,
            &mut posting_doc_ordinals,
            &mut posting_tfs,
        )?;
        term_offsets.push(u32::try_from(posting_doc_ordinals.len()).map_err(|_| {
            stale("persistent sparse CUDA posting count exceeds u32; use a sharded sparse index")
        })?);
    }
    let cached_bytes = cx_by_ord.len() * size_of::<CxId>()
        + doc_lengths.len() * size_of::<f32>()
        + term_offsets.len() * size_of::<u32>()
        + posting_doc_ordinals.len() * size_of::<u32>()
        + posting_tfs.len() * size_of::<f32>()
        + term_ord_by_idx.len() * (size_of::<u32>() * 2);
    Ok(SparseGpuCsr {
        cx_by_ord,
        doc_lengths,
        term_ord_by_idx,
        term_offsets,
        posting_doc_ordinals,
        posting_tfs,
        cached_bytes,
        #[cfg(feature = "cuda")]
        resident: OnceLock::new(),
    })
}

#[cfg(feature = "cuda")]
fn resident_index(csr: &SparseGpuCsr) -> CliResult<&Mutex<SparseBm25CudaIndex>> {
    if let Some(resident) = csr.resident.get() {
        return Ok(resident);
    }
    let candidate = SparseBm25CudaIndex::new(
        csr.cx_by_ord.len(),
        &csr.term_offsets,
        &csr.posting_doc_ordinals,
        &csr.posting_tfs,
        &csr.doc_lengths,
    )
    .map_err(|error| stale(format!("initialize resident sparse CUDA index: {error}")))?;
    let _ = csr.resident.set(Mutex::new(candidate));
    Ok(csr
        .resident
        .get()
        .expect("resident sparse index initialized"))
}

fn append_postings(
    ord_by_cx: &BTreeMap<CxId, u32>,
    postings: &[SparsePosting],
    posting_doc_ordinals: &mut Vec<u32>,
    posting_tfs: &mut Vec<f32>,
) -> CliResult {
    let mut sorted = postings
        .iter()
        .map(|posting| {
            let ord = *ord_by_cx.get(&posting.cx_id).ok_or_else(|| {
                stale(format!(
                    "persistent sparse CUDA posting references unknown {}; rebuild the vault search indexes",
                    posting.cx_id
                ))
            })?;
            Ok((ord, posting.tf))
        })
        .collect::<CliResult<Vec<_>>>()?;
    sorted.sort_by_key(|(ord, _)| *ord);
    for (ord, tf) in sorted {
        posting_doc_ordinals.push(ord);
        posting_tfs.push(tf);
    }
    Ok(())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn write_sparse_cuda_report(
    slot: SlotId,
    strict: bool,
    csr: &SparseGpuCsr,
    matched_query_terms: usize,
    matched_query_postings: usize,
    gpu: &SparseBm25CudaReport,
) -> CliResult {
    let Ok(path) = env::var("CALYX_SEARCH_SPARSE_CUDA_REPORT") else {
        return Ok(());
    };
    let report = SparseCudaPersistedReport {
        backend: "persistent-sparse-cuda-csr-v1",
        slot: slot.get(),
        strict,
        total_docs: csr.cx_by_ord.len(),
        term_count: csr.term_offsets.len().saturating_sub(1),
        indexed_postings: csr.posting_doc_ordinals.len(),
        matched_query_terms,
        matched_query_postings,
        csr_cached_bytes: csr.cached_bytes,
        gpu,
    };
    let bytes = serde_json::to_vec_pretty(&report)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn sparse_cuda_min_postings() -> usize {
    env::var("CALYX_SEARCH_SPARSE_CUDA_MIN_POSTINGS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SPARSE_CUDA_MIN_POSTINGS)
}

fn sparse_cuda_strict() -> bool {
    env_truthy("CALYX_SEARCH_SPARSE_CUDA_STRICT")
}

pub(super) fn strict_mode() -> bool {
    sparse_cuda_strict()
}

fn sparse_cuda_disabled() -> bool {
    env::var("CALYX_SEARCH_SPARSE_CUDA")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

fn env_truthy(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        matches!(
            value.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        )
    })
}
