use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use calyx_core::{CxId, SlotId};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{Kernel, LodestarError, Result};

const FORMAT_VERSION: u32 = 1;
const PANEL_FORMAT_VERSION: u32 = 2;
pub const PANEL_RRF_K: u32 = 60;

pub trait EmbeddingStore {
    fn embedding(&self, cx_id: CxId) -> Result<Option<Vec<f32>>>;
}

impl EmbeddingStore for BTreeMap<CxId, Vec<f32>> {
    fn embedding(&self, cx_id: CxId) -> Result<Option<Vec<f32>>> {
        Ok(self.get(&cx_id).cloned())
    }
}

pub trait KernelStore {
    fn write_index_bytes(&self, kernel_id: CxId, bytes: &[u8]) -> Result<()>;
    fn read_index_bytes(&self, kernel_id: CxId) -> Result<Option<Vec<u8>>>;
}

#[derive(Clone, Debug)]
pub struct FsKernelStore {
    root: PathBuf,
}

impl FsKernelStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn index_dir(&self, kernel_id: CxId) -> PathBuf {
        self.root
            .join("idx")
            .join("kernel")
            .join(kernel_id.to_string())
    }

    pub fn index_file_path(&self, kernel_id: CxId) -> PathBuf {
        self.index_dir(kernel_id).join("index.json")
    }

    pub fn kernel_file_path(&self, kernel_id: CxId) -> PathBuf {
        self.index_dir(kernel_id).join("kernel.json")
    }
}

impl KernelStore for FsKernelStore {
    fn write_index_bytes(&self, kernel_id: CxId, bytes: &[u8]) -> Result<()> {
        let dir = self.index_dir(kernel_id);
        let path = dir.join("index.json");
        install_immutable_file(&path, bytes)
    }

    fn read_index_bytes(&self, kernel_id: CxId) -> Result<Option<Vec<u8>>> {
        let path = self.index_file_path(kernel_id);
        if !Path::new(&path).exists() {
            return Ok(None);
        }
        fs::read(path).map(Some).map_err(io_error)
    }
}

pub(crate) fn install_immutable_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().ok_or_else(|| LodestarError::KernelIndexIo {
        detail: format!("immutable artifact path {} has no parent", path.display()),
    })?;
    fs::create_dir_all(parent).map_err(io_error)?;
    if path.exists() {
        let existing = fs::read(path).map_err(io_error)?;
        if existing == bytes {
            return Ok(());
        }
        return Err(LodestarError::KernelIndexIo {
            detail: format!(
                "refusing to replace immutable kernel artifact {} with different bytes",
                path.display()
            ),
        });
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| LodestarError::KernelIndexIo {
            detail: format!("immutable artifact path {} has no filename", path.display()),
        })?
        .to_string_lossy();
    let tmp = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(io_error)?;
    let publish = (|| {
        file.write_all(bytes).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::rename(&tmp, path).map_err(io_error)?;
        sync_parent(parent)
    })();
    if publish.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    publish
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> Result<()> {
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(io_error)
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> Result<()> {
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelVectorRow {
    pub cx_id: CxId,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct KernelIndex {
    pub kernel_id: CxId,
    pub dim: usize,
    rows: Vec<KernelVectorRow>,
}

impl KernelIndex {
    pub fn rows(&self) -> &[KernelVectorRow] {
        &self.rows
    }

    pub fn filter_to_nodes(&self, allowed_nodes: &BTreeSet<CxId>) -> Result<Self> {
        let rows = self
            .rows
            .iter()
            .filter(|row| allowed_nodes.contains(&row.cx_id))
            .cloned()
            .collect::<Vec<_>>();
        Self::from_rows(self.kernel_id, rows)
    }

    fn from_rows(kernel_id: CxId, rows: Vec<KernelVectorRow>) -> Result<Self> {
        let dim = validate_rows(&rows)?;
        Ok(Self {
            kernel_id,
            dim,
            rows,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct KernelIndexSnapshot {
    format_version: u32,
    kernel_id: CxId,
    dim: usize,
    rows: Vec<KernelVectorRow>,
}

pub fn build_kernel_index(kernel: &Kernel, embeddings: &dyn EmbeddingStore) -> Result<KernelIndex> {
    if kernel.members.is_empty() {
        return Err(LodestarError::KernelEmptyResult);
    }
    let rows = kernel
        .members
        .iter()
        .map(|cx_id| {
            let vector = embeddings
                .embedding(*cx_id)?
                .ok_or(LodestarError::KernelEmbeddingMissing { cx_id: *cx_id })?;
            Ok(KernelVectorRow {
                cx_id: *cx_id,
                vector,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    KernelIndex::from_rows(kernel.kernel_id, rows)
}

pub fn kernel_search(
    index: &KernelIndex,
    query_vec: &[f32],
    top_k: usize,
) -> Result<Vec<(CxId, f32)>> {
    if query_vec.len() != index.dim {
        return Err(LodestarError::KernelDimMismatch {
            expected: index.dim,
            actual: query_vec.len(),
        });
    }
    if let Some((offset, _)) = query_vec
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LodestarError::KernelInvalidParams {
            detail: format!("query vector has non-finite value at offset {offset}"),
        });
    }
    Ok(top_k_by_score(
        index
            .rows
            .par_iter()
            .map(|row| (row.cx_id, cosine(query_vec, &row.vector)))
            .collect(),
        top_k,
    ))
}

pub fn write_kernel_index(index: &KernelIndex, store: &dyn KernelStore) -> Result<()> {
    let snapshot = KernelIndexSnapshot {
        format_version: FORMAT_VERSION,
        kernel_id: index.kernel_id,
        dim: index.dim,
        rows: index.rows.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&snapshot).map_err(codec_error)?;
    store.write_index_bytes(index.kernel_id, &bytes)
}

pub fn load_kernel_index(kernel_id: CxId, store: &dyn KernelStore) -> Result<KernelIndex> {
    let Some(bytes) = store.read_index_bytes(kernel_id)? else {
        return Err(LodestarError::KernelIndexNotFound { kernel_id });
    };
    let snapshot: KernelIndexSnapshot = serde_json::from_slice(&bytes).map_err(codec_error)?;
    if snapshot.format_version != FORMAT_VERSION {
        return Err(LodestarError::KernelIndexCodec {
            detail: format!("unsupported format version {}", snapshot.format_version),
        });
    }
    if snapshot.kernel_id != kernel_id {
        return Err(LodestarError::KernelIndexCodec {
            detail: format!(
                "snapshot kernel id {} did not match requested {}",
                snapshot.kernel_id, kernel_id
            ),
        });
    }
    let actual_dim = validate_rows(&snapshot.rows)?;
    if snapshot.dim != actual_dim {
        return Err(LodestarError::KernelDimMismatch {
            expected: snapshot.dim,
            actual: actual_dim,
        });
    }
    KernelIndex::from_rows(snapshot.kernel_id, snapshot.rows)
}

fn validate_rows(rows: &[KernelVectorRow]) -> Result<usize> {
    if rows.is_empty() {
        return Err(LodestarError::KernelEmptyResult);
    }
    let dim = rows[0].vector.len();
    if dim == 0 {
        return Err(LodestarError::KernelInvalidParams {
            detail: "kernel vectors must have non-zero dimension".to_string(),
        });
    }
    let mut seen = BTreeSet::new();
    for row in rows {
        if !seen.insert(row.cx_id) {
            return Err(LodestarError::KernelInvalidParams {
                detail: format!("duplicate kernel row {}", row.cx_id),
            });
        }
        if row.vector.len() != dim {
            return Err(LodestarError::KernelDimMismatch {
                expected: dim,
                actual: row.vector.len(),
            });
        }
        if let Some((offset, _)) = row
            .vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(LodestarError::KernelInvalidParams {
                detail: format!("row {} has non-finite value at offset {offset}", row.cx_id),
            });
        }
    }
    Ok(dim)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut an = 0.0_f32;
    let mut bn = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        an += x * x;
        bn += y * y;
    }
    if an == 0.0 || bn == 0.0 {
        0.0
    } else {
        dot / (an.sqrt() * bn.sqrt())
    }
}

fn top_k_by_score(mut scored: Vec<(CxId, f32)>, top_k: usize) -> Vec<(CxId, f32)> {
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.to_string().cmp(&right.0.to_string()))
    });
    scored.truncate(top_k);
    scored
}

fn io_error(err: std::io::Error) -> LodestarError {
    LodestarError::KernelIndexIo {
        detail: err.to_string(),
    }
}

fn codec_error(err: serde_json::Error) -> LodestarError {
    LodestarError::KernelIndexCodec {
        detail: err.to_string(),
    }
}

/// A no-flatten panel measurement. Every contracted slot remains separately
/// addressable; fusion operates on ranks and never concatenates vectors.
pub type PanelVectors = BTreeMap<SlotId, Vec<f32>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PanelFusionLane {
    pub slot: SlotId,
    pub cosine: f32,
    pub rank: usize,
    pub rrf_contribution: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PanelFusionHit {
    pub cx_id: CxId,
    pub score: f32,
    pub lanes: Vec<PanelFusionLane>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PanelKernelVectorRow {
    pub cx_id: CxId,
    pub vectors: PanelVectors,
}

#[derive(Clone, Debug)]
pub struct PanelKernelIndex {
    pub kernel_id: CxId,
    slots: Vec<SlotId>,
    dims: BTreeMap<SlotId, usize>,
    rows: Vec<PanelKernelVectorRow>,
    candidates: BTreeMap<CxId, PanelVectors>,
}

impl PanelKernelIndex {
    pub fn slots(&self) -> &[SlotId] {
        &self.slots
    }

    pub fn dims(&self) -> &BTreeMap<SlotId, usize> {
        &self.dims
    }

    pub fn rows(&self) -> &[PanelKernelVectorRow] {
        &self.rows
    }

    fn from_rows(
        kernel_id: CxId,
        slots: Vec<SlotId>,
        rows: Vec<PanelKernelVectorRow>,
    ) -> Result<Self> {
        let candidates = rows
            .iter()
            .map(|row| (row.cx_id, row.vectors.clone()))
            .collect::<BTreeMap<_, _>>();
        if candidates.len() != rows.len() {
            return Err(LodestarError::KernelInvalidParams {
                detail: "panel kernel index contains duplicate constellation ids".to_string(),
            });
        }
        let dims = validate_panel(&slots, &candidates)?;
        Ok(Self {
            kernel_id,
            slots,
            dims,
            rows,
            candidates,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PanelKernelIndexSnapshot {
    format_version: u32,
    kernel_id: CxId,
    slots: Vec<SlotId>,
    dims: BTreeMap<SlotId, usize>,
    fusion: String,
    rrf_k: u32,
    rows: Vec<PanelKernelVectorRow>,
}

pub fn build_panel_kernel_index(
    kernel: &Kernel,
    slots: &[SlotId],
    embeddings: &BTreeMap<CxId, PanelVectors>,
) -> Result<PanelKernelIndex> {
    if kernel.members.is_empty() {
        return Err(LodestarError::KernelEmptyResult);
    }
    let rows = kernel
        .members
        .iter()
        .map(|cx_id| {
            let vectors = embeddings
                .get(cx_id)
                .cloned()
                .ok_or(LodestarError::KernelEmbeddingMissing { cx_id: *cx_id })?;
            Ok(PanelKernelVectorRow {
                cx_id: *cx_id,
                vectors,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    PanelKernelIndex::from_rows(kernel.kernel_id, slots.to_vec(), rows)
}

pub fn panel_kernel_search(
    index: &PanelKernelIndex,
    query: &PanelVectors,
    top_k: usize,
) -> Result<Vec<PanelFusionHit>> {
    let mut hits = rank_panel_candidates(query, &index.candidates, &index.slots, PANEL_RRF_K)?;
    hits.truncate(top_k);
    Ok(hits)
}

pub fn write_panel_kernel_index(index: &PanelKernelIndex, store: &dyn KernelStore) -> Result<()> {
    let snapshot = PanelKernelIndexSnapshot {
        format_version: PANEL_FORMAT_VERSION,
        kernel_id: index.kernel_id,
        slots: index.slots.clone(),
        dims: index.dims.clone(),
        fusion: "rrf".to_string(),
        rrf_k: PANEL_RRF_K,
        rows: index.rows.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&snapshot).map_err(codec_error)?;
    store.write_index_bytes(index.kernel_id, &bytes)
}

pub fn load_panel_kernel_index(
    kernel_id: CxId,
    store: &dyn KernelStore,
) -> Result<PanelKernelIndex> {
    let Some(bytes) = store.read_index_bytes(kernel_id)? else {
        return Err(LodestarError::KernelIndexNotFound { kernel_id });
    };
    let snapshot: PanelKernelIndexSnapshot = serde_json::from_slice(&bytes).map_err(codec_error)?;
    if snapshot.format_version != PANEL_FORMAT_VERSION
        || snapshot.kernel_id != kernel_id
        || snapshot.fusion != "rrf"
        || snapshot.rrf_k != PANEL_RRF_K
    {
        return Err(LodestarError::KernelIndexCodec {
            detail: format!(
                "invalid panel kernel index contract format={} kernel={} fusion={} rrf_k={}",
                snapshot.format_version, snapshot.kernel_id, snapshot.fusion, snapshot.rrf_k
            ),
        });
    }
    let index = PanelKernelIndex::from_rows(snapshot.kernel_id, snapshot.slots, snapshot.rows)?;
    if index.dims != snapshot.dims {
        return Err(LodestarError::KernelIndexCodec {
            detail: "panel kernel index dimensions differ from physical row readback".to_string(),
        });
    }
    Ok(index)
}

/// Deterministically rank a candidate set through every contracted lane using
/// normalized Reciprocal Rank Fusion. Missing, extra, malformed, or non-finite
/// lane values fail closed before any score is emitted.
pub fn rank_panel_candidates(
    query: &PanelVectors,
    candidates: &BTreeMap<CxId, PanelVectors>,
    slots: &[SlotId],
    rrf_k: u32,
) -> Result<Vec<PanelFusionHit>> {
    let dims = validate_panel(slots, candidates)?;
    validate_panel_vector("query", query, slots, &dims)?;
    if rrf_k == 0 {
        return Err(LodestarError::KernelInvalidParams {
            detail: "panel RRF K must be greater than zero".to_string(),
        });
    }
    let mut evidence = candidates
        .keys()
        .map(|id| (*id, Vec::with_capacity(slots.len())))
        .collect::<BTreeMap<_, _>>();
    let normalizer = (rrf_k as f32 + 1.0) / slots.len() as f32;
    for &slot in slots {
        let query_vector = query.get(&slot).expect("validated query slot");
        let mut lane = candidates
            .par_iter()
            .map(|(id, vectors)| {
                let vector = vectors.get(&slot).expect("validated candidate slot");
                (*id, cosine(query_vector, vector))
            })
            .collect::<Vec<_>>();
        lane.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (offset, (id, similarity)) in lane.into_iter().enumerate() {
            let rank = offset + 1;
            let rrf_contribution = normalizer / (rrf_k as f32 + rank as f32);
            evidence
                .get_mut(&id)
                .expect("candidate evidence")
                .push(PanelFusionLane {
                    slot,
                    cosine: similarity,
                    rank,
                    rrf_contribution,
                });
        }
    }
    let mut hits = evidence
        .into_iter()
        .map(|(cx_id, lanes)| PanelFusionHit {
            cx_id,
            score: lanes.iter().map(|lane| lane.rrf_contribution).sum(),
            lanes,
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.cx_id.cmp(&right.cx_id))
    });
    Ok(hits)
}

/// Borrowing variant used by graph construction so candidate generation does
/// not clone every high-dimensional panel for every source node.
pub fn rank_panel_candidate_refs(
    query: &PanelVectors,
    candidates: &BTreeMap<CxId, &PanelVectors>,
    slots: &[SlotId],
    rrf_k: u32,
) -> Result<Vec<PanelFusionHit>> {
    if slots.len() < 2 || candidates.is_empty() {
        return Err(LodestarError::KernelInvalidParams {
            detail: format!(
                "panel fusion needs at least two slots and one candidate; slots={} candidates={}",
                slots.len(),
                candidates.len()
            ),
        });
    }
    if !slots.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(LodestarError::KernelInvalidParams {
            detail: "panel fusion slots must be strictly increasing and unique".to_string(),
        });
    }
    let first = *candidates.values().next().expect("nonempty candidates");
    let dims = slots
        .iter()
        .map(|slot| (*slot, first.get(slot).map(Vec::len).unwrap_or(0)))
        .collect::<BTreeMap<_, _>>();
    validate_panel_vector("query", query, slots, &dims)?;
    for (id, vectors) in candidates {
        validate_panel_vector(&format!("candidate {id}"), vectors, slots, &dims)?;
    }
    if rrf_k == 0 {
        return Err(LodestarError::KernelInvalidParams {
            detail: "panel RRF K must be greater than zero".to_string(),
        });
    }
    let mut evidence = candidates
        .keys()
        .map(|id| (*id, Vec::with_capacity(slots.len())))
        .collect::<BTreeMap<_, _>>();
    let normalizer = (rrf_k as f32 + 1.0) / slots.len() as f32;
    for &slot in slots {
        let query_vector = query.get(&slot).expect("validated query slot");
        let mut lane = candidates
            .par_iter()
            .map(|(id, vectors)| {
                let vector = vectors.get(&slot).expect("validated candidate slot");
                (*id, cosine(query_vector, vector))
            })
            .collect::<Vec<_>>();
        lane.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (offset, (id, similarity)) in lane.into_iter().enumerate() {
            let rank = offset + 1;
            evidence
                .get_mut(&id)
                .expect("candidate evidence")
                .push(PanelFusionLane {
                    slot,
                    cosine: similarity,
                    rank,
                    rrf_contribution: normalizer / (rrf_k as f32 + rank as f32),
                });
        }
    }
    let mut hits = evidence
        .into_iter()
        .map(|(cx_id, lanes)| PanelFusionHit {
            cx_id,
            score: lanes.iter().map(|lane| lane.rrf_contribution).sum(),
            lanes,
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.cx_id.cmp(&right.cx_id))
    });
    Ok(hits)
}

fn validate_panel(
    slots: &[SlotId],
    candidates: &BTreeMap<CxId, PanelVectors>,
) -> Result<BTreeMap<SlotId, usize>> {
    if slots.len() < 2 || candidates.is_empty() {
        return Err(LodestarError::KernelInvalidParams {
            detail: format!(
                "panel fusion needs at least two slots and one candidate; slots={} candidates={}",
                slots.len(),
                candidates.len()
            ),
        });
    }
    let ordered = slots.windows(2).all(|pair| pair[0] < pair[1]);
    if !ordered {
        return Err(LodestarError::KernelInvalidParams {
            detail: "panel fusion slots must be strictly increasing and unique".to_string(),
        });
    }
    let first = candidates.values().next().expect("nonempty candidates");
    let dims = slots
        .iter()
        .map(|slot| {
            let dim = first.get(slot).map(Vec::len).unwrap_or(0);
            (*slot, dim)
        })
        .collect::<BTreeMap<_, _>>();
    for (id, vectors) in candidates {
        validate_panel_vector(&format!("candidate {id}"), vectors, slots, &dims)?;
    }
    Ok(dims)
}

fn validate_panel_vector(
    label: &str,
    vectors: &PanelVectors,
    slots: &[SlotId],
    dims: &BTreeMap<SlotId, usize>,
) -> Result<()> {
    if vectors.len() != slots.len() || vectors.keys().copied().ne(slots.iter().copied()) {
        return Err(LodestarError::KernelInvalidParams {
            detail: format!(
                "{label} panel slots {:?} differ from contract {:?}",
                vectors.keys().map(|slot| slot.get()).collect::<Vec<_>>(),
                slots.iter().map(|slot| slot.get()).collect::<Vec<_>>()
            ),
        });
    }
    for &slot in slots {
        let vector = vectors.get(&slot).expect("validated slot key");
        let expected = dims[&slot];
        if expected == 0 || vector.len() != expected {
            return Err(LodestarError::KernelDimMismatch {
                expected,
                actual: vector.len(),
            });
        }
        if let Some(offset) = vector.iter().position(|value| !value.is_finite()) {
            return Err(LodestarError::KernelInvalidParams {
                detail: format!("{label} slot {slot} has non-finite value at offset {offset}"),
            });
        }
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        if !norm.is_finite() || norm == 0.0 {
            return Err(LodestarError::KernelInvalidParams {
                detail: format!("{label} slot {slot} has invalid squared norm {norm}"),
            });
        }
    }
    Ok(())
}

pub fn panel_kernel_recall_test(
    index: &PanelKernelIndex,
    full: &BTreeMap<CxId, PanelVectors>,
    params: &crate::RecallTestParams,
    corpus_name: &str,
) -> Result<crate::RecallReport> {
    validate_panel_recall_params(params, full)?;
    let held_out = panel_held_out(full, params.held_out_fraction, params.rng_seed);
    if held_out.is_empty() {
        return Err(LodestarError::RecallEmptyCorpus);
    }
    let mut total = 0.0_f32;
    for id in &held_out {
        let query = full.get(id).expect("held-out panel row");
        let mut full_hits = rank_panel_candidates(query, full, &index.slots, PANEL_RRF_K)?;
        full_hits.truncate(params.top_k);
        let kernel_hits = panel_kernel_search(index, query, params.top_k)?;
        let kernel_ids = kernel_hits
            .iter()
            .map(|hit| hit.cx_id)
            .collect::<BTreeSet<_>>();
        let overlap = full_hits
            .iter()
            .filter(|hit| kernel_ids.contains(&hit.cx_id))
            .count();
        total += overlap as f32 / full_hits.len() as f32;
    }
    let ratio = total / held_out.len() as f32;
    Ok(crate::RecallReport {
        kernel_only: ratio,
        full: 1.0,
        ratio,
        approx_factor: 1.0,
        tau_star_estimate: 0,
        tau_star_exact: true,
        recall_test_params: Some(params.clone()),
        corpus_name: Some(corpus_name.to_string()),
        n_queries_tested: held_out.len(),
        held_out,
        warning: (ratio < params.min_recall_ratio).then(|| {
            format!(
                "{}: ratio={ratio:.6} min={:.6}",
                crate::recall_test::CALYX_KERNEL_RECALL_BELOW_GATE,
                params.min_recall_ratio
            )
        }),
    })
}

pub fn panel_kernel_recall_gate(
    index: &PanelKernelIndex,
    full: &BTreeMap<CxId, PanelVectors>,
    params: &crate::RecallTestParams,
    corpus_name: &str,
) -> Result<crate::RecallReport> {
    let report = panel_kernel_recall_test(index, full, params, corpus_name)?;
    if report.ratio < params.min_recall_ratio {
        return Err(LodestarError::RecallBelowGate {
            ratio: report.ratio,
            min: params.min_recall_ratio,
        });
    }
    Ok(report)
}

pub fn panel_full_topk_support_set(
    slots: &[SlotId],
    full: &BTreeMap<CxId, PanelVectors>,
    params: &crate::RecallTestParams,
) -> Result<crate::RecallSupportReport> {
    panel_rank_stabilization_support_set(slots, full, params, 0)
}

/// Extracts exact fused hits plus per-lane rank prefixes for the deterministic
/// held-out recall set. RRF ranks are relative to the candidate population, so
/// retaining only the fused winners can reorder a reduced kernel even when all
/// full-corpus winners are present. The lane prefixes preserve the rank
/// witnesses needed to stabilize that ordering without collapsing the panel.
pub fn panel_rank_stabilization_support_set(
    slots: &[SlotId],
    full: &BTreeMap<CxId, PanelVectors>,
    params: &crate::RecallTestParams,
    lane_depth: usize,
) -> Result<crate::RecallSupportReport> {
    validate_panel_recall_params(params, full)?;
    let held_out = panel_held_out(full, params.held_out_fraction, params.rng_seed);
    if held_out.is_empty() {
        return Err(LodestarError::RecallEmptyCorpus);
    }
    let mut members = BTreeSet::new();
    let mut candidate_hits = 0;
    for id in &held_out {
        let query = full.get(id).expect("held-out panel row");
        let hits = rank_panel_candidates(query, full, slots, PANEL_RRF_K)?;
        let mut query_support = hits
            .iter()
            .take(params.top_k)
            .map(|hit| hit.cx_id)
            .collect::<BTreeSet<_>>();
        if lane_depth > 0 {
            query_support.extend(
                hits.iter()
                    .filter(|hit| hit.lanes.iter().any(|lane| lane.rank <= lane_depth))
                    .map(|hit| hit.cx_id),
            );
        }
        candidate_hits += query_support.len();
        members.extend(query_support);
    }
    Ok(crate::RecallSupportReport {
        members: members.into_iter().collect(),
        n_queries_tested: held_out.len(),
        held_out,
        candidate_hits,
    })
}

fn validate_panel_recall_params(
    params: &crate::RecallTestParams,
    full: &BTreeMap<CxId, PanelVectors>,
) -> Result<()> {
    if full.is_empty() {
        return Err(LodestarError::RecallEmptyCorpus);
    }
    if !params.held_out_fraction.is_finite()
        || !(0.0..=1.0).contains(&params.held_out_fraction)
        || params.top_k == 0
        || !params.min_recall_ratio.is_finite()
        || !(0.0..=1.0).contains(&params.min_recall_ratio)
    {
        return Err(LodestarError::RecallInvalidParams {
            detail: "invalid panel recall parameters".to_string(),
        });
    }
    Ok(())
}

fn panel_held_out(full: &BTreeMap<CxId, PanelVectors>, fraction: f32, seed: u64) -> Vec<CxId> {
    let target = ((full.len() as f32) * fraction).ceil() as usize;
    let mut keyed = full
        .keys()
        .enumerate()
        .map(|(ordinal, id)| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&seed.to_be_bytes());
            hasher.update(&(ordinal as u64).to_be_bytes());
            hasher.update(id.as_bytes());
            (*hasher.finalize().as_bytes(), *id)
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    keyed
        .into_iter()
        .take(target.min(full.len()))
        .map(|(_, id)| id)
        .collect()
}
