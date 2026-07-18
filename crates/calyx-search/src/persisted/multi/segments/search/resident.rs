use std::sync::Arc;

use calyx_sextant::index::MaxSimCudaChunk;

use super::*;
use crate::persisted::multi::pinned::{PinnedCandidateSelection, PinnedMultiIndex};

pub(super) fn flatten_query(query_tokens: &[Vec<f32>], token_dim: usize) -> CliResult<Vec<f32>> {
    let mut out = Vec::with_capacity(query_tokens.len() * token_dim);
    for token in query_tokens {
        if token.len() != token_dim {
            return Err(stale(format!(
                "persistent MaxSim query token len {} != token_dim {token_dim}",
                token.len()
            )));
        }
        out.extend_from_slice(token);
    }
    Ok(out)
}

pub(super) struct ResidentCandidateChunkStream {
    index: Arc<PinnedMultiIndex>,
    selection: PinnedCandidateSelection,
    rows_read: usize,
}

impl ResidentCandidateChunkStream {
    pub(super) fn new(index: Arc<PinnedMultiIndex>, candidates: &BTreeSet<CxId>) -> Self {
        let selection = index.select_candidates(candidates);
        Self {
            index,
            selection,
            rows_read: 0,
        }
    }

    pub(super) fn row_count(&self) -> usize {
        self.selection.row_count()
    }

    pub(super) fn token_count(&self) -> usize {
        self.selection.token_count()
    }

    pub(super) fn next_chunk(
        &mut self,
        expected_row_start: usize,
        max_rows: usize,
        max_tokens: usize,
    ) -> CliResult<Option<MaxSimCudaChunk>> {
        if expected_row_start != self.rows_read {
            return Err(stale(format!(
                "resident MaxSim CUDA stream requested row {expected_row_start}, but cursor is at {}",
                self.rows_read
            )));
        }
        let mut chunk = ChunkBuilder::new(max_rows);
        while chunk.row_count() < max_rows {
            let Some(row_index) = self.selection.row_index(self.rows_read) else {
                break;
            };
            let row = self
                .index
                .selected_row(row_index)
                .ok_or_else(|| stale("resident MaxSim candidate row index is out of bounds"))?;
            if row.norms.len() > max_tokens {
                return Err(stale(format!(
                    "resident MaxSim row {} has {} tokens, exceeding CUDA chunk token budget {max_tokens}; raise CALYX_SEARCH_MAXSIM_CUDA_CHUNK_TOKENS",
                    row.cx_id,
                    row.norms.len()
                )));
            }
            if chunk.row_count() > 0 && chunk.token_count() + row.norms.len() > max_tokens {
                break;
            }
            chunk.push(row)?;
            self.rows_read += 1;
        }
        Ok(chunk.finish())
    }
}

struct ChunkBuilder {
    row_offsets: Vec<u32>,
    tokens: Vec<f32>,
    token_norms: Vec<f32>,
    id_hi: Vec<u64>,
    id_lo: Vec<u64>,
}

impl ChunkBuilder {
    fn new(max_rows: usize) -> Self {
        Self {
            row_offsets: Vec::with_capacity(max_rows + 1),
            tokens: Vec::new(),
            token_norms: Vec::new(),
            id_hi: Vec::with_capacity(max_rows),
            id_lo: Vec::with_capacity(max_rows),
        }
    }

    fn row_count(&self) -> usize {
        self.id_hi.len()
    }

    fn token_count(&self) -> usize {
        self.token_norms.len()
    }

    fn push(&mut self, row: pinned::PinnedCandidateRow<'_>) -> CliResult {
        if self.row_offsets.is_empty() {
            self.row_offsets.push(0);
        }
        let (hi, lo) = cx_id_halves(row.cx_id);
        self.id_hi.push(hi);
        self.id_lo.push(lo);
        self.tokens.extend_from_slice(row.tokens);
        self.token_norms.extend_from_slice(row.norms);
        self.row_offsets.push(
            u32::try_from(self.token_norms.len())
                .map_err(|_| stale("resident MaxSim CUDA chunk token offsets exceed u32"))?,
        );
        Ok(())
    }

    fn finish(self) -> Option<MaxSimCudaChunk> {
        let row_count = self.id_hi.len();
        (row_count > 0).then(|| MaxSimCudaChunk {
            row_count,
            token_count: self.token_norms.len(),
            row_offsets: self.row_offsets,
            tokens: self.tokens,
            token_norms: self.token_norms,
            id_hi: self.id_hi,
            id_lo: self.id_lo,
            candidate_mask: vec![1; row_count],
        })
    }
}
