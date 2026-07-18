use calyx_core::{CalyxError, CxId, Slot, SlotVector};

use super::contextual_assay_error;
use crate::error::CliResult;

const SPARSE_HASH_DIM: usize = 2_048;

pub(super) fn assay_vector(
    slot: &Slot,
    cx_id: CxId,
    stored: &SlotVector,
) -> CliResult<(Vec<f32>, String)> {
    stored.validate_schema().map_err(|error| {
        contextual_assay_error(
            error,
            format!(
                "stored vector schema failed for active slot {} ({}) on cx {cx_id}",
                slot.slot_id,
                slot.slot_key.key()
            ),
        )
    })?;
    let (mut vector, representation) = match stored {
        SlotVector::Dense { dim, data } => (data.clone(), format!("dense_native_dim_{dim}_l2")),
        SlotVector::Sparse { dim, entries } => {
            let mut projected = vec![0.0f32; SPARSE_HASH_DIM];
            for entry in entries {
                let (bucket, sign) = sparse_hash(slot.slot_id.get(), entry.idx);
                projected[bucket] += sign * entry.val;
            }
            (
                projected,
                format!("sparse_signed_blake3_hash_dim_{dim}_to_{SPARSE_HASH_DIM}_l2"),
            )
        }
        SlotVector::Multi { token_dim, tokens } => {
            let mut pooled = vec![0.0f32; *token_dim as usize];
            for token in tokens {
                for (sum, value) in pooled.iter_mut().zip(token) {
                    *sum += *value;
                }
            }
            let divisor = tokens.len() as f32;
            for value in &mut pooled {
                *value /= divisor;
            }
            (pooled, format!("multi_mean_pool_token_dim_{token_dim}_l2"))
        }
        SlotVector::Absent { reason } => {
            return Err(CalyxError::assay_degenerate_input(format!(
                "active slot {} ({}) is explicitly absent on cx {cx_id}: {reason:?}",
                slot.slot_id,
                slot.slot_key.key()
            ))
            .into());
        }
    };
    normalize_nonzero(&mut vector, slot, cx_id)?;
    Ok((vector, representation))
}

fn sparse_hash(slot: u16, index: u32) -> (usize, f32) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"calyx-assay-sparse-feature-hash-v1");
    hasher.update(&slot.to_le_bytes());
    hasher.update(&index.to_le_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let bucket = u64::from_le_bytes(bytes[0..8].try_into().expect("eight hash bytes")) as usize
        % SPARSE_HASH_DIM;
    let sign = if bytes[8] & 1 == 0 { 1.0 } else { -1.0 };
    (bucket, sign)
}

fn normalize_nonzero(vector: &mut [f32], slot: &Slot, cx_id: CxId) -> CliResult {
    let norm_sq = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return Err(CalyxError::assay_degenerate_input(format!(
            "active slot {} ({}) produces a non-finite or zero-norm assay vector on cx {cx_id}: norm_sq={norm_sq}",
            slot.slot_id,
            slot.slot_key.key()
        ))
        .into());
    }
    let norm = norm_sq.sqrt() as f32;
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

pub(super) fn cosine_to_centroid(vectors: &[Vec<f32>], slot: &Slot) -> CliResult<Vec<f32>> {
    let dim = vectors.first().map_or(0, Vec::len);
    let mut centroid = vec![0.0f64; dim];
    for vector in vectors {
        for (sum, value) in centroid.iter_mut().zip(vector) {
            *sum += f64::from(*value);
        }
    }
    let centroid_norm = centroid
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !centroid_norm.is_finite() || centroid_norm <= 0.0 {
        return Err(CalyxError::assay_degenerate_input(format!(
            "pairwise redundancy centroid is zero or non-finite for slot {} ({})",
            slot.slot_id,
            slot.slot_key.key()
        ))
        .into());
    }
    let mut scores = Vec::with_capacity(vectors.len());
    for (row_index, vector) in vectors.iter().enumerate() {
        let dot = vector
            .iter()
            .zip(&centroid)
            .map(|(left, right)| f64::from(*left) * *right)
            .sum::<f64>();
        let vector_norm = vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        let score = dot / (vector_norm * centroid_norm);
        if !score.is_finite() {
            return Err(CalyxError::assay_degenerate_input(format!(
                "pairwise redundancy cosine is non-finite for slot {} ({}) at sample row {row_index}",
                slot.slot_id,
                slot.slot_key.key()
            ))
            .into());
        }
        scores.push(score as f32);
    }
    Ok(scores)
}
