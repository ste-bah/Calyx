use calyx_core::Result;

use super::{DiskAnnPqBuildParams, corrupt, invalid};
use crate::error::{CALYX_INDEX_DIM_MISMATCH, sextant_error};

pub(super) fn rows(rows: &[(u32, Vec<f32>)], params: DiskAnnPqBuildParams) -> Result<()> {
    if rows.is_empty() {
        return Err(invalid("pq requires at least one row"));
    }
    let dim = rows[0].1.len();
    if dim == 0 {
        return Err(invalid("pq dim must be positive"));
    }
    if params.subvectors == 0 || params.centroids == 0 || params.iterations == 0 {
        return Err(invalid(
            "pq subvectors, centroids, and iterations must be positive",
        ));
    }
    if params.centroids > 256 {
        return Err(invalid("pq centroids must be <= 256 for u8 codes"));
    }
    if params.subvectors > dim || !dim.is_multiple_of(params.subvectors) {
        return Err(invalid(format!(
            "pq dim {dim} must be divisible by subvectors {}",
            params.subvectors
        )));
    }
    rows.len()
        .checked_mul(dim)
        .and_then(|cells| cells.checked_mul(size_of::<f32>()))
        .ok_or_else(|| invalid("pq corpus shape overflows address space"))?;
    for (idx, (id, vector)) in rows.iter().enumerate() {
        if *id as usize != idx {
            return Err(invalid(format!("pq row id {id} expected dense id {idx}")));
        }
        if vector.len() != dim {
            return Err(sextant_error(
                CALYX_INDEX_DIM_MISMATCH,
                format!("pq vector {id} dim {} expected {dim}", vector.len()),
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!("pq vector {id} contains non-finite value")));
        }
    }
    let centroids = params.centroids.min(rows.len());
    if centroids > 1 && rows[1..].iter().all(|(_, row)| same_bits(&rows[0].1, row)) {
        return Err(invalid(
            "pq training corpus is degenerate: all rows are bit-identical",
        ));
    }
    Ok(())
}

pub(super) fn header(
    dim: usize,
    node_count: usize,
    subvectors: usize,
    centroids: usize,
    subdim: usize,
    iterations: usize,
) -> Result<()> {
    if dim == 0
        || node_count == 0
        || subvectors == 0
        || centroids == 0
        || subdim == 0
        || iterations == 0
    {
        return Err(corrupt("pq header contains zero field"));
    }
    if centroids > 256 {
        return Err(corrupt("pq centroids exceed u8 code space"));
    }
    if subvectors.checked_mul(subdim) != Some(dim) {
        return Err(corrupt(format!(
            "pq header subvectors {subvectors} * subdim {subdim} != dim {dim}"
        )));
    }
    Ok(())
}

fn same_bits(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left.to_bits() == right.to_bits())
}
