//! Deterministic random projection pre-step for high-dimensional Assay inputs.

use calyx_core::{CalyxError, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
const GPU_PROJECTION_OP: &str = "Assay random projection";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionReport {
    pub input_rows: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub projected: Vec<Vec<f32>>,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionTransferBytes {
    pub input_bytes: usize,
    pub sign_bytes: usize,
    pub output_bytes: usize,
    pub total_bytes: usize,
}

pub fn target_projection_dim(rows: usize, input_dim: usize) -> usize {
    if rows <= 1 {
        return input_dim.min(1);
    }
    let log2 = (rows as f64).log2().ceil() as usize;
    input_dim.min((2 * log2).max(1))
}

pub fn projection_transfer_bytes(
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<ProjectionTransferBytes> {
    let input_bytes = checked_bytes(rows, input_dim, "projection input")?;
    let sign_bytes = checked_bytes(input_dim, output_dim, "projection sign matrix")?;
    let output_bytes = checked_bytes(rows, output_dim, "projection output")?;
    let total_bytes = input_bytes
        .checked_add(sign_bytes)
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .ok_or_else(|| CalyxError::forge_vram_budget("projection transfer byte total overflow"))?;
    Ok(ProjectionTransferBytes {
        input_bytes,
        sign_bytes,
        output_bytes,
        total_bytes,
    })
}

pub fn project_cpu(matrix: &[Vec<f32>], seed: u64) -> ProjectionReport {
    let input_dim = matrix.first().map_or(0, Vec::len);
    let output_dim = target_projection_dim(matrix.len(), input_dim);
    let scale = if output_dim == 0 {
        1.0
    } else {
        (output_dim as f32).sqrt()
    };
    let projected = matrix
        .iter()
        .map(|row| {
            (0..output_dim)
                .map(|out_col| {
                    row.iter()
                        .enumerate()
                        .map(|(in_col, value)| value * sign(seed, in_col, out_col) / scale)
                        .sum()
                })
                .collect()
        })
        .collect();
    ProjectionReport {
        input_rows: matrix.len(),
        input_dim,
        output_dim,
        projected,
        seed,
    }
}

pub fn project_gpu(matrix: &[Vec<f32>], seed: u64) -> Result<ProjectionReport> {
    project_gpu_impl(matrix, seed)
}

#[cfg(feature = "cuda")]
fn project_gpu_impl(matrix: &[Vec<f32>], seed: u64) -> Result<ProjectionReport> {
    use calyx_forge::Backend;

    let (rows, input_dim) = validate_projection_matrix(matrix)?;
    let output_dim = target_projection_dim(rows, input_dim);
    if output_dim == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "Assay random projection output dimension is zero",
        ));
    }
    let _ = projection_transfer_bytes(rows, input_dim, output_dim)?;

    let input = flatten_projection_matrix(matrix, rows, input_dim)?;
    let signs = projection_sign_matrix(seed, input_dim, output_dim)?;
    let mut out = vec![0.0; checked_len(rows, output_dim, "projection output")?];
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx(GPU_PROJECTION_OP, err))?;
    backend
        .gemm(&input, &signs, rows, input_dim, output_dim, &mut out)
        .map_err(|err| crate::cuda_strict::forge_to_calyx(GPU_PROJECTION_OP, err))?;

    if let Some((idx, value)) = out
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CalyxError::forge_numerical_invariant(format!(
            "{GPU_PROJECTION_OP} produced non-finite output at flat index {idx}: {value}"
        )));
    }

    let projected = unflatten_projection_output(&out, rows, output_dim);
    Ok(ProjectionReport {
        input_rows: rows,
        input_dim,
        output_dim,
        projected,
        seed,
    })
}

#[cfg(not(feature = "cuda"))]
fn project_gpu_impl(_matrix: &[Vec<f32>], _seed: u64) -> Result<ProjectionReport> {
    Err(CalyxError::forge_device_unavailable(
        "Assay random projection requires calyx-assay feature `cuda` and a working Forge CUDA runtime; no CPU fallback is used",
    ))
}

#[cfg(feature = "cuda")]
fn validate_projection_matrix(matrix: &[Vec<f32>]) -> Result<(usize, usize)> {
    let rows = matrix.len();
    if rows == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "Assay random projection requires at least one row",
        ));
    }
    let input_dim = matrix[0].len();
    if input_dim == 0 {
        return Err(CalyxError::assay_insufficient_samples(
            "Assay random projection requires at least one input dimension",
        ));
    }
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != input_dim {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "Assay random projection ragged row {row_idx}: expected input_dim={input_dim}, got {}",
                row.len()
            )));
        }
        if let Some((col_idx, value)) = row
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "Assay random projection non-finite input at row {row_idx} col {col_idx}: {value}"
            )));
        }
    }
    Ok((rows, input_dim))
}

#[cfg(feature = "cuda")]
fn flatten_projection_matrix(
    matrix: &[Vec<f32>],
    rows: usize,
    input_dim: usize,
) -> Result<Vec<f32>> {
    let len = checked_len(rows, input_dim, "projection input")?;
    let mut flat = Vec::with_capacity(len);
    for col in 0..input_dim {
        for row in matrix {
            flat.push(row[col]);
        }
    }
    Ok(flat)
}

#[cfg(feature = "cuda")]
fn projection_sign_matrix(seed: u64, input_dim: usize, output_dim: usize) -> Result<Vec<f32>> {
    let len = checked_len(input_dim, output_dim, "projection sign matrix")?;
    let scale = (output_dim as f32).sqrt();
    let mut signs = Vec::with_capacity(len);
    for out_col in 0..output_dim {
        for in_col in 0..input_dim {
            signs.push(sign(seed, in_col, out_col) / scale);
        }
    }
    Ok(signs)
}

#[cfg(feature = "cuda")]
fn unflatten_projection_output(out: &[f32], rows: usize, output_dim: usize) -> Vec<Vec<f32>> {
    let mut projected = vec![vec![0.0; output_dim]; rows];
    for out_col in 0..output_dim {
        for row in 0..rows {
            projected[row][out_col] = out[out_col * rows + row];
        }
    }
    projected
}

fn checked_len(rows: usize, cols: usize, label: &str) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| CalyxError::forge_vram_budget(format!("{label} length overflow")))
}

fn checked_bytes(rows: usize, cols: usize, label: &str) -> Result<usize> {
    checked_len(rows, cols, label)?
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| CalyxError::forge_vram_budget(format!("{label} byte length overflow")))
}

fn sign(seed: u64, in_col: usize, out_col: usize) -> f32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_be_bytes());
    hasher.update(&(in_col as u64).to_be_bytes());
    hasher.update(&(out_col as u64).to_be_bytes());
    if hasher.finalize().as_bytes()[0] & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}
