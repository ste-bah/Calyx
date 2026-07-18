use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys};

use crate::cpu::{check_finite, check_shape_2d};
use crate::cuda::gemm::new_blas;
use crate::{CudaContext, ForgeError, Result};

const PROFILE_REMEDIATION: &str = "Check CUDA/cuBLAS profile pairwise inputs, VRAM, and driver state; fail closed instead of CPU fallback";
const DEVICE_REMEDIATION: &str = "Check CUDA, CUDA GPU availability, and free VRAM";
const DEFAULT_TILE_HEADROOM_BYTES: usize = 128 * 1024 * 1024;
const DEFAULT_MAX_TILE_ROWS: usize = 4096;
const TILE_ROWS_ENV: &str = "CALYX_PROFILE_CUDA_TILE_ROWS";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfilePairwiseCudaStats {
    pub rows: usize,
    pub dim: usize,
    pub tile_rows: usize,
    pub tiles: usize,
    pub resident_matrices: usize,
    pub resident_matrix_bytes: usize,
    pub max_tile_bytes: usize,
    pub pairwise_values: usize,
}

pub fn pairwise_euclidean_gram_tiled_host(
    ctx: &CudaContext,
    values: &[f32],
    rows: usize,
    dim: usize,
    out: &mut [f32],
) -> Result<ProfilePairwiseCudaStats> {
    validate_inputs(values, rows, dim, out)?;
    out.fill(0.0);
    if rows == 0 {
        return Ok(ProfilePairwiseCudaStats {
            rows,
            dim,
            tile_rows: 0,
            tiles: 0,
            resident_matrices: 0,
            resident_matrix_bytes: 0,
            max_tile_bytes: 0,
            pairwise_values: 0,
        });
    }

    let tile_rows = choose_tile_rows(ctx, rows, dim)?;
    let norms = squared_norms(values, rows, dim);
    let stream = ctx.inner().default_stream();
    let resident = stream
        .clone_htod(values)
        .map_err(|err| device_unavailable(ctx, format!("profile matrix H2D copy failed: {err}")))?;
    let blas = new_blas(ctx)?;
    let mut tiles = 0_usize;
    let mut max_tile_bytes = 0_usize;

    for start in (0..rows).step_by(tile_rows) {
        let end = (start + tile_rows).min(rows);
        let current_rows = end - start;
        let tile = &values[start * dim..end * dim];
        let tile_dev = stream.clone_htod(tile).map_err(|err| {
            device_unavailable(ctx, format!("profile tile H2D copy failed: {err}"))
        })?;
        let mut dot_dev = stream.alloc_zeros(current_rows * rows).map_err(|err| {
            device_unavailable(ctx, format!("profile dot tile allocation failed: {err}"))
        })?;

        gram_tile(
            ctx,
            blas.as_ref(),
            &tile_dev,
            &resident,
            current_rows,
            dim,
            rows,
            &mut dot_dev,
        )?;
        stream.synchronize().map_err(|err| {
            device_unavailable(ctx, format!("profile gram stream sync failed: {err}"))
        })?;
        let dot = stream.clone_dtoh(&dot_dev).map_err(|err| {
            device_unavailable(ctx, format!("profile dot tile D2H copy failed: {err}"))
        })?;
        check_finite(&dot, "cuda.profile_pairwise_gram")?;
        write_distance_tile(start, current_rows, rows, &norms, &dot, out)?;
        max_tile_bytes = max_tile_bytes.max(
            current_rows
                .saturating_mul(dim)
                .saturating_add(current_rows.saturating_mul(rows))
                .saturating_mul(std::mem::size_of::<f32>()),
        );
        tiles += 1;
    }

    Ok(ProfilePairwiseCudaStats {
        rows,
        dim,
        tile_rows,
        tiles,
        resident_matrices: 1,
        resident_matrix_bytes: values.len().saturating_mul(std::mem::size_of::<f32>()),
        max_tile_bytes,
        pairwise_values: out.len(),
    })
}

fn validate_inputs(values: &[f32], rows: usize, dim: usize, out: &[f32]) -> Result<()> {
    check_shape_2d(values, rows, dim, "cuda.profile matrix")?;
    check_shape_2d(out, rows, rows, "cuda.profile pairwise output")?;
    check_finite(values, "cuda.profile_pairwise")?;
    if rows > 0 && dim == 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![rows, 1],
            got: vec![rows, dim],
            remediation: "profile pairwise matrix requires non-zero vector dimension".to_string(),
        });
    }
    Ok(())
}

fn choose_tile_rows(ctx: &CudaContext, rows: usize, dim: usize) -> Result<usize> {
    let requested = env_tile_rows()?;
    if let Some(tile_rows) = requested {
        if tile_rows == 0 {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![1],
                got: vec![0],
                remediation: format!("{TILE_ROWS_ENV} must be greater than zero"),
            });
        }
        ensure_tile_fits(ctx, rows, dim, tile_rows)?;
        return Ok(tile_rows.min(rows).max(1));
    }

    let (free_bytes, _) = ctx
        .inner()
        .mem_get_info()
        .map_err(|err| device_unavailable(ctx, format!("profile VRAM query failed: {err}")))?;
    let matrix_bytes = rows
        .checked_mul(dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| profile_shape("profile resident matrix byte count overflows usize"))?;
    let available_for_tiles = free_bytes
        .saturating_sub(matrix_bytes)
        .saturating_sub(DEFAULT_TILE_HEADROOM_BYTES);
    let bytes_per_tile_row = rows
        .checked_add(dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| profile_shape("profile tile byte count overflows usize"))?;
    if bytes_per_tile_row == 0 || available_for_tiles < bytes_per_tile_row {
        return Err(ForgeError::DeviceUnavailable {
            device: device_label(ctx),
            detail: format!(
                "insufficient VRAM for profile pairwise tile rows={rows} dim={dim} free_bytes={free_bytes} resident_bytes={matrix_bytes} headroom_bytes={DEFAULT_TILE_HEADROOM_BYTES}"
            ),
            remediation: DEVICE_REMEDIATION.to_string(),
        });
    }
    Ok(rows
        .min(DEFAULT_MAX_TILE_ROWS)
        .min(available_for_tiles / bytes_per_tile_row)
        .max(1))
}

fn ensure_tile_fits(ctx: &CudaContext, rows: usize, dim: usize, tile_rows: usize) -> Result<()> {
    let (free_bytes, _) = ctx
        .inner()
        .mem_get_info()
        .map_err(|err| device_unavailable(ctx, format!("profile VRAM query failed: {err}")))?;
    let resident_bytes = rows
        .checked_mul(dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| profile_shape("profile resident matrix byte count overflows usize"))?;
    let tile_bytes = tile_rows
        .checked_mul(rows.saturating_add(dim))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| profile_shape("profile tile byte count overflows usize"))?;
    let required = resident_bytes
        .saturating_add(tile_bytes)
        .saturating_add(DEFAULT_TILE_HEADROOM_BYTES);
    if required > free_bytes {
        return Err(ForgeError::DeviceUnavailable {
            device: device_label(ctx),
            detail: format!(
                "{TILE_ROWS_ENV}={tile_rows} requires {required} bytes but CUDA reports {free_bytes} free bytes"
            ),
            remediation: DEVICE_REMEDIATION.to_string(),
        });
    }
    Ok(())
}

fn env_tile_rows() -> Result<Option<usize>> {
    match std::env::var(TILE_ROWS_ENV) {
        Ok(raw) => raw
            .parse::<usize>()
            .map(Some)
            .map_err(|err| ForgeError::ShapeMismatch {
                expected: vec![1],
                got: vec![0],
                remediation: format!("parse {TILE_ROWS_ENV}={raw}: {err}"),
            }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(ForgeError::DeviceUnavailable {
            device: "cuda:env".to_string(),
            detail: format!("read {TILE_ROWS_ENV}: {err}"),
            remediation: DEVICE_REMEDIATION.to_string(),
        }),
    }
}

fn squared_norms(values: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            values[row * dim..(row + 1) * dim]
                .iter()
                .map(|value| value * value)
                .sum()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gram_tile(
    ctx: &CudaContext,
    blas: &CudaBlas,
    tile: &cudarc::driver::CudaSlice<f32>,
    resident: &cudarc::driver::CudaSlice<f32>,
    tile_rows: usize,
    dim: usize,
    rows: usize,
    out: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_T,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: to_i32(tile_rows, "tile_rows")?,
        n: to_i32(rows, "rows")?,
        k: to_i32(dim, "dim")?,
        alpha: 1.0,
        lda: to_i32(dim, "lda")?,
        ldb: to_i32(dim, "ldb")?,
        beta: 0.0,
        ldc: to_i32(tile_rows, "ldc")?,
    };
    unsafe { blas.gemm(cfg, tile, resident, out) }.map_err(|err| ForgeError::NumericalInvariant {
        op: "profile_pairwise_gram".to_string(),
        detail: format!("cublasSgemm_v2 failed: {err} on cuda:{}", ctx.device_idx()),
        remediation: PROFILE_REMEDIATION.to_string(),
    })
}

fn write_distance_tile(
    start: usize,
    tile_rows: usize,
    rows: usize,
    norms: &[f32],
    dot_col_major: &[f32],
    out_row_major: &mut [f32],
) -> Result<()> {
    for col in 0..rows {
        for local_row in 0..tile_rows {
            let row = start + local_row;
            let dot = dot_col_major[col * tile_rows + local_row];
            let raw = norms[row] + norms[col] - 2.0 * dot;
            let sq = if raw >= 0.0 {
                raw
            } else if raw.abs() <= 1e-4 * norms[row].abs().max(norms[col].abs()).max(1.0) {
                0.0
            } else {
                return Err(ForgeError::NumericalInvariant {
                    op: "profile_pairwise_distance".to_string(),
                    detail: format!("negative squared distance row={row} col={col} value={raw}"),
                    remediation: PROFILE_REMEDIATION.to_string(),
                });
            };
            out_row_major[row * rows + col] = sq.sqrt();
        }
    }
    Ok(())
}

fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![i32::MAX as usize],
        got: vec![value],
        remediation: format!("cuda.profile {name} exceeds cuBLAS i32 dimension limit"),
    })
}

fn profile_shape(detail: impl Into<String>) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: Vec::new(),
        got: Vec::new(),
        remediation: detail.into(),
    }
}

fn device_unavailable(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: device_label(ctx),
        detail,
        remediation: DEVICE_REMEDIATION.to_string(),
    }
}

fn device_label(ctx: &CudaContext) -> String {
    format!("cuda:{}", ctx.device_idx())
}
