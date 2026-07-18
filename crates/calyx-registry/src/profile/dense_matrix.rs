use std::collections::BTreeMap;
use std::env;

use calyx_core::{CalyxError, Result};

use super::{Observation, ProfileExecutionStats, ProfileMathBackend};

pub const CALYX_PROFILE_CUDA_MIN_ROWS_ENV: &str = "CALYX_PROFILE_CUDA_MIN_ROWS";
pub const CALYX_PROFILE_REQUIRE_CUDA_ENV: &str = "CALYX_PROFILE_REQUIRE_CUDA";
pub const DEFAULT_PROFILE_CUDA_MIN_ROWS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DenseObservationMatrix {
    values: Vec<f32>,
    labels: Vec<Option<String>>,
    rows: usize,
    dim: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairwiseDistanceMatrix {
    values: Vec<f32>,
    rows: usize,
    stats: ProfileExecutionStats,
}

impl DenseObservationMatrix {
    pub(crate) fn from_observations(observations: Vec<Observation>) -> Result<Self> {
        if observations.is_empty() {
            return Err(CalyxError::assay_insufficient_samples(
                "profile requires at least one measurable vector",
            ));
        }
        let dim = observations[0].data.len();
        validate_dim(dim)?;
        let rows = observations.len();
        let mut values = Vec::with_capacity(rows * dim);
        let mut labels = Vec::with_capacity(rows);
        for (idx, observation) in observations.into_iter().enumerate() {
            if observation.data.len() != dim {
                return Err(CalyxError::lens_dim_mismatch(format!(
                    "profile vector {idx} has dim {}, expected {dim}",
                    observation.data.len()
                )));
            }
            ensure_finite(&observation.data, idx)?;
            values.extend_from_slice(&observation.data);
            labels.push(observation.label);
        }
        Ok(Self {
            values,
            labels,
            rows,
            dim,
        })
    }

    pub(crate) fn from_vectors(
        vectors: Vec<Vec<f32>>,
        labels: Vec<Option<String>>,
    ) -> Result<Self> {
        let observations = vectors
            .into_iter()
            .enumerate()
            .map(|(idx, data)| Observation {
                data,
                label: labels.get(idx).cloned().unwrap_or(None),
            })
            .collect();
        Self::from_observations(observations)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    pub(crate) fn row(&self, idx: usize) -> &[f32] {
        &self.values[idx * self.dim..(idx + 1) * self.dim]
    }

    pub(crate) fn labels(&self) -> &[Option<String>] {
        &self.labels
    }

    pub(crate) fn pairwise_distances(&self) -> Result<PairwiseDistanceMatrix> {
        let mut stats = ProfileExecutionStats {
            measured_rows: self.rows,
            vector_dim: self.dim,
            ..ProfileExecutionStats::default()
        };
        let mut values = vec![0.0_f32; self.rows * self.rows];
        if use_cuda_for_rows(self.rows)? {
            pairwise_cuda(self, &mut values, &mut stats)?;
        } else {
            pairwise_cpu(self, &mut values);
            stats.pairwise_distance_backend = ProfileMathBackend::CpuFullMatrix;
            stats.pairwise_distance_matrices = 1;
            stats.pairwise_distance_values = values.len();
        }
        Ok(PairwiseDistanceMatrix {
            values,
            rows: self.rows,
            stats,
        })
    }

    pub(crate) fn label_groups(&self) -> BTreeMap<String, Vec<usize>> {
        let mut groups = BTreeMap::new();
        for (idx, label) in self.labels.iter().enumerate() {
            if let Some(label) = label {
                groups
                    .entry(label.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }
        groups
    }
}

impl PairwiseDistanceMatrix {
    pub(crate) fn get(&self, left: usize, right: usize) -> f32 {
        self.values[left * self.rows + right]
    }

    pub(crate) fn stats(&self) -> ProfileExecutionStats {
        self.stats
    }

    pub(crate) fn mean_upper_triangle(&self) -> f32 {
        if self.rows < 2 {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        let mut count = 0_usize;
        for left in 0..self.rows {
            for right in (left + 1)..self.rows {
                sum += self.get(left, right);
                count += 1;
            }
        }
        sum / count as f32
    }

    pub(crate) fn upper_triangle_signature(&self) -> Vec<f32> {
        let mut signature = Vec::with_capacity(self.rows.saturating_mul(self.rows) / 2);
        for left in 0..self.rows {
            for right in (left + 1)..self.rows {
                signature.push(self.get(left, right));
            }
        }
        signature
    }
}

fn validate_dim(dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(CalyxError::lens_dim_mismatch(
            "profile vectors must have non-zero dimension",
        ));
    }
    Ok(())
}

fn ensure_finite(values: &[f32], row: usize) -> Result<()> {
    if let Some((col, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(CalyxError::lens_numerical_invariant(format!(
            "profile vector row={row} col={col} contains non-finite value {value}"
        )));
    }
    Ok(())
}

fn pairwise_cpu(matrix: &DenseObservationMatrix, out: &mut [f32]) {
    for left in 0..matrix.rows {
        for right in left..matrix.rows {
            let distance = euclidean(matrix.row(left), matrix.row(right));
            out[left * matrix.rows + right] = distance;
            out[right * matrix.rows + left] = distance;
        }
    }
}

#[cfg(feature = "cuda")]
fn pairwise_cuda(
    matrix: &DenseObservationMatrix,
    out: &mut [f32],
    stats: &mut ProfileExecutionStats,
) -> Result<()> {
    let ctx = calyx_forge::init_cuda(0, false).map_err(map_forge_error)?;
    let cuda_stats = calyx_forge::pairwise_euclidean_gram_tiled_host(
        &ctx,
        &matrix.values,
        matrix.rows,
        matrix.dim,
        out,
    )
    .map_err(map_forge_error)?;
    stats.pairwise_distance_backend = ProfileMathBackend::CudaCublasTiledGram;
    stats.resident_matrices = cuda_stats.resident_matrices;
    stats.pairwise_distance_matrices = 1;
    stats.pairwise_distance_values = cuda_stats.pairwise_values;
    stats.pairwise_tile_rows = cuda_stats.tile_rows;
    stats.pairwise_tiles = cuda_stats.tiles;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn pairwise_cuda(
    _matrix: &DenseObservationMatrix,
    _out: &mut [f32],
    _stats: &mut ProfileExecutionStats,
) -> Result<()> {
    Err(CalyxError::forge_device_unavailable(
        "profile CUDA pairwise requested but calyx-registry was built without the cuda feature",
    ))
}

fn use_cuda_for_rows(rows: usize) -> Result<bool> {
    let min_rows = cuda_min_rows()?;
    if rows < min_rows {
        return Ok(false);
    }
    if cfg!(feature = "cuda") {
        return Ok(true);
    }
    if require_cuda()? {
        return Err(CalyxError::forge_device_unavailable(format!(
            "{CALYX_PROFILE_REQUIRE_CUDA_ENV}=true requires CUDA profile math for rows={rows} >= {min_rows}, but this binary lacks the cuda feature"
        )));
    }
    Ok(false)
}

fn cuda_min_rows() -> Result<usize> {
    match env::var(CALYX_PROFILE_CUDA_MIN_ROWS_ENV) {
        Ok(raw) => raw.parse::<usize>().map_err(|err| {
            CalyxError::forge_device_unavailable(format!(
                "parse {CALYX_PROFILE_CUDA_MIN_ROWS_ENV}={raw}: {err}"
            ))
        }),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PROFILE_CUDA_MIN_ROWS),
        Err(error) => Err(CalyxError::forge_device_unavailable(format!(
            "read {CALYX_PROFILE_CUDA_MIN_ROWS_ENV}: {error}"
        ))),
    }
}

fn require_cuda() -> Result<bool> {
    match env::var(CALYX_PROFILE_REQUIRE_CUDA_ENV) {
        Ok(raw) => match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(CalyxError::forge_device_unavailable(format!(
                "{CALYX_PROFILE_REQUIRE_CUDA_ENV} must be one of 1/0/true/false/yes/no/on/off, got {raw}"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(CalyxError::forge_device_unavailable(format!(
            "read {CALYX_PROFILE_REQUIRE_CUDA_ENV}: {error}"
        ))),
    }
}

fn euclidean(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = *a - *b;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(feature = "cuda")]
fn map_forge_error(error: calyx_forge::ForgeError) -> CalyxError {
    let message = format!("profile CUDA pairwise failed: {error}");
    match error.code() {
        "CALYX_FORGE_DEVICE_UNAVAILABLE" => CalyxError::forge_device_unavailable(message),
        "CALYX_FORGE_VRAM_BUDGET" => CalyxError::forge_vram_budget(message),
        "CALYX_FORGE_NUMERICAL_INVARIANT" | "CALYX_GPU_ERROR" => {
            CalyxError::forge_numerical_invariant(message)
        }
        "CALYX_FORGE_SHAPE_MISMATCH" => CalyxError::lens_dim_mismatch(message),
        _ => CalyxError::forge_numerical_invariant(message),
    }
}
