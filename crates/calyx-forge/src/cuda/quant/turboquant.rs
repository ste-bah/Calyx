mod encode;
mod ragged;

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::{CudaQuantContext, launch};
use crate::quant::{QuantLevel, QuantizedVec, SeedId};
use crate::{ForgeError, Result};

pub use ragged::CudaTurboQuantRow;

const MAX_ROTATION_WIDTH: usize = 4_096;

pub struct CudaTurboQuantBatch {
    quant: CudaQuantContext,
    rows: usize,
    dim: usize,
    rot_width: usize,
    level: QuantLevel,
    seed_id: SeedId,
    encoded_stride: usize,
    encoded: CudaSlice<u8>,
    codes: CudaSlice<u8>,
    signs: CudaSlice<u8>,
    scales: CudaSlice<f32>,
    residual_norms: CudaSlice<f32>,
    rotation_diagonal: CudaSlice<f32>,
    centroids: CudaSlice<f32>,
}

impl CudaTurboQuantBatch {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn rotation_width(&self) -> usize {
        self.rot_width
    }

    pub fn level(&self) -> QuantLevel {
        self.level
    }

    pub fn encoded_bytes_per_row(&self) -> usize {
        self.encoded_stride
    }

    pub fn read_encoded(&self) -> Result<Vec<QuantizedVec>> {
        let stream = self.quant.context().inner().default_stream();
        let encoded = stream.clone_dtoh(&self.encoded).map_err(|error| {
            device(
                &self.quant,
                format!("TurboQuant encoded readback failed: {error}"),
            )
        })?;
        let scales = stream.clone_dtoh(&self.scales).map_err(|error| {
            device(
                &self.quant,
                format!("TurboQuant scale readback failed: {error}"),
            )
        })?;
        self.quant
            .counters()
            .add_d2h(encoded.len() + scales.len() * size_of::<f32>());
        Ok(encoded
            .chunks_exact(self.encoded_stride)
            .zip(scales)
            .map(|(bytes, scale)| QuantizedVec {
                level: self.level,
                dim: self.dim,
                bytes: bytes.to_vec(),
                scale,
                seed_id: self.seed_id,
            })
            .collect())
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let output_len = checked_mul(self.rows, self.dim, "decoded output")?;
        let stream = self.quant.context().inner().default_stream();
        let mut output = stream.alloc_zeros(output_len).map_err(|error| {
            device(
                &self.quant,
                format!("TurboQuant decode allocation failed: {error}"),
            )
        })?;
        launch::decode(
            self.quant.context(),
            &self.codes,
            &self.scales,
            &self.centroids,
            &self.rotation_diagonal,
            self.dim,
            self.rot_width,
            self.rows,
            level_code(self.level)?,
            &mut output,
        )?;
        self.quant.counters().add_launches(1);
        let decoded = stream.clone_dtoh(&output).map_err(|error| {
            device(
                &self.quant,
                format!("TurboQuant decode readback failed: {error}"),
            )
        })?;
        self.quant
            .counters()
            .add_d2h(decoded.len() * size_of::<f32>());
        Ok(decoded)
    }

    pub fn score(&self, query: &Self) -> Result<CudaTurboQuantScores> {
        validate_score_pair(query, self)?;
        let stream = self.quant.context().inner().default_stream();
        let mut scores = stream.alloc_zeros(self.rows).map_err(|error| {
            device(
                &self.quant,
                format!("TurboQuant score allocation failed: {error}"),
            )
        })?;
        launch::score(
            self.quant.context(),
            &query.codes,
            &query.signs,
            &query.scales,
            &query.residual_norms,
            &self.codes,
            &self.signs,
            &self.scales,
            &self.residual_norms,
            &self.centroids,
            self.rot_width,
            self.rows,
            level_code(self.level)?,
            &mut scores,
        )?;
        let counters = self.quant.counters();
        counters.add_launches(1);
        counters.add_scored_candidates(self.rows);
        Ok(CudaTurboQuantScores::new(
            self.quant.clone(),
            scores,
            self.rows,
        ))
    }
}

impl std::fmt::Debug for CudaTurboQuantBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaTurboQuantBatch")
            .field("rows", &self.rows)
            .field("dim", &self.dim)
            .field("rot_width", &self.rot_width)
            .field("level", &self.level)
            .field("encoded_stride", &self.encoded_stride)
            .finish()
    }
}

type CudaTurboQuantScores = super::CudaQuantScores;

fn validate_score_pair(query: &CudaTurboQuantBatch, corpus: &CudaTurboQuantBatch) -> Result<()> {
    if query.rows != 1 {
        return Err(shape(
            "CUDA TurboQuant scoring requires exactly one query row",
        ));
    }
    if query.dim != corpus.dim
        || query.rot_width != corpus.rot_width
        || query.level != corpus.level
        || query.seed_id != corpus.seed_id
    {
        return Err(shape(
            "CUDA TurboQuant query and corpus must share dimension, level, and seed",
        ));
    }
    if !Arc::ptr_eq(
        query.quant.context().inner(),
        corpus.quant.context().inner(),
    ) {
        return Err(shape(
            "CUDA TurboQuant query and corpus must use the same CUDA context",
        ));
    }
    Ok(())
}

fn level_code(level: QuantLevel) -> Result<i32> {
    match level {
        QuantLevel::Bits3p5 => Ok(35),
        QuantLevel::Bits2p5 => Ok(25),
        _ => Err(shape("CUDA TurboQuant supports only Bits3p5 and Bits2p5")),
    }
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| shape(format!("CUDA TurboQuant {label} shape overflow")))
}

fn shape(remediation: impl Into<String>) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![1],
        got: vec![0],
        remediation: remediation.into(),
    }
}

fn device(quant: &CudaQuantContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", quant.context().device_idx()),
        detail,
        remediation: "Use the embedded sm_120 quant kernels with sufficient free VRAM".to_string(),
    }
}
