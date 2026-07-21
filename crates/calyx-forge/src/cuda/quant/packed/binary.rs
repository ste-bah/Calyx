use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::{alloc_f32, alloc_i32, alloc_u8, checked_mul, device, record_encode, shape};
use crate::Result;
use crate::cuda::quant::{CudaQuantContext, CudaQuantScores, packed_launch};
use crate::quant::{BinaryCodec, QuantLevel, QuantizedVec, Quantizer, SeedId};

pub struct CudaBinaryBatch {
    quant: CudaQuantContext,
    rows: usize,
    dim: usize,
    seed_id: SeedId,
    stride: usize,
    encoded: CudaSlice<u8>,
    diagonal: CudaSlice<f32>,
}

impl CudaBinaryBatch {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn encoded_bytes_per_row(&self) -> usize {
        self.stride
    }

    pub fn read_encoded(&self) -> Result<Vec<QuantizedVec>> {
        let bytes = self
            .quant
            .context()
            .inner()
            .default_stream()
            .clone_dtoh(&self.encoded)
            .map_err(|error| device(&self.quant, format!("binary readback failed: {error}")))?;
        self.quant.counters().add_d2h(bytes.len());
        let scale = binary_amplitude(self.dim);
        Ok(bytes
            .chunks_exact(self.stride)
            .map(|row| QuantizedVec {
                level: QuantLevel::Bits1,
                dim: self.dim,
                bytes: row.to_vec(),
                scale,
                seed_id: self.seed_id,
            })
            .collect())
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let output_len = checked_mul(self.rows, self.dim, "binary decoded output")?;
        let mut output = alloc_f32(&self.quant, output_len, "binary decoded output")?;
        packed_launch::binary_decode(
            self.quant.context(),
            &self.encoded,
            &self.diagonal,
            self.dim,
            self.rows,
            binary_amplitude(self.dim),
            &mut output,
        )?;
        self.quant.counters().add_launches(1);
        let decoded = self
            .quant
            .context()
            .inner()
            .default_stream()
            .clone_dtoh(&output)
            .map_err(|error| {
                device(
                    &self.quant,
                    format!("binary decode readback failed: {error}"),
                )
            })?;
        self.quant
            .counters()
            .add_d2h(decoded.len() * size_of::<f32>());
        Ok(decoded)
    }

    pub fn score(&self, query: &Self) -> Result<CudaBinaryScores> {
        validate_pair(query, self)?;
        let mut mismatches = alloc_i32(&self.quant, self.rows, "binary mismatches")?;
        let mut scores = alloc_f32(&self.quant, self.rows, "binary scores")?;
        packed_launch::binary_score(
            self.quant.context(),
            &query.encoded,
            &self.encoded,
            self.dim,
            self.rows,
            &mut mismatches,
            &mut scores,
        )?;
        let counters = self.quant.counters();
        counters.add_launches(1);
        counters.add_scored_candidates(self.rows);
        Ok(CudaBinaryScores {
            scores: CudaQuantScores::new(self.quant.clone(), scores, self.rows),
            mismatches,
        })
    }
}

impl std::fmt::Debug for CudaBinaryBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaBinaryBatch")
            .field("rows", &self.rows)
            .field("dim", &self.dim)
            .field("stride", &self.stride)
            .finish()
    }
}

pub struct CudaBinaryScores {
    scores: CudaQuantScores,
    mismatches: CudaSlice<i32>,
}

impl CudaBinaryScores {
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    pub fn read(&self) -> Result<Vec<f32>> {
        self.scores.read()
    }

    pub fn read_mismatch_counts(&self) -> Result<Vec<usize>> {
        let values = self
            .scores
            .quant
            .context()
            .inner()
            .default_stream()
            .clone_dtoh(&self.mismatches)
            .map_err(|error| {
                device(
                    &self.scores.quant,
                    format!("binary mismatch readback failed: {error}"),
                )
            })?;
        self.scores
            .quant
            .counters()
            .add_d2h(values.len() * size_of::<i32>());
        values
            .into_iter()
            .map(|value| {
                usize::try_from(value).map_err(|_| shape("negative binary mismatch count"))
            })
            .collect()
    }

    pub fn topk(&self, k: usize) -> Result<Vec<(usize, f32)>> {
        self.scores.topk(k)
    }
}

impl CudaQuantContext {
    pub fn encode_binary(&self, codec: &BinaryCodec, input: &[f32]) -> Result<CudaBinaryBatch> {
        let dim = codec.dim();
        let rows = super::validate_input(dim, input, "cuda_binary_encode")?;
        let stride = dim.div_ceil(8);
        let encoded_len = checked_mul(rows, stride, "binary encoded rows")?;
        let stream = self.context().inner().default_stream();
        let input_device = stream
            .clone_htod(input)
            .map_err(|error| device(self, format!("binary input upload failed: {error}")))?;
        let diagonal = stream
            .clone_htod(&codec.seed().diagonal)
            .map_err(|error| device(self, format!("binary diagonal upload failed: {error}")))?;
        let mut encoded = alloc_u8(self, encoded_len, "binary encoded rows")?;
        let mut bad = alloc_i32(self, rows, "binary status")?;
        packed_launch::binary_encode(
            self.context(),
            &input_device,
            &diagonal,
            dim,
            rows,
            &mut encoded,
            &mut bad,
        )?;
        let status = stream
            .clone_dtoh(&bad)
            .map_err(|error| device(self, format!("binary status readback failed: {error}")))?;
        super::validate_status(&status, "cuda_binary_encode")?;
        record_encode(
            self.counters(),
            input
                .len()
                .saturating_add(dim)
                .saturating_mul(size_of::<f32>()),
            rows,
            rows,
        );
        Ok(CudaBinaryBatch {
            quant: self.clone(),
            rows,
            dim,
            seed_id: codec.seed().id,
            stride,
            encoded,
            diagonal,
        })
    }
}

fn validate_pair(query: &CudaBinaryBatch, corpus: &CudaBinaryBatch) -> Result<()> {
    if query.rows != 1 {
        return Err(shape("binary scoring requires exactly one query row"));
    }
    if query.dim != corpus.dim || query.seed_id != corpus.seed_id {
        return Err(shape(
            "binary query and corpus must share dimension and seed",
        ));
    }
    if !Arc::ptr_eq(
        query.quant.context().inner(),
        corpus.quant.context().inner(),
    ) {
        return Err(shape("binary query and corpus must share one CUDA context"));
    }
    Ok(())
}

fn binary_amplitude(dim: usize) -> f32 {
    1.0 / (dim as f32).sqrt()
}
