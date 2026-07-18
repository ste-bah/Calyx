use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::{alloc_f32, alloc_i32, alloc_u8, checked_mul, device, record_encode, shape};
use crate::Result;
use crate::cuda::quant::{CudaQuantContext, CudaQuantScores, packed_launch};
use crate::quant::{QuantLevel, QuantizedVec, Quantizer, ScalarInt8Codec};

const ZERO_SEED: [u8; 32] = [0; 32];

pub struct CudaInt8Batch {
    quant: CudaQuantContext,
    rows: usize,
    dim: usize,
    encoded: CudaSlice<u8>,
    scales: CudaSlice<f32>,
}

impl CudaInt8Batch {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn encoded_bytes_per_row(&self) -> usize {
        self.dim
    }

    pub fn read_encoded(&self) -> Result<Vec<QuantizedVec>> {
        let stream = self.quant.context().inner().default_stream();
        let bytes = stream
            .clone_dtoh(&self.encoded)
            .map_err(|error| device(&self.quant, format!("int8 readback failed: {error}")))?;
        let scales = stream
            .clone_dtoh(&self.scales)
            .map_err(|error| device(&self.quant, format!("int8 scale readback failed: {error}")))?;
        self.quant
            .counters()
            .add_d2h(bytes.len() + scales.len() * size_of::<f32>());
        Ok(bytes
            .chunks_exact(self.dim)
            .zip(scales)
            .map(|(row, scale)| QuantizedVec {
                level: QuantLevel::Bits8,
                dim: self.dim,
                bytes: row.to_vec(),
                scale,
                seed_id: ZERO_SEED,
            })
            .collect())
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let output_len = checked_mul(self.rows, self.dim, "int8 decoded output")?;
        let mut output = alloc_f32(&self.quant, output_len, "int8 decoded output")?;
        packed_launch::int8_decode(
            self.quant.context(),
            &self.encoded,
            &self.scales,
            self.dim,
            self.rows,
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
                device(&self.quant, format!("int8 decode readback failed: {error}"))
            })?;
        self.quant
            .counters()
            .add_d2h(decoded.len() * size_of::<f32>());
        Ok(decoded)
    }

    pub fn score(&self, query: &Self) -> Result<CudaQuantScores> {
        validate_pair(query, self)?;
        let mut scores = alloc_f32(&self.quant, self.rows, "int8 scores")?;
        packed_launch::int8_score(
            self.quant.context(),
            &query.encoded,
            &query.scales,
            &self.encoded,
            &self.scales,
            self.dim,
            self.rows,
            &mut scores,
        )?;
        let counters = self.quant.counters();
        counters.add_launches(1);
        counters.add_scored_candidates(self.rows);
        Ok(CudaQuantScores::new(self.quant.clone(), scores, self.rows))
    }
}

impl std::fmt::Debug for CudaInt8Batch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaInt8Batch")
            .field("rows", &self.rows)
            .field("dim", &self.dim)
            .finish()
    }
}

impl CudaQuantContext {
    pub fn encode_int8(&self, codec: &ScalarInt8Codec, input: &[f32]) -> Result<CudaInt8Batch> {
        let dim = codec.dim();
        let rows = super::validate_input(dim, input, "cuda_int8_encode")?;
        let encoded_len = checked_mul(rows, dim, "int8 encoded rows")?;
        let stream = self.context().inner().default_stream();
        let input_device = stream
            .clone_htod(input)
            .map_err(|error| device(self, format!("int8 input upload failed: {error}")))?;
        let mut encoded = alloc_u8(self, encoded_len, "int8 encoded rows")?;
        let mut scales = alloc_f32(self, rows, "int8 scales")?;
        let mut bad = alloc_i32(self, rows, "int8 status")?;
        packed_launch::int8_encode(
            self.context(),
            &input_device,
            dim,
            rows,
            &mut encoded,
            &mut scales,
            &mut bad,
        )?;
        let status = stream
            .clone_dtoh(&bad)
            .map_err(|error| device(self, format!("int8 status readback failed: {error}")))?;
        super::validate_status(&status, "cuda_int8_encode")?;
        record_encode(
            self.counters(),
            input.len().saturating_mul(size_of::<f32>()),
            rows,
            rows,
        );
        Ok(CudaInt8Batch {
            quant: self.clone(),
            rows,
            dim,
            encoded,
            scales,
        })
    }
}

fn validate_pair(query: &CudaInt8Batch, corpus: &CudaInt8Batch) -> Result<()> {
    if query.rows != 1 {
        return Err(shape("int8 scoring requires exactly one query row"));
    }
    if query.dim != corpus.dim {
        return Err(shape("int8 query and corpus must share dimension"));
    }
    if !Arc::ptr_eq(
        query.quant.context().inner(),
        corpus.quant.context().inner(),
    ) {
        return Err(shape("int8 query and corpus must share one CUDA context"));
    }
    Ok(())
}
