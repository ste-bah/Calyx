use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::{CudaTurboQuantBatch, MAX_ROTATION_WIDTH, checked_mul, device, level_code, shape};
use crate::cuda::quant::{CudaQuantContext, QuantCounters, launch};
use crate::quant::qjl::qjl_bits_len;
use crate::quant::turboquant::packed_len;
use crate::quant::{Quantizer, TurboQuantCodec};
use crate::{ForgeError, Result};

impl CudaQuantContext {
    pub fn encode_turboquant(
        &self,
        codec: &TurboQuantCodec,
        input: &[f32],
    ) -> Result<CudaTurboQuantBatch> {
        let dim = codec.dim();
        let (rows, rot_width) = validate_encode(codec, input)?;
        let level = codec.level();
        let level_code = level_code(level)?;
        let row_elements = checked_mul(rows, rot_width, "rotated rows")?;
        let scalar_len = packed_len(rot_width, level);
        let signs_len = qjl_bits_len(rot_width);
        let encoded_stride = scalar_len
            .checked_add(37)
            .and_then(|value| value.checked_add(signs_len))
            .ok_or_else(|| shape("TurboQuant encoded stride overflow"))?;
        let encoded_len = checked_mul(rows, encoded_stride, "encoded batch")?;
        let sign_elements = checked_mul(rows, signs_len, "QJL sign batch")?;
        let stream = self.context().inner().default_stream();
        let input_device = stream
            .clone_htod(input)
            .map_err(|error| device(self, format!("TurboQuant input upload failed: {error}")))?;
        let rotation_diagonal = stream
            .clone_htod(&codec.rotation().diagonal)
            .map_err(|error| device(self, format!("rotation diagonal upload failed: {error}")))?;
        let rademacher_diagonal = stream
            .clone_htod(&codec.rademacher().diagonal)
            .map_err(|error| device(self, format!("QJL diagonal upload failed: {error}")))?;
        let (threshold_values, centroid_values) = codec.cuda_codebook_tables();
        let thresholds = stream
            .clone_htod(&threshold_values)
            .map_err(|error| device(self, format!("Lloyd threshold upload failed: {error}")))?;
        let centroids = stream
            .clone_htod(&centroid_values)
            .map_err(|error| device(self, format!("Lloyd centroid upload failed: {error}")))?;
        let seed = stream
            .clone_htod(&codec.rademacher().id)
            .map_err(|error| device(self, format!("QJL seed upload failed: {error}")))?;
        let mut rotated = alloc_f32(self, row_elements, "rotated rows")?;
        let mut decoded = alloc_f32(self, row_elements, "decoded scalar rows")?;
        let mut residual = alloc_f32(self, row_elements, "residual rows")?;
        let mut qjl_rotated = alloc_f32(self, row_elements, "QJL rotated rows")?;
        let mut scales = alloc_f32(self, rows, "row scales")?;
        let mut residual_norms = alloc_f32(self, rows, "residual norms")?;
        let mut codes = alloc_u8(self, row_elements, "scalar codes")?;
        let mut signs = alloc_u8(self, sign_elements, "QJL signs")?;
        let mut encoded = alloc_u8(self, encoded_len, "encoded rows")?;
        let mut primary_bad = alloc_i32(self, rows, "primary status")?;
        let mut qjl_bad = alloc_i32(self, rows, "QJL status")?;

        launch::rotate_fwht(
            self.context(),
            &input_device,
            &rotation_diagonal,
            dim,
            rot_width,
            rows,
            &mut rotated,
            &mut primary_bad,
        )?;
        launch::quantize_rows(
            self.context(),
            &rotated,
            &thresholds,
            &centroids,
            rot_width,
            rows,
            level_code,
            &mut scales,
            &mut codes,
            &mut decoded,
            &mut primary_bad,
        )?;
        launch::pack_scalar(
            self.context(),
            &codes,
            rot_width,
            rows,
            level_code,
            encoded_stride,
            &mut encoded,
        )?;
        launch::residual_rows(
            self.context(),
            &rotated,
            &decoded,
            rot_width,
            rows,
            &mut residual,
            &mut residual_norms,
            &mut primary_bad,
        )?;
        launch::rotate_fwht(
            self.context(),
            &residual,
            &rademacher_diagonal,
            rot_width,
            rot_width,
            rows,
            &mut qjl_rotated,
            &mut qjl_bad,
        )?;
        launch::pack_qjl(
            self.context(),
            &qjl_rotated,
            &residual_norms,
            &seed,
            rot_width,
            rows,
            scalar_len,
            encoded_stride,
            &mut signs,
            &mut encoded,
        )?;
        let primary_status = stream
            .clone_dtoh(&primary_bad)
            .map_err(|error| device(self, format!("TurboQuant status readback failed: {error}")))?;
        let qjl_status = stream
            .clone_dtoh(&qjl_bad)
            .map_err(|error| device(self, format!("QJL status readback failed: {error}")))?;
        validate_status(&primary_status, &qjl_status)?;
        record_encode_stats(
            self.counters(),
            input.len(),
            rot_width,
            threshold_values.len(),
            centroid_values.len(),
            rows,
        );
        Ok(CudaTurboQuantBatch {
            quant: self.clone(),
            rows,
            dim,
            rot_width,
            level,
            seed_id: codec.seed().id,
            encoded_stride,
            encoded,
            codes,
            signs,
            scales,
            residual_norms,
            rotation_diagonal,
            centroids,
        })
    }
}

fn validate_encode(codec: &TurboQuantCodec, input: &[f32]) -> Result<(usize, usize)> {
    let dim = codec.dim();
    if dim == 0 || input.is_empty() || !input.len().is_multiple_of(dim) {
        return Err(shape(
            "CUDA TurboQuant input must contain one or more complete rows",
        ));
    }
    if let Some(index) = input.iter().position(|value| !value.is_finite()) {
        return Err(ForgeError::NumericalInvariant {
            op: "cuda_turboquant_encode".to_string(),
            detail: format!("non-finite input coefficient at index {index}"),
            remediation: "Reject NaN/Inf vectors before CUDA quantization".to_string(),
        });
    }
    let rot_width = codec.rotation_width();
    if !rot_width.is_power_of_two() || rot_width > MAX_ROTATION_WIDTH {
        return Err(shape(format!(
            "CUDA TurboQuant rotation width must be a power of two <= {MAX_ROTATION_WIDTH}"
        )));
    }
    Ok((input.len() / dim, rot_width))
}

pub(super) fn validate_status(primary: &[i32], qjl: &[i32]) -> Result<()> {
    if let Some((row, status)) = primary
        .iter()
        .chain(qjl.iter())
        .copied()
        .enumerate()
        .find(|(_, status)| *status != 0)
    {
        return Err(ForgeError::NumericalInvariant {
            op: "cuda_turboquant_encode".to_string(),
            detail: format!("device validation failed at status row {row}: flags={status}"),
            remediation: "Reject non-finite inputs or overflowing quantization state".to_string(),
        });
    }
    Ok(())
}

fn record_encode_stats(
    counters: Arc<QuantCounters>,
    input_elements: usize,
    rot_width: usize,
    thresholds: usize,
    centroids: usize,
    rows: usize,
) {
    let h2d_floats = input_elements
        .saturating_add(rot_width.saturating_mul(2))
        .saturating_add(thresholds)
        .saturating_add(centroids);
    counters.add_h2d(
        h2d_floats
            .saturating_mul(size_of::<f32>())
            .saturating_add(32),
    );
    counters.add_d2h(rows.saturating_mul(size_of::<i32>() * 2));
    counters.add_launches(6);
    counters.add_encoded_rows(rows);
}

pub(super) fn alloc_f32(
    quant: &CudaQuantContext,
    len: usize,
    label: &str,
) -> Result<CudaSlice<f32>> {
    quant
        .context()
        .inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|error| device(quant, format!("{label} allocation failed: {error}")))
}

pub(super) fn alloc_u8(quant: &CudaQuantContext, len: usize, label: &str) -> Result<CudaSlice<u8>> {
    quant
        .context()
        .inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|error| device(quant, format!("{label} allocation failed: {error}")))
}

pub(super) fn alloc_i32(
    quant: &CudaQuantContext,
    len: usize,
    label: &str,
) -> Result<CudaSlice<i32>> {
    quant
        .context()
        .inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|error| device(quant, format!("{label} allocation failed: {error}")))
}
