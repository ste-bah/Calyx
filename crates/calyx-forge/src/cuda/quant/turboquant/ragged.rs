use super::encode::{alloc_f32, alloc_i32, alloc_u8, validate_status};
use super::{MAX_ROTATION_WIDTH, checked_mul, device, level_code, shape};
use crate::cuda::quant::{CudaQuantContext, launch};
use crate::quant::qjl::qjl_bits_len;
use crate::quant::turboquant::packed_len;
use crate::quant::{QuantLevel, QuantizedVec, Quantizer, TurboQuantCodec};
use crate::{ForgeError, Result};

/// One independently seeded row in a mixed-shape CUDA TurboQuant batch.
#[derive(Clone, Copy, Debug)]
pub struct CudaTurboQuantRow<'a> {
    codec: &'a TurboQuantCodec,
    input: &'a [f32],
}

impl<'a> CudaTurboQuantRow<'a> {
    pub fn new(codec: &'a TurboQuantCodec, input: &'a [f32]) -> Self {
        Self { codec, input }
    }
}

#[derive(Debug)]
struct ShapeGroup {
    dim: usize,
    rot_width: usize,
    level: QuantLevel,
    indices: Vec<usize>,
}

impl CudaQuantContext {
    /// Encodes mixed dimensions and per-row seeds in one six-kernel sequence
    /// per distinct `(dimension, level)` group. Output order matches `rows`.
    pub fn encode_turboquant_ragged(
        &self,
        rows: &[CudaTurboQuantRow<'_>],
    ) -> Result<Vec<QuantizedVec>> {
        let groups = validate_and_group(rows)?;
        let mut output = host_vec(rows.len(), "ragged output rows")?;
        output.resize_with(rows.len(), || None);
        for group in groups {
            self.encode_group(rows, &group, &mut output)?;
        }
        output
            .into_iter()
            .enumerate()
            .map(|(row, value)| {
                value.ok_or_else(|| shape(format!("missing ragged output row {row}")))
            })
            .collect()
    }

    fn encode_group(
        &self,
        rows: &[CudaTurboQuantRow<'_>],
        group: &ShapeGroup,
        output: &mut [Option<QuantizedVec>],
    ) -> Result<()> {
        let row_count = group.indices.len();
        let input_len = checked_mul(row_count, group.dim, "ragged input")?;
        let diagonal_len = checked_mul(row_count, group.rot_width, "ragged diagonals")?;
        let seed_len = checked_mul(row_count, 32, "ragged seed IDs")?;
        let mut input = host_vec(input_len, "ragged input")?;
        let mut rotation = host_vec(diagonal_len, "ragged rotation diagonals")?;
        let mut rademacher = host_vec(diagonal_len, "ragged QJL diagonals")?;
        let mut seeds = host_vec(seed_len, "ragged seed IDs")?;
        for &index in &group.indices {
            let row = rows[index];
            input.extend_from_slice(row.input);
            rotation.extend_from_slice(&row.codec.rotation().diagonal);
            rademacher.extend_from_slice(&row.codec.rademacher().diagonal);
            seeds.extend_from_slice(&row.codec.rademacher().id);
        }

        let scalar_len = packed_len(group.rot_width, group.level);
        let signs_len = qjl_bits_len(group.rot_width);
        let encoded_stride = scalar_len
            .checked_add(37)
            .and_then(|value| value.checked_add(signs_len))
            .ok_or_else(|| shape("TurboQuant ragged encoded stride overflow"))?;
        let encoded_len = checked_mul(row_count, encoded_stride, "ragged encoded batch")?;
        let row_elements = checked_mul(row_count, group.rot_width, "ragged rotated rows")?;
        let sign_elements = checked_mul(row_count, signs_len, "ragged QJL signs")?;
        let exemplar = rows[group.indices[0]].codec;
        let (threshold_values, centroid_values) = exemplar.cuda_codebook_tables();
        let stream = self.context().inner().default_stream();
        let input_device = stream
            .clone_htod(&input)
            .map_err(|error| device(self, format!("ragged input upload failed: {error}")))?;
        let rotation_device = stream
            .clone_htod(&rotation)
            .map_err(|error| device(self, format!("ragged rotation upload failed: {error}")))?;
        let rademacher_device = stream
            .clone_htod(&rademacher)
            .map_err(|error| device(self, format!("ragged QJL diagonal upload failed: {error}")))?;
        let thresholds = stream
            .clone_htod(&threshold_values)
            .map_err(|error| device(self, format!("ragged threshold upload failed: {error}")))?;
        let centroids = stream
            .clone_htod(&centroid_values)
            .map_err(|error| device(self, format!("ragged centroid upload failed: {error}")))?;
        let seed_device = stream
            .clone_htod(&seeds)
            .map_err(|error| device(self, format!("ragged seed upload failed: {error}")))?;

        let mut rotated = alloc_f32(self, row_elements, "ragged rotated rows")?;
        let mut decoded = alloc_f32(self, row_elements, "ragged decoded rows")?;
        let mut residual = alloc_f32(self, row_elements, "ragged residual rows")?;
        let mut qjl_rotated = alloc_f32(self, row_elements, "ragged QJL rows")?;
        let mut scales = alloc_f32(self, row_count, "ragged scales")?;
        let mut residual_norms = alloc_f32(self, row_count, "ragged residual norms")?;
        let mut codes = alloc_u8(self, row_elements, "ragged scalar codes")?;
        let mut signs = alloc_u8(self, sign_elements, "ragged QJL signs")?;
        let mut encoded = alloc_u8(self, encoded_len, "ragged encoded rows")?;
        let mut primary_bad = alloc_i32(self, row_count, "ragged primary status")?;
        let mut qjl_bad = alloc_i32(self, row_count, "ragged QJL status")?;
        let level = level_code(group.level)?;

        launch::rotate_fwht_rows(
            self.context(),
            &input_device,
            &rotation_device,
            group.dim,
            group.rot_width,
            row_count,
            &mut rotated,
            &mut primary_bad,
        )?;
        launch::quantize_rows(
            self.context(),
            &rotated,
            &thresholds,
            &centroids,
            group.rot_width,
            row_count,
            level,
            &mut scales,
            &mut codes,
            &mut decoded,
            &mut primary_bad,
        )?;
        launch::pack_scalar(
            self.context(),
            &codes,
            group.rot_width,
            row_count,
            level,
            encoded_stride,
            &mut encoded,
        )?;
        launch::residual_rows(
            self.context(),
            &rotated,
            &decoded,
            group.rot_width,
            row_count,
            &mut residual,
            &mut residual_norms,
            &mut primary_bad,
        )?;
        launch::rotate_fwht_rows(
            self.context(),
            &residual,
            &rademacher_device,
            group.rot_width,
            group.rot_width,
            row_count,
            &mut qjl_rotated,
            &mut qjl_bad,
        )?;
        launch::pack_qjl_rows(
            self.context(),
            &qjl_rotated,
            &residual_norms,
            &seed_device,
            group.rot_width,
            row_count,
            scalar_len,
            encoded_stride,
            &mut signs,
            &mut encoded,
        )?;

        let primary_status = stream.clone_dtoh(&primary_bad).map_err(|error| {
            device(
                self,
                format!("ragged primary status readback failed: {error}"),
            )
        })?;
        let qjl_status = stream
            .clone_dtoh(&qjl_bad)
            .map_err(|error| device(self, format!("ragged QJL status readback failed: {error}")))?;
        validate_status(&primary_status, &qjl_status)?;
        let encoded_host = stream
            .clone_dtoh(&encoded)
            .map_err(|error| device(self, format!("ragged encoded readback failed: {error}")))?;
        let scale_host = stream
            .clone_dtoh(&scales)
            .map_err(|error| device(self, format!("ragged scale readback failed: {error}")))?;
        self.record_group_stats(
            input.len() + rotation.len() + rademacher.len(),
            threshold_values.len() + centroid_values.len(),
            seeds.len(),
            encoded_host.len(),
            row_count,
        );
        for (position, &source_index) in group.indices.iter().enumerate() {
            let start = position * encoded_stride;
            output[source_index] = Some(QuantizedVec {
                level: group.level,
                dim: group.dim,
                bytes: encoded_host[start..start + encoded_stride].to_vec(),
                scale: scale_host[position],
                seed_id: rows[source_index].codec.seed().id,
            });
        }
        Ok(())
    }

    fn record_group_stats(
        &self,
        row_floats: usize,
        table_floats: usize,
        seed_bytes: usize,
        encoded_bytes: usize,
        rows: usize,
    ) {
        let counters = self.counters();
        counters.add_h2d(
            row_floats
                .saturating_add(table_floats)
                .saturating_mul(size_of::<f32>())
                .saturating_add(seed_bytes),
        );
        counters.add_d2h(
            rows.saturating_mul(size_of::<i32>() * 2 + size_of::<f32>())
                .saturating_add(encoded_bytes),
        );
        counters.add_launches(6);
        counters.add_encoded_rows(rows);
    }
}

fn validate_and_group(rows: &[CudaTurboQuantRow<'_>]) -> Result<Vec<ShapeGroup>> {
    if rows.is_empty() {
        return Err(shape("CUDA TurboQuant ragged input must contain rows"));
    }
    let mut groups = Vec::<ShapeGroup>::new();
    for (index, row) in rows.iter().copied().enumerate() {
        let dim = row.codec.dim();
        if row.input.len() != dim {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![dim],
                got: vec![row.input.len()],
                remediation: format!("ragged TurboQuant row {index} must match its codec"),
            });
        }
        if let Some(offset) = row.input.iter().position(|value| !value.is_finite()) {
            return Err(ForgeError::NumericalInvariant {
                op: "cuda_turboquant_ragged_encode".to_string(),
                detail: format!("non-finite coefficient at row {index} offset {offset}"),
                remediation: "Reject the complete microbatch before CUDA submission".to_string(),
            });
        }
        let rot_width = row.codec.rotation_width();
        if !rot_width.is_power_of_two() || rot_width > MAX_ROTATION_WIDTH {
            return Err(shape(format!(
                "ragged row {index} rotation width must be a power of two <= {MAX_ROTATION_WIDTH}"
            )));
        }
        let level = row.codec.level();
        level_code(level)?;
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.dim == dim && group.level == level)
        {
            group.indices.push(index);
        } else {
            groups.push(ShapeGroup {
                dim,
                rot_width,
                level,
                indices: vec![index],
            });
        }
    }
    Ok(groups)
}

fn host_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|error| ForgeError::VramBudget {
            detail: format!("{label} host allocation failed: {error}"),
            remediation: "Reduce the streaming microbatch size before CUDA submission".to_string(),
        })?;
    Ok(output)
}
