use cudarc::driver::{CudaSlice, PinnedHostSlice};
use rayon::prelude::*;

use super::launch;
use super::{CUDA_OLAP_PINNED_CHUNK_BYTES, CudaOlapContext, CudaOlapStats};
use crate::{CudaContext, ForgeError, Result};

#[derive(Debug)]
pub(super) struct TransposeBuffers {
    input_host: PinnedHostSlice<f32>,
    input_device: CudaSlice<f32>,
    output_device: CudaSlice<f32>,
}

impl TransposeBuffers {
    fn new(ctx: &CudaContext, values: usize) -> Result<Self> {
        let stream = ctx.inner().default_stream();
        Ok(Self {
            input_host: pinned_zeros(ctx, values, "transpose input")?,
            input_device: stream.alloc_zeros(values).map_err(|error| {
                launch::device(
                    ctx,
                    format!("OLAP transpose device input allocation failed: {error}"),
                )
            })?,
            output_device: stream.alloc_zeros(values).map_err(|error| {
                launch::device(
                    ctx,
                    format!("OLAP transpose device output allocation failed: {error}"),
                )
            })?,
        })
    }
}

impl CudaOlapContext {
    /// Transposes ragged host rows into column-major values with 32x32 CUDA tiles.
    pub fn transpose_rows(&self, rows: &[&[f32]]) -> Result<(Vec<f32>, CudaOlapStats)> {
        let (row_count, columns, values) = validate_rows(rows)?;
        let max_tile_values = CUDA_OLAP_PINNED_CHUNK_BYTES / size_of::<f32>();
        let tile_columns = columns.min(max_tile_values);
        let mut output = Vec::new();
        output.try_reserve_exact(values).map_err(|_| {
            launch::device(
                &self.ctx,
                format!("OLAP transpose output allocation failed for {values} values"),
            )
        })?;
        output.resize(values, 0.0);

        let stream = self.ctx.inner().default_stream();
        let mut cache = self.transpose_buffers.lock().map_err(|_| {
            launch::device(
                &self.ctx,
                "OLAP transpose staging lock is poisoned".to_string(),
            )
        })?;
        if cache.is_none() {
            *cache = Some(TransposeBuffers::new(&self.ctx, max_tile_values)?);
        }
        let buffers = cache.as_mut().expect("transpose buffers initialized");
        let mut launches = 0_usize;
        let mut peak_tile_values = 0_usize;
        for column_start in (0..columns).step_by(tile_columns) {
            let take_columns = (columns - column_start).min(tile_columns);
            let chunk_rows = (max_tile_values / take_columns).max(1).min(row_count);
            for row_start in (0..row_count).step_by(chunk_rows) {
                let take_rows = (row_count - row_start).min(chunk_rows);
                let tile_values = take_rows
                    .checked_mul(take_columns)
                    .ok_or_else(|| shape("OLAP transpose tile shape overflow", take_rows))?;
                stage_rows(
                    &self.ctx,
                    rows,
                    row_start,
                    take_rows,
                    column_start,
                    take_columns,
                    &mut buffers.input_host,
                )?;
                let input_host = buffers.input_host.as_slice().map_err(|error| {
                    launch::device(
                        &self.ctx,
                        format!("OLAP transpose pinned input access failed: {error}"),
                    )
                })?;
                stream
                    .memcpy_htod(&input_host[..tile_values], &mut buffers.input_device)
                    .map_err(|error| {
                        launch::device(&self.ctx, format!("OLAP transpose upload failed: {error}"))
                    })?;
                launch::transpose(
                    &self.ctx,
                    &buffers.input_device,
                    &mut buffers.output_device,
                    take_rows,
                    take_columns,
                )?;
                for local_column in 0..take_columns {
                    let source_start = local_column * take_rows;
                    let destination_start = (column_start + local_column) * row_count + row_start;
                    let source = buffers
                        .output_device
                        .slice(source_start..source_start + take_rows);
                    stream
                        .memcpy_dtoh(
                            &source,
                            &mut output[destination_start..destination_start + take_rows],
                        )
                        .map_err(|error| {
                            launch::device(
                                &self.ctx,
                                format!("OLAP transpose readback failed: {error}"),
                            )
                        })?;
                }
                launch::synchronize(&self.ctx, "OLAP transpose")?;
                peak_tile_values = peak_tile_values.max(tile_values);
                launches += 1;
            }
        }

        let transfer_bytes = values
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| shape("OLAP transpose transfer byte count overflow", values))?;
        Ok((
            output,
            CudaOlapStats {
                rows: row_count as u64,
                columns: columns as u64,
                chunks: launches as u64,
                dictionary_capacity: 0,
                kernel_launches: launches as u64,
                host_to_device_bytes: transfer_bytes as u64,
                device_to_host_bytes: transfer_bytes as u64,
                peak_pinned_staging_bytes: (peak_tile_values * size_of::<f32>()) as u64,
                peak_device_bytes: (peak_tile_values * size_of::<f32>() * 2) as u64,
            },
        ))
    }
}

fn validate_rows(rows: &[&[f32]]) -> Result<(usize, usize, usize)> {
    let columns = rows
        .first()
        .ok_or_else(|| shape("OLAP transpose requires at least one row", 0))?
        .len();
    if columns == 0 {
        return Err(shape("OLAP transpose requires non-empty rows", 0));
    }
    if rows.iter().any(|row| row.len() != columns) {
        return Err(shape(
            "OLAP transpose rows must have one fixed dimension",
            columns,
        ));
    }
    u32::try_from(rows.len())
        .map_err(|_| shape("OLAP transpose row count exceeds u32", rows.len()))?;
    u32::try_from(columns).map_err(|_| shape("OLAP transpose dimension exceeds u32", columns))?;
    let values = rows
        .len()
        .checked_mul(columns)
        .ok_or_else(|| shape("OLAP transpose value count overflow", rows.len()))?;
    Ok((rows.len(), columns, values))
}

fn stage_rows(
    ctx: &CudaContext,
    rows: &[&[f32]],
    row_start: usize,
    take_rows: usize,
    column_start: usize,
    take_columns: usize,
    pinned: &mut PinnedHostSlice<f32>,
) -> Result<()> {
    let values = take_rows
        .checked_mul(take_columns)
        .ok_or_else(|| shape("OLAP transpose staging overflow", take_rows))?;
    let output = pinned.as_mut_slice().map_err(|error| {
        launch::device(
            ctx,
            format!("OLAP transpose pinned input access failed: {error}"),
        )
    })?;
    const BLOCK_ROWS: usize = 4096;
    let block_values = BLOCK_ROWS * take_columns;
    output[..values]
        .par_chunks_mut(block_values)
        .zip(rows[row_start..row_start + take_rows].par_chunks(BLOCK_ROWS))
        .for_each(|(destination_block, source_block)| {
            for (destination, source) in destination_block
                .chunks_exact_mut(take_columns)
                .zip(source_block)
            {
                destination.copy_from_slice(&source[column_start..column_start + take_columns]);
            }
        });
    Ok(())
}

fn pinned_zeros(ctx: &CudaContext, len: usize, label: &str) -> Result<PinnedHostSlice<f32>> {
    let mut pinned = unsafe { ctx.inner().alloc_pinned::<f32>(len) }
        .map_err(|error| launch::device(ctx, format!("OLAP {label} allocation failed: {error}")))?;
    let pointer = pinned
        .as_mut_ptr()
        .map_err(|error| launch::device(ctx, format!("OLAP {label} access failed: {error}")))?;
    unsafe { pointer.write_bytes(0, len) };
    Ok(pinned)
}

fn shape(remediation: &str, got: usize) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![got],
        remediation: remediation.to_string(),
    }
}
