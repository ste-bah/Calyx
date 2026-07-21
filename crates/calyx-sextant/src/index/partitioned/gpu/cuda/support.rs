use std::sync::Arc;

use calyx_core::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice, ValidAsZeroBits};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{CALYX_INDEX_INVALID_PARAMS, CALYX_INDEX_IO, sextant_error};
use crate::index::SpannCentroidIndex;

use super::super::super::VectorSource;

pub(super) fn fill_range(
    source: &dyn VectorSource,
    start: usize,
    take: usize,
    dim: usize,
    host: &mut PinnedHostSlice<f32>,
) -> Result<()> {
    let destination = host
        .as_mut_slice()
        .map_err(cuda_error("corpus pinned access"))?;
    destination[..take * dim]
        .par_chunks_exact_mut(dim)
        .enumerate()
        .try_for_each(|(offset, row)| {
            let row_id = (start + offset) as u64;
            source.row_into(row_id, row);
            ensure_finite(row, row_id)
        })
}

pub(super) fn flatten_rows_seeded(
    rows: &[(u32, Vec<f32>)],
    seed: u64,
    host: &mut PinnedHostSlice<f32>,
) -> Result<()> {
    let dim = rows[0].1.len();
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let destination = host
        .as_mut_slice()
        .map_err(cuda_error("sample pinned access"))?;
    for (target, row_index) in destination.chunks_exact_mut(dim).zip(order) {
        target.copy_from_slice(&rows[row_index].1);
    }
    Ok(())
}

pub(super) fn validate_rows(rows: &[(u32, Vec<f32>)], dim: usize) -> Result<()> {
    for (id, row) in rows {
        if row.len() != dim {
            return Err(invalid(format!(
                "row {id} dim {} expected {dim}",
                row.len()
            )));
        }
        ensure_finite(row, u64::from(*id))?;
    }
    Ok(())
}

pub(super) fn ensure_finite(row: &[f32], row_id: u64) -> Result<()> {
    if row.iter().any(|value| !value.is_finite()) {
        Err(invalid(format!("row {row_id} has a non-finite component")))
    } else {
        Ok(())
    }
}

pub(super) fn validate_probe(centroids: &SpannCentroidIndex, probe: usize) -> Result<usize> {
    if centroids.centroid_count() == 0 || probe == 0 {
        return Err(invalid("routing requires centroids and probe > 0"));
    }
    Ok(probe.min(centroids.centroid_count()))
}

pub(super) fn corpus_bytes_if(corpus: &Option<CudaSlice<f32>>) -> usize {
    corpus
        .as_ref()
        .map_or(0, |values| values.len() * size_of::<f32>())
}

pub(super) fn byte_len<T>(values: usize) -> Result<usize> {
    values
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid("byte size overflow"))
}

pub(super) fn mib_bytes(name: &'static str, default: usize, allow_zero: bool) -> Result<usize> {
    let mib = match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| invalid(format!("{name} must be an integer MiB value")))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(invalid(format!("cannot read {name}: {error}"))),
    };
    if (!allow_zero && mib == 0) || mib > 65_536 {
        return Err(invalid(format!("{name}={mib} outside the supported range")));
    }
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| invalid(format!("{name} byte size overflow")))
}

pub(super) fn pinned_zeros<T>(
    context: &Arc<CudaContext>,
    len: usize,
    stage: &'static str,
) -> Result<PinnedHostSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    let mut pinned = unsafe { context.alloc_pinned::<T>(len) }.map_err(cuda_error(stage))?;
    let pointer = pinned.as_mut_ptr().map_err(cuda_error(stage))?;
    unsafe { pointer.write_bytes(0, len) };
    Ok(pinned)
}

pub(super) fn alloc_device<T>(
    stream: &Arc<CudaStream>,
    len: usize,
    stage: &'static str,
) -> Result<CudaSlice<T>>
where
    T: cudarc::driver::DeviceRepr + ValidAsZeroBits,
{
    stream.alloc_zeros(len).map_err(cuda_error(stage))
}

pub(super) fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid("counter exceeds u64"))
}

pub(super) fn invalid(detail: impl std::fmt::Display) -> calyx_core::CalyxError {
    sextant_error(
        CALYX_INDEX_INVALID_PARAMS,
        format!("partition CUDA: {detail}"),
    )
}

pub(super) fn cuda_error(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> calyx_core::CalyxError {
    move |error| sextant_error(CALYX_INDEX_IO, format!("partition CUDA {stage}: {error}"))
}
