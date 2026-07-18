use cudarc::driver::{CudaSlice, PinnedHostSlice};

use super::launch::{self, GroupBuffers, ReduceBuffers};
use super::{
    CUDA_OLAP_PINNED_CHUNK_BYTES, CudaOlapAggregate, CudaOlapContext, CudaOlapGroup, CudaOlapScan,
    CudaOlapStats, GROUP_CAP_OP, GROUP_NONFINITE_OP, VALUE_NONFINITE_OP,
};
use crate::{CudaContext, ForgeError, Result};

const VALUE_NONFINITE: u32 = 1;
const GROUP_NONFINITE: u32 = 2;
const GROUP_CAP: u32 = 4;
const DICTIONARY_FULL: u32 = 8;
const PARTIAL_BYTES: usize = 8 + 8 + 4 + 4;
const GROUP_SLOT_BYTES: usize = 8 + 8 + 8 + 4 + 4;

impl CudaOlapContext {
    /// Scans little-endian materialized `f32` columns through bounded pinned staging.
    pub fn scan_columns_le(
        &self,
        value_column: &[u8],
        group_column: Option<&[u8]>,
        max_groups: usize,
    ) -> Result<CudaOlapScan> {
        let rows = validate_columns(value_column, group_column, max_groups)?;
        let chunk_rows = (CUDA_OLAP_PINNED_CHUNK_BYTES / size_of::<f32>()).min(rows);
        let stream = self.ctx.inner().default_stream();
        let mut status = stream.alloc_zeros::<u32>(1).map_err(|error| {
            launch::device(&self.ctx, format!("OLAP status allocation failed: {error}"))
        })?;
        let mut groups = group_column
            .map(|_| allocate_groups(&self.ctx, rows, max_groups))
            .transpose()?;
        if let Some(groups) = groups.as_mut() {
            launch::group_init(&self.ctx, groups)?;
        }

        let mut merged = AggregateParts::default();
        let mut chunks = 0_usize;
        let mut partial_readback = 0_usize;
        let mut start = 0_usize;
        while start < rows {
            let take = (rows - start).min(chunk_rows);
            let value_host = stage_column(&self.ctx, value_column, start, take)?;
            let value_device = stream.clone_htod(&value_host).map_err(|error| {
                launch::device(&self.ctx, format!("OLAP value upload failed: {error}"))
            })?;
            let blocks = launch::reduce_blocks(take);
            let mut partials = allocate_partials(&self.ctx, blocks)?;
            launch::reduce(&self.ctx, &value_device, take, &mut partials, &mut status)?;

            let mut group_host = None;
            let mut group_device = None;
            if let (Some(bytes), Some(groups)) = (group_column, groups.as_mut()) {
                let host = stage_column(&self.ctx, bytes, start, take)?;
                let device = stream.clone_htod(&host).map_err(|error| {
                    launch::device(&self.ctx, format!("OLAP group upload failed: {error}"))
                })?;
                launch::group_reduce(
                    &self.ctx,
                    &value_device,
                    &device,
                    take,
                    max_groups.min(rows),
                    groups,
                    &mut status,
                )?;
                group_host = Some(host);
                group_device = Some(device);
            }

            launch::synchronize(&self.ctx, "OLAP scan")?;
            merge_partials(&self.ctx, &partials, &mut merged)?;
            partial_readback += blocks * PARTIAL_BYTES;
            drop(group_device);
            drop(group_host);
            chunks += 1;
            start += take;
        }

        let status_bits = read_one(&self.ctx, &status, "OLAP status")?;
        check_status(status_bits, max_groups)?;
        let aggregate = merged.finish()?;
        if aggregate.count != rows as u64 {
            return Err(numerical(
                "olap.row_count",
                format!(
                    "CUDA count {} differs from input rows {rows}",
                    aggregate.count
                ),
            ));
        }
        let output_groups = groups
            .as_ref()
            .map(|groups| read_groups(&self.ctx, groups, aggregate.count))
            .transpose()?
            .unwrap_or_default();

        let columns = 1 + usize::from(group_column.is_some());
        let dictionary_capacity = groups.as_ref().map_or(0, |groups| groups.capacity);
        let group_readback = dictionary_capacity * GROUP_SLOT_BYTES;
        let peak_input_bytes = chunk_rows * size_of::<f32>() * columns;
        let peak_device_bytes = peak_input_bytes
            + launch::reduce_blocks(chunk_rows) * PARTIAL_BYTES
            + dictionary_capacity * GROUP_SLOT_BYTES
            + size_of::<u32>() * 2;
        Ok(CudaOlapScan {
            aggregate,
            groups: output_groups,
            stats: CudaOlapStats {
                rows: rows as u64,
                columns: columns as u64,
                chunks: chunks as u64,
                dictionary_capacity: dictionary_capacity as u64,
                kernel_launches: (chunks * columns + usize::from(groups.is_some())) as u64,
                host_to_device_bytes: (rows * size_of::<f32>() * columns) as u64,
                device_to_host_bytes: (partial_readback + group_readback + size_of::<u32>()) as u64,
                peak_pinned_staging_bytes: peak_input_bytes as u64,
                peak_device_bytes: peak_device_bytes as u64,
            },
        })
    }
}

#[derive(Default)]
struct AggregateParts {
    count: u64,
    sum: f64,
    min: u32,
    max: u32,
    initialized: bool,
}

impl AggregateParts {
    fn push(&mut self, count: u64, sum: f64, min: u32, max: u32) -> Result<()> {
        self.count = self.count.checked_add(count).ok_or_else(|| {
            numerical(
                "olap.count_overflow",
                "OLAP count overflowed u64".to_string(),
            )
        })?;
        self.sum += sum;
        if count > 0 {
            if self.initialized {
                self.min = self.min.min(min);
                self.max = self.max.max(max);
            } else {
                self.min = min;
                self.max = max;
                self.initialized = true;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<CudaOlapAggregate> {
        if !self.initialized || self.count == 0 {
            return Err(numerical(
                "olap.empty",
                "OLAP aggregate contains no finite rows".to_string(),
            ));
        }
        Ok(CudaOlapAggregate {
            count: self.count,
            sum: self.sum,
            min: ordered_to_float(self.min),
            max: ordered_to_float(self.max),
        })
    }
}

fn validate_columns(
    value_column: &[u8],
    group_column: Option<&[u8]>,
    max_groups: usize,
) -> Result<usize> {
    if value_column.is_empty() || !value_column.len().is_multiple_of(size_of::<f32>()) {
        return Err(shape(
            "OLAP value column must contain complete non-empty f32 values",
            value_column.len(),
        ));
    }
    if let Some(groups) = group_column {
        if groups.len() != value_column.len() {
            return Err(shape(
                "OLAP value and group columns must have equal byte lengths",
                groups.len(),
            ));
        }
        if max_groups == 0 {
            return Err(shape("OLAP max_groups must be greater than zero", 0));
        }
    }
    Ok(value_column.len() / size_of::<f32>())
}

fn allocate_partials(ctx: &CudaContext, blocks: usize) -> Result<ReduceBuffers> {
    let stream = ctx.inner().default_stream();
    Ok(ReduceBuffers {
        counts: stream.alloc_zeros(blocks).map_err(|error| {
            launch::device(ctx, format!("OLAP count allocation failed: {error}"))
        })?,
        sums: stream
            .alloc_zeros(blocks)
            .map_err(|error| launch::device(ctx, format!("OLAP sum allocation failed: {error}")))?,
        mins: stream
            .alloc_zeros(blocks)
            .map_err(|error| launch::device(ctx, format!("OLAP min allocation failed: {error}")))?,
        maxs: stream
            .alloc_zeros(blocks)
            .map_err(|error| launch::device(ctx, format!("OLAP max allocation failed: {error}")))?,
    })
}

fn allocate_groups(ctx: &CudaContext, rows: usize, max_groups: usize) -> Result<GroupBuffers> {
    let capacity = dictionary_capacity(rows, max_groups)?;
    let stream = ctx.inner().default_stream();
    Ok(GroupBuffers {
        slots: alloc(ctx, capacity, "dictionary slots")?,
        counts: alloc(ctx, capacity, "group counts")?,
        sums: alloc(ctx, capacity, "group sums")?,
        mins: alloc(ctx, capacity, "group mins")?,
        maxs: alloc(ctx, capacity, "group maxs")?,
        unique_count: stream.alloc_zeros(1).map_err(|error| {
            launch::device(ctx, format!("OLAP unique-count allocation failed: {error}"))
        })?,
        capacity,
    })
}

fn alloc<T>(ctx: &CudaContext, len: usize, name: &str) -> Result<CudaSlice<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|error| launch::device(ctx, format!("OLAP {name} allocation failed: {error}")))
}

fn dictionary_capacity(rows: usize, max_groups: usize) -> Result<usize> {
    let required = max_groups
        .min(rows)
        .checked_mul(2)
        .ok_or_else(|| shape("OLAP dictionary capacity overflowed usize", max_groups))?;
    required
        .max(2)
        .checked_next_power_of_two()
        .ok_or_else(|| shape("OLAP dictionary power-of-two overflow", required))
}

fn stage_column(
    ctx: &CudaContext,
    bytes: &[u8],
    start: usize,
    rows: usize,
) -> Result<PinnedHostSlice<f32>> {
    let start_byte = start
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| shape("OLAP staging offset overflow", start))?;
    let byte_len = rows
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| shape("OLAP staging length overflow", rows))?;
    let source = bytes
        .get(start_byte..start_byte + byte_len)
        .ok_or_else(|| shape("OLAP staging range exceeds column bytes", bytes.len()))?;
    let mut pinned = unsafe { ctx.inner().alloc_pinned::<f32>(rows) }
        .map_err(|error| launch::device(ctx, format!("OLAP pinned allocation failed: {error}")))?;
    let output = pinned
        .as_mut_slice()
        .map_err(|error| launch::device(ctx, format!("OLAP pinned access failed: {error}")))?;
    if cfg!(target_endian = "little") {
        let output_bytes =
            unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), byte_len) };
        output_bytes.copy_from_slice(source);
    } else {
        for (value, encoded) in output.iter_mut().zip(source.chunks_exact(4)) {
            *value = f32::from_le_bytes(encoded.try_into().expect("four f32 bytes"));
        }
    }
    Ok(pinned)
}

fn merge_partials(
    ctx: &CudaContext,
    partials: &ReduceBuffers,
    merged: &mut AggregateParts,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let counts = stream
        .clone_dtoh(&partials.counts)
        .map_err(|error| launch::device(ctx, format!("OLAP count readback failed: {error}")))?;
    let sums = stream
        .clone_dtoh(&partials.sums)
        .map_err(|error| launch::device(ctx, format!("OLAP sum readback failed: {error}")))?;
    let mins = stream
        .clone_dtoh(&partials.mins)
        .map_err(|error| launch::device(ctx, format!("OLAP min readback failed: {error}")))?;
    let maxs = stream
        .clone_dtoh(&partials.maxs)
        .map_err(|error| launch::device(ctx, format!("OLAP max readback failed: {error}")))?;
    for index in 0..counts.len() {
        merged.push(counts[index], sums[index], mins[index], maxs[index])?;
    }
    Ok(())
}

fn read_groups(
    ctx: &CudaContext,
    groups: &GroupBuffers,
    expected_rows: u64,
) -> Result<Vec<CudaOlapGroup>> {
    let slots = read(ctx, &groups.slots, "dictionary slots")?;
    let counts = read(ctx, &groups.counts, "group counts")?;
    let sums = read(ctx, &groups.sums, "group sums")?;
    let mins = read(ctx, &groups.mins, "group mins")?;
    let maxs = read(ctx, &groups.maxs, "group maxs")?;
    let mut output = Vec::new();
    output
        .try_reserve(slots.len())
        .map_err(|_| launch::device(ctx, "OLAP group output allocation failed".to_string()))?;
    for (index, encoded) in slots.into_iter().enumerate() {
        if encoded == 0 {
            continue;
        }
        output.push(CudaOlapGroup {
            key_bits: encoded as u32,
            aggregate: CudaOlapAggregate {
                count: counts[index],
                sum: sums[index],
                min: ordered_to_float(mins[index]),
                max: ordered_to_float(maxs[index]),
            },
        });
    }
    output.sort_unstable_by_key(|group| group.key_bits);
    let grouped_rows = output.iter().try_fold(0_u64, |total, group| {
        total
            .checked_add(group.aggregate.count)
            .ok_or_else(|| numerical("olap.group_count", "group count overflowed u64".to_string()))
    })?;
    if grouped_rows != expected_rows {
        return Err(numerical(
            "olap.group_count",
            format!("grouped rows {grouped_rows} differ from total rows {expected_rows}"),
        ));
    }
    Ok(output)
}

fn read<T>(ctx: &CudaContext, input: &CudaSlice<T>, label: &str) -> Result<Vec<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    ctx.inner()
        .default_stream()
        .clone_dtoh(input)
        .map_err(|error| launch::device(ctx, format!("OLAP {label} readback failed: {error}")))
}

fn read_one(ctx: &CudaContext, input: &CudaSlice<u32>, label: &str) -> Result<u32> {
    read(ctx, input, label)?
        .into_iter()
        .next()
        .ok_or_else(|| numerical("olap.readback", format!("{label} readback was empty")))
}

fn check_status(status: u32, max_groups: usize) -> Result<()> {
    if status & VALUE_NONFINITE != 0 {
        return Err(numerical(
            VALUE_NONFINITE_OP,
            "OLAP value column contains NaN or Inf".to_string(),
        ));
    }
    if status & GROUP_NONFINITE != 0 {
        return Err(numerical(
            GROUP_NONFINITE_OP,
            "OLAP group column contains NaN or Inf".to_string(),
        ));
    }
    if status & GROUP_CAP != 0 {
        return Err(numerical(
            GROUP_CAP_OP,
            format!("OLAP group cap {max_groups} exceeded"),
        ));
    }
    if status & DICTIONARY_FULL != 0 {
        return Err(numerical(
            "olap.dictionary_full",
            "OLAP CUDA dictionary exhausted its bounded capacity".to_string(),
        ));
    }
    Ok(())
}

fn ordered_to_float(ordered: u32) -> f32 {
    let mask = if ordered & 0x8000_0000 != 0 {
        0x8000_0000
    } else {
        0xffff_ffff
    };
    f32::from_bits(ordered ^ mask)
}

fn numerical(op: &'static str, detail: String) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail,
        remediation: "reject corrupt/non-finite OLAP columns or lower the bounded scan shape"
            .to_string(),
    }
}

fn shape(remediation: &str, got: usize) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![got],
        remediation: remediation.to_string(),
    }
}
