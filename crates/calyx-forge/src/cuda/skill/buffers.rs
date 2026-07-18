use cudarc::driver::CudaSlice;

use super::{
    CUDA_SKILL_MAX_DEVICE_BYTES, CUDA_SKILL_MAX_POINTS, CUDA_SKILL_VRAM_RESERVE_BYTES,
    CudaSkillEdge, CudaSkillMst, CudaSkillSlot, CudaSkillStats,
};
use crate::{CudaContext, ForgeError, Result};

const NONFINITE: u32 = 1 << 0;
const ZERO_NORM: u32 = 1 << 1;
const NO_OVERLAP: u32 = 1 << 2;
const PRIM_INVARIANT: u32 = 1 << 3;

pub(super) struct SkillHost {
    values: Vec<f32>,
    offsets: Vec<i64>,
    dims: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SkillShape {
    pub points: usize,
    pub slots: usize,
    pub feature_values: usize,
    pub min_samples: usize,
    pub sort_length: usize,
    pub peak_device_bytes: usize,
    pub host_to_device_bytes: usize,
}

pub(super) struct SkillBuffers {
    pub values: CudaSlice<f32>,
    pub offsets: CudaSlice<i64>,
    pub dims: CudaSlice<i32>,
    pub distances: CudaSlice<f64>,
    pub core: CudaSlice<f64>,
    pub edge_sources: CudaSlice<u32>,
    pub edge_destinations: CudaSlice<u32>,
    pub edge_weights: CudaSlice<f64>,
    pub status: CudaSlice<u32>,
}

pub(super) fn validate_and_flatten(
    ctx: &CudaContext,
    point_count: usize,
    slots: &[CudaSkillSlot],
    min_samples: usize,
) -> Result<(SkillShape, SkillHost)> {
    if !(2..=CUDA_SKILL_MAX_POINTS).contains(&point_count) || slots.is_empty() || min_samples == 0 {
        return Err(shape(
            format!(
                "CUDA skill clustering requires 2..={CUDA_SKILL_MAX_POINTS} points, at least one slot, and min_samples >= 1"
            ),
            vec![point_count, slots.len(), min_samples],
        ));
    }
    let offset_count = point_count.checked_mul(slots.len()).ok_or_else(|| {
        shape(
            "CUDA skill offset shape overflowed usize",
            vec![point_count, slots.len()],
        )
    })?;
    let mut offsets = vec![-1_i64; offset_count];
    let mut values = Vec::new();
    let mut dims = Vec::with_capacity(slots.len());
    for (slot_index, slot) in slots.iter().enumerate() {
        if slot.dim == 0
            || slot.point_indices.is_empty()
            || slot.values.len() != slot.point_indices.len().saturating_mul(slot.dim)
        {
            return Err(shape(
                format!("CUDA skill slot {slot_index} has an invalid dense shape"),
                vec![slot.dim, slot.point_indices.len(), slot.values.len()],
            ));
        }
        dims.push(i32::try_from(slot.dim).map_err(|_| {
            shape(
                format!("CUDA skill slot {slot_index} dimension exceeds i32"),
                vec![slot.dim],
            )
        })?);
        let mut seen = vec![false; point_count];
        for (local_row, point) in slot.point_indices.iter().copied().enumerate() {
            let point = point as usize;
            if point >= point_count || std::mem::replace(&mut seen[point], true) {
                return Err(shape(
                    format!(
                        "CUDA skill slot {slot_index} point indexes must be unique and in range"
                    ),
                    vec![point],
                ));
            }
            let base = values.len();
            offsets[point * slots.len() + slot_index] = i64::try_from(base)
                .map_err(|_| shape("CUDA skill feature offset exceeds i64", vec![base]))?;
            let start = local_row * slot.dim;
            values.extend_from_slice(&slot.values[start..start + slot.dim]);
        }
    }
    let distance_values = point_count.checked_mul(point_count).ok_or_else(|| {
        shape(
            "CUDA skill distance shape overflowed usize",
            vec![point_count],
        )
    })?;
    let edge_count = point_count - 1;
    let peak_device_bytes = checked_byte_sum(&[
        byte_count::<f32>(values.len())?,
        byte_count::<i64>(offsets.len())?,
        byte_count::<i32>(dims.len())?,
        byte_count::<f64>(distance_values)?,
        byte_count::<f64>(point_count)?,
        byte_count::<u32>(edge_count * 2 + 1)?,
        byte_count::<f64>(edge_count)?,
    ])?;
    admit_vram(ctx, peak_device_bytes)?;
    let host_to_device_bytes = byte_count::<f32>(values.len())?
        + byte_count::<i64>(offsets.len())?
        + byte_count::<i32>(dims.len())?;
    Ok((
        SkillShape {
            points: point_count,
            slots: slots.len(),
            feature_values: values.len(),
            min_samples,
            sort_length: point_count.next_power_of_two(),
            peak_device_bytes,
            host_to_device_bytes,
        },
        SkillHost {
            values,
            offsets,
            dims,
        },
    ))
}

impl SkillBuffers {
    pub(super) fn allocate(
        ctx: &CudaContext,
        shape: &SkillShape,
        host: &SkillHost,
    ) -> Result<Self> {
        let stream = ctx.inner().default_stream();
        let upload = |label: &str, error: cudarc::driver::DriverError| {
            device(ctx, format!("CUDA skill {label} upload failed: {error}"))
        };
        Ok(Self {
            values: stream
                .clone_htod(&host.values)
                .map_err(|error| upload("values", error))?,
            offsets: stream
                .clone_htod(&host.offsets)
                .map_err(|error| upload("offsets", error))?,
            dims: stream
                .clone_htod(&host.dims)
                .map_err(|error| upload("dimensions", error))?,
            distances: alloc(ctx, shape.points * shape.points, "distance matrix")?,
            core: alloc(ctx, shape.points, "core distances")?,
            edge_sources: alloc(ctx, shape.points - 1, "edge sources")?,
            edge_destinations: alloc(ctx, shape.points - 1, "edge destinations")?,
            edge_weights: alloc(ctx, shape.points - 1, "edge weights")?,
            status: alloc(ctx, 1, "status")?,
        })
    }
}

pub(super) fn read_result(
    ctx: &CudaContext,
    shape: SkillShape,
    buffers: SkillBuffers,
    read_distances: bool,
) -> Result<CudaSkillMst> {
    let status = read(ctx, &buffers.status, "status")?[0];
    check_status(status)?;
    let sources = read(ctx, &buffers.edge_sources, "edge sources")?;
    let destinations = read(ctx, &buffers.edge_destinations, "edge destinations")?;
    let weights = read(ctx, &buffers.edge_weights, "edge weights")?;
    let mut edges = sources
        .into_iter()
        .zip(destinations)
        .zip(weights)
        .map(|((source, destination), weight)| CudaSkillEdge {
            source: source as usize,
            destination: destination as usize,
            weight,
        })
        .collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        left.weight
            .total_cmp(&right.weight)
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| left.destination.cmp(&right.destination))
    });
    let distances = read_distances
        .then(|| read(ctx, &buffers.distances, "distance matrix"))
        .transpose()?;
    let device_to_host_bytes = size_of::<u32>()
        + (shape.points - 1) * (size_of::<u32>() * 2 + size_of::<f64>())
        + distances
            .as_ref()
            .map_or(0, |values| byte_count::<f64>(values.len()).unwrap_or(0));
    Ok(CudaSkillMst {
        edges,
        distances,
        stats: CudaSkillStats {
            points: shape.points as u64,
            slots: shape.slots as u64,
            feature_values: shape.feature_values as u64,
            pairwise_values: (shape.points * shape.points) as u64,
            kernel_launches: 3,
            host_to_device_bytes: shape.host_to_device_bytes as u64,
            device_to_host_bytes: device_to_host_bytes as u64,
            peak_device_bytes: shape.peak_device_bytes as u64,
            full_distance_readback: read_distances,
        },
    })
}

fn admit_vram(ctx: &CudaContext, required: usize) -> Result<()> {
    let free = ctx.free_device_vram_bytes()?;
    if required <= CUDA_SKILL_MAX_DEVICE_BYTES
        && required <= free.saturating_sub(CUDA_SKILL_VRAM_RESERVE_BYTES)
    {
        return Ok(());
    }
    Err(device(
        ctx,
        format!(
            "CUDA skill clustering exceeds bounded admission; required_bytes={required} free_bytes={free} reserve_bytes={CUDA_SKILL_VRAM_RESERVE_BYTES} max_bytes={CUDA_SKILL_MAX_DEVICE_BYTES}"
        ),
    ))
}

fn check_status(status: u32) -> Result<()> {
    let (op, detail) = if status & NONFINITE != 0 {
        ("skill.vector_finite", "skill vector contains NaN or Inf")
    } else if status & ZERO_NORM != 0 {
        ("skill.vector_norm", "shared skill vector has zero norm")
    } else if status & NO_OVERLAP != 0 {
        (
            "skill.pair_no_overlap",
            "a point pair has no shared dense slot",
        )
    } else if status & PRIM_INVARIANT != 0 {
        (
            "skill.prim_invariant",
            "deterministic Prim traversal became disconnected",
        )
    } else {
        return Ok(());
    };
    Err(ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail: detail.to_string(),
        remediation: "reject invalid skill vectors or repair the strict CUDA clustering provider"
            .to_string(),
    })
}

fn alloc<T>(ctx: &CudaContext, len: usize, label: &str) -> Result<CudaSlice<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    ctx.inner()
        .default_stream()
        .alloc_zeros(len)
        .map_err(|error| {
            device(
                ctx,
                format!("CUDA skill {label} allocation failed: {error}"),
            )
        })
}

fn read<T>(ctx: &CudaContext, input: &CudaSlice<T>, label: &str) -> Result<Vec<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    ctx.inner()
        .default_stream()
        .clone_dtoh(input)
        .map_err(|error| device(ctx, format!("CUDA skill {label} readback failed: {error}")))
}

fn byte_count<T>(len: usize) -> Result<usize> {
    len.checked_mul(size_of::<T>())
        .ok_or_else(|| shape("CUDA skill byte count overflowed usize", vec![len]))
}

fn checked_byte_sum(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(0_usize, |total, part| total.checked_add(*part))
        .ok_or_else(|| {
            shape(
                "CUDA skill device footprint overflowed usize",
                parts.to_vec(),
            )
        })
}

pub(super) fn device(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation:
            "free VRAM or repair the strict CUDA skill provider; large-N CPU fallback is disabled"
                .to_string(),
    }
}

fn shape(remediation: impl Into<String>, got: Vec<usize>) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![CUDA_SKILL_MAX_POINTS],
        got,
        remediation: remediation.into(),
    }
}
