use cudarc::driver::{CudaSlice, PinnedHostSlice};

use super::{
    CUDA_ENERGY_MAX_DEVICE_BYTES, CUDA_ENERGY_PINNED_CHUNK_BYTES, CUDA_ENERGY_VRAM_RESERVE_BYTES,
    CudaEnergyDescent, CudaEnergyStats,
};
use crate::{CudaContext, ForgeError, Result};

const FLOAT_BYTES: usize = size_of::<f32>();
const CONTROL_WORDS: usize = 3;
const MEMBER_NONFINITE: u32 = 1 << 0;
const MEMBER_ZERO_NORM: u32 = 1 << 1;
const QUERY_NONFINITE: u32 = 1 << 2;
const QUERY_ZERO_NORM: u32 = 1 << 3;
const CENTROID_NONFINITE: u32 = 1 << 4;
const CENTROID_ZERO_NORM: u32 = 1 << 5;
const STATE_NONFINITE: u32 = 1 << 6;

#[derive(Clone, Copy, Debug)]
pub(super) struct EnergyShape {
    pub dim: usize,
    pub members: usize,
    pub elements: usize,
    pub padded_elements: usize,
    pub centroid_tiles: usize,
    pub peak_device_bytes: usize,
    pub pinned_elements: usize,
}

pub(super) struct EnergyBuffers {
    pub members: CudaSlice<f32>,
    pub member_inverse_norms: CudaSlice<f32>,
    pub query: CudaSlice<f32>,
    pub query_inverse_norm: CudaSlice<f32>,
    pub scores: CudaSlice<f32>,
    pub weights: CudaSlice<f32>,
    pub centroid_partials: CudaSlice<f32>,
    pub metadata: CudaSlice<f32>,
    pub control: CudaSlice<u32>,
    pub status: CudaSlice<u32>,
    pub host_to_device_bytes: usize,
}

pub(super) fn validate_and_admit(
    ctx: &CudaContext,
    initial: &[f32],
    members: &[&[f32]],
    beta: f32,
    max_steps: usize,
    eps: f32,
) -> Result<EnergyShape> {
    if initial.is_empty() || members.is_empty() {
        return Err(shape_error(
            "CUDA energy descent requires a non-empty vector and region",
            vec![initial.len(), members.len()],
        ));
    }
    if !beta.is_finite() || beta < 0.0 || !eps.is_finite() || eps < 0.0 {
        return Err(numerical(
            "energy.parameters",
            format!("beta={beta} and eps={eps} must be finite and non-negative"),
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(numerical(
            "energy.query_finite",
            "initial query contains NaN or Inf".to_string(),
        ));
    }
    let dim = initial.len();
    for (index, member) in members.iter().enumerate() {
        if member.len() != dim {
            return Err(shape_error(
                &format!("CUDA energy member {index} does not match query dimension {dim}"),
                vec![member.len()],
            ));
        }
    }
    ensure_i32(dim, "dimension")?;
    ensure_i32(members.len(), "member count")?;
    ensure_i32(max_steps, "step count")?;
    let elements = members.len().checked_mul(dim).ok_or_else(|| {
        shape_error(
            "CUDA energy member shape overflowed usize",
            vec![members.len(), dim],
        )
    })?;
    let pinned_elements = elements.min(CUDA_ENERGY_PINNED_CHUNK_BYTES / FLOAT_BYTES);
    let padded_elements = elements
        .div_ceil(pinned_elements)
        .checked_mul(pinned_elements)
        .ok_or_else(|| shape_error("CUDA energy padded shape overflowed usize", vec![elements]))?;
    let centroid_tiles = members.len().div_ceil(256);
    let partial_elements = centroid_tiles.checked_mul(dim).ok_or_else(|| {
        shape_error(
            "CUDA energy centroid scratch overflowed usize",
            vec![centroid_tiles, dim],
        )
    })?;
    let float_elements = padded_elements
        .checked_add(members.len().checked_mul(3).ok_or_else(|| {
            shape_error(
                "CUDA energy member scratch overflowed usize",
                vec![members.len()],
            )
        })?)
        .and_then(|value| value.checked_add(dim))
        .and_then(|value| value.checked_add(partial_elements))
        .and_then(|value| value.checked_add(2))
        .ok_or_else(|| {
            shape_error(
                "CUDA energy device footprint overflowed usize",
                vec![elements],
            )
        })?;
    let peak_device_bytes = float_elements
        .checked_mul(FLOAT_BYTES)
        .and_then(|value| value.checked_add((CONTROL_WORDS + 1) * size_of::<u32>()))
        .ok_or_else(|| {
            shape_error(
                "CUDA energy device bytes overflowed usize",
                vec![float_elements],
            )
        })?;
    admit_vram(ctx, peak_device_bytes)?;
    Ok(EnergyShape {
        dim,
        members: members.len(),
        elements,
        padded_elements,
        centroid_tiles,
        peak_device_bytes,
        pinned_elements,
    })
}

impl EnergyBuffers {
    pub(super) fn allocate(
        ctx: &CudaContext,
        initial: &[f32],
        members: &[&[f32]],
        shape: &EnergyShape,
    ) -> Result<Self> {
        let stream = ctx.inner().default_stream();
        let mut member_device = alloc(ctx, shape.padded_elements, "resident member matrix")?;
        upload_members(ctx, members, shape, &mut member_device)?;
        let query = stream.clone_htod(initial).map_err(|error| {
            device(
                ctx,
                format!("CUDA energy initial query upload failed: {error}"),
            )
        })?;
        let control = stream
            .clone_htod(&[1_u32, 0, 0])
            .map_err(|error| device(ctx, format!("CUDA energy control upload failed: {error}")))?;
        Ok(Self {
            members: member_device,
            member_inverse_norms: alloc(ctx, shape.members, "member inverse norms")?,
            query,
            query_inverse_norm: alloc(ctx, 1, "query inverse norm")?,
            scores: alloc(ctx, shape.members, "scaled similarities")?,
            weights: alloc(ctx, shape.members, "softmax weights")?,
            centroid_partials: alloc(ctx, shape.centroid_tiles * shape.dim, "centroid partials")?,
            metadata: alloc(ctx, 1, "descent metadata")?,
            control,
            status: alloc(ctx, 1, "descent status")?,
            host_to_device_bytes: shape.padded_elements * FLOAT_BYTES
                + initial.len() * FLOAT_BYTES
                + CONTROL_WORDS * size_of::<u32>(),
        })
    }
}

pub(super) fn read_result(
    ctx: &CudaContext,
    buffers: EnergyBuffers,
    shape: EnergyShape,
    max_steps: usize,
    kernel_launches: usize,
) -> Result<CudaEnergyDescent> {
    let stream = ctx.inner().default_stream();
    let status = read(ctx, &buffers.status, "status")?[0];
    check_status(status)?;
    let vector = read(ctx, &buffers.query, "final query")?;
    let metadata = read(ctx, &buffers.metadata, "metadata")?;
    let control = read(ctx, &buffers.control, "control")?;
    let final_energy = metadata[0];
    if !final_energy.is_finite() || vector.iter().any(|value| !value.is_finite()) {
        return Err(numerical(
            "energy.final_state",
            "CUDA energy descent returned NaN or Inf".to_string(),
        ));
    }
    stream.synchronize().map_err(|error| {
        device(
            ctx,
            format!("CUDA energy final synchronization failed: {error}"),
        )
    })?;
    let device_to_host_bytes = size_of::<u32>()
        + vector.len() * FLOAT_BYTES
        + metadata.len() * FLOAT_BYTES
        + control.len() * size_of::<u32>();
    Ok(CudaEnergyDescent {
        vector,
        steps_taken: control[1] as usize,
        converged: control[2] != 0,
        final_energy,
        stats: CudaEnergyStats {
            members: shape.members as u64,
            dim: shape.dim as u64,
            max_steps: max_steps as u64,
            kernel_launches: kernel_launches as u64,
            host_to_device_bytes: buffers.host_to_device_bytes as u64,
            device_to_host_bytes: device_to_host_bytes as u64,
            peak_pinned_staging_bytes: (shape.pinned_elements * FLOAT_BYTES) as u64,
            peak_device_bytes: shape.peak_device_bytes as u64,
        },
    })
}

fn upload_members(
    ctx: &CudaContext,
    members: &[&[f32]],
    shape: &EnergyShape,
    device_members: &mut CudaSlice<f32>,
) -> Result<()> {
    let stream = ctx.inner().default_stream();
    let mut pinned = pinned_zeros(ctx, shape.pinned_elements)?;
    let mut start = 0_usize;
    while start < shape.padded_elements {
        let logical_take = (shape.elements - start.min(shape.elements)).min(shape.pinned_elements);
        let host = pinned.as_mut_slice().map_err(|error| {
            device(
                ctx,
                format!("CUDA energy pinned staging access failed: {error}"),
            )
        })?;
        host.fill(0.0);
        fill_member_chunk(host, members, shape.dim, start, logical_take);
        let mut target = device_members.slice_mut(start..start + shape.pinned_elements);
        stream
            .memcpy_htod(&pinned, &mut target)
            .map_err(|error| device(ctx, format!("CUDA energy member upload failed: {error}")))?;
        start += shape.pinned_elements;
    }
    Ok(())
}

fn fill_member_chunk(
    output: &mut [f32],
    members: &[&[f32]],
    dim: usize,
    start: usize,
    take: usize,
) {
    let mut source = start;
    let mut written = 0_usize;
    while written < take {
        let row = source / dim;
        let col = source % dim;
        let copy = (dim - col).min(take - written);
        output[written..written + copy].copy_from_slice(&members[row][col..col + copy]);
        source += copy;
        written += copy;
    }
}

fn admit_vram(ctx: &CudaContext, required: usize) -> Result<()> {
    let free = ctx.free_device_vram_bytes()?;
    let admitted = free.saturating_sub(CUDA_ENERGY_VRAM_RESERVE_BYTES);
    if required <= CUDA_ENERGY_MAX_DEVICE_BYTES && required <= admitted {
        return Ok(());
    }
    Err(device(
        ctx,
        format!(
            "CUDA energy resident region exceeds bounded admission; required_bytes={required} free_bytes={free} reserve_bytes={CUDA_ENERGY_VRAM_RESERVE_BYTES} max_bytes={CUDA_ENERGY_MAX_DEVICE_BYTES}"
        ),
    ))
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
                format!("CUDA energy {label} allocation failed: {error}"),
            )
        })
}

fn pinned_zeros(ctx: &CudaContext, len: usize) -> Result<PinnedHostSlice<f32>> {
    let mut pinned = unsafe { ctx.inner().alloc_pinned::<f32>(len) }.map_err(|error| {
        device(
            ctx,
            format!("CUDA energy pinned allocation failed: {error}"),
        )
    })?;
    pinned
        .as_mut_slice()
        .map_err(|error| {
            device(
                ctx,
                format!("CUDA energy pinned initialization failed: {error}"),
            )
        })?
        .fill(0.0);
    Ok(pinned)
}

fn read<T>(ctx: &CudaContext, input: &CudaSlice<T>, label: &str) -> Result<Vec<T>>
where
    T: cudarc::driver::DeviceRepr,
{
    ctx.inner()
        .default_stream()
        .clone_dtoh(input)
        .map_err(|error| device(ctx, format!("CUDA energy {label} readback failed: {error}")))
}

fn check_status(status: u32) -> Result<()> {
    let (op, detail) = if status & MEMBER_NONFINITE != 0 {
        ("energy.member_finite", "region member contains NaN or Inf")
    } else if status & MEMBER_ZERO_NORM != 0 {
        ("energy.member_norm", "region member has zero norm")
    } else if status & QUERY_NONFINITE != 0 {
        ("energy.query_finite", "query contains NaN or Inf")
    } else if status & QUERY_ZERO_NORM != 0 {
        ("energy.query_norm", "query has zero norm")
    } else if status & CENTROID_NONFINITE != 0 {
        (
            "energy.centroid_finite",
            "weighted centroid contains NaN or Inf",
        )
    } else if status & CENTROID_ZERO_NORM != 0 {
        ("energy.centroid_norm", "weighted centroid has zero norm")
    } else if status & STATE_NONFINITE != 0 {
        (
            "energy.state_finite",
            "similarity, softmax, or energy became non-finite",
        )
    } else {
        return Ok(());
    };
    Err(numerical(op, detail.to_string()))
}

fn ensure_i32(value: usize, label: &str) -> Result<()> {
    if i32::try_from(value).is_ok() {
        Ok(())
    } else {
        Err(shape_error(
            &format!("CUDA energy {label} exceeds i32"),
            vec![value],
        ))
    }
}

pub(super) fn device(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation: "free VRAM or repair the strict CUDA energy provider; CPU fallback is disabled for admitted large regions".to_string(),
    }
}

fn numerical(op: &'static str, detail: String) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail,
        remediation: "reject invalid Oracle region vectors before CUDA energy descent".to_string(),
    }
}

fn shape_error(remediation: &str, got: Vec<usize>) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![i32::MAX as usize],
        got,
        remediation: remediation.to_string(),
    }
}
