use super::*;

pub(super) const THREADS: u32 = 256;
pub(super) const MAX_REDUCTION_BLOCKS: usize = 4096;
pub(super) const MAX_LINALG_VARIABLES: usize = 64;
pub(super) const MAX_HAWKES_PROCESSES: usize = 32;
pub(super) const MAX_GLS_PERMUTATION_CELLS: usize = 32 * 1024 * 1024;
pub(super) const VRAM_HEADROOM_BYTES: usize = 512 * 1024 * 1024;
pub(super) const DEVICE_REMEDIATION: &str =
    "Check CUDA, embedded assay PTX/CUBIN, and CUDA GPU availability";
pub(super) const NUMERICAL_REMEDIATION: &str =
    "Reject non-finite assay inputs and degenerate statistics; do not fall back to CPU";
pub(super) const VRAM_REMEDIATION: &str = "Reduce assay sample count/permutations/dimension or free GPU memory; strict assay CUDA never falls back to CPU";

pub(super) const FLAG_NONFINITE: u32 = 1;
pub(super) const FLAG_INVALID_INDEX: u32 = 1 << 1;

pub(super) fn assay_function(
    ctx: &CudaContext,
    cache_key: &'static str,
    function_name: &'static str,
) -> Result<Arc<cudarc::driver::CudaFunction>> {
    let module = assay_module(ctx)?;
    ctx.cached_function(&module, cache_key, function_name)
        .map_err(|err| device_unavailable(ctx, format!("{function_name} load failed: {err}")))
}

pub(super) fn assay_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.assay_module_cache().get() {
        return Ok(module.clone());
    }
    match ctx
        .inner()
        .load_module(Ptx::from_binary(ASSAY_CUBIN.to_vec()))
    {
        Ok(module) => {
            let _ = ctx.assay_module_cache().set(module.clone());
            Ok(module)
        }
        Err(cubin_err) => {
            let module = assay_ptx_module(ctx, cubin_err)?;
            let _ = ctx.assay_module_cache().set(module.clone());
            Ok(module)
        }
    }
}

pub(super) fn assay_ptx_module(
    ctx: &CudaContext,
    cubin_err: cudarc::driver::DriverError,
) -> Result<Arc<CudaModule>> {
    let ptx = str::from_utf8(ASSAY_PTX)
        .map_err(|err| device_unavailable(ctx, format!("assay PTX is not UTF-8: {err}")))?;
    ctx.inner()
        .load_module(Ptx::from_src(ptx))
        .map_err(|ptx_err| {
            device_unavailable(
                ctx,
                format!(
                    "assay CUBIN load failed: {cubin_err}; PTX fallback load failed: {ptx_err}"
                ),
            )
        })
}

pub(super) fn alloc_flags(ctx: &CudaContext, op: &'static str) -> Result<CudaSlice<u32>> {
    ctx.inner()
        .default_stream()
        .alloc_zeros(1)
        .map_err(|err| device_unavailable(ctx, format!("{op} flag allocation failed: {err}")))
}

pub(super) fn sync_and_decode(
    ctx: &CudaContext,
    op: &'static str,
    flags: &CudaSlice<u32>,
) -> Result<()> {
    ctx.inner()
        .default_stream()
        .synchronize()
        .map_err(|err| device_unavailable(ctx, format!("{op} sync failed: {err}")))?;
    let flags = ctx
        .inner()
        .default_stream()
        .clone_dtoh(flags)
        .map_err(|err| device_unavailable(ctx, format!("{op} flag readback failed: {err}")))?;
    decode_flags(op, flags.first().copied().unwrap_or_default())
}

pub(super) fn decode_flags(op: &'static str, flags: u32) -> Result<()> {
    if flags & FLAG_INVALID_INDEX != 0 {
        return Err(ForgeError::ShapeMismatch {
            expected: vec![0],
            got: vec![1],
            remediation: format!(
                "{op} CUDA kernel reported invalid indices, dimensions, or launch configuration"
            ),
        });
    }
    if flags & FLAG_NONFINITE != 0 {
        return Err(numerical(
            op,
            "assay CUDA kernel reported a non-finite intermediate or output".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn ensure_device_room(
    ctx: &CudaContext,
    op: &'static str,
    requested_bytes: usize,
) -> Result<()> {
    let free_bytes = ctx.free_device_vram_bytes()?;
    let available = free_bytes.saturating_sub(VRAM_HEADROOM_BYTES);
    if requested_bytes <= available {
        return Ok(());
    }
    Err(ForgeError::VramBudget {
        detail: format!(
            "{op} would exceed live CUDA VRAM headroom: requested_bytes={requested_bytes} free_bytes={free_bytes} reserved_headroom_bytes={VRAM_HEADROOM_BYTES}"
        ),
        remediation: VRAM_REMEDIATION.to_string(),
    })
}

pub(super) fn checked_square(n: usize, name: &str) -> Result<usize> {
    n.checked_mul(n).ok_or_else(|| shape_overflow(name))
}

pub(super) fn checked_sum_bytes(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |acc, part| {
        acc.checked_add(*part)
            .ok_or_else(|| shape_overflow("assay byte estimate overflow"))
    })
}

pub(super) fn bytes<T>(count: usize, name: &str) -> Result<usize> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| shape_overflow(name))
}

pub(super) fn shape_overflow(detail: &str) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![usize::MAX],
        got: vec![0],
        remediation: detail.to_string(),
    }
}

pub(super) fn reduction_blocks(len: usize) -> usize {
    len.div_ceil((THREADS as usize) * 4)
        .clamp(1, MAX_REDUCTION_BLOCKS)
}

pub(super) fn grid_blocks(len: usize) -> Result<u32> {
    to_u32(len.div_ceil(THREADS as usize), "assay grid blocks")
}

pub(super) fn to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![i32::MAX as usize],
        got: vec![value],
        remediation: format!("{name} exceeds CUDA i32 kernel argument limit"),
    })
}

pub(super) fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![value],
        remediation: format!("{name} exceeds CUDA grid dimension limit"),
    })
}

pub(super) fn numerical(op: &'static str, detail: String) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail,
        remediation: NUMERICAL_REMEDIATION.to_string(),
    }
}

pub(super) fn device_unavailable(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation: DEVICE_REMEDIATION.to_string(),
    }
}
