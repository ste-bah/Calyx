use std::str;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use serde::{Deserialize, Serialize};

use crate::cuda::gemm::gram_rows_cublas;
use crate::cuda::kernels::{LOOM_CUBIN, LOOM_PTX};
use crate::vram::RESERVED_HEADROOM_BYTES;
use crate::{CudaContext, ForgeError, Result, init_cuda};

mod workspace;
use workspace::{LoomShape, LoomWorkspace};

const THREADS: u32 = 256;
const FLAG_NONFINITE: u32 = 1;
const FLAG_ZERO_NORM: u32 = 1 << 1;
const FLAG_INVALID: u32 = 1 << 2;
const DEVICE_REMEDIATION: &str =
    "Check CUDA, embedded Loom PTX/CUBIN, cuBLAS, and available device memory";
const INPUT_REMEDIATION: &str =
    "Supply finite, equal-width, non-zero Loom rows and canonical pair indices";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CudaLoomVectorKind {
    Delta = 0,
    Interaction = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaLoomVectorRequest {
    pub left_row: usize,
    pub right_row: usize,
    pub kind: CudaLoomVectorKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaLoomStats {
    pub row_count: usize,
    pub dim: usize,
    pub agreement_pairs: usize,
    pub vector_terms: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub peak_device_bytes: usize,
    pub host_to_device_copies: usize,
    pub device_to_host_copies: usize,
    pub kernel_launches: usize,
    pub gemm_calls: usize,
    pub workspace_reused: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CudaLoomBatch {
    pub agreements: Vec<f32>,
    pub vector_terms: Vec<Vec<f32>>,
    pub stats: CudaLoomStats,
}

pub struct CudaLoomContext {
    context: CudaContext,
    workspace: Mutex<Option<LoomWorkspace>>,
}

impl CudaLoomContext {
    pub fn new(device_idx: u32) -> Result<Self> {
        Ok(Self::with_context(init_cuda(device_idx, false)?))
    }

    pub fn with_context(context: CudaContext) -> Self {
        Self {
            context,
            workspace: Mutex::new(None),
        }
    }

    pub fn context(&self) -> &CudaContext {
        &self.context
    }

    pub fn execute(
        &self,
        matrix: &[f32],
        row_count: usize,
        dim: usize,
        agreement_pairs: &[(usize, usize)],
        vector_requests: &[CudaLoomVectorRequest],
    ) -> Result<CudaLoomBatch> {
        validate_request(matrix, row_count, dim, agreement_pairs, vector_requests)?;
        if agreement_pairs.is_empty() && vector_requests.is_empty() {
            return Ok(CudaLoomBatch {
                agreements: Vec::new(),
                vector_terms: Vec::new(),
                stats: CudaLoomStats {
                    row_count,
                    dim,
                    ..CudaLoomStats::default()
                },
            });
        }
        let shape = LoomShape::new(row_count, dim, agreement_pairs.len(), vector_requests.len())?;
        ensure_device_room(&self.context, shape.device_bytes)?;
        let mut guard = self
            .workspace
            .lock()
            .map_err(|_| ForgeError::DeviceUnavailable {
                device: device_label(&self.context),
                detail: "Loom CUDA workspace lock is poisoned".to_string(),
                remediation: DEVICE_REMEDIATION.to_string(),
            })?;
        let reused = guard
            .as_ref()
            .is_some_and(|workspace| workspace.shape == shape);
        if !reused {
            *guard = Some(LoomWorkspace::new(&self.context, shape)?);
        }
        let workspace = guard.as_mut().expect("workspace initialized");
        workspace.upload(&self.context, matrix, agreement_pairs, vector_requests)?;
        workspace.reset_outputs(&self.context)?;
        let mut launches = 0;
        let mut gemm_calls = 0;
        if !agreement_pairs.is_empty() {
            launch_normalize(&self.context, workspace)?;
            launches += 1;
            gram_rows_cublas(
                &self.context,
                workspace.normalized.as_ref().expect("normalized buffer"),
                row_count,
                dim,
                workspace.gram.as_mut().expect("Gram buffer"),
            )?;
            launches += 1;
            gemm_calls += 1;
            launch_extract(&self.context, workspace)?;
            launches += 1;
        }
        if !vector_requests.is_empty() {
            launch_vectors(&self.context, workspace)?;
            launches += 1;
        }
        finish(
            &self.context,
            workspace,
            row_count,
            dim,
            agreement_pairs.len(),
            vector_requests.len(),
            launches,
            gemm_calls,
            reused,
        )
    }
}

fn launch_normalize(ctx: &CudaContext, workspace: &mut LoomWorkspace) -> Result<()> {
    let function = loom_function(ctx, "loom.normalize_rows", "loom_normalize_rows_f32")?;
    let rows = to_i32(workspace.shape.row_count, "row count")?;
    let dim = to_i32(workspace.shape.dim, "dimension")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(&workspace.matrix_device)
            .arg(&rows)
            .arg(&dim)
            .arg(workspace.normalized.as_mut().expect("normalized buffer"))
            .arg(&mut workspace.flags)
            .launch(LaunchConfig {
                grid_dim: (
                    to_u32(workspace.shape.row_count, "normalization grid")?,
                    1,
                    1,
                ),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map(|_| ())
    .map_err(|err| device_error(ctx, "normalization launch", err))
}

fn launch_extract(ctx: &CudaContext, workspace: &mut LoomWorkspace) -> Result<()> {
    let function = loom_function(ctx, "loom.extract_pairs", "loom_extract_pairs_f32")?;
    let pairs = workspace.agreements.as_ref().expect("agreement pairs");
    let pair_count = to_i32(workspace.shape.agreement_count, "agreement count")?;
    let rows = to_i32(workspace.shape.row_count, "row count")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(workspace.gram.as_ref().expect("Gram buffer"))
            .arg(&pairs.left_device)
            .arg(&pairs.right_device)
            .arg(&pair_count)
            .arg(&rows)
            .arg(
                workspace
                    .agreement_output
                    .as_mut()
                    .expect("agreement output"),
            )
            .arg(&mut workspace.flags)
            .launch(LaunchConfig {
                grid_dim: (grid_blocks(workspace.shape.agreement_count)?, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map(|_| ())
    .map_err(|err| device_error(ctx, "agreement extraction launch", err))
}

fn launch_vectors(ctx: &CudaContext, workspace: &mut LoomWorkspace) -> Result<()> {
    let function = loom_function(ctx, "loom.cross_terms", "loom_cross_terms_f32")?;
    let vectors = workspace.vectors.as_mut().expect("vector buffers");
    let count = to_i32(workspace.shape.vector_count, "vector request count")?;
    let rows = to_i32(workspace.shape.row_count, "row count")?;
    let dim = to_i32(workspace.shape.dim, "dimension")?;
    let stream = ctx.inner().default_stream();
    let mut launch = stream.launch_builder(function.as_ref());
    unsafe {
        launch
            .arg(&workspace.matrix_device)
            .arg(&vectors.pairs.left_device)
            .arg(&vectors.pairs.right_device)
            .arg(&vectors.kinds_device)
            .arg(&count)
            .arg(&rows)
            .arg(&dim)
            .arg(&mut vectors.output)
            .arg(&mut workspace.flags)
            .launch(LaunchConfig {
                grid_dim: (to_u32(workspace.shape.vector_count, "vector grid")?, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
    }
    .map(|_| ())
    .map_err(|err| device_error(ctx, "cross-term launch", err))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    ctx: &CudaContext,
    workspace: &LoomWorkspace,
    rows: usize,
    dim: usize,
    agreement_count: usize,
    vector_count: usize,
    launches: usize,
    gemm_calls: usize,
    reused: bool,
) -> Result<CudaLoomBatch> {
    let stream = ctx.inner().default_stream();
    stream
        .synchronize()
        .map_err(|err| device_error(ctx, "batch sync", err))?;
    let flags = stream
        .clone_dtoh(&workspace.flags)
        .map_err(|err| device_error(ctx, "flag readback", err))?;
    decode_flags(flags.first().copied().unwrap_or_default())?;
    let agreements = workspace
        .agreement_output
        .as_ref()
        .map(|output| {
            stream
                .clone_dtoh(output)
                .map_err(|err| device_error(ctx, "agreement readback", err))
        })
        .transpose()?
        .unwrap_or_default();
    let flat_vectors = workspace
        .vectors
        .as_ref()
        .map(|vectors| {
            stream
                .clone_dtoh(&vectors.output)
                .map_err(|err| device_error(ctx, "vector readback", err))
        })
        .transpose()?
        .unwrap_or_default();
    let vector_terms = flat_vectors
        .chunks_exact(dim)
        .map(<[f32]>::to_vec)
        .collect();
    let h2d_copies = 1 + usize::from(agreement_count > 0) * 2 + usize::from(vector_count > 0) * 3;
    let d2h_copies = 1 + usize::from(agreement_count > 0) + usize::from(vector_count > 0);
    let h2d_u32 = agreement_count * 2 + vector_count * 3;
    Ok(CudaLoomBatch {
        agreements,
        vector_terms,
        stats: CudaLoomStats {
            row_count: rows,
            dim,
            agreement_pairs: agreement_count,
            vector_terms: vector_count,
            host_to_device_bytes: workspace.shape.matrix_len * size_of::<f32>()
                + h2d_u32 * size_of::<u32>(),
            device_to_host_bytes: (agreement_count + workspace.shape.vector_len) * size_of::<f32>()
                + size_of::<u32>(),
            peak_device_bytes: workspace.shape.device_bytes,
            host_to_device_copies: h2d_copies,
            device_to_host_copies: d2h_copies,
            kernel_launches: launches,
            gemm_calls,
            workspace_reused: reused,
        },
    })
}

fn validate_request(
    matrix: &[f32],
    rows: usize,
    dim: usize,
    agreements: &[(usize, usize)],
    vectors: &[CudaLoomVectorRequest],
) -> Result<()> {
    let expected = checked_mul(rows, dim, "Loom input shape")?;
    if rows == 0 || dim == 0 || matrix.len() != expected {
        return Err(shape_error(
            vec![rows, dim],
            vec![matrix.len()],
            "matrix must be non-empty rows*dim",
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(numerical("Loom matrix contains NaN or infinity"));
    }
    let mut agreement_rows = vec![false; rows];
    for &(left, right) in agreements {
        validate_pair(left, right, rows)?;
        agreement_rows[left] = true;
        agreement_rows[right] = true;
    }
    for request in vectors {
        validate_pair(request.left_row, request.right_row, rows)?;
    }
    if agreement_rows
        .iter()
        .enumerate()
        .any(|(row, active)| *active && zero_norm(&matrix[row * dim..(row + 1) * dim]))
    {
        return Err(numerical("Loom agreement request contains a zero-norm row"));
    }
    Ok(())
}

fn zero_norm(row: &[f32]) -> bool {
    let mut norm = 0.0_f32;
    for value in row {
        norm += value * value;
    }
    norm <= f32::EPSILON
}

fn validate_pair(left: usize, right: usize, rows: usize) -> Result<()> {
    if left <= right && right < rows {
        return Ok(());
    }
    Err(shape_error(
        vec![0, rows],
        vec![left, right],
        "pair indices must satisfy left <= right < row_count",
    ))
}

fn loom_function(
    ctx: &CudaContext,
    key: &'static str,
    name: &'static str,
) -> Result<Arc<cudarc::driver::CudaFunction>> {
    let module = loom_module(ctx)?;
    ctx.cached_function(&module, key, name)
        .map_err(|err| device_error(ctx, "function load", err))
}

fn loom_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.loom_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(LOOM_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_error) => {
            let ptx = str::from_utf8(LOOM_PTX).map_err(|err| ForgeError::DeviceUnavailable {
                device: device_label(ctx),
                detail: format!("Loom PTX is not UTF-8: {err}"),
                remediation: DEVICE_REMEDIATION.to_string(),
            })?;
            ctx.inner()
                .load_module(Ptx::from_src(ptx))
                .map_err(|ptx_error| ForgeError::DeviceUnavailable {
                    device: device_label(ctx),
                    detail: format!(
                        "Loom CUBIN load failed: {cubin_error}; PTX load failed: {ptx_error}"
                    ),
                    remediation: DEVICE_REMEDIATION.to_string(),
                })?
        }
    };
    let _ = ctx.loom_module_cache().set(module.clone());
    Ok(module)
}

fn decode_flags(flags: u32) -> Result<()> {
    if flags & FLAG_INVALID != 0 {
        return Err(shape_error(
            vec![0],
            vec![1],
            "Loom CUDA kernel rejected dimensions or pair indices",
        ));
    }
    if flags & FLAG_ZERO_NORM != 0 {
        return Err(numerical("Loom CUDA normalization found a zero-norm row"));
    }
    if flags & FLAG_NONFINITE != 0 {
        return Err(numerical("Loom CUDA kernel produced a non-finite value"));
    }
    Ok(())
}

fn ensure_device_room(ctx: &CudaContext, bytes: usize) -> Result<()> {
    let free = ctx.free_device_vram_bytes()?;
    let usable = free.saturating_sub(RESERVED_HEADROOM_BYTES);
    if bytes <= usable {
        return Ok(());
    }
    Err(ForgeError::VramBudget {
        detail: format!(
            "Loom batch requested_bytes={bytes} exceeds usable_bytes={usable} free_bytes={free} reserved_headroom_bytes={RESERVED_HEADROOM_BYTES}"
        ),
        remediation: "Reduce Loom panel rows/dimension/materialized vectors or free GPU VRAM"
            .to_string(),
    })
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| shape_error(vec![left, right], vec![], label))
}

fn checked_sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |sum, value| {
        sum.checked_add(*value)
            .ok_or_else(|| shape_error(vec![usize::MAX], vec![], "Loom size sum overflow"))
    })
}

fn to_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| shape_error(vec![i32::MAX as usize], vec![value], label))
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| shape_error(vec![u32::MAX as usize], vec![value], label))
}

fn grid_blocks(len: usize) -> Result<u32> {
    to_u32(len.div_ceil(THREADS as usize), "grid blocks")
}

fn shape_error(expected: Vec<usize>, got: Vec<usize>, detail: &str) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected,
        got,
        remediation: format!("{INPUT_REMEDIATION}; {detail}"),
    }
}

fn numerical(detail: &str) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: "loom_cuda_batch".to_string(),
        detail: detail.to_string(),
        remediation: INPUT_REMEDIATION.to_string(),
    }
}

fn device_error(ctx: &CudaContext, stage: &str, error: cudarc::driver::DriverError) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: device_label(ctx),
        detail: format!("Loom {stage} failed: {error}"),
        remediation: DEVICE_REMEDIATION.to_string(),
    }
}

fn device_label(ctx: &CudaContext) -> String {
    format!("cuda:{}", ctx.device_idx())
}

#[cfg(test)]
mod tests;
