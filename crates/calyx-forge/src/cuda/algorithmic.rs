use std::str;
use std::sync::Arc;

use cudarc::driver::{CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use serde::{Deserialize, Serialize};

use crate::cuda::kernels::{ALGORITHMIC_CUBIN, ALGORITHMIC_PTX};
use crate::{CudaContext, ForgeError, Result, init_cuda};

const THREADS: u32 = 256;
const BYTE_RAW_STRIDE: usize = 15;
const DEVICE_REMEDIATION: &str =
    "Check CUDA, embedded algorithmic PTX/CUBIN, and CUDA GPU availability";

/// Largest keyword whose `u64 length || bytes` BLAKE3 message is one chunk.
pub const ALGORITHMIC_SPARSE_MAX_TOKEN_BYTES: usize = 1024 - 8;
/// Largest token whose domain-separated BLAKE3 message is one chunk.
pub const ALGORITHMIC_TOKEN_HASH_MAX_TOKEN_BYTES: usize = 1024 - 31 - 4;

/// Host-owned flattened bytes plus monotonic `u32` row offsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaByteRaggedBatch {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

impl CudaByteRaggedBatch {
    pub fn from_slices(rows: &[&[u8]]) -> Result<Self> {
        let total = rows.iter().try_fold(0_usize, |total, row| {
            total
                .checked_add(row.len())
                .ok_or_else(|| shape("ragged byte length overflows usize", rows.len()))
        })?;
        if total > u32::MAX as usize {
            return Err(shape("ragged byte length exceeds CUDA u32 offsets", total));
        }
        let mut bytes = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        for row in rows {
            bytes.extend_from_slice(row);
            offsets.push(bytes.len() as u32);
        }
        Ok(Self { bytes, offsets })
    }

    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn input_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn row(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)? as usize;
        let end = *self.offsets.get(index + 1)? as usize;
        self.bytes.get(start..end)
    }
}

/// Integer reductions returned by the byte-feature kernel before CPU `f32` conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaByteFeatureRaw {
    pub values: [u64; BYTE_RAW_STRIDE],
}

/// Persistable provider and transfer evidence for one algorithmic CUDA invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaAlgorithmicStats {
    pub rows: u64,
    pub work_items: u64,
    pub input_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub kernel_launches: u64,
}

#[derive(Clone, Debug)]
pub struct CudaAlgorithmicContext {
    ctx: CudaContext,
}

impl CudaAlgorithmicContext {
    pub fn new(device_idx: u32) -> Result<Self> {
        init_cuda(device_idx, true).map(Self::with_context)
    }

    pub fn with_context(ctx: CudaContext) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Reduces every byte row with one kernel launch.
    pub fn byte_features_raw(
        &self,
        batch: &CudaByteRaggedBatch,
    ) -> Result<(Vec<CudaByteFeatureRaw>, CudaAlgorithmicStats)> {
        if batch.rows() == 0 {
            return Ok((Vec::new(), empty_stats(batch)));
        }
        let rows = as_u32(batch.rows(), "byte row count")?;
        let output_len = checked_product(batch.rows(), BYTE_RAW_STRIDE, "byte output")?;
        let stream = self.ctx.inner().default_stream();
        let input = input_to_device(&self.ctx, batch)?;
        let mut output: CudaSlice<u64> = stream
            .alloc_zeros(output_len)
            .map_err(|err| device(&self.ctx, format!("byte output allocation failed: {err}")))?;
        let function = function(
            &self.ctx,
            "algorithmic.byte_features",
            "algorithmic_byte_features",
        )?;
        let mut launch = stream.launch_builder(function.as_ref());
        unsafe {
            launch
                .arg(&input.bytes)
                .arg(&input.offsets)
                .arg(&rows)
                .arg(&mut output)
                .launch(flat_threads(batch.rows())?)
        }
        .map_err(|err| device(&self.ctx, format!("byte kernel launch failed: {err}")))?;
        sync(&self.ctx, "byte features")?;
        let values = stream
            .clone_dtoh(&output)
            .map_err(|err| device(&self.ctx, format!("byte output readback failed: {err}")))?;
        let raw = values
            .chunks_exact(BYTE_RAW_STRIDE)
            .map(|chunk| CudaByteFeatureRaw {
                values: chunk.try_into().expect("exact byte raw stride"),
            })
            .collect();
        Ok((raw, stats(batch, batch.rows(), output_len * 8)))
    }

    /// Hashes all pre-tokenized sparse keywords with one kernel launch.
    pub fn sparse_keyword_hashes(
        &self,
        tokens: &CudaByteRaggedBatch,
    ) -> Result<(Vec<u32>, CudaAlgorithmicStats)> {
        validate_token_lengths(tokens, ALGORITHMIC_SPARSE_MAX_TOKEN_BYTES, "sparse")?;
        if tokens.rows() == 0 {
            return Ok((Vec::new(), empty_stats(tokens)));
        }
        let token_count = as_u32(tokens.rows(), "sparse token count")?;
        let stream = self.ctx.inner().default_stream();
        let input = input_to_device(&self.ctx, tokens)?;
        let mut output: CudaSlice<u32> = stream
            .alloc_zeros(tokens.rows())
            .map_err(|err| device(&self.ctx, format!("sparse output allocation failed: {err}")))?;
        let function = function(
            &self.ctx,
            "algorithmic.sparse_hashes",
            "algorithmic_sparse_hashes",
        )?;
        let mut launch = stream.launch_builder(function.as_ref());
        unsafe {
            launch
                .arg(&input.bytes)
                .arg(&input.offsets)
                .arg(&token_count)
                .arg(&mut output)
                .launch(flat_threads(tokens.rows())?)
        }
        .map_err(|err| device(&self.ctx, format!("sparse kernel launch failed: {err}")))?;
        sync(&self.ctx, "sparse hashes")?;
        let hashes = stream
            .clone_dtoh(&output)
            .map_err(|err| device(&self.ctx, format!("sparse output readback failed: {err}")))?;
        Ok((hashes, stats(tokens, tokens.rows(), tokens.rows() * 4)))
    }

    /// Expands all pre-tokenized TokenHash rows with one kernel launch.
    pub fn token_hash_words(
        &self,
        tokens: &CudaByteRaggedBatch,
        token_dim: u32,
    ) -> Result<(Vec<u32>, CudaAlgorithmicStats)> {
        validate_token_lengths(tokens, ALGORITHMIC_TOKEN_HASH_MAX_TOKEN_BYTES, "token hash")?;
        let token_dim = token_dim.max(1);
        if tokens.rows() == 0 {
            return Ok((Vec::new(), empty_stats(tokens)));
        }
        let token_count = as_u32(tokens.rows(), "token hash token count")?;
        let groups = token_dim.div_ceil(8);
        let jobs = checked_product(tokens.rows(), groups as usize, "token hash jobs")?;
        as_u32(jobs, "token hash jobs")?;
        let output_len = checked_product(tokens.rows(), token_dim as usize, "token hash output")?;
        as_u32(output_len, "token hash output words")?;
        let stream = self.ctx.inner().default_stream();
        let input = input_to_device(&self.ctx, tokens)?;
        let mut output: CudaSlice<u32> = stream
            .alloc_zeros(output_len)
            .map_err(|err| device(&self.ctx, format!("token output allocation failed: {err}")))?;
        let function = function(
            &self.ctx,
            "algorithmic.token_hash_words",
            "algorithmic_token_hash_words",
        )?;
        let mut launch = stream.launch_builder(function.as_ref());
        unsafe {
            launch
                .arg(&input.bytes)
                .arg(&input.offsets)
                .arg(&token_count)
                .arg(&token_dim)
                .arg(&groups)
                .arg(&mut output)
                .launch(flat_threads(jobs)?)
        }
        .map_err(|err| device(&self.ctx, format!("token kernel launch failed: {err}")))?;
        sync(&self.ctx, "token hashes")?;
        let words = stream
            .clone_dtoh(&output)
            .map_err(|err| device(&self.ctx, format!("token output readback failed: {err}")))?;
        Ok((words, stats(tokens, jobs, output_len * 4)))
    }
}

struct DeviceInput {
    bytes: CudaSlice<u8>,
    offsets: CudaSlice<u32>,
}

fn input_to_device(ctx: &CudaContext, batch: &CudaByteRaggedBatch) -> Result<DeviceInput> {
    let stream = ctx.inner().default_stream();
    let bytes = if batch.bytes.is_empty() {
        stream.clone_htod(&[0_u8])
    } else {
        stream.clone_htod(&batch.bytes)
    }
    .map_err(|err| device(ctx, format!("algorithmic input copy failed: {err}")))?;
    let offsets = stream
        .clone_htod(&batch.offsets)
        .map_err(|err| device(ctx, format!("algorithmic offset copy failed: {err}")))?;
    Ok(DeviceInput { bytes, offsets })
}

fn function(
    ctx: &CudaContext,
    cache_key: &'static str,
    name: &'static str,
) -> Result<Arc<cudarc::driver::CudaFunction>> {
    let module = algorithmic_module(ctx)?;
    ctx.cached_function(&module, cache_key, name)
        .map_err(|err| device(ctx, format!("{name} load failed: {err}")))
}

fn algorithmic_module(ctx: &CudaContext) -> Result<Arc<CudaModule>> {
    if let Some(module) = ctx.algorithmic_module_cache().get() {
        return Ok(module.clone());
    }
    let module = match ctx
        .inner()
        .load_module(Ptx::from_binary(ALGORITHMIC_CUBIN.to_vec()))
    {
        Ok(module) => module,
        Err(cubin_err) => {
            let ptx = str::from_utf8(ALGORITHMIC_PTX)
                .map_err(|err| device(ctx, format!("algorithmic PTX is not UTF-8: {err}")))?;
            ctx.inner().load_module(Ptx::from_src(ptx)).map_err(|ptx_err| {
                device(
                    ctx,
                    format!(
                        "algorithmic CUBIN load failed: {cubin_err}; PTX fallback failed: {ptx_err}"
                    ),
                )
            })?
        }
    };
    let _ = ctx.algorithmic_module_cache().set(module.clone());
    Ok(module)
}

fn validate_token_lengths(batch: &CudaByteRaggedBatch, max: usize, op: &str) -> Result<()> {
    for row in 0..batch.rows() {
        let len = batch.row(row).expect("validated ragged row").len();
        if len > max {
            return Err(ForgeError::ShapeMismatch {
                expected: vec![max],
                got: vec![len],
                remediation: format!(
                    "keep {op} BLAKE3 messages within one 1024-byte chunk or use the exact CPU path"
                ),
            });
        }
    }
    Ok(())
}

fn stats(
    batch: &CudaByteRaggedBatch,
    work_items: usize,
    output_bytes: usize,
) -> CudaAlgorithmicStats {
    let transferred_input = batch.input_bytes().max(1);
    CudaAlgorithmicStats {
        rows: batch.rows() as u64,
        work_items: work_items as u64,
        input_bytes: batch.input_bytes() as u64,
        host_to_device_bytes: (transferred_input + batch.offsets.len() * 4) as u64,
        device_to_host_bytes: output_bytes as u64,
        kernel_launches: 1,
    }
}

fn empty_stats(batch: &CudaByteRaggedBatch) -> CudaAlgorithmicStats {
    CudaAlgorithmicStats {
        rows: batch.rows() as u64,
        input_bytes: batch.input_bytes() as u64,
        ..CudaAlgorithmicStats::default()
    }
}

fn flat_threads(items: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (
            as_u32(items.div_ceil(THREADS as usize), "grid blocks")?,
            1,
            1,
        ),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn sync(ctx: &CudaContext, op: &str) -> Result<()> {
    ctx.inner()
        .default_stream()
        .synchronize()
        .map_err(|err| device(ctx, format!("{op} stream sync failed: {err}")))
}

fn checked_product(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| shape(&format!("{label} length overflows usize"), usize::MAX))
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| shape(&format!("{label} exceeds u32"), value))
}

fn shape(detail: &str, got: usize) -> ForgeError {
    ForgeError::ShapeMismatch {
        expected: vec![u32::MAX as usize],
        got: vec![got],
        remediation: detail.to_string(),
    }
}

fn device(ctx: &CudaContext, detail: String) -> ForgeError {
    ForgeError::DeviceUnavailable {
        device: format!("cuda:{}", ctx.device_idx()),
        detail,
        remediation: DEVICE_REMEDIATION.to_string(),
    }
}
