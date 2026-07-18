use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, Weak};

use calyx_core::{CalyxError, MeasurementGroupKey, Result, SlotShape};
use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model};

use super::super::super::cuda_guard::CudaDropGuard;
use super::super::super::{OnnxModelFiles, OnnxProviderPolicy};
#[cfg(feature = "cuda")]
use super::device;
use super::{
    BGE_M3_DENSE_DIM, BGE_M3_SPARSE_DIM, RUNTIMES, SharedBgem3Backend, SharedBgem3Runtime,
};
use crate::spec::FastembedBgem3Output;

pub(super) fn shared_runtime(
    key: MeasurementGroupKey,
    model_name: Bgem3Model,
    cache_dir: PathBuf,
    provider_policy: OnnxProviderPolicy,
) -> Result<Arc<SharedBgem3Runtime>> {
    let cache = RUNTIMES.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        CalyxError::lens_unreachable("BGE-M3 shared runtime cache mutex was poisoned")
    })?;
    cache.retain(|_, runtime| runtime.strong_count() > 0);
    if let Some(runtime) = cache.get(&key).and_then(Weak::upgrade) {
        return Ok(runtime);
    }
    let model = Bgem3Embedding::try_new(
        Bgem3InitOptions::new(model_name)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false)
            .with_intra_threads(1)
            .with_execution_providers(
                crate::runtime::onnx::fastembed_runtime::execution_providers(provider_policy)?,
            ),
    )
    .map_err(|error| CalyxError::lens_unreachable(format!("BGE-M3 init failed: {error}")))?;
    let model = CudaDropGuard::new(model, provider_policy);
    let runtime = Arc::new(SharedBgem3Runtime {
        key,
        provider_policy,
        backend: SharedBgem3Backend::Fastembed(Box::new(Some(Mutex::new(model.into_inner())))),
        forward_calls: AtomicU64::new(0),
        tokenization_calls: AtomicU64::new(0),
        dense_conversions: AtomicU64::new(0),
        sparse_conversions: AtomicU64::new(0),
        colbert_conversions: AtomicU64::new(0),
        active_runs: AtomicU64::new(0),
        max_concurrent_runs: AtomicU64::new(0),
    });
    cache.insert(key, Arc::downgrade(&runtime));
    Ok(runtime)
}

#[cfg(feature = "cuda")]
pub(super) fn shared_device_runtime(
    key: MeasurementGroupKey,
    prepared: device::DeviceBgem3Prepared,
) -> Result<Arc<SharedBgem3Runtime>> {
    let cache = RUNTIMES.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        CalyxError::lens_unreachable("BGE-M3 shared runtime cache mutex was poisoned")
    })?;
    cache.retain(|_, runtime| runtime.strong_count() > 0);
    if let Some(runtime) = cache.get(&key).and_then(Weak::upgrade) {
        return Ok(runtime);
    }
    let model = prepared.build()?;
    let runtime = Arc::new(SharedBgem3Runtime {
        key,
        provider_policy: OnnxProviderPolicy::CudaFailLoud,
        backend: SharedBgem3Backend::OnnxCuda(Box::new(Mutex::new(model))),
        forward_calls: AtomicU64::new(0),
        tokenization_calls: AtomicU64::new(0),
        dense_conversions: AtomicU64::new(0),
        sparse_conversions: AtomicU64::new(0),
        colbert_conversions: AtomicU64::new(0),
        active_runs: AtomicU64::new(0),
        max_concurrent_runs: AtomicU64::new(0),
    });
    cache.insert(key, Arc::downgrade(&runtime));
    Ok(runtime)
}

#[cfg(feature = "cuda")]
pub(super) fn device_runtime_key(
    prepared: &device::DeviceBgem3Prepared,
) -> Result<MeasurementGroupKey> {
    let provider = format!(
        "{};device={};gpu_mem_limit={:?};cuda_graphs={}",
        OnnxProviderPolicy::CudaFailLoud.as_str(),
        crate::runtime::onnx::session::configured_cuda_device()?,
        crate::runtime::onnx::arena::configured_gpu_mem_limit()?,
        crate::runtime::onnx::session::configured_cuda_graphs()?
    );
    let mut hasher = blake3::Hasher::new();
    let max_batch = prepared
        .max_batch
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let max_tokens = prepared.max_tokens.to_string();
    for part in [
        b"calyx:onnx-bgem3-cuda-runtime:v1".as_slice(),
        prepared.model_id.as_bytes(),
        prepared.weights_sha256.as_slice(),
        provider.as_bytes(),
        max_tokens.as_bytes(),
        max_batch.as_bytes(),
    ] {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    for path in prepared.files.artifact_paths() {
        let canonical = path.canonicalize().map_err(|error| {
            CalyxError::lens_unreachable(format!(
                "canonicalize CUDA BGE-M3 artifact {}: {error}",
                path.display()
            ))
        })?;
        let identity = canonical.to_string_lossy();
        hasher.update(&(identity.len() as u64).to_be_bytes());
        hasher.update(identity.as_bytes());
    }
    Ok(MeasurementGroupKey::from_bytes(
        *hasher.finalize().as_bytes(),
    ))
}

pub(super) fn runtime_key(
    files: &OnnxModelFiles,
    weights_sha256: [u8; 32],
    model_code: &str,
    model_file: &str,
    provider_policy: OnnxProviderPolicy,
    effective_max_batch: Option<usize>,
) -> Result<MeasurementGroupKey> {
    let cache_dir = files.cache_dir.canonicalize().map_err(|error| {
        CalyxError::lens_unreachable(format!(
            "canonicalize BGE-M3 cache root {}: {error}",
            files.cache_dir.display()
        ))
    })?;
    let provider = format!(
        "{};device=0;gpu_mem_limit={:?};cuda_graphs={}",
        provider_policy.as_str(),
        crate::runtime::onnx::arena::configured_gpu_mem_limit()?,
        crate::runtime::onnx::session::configured_cuda_graphs()?
    );
    let batch = effective_max_batch
        .map(|limit| limit.to_string())
        .unwrap_or_else(|| "none".to_string());
    let cache_identity = cache_dir.to_string_lossy().into_owned();
    let mut hasher = blake3::Hasher::new();
    for part in [
        b"calyx:fastembed-bgem3-runtime:v2".as_slice(),
        model_code.as_bytes(),
        model_file.as_bytes(),
        weights_sha256.as_slice(),
        cache_identity.as_bytes(),
        provider.as_bytes(),
        b"max_length=512;truncation=fastembed-default;intra_threads=1",
        batch.as_bytes(),
    ] {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    Ok(MeasurementGroupKey::from_bytes(
        *hasher.finalize().as_bytes(),
    ))
}

pub(super) fn output_for_shape(shape: SlotShape) -> Result<FastembedBgem3Output> {
    match shape {
        SlotShape::Dense(BGE_M3_DENSE_DIM) => Ok(FastembedBgem3Output::Dense),
        SlotShape::Sparse(BGE_M3_SPARSE_DIM) => Ok(FastembedBgem3Output::Sparse),
        SlotShape::Multi {
            token_dim: BGE_M3_DENSE_DIM,
        } => Ok(FastembedBgem3Output::Colbert),
        other => Err(CalyxError::lens_dim_mismatch(format!(
            "shape {other:?} is not a BGE-M3 projection"
        ))),
    }
}
