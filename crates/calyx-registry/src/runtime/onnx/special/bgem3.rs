use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use calyx_core::{
    CalyxError, GroupedLensRequest, Input, Lens, LensId, MeasurementGroupKey, Modality, Result,
    SlotShape, SlotVector,
};
use fastembed::{Bgem3Embedding, Bgem3Model};

use super::super::{OnnxModelFiles, OnnxProviderPolicy};
use super::models::{
    BGE_M3_DENSE_DIM, BGE_M3_SPARSE_DIM, bgem3_corpus_token, bgem3_model_from_name, bgem3_norm,
    bgem3_runtime_name, bgem3_shape,
};
#[cfg(feature = "cuda")]
use super::vectors::ensure_spec_match;
use super::vectors::{
    dense_batch, input_texts, leak_cuda_model, lock_model, multi_batch, sparse_batch, special_files,
};
use crate::frozen::{FrozenLensContract, LensDType, sha256_digest};
use crate::runtime::common::hash_files;
use crate::spec::{Bgem3Engine, FastembedBgem3Output, LensRuntime, LensSpec};

#[cfg(feature = "cuda")]
mod device;
mod runtime;

#[cfg(feature = "cuda")]
use runtime::{device_runtime_key, shared_device_runtime};
use runtime::{output_for_shape, runtime_key, shared_runtime};

static RUNTIMES: OnceLock<Mutex<BTreeMap<MeasurementGroupKey, Weak<SharedBgem3Runtime>>>> =
    OnceLock::new();

#[derive(Clone)]
pub struct FastembedBgem3Lens {
    id: LensId,
    output: FastembedBgem3Output,
    engine: Bgem3Engine,
    contract: FrozenLensContract,
    files: OnnxModelFiles,
    runtime: Arc<SharedBgem3Runtime>,
}

struct SharedBgem3Runtime {
    key: MeasurementGroupKey,
    provider_policy: OnnxProviderPolicy,
    backend: SharedBgem3Backend,
    forward_calls: AtomicU64,
    tokenization_calls: AtomicU64,
    dense_conversions: AtomicU64,
    sparse_conversions: AtomicU64,
    colbert_conversions: AtomicU64,
    active_runs: AtomicU64,
    max_concurrent_runs: AtomicU64,
}

enum SharedBgem3Backend {
    Fastembed(Box<Option<Mutex<Bgem3Embedding>>>),
    #[cfg(feature = "cuda")]
    OnnxCuda(Box<Mutex<device::DeviceBgem3Runtime>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bgem3RuntimeStats {
    pub session_initializations: u64,
    pub forward_calls: u64,
    pub tokenization_calls: u64,
    pub dense_conversions: u64,
    pub sparse_conversions: u64,
    pub colbert_conversions: u64,
    pub max_concurrent_runs: u64,
}

impl FastembedBgem3Lens {
    pub fn from_model_name_with_policy(
        name: impl Into<String>,
        model_name: &str,
        output: FastembedBgem3Output,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        Self::from_model_name_with_policy_and_batch(
            name,
            model_name,
            output,
            cache_dir,
            provider_policy,
            None,
        )
    }

    fn from_model_name_with_policy_and_batch(
        name: impl Into<String>,
        model_name: &str,
        output: FastembedBgem3Output,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
        effective_max_batch: Option<usize>,
    ) -> Result<Self> {
        let model_name = bgem3_model_from_name(model_name)?;
        Self::from_model_with_policy_and_batch(
            name,
            model_name,
            output,
            cache_dir,
            provider_policy,
            effective_max_batch,
        )
    }

    pub fn from_model_with_policy(
        name: impl Into<String>,
        model_name: Bgem3Model,
        output: FastembedBgem3Output,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        Self::from_model_with_policy_and_batch(
            name,
            model_name,
            output,
            cache_dir,
            provider_policy,
            None,
        )
    }

    fn from_model_with_policy_and_batch(
        name: impl Into<String>,
        model_name: Bgem3Model,
        output: FastembedBgem3Output,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
        effective_max_batch: Option<usize>,
    ) -> Result<Self> {
        super::super::dynamic_ort::ensure_dynamic_ort(provider_policy)?;
        let info = Bgem3Embedding::get_model_info(&model_name);
        let files = special_files(
            &cache_dir,
            &info.model_code,
            &info.model_file,
            &info.additional_files,
        )?;
        let weights_sha256 = hash_files(&files.artifact_paths())?;
        let key = runtime_key(
            &files,
            weights_sha256,
            &info.model_code,
            &info.model_file,
            provider_policy,
            effective_max_batch,
        )?;
        let runtime = shared_runtime(key, model_name, cache_dir, provider_policy)?;
        let contract = FrozenLensContract::new(
            name.into(),
            weights_sha256,
            sha256_digest(&[
                b"fastembed-bgem3-v1",
                info.model_code.as_bytes(),
                bgem3_corpus_token(output),
            ]),
            bgem3_shape(output),
            Modality::Text,
            LensDType::F32,
            bgem3_norm(output),
        );
        Ok(Self {
            id: contract.lens_id(),
            output,
            engine: Bgem3Engine::FastembedCpu,
            contract,
            files,
            runtime,
        })
    }

    pub fn from_lens_spec(spec: &LensSpec) -> Result<Self> {
        let LensRuntime::FastembedBgem3 {
            model_id,
            files,
            output,
            engine,
        } = &spec.runtime
        else {
            return Err(super::super::config_invalid(
                "LensSpec runtime is not fastembed-bgem3",
            ));
        };
        match engine {
            Bgem3Engine::FastembedCpu => Err(CalyxError::lens_unreachable(format!(
                "CALYX_BGE_M3_CPU_GRAPH_GPU_PLACEMENT: persisted lens {} uses the CPU-only FastEmbed gpahal/bge-m3-onnx-int8 graph and cannot start in the CUDA resident; commission a pinned onnx-bgem3-* artifact instead",
                spec.name
            ))),
            Bgem3Engine::OnnxCuda => {
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = (model_id, files, output);
                    Err(CalyxError::lens_unreachable(format!(
                        "CALYX_BGE_M3_CUDA_FEATURE_MISSING: persisted lens {} requires the calyx-registry `cuda` feature",
                        spec.name
                    )))
                }
                #[cfg(feature = "cuda")]
                {
                    let prepared = device::DeviceBgem3Runtime::prepare(spec, model_id, files)?;
                    let key = device_runtime_key(&prepared)?;
                    let artifact_files = prepared.files.clone();
                    let runtime = shared_device_runtime(key, prepared)?;
                    let contract = crate::derive_runtime_contract_from_spec(spec)?;
                    ensure_spec_match(contract.shape(), contract.weights_sha256(), spec)?;
                    Ok(Self {
                        id: contract.lens_id(),
                        output: *output,
                        engine: *engine,
                        contract,
                        files: artifact_files,
                        runtime,
                    })
                }
            }
        }
    }

    pub fn contract(&self) -> &FrozenLensContract {
        &self.contract
    }

    pub fn files(&self) -> &OnnxModelFiles {
        &self.files
    }

    pub fn provider_policy(&self) -> &'static str {
        self.runtime.provider_policy.as_str()
    }

    pub fn runtime_name(&self) -> &'static str {
        bgem3_runtime_name(self.output, self.engine)
    }

    pub fn runtime_stats(&self) -> Bgem3RuntimeStats {
        Bgem3RuntimeStats {
            session_initializations: 1,
            forward_calls: self.runtime.forward_calls.load(Ordering::Relaxed),
            tokenization_calls: self.runtime.tokenization_calls.load(Ordering::Relaxed),
            dense_conversions: self.runtime.dense_conversions.load(Ordering::Relaxed),
            sparse_conversions: self.runtime.sparse_conversions.load(Ordering::Relaxed),
            colbert_conversions: self.runtime.colbert_conversions.load(Ordering::Relaxed),
            max_concurrent_runs: self.runtime.max_concurrent_runs.load(Ordering::Relaxed),
        }
    }

    fn measure_requested(
        &self,
        requests: &[GroupedLensRequest],
        inputs: &[Input],
    ) -> Result<BTreeMap<LensId, Vec<SlotVector>>> {
        let mut outputs = BTreeSet::new();
        let mut lens_ids = BTreeSet::new();
        for request in requests {
            if !lens_ids.insert(request.lens_id) {
                return Err(CalyxError::lens_dim_mismatch(format!(
                    "BGE-M3 grouped request repeats lens {}",
                    request.lens_id
                )));
            }
            outputs.insert(output_for_shape(request.shape)?);
        }
        if inputs.is_empty() {
            return Ok(requests
                .iter()
                .map(|request| (request.lens_id, Vec::new()))
                .collect());
        }
        let texts = input_texts(self, inputs)?;
        let converted = match &self.runtime.backend {
            SharedBgem3Backend::Fastembed(model) => {
                if self.runtime.provider_policy == OnnxProviderPolicy::CudaFailLoud {
                    return Err(CalyxError::lens_unreachable(
                        "CALYX_BGE_M3_CPU_GRAPH_GPU_PLACEMENT: CPU-only FastEmbed BGE-M3 reached a CUDA fail-loud execution path",
                    ));
                }
                let mut model = lock_model(model, "BGE-M3 shared")?;
                let _active_run = self.start_active_run();
                self.runtime
                    .tokenization_calls
                    .fetch_add(1, Ordering::Relaxed);
                self.runtime.forward_calls.fetch_add(1, Ordering::Relaxed);
                let output = model.embed(texts, None).map_err(|error| {
                    CalyxError::lens_unreachable(format!(
                        "BGE-M3 grouped inference failed for {} projections: {error}",
                        requests.len()
                    ))
                })?;
                let mut converted = BTreeMap::new();
                if outputs.contains(&FastembedBgem3Output::Dense) {
                    self.runtime
                        .dense_conversions
                        .fetch_add(1, Ordering::Relaxed);
                    converted.insert(
                        FastembedBgem3Output::Dense,
                        dense_batch(output.dense, BGE_M3_DENSE_DIM, inputs.len())?,
                    );
                }
                if outputs.contains(&FastembedBgem3Output::Sparse) {
                    self.runtime
                        .sparse_conversions
                        .fetch_add(1, Ordering::Relaxed);
                    converted.insert(
                        FastembedBgem3Output::Sparse,
                        sparse_batch(output.sparse, BGE_M3_SPARSE_DIM, inputs.len())?,
                    );
                }
                if outputs.contains(&FastembedBgem3Output::Colbert) {
                    self.runtime
                        .colbert_conversions
                        .fetch_add(1, Ordering::Relaxed);
                    converted.insert(
                        FastembedBgem3Output::Colbert,
                        multi_batch(output.colbert, BGE_M3_DENSE_DIM, inputs.len())?,
                    );
                }
                converted
            }
            #[cfg(feature = "cuda")]
            SharedBgem3Backend::OnnxCuda(model) => {
                let mut model = model.lock().map_err(|_| {
                    CalyxError::lens_unreachable("CUDA BGE-M3 shared runtime mutex was poisoned")
                })?;
                let _active_run = self.start_active_run();
                self.runtime
                    .tokenization_calls
                    .fetch_add(1, Ordering::Relaxed);
                let measured = model.measure(self, &outputs, inputs)?;
                self.runtime
                    .forward_calls
                    .fetch_add(measured.forward_calls, Ordering::Relaxed);
                for output in &outputs {
                    self.conversion_counter(*output)
                        .fetch_add(measured.forward_calls, Ordering::Relaxed);
                }
                measured.outputs
            }
        };
        let mut measured = BTreeMap::new();
        for request in requests {
            let output = output_for_shape(request.shape)?;
            let vectors = converted.get(&output).ok_or_else(|| {
                CalyxError::lens_dim_mismatch(format!(
                    "BGE-M3 requested output was not converted for lens {}",
                    request.lens_id
                ))
            })?;
            if measured.insert(request.lens_id, vectors.clone()).is_some() {
                return Err(CalyxError::lens_dim_mismatch(format!(
                    "BGE-M3 grouped request repeats lens {}",
                    request.lens_id
                )));
            }
        }
        Ok(measured)
    }

    fn start_active_run(&self) -> ActiveRunGuard<'_> {
        let active = self.runtime.active_runs.fetch_add(1, Ordering::SeqCst) + 1;
        self.runtime
            .max_concurrent_runs
            .fetch_max(active, Ordering::SeqCst);
        ActiveRunGuard(&self.runtime.active_runs)
    }

    #[cfg(feature = "cuda")]
    fn conversion_counter(&self, output: FastembedBgem3Output) -> &AtomicU64 {
        match output {
            FastembedBgem3Output::Dense => &self.runtime.dense_conversions,
            FastembedBgem3Output::Sparse => &self.runtime.sparse_conversions,
            FastembedBgem3Output::Colbert => &self.runtime.colbert_conversions,
        }
    }
}

struct ActiveRunGuard<'a>(&'a AtomicU64);

impl Drop for ActiveRunGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Lens for FastembedBgem3Lens {
    fn id(&self) -> LensId {
        self.id
    }

    fn shape(&self) -> SlotShape {
        self.contract.shape()
    }

    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn measure(&self, input: &Input) -> Result<SlotVector> {
        let mut batch = self.measure_batch(std::slice::from_ref(input))?;
        batch.pop().ok_or_else(|| {
            CalyxError::lens_dim_mismatch(format!("lens {} returned no vector", self.id))
        })
    }

    fn measure_batch(&self, inputs: &[Input]) -> Result<Vec<SlotVector>> {
        self.measure_requested(
            &[GroupedLensRequest {
                lens_id: self.id,
                shape: self.shape(),
            }],
            inputs,
        )?
        .remove(&self.id)
        .ok_or_else(|| CalyxError::lens_dim_mismatch("BGE-M3 omitted standalone output"))
    }

    fn measurement_group_key(&self) -> Result<Option<MeasurementGroupKey>> {
        Ok(Some(self.runtime.key))
    }

    fn measure_grouped_batch(
        &self,
        requests: &[GroupedLensRequest],
        inputs: &[Input],
    ) -> Result<Option<BTreeMap<LensId, Vec<SlotVector>>>> {
        Ok(Some(self.measure_requested(requests, inputs)?))
    }
}

impl Drop for SharedBgem3Runtime {
    fn drop(&mut self) {
        match &mut self.backend {
            SharedBgem3Backend::Fastembed(model) => {
                leak_cuda_model(model, self.provider_policy);
            }
            #[cfg(feature = "cuda")]
            SharedBgem3Backend::OnnxCuda(_) => {}
        }
    }
}

#[cfg(test)]
mod tests;
