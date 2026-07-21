use std::path::PathBuf;
use std::sync::Mutex;

use calyx_core::{CalyxError, Input, Lens, LensId, Modality, Result, SlotShape, SlotVector};
use fastembed::{
    RerankInitOptions, RerankerModel, SparseInitOptions, SparseModel, SparseTextEmbedding,
    TextRerank,
};

use super::cuda_guard::CudaDropGuard;
use super::{OnnxModelFiles, OnnxProviderPolicy};
use crate::frozen::{FrozenLensContract, NormPolicy};
use crate::spec::{LensRuntime, LensSpec};

mod bgem3;
mod models;
mod vectors;

pub use bgem3::{Bgem3RuntimeStats, FastembedBgem3Lens};

use models::{reranker_model_from_name, sparse_dim, sparse_model_from_name};
use vectors::{
    contract, ensure_spec_match, input_texts, leak_cuda_model, lock_model, rerank_pair,
    single_vector, sparse_batch, sparse_shape_dim, special_files,
};

pub struct FastembedSparseLens {
    id: LensId,
    contract: FrozenLensContract,
    files: OnnxModelFiles,
    provider_policy: OnnxProviderPolicy,
    model: Option<Mutex<SparseTextEmbedding>>,
}

pub struct FastembedRerankerLens {
    id: LensId,
    contract: FrozenLensContract,
    files: OnnxModelFiles,
    provider_policy: OnnxProviderPolicy,
    model: Option<Mutex<TextRerank>>,
}

impl FastembedSparseLens {
    pub fn from_model_name_with_policy(
        name: impl Into<String>,
        model_name: &str,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        let model_name = sparse_model_from_name(model_name)?;
        Self::from_model_with_policy(name, model_name, cache_dir, provider_policy)
    }

    pub fn from_model_with_policy(
        name: impl Into<String>,
        model_name: SparseModel,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        super::dynamic_ort::ensure_dynamic_ort(provider_policy)?;
        let name = name.into();
        let info = SparseTextEmbedding::get_model_info(&model_name);
        let model = SparseTextEmbedding::try_new(
            SparseInitOptions::new(model_name.clone())
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(false)
                .with_intra_threads(1)
                .with_execution_providers(super::fastembed_runtime::execution_providers(
                    provider_policy,
                )?),
        )
        .map_err(|err| CalyxError::lens_unreachable(format!("sparse init failed: {err}")))?;
        let model = CudaDropGuard::new(model, provider_policy);
        let files = special_files(
            &cache_dir,
            &info.model_code,
            &info.model_file,
            &info.additional_files,
        )?;
        let shape = SlotShape::Sparse(sparse_dim(&model_name));
        let contract = contract(
            name,
            &files,
            shape,
            NormPolicy::Finite,
            &[b"fastembed-sparse-v1", info.model_code.as_bytes()],
        )?;
        Ok(Self::new(
            contract,
            files,
            provider_policy,
            model.into_inner(),
        ))
    }

    pub fn from_lens_spec(spec: &LensSpec) -> Result<Self> {
        let LensRuntime::FastembedSparse { model_id, .. } = &spec.runtime else {
            return Err(super::config_invalid(
                "LensSpec runtime is not fastembed-sparse",
            ));
        };
        let lens = Self::from_model_name_with_policy(
            spec.name.clone(),
            model_id,
            super::fastembed_runtime::default_cache_root(),
            OnnxProviderPolicy::CudaFailLoud,
        )?;
        ensure_spec_match(lens.shape(), lens.contract.weights_sha256(), spec)?;
        Ok(lens)
    }

    fn new(
        contract: FrozenLensContract,
        files: OnnxModelFiles,
        provider_policy: OnnxProviderPolicy,
        model: SparseTextEmbedding,
    ) -> Self {
        Self {
            id: contract.lens_id(),
            contract,
            files,
            provider_policy,
            model: Some(Mutex::new(model)),
        }
    }

    pub fn contract(&self) -> &FrozenLensContract {
        &self.contract
    }

    pub fn files(&self) -> &OnnxModelFiles {
        &self.files
    }

    pub fn provider_policy(&self) -> &'static str {
        self.provider_policy.as_str()
    }
}

impl FastembedRerankerLens {
    pub fn from_model_name_with_policy(
        name: impl Into<String>,
        model_name: &str,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        let model_name = reranker_model_from_name(model_name)?;
        Self::from_model_with_policy(name, model_name, cache_dir, provider_policy)
    }

    pub fn from_model_with_policy(
        name: impl Into<String>,
        model_name: RerankerModel,
        cache_dir: PathBuf,
        provider_policy: OnnxProviderPolicy,
    ) -> Result<Self> {
        super::dynamic_ort::ensure_dynamic_ort(provider_policy)?;
        let name = name.into();
        let info = TextRerank::get_model_info(&model_name);
        let model = TextRerank::try_new(
            RerankInitOptions::new(model_name)
                .with_cache_dir(cache_dir.clone())
                .with_show_download_progress(false)
                .with_intra_threads(1)
                .with_execution_providers(super::fastembed_runtime::execution_providers(
                    provider_policy,
                )?),
        )
        .map_err(|err| CalyxError::lens_unreachable(format!("reranker init failed: {err}")))?;
        let model = CudaDropGuard::new(model, provider_policy);
        let files = special_files(
            &cache_dir,
            &info.model_code,
            &info.model_file,
            &info.additional_files,
        )?;
        let contract = contract(
            name,
            &files,
            SlotShape::Dense(1),
            NormPolicy::Finite,
            &[b"fastembed-reranker-v1", info.model_code.as_bytes()],
        )?;
        Ok(Self::new(
            contract,
            files,
            provider_policy,
            model.into_inner(),
        ))
    }

    pub fn from_lens_spec(spec: &LensSpec) -> Result<Self> {
        let LensRuntime::FastembedReranker { model_id, .. } = &spec.runtime else {
            return Err(super::config_invalid(
                "LensSpec runtime is not fastembed-reranker",
            ));
        };
        let lens = Self::from_model_name_with_policy(
            spec.name.clone(),
            model_id,
            super::fastembed_runtime::default_cache_root(),
            OnnxProviderPolicy::CudaFailLoud,
        )?;
        ensure_spec_match(lens.shape(), lens.contract.weights_sha256(), spec)?;
        Ok(lens)
    }

    fn new(
        contract: FrozenLensContract,
        files: OnnxModelFiles,
        provider_policy: OnnxProviderPolicy,
        model: TextRerank,
    ) -> Self {
        Self {
            id: contract.lens_id(),
            contract,
            files,
            provider_policy,
            model: Some(Mutex::new(model)),
        }
    }

    pub fn contract(&self) -> &FrozenLensContract {
        &self.contract
    }

    pub fn files(&self) -> &OnnxModelFiles {
        &self.files
    }

    pub fn provider_policy(&self) -> &'static str {
        self.provider_policy.as_str()
    }
}

impl Lens for FastembedSparseLens {
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
        single_vector(self.id, self.measure_batch(std::slice::from_ref(input))?)
    }

    fn measure_batch(&self, inputs: &[Input]) -> Result<Vec<SlotVector>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if self.provider_policy == OnnxProviderPolicy::CudaFailLoud {
            return Err(super::fastembed_runtime::device_postprocess_unavailable(
                "fastembed-sparse",
            ));
        }
        let texts = input_texts(self, inputs)?;
        let mut model = lock_model(&self.model, "sparse")?;
        let embeddings = model.embed(texts, None).map_err(|err| {
            CalyxError::lens_unreachable(format!("sparse inference failed: {err}"))
        })?;
        sparse_batch(embeddings, sparse_shape_dim(self.shape()), inputs.len())
    }
}

impl Lens for FastembedRerankerLens {
    fn id(&self) -> LensId {
        self.id
    }

    fn shape(&self) -> SlotShape {
        SlotShape::Dense(1)
    }

    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn measure(&self, input: &Input) -> Result<SlotVector> {
        single_vector(self.id, self.measure_batch(std::slice::from_ref(input))?)
    }

    fn measure_batch(&self, inputs: &[Input]) -> Result<Vec<SlotVector>> {
        if self.provider_policy == OnnxProviderPolicy::CudaFailLoud && !inputs.is_empty() {
            return Err(super::fastembed_runtime::device_postprocess_unavailable(
                "fastembed-reranker",
            ));
        }
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            let text = crate::runtime::common::text_from_input(self, input)?;
            let (query, doc) = rerank_pair(text);
            let mut model = lock_model(&self.model, "reranker")?;
            let results = model
                .rerank(query, [doc], false, Some(1))
                .map_err(|err| CalyxError::lens_unreachable(format!("rerank failed: {err}")))?;
            let score = results
                .first()
                .ok_or_else(|| CalyxError::lens_dim_mismatch("reranker returned no score"))?
                .score;
            vectors::ensure_finite("reranker score", &[score])?;
            out.push(SlotVector::Dense {
                dim: 1,
                data: vec![score],
            });
        }
        Ok(out)
    }
}

impl Drop for FastembedSparseLens {
    fn drop(&mut self) {
        leak_cuda_model(&mut self.model, self.provider_policy);
    }
}

impl Drop for FastembedRerankerLens {
    fn drop(&mut self) {
        leak_cuda_model(&mut self.model, self.provider_policy);
    }
}
