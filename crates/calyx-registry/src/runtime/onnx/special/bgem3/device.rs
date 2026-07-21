use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, Input, Lens, Result, SlotVector, SparseEntry};
use ort::session::SessionOutputs;
use ort::value::ValueType;
use tokenizers::Tokenizer;

use super::super::super::cuda_guard::CudaDropGuard;
use super::super::super::custom::batch::{
    TokenBatch, max_tokens_from_config, session_inputs, token_batches,
};
use super::super::super::device_postprocess::{
    cuda_postprocess_context, device_tensor, forge_error, tensor_data_ptr, tensor_shape,
};
use super::super::super::io_binding::OnnxRunPlan;
use super::super::super::session::{ManagedOnnxSession, build_session};
use super::super::super::{OnnxModelFiles, OnnxProviderPolicy, config_invalid};
use super::super::models::{BGE_M3_DENSE_DIM, BGE_M3_SPARSE_DIM};
use crate::runtime::common::hash_files;
use crate::{FastembedBgem3Output, LensSpec};

const DENSE_OUTPUT: &str = "dense_vecs";
const SPARSE_OUTPUT: &str = "sparse_vecs";
const COLBERT_OUTPUT: &str = "colbert_vecs";

pub(super) struct DeviceBgem3Prepared {
    pub(super) files: OnnxModelFiles,
    pub(super) weights_sha256: [u8; 32],
    pub(super) max_tokens: usize,
    pub(super) max_batch: Option<usize>,
    pub(super) model_id: String,
}

pub(super) struct DeviceMeasureResult {
    pub(super) outputs: BTreeMap<FastembedBgem3Output, Vec<SlotVector>>,
    pub(super) forward_calls: u64,
}

pub(super) struct DeviceBgem3Runtime {
    session: Option<ManagedOnnxSession>,
    run_plan: OnnxRunPlan,
    tokenizer: Tokenizer,
    cuda: calyx_forge::CudaContext,
    max_tokens: usize,
    max_batch: Option<usize>,
}

impl DeviceBgem3Runtime {
    pub(super) fn prepare(
        spec: &LensSpec,
        model_id: &str,
        paths: &[PathBuf],
    ) -> Result<DeviceBgem3Prepared> {
        ensure_model_id(model_id)?;
        if spec.max_batch == Some(0) {
            return Err(config_invalid("BGE-M3 max_batch must be greater than zero"));
        }
        let files = model_files(model_id, paths)?;
        let artifacts = files.artifact_paths();
        let weights_sha256 = hash_files(&artifacts)?;
        if weights_sha256 != spec.weights_sha256 {
            return Err(CalyxError::lens_frozen_violation(format!(
                "CUDA BGE-M3 artifact hash drift for {}: runtime={} declared={}",
                spec.name,
                hex(&weights_sha256),
                hex(&spec.weights_sha256)
            )));
        }
        let config = read_config(&files.config)?;
        let max_tokens = max_tokens_from_config(&config)?;
        Ok(DeviceBgem3Prepared {
            files,
            weights_sha256,
            max_tokens,
            max_batch: spec.max_batch,
            model_id: model_id.to_string(),
        })
    }

    pub(super) fn measure(
        &mut self,
        lens: &dyn Lens,
        requested: &BTreeSet<FastembedBgem3Output>,
        inputs: &[Input],
    ) -> Result<DeviceMeasureResult> {
        let batches = token_batches(
            &self.tokenizer,
            lens,
            inputs,
            self.max_tokens,
            self.max_batch,
            self.run_plan.pads_batches(),
        )?;
        let mut rows = requested
            .iter()
            .map(|output| (*output, vec![None; inputs.len()]))
            .collect::<BTreeMap<_, _>>();
        let mut forward_calls = 0_u64;
        for batch in &batches {
            let measured = self.run_batch(batch, requested)?;
            forward_calls = forward_calls.saturating_add(1);
            for output in requested {
                let vectors = measured.get(output).ok_or_else(|| {
                    CalyxError::lens_dim_mismatch(format!(
                        "CUDA BGE-M3 omitted requested {output:?} batch"
                    ))
                })?;
                if vectors.len() != batch.batch {
                    return Err(CalyxError::lens_dim_mismatch(format!(
                        "CUDA BGE-M3 {output:?} returned {} rows for padded batch {}",
                        vectors.len(),
                        batch.batch
                    )));
                }
                let output_rows = rows.get_mut(output).ok_or_else(|| {
                    CalyxError::lens_dim_mismatch("CUDA BGE-M3 output row map disappeared")
                })?;
                for (input_index, vector) in batch.indices.iter().copied().zip(vectors.iter()) {
                    if output_rows[input_index].replace(vector.clone()).is_some() {
                        return Err(CalyxError::lens_dim_mismatch(format!(
                            "CUDA BGE-M3 measured input {input_index} twice for {output:?}"
                        )));
                    }
                }
            }
        }
        let outputs = rows
            .into_iter()
            .map(|(output, rows)| {
                rows.into_iter()
                    .enumerate()
                    .map(|(index, row)| {
                        row.ok_or_else(|| {
                            CalyxError::lens_dim_mismatch(format!(
                                "CUDA BGE-M3 omitted input {index} for {output:?}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(|rows| (output, rows))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(DeviceMeasureResult {
            outputs,
            forward_calls,
        })
    }

    fn run_batch(
        &mut self,
        batch: &TokenBatch,
        requested: &BTreeSet<FastembedBgem3Output>,
    ) -> Result<BTreeMap<FastembedBgem3Output, Vec<SlotVector>>> {
        let session = self.session.as_mut().ok_or_else(|| {
            CalyxError::lens_unreachable("CUDA BGE-M3 ONNX session is unavailable")
        })?;
        let input_tensors = session_inputs(session.as_ref(), batch)?;
        let cuda = self.cuda.clone();
        self.run_plan.run_extract_device(
            session.as_mut(),
            input_tensors,
            (batch.batch, batch.seq),
            |outputs| convert_outputs(outputs, batch, requested, &cuda),
        )
    }
}

impl DeviceBgem3Prepared {
    pub(super) fn build(self) -> Result<DeviceBgem3Runtime> {
        let artifacts = self.files.artifact_paths();
        let label = format!("onnx-bgem3:{}", self.model_id);
        super::super::super::arena::preflight_gpu_mem_limit_for_artifacts(
            &label,
            OnnxProviderPolicy::CudaFailLoud,
            artifacts.iter().map(PathBuf::as_path),
        )?;
        let session = build_session(
            &label,
            &self.files.model_file,
            OnnxProviderPolicy::CudaFailLoud,
        )?;
        validate_session_contract(session.as_ref())?;
        let session = CudaDropGuard::new(session, OnnxProviderPolicy::CudaFailLoud).into_inner();
        let run_plan = OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, label)?;
        let device_id = super::super::super::session::configured_cuda_device()?;
        let cuda = cuda_postprocess_context(OnnxProviderPolicy::CudaFailLoud, device_id)?
            .ok_or_else(|| config_invalid("CUDA BGE-M3 did not initialize a Forge CUDA context"))?;
        let tokenizer = Tokenizer::from_file(&self.files.tokenizer)
            .map_err(|error| config_invalid(format!("load BGE-M3 tokenizer failed: {error}")))?;
        Ok(DeviceBgem3Runtime {
            session: Some(session),
            run_plan,
            tokenizer,
            cuda,
            max_tokens: self.max_tokens,
            max_batch: self.max_batch,
        })
    }
}

impl Drop for DeviceBgem3Runtime {
    fn drop(&mut self) {
        // ORT CUDA teardown can execute after CUDA libraries have begun
        // process shutdown. Match the other fail-loud ONNX runtimes and keep
        // the verified session alive until process exit.
        if let Some(session) = self.session.take() {
            std::mem::forget(session);
        }
    }
}

fn convert_outputs(
    outputs: &SessionOutputs<'_>,
    batch: &TokenBatch,
    requested: &BTreeSet<FastembedBgem3Output>,
    cuda: &calyx_forge::CudaContext,
) -> Result<BTreeMap<FastembedBgem3Output, Vec<SlotVector>>> {
    let mut converted = BTreeMap::new();
    for output in requested {
        let vectors = match output {
            FastembedBgem3Output::Dense => dense_vectors(outputs, batch, cuda)?,
            FastembedBgem3Output::Sparse => sparse_vectors(outputs, batch, cuda)?,
            FastembedBgem3Output::Colbert => colbert_vectors(outputs, batch, cuda)?,
        };
        converted.insert(*output, vectors);
    }
    Ok(converted)
}

fn dense_vectors(
    outputs: &SessionOutputs<'_>,
    batch: &TokenBatch,
    cuda: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let (shape, ptr) = output_device_tensor(outputs, DENSE_OUTPUT)?;
    expect_shape(
        &shape,
        &[batch.batch, BGE_M3_DENSE_DIM as usize],
        DENSE_OUTPUT,
    )?;
    let flat = calyx_forge::cuda::dense_2d_from_external_f32(
        cuda,
        ptr,
        batch.batch,
        BGE_M3_DENSE_DIM as usize,
        true,
    )
    .map_err(forge_error)?;
    Ok(flat
        .chunks_exact(BGE_M3_DENSE_DIM as usize)
        .map(|row| SlotVector::Dense {
            dim: BGE_M3_DENSE_DIM,
            data: row.to_vec(),
        })
        .collect())
}

fn sparse_vectors(
    outputs: &SessionOutputs<'_>,
    batch: &TokenBatch,
    cuda: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let (shape, ptr) = output_device_tensor(outputs, SPARSE_OUTPUT)?;
    expect_shape(&shape, &[batch.batch, batch.seq, 1], SPARSE_OUTPUT)?;
    let rows = calyx_forge::cuda::bgem3_sparse_from_external_f32(
        cuda,
        ptr,
        &batch.ids,
        &batch.mask,
        batch.batch,
        batch.seq,
        BGE_M3_SPARSE_DIM as usize,
    )
    .map_err(forge_error)?;
    Ok(rows
        .rows
        .into_iter()
        .map(|row| SlotVector::Sparse {
            dim: BGE_M3_SPARSE_DIM,
            entries: row
                .into_iter()
                .map(|(idx, val)| SparseEntry { idx, val })
                .collect(),
        })
        .collect())
}

fn colbert_vectors(
    outputs: &SessionOutputs<'_>,
    batch: &TokenBatch,
    cuda: &calyx_forge::CudaContext,
) -> Result<Vec<SlotVector>> {
    let (shape, ptr) = output_device_tensor(outputs, COLBERT_OUTPUT)?;
    let seq = batch.seq.checked_sub(1).ok_or_else(|| {
        CalyxError::lens_dim_mismatch("CUDA BGE-M3 ColBERT requires at least one input token")
    })?;
    expect_shape(
        &shape,
        &[batch.batch, seq, BGE_M3_DENSE_DIM as usize],
        COLBERT_OUTPUT,
    )?;
    let shifted_mask = batch
        .mask
        .chunks_exact(batch.seq)
        .flat_map(|row| row.iter().skip(1).copied())
        .collect::<Vec<_>>();
    let rows = calyx_forge::cuda::bgem3_colbert_tokens_from_external_f32(
        cuda,
        ptr,
        &shifted_mask,
        batch.batch,
        seq,
        BGE_M3_DENSE_DIM as usize,
    )
    .map_err(forge_error)?;
    Ok(rows
        .rows
        .into_iter()
        .map(|tokens| SlotVector::Multi {
            token_dim: BGE_M3_DENSE_DIM,
            tokens,
        })
        .collect())
}

fn output_device_tensor(outputs: &SessionOutputs<'_>, name: &str) -> Result<(Vec<i64>, u64)> {
    let value = outputs
        .get(name)
        .ok_or_else(|| config_invalid(format!("CUDA BGE-M3 output {name} is missing")))?;
    let tensor = device_tensor(value, name)?;
    let shape = tensor_shape(&tensor, name)?;
    let ptr = tensor_data_ptr(&tensor, &shape, name)?;
    Ok((shape, ptr))
}

fn expect_shape(actual: &[i64], expected: &[usize], label: &str) -> Result<()> {
    let actual = actual
        .iter()
        .map(|dim| usize::try_from(*dim).ok())
        .collect::<Option<Vec<_>>>();
    if actual.as_deref() == Some(expected) {
        return Ok(());
    }
    Err(CalyxError::lens_dim_mismatch(format!(
        "CUDA BGE-M3 {label} shape {actual:?} != expected {expected:?}"
    )))
}

fn validate_session_contract(session: &ort::session::Session) -> Result<()> {
    for (name, final_dim) in [
        (DENSE_OUTPUT, BGE_M3_DENSE_DIM as i64),
        (SPARSE_OUTPUT, 1),
        (COLBERT_OUTPUT, BGE_M3_DENSE_DIM as i64),
    ] {
        let output = session
            .outputs()
            .iter()
            .find(|output| output.name() == name)
            .ok_or_else(|| config_invalid(format!("BGE-M3 ONNX graph is missing output {name}")))?;
        let ValueType::Tensor { shape, .. } = output.dtype() else {
            return Err(config_invalid(format!(
                "BGE-M3 output {name} is not a tensor"
            )));
        };
        if shape.last().copied() != Some(final_dim) {
            return Err(config_invalid(format!(
                "BGE-M3 output {name} final dimension {:?} != {final_dim}",
                shape.last()
            )));
        }
    }
    Ok(())
}

fn model_files(model_id: &str, paths: &[PathBuf]) -> Result<OnnxModelFiles> {
    let [model_file, tokenizer, config, rest @ ..] = paths else {
        return Err(config_invalid(
            "OnnxCuda BGE-M3 requires ordered model, tokenizer, config, and model data artifacts",
        ));
    };
    for path in paths {
        if !path.is_file() {
            return Err(config_invalid(format!(
                "CUDA BGE-M3 artifact {} is missing",
                path.display()
            )));
        }
    }
    if model_file.extension().and_then(|value| value.to_str()) != Some("onnx") {
        return Err(config_invalid(format!(
            "CUDA BGE-M3 first artifact {} must be an ONNX model",
            model_file.display()
        )));
    }
    let cache_dir = model_file
        .parent()
        .ok_or_else(|| config_invalid("CUDA BGE-M3 model has no parent directory"))?
        .to_path_buf();
    if paths
        .iter()
        .any(|path| path.parent() != Some(cache_dir.as_path()))
    {
        return Err(config_invalid(
            "CUDA BGE-M3 artifacts must share one directory so ONNX external data is resolved atomically",
        ));
    }
    let tokenizer_config = rest
        .iter()
        .find(|path| file_name(path) == "tokenizer_config.json")
        .cloned()
        .unwrap_or_else(|| tokenizer.clone());
    let special_tokens_map = rest
        .iter()
        .find(|path| file_name(path) == "special_tokens_map.json")
        .cloned()
        .unwrap_or_else(|| config.clone());
    Ok(OnnxModelFiles {
        cache_dir,
        model_code: model_id.to_string(),
        model_file: model_file.clone(),
        tokenizer: tokenizer.clone(),
        config: config.clone(),
        special_tokens_map,
        tokenizer_config,
        contract_paths: paths.to_vec(),
    })
}

fn read_config(path: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(path).map_err(|error| {
        config_invalid(format!(
            "read BGE-M3 config {} failed: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        config_invalid(format!(
            "parse BGE-M3 config {} failed: {error}",
            path.display()
        ))
    })
}

fn ensure_model_id(raw: &str) -> Result<()> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "baai/bge-m3" | "bge-m3" => Ok(()),
        other => Err(CalyxError::lens_unreachable(format!(
            "unsupported CUDA BGE-M3 ONNX model {other}; expected BAAI/bge-m3"
        ))),
    }
}

fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
