use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use calyx_core::{CalyxError, Result};
use serde::Deserialize;

use super::axis::MultimodalAxis;

const ADAPTER_SCHEMA: &str = "calyx-multimodal-adapter-v2";
const ENGINE_ONNX_EXTERNAL: &str = "onnx-external";
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_MAX_BATCH: usize = 32;
const PROVIDER_CPU_EXPLICIT: &str = "cpu_explicit";
const PROVIDER_CUDA_FAIL_LOUD: &str = "cuda_fail_loud";
const PROVIDER_TENSORRT_CUDA_FAIL_LOUD: &str = "tensorrt_cuda_fail_loud";
const PROVIDER_CUDA_DETAIL: &str = "cuda:0,error_on_failure,no_cpu_fallback";
const PROVIDER_TENSORRT_CUDA_DETAIL: &str = "tensorrt:0,cuda:0,error_on_failure,no_cpu_fallback";
const BATCH_POLICY_DYNAMIC_PADDED: &str = "dynamic_padded";

pub const CALYX_MULTIMODAL_ALLOW_CPU_ADAPTER_ENV: &str = "CALYX_MULTIMODAL_ALLOW_CPU_ADAPTER";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MultimodalAdapterProvider {
    CpuExplicit,
    CudaFailLoud,
    TensorRtCudaFailLoud,
}

impl MultimodalAdapterProvider {
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim() {
            PROVIDER_CPU_EXPLICIT => Ok(Self::CpuExplicit),
            PROVIDER_CUDA_FAIL_LOUD | PROVIDER_CUDA_DETAIL => Ok(Self::CudaFailLoud),
            "cuda_preferred" | "cuda:0,allow_cpu_fallback" => Err(config_invalid(
                "unsupported multimodal adapter provider cuda_preferred: CPU fallback is forbidden; use cuda_fail_loud or an audited cpu_explicit override",
            )),
            PROVIDER_TENSORRT_CUDA_FAIL_LOUD | PROVIDER_TENSORRT_CUDA_DETAIL => {
                Ok(Self::TensorRtCudaFailLoud)
            }
            other => Err(config_invalid(format!(
                "unsupported multimodal adapter provider {other}"
            ))),
        }
    }

    pub const fn config_value(self) -> &'static str {
        match self {
            Self::CpuExplicit => PROVIDER_CPU_EXPLICIT,
            Self::CudaFailLoud => PROVIDER_CUDA_FAIL_LOUD,
            Self::TensorRtCudaFailLoud => PROVIDER_TENSORRT_CUDA_FAIL_LOUD,
        }
    }

    pub const fn detail(self) -> &'static str {
        match self {
            Self::CpuExplicit => "cpu_explicit,no_cuda",
            Self::CudaFailLoud => PROVIDER_CUDA_DETAIL,
            Self::TensorRtCudaFailLoud => PROVIDER_TENSORRT_CUDA_DETAIL,
        }
    }

    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::CudaFailLoud | Self::TensorRtCudaFailLoud)
    }
}

#[derive(Clone, Debug)]
pub struct MultimodalAdapterConfig {
    pub path: PathBuf,
    pub axis: MultimodalAxis,
    pub model_id: String,
    pub processor_model_id: String,
    pub dim: u32,
    pub command: String,
    pub helper: PathBuf,
    pub model_file: PathBuf,
    pub provider: MultimodalAdapterProvider,
    pub max_batch: usize,
    pub batch_policy: String,
    pub timeout: Duration,
}

#[derive(Deserialize)]
struct RawAdapterConfig {
    schema: String,
    engine: String,
    axis: String,
    model_id: String,
    #[serde(default)]
    processor_model_id: Option<String>,
    dim: u32,
    #[serde(default)]
    python: Option<String>,
    helper: PathBuf,
    model_file: PathBuf,
    #[serde(default = "default_provider")]
    provider: String,
    #[serde(default = "default_max_batch")]
    max_batch: usize,
    #[serde(default = "default_batch_policy")]
    batch_policy: String,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

pub fn load_adapter_config(
    path: &Path,
    expected_axis: MultimodalAxis,
    expected_model_id: &str,
    expected_dim: Option<u32>,
) -> Result<MultimodalAdapterConfig> {
    let bytes = fs::read(path).map_err(|err| {
        config_invalid(format!(
            "read multimodal adapter config {} failed: {err}",
            path.display()
        ))
    })?;
    let raw: RawAdapterConfig = serde_json::from_slice(&bytes).map_err(|err| {
        config_invalid(format!(
            "parse multimodal adapter config {} failed: {err}",
            path.display()
        ))
    })?;
    if raw.schema != ADAPTER_SCHEMA {
        return Err(config_invalid(format!(
            "unsupported multimodal adapter schema {}",
            raw.schema
        )));
    }
    if raw.engine != ENGINE_ONNX_EXTERNAL {
        return Err(config_invalid(format!(
            "unsupported multimodal adapter engine {}",
            raw.engine
        )));
    }
    let axis = MultimodalAxis::parse(&raw.axis)?;
    if axis != expected_axis {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "multimodal adapter config axis {} != expected {}",
            axis.as_str(),
            expected_axis.as_str()
        )));
    }
    if raw.model_id != expected_model_id {
        return Err(CalyxError::lens_frozen_violation(format!(
            "multimodal adapter config model {} != expected {}",
            raw.model_id, expected_model_id
        )));
    }
    if raw.dim == 0 {
        return Err(config_invalid("multimodal adapter config dim must be > 0"));
    }
    if let Some(expected) = expected_dim
        && raw.dim != expected
    {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "multimodal adapter config dim {} != expected {}",
            raw.dim, expected
        )));
    }
    let provider = MultimodalAdapterProvider::parse(&raw.provider)?;
    if raw.max_batch == 0 {
        return Err(config_invalid("multimodal adapter max_batch must be > 0"));
    }
    if raw.batch_policy != BATCH_POLICY_DYNAMIC_PADDED {
        return Err(config_invalid(format!(
            "unsupported multimodal adapter batch_policy {}",
            raw.batch_policy
        )));
    }
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let helper = resolve_path(base, raw.helper);
    let model_file = resolve_path(base, raw.model_file);
    ensure_file("helper", &helper)?;
    ensure_file("model", &model_file)?;
    Ok(MultimodalAdapterConfig {
        path: path.to_path_buf(),
        axis,
        model_id: raw.model_id,
        processor_model_id: raw
            .processor_model_id
            .unwrap_or_else(|| expected_model_id.to_string()),
        dim: raw.dim,
        command: raw.python.unwrap_or_else(|| "python3".to_string()),
        helper,
        model_file,
        provider,
        max_batch: raw.max_batch,
        batch_policy: raw.batch_policy,
        timeout: Duration::from_millis(raw.timeout_ms),
    })
}

impl MultimodalAdapterConfig {
    pub fn contract_paths(&self) -> Vec<PathBuf> {
        vec![
            self.model_file.clone(),
            self.path.clone(),
            self.helper.clone(),
        ]
    }
}

fn default_provider() -> String {
    PROVIDER_CUDA_FAIL_LOUD.to_string()
}

const fn default_max_batch() -> usize {
    DEFAULT_MAX_BATCH
}

fn default_batch_policy() -> String {
    BATCH_POLICY_DYNAMIC_PADDED.to_string()
}

const fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

fn resolve_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn ensure_file(label: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    Err(config_invalid(format!(
        "multimodal adapter {label} file {} is missing",
        path.display()
    )))
}

pub fn config_invalid(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: "CALYX_LENS_CONFIG_INVALID",
        message: message.into(),
        remediation: "fix the multimodal adapter lens spec",
    }
}
