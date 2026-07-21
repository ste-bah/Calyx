use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, Result};
use serde_json::Value;

use super::{DEFAULT_ANSWERAI_COLBERT_MODEL, OnnxColbertFileSpec};
use crate::runtime::onnx::{OnnxModelFiles, config_invalid};

pub(super) fn answerai_colbert_model_id(raw: &str) -> Result<String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "answerdotai/answerai-colbert-small-v1" | "answerai-colbert-small-v1" => {
            Ok(DEFAULT_ANSWERAI_COLBERT_MODEL.to_string())
        }
        other => Err(CalyxError::lens_unreachable(format!(
            "unsupported onnx-colbert model {other}; expected {DEFAULT_ANSWERAI_COLBERT_MODEL}"
        ))),
    }
}

pub(super) fn model_files(spec: &OnnxColbertFileSpec) -> OnnxModelFiles {
    let cache_dir = spec
        .model_file
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    OnnxModelFiles {
        cache_dir,
        model_code: spec.model_id.clone(),
        model_file: spec.model_file.clone(),
        tokenizer: spec.tokenizer.clone(),
        config: spec.config.clone(),
        special_tokens_map: spec.config.clone(),
        tokenizer_config: spec.tokenizer.clone(),
        contract_paths: spec.contract_paths.clone(),
    }
}

pub(super) fn ensure_file(label: &str, path: &Path) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    Err(config_invalid(format!(
        "ONNX ColBERT {label} file {} is missing",
        path.display()
    )))
}

pub(super) fn validate_config(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|err| {
        config_invalid(format!(
            "read ONNX ColBERT config {} failed: {err}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|err| {
        config_invalid(format!(
            "parse ONNX ColBERT config {} failed: {err}",
            path.display()
        ))
    })
}
