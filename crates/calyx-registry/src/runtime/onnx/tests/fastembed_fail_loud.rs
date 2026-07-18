use std::path::PathBuf;

use calyx_core::{Lens, Modality};

use crate::frozen::FrozenLensContract;

use super::super::{OnnxLens, OnnxModelFiles, OnnxProviderPolicy, fastembed_runtime};

#[test]
fn empty_batch_short_circuits_before_backend() {
    for provider_policy in [
        OnnxProviderPolicy::CudaFailLoud,
        OnnxProviderPolicy::CpuExplicit,
    ] {
        let lens = empty_batch_lens(provider_policy);
        println!(
            "ISSUE1570_EMPTY_BATCH_BEFORE provider={} backend_present={}",
            provider_policy.as_str(),
            lens.backend.is_some()
        );

        let out = lens.measure_batch(&[]).unwrap();

        println!(
            "ISSUE1570_EMPTY_BATCH_AFTER provider={} output_len={} backend_present={}",
            provider_policy.as_str(),
            out.len(),
            lens.backend.is_some()
        );
        assert!(out.is_empty());
        assert!(lens.backend.is_none());
    }
}

#[test]
fn postprocess_error_names_host_output_root_cause() {
    for runtime in [
        "onnx-fastembed",
        "fastembed-sparse",
        "fastembed-reranker",
        "fastembed-bgem3",
    ] {
        let error = fastembed_runtime::device_postprocess_unavailable(runtime);
        println!(
            "ISSUE1570_FAIL_LOUD_ERROR runtime={runtime} code={} message={} remediation={}",
            error.code, error.message, error.remediation
        );
        assert_eq!(error.code, "CALYX_LENS_DEVICE_POSTPROCESS_UNAVAILABLE");
        assert!(error.message.contains(runtime));
        assert!(error.message.contains("FastEmbed 5.16 host-owned outputs"));
        assert!(error.message.contains("CudaFailLoud"));
        assert!(error.remediation.contains("Calyx-owned"));
        assert!(error.remediation.contains("CUDA output buffers"));
    }
}

fn empty_batch_lens(provider_policy: OnnxProviderPolicy) -> OnnxLens {
    let contract = FrozenLensContract::tei_http(
        "issue1570-empty-batch",
        "fastembed://issue1570-empty-batch",
        Modality::Text,
        3,
    );
    OnnxLens {
        id: contract.lens_id(),
        dim: 3,
        contract,
        files: OnnxModelFiles {
            cache_dir: PathBuf::new(),
            model_code: "issue1570-empty-batch".to_string(),
            model_file: PathBuf::new(),
            tokenizer: PathBuf::new(),
            config: PathBuf::new(),
            special_tokens_map: PathBuf::new(),
            tokenizer_config: PathBuf::new(),
            contract_paths: Vec::new(),
        },
        provider_policy,
        max_batch: None,
        backend: None,
    }
}
