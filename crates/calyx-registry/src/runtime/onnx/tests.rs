use std::fs;

use calyx_core::{Input, Lens, Modality, SlotShape, SlotVector};
use proptest::prelude::*;

use super::custom::pool_output;
use super::*;

mod arena_env;
mod fastembed_fail_loud;
mod fixture;
mod runtime_guard;

use fixture::{Fixture, hex32, lens_error};
#[cfg(feature = "cuda")]
use fixture::{assert_close, write_onnx_fsv_readback};

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn persisted_onnx_manifest_defaults_to_cuda_fail_loud() {
    let fixture = Fixture::new("manifest-provider", &[3.0, 4.0, 0.0]);
    let spec = OnnxLens::from_files(fixture.spec("custom-provider"))
        .unwrap()
        .lens_spec();

    let file_spec = OnnxFileSpec::from_lens_spec(&spec).unwrap();

    assert_eq!(file_spec.provider_policy, OnnxProviderPolicy::CudaFailLoud);
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_from_files_measures_unit_norm_vector() {
    let fixture = Fixture::new("unit-norm", &[3.0, 4.0, 0.0]);
    let lens = OnnxLens::from_files(
        fixture
            .spec("custom-unit")
            .with_expected_shape(SlotShape::Dense(3)),
    )
    .unwrap();

    assert_eq!(lens.shape(), SlotShape::Dense(3));
    assert_eq!(lens.runtime_name(), "onnx-custom");
    let vector = lens
        .measure(&Input::new(Modality::Text, b"hello calyx".to_vec()))
        .unwrap();

    let SlotVector::Dense { dim, data } = vector else {
        panic!("expected dense custom ONNX vector");
    };
    assert_eq!(dim, 3);
    let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1.0e-6);
    assert!((data[0] - 0.6).abs() < 1.0e-6);
    assert!((data[1] - 0.8).abs() < 1.0e-6);
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_lens_spec_round_trips_runtime_files() {
    let fixture = Fixture::new("spec-roundtrip", &[3.0, 4.0, 0.0]);
    let lens = OnnxLens::from_files(fixture.spec("custom-spec")).unwrap();
    let spec = lens.lens_spec();

    let reloaded = OnnxLens::from_lens_spec(&spec).unwrap();
    assert_eq!(reloaded.id(), lens.id());
    assert_eq!(reloaded.runtime_name(), "onnx-custom");
    let vector = reloaded
        .measure(&Input::new(Modality::Text, b"calyx".to_vec()))
        .unwrap();

    lens.contract()
        .verify_vector(reloaded.id(), &vector)
        .unwrap();
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_file_hash_controls_lens_id_and_frozen_violation() {
    let fixture = Fixture::new("hash", &[3.0, 4.0, 0.0]);
    let first = OnnxLens::from_files(fixture.spec("custom-hash")).unwrap();
    let second = OnnxLens::from_files(fixture.spec("custom-hash")).unwrap();
    assert_eq!(first.id(), second.id());

    let expected = first.contract().weights_sha256();
    fs::write(
        &fixture.config,
        r#"{"model_type":"calyx-test","pooling":"cls"}"#,
    )
    .unwrap();
    let changed = OnnxLens::from_files(fixture.spec("custom-hash")).unwrap();
    assert_ne!(first.id(), changed.id());

    let error = lens_error(OnnxLens::from_files(
        fixture
            .spec("custom-hash")
            .with_expected_weights_sha256(expected),
    ));
    assert_eq!(error.code, "CALYX_LENS_FROZEN_VIOLATION");
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_missing_tokenizer_is_config_invalid() {
    let fixture = Fixture::new("missing-tokenizer", &[3.0, 4.0, 0.0]);
    fs::remove_file(&fixture.tokenizer).unwrap();

    let error = lens_error(OnnxLens::from_files(fixture.spec("custom-missing")));

    assert_eq!(error.code, "CALYX_LENS_CONFIG_INVALID");
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_declared_dim_mismatch_fails_closed() {
    let fixture = Fixture::new("dim-mismatch", &[3.0, 4.0, 0.0]);

    let error = lens_error(OnnxLens::from_files(
        fixture
            .spec("custom-dim")
            .with_expected_shape(SlotShape::Dense(4)),
    ));

    assert_eq!(error.code, "CALYX_LENS_DIM_MISMATCH");
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and ORT editor/session runtime"]
fn custom_onnx_non_finite_output_is_numerical_invariant() {
    let fixture = Fixture::new("nan", &[f32::NAN, 1.0, 0.0]);
    let lens = OnnxLens::from_files(fixture.spec("custom-nan")).unwrap();

    let error = lens
        .measure(&Input::new(Modality::Text, b"hello".to_vec()))
        .unwrap_err();

    assert_eq!(error.code, "CALYX_LENS_NUMERICAL_INVARIANT");
}

proptest! {
    #[test]
    fn pooling_is_deterministic(values in proptest::collection::vec(-10.0f32..10.0, 12)) {
        let shape = [1, 4, 3];
        let mask = [1, 1, 0, 1];
        for policy in [PoolingPolicy::Mean, PoolingPolicy::Cls, PoolingPolicy::LastToken] {
            let first = pool_output(&shape, &values, &mask, policy, 3).unwrap();
            for _ in 0..100 {
                prop_assert_eq!(pool_output(&shape, &values, &mask, policy, 3).unwrap(), first.clone());
            }
        }
    }
}

#[test]
fn pooling_rejects_short_attention_mask_for_masked_policies() {
    let shape = [1, 4, 3];
    let values = vec![1.0; 12];
    let short_mask = [1, 1];

    for policy in [
        PoolingPolicy::Cls,
        PoolingPolicy::Mean,
        PoolingPolicy::LastToken,
    ] {
        let error = pool_output(&shape, &values, &short_mask, policy, 3).unwrap_err();
        assert_eq!(error.code, "CALYX_LENS_DIM_MISMATCH");
        assert!(error.message.contains("seq"));
    }
}

#[test]
#[ignore = "requires manual HF cache/network and downloads ONNX all-MiniLM"]
fn onnx_all_minilm_manual_fsv() {
    let lens = OnnxLens::all_minilm_l6_v2_cpu_explicit("onnx-manual-fsv").unwrap();
    let input = Input::new(Modality::Text, b"Calyx PH19 ONNX local probe".to_vec());
    let vector = lens.measure(&input).unwrap();

    if let SlotVector::Dense { dim, data } = vector {
        println!("ONNX_FSV_CACHE={}", lens.files().cache_dir.display());
        println!("ONNX_FSV_MODEL={}", lens.files().model_file.display());
        println!("ONNX_FSV_PROVIDER_POLICY={}", lens.provider_policy());
        println!("ONNX_FSV_DIM={dim}");
        println!("ONNX_FSV_FIRST3={:?}", &data[..3]);
        let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("ONNX_FSV_NORM={norm:.8}");
        assert!((norm - 1.0).abs() < 1.0e-3);
    } else {
        panic!("expected dense ONNX vector");
    }
}

#[test]
#[ignore = "requires manual CUDA/ONNX stack; validates fail-loud GPU policy"]
fn onnx_cuda_fail_loud_manual_fsv() {
    let input = Input::new(Modality::Text, b"Calyx PH19 CUDA fail-loud probe".to_vec());
    match OnnxLens::all_minilm_l6_v2("onnx-manual-cuda-fail-loud") {
        Ok(lens) => match lens.measure(&input) {
            Ok(vector) => {
                println!("ONNX_CUDA_RESULT=success");
                if let SlotVector::Dense { dim, data } = vector {
                    let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
                    println!("ONNX_CUDA_DIM={dim}");
                    println!("ONNX_CUDA_NORM={norm:.8}");
                    assert!((norm - 1.0).abs() < 1.0e-3);
                }
            }
            Err(error) => {
                println!("ONNX_CUDA_RESULT=fail_loud");
                println!("ONNX_CUDA_ERROR_CODE={}", error.code);
                println!("ONNX_CUDA_ERROR_MESSAGE={}", error.message);
                assert_eq!(error.code, "CALYX_LENS_UNREACHABLE");
            }
        },
        Err(error) => {
            println!("ONNX_CUDA_RESULT=fail_loud_init");
            println!("ONNX_CUDA_ERROR_CODE={}", error.code);
            println!("ONNX_CUDA_ERROR_MESSAGE={}", error.message);
            assert_eq!(error.code, "CALYX_LENS_UNREACHABLE");
        }
    }
}

#[test]
#[ignore = "requires manual HF cache/network and downloads ONNX all-MiniLM"]
fn onnx_dim_guard_manual_fsv() {
    let lens = OnnxLens::all_minilm_l6_v2_cpu_explicit("onnx-manual-dim-guard").unwrap();
    let error = lens
        .contract()
        .verify_vector(
            lens.id(),
            &SlotVector::Dense {
                dim: 3,
                data: vec![1.0, 0.0, 0.0],
            },
        )
        .unwrap_err();

    println!("ONNX_DIM_GUARD_ERROR={}", error.code);
    assert_eq!(error.code, "CALYX_LENS_DIM_MISMATCH");
}

#[test]
#[ignore = "requires explicit custom ONNX env paths in a manual verification run"]
fn custom_onnx_manual_fsv_from_files() {
    let model = std::env::var("CALYX_CUSTOM_ONNX_MODEL").unwrap();
    let tokenizer = std::env::var("CALYX_CUSTOM_ONNX_TOKENIZER").unwrap();
    let config = std::env::var("CALYX_CUSTOM_ONNX_CONFIG").unwrap();
    let lens = OnnxLens::from_files(
        OnnxFileSpec::text(
            "onnx-custom-manual-fsv",
            "Xenova/bge-small-en-v1.5",
            model,
            tokenizer,
            config,
            PoolingPolicy::Mean,
            NormPolicy::unit(),
        )
        .with_provider_policy(OnnxProviderPolicy::CpuExplicit),
    )
    .unwrap();
    let vector = lens
        .measure(&Input::new(
            Modality::Text,
            b"Calyx PH73 custom ONNX explicit file probe".to_vec(),
        ))
        .unwrap();
    let spec = lens.lens_spec();
    let reloaded = OnnxLens::from_lens_spec(&spec).unwrap();
    assert_eq!(lens.id(), reloaded.id());

    let SlotVector::Dense { dim, data } = vector else {
        panic!("expected dense custom ONNX vector");
    };
    let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("ONNX_CUSTOM_FSV_RUNTIME={}", lens.runtime_name());
    println!("ONNX_CUSTOM_FSV_MODEL_ID={}", lens.files().model_code);
    println!("ONNX_CUSTOM_FSV_LENS_ID={}", lens.id());
    println!("ONNX_CUSTOM_FSV_CORPUS_HASH={}", hex32(&spec.corpus_hash));
    println!(
        "ONNX_CUSTOM_FSV_WEIGHTS_SHA256={}",
        hex32(&spec.weights_sha256)
    );
    println!("ONNX_CUSTOM_FSV_DIM={dim}");
    println!("ONNX_CUSTOM_FSV_DTYPE=int8");
    println!("ONNX_CUSTOM_FSV_NORM={norm:.8}");
    println!("ONNX_CUSTOM_FSV_FIRST3={:?}", &data[..3]);
    println!(
        "ONNX_CUSTOM_FSV_SPEC_RELOAD_RUNTIME={}",
        reloaded.runtime_name()
    );
    assert_eq!(lens.runtime_name(), "onnx-custom");
    assert!((norm - 1.0).abs() < 1.0e-3);
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH, CUDA, and CALYX_FSV_ROOT"]
#[cfg(feature = "cuda")]
fn custom_onnx_cuda_device_postprocess_manual_fsv() {
    let fixture = Fixture::new_cuda_token_matmul("cuda-device-output");
    let input_text = b"hello calyx".to_vec();
    let expected = vec![0.6_f32, 0.8_f32];
    println!(
        "CUDA_DEVICE_FSV_BEFORE model={} tokenizer={} config={} input_text={}",
        fixture.model.display(),
        fixture.tokenizer.display(),
        fixture.config.display(),
        String::from_utf8_lossy(&input_text)
    );

    let lens = OnnxLens::from_files(
        fixture
            .spec("custom-cuda-device")
            .with_provider_policy(OnnxProviderPolicy::CudaFailLoud)
            .with_expected_shape(SlotShape::Dense(2))
            .with_max_batch(1),
    )
    .unwrap();
    let vector = lens
        .measure(&Input::new(Modality::Text, input_text.clone()))
        .unwrap();
    let SlotVector::Dense { dim, data } = vector else {
        panic!("expected dense custom ONNX CUDA vector");
    };
    println!("CUDA_DEVICE_FSV_AFTER dim={dim} data={data:?}");
    assert_eq!(dim, 2);
    assert_close(&data, &expected, 1.0e-5);
    let norm = data.iter().map(|value| value * value).sum::<f32>().sqrt();

    let payload = serde_json::json!({
        "source_of_truth": "SlotVector returned after ONNX Runtime CUDA output binding and Forge CUDA postprocess, persisted to this CALYX_FSV_ROOT JSON readback",
        "before": {
            "model": fixture.model.display().to_string(),
            "tokenizer": fixture.tokenizer.display().to_string(),
            "config": fixture.config.display().to_string(),
            "input_text": String::from_utf8_lossy(&input_text).to_string(),
            "input_token_ids": [1, 2],
            "expected_device_output_before_pool": [[[3.0, 4.0], [6.0, 8.0]]],
            "expected_pooling": "mean over two unmasked tokens",
        },
        "expected": {
            "dim": 2,
            "data": expected,
            "norm": 1.0,
        },
        "after": {
            "runtime": lens.runtime_name(),
            "provider_policy": lens.provider_policy(),
            "lens_id": format!("{:?}", lens.id()),
            "shape": format!("{:?}", lens.shape()),
            "dim": dim,
            "data": data,
            "norm": norm,
        },
        "checks": {
            "device_output_audit": "device_tensor() rejects CPU outputs before Forge postprocess; reaching this readback proves ORT returned CUDA memory",
            "postprocess": "Forge CUDA mean-pooling + L2 normalization returned the expected compact vector",
        }
    });
    let readback_path = write_onnx_fsv_readback("custom-onnx-cuda-device-readback.json", payload);
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&readback_path).unwrap()).unwrap();
    assert_eq!(persisted["after"]["dim"], serde_json::json!(2));
    assert_eq!(persisted["after"]["data"].as_array().unwrap().len(), 2);
    println!("CUDA_DEVICE_FSV_READBACK={}", readback_path.display());
    println!("CUDA_DEVICE_FSV_READBACK_JSON={persisted}");
}

#[test]
#[ignore = "manual PH73 edge FSV prints source-of-truth file states"]
fn custom_onnx_edges_manual_fsv() {
    let missing = Fixture::new("edge-missing-tokenizer", &[3.0, 4.0, 0.0]);
    println!(
        "EDGE_MISSING_TOKENIZER_BEFORE_EXISTS={}",
        missing.tokenizer.is_file()
    );
    fs::remove_file(&missing.tokenizer).unwrap();
    let missing_error = lens_error(OnnxLens::from_files(missing.spec("edge-missing")));
    println!(
        "EDGE_MISSING_TOKENIZER_AFTER_EXISTS={}",
        missing.tokenizer.is_file()
    );
    println!("EDGE_MISSING_TOKENIZER_ERROR={}", missing_error.code);
    assert_eq!(missing_error.code, "CALYX_LENS_CONFIG_INVALID");

    let dim = Fixture::new("edge-dim", &[3.0, 4.0, 0.0]);
    let dim_error = lens_error(OnnxLens::from_files(
        dim.spec("edge-dim")
            .with_expected_shape(SlotShape::Dense(4)),
    ));
    println!(
        "EDGE_DECLARED_DIM_ACTUAL=3 DECLARED=4 ERROR={}",
        dim_error.code
    );
    assert_eq!(dim_error.code, "CALYX_LENS_DIM_MISMATCH");

    let nan = Fixture::new("edge-nan", &[f32::NAN, 1.0, 0.0]);
    let nan_lens = OnnxLens::from_files(nan.spec("edge-nan")).unwrap();
    let nan_error = nan_lens
        .measure(&Input::new(Modality::Text, b"hello".to_vec()))
        .unwrap_err();
    println!("EDGE_NON_FINITE_OUTPUT_ERROR={}", nan_error.code);
    assert_eq!(nan_error.code, "CALYX_LENS_NUMERICAL_INVARIANT");

    let drift = Fixture::new("edge-hash", &[3.0, 4.0, 0.0]);
    let original = OnnxLens::from_files(drift.spec("edge-hash")).unwrap();
    let expected = original.contract().weights_sha256();
    fs::write(
        &drift.config,
        r#"{"model_type":"calyx-test","pooling":"cls"}"#,
    )
    .unwrap();
    let drift_error = lens_error(OnnxLens::from_files(
        drift
            .spec("edge-hash")
            .with_expected_weights_sha256(expected),
    ));
    println!(
        "EDGE_HASH_DRIFT_EXPECTED={} ERROR={}",
        hex32(&expected),
        drift_error.code
    );
    assert_eq!(drift_error.code, "CALYX_LENS_FROZEN_VIOLATION");
}
