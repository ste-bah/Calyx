use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use calyx_core::{Lens, Result};

use crate::frozen::NormPolicy;
use crate::runtime::onnx::{OnnxFileSpec, OnnxLens, OnnxProviderPolicy, PoolingPolicy};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

pub(super) struct Fixture {
    root: PathBuf,
    pub(super) model: PathBuf,
    pub(super) tokenizer: PathBuf,
    pub(super) config: PathBuf,
}

impl Fixture {
    pub(super) fn new(name: &str, output: &[f32]) -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "calyx-custom-onnx-{name}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let model = root.join("model.onnx");
        let tokenizer = root.join("tokenizer.json");
        let config = root.join("config.json");
        write_tokenizer(&tokenizer);
        fs::write(
            &config,
            r#"{"model_type":"calyx-test","hidden_size":3,"pooling":"mean"}"#,
        )
        .unwrap();
        write_model(&model, output);
        Self {
            root,
            model,
            tokenizer,
            config,
        }
    }

    #[cfg(feature = "cuda")]
    pub(super) fn new_cuda_token_matmul(name: &str) -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "calyx-custom-onnx-{name}-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let model = root.join("model.onnx");
        let tokenizer = root.join("tokenizer.json");
        let config = root.join("config.json");
        write_tokenizer(&tokenizer);
        fs::write(
            &config,
            r#"{"model_type":"calyx-test","hidden_size":2,"pooling":"mean","max_position_embeddings":2}"#,
        )
        .unwrap();
        write_cuda_token_matmul_model(&model);
        Self {
            root,
            model,
            tokenizer,
            config,
        }
    }

    pub(super) fn spec(&self, name: &str) -> OnnxFileSpec {
        OnnxFileSpec::text(
            name,
            "calyx-test-custom-onnx",
            self.model.clone(),
            self.tokenizer.clone(),
            self.config.clone(),
            PoolingPolicy::Mean,
            NormPolicy::unit(),
        )
        .with_provider_policy(OnnxProviderPolicy::CpuExplicit)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn write_tokenizer(path: &Path) {
    fs::write(
        path,
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"calyx":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
}

fn write_model(path: &Path, output: &[f32]) {
    use ort::editor::{Graph, Model, ONNX_DOMAIN, Opset};
    use ort::memory::Allocator;
    use ort::session::Session;
    use ort::value::{Outlet, Shape, SymbolicDimensions, Tensor, TensorElementType, ValueType};

    let mut graph = Graph::new().unwrap();
    graph
        .set_inputs([Outlet::new(
            "input_ids",
            ValueType::Tensor {
                ty: TensorElementType::Int64,
                shape: Shape::new([1, -1]),
                dimension_symbols: SymbolicDimensions::empty(2),
            },
        )])
        .unwrap();
    graph
        .set_outputs([Outlet::new(
            "sentence_embedding",
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new([1, output.len() as i64]),
                dimension_symbols: SymbolicDimensions::empty(2),
            },
        )])
        .unwrap();
    let mut tensor =
        Tensor::<f32>::new(&Allocator::default(), [1_i64, output.len() as i64]).unwrap();
    tensor.extract_tensor_mut().1.copy_from_slice(output);
    graph
        .add_initializer("sentence_embedding", tensor, false)
        .unwrap();
    let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22).unwrap()]).unwrap();
    model.add_graph(graph).unwrap();
    let builder = Session::builder()
        .unwrap()
        .with_optimized_model_path(path)
        .unwrap();
    let session = model.into_session(&builder).unwrap();
    drop(session);
    assert!(
        path.is_file(),
        "expected ORT to materialize {}",
        path.display()
    );
}

#[cfg(feature = "cuda")]
fn write_cuda_token_matmul_model(path: &Path) {
    use ort::editor::{Graph, Model, Node, ONNX_DOMAIN, Opset};
    use ort::memory::Allocator;
    use ort::operator::Attribute;
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::value::{Outlet, Shape, SymbolicDimensions, Tensor, TensorElementType, ValueType};

    let mut graph = Graph::new().unwrap();
    graph
        .set_inputs([Outlet::new(
            "input_ids",
            ValueType::Tensor {
                ty: TensorElementType::Int64,
                shape: Shape::new([1, 2]),
                dimension_symbols: SymbolicDimensions::empty(2),
            },
        )])
        .unwrap();
    graph
        .set_outputs([Outlet::new(
            "last_hidden_state",
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new([1, 2, 2]),
                dimension_symbols: SymbolicDimensions::empty(3),
            },
        )])
        .unwrap();
    graph
        .add_node(
            Node::new(
                "Cast",
                ONNX_DOMAIN,
                "cast_input_ids",
                ["input_ids"],
                ["ids_f32"],
                [Attribute::new("to", 1_i64).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();

    let mut axes = Tensor::<i64>::new(&Allocator::default(), [1_i64]).unwrap();
    axes.extract_tensor_mut().1.copy_from_slice(&[2_i64]);
    graph
        .add_initializer("unsqueeze_axes", axes, false)
        .unwrap();
    graph
        .add_node(
            Node::new(
                "Unsqueeze",
                ONNX_DOMAIN,
                "unsqueeze_tokens",
                ["ids_f32", "unsqueeze_axes"],
                ["tokens_rank3"],
                [],
            )
            .unwrap(),
        )
        .unwrap();

    let mut first_weight = Tensor::<f32>::new(&Allocator::default(), [1_i64, 2]).unwrap();
    first_weight
        .extract_tensor_mut()
        .1
        .copy_from_slice(&[3.0_f32, 4.0]);
    graph
        .add_initializer("first_weight", first_weight, false)
        .unwrap();
    graph
        .add_node(
            Node::new(
                "MatMul",
                ONNX_DOMAIN,
                "matmul_project_tokens",
                ["tokens_rank3", "first_weight"],
                ["tokens_mm_0"],
                [],
            )
            .unwrap(),
        )
        .unwrap();

    let mut identity = Tensor::<f32>::new(&Allocator::default(), [2_i64, 2]).unwrap();
    identity
        .extract_tensor_mut()
        .1
        .copy_from_slice(&[1.0_f32, 0.0, 0.0, 1.0]);
    graph
        .add_initializer("identity_weight", identity, false)
        .unwrap();
    let mut previous = "tokens_mm_0".to_string();
    for index in 1..=24 {
        let output = if index == 24 {
            "last_hidden_state".to_string()
        } else {
            format!("tokens_mm_{index}")
        };
        graph
            .add_node(
                Node::new(
                    "MatMul",
                    ONNX_DOMAIN,
                    format!("matmul_identity_{index}"),
                    [previous.as_str(), "identity_weight"],
                    [output.as_str()],
                    [],
                )
                .unwrap(),
            )
            .unwrap();
        previous = output;
    }

    let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22).unwrap()]).unwrap();
    model.add_graph(graph).unwrap();
    let builder = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .unwrap()
        .with_optimized_model_path(path)
        .unwrap();
    let session = model.into_session(&builder).unwrap();
    drop(session);
    assert!(
        path.is_file(),
        "expected ORT to materialize {}",
        path.display()
    );
}

pub(super) fn lens_error(result: Result<OnnxLens>) -> calyx_core::CalyxError {
    match result {
        Ok(lens) => panic!("expected error, got lens {}", lens.id()),
        Err(error) => error,
    }
}

pub(super) fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(feature = "cuda")]
pub(super) fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "value {index} expected {expected} got {actual}"
        );
    }
}

#[cfg(feature = "cuda")]
pub(super) fn write_onnx_fsv_readback(name: &str, payload: serde_json::Value) -> PathBuf {
    let root = calyx_fsv::fsv_root("CALYX_FSV_ROOT")
        .expect("CALYX_FSV_ROOT must be set for ONNX CUDA FSV readback");
    fs::create_dir_all(&root).expect("create CALYX_FSV_ROOT");
    let path = root.join(name);
    let bytes = serde_json::to_vec_pretty(&payload).expect("serialize ONNX CUDA FSV readback");
    fs::write(&path, bytes).expect("write ONNX CUDA FSV readback");
    path
}
