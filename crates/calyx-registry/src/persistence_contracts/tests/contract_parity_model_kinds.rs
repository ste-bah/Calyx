//! Issue #1489 regression net, part 2: static-contract formula pins for the
//! remaining model-backed runtime kinds (ColBERT, fastembed, qwen3). See
//! `contract_parity.rs` for the cheap-runtime round-trips and the candle/ONNX
//! shared-formula checks.

use fastembed::{
    Bgem3Embedding, Bgem3Model, RerankerModel, SparseModel, SparseTextEmbedding, TextRerank,
};

use super::contract_parity::{fixture_dir, spec_with};
use super::*;
use crate::{Bgem3Engine, FastembedBgem3Output};

#[test]
fn onnx_colbert_static_contract_matches_runtime_formula() {
    let dir = fixture_dir("parity-colbert");
    let model = dir.join("model_fp16.onnx");
    let tokenizer = dir.join("tokenizer.json");
    let config = dir.join("config.json");
    fs::write(&model, b"colbert fixture bytes").unwrap();
    fs::write(&tokenizer, br#"{"tokenizer":"fixture"}"#).unwrap();
    fs::write(&config, br#"{"hidden_size":128}"#).unwrap();
    let model_id = "answerdotai/answerai-colbert-small-v1";
    let spec = spec_with(
        "parity-colbert",
        LensRuntime::OnnxColbert {
            model_id: model_id.to_string(),
            files: vec![model, tokenizer, config],
        },
        SlotShape::Multi { token_dim: 96 },
        [0x66; 32],
        NormPolicy::Finite,
    );

    let derived = derive_runtime_contract_from_spec(&spec).unwrap();
    let expected_corpus = sha256_digest(&[
        b"onnx-colbert-token-v1",
        model_id.as_bytes(),
        b"onnx/model_fp16.onnx",
        b"attention-mask-unpooled-finite",
    ]);
    assert_eq!(derived.corpus_hash(), expected_corpus);
    assert_eq!(derived.shape(), SlotShape::Multi { token_dim: 96 });
    assert_eq!(derived.norm_policy(), NormPolicy::Finite);
}

#[test]
fn fastembed_static_contracts_match_runtime_formulas() {
    let dir = fixture_dir("parity-fastembed");
    let artifact = dir.join("model.onnx");
    fs::write(&artifact, b"fastembed fixture bytes").unwrap();
    let files = vec![artifact];

    let sparse_info = SparseTextEmbedding::get_model_info(&SparseModel::SPLADEPPV1);
    let sparse_spec = spec_with(
        "parity-fastembed-sparse",
        LensRuntime::FastembedSparse {
            model_id: "prithivida/Splade_PP_en_v1".to_string(),
            files: files.clone(),
        },
        SlotShape::Sparse(30_522),
        [0x77; 32],
        NormPolicy::Finite,
    );
    let derived = derive_runtime_contract_from_spec(&sparse_spec).unwrap();
    assert_eq!(
        derived.corpus_hash(),
        sha256_digest(&[b"fastembed-sparse-v1", sparse_info.model_code.as_bytes()])
    );
    assert_eq!(derived.shape(), SlotShape::Sparse(30_522));
    assert_eq!(derived.norm_policy(), NormPolicy::Finite);

    let bgem3_info = Bgem3Embedding::get_model_info(&Bgem3Model::BGEM3Q);
    for (output, shape, norm, token) in [
        (
            FastembedBgem3Output::Dense,
            SlotShape::Dense(1024),
            NormPolicy::unit(),
            b"dense" as &[u8],
        ),
        (
            FastembedBgem3Output::Sparse,
            SlotShape::Sparse(250_002),
            NormPolicy::Finite,
            b"sparse",
        ),
        (
            FastembedBgem3Output::Colbert,
            SlotShape::Multi { token_dim: 1024 },
            NormPolicy::Finite,
            b"colbert",
        ),
    ] {
        let spec = spec_with(
            "parity-fastembed-bgem3",
            LensRuntime::FastembedBgem3 {
                model_id: "BAAI/bge-m3".to_string(),
                files: files.clone(),
                output,
                engine: Bgem3Engine::FastembedCpu,
            },
            shape,
            [0x78; 32],
            norm,
        );
        let derived = derive_runtime_contract_from_spec(&spec).unwrap();
        assert_eq!(
            derived.corpus_hash(),
            sha256_digest(&[
                b"fastembed-bgem3-v1",
                bgem3_info.model_code.as_bytes(),
                token
            ]),
            "bgem3 {output:?}"
        );
        assert_eq!(derived.shape(), shape, "bgem3 {output:?}");
        assert_eq!(derived.norm_policy(), norm, "bgem3 {output:?}");
    }

    let reranker_info = TextRerank::get_model_info(&RerankerModel::BGERerankerV2M3);
    let reranker_spec = spec_with(
        "parity-fastembed-reranker",
        LensRuntime::FastembedReranker {
            model_id: "BAAI/bge-reranker-v2-m3".to_string(),
            files: files.clone(),
        },
        SlotShape::Dense(1),
        [0x79; 32],
        NormPolicy::Finite,
    );
    let derived = derive_runtime_contract_from_spec(&reranker_spec).unwrap();
    assert_eq!(
        derived.corpus_hash(),
        sha256_digest(&[
            b"fastembed-reranker-v1",
            reranker_info.model_code.as_bytes()
        ])
    );
    assert_eq!(derived.shape(), SlotShape::Dense(1));
}

#[test]
fn qwen3_static_contract_matches_runtime_formula() {
    let dir = fixture_dir("parity-qwen3");
    let weights_file = dir.join("model.safetensors");
    let tokenizer = dir.join("tokenizer.json");
    let config = dir.join("config.json");
    fs::write(&weights_file, b"qwen3 fixture bytes").unwrap();
    fs::write(&tokenizer, br#"{"tokenizer":"fixture"}"#).unwrap();
    fs::write(&config, br#"{"hidden_size":1024}"#).unwrap();
    let spec = spec_with(
        "parity-qwen3",
        LensRuntime::FastembedQwen3 {
            model_id: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            files: vec![weights_file, tokenizer, config],
            dtype: "f16".to_string(),
            max_tokens: 8_192,
        },
        SlotShape::Dense(1024),
        [0x7A; 32],
        NormPolicy::unit(),
    );

    let derived = derive_runtime_contract_from_spec(&spec).unwrap();
    let expected_corpus = sha256_digest(&[
        b"fastembed-qwen3-text-v1",
        b"Qwen/Qwen3-Embedding-0.6B",
        b"f16",
        b"8192",
        b"left-padding,last-token,l2",
    ]);
    assert_eq!(derived.corpus_hash(), expected_corpus);
    assert_eq!(derived.shape(), SlotShape::Dense(1024));
    assert_eq!(derived.norm_policy(), NormPolicy::unit());
}
