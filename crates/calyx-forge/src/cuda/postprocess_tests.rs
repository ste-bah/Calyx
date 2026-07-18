use std::fs;
use std::path::Path;

use cudarc::driver::{CudaSlice, DevicePtr};
use serde_json::json;

use super::test_lock;
use crate::cuda::{
    CudaDenseTokenPostprocess, CudaPostprocessPooling, bgem3_colbert_tokens_from_external_f32,
    bgem3_sparse_from_external_f32, colbert_tokens_from_external_f32, dense_2d_from_external_f32,
    dense_tokens_from_external_f32, init_cuda, sparse_positive_from_external_f32,
};
use crate::{CudaContext, ForgeError, Result};

fn fsv_error(op: &str, path: &Path, detail: impl ToString) -> ForgeError {
    ForgeError::CacheError {
        op: op.to_string(),
        path: path.display().to_string(),
        detail: detail.to_string(),
        remediation: "repair CALYX_FSV_ROOT and rerun the CUDA postprocess readback".to_string(),
    }
}

fn write_fsv_readback(name: &str, payload: serde_json::Value) -> Result<()> {
    let Some(root) = calyx_fsv::fsv_root("CALYX_FSV_ROOT") else {
        return Ok(());
    };
    fs::create_dir_all(&root).map_err(|err| fsv_error("fsv_mkdir", &root, err))?;
    let path = root.join(name);
    let bytes = serde_json::to_vec_pretty(&payload).map_err(|err| {
        fsv_error(
            "fsv_serialize",
            &path,
            format!("serialize JSON readback failed: {err}"),
        )
    })?;
    fs::write(&path, &bytes).map_err(|err| fsv_error("fsv_write", &path, err))?;
    let readback = fs::read(&path).map_err(|err| fsv_error("fsv_read", &path, err))?;
    assert_eq!(readback, bytes);
    println!(
        "CUDA_POSTPROCESS_FSV_READBACK path={} bytes={}",
        path.display(),
        readback.len()
    );
    Ok(())
}

fn upload_f32(ctx: &CudaContext, values: &[f32]) -> Result<CudaSlice<f32>> {
    ctx.inner()
        .default_stream()
        .clone_htod(values)
        .map_err(|err| ForgeError::DeviceUnavailable {
            device: format!("cuda:{}", ctx.device_idx()),
            detail: format!("test upload failed: {err}"),
            remediation: "verify CUDA driver and rerun CUDA postprocess tests".to_string(),
        })
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tol,
            "idx={idx} actual={actual} expected={expected}"
        );
    }
}

fn error_code(error: &ForgeError) -> &'static str {
    error.code()
}

#[test]
fn bgem3_joint_heads_compact_normalize_and_fail_closed_on_device() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let sparse_values = vec![9.0, 0.2, 0.4, 0.8, 9.0, -1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let token_ids = vec![0, 10, 7, 10, 3, 11, 4, 5, 4, 250_001, 8, 9];
    let mask = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
    println!(
        "CUDA_BGEM3_BEFORE sparse batch=2 seq=6 vocab=250002 values={sparse_values:?} ids={token_ids:?} mask={mask:?}"
    );
    let sparse_dev = upload_f32(&ctx, &sparse_values)?;
    let stream = ctx.inner().default_stream();
    let (sparse_ptr, _sparse_guard) = sparse_dev.device_ptr(&stream);
    let sparse =
        bgem3_sparse_from_external_f32(&ctx, sparse_ptr, &token_ids, &mask, 2, 6, 250_002)?;
    let expected_sparse = vec![
        vec![(7, 0.4), (10, 0.8)],
        vec![(4, 3.0), (5, 2.0), (250_001, 4.0)],
    ];
    println!("CUDA_BGEM3_AFTER sparse readback={:?}", sparse.rows);
    assert_eq!(sparse.rows, expected_sparse);
    assert_eq!(sparse.input_floats, 12);
    assert_eq!(sparse.host_value_floats, 5);

    let colbert_values = vec![3.0, 4.0, 8.0, 6.0, 0.0, 5.0, 12.0, 5.0];
    let colbert_mask = vec![1, 0, 1, 1];
    println!(
        "CUDA_BGEM3_BEFORE colbert batch=1 seq=4 dim=2 values={colbert_values:?} mask={colbert_mask:?}"
    );
    let colbert_dev = upload_f32(&ctx, &colbert_values)?;
    let stream = ctx.inner().default_stream();
    let (colbert_ptr, _colbert_guard) = colbert_dev.device_ptr(&stream);
    let colbert =
        bgem3_colbert_tokens_from_external_f32(&ctx, colbert_ptr, &colbert_mask, 1, 4, 2)?;
    let expected_colbert = vec![vec![
        vec![0.6, 0.8],
        vec![0.0, 1.0],
        vec![12.0 / 13.0, 5.0 / 13.0],
    ]];
    println!("CUDA_BGEM3_AFTER colbert readback={:?}", colbert.rows);
    for (actual, expected) in colbert.rows[0].iter().zip(&expected_colbert[0]) {
        assert_close(actual, expected, 1e-5);
    }
    assert_eq!(colbert.input_floats, 8);
    assert_eq!(colbert.host_value_floats, 6);

    let mut edges = Vec::new();
    for (case, values, ids) in [
        ("invalid_token_id", vec![1.0], vec![250_002]),
        ("nonfinite_sparse_weight", vec![f32::NAN], vec![42]),
    ] {
        let edge_mask = vec![1];
        println!(
            "CUDA_BGEM3_EDGE_BEFORE case={case} values={values:?} ids={ids:?} mask={edge_mask:?}"
        );
        let device = upload_f32(&ctx, &values)?;
        let stream = ctx.inner().default_stream();
        let (ptr, _guard) = device.device_ptr(&stream);
        let error = bgem3_sparse_from_external_f32(&ctx, ptr, &ids, &edge_mask, 1, 1, 250_002)
            .expect_err("invalid BGE-M3 sparse output must fail closed");
        assert!(matches!(error, ForgeError::NumericalInvariant { .. }));
        println!(
            "CUDA_BGEM3_EDGE_AFTER case={case} error_code={} error={error}",
            error_code(&error)
        );
        edges.push(json!({
            "case": case,
            "before": {"values": values.iter().map(|value| format!("{value:?}")).collect::<Vec<_>>(), "ids": ids, "mask": edge_mask},
            "after": {"error_code": error_code(&error), "error": error.to_string()}
        }));
    }

    let zero_values = vec![0.0, 0.0];
    let zero_mask = vec![1];
    println!(
        "CUDA_BGEM3_EDGE_BEFORE case=zero_norm_colbert values={zero_values:?} mask={zero_mask:?}"
    );
    let zero_dev = upload_f32(&ctx, &zero_values)?;
    let stream = ctx.inner().default_stream();
    let (zero_ptr, _zero_guard) = zero_dev.device_ptr(&stream);
    let zero_error = bgem3_colbert_tokens_from_external_f32(&ctx, zero_ptr, &zero_mask, 1, 1, 2)
        .expect_err("zero-norm BGE-M3 ColBERT token must fail closed");
    assert!(matches!(zero_error, ForgeError::NumericalInvariant { .. }));
    println!(
        "CUDA_BGEM3_EDGE_AFTER case=zero_norm_colbert error_code={} error={zero_error}",
        error_code(&zero_error)
    );
    edges.push(json!({
        "case": "zero_norm_colbert",
        "before": {"values": zero_values, "mask": zero_mask},
        "after": {"error_code": error_code(&zero_error), "error": zero_error.to_string()}
    }));

    write_fsv_readback(
        "cuda-bgem3-postprocess-readback.json",
        json!({
            "issues": [1497, 1631],
            "source_of_truth": "real CUDA device buffers independently read back after BGE-M3 postprocess kernels",
            "happy": {
                "sparse": {"expected": expected_sparse, "after_readback": sparse.rows},
                "colbert": {"expected": expected_colbert, "after_readback": colbert.rows}
            },
            "edges": edges
        }),
    )?;
    Ok(())
}

#[test]
fn postprocess_dense_sparse_and_colbert_read_back_expected_device_outputs() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let dense_tokens = vec![3.0, 4.0, 0.0, 0.0, 6.0, 8.0, 1.0, 0.0, 0.0, 2.0, 9.0, 9.0];
    let dense_mask = vec![1, 0, 1, 1, 1, 0];
    println!(
        "CUDA_POSTPROCESS_BEFORE happy_dense_tokens source=device_input batch=2 seq=3 dim=2 values={dense_tokens:?} mask={dense_mask:?}"
    );
    let dense_dev = upload_f32(&ctx, &dense_tokens)?;
    let stream = ctx.inner().default_stream();
    let (dense_ptr, _dense_guard) = dense_dev.device_ptr(&stream);
    let dense = dense_tokens_from_external_f32(
        &ctx,
        CudaDenseTokenPostprocess {
            values_ptr: dense_ptr,
            mask: &dense_mask,
            batch: 2,
            seq: 3,
            dim: 2,
            pooling: CudaPostprocessPooling::Mean,
            normalize: true,
        },
    )?;
    let expected_dense = vec![0.6, 0.8, 0.4472136, 0.8944272];
    println!("CUDA_POSTPROCESS_AFTER happy_dense_tokens readback={dense:?}");
    assert_close(&dense, &expected_dense, 1e-5);

    let sparse_values = vec![0.0, 1.25, -2.0, 3.5, 0.0, 4.0, 0.0, -1.0, 0.75, 0.0];
    println!(
        "CUDA_POSTPROCESS_BEFORE happy_sparse source=device_input batch=2 dim=5 values={sparse_values:?}"
    );
    let sparse_dev = upload_f32(&ctx, &sparse_values)?;
    let stream = ctx.inner().default_stream();
    let (sparse_ptr, _sparse_guard) = sparse_dev.device_ptr(&stream);
    let sparse = sparse_positive_from_external_f32(&ctx, sparse_ptr, 2, 5)?;
    println!(
        "CUDA_POSTPROCESS_AFTER happy_sparse readback={:?}",
        sparse.rows
    );
    assert_eq!(
        sparse.rows,
        vec![vec![(1, 1.25), (3, 3.5)], vec![(0, 4.0), (3, 0.75)]]
    );
    assert_eq!(sparse.input_floats, 10);
    assert_eq!(sparse.host_value_floats, 4);

    let colbert_values = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let colbert_mask = vec![1, 0, 1, 0, 1, 1];
    println!(
        "CUDA_POSTPROCESS_BEFORE happy_colbert source=device_input batch=2 seq=3 dim=2 values={colbert_values:?} mask={colbert_mask:?}"
    );
    let colbert_dev = upload_f32(&ctx, &colbert_values)?;
    let stream = ctx.inner().default_stream();
    let (colbert_ptr, _colbert_guard) = colbert_dev.device_ptr(&stream);
    let colbert = colbert_tokens_from_external_f32(&ctx, colbert_ptr, &colbert_mask, 2, 3, 2)?;
    println!(
        "CUDA_POSTPROCESS_AFTER happy_colbert readback={:?}",
        colbert.rows
    );
    assert_eq!(
        colbert.rows,
        vec![
            vec![vec![1.0, 2.0], vec![5.0, 6.0]],
            vec![vec![9.0, 10.0], vec![11.0, 12.0]]
        ]
    );
    assert_eq!(colbert.input_floats, 12);
    assert_eq!(colbert.host_value_floats, 8);

    write_fsv_readback(
        "cuda-postprocess-happy-readback.json",
        json!({
            "issue": 1497,
            "source_of_truth": "CUDA device buffers read back after postprocess kernels",
            "dense": {
                "before": {"batch": 2, "seq": 3, "dim": 2, "values": dense_tokens, "mask": dense_mask},
                "expected": expected_dense,
                "after_readback": dense
            },
            "sparse": {
                "before": {"batch": 2, "dim": 5, "values": sparse_values},
                "expected": [[{"idx": 1, "val": 1.25}, {"idx": 3, "val": 3.5}], [{"idx": 0, "val": 4.0}, {"idx": 3, "val": 0.75}]],
                "after_readback": sparse.rows,
                "input_floats": sparse.input_floats,
                "host_value_floats": sparse.host_value_floats
            },
            "colbert": {
                "before": {"batch": 2, "seq": 3, "dim": 2, "values": colbert_values, "mask": colbert_mask},
                "expected": [[[1.0, 2.0], [5.0, 6.0]], [[9.0, 10.0], [11.0, 12.0]]],
                "after_readback": colbert.rows,
                "input_floats": colbert.input_floats,
                "host_value_floats": colbert.host_value_floats
            }
        }),
    )?;
    Ok(())
}

#[test]
fn postprocess_edges_fail_loud_with_before_after_state() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let mut edge_reports = Vec::new();

    let empty_values = vec![1.0, 2.0, 3.0, 4.0];
    let empty_mask = vec![0, 0];
    println!(
        "CUDA_POSTPROCESS_EDGE_BEFORE empty_mask source=device_input batch=1 seq=2 dim=2 values={empty_values:?} mask={empty_mask:?}"
    );
    let empty_dev = upload_f32(&ctx, &empty_values)?;
    let stream = ctx.inner().default_stream();
    let (empty_ptr, _empty_guard) = empty_dev.device_ptr(&stream);
    let empty_error = dense_tokens_from_external_f32(
        &ctx,
        CudaDenseTokenPostprocess {
            values_ptr: empty_ptr,
            mask: &empty_mask,
            batch: 1,
            seq: 2,
            dim: 2,
            pooling: CudaPostprocessPooling::Mean,
            normalize: true,
        },
    )
    .expect_err("empty mask must fail closed");
    println!(
        "CUDA_POSTPROCESS_EDGE_AFTER empty_mask error_code={} error={empty_error}",
        error_code(&empty_error)
    );
    assert!(matches!(empty_error, ForgeError::NumericalInvariant { .. }));
    edge_reports.push(json!({
        "case": "empty_mask",
        "before": {"batch": 1, "seq": 2, "dim": 2, "values": empty_values, "mask": empty_mask},
        "after": {"error_code": error_code(&empty_error), "error": empty_error.to_string()}
    }));

    let nonfinite_values = vec![1.0, f32::NAN];
    println!(
        "CUDA_POSTPROCESS_EDGE_BEFORE nonfinite_dense source=device_input batch=1 dim=2 values={nonfinite_values:?}"
    );
    let nonfinite_dev = upload_f32(&ctx, &nonfinite_values)?;
    let stream = ctx.inner().default_stream();
    let (nonfinite_ptr, _nonfinite_guard) = nonfinite_dev.device_ptr(&stream);
    let nonfinite_error = dense_2d_from_external_f32(&ctx, nonfinite_ptr, 1, 2, false)
        .expect_err("non-finite dense output must fail closed");
    println!(
        "CUDA_POSTPROCESS_EDGE_AFTER nonfinite_dense error_code={} error={nonfinite_error}",
        error_code(&nonfinite_error)
    );
    assert!(matches!(
        nonfinite_error,
        ForgeError::NumericalInvariant { .. }
    ));
    edge_reports.push(json!({
        "case": "nonfinite_dense",
        "before": {"batch": 1, "dim": 2, "values": ["1.0", "NaN"]},
        "after": {"error_code": error_code(&nonfinite_error), "error": nonfinite_error.to_string()}
    }));

    let mismatch_values = vec![1.0, 2.0, 3.0, 4.0];
    let mismatch_mask = vec![1];
    println!(
        "CUDA_POSTPROCESS_EDGE_BEFORE mask_len_mismatch source=device_input batch=1 seq=2 dim=2 values={mismatch_values:?} mask={mismatch_mask:?}"
    );
    let mismatch_dev = upload_f32(&ctx, &mismatch_values)?;
    let stream = ctx.inner().default_stream();
    let (mismatch_ptr, _mismatch_guard) = mismatch_dev.device_ptr(&stream);
    let mismatch_error = dense_tokens_from_external_f32(
        &ctx,
        CudaDenseTokenPostprocess {
            values_ptr: mismatch_ptr,
            mask: &mismatch_mask,
            batch: 1,
            seq: 2,
            dim: 2,
            pooling: CudaPostprocessPooling::Mean,
            normalize: false,
        },
    )
    .expect_err("mask length mismatch must fail before kernel launch");
    println!(
        "CUDA_POSTPROCESS_EDGE_AFTER mask_len_mismatch error_code={} error={mismatch_error}",
        error_code(&mismatch_error)
    );
    assert!(matches!(mismatch_error, ForgeError::ShapeMismatch { .. }));
    edge_reports.push(json!({
        "case": "mask_len_mismatch",
        "before": {"batch": 1, "seq": 2, "dim": 2, "values": mismatch_values, "mask": mismatch_mask},
        "after": {"error_code": error_code(&mismatch_error), "error": mismatch_error.to_string()}
    }));

    write_fsv_readback(
        "cuda-postprocess-edge-readback.json",
        json!({
            "issue": 1497,
            "source_of_truth": "CUDA device buffers plus structured Forge errors after postprocess trigger",
            "edges": edge_reports
        }),
    )?;
    Ok(())
}
