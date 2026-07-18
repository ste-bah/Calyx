use calyx_assay::project_gpu;
use serde_json::json;

#[cfg(feature = "cuda")]
use calyx_assay::{project_cpu, projection_transfer_bytes, target_projection_dim};
#[cfg(feature = "cuda")]
use calyx_forge::Backend;
#[cfg(feature = "cuda")]
use std::time::Instant;

fn deterministic_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let raw = ((row * 31 + col * 17 + row * col * 3) % 257) as f32;
                    (raw - 128.0) / 37.0
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn max_abs_diff(left: &[Vec<f32>], right: &[Vec<f32>]) -> f32 {
    assert_eq!(left.len(), right.len());
    let mut max_diff = 0.0_f32;
    for (left_row, right_row) in left.iter().zip(right) {
        assert_eq!(left_row.len(), right_row.len());
        for (left, right) in left_row.iter().zip(right_row) {
            max_diff = max_diff.max((left - right).abs());
        }
    }
    max_diff
}

#[cfg(feature = "cuda")]
fn assert_projection_close(
    cpu: &calyx_assay::ProjectionReport,
    gpu: &calyx_assay::ProjectionReport,
    tolerance: f32,
    label: &str,
) -> f32 {
    assert_eq!(gpu.input_rows, cpu.input_rows, "{label} row mismatch");
    assert_eq!(gpu.input_dim, cpu.input_dim, "{label} input dim mismatch");
    assert_eq!(
        gpu.output_dim, cpu.output_dim,
        "{label} output dim mismatch"
    );
    assert_eq!(gpu.seed, cpu.seed, "{label} seed mismatch");
    let diff = max_abs_diff(&cpu.projected, &gpu.projected);
    assert!(
        diff <= tolerance,
        "{label} max_abs_diff={diff} tolerance={tolerance}"
    );
    diff
}

fn error_state(err: &calyx_core::CalyxError) -> serde_json::Value {
    json!({
        "code": err.code,
        "message": err.message,
        "remediation": err.remediation,
    })
}

#[cfg(not(feature = "cuda"))]
#[test]
fn projection_gpu_fails_loud_without_cuda_feature() {
    let matrix = deterministic_matrix(8, 12);
    let before = json!({
        "rows": matrix.len(),
        "input_dim": matrix[0].len(),
        "seed": 1503_u64,
    });
    let err = project_gpu(&matrix, 1503)
        .expect_err("project_gpu must fail loud without the cuda feature");
    let after = error_state(&err);
    println!("ISSUE1503_NO_CUDA_BEFORE {before}");
    println!("ISSUE1503_NO_CUDA_AFTER {after}");
    assert_eq!(err.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    assert!(err.message.contains("no CPU fallback"));
}

#[cfg(feature = "cuda")]
#[test]
fn projection_cuda_matches_cpu_oracle_and_edges_fail_closed() {
    let seed = 1503_u64;
    let backend = calyx_forge::CudaBackend::new().expect("CUDA backend must initialize");
    let device_info = backend.device_info();

    let matrix = deterministic_matrix(64, 32);
    let output_dim = target_projection_dim(matrix.len(), matrix[0].len());
    let transfer = projection_transfer_bytes(matrix.len(), matrix[0].len(), output_dim).unwrap();
    let happy_before = json!({
        "rows": matrix.len(),
        "input_dim": matrix[0].len(),
        "output_dim": output_dim,
        "seed": seed,
        "transfer_bytes": transfer,
    });
    let cpu = project_cpu(&matrix, seed);
    let gpu = project_gpu(&matrix, seed).unwrap();
    let happy_diff = assert_projection_close(&cpu, &gpu, 2.0e-4, "happy path");
    let happy_after = json!({
        "cpu_shape": {"rows": cpu.input_rows, "input_dim": cpu.input_dim, "output_dim": cpu.output_dim},
        "gpu_shape": {"rows": gpu.input_rows, "input_dim": gpu.input_dim, "output_dim": gpu.output_dim},
        "max_abs_diff": happy_diff,
        "gpu_first_row": gpu.projected.first(),
    });
    println!("ISSUE1503_HAPPY_BEFORE {happy_before}");
    println!("ISSUE1503_HAPPY_AFTER {happy_after}");

    let representative = deterministic_matrix(200, 1_536);
    let representative_output_dim =
        target_projection_dim(representative.len(), representative[0].len());
    let representative_transfer = projection_transfer_bytes(
        representative.len(),
        representative[0].len(),
        representative_output_dim,
    )
    .unwrap();
    let representative_before = json!({
        "rows": representative.len(),
        "input_dim": representative[0].len(),
        "output_dim": representative_output_dim,
        "seed": seed,
        "transfer_bytes": representative_transfer,
    });
    let warmup = project_gpu(&representative, seed).expect("warm CUDA projection");
    assert_eq!(warmup.output_dim, representative_output_dim);

    let cpu_start = Instant::now();
    let representative_cpu = project_cpu(&representative, seed);
    let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
    let gpu_start = Instant::now();
    let representative_gpu = project_gpu(&representative, seed).unwrap();
    let gpu_ms = gpu_start.elapsed().as_secs_f64() * 1_000.0;
    let representative_diff = assert_projection_close(
        &representative_cpu,
        &representative_gpu,
        2.0e-3,
        "representative Assay shape",
    );
    assert!(
        gpu_ms < cpu_ms,
        "expected GPU projection to beat CPU at representative shape: cpu_ms={cpu_ms} gpu_ms={gpu_ms}"
    );
    let benchmark_after = json!({
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "speedup": cpu_ms / gpu_ms,
        "max_abs_diff": representative_diff,
        "gpu_first_row": representative_gpu.projected.first(),
    });
    println!("ISSUE1503_BENCH_BEFORE {representative_before}");
    println!("ISSUE1503_BENCH_AFTER {benchmark_after}");

    let empty_before = json!({"rows": 0, "input_dim": 0});
    let empty_err = project_gpu(&[], seed).expect_err("empty projection input must fail closed");
    let empty_after = error_state(&empty_err);
    assert_eq!(empty_err.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");
    println!("ISSUE1503_EDGE_EMPTY_BEFORE {empty_before}");
    println!("ISSUE1503_EDGE_EMPTY_AFTER {empty_after}");

    let zero_dim = vec![Vec::<f32>::new(), Vec::<f32>::new()];
    let zero_dim_before = json!({"rows": zero_dim.len(), "input_dim": 0});
    let zero_dim_err =
        project_gpu(&zero_dim, seed).expect_err("zero-dimension projection input must fail closed");
    let zero_dim_after = error_state(&zero_dim_err);
    assert_eq!(zero_dim_err.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");
    println!("ISSUE1503_EDGE_ZERO_DIM_BEFORE {zero_dim_before}");
    println!("ISSUE1503_EDGE_ZERO_DIM_AFTER {zero_dim_after}");

    let ragged = vec![vec![1.0_f32, 2.0], vec![3.0]];
    let ragged_before = json!({"rows": ragged.len(), "row_dims": [2, 1]});
    let ragged_err = project_gpu(&ragged, seed).expect_err("ragged input must fail closed");
    let ragged_after = error_state(&ragged_err);
    assert_eq!(ragged_err.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");
    println!("ISSUE1503_EDGE_RAGGED_BEFORE {ragged_before}");
    println!("ISSUE1503_EDGE_RAGGED_AFTER {ragged_after}");

    let nonfinite = vec![vec![1.0_f32, f32::INFINITY], vec![3.0, 4.0]];
    let nonfinite_before =
        json!({"rows": nonfinite.len(), "input_dim": 2, "nonfinite": "row=0 col=1 +inf"});
    let nonfinite_err =
        project_gpu(&nonfinite, seed).expect_err("non-finite input must fail closed");
    let nonfinite_after = error_state(&nonfinite_err);
    assert_eq!(nonfinite_err.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");
    println!("ISSUE1503_EDGE_NONFINITE_BEFORE {nonfinite_before}");
    println!("ISSUE1503_EDGE_NONFINITE_AFTER {nonfinite_after}");

    let overflow_before = json!({"rows": usize::MAX, "input_dim": 2, "output_dim": 2});
    let overflow_err = projection_transfer_bytes(usize::MAX, 2, 2)
        .expect_err("overflowing transfer accounting must fail closed");
    let overflow_after = error_state(&overflow_err);
    assert_eq!(overflow_err.code, "CALYX_FORGE_VRAM_BUDGET");
    println!("ISSUE1503_EDGE_OVERFLOW_BEFORE {overflow_before}");
    println!("ISSUE1503_EDGE_OVERFLOW_AFTER {overflow_after}");

    write_fsv_artifact(json!({
        "artifact_kind": "issue1503.assay-projection-cuda-fsv.v1",
        "source_of_truth": "CALYX_ASSAY_ISSUE1503_FSV_DIR/issue1503-fsv-readback.json",
        "trigger": "cargo test -p calyx-assay --features cuda --test __calyx_integration_suite_issue1502_1504_cuda issue1503_projection_cuda -- --nocapture",
        "device": device_info,
        "happy_path": {
            "before": happy_before,
            "after": happy_after,
            "cpu": cpu,
            "gpu": gpu,
        },
        "representative_benchmark": {
            "before": representative_before,
            "after": benchmark_after,
        },
        "edges": [
            {"name": "empty", "before": empty_before, "after": empty_after},
            {"name": "zero_dim", "before": zero_dim_before, "after": zero_dim_after},
            {"name": "ragged", "before": ragged_before, "after": ragged_after},
            {"name": "nonfinite", "before": nonfinite_before, "after": nonfinite_after},
            {"name": "overflow_transfer_bytes", "before": overflow_before, "after": overflow_after},
        ],
    }));
}

#[cfg(feature = "cuda")]
fn write_fsv_artifact(value: serde_json::Value) {
    let Ok(root) = std::env::var("CALYX_ASSAY_ISSUE1503_FSV_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root).expect("create issue1503 FSV dir");
    let path = root.join("issue1503-fsv-readback.json");
    let bytes = serde_json::to_vec_pretty(&value).expect("serialize issue1503 FSV artifact");
    std::fs::write(&path, bytes).expect("write issue1503 FSV artifact");
    let readback = std::fs::read(&path).expect("read issue1503 FSV artifact");
    let restored: serde_json::Value =
        serde_json::from_slice(&readback).expect("parse issue1503 FSV artifact");
    assert_eq!(restored, value);
    println!(
        "ISSUE1503_FSV_READBACK path={} bytes={}",
        path.display(),
        readback.len()
    );
    assert!(!readback.is_empty());
}
