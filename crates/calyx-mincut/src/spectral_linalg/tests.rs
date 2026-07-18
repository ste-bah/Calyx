use super::*;
use std::cell::Cell;

#[test]
fn decomposition_applies_operator_once_per_accepted_basis_vector() {
    let matrix = vec![
        vec![4.0, 1.0, 0.0, 0.0],
        vec![1.0, 3.0, 1.0, 0.0],
        vec![0.0, 1.0, 2.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
    ];
    let calls = Cell::new(0_usize);
    let decomposition = lanczos_decomposition_operator(4, 4, 4, &mut |vector| {
        calls.set(calls.get() + 1);
        dense_mat_vec(&matrix, vector)
    })
    .expect("decompose symmetric operator");

    assert_eq!(decomposition.basis.len(), 4);
    assert_eq!(
        calls.get(),
        4,
        "one sparse matvec per accepted basis vector"
    );
}

#[test]
fn recorded_projection_matches_reference_qtaq_after_restart() {
    let matrix = vec![
        vec![2.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 5.0, 1.0],
        vec![0.0, 0.0, 1.0, 4.0],
    ];
    let decomposition =
        lanczos_decomposition_operator(4, 4, 4, &mut |vector| dense_mat_vec(&matrix, vector))
            .expect("restart decomposition");
    let reference = reference_projection(&decomposition.basis, &matrix);

    for (row, reference_row) in reference.iter().enumerate() {
        for (column, reference_value) in reference_row.iter().enumerate() {
            assert_close(
                decomposition.projected[row][column],
                *reference_value,
                2.0e-5,
            );
            assert_close(
                decomposition.projected[row][column],
                decomposition.projected[column][row],
                1.0e-7,
            );
        }
    }
}

#[test]
fn malformed_operator_product_fails_loud() {
    let error = lanczos_decomposition_operator(3, 2, 2, &mut |_vector| vec![1.0, f32::NAN])
        .expect_err("malformed operator result must fail");
    assert_eq!(error.code(), "CALYX_SPECTRAL_INVALID_OPERATOR");
}

#[test]
#[ignore = "manual full-state verification for issue #1531"]
fn issue1531_manual_fsv_records_operator_calls_and_projection_bytes() {
    let artifact = std::env::var_os("CALYX_ISSUE1531_FSV_ARTIFACT")
        .map(std::path::PathBuf::from)
        .expect("set CALYX_ISSUE1531_FSV_ARTIFACT to a fresh JSON path");
    assert!(!artifact.exists(), "artifact path must be fresh");
    let matrix = vec![
        vec![4.0, 1.0, 0.0, 0.0],
        vec![1.0, 3.0, 1.0, 0.0],
        vec![0.0, 1.0, 2.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
    ];
    let calls = Cell::new(0usize);
    let decomposition = lanczos_decomposition_operator(4, 4, 4, &mut |vector| {
        calls.set(calls.get() + 1);
        dense_mat_vec(&matrix, vector)
    })
    .expect("happy-path decomposition");
    let reference = reference_projection(&decomposition.basis, &matrix);
    let max_projection_error = decomposition
        .projected
        .iter()
        .flatten()
        .zip(reference.iter().flatten())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(calls.get(), decomposition.basis.len());
    assert!(max_projection_error <= 2.0e-5);

    let zero_calls = Cell::new(0usize);
    let zero = lanczos_eigen_operator(4, 0, 4, |_vector| {
        zero_calls.set(zero_calls.get() + 1);
        vec![0.0; 4]
    })
    .expect("zero target is empty");
    assert_eq!(zero_calls.get(), 0);

    let malformed_calls = Cell::new(0usize);
    let malformed = lanczos_decomposition_operator(3, 2, 2, &mut |_vector| {
        malformed_calls.set(malformed_calls.get() + 1);
        vec![1.0, f32::NAN]
    })
    .expect_err("malformed result fails closed");

    let restart_matrix = vec![
        vec![2.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 5.0, 1.0],
        vec![0.0, 0.0, 1.0, 4.0],
    ];
    let restart_calls = Cell::new(0usize);
    let restart = lanczos_decomposition_operator(4, 4, 4, &mut |vector| {
        restart_calls.set(restart_calls.get() + 1);
        dense_mat_vec(&restart_matrix, vector)
    })
    .expect("rank-block restart");
    let report = serde_json::json!({
        "issue": 1531,
        "source_of_truth": "matvec call counters and persisted projection JSON",
        "before": {"operator_calls": 0, "basis_vectors": 0},
        "happy_after": {
            "operator_calls": calls.get(),
            "basis_vectors": decomposition.basis.len(),
            "projected": decomposition.projected,
            "max_projection_error": max_projection_error,
        },
        "edge_zero_target_after": {"operator_calls": zero_calls.get(), "values": zero.0, "vectors": zero.1},
        "edge_malformed_after": {"operator_calls": malformed_calls.get(), "error_code": malformed.code()},
        "edge_restart_after": {"operator_calls": restart_calls.get(), "basis_vectors": restart.basis.len(), "projected": restart.projected},
    });
    if let Some(parent) = artifact.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&artifact, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    let readback: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&artifact).unwrap()).unwrap();
    assert_eq!(readback["happy_after"]["operator_calls"], 4);
    println!(
        "ISSUE1531_FSV={}",
        serde_json::to_string(&readback).unwrap()
    );
}

fn reference_projection(basis: &[Vec<f32>], matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut projected = vec![vec![0.0; basis.len()]; basis.len()];
    for (column, vector) in basis.iter().enumerate() {
        let product = dense_mat_vec(matrix, vector);
        for (row, basis_vector) in basis.iter().enumerate() {
            projected[row][column] = dot(basis_vector, &product);
        }
    }
    projected
}

fn dense_mat_vec(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn assert_close(left: f32, right: f32, tolerance: f32) {
    assert!(
        (left - right).abs() <= tolerance,
        "left={left} right={right} tolerance={tolerance}"
    );
}
