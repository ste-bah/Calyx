use super::*;
use crate::cuda::test_lock;

const ROWS: usize = 12;
const DIM: usize = 64;

fn panel() -> Vec<f32> {
    (0..ROWS)
        .flat_map(|row| {
            (0..DIM).map(move |col| {
                let stripe = (col % 9) as f32 - 4.0;
                (row + 1) as f32 * 0.13 + stripe * 0.071
            })
        })
        .collect()
}

fn agreement_cpu(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (&a, &b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}

#[test]
fn loom_batch_matches_cpu_and_reuses_constant_dispatch_workspace() -> Result<()> {
    let _guard = test_lock();
    let context = CudaLoomContext::new(0)?;
    let matrix = panel();
    let pairs: Vec<_> = (0..ROWS)
        .flat_map(|left| (left + 1..ROWS).map(move |right| (left, right)))
        .collect();
    let vectors = [
        CudaLoomVectorRequest {
            left_row: 0,
            right_row: 1,
            kind: CudaLoomVectorKind::Delta,
        },
        CudaLoomVectorRequest {
            left_row: 2,
            right_row: 11,
            kind: CudaLoomVectorKind::Interaction,
        },
    ];

    let first = context.execute(&matrix, ROWS, DIM, &pairs, &vectors)?;
    let second = context.execute(&matrix, ROWS, DIM, &pairs, &vectors)?;

    for (&actual, &(left, right)) in first.agreements.iter().zip(&pairs) {
        let left_row = &matrix[left * DIM..(left + 1) * DIM];
        let right_row = &matrix[right * DIM..(right + 1) * DIM];
        assert_close(actual, agreement_cpu(left_row, right_row), 2.0e-4);
    }
    for col in 0..DIM {
        assert_close(
            first.vector_terms[0][col],
            matrix[col] - matrix[DIM + col],
            1.0e-6,
        );
        assert_close(
            first.vector_terms[1][col],
            matrix[2 * DIM + col] * matrix[11 * DIM + col],
            1.0e-6,
        );
    }
    assert_eq!(first, second.clone().with_workspace_reused(false));
    assert!(!first.stats.workspace_reused);
    assert!(second.stats.workspace_reused);
    assert_eq!(second.stats.agreement_pairs, 66);
    assert_eq!(second.stats.vector_terms, 2);
    assert_eq!(second.stats.kernel_launches, 4);
    assert_eq!(second.stats.gemm_calls, 1);
    assert_eq!(second.stats.host_to_device_copies, 6);
    assert_eq!(second.stats.device_to_host_copies, 3);
    assert!(second.stats.host_to_device_copies < pairs.len());
    println!(
        "CUDA_LOOM_BATCH rows={} dim={} pairs={} vectors={} launches={} h2d_copies={} d2h_copies={} workspace_reused={}",
        ROWS,
        DIM,
        pairs.len(),
        vectors.len(),
        second.stats.kernel_launches,
        second.stats.host_to_device_copies,
        second.stats.device_to_host_copies,
        second.stats.workspace_reused,
    );
    Ok(())
}

#[test]
fn loom_batch_fails_closed_on_shape_order_zero_norm_and_nonfinite() -> Result<()> {
    let _guard = test_lock();
    let context = CudaLoomContext::new(0)?;
    let mut matrix = panel();

    let order = context
        .execute(&matrix, ROWS, DIM, &[(2, 1)], &[])
        .expect_err("non-canonical pair must fail");
    assert!(matches!(order, ForgeError::ShapeMismatch { .. }));

    let shape = context
        .execute(&matrix[..matrix.len() - 1], ROWS, DIM, &[(0, 1)], &[])
        .expect_err("matrix shape drift must fail");
    assert!(matches!(shape, ForgeError::ShapeMismatch { .. }));

    matrix[..DIM].fill(0.0);
    let zero = context
        .execute(&matrix, ROWS, DIM, &[(0, 1)], &[])
        .expect_err("zero-norm agreement row must fail");
    assert!(matches!(zero, ForgeError::NumericalInvariant { .. }));

    matrix[0] = f32::NAN;
    let nonfinite = context
        .execute(
            &matrix,
            ROWS,
            DIM,
            &[],
            &[CudaLoomVectorRequest {
                left_row: 0,
                right_row: 1,
                kind: CudaLoomVectorKind::Delta,
            }],
        )
        .expect_err("non-finite vector input must fail");
    assert!(matches!(nonfinite, ForgeError::NumericalInvariant { .. }));
    println!("CUDA_LOOM_EDGES ordering shape zero_norm nonfinite fail_closed");
    Ok(())
}

trait BatchTestExt {
    fn with_workspace_reused(self, reused: bool) -> Self;
}

impl BatchTestExt for CudaLoomBatch {
    fn with_workspace_reused(mut self, reused: bool) -> Self {
        self.stats.workspace_reused = reused;
        self
    }
}
