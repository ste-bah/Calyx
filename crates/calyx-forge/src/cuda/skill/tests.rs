use super::test_support::{cpu_distances, cpu_mst, dense_slot};
use super::*;
use crate::{ForgeError, Result};

#[test]
fn skill_limits_cover_the_production_cap() {
    assert_eq!(CUDA_SKILL_MAX_POINTS, 2_048);
    assert!(SKILL_CUDA_MIN_POINTS < CUDA_SKILL_MAX_POINTS);
    assert!(CUDA_SKILL_MAX_DEVICE_BYTES > 2_048 * 2_048 * size_of::<f64>());
}

#[test]
#[ignore = "requires a CUDA device"]
fn fused_distances_and_mst_match_cpu_and_are_deterministic() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaSkillContext::new(0)?;
    let points = 257;
    let slots = vec![dense_slot(points, 48, 17), dense_slot(points, 16, 31)];
    let expected_distances = cpu_distances(points, &slots)?;
    let expected_edges = cpu_mst(points, &expected_distances, 5);
    let first = context.minimum_spanning_tree_with_distances(points, &slots, 5)?;
    let second = context.minimum_spanning_tree(points, &slots, 5)?;
    let actual_distances = first.distances.as_ref().expect("debug distance readback");
    let max_distance_delta = actual_distances
        .iter()
        .zip(&expected_distances)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_distance_delta <= 1.0e-7,
        "distance delta={max_distance_delta}"
    );
    assert_eq!(first.edges.len(), expected_edges.len());
    for (actual, expected) in first.edges.iter().zip(&expected_edges) {
        assert_eq!(
            (actual.source, actual.destination),
            (expected.0, expected.1)
        );
        assert!((actual.weight - expected.2).abs() <= 1.0e-7);
    }
    assert_eq!(first.edges, second.edges);
    assert!(!second.stats.full_distance_readback);
    assert!(second.stats.device_to_host_bytes < first.stats.device_to_host_bytes);
    assert_eq!(second.stats.kernel_launches, 3);
    println!(
        "ISSUE1517_CUDA_PARITY points={} slots={} distance_delta={:.12} h2d={} d2h={} peak_device={} edges={}",
        points,
        slots.len(),
        max_distance_delta,
        second.stats.host_to_device_bytes,
        second.stats.device_to_host_bytes,
        second.stats.peak_device_bytes,
        second.edges.len(),
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn invalid_degenerate_and_disconnected_inputs_fail_closed() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaSkillContext::new(0)?;
    assert!(matches!(
        context.minimum_spanning_tree(
            CUDA_SKILL_MAX_POINTS + 1,
            &[dense_slot(CUDA_SKILL_MAX_POINTS + 1, 1, 1)],
            1,
        ),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let mut zero = dense_slot(2, 4, 7);
    zero.values[..4].fill(0.0);
    assert!(matches!(
        context.minimum_spanning_tree(2, &[zero], 1),
        Err(ForgeError::NumericalInvariant { .. })
    ));

    let mut nonfinite = dense_slot(2, 4, 9);
    nonfinite.values[3] = f32::NAN;
    assert!(matches!(
        context.minimum_spanning_tree(2, &[nonfinite], 1),
        Err(ForgeError::NumericalInvariant { .. })
    ));

    let disconnected = vec![
        CudaSkillSlot {
            dim: 2,
            point_indices: vec![0],
            values: vec![1.0, 0.0],
        },
        CudaSkillSlot {
            dim: 2,
            point_indices: vec![1],
            values: vec![0.0, 1.0],
        },
    ];
    assert!(matches!(
        context.minimum_spanning_tree(2, &disconnected, 1),
        Err(ForgeError::NumericalInvariant { .. })
    ));

    let duplicate_rows = CudaSkillSlot {
        dim: 2,
        point_indices: vec![0, 1, 2],
        values: vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    };
    let duplicate = context.minimum_spanning_tree(3, &[duplicate_rows], 1)?;
    assert_eq!(duplicate.edges[0].weight, 0.0);
    Ok(())
}
