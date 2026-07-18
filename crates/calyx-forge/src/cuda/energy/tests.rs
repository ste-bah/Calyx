use super::test_support::{cpu_descent, deterministic_initial, deterministic_members};
use super::*;
use crate::{ForgeError, Result};

#[test]
fn energy_limits_are_bounded_and_nonzero() {
    assert!(ENERGY_CUDA_MIN_ELEMENTS > 0);
    assert!(CUDA_ENERGY_PINNED_CHUNK_BYTES > 0);
    assert!(CUDA_ENERGY_PINNED_CHUNK_BYTES < CUDA_ENERGY_MAX_DEVICE_BYTES);
}

#[test]
#[ignore = "requires a CUDA device"]
fn resident_descent_is_deterministic_for_fixed_inputs() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaEnergyContext::new(0)?;
    let dim = 64_usize;
    let members = deterministic_members(257, dim);
    let member_refs = members.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let initial = deterministic_initial(dim);
    let first = context.descend(&initial, &member_refs, 1.0, 20, 1.0e-4)?;
    let second = context.descend(&initial, &member_refs, 1.0, 20, 1.0e-4)?;
    assert_eq!(first.vector, second.vector);
    assert_eq!(first.steps_taken, second.steps_taken);
    assert_eq!(first.converged, second.converged);
    assert_eq!(first.final_energy, second.final_energy);
    assert_eq!(
        first.stats.host_to_device_bytes,
        second.stats.host_to_device_bytes
    );
    println!(
        "ISSUE1521_CUDA_DETERMINISM members={} dim={} steps={} converged={} energy={:.8} h2d={} d2h={} kernels={} peak_device={} peak_pinned={}",
        first.stats.members,
        first.stats.dim,
        first.steps_taken,
        first.converged,
        first.final_energy,
        first.stats.host_to_device_bytes,
        first.stats.device_to_host_bytes,
        first.stats.kernel_launches,
        first.stats.peak_device_bytes,
        first.stats.peak_pinned_staging_bytes,
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn resident_descent_matches_cpu_contract_and_edges() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaEnergyContext::new(0)?;
    let dim = 64_usize;
    let members = deterministic_members(1_025, dim);
    let member_refs = members.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let initial = deterministic_initial(dim);
    let expected = cpu_descent(&initial, &member_refs, 1.75, 20, 1.0e-4)?;
    let actual = context.descend(&initial, &member_refs, 1.75, 20, 1.0e-4)?;
    assert_eq!(actual.steps_taken, expected.steps_taken);
    assert_eq!(actual.converged, expected.converged);
    assert!((actual.final_energy - expected.final_energy).abs() <= 2.0e-4);
    let max_delta = actual
        .vector
        .iter()
        .zip(&expected.vector)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_delta <= 2.0e-4, "max vector delta={max_delta}");
    let matrix_bytes = members.len() * dim * size_of::<f32>();
    assert!(actual.stats.host_to_device_bytes < (matrix_bytes * 2) as u64);
    assert_eq!(
        actual.stats.device_to_host_bytes,
        (size_of::<u32>() + dim * size_of::<f32>() + size_of::<f32>() + 3 * size_of::<u32>())
            as u64
    );

    let mut previous = context
        .descend(&initial, &member_refs, 1.75, 0, 1.0e-4)?
        .final_energy;
    for steps in 1..=actual.steps_taken {
        let next = context
            .descend(&initial, &member_refs, 1.75, steps, 1.0e-4)?
            .final_energy;
        assert!(
            next <= previous + 2.0e-4,
            "step={steps} {next} > {previous}"
        );
        previous = next;
    }

    let zero = vec![0.0_f32; dim];
    let axis = (0..dim)
        .map(|col| if col == 0 { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let uniform = context.descend(&initial, &[&zero, &axis], 0.0, 20, 1.0e-4)?;
    assert!(uniform.converged);
    assert!((uniform.vector[0] - 1.0).abs() <= 1.0e-6);
    assert!(
        uniform.vector[1..]
            .iter()
            .all(|value| value.abs() <= 1.0e-6)
    );

    let zero_error = context
        .descend(&initial, &[&zero], 1.0, 20, 1.0e-4)
        .expect_err("positive-beta zero member must fail");
    assert!(matches!(zero_error, ForgeError::NumericalInvariant { .. }));
    let no_steps = context.descend(&initial, &member_refs, 1.0, 0, 1.0e-4)?;
    assert_eq!(no_steps.vector, initial);
    assert_eq!(no_steps.steps_taken, 0);
    assert!(!no_steps.converged);
    println!(
        "ISSUE1521_CUDA_PARITY members={} dim={} steps={} energy_cpu={:.8} energy_gpu={:.8} max_delta={:.8} h2d={} d2h={} matrix_bytes={}",
        members.len(),
        dim,
        actual.steps_taken,
        expected.final_energy,
        actual.final_energy,
        max_delta,
        actual.stats.host_to_device_bytes,
        actual.stats.device_to_host_bytes,
        matrix_bytes,
    );
    Ok(())
}
