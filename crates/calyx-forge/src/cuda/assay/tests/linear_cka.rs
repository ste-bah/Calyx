use super::*;

#[test]
fn linear_cka_pair_estimates_match_cpu_oracle() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let (values, offsets, dimensions, row_count, tuples) = cka_fixture();
    let expected =
        cpu_linear_cka_pair_estimates(&values, &offsets, &dimensions, row_count, &tuples, false);
    let actual = linear_cka_pair_estimates_host(
        &ctx,
        &values,
        &offsets,
        &dimensions,
        row_count,
        &tuples,
        false,
    )?;

    assert_close_vec(
        "linear CKA raw",
        &actual.raw_signed_point,
        &expected.raw_signed_point,
        5e-5,
    );
    assert_close_vec(
        "linear CKA point",
        &actual.redundancy_point,
        &expected.redundancy_point,
        5e-5,
    );
    assert_close_vec(
        "linear CKA SE",
        &actual.mc_standard_error,
        &expected.mc_standard_error,
        5e-5,
    );
    assert_close_vec(
        "linear CKA gate",
        &actual.mc_gate_upper_estimate,
        &expected.mc_gate_upper_estimate,
        5e-5,
    );
    println!(
        "FORGE_CKA_SOT source=linear_cka_pair_estimates_host_readback pairs={} raw={:?} point={:?} se={:?} gate={:?}",
        actual.raw_signed_point.len(),
        actual.raw_signed_point,
        actual.redundancy_point,
        actual.mc_standard_error,
        actual.mc_gate_upper_estimate
    );
    Ok(())
}

#[test]
fn linear_cka_pair_estimates_fail_loud_on_edges() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let (values, offsets, dimensions, row_count, tuples) = cka_fixture();

    let mut nonfinite = values.clone();
    nonfinite[11] = f32::NAN;
    let nan = linear_cka_pair_estimates_host(
        &ctx,
        &nonfinite,
        &offsets,
        &dimensions,
        row_count,
        &tuples,
        false,
    )
    .expect_err("non-finite CKA values must fail");
    assert!(matches!(nan, ForgeError::NumericalInvariant { .. }));
    println!("FORGE_CKA_EDGE nonfinite before=nan_at_flat_11 after={nan}");

    let mut bad_tuple = tuples.clone();
    bad_tuple[2] = bad_tuple[1];
    let invalid_tuple = linear_cka_pair_estimates_host(
        &ctx,
        &values,
        &offsets,
        &dimensions,
        row_count,
        &bad_tuple,
        false,
    )
    .expect_err("duplicate CKA tuple index must fail");
    assert!(matches!(invalid_tuple, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_CKA_EDGE invalid_tuple before=duplicate_tuple0 after={invalid_tuple}");

    let mut bad_offsets = offsets.clone();
    bad_offsets[1] -= 1;
    let invalid_offsets = linear_cka_pair_estimates_host(
        &ctx,
        &values,
        &bad_offsets,
        &dimensions,
        row_count,
        &tuples,
        false,
    )
    .expect_err("bad CKA lens offsets must fail");
    assert!(matches!(invalid_offsets, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_CKA_EDGE invalid_offsets before=offset1_minus1 after={invalid_offsets}");

    let constant = vec![1.0_f32; values.len()];
    let zero_energy = linear_cka_pair_estimates_host(
        &ctx,
        &constant,
        &offsets,
        &dimensions,
        row_count,
        &tuples,
        false,
    )
    .expect_err("zero centered CKA energy must fail");
    assert!(matches!(zero_energy, ForgeError::NumericalInvariant { .. }));
    println!("FORGE_CKA_EDGE zero_energy before=constant_values after={zero_energy}");
    Ok(())
}
