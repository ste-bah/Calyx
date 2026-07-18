use super::*;

#[test]
fn neighbor_wrappers_match_cpu_oracles() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let n = 64usize;
    let k = 3usize;
    let x = matrix(n, 2, 0.17);
    let y = matrix(n, 1, 0.41);
    let ksg = ksg_continuous_counts_host(&ctx, &x, &y, n, 2, 1, k)?;
    let (expected_radii, expected_nx, expected_ny) = cpu_ksg_counts(&x, &y, n, 2, 1, k);
    assert_close_vec("ksg radii", &ksg.radii, &expected_radii, 0.0);
    assert_eq!(ksg.nx, expected_nx);
    assert_eq!(ksg.ny, expected_ny);
    println!(
        "FORGE_NEIGHBOR_SOT ksg rows={} first_radius={} first_nx={} first_ny={}",
        ksg.radii.len(),
        ksg.radii[0],
        ksg.nx[0],
        ksg.ny[0]
    );

    let entropy = entropy_radii_host(&ctx, &x, n, 2, k)?;
    let expected_entropy = cpu_entropy_radii(&x, n, 2, k);
    assert_close_vec("entropy radii", &entropy, &expected_entropy, 0.0);
    println!(
        "FORGE_NEIGHBOR_SOT entropy rows={} first_radius={}",
        entropy.len(),
        entropy[0]
    );

    let labels = (0..n)
        .map(|idx| if idx % 2 == 0 { 0 } else { 1 })
        .collect::<Vec<_>>();
    let mixed = mixed_ksg_counts_host(&ctx, &x, &labels, n, 2, k)?;
    let (mixed_radii, same, full) = cpu_mixed_counts(&x, &labels, n, 2, k);
    assert_close_vec("mixed radii", &mixed.radii, &mixed_radii, 0.0);
    assert_eq!(mixed.same_class_counts, same);
    assert_eq!(mixed.full_counts, full);
    println!(
        "FORGE_NEIGHBOR_SOT mixed rows={} first_radius={} first_same={} first_full={}",
        mixed.radii.len(),
        mixed.radii[0],
        mixed.same_class_counts[0],
        mixed.full_counts[0]
    );

    let ccm_embedding = matrix(48, 3, 0.29);
    let target = (0..48)
        .map(|idx| ((idx as f32) * 0.13).sin() + 0.25 * ((idx as f32) * 0.07).cos())
        .collect::<Vec<_>>();
    let libraries = [18usize, 36usize];
    let predictions =
        ccm_simplex_predictions_host(&ctx, &ccm_embedding, &target, 48, 3, 4, &libraries)?;
    assert_eq!(predictions.library_predictions.len(), libraries.len());
    for (lib_idx, &library_size) in libraries.iter().enumerate() {
        let expected = cpu_ccm_predictions(&ccm_embedding, &target, 48, 3, 4, library_size);
        assert_close_vec(
            "ccm predictions",
            &predictions.library_predictions[lib_idx],
            &expected,
            1e-5,
        );
    }
    println!(
        "FORGE_NEIGHBOR_SOT ccm groups={} first_len={} first_prediction={}",
        predictions.library_predictions.len(),
        predictions.library_predictions[0].len(),
        predictions.library_predictions[0][0]
    );
    Ok(())
}

#[test]
fn neighbor_wrappers_fail_loud_on_edges() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let valid = matrix(64, 2, 0.11);
    let labels = (0..64)
        .map(|idx| if idx % 2 == 0 { 0 } else { 1 })
        .collect::<Vec<_>>();

    let mut nonfinite = valid.clone();
    nonfinite[7] = f32::NAN;
    assert!(matches!(
        ksg_continuous_counts_host(&ctx, &nonfinite, &valid, 64, 2, 2, 3),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    println!("FORGE_NEIGHBOR_EDGE nonfinite before=nan_at_7 after=NumericalInvariant");

    assert!(matches!(
        entropy_radii_host(&ctx, &valid[..63], 64, 2, 3),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    println!("FORGE_NEIGHBOR_EDGE length_mismatch before=127_values after=ShapeMismatch");

    assert!(matches!(
        mixed_ksg_counts_host(&ctx, &valid, &labels, 64, 2, 33),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    println!("FORGE_NEIGHBOR_EDGE k_over_limit before=k33 after=ShapeMismatch");

    assert!(matches!(
        ccm_simplex_predictions_host(&ctx, &valid[..64], &valid[..32], 32, 2, 3, &[]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    println!("FORGE_NEIGHBOR_EDGE empty_libraries before=0 after=ShapeMismatch");
    Ok(())
}
