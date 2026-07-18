use super::*;

#[test]
fn dependence_bulk_wrappers_match_exact_oracles() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let x = (0..64).map(|index| index as f32).collect::<Vec<_>>();
    let y = (0..64).map(|index| (63 - index) as f32).collect::<Vec<_>>();
    let histogram = histogram_counts_host(&ctx, &x, &y, 8)?;
    assert_eq!(histogram.x_counts, vec![8; 8]);
    assert_eq!(histogram.y_counts, vec![8; 8]);
    assert_eq!(histogram.joint_counts.iter().sum::<u64>(), 64);
    assert_eq!(histogram.stats.kernel_launches, 1);

    let categories_x = [0, 0, 0, 1, 1, 2, 2, 2];
    let categories_y = [0, 0, 1, 1, 1, 0, 1, 1];
    let categorical = categorical_counts_host(&ctx, &categories_x, &categories_y, 3, 2)?;
    assert_eq!(categorical.table, vec![2, 1, 0, 2, 1, 2]);

    let rank_x = [1.0, 2.0, 2.0, 4.0];
    let rank_y = [1.0, 2.0, 3.0, 4.0];
    let rank = rank_pair_host(&ctx, &rank_x, &rank_y)?;
    assert_eq!(rank.x_ranks, vec![1.0, 2.5, 2.5, 4.0]);
    assert_eq!(rank.y_ranks, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(rank.x_tie_sizes, vec![0, 2, 0, 0]);
    assert_eq!(rank.y_tie_sizes, vec![0; 4]);
    assert_eq!((rank.concordant, rank.discordant), (5, 0));
    assert_eq!(rank.stats.kernel_launches, 2);

    let copula_x = (0..20).map(|index| index as f64).collect::<Vec<_>>();
    let copula_y = copula_x.clone();
    let copula = copula_terms_host(&ctx, &copula_x, &copula_y, 0.1)?;
    assert_eq!(copula.c_mid_count, 10);
    assert_eq!(copula.lower_tail_count, 2);
    assert_eq!(copula.upper_tail_count, 2);
    assert!(copula.x_tie_sizes.iter().all(|&size| size == 0));
    let expected_hoeffding = cpu_hoeffding_terms(20);
    for (actual, expected) in copula.hoeffding_terms.iter().zip(expected_hoeffding) {
        assert!((actual - expected).abs() <= 1e-15);
    }

    let mic_x = (0..40).map(|index| index as f64).collect::<Vec<_>>();
    let mic = mic_pair_host(&ctx, &mic_x, &mic_x, 9)?;
    assert!((mic.primary_x.score - 1.0).abs() <= 1e-12, "{mic:?}");
    assert!((mic.primary_y.score - 1.0).abs() <= 1e-12, "{mic:?}");
    assert_eq!(mic.stats.kernel_launches, 1);
    assert!(mic.stats.peak_device_bytes >= mic.stats.host_to_device_bytes);

    println!(
        "FORGE_DEPENDENCE_SOT histogram_launches={} rank_launches={} copula_launches={} mic_launches={} mic_score={}",
        histogram.stats.kernel_launches,
        rank.stats.kernel_launches,
        copula.stats.kernel_launches,
        mic.stats.kernel_launches,
        mic.primary_x.score
    );
    Ok(())
}

#[test]
fn dependence_bulk_wrappers_fail_loud_on_edges() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let valid = (0..20).map(|index| index as f64).collect::<Vec<_>>();

    let mut nonfinite = valid.clone();
    nonfinite[7] = f64::NAN;
    assert!(matches!(
        rank_pair_host(&ctx, &nonfinite, &valid),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    assert!(matches!(
        copula_terms_host(&ctx, &valid, &valid, 0.5),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    assert!(matches!(
        categorical_counts_host(&ctx, &[0, 2], &[0, 1], 2, 2),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        mic_pair_host(&ctx, &[1.0; 20], &valid, 7),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    println!(
        "FORGE_DEPENDENCE_EDGE nonfinite=NumericalInvariant invalid_dense=ShapeMismatch constant_mic=NumericalInvariant"
    );
    Ok(())
}

fn cpu_hoeffding_terms(n: usize) -> Vec<f64> {
    let scale = 1.0 / (n + 1) as f64;
    (1..=n)
        .map(|rank| {
            let u = rank as f64 * scale;
            let c = rank as f64 / n as f64;
            (c - u * u).powi(2)
        })
        .collect()
}
