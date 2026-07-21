use super::*;

#[test]
fn hawkes_em_matches_cpu_oracle() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;
    let events = [0.25, 1.0, 2.5, 0.5, 1.75, 3.0];
    let offsets = [0, 3, 6];
    let observation_end = 4.0;
    let decay = 1.25;
    let iterations = 3;
    let expected = cpu_hawkes_em(&events, &offsets, observation_end, decay, iterations);
    println!(
        "FORGE_HAWKES_BEFORE source=synthetic_oracle events={events:?} offsets={offsets:?} observation_end={observation_end} decay={decay} iterations={iterations} expected={expected:?}"
    );

    let actual = hawkes_em_host(&ctx, &events, &offsets, observation_end, decay, iterations)?;

    assert_close_vec(
        "Hawkes baseline",
        &actual.baseline_rates,
        &expected.baseline_rates,
        1.0e-5,
    );
    assert_close_vec(
        "Hawkes branching",
        &actual.branching_matrix,
        &expected.branching_matrix,
        1.0e-5,
    );
    assert!(
        (actual.spectral_radius - expected.spectral_radius).abs() <= 1.0e-5,
        "Hawkes spectral radius mismatch: actual={} expected={}",
        actual.spectral_radius,
        expected.spectral_radius
    );
    println!("FORGE_HAWKES_AFTER source=device_readback actual={actual:?}");
    Ok(())
}

#[test]
fn hawkes_em_fails_loud_on_edges() -> Result<()> {
    let _guard = test_lock();
    let ctx = init_cuda(0, false)?;

    let empty = hawkes_em_host(&ctx, &[], &[], 4.0, 1.25, 3)
        .expect_err("empty Hawkes process set must fail");
    assert!(matches!(empty, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_HAWKES_EDGE empty before=events0_offsets0 after={empty}");

    let duplicate = hawkes_em_host(&ctx, &[0.5, 0.5], &[0, 2], 4.0, 1.25, 3)
        .expect_err("duplicate process event must fail");
    assert!(matches!(duplicate, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_HAWKES_EDGE duplicate before=events[0.5,0.5]_offsets[0,2] after={duplicate}");

    let nonfinite = hawkes_em_host(&ctx, &[0.5, f64::NAN], &[0, 2], 4.0, 1.25, 3)
        .expect_err("non-finite process event must fail");
    assert!(matches!(nonfinite, ForgeError::NumericalInvariant { .. }));
    println!("FORGE_HAWKES_EDGE nonfinite before=event1_NaN after={nonfinite}");

    let iterations = hawkes_em_host(&ctx, &[0.5, 1.0], &[0, 2], 4.0, 1.25, 1_001)
        .expect_err("iteration limit must fail");
    assert!(matches!(iterations, ForgeError::ShapeMismatch { .. }));
    println!("FORGE_HAWKES_EDGE max_iterations before=1001 after={iterations}");
    Ok(())
}

fn cpu_hawkes_em(
    events: &[f64],
    offsets: &[i32],
    observation_end: f64,
    decay: f64,
    iterations: usize,
) -> CudaHawkesFit {
    let dimensions = offsets.len() - 1;
    let mut baseline = (0..dimensions)
        .map(|source| {
            let count = f64::from(offsets[source + 1] - offsets[source]);
            0.5 * count / observation_end
        })
        .collect::<Vec<_>>();
    let mut branching = vec![0.05_f64; dimensions * dimensions];
    let exposures = (0..dimensions)
        .map(|source| {
            (offsets[source] as usize..offsets[source + 1] as usize)
                .map(|index| 1.0 - (-decay * (observation_end - events[index])).exp())
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut event_process = vec![0_usize; events.len()];
    for source in 0..dimensions {
        for slot in &mut event_process[offsets[source] as usize..offsets[source + 1] as usize] {
            *slot = source;
        }
    }
    let mut kernel_sums = vec![0.0_f64; events.len() * dimensions];
    for (event_index, &target_time) in events.iter().enumerate() {
        for source in 0..dimensions {
            kernel_sums[event_index * dimensions + source] = (offsets[source] as usize
                ..offsets[source + 1] as usize)
                .filter_map(|source_index| {
                    let source_time = events[source_index];
                    (source_time < target_time)
                        .then(|| decay * (-decay * (target_time - source_time)).exp())
                })
                .sum();
        }
    }

    for _ in 0..iterations {
        let mut background = vec![0.0_f64; dimensions];
        let mut triggered = vec![0.0_f64; dimensions * dimensions];
        for event_index in 0..events.len() {
            let target = event_process[event_index];
            let intensity = baseline[target]
                + (0..dimensions)
                    .map(|source| {
                        branching[target * dimensions + source]
                            * kernel_sums[event_index * dimensions + source]
                    })
                    .sum::<f64>();
            background[target] += baseline[target] / intensity;
            for source in 0..dimensions {
                triggered[target * dimensions + source] += branching[target * dimensions + source]
                    * kernel_sums[event_index * dimensions + source]
                    / intensity;
            }
        }
        baseline = background
            .into_iter()
            .map(|count| count / observation_end)
            .collect();
        branching = (0..dimensions * dimensions)
            .map(|index| triggered[index] / exposures[index % dimensions])
            .collect();
    }

    let spectral_radius = power_iteration(&branching, dimensions);
    CudaHawkesFit {
        baseline_rates: baseline.into_iter().map(|value| value as f32).collect(),
        branching_matrix: branching.into_iter().map(|value| value as f32).collect(),
        spectral_radius: spectral_radius as f32,
    }
}

fn power_iteration(matrix: &[f64], dimensions: usize) -> f64 {
    let mut vector = vec![1.0 / dimensions as f64; dimensions];
    let mut eigenvalue = 0.0;
    for _ in 0..100 {
        let next = (0..dimensions)
            .map(|row| {
                (0..dimensions)
                    .map(|column| matrix[row * dimensions + column] * vector[column])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let norm = next.iter().sum::<f64>();
        if norm <= 1.0e-15 {
            return 0.0;
        }
        for row in 0..dimensions {
            vector[row] = next[row] / norm;
        }
        eigenvalue = norm;
    }
    eigenvalue
}
