use super::*;

pub(super) fn matrix(rows: usize, dim: usize, phase: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * dim);
    for row in 0..rows {
        for col in 0..dim {
            let r = row as f32 + 1.0;
            let c = col as f32 + 1.0;
            out.push((r * phase + c * 0.19).sin() + (r * c * 0.013).cos() * 0.5);
        }
    }
    out
}

pub(super) type LogisticFixture = (
    Vec<f32>,
    Vec<i32>,
    usize,
    usize,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
    Vec<i32>,
);

#[derive(Debug)]
pub(super) struct CpuLogisticSummaries {
    pub(super) bits: Vec<f32>,
    pub(super) accuracy: Vec<f32>,
}

pub(super) fn logistic_fixture() -> LogisticFixture {
    let n = 48usize;
    let dim = 4usize;
    let mut samples = Vec::with_capacity(n * dim);
    let mut labels = Vec::with_capacity(n);
    for row in 0..n {
        let label = if row % 2 == 0 { 0 } else { 1 };
        let sign = if label == 1 { 1.0 } else { -1.0 };
        labels.push(label);
        samples.push(sign * 3.0);
        samples.push((row as f32 + 1.0) * 0.01);
        samples.push(sign * 0.5 + (row % 3) as f32 * 0.02);
        samples.push(1.0 - row as f32 * 0.005);
    }

    let fit0_train = (0..32).collect::<Vec<_>>();
    let fit0_test = (32..48).collect::<Vec<_>>();
    let fit1_train = (8..48).collect::<Vec<_>>();
    let fit1_test = (0..8).collect::<Vec<_>>();
    let fit2_test = vec![3, 4, 11, 12, 21, 22, 35, 36];
    let fit2_train = (0..n)
        .filter(|row| !fit2_test.contains(row))
        .collect::<Vec<_>>();

    let mut train_offsets = vec![0];
    let mut train_indices = Vec::new();
    let mut test_offsets = vec![0];
    let mut test_indices = Vec::new();
    push_split(&mut train_offsets, &mut train_indices, &fit0_train);
    push_split(&mut test_offsets, &mut test_indices, &fit0_test);
    push_split(&mut train_offsets, &mut train_indices, &fit1_train);
    push_split(&mut test_offsets, &mut test_indices, &fit1_test);
    push_split(&mut train_offsets, &mut train_indices, &fit2_train);
    push_split(&mut test_offsets, &mut test_indices, &fit2_test);
    (
        samples,
        labels,
        n,
        dim,
        train_offsets,
        train_indices,
        test_offsets,
        test_indices,
    )
}

pub(super) fn push_split(offsets: &mut Vec<i32>, indices: &mut Vec<i32>, rows: &[usize]) {
    indices.extend(rows.iter().map(|&row| row as i32));
    offsets.push(indices.len() as i32);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cpu_logistic_summaries(
    samples: &[f32],
    labels: &[i32],
    _n: usize,
    dim: usize,
    train_offsets: &[i32],
    train_indices: &[i32],
    test_offsets: &[i32],
    test_indices: &[i32],
    steps: usize,
    lr: f32,
    l2: f32,
) -> CpuLogisticSummaries {
    let fit_count = train_offsets.len() - 1;
    let mut bits = Vec::with_capacity(fit_count);
    let mut accuracy = Vec::with_capacity(fit_count);
    for fit in 0..fit_count {
        let train = &train_indices[train_offsets[fit] as usize..train_offsets[fit + 1] as usize];
        let test = &test_indices[test_offsets[fit] as usize..test_offsets[fit + 1] as usize];
        let mut weights = vec![0.0_f32; dim];
        let mut bias = 0.0_f32;
        let train_n = train.len().max(1) as f32;
        for _ in 0..steps {
            let mut grad = vec![0.0_f32; dim];
            let mut bias_grad = 0.0_f32;
            for &row_i32 in train {
                let row = row_i32 as usize;
                let p = logistic_sigmoid(logistic_dot(samples, row, dim, &weights) + bias);
                let error = p - labels[row] as f32;
                for col in 0..dim {
                    grad[col] += error * samples[row * dim + col];
                }
                bias_grad += error;
            }
            for col in 0..dim {
                weights[col] -= lr * (grad[col] / train_n + l2 * weights[col]);
            }
            bias -= lr * bias_grad / train_n;
        }

        let mut predictions = Vec::with_capacity(test.len());
        let mut truths = Vec::with_capacity(test.len());
        let mut correct = 0usize;
        for &row_i32 in test {
            let row = row_i32 as usize;
            let prediction =
                logistic_sigmoid(logistic_dot(samples, row, dim, &weights) + bias) >= 0.5;
            let truth = labels[row] != 0;
            correct += usize::from(prediction == truth);
            predictions.push(prediction);
            truths.push(truth);
        }
        bits.push(binary_mi_for_bools(&truths, &predictions));
        accuracy.push(correct as f32 / test.len().max(1) as f32);
    }
    CpuLogisticSummaries { bits, accuracy }
}

pub(super) fn cka_fixture() -> (Vec<f32>, Vec<i32>, Vec<i32>, usize, Vec<i32>) {
    let row_count = 64usize;
    let dimensions = vec![3_i32, 4, 2];
    let mut values = Vec::new();
    let mut offsets = vec![0_i32];
    for (lens, &dim) in dimensions.iter().enumerate() {
        let dim = dim as usize;
        for row in 0..row_count {
            for col in 0..dim {
                let t = row as f32 * 0.071 + col as f32 * 0.19;
                let base = t.sin() + (1.7 * t).cos() * 0.25;
                let value = match lens {
                    0 => base + col as f32 * 0.03,
                    1 => {
                        let source = base_rows_for_cka(row, col % 3);
                        source * if col % 2 == 0 { 1.8 } else { -1.2 } + 0.4
                    }
                    _ => (base * 0.4 + (row as f32 * 0.13).cos() * 0.6) + col as f32 * 0.02,
                };
                values.push(value);
            }
        }
        offsets.push(values.len() as i32);
    }
    let tuples = sampled_cka_tuples(row_count, 4_096);
    (values, offsets, dimensions, row_count, tuples)
}

pub(super) fn base_rows_for_cka(row: usize, col: usize) -> f32 {
    let t = row as f32 * 0.071 + col as f32 * 0.19;
    t.sin() + (1.7 * t).cos() * 0.25 + col as f32 * 0.03
}

pub(super) fn sampled_cka_tuples(row_count: usize, count: usize) -> Vec<i32> {
    let mut tuples = Vec::with_capacity(count * 4);
    for seed in 0..count {
        let mut tuple = [usize::MAX; 4];
        for position in 0..4 {
            let mut candidate = (seed * 37 + position * 11 + position * position) % row_count;
            while tuple[..position].contains(&candidate) {
                candidate = (candidate + 1) % row_count;
            }
            tuple[position] = candidate;
        }
        tuple.sort_unstable();
        tuples.extend(tuple.iter().map(|&value| value as i32));
    }
    tuples
}

pub(super) fn cpu_linear_cka_pair_estimates(
    values: &[f32],
    offsets: &[i32],
    dimensions: &[i32],
    row_count: usize,
    tuples: &[i32],
    exact: bool,
) -> CudaLinearCkaPairEstimates {
    let lens_count = dimensions.len();
    let tuple_count = tuples.len() / 4;
    let mut sketches = Vec::with_capacity(lens_count);
    for lens in 0..lens_count {
        let dim = dimensions[lens] as usize;
        let offset = offsets[lens] as usize;
        let inverse = cpu_cka_inverse_energy(values, offset, row_count, dim);
        let mut sketch = Vec::with_capacity(tuple_count);
        for tuple in tuples.chunks_exact(4) {
            sketch.push(cpu_cka_tuple_z(
                values, offset, row_count, dim, tuple, inverse,
            ));
        }
        sketches.push(sketch);
    }

    let pair_count = lens_count * (lens_count - 1) / 2;
    let mut raw_signed_point = Vec::with_capacity(pair_count);
    let mut redundancy_point = Vec::with_capacity(pair_count);
    let mut mc_standard_error = Vec::with_capacity(pair_count);
    let mut mc_gate_upper_estimate = Vec::with_capacity(pair_count);
    for left in 0..lens_count {
        for right in (left + 1)..lens_count {
            let estimate =
                cpu_cka_pair_estimate(&sketches[left], &sketches[right], tuple_count, exact);
            raw_signed_point.push(estimate.0 as f32);
            redundancy_point.push(estimate.1 as f32);
            mc_standard_error.push(estimate.2 as f32);
            mc_gate_upper_estimate.push(estimate.3 as f32);
        }
    }
    CudaLinearCkaPairEstimates {
        raw_signed_point,
        redundancy_point,
        mc_standard_error,
        mc_gate_upper_estimate,
    }
}

pub(super) fn cpu_cka_inverse_energy(
    values: &[f32],
    offset: usize,
    row_count: usize,
    dim: usize,
) -> f64 {
    let mut energy = 0.0;
    for col in 0..dim {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for row in 0..row_count {
            let value = values[offset + row * dim + col] as f64;
            sum += value;
            sum_sq += value * value;
        }
        energy += sum_sq - sum * sum / row_count as f64;
    }
    1.0 / energy
}

pub(super) fn cpu_cka_tuple_z(
    values: &[f32],
    offset: usize,
    _row_count: usize,
    dim: usize,
    tuple: &[i32],
    inverse: f64,
) -> [f64; 3] {
    let rows = [
        tuple[0] as usize,
        tuple[1] as usize,
        tuple[2] as usize,
        tuple[3] as usize,
    ];
    let mut r = 0.0;
    let mut s = 0.0;
    for col in 0..dim {
        let a = values[offset + rows[0] * dim + col] as f64;
        let b = values[offset + rows[1] * dim + col] as f64;
        let c = values[offset + rows[2] * dim + col] as f64;
        let d = values[offset + rows[3] * dim + col] as f64;
        r += (a - d) * (b - c);
        s += (a - c) * (b - d);
    }
    let r = r * inverse;
    let s = s * inverse;
    [(r + s) / 6.0, (-2.0 * r + s) / 6.0, (r - 2.0 * s) / 6.0]
}

pub(super) fn cpu_cka_pair_estimate(
    left: &[[f64; 3]],
    right: &[[f64; 3]],
    tuple_count: usize,
    exact: bool,
) -> (f64, f64, f64, f64) {
    let mut blocks = [(0.0, 0.0, 0.0); 32];
    for (block, slot) in blocks.iter_mut().enumerate() {
        let start = block * tuple_count / 32;
        let end = (block + 1) * tuple_count / 32;
        for idx in start..end {
            let cross = dot3_f64(&left[idx], &right[idx]);
            let self_a = dot3_f64(&left[idx], &left[idx]);
            let self_b = dot3_f64(&right[idx], &right[idx]);
            slot.0 += cross;
            slot.1 += self_a;
            slot.2 += self_b;
        }
    }
    let total = blocks.iter().fold((0.0, 0.0, 0.0), |mut acc, value| {
        acc.0 += value.0;
        acc.1 += value.1;
        acc.2 += value.2;
        acc
    });
    let raw = checked_cosine_f64(total).clamp(-1.0, 1.0);
    let point = raw.max(0.0);
    let se = if exact {
        0.0
    } else {
        let mut leave_out = [0.0; 32];
        for block in 0..32 {
            leave_out[block] = checked_cosine_f64((
                total.0 - blocks[block].0,
                total.1 - blocks[block].1,
                total.2 - blocks[block].2,
            ));
        }
        let mean = leave_out.iter().sum::<f64>() / leave_out.len() as f64;
        let squared = leave_out
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        ((31.0 / 32.0) * squared).sqrt()
    };
    let gate = (point + 4.0 * se).min(1.0);
    (raw, point, se, gate)
}

pub(super) fn dot3_f64(left: &[f64; 3], right: &[f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn checked_cosine_f64((cross, self_a, self_b): (f64, f64, f64)) -> f64 {
    cross / (self_a * self_b).sqrt()
}

pub(super) fn logistic_sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit.clamp(-40.0, 40.0)).exp())
}

pub(super) fn logistic_dot(samples: &[f32], row: usize, dim: usize, weights: &[f32]) -> f32 {
    (0..dim)
        .map(|col| samples[row * dim + col] * weights[col])
        .sum()
}

pub(super) fn binary_mi_for_bools(labels: &[bool], predictions: &[bool]) -> f32 {
    let n = labels.len().max(1) as f32;
    let mut joint = [[0.0_f32; 2]; 2];
    for (label, prediction) in labels.iter().zip(predictions) {
        joint[*label as usize][*prediction as usize] += 1.0;
    }
    let py = [
        (joint[0][0] + joint[0][1]) / n,
        (joint[1][0] + joint[1][1]) / n,
    ];
    let pp = [
        (joint[0][0] + joint[1][0]) / n,
        (joint[0][1] + joint[1][1]) / n,
    ];
    let mut mi = 0.0;
    for y in 0..2 {
        for p in 0..2 {
            let joint_p = joint[y][p] / n;
            if joint_p > 0.0 && py[y] > 0.0 && pp[p] > 0.0 {
                mi += joint_p * (joint_p / (py[y] * pp[p])).log2();
            }
        }
    }
    mi.max(0.0)
}
