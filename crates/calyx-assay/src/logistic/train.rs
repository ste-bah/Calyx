use super::*;

pub(super) struct LogisticSummary {
    pub(super) bits: f32,
    pub(super) accuracy: f32,
}

pub(super) fn logistic_heldout_summary(
    samples: &[Vec<f32>],
    labels: &[bool],
    dim: usize,
    split: &GroupSplit,
) -> LogisticSummary {
    let train_samples = split
        .train
        .iter()
        .map(|&idx| samples[idx].clone())
        .collect::<Vec<_>>();
    let train_labels = split
        .train
        .iter()
        .map(|&idx| labels[idx])
        .collect::<Vec<_>>();
    let test_samples = split
        .test
        .iter()
        .map(|&idx| samples[idx].clone())
        .collect::<Vec<_>>();
    let test_labels = split
        .test
        .iter()
        .map(|&idx| labels[idx])
        .collect::<Vec<_>>();
    logistic_train_test_summary(
        &train_samples,
        &train_labels,
        &test_samples,
        &test_labels,
        dim,
    )
}

pub(super) fn logistic_train_test_summary(
    train_samples: &[Vec<f32>],
    train_labels: &[bool],
    test_samples: &[Vec<f32>],
    test_labels: &[bool],
    dim: usize,
) -> LogisticSummary {
    let model = fit_logistic(train_samples, train_labels, dim);
    score_logistic(&model, test_samples, test_labels)
}

fn fit_logistic(samples: &[Vec<f32>], labels: &[bool], dim: usize) -> (Vec<f32>, f32) {
    let mut weights = vec![0.0; dim];
    let mut bias = 0.0;
    let n = labels.len().max(1) as f32;
    for _ in 0..LOGISTIC_STEPS {
        let mut grad = vec![0.0; dim];
        let mut bias_grad = 0.0;
        for (row, label) in samples.iter().zip(labels) {
            let p = sigmoid(dot(row, &weights) + bias);
            let error = p - f32::from(*label);
            for (slot, value) in grad.iter_mut().zip(row) {
                *slot += error * value;
            }
            bias_grad += error;
        }
        for (weight, grad) in weights.iter_mut().zip(grad) {
            *weight -= LOGISTIC_LR * (grad / n + LOGISTIC_L2 * *weight);
        }
        bias -= LOGISTIC_LR * bias_grad / n;
    }
    (weights, bias)
}

fn score_logistic(
    model: &(Vec<f32>, f32),
    samples: &[Vec<f32>],
    labels: &[bool],
) -> LogisticSummary {
    let predictions = samples
        .iter()
        .map(|row| sigmoid(dot(row, &model.0) + model.1) >= 0.5)
        .collect::<Vec<_>>();
    let accuracy = predictions
        .iter()
        .zip(labels)
        .filter(|(prediction, label)| **prediction == **label)
        .count() as f32
        / labels.len().max(1) as f32;
    LogisticSummary {
        bits: binary_mi(labels, &predictions),
        accuracy,
    }
}

fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit.clamp(-40.0, 40.0)).exp())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

pub(super) fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

pub(super) fn sample_sigma(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / (values.len() - 1) as f32;
    variance.sqrt()
}

pub(super) fn seed_ci(mean: f32, sigma: f32, n: usize) -> (f32, f32) {
    let t = match n.saturating_sub(1) {
        0 => 0.0,
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        _ => 1.960,
    };
    let half_width = t * sigma / (n.max(1) as f32).sqrt();
    ((mean - half_width).max(0.0), mean + half_width)
}

pub(super) fn binary_mi(labels: &[bool], predictions: &[bool]) -> f32 {
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
