pub(super) struct KernelMatrix {
    pub(super) n: usize,
    pub(super) values: Vec<f64>,
}

impl KernelMatrix {
    pub(super) fn new(samples: &[Vec<f64>], bandwidth: f64) -> Self {
        let n = samples.len();
        let mut values = vec![0.0; n * n];
        for i in 0..n {
            values[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let value = gaussian_kernel(&samples[i], &samples[j], bandwidth);
                values[i * n + j] = value;
                values[j * n + i] = value;
            }
        }
        Self { n, values }
    }

    pub(super) fn mmd2(&self, x: &[usize], y: &[usize]) -> f64 {
        self.mean(x, x) + self.mean(y, y) - 2.0 * self.mean(x, y)
    }

    pub(super) fn mmd2_unbiased(&self, x: &[usize], y: &[usize]) -> f64 {
        self.off_diagonal_mean(x) + self.off_diagonal_mean(y) - 2.0 * self.mean(x, y)
    }

    pub(super) fn off_diagonal_mean(&self, indices: &[usize]) -> f64 {
        debug_assert!(indices.len() > 1);
        let mut sum = 0.0;
        for &i in indices {
            for &j in indices {
                if i != j {
                    sum += self.values[i * self.n + j];
                }
            }
        }
        sum / (indices.len() * (indices.len() - 1)) as f64
    }

    pub(super) fn mean(&self, left: &[usize], right: &[usize]) -> f64 {
        let mut sum = 0.0;
        for &i in left {
            for &j in right {
                sum += self.values[i * self.n + j];
            }
        }
        sum / (left.len() * right.len()) as f64
    }
}

fn gaussian_kernel(a: &[f64], b: &[f64], bandwidth: f64) -> f64 {
    (-squared_distance(a, b) / (2.0 * bandwidth * bandwidth)).exp()
}

pub(super) fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum()
}

pub(super) fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    let rank = ((sorted_values.len() - 1) as f64 * q).ceil() as usize;
    sorted_values[rank.min(sorted_values.len() - 1)]
}
