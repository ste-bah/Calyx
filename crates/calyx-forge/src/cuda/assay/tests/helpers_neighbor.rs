pub(super) fn cpu_ksg_counts(
    x: &[f32],
    y: &[f32],
    n: usize,
    dim_x: usize,
    dim_y: usize,
    k: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut radii = Vec::with_capacity(n);
    let mut nx = Vec::with_capacity(n);
    let mut ny = Vec::with_capacity(n);
    for row in 0..n {
        let mut distances = (0..n)
            .filter(|&col| col != row)
            .map(|col| cheb(x, row, col, dim_x).max(cheb(y, row, col, dim_y)))
            .collect::<Vec<_>>();
        distances.sort_by(f32::total_cmp);
        let radius = distances[k - 1];
        radii.push(radius);
        nx.push(
            (0..n)
                .filter(|&col| col != row && cheb(x, row, col, dim_x) < radius)
                .count(),
        );
        ny.push(
            (0..n)
                .filter(|&col| col != row && cheb(y, row, col, dim_y) < radius)
                .count(),
        );
    }
    (radii, nx, ny)
}

pub(super) fn cpu_entropy_radii(values: &[f32], n: usize, dim: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|row| {
            let mut distances = (0..n)
                .filter(|&col| col != row)
                .map(|col| cheb(values, row, col, dim))
                .collect::<Vec<_>>();
            distances.sort_by(f32::total_cmp);
            distances[k - 1]
        })
        .collect()
}

pub(super) fn cpu_mixed_counts(
    values: &[f32],
    labels: &[i32],
    n: usize,
    dim: usize,
    k: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut radii = Vec::with_capacity(n);
    let mut same = Vec::with_capacity(n);
    let mut full = Vec::with_capacity(n);
    for row in 0..n {
        let mut distances = (0..n)
            .filter(|&col| col != row && labels[col] == labels[row])
            .map(|col| cheb(values, row, col, dim))
            .collect::<Vec<_>>();
        distances.sort_by(f32::total_cmp);
        let radius = distances[k - 1];
        radii.push(radius);
        same.push(
            (0..n)
                .filter(|&col| {
                    col != row
                        && labels[col] == labels[row]
                        && cheb(values, row, col, dim) <= radius
                })
                .count(),
        );
        full.push(
            (0..n)
                .filter(|&col| col != row && cheb(values, row, col, dim) <= radius)
                .count(),
        );
    }
    (radii, same, full)
}

pub(super) fn cpu_ccm_predictions(
    embedding: &[f32],
    target: &[f32],
    _n: usize,
    dim: usize,
    neighbor_count: usize,
    library_size: usize,
) -> Vec<f32> {
    let mut predictions = Vec::with_capacity(library_size);
    for row in 0..library_size {
        let mut distances = (0..library_size)
            .filter(|&col| col != row)
            .map(|col| (euclid(embedding, row, col, dim), col))
            .collect::<Vec<_>>();
        distances.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        predictions.push(simplex_prediction(&distances[..neighbor_count], target));
    }
    predictions
}

pub(super) fn simplex_prediction(nearest: &[(f64, usize)], target: &[f32]) -> f32 {
    const EPS: f64 = 1.0e-12;
    let d1 = nearest[0].0;
    if d1 <= EPS {
        let zeros = nearest
            .iter()
            .filter(|(dist, _)| *dist <= EPS)
            .map(|(_, idx)| target[*idx] as f64)
            .collect::<Vec<_>>();
        return (zeros.iter().sum::<f64>() / zeros.len() as f64) as f32;
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for &(distance, idx) in nearest {
        let weight = (-distance / d1).exp();
        weighted_sum += weight * target[idx] as f64;
        weight_sum += weight;
    }
    (weighted_sum / weight_sum) as f32
}

pub(super) fn cheb(values: &[f32], left: usize, right: usize, dim: usize) -> f32 {
    (0..dim)
        .map(|offset| (values[left * dim + offset] - values[right * dim + offset]).abs())
        .fold(0.0, f32::max)
}

pub(super) fn euclid(values: &[f32], left: usize, right: usize, dim: usize) -> f64 {
    (0..dim)
        .map(|offset| {
            let diff = values[left * dim + offset] as f64 - values[right * dim + offset] as f64;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}
