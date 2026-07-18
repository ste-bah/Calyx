pub(super) fn linalg_columns(n: usize) -> Vec<f32> {
    let mut columns = Vec::with_capacity(n * 4);
    let c0 = (0..n)
        .map(|row| {
            let t = row as f32 * 0.071;
            t.sin() + 0.25 * (t * 1.7).cos()
        })
        .collect::<Vec<_>>();
    let c1 = (0..n)
        .map(|row| c0[row] * 0.55 + ((row as f32) * 0.137).cos() * 0.33)
        .collect::<Vec<_>>();
    let c2 = (0..n)
        .map(|row| ((row as f32) * 0.049 + 0.7).cos() - 0.2 * ((row as f32) * 0.19).sin())
        .collect::<Vec<_>>();
    let c3 = (0..n)
        .map(|row| {
            let t = row as f32 * 0.031;
            (t + 0.4).sin() * 0.4 + (row % 5) as f32 * 0.03
        })
        .collect::<Vec<_>>();
    columns.extend_from_slice(&c0);
    columns.extend_from_slice(&c1);
    columns.extend_from_slice(&c2);
    columns.extend_from_slice(&c3);
    columns
}

pub(super) fn cpu_corr_matrix(columns: &[f32], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        out[i * d + i] = 1.0;
        for j in (i + 1)..d {
            let xi = &columns[i * n..(i + 1) * n];
            let yj = &columns[j * n..(j + 1) * n];
            let mean_x = xi.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let mean_y = yj.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let mut cov = 0.0;
            let mut vx = 0.0;
            let mut vy = 0.0;
            for (&x, &y) in xi.iter().zip(yj) {
                let dx = x as f64 - mean_x;
                let dy = y as f64 - mean_y;
                cov += dx * dy;
                vx += dx * dx;
                vy += dy * dy;
            }
            let r = (cov / (vx.sqrt() * vy.sqrt())).clamp(-1.0, 1.0);
            out[i * d + j] = r;
            out[j * d + i] = r;
        }
    }
    out
}

pub(super) fn cpu_invert_symmetric(matrix: &[f64], d: usize) -> Option<Vec<f64>> {
    let width = 2 * d;
    let mut a = vec![0.0; d * width];
    for i in 0..d {
        for j in 0..d {
            a[i * width + j] = matrix[i * d + j];
        }
        a[i * width + d + i] = 1.0;
    }
    for col in 0..d {
        let mut pivot = col;
        let mut best = a[col * width + col].abs();
        for row in (col + 1)..d {
            let value = a[row * width + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..width {
                a.swap(col * width + j, pivot * width + j);
            }
        }
        let inv_p = 1.0 / a[col * width + col];
        for j in 0..width {
            a[col * width + j] *= inv_p;
        }
        for row in 0..d {
            if row == col {
                continue;
            }
            let factor = a[row * width + col];
            for j in 0..width {
                a[row * width + j] -= factor * a[col * width + j];
            }
        }
    }
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = a[i * width + d + j];
        }
    }
    Some(out)
}

pub(super) fn granger_fixture(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (idx, value) in x.iter_mut().enumerate() {
        *value = (splitmix(idx as u64) - 0.5) as f32;
    }
    for idx in 2..n {
        let noise = (splitmix(9000 + idx as u64) - 0.5) * 0.08;
        y[idx] = 0.45 * y[idx - 1] + 0.9 * x[idx - 1] + 0.35 * x[idx - 2] + noise as f32;
    }
    (x, y)
}

pub(super) fn splitmix(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64) / ((1_u64 << 53) as f64)
}

pub(super) fn cpu_granger_rss(x: &[f32], y: &[f32], p: usize) -> Option<(f64, f64)> {
    let n = x.len();
    let t_rows = n - p;
    let kr = 1 + p;
    let ku = 1 + 2 * p;
    let mut ar = vec![0.0; kr * kr];
    let mut au = vec![0.0; ku * ku];
    let mut br = vec![0.0; kr];
    let mut bu = vec![0.0; ku];
    for target in p..n {
        let yi = y[target] as f64;
        for c in 0..kr {
            let vc = granger_restricted_value(y, target, c);
            br[c] += vc * yi;
            for d in c..kr {
                ar[c * kr + d] += vc * granger_restricted_value(y, target, d);
            }
        }
        for c in 0..ku {
            let vc = granger_unrestricted_value(x, y, target, c, p);
            bu[c] += vc * yi;
            for d in c..ku {
                au[c * ku + d] += vc * granger_unrestricted_value(x, y, target, d, p);
            }
        }
    }
    for c in 0..kr {
        for d in (c + 1)..kr {
            ar[d * kr + c] = ar[c * kr + d];
        }
    }
    for c in 0..ku {
        for d in (c + 1)..ku {
            au[d * ku + c] = au[c * ku + d];
        }
    }
    let beta_r = cpu_solve(&mut ar, &mut br, kr)?;
    let beta_u = cpu_solve(&mut au, &mut bu, ku)?;
    let mut rss_r = 0.0;
    let mut rss_u = 0.0;
    for target in p..n {
        let yi = y[target] as f64;
        let fit_r = (0..kr)
            .map(|c| granger_restricted_value(y, target, c) * beta_r[c])
            .sum::<f64>();
        let fit_u = (0..ku)
            .map(|c| granger_unrestricted_value(x, y, target, c, p) * beta_u[c])
            .sum::<f64>();
        rss_r += (yi - fit_r).powi(2);
        rss_u += (yi - fit_u).powi(2);
    }
    assert_eq!(t_rows, n - p);
    Some((rss_r, rss_u))
}

pub(super) fn granger_restricted_value(y: &[f32], target: usize, col: usize) -> f64 {
    if col == 0 {
        1.0
    } else {
        y[target - col] as f64
    }
}

pub(super) fn granger_unrestricted_value(
    x: &[f32],
    y: &[f32],
    target: usize,
    col: usize,
    p: usize,
) -> f64 {
    if col == 0 {
        1.0
    } else if col <= p {
        y[target - col] as f64
    } else {
        x[target - (col - p)] as f64
    }
}

pub(super) fn cpu_solve(a: &mut [f64], rhs: &mut [f64], k: usize) -> Option<Vec<f64>> {
    let scale = (0..k).map(|i| a[i * k + i].abs()).fold(0.0, f64::max);
    let eps = 1.0e-12 * scale.max(1.0);
    for col in 0..k {
        let mut pivot = col;
        let mut best = a[col * k + col].abs();
        for row in (col + 1)..k {
            let value = a[row * k + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < eps {
            return None;
        }
        if pivot != col {
            for j in 0..k {
                a.swap(col * k + j, pivot * k + j);
            }
            rhs.swap(col, pivot);
        }
        let inv_p = 1.0 / a[col * k + col];
        for j in 0..k {
            a[col * k + j] *= inv_p;
        }
        rhs[col] *= inv_p;
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = a[row * k + col];
            for j in 0..k {
                a[row * k + j] -= factor * a[col * k + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    Some(rhs.to_vec())
}
