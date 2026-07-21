use crate::Result;

#[derive(Debug)]
pub(super) struct CpuDescent {
    pub vector: Vec<f32>,
    pub steps_taken: usize,
    pub converged: bool,
    pub final_energy: f32,
}

pub(super) fn cpu_descent(
    initial: &[f32],
    members: &[&[f32]],
    beta: f32,
    max_steps: usize,
    eps: f32,
) -> Result<CpuDescent> {
    let dim = initial.len();
    let flattened = members
        .iter()
        .flat_map(|member| member.iter().copied())
        .collect::<Vec<_>>();
    let mut vector = initial.to_vec();
    let mut scaled = cpu_scaled(&vector, &flattened, dim, members.len(), beta)?;
    let mut previous = cpu_energy(&scaled, members.len(), beta);
    for step in 1..=max_steps {
        let weights = cpu_weights(&scaled, members.len(), beta);
        let mut next = vec![0.0_f32; dim];
        for (weight, member) in weights.iter().zip(flattened.chunks_exact(dim)) {
            for (dst, src) in next.iter_mut().zip(member) {
                *dst += weight * src;
            }
        }
        crate::cpu::normalize_f32(&mut next, dim)?;
        vector = next;
        scaled = cpu_scaled(&vector, &flattened, dim, members.len(), beta)?;
        let energy = cpu_energy(&scaled, members.len(), beta);
        if members.len() == 1 || (energy - previous).abs() < eps {
            return Ok(CpuDescent {
                vector,
                steps_taken: step,
                converged: true,
                final_energy: energy,
            });
        }
        previous = energy;
    }
    Ok(CpuDescent {
        vector,
        steps_taken: max_steps,
        converged: false,
        final_energy: previous,
    })
}

fn cpu_scaled(
    query: &[f32],
    members: &[f32],
    dim: usize,
    count: usize,
    beta: f32,
) -> Result<Vec<f32>> {
    if beta == 0.0 {
        return Ok(vec![0.0; count]);
    }
    let mut output = vec![0.0; count];
    crate::cpu::cosine_batch(query, members, dim, &mut output)?;
    output.iter_mut().for_each(|score| *score *= beta);
    Ok(output)
}

fn cpu_energy(scaled: &[f32], count: usize, beta: f32) -> f32 {
    if beta == 0.0 {
        return -(count as f32).ln();
    }
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = scaled.iter().map(|score| (*score - max).exp()).sum::<f32>();
    -(max + sum.ln())
}

fn cpu_weights(scaled: &[f32], count: usize, beta: f32) -> Vec<f32> {
    if beta == 0.0 {
        return vec![1.0 / count as f32; count];
    }
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = scaled
        .iter()
        .map(|score| (*score - max).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    weights.iter_mut().for_each(|weight| *weight /= sum);
    weights
}

pub(super) fn deterministic_members(rows: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..dim)
                .map(|col| ((row * 17 + col * 13 + 1) % 97) as f32 / 97.0 + 0.01)
                .collect()
        })
        .collect()
}

pub(super) fn deterministic_initial(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|col| ((col * 11 + 3) % 43) as f32 / 43.0 + 0.01)
        .collect()
}
