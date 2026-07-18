use super::CudaSkillSlot;
use crate::{ForgeError, Result};

pub(super) fn dense_slot(points: usize, dim: usize, salt: usize) -> CudaSkillSlot {
    CudaSkillSlot {
        dim,
        point_indices: (0..points as u32).collect(),
        values: (0..points)
            .flat_map(|point| {
                (0..dim).map(move |feature| {
                    ((point * 17 + feature * 13 + salt) % 101) as f32 / 101.0 + 0.01
                })
            })
            .collect(),
    }
}

pub(super) fn cpu_distances(points: usize, slots: &[CudaSkillSlot]) -> Result<Vec<f64>> {
    let mut offsets = vec![vec![None; points]; slots.len()];
    for (slot_index, slot) in slots.iter().enumerate() {
        for (local, point) in slot.point_indices.iter().copied().enumerate() {
            offsets[slot_index][point as usize] = Some(local * slot.dim);
        }
    }
    let mut output = vec![0.0_f64; points * points];
    for left in 0..points {
        for right in left + 1..points {
            let mut sum = 0.0_f64;
            let mut count = 0_usize;
            for (slot_index, slot) in slots.iter().enumerate() {
                let (Some(left_offset), Some(right_offset)) =
                    (offsets[slot_index][left], offsets[slot_index][right])
                else {
                    continue;
                };
                let left_row = &slot.values[left_offset..left_offset + slot.dim];
                let right_row = &slot.values[right_offset..right_offset + slot.dim];
                sum += f64::from(cosine(left_row, right_row)?);
                count += 1;
            }
            if count == 0 {
                return Err(numerical(
                    "skill.pair_no_overlap",
                    "CPU pair has no shared slot",
                ));
            }
            let distance = (1.0 - sum / count as f64).clamp(0.0, 2.0);
            output[left * points + right] = distance;
            output[right * points + left] = distance;
        }
    }
    Ok(output)
}

pub(super) fn cpu_mst(
    points: usize,
    distances: &[f64],
    min_samples: usize,
) -> Vec<(usize, usize, f64)> {
    let rank = min_samples.min(points - 1);
    let core = (0..points)
        .map(|row| {
            let mut values = (0..points)
                .filter(|column| *column != row)
                .map(|column| distances[row * points + column])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values[rank - 1]
        })
        .collect::<Vec<_>>();
    let mut in_tree = vec![false; points];
    let mut keys = vec![f64::INFINITY; points];
    let mut parents = vec![usize::MAX; points];
    keys[0] = 0.0;
    let mut edges = Vec::with_capacity(points - 1);
    for _ in 0..points {
        let point = (0..points)
            .filter(|point| !in_tree[*point])
            .min_by(|left, right| {
                keys[*left]
                    .total_cmp(&keys[*right])
                    .then_with(|| left.cmp(right))
            })
            .expect("connected complete graph");
        in_tree[point] = true;
        if parents[point] != usize::MAX {
            edges.push((
                parents[point].min(point),
                parents[point].max(point),
                keys[point],
            ));
        }
        for destination in 0..points {
            if !in_tree[destination] {
                let weight = distances[point * points + destination]
                    .max(core[point])
                    .max(core[destination]);
                if weight < keys[destination] {
                    keys[destination] = weight;
                    parents[destination] = point;
                }
            }
        }
    }
    edges.sort_by(|left, right| {
        left.2
            .total_cmp(&right.2)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });
    edges
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        if !left.is_finite() || !right.is_finite() {
            return Err(numerical("skill.vector_finite", "CPU row is non-finite"));
        }
        dot += f64::from(*left) * f64::from(*right);
        left_norm += f64::from(*left) * f64::from(*left);
        right_norm += f64::from(*right) * f64::from(*right);
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator <= f64::EPSILON {
        return Err(numerical("skill.vector_norm", "CPU row has zero norm"));
    }
    Ok((dot / denominator) as f32)
}

fn numerical(op: &str, detail: &str) -> ForgeError {
    ForgeError::NumericalInvariant {
        op: op.to_string(),
        detail: detail.to_string(),
        remediation: "repair skill test fixture".to_string(),
    }
}
