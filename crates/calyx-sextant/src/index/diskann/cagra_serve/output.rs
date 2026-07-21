use calyx_core::Result;

use super::CagraServingMetric;
use crate::error::{
    CALYX_INDEX_INVALID_PARAMS, CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE, sextant_error,
};

pub(super) fn validated_bitset(ids: &[u32], rows: usize) -> Result<(Vec<u32>, usize)> {
    let mut words = vec![0_u32; rows.div_ceil(32)];
    let mut count = 0usize;
    for &id in ids {
        let id = id as usize;
        if id >= rows {
            return Err(sextant_error(
                CALYX_INDEX_INVALID_PARAMS,
                format!("CAGRA filter id {id} exceeds row count {rows}"),
            ));
        }
        let mask = 1_u32 << (id % 32);
        let word = &mut words[id / 32];
        if *word & mask == 0 {
            *word |= mask;
            count += 1;
        }
    }
    Ok((words, count))
}

pub(super) fn validate_output(
    ids: Vec<i64>,
    distances: Vec<f32>,
    query_count: usize,
    k: usize,
    rows: usize,
    metric: CagraServingMetric,
) -> Result<Vec<Vec<(u32, f32)>>> {
    let mut output = Vec::with_capacity(query_count);
    for query_idx in 0..query_count {
        let start = query_idx * k;
        let mut row = Vec::with_capacity(k);
        for (&id, &distance) in ids[start..start + k]
            .iter()
            .zip(&distances[start..start + k])
        {
            if id < 0 || id as usize >= rows || !distance.is_finite() || distance < -1e-4 {
                return Err(sextant_error(
                    CALYX_SEXTANT_GPU_SERVING_UNAVAILABLE,
                    format!("cuVS returned invalid pair id={id} distance={distance} rows={rows}"),
                ));
            }
            let distance = match metric {
                CagraServingMetric::UnitL2 => 0.5 * distance.max(0.0),
                CagraServingMetric::RawL2 => distance.max(0.0),
            };
            row.push((id as u32, distance));
        }
        row.sort_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        row.dedup_by_key(|(id, _)| *id);
        row.truncate(k);
        output.push(row);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_deduplicates_and_uses_include_bits() {
        let (words, count) = validated_bitset(&[0, 31, 32, 32, 64], 65).expect("bitset");
        assert_eq!(count, 4);
        assert_eq!(words, vec![0x8000_0001, 1, 1]);
    }

    #[test]
    fn unit_l2_output_maps_to_cosine_distance_and_sorts_ties() {
        let output = validate_output(
            vec![2, 0, 1],
            vec![0.5, 0.5, 0.25],
            1,
            3,
            3,
            CagraServingMetric::UnitL2,
        )
        .expect("output");
        assert_eq!(output[0], vec![(1, 0.125), (0, 0.25), (2, 0.25)]);
    }
}
