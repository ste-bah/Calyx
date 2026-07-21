use std::collections::BTreeMap;

use super::*;
use crate::ForgeError;

#[derive(Clone, Copy, Debug, Default)]
struct Oracle {
    count: u64,
    sum: f64,
    min: f32,
    max: f32,
}

impl Oracle {
    fn push(&mut self, value: f32) {
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count += 1;
        self.sum += f64::from(value);
    }
}

#[test]
fn cuda_olap_matches_count_extrema_keys_and_bounded_sum() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let rows = 131_071;
    let values = (0..rows)
        .map(|row| ((row as f32) * 0.03125).sin() * 17.0)
        .collect::<Vec<_>>();
    let keys = (0..rows)
        .map(|row| match row % 5 {
            0 => -0.0_f32,
            1 => 0.0_f32,
            2 => -2.0_f32,
            3 => 1.0_f32,
            _ => 9.0_f32,
        })
        .collect::<Vec<_>>();
    let mut total = Oracle::default();
    let mut grouped = BTreeMap::<u32, Oracle>::new();
    for (&value, &key) in values.iter().zip(&keys) {
        total.push(value);
        grouped.entry(key.to_bits()).or_default().push(value);
    }

    let context = CudaOlapContext::new(0)?;
    let result = context.scan_columns_le(&le_bytes(&values), Some(&le_bytes(&keys)), 8)?;
    assert_aggregate(result.aggregate, total);
    assert_eq!(result.groups.len(), grouped.len());
    for (got, (key_bits, expected)) in result.groups.iter().zip(grouped) {
        assert_eq!(got.key_bits, key_bits);
        assert_aggregate(got.aggregate, expected);
    }
    assert!(result.stats.peak_pinned_staging_bytes <= (2 * CUDA_OLAP_PINNED_CHUNK_BYTES) as u64);
    assert_eq!(result.stats.host_to_device_bytes, (rows * 8) as u64);
    Ok(())
}

#[test]
fn cuda_olap_nonfinite_and_group_cap_fail_closed() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let context = CudaOlapContext::new(0)?;
    let mut values = vec![1.0_f32; 4096];
    values[4000] = f32::NAN;
    assert_op(
        context
            .scan_columns_le(&le_bytes(&values), None, 0)
            .expect_err("NaN must fail"),
        VALUE_NONFINITE_OP,
    );

    let values = vec![1.0_f32; 4096];
    let keys = (0..4096).map(|row| row as f32).collect::<Vec<_>>();
    assert_op(
        context
            .scan_columns_le(&le_bytes(&values), Some(&le_bytes(&keys)), 32)
            .expect_err("group cap must fail"),
        GROUP_CAP_OP,
    );
    Ok(())
}

#[test]
fn cuda_tiled_transpose_is_bit_exact_for_multiple_tiles() -> Result<()> {
    let _guard = crate::cuda::test_lock();
    let rows = 70_001;
    let columns = 67;
    let matrix = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| f32::from_bits(((row * columns + column) as u32) | 1))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let refs = matrix.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let context = CudaOlapContext::new(0)?;
    let (transposed, stats) = context.transpose_rows(&refs)?;
    for column in 0..columns {
        for row in 0..rows {
            assert_eq!(
                transposed[column * rows + row].to_bits(),
                matrix[row][column].to_bits()
            );
        }
    }
    assert!(stats.kernel_launches >= 2);
    assert!(stats.peak_pinned_staging_bytes <= CUDA_OLAP_PINNED_CHUNK_BYTES as u64);
    Ok(())
}

fn assert_aggregate(got: CudaOlapAggregate, expected: Oracle) {
    assert_eq!(got.count, expected.count);
    assert_eq!(got.min.to_bits(), expected.min.to_bits());
    assert_eq!(got.max.to_bits(), expected.max.to_bits());
    let tolerance = olap_sum_tolerance(expected.count, expected.min, expected.max);
    assert!(
        (got.sum - expected.sum).abs() <= tolerance,
        "sum delta={} tolerance={tolerance}",
        (got.sum - expected.sum).abs()
    );
}

fn assert_op(error: ForgeError, expected: &str) {
    let ForgeError::NumericalInvariant { op, .. } = error else {
        panic!("expected numerical invariant, got {error}");
    };
    assert_eq!(op, expected);
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
