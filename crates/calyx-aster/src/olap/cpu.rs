use std::collections::BTreeMap;

use super::{OlapAggregate, OlapExecutionStats, OlapGroupAggregate, OlapScanPlan, olap_error};
use crate::sst::arrow::ArrowColumnView;
use calyx_core::Result;

pub(super) fn scan(
    chunk: &ArrowColumnView<'_>,
    plan: OlapScanPlan,
) -> Result<(OlapAggregate, Vec<OlapGroupAggregate>, OlapExecutionStats)> {
    let aggregate = scan_total(chunk, plan.value_column)?;
    let groups = scan_groups(chunk, plan)?;
    Ok((
        aggregate,
        groups,
        OlapExecutionStats {
            backend: "cpu".to_string(),
            pinned_staging: false,
            chunks: 0,
            dictionary_capacity: 0,
            kernel_launches: 0,
            host_to_device_bytes: 0,
            device_to_host_bytes: 0,
            peak_pinned_staging_bytes: 0,
            peak_device_bytes: 0,
            sum_abs_tolerance: 0.0,
            avg_abs_tolerance: 0.0,
        },
    ))
}

fn scan_total(chunk: &ArrowColumnView<'_>, column: usize) -> Result<OlapAggregate> {
    let mut acc = Accumulator::default();
    for value in chunk.column_values(column)? {
        acc.push(value)?;
    }
    acc.finish()
}

fn scan_groups(chunk: &ArrowColumnView<'_>, plan: OlapScanPlan) -> Result<Vec<OlapGroupAggregate>> {
    let Some(group_column) = plan.group_by_column else {
        return Ok(Vec::new());
    };
    let mut groups = BTreeMap::<u32, Accumulator>::new();
    for row in 0..chunk.n_rows() {
        let group_key = finite(chunk.value(group_column, row)?)?;
        let value = chunk.value(plan.value_column, row)?;
        if !groups.contains_key(&group_key.to_bits()) && groups.len() == plan.max_groups {
            return Err(olap_error(
                "CALYX_OLAP_SCAN_LIMIT",
                format!("group cap {} exceeded", plan.max_groups),
            ));
        }
        groups.entry(group_key.to_bits()).or_default().push(value)?;
    }
    groups
        .into_iter()
        .map(|(group_key_bits, acc)| {
            Ok(OlapGroupAggregate {
                group_key_bits,
                group_key: f32::from_bits(group_key_bits),
                aggregate: acc.finish()?,
            })
        })
        .collect()
}

#[derive(Debug, Default, Clone)]
struct Accumulator {
    count: usize,
    sum: f64,
    min: f32,
    max: f32,
}

impl Accumulator {
    fn push(&mut self, value: f32) -> Result<()> {
        let value = finite(value)?;
        if self.count == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.count = self
            .count
            .checked_add(1)
            .ok_or_else(|| olap_error("CALYX_OLAP_SCAN_LIMIT", "aggregate count overflow"))?;
        self.sum += f64::from(value);
        Ok(())
    }

    fn finish(self) -> Result<OlapAggregate> {
        if self.count == 0 {
            return Err(olap_error("CALYX_OLAP_EMPTY", "aggregate has no rows"));
        }
        Ok(OlapAggregate {
            count: self.count,
            sum: self.sum,
            min: self.min,
            max: self.max,
            avg: self.sum / self.count as f64,
        })
    }
}

fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(olap_error(
            "CALYX_OLAP_NONFINITE_VALUE",
            "column aggregate encountered NaN or Inf",
        ))
    }
}
