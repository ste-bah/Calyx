use super::{
    OLAP_CUDA_MIN_ROWS, OlapAggregate, OlapExecutionStats, OlapGroupAggregate, OlapScanPlan, cpu,
};
#[cfg(feature = "cuda")]
use super::{olap_error, olap_sum_tolerance};
use crate::sst::arrow::ArrowColumnView;
use calyx_core::{CalyxError, Result};

pub(super) fn scan(
    chunk: &ArrowColumnView<'_>,
    plan: OlapScanPlan,
) -> Result<(OlapAggregate, Vec<OlapGroupAggregate>, OlapExecutionStats)> {
    if chunk.n_rows() < OLAP_CUDA_MIN_ROWS {
        return cpu::scan(chunk, plan);
    }
    scan_cuda(chunk, plan)
}

#[cfg(feature = "cuda")]
fn scan_cuda(
    chunk: &ArrowColumnView<'_>,
    plan: OlapScanPlan,
) -> Result<(OlapAggregate, Vec<OlapGroupAggregate>, OlapExecutionStats)> {
    debug_assert_eq!(OLAP_CUDA_MIN_ROWS, calyx_forge::OLAP_CUDA_MIN_ROWS);
    let value_column = chunk.column_bytes(plan.value_column)?;
    let group_column = plan
        .group_by_column
        .map(|column| chunk.column_bytes(column))
        .transpose()?;
    let result = crate::cuda_olap::with_context(|context| {
        context.scan_columns_le(value_column, group_column, plan.max_groups)
    })
    .map_err(map_forge_error)?;
    let aggregate = convert_aggregate(result.aggregate)?;
    let groups = result
        .groups
        .into_iter()
        .map(|group| {
            Ok(OlapGroupAggregate {
                group_key_bits: group.key_bits,
                group_key: f32::from_bits(group.key_bits),
                aggregate: convert_aggregate(group.aggregate)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let sum_abs_tolerance = olap_sum_tolerance(aggregate.count, aggregate.min, aggregate.max);
    let avg_abs_tolerance = sum_abs_tolerance / aggregate.count as f64;
    Ok((
        aggregate,
        groups,
        OlapExecutionStats {
            backend: "cuda-hash-dictionary".to_string(),
            pinned_staging: true,
            chunks: result.stats.chunks,
            dictionary_capacity: result.stats.dictionary_capacity,
            kernel_launches: result.stats.kernel_launches,
            host_to_device_bytes: result.stats.host_to_device_bytes,
            device_to_host_bytes: result.stats.device_to_host_bytes,
            peak_pinned_staging_bytes: result.stats.peak_pinned_staging_bytes,
            peak_device_bytes: result.stats.peak_device_bytes,
            sum_abs_tolerance,
            avg_abs_tolerance,
        },
    ))
}

#[cfg(feature = "cuda")]
fn convert_aggregate(input: calyx_forge::CudaOlapAggregate) -> Result<OlapAggregate> {
    let count = usize::try_from(input.count)
        .map_err(|_| olap_error("CALYX_OLAP_SCAN_LIMIT", "CUDA count exceeds usize"))?;
    if count == 0 {
        return Err(olap_error("CALYX_OLAP_EMPTY", "aggregate has no rows"));
    }
    Ok(OlapAggregate {
        count,
        sum: input.sum,
        min: input.min,
        max: input.max,
        avg: input.sum / count as f64,
    })
}

#[cfg(feature = "cuda")]
fn map_forge_error(error: calyx_forge::ForgeError) -> CalyxError {
    if let calyx_forge::ForgeError::NumericalInvariant { op, .. } = &error {
        let code = match op.as_str() {
            "olap.value_finite" | "olap.group_finite" => "CALYX_OLAP_NONFINITE_VALUE",
            "olap.group_cap" => "CALYX_OLAP_SCAN_LIMIT",
            _ => error.code(),
        };
        return CalyxError {
            code,
            message: error.to_string(),
            remediation: "fix the OLAP column values or bounded group plan; CUDA faults never fall back to CPU",
        };
    }
    CalyxError {
        code: error.code(),
        message: error.to_string(),
        remediation: "restore CUDA availability or keep the scan below the measured CPU/GPU crossover",
    }
}

#[cfg(not(feature = "cuda"))]
fn scan_cuda(
    _chunk: &ArrowColumnView<'_>,
    _plan: OlapScanPlan,
) -> Result<(OlapAggregate, Vec<OlapGroupAggregate>, OlapExecutionStats)> {
    Err(CalyxError {
        code: "CALYX_FORGE_DEVICE_UNAVAILABLE",
        message: format!(
            "OLAP scans at or above {OLAP_CUDA_MIN_ROWS} rows require the CUDA Aster feature"
        ),
        remediation: "build Aster with feature cuda or keep the scan below the measured CPU/GPU crossover",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sst::arrow::{decode_column_shape, encode_column_chunk};

    #[test]
    fn small_scan_uses_cpu() {
        let rows = [&[1.0_f32][..], &[2.0_f32][..]];
        let bytes = encode_column_chunk(&rows).expect("encode");
        let chunk = decode_column_shape(&bytes).expect("decode");
        let (aggregate, _, execution) = scan(&chunk, OlapScanPlan::new(0)).expect("scan");
        assert_eq!(aggregate.sum, 3.0);
        assert_eq!(execution.backend, "cpu");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn default_million_row_path_never_silently_scans_on_cpu() {
        let row = [1.0_f32];
        let rows = vec![row.as_slice(); OLAP_CUDA_MIN_ROWS];
        let bytes = encode_column_chunk(&rows).expect("encode");
        let chunk = decode_column_shape(&bytes).expect("decode");
        let error = scan(&chunk, OlapScanPlan::new(0)).expect_err("CUDA required");
        assert_eq!(error.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    }
}
