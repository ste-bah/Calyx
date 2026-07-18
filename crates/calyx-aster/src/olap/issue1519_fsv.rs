use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;
use sha2::{Digest, Sha256};

use super::{OlapAggregate, OlapScanPlan, cpu, dispatch, olap_sum_tolerance};
use crate::sst::arrow::{
    decode_column_shape, encode_column_chunk, encode_column_chunk_accelerated,
};

const DEFAULT_ROWS: usize = 1_000_000;
const COLUMNS: usize = 8;
const GROUPS: usize = 4096;
const BENCH_SAMPLES: usize = 5;

#[test]
#[ignore = "manual GPU-host FSV for issue #1519 CUDA OLAP and transpose"]
fn issue1519_cuda_olap_and_transpose_fsv() {
    let root = PathBuf::from(
        std::env::var("CALYX_FSV_ROOT").expect("CALYX_FSV_ROOT must name the issue FSV root"),
    );
    let rows = std::env::var("CALYX_ISSUE1519_ROWS")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid row count"))
        .unwrap_or(DEFAULT_ROWS);
    assert!(rows >= super::OLAP_CUDA_MIN_ROWS);
    fs::create_dir_all(&root).expect("create FSV root");

    let matrix = fixture(rows);
    let row_refs = matrix.chunks_exact(COLUMNS).collect::<Vec<_>>();
    warm_cuda(&row_refs);
    encode_column_chunk(&row_refs).expect("warm CPU transpose");

    let mut cpu_bytes = None;
    let mut gpu_bytes = None;
    let mut transpose_stats = None;
    let mut cpu_transpose_samples = Vec::with_capacity(BENCH_SAMPLES);
    let mut gpu_transpose_samples = Vec::with_capacity(BENCH_SAMPLES);
    for _ in 0..BENCH_SAMPLES {
        let (candidate, elapsed) = timed(|| encode_column_chunk(&row_refs).expect("CPU transpose"));
        cpu_transpose_samples.push(elapsed);
        if let Some(expected) = cpu_bytes.as_ref() {
            assert_eq!(&candidate, expected, "CPU transpose must be deterministic");
        } else {
            cpu_bytes = Some(candidate);
        }

        let ((candidate, stats), elapsed) =
            timed(|| encode_column_chunk_accelerated(&row_refs).expect("CUDA transpose"));
        gpu_transpose_samples.push(elapsed);
        assert_eq!(
            &candidate,
            cpu_bytes.as_ref().expect("CPU bytes initialized"),
            "persisted column bytes must be exact"
        );
        if let Some(expected) = gpu_bytes.as_ref() {
            assert_eq!(&candidate, expected, "CUDA transpose must be deterministic");
            assert_eq!(Some(stats), transpose_stats);
        } else {
            gpu_bytes = Some(candidate);
            transpose_stats = Some(stats);
        }
    }
    let cpu_bytes = cpu_bytes.expect("CPU transpose sampled");
    let gpu_bytes = gpu_bytes.expect("CUDA transpose sampled");
    let transpose_stats = transpose_stats.expect("CUDA transpose stats sampled");
    let cpu_transpose_us = median(&cpu_transpose_samples);
    let gpu_transpose_us = median(&gpu_transpose_samples);
    assert_eq!(gpu_bytes, cpu_bytes, "persisted column bytes must be exact");
    assert_eq!(transpose_stats.backend, "cuda");

    let chunk_path = root.join("slot-column.cxa1");
    fs::write(&chunk_path, &gpu_bytes).expect("persist GPU column bytes");
    let persisted = fs::read(&chunk_path).expect("read persisted GPU column bytes");
    assert_eq!(persisted, cpu_bytes);
    let chunk = decode_column_shape(&persisted).expect("decode persisted column");

    let ungrouped_plan = OlapScanPlan::new(0).with_limits(rows, GROUPS);
    let (gpu_ungrouped, cpu_ungrouped_samples, gpu_ungrouped_samples) =
        benchmark_scan(&chunk, ungrouped_plan, "ungrouped");
    let cpu_ungrouped_us = median(&cpu_ungrouped_samples);
    let gpu_ungrouped_us = median(&gpu_ungrouped_samples);

    let grouped_plan = OlapScanPlan::new(0)
        .with_group_by(1)
        .with_limits(rows, GROUPS);
    let (gpu_grouped, cpu_grouped_samples, gpu_grouped_samples) =
        benchmark_scan(&chunk, grouped_plan, "grouped");
    let cpu_grouped_us = median(&cpu_grouped_samples);
    let gpu_grouped_us = median(&gpu_grouped_samples);

    let edge_cases = edge_cases(&persisted, rows);
    let transpose_speedup = ratio(cpu_transpose_us, gpu_transpose_us);
    let ungrouped_speedup = ratio(cpu_ungrouped_us, gpu_ungrouped_us);
    let grouped_speedup = ratio(cpu_grouped_us, gpu_grouped_us);
    println!(
        "ISSUE1519_TIMINGS transpose_cpu_us={cpu_transpose_us} transpose_gpu_us={gpu_transpose_us} ungrouped_cpu_us={cpu_ungrouped_us} ungrouped_gpu_us={gpu_ungrouped_us} grouped_cpu_us={cpu_grouped_us} grouped_gpu_us={gpu_grouped_us}"
    );
    assert!(
        transpose_speedup > 1.0,
        "transpose speedup={transpose_speedup}"
    );
    assert!(
        ungrouped_speedup > 1.0,
        "ungrouped speedup={ungrouped_speedup}"
    );
    assert!(grouped_speedup > 1.0, "grouped speedup={grouped_speedup}");

    let report = json!({
        "format": "calyx-issue1519-cuda-olap-fsv-v1",
        "rows": rows,
        "columns": COLUMNS,
        "groups": GROUPS,
        "chunk": {
            "path": chunk_path,
            "bytes": persisted.len(),
            "sha256": sha256(&persisted),
            "matches_cpu_oracle": persisted == cpu_bytes,
        },
        "transpose": {
            "cpu_us": cpu_transpose_us,
            "gpu_us": gpu_transpose_us,
            "cpu_samples_us": cpu_transpose_samples,
            "gpu_samples_us": gpu_transpose_samples,
            "speedup": transpose_speedup,
            "kernel_launches": transpose_stats.kernel_launches,
            "h2d_bytes": transpose_stats.host_to_device_bytes,
            "d2h_bytes": transpose_stats.device_to_host_bytes,
            "peak_pinned_staging_bytes": transpose_stats.peak_pinned_staging_bytes,
        },
        "benchmark": { "samples": BENCH_SAMPLES, "statistic": "median" },
        "ungrouped": measurement(
            cpu_ungrouped_us,
            gpu_ungrouped_us,
            &cpu_ungrouped_samples,
            &gpu_ungrouped_samples,
            &gpu_ungrouped,
        ),
        "grouped": measurement(
            cpu_grouped_us,
            gpu_grouped_us,
            &cpu_grouped_samples,
            &gpu_grouped_samples,
            &gpu_grouped,
        ),
        "edge_cases": edge_cases,
    });
    let report_path = root.join("issue1519-cuda-olap-fsv.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap()).expect("write report");
    println!("ISSUE1519_FSV={}", report_path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn fixture(rows: usize) -> Vec<f32> {
    let mut matrix = Vec::with_capacity(rows * COLUMNS);
    for row in 0..rows {
        let value = ((row % 20_003) as f32 - 10_001.0) * 0.03125;
        matrix.extend_from_slice(&[
            value,
            (row % GROUPS) as f32,
            row as f32,
            -value,
            value * 0.5,
            (row % 17) as f32,
            0.0,
            -0.0,
        ]);
    }
    matrix
}

fn warm_cuda(rows: &[&[f32]]) {
    let take = rows.len().min(4096);
    crate::cuda_olap::with_context(|context| context.transpose_rows(&rows[..take]))
        .expect("warm OLAP CUDA module");
}

type ScanOutput = (
    OlapAggregate,
    Vec<super::OlapGroupAggregate>,
    super::OlapExecutionStats,
);

fn benchmark_scan(
    chunk: &crate::sst::arrow::ArrowColumnView<'_>,
    plan: OlapScanPlan,
    label: &str,
) -> (ScanOutput, Vec<u128>, Vec<u128>) {
    cpu::scan(chunk, plan).unwrap_or_else(|error| panic!("warm CPU {label}: {error}"));
    dispatch::scan(chunk, plan).unwrap_or_else(|error| panic!("warm CUDA {label}: {error}"));

    let mut canonical_gpu = None;
    let mut cpu_samples = Vec::with_capacity(BENCH_SAMPLES);
    let mut gpu_samples = Vec::with_capacity(BENCH_SAMPLES);
    for _ in 0..BENCH_SAMPLES {
        let (cpu_result, cpu_us) =
            timed(|| cpu::scan(chunk, plan).unwrap_or_else(|error| panic!("CPU {label}: {error}")));
        let (gpu_result, gpu_us) = timed(|| {
            dispatch::scan(chunk, plan).unwrap_or_else(|error| panic!("CUDA {label}: {error}"))
        });
        assert_scan(&gpu_result, &cpu_result);
        cpu_samples.push(cpu_us);
        gpu_samples.push(gpu_us);
        if let Some(expected) = canonical_gpu.as_ref() {
            assert_eq!(&gpu_result, expected, "CUDA {label} must be deterministic");
        } else {
            canonical_gpu = Some(gpu_result);
        }
    }
    (
        canonical_gpu.expect("CUDA scan sampled"),
        cpu_samples,
        gpu_samples,
    )
}

fn assert_scan(gpu: &ScanOutput, cpu: &ScanOutput) {
    assert_aggregate(&gpu.0, &cpu.0);
    assert_eq!(gpu.1.len(), cpu.1.len());
    for (gpu, cpu) in gpu.1.iter().zip(&cpu.1) {
        assert_eq!(gpu.group_key_bits, cpu.group_key_bits);
        assert_eq!(gpu.group_key.to_bits(), cpu.group_key.to_bits());
        assert_aggregate(&gpu.aggregate, &cpu.aggregate);
    }
}

fn assert_aggregate(gpu: &OlapAggregate, cpu: &OlapAggregate) {
    assert_eq!(gpu.count, cpu.count);
    assert_eq!(gpu.min.to_bits(), cpu.min.to_bits());
    assert_eq!(gpu.max.to_bits(), cpu.max.to_bits());
    let tolerance = olap_sum_tolerance(cpu.count, cpu.min, cpu.max);
    assert!((gpu.sum - cpu.sum).abs() <= tolerance);
    assert!((gpu.avg - cpu.avg).abs() <= tolerance / cpu.count as f64);
}

fn edge_cases(persisted: &[u8], rows: usize) -> serde_json::Value {
    let empty = encode_column_chunk(&[]).expect_err("empty rejected");
    let mut nonfinite = persisted.to_vec();
    nonfinite[16 + (rows - 1) * 4..16 + rows * 4].copy_from_slice(&f32::NAN.to_le_bytes());
    let chunk = decode_column_shape(&nonfinite).expect("decode NaN chunk");
    let nonfinite = dispatch::scan(&chunk, OlapScanPlan::new(0).with_limits(rows, GROUPS))
        .expect_err("NaN rejected");
    let chunk = decode_column_shape(persisted).expect("decode grouped chunk");
    let high_cardinality = dispatch::scan(
        &chunk,
        OlapScanPlan::new(0).with_group_by(2).with_limits(rows, 128),
    )
    .expect_err("high cardinality rejected");
    let corrupt = decode_column_shape(&persisted[..persisted.len() - 1])
        .expect_err("truncated column rejected");
    let mut overflow = Vec::from(b"CXA1".as_slice());
    overflow.extend_from_slice(&1_u32.to_le_bytes());
    overflow.extend_from_slice(&u32::MAX.to_le_bytes());
    overflow.extend_from_slice(&u32::MAX.to_le_bytes());
    let overflow = decode_column_shape(&overflow).expect_err("overflow shape rejected");
    json!({
        "empty": empty.code,
        "nonfinite": nonfinite.code,
        "high_cardinality": high_cardinality.code,
        "corrupt": corrupt.code,
        "overflow": overflow.code,
    })
}

fn measurement(
    cpu_us: u128,
    gpu_us: u128,
    cpu_samples_us: &[u128],
    gpu_samples_us: &[u128],
    result: &(
        OlapAggregate,
        Vec<super::OlapGroupAggregate>,
        super::OlapExecutionStats,
    ),
) -> serde_json::Value {
    json!({
        "cpu_us": cpu_us,
        "gpu_us": gpu_us,
        "cpu_samples_us": cpu_samples_us,
        "gpu_samples_us": gpu_samples_us,
        "speedup": ratio(cpu_us, gpu_us),
        "aggregate": result.0,
        "group_count": result.1.len(),
        "execution": result.2,
    })
}

fn timed<T>(operation: impl FnOnce() -> T) -> (T, u128) {
    let started = Instant::now();
    let result = operation();
    (result, started.elapsed().as_micros())
}

fn median(samples: &[u128]) -> u128 {
    assert_eq!(samples.len(), BENCH_SAMPLES);
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn ratio(cpu_us: u128, gpu_us: u128) -> f64 {
    cpu_us as f64 / gpu_us.max(1) as f64
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
