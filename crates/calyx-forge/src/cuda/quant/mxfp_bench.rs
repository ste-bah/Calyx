use super::{CudaMxFpBatch, CudaQuantContext, MXFP4_CUDA_MIN_ELEMENTS, MXFP8_CUDA_MIN_ELEMENTS};
use crate::cuda::{init_cuda, test_lock};
use crate::quant::{MxFp4Codec, QuantLevel, QuantizedVec, Quantizer};
use crate::{ForgeError, Result};
use serde_json::{Value, json};
use std::fs;
use std::time::{Duration, Instant};

const ZERO_SEED: [u8; 32] = [0; 32];

#[test]
#[ignore = "release-only production GPU benchmark/FSV; set CALYX_MXFP_BENCH_* variables"]
fn mxfp_cuda_benchmark_fsv() -> Result<()> {
    let _guard = test_lock();
    let dims = bench_sizes("CALYX_MXFP_BENCH_DIMS", &[768, 1_536, 2_048])?;
    let rows = bench_sizes("CALYX_MXFP_BENCH_ROWS", &[1_024, 100_000, 1_000_000])?;
    let root = required_env("CALYX_FSV_ROOT")?;
    let git_sha = required_env("CALYX_BENCH_GIT_SHA")?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let mut measurements = Vec::new();
    for dim in dims {
        let codec = MxFp4Codec::new(dim);
        for &candidate_rows in &rows {
            measurements.push(benchmark_shape(&quant, &codec, dim, candidate_rows)?);
        }
    }
    let payload = serde_json::to_vec_pretty(&json!({
        "issue": 1768,
        "git_sha": git_sha,
        "mxfp4_dispatch_threshold_elements": MXFP4_CUDA_MIN_ELEMENTS,
        "mxfp8_dispatch_threshold_elements": MXFP8_CUDA_MIN_ELEMENTS,
        "measurements": measurements,
    }))
    .map_err(|error| bench_error(format!("benchmark JSON serialization failed: {error}")))?;
    fs::create_dir_all(&root)
        .map_err(|error| bench_error(format!("create FSV root {root}: {error}")))?;
    let path = std::path::Path::new(&root).join("mxfp-cuda-benchmark.json");
    fs::write(&path, &payload)
        .map_err(|error| bench_error(format!("write {}: {error}", path.display())))?;
    assert_eq!(
        fs::read(&path)
            .map_err(|error| bench_error(format!("read {}: {error}", path.display())))?,
        payload
    );
    println!("MXFP_CUDA_FSV path={}", path.display());
    Ok(())
}

fn benchmark_shape(
    quant: &CudaQuantContext,
    codec: &MxFp4Codec,
    dim: usize,
    rows: usize,
) -> Result<Value> {
    let build4 = Instant::now();
    let cpu4 = packed_rows(QuantLevel::Bits4Fp, dim, rows);
    let cpu4_build_seconds = build4.elapsed().as_secs_f64();
    let build8 = Instant::now();
    let cpu8 = packed_rows(QuantLevel::Bits8Fp, dim, rows);
    let cpu8_build_seconds = build8.elapsed().as_secs_f64();
    let upload4 = Instant::now();
    let gpu4 = quant.upload_mxfp(codec, &cpu4)?;
    let gpu4_upload_seconds = upload4.elapsed().as_secs_f64();
    let upload8 = Instant::now();
    let gpu8 = quant.upload_mxfp(codec, &cpu8)?;
    let gpu8_upload_seconds = upload8.elapsed().as_secs_f64();
    let query4 = cpu4[0].clone();
    let query8 = cpu8[0].clone();
    let repeats = repetitions(rows);
    let fp4_fp4 = benchmark_combo(quant, codec, "mxfp4_mxfp4", &gpu4, &query4, &cpu4, repeats)?;
    let fp8_fp8 = benchmark_combo(quant, codec, "mxfp8_mxfp8", &gpu8, &query8, &cpu8, repeats)?;
    let fp4_fp8 = benchmark_combo(quant, codec, "mxfp4_mxfp8", &gpu8, &query4, &cpu8, repeats)?;
    let fp8_fp4 = benchmark_combo(quant, codec, "mxfp8_mxfp4", &gpu4, &query8, &cpu4, repeats)?;
    Ok(json!({
        "dim": dim,
        "candidate_rows": rows,
        "elements": dim.saturating_mul(rows),
        "repetitions": repeats,
        "cpu_mxfp4_build_seconds": cpu4_build_seconds,
        "cpu_mxfp8_build_seconds": cpu8_build_seconds,
        "gpu_mxfp4_upload_seconds": gpu4_upload_seconds,
        "gpu_mxfp8_upload_seconds": gpu8_upload_seconds,
        "combinations": [fp4_fp4, fp8_fp8, fp4_fp8, fp8_fp4],
    }))
}

fn benchmark_combo(
    quant: &CudaQuantContext,
    codec: &MxFp4Codec,
    name: &str,
    gpu_corpus: &CudaMxFpBatch,
    cpu_query: &QuantizedVec,
    cpu_corpus: &[QuantizedVec],
    repeats: usize,
) -> Result<Value> {
    let cpu_started = Instant::now();
    let mut cpu_top = Vec::new();
    for _ in 0..repeats {
        let scores = cpu_corpus
            .iter()
            .map(|candidate| codec.dot_estimate(cpu_query, candidate))
            .collect::<Result<Vec<_>>>()?;
        cpu_top = host_topk(&scores, 10);
    }
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();

    quant.reset_stats();
    let gpu_started = Instant::now();
    let mut gpu_top = Vec::new();
    for _ in 0..repeats {
        let gpu_query = quant.upload_mxfp(codec, std::slice::from_ref(cpu_query))?;
        gpu_top = gpu_corpus.score(&gpu_query)?.topk(10)?;
    }
    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
    assert_eq!(
        gpu_top.iter().map(|pair| pair.0).collect::<Vec<_>>(),
        cpu_top.iter().map(|pair| pair.0).collect::<Vec<_>>(),
        "MXFP top-k {name} rows={}",
        cpu_corpus.len()
    );
    for ((_, gpu), (_, cpu)) in gpu_top.iter().zip(&cpu_top) {
        let tolerance = 2e-6 * cpu.abs().max(1.0);
        assert!((gpu - cpu).abs() <= tolerance);
    }
    let stats = quant.stats();
    let full_scores = cpu_corpus
        .len()
        .saturating_mul(repeats)
        .saturating_mul(size_of::<f32>()) as u64;
    if cpu_corpus.len() > 32 && stats.d2h_bytes >= full_scores {
        return Err(bench_error(format!(
            "{name} D2H was not compact: {} >= {full_scores}",
            stats.d2h_bytes
        )));
    }
    println!(
        "MXFP_CUDA_BENCH combo={name} dim={} rows={} repeats={repeats} cpu={cpu_seconds:.6}s gpu={gpu_seconds:.6}s speedup={:.3}x d2h={}",
        codec.dim(),
        cpu_corpus.len(),
        cpu_seconds / gpu_seconds,
        stats.d2h_bytes,
    );
    Ok(json!({
        "name": name,
        "cpu_search_seconds": cpu_seconds,
        "gpu_search_seconds": gpu_seconds,
        "search_speedup": cpu_seconds / gpu_seconds,
        "topk_exact": true,
        "gpu_stats": stats,
    }))
}

#[test]
#[ignore = "Nsight-only resident search trace; set CALYX_MXFP_TRACE_* variables"]
fn mxfp_cuda_resident_trace_fsv() -> Result<()> {
    let _guard = test_lock();
    let dim = bench_usize("CALYX_MXFP_TRACE_DIM", 768, 4_096)?;
    let rows = bench_usize("CALYX_MXFP_TRACE_ROWS", 100_000, i32::MAX as usize / dim)?;
    let pause = bench_usize("CALYX_MXFP_TRACE_PAUSE_SECONDS", 6, 60)?;
    let root = required_env("CALYX_FSV_ROOT")?;
    let git_sha = required_env("CALYX_BENCH_GIT_SHA")?;
    let codec = MxFp4Codec::new(dim);
    let cpu4 = packed_rows(QuantLevel::Bits4Fp, dim, rows);
    let cpu8 = packed_rows(QuantLevel::Bits8Fp, dim, rows);
    let query4 = cpu4[0].clone();
    let query8 = cpu8[0].clone();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let gpu4 = quant.upload_mxfp(&codec, &cpu4)?;
    let gpu8 = quant.upload_mxfp(&codec, &cpu8)?;
    drop(cpu4);
    drop(cpu8);
    let warm4 = quant.upload_mxfp(&codec, std::slice::from_ref(&query4))?;
    let warm8 = quant.upload_mxfp(&codec, std::slice::from_ref(&query8))?;
    let _ = gpu4.score(&warm4)?.topk(10)?;
    let _ = gpu8.score(&warm8)?.topk(10)?;
    let _ = gpu8.score(&warm4)?.topk(10)?;
    let _ = gpu4.score(&warm8)?.topk(10)?;
    quant.reset_stats();
    println!("MXFP_TRACE_READY rows={rows} dim={dim} pause_seconds={pause}");
    std::thread::sleep(Duration::from_secs(pause as u64));
    let repeats = 5;
    for _ in 0..repeats {
        let query_gpu4 = quant.upload_mxfp(&codec, std::slice::from_ref(&query4))?;
        let query_gpu8 = quant.upload_mxfp(&codec, std::slice::from_ref(&query8))?;
        let _ = gpu4.score(&query_gpu4)?.topk(10)?;
        let _ = gpu8.score(&query_gpu8)?.topk(10)?;
        let _ = gpu8.score(&query_gpu4)?.topk(10)?;
        let _ = gpu4.score(&query_gpu8)?.topk(10)?;
    }
    let stats = quant.stats();
    let payload = serde_json::to_vec_pretty(&json!({
        "issue": 1768,
        "git_sha": git_sha,
        "dim": dim,
        "candidate_rows": rows,
        "repetitions_per_combination": repeats,
        "corpora_resident_before_capture_pause": true,
        "gpu_stats": stats,
    }))
    .map_err(|error| bench_error(format!("trace JSON serialization failed: {error}")))?;
    fs::create_dir_all(&root)
        .map_err(|error| bench_error(format!("create trace root {root}: {error}")))?;
    let path = std::path::Path::new(&root).join("mxfp-cuda-trace.json");
    fs::write(&path, payload)
        .map_err(|error| bench_error(format!("write {}: {error}", path.display())))?;
    println!(
        "MXFP_TRACE_DONE d2h={} h2d={} path={}",
        stats.d2h_bytes,
        stats.h2d_bytes,
        path.display()
    );
    Ok(())
}

fn packed_rows(level: QuantLevel, dim: usize, rows: usize) -> Vec<QuantizedVec> {
    (0..rows)
        .map(|row| packed_row(level, dim, if row < 2 { 0 } else { row }))
        .collect()
}

fn packed_row(level: QuantLevel, dim: usize, row: usize) -> QuantizedVec {
    let blocks = dim.div_ceil(32);
    let block_bytes = match level {
        QuantLevel::Bits4Fp => 17,
        QuantLevel::Bits8Fp => 33,
        _ => unreachable!("benchmark uses MXFP levels"),
    };
    let mut bytes = vec![0; blocks * block_bytes];
    for block in 0..blocks {
        let used = (dim - block * 32).min(32);
        match level {
            QuantLevel::Bits4Fp => {
                for index in 0..32 {
                    let code = if index < used {
                        ((row.wrapping_mul(17) + block * 5 + index * 3) % 15) as u8
                    } else {
                        7
                    };
                    let byte = &mut bytes[block * block_bytes + index / 2];
                    if index.is_multiple_of(2) {
                        *byte |= code;
                    } else {
                        *byte |= code << 4;
                    }
                }
                bytes[block * block_bytes + 16] = 124 + ((row + block) % 7) as u8;
            }
            QuantLevel::Bits8Fp => {
                for index in 0..used {
                    bytes[block * block_bytes + index] =
                        row.wrapping_mul(29).wrapping_add(block * 11 + index * 7) as u8;
                }
                bytes[block * block_bytes + 32] = 124 + ((row + block) % 7) as u8;
            }
            _ => unreachable!("benchmark uses MXFP levels"),
        }
    }
    QuantizedVec {
        level,
        dim,
        bytes,
        scale: 0.0,
        seed_id: ZERO_SEED,
    }
}

fn repetitions(rows: usize) -> usize {
    match rows {
        0..=128 => 200,
        129..=1_024 => 50,
        1_025..=100_000 => 3,
        _ => 1,
    }
}

fn host_topk(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs = scores.iter().copied().enumerate().collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    pairs.truncate(k.min(pairs.len()));
    pairs
}

fn bench_sizes(name: &str, default: &[usize]) -> Result<Vec<usize>> {
    let Ok(value) = std::env::var(name) else {
        return Ok(default.to_vec());
    };
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| bench_error(format!("{name} contains invalid size {part:?}")))
        })
        .collect()
}

fn bench_usize(name: &str, default: usize, max: usize) -> Result<usize> {
    let Ok(value) = std::env::var(name) else {
        return Ok(default);
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0 && *value <= max)
        .ok_or_else(|| bench_error(format!("{name} must be in 1..={max}")))
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| bench_error(format!("{name} must be set")))
}

fn bench_error(detail: impl Into<String>) -> ForgeError {
    ForgeError::CacheError {
        op: "cuda_mxfp_benchmark".to_string(),
        path: "CALYX_FSV_ROOT".to_string(),
        detail: detail.into(),
        remediation: "Run the release benchmark on the production CUDA host".to_string(),
    }
}
