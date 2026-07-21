use super::{BINARY_CUDA_MIN_ELEMENTS, CudaQuantContext, INT8_CUDA_MIN_ELEMENTS};
use crate::cuda::{init_cuda, test_lock};
use crate::quant::{BinaryCodec, Quantizer, ScalarInt8Codec, binary_prefilter, new_seed};
use crate::{ForgeError, Result};
use serde_json::{Value, json};
use std::fs;
use std::time::{Duration, Instant};

fn fixture(rows: usize, dim: usize, salt: u32) -> Vec<f32> {
    let mut state = salt;
    (0..rows * dim)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0;
            random + (index % 7) as f32 * 0.00390625
        })
        .collect()
}

#[test]
#[ignore = "release-only production GPU benchmark/FSV; set CALYX_PACKED_BENCH_* variables"]
fn packed_cuda_benchmark_fsv() -> Result<()> {
    let _guard = test_lock();
    let dim = bench_usize("CALYX_PACKED_BENCH_DIM", 768)?;
    let rows = bench_sizes(
        "CALYX_PACKED_BENCH_ROWS",
        &[1, 8, 32, 64, 128, 1_024, 100_000, 1_000_000],
    )?;
    let root = std::env::var("CALYX_FSV_ROOT")
        .map_err(|_| bench_error("CALYX_FSV_ROOT must name the durable evidence directory"))?;
    let git_sha = std::env::var("CALYX_BENCH_GIT_SHA")
        .map_err(|_| bench_error("CALYX_BENCH_GIT_SHA must identify the tested commit"))?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    warm_packed_kernels(&quant, dim)?;
    let binary = benchmark_binary(&quant, dim, &rows)?;
    let int8 = benchmark_int8(&quant, dim, &rows)?;
    let payload = serde_json::to_vec_pretty(&json!({
        "issue": 1767,
        "git_sha": git_sha,
        "dim": dim,
        "binary_dispatch_threshold_elements": BINARY_CUDA_MIN_ELEMENTS,
        "int8_dispatch_threshold_elements": INT8_CUDA_MIN_ELEMENTS,
        "binary": binary,
        "int8": int8,
    }))
    .map_err(|error| bench_error(format!("benchmark JSON serialization failed: {error}")))?;
    fs::create_dir_all(&root)
        .map_err(|error| bench_error(format!("create FSV root {root}: {error}")))?;
    let path = std::path::Path::new(&root).join("packed-quant-cuda-benchmark.json");
    fs::write(&path, &payload)
        .map_err(|error| bench_error(format!("write {}: {error}", path.display())))?;
    assert_eq!(
        fs::read(&path)
            .map_err(|error| bench_error(format!("read {}: {error}", path.display())))?,
        payload
    );
    println!("PACKED_QUANT_CUDA_FSV path={}", path.display());
    Ok(())
}

#[test]
#[ignore = "Nsight-only resident search trace; set CALYX_PACKED_TRACE_* variables"]
fn packed_cuda_resident_trace_fsv() -> Result<()> {
    let _guard = test_lock();
    let dim = bench_usize("CALYX_PACKED_TRACE_DIM", 768)?;
    let rows = bench_sizes("CALYX_PACKED_TRACE_ROWS", &[100_000])?[0];
    let pause_seconds = bench_usize("CALYX_PACKED_TRACE_PAUSE_SECONDS", 6)?;
    let root = std::env::var("CALYX_FSV_ROOT")
        .map_err(|_| bench_error("CALYX_FSV_ROOT must name the durable trace directory"))?;
    let git_sha = std::env::var("CALYX_BENCH_GIT_SHA")
        .map_err(|_| bench_error("CALYX_BENCH_GIT_SHA must identify the traced commit"))?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let query = (0..dim)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let mut corpus = fixture(rows, dim, 0x1767);
    corpus.iter_mut().for_each(|value| *value *= 0.25);
    corpus[..dim].copy_from_slice(&query);
    if rows > 1 {
        corpus[dim..2 * dim].copy_from_slice(&query);
    }
    let binary = BinaryCodec::new(new_seed(dim, b"issue-1767-resident-trace"))?;
    let int8 = ScalarInt8Codec::new(dim);
    let binary_corpus = quant.encode_binary(&binary, &corpus)?;
    let int8_corpus = quant.encode_int8(&int8, &corpus)?;
    drop(corpus);
    let warm_binary_query = quant.encode_binary(&binary, &query)?;
    let warm_int8_query = quant.encode_int8(&int8, &query)?;
    let _ = binary_corpus.score(&warm_binary_query)?.topk(10)?;
    let _ = int8_corpus.score(&warm_int8_query)?.topk(10)?;

    quant.reset_stats();
    println!("PACKED_QUANT_TRACE_READY rows={rows} dim={dim} pause_seconds={pause_seconds}");
    std::thread::sleep(Duration::from_secs(pause_seconds as u64));
    let repeats = 5;
    for _ in 0..repeats {
        let binary_query = quant.encode_binary(&binary, &query)?;
        let binary_top = binary_corpus.score(&binary_query)?.topk(10)?;
        assert_eq!(binary_top[0].0, 0);
        assert_eq!(binary_top[1].0, 1);
        let int8_query = quant.encode_int8(&int8, &query)?;
        let int8_top = int8_corpus.score(&int8_query)?.topk(10)?;
        assert_eq!(int8_top[0].0, 0);
        assert_eq!(int8_top[1].0, 1);
    }
    let stats = quant.stats();
    assert_compact_readback(stats.d2h_bytes, rows, repeats, "resident trace")?;
    let payload = serde_json::to_vec_pretty(&json!({
        "issue": 1767,
        "git_sha": git_sha,
        "dim": dim,
        "candidate_rows": rows,
        "repetitions_per_codec": repeats,
        "corpora_resident_before_capture_pause": true,
        "gpu_stats": stats,
    }))
    .map_err(|error| bench_error(format!("trace JSON serialization failed: {error}")))?;
    fs::create_dir_all(&root)
        .map_err(|error| bench_error(format!("create trace root {root}: {error}")))?;
    let path = std::path::Path::new(&root).join("packed-quant-cuda-trace.json");
    fs::write(&path, &payload)
        .map_err(|error| bench_error(format!("write {}: {error}", path.display())))?;
    println!(
        "PACKED_QUANT_TRACE_DONE d2h={} h2d={} path={}",
        stats.d2h_bytes,
        stats.h2d_bytes,
        path.display()
    );
    Ok(())
}

fn warm_packed_kernels(quant: &CudaQuantContext, dim: usize) -> Result<()> {
    let input = fixture(2, dim, 0x1767);
    let query = &input[..dim];
    let binary = BinaryCodec::new(new_seed(dim, b"issue-1767-benchmark"))?;
    let binary_corpus = quant.encode_binary(&binary, &input)?;
    let binary_query = quant.encode_binary(&binary, query)?;
    let _ = binary_corpus.score(&binary_query)?.topk(1)?;
    let int8 = ScalarInt8Codec::new(dim);
    let int8_corpus = quant.encode_int8(&int8, &input)?;
    let int8_query = quant.encode_int8(&int8, query)?;
    let _ = int8_corpus.score(&int8_query)?.topk(1)?;
    Ok(())
}

fn benchmark_binary(
    quant: &CudaQuantContext,
    dim: usize,
    row_sizes: &[usize],
) -> Result<Vec<Value>> {
    let codec = BinaryCodec::new(new_seed(dim, b"issue-1767-benchmark"))?;
    let query = fixture(1, dim, 0xB1767);
    let mut measurements = Vec::new();
    for &rows in row_sizes {
        let mut corpus = fixture(rows, dim, 0xB100 ^ rows as u32);
        corpus[..dim].copy_from_slice(&query);
        if rows > 1 {
            corpus[dim..2 * dim].copy_from_slice(&query);
        }
        let cpu_build_started = Instant::now();
        let cpu_corpus = corpus
            .chunks_exact(dim)
            .map(|row| codec.encode(row))
            .collect::<Result<Vec<_>>>()?;
        let cpu_build_seconds = cpu_build_started.elapsed().as_secs_f64();
        let gpu_build_started = Instant::now();
        let gpu_corpus = quant.encode_binary(&codec, &corpus)?;
        let gpu_build_seconds = gpu_build_started.elapsed().as_secs_f64();
        drop(corpus);
        let repeats = repetitions(rows);

        let cpu_started = Instant::now();
        let cpu_query = codec.encode(&query)?;
        let mut cpu_top = Vec::new();
        for _ in 0..repeats {
            cpu_top = binary_prefilter(&cpu_query, &cpu_corpus, 10)?;
        }
        let cpu_seconds = cpu_started.elapsed().as_secs_f64();

        quant.reset_stats();
        let gpu_started = Instant::now();
        let mut gpu_top = Vec::new();
        for _ in 0..repeats {
            let gpu_query = quant.encode_binary(&codec, &query)?;
            gpu_top = gpu_corpus
                .score(&gpu_query)?
                .topk(10)?
                .into_iter()
                .map(|(index, _)| index)
                .collect();
        }
        let gpu_seconds = gpu_started.elapsed().as_secs_f64();
        assert_eq!(gpu_top, cpu_top, "binary top-k rows={rows}");
        let stats = quant.stats();
        assert_compact_readback(stats.d2h_bytes, rows, repeats, "binary")?;
        emit_measurement(
            "binary",
            rows,
            repeats,
            cpu_seconds,
            gpu_seconds,
            stats.d2h_bytes,
        );
        measurements.push(json!({
            "candidate_rows": rows,
            "elements": rows.saturating_mul(dim),
            "repetitions": repeats,
            "cpu_build_seconds": cpu_build_seconds,
            "gpu_build_seconds": gpu_build_seconds,
            "cpu_search_seconds": cpu_seconds,
            "gpu_search_seconds": gpu_seconds,
            "search_speedup": cpu_seconds / gpu_seconds,
            "topk_exact": true,
            "gpu_stats": stats,
        }));
    }
    Ok(measurements)
}

fn benchmark_int8(quant: &CudaQuantContext, dim: usize, row_sizes: &[usize]) -> Result<Vec<Value>> {
    let codec = ScalarInt8Codec::new(dim);
    let query = fixture(1, dim, 0x81767);
    let mut measurements = Vec::new();
    for &rows in row_sizes {
        let mut corpus = fixture(rows, dim, 0x8100 ^ rows as u32);
        corpus[..dim].copy_from_slice(&query);
        if rows > 1 {
            corpus[dim..2 * dim].copy_from_slice(&query);
        }
        let cpu_build_started = Instant::now();
        let cpu_corpus = corpus
            .chunks_exact(dim)
            .map(|row| codec.encode(row))
            .collect::<Result<Vec<_>>>()?;
        let cpu_build_seconds = cpu_build_started.elapsed().as_secs_f64();
        let gpu_build_started = Instant::now();
        let gpu_corpus = quant.encode_int8(&codec, &corpus)?;
        let gpu_build_seconds = gpu_build_started.elapsed().as_secs_f64();
        drop(corpus);
        let repeats = repetitions(rows);

        let cpu_started = Instant::now();
        let cpu_query = codec.encode(&query)?;
        let mut cpu_top = Vec::new();
        for _ in 0..repeats {
            let scores = cpu_corpus
                .iter()
                .map(|candidate| codec.dot_estimate(&cpu_query, candidate))
                .collect::<Result<Vec<_>>>()?;
            cpu_top = host_topk(&scores, 10)
                .into_iter()
                .map(|(index, _)| index)
                .collect();
        }
        let cpu_seconds = cpu_started.elapsed().as_secs_f64();

        quant.reset_stats();
        let gpu_started = Instant::now();
        let mut gpu_top = Vec::new();
        for _ in 0..repeats {
            let gpu_query = quant.encode_int8(&codec, &query)?;
            gpu_top = gpu_corpus
                .score(&gpu_query)?
                .topk(10)?
                .into_iter()
                .map(|(index, _)| index)
                .collect();
        }
        let gpu_seconds = gpu_started.elapsed().as_secs_f64();
        assert_eq!(gpu_top, cpu_top, "int8 top-k rows={rows}");
        let stats = quant.stats();
        assert_compact_readback(stats.d2h_bytes, rows, repeats, "int8")?;
        emit_measurement(
            "int8",
            rows,
            repeats,
            cpu_seconds,
            gpu_seconds,
            stats.d2h_bytes,
        );
        measurements.push(json!({
            "candidate_rows": rows,
            "elements": rows.saturating_mul(dim),
            "repetitions": repeats,
            "cpu_build_seconds": cpu_build_seconds,
            "gpu_build_seconds": gpu_build_seconds,
            "cpu_search_seconds": cpu_seconds,
            "gpu_search_seconds": gpu_seconds,
            "search_speedup": cpu_seconds / gpu_seconds,
            "topk_exact": true,
            "gpu_stats": stats,
        }));
    }
    Ok(measurements)
}

fn repetitions(rows: usize) -> usize {
    match rows {
        0..=128 => 200,
        129..=1_024 => 50,
        1_025..=100_000 => 5,
        _ => 1,
    }
}

fn assert_compact_readback(bytes: u64, rows: usize, repeats: usize, codec: &str) -> Result<()> {
    let full_scores = rows
        .saturating_mul(repeats)
        .saturating_mul(size_of::<f32>()) as u64;
    if rows > 32 && bytes >= full_scores {
        return Err(bench_error(format!(
            "{codec} D2H was not compact: bytes={bytes} full_scores={full_scores}"
        )));
    }
    Ok(())
}

fn emit_measurement(
    codec: &str,
    rows: usize,
    repeats: usize,
    cpu_seconds: f64,
    gpu_seconds: f64,
    d2h_bytes: u64,
) {
    println!(
        "PACKED_QUANT_CUDA_BENCH codec={codec} rows={rows} repeats={repeats} cpu={cpu_seconds:.6}s gpu={gpu_seconds:.6}s speedup={:.3}x d2h={d2h_bytes}",
        cpu_seconds / gpu_seconds,
    );
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

fn bench_usize(name: &str, default: usize) -> Result<usize> {
    let Ok(value) = std::env::var(name) else {
        return Ok(default);
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0 && *value <= 4_096)
        .ok_or_else(|| bench_error(format!("{name} must be in 1..=4096")))
}

fn bench_error(detail: impl Into<String>) -> ForgeError {
    ForgeError::CacheError {
        op: "cuda_packed_quant_benchmark".to_string(),
        path: "CALYX_FSV_ROOT".to_string(),
        detail: detail.into(),
        remediation: "Run the release benchmark on the production CUDA host".to_string(),
    }
}
