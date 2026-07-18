use super::{
    CudaQuantContext, CudaTurboQuantRow, QuantDispatch, TURBOQUANT_CUDA_MIN_ELEMENTS,
    turboquant_dispatch,
};
use crate::cuda::{init_cuda, test_lock};
use crate::quant::{QuantLevel, Quantizer, TurboQuantCodec, new_seed};
use crate::{ForgeError, Result};
use std::fs;
use std::time::Instant;

fn fixture(rows: usize, dim: usize, salt: u32) -> Vec<f32> {
    let mut state = salt;
    (0..rows * dim)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0;
            random + (index % 11) as f32 * 0.0078125
        })
        .collect()
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32, label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length");
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "{label}[{index}] left={left} right={right} tolerance={tolerance}"
        );
    }
}

#[test]
fn dispatch_boundary_is_explicit_and_overflow_safe() {
    assert_eq!(turboquant_dispatch(1, 1), QuantDispatch::Cpu);
    assert_eq!(
        turboquant_dispatch(1, TURBOQUANT_CUDA_MIN_ELEMENTS),
        QuantDispatch::Cuda
    );
    assert_eq!(turboquant_dispatch(usize::MAX, 2), QuantDispatch::Cpu);
}

#[test]
fn turboquant_cuda_encoding_and_decode_match_cpu() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    for dim in [1, 768, 1_536, 2_048] {
        let mut input = fixture(3, dim, dim as u32 ^ 0x1494);
        input[..dim].fill(0.0);
        for level in [QuantLevel::Bits3p5, QuantLevel::Bits2p5] {
            let codec = TurboQuantCodec::new(new_seed(dim, b"issue-1766-parity"), level)?;
            let gpu = quant.encode_turboquant(&codec, &input)?;
            assert_eq!(gpu.rows(), 3);
            assert_eq!(gpu.dim(), dim);
            assert_eq!(gpu.rotation_width(), dim.next_power_of_two());
            assert_eq!(gpu.level(), level);

            let gpu_encoded = gpu.read_encoded()?;
            let cpu_encoded = input
                .chunks_exact(dim)
                .map(|row| codec.encode(row))
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(gpu_encoded.len(), cpu_encoded.len());
            for (row, (gpu_row, cpu_row)) in gpu_encoded.iter().zip(&cpu_encoded).enumerate() {
                assert_eq!(gpu_row.level, cpu_row.level, "level row {row}");
                assert_eq!(gpu_row.dim, cpu_row.dim, "dim row {row}");
                assert_eq!(gpu_row.seed_id, cpu_row.seed_id, "seed row {row}");
                assert_eq!(
                    gpu_row.scale.to_bits(),
                    cpu_row.scale.to_bits(),
                    "scale row {row}"
                );
                assert_eq!(gpu_row.bytes, cpu_row.bytes, "encoded bytes row {row}");
            }

            let gpu_decoded = gpu.decode()?;
            let cpu_decoded = cpu_encoded
                .iter()
                .map(|row| codec.decode(row))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            assert_close(&gpu_decoded, &cpu_decoded, 2e-6, "decode");
        }
    }
    Ok(())
}

#[test]
fn turboquant_cuda_ragged_rows_preserve_shape_seed_and_bytes() -> Result<()> {
    let _guard = test_lock();
    let dims = [64, 128, 64, 257];
    let levels = [
        QuantLevel::Bits3p5,
        QuantLevel::Bits3p5,
        QuantLevel::Bits3p5,
        QuantLevel::Bits2p5,
    ];
    let codecs = dims
        .iter()
        .zip(levels)
        .enumerate()
        .map(|(row, (&dim, level))| {
            let entropy = format!("issue-1518-row-{row}");
            TurboQuantCodec::new(new_seed(dim, entropy.as_bytes()), level)
        })
        .collect::<Result<Vec<_>>>()?;
    let inputs = dims
        .iter()
        .enumerate()
        .map(|(row, &dim)| fixture(1, dim, 0x1518 + row as u32))
        .collect::<Vec<_>>();
    let rows = codecs
        .iter()
        .zip(&inputs)
        .map(|(codec, input)| CudaTurboQuantRow::new(codec, input))
        .collect::<Vec<_>>();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    quant.reset_stats();
    let gpu = quant.encode_turboquant_ragged(&rows)?;
    let cpu = codecs
        .iter()
        .zip(&inputs)
        .map(|(codec, input)| codec.encode(input))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        gpu, cpu,
        "ragged CUDA output must be the CPU canonical bytes"
    );
    assert_ne!(gpu[0].seed_id, gpu[2].seed_id);
    let stats = quant.stats();
    assert_eq!(stats.encoded_rows, rows.len() as u64);
    assert_eq!(
        stats.kernel_launches, 18,
        "three shape groups x six kernels"
    );
    assert!(stats.h2d_bytes > inputs.iter().map(Vec::len).sum::<usize>() as u64 * 4);
    assert!(stats.d2h_bytes > gpu.iter().map(|row| row.bytes.len()).sum::<usize>() as u64);
    Ok(())
}

#[test]
fn turboquant_cuda_ragged_validation_fails_before_launch() -> Result<()> {
    let _guard = test_lock();
    let codec = TurboQuantCodec::new(new_seed(8, b"issue-1518-validation"), QuantLevel::Bits3p5)?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    assert!(matches!(
        quant.encode_turboquant_ragged(&[]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let short = [0.0; 7];
    let wrong = [CudaTurboQuantRow::new(&codec, &short)];
    assert!(matches!(
        quant.encode_turboquant_ragged(&wrong),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let good = [0.0; 8];
    let mut bad = [0.0; 8];
    bad[3] = f32::INFINITY;
    let mixed = [
        CudaTurboQuantRow::new(&codec, &good),
        CudaTurboQuantRow::new(&codec, &bad),
    ];
    quant.reset_stats();
    assert!(matches!(
        quant.encode_turboquant_ragged(&mixed),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    assert_eq!(quant.stats().kernel_launches, 0);
    Ok(())
}

#[test]
fn turboquant_cuda_scores_stay_resident_until_read_or_topk() -> Result<()> {
    let _guard = test_lock();
    let dim = 128;
    let rows = 2_049;
    let codec = TurboQuantCodec::new(
        new_seed(dim, b"issue-1766-resident-score"),
        QuantLevel::Bits3p5,
    )?;
    let query = fixture(1, dim, 0xC0DE);
    let mut corpus = fixture(rows, dim, 0xCAFE);
    corpus[..dim].copy_from_slice(&query);
    corpus[dim..2 * dim].copy_from_slice(&query);
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let gpu_query = quant.encode_turboquant(&codec, &query)?;
    let gpu_corpus = quant.encode_turboquant(&codec, &corpus)?;

    quant.reset_stats();
    let scores = gpu_corpus.score(&gpu_query)?;
    let resident = quant.stats();
    assert_eq!(resident.kernel_launches, 1);
    assert_eq!(resident.scored_candidates, rows as u64);
    assert_eq!(resident.d2h_bytes, 0, "full scores must remain on device");

    let gpu_values = scores.read()?;
    let q = codec.encode(&query)?;
    let cpu_rows = corpus
        .chunks_exact(dim)
        .map(|row| codec.encode(row))
        .collect::<Result<Vec<_>>>()?;
    let cpu_values = codec.dot_estimate_batch(&q, &cpu_rows)?;
    assert_close(&gpu_values, &cpu_values, 3e-5, "score");
    assert_eq!(gpu_values[0].to_bits(), cpu_values[0].to_bits());

    quant.reset_stats();
    let top = scores.topk(8)?;
    let top_stats = quant.stats();
    assert_eq!(top[0].0, 0);
    assert_eq!(top[1].0, 1, "equal scores must use ascending index ties");
    assert_eq!(top_stats.kernel_launches, 1);
    assert_eq!(top_stats.compact_topk_rows, 24);
    assert_eq!(top_stats.d2h_bytes, 24 * 8);
    assert!(top_stats.d2h_bytes < rows as u64 * size_of::<f32>() as u64);
    Ok(())
}

#[test]
fn turboquant_cuda_rejects_bad_shapes_values_and_pairs() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let codec = TurboQuantCodec::new(new_seed(8, b"issue-1766-errors"), QuantLevel::Bits2p5)?;
    let shape = quant
        .encode_turboquant(&codec, &[0.0; 7])
        .expect_err("incomplete rows must fail");
    assert!(matches!(shape, ForgeError::ShapeMismatch { .. }));
    let mut nonfinite = [0.0; 8];
    nonfinite[3] = f32::NAN;
    let numerical = quant
        .encode_turboquant(&codec, &nonfinite)
        .expect_err("NaN must fail");
    assert!(matches!(numerical, ForgeError::NumericalInvariant { .. }));
    let oversized_codec = TurboQuantCodec::new(
        new_seed(4_097, b"issue-1766-oversized"),
        QuantLevel::Bits2p5,
    )?;
    let oversized = quant
        .encode_turboquant(&oversized_codec, &vec![0.0; 4_097])
        .expect_err("unsupported shared-memory shape must fail before allocation");
    assert!(matches!(oversized, ForgeError::ShapeMismatch { .. }));

    let multi_query = quant.encode_turboquant(&codec, &[0.0; 16])?;
    let corpus = quant.encode_turboquant(&codec, &[0.0; 8])?;
    assert!(corpus.score(&multi_query).is_err());
    let other_codec =
        TurboQuantCodec::new(new_seed(8, b"issue-1766-other-seed"), QuantLevel::Bits2p5)?;
    let other = quant.encode_turboquant(&other_codec, &[0.0; 8])?;
    assert!(corpus.score(&other).is_err());
    Ok(())
}

#[test]
#[ignore = "release-only production GPU benchmark/FSV; set CALYX_QUANT_BENCH_* variables"]
fn turboquant_cuda_benchmark_fsv() -> Result<()> {
    let _guard = test_lock();
    let dims = bench_sizes("CALYX_QUANT_BENCH_DIMS", &[768, 1_536, 2_048])?;
    let rows = bench_sizes("CALYX_QUANT_BENCH_ROWS", &[1_024])?;
    let root = std::env::var("CALYX_FSV_ROOT")
        .map_err(|_| bench_error("CALYX_FSV_ROOT must name the durable evidence directory"))?;
    let git_sha = std::env::var("CALYX_BENCH_GIT_SHA")
        .map_err(|_| bench_error("CALYX_BENCH_GIT_SHA must identify the tested commit"))?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let mut measurements = Vec::new();

    for dim in dims {
        let codec =
            TurboQuantCodec::new(new_seed(dim, b"issue-1766-benchmark"), QuantLevel::Bits3p5)?;
        let query = fixture(1, dim, 0xBEEF ^ dim as u32);
        let warm_corpus = fixture(2, dim, 0x1766 ^ dim as u32);
        let warm_query = quant.encode_turboquant(&codec, &query)?;
        let warm_corpus = quant.encode_turboquant(&codec, &warm_corpus)?;
        let _ = warm_corpus.score(&warm_query)?.topk(1)?;
        for &candidate_rows in &rows {
            let mut corpus = fixture(candidate_rows, dim, 0xFACE ^ candidate_rows as u32);
            corpus[..dim].copy_from_slice(&query);
            if candidate_rows > 1 {
                corpus[dim..2 * dim].copy_from_slice(&query);
            }

            let cpu_started = Instant::now();
            let cpu_query = codec.encode(&query)?;
            let cpu_corpus = corpus
                .chunks_exact(dim)
                .map(|row| codec.encode(row))
                .collect::<Result<Vec<_>>>()?;
            let cpu_scores = codec.dot_estimate_batch(&cpu_query, &cpu_corpus)?;
            let cpu_top = host_topk(&cpu_scores, 10);
            let cpu_elapsed = cpu_started.elapsed();
            drop(cpu_scores);
            drop(cpu_corpus);

            quant.reset_stats();
            let gpu_started = Instant::now();
            let gpu_query = quant.encode_turboquant(&codec, &query)?;
            let gpu_corpus = quant.encode_turboquant(&codec, &corpus)?;
            let gpu_scores = gpu_corpus.score(&gpu_query)?;
            let gpu_top = gpu_scores.topk(10)?;
            let gpu_elapsed = gpu_started.elapsed();
            let stats = quant.stats();
            assert_eq!(gpu_top[0].0, 0);
            if candidate_rows > 1 {
                assert_eq!(gpu_top[1].0, 1);
            }
            let overlap = gpu_top
                .iter()
                .filter(|(index, _)| cpu_top.iter().any(|(cpu_index, _)| cpu_index == index))
                .count();
            let cpu_seconds = cpu_elapsed.as_secs_f64();
            let gpu_seconds = gpu_elapsed.as_secs_f64();
            measurements.push(serde_json::json!({
                "dim": dim,
                "candidate_rows": candidate_rows,
                "elements": dim.saturating_mul(candidate_rows),
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
                "speedup": cpu_seconds / gpu_seconds,
                "top10_overlap": overlap,
                "gpu_stats": stats,
            }));
            println!(
                "TURBOQUANT_CUDA_BENCH dim={dim} rows={candidate_rows} cpu={cpu_seconds:.6}s gpu={gpu_seconds:.6}s speedup={:.3}x overlap={overlap}/{} d2h={}",
                cpu_seconds / gpu_seconds,
                gpu_top.len(),
                stats.d2h_bytes,
            );
        }
    }

    let payload = serde_json::to_vec_pretty(&serde_json::json!({
        "issue": 1766,
        "git_sha": git_sha,
        "level": "bits3p5",
        "dispatch_threshold_elements": TURBOQUANT_CUDA_MIN_ELEMENTS,
        "measurements": measurements,
    }))
    .map_err(|error| bench_error(format!("benchmark JSON serialization failed: {error}")))?;
    fs::create_dir_all(&root)
        .map_err(|error| bench_error(format!("create FSV root {root}: {error}")))?;
    let path = std::path::Path::new(&root).join("turboquant-cuda-benchmark.json");
    fs::write(&path, &payload)
        .map_err(|error| bench_error(format!("write {}: {error}", path.display())))?;
    let readback = fs::read(&path)
        .map_err(|error| bench_error(format!("read {}: {error}", path.display())))?;
    assert_eq!(readback, payload);
    println!("TURBOQUANT_CUDA_FSV path={}", path.display());
    Ok(())
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

fn bench_error(detail: impl Into<String>) -> ForgeError {
    ForgeError::CacheError {
        op: "cuda_turboquant_benchmark".to_string(),
        path: "CALYX_FSV_ROOT".to_string(),
        detail: detail.into(),
        remediation: "Set durable benchmark paths and rerun on the production CUDA host"
            .to_string(),
    }
}
