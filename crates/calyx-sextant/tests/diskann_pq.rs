use calyx_sextant::index::{
    DISKANN_PQ_SMALL_CORPUS_ROWS, DiskAnnPqBuildExecution, DiskAnnPqBuildParams, DiskAnnPqIndex,
};

#[cfg(sextant_cuvs)]
const CODEBOOK_ABS_TOLERANCE: f32 = 2.0e-4;
#[cfg(sextant_cuvs)]
const ENCODED_DISTANCE_TOLERANCE: f32 = 5.0e-3;
#[cfg(sextant_cuvs)]
const RECALL_AT_20_TOLERANCE: f64 = 0.02;

fn rows() -> Vec<(u32, Vec<f32>)> {
    (0..32)
        .map(|idx| {
            let x = idx as f32 / 32.0;
            (idx, vec![x, x + 0.1, 1.0 - x, 0.9 - x])
        })
        .collect()
}

#[test]
fn pq_build_encode_lut_and_readback() {
    let dir = std::env::temp_dir().join(format!("calyx-pq-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("graph.pq");
    let index = DiskAnnPqIndex::build(
        &rows(),
        DiskAnnPqBuildParams {
            subvectors: 2,
            centroids: 8,
            iterations: 3,
        },
    )
    .expect("build pq");
    index.write_atomic(&path).expect("write pq");
    let read = DiskAnnPqIndex::read(&path).expect("read pq");
    assert_eq!(read.node_count(), 32);
    assert_eq!(read.subvectors(), 2);
    assert_eq!(read.centroids(), 8);
    assert!(read.ram_bytes() > 32 * 2);
    let query = read.query(&rows()[7].1).expect("query");
    let self_distance = query.distance_l2(7).expect("self distance");
    let far_distance = query.distance_l2(31).expect("far distance");
    assert!(self_distance <= far_distance);
    assert_eq!(index.build_diagnostics().backend, "cpu-reference-v1");
    assert_eq!(read.build_diagnostics().backend, "sidecar-v1-read");
    assert_eq!(read.codebook(), index.codebook());
    assert_eq!(read.codes(), index.codes());
}

#[test]
fn pq_rejects_non_finite_and_degenerate_training_rows() {
    let mut non_finite = rows();
    non_finite[4].1[2] = f32::NAN;
    let err = DiskAnnPqIndex::build(&non_finite, params()).expect_err("NaN must fail closed");
    assert_eq!(err.code, "CALYX_INDEX_INVALID_PARAMS");
    assert!(err.message.contains("non-finite"));

    let degenerate: Vec<_> = (0..16).map(|id| (id, vec![1.0; 4])).collect();
    let err = DiskAnnPqIndex::build(&degenerate, params()).expect_err("degenerate must fail");
    assert_eq!(err.code, "CALYX_INDEX_INVALID_PARAMS");
    assert!(err.message.contains("degenerate"));
}

#[test]
fn pq_reader_rejects_out_of_range_code() {
    let dir = std::env::temp_dir().join(format!("calyx-pq-corrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("graph.pq");
    DiskAnnPqIndex::build(&rows(), params())
        .expect("build")
        .write_atomic(&path)
        .expect("write");
    let mut bytes = std::fs::read(&path).expect("read bytes");
    *bytes.last_mut().expect("code byte") = u8::MAX;
    std::fs::write(&path, bytes).expect("corrupt code");
    let err = DiskAnnPqIndex::read(&path).expect_err("out-of-range code must fail");
    assert_eq!(err.code, "CALYX_INDEX_CORRUPT");
    assert!(err.message.contains("out-of-range code"));
}

#[cfg(not(sextant_cuvs))]
#[test]
fn pq_auto_refuses_large_silent_cpu_fallback() {
    let rows = synthetic_rows(DISKANN_PQ_SMALL_CORPUS_ROWS + 1, 4);
    let err = DiskAnnPqIndex::build(&rows, params()).expect_err("large auto build needs CUDA");
    assert_eq!(err.code, "CALYX_INDEX_INVALID_PARAMS");
    assert!(err.message.contains("refusing silent CPU training"));
}

#[test]
fn pq_explicit_cpu_reference_is_auditable_above_threshold() {
    let rows = synthetic_rows(DISKANN_PQ_SMALL_CORPUS_ROWS + 1, 4);
    let index = DiskAnnPqIndex::build_with_execution(
        &rows,
        DiskAnnPqBuildParams {
            subvectors: 2,
            centroids: 4,
            iterations: 1,
        },
        DiskAnnPqBuildExecution::CpuReference,
    )
    .expect("explicit reference build");
    assert_eq!(index.build_diagnostics().backend, "cpu-reference-v1");
    assert_eq!(
        index.build_diagnostics().requested_execution,
        "cpu-reference"
    );
}

#[test]
fn pq_exact_ties_choose_the_lowest_centroid_id() {
    let rows = vec![(0, vec![0.0]), (1, vec![2.0]), (2, vec![1.0])];
    let params = DiskAnnPqBuildParams {
        subvectors: 1,
        centroids: 2,
        iterations: 1,
    };
    let cpu =
        DiskAnnPqIndex::build_with_execution(&rows, params, DiskAnnPqBuildExecution::CpuReference)
            .expect("CPU tie build");
    assert_eq!(cpu.codebook(), &[0.5, 2.0]);
    assert_eq!(cpu.codes(), &[0, 1, 0]);

    #[cfg(sextant_cuvs)]
    {
        let gpu = DiskAnnPqIndex::build_with_execution(
            &rows,
            params,
            DiskAnnPqBuildExecution::CudaRequired,
        )
        .expect("CUDA tie build");
        assert_eq!(gpu.codebook(), cpu.codebook());
        assert_eq!(gpu.codes(), cpu.codes());
    }
}

#[cfg(sextant_cuvs)]
#[test]
fn pq_cuda_matches_cpu_and_reuses_the_resident_upload() {
    let rows = synthetic_rows(8_192, 32);
    let params = DiskAnnPqBuildParams {
        subvectors: 4,
        centroids: 32,
        iterations: 4,
    };
    let cpu =
        DiskAnnPqIndex::build_with_execution(&rows, params, DiskAnnPqBuildExecution::CpuReference)
            .expect("CPU reference");
    let gpu =
        DiskAnnPqIndex::build_with_execution(&rows, params, DiskAnnPqBuildExecution::CudaRequired)
            .expect("CUDA build");
    let gpu_repeat =
        DiskAnnPqIndex::build_with_execution(&rows, params, DiskAnnPqBuildExecution::CudaRequired)
            .expect("repeat CUDA build");
    let max_codebook_delta = cpu
        .codebook()
        .iter()
        .zip(gpu.codebook())
        .map(|(cpu, gpu)| (cpu - gpu).abs())
        .fold(0.0_f32, f32::max);
    let code_agreement = cpu
        .codes()
        .iter()
        .zip(gpu.codes())
        .filter(|(cpu, gpu)| cpu == gpu)
        .count() as f64
        / cpu.codes().len() as f64;
    assert!(
        max_codebook_delta <= CODEBOOK_ABS_TOLERANCE,
        "delta={max_codebook_delta}"
    );
    assert!(code_agreement >= 0.999, "agreement={code_agreement}");
    assert_eq!(gpu.codes(), gpu_repeat.codes(), "CUDA codes must repeat");
    let repeat_delta = gpu
        .codebook()
        .iter()
        .zip(gpu_repeat.codebook())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(repeat_delta <= 2.0e-5, "repeat delta={repeat_delta}");
    let mut max_encoded_distance_ratio = 0.0_f32;
    let mut cpu_recall = 0.0;
    let mut gpu_recall = 0.0;
    let query_ids = [0, 97, 1_024, 2_731, 4_095, 6_001, 7_777, 8_191];
    for query_id in query_ids {
        let cpu_query = cpu.query(&rows[query_id].1).expect("CPU PQ query");
        let gpu_query = gpu.query(&rows[query_id].1).expect("GPU PQ query");
        for id in 0..rows.len() {
            let cpu_distance = cpu_query.distance_l2(id as u32).expect("CPU distance");
            let gpu_distance = gpu_query.distance_l2(id as u32).expect("GPU distance");
            let ratio = (cpu_distance - gpu_distance).abs() / (1.0 + cpu_distance.abs());
            max_encoded_distance_ratio = max_encoded_distance_ratio.max(ratio);
        }
        cpu_recall += recall_at(&rows, &cpu, query_id, 20);
        gpu_recall += recall_at(&rows, &gpu, query_id, 20);
    }
    cpu_recall /= query_ids.len() as f64;
    gpu_recall /= query_ids.len() as f64;
    assert!(
        max_encoded_distance_ratio <= ENCODED_DISTANCE_TOLERANCE,
        "encoded distance ratio={max_encoded_distance_ratio}"
    );
    assert!(
        (cpu_recall - gpu_recall).abs() <= RECALL_AT_20_TOLERANCE,
        "CPU recall={cpu_recall}, GPU recall={gpu_recall}"
    );

    let dir = std::env::temp_dir().join(format!("calyx-pq-gpu-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("sidecar dir");
    let path = dir.join("graph.pq");
    gpu.write_atomic(&path).expect("write GPU sidecar");
    assert_eq!(
        &std::fs::read(&path).expect("sidecar bytes")[..8],
        b"CLXPQ001"
    );
    let read = DiskAnnPqIndex::read(&path).expect("read GPU sidecar");
    assert_eq!(read.codebook(), gpu.codebook());
    assert_eq!(read.codes(), gpu.codes());
    std::fs::remove_dir_all(&dir).expect("remove sidecar dir");

    let diagnostics = gpu.build_diagnostics();
    assert_eq!(diagnostics.backend, "cuda-lloyd-tiled-v1");
    assert!(diagnostics.strict_gpu_required);
    assert!(diagnostics.pinned_staging);
    assert!(diagnostics.resident_corpus);
    assert!(diagnostics.subspace_upload_reuse);
    assert_eq!(diagnostics.corpus_uploads, 1);
    assert_eq!(
        diagnostics.assignment_kernel_launches,
        params.iterations + 1
    );
    assert_eq!(diagnostics.accumulation_kernel_launches, params.iterations);
    assert_eq!(diagnostics.centroid_kernel_launches, params.iterations);
    println!(
        "{}",
        serde_json::json!({
            "code_agreement": code_agreement,
            "cpu_recall_at_20": cpu_recall,
            "gpu_recall_at_20": gpu_recall,
            "max_codebook_delta": max_codebook_delta,
            "max_encoded_distance_ratio": max_encoded_distance_ratio,
            "diagnostics": diagnostics,
        })
    );
}

#[cfg(sextant_cuvs)]
#[test]
#[ignore = "manual CUDA streaming probe; set resident MiB to zero and chunk MiB to one"]
fn pq_cuda_streams_bounded_chunks() {
    assert_eq!(
        std::env::var("CALYX_DISKANN_PQ_GPU_RESIDENT_MIB").as_deref(),
        Ok("0")
    );
    assert_eq!(
        std::env::var("CALYX_DISKANN_PQ_GPU_CHUNK_MIB").as_deref(),
        Ok("1")
    );
    let rows = synthetic_rows(16_384, 32);
    let params = DiskAnnPqBuildParams {
        subvectors: 4,
        centroids: 32,
        iterations: 2,
    };
    let gpu =
        DiskAnnPqIndex::build_with_execution(&rows, params, DiskAnnPqBuildExecution::CudaRequired)
            .expect("streaming CUDA build");
    let diagnostics = gpu.build_diagnostics();
    assert!(!diagnostics.resident_corpus);
    assert_eq!(diagnostics.chunk_rows, 8_192);
    assert_eq!(diagnostics.chunks_per_pass, 2);
    assert_eq!(diagnostics.corpus_uploads, 6);
    assert_eq!(diagnostics.assignment_kernel_launches, 6);
    assert_eq!(diagnostics.accumulation_kernel_launches, 4);
    assert_eq!(diagnostics.centroid_kernel_launches, 2);
    assert_eq!(diagnostics.h2d_transfers, 7);
    assert_eq!(diagnostics.d2h_transfers, 3);
    println!(
        "{}",
        serde_json::to_string(diagnostics).expect("serialize diagnostics")
    );
}

#[cfg(sextant_cuvs)]
#[test]
#[ignore = "manual release-mode CUDA crossover benchmark"]
fn pq_cuda_crossover_benchmark() {
    let params = DiskAnnPqBuildParams::default();
    let mut measurements = Vec::new();
    for row_count in [256, 512, 1_024, 2_048, 4_096, 32_768] {
        let rows = synthetic_rows(row_count, 128);
        let cpu_started = std::time::Instant::now();
        let cpu = DiskAnnPqIndex::build_with_execution(
            &rows,
            params,
            DiskAnnPqBuildExecution::CpuReference,
        )
        .expect("CPU reference");
        let cpu_us = cpu_started.elapsed().as_micros();
        let gpu_started = std::time::Instant::now();
        let gpu = DiskAnnPqIndex::build_with_execution(
            &rows,
            params,
            DiskAnnPqBuildExecution::CudaRequired,
        )
        .expect("CUDA benchmark");
        let gpu_us = gpu_started.elapsed().as_micros();
        let cpu_diagnostics = cpu.build_diagnostics();
        let gpu_diagnostics = gpu.build_diagnostics();
        let agreement = cpu
            .codes()
            .iter()
            .zip(gpu.codes())
            .filter(|(cpu, gpu)| cpu == gpu)
            .count() as f64
            / cpu.codes().len() as f64;
        measurements.push(serde_json::json!({
            "rows": row_count,
            "cpu_us": cpu_us,
            "gpu_us": gpu_us,
            "speedup": cpu_us as f64 / gpu_us as f64,
            "training_speedup": cpu_diagnostics.training_us as f64
                / gpu_diagnostics.training_us.max(1) as f64,
            "encoding_speedup": cpu_diagnostics.encoding_us as f64
                / gpu_diagnostics.encoding_us.max(1) as f64,
            "code_agreement": agreement,
            "cpu_diagnostics": cpu_diagnostics,
            "gpu_diagnostics": gpu_diagnostics,
        }));
    }
    println!("{}", serde_json::to_string(&measurements).expect("JSON"));
}

#[cfg(sextant_cuvs)]
#[test]
#[ignore = "manual release-mode CUDA scale benchmark"]
fn pq_cuda_scale_benchmark() {
    let row_count = std::env::var("CALYX_ISSUE1516_SCALE_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(262_144);
    let rows = synthetic_rows(row_count, 128);
    let started = std::time::Instant::now();
    let gpu = DiskAnnPqIndex::build_with_execution(
        &rows,
        DiskAnnPqBuildParams::default(),
        DiskAnnPqBuildExecution::CudaRequired,
    )
    .expect("CUDA scale build");
    println!(
        "{}",
        serde_json::json!({
            "wall_us": started.elapsed().as_micros(),
            "diagnostics": gpu.build_diagnostics(),
        })
    );
}

fn params() -> DiskAnnPqBuildParams {
    DiskAnnPqBuildParams {
        subvectors: 2,
        centroids: 8,
        iterations: 3,
    }
}

fn synthetic_rows(count: usize, dim: usize) -> Vec<(u32, Vec<f32>)> {
    (0..count)
        .map(|row| {
            let vector = (0..dim)
                .map(|axis| {
                    let phase = (row * 131 + axis * 17) as f32 * 0.001_953_125;
                    phase.sin() + (phase * 0.37).cos() * 0.25 + row as f32 * 1.0e-7
                })
                .collect();
            (row as u32, vector)
        })
        .collect()
}

#[cfg(sextant_cuvs)]
fn recall_at(rows: &[(u32, Vec<f32>)], index: &DiskAnnPqIndex, query_id: usize, k: usize) -> f64 {
    let query = &rows[query_id].1;
    let mut exact: Vec<_> = rows
        .iter()
        .map(|(id, row)| (*id, l2_sq(query, row)))
        .collect();
    exact.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let pq_query = index.query(query).expect("PQ recall query");
    let mut approximate: Vec<_> = rows
        .iter()
        .map(|(id, _)| (*id, pq_query.distance_l2(*id).expect("PQ recall distance")))
        .collect();
    approximate.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let exact = &exact[..k];
    approximate[..k]
        .iter()
        .filter(|candidate| exact.iter().any(|truth| truth.0 == candidate.0))
        .count() as f64
        / k as f64
}

#[cfg(sextant_cuvs)]
fn l2_sq(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

#[test]
fn pq_rejects_non_divisible_subvectors() {
    let err = DiskAnnPqIndex::build(
        &rows(),
        DiskAnnPqBuildParams {
            subvectors: 3,
            centroids: 8,
            iterations: 1,
        },
    )
    .expect_err("bad subvectors");
    assert_eq!(err.code, "CALYX_INDEX_INVALID_PARAMS");
}
