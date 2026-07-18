#![cfg(sextant_cuvs)]

use std::path::PathBuf;

use calyx_core::{CxId, SlotId};
use calyx_sextant::index::{
    DiskAnnBuildBackend, DiskAnnBuildParams, DiskAnnPqBuildExecution, DiskAnnPqBuildParams,
    DiskAnnPqIndex, DiskAnnPqSearchBuild, DiskAnnSearch, DiskAnnSearchParams, l2_normalize,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("calyx-diskann-pq-e2e")
        .join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn vectors(n: usize, dim: usize) -> Vec<(CxId, Vec<f32>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(1_516);
    (0..n)
        .map(|idx| {
            let mut vector: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            vector[idx % dim] += 4.0;
            let mut bytes = [0_u8; 16];
            bytes[8..].copy_from_slice(&(idx as u64).to_be_bytes());
            (CxId::from_bytes(bytes), vector)
        })
        .collect()
}

fn build_params() -> DiskAnnBuildParams {
    DiskAnnBuildParams {
        dim: 128,
        m_max: 16,
        ef_construction: 64,
        alpha: 1.2,
    }
}

fn search_params() -> DiskAnnSearchParams {
    DiskAnnSearchParams {
        beamwidth: 32,
        ef_search: 64,
        rescore_k: 32,
        rescore_from_raw: false,
    }
}

#[test]
#[ignore = "manual release-mode CAGRA plus PQ end-to-end benchmark"]
fn cagra_pq_gpu_improves_end_to_end_build() {
    let row_count = std::env::var("CALYX_ISSUE1516_E2E_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32_768);
    let rows = vectors(row_count, 128);
    let normalized: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(id, (_, row))| (id as u32, l2_normalize(row)))
        .collect();
    let pq = DiskAnnPqBuildParams::default();
    let search = search_params();

    let baseline_dir = scratch("cpu");
    let baseline_graph = baseline_dir.join("graph.cda");
    let baseline_started = std::time::Instant::now();
    drop(
        DiskAnnSearch::build_with_backend(
            SlotId::new(0),
            &baseline_graph,
            &rows,
            build_params(),
            Some(baseline_dir.join("raw")),
            search,
            DiskAnnBuildBackend::CuvsCagra,
        )
        .expect("baseline CAGRA build"),
    );
    let baseline_pq = DiskAnnPqIndex::build_with_execution(
        &normalized,
        pq,
        DiskAnnPqBuildExecution::CpuReference,
    )
    .expect("baseline CPU PQ build");
    baseline_pq
        .write_atomic(&baseline_graph.with_extension("pq"))
        .expect("baseline PQ sidecar");
    let baseline_us = baseline_started.elapsed().as_micros();

    let candidate_dir = scratch("gpu");
    let candidate_started = std::time::Instant::now();
    let candidate = DiskAnnSearch::build_with_pq_plan(
        SlotId::new(0),
        candidate_dir.join("graph.cda"),
        &rows,
        build_params(),
        Some(candidate_dir.join("raw")),
        DiskAnnPqSearchBuild {
            search,
            pq,
            backend: DiskAnnBuildBackend::CuvsCagra,
        },
    )
    .expect("candidate CAGRA plus GPU PQ build");
    let candidate_us = candidate_started.elapsed().as_micros();
    let candidate_diagnostics = candidate
        .pq_build_diagnostics()
        .expect("candidate PQ diagnostics");
    assert_eq!(candidate_diagnostics.backend, "cuda-lloyd-tiled-v1");
    assert!(
        candidate_us < baseline_us,
        "baseline={baseline_us}us candidate={candidate_us}us"
    );
    println!(
        "{}",
        serde_json::json!({
            "rows": row_count,
            "baseline_cagra_plus_cpu_pq_us": baseline_us,
            "candidate_cagra_plus_gpu_pq_us": candidate_us,
            "end_to_end_speedup": baseline_us as f64 / candidate_us as f64,
            "baseline_pq_diagnostics": baseline_pq.build_diagnostics(),
            "candidate_pq_diagnostics": candidate_diagnostics,
        })
    );
}
