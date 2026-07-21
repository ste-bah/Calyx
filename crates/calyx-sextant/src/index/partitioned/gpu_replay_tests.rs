use super::*;

#[test]
#[ignore = "requires explicit real corpus and output paths"]
fn replay_real_gpu_assignment_contract_from_env() {
    let source_path = std::env::var("CALYX_PARTITION_REPLAY_SOURCE").expect("replay source");
    let centroid_path = std::env::var("CALYX_PARTITION_REPLAY_CENTROIDS").ok();
    let root = std::path::PathBuf::from(
        std::env::var("CALYX_PARTITION_REPLAY_ROOT").expect("replay root"),
    );
    let max_factor: f64 = std::env::var("CALYX_PARTITION_REPLAY_MAX_FACTOR")
        .unwrap_or_else(|_| "1.09828821".to_string())
        .parse()
        .expect("max replication factor");
    let boundary_epsilon: f32 = std::env::var("CALYX_PARTITION_REPLAY_EPSILON")
        .unwrap_or_else(|_| "0.3".to_string())
        .parse()
        .expect("replay boundary epsilon");
    assert!(!root.exists(), "replay root must not exist");

    let source =
        I8BinSource::open_raw(std::path::Path::new(&source_path)).expect("open replay source");
    let loaded = centroid_path
        .map(|path| SpannCentroidIndex::open_from_path(path).expect("open replay centroids"));
    let sample = 200_000usize.min(source.len() as usize);
    let initial_regions = loaded.as_ref().map_or_else(
        || {
            gpu::initial_cluster_count(
                DiskAnnBuildBackend::CuvsCagra,
                source.len(),
                1_024,
                Some(8_192),
                sample,
            )
        },
        SpannCentroidIndex::centroid_count,
    );
    let mut session = gpu::for_backend(
        DiskAnnBuildBackend::CuvsCagra,
        &source,
        100_000,
        sample,
        initial_regions,
    )
    .expect("create replay session")
    .expect("CUDA session");
    let fitted = loaded.is_none();
    let centroids = loaded.unwrap_or_else(|| {
        let stride = (source.len() / sample as u64).max(1);
        let rows: Vec<(u32, Vec<f32>)> = (0..sample)
            .into_par_iter()
            .map(|row| {
                let source_row = (row as u64 * stride) % source.len();
                (row as u32, source.row(source_row))
            })
            .collect();
        let centroids = session
            .fit_centroids(&rows, initial_regions, 42)
            .expect("fit replay centroids");
        centroids
            .save(root.join("centroids"))
            .expect("save replay centroids");
        centroids
    });
    let effective_boundary_epsilon = gpu::supported_boundary_epsilon(
        boundary_epsilon,
        sample,
        centroids.centroid_count(),
        source.dim(),
    );
    let (regions, stats) = assignment::stream_assign_to_ids_bounded_gpu(
        &root,
        AssignmentSink::Final,
        &centroids,
        &source,
        BoundedAssignmentConfig {
            cap: 16_384,
            routing_probe: 64,
            routing: AssignmentRouting::Exact,
            boundary_epsilon: effective_boundary_epsilon,
            max_replication: 2,
            apply_rng_rule: true,
            rng_factor: 1.0,
        },
        8_192,
        &mut session,
    )
    .expect("replay bounded GPU assignment");
    let stored: usize = regions.iter().map(|region| region.count).sum();
    let factor = stored as f64 / source.len() as f64;
    let report = serde_json::json!({
        "rows": source.len(),
        "regions": regions.len(),
        "stored": stored,
        "replication_factor": factor,
        "max_region_count": regions.iter().map(|region| region.count).max(),
        "fitted_centroids": fitted,
        "requested_boundary_epsilon": boundary_epsilon,
        "effective_boundary_epsilon": effective_boundary_epsilon,
        "closure": stats,
        "diagnostics": session.diagnostics_mut().clone(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("replay report")
    );
    assert!(regions.iter().all(|region| region.count <= 16_384));
    assert_eq!(stats.rows, source.len());
    assert!(
        factor <= max_factor,
        "replication factor {factor} > {max_factor}"
    );
}
