use super::*;

fn test_root(label: &str) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "calyx-part-gpu-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn test_params(n_cx: u64, dim: usize) -> PartitionBuildParams {
    PartitionBuildParams {
        n_cx,
        dim,
        n_regions: 8,
        seed: 1_515,
        sample: 256.min(n_cx as usize),
        chunk: 128,
        m_max: 8,
        ef_construction: 32,
        region_build_parallelism: 1,
        final_assignment_probe: 8,
        final_assignment_cap: Some(128),
        balance_cap: Some(2_048),
        assignment_boundary_epsilon: 0.10,
        assignment_max_replication: 1,
        assignment_rng_rule: true,
        assignment_rng_factor: 1.0,
    }
}

#[test]
fn gpu_cluster_plan_adds_cap_headroom_without_changing_cpu_shape() {
    assert!(gpu::has_balance_headroom(19_533, 100_000_000, 8_192));
    assert!(!gpu::has_balance_headroom(32, 8_192, 64));
    assert_eq!(
        gpu::initial_cluster_count(
            DiskAnnBuildBackend::CuvsCagra,
            1_000_000,
            128,
            Some(4_096),
            200_000,
        ),
        392
    );
    assert_eq!(
        gpu::initial_cluster_count(
            DiskAnnBuildBackend::CpuVamana,
            1_000_000,
            128,
            Some(4_096),
            200_000,
        ),
        128
    );
    assert_eq!(
        gpu::initial_cluster_count(
            DiskAnnBuildBackend::CuvsCagra,
            1_000_000,
            128,
            None,
            200_000,
        ),
        128
    );
}

#[test]
fn bounded_assignment_counts_the_sorted_epsilon_tail() {
    let primary_counts = vec![0; 4];
    let stored_counts = vec![0; 4];
    let candidates = vec![(0, 1.0), (1, 1.1), (2, 2.0), (3, 3.0)];
    let mut stats = ClosureAssignmentStats::default();
    let mut selected = vec![usize::MAX];

    assert!(assignment::choose_bounded_regions(
        &primary_counts,
        &stored_counts,
        10,
        10,
        &candidates,
        None,
        0.1,
        4,
        3,
        None,
        &mut stats,
        &mut selected,
    ));
    assert_eq!(selected, [0, 1]);
    assert_eq!(stats.rows, 1);
    assert_eq!(stats.epsilon_filtered, 2);
    assert_eq!(stats.replicas_stored, 1);
    assert_eq!(stats.replica_histogram, [0, 1]);
}

#[test]
fn primary_capacity_probe_can_select_beyond_a_full_requested_prefix() {
    let primary_counts = vec![1, 1, 0, 0];
    let stored_counts = vec![0; 4];
    let candidates = vec![(0, 1.0), (1, 1.1), (2, 1.2), (3, 1.3)];
    let mut stats = ClosureAssignmentStats::default();
    let mut selected = Vec::new();

    assert!(!assignment::choose_bounded_regions(
        &primary_counts,
        &stored_counts,
        1,
        2,
        &candidates[..2],
        None,
        0.1,
        1,
        0,
        None,
        &mut stats,
        &mut selected,
    ));
    assert_eq!(stats.rows, 0);
    assert!(assignment::choose_bounded_regions(
        &primary_counts,
        &stored_counts,
        1,
        2,
        &candidates,
        None,
        0.1,
        1,
        0,
        None,
        &mut stats,
        &mut selected,
    ));
    assert_eq!(selected, [2]);
    assert_eq!(stats.rows, 1);
}

#[test]
fn capacity_displacement_does_not_widen_gpu_boundary_radius() {
    let primary_counts = vec![1, 0, 0];
    let stored_counts = primary_counts.clone();
    let candidates = vec![(0, 1.0), (1, 1.3), (2, 1.4)];
    let mut stats = ClosureAssignmentStats::default();
    let mut selected = Vec::new();

    assert!(assignment::choose_bounded_regions(
        &primary_counts,
        &stored_counts,
        1,
        1,
        &candidates,
        Some(candidates[0].1),
        0.1,
        2,
        1,
        None,
        &mut stats,
        &mut selected,
    ));
    assert_eq!(selected, [1]);
    assert_eq!(stats.replicas_stored, 0);
}

#[cfg(not(sextant_cuvs))]
#[test]
fn cagra_partition_build_refuses_cpu_fallback_before_mutation() {
    let root = test_root("strict-no-cuvs");
    let error = build_partitioned_vault_with_backend(
        &root,
        test_params(128, 16),
        DiskAnnBuildBackend::CuvsCagra,
    )
    .expect_err("CAGRA partition build must fail closed without cuVS");

    assert!(error.message.contains("strict CUDA partition execution"));
    assert!(error.message.contains("refusing CPU centroid training"));
    assert!(error.message.contains("whole-corpus scans"));
    assert!(
        !root.exists(),
        "strict failure must precede filesystem mutation"
    );
}

#[cfg(sextant_cuvs)]
mod cuda {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use super::*;
    use crate::index::SpannCentroidIndex;

    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct NonFiniteSource {
        rows: Vec<Vec<f32>>,
    }

    impl VectorSource for NonFiniteSource {
        fn dim(&self) -> usize {
            self.rows[0].len()
        }

        fn len(&self) -> u64 {
            self.rows.len() as u64
        }

        fn row(&self, idx: u64) -> Vec<f32> {
            self.rows[idx as usize].clone()
        }
    }

    #[test]
    fn resident_cuda_build_preserves_partition_and_recall_contracts() {
        let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let cpu_root = test_root("cpu-reference");
        let gpu_root = test_root("cuda-resident");
        let repeat_root = test_root("cuda-repeat");
        let params = test_params(512, 16);

        let cpu =
            build_partitioned_vault_with_backend(&cpu_root, params, DiskAnnBuildBackend::CpuVamana)
                .expect("CPU reference build");
        let gpu =
            build_partitioned_vault_with_backend(&gpu_root, params, DiskAnnBuildBackend::CuvsCagra)
                .expect("CUDA partition build");
        let repeat = build_partitioned_vault_with_backend(
            &repeat_root,
            params,
            DiskAnnBuildBackend::CuvsCagra,
        )
        .expect("repeat CUDA partition build");

        assert_manifest_contract(&cpu, params.n_cx, 128);
        assert_manifest_contract(&gpu, params.n_cx, 128);
        assert_manifest_contract(&repeat, params.n_cx, 128);
        assert!(cpu.partition_build_diagnostics.is_none());
        assert_resident_diagnostics(&gpu, params.n_cx);
        assert_resident_diagnostics(&repeat, params.n_cx);

        let centroid_delta = max_centroid_delta(&gpu_root, &repeat_root);
        assert!(
            centroid_delta <= 5e-4,
            "repeat CUDA centroid max delta {centroid_delta} exceeds 5e-4"
        );
        let routing_agreement = assignment_agreement(&gpu_root, &gpu, &repeat_root, &repeat);
        assert!(
            routing_agreement >= 0.99,
            "repeat CUDA primary-routing agreement {routing_agreement} < 0.99"
        );

        let cpu_recall = recall_at_10(&cpu_root, params);
        let gpu_recall = recall_at_10(&gpu_root, params);
        assert!(
            cpu_recall >= 0.85,
            "CPU reference recall@10 {cpu_recall} < 0.85"
        );
        assert!(
            gpu_recall >= 0.85,
            "CUDA build recall@10 {gpu_recall} < 0.85"
        );
        assert!(
            (cpu_recall - gpu_recall).abs() <= 0.10,
            "CPU/CUDA recall@10 delta exceeds 0.10: {cpu_recall} vs {gpu_recall}"
        );

        for root in [&cpu_root, &gpu_root, &repeat_root] {
            std::fs::remove_dir_all(root).expect("remove test vault");
        }
    }

    #[test]
    fn batched_cuda_balance_enforces_cap_in_few_rounds() {
        let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let root = test_root("cuda-batched-balance");
        let mut params = test_params(8_192, 16);
        params.sample = 32;
        params.chunk = 8_192;
        params.balance_cap = Some(64);
        params.final_assignment_cap = Some(128);

        let manifest =
            build_partitioned_vault_with_backend(&root, params, DiskAnnBuildBackend::CuvsCagra)
                .expect("batched CUDA balance build");
        assert_manifest_contract(&manifest, params.n_cx, 128);
        let diagnostics = manifest
            .partition_build_diagnostics
            .as_ref()
            .expect("CUDA diagnostics");
        assert_eq!(
            manifest.provisional_assignment_routing,
            "cuda-cuvs-bruteforce-l2"
        );
        assert!(diagnostics.kmeans_calls > 1, "balance did not run");
        assert!(
            diagnostics.kmeans_calls <= 9,
            "balance used {} k-means calls",
            diagnostics.kmeans_calls
        );
        std::fs::remove_dir_all(root).expect("remove test vault");
    }

    #[test]
    fn resident_cuda_rejects_non_finite_corpus_before_mutation() {
        let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let root = test_root("non-finite");
        let mut rows = vec![vec![0.0; 16]; 32];
        rows[17][4] = f32::NAN;
        let source = NonFiniteSource { rows };
        let error = build_partitioned_vault_from_source_with_backend(
            &root,
            &source,
            test_params(32, 16),
            DiskAnnBuildBackend::CuvsCagra,
        )
        .expect_err("non-finite corpus must be rejected");

        assert!(error.message.contains("non-finite"), "{error:?}");
        assert!(
            !root.exists(),
            "corpus validation must precede filesystem mutation"
        );
    }

    #[test]
    #[ignore = "requires CALYX_PARTITION_GPU_RESIDENT_MIB=0 and CALYX_PARTITION_GPU_CHUNK_MIB=1"]
    fn streaming_cuda_skips_reupload_when_initial_plan_has_headroom() {
        let _guard = GPU_TEST_LOCK.lock().expect("GPU test lock");
        assert_eq!(
            std::env::var("CALYX_PARTITION_GPU_RESIDENT_MIB").as_deref(),
            Ok("0")
        );
        assert_eq!(
            std::env::var("CALYX_PARTITION_GPU_CHUNK_MIB").as_deref(),
            Ok("1")
        );
        let root = test_root("cuda-streaming");
        let mut params = test_params(8_192, 64);
        params.sample = 512;
        params.chunk = 8_192;
        params.final_assignment_cap = Some(2_048);

        let manifest =
            build_partitioned_vault_with_backend(&root, params, DiskAnnBuildBackend::CuvsCagra)
                .expect("streaming CUDA build");
        assert_manifest_contract(&manifest, params.n_cx, 2_048);
        let diagnostics = manifest
            .partition_build_diagnostics
            .as_ref()
            .expect("CUDA diagnostics");
        assert!(!diagnostics.resident_corpus);
        assert!(!diagnostics.resident_reused_across_scans);
        assert_eq!(diagnostics.chunk_rows, 4_096);
        assert_eq!(diagnostics.corpus_passes, 1);
        assert_eq!(diagnostics.corpus_uploads, 2);
        assert_eq!(diagnostics.rows_uploaded, 8_192);
        assert_eq!(
            manifest.provisional_assignment_routing,
            "skipped-cap-headroom"
        );
        std::fs::remove_dir_all(root).expect("remove test vault");
    }

    fn assert_manifest_contract(manifest: &PartitionedManifest, rows: u64, cap: usize) {
        assert_eq!(manifest.n_cx, rows);
        assert_eq!(manifest.stored_region_members, rows as usize);
        assert_eq!(manifest.final_assignment_cap, Some(cap));
        assert_eq!(manifest.final_assignment_max_replication, 1);
        assert!(manifest.regions.iter().all(|region| region.count <= cap));
        let closure = manifest
            .final_assignment_closure
            .as_ref()
            .expect("closure diagnostics");
        assert_eq!(closure.rows, rows);
        assert_eq!(closure.replicas_stored, 0);
        assert_eq!(closure.replication_factor(), 1.0);
    }

    fn assert_resident_diagnostics(manifest: &PartitionedManifest, rows: u64) {
        let diagnostics = manifest
            .partition_build_diagnostics
            .as_ref()
            .expect("CUDA diagnostics");
        assert_eq!(diagnostics.backend, "cuvs-balanced-kmeans-bruteforce-v1");
        assert!(diagnostics.strict_gpu_required);
        assert_eq!(
            manifest.provisional_assignment_routing,
            "skipped-cap-headroom"
        );
        assert!(diagnostics.resident_corpus);
        assert!(!diagnostics.resident_reused_across_scans);
        assert_eq!(diagnostics.kmeans_calls, 1);
        assert_eq!(diagnostics.routing_calls, 1);
        assert_eq!(diagnostics.corpus_passes, 1);
        assert_eq!(diagnostics.corpus_uploads, 1);
        assert_eq!(diagnostics.rows_uploaded, rows);
        assert!(diagnostics.h2d_transfers >= 3);
        assert!(diagnostics.d2h_transfers >= 3);
        assert!(diagnostics.peak_device_bytes > 0);
        assert!(diagnostics.peak_pinned_host_bytes > 0);
        assert!(diagnostics.centroid_training_us > 0);
        assert_eq!(diagnostics.provisional_assignment_us, 0);
        assert!(diagnostics.final_assignment_us > 0);
        assert!(diagnostics.pre_region_build_us > 0);
        assert!(diagnostics.end_to_end_build_us >= diagnostics.pre_region_build_us);
    }

    fn max_centroid_delta(left: &std::path::Path, right: &std::path::Path) -> f32 {
        let left = SpannCentroidIndex::open(left.join(CENTROID_DIR)).expect("left centroids");
        let right = SpannCentroidIndex::open(right.join(CENTROID_DIR)).expect("right centroids");
        assert_eq!(left.centroid_count(), right.centroid_count());
        left.centroids()
            .iter()
            .flatten()
            .zip(right.centroids().iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn assignment_agreement(
        left_root: &std::path::Path,
        left: &PartitionedManifest,
        right_root: &std::path::Path,
        right: &PartitionedManifest,
    ) -> f32 {
        let left = primary_assignments(left_root, left);
        let right = primary_assignments(right_root, right);
        assert_eq!(left.len(), right.len());
        left.iter().zip(&right).filter(|(a, b)| a == b).count() as f32 / left.len() as f32
    }

    fn primary_assignments(root: &std::path::Path, manifest: &PartitionedManifest) -> Vec<u32> {
        let mut assignments = vec![u32::MAX; manifest.n_cx as usize];
        for region in &manifest.regions {
            for row in assignment::read_ids(&root.join(&region.ids_rel)).expect("region ids") {
                let slot = &mut assignments[row as usize];
                assert_eq!(*slot, u32::MAX, "row stored more than once");
                *slot = region.id;
            }
        }
        assert!(assignments.iter().all(|&region| region != u32::MAX));
        assignments
    }

    fn recall_at_10(root: &std::path::Path, params: PartitionBuildParams) -> f32 {
        let search = PartitionedSearch::open(root).expect("open partitioned search");
        let mut found = 0usize;
        let mut expected = 0usize;
        for sample in 0..32u64 {
            let row = (sample * 17) % params.n_cx;
            let query = gen_row(params.seed, row, params.dim);
            let truth = exact_topk(&query, params, 10);
            let actual: BTreeSet<u64> = search
                .search(&query, 10, search.manifest().n_regions, 128)
                .expect("partitioned search")
                .into_iter()
                .map(|(row, _)| row)
                .collect();
            found += truth.iter().filter(|row| actual.contains(row)).count();
            expected += truth.len();
        }
        found as f32 / expected as f32
    }

    fn exact_topk(query: &[f32], params: PartitionBuildParams, k: usize) -> Vec<u64> {
        let mut rows: Vec<(u64, f32)> = (0..params.n_cx)
            .map(|row| {
                let vector = gen_row(params.seed, row, params.dim);
                let distance = vector
                    .iter()
                    .zip(query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (row, distance)
            })
            .collect();
        rows.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        rows.into_iter().take(k).map(|(row, _)| row).collect()
    }
}
