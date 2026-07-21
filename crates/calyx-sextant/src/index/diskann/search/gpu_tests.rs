use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use calyx_core::{CxId, SlotId};

use super::{DiskAnnSearch, DiskAnnSearchParams};
use crate::index::diskann::{
    CagraPartitionRegion, CagraPartitionSearchRequest, CagraServingMetric, DiskAnnBuildBackend,
    DiskAnnBuildParams, cagra_dataset_sidecar_path, cagra_partitioned_search,
    cagra_serving_diagnostics, cagra_serving_region, cagra_sidecar_path,
};

#[test]
fn serialized_cagra_serves_batches_and_exact_device_filters() {
    let root = temp_root();
    let graph = root.join("missing-parent").join("graph.cda");
    let rows = (0..128_u32)
        .map(|id| {
            let mut vector = vec![0.0_f32; 8];
            vector[id as usize % 8] = 1.0;
            vector[(id as usize * 3 + 1) % 8] += id as f32 / 512.0;
            (cx(id), vector)
        })
        .collect::<Vec<_>>();
    let params = DiskAnnSearchParams {
        beamwidth: 64,
        ef_search: 128,
        rescore_k: 128,
        rescore_from_raw: false,
    };
    let built = DiskAnnSearch::build_with_backend(
        SlotId::new(0),
        &graph,
        &rows,
        DiskAnnBuildParams {
            dim: 8,
            m_max: 8,
            ef_construction: 32,
            alpha: 1.2,
        },
        None,
        params,
        DiskAnnBuildBackend::CuvsCagra,
    )
    .expect("CAGRA build");
    drop(built);
    let asset = cagra_sidecar_path(&graph);
    let dataset_asset = cagra_dataset_sidecar_path(&graph);
    assert!(fs::metadata(&asset).expect("CAGRA asset").len() > 0);
    assert!(
        fs::metadata(&dataset_asset)
            .expect("CUDA dataset asset")
            .len()
            > 0
    );

    let search = DiskAnnSearch::open_gpu_serving(
        SlotId::new(0),
        &graph,
        rows.iter().map(|(id, _)| *id).collect(),
        None,
        params,
    )
    .expect("GPU serving open");
    let queries = [rows[7].1.as_slice(), rows[41].1.as_slice()].concat();
    let batch = search
        .search_ids_batch(&queries, 2, 10, &params)
        .expect("CAGRA batch");
    assert!(batch[0].iter().any(|(id, _)| *id == 7));
    assert!(batch[1].iter().any(|(id, _)| *id == 41));

    let filtered = search
        .search_ids_filtered_cuda(&rows[7].1, 2, &params, &[7, 41])
        .expect("device-filtered exact search");
    assert_eq!(filtered[0].0, 7);
    assert!(filtered.iter().all(|(id, _)| [7, 41].contains(id)));
    let diagnostics = cagra_serving_diagnostics();
    assert!(diagnostics.cagra_kernel_launches >= 1);
    assert!(diagnostics.exact_filter_kernel_launches >= 1);
    assert_eq!(diagnostics.intermediate_readback_pairs, 0);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn partitioned_cuda_batches_regions_dedupes_and_reads_final_topk_only() {
    let root = temp_root();
    let first_graph = root.join("first").join("graph.cda");
    let second_graph = root.join("second").join("graph.cda");
    let params = DiskAnnSearchParams {
        beamwidth: 32,
        ef_search: 32,
        rescore_k: 32,
        rescore_from_raw: false,
    };
    build_region(
        &first_graph,
        &[(cx(0), vec![1.0, 0.0]), (cx(1), vec![0.0, 0.0])],
        params,
    );
    build_region(
        &second_graph,
        &[(cx(0), vec![1.0, 0.0]), (cx(1), vec![0.0, 1.0])],
        params,
    );
    let first_ids = [20_u64, 10];
    let second_ids = [20_u64, 30];
    let first_serving =
        cagra_serving_region(&first_graph, &first_ids, CagraServingMetric::RawL2, 2)
            .expect("first serving generation");
    let second_serving =
        cagra_serving_region(&second_graph, &second_ids, CagraServingMetric::RawL2, 2)
            .expect("second serving generation");
    let regions = [
        CagraPartitionRegion {
            serving: &first_serving,
            global_ids: &first_ids,
        },
        CagraPartitionRegion {
            serving: &second_serving,
            global_ids: &second_ids,
        },
    ];
    let before = cagra_serving_diagnostics();
    let hits = cagra_partitioned_search(CagraPartitionSearchRequest {
        metric: CagraServingMetric::RawL2,
        query: &[1.0, 0.0],
        k: 3,
        regions: &regions,
    })
    .expect("partitioned CUDA search");
    assert_eq!(
        hits.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        [20, 10, 30]
    );
    assert!(hits.iter().all(|(_, distance)| distance.is_finite()));
    let after = cagra_serving_diagnostics();
    assert!(after.partitioned_exact_kernel_launches > before.partitioned_exact_kernel_launches);
    assert!(after.partitioned_merge_kernel_launches > before.partitioned_merge_kernel_launches);
    assert!(after.partitioned_scratch_bytes > 0);
    assert!(after.partitioned_scratch_bytes <= after.partitioned_scratch_max_bytes);
    assert!(after.partitioned_i8_dataset_loads >= before.partitioned_i8_dataset_loads + 2);
    assert!(after.partitioned_pool_reserved_bytes > 0);
    assert!(after.partitioned_pool_reserved_max_bytes >= after.partitioned_pool_reserved_bytes);
    assert!(after.partitioned_pool_used_bytes > 0);
    assert!(after.partitioned_pool_used_max_bytes >= after.partitioned_pool_used_bytes);
    assert_eq!(after.intermediate_readback_pairs, 0);
    assert!(after.final_readback_pairs >= before.final_readback_pairs + 3);

    build_region(
        &first_graph,
        &[(cx(0), vec![2.0, 0.0]), (cx(1), vec![0.0, 0.0])],
        params,
    );
    let rebuilt_first =
        cagra_serving_region(&first_graph, &first_ids, CagraServingMetric::RawL2, 2)
            .expect("rebuilt serving generation");
    let rebuilt_regions = [
        CagraPartitionRegion {
            serving: &rebuilt_first,
            global_ids: &first_ids,
        },
        CagraPartitionRegion {
            serving: &second_serving,
            global_ids: &second_ids,
        },
    ];
    cagra_partitioned_search(CagraPartitionSearchRequest {
        metric: CagraServingMetric::RawL2,
        query: &[1.0, 0.0],
        k: 3,
        regions: &rebuilt_regions,
    })
    .expect("rebuilt generation search");
    let rebuilt = cagra_serving_diagnostics();
    assert!(rebuilt.cache_invalidations > after.cache_invalidations);

    let remapped_ids = [40_u64, 10];
    let remapped_first =
        cagra_serving_region(&first_graph, &remapped_ids, CagraServingMetric::RawL2, 2)
            .expect("remapped serving generation");
    let remapped_regions = [
        CagraPartitionRegion {
            serving: &remapped_first,
            global_ids: &remapped_ids,
        },
        CagraPartitionRegion {
            serving: &second_serving,
            global_ids: &second_ids,
        },
    ];
    let remapped = cagra_partitioned_search(CagraPartitionSearchRequest {
        metric: CagraServingMetric::RawL2,
        query: &[1.0, 0.0],
        k: 3,
        regions: &remapped_regions,
    })
    .expect("remapped generation search");
    assert_eq!(
        remapped.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        [20, 10, 40]
    );
    let remapped_diagnostics = cagra_serving_diagnostics();
    assert!(remapped_diagnostics.cache_invalidations > rebuilt.cache_invalidations);
    let _ = fs::remove_dir_all(root);
}

fn build_region(graph: &std::path::Path, rows: &[(CxId, Vec<f32>)], params: DiskAnnSearchParams) {
    let built = DiskAnnSearch::build_raw_l2_without_default_raw_sidecar_with_backend(
        SlotId::new(0),
        graph,
        rows,
        DiskAnnBuildParams {
            dim: 2,
            m_max: 1,
            ef_construction: 8,
            alpha: 1.2,
        },
        None,
        params,
        DiskAnnBuildBackend::CuvsCagra,
    )
    .expect("partition region build");
    drop(built);
}

fn cx(id: u32) -> CxId {
    let mut bytes = [0_u8; 16];
    bytes[12..].copy_from_slice(&id.to_be_bytes());
    CxId::from_bytes(bytes)
}

fn temp_root() -> PathBuf {
    let mut root = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    root.push(format!("calyx-issue1510-cagra-serving-{nanos}"));
    fs::create_dir_all(&root).expect("temp root");
    root
}
