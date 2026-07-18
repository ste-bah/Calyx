//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "reranker_support/mod.rs"]
mod __calyx_shared_reranker_support_mod_rs;

#[path = "sextant_support/mod.rs"]
mod __calyx_shared_sextant_support_mod_rs;

#[path = "diskann_concat.rs"]
mod diskann_concat;
#[path = "diskann_dual.rs"]
mod diskann_dual;
#[path = "diskann_graph.rs"]
mod diskann_graph;
#[path = "diskann_pq.rs"]
mod diskann_pq;
#[path = "e3_periodic_scoring_fsv.rs"]
mod e3_periodic_scoring_fsv;
#[path = "funnel.rs"]
mod funnel;
#[path = "guarded_inert.rs"]
mod guarded_inert;
#[path = "guarded_multislot_readback.rs"]
mod guarded_multislot_readback;
#[path = "hnsw_recall.rs"]
mod hnsw_recall;
#[path = "issue723_multimodal_rrf_fsv.rs"]
mod issue723_multimodal_rrf_fsv;
#[path = "ph58_ann_gc_fsv.rs"]
mod ph58_ann_gc_fsv;
#[path = "ph63_navigation_fsv.rs"]
mod ph63_navigation_fsv;
#[path = "pipeline_recall_headroom_fsv.rs"]
mod pipeline_recall_headroom_fsv;
#[path = "query_admission_fsv.rs"]
mod query_admission_fsv;
#[path = "query_validation.rs"]
mod query_validation;
#[path = "recurrence_boost_fsv.rs"]
mod recurrence_boost_fsv;
#[path = "sparse_vector_readback_fsv.rs"]
mod sparse_vector_readback_fsv;
#[path = "stage4_real_qrels_fsv.rs"]
mod stage4_real_qrels_fsv;
#[path = "temporal_search_fsv.rs"]
mod temporal_search_fsv;
#[path = "temporal_window_fsv.rs"]
mod temporal_window_fsv;
#[path = "temporal_window_recall.rs"]
mod temporal_window_recall;
