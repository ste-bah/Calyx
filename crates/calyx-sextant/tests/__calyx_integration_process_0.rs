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

#[path = "embedded_scale_perf_fsv.rs"]
mod embedded_scale_perf_fsv;
#[path = "reranker_nonpersistence_fsv.rs"]
mod reranker_nonpersistence_fsv;
#[path = "reranker_search_fsv.rs"]
mod reranker_search_fsv;
#[path = "reranker_tei_nonpersistence_fsv.rs"]
mod reranker_tei_nonpersistence_fsv;
#[path = "stage4_fsv.rs"]
mod stage4_fsv;
