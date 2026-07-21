//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "support/real_corpora.rs"]
mod __calyx_shared_support_real_corpora_rs;

#[path = "fsv_multi_scope.rs"]
mod fsv_multi_scope;
#[path = "fsv_recall_real_corpora.rs"]
mod fsv_recall_real_corpora;
#[path = "ph33_real_anchor_search_fsv.rs"]
mod ph33_real_anchor_search_fsv;
#[path = "ph33_real_ledger_answer_fsv.rs"]
mod ph33_real_ledger_answer_fsv;
