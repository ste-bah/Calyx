//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "fsv_support/mod.rs"]
mod __calyx_shared_fsv_support_mod_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "support/fsv_bad_change.rs"]
mod __calyx_shared_support_fsv_bad_change_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "support/propose_lens.rs"]
mod __calyx_shared_support_propose_lens_rs;

#[path = "ab_runner_fsv.rs"]
mod ab_runner_fsv;
#[path = "bandit.rs"]
mod bandit;
#[path = "budget.rs"]
mod budget;
#[path = "deficit_localize.rs"]
mod deficit_localize;
#[path = "degrade.rs"]
mod degrade;
#[path = "differentiation_gate.rs"]
mod differentiation_gate;
#[path = "frozen_guard.rs"]
mod frozen_guard;
#[path = "frozen_guard_fsv.rs"]
mod frozen_guard_fsv;
#[path = "fsv_corrupt_rebuild.rs"]
mod fsv_corrupt_rebuild;
#[path = "fsv_j_growth.rs"]
mod fsv_j_growth;
#[path = "growth_curve.rs"]
mod growth_curve;
#[path = "growth_curve_fsv.rs"]
mod growth_curve_fsv;
#[path = "issue1246_online_learning_fsv.rs"]
mod issue1246_online_learning_fsv;
#[path = "issue392_fsv.rs"]
mod issue392_fsv;
#[path = "j_composite.rs"]
mod j_composite;
#[path = "ledger_anneal_fsv.rs"]
mod ledger_anneal_fsv;
#[path = "mistake_log_fsv.rs"]
mod mistake_log_fsv;
#[path = "online_head.rs"]
mod online_head;
#[path = "operator_synth.rs"]
mod operator_synth;
#[path = "propose_lens.rs"]
mod propose_lens;
#[path = "rebuild.rs"]
mod rebuild;
#[path = "record_outcome.rs"]
mod record_outcome;
#[path = "regression_assert.rs"]
mod regression_assert;
#[path = "regression_assert_fsv.rs"]
mod regression_assert_fsv;
#[path = "regression_guard.rs"]
mod regression_guard;
#[path = "replay_buffer_fsv.rs"]
mod replay_buffer_fsv;
#[path = "rollback_fsv.rs"]
mod rollback_fsv;
#[path = "scope_forge.rs"]
mod scope_forge;
#[path = "scope_loom_fsv.rs"]
mod scope_loom_fsv;
#[path = "scope_storage.rs"]
mod scope_storage;
#[path = "scope_storage_fsv.rs"]
mod scope_storage_fsv;
#[path = "shadow.rs"]
mod shadow;
#[path = "shadow_fsv.rs"]
mod shadow_fsv;
#[path = "triggers.rs"]
mod triggers;
#[path = "tripwire.rs"]
mod tripwire;
