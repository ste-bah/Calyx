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

#[path = "issue1369_online_head_contract_fsv.rs"]
mod issue1369_online_head_contract_fsv;
