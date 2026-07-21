//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "stage5_helpers/mod.rs"]
mod __calyx_shared_stage5_helpers_mod_rs;

#[path = "stage5_fsv.rs"]
mod stage5_fsv;
