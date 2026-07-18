//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "support/fsv_io.rs"]
mod __calyx_shared_support_fsv_io_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "support/mod.rs"]
mod __calyx_shared_support_mod_rs;

#[path = "verify_chain.rs"]
mod verify_chain;
