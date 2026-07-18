//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "support/dedup_fsv_io.rs"]
mod __calyx_shared_support_dedup_fsv_io_rs;

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

#[path = "dedup_ingest_at_readback.rs"]
mod dedup_ingest_at_readback;
#[path = "dedup_invariants_readback.rs"]
mod dedup_invariants_readback;
#[path = "issue1243_grounded_summary_replay_fsv.rs"]
mod issue1243_grounded_summary_replay_fsv;
#[path = "ph36_fsv_integration.rs"]
mod ph36_fsv_integration;
#[path = "readme_assets_contract.rs"]
mod readme_assets_contract;
