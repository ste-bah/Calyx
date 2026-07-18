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

#[path = "cli_error_emit_fsv.rs"]
mod cli_error_emit_fsv;
#[path = "compact_cli_recovery.rs"]
mod compact_cli_recovery;
#[path = "compression_report_readback.rs"]
mod compression_report_readback;
#[path = "cx_list_include_slots_readback.rs"]
mod cx_list_include_slots_readback;
#[path = "dedup_check_readback.rs"]
mod dedup_check_readback;
#[path = "issue1108_build_info_fsv.rs"]
mod issue1108_build_info_fsv;
#[path = "living_concert_fsv.rs"]
mod living_concert_fsv;
#[path = "merkle_vault.rs"]
mod merkle_vault;
#[path = "recurrence_series_readback.rs"]
mod recurrence_series_readback;
#[path = "verify_chain_physical.rs"]
mod verify_chain_physical;
