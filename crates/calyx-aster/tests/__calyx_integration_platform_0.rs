//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "fsv_support/mod.rs"]
mod __calyx_shared_fsv_support_mod_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "soak_ph58/serving.rs"]
mod __calyx_shared_soak_ph58_serving_rs;

#[path = "durable_manifest_assets.rs"]
mod durable_manifest_assets;
#[path = "issue472_mmap_fsv.rs"]
mod issue472_mmap_fsv;
#[path = "issue473_disk_pressure.rs"]
mod issue473_disk_pressure;
#[path = "panel_manifest_persistence_fsv.rs"]
mod panel_manifest_persistence_fsv;
#[path = "ph58_compaction_gc_fsv.rs"]
mod ph58_compaction_gc_fsv;
#[path = "ph58_mvcc_snapshot_gc_fsv.rs"]
mod ph58_mvcc_snapshot_gc_fsv;
#[path = "ph58_reader_leases_fsv.rs"]
mod ph58_reader_leases_fsv;
#[path = "ph58_tombstone_perf_gate.rs"]
mod ph58_tombstone_perf_gate;
#[path = "soak_ph56.rs"]
mod soak_ph56;
#[path = "soak_ph58.rs"]
mod soak_ph58;
