//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "novelty_recurrence_support/mod.rs"]
mod __calyx_shared_novelty_recurrence_support_mod_rs;

#[path = "derive_required.rs"]
mod derive_required;
#[path = "generate.rs"]
mod generate;
#[path = "guard_health_serde.rs"]
mod guard_health_serde;
#[path = "guard_kofn.rs"]
mod guard_kofn;
#[path = "guard_no_flatten.rs"]
mod guard_no_flatten;
#[path = "guard_ph37_fsv.rs"]
mod guard_ph37_fsv;
#[path = "identity_profile.rs"]
mod identity_profile;
#[path = "injection_guard_runtime_fsv.rs"]
mod injection_guard_runtime_fsv;
#[path = "ledger_provenance.rs"]
mod ledger_provenance;
#[path = "novelty_recurrence.rs"]
mod novelty_recurrence;
#[path = "novelty_recurrence_fsv.rs"]
mod novelty_recurrence_fsv;
#[path = "polis_civic_fsv.rs"]
mod polis_civic_fsv;
