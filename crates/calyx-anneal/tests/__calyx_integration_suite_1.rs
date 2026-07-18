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

#[path = "ab_runner.rs"]
mod ab_runner;
#[path = "admission_record.rs"]
mod admission_record;
#[path = "artifact_shadow.rs"]
mod artifact_shadow;
#[path = "bandit_fsv.rs"]
mod bandit_fsv;
#[path = "budget_fsv.rs"]
mod budget_fsv;
#[path = "candidate_synth.rs"]
mod candidate_synth;
#[path = "fsv_bad_change.rs"]
mod fsv_bad_change;
#[path = "fsv_lens_proposal.rs"]
mod fsv_lens_proposal;
#[path = "fsv_mistake_closure.rs"]
mod fsv_mistake_closure;
#[path = "fsv_soak.rs"]
mod fsv_soak;
#[path = "goodhart.rs"]
mod goodhart;
#[path = "gradient.rs"]
mod gradient;
#[path = "intelligence_report.rs"]
mod intelligence_report;
#[path = "issue486_janitor.rs"]
mod issue486_janitor;
#[path = "issue486_janitor_fsv.rs"]
mod issue486_janitor_fsv;
#[path = "issue791_conversion_flywheel_fsv.rs"]
mod issue791_conversion_flywheel_fsv;
#[path = "ledger_anneal.rs"]
mod ledger_anneal;
#[path = "mistake_log.rs"]
mod mistake_log;
#[path = "online_head_fsv.rs"]
mod online_head_fsv;
#[path = "operator_synth_fsv.rs"]
mod operator_synth_fsv;
#[path = "rebuild_fsv.rs"]
mod rebuild_fsv;
#[path = "recalibrate.rs"]
mod recalibrate;
#[path = "recalibrate_fsv.rs"]
mod recalibrate_fsv;
#[path = "recurrence_schedule.rs"]
mod recurrence_schedule;
#[path = "replay_buffer.rs"]
mod replay_buffer;
#[path = "restore.rs"]
mod restore;
#[path = "restore_fsv.rs"]
mod restore_fsv;
#[path = "rollback.rs"]
mod rollback;
#[path = "scope_forge_fsv.rs"]
mod scope_forge_fsv;
#[path = "scope_index.rs"]
mod scope_index;
#[path = "scope_index_fsv.rs"]
mod scope_index_fsv;
#[path = "scope_loom.rs"]
mod scope_loom;
#[path = "soak_harness.rs"]
mod soak_harness;
#[path = "tripwire_fsv.rs"]
mod tripwire_fsv;
