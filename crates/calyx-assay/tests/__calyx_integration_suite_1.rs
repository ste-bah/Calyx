//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "ph52_signal_support/mod.rs"]
mod __calyx_shared_ph52_signal_support_mod_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "ph52_support/mod.rs"]
mod __calyx_shared_ph52_support_mod_rs;

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "stage5_helpers/mod.rs"]
mod __calyx_shared_stage5_helpers_mod_rs;

#[path = "assay_scope_fsv.rs"]
mod assay_scope_fsv;
#[path = "assay_trust_fsv.rs"]
mod assay_trust_fsv;
#[path = "aster_materialization_gate_fsv.rs"]
mod aster_materialization_gate_fsv;
#[path = "bootstrap_ci_fsv.rs"]
mod bootstrap_ci_fsv;
#[path = "formula_coverage_fsv.rs"]
mod formula_coverage_fsv;
#[path = "granger_fsv.rs"]
mod granger_fsv;
#[path = "hsic_fsv.rs"]
mod hsic_fsv;
#[path = "issue061_variable_lag_fsv.rs"]
mod issue061_variable_lag_fsv;
#[path = "issue062_cross_correlation_fsv.rs"]
mod issue062_cross_correlation_fsv;
#[path = "issue063_conditional_mi_fsv.rs"]
mod issue063_conditional_mi_fsv;
#[path = "issue064_ccm_fsv.rs"]
mod issue064_ccm_fsv;
#[path = "issue066_hawkes_fsv.rs"]
mod issue066_hawkes_fsv;
#[path = "issue1312_tc_subsample_fsv.rs"]
mod issue1312_tc_subsample_fsv;
#[path = "issue1313_pc_stable_fsv.rs"]
mod issue1313_pc_stable_fsv;
#[path = "ph52_bayesian_tests.rs"]
mod ph52_bayesian_tests;
#[path = "power_gate_fsv.rs"]
mod power_gate_fsv;
#[path = "real_labeled_classification_fsv.rs"]
mod real_labeled_classification_fsv;
#[path = "recurrence_anchor_fsv.rs"]
mod recurrence_anchor_fsv;
#[path = "recurrence_hazard_fsv.rs"]
mod recurrence_hazard_fsv;
