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

#[path = "advanced_math_fsv.rs"]
mod advanced_math_fsv;
#[path = "categorical_association_fsv.rs"]
mod categorical_association_fsv;
#[path = "coverage_scope.rs"]
mod coverage_scope;
#[path = "distance_correlation_fsv.rs"]
mod distance_correlation_fsv;
#[path = "estimator_contract_fsv.rs"]
mod estimator_contract_fsv;
#[path = "issue059_copula_tail_fsv.rs"]
mod issue059_copula_tail_fsv;
#[path = "issue067_point_process_cointensity_fsv.rs"]
mod issue067_point_process_cointensity_fsv;
#[path = "issue068_pc_stable_fsv.rs"]
mod issue068_pc_stable_fsv;
#[path = "issue069_partial_network_fsv.rs"]
mod issue069_partial_network_fsv;
#[path = "issue1313_pc_stable_order_tests.rs"]
mod issue1313_pc_stable_order_tests;
#[path = "ksg_mixed_discrete_fsv.rs"]
mod ksg_mixed_discrete_fsv;
#[path = "mic_fsv.rs"]
mod mic_fsv;
#[path = "panel_width_degradation_fsv.rs"]
mod panel_width_degradation_fsv;
#[path = "partial_correlation_fsv.rs"]
mod partial_correlation_fsv;
#[path = "ph42_exit_gate_fsv.rs"]
mod ph42_exit_gate_fsv;
#[path = "ph52_total_correlation_tests.rs"]
mod ph52_total_correlation_tests;
#[path = "ph52_transfer_entropy_tests.rs"]
mod ph52_transfer_entropy_tests;
#[path = "rank_correlation_fsv.rs"]
mod rank_correlation_fsv;
#[path = "recurrence_anchor.rs"]
mod recurrence_anchor;
