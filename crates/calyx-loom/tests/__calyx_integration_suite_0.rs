//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "blind_spot_calibration_fsv.rs"]
mod blind_spot_calibration_fsv;
#[path = "cross_term_fail_closed.rs"]
mod cross_term_fail_closed;
#[path = "issue573_reactive_subscription_tests.rs"]
mod issue573_reactive_subscription_tests;
#[path = "issue755_reactive_durable_fsv.rs"]
mod issue755_reactive_durable_fsv;
#[path = "periodic_recall_bounded_fsv.rs"]
mod periodic_recall_bounded_fsv;
#[path = "recurrence_cross_terms.rs"]
mod recurrence_cross_terms;
#[path = "recurrence_cross_terms_fsv.rs"]
mod recurrence_cross_terms_fsv;
#[path = "rolled_recurrence.rs"]
mod rolled_recurrence;
