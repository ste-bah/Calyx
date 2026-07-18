//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "calibrate_unit.rs"]
mod calibrate_unit;
#[path = "calibration_slot_validation.rs"]
mod calibration_slot_validation;
#[path = "drift_monitor.rs"]
mod drift_monitor;
#[path = "drift_retry.rs"]
mod drift_retry;
#[path = "drift_uncalibrated.rs"]
mod drift_uncalibrated;
#[path = "guard_inert.rs"]
mod guard_inert;
#[path = "guard_provisional.rs"]
mod guard_provisional;
#[path = "guard_query.rs"]
mod guard_query;
#[allow(
    dead_code,
    reason = "legacy zero-test helper target retained for compile coverage"
)]
#[path = "identity_fsv.rs"]
mod identity_fsv;
#[path = "novelty_handler.rs"]
mod novelty_handler;
#[path = "speaker_lens.rs"]
mod speaker_lens;
#[path = "style_lens.rs"]
mod style_lens;
#[path = "timestamp_units.rs"]
mod timestamp_units;
