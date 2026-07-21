//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "reproduce_support/mod.rs"]
mod __calyx_shared_reproduce_support_mod_rs;

#[path = "appender_fsv.rs"]
mod appender_fsv;
#[path = "audit_tests.rs"]
mod audit_tests;
#[path = "checkpoint_tests.rs"]
mod checkpoint_tests;
#[path = "merkle_tests.rs"]
mod merkle_tests;
#[path = "reproduce_anchor_fsv.rs"]
mod reproduce_anchor_fsv;
#[path = "reproduce_fusion_fsv.rs"]
mod reproduce_fusion_fsv;
#[path = "reproduce_fusion_tests.rs"]
mod reproduce_fusion_tests;
#[path = "reproduce_tests.rs"]
mod reproduce_tests;
