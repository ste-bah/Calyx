//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "fsv_support/mod.rs"]
mod __calyx_shared_fsv_support_mod_rs;

#[path = "btree_index_fsv.rs"]
mod btree_index_fsv;
#[path = "btree_query_fsv.rs"]
mod btree_query_fsv;
#[path = "dedup_manifest_fsv.rs"]
mod dedup_manifest_fsv;
#[path = "input_validation.rs"]
mod input_validation;
#[path = "issue1210_binary_csr_fsv.rs"]
mod issue1210_binary_csr_fsv;
#[path = "issue471_memtable_fsv.rs"]
mod issue471_memtable_fsv;
#[path = "issue485_orphan_panel_gc_fsv.rs"]
mod issue485_orphan_panel_gc_fsv;
#[path = "issue883_anchor_grounding_flag.rs"]
mod issue883_anchor_grounding_flag;
#[path = "ph60_integration.rs"]
mod ph60_integration;
#[path = "residency_fsv.rs"]
mod residency_fsv;
#[path = "slot_column_fsv.rs"]
mod slot_column_fsv;
#[path = "string_metadata_fsv.rs"]
mod string_metadata_fsv;
#[path = "temporal_manifest_fsv.rs"]
mod temporal_manifest_fsv;
#[path = "timetravel_fsv.rs"]
mod timetravel_fsv;
