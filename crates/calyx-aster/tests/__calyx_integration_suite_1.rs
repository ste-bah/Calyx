//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[path = "fsv_support/mod.rs"]
mod __calyx_shared_fsv_support_mod_rs;

#[path = "compression_ratio.rs"]
mod compression_ratio;
#[path = "inverted_index_fsv.rs"]
mod inverted_index_fsv;
#[path = "issue1213_weighted_csr_fsv.rs"]
mod issue1213_weighted_csr_fsv;
#[path = "issue460_atomic_index_write_fsv.rs"]
mod issue460_atomic_index_write_fsv;
#[path = "issue461_index_rebuild_fsv.rs"]
mod issue461_index_rebuild_fsv;
#[path = "issue575_retention_horizon_tests.rs"]
mod issue575_retention_horizon_tests;
#[path = "issue597_real_restic_crypto_shred.rs"]
mod issue597_real_restic_crypto_shred;
#[path = "issue750_u64_index_fsv.rs"]
mod issue750_u64_index_fsv;
#[path = "issue815_nonce_envelope_fsv.rs"]
mod issue815_nonce_envelope_fsv;
#[path = "issue816_ledger_head_anchor_fsv.rs"]
mod issue816_ledger_head_anchor_fsv;
#[path = "ph53_fsv.rs"]
mod ph53_fsv;
#[path = "ph54_fsv.rs"]
mod ph54_fsv;
#[path = "ph58_wal_recycler_fsv.rs"]
mod ph58_wal_recycler_fsv;
#[path = "plain_column_fsv.rs"]
mod plain_column_fsv;
