//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "issue790_vector_compression_fsv/support.rs"]
mod __calyx_shared_issue790_vector_compression_fsv_support_rs;

#[path = "backfill_atomic_fsv.rs"]
mod backfill_atomic_fsv;
#[path = "capability_gate_fsv.rs"]
mod capability_gate_fsv;
#[path = "colbert_manifest.rs"]
mod colbert_manifest;
#[path = "embedder_zoo_fsv.rs"]
mod embedder_zoo_fsv;
#[path = "fastembed_manifest.rs"]
mod fastembed_manifest;
#[path = "hot_swap_fsv.rs"]
mod hot_swap_fsv;
#[path = "hot_swap_registered_fsv.rs"]
mod hot_swap_registered_fsv;
#[path = "input_validation_fsv.rs"]
mod input_validation_fsv;
#[path = "issue1489_contract_parity_fsv.rs"]
mod issue1489_contract_parity_fsv;
#[path = "issue752_vault_panel_persistence_fsv.rs"]
mod issue752_vault_panel_persistence_fsv;
#[path = "issue788_multimodal_lens_pack_fsv.rs"]
mod issue788_multimodal_lens_pack_fsv;
#[path = "issue789_default_domain_panels_fsv.rs"]
mod issue789_default_domain_panels_fsv;
#[path = "issue790_vector_compression_fsv.rs"]
mod issue790_vector_compression_fsv;
#[path = "issue925_raw_f32_envelope_fsv.rs"]
mod issue925_raw_f32_envelope_fsv;
#[path = "issue934_mxfp_no_fallback_fsv.rs"]
mod issue934_mxfp_no_fallback_fsv;
#[path = "issue990_builtin_template_registry_fsv.rs"]
mod issue990_builtin_template_registry_fsv;
#[path = "panels_temporal_fsv.rs"]
mod panels_temporal_fsv;
#[path = "ph57_ingest_microbatch_fsv.rs"]
mod ph57_ingest_microbatch_fsv;
#[path = "registry_sextant_integration_fsv.rs"]
mod registry_sextant_integration_fsv;
