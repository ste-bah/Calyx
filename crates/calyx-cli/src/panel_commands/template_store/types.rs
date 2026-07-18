use std::path::PathBuf;

use serde::Serialize;

use super::ResidentSwapAttestation;
use super::SavedPanelTemplate;

#[derive(Clone, Debug)]
pub(in crate::panel_commands) struct TemplateLensProgress {
    pub phase: &'static str,
    pub ordinal: usize,
    pub total: usize,
    pub slot_key: String,
    pub lens_name: String,
    pub lens_id: String,
    pub runtime_lens_id: Option<String>,
    pub runtime: String,
    pub modality: String,
    pub shape: String,
    pub placement: String,
    pub manifest: String,
}

#[derive(Clone, Debug)]
pub(in crate::panel_commands) struct TemplateStore {
    pub(super) root: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub(in crate::panel_commands) struct TemplateSave {
    pub template_id: String,
    pub object_path: PathBuf,
    pub index_path: PathBuf,
    pub template: SavedPanelTemplate,
}

#[derive(Clone, Debug, Serialize)]
pub(in crate::panel_commands) struct TemplateSummary {
    pub name: String,
    pub active_template_id: String,
    pub version: u32,
    pub object_schema_version: u16,
    pub migration_required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refresh_command: Option<String>,
    pub content_lens_count: usize,
    pub time_control_count: usize,
    pub has_ensemble_card: bool,
    pub a37_gate_eligible: bool,
    pub a37_status: String,
    pub object_path: String,
}

#[derive(Clone, Debug, Serialize)]
pub(in crate::panel_commands) struct TemplateSwap {
    pub template_id: String,
    pub template_name: String,
    pub vault: PathBuf,
    pub content_lens_count: usize,
    pub a37_gate_eligible: bool,
    pub a37_status: String,
    pub registered_lenses_added: usize,
    pub manifest_seq: u64,
    pub panel_ref: String,
    pub registry_ref: String,
    pub diff: calyx_registry::PanelDiff,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_attestation: Option<ResidentSwapAttestation>,
    pub readback_verified: bool,
}
