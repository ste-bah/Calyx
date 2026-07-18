use crate::error::{CliError, CliResult};

use super::SavedPanelTemplate;

pub(in crate::panel_commands) fn id_for_loaded(template: &SavedPanelTemplate) -> CliResult<String> {
    Ok(blake3::hash(&object_bytes(template)?).to_hex().to_string())
}

pub(in crate::panel_commands) fn object_bytes(template: &SavedPanelTemplate) -> CliResult<Vec<u8>> {
    serde_json::to_vec_pretty(template)
        .map_err(|error| CliError::runtime(format!("serialize template object: {error}")))
}
