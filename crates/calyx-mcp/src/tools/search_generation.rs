use std::path::Path;

use calyx_aster::vault::AsterVault;
use calyx_core::{CalyxError, VaultStore};
use calyx_registry::VaultPanelState;

use crate::server::{ToolError, ToolResult};

pub(super) fn publish_search_generation(
    vault_dir: &Path,
    vault: &AsterVault,
    state: &VaultPanelState,
) -> ToolResult<()> {
    if let Ok(indexes) = calyx_search::PersistedSearchIndexes::open(vault_dir) {
        let snapshot = vault.snapshot();
        let reusable = indexes
            .ensure_fresh_at_snapshot(snapshot, vault.derived_content_seq().min(snapshot))
            .and_then(|_| indexes.ensure_search_bounded())
            .and_then(|_| indexes.generation().map(|_| ()));
        if reusable.is_ok() {
            return Ok(());
        }
    }
    calyx_search::rebuild_for_vault_with_panel_state(vault_dir, vault, state).map_err(search_error)
}

fn search_error(error: calyx_search::SearchError) -> ToolError {
    match error {
        calyx_search::SearchError::Calyx(error) => error.into(),
        calyx_search::SearchError::Usage(message) => ToolError::invalid_params(message),
        calyx_search::SearchError::Io(message) => {
            CalyxError::stale_derived(format!("publish persistent search generation: {message}"))
                .into()
        }
    }
}
