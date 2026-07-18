use std::path::Path;

use calyx_registry::{Registry, load_vault_panel_state};

use crate::error::{CliError, CliResult};

pub(super) fn verify_persisted_swap(
    vault_dir: &Path,
    expected_panel: &calyx_core::Panel,
    expected_registry: &Registry,
    expected_panel_ref: &calyx_aster::manifest::ImmutableRef,
) -> CliResult {
    let readback = load_vault_panel_state(vault_dir)?;
    if &readback.panel != expected_panel {
        return Err(CliError::runtime(format!(
            "panel template swap physical panel readback mismatch at {}",
            vault_dir.display()
        )));
    }
    if readback.registry.lens_snapshots() != expected_registry.lens_snapshots() {
        return Err(CliError::runtime(format!(
            "panel template swap physical registry readback mismatch at {}",
            vault_dir.display()
        )));
    }
    let snapshot = readback.registry_snapshot.ok_or_else(|| {
        CliError::runtime(format!(
            "panel template swap physical registry snapshot is absent at {}",
            vault_dir.display()
        ))
    })?;
    if &snapshot.panel_ref != expected_panel_ref {
        return Err(CliError::runtime(format!(
            "panel template swap physical registry snapshot points at a different panel at {}",
            vault_dir.display()
        )));
    }
    Ok(())
}
