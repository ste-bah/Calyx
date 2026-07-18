use std::fs;

use crate::error::{CliError, CliResult};

pub(super) fn write(output: &serde_json::Value) -> CliResult {
    let Some(root) = calyx_fsv::env_fsv_root("CALYX_FSV_ROOT")
        .map_err(|error| CliError::usage(error.to_string()))?
    else {
        return Ok(());
    };
    let dir = root.join("weave-loom");
    fs::create_dir_all(&dir)?;
    let path = dir.join("weave_loom_report.json");
    let bytes = serde_json::to_vec_pretty(output)
        .map_err(|error| CliError::runtime(format!("serialize {}: {error}", path.display())))?;
    fs::write(&path, bytes)?;
    eprintln!("WEAVE_LOOM_READBACK={}", path.display());
    Ok(())
}
