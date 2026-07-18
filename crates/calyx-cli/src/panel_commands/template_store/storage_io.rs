use std::fs;
use std::io;
use std::path::Path;

use crate::error::CliResult;

use super::{TEMPLATE_INVALID, template_error};

pub(super) fn object_rel_path(template_id: &str) -> String {
    format!("objects/{template_id}.json")
}

pub(super) fn write_immutable(path: &Path, bytes: &[u8]) -> CliResult {
    match fs::read(path) {
        Ok(existing) if existing == bytes => return Ok(()),
        Ok(_) => {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "immutable template object {} already exists with different bytes",
                    path.display()
                ),
                "do not edit immutable template objects; save a new template version",
            ));
        }
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error.into()),
        Err(_) => {}
    }
    write_atomic(path, bytes)
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> CliResult {
    crate::durable_write::write_bytes_atomic(path, bytes, "panel template object/catalog")
}
