use super::*;

const MANIFEST_MISSING: &str = "CALYX_ASTER_MANIFEST_MISSING";
const MANIFEST_IO: &str = "CALYX_ASTER_MANIFEST_IO";

pub(super) fn storage_error(context: &str, path: &Path, error: io::Error) -> CalyxError {
    let message = format!(
        "{context} {}: kind={:?}: {error}",
        path.display(),
        error.kind()
    );
    match error.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => {
            CalyxError::disk_pressure(message)
        }
        io::ErrorKind::NotFound => CalyxError {
            code: MANIFEST_MISSING,
            message,
            remediation: "restore the named manifest member from a verified snapshot or pass the correct existing vault; do not synthesize manifest state",
        },
        _ => CalyxError {
            code: MANIFEST_IO,
            message,
            remediation: "inspect the named path, OS error kind, permissions, mount health, and filesystem state; repair the physical cause before retrying",
        },
    }
}

pub(super) fn format_version_unsupported(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: "CALYX_FORMAT_VERSION_UNSUPPORTED",
        message: message.into(),
        remediation: "refuse unknown format major; migrate through a compatible reader",
    }
}
