use calyx_core::CalyxError;

use super::CALYX_ANNEAL_REPLAY_INVALID_ROW;
use crate::ledger_anneal::CALYX_ASTER_CF_UNAVAILABLE;

pub(super) fn invalid_row(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: CALYX_ANNEAL_REPLAY_INVALID_ROW,
        message: message.into(),
        remediation: "repair or quarantine anneal_replay CF rows before learning",
    }
}

pub(super) fn cf_unavailable(context: &str, error: CalyxError) -> CalyxError {
    CalyxError {
        code: CALYX_ASTER_CF_UNAVAILABLE,
        message: format!("{context}: {}: {}", error.code, error.message),
        remediation: "restore Aster anneal_replay CF availability",
    }
}
