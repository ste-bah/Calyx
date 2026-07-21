use std::collections::BTreeSet;
use std::path::Path;

use calyx_core::{CalyxError, Result};
use calyx_ledger::{LedgerEntry, SubjectId, decode};

use super::LedgerQueryIndex;

const TAIL_BATCH_ROWS: u64 = 4_096;

pub(super) fn extend_index(
    vault: &Path,
    index: &mut LedgerQueryIndex,
    new_height: u64,
    new_tip: &[u8; 32],
) -> Result<u64> {
    let original_height = index.height;
    let mut expected_prev = index.tip_hash;
    while index.height < new_height {
        let end = index.height.saturating_add(TAIL_BATCH_ROWS).min(new_height);
        let wanted = (index.height..end).collect::<BTreeSet<_>>();
        let (rows, _) = super::super::read_ledger_seqs_unlocked_traced(vault, &wanted, None)?;
        for seq in index.height..end {
            let row = rows.get(&seq).ok_or_else(|| {
                CalyxError::ledger_chain_broken(format!(
                    "ledger query incremental index is missing seq {seq}"
                ))
            })?;
            let entry = decode_checked(row.seq, &row.bytes)?;
            if entry.prev_hash != expected_prev {
                return Err(CalyxError::ledger_chain_broken(format!(
                    "ledger query incremental index prev hash mismatch at seq {seq}"
                )));
            }
            expected_prev = entry.entry_hash;
            index.push(&entry)?;
        }
    }
    index.validate_generation(new_height, new_tip)?;
    Ok(index.height - original_height)
}

pub(super) fn decode_checked(seq: u64, bytes: &[u8]) -> Result<LedgerEntry> {
    let entry = decode(bytes)?;
    if entry.seq != seq {
        return Err(CalyxError::ledger_chain_broken(format!(
            "ledger query physical key {seq} does not match encoded seq {}",
            entry.seq
        )));
    }
    if !entry.verify() {
        return Err(CalyxError::ledger_corrupt(format!(
            "ledger query entry seq {seq} hash mismatch"
        )));
    }
    Ok(entry)
}

pub(super) fn subject_bytes(subject: &SubjectId) -> &[u8] {
    match subject {
        SubjectId::Cx(id) => id.as_bytes(),
        SubjectId::Lens(id) => id.as_bytes(),
        SubjectId::Kernel(bytes) | SubjectId::Guard(bytes) | SubjectId::Query(bytes) => bytes,
    }
}
