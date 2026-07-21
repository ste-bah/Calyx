use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use calyx_aster::ledger_view::LedgerQuerySnapshot;
use calyx_core::{CalyxError, CxId};
use calyx_ledger::{EntryKind, LedgerEntry, SubjectId};
use serde_json::Value;

use crate::server::ToolResult;

pub(super) fn indexed_answer_entries(
    query: &LedgerQuerySnapshot,
    answer_id: &[u8],
) -> ToolResult<Vec<LedgerEntry>> {
    let mut entries = BTreeMap::<u64, LedgerEntry>::new();
    for entry in query.entries_for_subject(&SubjectId::Query(answer_id.to_vec()))? {
        entries.insert(entry.seq, entry);
    }
    let mut linked_seqs = BTreeSet::new();
    let mut linked_ids = BTreeSet::<Vec<u8>>::new();
    for entry in entries
        .values()
        .filter(|entry| entry.kind == EntryKind::Answer)
    {
        let payload: Value = serde_json::from_slice(&entry.payload).map_err(|error| {
            CalyxError::ledger_corrupt(format!("decode answer payload seq={}: {error}", entry.seq))
        })?;
        for field in ["kernel_ref", "guard_ref"] {
            if let Some(seq) = payload
                .get(field)
                .and_then(|reference| reference.get("seq"))
                .and_then(Value::as_u64)
            {
                linked_seqs.insert(seq);
            }
        }
        for field in ["kernel_id", "guard_id"] {
            if let Some(raw) = payload.get(field).and_then(Value::as_str) {
                linked_ids.extend(identifier_variants(raw));
            }
        }
    }
    for entry in query.read_selected(&linked_seqs)? {
        entries.insert(entry.seq, entry);
    }
    for id in linked_ids {
        for entry in query.entries_for_subject_bytes(&id)? {
            entries.insert(entry.seq, entry);
        }
    }
    Ok(entries.into_values().collect())
}

fn identifier_variants(raw: &str) -> BTreeSet<Vec<u8>> {
    let mut out = BTreeSet::from([raw.as_bytes().to_vec()]);
    if let Ok(id) = CxId::from_str(raw) {
        out.insert(id.as_bytes().to_vec());
    }
    if raw.len().is_multiple_of(2)
        && raw.bytes().all(|byte| byte.is_ascii_hexdigit())
        && let Some(bytes) = raw
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| {
                let high = (chunk[0] as char).to_digit(16)? as u8;
                let low = (chunk[1] as char).to_digit(16)? as u8;
                Some((high << 4) | low)
            })
            .collect::<Option<Vec<_>>>()
    {
        out.insert(bytes);
    }
    out
}
