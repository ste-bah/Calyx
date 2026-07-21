use std::collections::BTreeSet;
use std::str::FromStr;

use calyx_core::CxId;
use serde_json::Value;

use crate::entry::{LedgerEntry, SubjectId};

pub(super) fn entry_mentions_cx(entry: &LedgerEntry, cx_id: CxId) -> bool {
    entry_cx_mentions(entry).contains(&cx_id)
}

/// Returns every CX referenced by the entry's primary subject or recognized
/// provenance payload fields. The traversal intentionally mirrors
/// `get_provenance`: unrelated strings are not interpreted as CX identifiers.
pub fn entry_cx_mentions(entry: &LedgerEntry) -> BTreeSet<CxId> {
    let mut out = BTreeSet::new();
    if let SubjectId::Cx(cx_id) = entry.subject {
        out.insert(cx_id);
    }
    if let Ok(payload) = serde_json::from_slice::<Value>(&entry.payload) {
        collect_value_mentions(&payload, &mut out);
    }
    out
}

fn collect_value_mentions(value: &Value, out: &mut BTreeSet<CxId>) {
    match value {
        Value::Object(map) => map.iter().for_each(|(key, value)| {
            if is_cx_payload_field(key) {
                collect_cx_values(value, out);
            } else if matches!(value, Value::Object(_) | Value::Array(_)) {
                collect_value_mentions(value, out);
            }
        }),
        Value::Array(values) => values.iter().for_each(|value| {
            if matches!(value, Value::Object(_)) {
                collect_value_mentions(value, out);
            }
        }),
        _ => {}
    }
}

fn collect_cx_values(value: &Value, out: &mut BTreeSet<CxId>) {
    match value {
        Value::String(value) => {
            if let Ok(cx_id) = CxId::from_str(value) {
                out.insert(cx_id);
            }
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_cx_values(value, out)),
        Value::Object(_) => collect_value_mentions(value, out),
        _ => {}
    }
}

fn is_cx_payload_field(key: &str) -> bool {
    matches!(
        key,
        "cx_id"
            | "from_id"
            | "to_id"
            | "source_cx_id"
            | "target_cx_id"
            | "nearest_cx"
            | "matched_cx_id"
            | "query_id"
            | "anchor_kernel_node_id"
    )
}
