use super::*;

pub(super) fn parse_cx_field(value: &Value, field: &str) -> Option<Result<CxId>> {
    value.get(field).and_then(Value::as_str).map(|raw| {
        CxId::from_str(raw).map_err(|error| CalyxError::ledger_corrupt(error.to_string()))
    })
}

pub(super) fn parse_lens_field(value: &Value, field: &str) -> Option<Result<LensId>> {
    value.get(field).and_then(Value::as_str).map(|raw| {
        LensId::from_str(raw).map_err(|error| CalyxError::ledger_corrupt(error.to_string()))
    })
}

pub(super) fn expected_hops_match(payload: &Value, path_len: usize) -> bool {
    payload
        .get("expected_hops")
        .or_else(|| payload.get("hop_count"))
        .or_else(|| payload.get("path_len"))
        .and_then(Value::as_u64)
        .is_none_or(|expected| expected as usize == path_len)
}

pub(super) fn linked_entry(
    entries: &[LedgerEntry],
    payload: &Value,
    kind: EntryKind,
    id_fields: &[&str],
    ref_field: &str,
) -> Option<LedgerEntry> {
    if let Some(entry) = linked_entry_by_ref(entries, payload, kind, ref_field) {
        return Some(entry.clone());
    }
    let ids = id_fields
        .iter()
        .filter_map(|field| payload.get(*field).and_then(Value::as_str))
        .flat_map(identifier_variants)
        .collect::<Vec<_>>();
    entries
        .iter()
        .find(|entry| entry.kind == kind && subject_bytes_match(&entry.subject, &ids))
        .cloned()
}

pub(super) fn linked_entry_by_ref<'a>(
    entries: &'a [LedgerEntry],
    payload: &Value,
    kind: EntryKind,
    ref_field: &str,
) -> Option<&'a LedgerEntry> {
    let reference = payload.get(ref_field)?;
    let seq = reference.get("seq").and_then(Value::as_u64)?;
    entries.iter().find(|entry| {
        entry.kind == kind
            && entry.seq == seq
            && reference
                .get("hash")
                .and_then(Value::as_str)
                .is_none_or(|hash| hash.eq_ignore_ascii_case(&hex(&entry.entry_hash)))
    })
}

pub(super) fn linked_payload_present(payload: &Value, id_fields: &[&str], ref_field: &str) -> bool {
    payload.get(ref_field).is_some()
        || id_fields
            .iter()
            .any(|field| payload.get(*field).and_then(Value::as_str).is_some())
}

pub(super) fn subject_bytes_match(subject: &SubjectId, candidates: &[Vec<u8>]) -> bool {
    let bytes: &[u8] = match subject {
        SubjectId::Kernel(bytes) | SubjectId::Guard(bytes) | SubjectId::Query(bytes) => bytes,
        SubjectId::Cx(id) => id.as_bytes(),
        SubjectId::Lens(id) => id.as_bytes(),
    };
    candidates.iter().any(|candidate| candidate == bytes)
}

pub(super) fn identifier_variants(raw: &str) -> Vec<Vec<u8>> {
    let mut out = vec![raw.as_bytes().to_vec()];
    if let Ok(id) = CxId::from_str(raw) {
        out.push(id.as_bytes().to_vec());
    }
    if raw.len().is_multiple_of(2)
        && raw.bytes().all(|byte| byte.is_ascii_hexdigit())
        && let Some(bytes) = decode_hex(raw)
    {
        out.push(bytes);
    }
    out.sort();
    out.dedup();
    out
}

pub(super) fn decode_hex(value: &str) -> Option<Vec<u8>> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| Some((hex_value(chunk[0])? << 4) | hex_value(chunk[1])?))
        .collect()
}

pub(super) fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
