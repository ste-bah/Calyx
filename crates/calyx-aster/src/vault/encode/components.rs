use super::*;

pub(super) fn encode_input_ref_tail(input: &InputRef, out: &mut Vec<u8>) -> Result<()> {
    out.push(u8::from(input.redacted));
    match &input.pointer {
        Some(pointer) => {
            out.push(1);
            put_string(out, pointer)?;
        }
        None => out.push(0),
    }
    Ok(())
}

pub(super) fn decode_input_ref_tail(cursor: &mut Cursor<'_>, hash: [u8; 32]) -> Result<InputRef> {
    let redacted = cursor.u8()? != 0;
    let pointer = match cursor.u8()? {
        0 => None,
        1 => Some(cursor.string()?),
        tag => {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "unknown input pointer tag {tag}"
            )));
        }
    };
    Ok(InputRef {
        hash,
        pointer,
        redacted,
    })
}

pub(super) fn encode_absent_reason(reason: &AbsentReason, out: &mut Vec<u8>) -> Result<()> {
    match reason {
        AbsentReason::NotApplicable => out.push(0),
        AbsentReason::Redacted => out.push(1),
        AbsentReason::LensUnavailable => out.push(2),
        AbsentReason::Deferred => out.push(3),
        AbsentReason::LensInactive => out.push(4),
        AbsentReason::Error(value) => {
            out.push(5);
            put_string(out, value)?;
        }
    }
    Ok(())
}

pub(super) fn decode_absent_reason(cursor: &mut Cursor<'_>) -> Result<AbsentReason> {
    Ok(match cursor.u8()? {
        0 => AbsentReason::NotApplicable,
        1 => AbsentReason::Redacted,
        2 => AbsentReason::LensUnavailable,
        3 => AbsentReason::Deferred,
        4 => AbsentReason::LensInactive,
        5 => AbsentReason::Error(cursor.string()?),
        tag => {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "unknown absent tag {tag}"
            )));
        }
    })
}

pub(super) fn modality_tag(modality: Modality) -> u8 {
    match modality {
        Modality::Text => 0,
        Modality::Code => 1,
        Modality::Image => 2,
        Modality::Audio => 3,
        Modality::Video => 4,
        Modality::Structured => 5,
        Modality::Mixed => 6,
        Modality::Protein => 7,
        Modality::Dna => 8,
        Modality::Molecule => 9,
    }
}

pub(super) fn decode_modality(tag: u8) -> Result<Modality> {
    Ok(match tag {
        0 => Modality::Text,
        1 => Modality::Code,
        2 => Modality::Image,
        3 => Modality::Audio,
        4 => Modality::Video,
        5 => Modality::Structured,
        6 => Modality::Mixed,
        7 => Modality::Protein,
        8 => Modality::Dna,
        9 => Modality::Molecule,
        _ => return Err(CalyxError::aster_corrupt_shard("unknown modality tag")),
    })
}

pub(super) fn flags_bits(flags: CxFlags) -> u8 {
    u8::from(flags.ungrounded)
        | (u8::from(flags.degraded) << 1)
        | (u8::from(flags.novel_region) << 2)
        | (u8::from(flags.redacted_input) << 3)
}

pub(super) fn decode_flags(bits: u8) -> CxFlags {
    CxFlags {
        ungrounded: bits & 1 != 0,
        degraded: bits & 2 != 0,
        novel_region: bits & 4 != 0,
        redacted_input: bits & 8 != 0,
    }
}

pub(super) fn put_string(out: &mut Vec<u8>, value: &str) -> Result<()> {
    put_bytes(out, value.as_bytes())
}

pub(super) fn encode_string_metadata(
    metadata: &BTreeMap<String, String>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let count = u32::try_from(metadata.len())
        .map_err(|_| CalyxError::aster_corrupt_shard("metadata map too large"))?;
    out.extend_from_slice(&count.to_be_bytes());
    for (key, value) in metadata {
        put_string(out, key)?;
        put_string(out, value)?;
    }
    Ok(())
}

pub(super) fn decode_string_metadata(cursor: &mut Cursor<'_>) -> Result<BTreeMap<String, String>> {
    let count = cursor.u32()? as usize;
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = cursor.string()?;
        let value = cursor.string()?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

pub(super) fn require_nonzero_slot_count(field: &str, value: u32) -> Result<()> {
    if value == 0 {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "encoded slot vector {field} must be greater than zero"
        )));
    }
    Ok(())
}

pub(super) fn checked_encoded_len(
    header: usize,
    count: usize,
    width: usize,
    field: &str,
) -> Result<usize> {
    count
        .checked_mul(width)
        .and_then(|payload| header.checked_add(payload))
        .ok_or_else(|| {
            CalyxError::aster_corrupt_shard(format!(
                "encoded slot vector {field} byte count overflow: count={count}, width={width}, header={header}"
            ))
        })
}

pub(super) fn require_slot_vector_len(
    shape: &str,
    actual: usize,
    expected: usize,
    detail: String,
) -> Result<()> {
    if actual != expected {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "{shape} slot vector byte length mismatch: expected={expected}, actual={actual}, {detail}"
        )));
    }
    Ok(())
}

pub(super) fn reserved_vec<T>(capacity: usize, field: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        CalyxError::aster_corrupt_shard(format!(
            "cannot reserve {capacity} values for {field}: {error}"
        ))
    })?;
    Ok(values)
}

pub(super) fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| CalyxError::aster_corrupt_shard("encoded field too large"))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}
