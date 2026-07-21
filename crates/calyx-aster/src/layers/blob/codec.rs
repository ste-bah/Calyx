use super::*;

pub fn collection_id(col: &Collection) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"calyx:blob:collection:v1");
    hasher.update(&col.tenant.0.to_be_bytes());
    hasher.update(&(col.name.len() as u16).to_be_bytes());
    hasher.update(col.name.as_bytes());
    u64::from_be_bytes(hasher.finalize().as_bytes()[0..8].try_into().unwrap())
}

/// `0x05 | 0x00 | cid | blob_id | chunk_idx`.
pub fn chunk_key(col: &Collection, blob_id: BlobId, idx: u32) -> Vec<u8> {
    let mut key = chunk_prefix(col, blob_id);
    key.extend_from_slice(&idx.to_be_bytes());
    key
}

/// `0x05 | 0x01 | cid | blob_id`.
pub fn manifest_key(col: &Collection, blob_id: BlobId) -> Vec<u8> {
    let mut key = Vec::with_capacity(2 + 8 + BLOB_ID_BYTES);
    key.push(DISC_BLOB);
    key.push(KIND_MANIFEST);
    key.extend_from_slice(&collection_id(col).to_be_bytes());
    key.extend_from_slice(blob_id.as_bytes());
    key
}

pub(super) fn chunk_prefix(col: &Collection, blob_id: BlobId) -> Vec<u8> {
    let mut key = Vec::with_capacity(2 + 8 + BLOB_ID_BYTES + 4);
    key.push(DISC_BLOB);
    key.push(KIND_CHUNK);
    key.extend_from_slice(&collection_id(col).to_be_bytes());
    key.extend_from_slice(blob_id.as_bytes());
    key
}

pub(super) fn encode_manifest(manifest: &BlobManifest) -> Vec<u8> {
    let mut out = Vec::with_capacity(MANIFEST_VALUE_BYTES);
    out.extend_from_slice(&manifest.total_bytes.to_be_bytes());
    out.extend_from_slice(&manifest.chunk_count.to_be_bytes());
    out.extend_from_slice(&manifest.content_hash);
    out.push(u8::from(manifest.cold_tier));
    out.extend_from_slice(&manifest.created_at_ms.unwrap_or(0).to_be_bytes());
    out
}

pub(super) fn decode_manifest(bytes: &[u8]) -> Result<BlobManifest> {
    if !matches!(bytes.len(), MANIFEST_VALUE_BYTES_V1 | MANIFEST_VALUE_BYTES) {
        return Err(corrupt(format!(
            "blob manifest must be {MANIFEST_VALUE_BYTES_V1} or {MANIFEST_VALUE_BYTES} bytes, got {}",
            bytes.len()
        )));
    }
    let total_bytes = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
    let chunk_count = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    if total_bytes > MAX_BLOB_BYTES as u64 {
        return Err(corrupt(format!(
            "blob manifest total_bytes {total_bytes} exceeds the {MAX_BLOB_BYTES}-byte ceiling"
        )));
    }
    let expected_chunks = if total_bytes == 0 {
        0
    } else {
        total_bytes.div_ceil(BLOB_CHUNK_SIZE as u64)
    };
    if u64::from(chunk_count) != expected_chunks {
        return Err(corrupt(format!(
            "blob manifest chunk_count {chunk_count} does not match {total_bytes} bytes at {BLOB_CHUNK_SIZE} bytes per chunk (expected {expected_chunks})"
        )));
    }
    let mut content_hash = [0_u8; HASH_BYTES];
    content_hash.copy_from_slice(&bytes[12..44]);
    let cold_tier = match bytes[44] {
        0 => false,
        1 => true,
        other => {
            return Err(corrupt(format!(
                "blob manifest cold_tier byte {other} is not 0/1"
            )));
        }
    };
    let created_at_ms = if bytes.len() == MANIFEST_VALUE_BYTES {
        Some(u64::from_be_bytes(bytes[45..53].try_into().unwrap()))
    } else {
        None
    };
    Ok(BlobManifest {
        total_bytes,
        chunk_count,
        content_hash,
        cold_tier,
        created_at_ms,
    })
}

pub(super) fn hash_payload(data: &[u8]) -> [u8; HASH_BYTES] {
    #[cfg(test)]
    {
        HASH_CALLS.set(HASH_CALLS.get() + 1);
        HASHED_BYTES.set(HASHED_BYTES.get() + data.len());
    }
    *blake3::hash(data).as_bytes()
}

/// Half-open range covering every row (chunks + manifest) of one blob. Used by
/// callers that need to scan a blob's physical footprint.
pub fn blob_row_range(col: &Collection, blob_id: BlobId) -> KeyRange {
    crate::cf::prefix_range(&{
        let mut prefix = Vec::with_capacity(1 + 8 + BLOB_ID_BYTES);
        prefix.push(DISC_BLOB);
        // Spans both KIND_CHUNK (0x00) and KIND_MANIFEST (0x01) for this blob.
        prefix.push(KIND_CHUNK);
        prefix.extend_from_slice(&collection_id(col).to_be_bytes());
        prefix.extend_from_slice(blob_id.as_bytes());
        prefix
    })
}

pub(super) fn require_blob_mode(col: &Collection) -> Result<()> {
    if col.mode == CollectionMode::Blob {
        Ok(())
    } else {
        Err(invalid_argument(format!(
            "blob layer requires a Blob collection, got {:?}",
            col.mode
        )))
    }
}

pub(super) fn ledger_subject(manifest_key: &[u8]) -> SubjectId {
    SubjectId::Query(blake3::hash(manifest_key).as_bytes().to_vec())
}

pub(super) fn ledger_payload(
    col: &Collection,
    blob_id: BlobId,
    manifest: &BlobManifest,
) -> Vec<u8> {
    let mut payload = PayloadBuilder::default();
    payload
        .insert_str("collection_id", format!("{:016x}", collection_id(col)))
        .insert_str("blob_id", hex_bytes(blob_id.as_bytes()))
        .insert_str("total_bytes", manifest.total_bytes.to_string())
        .insert_str("chunk_count", manifest.chunk_count.to_string())
        .insert_str("content_hash", hex_bytes(&manifest.content_hash));
    RedactionPolicy::default().apply_to_payload(&payload)
}

pub(super) fn hex_bytes(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(super) fn blob_too_large(len: usize) -> CalyxError {
    CalyxError {
        code: CALYX_BLOB_TOO_LARGE,
        message: format!("blob of {len} bytes exceeds the {MAX_BLOB_BYTES}-byte ceiling"),
        remediation: "split the payload or raise MAX_BLOB_BYTES",
    }
}

pub(super) fn invalid_argument(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: CALYX_INVALID_ARGUMENT,
        message: message.into(),
        remediation: "fix the blob input",
    }
}

pub(super) fn corrupt(message: impl Into<String>) -> CalyxError {
    CalyxError::aster_corrupt_shard(message)
}
