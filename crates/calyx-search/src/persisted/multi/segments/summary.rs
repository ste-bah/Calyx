use super::*;

#[cfg(test)]
pub(super) fn summarize_segment_files(
    vault_dir: &Path,
    slot: SlotId,
    token_dim: u32,
    manifest: &MultiSegmentsManifest,
    verify_binary: bool,
) -> CliResult<ReusedMultiSegments> {
    let mut ids = BTreeSet::new();
    let mut token_count = 0usize;
    let mut refs = Vec::with_capacity(manifest.segments.len());
    for segment in &manifest.segments {
        let path = checked_segment_path(vault_dir, &segment.index_rel, slot)?;
        let mut segment_ref = segment.clone();
        if !verify_binary && !segment.ids.is_empty() {
            if segment.ids.len() != segment.row_count {
                return Err(stale(format!(
                    "persistent segmented multi manifest {} id count {} != row_count {}; rebuild the vault search indexes",
                    segment.index_rel,
                    segment.ids.len(),
                    segment.row_count
                )));
            }
            for cx_id in &segment.ids {
                if !ids.insert(*cx_id) {
                    return Err(stale(format!(
                        "persistent segmented multi sidecars repeat {cx_id}; rebuild the vault search indexes"
                    )));
                }
            }
            token_count = token_count
                .checked_add(segment.token_count)
                .ok_or_else(|| stale("persistent segmented multi sidecar token_count overflow"))?;
            refs.push(segment_ref);
            continue;
        }
        let summary = binary::summarize_binary_path(
            &path,
            &segment.sha256,
            slot,
            token_dim,
            Some(segment.row_count as u64),
            Some(segment.token_count as u64),
        )?;
        if summary.base_seq != segment.base_seq {
            return Err(stale(format!(
                "persistent segmented multi sidecar {} seq {} != segment manifest seq {}; rebuild the vault search indexes",
                segment.index_rel, summary.base_seq, segment.base_seq
            )));
        }
        segment_ref.ids = summary.ids.iter().copied().collect();
        for cx_id in summary.ids {
            if !ids.insert(cx_id) {
                return Err(stale(format!(
                    "persistent segmented multi sidecars repeat {cx_id}; rebuild the vault search indexes"
                )));
            }
        }
        token_count = token_count
            .checked_add(segment.token_count)
            .ok_or_else(|| stale("persistent segmented multi sidecar token_count overflow"))?;
        refs.push(segment_ref);
    }
    if ids.len() != manifest.row_count {
        return Err(stale(format!(
            "persistent segmented multi manifest row_count {} != unique row count {}; rebuild the vault search indexes",
            manifest.row_count,
            ids.len()
        )));
    }
    if token_count != manifest.token_count {
        return Err(stale(format!(
            "persistent segmented multi manifest token_count {} != sidecar token count {token_count}; rebuild the vault search indexes",
            manifest.token_count
        )));
    }
    Ok(ReusedMultiSegments {
        refs,
        ids,
        token_count,
    })
}
