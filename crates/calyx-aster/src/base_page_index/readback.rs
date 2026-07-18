use super::{
    BASE_PAGE_INDEX_DIR, BasePageIndexEntry, BasePageIndexPage, BasePageIndexPageRef,
    BasePageIndexSource, corrupt, sha256_hex, stale,
};
use crate::mvcc::is_tombstone_value;
use crate::sst::SstPointReader;
use crate::wal::WalWriteRowPointReader;
use calyx_core::{CalyxError, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

pub(super) fn read_page(
    vault: &Path,
    page_ref: &BasePageIndexPageRef,
) -> Result<BasePageIndexPage> {
    let path = vault.join(BASE_PAGE_INDEX_DIR).join(&page_ref.path);
    let bytes = fs::read(&path).map_err(|error| {
        CalyxError::disk_pressure(format!("read Base page index page: {error}"))
    })?;
    let actual = sha256_hex(&bytes);
    if actual != page_ref.sha256_hex {
        return Err(corrupt(format!(
            "Base page index page {} sha256 mismatch: expected {}, got {}",
            path.display(),
            page_ref.sha256_hex,
            actual
        )));
    }
    let page: BasePageIndexPage = serde_json::from_slice(&bytes)
        .map_err(|error| corrupt(format!("decode Base page index page: {error}")))?;
    if page.entries.len() != page_ref.entry_count {
        return Err(corrupt(format!(
            "Base page index page {} expected {} entries, got {}",
            path.display(),
            page_ref.entry_count,
            page.entries.len()
        )));
    }
    Ok(page)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SourceValueVisit {
    pub(super) source_files: usize,
    pub(super) values: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SourceFile {
    Sst(String),
    Wal(String),
}

/// Visits exact physical Base values while opening each backing file once.
///
/// Entry metadata may be retained so requests can be grouped by immutable
/// source file, but values are authenticated and handed to the visitor one at
/// a time. This keeps memory bounded even when one compacted SST owns every
/// selected key.
pub(super) fn visit_source_values<E>(
    vault: &Path,
    entries: Vec<(Vec<u8>, BasePageIndexEntry)>,
    mut visitor: impl FnMut(&[u8], Vec<u8>) -> std::result::Result<(), E>,
) -> std::result::Result<SourceValueVisit, E>
where
    E: From<CalyxError>,
{
    let mut groups = BTreeMap::<SourceFile, Vec<(Vec<u8>, BasePageIndexEntry)>>::new();
    for (key, entry) in entries {
        let source = match &entry.source {
            BasePageIndexSource::Sst { path, .. } => SourceFile::Sst(path.clone()),
            BasePageIndexSource::Wal { path, .. } => SourceFile::Wal(path.clone()),
        };
        groups.entry(source).or_default().push((key, entry));
    }
    let mut stats = SourceValueVisit {
        source_files: groups.len(),
        ..SourceValueVisit::default()
    };
    for (source, mut group) in groups {
        match source {
            SourceFile::Sst(path) => {
                require_sst_offsets(&group)?;
                group.sort_by_key(|(_, entry)| match &entry.source {
                    BasePageIndexSource::Sst {
                        record_offset: Some(offset),
                        ..
                    } => *offset,
                    _ => u64::MAX,
                });
                let source_path = checked_source_path(vault, &path)?;
                let mut reader = SstPointReader::open(&source_path)?;
                for (key, entry) in group {
                    let record_offset = match &entry.source {
                        BasePageIndexSource::Sst {
                            record_offset: Some(record_offset),
                            ..
                        } => *record_offset,
                        _ => {
                            return Err(
                                corrupt("Base page index SST source group changed type").into()
                            );
                        }
                    };
                    let value = reader.read_value(record_offset, &key)?;
                    validate_entry_value(&entry, &value)?;
                    stats.values += 1;
                    visitor(&key, value)?;
                }
            }
            SourceFile::Wal(path) => {
                require_wal_offsets(&group)?;
                group.sort_by_key(|(_, entry)| match &entry.source {
                    BasePageIndexSource::Wal {
                        row_offset: Some(offset),
                        ..
                    } => *offset,
                    _ => u64::MAX,
                });
                let source_path = checked_source_path(vault, &path)?;
                let mut reader = WalWriteRowPointReader::open(&source_path)?;
                for (key, entry) in group {
                    let (seq, start_offset, end_offset, row_offset) = match &entry.source {
                        BasePageIndexSource::Wal {
                            seq,
                            start_offset,
                            end_offset,
                            row_offset: Some(row_offset),
                            ..
                        } => (*seq, *start_offset, *end_offset, *row_offset),
                        _ => {
                            return Err(
                                corrupt("Base page index WAL source group changed type").into()
                            );
                        }
                    };
                    let value =
                        reader.read_base_value(seq, start_offset, end_offset, row_offset, &key)?;
                    validate_entry_value(&entry, &value)?;
                    stats.values += 1;
                    visitor(&key, value)?;
                }
            }
        }
    }
    Ok(stats)
}

fn require_sst_offsets(entries: &[(Vec<u8>, BasePageIndexEntry)]) -> Result<()> {
    if entries.iter().any(|(_, entry)| {
        !matches!(
            entry.source,
            BasePageIndexSource::Sst {
                record_offset: Some(_),
                ..
            }
        )
    }) {
        return Err(stale(
            "Base page index SST source lacks an exact record offset; rebuild the index to migrate it",
        ));
    }
    Ok(())
}

fn require_wal_offsets(entries: &[(Vec<u8>, BasePageIndexEntry)]) -> Result<()> {
    if entries.iter().any(|(_, entry)| {
        !matches!(
            entry.source,
            BasePageIndexSource::Wal {
                row_offset: Some(_),
                ..
            }
        )
    }) {
        return Err(stale(
            "Base page index WAL source lacks an exact row offset; rebuild the index to migrate it",
        ));
    }
    Ok(())
}

fn checked_source_path(vault: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(corrupt(format!(
            "Base page index source path {relative:?} is not a canonical relative path"
        )));
    }
    let source_path = vault.join(path);
    if !source_path.exists() {
        return Err(stale(format!(
            "Base page index source {} no longer exists",
            source_path.display()
        )));
    }
    Ok(source_path)
}

fn validate_entry_value(entry: &BasePageIndexEntry, value: &[u8]) -> Result<()> {
    let hash = sha256_hex(value);
    if hash != entry.value_sha256_hex {
        return Err(corrupt(format!(
            "Base page index key {} source value sha256 mismatch: expected {}, got {}",
            entry.key_hex, entry.value_sha256_hex, hash
        )));
    }
    let tombstoned = is_tombstone_value(value);
    if tombstoned != entry.tombstoned {
        return Err(corrupt(format!(
            "Base page index key {} tombstone state mismatch: manifest {}, source {}",
            entry.key_hex, entry.tombstoned, tombstoned
        )));
    }
    Ok(())
}
