use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, CalyxErrorCode, Result};

use crate::cf::ColumnFamily;
use crate::file_lock::FileLockGuard;

use super::record::HeaderStatus;
use super::{TornTail, record, storage_error};

pub(crate) struct WalWriteRowPointReader {
    _lock: FileLockGuard,
    file: File,
    path: PathBuf,
}

impl WalWriteRowPointReader {
    pub(crate) fn open(segment_path: impl AsRef<Path>) -> Result<Self> {
        let path = segment_path.as_ref();
        let dir = path
            .parent()
            .ok_or_else(|| CalyxError::disk_pressure("WAL segment path has no parent"))?;
        let lock = FileLockGuard::acquire(&dir.join(".append.lock"))?;
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|error| storage_error("open WAL segment for indexed row read", error))?;
        Ok(Self {
            _lock: lock,
            file,
            path: path.to_path_buf(),
        })
    }

    /// Reads one encoded Base row without materializing unrelated slot rows
    /// from the same WAL write batch. The page index independently authenticates
    /// the returned value with SHA-256; this reader validates the WAL framing,
    /// record identity, CF tag, encoded key, and physical row bounds.
    pub(crate) fn read_base_value(
        &mut self,
        seq: u64,
        start_offset: u64,
        end_offset: u64,
        row_offset: u64,
        expected_key: &[u8],
    ) -> Result<Vec<u8>> {
        let header = match record::read_header_at(&mut self.file, start_offset)
            .map_err(|error| storage_error("read WAL header for indexed row", error))?
        {
            HeaderStatus::Complete(header) => header,
            HeaderStatus::Eof => {
                return Err(CalyxError::aster_corrupt_shard(format!(
                    "WAL record {seq} at {}:{start_offset}..{end_offset} is beyond EOF",
                    self.path.display()
                )));
            }
            HeaderStatus::Torn { offset, message } => {
                return Err(TornTail {
                    segment_path: self.path.clone(),
                    offset,
                    code: CalyxErrorCode::AsterTornWal.code(),
                    message,
                }
                .error());
            }
        };
        if header.seq != seq
            || header.start_offset != start_offset
            || header.end_offset != end_offset
        {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "WAL record {} decoded as seq {} range {}..{} instead of seq {seq} range {start_offset}..{end_offset}",
                self.path.display(),
                header.seq,
                header.start_offset,
                header.end_offset
            )));
        }
        let payload_start = start_offset
            .checked_add(super::RECORD_HEADER_BYTES)
            .ok_or_else(|| CalyxError::aster_corrupt_shard("WAL payload offset overflow"))?;
        if row_offset < payload_start || row_offset >= end_offset {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "WAL Base row offset {row_offset} is outside record {seq} payload {payload_start}..{end_offset} in {}",
                self.path.display()
            )));
        }
        self.file
            .seek(SeekFrom::Start(row_offset))
            .map_err(|error| storage_error("seek WAL indexed Base row", error))?;
        let mut prefix = [0_u8; 5];
        read_exact_indexed(&mut self.file, &mut prefix, &self.path, row_offset)?;
        let cf = crate::vault::cf_codec::decode_cf(prefix[0])?;
        if cf != ColumnFamily::Base {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "WAL indexed Base row at {}:{row_offset} has CF {cf:?}",
                self.path.display()
            )));
        }
        let key_len = u32::from_be_bytes(prefix[1..5].try_into().expect("key length")) as usize;
        let mut key = vec![0_u8; key_len];
        read_exact_indexed(&mut self.file, &mut key, &self.path, row_offset + 5)?;
        if key != expected_key {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "WAL indexed Base row at {}:{row_offset} has key {} instead of {}",
                self.path.display(),
                hex_bytes(&key),
                hex_bytes(expected_key)
            )));
        }
        let mut value_len_bytes = [0_u8; 4];
        let value_len_offset = row_offset
            .checked_add(5)
            .and_then(|offset| offset.checked_add(key_len as u64))
            .ok_or_else(|| CalyxError::aster_corrupt_shard("WAL Base row key offset overflow"))?;
        read_exact_indexed(
            &mut self.file,
            &mut value_len_bytes,
            &self.path,
            value_len_offset,
        )?;
        let value_len = u32::from_be_bytes(value_len_bytes) as usize;
        let value_start = value_len_offset
            .checked_add(4)
            .ok_or_else(|| CalyxError::aster_corrupt_shard("WAL Base value offset overflow"))?;
        let value_end = value_start
            .checked_add(value_len as u64)
            .ok_or_else(|| CalyxError::aster_corrupt_shard("WAL Base value length overflow"))?;
        if value_end > end_offset {
            return Err(CalyxError::aster_corrupt_shard(format!(
                "WAL indexed Base value at {}:{value_start}..{value_end} exceeds record {seq} end {end_offset}",
                self.path.display()
            )));
        }
        let mut value = vec![0_u8; value_len];
        read_exact_indexed(&mut self.file, &mut value, &self.path, value_start)?;
        Ok(value)
    }
}

fn read_exact_indexed(file: &mut File, out: &mut [u8], path: &Path, offset: u64) -> Result<()> {
    file.read_exact(out).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            CalyxError::aster_corrupt_shard(format!(
                "WAL indexed row is truncated at {}:{offset} while reading {} bytes",
                path.display(),
                out.len()
            ))
        } else {
            storage_error("read WAL indexed row", error)
        }
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
