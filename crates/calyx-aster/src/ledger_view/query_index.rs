//! Head-anchored ledger subject/kind side index.
//!
//! The index is a derived, checksummed generation. Its file name and payload
//! both bind it to the durable ledger head. Missing generations are rebuilt;
//! malformed generations fail closed. Once a generation exists, appends are
//! incorporated from targeted point reads of only the new tail.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, CxId, Result};
use calyx_ledger::{
    EntryKind, LedgerEntry, SubjectId, VerifyResult, entry_cx_mentions, verify_snapshot,
};
use serde::{Deserialize, Serialize};

use super::{AsterLedgerCfStore, AsterVaultLayout, durable_commit_lock_path};
use build::{decode_checked, extend_index, subject_bytes};
use persistence::{index_path, newest_previous_index, read_index, write_index};

mod build;
mod persistence;

const INDEX_VERSION: u16 = 1;
const EMPTY_SEQUENCES: &[u64] = &[];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LedgerQueryOpenStats {
    pub index_rebuilt: bool,
    pub rows_indexed: u64,
    pub rows_reused: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LedgerQueryVisitStats {
    pub snapshot_height: u64,
    pub batches_read: u64,
    pub matching_rows_visited: u64,
    pub physical_rows_read: u64,
}

#[derive(Clone, Debug)]
pub struct LedgerQuerySnapshot {
    vault: PathBuf,
    index: LedgerQueryIndex,
    open_stats: LedgerQueryOpenStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerQueryIndex {
    version: u16,
    height: u64,
    tip_hash: [u8; 32],
    entry_count: u64,
    subjects: BTreeMap<SubjectKey, Vec<u64>>,
    subject_bytes: BTreeMap<Vec<u8>, Vec<u64>>,
    cx_mentions: BTreeMap<[u8; 16], Vec<u64>>,
    kinds: BTreeMap<u8, Vec<u64>>,
    answers: BTreeMap<Vec<u8>, Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct SubjectKey {
    tag: u8,
    bytes: Vec<u8>,
}

impl LedgerQuerySnapshot {
    pub fn open(vault: &Path) -> Result<Self> {
        let _commit_guard =
            crate::file_lock::FileLockGuard::acquire(&durable_commit_lock_path(vault))?;
        let layout = AsterVaultLayout::read(vault)?;
        let anchor = crate::ledger_head::read_head_anchor(vault)?;
        let Some(anchor) = anchor else {
            let store = AsterLedgerCfStore::open_with_layout(vault, layout, None)?;
            if !store.rows.is_empty() {
                return Err(CalyxError::ledger_corrupt(
                    "ledger query found rows without a durable head anchor",
                ));
            }
            return Ok(Self {
                vault: vault.to_path_buf(),
                index: LedgerQueryIndex::empty(),
                open_stats: LedgerQueryOpenStats::default(),
            });
        };

        let expected = index_path(vault, anchor.height, &anchor.tip_hash);
        if expected.exists() {
            let index = read_index(&expected)?;
            index.validate_generation(anchor.height, &anchor.tip_hash)?;
            return Ok(Self {
                vault: vault.to_path_buf(),
                open_stats: LedgerQueryOpenStats {
                    rows_reused: index.height,
                    ..LedgerQueryOpenStats::default()
                },
                index,
            });
        }

        let previous = newest_previous_index(vault, anchor.height)?;
        let (index, open_stats) = if let Some(previous) = previous {
            let mut index = read_index(&previous)?;
            index.validate_internal()?;
            if index.height >= anchor.height {
                return Err(CalyxError::ledger_corrupt(format!(
                    "ledger query previous generation {} is not older than head {}",
                    index.height, anchor.height
                )));
            }
            let reused = index.height;
            let rows_indexed = extend_index(vault, &mut index, anchor.height, &anchor.tip_hash)?;
            (
                index,
                LedgerQueryOpenStats {
                    index_rebuilt: true,
                    rows_indexed,
                    rows_reused: reused,
                },
            )
        } else {
            let store = AsterLedgerCfStore::open_with_layout(vault, layout, None)?;
            let snapshot = calyx_ledger::LedgerCfStore::snapshot(&store)?;
            match verify_snapshot(&snapshot, 0..anchor.height)? {
                VerifyResult::Intact { count } if count == anchor.height => {}
                VerifyResult::Intact { count } => {
                    return Err(CalyxError::ledger_chain_broken(format!(
                        "ledger query verified {count} rows at head height {}",
                        anchor.height
                    )));
                }
                VerifyResult::Broken {
                    at_seq,
                    expected,
                    found,
                } => {
                    return Err(CalyxError::ledger_chain_broken(format!(
                        "ledger query index build broke at seq {at_seq}: expected={} found={}",
                        hex(&expected),
                        hex(&found)
                    )));
                }
                VerifyResult::Corrupt { at_seq, reason } => {
                    return Err(CalyxError::ledger_corrupt(format!(
                        "ledger query index build found corrupt seq {at_seq}: {reason}"
                    )));
                }
            }
            let mut index = LedgerQueryIndex::empty();
            for row in &store.rows {
                let entry = decode_checked(row.seq, &row.bytes)?;
                index.push(&entry)?;
            }
            index.tip_hash = anchor.tip_hash;
            index.validate_generation(anchor.height, &anchor.tip_hash)?;
            (
                index,
                LedgerQueryOpenStats {
                    index_rebuilt: true,
                    rows_indexed: anchor.height,
                    rows_reused: 0,
                },
            )
        };
        write_index(&expected, &index)?;
        Ok(Self {
            vault: vault.to_path_buf(),
            index,
            open_stats,
        })
    }

    pub const fn height(&self) -> u64 {
        self.index.height
    }

    pub const fn tip_hash(&self) -> [u8; 32] {
        self.index.tip_hash
    }

    pub const fn open_stats(&self) -> LedgerQueryOpenStats {
        self.open_stats
    }

    pub fn subject_sequences(&self, subject: &SubjectId) -> &[u64] {
        self.index
            .subjects
            .get(&SubjectKey::from_subject(subject))
            .map_or(EMPTY_SEQUENCES, Vec::as_slice)
    }

    pub fn cx_sequences(&self, cx_id: CxId) -> &[u64] {
        self.index
            .cx_mentions
            .get(cx_id.as_bytes())
            .map_or(EMPTY_SEQUENCES, Vec::as_slice)
    }

    pub fn kind_sequences(&self, kind: EntryKind) -> &[u64] {
        self.index
            .kinds
            .get(&kind.wire_code())
            .map_or(EMPTY_SEQUENCES, Vec::as_slice)
    }

    pub fn answer_ids(&self) -> impl Iterator<Item = &[u8]> {
        self.index.answers.keys().map(Vec::as_slice)
    }

    pub fn contains_answer(&self, answer_id: &[u8]) -> bool {
        self.index.answers.contains_key(answer_id)
    }

    pub fn entries_for_subject(&self, subject: &SubjectId) -> Result<Vec<LedgerEntry>> {
        let entries = self.read_entries(self.subject_sequences(subject))?;
        if entries.iter().any(|entry| &entry.subject != subject) {
            return Err(CalyxError::ledger_corrupt(
                "ledger query subject index points at a mismatched entry",
            ));
        }
        Ok(entries)
    }

    pub fn entries_for_subject_bytes(&self, bytes: &[u8]) -> Result<Vec<LedgerEntry>> {
        let seqs: &[u64] = self
            .index
            .subject_bytes
            .get(bytes)
            .map_or(EMPTY_SEQUENCES, Vec::as_slice);
        let entries = self.read_entries(seqs)?;
        if entries
            .iter()
            .any(|entry| subject_bytes(&entry.subject) != bytes)
        {
            return Err(CalyxError::ledger_corrupt(
                "ledger subject-byte index points at a mismatched entry",
            ));
        }
        Ok(entries)
    }

    pub fn entries_for_cx(&self, cx_id: CxId) -> Result<Vec<LedgerEntry>> {
        let entries = self.read_entries(self.cx_sequences(cx_id))?;
        if entries
            .iter()
            .any(|entry| !entry_cx_mentions(entry).contains(&cx_id))
        {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger CX index for {cx_id} points at a mismatched entry"
            )));
        }
        Ok(entries)
    }

    pub fn read_selected(&self, seqs: &BTreeSet<u64>) -> Result<Vec<LedgerEntry>> {
        self.read_entries(&seqs.iter().copied().collect::<Vec<_>>())
    }

    pub fn visit_kind_reverse(
        &self,
        kind: EntryKind,
        batch_size: usize,
        mut visit: impl FnMut(&LedgerEntry) -> Result<bool>,
    ) -> Result<LedgerQueryVisitStats> {
        if batch_size == 0 {
            return Err(CalyxError::ledger_corrupt(
                "ledger query kind batch_size must be > 0",
            ));
        }
        let seqs = self.kind_sequences(kind);
        let mut stats = LedgerQueryVisitStats {
            snapshot_height: self.height(),
            ..LedgerQueryVisitStats::default()
        };
        for chunk in seqs.rchunks(batch_size) {
            let entries = self.read_entries(chunk)?;
            stats.batches_read += 1;
            stats.physical_rows_read = stats
                .physical_rows_read
                .saturating_add(entries.len() as u64)
                .saturating_add(u64::from(self.height() != 0));
            for entry in entries.iter().rev() {
                if entry.kind != kind {
                    return Err(CalyxError::ledger_corrupt(format!(
                        "ledger kind index {:?} points at seq {} kind {:?}",
                        kind, entry.seq, entry.kind
                    )));
                }
                stats.matching_rows_visited += 1;
                if visit(entry)? {
                    return Ok(stats);
                }
            }
        }
        Ok(stats)
    }

    fn read_entries(&self, seqs: &[u64]) -> Result<Vec<LedgerEntry>> {
        if seqs.is_empty() {
            return Ok(Vec::new());
        }
        let mut wanted = seqs.iter().copied().collect::<BTreeSet<_>>();
        if self.height() != 0 {
            wanted.insert(self.height() - 1);
        }
        let (rows, _) = super::read_ledger_seqs_traced(&self.vault, &wanted)?;
        for seq in &wanted {
            if !rows.contains_key(seq) {
                return Err(CalyxError::ledger_chain_broken(format!(
                    "ledger query generation at height {} is missing seq {seq}",
                    self.height()
                )));
            }
        }
        if self.height() != 0 {
            let head = rows.get(&(self.height() - 1)).expect("checked above");
            let decoded = decode_checked(head.seq, &head.bytes)?;
            if decoded.entry_hash != self.tip_hash() {
                return Err(CalyxError::ledger_chain_broken(format!(
                    "ledger query generation head hash mismatch at seq {}",
                    head.seq
                )));
            }
        }
        seqs.iter()
            .map(|seq| {
                let row = rows.get(seq).expect("checked above");
                decode_checked(row.seq, &row.bytes)
            })
            .collect()
    }
}

impl LedgerQueryIndex {
    fn empty() -> Self {
        Self {
            version: INDEX_VERSION,
            height: 0,
            tip_hash: [0; 32],
            entry_count: 0,
            subjects: BTreeMap::new(),
            subject_bytes: BTreeMap::new(),
            cx_mentions: BTreeMap::new(),
            kinds: BTreeMap::new(),
            answers: BTreeMap::new(),
        }
    }

    fn push(&mut self, entry: &LedgerEntry) -> Result<()> {
        if entry.seq != self.height {
            return Err(CalyxError::ledger_chain_broken(format!(
                "ledger query index expected seq {}, found {}",
                self.height, entry.seq
            )));
        }
        self.subjects
            .entry(SubjectKey::from_subject(&entry.subject))
            .or_default()
            .push(entry.seq);
        self.subject_bytes
            .entry(subject_bytes(&entry.subject).to_vec())
            .or_default()
            .push(entry.seq);
        for cx_id in entry_cx_mentions(entry) {
            self.cx_mentions
                .entry(cx_id.to_bytes())
                .or_default()
                .push(entry.seq);
        }
        self.kinds
            .entry(entry.kind.wire_code())
            .or_default()
            .push(entry.seq);
        if entry.kind == EntryKind::Answer
            && let SubjectId::Query(answer_id) = &entry.subject
        {
            self.answers
                .entry(answer_id.clone())
                .or_default()
                .push(entry.seq);
        }
        self.height += 1;
        self.entry_count += 1;
        self.tip_hash = entry.entry_hash;
        Ok(())
    }

    fn validate_generation(&self, height: u64, tip_hash: &[u8; 32]) -> Result<()> {
        self.validate_internal()?;
        if self.height != height || &self.tip_hash != tip_hash {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger query index generation mismatch: index height={} tip={} head height={} tip={}",
                self.height,
                hex(&self.tip_hash),
                height,
                hex(tip_hash)
            )));
        }
        Ok(())
    }

    fn validate_internal(&self) -> Result<()> {
        if self.version != INDEX_VERSION {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger query index version {} is unsupported",
                self.version
            )));
        }
        if self.entry_count != self.height {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger query index count {} does not match height {}",
                self.entry_count, self.height
            )));
        }
        validate_postings("subject", self.subjects.values(), self.height, self.height)?;
        validate_postings(
            "subject bytes",
            self.subject_bytes.values(),
            self.height,
            self.height,
        )?;
        validate_postings("kind", self.kinds.values(), self.height, self.height)?;
        validate_postings(
            "CX mention",
            self.cx_mentions.values(),
            self.height,
            u64::MAX,
        )?;
        validate_postings("answer", self.answers.values(), self.height, u64::MAX)?;
        Ok(())
    }
}

impl SubjectKey {
    fn from_subject(subject: &SubjectId) -> Self {
        match subject {
            SubjectId::Cx(id) => Self {
                tag: 0,
                bytes: id.as_bytes().to_vec(),
            },
            SubjectId::Lens(id) => Self {
                tag: 1,
                bytes: id.as_bytes().to_vec(),
            },
            SubjectId::Kernel(bytes) => Self {
                tag: 2,
                bytes: bytes.clone(),
            },
            SubjectId::Guard(bytes) => Self {
                tag: 3,
                bytes: bytes.clone(),
            },
            SubjectId::Query(bytes) => Self {
                tag: 4,
                bytes: bytes.clone(),
            },
        }
    }
}

fn validate_postings<'a>(
    label: &str,
    postings: impl Iterator<Item = &'a Vec<u64>>,
    height: u64,
    exact_total: u64,
) -> Result<()> {
    let mut total = 0_u64;
    for seqs in postings {
        if seqs.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger query {label} postings are not strictly increasing"
            )));
        }
        if seqs.iter().any(|seq| *seq >= height) {
            return Err(CalyxError::ledger_corrupt(format!(
                "ledger query {label} posting exceeds height {height}"
            )));
        }
        total = total.saturating_add(seqs.len() as u64);
    }
    if exact_total != u64::MAX && total != exact_total {
        return Err(CalyxError::ledger_corrupt(format!(
            "ledger query {label} postings total {total}, expected {exact_total}"
        )));
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests;
