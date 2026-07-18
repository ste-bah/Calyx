//! Group-commit hook for adding Ledger rows to a storage write batch.

use calyx_core::{CalyxError, Clock, LedgerRef, Result};

use crate::append::{LedgerAppender, LedgerCfStore, MemoryLedgerStore, PreparedLedgerEntry};
use crate::checkpoint::{CheckpointConfig, CheckpointScheduler, OverlayLedgerStore};
use crate::entry::{ActorId, SubjectId};
use crate::kind::EntryKind;

/// Storage batch surface required by the Ledger group-commit hook.
pub trait LedgerWriteBatch {
    fn put_ledger_row(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()>;
}

/// Legacy direct hook surface kept crate-private so storage engines cannot
/// advance Ledger state outside their durable staged commit path.
#[allow(dead_code)] // #652: retained only to fail closed if crate-local legacy code calls it.
pub(crate) trait LedgerGroupCommitHook: Send + Sync {
    fn on_commit(
        &mut self,
        batch: &mut dyn LedgerWriteBatch,
        kind: EntryKind,
        subject: SubjectId,
        payload: Vec<u8>,
        actor: ActorId,
    ) -> Result<LedgerRef>;
}

/// In-memory batch used by unit tests and adapters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteBatch {
    ledger_rows: Vec<LedgerBatchRow>,
}

impl WriteBatch {
    pub fn ledger_rows(&self) -> &[LedgerBatchRow] {
        &self.ledger_rows
    }
}

impl LedgerWriteBatch for WriteBatch {
    fn put_ledger_row(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.ledger_rows.push(LedgerBatchRow { key, value });
        Ok(())
    }
}

/// One ledger row staged into a group-commit batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerBatchRow {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// One ledger row prepared for a storage batch but not yet committed to the appender tip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedLedgerRow {
    key: Vec<u8>,
    value: Vec<u8>,
    ledger_ref: LedgerRef,
    prepared: PreparedLedgerEntry,
    checkpoint_range_end: Option<u64>,
}

impl StagedLedgerRow {
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    pub fn value(&self) -> &[u8] {
        &self.value
    }

    pub fn ledger_ref(&self) -> LedgerRef {
        self.ledger_ref.clone()
    }
}

/// Default hook backed by a `LedgerAppender`.
#[derive(Debug)]
pub struct DefaultLedgerHook<S = MemoryLedgerStore, C = calyx_core::SystemClock> {
    appender: LedgerAppender<S, C>,
    checkpoint: Option<CheckpointScheduler>,
}

impl<S, C> DefaultLedgerHook<S, C>
where
    S: LedgerCfStore,
    C: Clock,
{
    pub const fn new(appender: LedgerAppender<S, C>) -> Self {
        Self {
            appender,
            checkpoint: None,
        }
    }

    pub fn with_checkpoint_config(
        appender: LedgerAppender<S, C>,
        config: CheckpointConfig,
    ) -> Result<Self> {
        let checkpoint = CheckpointScheduler::recover(config, appender.store())?;
        Ok(Self {
            appender,
            checkpoint: Some(checkpoint),
        })
    }

    pub const fn with_checkpoint_scheduler(
        appender: LedgerAppender<S, C>,
        checkpoint: CheckpointScheduler,
    ) -> Self {
        Self {
            appender,
            checkpoint: Some(checkpoint),
        }
    }

    pub const fn appender(&self) -> &LedgerAppender<S, C> {
        &self.appender
    }

    pub const fn checkpoint(&self) -> Option<&CheckpointScheduler> {
        self.checkpoint.as_ref()
    }

    pub fn stage(
        &self,
        kind: EntryKind,
        subject: SubjectId,
        payload: Vec<u8>,
        actor: ActorId,
    ) -> Result<StagedLedgerRow> {
        let prepared = self
            .appender
            .prepare(kind, subject, payload, actor)
            .map_err(group_commit_failed)?;
        Ok(StagedLedgerRow {
            key: ledger_batch_key(prepared.seq()),
            value: prepared.bytes().to_vec(),
            ledger_ref: prepared.ledger_ref(),
            prepared,
            checkpoint_range_end: None,
        })
    }

    pub fn stage_with_checkpoints(
        &self,
        kind: EntryKind,
        subject: SubjectId,
        payload: Vec<u8>,
        actor: ActorId,
    ) -> Result<Vec<StagedLedgerRow>> {
        let first = self.stage(kind, subject, payload, actor)?;
        let range_end = first
            .ledger_ref
            .seq
            .checked_add(1)
            .ok_or_else(|| CalyxError::ledger_chain_broken("ledger sequence exhausted"))?;
        let mut staged = vec![first];

        if let Some(checkpoint) = &self.checkpoint
            && checkpoint.should_checkpoint(range_end)
        {
            let overlay = OverlayLedgerStore::new(
                self.appender.store(),
                staged.iter().map(|row| row.prepared.clone()),
            )?;
            let predecessor = &staged.last().expect("staged data row").prepared;
            let prepared = checkpoint.prepare_checkpoint_after(
                &self.appender,
                &overlay,
                predecessor,
                range_end,
            )?;
            staged.push(StagedLedgerRow {
                key: ledger_batch_key(prepared.seq()),
                value: prepared.bytes().to_vec(),
                ledger_ref: prepared.ledger_ref(),
                prepared,
                checkpoint_range_end: Some(range_end),
            });
        }

        Ok(staged)
    }

    /// Stages an ordered set of ledger entries as one uncommitted chain.
    ///
    /// Every entry is prepared after the prior staged data/checkpoint row, so
    /// callers can place all returned rows in one storage group commit without
    /// duplicate sequence numbers or advancing the in-memory ledger tip early.
    pub fn stage_many_with_checkpoints<I>(&self, entries: I) -> Result<Vec<StagedLedgerRow>>
    where
        I: IntoIterator<Item = (EntryKind, SubjectId, Vec<u8>, ActorId)>,
    {
        let mut staged = Vec::<StagedLedgerRow>::new();
        let mut checkpoint = self.checkpoint.clone();
        for (kind, subject, payload, actor) in entries {
            self.stage_one_with_checkpoint(
                &mut staged,
                &mut checkpoint,
                kind,
                subject,
                payload,
                actor,
            )?;
        }
        Ok(staged)
    }

    /// Stages prefix entries and one ref-dependent final entry as a single
    /// uncommitted chain. The final builder runs after all prefix refs are
    /// known but before any storage mutation, which lets callers seal an
    /// atomic completion record that references every staged data row.
    pub fn stage_many_with_checkpoints_and_final<I, F>(
        &self,
        entries: I,
        final_entry: F,
    ) -> Result<(Vec<StagedLedgerRow>, Vec<LedgerRef>)>
    where
        I: IntoIterator<Item = (EntryKind, SubjectId, Vec<u8>, ActorId)>,
        F: FnOnce(&[LedgerRef]) -> Result<(EntryKind, SubjectId, Vec<u8>, ActorId)>,
    {
        let mut staged = Vec::<StagedLedgerRow>::new();
        let mut checkpoint = self.checkpoint.clone();
        let mut data_refs = Vec::<LedgerRef>::new();
        for (kind, subject, payload, actor) in entries {
            data_refs.push(self.stage_one_with_checkpoint(
                &mut staged,
                &mut checkpoint,
                kind,
                subject,
                payload,
                actor,
            )?);
        }
        let (kind, subject, payload, actor) = final_entry(&data_refs)?;
        data_refs.push(self.stage_one_with_checkpoint(
            &mut staged,
            &mut checkpoint,
            kind,
            subject,
            payload,
            actor,
        )?);
        Ok((staged, data_refs))
    }

    fn stage_one_with_checkpoint(
        &self,
        staged: &mut Vec<StagedLedgerRow>,
        checkpoint: &mut Option<CheckpointScheduler>,
        kind: EntryKind,
        subject: SubjectId,
        payload: Vec<u8>,
        actor: ActorId,
    ) -> Result<LedgerRef> {
        let prepared = match staged.last() {
            Some(predecessor) => {
                self.appender
                    .prepare_after(&predecessor.prepared, kind, subject, payload, actor)?
            }
            None => self.appender.prepare(kind, subject, payload, actor)?,
        };
        let range_end = prepared
            .seq()
            .checked_add(1)
            .ok_or_else(|| CalyxError::ledger_chain_broken("ledger sequence exhausted"))?;
        let data_ref = prepared.ledger_ref();
        staged.push(StagedLedgerRow {
            key: ledger_batch_key(prepared.seq()),
            value: prepared.bytes().to_vec(),
            ledger_ref: data_ref.clone(),
            prepared,
            checkpoint_range_end: None,
        });

        if let Some(scheduler) = checkpoint.as_mut()
            && scheduler.should_checkpoint(range_end)
        {
            let overlay = OverlayLedgerStore::new(
                self.appender.store(),
                staged.iter().map(|row| row.prepared.clone()),
            )?;
            let predecessor = &staged.last().expect("staged data row").prepared;
            let prepared = scheduler.prepare_checkpoint_after(
                &self.appender,
                &overlay,
                predecessor,
                range_end,
            )?;
            staged.push(StagedLedgerRow {
                key: ledger_batch_key(prepared.seq()),
                value: prepared.bytes().to_vec(),
                ledger_ref: prepared.ledger_ref(),
                prepared,
                checkpoint_range_end: Some(range_end),
            });
            scheduler.advance_after_checkpoint(range_end)?;
        }
        Ok(data_ref)
    }

    pub fn commit_staged(&mut self, staged: &StagedLedgerRow) -> Result<LedgerRef> {
        let ledger_ref = self
            .appender
            .commit_prepared(&staged.prepared)
            .map_err(group_commit_failed)?;
        if let (Some(checkpoint), Some(range_end)) =
            (self.checkpoint.as_mut(), staged.checkpoint_range_end)
        {
            checkpoint
                .advance_after_checkpoint(range_end)
                .map_err(group_commit_failed)?;
        }
        Ok(ledger_ref)
    }
}

impl<S, C> LedgerGroupCommitHook for DefaultLedgerHook<S, C>
where
    S: LedgerCfStore + Send + Sync,
    C: Clock + Send + Sync,
{
    fn on_commit(
        &mut self,
        _batch: &mut dyn LedgerWriteBatch,
        _kind: EntryKind,
        _subject: SubjectId,
        _payload: Vec<u8>,
        _actor: ActorId,
    ) -> Result<LedgerRef> {
        Err(group_commit_failed(
            "direct LedgerGroupCommitHook::on_commit is disabled; use \
             stage_with_checkpoints and commit_staged after durable storage commit",
        ))
    }
}

/// Big-endian ledger CF key; must match Aster `ledger_key`.
pub fn ledger_batch_key(seq: u64) -> Vec<u8> {
    seq.to_be_bytes().to_vec()
}

/// Storage operation categories mapped to ledger entry kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteOp {
    Ingest,
    VaultAdmin,
    Erase,
}

pub const fn ingest_kind_for(op: WriteOp) -> EntryKind {
    match op {
        WriteOp::Ingest => EntryKind::Ingest,
        WriteOp::VaultAdmin => EntryKind::Admin,
        WriteOp::Erase => EntryKind::Erase,
    }
}

fn group_commit_failed(message: impl ToString) -> CalyxError {
    CalyxError::ledger_group_commit_failed(message.to_string())
}

#[cfg(test)]
mod tests;
