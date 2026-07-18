use std::path::Path;
use std::time::Duration;

use rusqlite::OpenFlags;

use crate::protocol::{ObjectId, RequestId, RunId, StageId};

use super::support::{
    configure, corrupt_protocol, initialize_schema, preflight_schema, prepare_journal_file,
    query_text_column, sql, sync_file, sync_parent, transaction_transition_allowed,
    validate_journal_file, validate_journal_path, validate_schema, validate_stage_event_chain,
};
use super::{JOURNAL_APPLICATION_ID, Journal, JournalError};

impl Journal {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, JournalError> {
        let path = path.as_ref();
        validate_journal_path(path)?;
        let created = prepare_journal_file(path)?;
        if created {
            sync_parent(path)?;
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_FULL_MUTEX;
        let connection = rusqlite::Connection::open_with_flags(path, flags)
            .map_err(|source| sql("open", source))?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|source| sql("set busy timeout", source))?;
        preflight_schema(&connection)?;
        configure(&connection)?;
        initialize_schema(&connection)?;
        validate_journal_file(path)?;
        let journal = Self {
            path: path.to_path_buf(),
            connection,
        };
        journal.verify_durability_settings()?;
        journal.integrity_check()?;
        Ok(journal)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn verify_durability_settings(&self) -> Result<(), JournalError> {
        let mode: String = self
            .connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .map_err(|source| sql("read journal_mode", source))?;
        let synchronous: i64 = self
            .connection
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(|source| sql("read synchronous", source))?;
        let foreign_keys: i64 = self
            .connection
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .map_err(|source| sql("read foreign_keys", source))?;
        let trusted_schema: i64 = self
            .connection
            .query_row("PRAGMA trusted_schema", [], |row| row.get(0))
            .map_err(|source| sql("read trusted_schema", source))?;
        let fullfsync: i64 = self
            .connection
            .query_row("PRAGMA fullfsync", [], |row| row.get(0))
            .map_err(|source| sql("read fullfsync", source))?;
        let checkpoint_fullfsync: i64 = self
            .connection
            .query_row("PRAGMA checkpoint_fullfsync", [], |row| row.get(0))
            .map_err(|source| sql("read checkpoint_fullfsync", source))?;
        let application_id: i64 = self
            .connection
            .query_row("PRAGMA application_id", [], |row| row.get(0))
            .map_err(|source| sql("read application_id", source))?;
        if !mode.eq_ignore_ascii_case("wal")
            || synchronous != 2
            || foreign_keys != 1
            || trusted_schema != 0
            || fullfsync != 1
            || checkpoint_fullfsync != 1
            || application_id != JOURNAL_APPLICATION_ID
        {
            return Err(JournalError::Durability(format!(
                "expected WAL/FULL/foreign_keys=ON/trusted_schema=OFF/fullfsync=ON/checkpoint_fullfsync=ON/application_id={JOURNAL_APPLICATION_ID}; got {mode}/{synchronous}/{foreign_keys}/{trusted_schema}/{fullfsync}/{checkpoint_fullfsync}/{application_id}"
            )));
        }
        Ok(())
    }

    pub fn integrity_check(&self) -> Result<(), JournalError> {
        let result: String = self
            .connection
            .query_row("PRAGMA integrity_check(1)", [], |row| row.get(0))
            .map_err(|source| sql("integrity_check", source))?;
        if result != "ok" {
            return Err(JournalError::Corrupt(format!(
                "integrity_check returned {result:?}"
            )));
        }
        validate_schema(&self.connection)?;
        self.semantic_integrity_check()
    }

    fn semantic_integrity_check(&self) -> Result<(), JournalError> {
        for value in query_text_column(
            &self.connection,
            "SELECT run_id FROM runs ORDER BY run_id",
            "scan run ids",
        )? {
            let id = RunId::new(value).map_err(corrupt_protocol)?;
            self.get_run(&id)?
                .ok_or_else(|| JournalError::Corrupt(format!("run {id} disappeared")))?;
        }
        for value in query_text_column(
            &self.connection,
            "SELECT object_id FROM object_transactions ORDER BY object_id",
            "scan object ids",
        )? {
            let id = ObjectId::new(value).map_err(corrupt_protocol)?;
            let record = self
                .get(&id)?
                .ok_or_else(|| JournalError::Corrupt(format!("object {id} disappeared")))?;
            let events = self.events(&id)?;
            let mut previous = None;
            for event in &events {
                if event.from_state != previous {
                    return Err(JournalError::Corrupt(format!(
                        "object {id} event {} starts at {:?}, expected {:?}",
                        event.event_id, event.from_state, previous
                    )));
                }
                if let Some(from) = event.from_state
                    && !transaction_transition_allowed(from, event.to_state)
                {
                    return Err(JournalError::Corrupt(format!(
                        "object {id} event {} records forbidden transition {from:?} -> {:?}",
                        event.event_id, event.to_state
                    )));
                }
                previous = Some(event.to_state);
            }
            if previous != Some(record.state) {
                return Err(JournalError::Corrupt(format!(
                    "object {id} event chain ends at {previous:?}, row is {:?}",
                    record.state
                )));
            }
        }
        for value in query_text_column(
            &self.connection,
            "SELECT stage_id FROM stages ORDER BY stage_id",
            "scan stage ids",
        )? {
            let id = StageId::new(value).map_err(corrupt_protocol)?;
            let record = self
                .get_stage(&id)?
                .ok_or_else(|| JournalError::Corrupt(format!("stage {id} disappeared")))?;
            validate_stage_event_chain(&self.connection, &record)?;
        }
        for value in query_text_column(
            &self.connection,
            "SELECT request_id FROM operations ORDER BY request_id",
            "scan operation ids",
        )? {
            let id = RequestId::new(value).map_err(corrupt_protocol)?;
            self.get_operation(&id)?
                .ok_or_else(|| JournalError::Corrupt(format!("operation {id} disappeared")))?;
        }
        Ok(())
    }

    pub fn checkpoint(&self) -> Result<(), JournalError> {
        let (busy, log, checkpointed): (i64, i64, i64) = self
            .connection
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|source| sql("WAL checkpoint", source))?;
        if busy != 0 || log != checkpointed {
            return Err(JournalError::Durability(format!(
                "WAL checkpoint incomplete: busy={busy}, log={log}, checkpointed={checkpointed}"
            )));
        }
        sync_file(&self.path)
    }
}
