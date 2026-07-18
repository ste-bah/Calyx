use rusqlite::{Connection, OptionalExtension};

use crate::fs_tx::{ObjectIdentity, OpaqueHandle};
use crate::protocol::{
    LeafName, ObjectId, ProfileName, RequestId, RoleName, RootAlias, RunId, RunToken,
};

use super::super::stage::stage_transition_allowed;
use super::super::{
    IntentRecord, JournalError, RunIntent, RunRecord, RunState, StageRecord, StageState,
    TransactionRecord, TransactionState,
};
use super::util::{corrupt_protocol, corrupt_sql, parse_u64, sql};

pub(in crate::journal) fn query_text_column(
    connection: &Connection,
    query: &'static str,
    operation: &'static str,
) -> Result<Vec<String>, JournalError> {
    let mut statement = connection
        .prepare(query)
        .map_err(|source| sql(operation, source))?;
    statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|source| sql(operation, source))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| sql(operation, source))
}

pub(in crate::journal) fn validate_stage_event_chain(
    connection: &Connection,
    record: &StageRecord,
) -> Result<(), JournalError> {
    let mut statement = connection
        .prepare(
            "SELECT event_id,from_state,to_state FROM stage_events WHERE stage_id=?1 ORDER BY event_id",
        )
        .map_err(|source| sql("prepare stage event integrity query", source))?;
    let events = statement
        .query_map([record.intent.stage_id.as_str()], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|source| sql("query stage event integrity", source))?;
    let mut previous = None;
    for event in events {
        let (event_id, from, to) =
            event.map_err(|source| sql("read stage event integrity", source))?;
        let from = from.map(|value| StageState::parse(&value)).transpose()?;
        let to = StageState::parse(&to)?;
        if from != previous {
            return Err(JournalError::Corrupt(format!(
                "stage {} event {event_id} starts at {from:?}, expected {previous:?}",
                record.intent.stage_id
            )));
        }
        if let Some(from) = from
            && !stage_transition_allowed(from, to)
        {
            return Err(JournalError::Corrupt(format!(
                "stage {} event {event_id} records forbidden transition {from:?} -> {to:?}",
                record.intent.stage_id
            )));
        }
        previous = Some(to);
    }
    if previous != Some(record.state) {
        return Err(JournalError::Corrupt(format!(
            "stage {} event chain ends at {previous:?}, row is {:?}",
            record.intent.stage_id, record.state
        )));
    }
    Ok(())
}

pub(in crate::journal) fn read_transaction(
    connection: &Connection,
    object_id: &ObjectId,
) -> Result<Option<TransactionRecord>, JournalError> {
    connection.query_row(
        "SELECT request_id,run_id,role,root_alias,leaf,state,device,inode,owner_uid,owner_gid,mode,mount_id,handle_type,handle,quarantine_name,error_code,detail,created_ms,updated_ms FROM object_transactions WHERE object_id=?1",
        [object_id.as_str()], |row| transaction_from_row(object_id.clone(), row),
    ).optional().map_err(|source| sql("read transaction", source))?.transpose()
}

pub(in crate::journal) fn transaction_from_row(
    object_id: ObjectId,
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<Result<TransactionRecord, JournalError>> {
    let request_id = RequestId::new(row.get::<_, String>(0)?);
    let run_id = RunId::new(row.get::<_, String>(1)?);
    let role = RoleName::new(row.get::<_, String>(2)?);
    let root_alias = RootAlias::new(row.get::<_, String>(3)?);
    let leaf = LeafName::new(row.get::<_, String>(4)?);
    let state = TransactionState::parse(&row.get::<_, String>(5)?);
    let device: Option<String> = row.get(6)?;
    let identity = if let Some(device) = device {
        let inode: String = row.get(7)?;
        let opaque = OpaqueHandle::new(row.get(11)?, row.get(12)?, row.get(13)?)
            .map_err(|error| JournalError::Corrupt(error.to_string()));
        Some(opaque.and_then(|opaque| {
            Ok(ObjectIdentity {
                device: parse_u64(&device, "device")?,
                inode: parse_u64(&inode, "inode")?,
                owner_uid: row
                    .get(8)
                    .map_err(|e| JournalError::Corrupt(e.to_string()))?,
                owner_gid: row
                    .get(9)
                    .map_err(|e| JournalError::Corrupt(e.to_string()))?,
                mode: row
                    .get(10)
                    .map_err(|e| JournalError::Corrupt(e.to_string()))?,
                opaque,
            })
        }))
    } else {
        None
    };
    Ok((|| {
        let record = TransactionRecord {
            intent: IntentRecord {
                object_id,
                request_id: request_id.map_err(corrupt_protocol)?,
                run_id: run_id.map_err(corrupt_protocol)?,
                role: role.map_err(corrupt_protocol)?,
                root_alias: root_alias.map_err(corrupt_protocol)?,
                leaf: leaf.map_err(corrupt_protocol)?,
            },
            state: state?,
            identity: identity.transpose()?,
            quarantine_name: row.get(14).map_err(corrupt_sql)?,
            error_code: row.get(15).map_err(corrupt_sql)?,
            detail: row.get(16).map_err(corrupt_sql)?,
            created_ms: row.get(17).map_err(corrupt_sql)?,
            updated_ms: row.get(18).map_err(corrupt_sql)?,
        };
        validate_transaction_record(&record)?;
        Ok(record)
    })())
}

fn validate_transaction_record(record: &TransactionRecord) -> Result<(), JournalError> {
    let has_identity = record.identity.is_some();
    let has_quarantine = record.quarantine_name.is_some();
    let has_error = record.error_code.is_some();
    let valid = match record.state {
        TransactionState::Intent => !has_identity && !has_quarantine && !has_error,
        TransactionState::Prepared
        | TransactionState::Published
        | TransactionState::Committed
        | TransactionState::DeleteIntent => has_identity && !has_error,
        TransactionState::Quarantined | TransactionState::Deleted => {
            has_identity && has_quarantine && !has_error
        }
        TransactionState::MismatchPreserved => has_identity && has_error,
        TransactionState::Failed => !has_identity && !has_quarantine && has_error,
    };
    if !valid {
        return Err(JournalError::Corrupt(format!(
            "object {} state {:?} has inconsistent identity, quarantine, or error metadata",
            record.intent.object_id, record.state
        )));
    }
    if let Some(name) = record.quarantine_name.as_deref()
        && name != format!("q-{}", record.intent.object_id)
    {
        return Err(JournalError::Corrupt(format!(
            "object {} has unexpected quarantine name {name:?}",
            record.intent.object_id
        )));
    }
    Ok(())
}

pub(in crate::journal) fn run_from_row(
    run_id: RunId,
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<Result<RunRecord, JournalError>> {
    let request = RequestId::new(row.get::<_, String>(0)?);
    let token = RunToken::new(row.get::<_, String>(1)?);
    let profile = ProfileName::new(row.get::<_, String>(2)?);
    let starttime: String = row.get(5)?;
    let state: String = row.get(6)?;
    Ok((|| {
        Ok(RunRecord {
            intent: RunIntent {
                run_id,
                request_id: request.map_err(corrupt_protocol)?,
                run_token: token.map_err(corrupt_protocol)?,
                profile: profile.map_err(corrupt_protocol)?,
                owner_uid: row.get(3).map_err(corrupt_sql)?,
                owner_pid: row.get(4).map_err(corrupt_sql)?,
                owner_starttime: parse_u64(&starttime, "owner_starttime")?,
            },
            state: RunState::parse(&state)?,
            created_ms: row.get(7).map_err(corrupt_sql)?,
            updated_ms: row.get(8).map_err(corrupt_sql)?,
            detail: row.get(9).map_err(corrupt_sql)?,
        })
    })())
}
