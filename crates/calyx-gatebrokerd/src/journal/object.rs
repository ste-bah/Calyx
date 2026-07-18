use rusqlite::{OptionalExtension, TransactionBehavior, params};

use crate::protocol::{ObjectId, RequestId, RunId};

use super::support::{
    corrupt_protocol, identity_fields, now_ms, read_transaction, require_active_run, sql,
    validate_transition,
};
use super::{
    IntentRecord, Journal, JournalError, JournalEvent, TransactionRecord, TransactionState,
    TransitionUpdate,
};

impl Journal {
    pub fn begin_intent(&mut self, intent: &IntentRecord) -> Result<(), JournalError> {
        let now = now_ms()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|source| sql("begin object intent", source))?;
        require_active_run(&transaction, &intent.run_id, "object intent")?;
        transaction.execute(
            "INSERT INTO object_transactions(object_id,request_id,run_id,role,root_alias,leaf,state,created_ms,updated_ms) VALUES(?1,?2,?3,?4,?5,?6,'intent',?7,?7)",
            params![intent.object_id.as_str(), intent.request_id.as_str(), intent.run_id.as_str(), intent.role.as_str(), intent.root_alias.as_str(), intent.leaf.as_str(), now],
        ).map_err(|source| sql("insert object intent", source))?;
        transaction
            .execute(
                "INSERT INTO journal_events(object_id,to_state,at_ms) VALUES(?1,'intent',?2)",
                params![intent.object_id.as_str(), now],
            )
            .map_err(|source| sql("insert intent event", source))?;
        transaction
            .commit()
            .map_err(|source| sql("commit object intent", source))
    }

    pub fn transition(
        &mut self,
        object_id: &ObjectId,
        expected: TransactionState,
        next: TransactionState,
        update: TransitionUpdate,
    ) -> Result<(), JournalError> {
        validate_transition(expected, next, &update)?;
        let now = now_ms()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|source| sql("begin state transition", source))?;
        let current = read_transaction(&transaction, object_id)?
            .ok_or_else(|| JournalError::NotFound(object_id.to_string()))?;
        if current.state != expected {
            return Err(JournalError::InvalidTransition {
                object_id: object_id.to_string(),
                expected,
                actual: current.state,
                next,
            });
        }
        if let (Some(current), Some(replacement)) =
            (current.identity.as_ref(), update.identity.as_ref())
            && (!current.same_authority(replacement)
                || current.device != replacement.device
                || current.inode != replacement.inode)
        {
            return Err(JournalError::InvalidMetadata(format!(
                "object {object_id} identity authority cannot change during a journal transition"
            )));
        }
        if let (Some(current), Some(replacement)) = (
            current.quarantine_name.as_deref(),
            update.quarantine_name.as_deref(),
        ) && current != replacement
        {
            return Err(JournalError::InvalidMetadata(format!(
                "object {object_id} quarantine name is immutable"
            )));
        }
        if let Some(name) = update.quarantine_name.as_deref()
            && name != format!("q-{object_id}")
        {
            return Err(JournalError::InvalidMetadata(format!(
                "object {object_id} quarantine name must be q-{object_id}"
            )));
        }
        let identity = update.identity.as_ref().or(current.identity.as_ref());
        let fields = identity.map(identity_fields);
        let changed = transaction.execute(
            "UPDATE object_transactions SET state=?1,device=?2,inode=?3,owner_uid=?4,owner_gid=?5,mode=?6,mount_id=?7,handle_type=?8,handle=?9,quarantine_name=COALESCE(?10,quarantine_name),error_code=?11,detail=?12,updated_ms=?13 WHERE object_id=?14 AND state=?15",
            params![next.as_db(), fields.as_ref().map(|v| v.0.as_str()), fields.as_ref().map(|v| v.1.as_str()), fields.as_ref().map(|v| v.2), fields.as_ref().map(|v| v.3), fields.as_ref().map(|v| v.4), fields.as_ref().map(|v| v.5), fields.as_ref().map(|v| v.6), fields.as_ref().map(|v| v.7.as_slice()), update.quarantine_name, update.error_code, update.detail, now, object_id.as_str(), expected.as_db()],
        ).map_err(|source| sql("update transaction state", source))?;
        if changed != 1 {
            return Err(JournalError::Durability(
                "compare-and-set transition changed no rows".into(),
            ));
        }
        transaction.execute(
            "INSERT INTO journal_events(object_id,from_state,to_state,error_code,detail,at_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![object_id.as_str(), expected.as_db(), next.as_db(), update.error_code, update.detail, now],
        ).map_err(|source| sql("insert transition event", source))?;
        transaction
            .commit()
            .map_err(|source| sql("commit state transition", source))
    }

    pub fn get(&self, object_id: &ObjectId) -> Result<Option<TransactionRecord>, JournalError> {
        read_transaction(&self.connection, object_id)
    }

    pub fn by_request(
        &self,
        request_id: &RequestId,
    ) -> Result<Option<TransactionRecord>, JournalError> {
        let object: Option<String> = self
            .connection
            .query_row(
                "SELECT object_id FROM object_transactions WHERE request_id=?1",
                [request_id.as_str()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|source| sql("find transaction by request", source))?;
        object
            .map(|value| {
                ObjectId::new(value).map_err(|error| JournalError::Corrupt(error.to_string()))
            })
            .transpose()?
            .map(|id| self.get(&id))
            .transpose()
            .map(Option::flatten)
    }

    pub fn list_incomplete(&self) -> Result<Vec<TransactionRecord>, JournalError> {
        let mut statement = self.connection.prepare(
            "SELECT object_id FROM object_transactions WHERE state IN ('intent','prepared','published','delete_intent','quarantined') ORDER BY created_ms,object_id",
        ).map_err(|source| sql("prepare incomplete query", source))?;
        let ids = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|source| sql("query incomplete transactions", source))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| sql("read incomplete transaction ids", source))?;
        ids.into_iter()
            .map(|id| {
                let id =
                    ObjectId::new(id).map_err(|error| JournalError::Corrupt(error.to_string()))?;
                self.get(&id)?
                    .ok_or_else(|| JournalError::NotFound(id.to_string()))
            })
            .collect()
    }

    pub fn list_objects_for_run(
        &self,
        run_id: &RunId,
    ) -> Result<Vec<TransactionRecord>, JournalError> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT object_id FROM object_transactions WHERE run_id=?1 ORDER BY created_ms,object_id",
            )
            .map_err(|source| sql("prepare run object query", source))?;
        let ids = statement
            .query_map([run_id.as_str()], |row| row.get::<_, String>(0))
            .map_err(|source| sql("query run objects", source))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| sql("read run object ids", source))?;
        ids.into_iter()
            .map(|value| {
                let id = ObjectId::new(value).map_err(corrupt_protocol)?;
                self.get(&id)?
                    .ok_or_else(|| JournalError::Corrupt(format!("run object {id} disappeared")))
            })
            .collect()
    }

    pub fn events(&self, object_id: &ObjectId) -> Result<Vec<JournalEvent>, JournalError> {
        let mut statement = self.connection.prepare(
            "SELECT event_id,from_state,to_state,error_code,detail,at_ms FROM journal_events WHERE object_id=?1 ORDER BY event_id",
        ).map_err(|source| sql("prepare event query", source))?;
        let rows = statement
            .query_map([object_id.as_str()], |row| {
                Ok((
                    row.get(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })
            .map_err(|source| sql("query events", source))?;
        rows.map(|row| {
            let (event_id, from, to, error_code, detail, at_ms) =
                row.map_err(|source| sql("read event", source))?;
            Ok(JournalEvent {
                event_id,
                object_id: object_id.clone(),
                from_state: from.map(|v| TransactionState::parse(&v)).transpose()?,
                to_state: TransactionState::parse(&to)?,
                error_code,
                detail,
                at_ms,
            })
        })
        .collect()
    }
}
