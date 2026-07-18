use std::error::Error;
use std::path::PathBuf;

use calyx_aster::ledger_view::AsterLedgerCfStore;
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{LedgerRef, VaultId};
use calyx_ledger::{
    ActorId, EntryKind, LedgerCfStore, SubjectId, VerifyResult, decode, verify_chain,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let root = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: issue1856_atomic_answer_fsv <new-vault-dir>")?;
    let failure_only = args.next().as_deref() == Some(std::ffi::OsStr::new("--failure-only"));
    if root.exists() {
        return Err(format!("FSV target already exists: {}", root.display()).into());
    }
    let vault_id: VaultId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse()?;
    let vault = AsterVault::new_durable(&root, vault_id, [0x56; 32], VaultOptions::default())?;
    let answer_id = vec![0x18; 32];
    let subject = SubjectId::Query(answer_id.clone());
    let actor = ActorId::Service("issue1856-manual-fsv".to_string());
    let prefix = (0..3_u32)
        .map(|hop_index| {
            Ok((
                EntryKind::Answer,
                subject.clone(),
                serde_json::to_vec(&json!({
                    "type": "kernel_answer_hop_v1",
                    "answer_id": hex(&answer_id),
                    "hop_index": hop_index,
                    "from_id": format!("{:032x}", hop_index + 1),
                    "to_id": format!("{:032x}", hop_index + 2),
                    "edge_weight": 0.9,
                    "hop_score": 0.8,
                }))?,
                actor.clone(),
            ))
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    let failure = vault
        .append_ledger_entries_with_final(prefix.clone(), |_| {
            Ok((
                EntryKind::Answer,
                subject.clone(),
                br#"{"type":"kernel_answer_v2","api_key":"sk-forced-late-completion-failure"}"#
                    .to_vec(),
                actor.clone(),
            ))
        })
        .expect_err("secret-bearing completion must fail closed");
    let after_failure_store = AsterLedgerCfStore::open(&root)?;
    let rows_after_failure = after_failure_store.scan()?;
    if !rows_after_failure.is_empty() {
        return Err(format!(
            "atomic failure leaked {} physical ledger rows",
            rows_after_failure.len()
        )
        .into());
    }
    if failure_only {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": "ok",
                "mode": "failure_only",
                "vault_dir": root,
                "answer_id": hex(&answer_id),
                "forced_failure": {
                    "code": failure.code,
                    "message": failure.message,
                    "remediation": failure.remediation,
                    "visible_rows_after_failure": rows_after_failure.len(),
                },
            }))?
        );
        return Ok(());
    }

    let refs = vault.append_ledger_entries_with_final(prefix, |hop_refs| {
        let path = hop_refs
            .iter()
            .enumerate()
            .map(|(hop_index, reference)| {
                json!({
                    "hop_index": hop_index,
                    "ledger_ref": ledger_ref_json(reference),
                })
            })
            .collect::<Vec<_>>();
        Ok((
            EntryKind::Answer,
            subject,
            serde_json::to_vec(&json!({
                "type": "kernel_answer_v2",
                "answer_id": hex(&answer_id),
                "complete": true,
                "expected_hops": hop_refs.len(),
                "path": path,
            }))
            .map_err(|error| {
                calyx_core::CalyxError::ledger_group_commit_failed(error.to_string())
            })?,
            actor,
        ))
    })?;
    let physical = AsterLedgerCfStore::open(&root)?;
    let rows = physical.scan()?;
    let verification = verify_chain(&physical, 0..rows.len() as u64)?;
    let decoded = rows
        .iter()
        .map(|row| {
            let entry = decode(&row.bytes)?;
            let payload: serde_json::Value = serde_json::from_slice(&entry.payload)?;
            Ok(json!({
                "seq": entry.seq,
                "kind": entry.kind.as_str(),
                "prev_hash": hex(&entry.prev_hash),
                "entry_hash": hex(&entry.entry_hash),
                "payload_type": payload["type"],
                "complete": payload.get("complete"),
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    if refs.len() != 4
        || rows.len() != 4
        || !matches!(verification, VerifyResult::Intact { count: 4 })
    {
        return Err(format!(
            "successful atomic publication mismatch refs={} rows={} verification={verification:?}",
            refs.len(),
            rows.len()
        )
        .into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "ok",
            "vault_dir": root,
            "forced_failure": {
                "code": failure.code,
                "message": failure.message,
                "remediation": failure.remediation,
                "visible_rows_after_failure": rows_after_failure.len(),
            },
            "successful_publication": {
                "returned_refs": refs,
                "physical_rows": rows.len(),
                "verify_chain": format!("{verification:?}"),
                "decoded_rows": decoded,
            },
        }))?
    );
    Ok(())
}

fn ledger_ref_json(reference: &LedgerRef) -> serde_json::Value {
    json!({
        "seq": reference.seq,
        "hash": hex(&reference.hash),
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
