use super::{AsterVault, encode, ledger_hook};
use crate::cf::ColumnFamily;
use crate::manifest::ManifestStore;
#[cfg(test)]
use calyx_core::VaultStore;
use calyx_core::{CalyxError, Clock, LedgerRef, Result};
use calyx_ledger::{ActorId, EntryKind, MAX_UNCLASSIFIED_TOKEN_LEN, RedactionPolicy, SubjectId};
use serde::{Deserialize, Serialize};

pub const CALYX_INGEST_PRECONDITION_FAILED: &str = "CALYX_INGEST_PRECONDITION_FAILED";
pub const CALYX_INGEST_PRECONDITION_INVALID: &str = "CALYX_INGEST_PRECONDITION_INVALID";

const BASE_COUNT_PAGE_SIZE: usize = 8_192;
const CLAIM_FORMAT: &str = "calyx-ingest-precondition-claim-v1";

/// Caller-supplied compare values for an atomic ingest-state claim.
///
/// Every populated field is evaluated while Aster holds the durable commit
/// lock. An empty precondition is rejected by the claim API so callers cannot
/// accidentally present an unguarded ingest as guarded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestPrecondition {
    pub expected_durable_seq: Option<u64>,
    pub expected_manifest_seq: Option<u64>,
    pub expected_base_count: Option<u64>,
}

impl IngestPrecondition {
    pub fn is_empty(&self) -> bool {
        self.expected_durable_seq.is_none()
            && self.expected_manifest_seq.is_none()
            && self.expected_base_count.is_none()
    }
}

/// Physical state observed under Aster's durable commit lock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestVaultState {
    /// Live durable WAL/MVCC tip, including commits newer than the manifest.
    pub durable_seq: u64,
    /// CURRENT manifest generation, or zero before the first manifest.
    pub manifest_seq: u64,
    /// Durable coverage recorded by CURRENT, or zero without a manifest.
    pub manifest_durable_seq: u64,
    /// Visible, non-tombstoned Base rows at `durable_seq`.
    pub base_count: u64,
}

/// Non-sensitive provenance attached to the atomic claim Ledger entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestPreconditionContext {
    pub session_id: String,
    pub batch_sha256: String,
    pub planned_row_count: usize,
}

/// Receipt for the Ledger claim that won the compare-and-claim operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestPreconditionClaim {
    pub format: String,
    pub expected: IngestPrecondition,
    pub observed_before_claim: IngestVaultState,
    pub context: IngestPreconditionContext,
    pub ledger_ref: LedgerRef,
}

impl<C> AsterVault<C>
where
    C: Clock,
{
    /// Atomically compares physical vault state and appends an Ingest Ledger
    /// claim. A mismatch returns before WAL, Base, slot, or Ledger mutation.
    pub fn claim_ingest_precondition(
        &self,
        expected: IngestPrecondition,
        context: IngestPreconditionContext,
    ) -> Result<IngestPreconditionClaim> {
        validate_expected(&expected)?;
        validate_context(&context)?;
        self.with_durable_commit_lock(|| {
            let actual = self.ingest_vault_state_locked()?;
            ensure_matches(&expected, &actual)?;
            let payload = serde_json::to_vec(&serde_json::json!({
                "format": CLAIM_FORMAT,
                "expected": &expected,
                "observed_before_claim": &actual,
                "context": &context,
                "source_of_truth": "Aster durable commit lock + live WAL/MVCC tip + CURRENT manifest + paged Base CF readback",
            }))
            .map_err(|error| invalid(format!("encode ingest precondition claim: {error}")))?;
            RedactionPolicy::check_payload(&payload)?;
            let ledger_ref = self.commit_claim_ledger_locked(payload)?;
            Ok(IngestPreconditionClaim {
                format: CLAIM_FORMAT.to_string(),
                expected,
                observed_before_claim: actual,
                context,
                ledger_ref,
            })
        })
    }

    /// Atomically validates a precondition without mutation. This is intended
    /// for an empty batch, where there is no ingest operation to claim.
    pub fn verify_ingest_precondition(
        &self,
        expected: &IngestPrecondition,
    ) -> Result<IngestVaultState> {
        validate_expected(expected)?;
        self.with_durable_commit_lock(|| {
            let actual = self.ingest_vault_state_locked()?;
            ensure_matches(expected, &actual)?;
            Ok(actual)
        })
    }

    fn ingest_vault_state_locked(&self) -> Result<IngestVaultState> {
        let durable_seq = self.latest_seq();
        let (manifest_seq, manifest_durable_seq) = match &self.durable {
            Some(durable) if durable.root().join("CURRENT").exists() => {
                let manifest = ManifestStore::open(durable.root()).load_current()?;
                (manifest.manifest_seq, manifest.durable_seq)
            }
            _ => (0, 0),
        };
        let snapshot = self.snapshot_handle(durable_seq);
        let mut base_count = 0u64;
        self.rows.scan_cf_pages_at(
            snapshot.snapshot(),
            ColumnFamily::Base,
            BASE_COUNT_PAGE_SIZE,
            &self.clock,
            |page| {
                base_count = base_count.checked_add(page.len() as u64).ok_or_else(|| {
                    CalyxError::aster_corrupt_shard("Base CF count overflow during ingest claim")
                })?;
                Ok::<(), CalyxError>(())
            },
        )?;
        Ok(IngestVaultState {
            durable_seq,
            manifest_seq,
            manifest_durable_seq,
            base_count,
        })
    }

    fn commit_claim_ledger_locked(&self, payload: Vec<u8>) -> Result<LedgerRef> {
        let subject = SubjectId::Guard(self.vault_id.as_ulid().to_bytes().to_vec());
        let actor = ActorId::Service("calyx-cli-ingest-precondition".to_string());
        let Some(hook) = &self.ledger_hook else {
            let mut rows = Vec::<encode::WriteRow>::new();
            let ledger_ref = self.stage_raw_ledger_entry_locked(
                &mut rows,
                EntryKind::Ingest,
                subject,
                payload,
                actor,
            )?;
            self.commit_rows_locked(&rows)?;
            return Ok(ledger_ref);
        };
        let mut guard = ledger_hook::lock_hook(hook)?;
        let staged = guard.stage_with_checkpoints(EntryKind::Ingest, subject, payload, actor)?;
        let ledger_ref = staged
            .first()
            .ok_or_else(|| CalyxError::ledger_group_commit_failed("no staged claim ledger rows"))?
            .ledger_ref();
        let rows = staged
            .iter()
            .map(|row| encode::WriteRow {
                cf: ColumnFamily::Ledger,
                key: row.key().to_vec(),
                value: row.value().to_vec(),
            })
            .collect::<Vec<_>>();
        self.commit_rows_locked(&rows)?;
        for row in &staged {
            guard.commit_staged(row)?;
        }
        Ok(ledger_ref)
    }
}

fn validate_expected(expected: &IngestPrecondition) -> Result<()> {
    if expected.is_empty() {
        return Err(invalid(
            "ingest precondition must specify durable seq, manifest seq, or Base count",
        ));
    }
    Ok(())
}

fn validate_context(context: &IngestPreconditionContext) -> Result<()> {
    let path_safe = !context.session_id.is_empty()
        && context.session_id != "."
        && context.session_id != ".."
        && context
            .session_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !path_safe {
        return Err(invalid(format!(
            "ingest precondition claim session id {:?} is not path-safe ASCII",
            context.session_id,
        )));
    }
    RedactionPolicy::check_public_identifier("session_id", &context.session_id).map_err(
        |error| {
            invalid(format!(
                "ingest precondition claim session id cannot be persisted by the durable Ledger policy: {}; use at most {} characters for a generic path-safe id",
                error.message,
                MAX_UNCLASSIFIED_TOKEN_LEN,
            ))
        },
    )?;
    if context.planned_row_count == 0 {
        return Err(invalid(
            "ingest precondition claim requires a non-empty planned batch",
        ));
    }
    if context.batch_sha256.len() != 64
        || !context
            .batch_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid(
            "ingest precondition claim batch_sha256 must be 64 hexadecimal characters",
        ));
    }
    Ok(())
}

fn ensure_matches(expected: &IngestPrecondition, actual: &IngestVaultState) -> Result<()> {
    let matches = expected
        .expected_durable_seq
        .is_none_or(|value| value == actual.durable_seq)
        && expected
            .expected_manifest_seq
            .is_none_or(|value| value == actual.manifest_seq)
        && expected
            .expected_base_count
            .is_none_or(|value| value == actual.base_count);
    if matches {
        return Ok(());
    }
    let expected_json =
        serde_json::to_string(expected).unwrap_or_else(|error| format!("<encode-error:{error}>"));
    let actual_json =
        serde_json::to_string(actual).unwrap_or_else(|error| format!("<encode-error:{error}>"));
    Err(CalyxError {
        code: CALYX_INGEST_PRECONDITION_FAILED,
        message: format!(
            "atomic ingest precondition did not match physical vault state; expected={expected_json} actual={actual_json}; no claim or ingest mutation was committed"
        ),
        remediation: "read CURRENT/MANIFEST, the live durable sequence, and Base CF count; investigate the intervening writer and start a new ingest session with the accepted physical state",
    })
}

fn invalid(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: CALYX_INGEST_PRECONDITION_INVALID,
        message: message.into(),
        remediation: "supply at least one exact expected vault-state value and a real batch session identity/hash",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cf::ColumnFamily;
    use crate::vault::VaultOptions;
    use calyx_core::VaultId;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn test_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "calyx-ingest-precondition-{label}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("remove stale test vault");
        }
        dir
    }

    fn open(dir: &PathBuf) -> AsterVault {
        let vault_id: VaultId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().expect("vault id");
        AsterVault::open(
            dir,
            vault_id,
            b"real-durable-ingest-claim",
            VaultOptions::default(),
        )
        .expect("open real durable Aster vault")
    }

    fn empty() -> IngestPrecondition {
        IngestPrecondition {
            expected_durable_seq: Some(0),
            expected_manifest_seq: Some(0),
            expected_base_count: Some(0),
        }
    }

    fn context(session: &str) -> IngestPreconditionContext {
        IngestPreconditionContext {
            session_id: session.to_string(),
            batch_sha256: "a".repeat(64),
            planned_row_count: 1,
        }
    }

    fn cf_count(vault: &AsterVault, cf: ColumnFamily) -> usize {
        vault
            .scan_cf_at(vault.snapshot(), cf)
            .expect("scan physical CF")
            .len()
    }

    #[test]
    fn expected_empty_claim_succeeds_once_and_stale_second_handle_is_unchanged() {
        let dir = test_dir("compare-and-claim");
        let first = open(&dir);
        let stale = open(&dir);

        let claim = first
            .claim_ingest_precondition(empty(), context("first-real-session"))
            .expect("first exact-empty claim");
        assert_eq!(claim.observed_before_claim.base_count, 0);
        assert_eq!(claim.observed_before_claim.durable_seq, 0);
        assert_eq!(cf_count(&first, ColumnFamily::Base), 0);
        assert_eq!(cf_count(&first, ColumnFamily::Ledger), 1);

        let ledger_before = cf_count(&first, ColumnFamily::Ledger);
        let error = stale
            .claim_ingest_precondition(empty(), context("stale-real-session"))
            .expect_err("stale exact-empty claim must fail");
        assert_eq!(error.code, CALYX_INGEST_PRECONDITION_FAILED);
        assert_eq!(cf_count(&stale, ColumnFamily::Base), 0);
        assert_eq!(cf_count(&stale, ColumnFamily::Ledger), ledger_before);
        drop(first);
        drop(stale);
        fs::remove_dir_all(dir).expect("remove test vault");
    }

    #[test]
    fn mismatch_and_invalid_context_leave_every_physical_cf_unchanged() {
        let dir = test_dir("no-mutation");
        let vault = open(&dir);
        let before_seq = vault.snapshot();
        let before_base = cf_count(&vault, ColumnFamily::Base);
        let before_ledger = cf_count(&vault, ColumnFamily::Ledger);

        let mismatch = IngestPrecondition {
            expected_durable_seq: Some(0),
            expected_manifest_seq: Some(0),
            expected_base_count: Some(1),
        };
        let error = vault
            .claim_ingest_precondition(mismatch, context("mismatch-real-session"))
            .expect_err("wrong Base count must fail");
        assert_eq!(error.code, CALYX_INGEST_PRECONDITION_FAILED);

        let mut invalid_context = context("invalid-real-session");
        invalid_context.batch_sha256 = "not-a-sha256".to_string();
        let error = vault
            .claim_ingest_precondition(empty(), invalid_context)
            .expect_err("invalid provenance must fail");
        assert_eq!(error.code, CALYX_INGEST_PRECONDITION_INVALID);

        let too_long = context(&("a".repeat(MAX_UNCLASSIFIED_TOKEN_LEN - 1) + "-x"));
        let error = vault
            .claim_ingest_precondition(empty(), too_long)
            .expect_err("session id at the Ledger token boundary must fail before claim");
        assert_eq!(error.code, CALYX_INGEST_PRECONDITION_INVALID);
        assert!(error.message.contains("use at most 39 characters"));

        let invalid_characters = context("not/a/path-safe-session");
        let error = vault
            .claim_ingest_precondition(empty(), invalid_characters)
            .expect_err("non-path-safe session id must fail before claim");
        assert_eq!(error.code, CALYX_INGEST_PRECONDITION_INVALID);
        assert_eq!(vault.snapshot(), before_seq);
        assert_eq!(cf_count(&vault, ColumnFamily::Base), before_base);
        assert_eq!(cf_count(&vault, ColumnFamily::Ledger), before_ledger);
        drop(vault);
        fs::remove_dir_all(dir).expect("remove test vault");
    }

    #[test]
    fn empty_batch_validation_is_atomic_and_has_no_ledger_side_effect() {
        let dir = test_dir("verify-only");
        let vault = open(&dir);
        let state = vault
            .verify_ingest_precondition(&empty())
            .expect("verify exact empty state");
        assert_eq!(state.base_count, 0);
        assert_eq!(state.durable_seq, 0);
        assert_eq!(vault.snapshot(), 0);
        assert_eq!(cf_count(&vault, ColumnFamily::Ledger), 0);
        drop(vault);
        fs::remove_dir_all(dir).expect("remove test vault");
    }
}
