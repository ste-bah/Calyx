use calyx_anneal::{
    AnnealLedger, AnnealLedgerAction, AnnealLedgerEntry, AsterAnnealLedgerStore,
    CALYX_ASTER_CF_UNAVAILABLE, CALYX_LEDGER_ENTRY_TOO_LARGE, ChangeId, MetricComparison,
    MetricSnapshot, TripwireMetric,
};
use calyx_aster::cf::{ColumnFamily, ledger_key};
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{CalyxError, FixedClock, Result, VaultId};
use calyx_ledger::{
    ActorId, EntryKind, LedgerAppender, LedgerCfStore, LedgerEntry, LedgerRow, MemoryLedgerStore,
    SubjectId, decode as decode_ledger,
};
use proptest::prelude::*;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

const TEST_TS: u64 = 1_785_500_398;

#[test]
fn promote_entry_roundtrips_from_ledger_payload() {
    let mut ledger = memory_ledger();
    let entry = sample_entry(ChangeId(101), AnnealLedgerAction::Promote, Some([0; 32]));

    let reference = ledger.write(entry.clone()).expect("write promote");
    let readback = ledger.read_recent_with_refs(1).expect("read recent");

    assert_eq!(readback.len(), 1);
    assert_eq!(readback[0].ledger_ref, reference);
    assert_eq!(readback[0].entry, entry);
}

#[test]
fn promote_revert_read_in_order_and_find_by_change_id() {
    let mut ledger = memory_ledger();
    let promote = sample_entry(ChangeId(201), AnnealLedgerAction::Promote, Some([0; 32]));
    let revert = sample_entry(ChangeId(202), AnnealLedgerAction::Revert, None);

    let first_ref = ledger.write(promote.clone()).expect("write promote");
    let mut expected_revert = revert.clone();
    expected_revert.prev_hash = Some(first_ref.hash);
    let second_ref = ledger.write(revert).expect("write revert");

    let recent = ledger.read_recent_with_refs(2).expect("read recent");
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].entry, promote);
    assert_eq!(recent[1].entry, expected_revert);
    assert!(recent[0].ledger_ref.seq < recent[1].ledger_ref.seq);
    assert_eq!(
        ledger.find_by_change_id(ChangeId(201)).unwrap(),
        Some(recent[0].entry.clone())
    );
    assert_eq!(
        ledger.find_by_change_id_with_ref(ChangeId(202)).unwrap(),
        Some(recent[1].clone())
    );
    assert_eq!(recent[1].ledger_ref, second_ref);
}

#[test]
fn repeated_change_lookup_returns_latest_event() {
    let mut ledger = memory_ledger();
    let change_id = ChangeId(303);
    ledger
        .write(sample_entry(
            change_id,
            AnnealLedgerAction::Promote,
            Some([0; 32]),
        ))
        .expect("write promote");
    ledger
        .write(sample_entry(change_id, AnnealLedgerAction::Revert, None))
        .expect("write revert");

    let found = ledger
        .find_by_change_id(change_id)
        .expect("lookup")
        .expect("entry");

    assert_eq!(found.action, AnnealLedgerAction::Revert);
}

#[test]
fn empty_description_and_empty_read_are_allowed() {
    let mut ledger = memory_ledger();
    assert_eq!(ledger.read_recent(10).unwrap(), Vec::new());

    let mut entry = sample_entry(ChangeId(404), AnnealLedgerAction::Park, Some([0; 32]));
    entry.description.clear();

    ledger
        .write(entry.clone())
        .expect("write empty description");

    assert_eq!(ledger.read_recent(1).unwrap(), vec![entry]);
}

#[test]
fn cf_unavailable_error_propagates() {
    let appender = LedgerAppender::open(FailingStore, FixedClock::new(TEST_TS)).unwrap();
    let mut ledger =
        AnnealLedger::new(appender, ActorId::Service("calyx-anneal-test".to_string())).unwrap();

    let error = ledger
        .write(sample_entry(
            ChangeId(505),
            AnnealLedgerAction::Propose,
            Some([0; 32]),
        ))
        .unwrap_err();

    assert_eq!(error.code, CALYX_ASTER_CF_UNAVAILABLE);
}

#[test]
fn oversized_payload_fails_closed_without_truncation() {
    let mut ledger = memory_ledger();
    let mut entry = sample_entry(
        ChangeId(606),
        AnnealLedgerAction::Recalibrate,
        Some([0; 32]),
    );
    entry.description = "oversized ".repeat(4096);

    let error = ledger.write(entry).unwrap_err();

    assert_eq!(error.code, CALYX_LEDGER_ENTRY_TOO_LARGE);
    assert_eq!(ledger.read_recent(10).unwrap(), Vec::new());
}

#[test]
fn mismatched_prev_hash_fails_closed() {
    let mut ledger = memory_ledger();
    let error = ledger
        .write(sample_entry(
            ChangeId(707),
            AnnealLedgerAction::MistakeUpdate,
            Some([9; 32]),
        ))
        .unwrap_err();

    assert_eq!(error.code, "CALYX_LEDGER_CHAIN_BROKEN");
    assert_eq!(ledger.read_recent(10).unwrap(), Vec::new());
}

#[test]
fn aster_anneal_append_keeps_live_vault_ledger_hook_in_sync() {
    let root = test_vault_dir("issue1571-anneal-vault-hook");
    let vault = AsterVault::new_durable(
        &root,
        vault_id(),
        b"issue1571-anneal-vault-hook",
        VaultOptions::default(),
    )
    .expect("open durable vault");
    let before = physical_ledger_entries(&vault);
    println!("ISSUE1571_BEFORE rows={}", before.len());

    let mut anneal = aster_anneal_ledger(&vault);
    anneal
        .write(sample_entry(
            ChangeId(1_571_001),
            AnnealLedgerAction::Promote,
            Some([0; 32]),
        ))
        .expect("anneal append first row");
    vault
        .append_ledger_entry(
            EntryKind::Admin,
            SubjectId::Query(b"issue1571-vault-append".to_vec()),
            b"vault append after anneal".to_vec(),
            ActorId::Service("calyx-aster-test".to_string()),
        )
        .expect("vault append after anneal must use refreshed hook");
    anneal
        .write(sample_entry(
            ChangeId(1_571_002),
            AnnealLedgerAction::Revert,
            None,
        ))
        .expect("same anneal appender append after vault row");
    vault.flush().expect("flush issue1571 rows");

    let after = physical_ledger_entries(&vault);
    println!(
        "ISSUE1571_AFTER rows={} seqs={:?} hashes={:?}",
        after.len(),
        after.iter().map(|entry| entry.seq).collect::<Vec<_>>(),
        after
            .iter()
            .map(|entry| entry.entry_hash)
            .collect::<Vec<_>>()
    );
    assert_eq!(after.len(), 3);
    for (idx, entry) in after.iter().enumerate() {
        assert_eq!(entry.seq, idx as u64);
        if idx == 0 {
            assert_eq!(entry.prev_hash, [0; 32]);
        } else {
            assert_eq!(entry.prev_hash, after[idx - 1].entry_hash);
        }
    }

    fs::remove_dir_all(root).ok();
}

proptest! {
    #![proptest_config(calyx_testkit::integration_proptest_config(32))]

    #[test]
    fn written_entries_remain_in_insertion_order(actions in prop::collection::vec(action_strategy(), 1..32)) {
        let mut ledger = memory_ledger();
        let mut expected_ids = Vec::new();

        for (index, action) in actions.into_iter().enumerate() {
            let change_id = ChangeId(10_000 + index as u64);
            ledger.write(sample_entry(change_id, action, None)).unwrap();
            expected_ids.push(change_id);
        }

        let readback = ledger.read_recent_with_refs(usize::MAX).unwrap();
        let actual_ids = readback
            .iter()
            .map(|entry| entry.entry.change_id)
            .collect::<Vec<_>>();
        prop_assert_eq!(actual_ids, expected_ids);
        prop_assert!(
            readback
                .windows(2)
                .all(|pair| pair[0].ledger_ref.seq < pair[1].ledger_ref.seq)
        );
    }
}

struct FailingStore;

impl LedgerCfStore for FailingStore {
    fn scan(&self) -> Result<Vec<LedgerRow>> {
        Ok(Vec::new())
    }

    fn put_new(&mut self, _seq: u64, _bytes: &[u8]) -> Result<()> {
        Err(CalyxError {
            code: CALYX_ASTER_CF_UNAVAILABLE,
            message: "injected ledger CF outage".to_string(),
            remediation: "restore Aster ledger CF availability",
        })
    }
}

fn memory_ledger() -> AnnealLedger<MemoryLedgerStore, FixedClock> {
    let appender =
        LedgerAppender::open(MemoryLedgerStore::default(), FixedClock::new(TEST_TS)).unwrap();
    AnnealLedger::new(appender, ActorId::Service("calyx-anneal-test".to_string())).unwrap()
}

fn aster_anneal_ledger(
    vault: &AsterVault,
) -> AnnealLedger<AsterAnnealLedgerStore<'_, calyx_core::SystemClock>, FixedClock> {
    let store = AsterAnnealLedgerStore::new(vault);
    let appender = LedgerAppender::open(store, FixedClock::new(TEST_TS)).unwrap();
    AnnealLedger::new(appender, ActorId::Service("calyx-anneal-test".to_string())).unwrap()
}

fn physical_ledger_entries(vault: &AsterVault) -> Vec<LedgerEntry> {
    vault
        .scan_cf_at(vault.latest_seq(), ColumnFamily::Ledger)
        .expect("scan physical Ledger CF")
        .into_iter()
        .map(|(key, bytes)| {
            let entry = decode_ledger(&bytes).expect("decode physical ledger entry");
            assert_eq!(key, ledger_key(entry.seq));
            entry
        })
        .collect()
}

fn test_vault_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "calyx-anneal-{name}-{}-{nanos}",
        std::process::id()
    ))
}

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
}

fn sample_entry(
    change_id: ChangeId,
    action: AnnealLedgerAction,
    prev_hash: Option<[u8; 32]>,
) -> AnnealLedgerEntry {
    AnnealLedgerEntry {
        action,
        change_id,
        artifact_id: format!("artifact-{}", change_id.0),
        prior_ptr_hash: [1; 32],
        candidate_ptr_hash: [2; 32],
        metrics: MetricSnapshot {
            evaluated_at: TEST_TS,
            query_count: 8,
            metrics: vec![MetricComparison {
                metric: TripwireMetric::RecallAtK,
                candidate_value: 0.91,
                incumbent_value: 0.89,
            }],
        },
        ts: TEST_TS,
        description: "synthetic anneal ledger event".to_string(),
        fault: None,
        proposal: None,
        details: None,
        prev_hash,
    }
}

fn action_strategy() -> impl Strategy<Value = AnnealLedgerAction> {
    prop_oneof![
        Just(AnnealLedgerAction::Promote),
        Just(AnnealLedgerAction::Revert),
        Just(AnnealLedgerAction::Propose),
        Just(AnnealLedgerAction::LensAdmitted),
        Just(AnnealLedgerAction::LensRejected),
        Just(AnnealLedgerAction::Park),
        Just(AnnealLedgerAction::DegradeChange),
        Just(AnnealLedgerAction::FaultEvent),
        Just(AnnealLedgerAction::Rebuild),
        Just(AnnealLedgerAction::BaseCorruptAlert),
        Just(AnnealLedgerAction::BaseRestored),
        Just(AnnealLedgerAction::Recalibrate),
        Just(AnnealLedgerAction::TauRecalibrated),
        Just(AnnealLedgerAction::TauRecalibrationReverted),
        Just(AnnealLedgerAction::LensPark),
        Just(AnnealLedgerAction::LensUnpark),
        Just(AnnealLedgerAction::MistakeUpdate),
        Just(AnnealLedgerAction::HeadUpdate),
        Just(AnnealLedgerAction::HeadUpdateReverted),
        Just(AnnealLedgerAction::OperatorPromoted),
        Just(AnnealLedgerAction::OperatorReverted),
        Just(AnnealLedgerAction::SleepPassDeferred),
        Just(AnnealLedgerAction::OutcomeReward),
        Just(AnnealLedgerAction::OutcomeContradiction),
        Just(AnnealLedgerAction::AutotuneAB),
        Just(AnnealLedgerAction::AutotunePromote),
        Just(AnnealLedgerAction::GoodhartPassed),
        Just(AnnealLedgerAction::GoodhartFailed),
    ]
}
