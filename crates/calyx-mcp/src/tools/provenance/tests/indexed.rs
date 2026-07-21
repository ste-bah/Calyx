use super::*;

#[test]
fn answer_trace_scans_home_and_returns_retrieval_steps() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (root, _resolved, vault) = test_vault("answer-trace");
    let old_home = std::env::var_os("CALYX_HOME");
    unsafe {
        std::env::set_var("CALYX_HOME", &root);
    }
    let cx_id = CxId::from_bytes([9; 16]);
    let answer_id = b"answer-523".to_vec();
    let payload = json!({
        "complete": true,
        "expected_hops": 1,
        "path": [{
            "hop": 0,
            "cx_id": cx_id.to_string(),
            "score": 0.75,
            "ledger_ref": {"seq": 0}
        }],
        "fusion_weights": FusionWeights {
            mode: FusionMode::Rrf,
            k: 1,
            candidates: vec![cx_id],
            weights: Vec::new(),
            single_slot: None,
        },
    });
    vault
        .append_ledger_entry(
            EntryKind::Answer,
            SubjectId::Query(answer_id.clone()),
            serde_json::to_vec(&payload).unwrap(),
            ActorId::Service("unit".to_string()),
        )
        .unwrap();
    vault.flush().unwrap();

    for seed in [0x31, 0x32] {
        let unrelated_id = VaultId::from_ulid(Ulid::from_bytes([seed; 16]));
        let unrelated_path = root.join("vaults").join(unrelated_id.to_string());
        let unrelated = AsterVault::new_durable(
            &unrelated_path,
            unrelated_id,
            vec![seed],
            VaultOptions {
                panel: Some(panel()),
                ..VaultOptions::default()
            },
        )
        .unwrap();
        for row in 0..32_u64 {
            unrelated
                .append_ledger_entry(
                    EntryKind::Admin,
                    SubjectId::Query(format!("unrelated-{seed}-{row}").into_bytes()),
                    b"not an answer".to_vec(),
                    ActorId::System,
                )
                .unwrap();
        }
        unrelated
            .append_ledger_entry(
                EntryKind::Answer,
                SubjectId::Query(format!("directory-answer-{seed}").into_bytes()),
                serde_json::to_vec(&json!({"complete": false})).unwrap(),
                ActorId::System,
            )
            .unwrap();
    }

    let vault_root = root.join("vaults");
    let (candidates, first_stats) =
        super::super::answer_directory::resolve_answer_vaults(&vault_root, &answer_id).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(first_stats.directory_rebuilt);
    assert_eq!(first_stats.vaults_opened, 3);
    assert_eq!(first_stats.candidates, 1);
    let (cached_candidates, second_stats) =
        super::super::answer_directory::resolve_answer_vaults(&vault_root, &answer_id).unwrap();
    assert_eq!(cached_candidates, candidates);
    assert!(!second_stats.directory_rebuilt);
    assert_eq!(second_stats.vaults_opened, 0);
    assert_eq!(second_stats.candidates, 1);
    for seed in [0x31, 0x32] {
        let id = format!("directory-answer-{seed}");
        let (paths, stats) =
            super::super::answer_directory::resolve_answer_vaults(&vault_root, id.as_bytes())
                .unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(stats.vaults_opened, 0);
    }
    let (missing, missing_stats) =
        super::super::answer_directory::resolve_answer_vaults(&vault_root, b"missing-answer")
            .unwrap();
    assert!(missing.is_empty());
    assert_eq!(missing_stats.vaults_opened, 0);

    let out = serde_json::to_value(core::answer_trace("answer-523").unwrap()).unwrap();
    assert_eq!(out["answer_id"], core::hex(&answer_id));
    assert_eq!(out["complete"], true);
    assert_eq!(out["trusted"], true);
    assert_eq!(out["answer_seq"], 0);
    assert_eq!(out["retrieval_steps"][0]["cx_id"], cx_id.to_string());
    assert_eq!(out["kernel_cx_ids"][0], cx_id.to_string());
    restore_home(old_home);
    fs::remove_dir_all(root).ok();
}

#[test]
fn answer_trace_rejects_duplicate_authoritative_vaults() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (root, _resolved, first) = test_vault("answer-trace-duplicate-first");
    let old_home = std::env::var_os("CALYX_HOME");
    unsafe {
        std::env::set_var("CALYX_HOME", &root);
    }
    let answer_id = b"answer-duplicate".to_vec();
    first
        .append_ledger_entry(
            EntryKind::Answer,
            SubjectId::Query(answer_id.clone()),
            serde_json::to_vec(&json!({"complete": true})).unwrap(),
            ActorId::System,
        )
        .unwrap();
    first.flush().unwrap();

    let second_id = VaultId::from_ulid(Ulid::from_bytes([0xD2; 16]));
    let second_path = root.join("vaults").join(second_id.to_string());
    let second = AsterVault::new_durable(
        &second_path,
        second_id,
        b"answer-trace-duplicate-second".to_vec(),
        VaultOptions {
            panel: Some(panel()),
            ..VaultOptions::default()
        },
    )
    .unwrap();
    second
        .append_ledger_entry(
            EntryKind::Answer,
            SubjectId::Query(answer_id.clone()),
            serde_json::to_vec(&json!({"complete": true})).unwrap(),
            ActorId::System,
        )
        .unwrap();
    second.flush().unwrap();

    let vault_root = root.join("vaults");
    let (paths, stats) =
        super::super::answer_directory::resolve_answer_vaults(&vault_root, &answer_id).unwrap();
    assert_eq!(paths.len(), 2);
    assert_eq!(stats.candidates, 2);
    assert!(paths.iter().all(|path| path.exists()));
    let error = core::answer_trace("answer-duplicate").unwrap_err();
    assert_eq!(err_code(&error), "CALYX_LEDGER_CORRUPT");
    let message = tool_error_message(&error);
    assert!(message.contains("ambiguous across 2 authoritative vaults"));
    assert!(message.contains(&paths[0].display().to_string()));
    assert!(message.contains(&paths[1].display().to_string()));

    restore_home(old_home);
    fs::remove_dir_all(root).ok();
}

#[test]
#[ignore = "manual full-state verification for issue #1532"]
fn issue1532_manual_fsv_uses_one_bounded_consistent_anneal_snapshot() {
    let artifact = std::env::var_os("CALYX_ISSUE1532_FSV_ARTIFACT")
        .map(std::path::PathBuf::from)
        .expect("set CALYX_ISSUE1532_FSV_ARTIFACT to a fresh JSON path");
    assert!(!artifact.exists(), "artifact path must be fresh");
    let (root, resolved, vault) = test_vault("issue1532-manual");

    let (empty, empty_p99, empty_stats) = status::anneal_ledger_status(&resolved.path).unwrap();
    assert!(empty.is_empty() && empty_p99.is_none());
    assert_eq!(empty_stats.matching_rows_visited, 0);
    let appender = LedgerAppender::open(
        AsterAnnealLedgerStore::new(&vault),
        FixedClock::new(1_532_000),
    )
    .unwrap();
    let mut ledger = AnnealLedger::new(appender, ActorId::System).unwrap();
    ledger.write(anneal_event(0, Some(73.5))).unwrap();
    let (one, one_p99, one_stats) = status::anneal_ledger_status(&resolved.path).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one_p99, Some(73.5));
    assert_eq!(one_stats.matching_rows_visited, 1);

    for unrelated in 0..300_u64 {
        ledger
            .appender_mut()
            .append(
                EntryKind::Admin,
                SubjectId::Query(format!("issue1532-unrelated-{unrelated}").into_bytes()),
                b"unrelated non-JSON payload".to_vec(),
                ActorId::System,
            )
            .unwrap();
    }
    for change in 1..=17u64 {
        ledger.write(anneal_event(change, None)).unwrap();
    }
    let (recent, p99, stats) = status::anneal_ledger_status(&resolved.path).unwrap();
    assert_eq!(recent.len(), 16);
    assert_eq!(p99, Some(73.5));
    assert_eq!(stats.snapshot_height, 318);
    assert_eq!(stats.matching_rows_visited, 18);
    assert_eq!(stats.batches_read, 1);
    assert!(stats.physical_rows_read <= 19);
    let wire =
        serde_json::to_value(status::anneal_status_for_resolved(&resolved).unwrap()).unwrap();
    assert_eq!(wire["recent_changes"].as_array().unwrap().len(), 16);
    assert_eq!(wire["p99_latency_ms"], 73.5);
    let physical = AsterLedgerCfStore::open(&resolved.path)
        .unwrap()
        .scan()
        .unwrap();
    assert_eq!(physical.len(), 318);
    let report = json!({
        "issue": 1532,
        "source_of_truth": resolved.path.display().to_string(),
        "edge_empty": {"rows_visited": empty_stats.matching_rows_visited, "physical_rows_read": empty_stats.physical_rows_read, "recent": empty.len(), "p99": empty_p99},
        "edge_one": {"rows_visited": one_stats.matching_rows_visited, "physical_rows_read": one_stats.physical_rows_read, "recent": one.len(), "p99": one_p99},
        "edge_p99_older_than_last16": {
            "snapshot_height": stats.snapshot_height,
            "batches_read": stats.batches_read,
            "rows_visited": stats.matching_rows_visited,
            "physical_rows_read": stats.physical_rows_read,
            "recent": recent.len(),
            "p99": p99,
        },
        "physical_ledger_rows": physical.len(),
        "query_index_files": fs::read_dir(resolved.path.join("ledger_query_index"))
            .unwrap()
            .map(|entry| entry.unwrap().path().display().to_string())
            .collect::<Vec<_>>(),
        "wire": wire,
    });
    if let Some(parent) = artifact.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&artifact, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    let readback: Value = serde_json::from_slice(&fs::read(&artifact).unwrap()).unwrap();
    assert_eq!(readback["physical_ledger_rows"], 318);
    println!(
        "ISSUE1532_FSV={}",
        serde_json::to_string(&readback).unwrap()
    );
    let _ = root;
}
