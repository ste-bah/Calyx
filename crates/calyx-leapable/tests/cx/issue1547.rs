use super::*;

#[test]
#[ignore = "manual full-state verification for issues #1547 and #1549"]
fn issue1547_put_outcome_stdio_fsv() {
    let fsv_root = std::env::var_os("CALYX_FSV_ROOT")
        .map(PathBuf::from)
        .expect("set CALYX_FSV_ROOT to a writable evidence directory");
    fs::create_dir_all(&fsv_root).expect("create FSV root");
    let root = fsv_root.join("issue1547-leapable");
    fs::remove_dir_all(&root).ok();
    fs::create_dir_all(&root).expect("create Leapable FSV root");

    let create = request(
        1,
        "vault.create",
        json!({"vault_ref": "outcomes", "ts": 1_785_547_000_000_u64}),
    );
    let create_responses = invoke(&create, &root);
    assert!(create_responses[0].get("error").is_none());

    let empty = [
        request(
            2,
            "vault.open",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_001_000_u64}),
        ),
        request(
            3,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_001_100_u64, "limit": 10}),
        ),
        request(
            4,
            "cx.put_batch",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_001_200_u64, "items": []}),
        ),
        request(
            5,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_001_300_u64, "limit": 10}),
        ),
    ]
    .concat();
    let empty_responses = invoke(&empty, &root);
    assert_eq!(
        empty_responses[1]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(empty_responses[2]["result"]["count"], 0);
    assert_eq!(
        empty_responses[3]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    println!(
        "ISSUE1547_EDGE_EMPTY before_rows=0 outcome_count=0 after_rows=0 before_snapshot={} after_snapshot={}",
        empty_responses[1]["result"]["snapshot"], empty_responses[3]["result"]["snapshot"]
    );

    let text = "issue 1547 known duplicate";
    let duplicate_batch = [
        request(
            6,
            "vault.open",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_002_000_u64}),
        ),
        request(
            7,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_002_100_u64, "limit": 10}),
        ),
        request(
            8,
            "cx.put_batch",
            json!({
                "vault_ref": "outcomes",
                "ts": 1_785_547_002_200_u64,
                "items": [put_item(text, "first-payload"), put_item(text, "later-payload")]
            }),
        ),
        request(
            9,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_002_300_u64, "limit": 10}),
        ),
    ]
    .concat();
    let duplicate_responses = invoke(&duplicate_batch, &root);
    let outcomes = duplicate_responses[2]["result"]["items"]
        .as_array()
        .expect("batch outcomes");
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0]["deduped"], false);
    assert_eq!(outcomes[1]["deduped"], true);
    assert_eq!(outcomes[0]["cx_id"], outcomes[1]["cx_id"]);
    let cx_id = outcomes[0]["cx_id"].as_str().unwrap().to_string();
    assert_eq!(
        duplicate_responses[1]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        duplicate_responses[3]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    println!(
        "ISSUE1547_HAPPY before_rows=0 submitted=2 inserted=1 deduped=1 after_rows=1 cx_id={cx_id}"
    );

    let invalid = [
        request(
            10,
            "vault.open",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_003_000_u64}),
        ),
        request(
            11,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_003_100_u64, "limit": 10}),
        ),
        request(
            12,
            "cx.put",
            json!({
                "vault_ref": "outcomes",
                "ts": 1_785_547_003_200_u64,
                "panel_version": 7,
                "modality": "text",
                "input": {"text": "reserved derived state"},
                "scalars": {"recurrence.frequency": 99.0}
            }),
        ),
        request(
            13,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_003_300_u64, "limit": 10}),
        ),
    ]
    .concat();
    let invalid_responses = invoke(&invalid, &root);
    assert_calyx_code(&invalid_responses[2], "CALYX_LEAPABLE_CX_INPUT_INVALID");
    assert_eq!(
        invalid_responses[1]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        invalid_responses[3]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        invalid_responses[1]["result"]["snapshot"],
        invalid_responses[3]["result"]["snapshot"]
    );
    println!(
        "ISSUE1547_EDGE_RESERVED before_rows=1 error=CALYX_LEAPABLE_CX_INPUT_INVALID after_rows=1 before_snapshot={} after_snapshot={}",
        invalid_responses[1]["result"]["snapshot"], invalid_responses[3]["result"]["snapshot"]
    );

    let cold_read = [
        request(
            14,
            "vault.open",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_004_000_u64}),
        ),
        request(
            15,
            "cx.scan",
            json!({"vault_ref": "outcomes", "ts": 1_785_547_004_100_u64, "limit": 10}),
        ),
        request(
            16,
            "cx.get",
            json!({
                "vault_ref": "outcomes",
                "ts": 1_785_547_004_200_u64,
                "cx_id": cx_id,
                "include_input": true
            }),
        ),
    ]
    .concat();
    let cold_responses = invoke(&cold_read, &root);
    assert_eq!(
        cold_responses[1]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let stored = &cold_responses[2]["result"]["item"];
    assert_eq!(stored["input_text"], text);
    assert_eq!(
        stored["constellation"]["metadata"]["chunk_id"],
        "first-payload"
    );
    assert_eq!(
        stored["constellation"]["input_ref"]["pointer"],
        "leapable://first-payload"
    );

    let vault_dir = storage_dir(&root, "outcomes");
    let inventory = physical_inventory(&vault_dir);
    assert!(vault_dir.join("cf/base").exists());
    assert!(vault_dir.join("cf/leapable").exists());
    assert!(vault_dir.join("cf/ledger").exists());
    assert!(!wal_files(&vault_dir).is_empty());
    let evidence = json!({
        "issue": 1547,
        "source_of_truth": vault_dir.display().to_string(),
        "empty_edge": empty_responses,
        "duplicate_batch": duplicate_responses,
        "invalid_reserved_scalar": invalid_responses,
        "cold_reopen_readback": cold_responses,
        "physical_files": inventory,
    });
    let evidence_path = fsv_root.join("issue1547-leapable-readback.json");
    fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .expect("write issue 1547 evidence");
    let reread: Value = serde_json::from_slice(&fs::read(&evidence_path).unwrap()).unwrap();
    assert_eq!(
        reread["cold_reopen_readback"][1]["result"]["items"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    println!("ISSUE1547_FSV_EVIDENCE={}", evidence_path.display());
}

fn invoke(input: &str, root: &Path) -> Vec<Value> {
    let (stdout, stderr, ok) = run_engine(input, root);
    assert!(ok, "Leapable process failed: {stderr}");
    assert!(
        !stderr.contains('{'),
        "stderr must not contain protocol JSON: {stderr}"
    );
    json_lines(&stdout)
}

fn physical_inventory(root: &Path) -> Vec<Value> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("read physical state directory") {
            let path = entry.expect("physical state entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let bytes = fs::read(&path).expect("read physical state file");
                files.push(json!({
                    "path": path.strip_prefix(root).unwrap().to_string_lossy(),
                    "bytes": bytes.len(),
                    "blake3": blake3::hash(&bytes).to_hex().to_string(),
                }));
            }
        }
    }
    files.sort_by_key(|row| row["path"].as_str().unwrap_or_default().to_string());
    files
}
