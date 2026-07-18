use super::*;

#[test]
fn nonempty_vault_missing_manifest_fails_loud() {
    let env = TestEnv::new("generation-missing");
    let server = server();
    let created = call_ok(&server, 1, "calyx.create_vault", json!({"name": "v"}));
    call_ok(
        &server,
        2,
        "calyx.add_lens",
        json!({"vault": "v", "name": "byte_axis", "runtime": "algorithmic"}),
    );
    call_ok(
        &server,
        3,
        "calyx.ingest",
        json!({"vault": "v", "input": "alpha"}),
    );
    let manifest = env
        .vault_path(created["vault_id"].as_str().unwrap())
        .join("idx/search/manifest.json");
    fs::remove_file(&manifest).unwrap();

    let error = call_err(
        &server,
        4,
        "calyx.search",
        json!({"vault": "v", "query": "alpha"}),
    );
    assert_eq!(error.code, -32000);
    assert_eq!(error.data.unwrap()["calyx_code"], "CALYX_STALE_DERIVED");
    assert!(error.message.contains("manifest missing"));
}

#[test]
fn corrupt_manifest_fails_loud_without_rebuild() {
    let env = TestEnv::new("generation-corrupt");
    let server = server();
    let created = call_ok(&server, 1, "calyx.create_vault", json!({"name": "v"}));
    call_ok(
        &server,
        2,
        "calyx.add_lens",
        json!({"vault": "v", "name": "byte_axis", "runtime": "algorithmic"}),
    );
    call_ok(
        &server,
        3,
        "calyx.ingest",
        json!({"vault": "v", "input": "alpha"}),
    );
    let manifest = env
        .vault_path(created["vault_id"].as_str().unwrap())
        .join("idx/search/manifest.json");
    fs::write(&manifest, b"{not-json").unwrap();

    let error = call_err(
        &server,
        4,
        "calyx.search",
        json!({"vault": "v", "query": "alpha"}),
    );
    assert_eq!(error.code, -32000);
    assert_eq!(error.data.unwrap()["calyx_code"], "CALYX_STALE_DERIVED");
    assert!(error.message.contains("persistent search I/O failure"));
    assert_eq!(fs::read(&manifest).unwrap(), b"{not-json");
}

#[test]
fn fresh_search_rejects_stale_generation_and_stale_ok_tags_hits() {
    let env = TestEnv::new("generation-stale");
    let server = server();
    let created = call_ok(&server, 1, "calyx.create_vault", json!({"name": "v"}));
    call_ok(
        &server,
        2,
        "calyx.add_lens",
        json!({"vault": "v", "name": "byte_axis", "runtime": "algorithmic"}),
    );
    call_ok(
        &server,
        3,
        "calyx.ingest",
        json!({"vault": "v", "input": "alpha"}),
    );
    let vault_id = created["vault_id"].as_str().unwrap();
    let vault_path = env.vault_path(vault_id);
    let parsed_id = vault_id.parse().unwrap();
    let vault = AsterVault::open(
        &vault_path,
        parsed_id,
        crate::tools::vault::store::vault_salt(parsed_id, "v"),
        VaultOptions::default(),
    )
    .unwrap();
    vault
        .write_cf(ColumnFamily::Guard, b"issue1513-stale".to_vec(), vec![1])
        .unwrap();
    vault.flush().unwrap();

    let error = call_err(
        &server,
        4,
        "calyx.search",
        json!({"vault": "v", "query": "alpha"}),
    );
    assert_eq!(error.data.unwrap()["calyx_code"], "CALYX_STALE_DERIVED");
    assert!(error.message.contains("behind derived content seq"));

    let stale = call_ok(
        &server,
        5,
        "calyx.search",
        json!({"vault": "v", "query": "alpha", "fresh": false}),
    );
    assert!(stale["hits"][0]["freshness"]["stale_by"].as_u64().unwrap() > 0);
    assert_eq!(stale["execution"]["request_index_builds"], 0);
}
