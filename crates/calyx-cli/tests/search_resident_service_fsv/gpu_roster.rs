use super::*;

#[test]
fn search_vault_resident_serves_gpu_roster_and_includes_cpu_opt_in_slot() {
    let root = temp_root("search-resident-mixed-fsv");
    fs::create_dir_all(&root).expect("create mixed search-resident FSV root");
    let template = "search-resident-mixed-fsv";
    let cpu_lenses = write_mixed_algorithmic_catalog(&root, 9, 1);
    let allow_cpu = cpu_lenses.join(",");

    let refused = run_fail(
        Command::new(calyx_exe())
            .arg("panel")
            .arg("template")
            .arg("save")
            .arg("--home")
            .arg(&root)
            .arg("--name")
            .arg(template)
            .arg("--all-current")
            .arg("--modality")
            .arg("text"),
        "save mixed template without CPU opt-in",
    );
    let refused_index_path = root.join("panels").join("templates").join("index.json");
    let refused_index_exists = refused_index_path.exists();
    println!(
        "search_resident_mixed_fsv_edge_no_cpu_opt_in status={:?} stderr={} template_index_exists={}",
        refused.status.code(),
        String::from_utf8_lossy(&refused.stderr),
        refused_index_exists
    );
    assert!(!refused_index_exists);
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("CALYX_PANEL_TEMPLATE_CPU_LENS_REFUSED")
    );

    let save = run_ok(
        Command::new(calyx_exe())
            .env("CALYX_PANEL_ALLOW_CPU_LENS", &allow_cpu)
            .arg("panel")
            .arg("template")
            .arg("save")
            .arg("--home")
            .arg(&root)
            .arg("--name")
            .arg(template)
            .arg("--all-current")
            .arg("--modality")
            .arg("text"),
        "save mixed template with CPU opt-in",
    );
    let create = run_ok(
        Command::new(calyx_exe())
            .env("CALYX_HOME", &root)
            .env("CALYX_PANEL_ALLOW_CPU_LENS", &allow_cpu)
            .arg("create-vault")
            .arg("search-resident-mixed-vault")
            .arg("--panel-template")
            .arg(template),
        "create mixed search-resident vault",
    );
    let create_json: Value = serde_json::from_slice(&create.stdout).expect("parse create-vault");
    let vault_id: VaultId = create_json["vault_id"].as_str().unwrap().parse().unwrap();
    let vault_path = root.join("vaults").join(vault_id.to_string());
    println!("search_resident_mixed_fsv_root={}", root.display());
    println!("search_resident_mixed_fsv_vault={}", vault_path.display());
    let before = cf_state(&vault_path, vault_id, "search-resident-mixed-vault");
    println!("search_resident_mixed_fsv_before={before}");

    let vault_progress = root.join("resident-vault-mixed-progress.jsonl");
    let vault_stderr = root.join("resident-vault-mixed.stderr.log");
    let mut vault_service =
        spawn_vault_resident_service(&root, &vault_path, &vault_progress, &vault_stderr);
    let vault_ready = read_ready(&mut vault_service);
    let vault_addr = vault_ready["bind"].as_str().unwrap().to_string();
    let cpu_excluded = vault_ready["cpu_excluded_slots"]
        .as_array()
        .expect("resident ready cpu_excluded_slots array");
    println!("search_resident_mixed_fsv_vault_ready={vault_ready}");
    assert_eq!(cpu_excluded.len(), 1);
    assert!(cpu_excluded[0].as_str().unwrap().contains("slot=9"));

    let batch = write_batch(
        &root,
        "search-resident-mixed.jsonl",
        &[
            "calyx mixed resident search alpha exact marker 9141",
            "calyx mixed resident search beta distractor 8202",
            "calyx mixed resident search gamma distractor 7303",
        ],
    );
    let ingest = run_ok(
        Command::new(calyx_exe())
            .env("CALYX_HOME", &root)
            .arg("ingest")
            .arg(&vault_path)
            .arg("--batch")
            .arg(&batch)
            .arg("--resident-addr")
            .arg(&vault_addr),
        "ingest mixed search-resident corpus",
    );
    let after_ingest = cf_state(&vault_path, vault_id, "search-resident-mixed-vault");
    println!("search_resident_mixed_fsv_after_ingest={after_ingest}");
    let rebuild = run_ok(
        Command::new(calyx_exe())
            .env("CALYX_HOME", &root)
            .arg("rebuild-search-index")
            .arg(&vault_path),
        "rebuild mixed search-resident index",
    );
    let rebuild_json = parse_json(&rebuild.stdout);
    let rebuild_progress_path = PathBuf::from(
        rebuild_json["progress_artifact"]
            .as_str()
            .expect("rebuild progress artifact path"),
    );
    let rebuild_progress_text = fs::read_to_string(&rebuild_progress_path)
        .expect("read mixed rebuild progress artifact from source of truth");
    let after_rebuild = cf_state(&vault_path, vault_id, "search-resident-mixed-vault");
    let index = search_index_state(&vault_path);
    println!("search_resident_mixed_fsv_after_rebuild={after_rebuild}");
    println!("search_resident_mixed_fsv_index={index}");
    println!("search_resident_mixed_fsv_rebuild_progress={rebuild_progress_text}");

    let search = run_ok(
        Command::new(calyx_exe())
            .env("CALYX_HOME", &root)
            .arg("search")
            .arg(&vault_path)
            .arg("calyx mixed resident search alpha exact marker 9141")
            .arg("--k")
            .arg("1")
            .arg("--explain")
            .arg("--resident-addr")
            .arg(&vault_addr),
        "search mixed vault with resident GPU roster and local CPU slot",
    );
    let after_search = cf_state(&vault_path, vault_id, "search-resident-mixed-vault");
    let search_output: Value = serde_json::from_slice(&search.stdout).expect("parse mixed search");
    println!("search_resident_mixed_fsv_search_stdout={search_output}");
    println!(
        "search_resident_mixed_fsv_search_stderr={}",
        String::from_utf8_lossy(&search.stderr)
    );
    println!("search_resident_mixed_fsv_after_search={after_search}");
    let vault_service_output = stop_resident_service(&vault_addr, vault_service, &vault_stderr);
    let vault_progress_text = fs::read_to_string(&vault_progress).unwrap_or_default();
    println!("search_resident_mixed_fsv_vault_progress_log={vault_progress_text}");

    assert_eq!(before["base_rows"], 0);
    assert_eq!(after_ingest["base_rows"], 3);
    assert_eq!(after_ingest["slot_00_rows"], 3);
    assert_eq!(after_ingest["slot_09_rows"], 3);
    assert_eq!(after_ingest["ledger_rows"], 1);
    assert_eq!(after_search, after_rebuild);
    assert_index_matches_manifest(&index, 10, 3);
    let slots = &search_output["slots"];
    assert_eq!(slots["resident_gpu"].as_array().unwrap().len(), 9);
    assert_eq!(slots["local_cpu"].as_array().unwrap().len(), 1);
    assert_eq!(slots["local_cpu"][0]["slot"], 9);
    let hits = search_output["hits"].as_array().expect("mixed search hits");
    assert_eq!(hits.len(), 1);
    let per_lens = hits[0]["per_lens"].as_array().expect("per-lens explain");
    assert_eq!(per_lens.len(), 10);
    assert!(per_lens.iter().any(|item| item["slot"] == 9));
    let hit_id = hits[0]["cx_id"].as_str().expect("hit cx_id");
    assert!(
        after_search["base_ids"]
            .as_array()
            .unwrap()
            .iter()
            .any(|id| id == hit_id)
    );
    let search_stderr = String::from_utf8_lossy(&search.stderr);
    assert!(search_stderr.contains("phase=search_local_cpu_measure slot=9"));
    assert!(search_stderr.contains("demanded_gpu_slots=9"));
    assert!(search_stderr.contains("local_cpu_slots=1"));
    assert!(search_stderr.contains("phase=search_resident_service_ok"));
    let vault_service_stderr = String::from_utf8_lossy(&vault_service_output.stderr);
    assert!(vault_service_stderr.contains("resident_cpu_lens_excluded"));
    assert!(vault_progress_text.contains("\"phase\":\"resident_cpu_lens_excluded\""));
    assert_eq!(
        vault_progress_text
            .matches("\"phase\":\"probe_ok\"")
            .count(),
        9
    );
    assert!(!vault_progress_text.contains("\"phase\":\"probe_error\""));

    let evidence = json!({
        "source_of_truth": {
            "vault": vault_path,
            "base_ledger_slot_cfs": after_search,
            "search_index": index,
            "resident_ready": vault_ready,
            "evidence_file": root.join("search-resident-mixed-fsv-evidence.json"),
        },
        "commands": {
            "refused_without_cpu_opt_in_stderr": String::from_utf8_lossy(&refused.stderr),
            "template_save_stdout": parse_json(&save.stdout),
            "create_stdout": create_json,
            "ingest_stdout": parse_json(&ingest.stdout),
            "rebuild_stdout": rebuild_json,
            "rebuild_progress_log": rebuild_progress_text,
            "search_stdout": search_output,
            "search_stderr": String::from_utf8_lossy(&search.stderr),
            "vault_service_stderr": vault_service_stderr,
            "vault_progress_log": vault_progress_text,
        },
        "states": {
            "before": before,
            "after_ingest": after_ingest,
            "after_rebuild": after_rebuild,
            "after_search": after_search,
        },
    });
    let evidence_path = root.join("search-resident-mixed-fsv-evidence.json");
    fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).unwrap(),
    )
    .unwrap();
    let evidence_readback: Value =
        serde_json::from_slice(&fs::read(&evidence_path).unwrap()).unwrap();
    println!("search_resident_mixed_fsv_evidence={evidence_readback}");
    assert_eq!(evidence_readback["states"]["after_search"], after_rebuild);
    assert_eq!(
        evidence_readback["source_of_truth"]["resident_ready"]["cpu_excluded_slots"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    if std::env::var("CALYX_KEEP_SEARCH_RESIDENT_FSV_ROOT").as_deref() == Ok("1") {
        println!(
            "search_resident_mixed_fsv_preserved_root={}",
            root.display()
        );
    } else {
        fs::remove_dir_all(root).ok();
    }
}
