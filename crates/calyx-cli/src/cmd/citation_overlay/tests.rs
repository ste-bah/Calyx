use std::fs;
use std::path::{Path, PathBuf};

use calyx_aster::plain_graph::PhysicalPlainGraph;
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{CxId, VaultId};
use serde_json::{Value, json};
use ulid::Ulid;

use super::{
    DEFAULT_COLLECTION, EDGE_TYPE, MaterializeCitationOverlayArgs, materialize_with_home,
    parse_materialize_citation_overlay, preflight_report_paths,
};
use crate::cmd::Subcommand;
use crate::cmd::vault::vault_salt;

#[test]
fn parses_required_and_optional_flags() {
    let args = [
        "legal-cuyahoga",
        "--idmap",
        "/fsv/idmap.csv",
        "--citations",
        "/fsv/citations_cuyahoga.csv",
        "--collection",
        "legal-citations-v1",
        "--skip-report",
        "/fsv/skips.json",
        "--report",
        "/fsv/report.json",
        "--home",
        "/home/calyx",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let parsed = parse_materialize_citation_overlay(&args).expect("parse command");
    assert_eq!(
        parsed,
        Subcommand::MaterializeCitationOverlay(MaterializeCitationOverlayArgs {
            vault: "legal-cuyahoga".to_string(),
            idmap: "/fsv/idmap.csv".into(),
            citations: "/fsv/citations_cuyahoga.csv".into(),
            collection: Some("legal-citations-v1".to_string()),
            skip_report: Some("/fsv/skips.json".into()),
            report: Some("/fsv/report.json".into()),
            home: Some("/home/calyx".into()),
        })
    );
}

#[test]
fn rejects_missing_idmap() {
    let args = ["legal-cuyahoga", "--citations", "/fsv/c.csv"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let error = parse_materialize_citation_overlay(&args).expect_err("missing idmap");
    assert!(error.message().contains("--idmap"));
}

#[test]
fn rejects_report_path_aliases_before_materialization() {
    let home = temp_root("report-alias");
    fs::create_dir_all(home.join("alias-parent")).expect("create physical parent");
    let report = home.join("report.json");
    let alias = home.join("alias-parent").join("..").join("report.json");
    let error = preflight_report_paths(&MaterializeCitationOverlayArgs {
        vault: "legal-cuyahoga".to_string(),
        idmap: home.join("idmap.csv"),
        citations: home.join("citations.csv"),
        collection: None,
        skip_report: Some(report),
        report: Some(alias),
        home: None,
    })
    .expect_err("canonical aliases must fail before DB mutation");

    assert!(error.message().contains("must identify distinct"));
    fs::remove_dir_all(&home).expect("clean physical parent");
}

fn temp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "calyx-citation-overlay-{tag}-{}-{}",
        std::process::id(),
        Ulid::new()
    ))
}

/// Creates a real durable vault + CLI index so the overlay materializer can
/// open it exactly like the production Cuyahoga vault (#1469) would present.
fn make_vault(home: &Path, name: &str) -> VaultId {
    let vault_id = VaultId::from_ulid(Ulid::new());
    let relative = format!("vaults/{vault_id}");
    let vault_dir = home.join(&relative);
    AsterVault::new_durable(
        &vault_dir,
        vault_id,
        vault_salt(vault_id, name),
        VaultOptions::default(),
    )
    .expect("create durable vault");
    let index = json!({
        "vaults": [ {
            "name": name,
            "vault_id": vault_id.to_string(),
            "path": relative,
            "panel_template": "legal-v1",
        } ]
    });
    let index_path = home.join("vaults").join("index.json");
    fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    vault_id
}

fn cx(seed: &str) -> CxId {
    CxId::from_input(seed.as_bytes(), 0, b"citation-overlay-test-salt")
}

#[test]
fn materializes_cites_overlay_and_reads_back_from_cf_bytes() {
    let home = temp_root("happy");
    fs::create_dir_all(home.join("vaults")).unwrap();
    let name = "legal-cuyahoga";
    make_vault(&home, name);

    // Four in-slice opinions resolve to existing constellation CxIds; a fifth
    // (op-outside) has no cx mapping to force an unresolved skip.
    let a = cx("opinion-A");
    let b = cx("opinion-B");
    let c = cx("opinion-C");
    let d = cx("opinion-D");
    let idmap = home.join("idmap.csv");
    fs::write(
        &idmap,
        format!("opinion_id,cx_id\n1001,{a}\n1002,{b}\n1003,{c}\n1004,{d}\n"),
    )
    .unwrap();

    // Rows: 3 clean edges, one depth>10 (capped weight=1.0), one unresolved
    // cited endpoint (frontier), one duplicate pair, one zero-depth invalid.
    let citations = home.join("citations_cuyahoga.csv");
    fs::write(
        &citations,
        "id,depth,citing_opinion_id,cited_opinion_id\n\
         r1,3,1001,1002\n\
         r2,7,1001,1003\n\
         r3,15,1002,1004\n\
         r4,2,1001,9999\n\
         r5,4,1001,1002\n\
         r6,0,1003,1004\n",
    )
    .unwrap();

    let report = materialize_with_home(
        &home,
        MaterializeCitationOverlayArgs {
            vault: name.to_string(),
            idmap,
            citations,
            collection: None,
            skip_report: Some(home.join("skips.json")),
            report: Some(home.join("report.json")),
            home: None,
        },
    )
    .expect("materialize overlay");

    // 6 data rows: 3 edges built; 3 skipped (unresolved_cited, duplicate, invalid_depth).
    assert_eq!(report.skip_report.total_rows, 6);
    assert_eq!(report.skip_report.edges_built, 3);
    assert_eq!(report.skip_report.skipped_total, 3);
    assert_eq!(report.skip_report.skipped_unresolved_cited, 1);
    assert_eq!(report.skip_report.skipped_duplicate_pair, 1);
    assert_eq!(report.skip_report.skipped_invalid_depth, 1);
    // Acceptance identity: edges == in-slice rows - skips.
    assert_eq!(
        report.skip_report.edges_built,
        report.skip_report.total_rows - report.skip_report.skipped_total
    );

    // Readback counts from the physical Graph CF + CSR.
    assert_eq!(report.readback.node_rows_written, 4);
    assert_eq!(report.readback.edge_rows_written, 3);
    assert_eq!(report.readback.physical_edge_out_keys, 3);
    assert_eq!(report.readback.csr_edges, 3);
    assert_eq!(report.readback.csr_nodes, 4);
    assert!(report.readback.all_edge_values_read_back);

    // Spot-verify a known pair in the accepted generation's CF bytes.
    let vault_dir = home.join("vaults").join(&report.vault_id);
    let physical = PhysicalPlainGraph::open_latest_accepted(&vault_dir, DEFAULT_COLLECTION)
        .expect("open accepted");
    let edge_bytes = physical
        .get_edge(a, EDGE_TYPE, b)
        .expect("read edge")
        .expect("edge A->B present");
    let edge: Value = serde_json::from_slice(&edge_bytes).unwrap();
    assert_eq!(edge["edge_type"], "cites");
    assert_eq!(edge["depth"], 3);
    assert_eq!(edge["weight"], json!(0.3));
    assert_eq!(
        edge["provenance_dataset"],
        "courtlistener-citation-map-2026-06-30"
    );
    assert_eq!(edge["source_row_id"], "r1");

    // depth>10 capped to weight 1.0 in the CF bytes.
    let capped = physical.get_edge(b, EDGE_TYPE, d).unwrap().unwrap();
    let capped: Value = serde_json::from_slice(&capped).unwrap();
    assert_eq!(capped["depth"], 15);
    assert_eq!(capped["weight"], json!(1.0));

    // Skip report persisted to disk with matching counts.
    let skip_disk: Value =
        serde_json::from_slice(&fs::read(home.join("skips.json")).unwrap()).unwrap();
    assert_eq!(skip_disk["edges_built"], 3);
    assert_eq!(skip_disk["skipped_total"], 3);

    fs::remove_dir_all(&home).ok();
}

#[test]
fn refuses_when_no_edges_resolve() {
    let home = temp_root("empty");
    fs::create_dir_all(home.join("vaults")).unwrap();
    let name = "legal-cuyahoga";
    make_vault(&home, name);
    let a = cx("only-opinion");
    let idmap = home.join("idmap.csv");
    fs::write(&idmap, format!("opinion_id,cx_id\n1001,{a}\n")).unwrap();
    let citations = home.join("citations.csv");
    // Both endpoints out of slice -> unresolved_both, zero edges.
    fs::write(
        &citations,
        "id,depth,citing_opinion_id,cited_opinion_id\nr1,3,8000,9000\n",
    )
    .unwrap();
    let error = materialize_with_home(
        &home,
        MaterializeCitationOverlayArgs {
            vault: name.to_string(),
            idmap,
            citations,
            collection: None,
            skip_report: None,
            report: None,
            home: None,
        },
    )
    .expect_err("no edges resolved");
    assert!(error.message().contains("no citation edges resolved"));
    fs::remove_dir_all(&home).ok();
}
