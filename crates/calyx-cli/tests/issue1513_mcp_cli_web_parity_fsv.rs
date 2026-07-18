use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use calyx_core::AuthN;
use calyx_mcp::jsonrpc::decode_jsonrpc_request;
use calyx_mcp::server::McpServer;
use calyx_web_api::{AuthCtx, Guardrails, MeasureCtx, build_app_with_search};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

#[tokio::test(flavor = "multi_thread")]
async fn mcp_cli_and_web_share_filtered_persisted_generation() {
    let home = TestHome::new();
    let mut server = McpServer::new();
    calyx_mcp::tools::register_all(&mut server).unwrap();
    let created = mcp_call(&server, 1, "calyx.create_vault", json!({"name": "parity"}));
    mcp_call(
        &server,
        2,
        "calyx.add_lens",
        json!({
            "vault": "parity",
            "name": "byte_axis",
            "runtime": "algorithmic"
        }),
    );
    mcp_call(
        &server,
        3,
        "calyx.ingest",
        json!({
            "vault": "parity",
            "batch": ["alpha alpha", "alpha nearby", "beta different", "gamma remote"]
        }),
    );

    let vault_id = created["vault_id"].as_str().unwrap();
    let vault_dir = home.path.join("vaults").join(vault_id);
    let manifest_path = vault_dir.join("idx/search/manifest.json");
    let manifest_before = fs::read(&manifest_path).unwrap();
    let filter = json!({"metadata": [{"modality": "text"}]});

    let mcp = mcp_call(
        &server,
        4,
        "calyx.search",
        json!({
            "vault": "parity",
            "query": "alpha",
            "k": 3,
            "fusion": "rrf",
            "filter": filter
        }),
    );
    assert_eq!(mcp["execution"]["executor"], "calyx-search/persisted");
    assert_eq!(mcp["execution"]["request_index_builds"], 0);
    assert_eq!(mcp["execution"]["slot_cache_enabled"], false);
    let cache = &mcp["execution"]["cache_after"];
    assert_eq!(cache["entry_count"], 0);
    assert!(cache["entry_count"].as_u64().unwrap() <= cache["max_entries"].as_u64().unwrap());

    let filter_json = serde_json::to_string(&filter).unwrap();
    let cli_output = Command::new(env!("CARGO_BIN_EXE_calyx"))
        .env("CALYX_HOME", &home.path)
        .args([
            "search",
            "parity",
            "alpha",
            "--k",
            "3",
            "--fusion",
            "rrf",
            "--filter",
            &filter_json,
        ])
        .output()
        .unwrap();
    assert!(
        cli_output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&cli_output.stderr)
    );
    let cli: Value = serde_json::from_slice(&cli_output.stdout).unwrap();
    let cli_stderr = String::from_utf8(cli_output.stderr).unwrap();
    let cli_generation = cli_generation(&cli_stderr);

    let secret = "issue1513-parity-secret";
    let app = build_app_with_search(
        Arc::new(Guardrails::production()),
        Arc::new(MeasureCtx::load(&vault_dir, "parity").unwrap()),
        Arc::new(AuthCtx::new(secret).unwrap()),
    );
    let response = app
        .oneshot(
            Request::post("/v1/search")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, format!("Bearer {secret}"))
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "query": "alpha",
                        "k": 3,
                        "fusion": "rrf",
                        "filter": filter
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let web: Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();

    let mcp_hits = normalize_hits(&mcp["hits"], Surface::Snake);
    let cli_hits = normalize_hits(&cli, Surface::Snake);
    let web_hits = normalize_hits(&web["hits"], Surface::Camel);
    assert!(!mcp_hits.is_empty());
    assert_eq!(mcp_hits, cli_hits);
    assert_eq!(mcp_hits, web_hits);
    assert_eq!(mcp["execution"]["generation"], web["generation"]);
    assert_eq!(
        cli_generation.0,
        mcp["execution"]["generation"]["base_seq"].as_u64().unwrap()
    );
    assert_eq!(
        cli_generation.1,
        mcp["execution"]["generation"]["manifest_sha256"]
            .as_str()
            .unwrap()
    );
    let manifest_after = fs::read(&manifest_path).unwrap();
    assert_eq!(
        manifest_before, manifest_after,
        "search must not rebuild indexes"
    );

    let report = json!({
        "schema": "issue1513-mcp-cli-web-parity-v1",
        "vault_id": vault_id,
        "filter": filter,
        "manifest_sha256": sha256_hex(&manifest_after),
        "manifest_unchanged": true,
        "mcp_execution": mcp["execution"],
        "cli_generation": {
            "base_seq": cli_generation.0,
            "manifest_sha256": cli_generation.1,
        },
        "web_generation": web["generation"],
        "canonical_hits": mcp_hits,
        "cli_stderr": cli_stderr,
    });
    if let Some(root) = std::env::var_os("CALYX_FSV_ROOT") {
        let root = PathBuf::from(root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("issue1513_mcp_cli_web_parity.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    println!("ISSUE1513_MCP_CLI_WEB_PARITY_FSV {report}");
}

#[derive(Clone, Copy)]
enum Surface {
    Snake,
    Camel,
}

fn normalize_hits(hits: &Value, surface: Surface) -> Vec<Value> {
    hits.as_array()
        .unwrap()
        .iter()
        .map(|hit| match surface {
            Surface::Snake => json!({
                "rank": hit["rank"],
                "cx_id": hit["cx_id"],
                "score_f32_bits": (hit["score"].as_f64().unwrap() as f32).to_bits(),
                "ledger_seq": hit["provenance"]["ledger_seq"],
                "chain_hash": hit["provenance"]["chain_hash"],
            }),
            Surface::Camel => json!({
                "rank": hit["rank"],
                "cx_id": hit["cxId"],
                "score_f32_bits": (hit["score"].as_f64().unwrap() as f32).to_bits(),
                "ledger_seq": hit["provenance"]["ledgerSeq"],
                "chain_hash": hit["provenance"]["chainHash"],
            }),
        })
        .collect()
}

fn mcp_call(server: &McpServer, id: u64, name: &str, arguments: Value) -> Value {
    let request = decode_jsonrpc_request(
        serde_json::to_string(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }))
        .unwrap()
        .as_bytes(),
    )
    .unwrap();
    let authn = AuthN::InProcess {
        host_app_id: "issue1513-fsv".to_string(),
    };
    let response = server.dispatch_with_authn(request, Some(&authn));
    assert!(response.error.is_none(), "MCP error: {:?}", response.error);
    let text = response.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    serde_json::from_str(&text).unwrap()
}

fn cli_generation(stderr: &str) -> (u64, String) {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("CALYX_SEARCH_GENERATION "))
        .expect("CLI generation instrumentation");
    let value = |name: &str| {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix(&format!("{name}=")))
            .unwrap()
            .to_string()
    };
    (value("base_seq").parse().unwrap(), value("manifest_sha256"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct TestHome {
    path: PathBuf,
    old: Option<OsString>,
}

impl TestHome {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "calyx-issue1513-parity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        let old = std::env::var_os("CALYX_HOME");
        unsafe { std::env::set_var("CALYX_HOME", &path) };
        Self { path, old }
    }
}

impl Drop for TestHome {
    fn drop(&mut self) {
        match &self.old {
            Some(value) => unsafe { std::env::set_var("CALYX_HOME", value) },
            None => unsafe { std::env::remove_var("CALYX_HOME") },
        }
        if self.path.starts_with(std::env::temp_dir()) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
