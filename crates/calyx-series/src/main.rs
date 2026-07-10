//! calyx-series — series/saga memory for multi-book sagas.
//!
//! An additive tool that COMPOSES the stable `calyx` CLI (it links no Calyx
//! library, so upstream refactors never break it via merge conflicts). It gives
//! the book writer real cross-book memory: `absorb` a finished book into a
//! per-series vault with `(series, book, chapter, page)` provenance, `recall`
//! relevant past events when planning a sequel, and emit a structural `bible`.
//!
//! The series vault uses a CPU external-cmd semantic lens (resident-free), so
//! absorb/recall never contend with the book's GPU resident.
//!
//! Errors fail loud with full context; there are no silent fallbacks.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use sha2::{Digest, Sha256};

type R<T> = Result<T, String>;

fn calyx_bin() -> String {
    env::var("CALYX_BIN").unwrap_or_else(|_| "calyx".to_string())
}

fn semantic_cmd() -> String {
    env::var("CALYX_SERIES_SEMANTIC_CMD")
        .unwrap_or_else(|_| "/home/unixdude/calyx-book/stylo/semantic_lens_cmd.sh".to_string())
}

fn calyx_home() -> R<PathBuf> {
    env::var("CALYX_HOME")
        .map(PathBuf::from)
        .map_err(|_| "CALYX_HOME is not set; it must point at the Calyx data root".to_string())
}

/// series display name -> vault-safe slug -> vault name `series-<slug>`.
fn slugify(name: &str) -> String {
    let mut s = String::new();
    let mut dash = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c);
            dash = false;
        } else if !dash && !s.is_empty() {
            s.push('-');
            dash = true;
        }
    }
    s.trim_matches('-').to_string()
}

fn vault_name(series: &str) -> String {
    format!("series-{}", slugify(series))
}

/// Run the `calyx` CLI, inheriting the parent env (CALYX_HOME, LD_LIBRARY_PATH,
/// ORT_DYLIB_PATH, ...). Fail loud with full stderr/stdout context.
/// Returns (stdout, stderr). Fails loud with full context on non-zero exit.
fn run_calyx(args: &[&str]) -> R<(String, String)> {
    let bin = calyx_bin();
    let out = Command::new(&bin)
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn `{bin}` (is CALYX_BIN correct / on PATH?): {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        let tail = |s: &str| -> String {
            let s = s.trim();
            let n = s.len().saturating_sub(1200);
            s[n..].to_string()
        };
        return Err(format!(
            "`calyx {}` failed (exit {:?})\n--- stderr ---\n{}\n--- stdout ---\n{}",
            args.join(" "),
            out.status.code(),
            tail(&stderr),
            tail(&stdout),
        ));
    }
    Ok((stdout, stderr))
}

/// Resolve a vault NAME to its on-disk directory via CALYX_HOME/vaults/index.json.
fn vault_path(name: &str) -> R<PathBuf> {
    let home = calyx_home()?;
    let idx_path = home.join("vaults").join("index.json");
    let raw = fs::read_to_string(&idx_path)
        .map_err(|e| format!("read {}: {e}", idx_path.display()))?;
    let idx: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse {}: {e}", idx_path.display()))?;
    let vaults = idx
        .get("vaults")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{} has no `vaults` array", idx_path.display()))?;
    for v in vaults {
        if v.get("name").and_then(|n| n.as_str()) == Some(name) {
            let p = v
                .get("path")
                .and_then(|p| p.as_str())
                .ok_or_else(|| format!("vault {name} entry has no `path`"))?;
            return Ok(home.join(p));
        }
    }
    Err(format!(
        "no vault named {name:?} in {} (run `calyx-series init --series ...` first)",
        idx_path.display()
    ))
}

/// Deterministic per-page retrieval timestamp = the page file's mtime (epoch secs).
/// This MUST NOT use wall-clock `now()`: absorb is called idempotently (on book
/// finish and on every `/api/sequel`), and Calyx fails closed if an already-stored
/// anchor is re-ingested with different metadata. mtime only changes when the page
/// is actually rewritten -- and a rewritten page has a new content hash (new anchor)
/// anyway -- so this keeps re-absorb a byte-identical no-op.
fn file_mtime_secs(path: &std::path::Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ----------------------------------------------------------------- init

fn cmd_init(series: &str) -> R<()> {
    let vault = vault_name(series);
    eprintln!("[init] series={series:?} vault={vault}");
    match run_calyx(&["create-vault", &vault, "--panel-template", "text-default"]) {
        Ok(out) => {
            let vid = out.0
                .lines()
                .find_map(|l| l.trim().starts_with('{').then(|| l.trim()))
                .unwrap_or("");
            eprintln!("[init] created vault: {vid}");
        }
        Err(e) if e.contains("already exists") => {
            // registered already; verify it is actually usable (dir present) so a
            // dangling/corrupt registration fails loud instead of faking success.
            let path = vault_path(&vault)?;
            if !path.is_dir() {
                return Err(format!(
                    "vault {vault} is registered in the index but its directory {} is \
                     missing or corrupt; retire it before re-initializing",
                    path.display()
                ));
            }
            println!(
                "{}",
                serde_json::json!({"status":"already_initialized","series":series,"vault":vault})
            );
            return Ok(());
        }
        Err(e) => return Err(format!("create-vault failed: {e}")),
    }
    // text-default activates only keyword_splade (GPU sparse) at slot 1; park it
    // so the ONLY active content lens is the CPU semantic lens -> resident-free.
    run_calyx(&["park-lens", &vault, "--slot", "1"])
        .map_err(|e| format!("park splade (slot 1) failed: {e}"))?;
    // add the resident-free CPU semantic lens (all-MiniLM via external-cmd worker)
    let cmd = semantic_cmd();
    if !PathBuf::from(&cmd).exists() {
        return Err(format!(
            "semantic worker not found at {cmd}; set CALYX_SERIES_SEMANTIC_CMD"
        ));
    }
    run_calyx(&[
        "add-lens", &vault, "--name", "series-semantic", "--runtime", "external-cmd",
        "--endpoint", &cmd, "--shape", "Dense(384)", "--modality", "text",
    ])
    .map_err(|e| format!("add semantic lens failed: {e}"))?;
    println!(
        "{}",
        serde_json::json!({"status":"initialized","series":series,"vault":vault,"lens":"series-semantic(Dense384,cpu)"})
    );
    Ok(())
}

// ----------------------------------------------------------------- absorb

fn cmd_absorb(series: &str, book_dir: &str, book: u32) -> R<()> {
    let vault = vault_name(series);
    let dir = PathBuf::from(book_dir);
    if !dir.is_dir() {
        return Err(format!("book-dir does not exist or is not a directory: {book_dir}"));
    }
    let state_path = dir.join("state.json");
    let state_raw = fs::read_to_string(&state_path)
        .map_err(|e| format!("read {}: {e} (is this a book directory?)", state_path.display()))?;
    let state: serde_json::Value = serde_json::from_str(&state_raw)
        .map_err(|e| format!("parse {}: {e}", state_path.display()))?;

    // map page number -> chapter number from state.pages[]
    let mut page_chapter: BTreeMap<u64, u64> = BTreeMap::new();
    if let Some(pages) = state.get("pages").and_then(|p| p.as_array()) {
        for p in pages {
            if let Some(n) = p.get("n").and_then(|n| n.as_u64()) {
                let ch = p.get("chapter").and_then(|c| c.as_u64()).unwrap_or(0);
                page_chapter.insert(n, ch);
            }
        }
    }

    let pages_dir = dir.join("pages");
    if !pages_dir.is_dir() {
        return Err(format!("no pages/ directory under {book_dir}"));
    }
    let mut files: Vec<(u64, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&pages_dir)
        .map_err(|e| format!("read {}: {e}", pages_dir.display()))?
    {
        let path = entry.map_err(|e| format!("dir entry: {e}"))?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if let Ok(n) = stem.trim_start_matches('0').parse::<u64>() {
                files.push((n, path));
            } else if stem.chars().all(|c| c == '0') && !stem.is_empty() {
                files.push((0, path));
            }
        }
    }
    files.sort_by_key(|(n, _)| *n);
    if files.is_empty() {
        return Err(format!("book at {book_dir} has zero pages/*.txt files; nothing to absorb"));
    }

    // build provenance-tagged ingest JSONL
    let mut jsonl = String::new();
    for (n, path) in &files {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let sha = hex(&hasher.finalize());
        let chapter = page_chapter.get(n).copied().unwrap_or(0);
        // Deterministic per-page timestamp so re-absorb is a byte-identical no-op.
        let ts = file_mtime_secs(path).to_string();
        let row = serde_json::json!({
            "text": text,
            "metadata": {
                "source_dataset": format!("series:{series}"),
                "source_sha256": sha,
                "license": "series-memory",
                "retrieval_ts": ts,
                "source_url": format!("local://series/{}/book/{}/page/{}", slugify(series), book, n),
                "series": series,
                "book": book.to_string(),
                "chapter": chapter.to_string(),
                "page": n.to_string(),
                // store the page text so recall can return the passage itself
                "text": text,
            }
        });
        jsonl.push_str(&row.to_string());
        jsonl.push('\n');
    }
    let tmp = calyx_home()?.join(format!(".series-absorb-{}-b{}.jsonl", slugify(series), book));
    fs::write(&tmp, &jsonl).map_err(|e| format!("write {}: {e}", tmp.display()))?;

    let tmp_s = tmp.to_string_lossy().into_owned();
    let (stdout, stderr) = run_calyx(&["ingest", &vault, "--batch", &tmp_s, "--output", "summary"])
        .map_err(|e| format!("ingest failed: {e}"))?;
    let _ = fs::remove_file(&tmp);

    // `calyx ingest` fails closed on any error, so exit-0 means every prepared
    // row was ingested. The independent readback is the source of truth for the
    // physical count; here we report what we sent plus the CLI's new/already
    // counts (emitted on the batch-summary log line).
    let combined = format!("{stdout}\n{stderr}");
    let grab = |key: &str| -> Option<u64> {
        combined
            .split_whitespace()
            .find_map(|t| t.strip_prefix(key).and_then(|v| v.parse().ok()))
    };
    println!(
        "{}",
        serde_json::json!({
            "status":"absorbed","series":series,"book":book,"vault":vault,
            "pages_found":files.len(),
            "rows_ingested":files.len(),
            "new":grab("new_count="),
            "already":grab("already_count=")
        })
    );
    Ok(())
}

// ----------------------------------------------------------------- recall

fn cmd_recall(series: &str, query: &str, k: u32) -> R<()> {
    let vault = vault_name(series);
    let path = vault_path(&vault)?;
    let path_s = path.to_string_lossy().into_owned();
    let ks = k.to_string();
    let (out, _) = run_calyx(&["search", &vault, query, "--k", &ks, "--stale-ok"])
        .map_err(|e| format!("search failed: {e}"))?;
    let a = out.find('[');
    let b = out.rfind(']');
    let hits: serde_json::Value = match (a, b) {
        (Some(a), Some(b)) if b > a => serde_json::from_str(&out[a..=b])
            .map_err(|e| format!("parse search hits: {e}\nraw: {out}"))?,
        _ => return Err(format!("search produced no hit array; raw output:\n{out}")),
    };
    let hits = hits.as_array().cloned().unwrap_or_default();

    let mut results = Vec::new();
    for h in &hits {
        let cx = h.get("cx_id").and_then(|c| c.as_str()).unwrap_or("");
        let rank = h.get("rank").and_then(|r| r.as_u64()).unwrap_or(0);
        let score = h.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
        if cx.is_empty() {
            continue;
        }
        // pull provenance + stored text for this exact cx from the vault
        let (rb, _) = run_calyx(&[
            "readback", "cx-list", "--vault", &path_s, "--cx-id", cx,
            "--include-slots", "--rebuild-base-page-index",
        ])
        .map_err(|e| format!("readback {cx} failed: {e}"))?;
        let (ma, mb) = (rb.find('['), rb.rfind(']'));
        let meta = match (ma, mb) {
            (Some(a), Some(b)) if b > a => {
                let arr: serde_json::Value = serde_json::from_str(&rb[a..=b])
                    .map_err(|e| format!("parse readback for {cx}: {e}"))?;
                arr.as_array()
                    .and_then(|v| v.first())
                    .and_then(|c| c.get("metadata").cloned())
                    .unwrap_or(serde_json::Value::Null)
            }
            _ => serde_json::Value::Null,
        };
        let g = |k: &str| meta.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        results.push(serde_json::json!({
            "rank": rank, "score": score, "cx_id": cx,
            "book": g("book"), "chapter": g("chapter"), "page": g("page"),
            "text": g("text"),
        }));
    }
    // machine-readable (for the orchestrator to inject) + human summary on stderr
    println!("{}", serde_json::json!({"series":series,"query":query,"hits":results}));
    eprintln!("[recall] {} hit(s) for {query:?}:", results.len());
    for r in &results {
        eprintln!(
            "  book{} ch{} pg{} (score {:.4}): {}",
            r["book"], r["chapter"], r["page"],
            r["score"].as_f64().unwrap_or(0.0),
            r["text"].as_str().unwrap_or("").chars().take(90).collect::<String>()
        );
    }
    Ok(())
}

// ----------------------------------------------------------------- bible

fn cmd_bible(series: &str) -> R<()> {
    let vault = vault_name(series);
    let path = vault_path(&vault)?;
    let path_s = path.to_string_lossy().into_owned();
    let (rb, _) = run_calyx(&[
        "readback", "cx-list", "--vault", &path_s, "--limit", "100000",
        "--include-slots", "--allow-unbounded", "--rebuild-base-page-index",
    ])
    .map_err(|e| format!("readback failed: {e}"))?;
    let (a, b) = (rb.find('['), rb.rfind(']'));
    let arr: serde_json::Value = match (a, b) {
        (Some(a), Some(b)) if b > a => {
            serde_json::from_str(&rb[a..=b]).map_err(|e| format!("parse readback: {e}"))?
        }
        _ => return Err(format!("bible: readback produced no array; raw:\n{rb}")),
    };
    let rows = arr.as_array().cloned().unwrap_or_default();
    // aggregate book -> chapter -> page count
    let mut books: BTreeMap<u64, BTreeMap<u64, u64>> = BTreeMap::new();
    for r in &rows {
        let m = r.get("metadata");
        let gi = |k: &str| -> u64 {
            m.and_then(|m| m.get(k))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        };
        *books.entry(gi("book")).or_default().entry(gi("chapter")).or_default() += 1;
    }
    let books_json: Vec<_> = books
        .iter()
        .map(|(book, chapters)| {
            let total: u64 = chapters.values().sum();
            serde_json::json!({
                "book": book,
                "page_count": total,
                "chapters": chapters.iter()
                    .map(|(c, n)| serde_json::json!({"chapter": c, "pages": n}))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "series": series, "vault": vault,
            "book_count": books.len(),
            "total_pages": rows.len(),
            "books": books_json
        })
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ----------------------------------------------------------------- CLI

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn need<'a>(args: &'a [String], name: &str) -> R<&'a str> {
    flag(args, name).ok_or_else(|| format!("missing required flag {name}"))
}

fn usage() -> String {
    "calyx-series — series/saga memory (composes the calyx CLI)\n\
     usage:\n  \
     calyx-series init   --series <name>\n  \
     calyx-series absorb --series <name> --book-dir <path> --book <n>\n  \
     calyx-series recall --series <name> --query <text> [--k <n>]\n  \
     calyx-series bible  --series <name>\n\
     env: CALYX_BIN (calyx binary), CALYX_HOME (data root), CALYX_SERIES_SEMANTIC_CMD"
        .to_string()
}

fn dispatch() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("");
    match cmd {
        "init" => cmd_init(need(&args, "--series")?),
        "absorb" => cmd_absorb(
            need(&args, "--series")?,
            need(&args, "--book-dir")?,
            need(&args, "--book")?
                .parse()
                .map_err(|e| format!("--book must be an integer: {e}"))?,
        ),
        "recall" => cmd_recall(
            need(&args, "--series")?,
            need(&args, "--query")?,
            flag(&args, "--k").and_then(|s| s.parse().ok()).unwrap_or(5),
        ),
        "bible" => cmd_bible(need(&args, "--series")?),
        "" => Err(format!("no subcommand given\n\n{}", usage())),
        other => Err(format!("unknown subcommand {other:?}\n\n{}", usage())),
    }
}

fn main() {
    if let Err(e) = dispatch() {
        eprintln!("calyx-series ERROR: {e}");
        std::process::exit(1);
    }
}
