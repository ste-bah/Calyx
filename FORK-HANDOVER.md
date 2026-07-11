# Calyx Fork Handover (ste-bah/Calyx) — for Hermes

How our fork differs from ChrisRoyse/Calyx, and how to import his new commits safely.

## 1. Layout & source of truth
- **`/home/unixdude/Calyx` (WSL) is the SOURCE OF TRUTH.** Windows `F:\calyx\Calyx` is stale/diverged — ignore it.
- **Remotes:** `fork` = `ste-bah/Calyx` (ours — push here). `origin` = `upstream` = `ChrisRoyse/Calyx` (import from here).
- **The book-writer app is separate:** `~/calyx-book/` (NOT a git repo; deployed live). It only *composes* the `calyx` CLI. Don't confuse it with the Rust repo.

## 2. Our overrides — MUST survive every import
After each merge, confirm these are intact (build fails loudly if not):
- **`crates/calyx-registry/src/runtime/onnx/session.rs`** — GPU-lens **CPU-EP-fallback** override (the "GPU lens thing"). Without it every `CudaFailLoud` resident aborts.
- **`crates/calyx-mcp/src/tools/search/engine.rs` + `tests.rs`** — MCP search **one-path** port (resident auto-discovery).
- **`crates/calyx-mcp/src/tools/search/extensions/guard_generate.rs`** — `generated_vectors()` measures the candidate through the **resident** (see §4).
- **`crates/calyx-registry/src/runtime/algorithmic/stylometry.rs`** — stylometric pacing lens.
- **`crates/calyx-cli/src/cmd/lens.rs`** — add-lens manifest.json path (style gates use it).
- **`crates/calyx-series/`** — our additive sequel/series-memory crate (composes the CLI, links no Calyx lib; auto-included via `members=["crates/*"]`).

## 3. Import procedure (safe)
```bash
cd /home/unixdude/Calyx
git fetch origin
git log --oneline main..origin/main            # REVIEW incoming first
git branch pre-import-$(date +%F) main          # safety snapshot
git merge --no-commit --no-ff origin/main       # merge; resolve conflicts (§4)
git add <resolved files> && git commit          # complete merge (LOCAL only)
bash _build_cuda.sh                              # BUILD = the real test; need cargo_rc=0
unset LD_LIBRARY_PATH; ./target/release/calyx --help   # smoke-test the binary
git commit --amend -F <msgfile>                 # descriptive merge message
git push fork main                              # push ONLY after build+smoke pass
git branch -D pre-import-$(date +%F)            # delete snapshot post-verify
```
Never push an unbuilt merge. The running book stays on the OLD binary until a resident restart (§5).

## 4. Recurring conflicts — how to resolve
- **`calyx-leapable` (E0583 + rustc ICE):** the public mirror chronically omits `engine/storage/{codec,params}.rs`. If it reappears (modify/delete conflict), **`git rm -rf crates/calyx-leapable`** — nothing depends on it (our leapable lives in `calyx-cli/src/leapable`). (It did NOT reappear in the 2026-07-11 import.)
- **`guard_generate.rs`:** upstream keeps moving guard measurement to a **local** path (`required_dense_vectors` in `tools/guard_measure.rs` → `state.registry.measure`). **KEEP OURS** (`generated_vectors` → `measure_query_vectors_resident_hybrid`): our book runs `calyx guard generate` as a fresh subprocess per passage with **GPU lenses**, so a local measure would cold-load GPU models every call. After keeping ours: (a) ensure **`ProducedSlots`** is in the `use calyx_ward::{...}` import (upstream drops it → `E0425 cannot find type ProducedSlots`; it is `calyx_ward`'s pub alias `BTreeMap<SlotId,Vec<f32>>`), (b) delete any now-unused `use crate::tools::guard_measure::required_dense_vectors;`.
- Everything else (audit hardening, forge/aster, calyxd, forecast) has merged cleanly historically.

## 5. Build & deploy (critical — a plain build makes a BROKEN binary)
- Build **exactly**: `cargo build --release --workspace --features cuda -j4` (with `nice`), env: `LIBCUVS_USE_PYTHON=1`, `cuvs_DIR=/home/unixdude/.local/lib/python3.12/site-packages/libcuvs/lib64/cmake/cuvs`, and the full `LD_LIBRARY_PATH` (cudnn/libcuvs/librmm/rapids_logger/libraft/cuda-13.1/wsl). **`_build_cuda.sh` does all this** and bakes the RPATH (patchelf `--force-rpath`) so `calyx` runs with no `LD_LIBRARY_PATH`.
- **NEVER** `cargo build -p calyx-cli` or `--bin calyx` — it silently drops the `candle-cuda` feature and the next resident dies on slot 5 ("built without feature candle-cuda"). Good cuda binary is ~63 MB.
- **Deploy** = `bash ~/calyx-book/stylo/restart2.sh` (kills the orchestrator MainPID; systemd `calyx-book` auto-restarts + auto-resumes the book → a fresh resident loads the new binary). Confirm `calyx build-info` git_sha == HEAD and resident readiness shows `gpu_content_lens_count=7`. **The book keeps running on the old binary until this restart** (each restart discards only the in-progress page).

## 6. Gotchas
- WSL: cargo builds need `-j4` + `nice` or WSL dies. One GPU resident at a time; never build a vault's GPU stage while a book runs against another.
- `gh`/heredocs through `wsl.exe` mangle backticks/`$` — use `gh ... --body-file <file>` and `git commit -F <file>`.
- `_build_cuda.sh` is untracked (local build helper); keep it.

_Last import: 2026-07-11, ste-bah/Calyx `04c77ab` (upstream `d040c50`), 20 commits, 1 conflict (guard_generate.rs), built clean._
