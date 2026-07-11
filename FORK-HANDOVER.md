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
- **`guard_generate.rs` — BOTH backends now coexist (as of `16f6e3c`+).** Upstream measures guard candidates **locally** (`required_dense_vectors` in `tools/guard_measure.rs` → `state.registry.measure`); we also need the **resident-hybrid** path (`generated_vectors` → `measure_query_vectors_resident_hybrid`) because the book runs `calyx guard generate` as a fresh subprocess per passage with **GPU lenses**, so a local measure would cold-load GPU models every call. Resolution: keep **both** functions; the call site selects via `guard_prefers_resident()` (env `CALYX_GUARD_MEASURE`: `resident`=ours, else upstream's local). The book-writer sets `CALYX_GUARD_MEASURE=resident` in `CALYX_ENV`. **This shrinks the recurring conflict** — upstream's `guard_measure.rs`/`required_dense_vectors` now merge cleanly (we keep them); only the ~5-line call-site selector may conflict (keep the `if guard_prefers_resident() {...} else {...}` form). Still ensure **`ProducedSlots`** stays in the `use calyx_ward::{...}` import (upstream sometimes drops it → `E0425`; it's `calyx_ward`'s pub alias `BTreeMap<SlotId,Vec<f32>>`).
- Everything else (audit hardening, forge/aster, calyxd, forecast) has merged cleanly historically.

## 5. Build & deploy (critical — a plain build makes a BROKEN binary)
- Build **exactly**: `cargo build --release --workspace --features cuda -j4` (with `nice`), env: `LIBCUVS_USE_PYTHON=1`, `cuvs_DIR=/home/unixdude/.local/lib/python3.12/site-packages/libcuvs/lib64/cmake/cuvs`, and the full `LD_LIBRARY_PATH` (cudnn/libcuvs/librmm/rapids_logger/libraft/cuda-13.1/wsl). **`_build_cuda.sh` does all this** and bakes the RPATH (patchelf `--force-rpath`) so `calyx` runs with no `LD_LIBRARY_PATH`. Good cuda binary is ~63 MB with caps `forge-cuda/registry-candle-cuda/sextant-cuvs` all true (`calyx build-info`).
- **NEVER** `cargo build -p calyx-cli` or `--bin calyx` — it silently drops the `candle-cuda` feature and the next resident dies on slot 5 ("built without feature candle-cuda").
- **Deploy = cycle the GPU RESIDENT so it re-execs the new binary.** The systemd `calyx-book` service is `KillMode=process`, so `restart2.sh` restarts the ORCHESTRATOR but the **resident SURVIVES on the OLD binary** — you must kill it too. Full sequence (helper: `~/calyx-book/stylo/redeploy_both.sh`): (1) `POST /api/stop` + wait until not `running` (so no `calyx guard` subprocess collides with the build); (2) `_build_cuda.sh`, need `cargo_rc=0`; (3) `restart2.sh` (reloads the orchestrator's `CALYX_ENV`, e.g. `CALYX_GUARD_MEASURE`); (4) kill any resident on :8765 by PID (`ss -ltnp | grep :8765` → `pid=`) and `rm resident/discovery.json`; (5) `POST /api/start` with the book's FULL stored config → the worker starts a fresh resident that execs the new binary. Confirm a page then generates cleanly.
- **A resident restart costs a ~15 min GPU warm-up** (reloads every lens model + re-warms search pins; the book waits). So **BATCH all Calyx-crate changes into ONE build + ONE deploy** — never rebuild/redeploy per change. Edit all the `.rs` files, build once, cycle the resident once. (Lesson learned 2026-07-11: a merge deploy then a separate guard change = two warm-ups; should have been one.)

## 6. Search-index corruption recovery
Symptom: book `status=error`, log `grounding search failed ... CALYX_STALE_DERIVED ... sparse sidecar .../slot_000NN_seq_..._n_....sparse.json is not valid JSON: missing field doc_lengths ...; rebuild the vault search indexes`. Cause: a sidecar was **truncated** (e.g. a host restart mid-write); `--stale-ok` cannot bypass corrupt JSON. Fix: **`calyx rebuild-search-index <vault-name>`** (env `CALYX_HOME`) — it needs the resident **DOWN** (it holds the index pins), so do it inside the deploy window (resident already killed). ~30 s for tolkien-full; writes a fresh sidecar and supersedes the corrupt one. Verify: the newest `slot_000NN_*.sparse.json` `json.load`s and has `doc_lengths` with len == the vault's constellation count.

## 7. Gotchas
- WSL: cargo builds need `-j4` + `nice` or WSL dies. One GPU resident at a time; never build a vault's GPU stage while a book runs against another.
- `gh`/heredocs and nested `$( )` **through `wsl.exe -lc "..."` mangle backticks/`$`/quotes** — put multi-step logic in a `.sh`/`.py` file and run that; use `gh ... --body-file <file>` and `git commit -F <file>`.
- `pkill -f 'panel resident serve'` from a shell whose own command line contains that string **kills itself** — kill the resident by PID (`ss -ltnp | grep :8765`).
- `_build_cuda.sh` is untracked (local build helper); keep it. It now also runs the post-build `patchelf` RPATH step.

_Last import: 2026-07-11, ste-bah/Calyx `04c77ab` (upstream `d040c50`), 20 commits, 1 conflict (guard_generate.rs). Then `f956db0`: both guard backends coexist via `CALYX_GUARD_MEASURE`. Deployed on a rebuilt tolkien-full search index (recovered a truncated slot-7 sidecar)._
