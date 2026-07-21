//! Issue #1489 FSV: round-trip every lens of a real vault's persisted
//! registry through BOTH contract paths and prove they agree.
//!
//! - Static path: `audit_vault_registry_contracts` (search preflight,
//!   rebuild-search-index, registry-repair) reconstructs each contract from
//!   the persisted `LensSpec` without loading runtime sessions.
//! - Warm path: `measure_registry_snapshot_lens_batch` rebuilds the actual
//!   runtime lens (`load_runtime_lens`), which fails loud with
//!   `CALYX_LENS_FROZEN_VIOLATION` if the constructed contract differs from
//!   the persisted one.
//!
//! If both pass for every lens, no audit-vs-warm fixpoint gap exists for that
//! vault's catalog.
//!
//! Requires real model artifacts and (for GPU lens kinds) CUDA, so it only
//! runs when `CALYX_ISSUE1489_PARITY_VAULT` points at a vault directory:
//!
//! ```text
//! CALYX_ISSUE1489_PARITY_VAULT=/zfs/hot/calyx/vaults/<ULID> \
//!     cargo test --features cuda -p calyx-registry \
//!     --test __calyx_integration_suite_0 issue1489_contract_parity_fsv -- --nocapture
//! ```

use calyx_core::Input;
use calyx_registry::{
    audit_vault_registry_contracts, load_vault_panel_state, measure_registry_snapshot_lens_batch,
};

const VAULT_ENV: &str = "CALYX_ISSUE1489_PARITY_VAULT";

#[test]
fn issue1489_vault_registry_contract_parity_fsv() {
    let Ok(vault_dir) = std::env::var(VAULT_ENV) else {
        eprintln!("skipping issue1489 contract parity FSV: {VAULT_ENV} is not set");
        return;
    };

    let audit = audit_vault_registry_contracts(&vault_dir).unwrap();
    println!(
        "static audit: checked_count={} valid={} diffs={:?}",
        audit.checked_count, audit.valid, audit.diffs
    );
    assert!(
        audit.valid,
        "static contract audit must reconstruct the persisted contracts: {:?}",
        audit.diffs
    );
    assert!(audit.checked_count > 0, "vault registry has no lenses");

    let state = load_vault_panel_state(&vault_dir).unwrap();
    let snapshot = state
        .registry_snapshot
        .expect("vault must persist a registry snapshot");
    let mut warmed = 0usize;
    for lens in &snapshot.lenses {
        let Some(spec) = &lens.spec else {
            panic!("lens {} persisted without LensSpec", lens.lens_id);
        };
        let input = Input::new(
            lens.contract.modality(),
            b"issue1489 contract parity probe".to_vec(),
        );
        // Rebuilds the runtime lens and fails loud on contract drift before
        // measuring, which is exactly the warm-path check panel resident
        // serve performs.
        let vectors = measure_registry_snapshot_lens_batch(lens, std::slice::from_ref(&input))
            .unwrap_or_else(|err| {
                panic!(
                    "warm-path reconstruction failed for lens {} ({}): {err:?}",
                    lens.lens_id, spec.name
                )
            });
        println!(
            "warm parity ok: lens={} name={} shape={:?} vectors={}",
            lens.lens_id,
            spec.name,
            lens.contract.shape(),
            vectors.len()
        );
        warmed += 1;
    }
    assert_eq!(warmed, snapshot.lenses.len());
    println!(
        "issue1489 parity: {} lenses agree on static and warm contracts",
        warmed
    );
}
