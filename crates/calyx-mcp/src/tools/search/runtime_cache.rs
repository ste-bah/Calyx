//! Generation-bound process caches for the expensive search runtime state.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use calyx_aster::ledger_view::AsterLedgerCfStore;
use calyx_aster::manifest::{ImmutableRef, ManifestStore, VaultManifest};
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::{CalyxError, VaultId};
use calyx_ledger::LedgerCfStore;
use calyx_registry::{VaultPanelState, load_vault_panel_state};

use crate::server::ToolResult;
use crate::tools::vault::store::{ResolvedVault, vault_salt};

#[derive(Clone, PartialEq, Eq)]
struct PanelGenerationKey {
    panel_ref: ImmutableRef,
    registry_ref: Option<ImmutableRef>,
}

#[derive(Clone)]
struct CachedPanelState {
    key: PanelGenerationKey,
    state: VaultPanelState,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct VaultGenerationKey {
    path: PathBuf,
    name: String,
    vault_id: VaultId,
    manifest_seq: u64,
    durable_seq: u64,
    pub(super) ledger_head_height: Option<u64>,
    pub(super) ledger_head_hash: Option<[u8; 32]>,
}

#[derive(Clone)]
struct CachedVaultState {
    key: VaultGenerationKey,
    vault: Arc<AsterVault>,
    ledger_view: Arc<AsterLedgerCfStore>,
}

/// Keep one manifest-bound Aster vault open for the MCP process.
///
/// Opening a durable vault reconstructs its checkpointed rows and is the
/// database equivalent of opening a shared engine, not request work. Reuse is
/// allowed only while the exact CURRENT manifest sequence, durable / derived
/// tips, and external Ledger head anchor remain unchanged. A miss is serialized
/// so concurrent requests cannot reconstruct duplicate in-memory copies of the
/// same physical vault.
pub(super) fn cached_vault(
    resolved: &ResolvedVault,
) -> ToolResult<(
    Arc<AsterVault>,
    Arc<AsterLedgerCfStore>,
    bool,
    VaultGenerationKey,
)> {
    let manifest_before = ManifestStore::open(&resolved.path).load_current()?;
    let key_before = vault_generation_key(resolved, &manifest_before)?;

    static CACHE: OnceLock<Mutex<Option<CachedVaultState>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut cache = cache.lock().map_err(|_| {
        CalyxError::stale_derived(
            "MCP search vault runtime cache mutex was poisoned; restart calyx-mcp",
        )
    })?;
    if let Some(cached) = cache.as_ref()
        && cached.key == key_before
    {
        let vault = Arc::clone(&cached.vault);
        let ledger_view = Arc::clone(&cached.ledger_view);
        drop(cache);
        validate_cached_vault(&vault, &key_before)?;
        validate_cached_ledger_view(&ledger_view, &key_before)?;
        ensure_vault_generation_unchanged(resolved, &key_before)?;
        return Ok((vault, ledger_view, true, key_before));
    }

    let vault = Arc::new(AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        VaultOptions::default(),
    )?);
    let ledger_view = Arc::new(AsterLedgerCfStore::open(&resolved.path)?);
    validate_cached_vault(&vault, &key_before)?;
    validate_cached_ledger_view(&ledger_view, &key_before)?;
    ensure_vault_generation_unchanged(resolved, &key_before)?;
    *cache = Some(CachedVaultState {
        key: key_before.clone(),
        vault: Arc::clone(&vault),
        ledger_view: Arc::clone(&ledger_view),
    });
    Ok((vault, ledger_view, false, key_before))
}

fn vault_generation_key(
    resolved: &ResolvedVault,
    manifest: &VaultManifest,
) -> ToolResult<VaultGenerationKey> {
    let ledger_head = calyx_aster::ledger_head::read_head_anchor(&resolved.path)?;
    Ok(VaultGenerationKey {
        path: resolved.path.clone(),
        name: resolved.name.clone(),
        vault_id: resolved.vault_id,
        manifest_seq: manifest.manifest_seq,
        durable_seq: manifest.durable_seq,
        ledger_head_height: ledger_head.as_ref().map(|head| head.height),
        ledger_head_hash: ledger_head.map(|head| head.tip_hash),
    })
}

fn validate_cached_vault(vault: &AsterVault, expected: &VaultGenerationKey) -> ToolResult<()> {
    // A legacy manifest can retain the pre-#1808 broad derived-content field
    // while `AsterVault::open` physically migrates the in-memory frontier.
    // The immutable manifest sequence identifies those exact source bytes;
    // compare the opened identity/durable tip here and let Aster's recovered
    // derived frontier remain authoritative for search freshness.
    if vault.vault_id() != expected.vault_id || vault.latest_seq() != expected.durable_seq {
        return Err(CalyxError::stale_derived(format!(
            "opened MCP search vault identity/tip does not match CURRENT: expected vault={} durable_seq={}, opened vault={} durable_seq={}",
            expected.vault_id,
            expected.durable_seq,
            vault.vault_id(),
            vault.latest_seq()
        ))
        .into());
    }
    Ok(())
}

fn validate_cached_ledger_view(
    ledger_view: &AsterLedgerCfStore,
    expected: &VaultGenerationKey,
) -> ToolResult<()> {
    let actual = ledger_view
        .head_anchor()?
        .map(|head| (head.height, head.tip_hash));
    let expected = expected.ledger_head_height.zip(expected.ledger_head_hash);
    if actual != expected {
        return Err(CalyxError::ledger_chain_broken(
            "cached MCP search ledger view does not match the generation-bound head anchor",
        )
        .into());
    }
    Ok(())
}

pub(super) fn ensure_vault_generation_unchanged(
    resolved: &ResolvedVault,
    expected: &VaultGenerationKey,
) -> ToolResult<()> {
    let manifest = ManifestStore::open(&resolved.path).load_current()?;
    let actual = vault_generation_key(resolved, &manifest)?;
    if &actual != expected {
        return Err(CalyxError::stale_derived(
            "vault generation changed while validating the cached MCP search vault; retry against one stable CURRENT manifest",
        )
        .into());
    }
    Ok(())
}

/// Keep one physically validated panel generation alive for the process.
///
/// `VaultPanelState::clone` shares each registry entry's `Arc<dyn Lens>`, so a
/// cache hit reuses the exact lazy runtime/session objects initialized by the
/// preceding request. Hits rehash the referenced immutable panel/registry
/// assets without reconstructing their decoded runtime state. A changed panel
/// or registry reference performs the full validated load and replaces the
/// single entry, bounding retained generations to one.
pub(super) fn cached_panel_state(vault_dir: &Path) -> ToolResult<(VaultPanelState, bool)> {
    let manifest_before = ManifestStore::open(vault_dir).load_current()?;
    let key_before = PanelGenerationKey {
        panel_ref: manifest_before.panel_ref.clone(),
        registry_ref: manifest_before.registry_ref.clone(),
    };

    static CACHE: OnceLock<Mutex<Option<CachedPanelState>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let cached = cache
        .lock()
        .map_err(|_| {
            CalyxError::registry_unavailable(
                "MCP search panel runtime cache mutex was poisoned; restart calyx-mcp",
            )
        })?
        .as_ref()
        .filter(|cached| cached.key == key_before)
        .map(|cached| cached.state.clone());
    if let Some(cached) = cached {
        validate_immutable_ref(vault_dir, &key_before.panel_ref)?;
        if let Some(registry_ref) = key_before.registry_ref.as_ref() {
            validate_immutable_ref(vault_dir, registry_ref)?;
        }
        ensure_panel_generation_unchanged(vault_dir, &key_before)?;
        return Ok((cached, true));
    }

    // Serialize cache misses so concurrent callers cannot construct duplicate
    // CUDA runtime generations before either one publishes the entry.
    let mut cache = cache.lock().map_err(|_| {
        CalyxError::registry_unavailable(
            "MCP search panel runtime cache mutex was poisoned; restart calyx-mcp",
        )
    })?;
    if let Some(cached) = cache.as_ref()
        && cached.key == key_before
    {
        let cached = cached.state.clone();
        drop(cache);
        validate_immutable_ref(vault_dir, &key_before.panel_ref)?;
        if let Some(registry_ref) = key_before.registry_ref.as_ref() {
            validate_immutable_ref(vault_dir, registry_ref)?;
        }
        ensure_panel_generation_unchanged(vault_dir, &key_before)?;
        return Ok((cached, true));
    }
    let loaded = load_vault_panel_state(vault_dir)?;
    ensure_panel_generation_unchanged(vault_dir, &key_before)?;
    *cache = Some(CachedPanelState {
        key: key_before,
        state: loaded.clone(),
    });
    Ok((loaded, false))
}

fn validate_immutable_ref(vault_dir: &Path, reference: &ImmutableRef) -> ToolResult<()> {
    let path = vault_dir.join(&reference.logical_path);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        CalyxError::aster_corrupt_shard(format!(
            "read immutable MCP panel runtime asset metadata {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "immutable MCP panel runtime asset is not a regular non-symlink file: {}",
            path.display()
        ))
        .into());
    }
    let bytes = fs::read(&path).map_err(|error| {
        CalyxError::aster_corrupt_shard(format!(
            "read immutable MCP panel runtime asset {}: {error}",
            path.display()
        ))
    })?;
    let actual = blake3::hash(&bytes).to_hex().to_string();
    if actual != reference.blake3_hex {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "immutable MCP panel runtime asset {} BLAKE3 {actual} != manifest {}",
            path.display(),
            reference.blake3_hex
        ))
        .into());
    }
    Ok(())
}

fn ensure_panel_generation_unchanged(
    vault_dir: &Path,
    expected: &PanelGenerationKey,
) -> ToolResult<()> {
    let manifest = ManifestStore::open(vault_dir).load_current()?;
    let actual = PanelGenerationKey {
        panel_ref: manifest.panel_ref,
        registry_ref: manifest.registry_ref,
    };
    if &actual != expected {
        return Err(CalyxError::stale_derived(
            "vault panel or registry generation changed while validating the MCP search runtime; retry against one stable manifest generation",
        )
        .into());
    }
    Ok(())
}
