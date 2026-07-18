use std::path::{Path, PathBuf};

use calyx_core::{CalyxError, VaultId};

use super::{
    ResolvedVault, VaultIndex, VaultIndexEntry, index_path, read_index, read_vault_identity,
    vault_identity_mismatch, vault_identity_missing,
};
use crate::error::CliResult;

pub(crate) fn resolve_vault(home: &Path, vault: &str) -> CliResult<PathBuf> {
    resolve_vault_info(home, vault).map(|resolved| resolved.path)
}

pub(crate) fn resolve_vault_info(home: &Path, vault: &str) -> CliResult<ResolvedVault> {
    let index = read_index(home)?;
    let checked_index = index_path(home);
    let direct = PathBuf::from(vault);
    // A bare argument (one path component) is a logical vault reference —
    // vault id or CLI-index name — and must never be captured by an
    // incidental same-named entry in the process cwd (#1082). Filesystem
    // paths must be explicit: absolute, or multi-component like ./name.
    let explicit_path = direct.is_absolute() || direct.components().count() > 1;
    if explicit_path {
        if !path_exists(&direct, "direct vault path")? {
            return Err(CalyxError::vault_access_denied(format!(
                "direct vault path {} does not exist; pass an existing vault directory, a vault id, or a name from CLI index {}",
                direct.display(),
                checked_index.display()
            ))
            .into());
        }
        let indexed = resolve_direct_indexed(home, &index, &direct)?;
        return resolve_bound_path(&direct, indexed, &checked_index);
    }
    if let Ok(vault_id) = vault.parse::<VaultId>() {
        let path = home.join("vaults").join(vault_id.to_string());
        if path_exists(&path, "vault id path")? {
            let indexed = index
                .vaults
                .iter()
                .find(|entry| entry.vault_id == vault_id)
                .map(|entry| resolve_entry(home, entry));
            return resolve_bound_path(&path, indexed, &checked_index);
        }
    }
    if let Some(entry) = index
        .vaults
        .iter()
        .find(|entry| entry.name == vault || entry.vault_id.to_string() == vault)
    {
        let resolved = resolve_entry(home, entry);
        if path_exists(&resolved.path, "indexed vault path")? {
            let path = resolved.path.clone();
            return resolve_bound_path(&path, Some(resolved), &checked_index);
        }
        return Err(CalyxError::vault_access_denied(format!(
            "vault {vault} resolved through CLI index {} to missing path {}",
            checked_index.display(),
            resolved.path.display()
        ))
        .into());
    }
    Err(CalyxError::vault_access_denied(format!(
        "vault {vault} does not exist; checked CLI index {} for vault ids and names; pass an absolute or ./-prefixed path for a direct vault directory",
        checked_index.display()
    ))
    .into())
}

/// Resolves the namespace that is safe to feed into `vault_salt`.
///
/// A physical identity is authoritative for new vaults. An exact active-index
/// binding keeps legacy vaults usable, but an unbound legacy path is refused:
/// synthesizing a name here changes all deterministic Cx IDs (#1794).
fn resolve_bound_path(
    direct: &Path,
    indexed: Option<ResolvedVault>,
    checked_index: &Path,
) -> CliResult<ResolvedVault> {
    let physical = read_vault_identity(direct)?;
    match (physical, indexed) {
        (Some(identity), Some(indexed)) => {
            if identity.vault_id != indexed.vault_id || identity.canonical_name != indexed.name {
                return Err(vault_identity_mismatch(format!(
                    "physical identity for {} binds vault_id={} canonical_name={:?}, but active index {} binds vault_id={} name={:?}",
                    direct.display(),
                    identity.vault_id,
                    identity.canonical_name,
                    checked_index.display(),
                    indexed.vault_id,
                    indexed.name
                )));
            }
            Ok(ResolvedVault {
                path: direct.to_path_buf(),
                name: identity.canonical_name,
                vault_id: identity.vault_id,
            })
        }
        (Some(identity), None) => Ok(ResolvedVault {
            path: direct.to_path_buf(),
            name: identity.canonical_name,
            vault_id: identity.vault_id,
        }),
        (None, Some(indexed)) => Ok(indexed),
        (None, None) => Err(vault_identity_missing(direct, checked_index)),
    }
}

fn resolve_direct_indexed(
    home: &Path,
    index: &VaultIndex,
    direct: &Path,
) -> CliResult<Option<ResolvedVault>> {
    let direct = direct.canonicalize()?;
    for entry in &index.vaults {
        let path = home.join(&entry.path);
        if path_exists(&path, "indexed vault path")? && path.canonicalize()? == direct {
            return Ok(Some(resolve_entry(home, entry)));
        }
    }
    Ok(None)
}

fn path_exists(path: &Path, context: &str) -> CliResult<bool> {
    path.try_exists().map_err(|error| {
        crate::error::CliError::io(format!("inspect {context} {}: {error}", path.display()))
    })
}

fn resolve_entry(home: &Path, entry: &VaultIndexEntry) -> ResolvedVault {
    ResolvedVault {
        path: home.join(&entry.path),
        name: entry.name.clone(),
        vault_id: entry.vault_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmd::vault::{read_vault_identity, vault_salt, write_index, write_vault_identity};
    use calyx_aster::vault::{AsterVault, VaultOptions};
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NONCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn physical_identity_keeps_cx_namespace_stable_across_home_and_path_forms() {
        let root = temp_root("stable-namespace");
        let home = root.join("home-a");
        let other_home = root.join("home-b");
        let vault_id = id("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let name = "legal-cuyahoga-stable";
        let vault = home.join("vaults").join(vault_id.to_string());
        fs::create_dir_all(&other_home).unwrap();
        create_real_vault(&vault, vault_id, name);
        write_vault_identity(&vault, vault_id, name).unwrap();
        write_index(
            &home,
            &VaultIndex {
                vaults: vec![VaultIndexEntry {
                    name: name.to_string(),
                    vault_id,
                    path: format!("vaults/{vault_id}"),
                    panel_template: "legal-default".to_string(),
                }],
                retired_vaults: Vec::new(),
            },
        )
        .unwrap();

        let by_name = resolve_vault_info(&home, name).unwrap();
        let by_indexed_path = resolve_vault_info(&home, vault.to_str().unwrap()).unwrap();
        let by_foreign_home_path =
            resolve_vault_info(&other_home, vault.to_str().unwrap()).unwrap();
        assert_eq!(by_name.name, name);
        assert_eq!(by_indexed_path.name, name);
        assert_eq!(by_foreign_home_path.name, name);
        assert_eq!(
            vault_salt(by_name.vault_id, &by_name.name),
            vault_salt(by_foreign_home_path.vault_id, &by_foreign_home_path.name)
        );

        let input = b"real synthetic Cuyahoga opinion text for stable identity";
        let first = open(&by_name).cx_id_for_input(input, 9);
        let second = open(&by_indexed_path).cx_id_for_input(input, 9);
        let third = open(&by_foreign_home_path).cx_id_for_input(input, 9);
        assert_eq!(first, second);
        assert_eq!(first, third);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn physical_identity_and_index_mismatch_fails_closed() {
        let root = temp_root("mismatch");
        let home = root.join("home");
        let vault_id = id("01ARZ3NDEKTSV4RRFFQ69G5FAA");
        let vault = home.join("vaults").join(vault_id.to_string());
        create_real_vault(&vault, vault_id, "physical-name");
        write_vault_identity(&vault, vault_id, "physical-name").unwrap();
        write_index(
            &home,
            &VaultIndex {
                vaults: vec![VaultIndexEntry {
                    name: "different-index-name".to_string(),
                    vault_id,
                    path: format!("vaults/{vault_id}"),
                    panel_template: "legal-default".to_string(),
                }],
                retired_vaults: Vec::new(),
            },
        )
        .unwrap();

        let error = resolve_vault_info(&home, vault.to_str().unwrap()).unwrap_err();
        assert_eq!(error.code(), "CALYX_VAULT_IDENTITY_MISMATCH");
        assert!(error.message().contains("physical-name"));
        assert!(error.message().contains("different-index-name"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unbound_legacy_direct_path_fails_instead_of_guessing_namespace() {
        let root = temp_root("unbound-legacy");
        let home = root.join("unrelated-home");
        let vault_id = id("01ARZ3NDEKTSV4RRFFQ69G5FAB");
        let vault = root.join("storage").join(vault_id.to_string());
        fs::create_dir_all(&home).unwrap();
        create_real_vault(&vault, vault_id, "legacy-original-name");
        let files_before = physical_files(&vault);

        let error = resolve_vault_info(&home, vault.to_str().unwrap()).unwrap_err();
        assert_eq!(error.code(), "CALYX_VAULT_IDENTITY_MISSING");
        assert_eq!(physical_files(&vault), files_before);
        assert!(!vault.join("VAULT_IDENTITY.json").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn indexed_legacy_vault_retains_its_original_namespace() {
        let root = temp_root("indexed-legacy");
        let home = root.join("home");
        let vault_id = id("01ARZ3NDEKTSV4RRFFQ69G5FAC");
        let name = "legacy-indexed-name";
        let vault = home.join("vaults").join(vault_id.to_string());
        create_real_vault(&vault, vault_id, name);
        write_index(
            &home,
            &VaultIndex {
                vaults: vec![VaultIndexEntry {
                    name: name.to_string(),
                    vault_id,
                    path: format!("vaults/{vault_id}"),
                    panel_template: "legal-default".to_string(),
                }],
                retired_vaults: Vec::new(),
            },
        )
        .unwrap();

        let resolved = resolve_vault_info(&home, vault.to_str().unwrap()).unwrap();
        assert_eq!(resolved.name, name);
        assert!(read_vault_identity(&vault).unwrap().is_none());
        let expected = AsterVault::open(
            &vault,
            vault_id,
            vault_salt(vault_id, name),
            VaultOptions::default(),
        )
        .unwrap()
        .cx_id_for_input(b"legacy content", 1);
        assert_eq!(
            open(&resolved).cx_id_for_input(b"legacy content", 1),
            expected
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_physical_identity_is_typed_and_never_ignored() {
        let root = temp_root("corrupt");
        let home = root.join("home");
        let vault_id = id("01ARZ3NDEKTSV4RRFFQ69G5FAD");
        let vault = root.join("storage").join(vault_id.to_string());
        fs::create_dir_all(&home).unwrap();
        create_real_vault(&vault, vault_id, "corrupt-test");
        fs::write(vault.join("VAULT_IDENTITY.json"), b"{not-json\n").unwrap();

        let error = resolve_vault_info(&home, vault.to_str().unwrap()).unwrap_err();
        assert_eq!(error.code(), "CALYX_VAULT_IDENTITY_CORRUPT");
        fs::remove_dir_all(root).unwrap();
    }

    fn create_real_vault(path: &Path, vault_id: VaultId, name: &str) {
        AsterVault::new_durable(
            path,
            vault_id,
            vault_salt(vault_id, name),
            VaultOptions::default(),
        )
        .unwrap();
    }

    fn open(resolved: &ResolvedVault) -> AsterVault {
        AsterVault::open(
            &resolved.path,
            resolved.vault_id,
            vault_salt(resolved.vault_id, &resolved.name),
            VaultOptions::default(),
        )
        .unwrap()
    }

    fn id(value: &str) -> VaultId {
        value.parse().unwrap()
    }

    fn physical_files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        fn walk(root: &Path, path: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
            let mut entries = fs::read_dir(path)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>();
            entries.sort();
            for entry in entries {
                if entry.is_dir() {
                    walk(root, &entry, out);
                } else {
                    out.insert(
                        entry.strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(entry).unwrap(),
                    );
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(root, root, &mut out);
        out
    }

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "calyx-vault-identity-{name}-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
