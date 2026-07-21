//! Durable answer-to-vault directory tied to every vault ledger head.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use calyx_aster::ledger_view::LedgerQuerySnapshot;
use calyx_core::{CalyxError, Result};
use serde::{Deserialize, Serialize};

const VERSION: u16 = 1;
const MAGIC: &[u8] = b"calyx_answer_vault_directory_v1\0";
const DIRECTORY: &str = ".answer_vault_directory";
const MAX_FILE_BYTES: u64 = 1 << 30;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct AnswerDirectoryStats {
    pub vaults_opened: u64,
    pub candidates: u64,
    pub directory_rebuilt: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct VaultGeneration {
    name: String,
    height: u64,
    tip_hash: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AnswerDirectory {
    version: u16,
    generations: Vec<VaultGeneration>,
    answers: BTreeMap<String, Vec<String>>,
}

pub(super) fn resolve_answer_vaults(
    vault_root: &Path,
    answer_id: &[u8],
) -> Result<(Vec<PathBuf>, AnswerDirectoryStats)> {
    let generations = read_generations(vault_root)?;
    let generation_bytes = serde_json::to_vec(&generations).map_err(|error| {
        CalyxError::ledger_corrupt(format!("encode answer directory generation: {error}"))
    })?;
    let generation_hash = hex(blake3::hash(&generation_bytes).as_bytes());
    let path = vault_root
        .join(DIRECTORY)
        .join(format!("{generation_hash}.bin"));
    let (directory, mut stats) = if path.exists() {
        (
            read_directory(&path, &generations)?,
            AnswerDirectoryStats::default(),
        )
    } else {
        let mut answers = BTreeMap::<String, Vec<String>>::new();
        let mut stats = AnswerDirectoryStats {
            directory_rebuilt: true,
            ..AnswerDirectoryStats::default()
        };
        for generation in &generations {
            let query = LedgerQuerySnapshot::open(&vault_root.join(&generation.name))?;
            stats.vaults_opened += 1;
            if query.height() != generation.height || query.tip_hash() != generation.tip_hash {
                return Err(CalyxError::ledger_chain_broken(format!(
                    "answer directory vault {} changed head during rebuild",
                    generation.name
                )));
            }
            for answer_id in query.answer_ids() {
                answers
                    .entry(hex(answer_id))
                    .or_default()
                    .push(generation.name.clone());
            }
        }
        for vaults in answers.values_mut() {
            vaults.sort();
            vaults.dedup();
        }
        let directory = AnswerDirectory {
            version: VERSION,
            generations: generations.clone(),
            answers,
        };
        write_directory(&path, &directory)?;
        (directory, stats)
    };
    let names = directory
        .answers
        .get(&hex(answer_id))
        .cloned()
        .unwrap_or_default();
    stats.candidates = names.len() as u64;
    Ok((
        names
            .into_iter()
            .map(|name| vault_root.join(name))
            .collect(),
        stats,
    ))
}

fn read_generations(vault_root: &Path) -> Result<Vec<VaultGeneration>> {
    if !vault_root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(vault_root).map_err(|error| {
        CalyxError::disk_pressure(format!(
            "read answer directory vault root {}: {error}",
            vault_root.display()
        ))
    })? {
        let entry = entry.map_err(|error| {
            CalyxError::disk_pressure(format!("read answer directory vault entry: {error}"))
        })?;
        if !entry
            .file_type()
            .map_err(|error| {
                CalyxError::disk_pressure(format!("stat answer directory vault entry: {error}"))
            })?
            .is_dir()
            || entry.file_name() == DIRECTORY
        {
            continue;
        }
        let name = entry.file_name().into_string().map_err(|_| {
            CalyxError::ledger_corrupt("answer directory vault name is not valid UTF-8")
        })?;
        let anchor = calyx_aster::ledger_head::read_head_anchor(&entry.path())?;
        let (height, tip_hash) =
            anchor.map_or((0, [0; 32]), |anchor| (anchor.height, anchor.tip_hash));
        out.push(VaultGeneration {
            name,
            height,
            tip_hash,
        });
    }
    out.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(out)
}

fn write_directory(path: &Path, directory: &AnswerDirectory) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        CalyxError::disk_pressure("answer directory path has no parent directory")
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        CalyxError::disk_pressure(format!(
            "create answer directory {}: {error}",
            parent.display()
        ))
    })?;
    let payload = serde_json::to_vec(directory)
        .map_err(|error| CalyxError::ledger_corrupt(format!("encode answer directory: {error}")))?;
    let mut bytes = Vec::with_capacity(MAGIC.len() + 8 + payload.len() + 32);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(blake3::hash(&payload).as_bytes());
    let temp = parent.join(format!(
        ".answer-directory-{}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|error| {
                CalyxError::disk_pressure(format!(
                    "create answer directory temp {}: {error}",
                    temp.display()
                ))
            })?;
        file.write_all(&bytes).map_err(|error| {
            CalyxError::disk_pressure(format!(
                "write answer directory temp {}: {error}",
                temp.display()
            ))
        })?;
        file.sync_all().map_err(|error| {
            CalyxError::disk_pressure(format!(
                "sync answer directory temp {}: {error}",
                temp.display()
            ))
        })?;
        drop(file);
        fs::rename(&temp, path).map_err(|error| {
            CalyxError::disk_pressure(format!(
                "publish answer directory {} -> {}: {error}",
                temp.display(),
                path.display()
            ))
        })?;
        remove_old_directories(parent, path)
    })();
    if result.is_err()
        && temp.exists()
        && let Err(cleanup) = fs::remove_file(&temp)
    {
        eprintln!(
            "CALYX_ANSWER_DIRECTORY_TEMP_CLEANUP_FAILED path={} error={cleanup}",
            temp.display()
        );
    }
    result
}

fn remove_old_directories(directory: &Path, keep: &Path) -> Result<()> {
    for entry in fs::read_dir(directory).map_err(|error| {
        CalyxError::disk_pressure(format!(
            "read answer directory generations {}: {error}",
            directory.display()
        ))
    })? {
        let path = entry
            .map_err(|error| {
                CalyxError::disk_pressure(format!("read answer directory generation: {error}"))
            })?
            .path();
        if path != keep && path.extension().is_some_and(|extension| extension == "bin") {
            fs::remove_file(&path).map_err(|error| {
                CalyxError::disk_pressure(format!(
                    "remove stale answer directory {}: {error}",
                    path.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn read_directory(path: &Path, generations: &[VaultGeneration]) -> Result<AnswerDirectory> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        CalyxError::disk_pressure(format!("stat answer directory {}: {error}", path.display()))
    })?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_FILE_BYTES {
        return Err(CalyxError::ledger_corrupt(format!(
            "answer directory {} is not a bounded regular file (bytes={})",
            path.display(),
            metadata.len()
        )));
    }
    let bytes = fs::read(path).map_err(|error| {
        CalyxError::disk_pressure(format!("read answer directory {}: {error}", path.display()))
    })?;
    let header = MAGIC.len() + 8;
    if bytes.len() < header + 32 || &bytes[..MAGIC.len()] != MAGIC {
        return Err(CalyxError::ledger_corrupt(format!(
            "answer directory {} has an invalid header",
            path.display()
        )));
    }
    let length = u64::from_be_bytes(
        bytes[MAGIC.len()..header]
            .try_into()
            .expect("eight-byte length"),
    );
    let length = usize::try_from(length)
        .map_err(|_| CalyxError::ledger_corrupt("answer directory length exceeds usize"))?;
    if bytes.len() != header + length + 32 {
        return Err(CalyxError::ledger_corrupt(format!(
            "answer directory {} length mismatch",
            path.display()
        )));
    }
    let payload = &bytes[header..header + length];
    if blake3::hash(payload).as_bytes() != &bytes[header + length..] {
        return Err(CalyxError::ledger_corrupt(format!(
            "answer directory {} checksum mismatch",
            path.display()
        )));
    }
    let directory: AnswerDirectory = serde_json::from_slice(payload).map_err(|error| {
        CalyxError::ledger_corrupt(format!(
            "decode answer directory {}: {error}",
            path.display()
        ))
    })?;
    if directory.version != VERSION || directory.generations != generations {
        return Err(CalyxError::ledger_corrupt(format!(
            "answer directory {} generation payload mismatch",
            path.display()
        )));
    }
    Ok(directory)
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
