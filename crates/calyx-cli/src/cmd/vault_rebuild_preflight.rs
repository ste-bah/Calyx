use std::fs;
use std::path::{Path, PathBuf};

use calyx_aster::pressure::disk_capacity_bytes;
use calyx_core::CalyxError;
use serde::Serialize;

use super::vault::{home_dir, resolve_vault_info};
use crate::durable_write::write_bytes_atomic_new;
use crate::error::{CliError, CliResult};
use crate::output::print_json;

const DEFAULT_TEMPORARY_BPS: u64 = 2_500;
const DEFAULT_SAFETY_BPS: u64 = 1_000;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    vault: String,
    target: PathBuf,
    projected_bytes: Option<u64>,
    rollback_reserve_bytes: Option<u64>,
    temporary_bps: u64,
    safety_bps: u64,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct TreeSize {
    logical_bytes: u64,
    physical_bytes: u64,
    files: u64,
    directories: u64,
    physical_size_basis: &'static str,
}

#[derive(Debug, Serialize)]
struct Report {
    status: &'static str,
    verdict: &'static str,
    source_vault: String,
    source_vault_id: String,
    source_path: String,
    source: TreeSize,
    target_path: String,
    target_total_bytes: u64,
    target_available_bytes: u64,
    projected_rebuild_bytes: u64,
    rollback_reserve_bytes: u64,
    temporary_generation_reserve_bytes: u64,
    safety_reserve_bytes: u64,
    total_required_bytes: u64,
    available_after_bytes: Option<u64>,
    temporary_bps: u64,
    safety_bps: u64,
    source_and_target_same_filesystem: Option<bool>,
    prior_vault_policy: &'static str,
    constellation_policy: &'static str,
}

pub(crate) fn try_run(args: &[String]) -> Option<CliResult> {
    if args.first().map(String::as_str) != Some("vault-rebuild-preflight") {
        return None;
    }
    if matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) {
        return Some(crate::usage::print_command_usage("vault-rebuild-preflight"));
    }
    Some(parse(&args[1..]).and_then(run))
}

fn parse(args: &[String]) -> CliResult<Args> {
    let vault = args
        .first()
        .filter(|value| !value.starts_with('-'))
        .cloned()
        .ok_or_else(|| CliError::usage("vault-rebuild-preflight requires <source-vault>"))?;
    let mut target = None;
    let mut projected_bytes = None;
    let mut rollback_reserve_bytes = None;
    let mut temporary_bps = DEFAULT_TEMPORARY_BPS;
    let mut safety_bps = DEFAULT_SAFETY_BPS;
    let mut out = None;
    let mut home = None;
    let mut index = 1;
    while index < args.len() {
        let flag = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| CliError::usage(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--target" => target = Some(PathBuf::from(value)),
            "--projected-bytes" => {
                projected_bytes = Some(parse_positive_u64(flag, value)?);
            }
            "--rollback-reserve-bytes" => {
                rollback_reserve_bytes = Some(parse_positive_u64(flag, value)?);
            }
            "--temporary-bps" => temporary_bps = parse_bps(flag, value)?,
            "--safety-bps" => safety_bps = parse_bps(flag, value)?,
            "--out" => out = Some(PathBuf::from(value)),
            "--home" => home = Some(PathBuf::from(value)),
            other => {
                return Err(CliError::usage(format!(
                    "unexpected vault-rebuild-preflight flag {other}"
                )));
            }
        }
        index += 2;
    }
    Ok(Args {
        vault,
        target: target.ok_or_else(|| {
            CliError::usage("vault-rebuild-preflight requires --target <existing-dir>")
        })?,
        projected_bytes,
        rollback_reserve_bytes,
        temporary_bps,
        safety_bps,
        out,
        home,
    })
}

fn parse_positive_u64(flag: &str, value: &str) -> CliResult<u64> {
    value
        .parse::<u64>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| CliError::usage(format!("{flag} requires an integer greater than zero")))
}

fn parse_bps(flag: &str, value: &str) -> CliResult<u64> {
    value
        .parse::<u64>()
        .ok()
        .filter(|parsed| *parsed <= 10_000)
        .ok_or_else(|| CliError::usage(format!("{flag} requires an integer in 0..=10000")))
}

fn run(args: Args) -> CliResult {
    let home = args.home.clone().map_or_else(home_dir, Ok)?;
    let resolved = resolve_vault_info(&home, &args.vault)?;
    let target_metadata = fs::symlink_metadata(&args.target).map_err(|error| {
        CliError::io(format!(
            "inspect rebuild target {} failed: {error}",
            args.target.display()
        ))
    })?;
    if target_metadata.file_type().is_symlink() || !target_metadata.is_dir() {
        return Err(CliError::from(CalyxError {
            code: "CALYX_VAULT_REBUILD_TARGET_INVALID",
            message: format!(
                "rebuild target {} must be an existing non-symlink directory",
                args.target.display()
            ),
            remediation: "choose the explicit existing filesystem or dataset root that will own the candidate vault",
        }));
    }
    let target = fs::canonicalize(&args.target).map_err(|error| {
        CliError::io(format!(
            "canonicalize rebuild target {} failed: {error}",
            args.target.display()
        ))
    })?;
    let source = measure_tree(&resolved.path)?;
    let measured_full_vault = source.logical_bytes.max(source.physical_bytes);
    let projected_rebuild_bytes = args.projected_bytes.unwrap_or(measured_full_vault);
    let rollback_reserve_bytes = args.rollback_reserve_bytes.unwrap_or(measured_full_vault);
    let temporary_generation_reserve_bytes =
        bps_reserve(projected_rebuild_bytes, args.temporary_bps)?;
    let safety_reserve_bytes = bps_reserve(projected_rebuild_bytes, args.safety_bps)?;
    let total_required_bytes = [
        projected_rebuild_bytes,
        rollback_reserve_bytes,
        temporary_generation_reserve_bytes,
        safety_reserve_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| total.checked_add(value))
    .ok_or_else(|| CliError::from(budget_overflow("total required bytes")))?;
    let capacity = disk_capacity_bytes(&target)?;
    let allowed = capacity.available >= total_required_bytes;
    let report = Report {
        status: "complete",
        verdict: if allowed { "allow" } else { "refuse" },
        source_vault: resolved.name,
        source_vault_id: resolved.vault_id.to_string(),
        source_path: resolved.path.display().to_string(),
        source,
        target_path: target.display().to_string(),
        target_total_bytes: capacity.total,
        target_available_bytes: capacity.available,
        projected_rebuild_bytes,
        rollback_reserve_bytes,
        temporary_generation_reserve_bytes,
        safety_reserve_bytes,
        total_required_bytes,
        available_after_bytes: capacity.available.checked_sub(total_required_bytes),
        temporary_bps: args.temporary_bps,
        safety_bps: args.safety_bps,
        source_and_target_same_filesystem: same_filesystem(&resolved.path, &target)?,
        prior_vault_policy: "preserve sacred prior vault bytes until an explicit, separately authorized retirement",
        constellation_policy: "capacity planning never narrows, flattens, concatenates, averages, or substitutes the panel's separate typed lens slots",
    };
    if let Some(path) = args.out.as_deref() {
        let mut bytes = serde_json::to_vec_pretty(&report)
            .map_err(|error| CliError::runtime(format!("serialize rebuild preflight: {error}")))?;
        bytes.push(b'\n');
        write_bytes_atomic_new(path, &bytes, "vault rebuild preflight report")?;
    }
    print_json(&report)?;
    if !allowed {
        return Err(CliError::from(CalyxError {
            code: "CALYX_VAULT_REBUILD_HEADROOM",
            message: format!(
                "rebuild target {} has {} available bytes but requires {}: projected={} rollback={} temporary={} safety={}",
                target.display(),
                capacity.available,
                total_required_bytes,
                projected_rebuild_bytes,
                rollback_reserve_bytes,
                temporary_generation_reserve_bytes,
                safety_reserve_bytes,
            ),
            remediation: "expand the target dataset/quota or choose an explicitly measured tiered-build target; never delete a sacred prior vault implicitly",
        }));
    }
    Ok(())
}

fn bps_reserve(value: u64, bps: u64) -> CliResult<u64> {
    value
        .checked_mul(bps)
        .and_then(|scaled| scaled.checked_add(9_999))
        .map(|scaled| scaled / 10_000)
        .ok_or_else(|| CliError::from(budget_overflow("basis-point reserve")))
}

fn budget_overflow(context: &str) -> CalyxError {
    CalyxError {
        code: "CALYX_VAULT_REBUILD_BUDGET_OVERFLOW",
        message: format!("vault rebuild {context} overflowed u64"),
        remediation: "correct the projected/reserve byte inputs before creating a candidate vault",
    }
}

fn measure_tree(root: &Path) -> CliResult<TreeSize> {
    let mut stack = vec![root.to_path_buf()];
    let mut logical_bytes = 0_u64;
    let mut physical_bytes = 0_u64;
    let mut files = 0_u64;
    let mut directories = 0_u64;
    while let Some(path) = stack.pop() {
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            CliError::io(format!(
                "inspect vault member {} failed: {error}",
                path.display()
            ))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(CliError::from(CalyxError {
                code: "CALYX_VAULT_REBUILD_SIZE_UNBOUNDED",
                message: format!(
                    "vault member {} is a symlink, so local byte measurement is incomplete",
                    path.display()
                ),
                remediation: "measure every explicit tier/source root and pass a complete projected byte forecast",
            }));
        }
        physical_bytes = physical_bytes
            .checked_add(allocated_bytes(&metadata))
            .ok_or_else(|| CliError::from(budget_overflow("physical source size")))?;
        if metadata.is_dir() {
            directories = directories
                .checked_add(1)
                .ok_or_else(|| CliError::from(budget_overflow("directory count")))?;
            for entry in fs::read_dir(&path).map_err(|error| {
                CliError::io(format!(
                    "list vault directory {} failed: {error}",
                    path.display()
                ))
            })? {
                stack.push(
                    entry
                        .map_err(|error| {
                            CliError::io(format!(
                                "read vault directory entry in {} failed: {error}",
                                path.display()
                            ))
                        })?
                        .path(),
                );
            }
        } else if metadata.is_file() {
            files = files
                .checked_add(1)
                .ok_or_else(|| CliError::from(budget_overflow("file count")))?;
            logical_bytes = logical_bytes
                .checked_add(metadata.len())
                .ok_or_else(|| CliError::from(budget_overflow("logical source size")))?;
        }
    }
    Ok(TreeSize {
        logical_bytes,
        physical_bytes,
        files,
        directories,
        physical_size_basis: physical_size_basis(),
    })
}

#[cfg(unix)]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.blocks().saturating_mul(512)
}

#[cfg(not(unix))]
fn allocated_bytes(metadata: &fs::Metadata) -> u64 {
    metadata.len()
}

#[cfg(unix)]
const fn physical_size_basis() -> &'static str {
    "st_blocks_x_512"
}

#[cfg(not(unix))]
const fn physical_size_basis() -> &'static str {
    "logical_bytes_platform_fallback"
}

#[cfg(unix)]
fn same_filesystem(source: &Path, target: &Path) -> CliResult<Option<bool>> {
    use std::os::unix::fs::MetadataExt;
    let source = fs::metadata(source).map_err(|error| {
        CliError::io(format!(
            "stat source filesystem {} failed: {error}",
            source.display()
        ))
    })?;
    let target = fs::metadata(target).map_err(|error| {
        CliError::io(format!(
            "stat target filesystem {} failed: {error}",
            target.display()
        ))
    })?;
    Ok(Some(source.dev() == target.dev()))
}

#[cfg(not(unix))]
fn same_filesystem(_source: &Path, _target: &Path) -> CliResult<Option<bool>> {
    Ok(None)
}
