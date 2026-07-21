mod cpu;
mod dispatch;
mod types;

#[cfg(all(test, feature = "cuda"))]
mod issue1519_fsv;
#[cfg(test)]
mod tests;

pub use types::{
    DEFAULT_MAX_GROUPS, DEFAULT_MAX_ROWS, OLAP_CUDA_MIN_ROWS, OlapAggregate, OlapExecutionStats,
    OlapGroupAggregate, OlapScanPlan, OlapScanResult, olap_sum_tolerance,
};

use crate::mmap_col::MmapColumn;
use crate::sst::arrow::{ArrowColumnView, decode_column_shape};
use crate::vault::{AsterVault, SlotColumnManifest};
use calyx_core::{CalyxError, Clock, Result, Seq, SlotId};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_MAGIC: &str = "CXSC1";
const MANIFEST_VERSION: u32 = 1;
const CHUNK_FILE: &str = "slot-column.cxa1";

impl<C> AsterVault<C>
where
    C: Clock,
{
    pub fn olap_scan_aggregate_slot_at(
        &self,
        snapshot: Seq,
        slot: SlotId,
        output_dir: impl AsRef<Path>,
        plan: OlapScanPlan,
    ) -> Result<OlapScanResult> {
        let materialized = self.materialize_slot_column_at(snapshot, slot, output_dir)?;
        scan_materialized_slot_column_aggregate(&materialized.manifest_path, plan)
    }
}

pub fn scan_materialized_slot_column_aggregate(
    manifest_path: impl AsRef<Path>,
    plan: OlapScanPlan,
) -> Result<OlapScanResult> {
    let manifest_path = manifest_path.as_ref();
    let manifest = read_manifest(manifest_path)?;
    let chunk_path = chunk_path_for(manifest_path, &manifest)?;
    let column = MmapColumn::open(&chunk_path)?;
    let chunk_sha256 = sha256_hex(column.as_bytes());
    if chunk_sha256 != manifest.chunk_sha256 {
        return Err(CalyxError::aster_corrupt_shard(
            "slot column chunk sha256 mismatch",
        ));
    }
    let chunk = decode_column_shape(column.as_bytes())?;
    validate_manifest_shape(&manifest, &chunk)?;
    validate_plan(plan, chunk.dim(), chunk.n_rows())?;
    let (aggregate, groups, execution) = dispatch::scan(&chunk, plan)?;
    Ok(OlapScanResult {
        source_manifest_path: manifest_path.to_path_buf(),
        source_chunk_path: chunk_path,
        chunk_sha256,
        rows_scanned: chunk.n_rows(),
        dim: chunk.dim(),
        value_column: plan.value_column,
        group_by_column: plan.group_by_column,
        aggregate,
        groups,
        execution,
    })
}

fn validate_plan(plan: OlapScanPlan, dim: usize, rows: usize) -> Result<()> {
    if plan.max_rows == 0 {
        return Err(olap_error(
            "CALYX_OLAP_INVALID_PLAN",
            "max_rows must be > 0",
        ));
    }
    if rows > plan.max_rows {
        return Err(olap_error(
            "CALYX_OLAP_SCAN_LIMIT",
            format!("row cap {} exceeded by {rows}", plan.max_rows),
        ));
    }
    if plan.value_column >= dim {
        return Err(olap_error(
            "CALYX_OLAP_INVALID_PLAN",
            format!("value column {} outside dim {dim}", plan.value_column),
        ));
    }
    if let Some(group_by) = plan.group_by_column {
        if group_by >= dim {
            return Err(olap_error(
                "CALYX_OLAP_INVALID_PLAN",
                format!("group column {group_by} outside dim {dim}"),
            ));
        }
        if plan.max_groups == 0 {
            return Err(olap_error(
                "CALYX_OLAP_INVALID_PLAN",
                "max_groups must be > 0 when group_by is set",
            ));
        }
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<SlotColumnManifest> {
    let bytes = fs::read(path)
        .map_err(|error| olap_error("CALYX_OLAP_IO", format!("read manifest: {error}")))?;
    let manifest: SlotColumnManifest = serde_json::from_slice(&bytes).map_err(|error| {
        CalyxError::aster_corrupt_shard(format!("decode slot column manifest: {error}"))
    })?;
    if manifest.magic != MANIFEST_MAGIC || manifest.version != MANIFEST_VERSION {
        return Err(CalyxError::aster_corrupt_shard(
            "slot column manifest version mismatch",
        ));
    }
    Ok(manifest)
}

fn chunk_path_for(manifest_path: &Path, manifest: &SlotColumnManifest) -> Result<PathBuf> {
    if manifest.chunk_file != CHUNK_FILE {
        return Err(CalyxError::aster_corrupt_shard(
            "slot column manifest chunk path invalid",
        ));
    }
    let parent = manifest_path
        .parent()
        .ok_or_else(|| olap_error("CALYX_OLAP_IO", "slot manifest has no parent"))?;
    Ok(parent.join(CHUNK_FILE))
}

fn validate_manifest_shape(
    manifest: &SlotColumnManifest,
    chunk: &ArrowColumnView<'_>,
) -> Result<()> {
    if chunk.n_rows() != manifest.rows || chunk.dim() != manifest.dim as usize {
        return Err(CalyxError::aster_corrupt_shard(
            "slot column manifest shape mismatch",
        ));
    }
    if manifest.cx_ids.len() != manifest.rows {
        return Err(CalyxError::aster_corrupt_shard(
            "slot column cx_id count mismatch",
        ));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn olap_error(code: &'static str, message: impl Into<String>) -> CalyxError {
    CalyxError {
        code,
        message: message.into(),
        remediation: "fix OLAP scan input or rebuild the materialized column chunk",
    }
}
