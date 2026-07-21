use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_MAX_ROWS: usize = 1_000_000;
pub const DEFAULT_MAX_GROUPS: usize = 4096;
pub const OLAP_CUDA_MIN_ROWS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OlapScanPlan {
    pub value_column: usize,
    pub group_by_column: Option<usize>,
    pub max_rows: usize,
    pub max_groups: usize,
}

impl OlapScanPlan {
    pub const fn new(value_column: usize) -> Self {
        Self {
            value_column,
            group_by_column: None,
            max_rows: DEFAULT_MAX_ROWS,
            max_groups: DEFAULT_MAX_GROUPS,
        }
    }

    pub const fn with_group_by(mut self, group_by_column: usize) -> Self {
        self.group_by_column = Some(group_by_column);
        self
    }

    pub const fn with_limits(mut self, max_rows: usize, max_groups: usize) -> Self {
        self.max_rows = max_rows;
        self.max_groups = max_groups;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OlapAggregate {
    pub count: usize,
    pub sum: f64,
    pub min: f32,
    pub max: f32,
    pub avg: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OlapGroupAggregate {
    pub group_key_bits: u32,
    pub group_key: f32,
    pub aggregate: OlapAggregate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OlapScanResult {
    pub source_manifest_path: PathBuf,
    pub source_chunk_path: PathBuf,
    pub chunk_sha256: String,
    pub rows_scanned: usize,
    pub dim: usize,
    pub value_column: usize,
    pub group_by_column: Option<usize>,
    pub aggregate: OlapAggregate,
    pub groups: Vec<OlapGroupAggregate>,
    pub execution: OlapExecutionStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OlapExecutionStats {
    pub backend: String,
    pub pinned_staging: bool,
    pub chunks: u64,
    pub dictionary_capacity: u64,
    pub kernel_launches: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub peak_pinned_staging_bytes: u64,
    pub peak_device_bytes: u64,
    pub sum_abs_tolerance: f64,
    pub avg_abs_tolerance: f64,
}

pub fn olap_sum_tolerance(count: usize, min: f32, max: f32) -> f64 {
    let n = count as f64;
    let max_abs = f64::from(min).abs().max(f64::from(max).abs());
    (8.0 * f64::EPSILON * n * n * max_abs).max(1.0e-12)
}
