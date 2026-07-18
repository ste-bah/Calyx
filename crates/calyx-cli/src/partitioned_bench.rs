//! `calyx build-partitioned-vault` + `calyx bench partitioned-search` (#550).
//!
//! Non-materializing CLI surfaces over the memory-bounded partitioned vault
//! (`calyx_sextant::index::partitioned`). The builder streams rows per region; the
//! search generates query vectors on the fly via `gen_row` and routes to a few
//! region graphs — neither holds the full dataset. This is the path to the 1e8
//! KernelFirst SLO soak (the flat `build-bench-vault`/`bench` paths materialize
//! everything and cannot scale — see #703).

use calyx_sextant::index::{DenseVectorFile, I32BinMatrix, PartitionDistanceMetric};

use crate::error::{CliError, CliResult};

#[path = "partitioned_bench/args.rs"]
mod args;
mod brute_force;
#[path = "partitioned_bench/build.rs"]
mod build;
#[path = "partitioned_bench/multi_rrf.rs"]
mod multi_rrf;
#[path = "partitioned_bench/progress.rs"]
mod progress;
#[path = "partitioned_bench/report_readback.rs"]
mod report_readback;
#[path = "partitioned_bench/rrf_plan.rs"]
pub(crate) mod rrf_plan;
#[path = "partitioned_bench/rrf_plan_remap.rs"]
mod rrf_plan_remap;
#[path = "partitioned_bench/search.rs"]
mod search;
#[path = "partitioned_bench/slot_truth_generate.rs"]
mod slot_truth_generate;
#[path = "partitioned_bench/slot_truth_store.rs"]
mod slot_truth_store;
#[path = "partitioned_bench/summary.rs"]
mod summary;
#[path = "partitioned_bench/timeline_import.rs"]
mod timeline_import;
#[path = "partitioned_bench/timeline_store.rs"]
pub(crate) mod timeline_store;
#[path = "partitioned_bench/tuner_status.rs"]
mod tuner_status;
use args::{SearchArgs, parse, parse_pruning_epsilon, parse_recall_floor};
#[cfg(test)]
pub(crate) use build::BuildArgs;
pub(crate) use build::run as run_build;
use summary::{percentiles, summarize_u64};

const METRIC_CLASS_ANN_CORRECTNESS: &str = "ann_correctness";
const GROUNDED_PHASE_EXIT_ELIGIBLE: bool = false;

fn partitioned_error(
    code: &'static str,
    message: impl Into<String>,
    remediation: &'static str,
) -> CliError {
    CliError::Calyx(calyx_core::CalyxError {
        code,
        message: message.into(),
        remediation,
    })
}

fn enforce_recall_floor(floor: Option<f32>, gt_n: usize, recall: Option<f32>) -> CliResult {
    let Some(floor) = floor else {
        return Ok(());
    };
    let Some(measured) = recall.filter(|_| gt_n > 0) else {
        return Err(partitioned_error(
            "CALYX_FSV_PARTITIONED_GROUND_TRUTH_REQUIRED",
            format!("--recall-floor {floor:.6} requires --ground-truth > 0"),
            "rerun with --ground-truth N and read ground_truth_recall_at_k",
        ));
    };
    if measured + f32::EPSILON < floor {
        return Err(partitioned_error(
            "CALYX_FSV_PARTITIONED_RECALL_BELOW_FLOOR",
            format!("ground_truth_recall_at_k={measured:.6} below recall_floor={floor:.6}"),
            "increase n-probe/region-beam or rebuild/tune before claiming the recall gate",
        ));
    }
    Ok(())
}

fn row_for_metric(
    vectors: &DenseVectorFile,
    idx: u64,
    distance_metric: PartitionDistanceMetric,
) -> Vec<f32> {
    match distance_metric {
        PartitionDistanceMetric::UnitL2 => vectors.row_f32(idx),
        PartitionDistanceMetric::RawL2 => vectors.row_f32_raw(idx),
    }
}

fn recall_from_i32bin_ground_truth(
    path: &std::path::Path,
    ann: &[Vec<u64>],
    k: usize,
) -> CliResult<f32> {
    let truth = I32BinMatrix::open(path).map_err(CliError::Calyx)?;
    if truth.count() < ann.len() as u64 {
        return Err(CliError::usage(format!(
            "ground-truth file has {} rows, need {}",
            truth.count(),
            ann.len()
        )));
    }
    if truth.width() < k {
        return Err(CliError::usage(format!(
            "ground-truth file width {} is smaller than k {k}",
            truth.width()
        )));
    }
    let truth_sets = (0..ann.len())
        .map(|idx| {
            let row = truth.row(idx as u64);
            let mut set = std::collections::HashSet::with_capacity(k);
            for value in row.into_iter().take(k) {
                if value < 0 {
                    return Err(CliError::usage(format!(
                        "ground-truth row {idx} contains negative id {value}"
                    )));
                }
                set.insert(value as u64);
            }
            Ok(set)
        })
        .collect::<CliResult<Vec<_>>>()?;
    Ok(recall_from_truth_sets(ann, &truth_sets))
}

fn map_ann_rows_to_ground_truth_ids(
    path: &std::path::Path,
    ann: &[Vec<u64>],
    expected_rows: u64,
) -> CliResult<Vec<Vec<u64>>> {
    let id_map = I32BinMatrix::open(path).map_err(CliError::Calyx)?;
    if id_map.width() != 1 {
        return Err(CliError::usage(format!(
            "--ground-truth-id-map width must be 1, got {}",
            id_map.width()
        )));
    }
    if id_map.count() != expected_rows {
        return Err(CliError::usage(format!(
            "--ground-truth-id-map row count {} != vault n_cx {expected_rows}",
            id_map.count()
        )));
    }
    ann.iter()
        .map(|row| {
            row.iter()
                .map(|&row_id| {
                    if row_id >= id_map.count() {
                        return Err(CliError::usage(format!(
                            "ANN row id {row_id} is outside --ground-truth-id-map count {}",
                            id_map.count()
                        )));
                    }
                    let external_id = id_map.row(row_id)[0];
                    if external_id < 0 {
                        return Err(CliError::usage(format!(
                            "--ground-truth-id-map row {row_id} contains negative id {external_id}"
                        )));
                    }
                    Ok(external_id as u64)
                })
                .collect()
        })
        .collect()
}

fn recall_from_truth_sets(ann: &[Vec<u64>], truth: &[std::collections::HashSet<u64>]) -> f32 {
    let mut found = 0usize;
    let mut total = 0usize;
    for (ann, truth_set) in ann.iter().zip(truth.iter()) {
        found += ann.iter().filter(|cx| truth_set.contains(cx)).count();
        total += truth_set.len();
    }
    found as f32 / total.max(1) as f32
}

pub(crate) fn run_search(args: &[String]) -> CliResult {
    let args = SearchArgs::parse(args)?;
    if args.queries.is_some() {
        search::run_real(&args)
    } else {
        search::run_synthetic(&args)
    }
}

pub(crate) fn run_rrf(args: &[String]) -> CliResult {
    multi_rrf::run(args)
}

pub(crate) fn run_rrf_plan(args: &[String]) -> CliResult {
    rrf_plan::run_import(args)
}

pub(crate) fn run_rrf_plan_remap(args: &[String]) -> CliResult {
    rrf_plan_remap::run(args)
}

pub(crate) fn run_rrf_report_readback(args: &[String]) -> CliResult {
    report_readback::run(args)
}

pub(crate) fn run_rrf_slot_truth(args: &[String]) -> CliResult {
    slot_truth_generate::run(args)
}

pub(crate) fn run_rrf_timeline(args: &[String]) -> CliResult {
    timeline_import::run(args)
}

pub(crate) fn is_topic(topic: &str) -> bool {
    matches!(
        topic,
        "partitioned-search"
            | "partitioned-rrf"
            | "partitioned-rrf-plan"
            | "partitioned-rrf-plan-remap"
            | "partitioned-rrf-report-readback"
            | "partitioned-rrf-slot-truth"
            | "partitioned-rrf-timeline"
    )
}

pub(crate) fn run_topic(topic: &str, args: &[String]) -> CliResult {
    match topic {
        "partitioned-search" => run_search(args),
        "partitioned-rrf" => run_rrf(args),
        "partitioned-rrf-plan" => run_rrf_plan(args),
        "partitioned-rrf-plan-remap" => run_rrf_plan_remap(args),
        "partitioned-rrf-report-readback" => run_rrf_report_readback(args),
        "partitioned-rrf-slot-truth" => run_rrf_slot_truth(args),
        "partitioned-rrf-timeline" => run_rrf_timeline(args),
        _ => Err(CliError::usage(format!(
            "unknown partitioned bench topic: {topic}"
        ))),
    }
}

#[cfg(test)]
#[path = "partitioned_bench/progress_tests.rs"]
mod partitioned_bench_progress_tests;
#[cfg(test)]
#[path = "partitioned_bench_tests.rs"]
mod partitioned_bench_tests;
#[cfg(test)]
#[path = "partitioned_bench/rrf_failed_truth_write_tests.rs"]
mod rrf_failed_truth_write_tests;
#[cfg(test)]
#[path = "partitioned_bench/spann_knob_tests.rs"]
mod spann_knob_tests;
