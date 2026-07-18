use std::time::Instant;

use calyx_sextant::index::{DenseVectorFile, PartitionedSearch, PartitionedSearchOptions, gen_row};
use serde_json::json;

use super::args::SearchArgs;
use super::brute_force::{exact_topk_synthetic, exact_topk_vecfile};
use super::{
    GROUNDED_PHASE_EXIT_ELIGIBLE, METRIC_CLASS_ANN_CORRECTNESS, enforce_recall_floor,
    map_ann_rows_to_ground_truth_ids, partitioned_error, percentiles,
    recall_from_i32bin_ground_truth, recall_from_truth_sets, row_for_metric, summarize_u64,
    tuner_status,
};
use crate::error::{CliError, CliResult};

/// Real query embeddings and exact ground truth over the real corpus.
pub(super) fn run_real(args: &SearchArgs) -> CliResult {
    if args.anneal_vault.is_some() {
        return Err(partitioned_error(
            "CALYX_FSV_PARTITIONED_SEARCH_TUNER_REAL_RECALL_REQUIRED",
            "partitioned-search tuner status needs per-query recall; real multi-lens gates should use partitioned-rrf",
            "rerun real fused evidence with bench partitioned-rrf --anneal-vault and ground truth",
        ));
    }
    let search = PartitionedSearch::open(&args.vault).map_err(CliError::Calyx)?;
    let manifest = search.manifest().clone();
    let distance_metric = manifest.distance_metric;
    let queries_path = args.queries.as_ref().expect("real mode");
    let q_vecs = DenseVectorFile::open(queries_path).map_err(CliError::Calyx)?;
    if q_vecs.dim() != manifest.dim {
        return Err(CliError::usage(format!(
            "query dim {} != vault dim {}",
            q_vecs.dim(),
            manifest.dim
        )));
    }
    let n = args.n.min(q_vecs.count() as usize);
    let mut latencies_us: Vec<u64> = Vec::with_capacity(n);
    let mut region_touch_counts = Vec::with_capacity(n);
    let mut first_touched_regions = Vec::new();
    let gt_n = args.ground_truth.min(n);
    let mut gt_queries: Vec<Vec<f32>> = Vec::with_capacity(gt_n);
    let mut gt_ann: Vec<Vec<u64>> = Vec::with_capacity(gt_n);
    let search_opts = PartitionedSearchOptions {
        n_probe: args.n_probe,
        region_beam: args.region_beam,
        pruning_epsilon: args.pruning_epsilon,
    };
    for i in 0..n {
        let q = row_for_metric(&q_vecs, i as u64, distance_metric);
        let started = Instant::now();
        let readback = search
            .search_with_readback_opts(&q, args.k, search_opts)
            .map_err(CliError::Calyx)?;
        let hits = readback.hits;
        if first_touched_regions.is_empty() {
            first_touched_regions = readback.touched_regions.clone();
        }
        region_touch_counts.push(readback.touched_regions.len() as u64);
        latencies_us.push((started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64).max(1));
        if i < gt_n {
            gt_ann.push(hits.iter().map(|(cx, _)| *cx).collect());
            gt_queries.push(q);
        }
    }
    let summary = percentiles(&latencies_us);

    let (ground_truth_recall, ground_truth_execution) = if gt_n == 0 {
        (None, None)
    } else if let Some(path) = &args.ground_truth_file {
        let mapped_ann;
        let ann = if let Some(id_map) = &args.ground_truth_id_map {
            mapped_ann = map_ann_rows_to_ground_truth_ids(id_map, &gt_ann, manifest.n_cx)?;
            &mapped_ann
        } else {
            &gt_ann
        };
        (
            Some(recall_from_i32bin_ground_truth(path, ann, args.k)?),
            None,
        )
    } else {
        let corpus_path = args.corpus.as_ref().ok_or_else(|| {
            CliError::usage(
                "--corpus <file.fbin|file.i8bin> or --ground-truth-file <file.i32bin> is required with --ground-truth in real mode",
            )
        })?;
        let corpus = DenseVectorFile::open(corpus_path).map_err(CliError::Calyx)?;
        if corpus.dim() != manifest.dim {
            return Err(CliError::usage(format!(
                "corpus dim {} != vault dim {}",
                corpus.dim(),
                manifest.dim
            )));
        }
        let truth = exact_topk_vecfile(&corpus, &gt_queries, args.k, distance_metric)?;
        let recall = recall_from_truth_sets(&gt_ann, &truth.sets());
        (Some(recall), Some(truth.execution))
    };
    enforce_recall_floor(args.recall_floor, gt_n, ground_truth_recall)?;
    let serving = search.serving_diagnostics();

    let report = json!({
        "trigger": "calyx bench partitioned-search",
        "mode": "real",
        "metric_class": METRIC_CLASS_ANN_CORRECTNESS,
        "grounded_phase_exit_eligible": GROUNDED_PHASE_EXIT_ELIGIBLE,
        "vault": args.vault.to_string_lossy(),
        "queries_file": queries_path.to_string_lossy(),
        "n_cx": manifest.n_cx,
        "dim": manifest.dim,
        "n_regions": manifest.n_regions,
        "queries": n,
        "k": args.k,
        "n_probe": args.n_probe,
        "region_beam": args.region_beam,
        "pruning_epsilon": args.pruning_epsilon,
        "strategy": "KernelFirstPartitioned",
        "distance_metric": distance_metric.as_str(),
        "graph_build_backend": manifest.graph_build_backend.as_str(),
        "gpu_serving": serving,
        "region_touch_count": summarize_u64(&region_touch_counts),
        "max_touched_regions": region_touch_counts.iter().copied().max().unwrap_or(0),
        "first_touched_regions": first_touched_regions,
        "region_touch_bound": args.n_probe.min(manifest.n_regions),
        "latency_us": summary,
        "ground_truth_queries": gt_n,
        "ground_truth_file": args.ground_truth_file.as_ref().map(|path| path.to_string_lossy()),
        "ground_truth_id_map": args.ground_truth_id_map.as_ref().map(|path| path.to_string_lossy()),
        "ground_truth_recall_at_k": ground_truth_recall,
        "ground_truth_execution": ground_truth_execution,
        "recall_floor": args.recall_floor,
    });
    print_report(&report)
}

/// Synthetic search is a builder/latency harness, not a real-data recall claim.
pub(super) fn run_synthetic(args: &SearchArgs) -> CliResult {
    let search = PartitionedSearch::open(&args.vault).map_err(CliError::Calyx)?;
    let manifest = search.manifest().clone();
    let dim = manifest.dim;
    let n_cx = manifest.n_cx;
    let seed = manifest.seed;

    let mut latencies_us: Vec<u64> = Vec::with_capacity(args.n);
    let mut region_touch_counts = Vec::with_capacity(args.n);
    let mut first_touched_regions = Vec::new();
    let mut self_hits = 0usize;
    let mut self_recall_rows = Vec::with_capacity(args.n);
    let gt_n = args.ground_truth.min(args.n);
    let mut gt_queries: Vec<Vec<f32>> = Vec::with_capacity(gt_n);
    let mut gt_ann: Vec<Vec<u64>> = Vec::with_capacity(gt_n);
    let search_opts = PartitionedSearchOptions {
        n_probe: args.n_probe,
        region_beam: args.region_beam,
        pruning_epsilon: args.pruning_epsilon,
    };
    for i in 0..args.n {
        let idx = (seed.wrapping_add(i as u64 * 7919)) % n_cx;
        let q = gen_row(seed, idx, dim);
        let started = Instant::now();
        let readback = search
            .search_with_readback_opts(&q, args.k, search_opts)
            .map_err(CliError::Calyx)?;
        let hits = readback.hits;
        if first_touched_regions.is_empty() {
            first_touched_regions = readback.touched_regions.clone();
        }
        region_touch_counts.push(readback.touched_regions.len() as u64);
        let us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;
        latencies_us.push(us.max(1));
        let self_hit = hits.iter().any(|(cx, _)| *cx == idx);
        self_hits += usize::from(self_hit);
        self_recall_rows.push(if self_hit { 1.0 } else { 0.0 });
        if i < gt_n {
            gt_ann.push(hits.iter().map(|(cx, _)| *cx).collect());
            gt_queries.push(q);
        }
    }
    let summary = percentiles(&latencies_us);
    let self_recall = self_hits as f32 / args.n.max(1) as f32;

    let (ground_truth_recall, ground_truth_execution) = if gt_n > 0 {
        let truth = exact_topk_synthetic(
            seed,
            n_cx,
            dim,
            &gt_queries,
            args.k,
            manifest.distance_metric,
        )?;
        let recall = recall_from_truth_sets(&gt_ann, &truth.sets());
        (Some(recall), Some(truth.execution))
    } else {
        (None, None)
    };
    enforce_recall_floor(args.recall_floor, gt_n, ground_truth_recall)?;
    let serving = search.serving_diagnostics();
    let tuner_status_path = if let Some(vault) = &args.anneal_vault {
        Some(tuner_status::write(tuner_status::Request {
            vault,
            latencies_us: &latencies_us,
            per_query_recall: &self_recall_rows,
            region_beam: args.region_beam,
            n_probe: args.n_probe,
            tuner_slo_us: args.tuner_slo_us,
            recall_floor: args.recall_floor,
            aggregate_recall: self_recall,
            latency_us: &summary,
            queries: args.n,
            k: args.k,
        })?)
    } else {
        None
    };

    let report = json!({
        "trigger": "calyx bench partitioned-search",
        "mode": "synthetic",
        "metric_class": METRIC_CLASS_ANN_CORRECTNESS,
        "grounded_phase_exit_eligible": GROUNDED_PHASE_EXIT_ELIGIBLE,
        "vault": args.vault.to_string_lossy(),
        "n_cx": n_cx,
        "dim": dim,
        "n_regions": manifest.n_regions,
        "queries": args.n,
        "k": args.k,
        "n_probe": args.n_probe,
        "region_beam": args.region_beam,
        "pruning_epsilon": args.pruning_epsilon,
        "strategy": "KernelFirstPartitioned",
        "graph_build_backend": manifest.graph_build_backend.as_str(),
        "gpu_serving": serving,
        "region_touch_count": summarize_u64(&region_touch_counts),
        "max_touched_regions": region_touch_counts.iter().copied().max().unwrap_or(0),
        "first_touched_regions": first_touched_regions,
        "region_touch_bound": args.n_probe.min(manifest.n_regions),
        "latency_us": summary,
        "self_recall_at_k": self_recall,
        "ground_truth_queries": gt_n,
        "ground_truth_recall_at_k": ground_truth_recall,
        "ground_truth_execution": ground_truth_execution,
        "recall_floor": args.recall_floor,
        "tuner_status_path": tuner_status_path,
    });
    print_report(&report)
}

fn print_report(report: &serde_json::Value) -> CliResult {
    println!(
        "{}",
        serde_json::to_string_pretty(report)
            .map_err(|error| CliError::runtime(format!("serialize bench report: {error}")))?
    );
    Ok(())
}
