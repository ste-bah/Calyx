//! Ground the MFVS kernel on the persisted association graph (#871).

mod admission;
mod admission_select;
mod model;
mod parse;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use calyx_aster::ledger_view::AsterLedgerCfStore;
use calyx_aster::plain_graph::PhysicalPlainGraph;
use calyx_aster::vault::{AsterVault, VaultOptions};
use calyx_core::CxId;
use calyx_ledger::{ActorId, EntryKind, LedgerCfStore, SubjectId, decode};
use calyx_lodestar::{
    ASTER_ASSOC_METADATA_KEY, AsterAssocMetadata, AsterAssocNodeProps, FsKernelStore, KernelParams,
    LodestarError, PANEL_ASTER_ASSOC_COLLECTION, PANEL_RRF_K, PanelVectors, RecallTestParams,
    build_kernel_pipeline, build_panel_kernel_index, kernel_health, kernel_members_hash,
    load_panel_kernel_index, panel_kernel_recall_gate, panel_kernel_recall_test,
    panel_rank_stabilization_support_set, read_kernel_artifact, refine_kernel_with_recall_support,
    seal_completed_kernel_identity, write_kernel_artifact, write_panel_kernel_index,
};
use calyx_registry::load_vault_panel_state;
use rayon::prelude::*;
use serde_json::json;

use super::Subcommand;
use super::kernel_generation::{
    KernelAdmissionContract, KernelGenerationManifest, KernelGraphContract, acquire_build_lock,
    artifact_ref, hex32, ledger_hash, physical_graph_contract, publish_current_generation,
    sha256_bytes,
};
use super::vault::{home_dir, now_ms, resolve_vault_info, vault_salt};
use crate::error::{CliError, CliResult};
use crate::output::print_json;

use self::model::{KernelBuildNodeProps, KernelBuildRefinement};

const DEFAULT_HELD_OUT_FRACTION: f32 = 0.005;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_MIN_RECALL: f32 = 0.95;
const RNG_SEED: u64 = 871;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KernelBuildArgs {
    pub vault: String,
    pub held_out_fraction: f32,
    pub top_k: usize,
    pub min_recall: f32,
    pub admission_queries: Option<PathBuf>,
    pub resident_addr: Option<SocketAddr>,
}

pub(crate) use parse::parse_kernel_build;

impl Default for KernelBuildArgs {
    fn default() -> Self {
        Self {
            vault: String::new(),
            held_out_fraction: DEFAULT_HELD_OUT_FRACTION,
            top_k: DEFAULT_TOP_K,
            min_recall: DEFAULT_MIN_RECALL,
            admission_queries: None,
            resident_addr: None,
        }
    }
}

pub(crate) fn run(command: Subcommand) -> CliResult {
    let Subcommand::KernelBuild(args) = command else {
        unreachable!("non-kernel-build command routed to kernel_build module");
    };
    run_kernel_build_with_home(&home_dir()?, args)
}

fn run_kernel_build_with_home(home: &Path, args: KernelBuildArgs) -> CliResult {
    let started = Instant::now();
    let resolved = resolve_vault_info(home, &args.vault)?;
    let _generation_lock = acquire_build_lock(&resolved.path)?;
    eprintln!(
        "kernel-build: opening physical graph name={} id={} path={}",
        resolved.name,
        resolved.vault_id,
        resolved.path.display()
    );
    let plain = PhysicalPlainGraph::open_latest(&resolved.path, PANEL_ASTER_ASSOC_COLLECTION)?;
    let metadata_bytes = plain
        .get_metadata(ASTER_ASSOC_METADATA_KEY)?
        .ok_or_else(|| {
            CliError::usage(
                "kernel-build requires a sealed weave graph contract; re-run `calyx weave-loom`",
            )
        })?;
    let graph_metadata: AsterAssocMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|error| CliError::runtime(format!("decode weave graph metadata: {error}")))?;
    if graph_metadata.embedding_slot.is_some() {
        return Err(CliError::usage(
            "woven graph uses the forbidden legacy single-vector contract; re-run `calyx weave-loom`",
        ));
    }
    let embedding_slots = graph_metadata.embedding_slots.clone();
    if embedding_slots.len() < 2
        || !embedding_slots.windows(2).all(|pair| pair[0] < pair[1])
        || graph_metadata.fusion.as_deref() != Some("rrf")
        || graph_metadata.rrf_k != Some(PANEL_RRF_K)
    {
        return Err(CliError::usage(
            "woven graph has no valid ordered no-flatten panel/RRF contract; re-run `calyx weave-loom`",
        ));
    }
    let panel_version = graph_metadata.panel_version.ok_or_else(|| {
        CliError::usage("woven graph metadata has no panel_version; re-run `calyx weave-loom`")
    })?;
    let graph_source_seq = graph_metadata.graph_source_seq.ok_or_else(|| {
        CliError::usage("woven graph metadata has no graph_source_seq; re-run `calyx weave-loom`")
    })?;
    let knn = graph_metadata
        .knn
        .filter(|value| *value > 0)
        .ok_or_else(|| CliError::usage("woven graph metadata has no valid knn"))?;
    let edge_score_threshold = graph_metadata
        .edge_score_threshold
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .ok_or_else(|| CliError::usage("woven graph metadata has no valid edge_score_threshold"))?;
    let panel_state = load_vault_panel_state(&resolved.path)?;
    if u64::from(panel_state.panel.version) != panel_version {
        return Err(CliError::usage(format!(
            "woven graph panel version {panel_version} differs from active vault panel {}; re-run `calyx weave-loom`",
            panel_state.panel.version
        )));
    }
    let mut expected_slots = panel_state
        .panel
        .slots
        .iter()
        .filter(|slot| {
            slot.state == calyx_core::SlotState::Active
                && !slot.retrieval_only
                && matches!(slot.shape, calyx_core::SlotShape::Dense(_))
        })
        .map(|slot| slot.slot_id)
        .collect::<Vec<_>>();
    expected_slots.sort();
    expected_slots.dedup();
    if embedding_slots != expected_slots {
        return Err(CliError::usage(format!(
            "woven graph slots {:?} differ from every active dense content slot {:?}; re-run `calyx weave-loom`",
            embedding_slots
                .iter()
                .map(|slot| slot.get())
                .collect::<Vec<_>>(),
            expected_slots
                .iter()
                .map(|slot| slot.get())
                .collect::<Vec<_>>()
        )));
    }
    let stage = Instant::now();
    let graph = plain.assoc_graph()?;
    eprintln!(
        "kernel-build: loaded graph nodes={} edges={} elapsed_ms={}",
        graph.node_count(),
        graph.edge_count(),
        stage.elapsed().as_millis()
    );
    if graph.node_count() < 2 {
        return Err(CliError::usage(format!(
            "kernel-build needs a woven association graph (>=2 nodes); graph CF has {} — run `calyx weave-loom` first",
            graph.node_count()
        )));
    }

    // Complete per-node constellation panel + anchors from physical graph props.
    let mut embeddings: BTreeMap<CxId, PanelVectors> = BTreeMap::new();
    let mut anchors: Vec<CxId> = Vec::new();
    let stage = Instant::now();
    let node_props = plain.node_props()?;
    let csr_bytes = plain.read_csr_bytes()?.ok_or_else(|| {
        CliError::usage(
            "kernel-build requires a persisted association CSR; re-run `calyx weave-loom`",
        )
    })?;
    let physical_node_count = plain.node_key_count()?;
    let physical_edge_count = plain.edge_out_key_count()?;
    if physical_node_count != graph.node_count() || physical_edge_count != graph.edge_count() {
        return Err(CliError::runtime(format!(
            "physical graph rows differ from CSR: node_rows={physical_node_count} csr_nodes={} edge_rows={physical_edge_count} csr_edges={}",
            graph.node_count(),
            graph.edge_count()
        )));
    }
    let (physical_contract_hash, node_props_sha256) =
        physical_graph_contract(&metadata_bytes, &node_props, &csr_bytes);
    let graph_node_ids = graph.node_ids().collect::<Vec<_>>();
    let prop_node_ids = node_props.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    if graph_node_ids != prop_node_ids {
        return Err(CliError::usage(format!(
            "graph topology has {} node keys but node props scan returned {}; rebuild the graph CF",
            graph_node_ids.len(),
            prop_node_ids.len()
        )));
    }
    let parsed_props = node_props
        .into_par_iter()
        .map(|(id, bytes)| -> CliResult<KernelBuildNodeProps> {
            let props: AsterAssocNodeProps = serde_json::from_slice(&bytes).map_err(|error| {
                CliError::runtime(format!("parse graph node {id} props: {error}"))
            })?;
            if props.embedding.is_some() {
                return Err(CliError::usage(format!(
                    "graph node {id} contains a forbidden legacy single embedding; re-run weave-loom"
                )));
            }
            if props.embeddings.keys().copied().ne(embedding_slots.iter().copied()) {
                return Err(CliError::usage(format!(
                    "graph node {id} slots {:?} differ from graph contract {:?}; re-run weave-loom",
                    props.embeddings.keys().map(|slot| slot.get()).collect::<Vec<_>>(),
                    embedding_slots.iter().map(|slot| slot.get()).collect::<Vec<_>>()
                )));
            }
            Ok(KernelBuildNodeProps {
                id,
                embeddings: props.embeddings,
                anchored: !props.anchors.is_empty(),
                metadata: props.metadata,
            })
        })
        .collect::<CliResult<Vec<_>>>()?;
    let jurisdiction = admission::jurisdiction(&parsed_props)?;
    for parsed in parsed_props {
        if parsed.anchored {
            anchors.push(parsed.id);
        }
        embeddings.insert(parsed.id, parsed.embeddings);
    }
    if embeddings.len() != graph.node_count() {
        return Err(CliError::usage(format!(
            "graph topology has {} nodes but parsed {} complete panel constellations; rebuild the graph CF",
            graph.node_count(),
            embeddings.len()
        )));
    }
    eprintln!(
        "kernel-build: loaded node props rows={} anchors={} panel_constellations={} lanes={} elapsed_ms={}",
        embeddings.len(),
        anchors.len(),
        embeddings.len(),
        embedding_slots.len(),
        stage.elapsed().as_millis()
    );
    if anchors.is_empty() {
        return Err(CliError::usage(
            "kernel-build found no anchored nodes in the graph; anchor the corpus before grounding a kernel",
        ));
    }

    let kernel_params = KernelParams {
        panel_version,
        anchor_kind: Some("any".to_string()),
        corpus_shard_hash: physical_contract_hash,
        built_at_millis: now_ms(),
        ..KernelParams::default()
    };
    let stage = Instant::now();
    let mut kernel = build_kernel_pipeline(&graph, &anchors, &kernel_params)?;
    eprintln!(
        "kernel-build: built kernel id={} members={} kernel_graph={} groundedness={:.6} elapsed_ms={}",
        kernel.kernel_id,
        kernel.members.len(),
        kernel.kernel_graph.len(),
        kernel.groundedness.reached_anchor,
        stage.elapsed().as_millis()
    );

    let recall_params = RecallTestParams {
        held_out_fraction: args.held_out_fraction,
        top_k: args.top_k,
        rng_seed: RNG_SEED,
        min_recall_ratio: args.min_recall,
    };
    let admission = admission_select::select(
        &resolved,
        &panel_state,
        &embedding_slots,
        &embeddings,
        jurisdiction.as_ref(),
        &args,
    )?;
    admission::emit(&admission);
    let corpus_name = format!("kernel-build:{}:panel-rrf", resolved.name);
    let mut kernel_index = build_panel_kernel_index(&kernel, &embedding_slots, &embeddings)?;
    let stage = Instant::now();
    let initial_recall =
        panel_kernel_recall_test(&kernel_index, &embeddings, &recall_params, &corpus_name)?;
    let mut refinement = None;
    if initial_recall.ratio < args.min_recall {
        eprintln!(
            "kernel-build: recall below gate before refinement ratio={:.6} min={:.6}; extracting exact full top-k support",
            initial_recall.ratio, args.min_recall
        );
        let initial_members = kernel.members.len();
        let initial_kernel_graph = kernel.kernel_graph.len();
        let mut lane_depth = recall_params.top_k.min(embeddings.len());
        let (support, refined_recall, final_lane_depth) = loop {
            let support = panel_rank_stabilization_support_set(
                &embedding_slots,
                &embeddings,
                &recall_params,
                lane_depth,
            )?;
            kernel = refine_kernel_with_recall_support(
                kernel,
                &support.members,
                &graph,
                &anchors,
                &kernel_params,
                &format!("exact_panel_rank_support_depth_{lane_depth}"),
            )?;
            kernel_index = build_panel_kernel_index(&kernel, &embedding_slots, &embeddings)?;
            let refined =
                panel_kernel_recall_test(&kernel_index, &embeddings, &recall_params, &corpus_name)?;
            eprintln!(
                "kernel-build: panel recall refinement lane_depth={} support_members={} candidate_hits={} ratio={:.6}",
                lane_depth,
                support.members.len(),
                support.candidate_hits,
                refined.ratio
            );
            if refined.ratio >= args.min_recall || lane_depth == embeddings.len() {
                break (support, refined, lane_depth);
            }
            lane_depth = lane_depth.saturating_mul(2).min(embeddings.len());
        };
        refinement = Some(KernelBuildRefinement {
            initial_ratio: initial_recall.ratio,
            initial_kernel_only: initial_recall.kernel_only,
            initial_members,
            initial_kernel_graph,
            support_members: support.members.len(),
            support_candidate_hits: support.candidate_hits,
            support_queries: support.n_queries_tested,
            final_members: kernel.members.len(),
            final_kernel_graph: kernel.kernel_graph.len(),
        });
        eprintln!(
            "kernel-build: recall refinement support_members={} candidate_hits={} queries={} lane_depth={} ratio={:.6} members {}->{} kernel_graph {}->{}",
            support.members.len(),
            support.candidate_hits,
            support.n_queries_tested,
            final_lane_depth,
            refined_recall.ratio,
            initial_members,
            kernel.members.len(),
            initial_kernel_graph,
            kernel.kernel_graph.len()
        );
    }
    let mut recall =
        panel_kernel_recall_gate(&kernel_index, &embeddings, &recall_params, &corpus_name)?;
    recall.approx_factor = kernel.recall.approx_factor;
    recall.tau_star_estimate = kernel.recall.tau_star_estimate;
    recall.tau_star_exact = kernel.recall.tau_star_exact;
    kernel.recall = recall.clone();
    seal_completed_kernel_identity(&mut kernel, &physical_contract_hash)?;
    kernel_index = build_panel_kernel_index(&kernel, &embedding_slots, &embeddings)?;
    eprintln!(
        "kernel-build: recall ratio={:.6} kernel_only={:.6} full={:.6} queries={} elapsed_ms={}",
        recall.ratio,
        recall.kernel_only,
        recall.full,
        recall.n_queries_tested,
        stage.elapsed().as_millis()
    );

    let store = FsKernelStore::new(&resolved.path);
    let stage = Instant::now();
    write_panel_kernel_index(&kernel_index, &store)?;
    write_kernel_artifact(&kernel, &store)?;

    let persisted_kernel = read_kernel_artifact(kernel.kernel_id, &store)?;
    if persisted_kernel != kernel {
        return Err(LodestarError::KernelArtifactCodec {
            detail: format!(
                "readback mismatch for kernel {}; persisted kernel.json did not match built kernel",
                kernel.kernel_id
            ),
        }
        .into());
    }
    let persisted_index = load_panel_kernel_index(kernel.kernel_id, &store)?;
    if persisted_index.rows().len() != kernel.members.len() {
        return Err(LodestarError::KernelIndexCodec {
            detail: format!(
                "readback mismatch for kernel {}; index rows {} did not match members {}",
                kernel.kernel_id,
                persisted_index.rows().len(),
                kernel.members.len()
            ),
        }
        .into());
    }
    let health = kernel_health(kernel.kernel_id, &store)?;
    let vault = AsterVault::open(
        &resolved.path,
        resolved.vault_id,
        vault_salt(resolved.vault_id, &resolved.name),
        VaultOptions::default(),
    )?;
    let admission_ledger = kernel_admission_ledger_payload(&admission);
    let build_payload = serde_json::to_vec_pretty(&json!({
        "type": "kernel_build_v3",
        "kernel_id": kernel.kernel_id.to_string(),
        "members_hash": hex32(&kernel_members_hash(&kernel)),
        "graph_source_seq": graph_source_seq,
        "physical_graph_contract_sha256": hex32(&physical_contract_hash),
        "embedding_slots": embedding_slots.iter().map(|slot| slot.get()).collect::<Vec<_>>(),
        "fusion": "rrf",
        "rrf_k": PANEL_RRF_K,
        "kernel_json_sha256": super::kernel_generation::sha256_file(&store.kernel_file_path(kernel.kernel_id))?,
        "index_json_sha256": super::kernel_generation::sha256_file(&store.index_file_path(kernel.kernel_id))?,
        "recall_ratio": recall.ratio,
        "groundedness_fraction": kernel.groundedness.reached_anchor,
        "admission": admission_ledger,
        "jurisdiction": jurisdiction,
    }))
    .map_err(|error| CliError::runtime(format!("encode kernel build ledger payload: {error}")))?;
    let build_ledger_ref = vault.append_ledger_entry(
        EntryKind::Kernel,
        SubjectId::Kernel(kernel.kernel_id.as_bytes().to_vec()),
        build_payload.clone(),
        ActorId::Service("calyx-kernel-build".to_string()),
    )?;
    let physical_ledger = AsterLedgerCfStore::open(&resolved.path)?;
    let ledger_row = physical_ledger
        .read_seq(build_ledger_ref.seq)?
        .ok_or_else(|| CliError::runtime("kernel build ledger row missing after append"))?;
    let ledger_entry = decode(&ledger_row.bytes)?;
    if ledger_entry.entry_hash != build_ledger_ref.hash
        || ledger_entry.kind != EntryKind::Kernel
        || ledger_entry.payload != build_payload
    {
        return Err(CliError::runtime(
            "kernel build ledger physical readback differs from the appended row",
        ));
    }
    let graph_contract = KernelGraphContract {
        collection: PANEL_ASTER_ASSOC_COLLECTION.to_string(),
        nodes: graph.node_count(),
        edges: graph.edge_count(),
        source_seq: graph_source_seq,
        embedding_slots: embedding_slots.clone(),
        fusion: "rrf".to_string(),
        rrf_k: PANEL_RRF_K,
        panel_version,
        knn,
        edge_score_threshold,
        metadata_sha256: sha256_bytes(&metadata_bytes),
        node_props_sha256,
        csr_sha256: sha256_bytes(&csr_bytes),
        physical_contract_sha256: hex32(&physical_contract_hash),
    };
    let published = publish_current_generation(
        &resolved.path,
        KernelGenerationManifest {
            schema_version: 3,
            kernel_id: kernel.kernel_id,
            kernel_json: artifact_ref(&resolved.path, &store.kernel_file_path(kernel.kernel_id))?,
            index_json: artifact_ref(&resolved.path, &store.index_file_path(kernel.kernel_id))?,
            graph: graph_contract,
            admission: admission.clone(),
            jurisdiction: jurisdiction.clone(),
            build_ledger_seq: build_ledger_ref.seq,
            build_ledger_hash: ledger_hash(&build_ledger_ref),
        },
    )?;
    eprintln!(
        "kernel-build: persisted artifacts kernel_json={} index_json={} elapsed_ms={} total_elapsed_ms={}",
        store.kernel_file_path(kernel.kernel_id).display(),
        store.index_file_path(kernel.kernel_id).display(),
        stage.elapsed().as_millis(),
        started.elapsed().as_millis()
    );

    let groundedness_fraction = kernel.groundedness.reached_anchor;
    let kernel_file = store.kernel_file_path(kernel.kernel_id);
    let index_file = store.index_file_path(kernel.kernel_id);
    let output = json!({
        "status": "ok",
        "vault": resolved.name,
        "graph": { "nodes": graph.node_count(), "edges": graph.edge_count() },
        "kernel": {
            "kernel_id": kernel.kernel_id.to_string(),
            "members": kernel.members.len(),
            "kernel_graph": kernel.kernel_graph.len(),
            "groundedness_fraction": groundedness_fraction,
        },
        "recall": {
            "kernel_only": recall.kernel_only,
            "full": recall.full,
            "ratio": recall.ratio,
            "tau_star_estimate": recall.tau_star_estimate,
            "tau_star_exact": recall.tau_star_exact,
            "n_queries_tested": recall.n_queries_tested,
            "held_out_fraction": args.held_out_fraction,
            "min_recall_ratio": args.min_recall,
            "gate_passed": true,
        },
        "admission": admission,
        "jurisdiction": jurisdiction,
        "refinement": refinement.as_ref().map(|refinement| json!({
            "initial_ratio": refinement.initial_ratio,
            "initial_kernel_only": refinement.initial_kernel_only,
            "initial_members": refinement.initial_members,
            "initial_kernel_graph": refinement.initial_kernel_graph,
            "support_members": refinement.support_members,
            "support_candidate_hits": refinement.support_candidate_hits,
            "support_queries": refinement.support_queries,
            "final_members": refinement.final_members,
            "final_kernel_graph": refinement.final_kernel_graph,
        })),
        "artifacts": {
            "store_root": resolved.path,
            "kernel_json": kernel_file,
            "kernel_json_bytes": std::fs::metadata(&kernel_file)?.len(),
            "index_json": index_file,
            "index_json_bytes": std::fs::metadata(&index_file)?.len(),
            "readback": {
                "kernel_id": persisted_kernel.kernel_id.to_string(),
                "kernel_members": persisted_kernel.members.len(),
                "index_rows": persisted_index.rows().len(),
                "health_recall_pass_mode": format!("{:?}", health.recall.pass_mode).to_ascii_lowercase(),
                "health_grounded_fraction": health.grounded_fraction,
            },
            "current_manifest": published.manifest_path,
            "current_manifest_sha256": published.manifest_sha256,
            "build_ledger_seq": build_ledger_ref.seq,
            "build_ledger_hash": ledger_hash(&build_ledger_ref),
        },
    });
    print_json(&output)
}

fn kernel_admission_ledger_payload(admission: &KernelAdmissionContract) -> serde_json::Value {
    json!({
        "schema_version": admission.schema_version,
        "method": admission.method,
        "corpus_count": admission.corpus_count,
        "sample_count": admission.sample_count,
        "sample_limit": admission.sample_limit,
        "sample_seed": admission.sample_seed,
        "lower_tail_quantile": admission.lower_tail_quantile,
        "threshold": admission.threshold,
        "min_score": admission.min_score,
        "median_score": admission.median_score,
        "max_score": admission.max_score,
        "sample_ids_sha256": admission.sample_ids_sha256,
        "observations_sha256": admission.observations_sha256,
        "calibration_queries_bytes": admission.calibration_queries.as_ref().map(|artifact| artifact.bytes),
        "calibration_queries_sha256": admission.calibration_queries.as_ref().map(|artifact| &artifact.sha256),
    })
}

#[cfg(test)]
mod tests;
