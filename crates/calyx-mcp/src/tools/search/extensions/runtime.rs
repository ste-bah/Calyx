use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use calyx_aster::vault::AsterVault;
use calyx_core::{CalyxError, Constellation, CxId, SlotId, SlotShape, SlotVector, VaultStore};
use calyx_registry::{VaultPanelState, load_vault_panel_state};
use calyx_search::{
    PersistedSearchGeneration, PersistedSearchIndexes, PersistedSearchSlot, SearchError,
};
use calyx_sextant::{
    IndexSearchHit, IndexStats, SearchEngine, SextantIndex, SkillParams, SkillTree, SlotIndexMap,
};

use crate::server::{ToolError, ToolResult};

use super::super::engine;

pub(super) struct NavRuntime {
    pub(super) path: std::path::PathBuf,
    pub(super) vault: AsterVault,
    pub(super) state: VaultPanelState,
    pub(super) docs: Arc<BTreeMap<CxId, Constellation>>,
    pub(super) engine: SearchEngine,
}

pub(super) fn load_runtime(vault: &str) -> ToolResult<NavRuntime> {
    let resolved = engine::resolve_requested_vault(vault)?;
    let path = resolved.path.clone();
    let vault = engine::open_vault(&resolved)?;
    let state = load_vault_panel_state(&resolved.path)?;
    let docs = Arc::new(map_search(calyx_search::load_docs(&vault))?);
    let engine = match PersistedSearchIndexes::open(&resolved.path) {
        Ok(indexes) => {
            let indexes = Arc::new(indexes);
            validate_generation(&vault, &indexes)?;
            let generation = map_search(indexes.generation())?;
            build_search_engine(Arc::clone(&docs), indexes, &generation)?
        }
        Err(error) if docs.is_empty() && is_missing_manifest(&error) => {
            build_empty_search_engine(&docs)?
        }
        Err(error) => return Err(map_search_error(error)),
    };
    Ok(NavRuntime {
        path,
        vault,
        state,
        docs,
        engine,
    })
}

pub(super) fn skill_tree(search: &SearchEngine) -> ToolResult<SkillTree> {
    Ok(calyx_sextant::skills(
        search,
        &SkillParams {
            min_cluster_size: 2,
            min_samples: 1,
            max_constellations: 2048,
            slots: None,
            allow_single_cluster: true,
        },
    )?)
}

pub(super) fn query_vector_for_skill(
    runtime: &NavRuntime,
    text: &str,
) -> ToolResult<Option<(SlotId, SlotVector)>> {
    let slots: BTreeSet<_> = runtime.engine.indexes.slots().into_iter().collect();
    Ok(
        map_search(calyx_search::measure_query_vectors(&runtime.state, text))?
            .into_iter()
            .find(|(slot, vector)| {
                slots.contains(slot) && !matches!(vector, SlotVector::Sparse { .. })
            }),
    )
}

pub(super) fn parse_cx_id(value: &str) -> ToolResult<CxId> {
    value
        .parse::<CxId>()
        .map_err(|err| ToolError::invalid_params(format!("parse cx_id {value}: {err}")))
}

pub(super) fn ensure_doc_exists(
    docs: &BTreeMap<CxId, Constellation>,
    cx_id: CxId,
) -> ToolResult<()> {
    if docs.contains_key(&cx_id) {
        Ok(())
    } else {
        Err(
            CalyxError::vault_access_denied(format!("cx_id {cx_id} does not exist in vault"))
                .into(),
        )
    }
}

pub(super) fn score01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn validate_generation(vault: &AsterVault, indexes: &PersistedSearchIndexes) -> ToolResult<()> {
    let snapshot = vault.snapshot();
    map_search(
        indexes.ensure_fresh_at_snapshot(snapshot, vault.derived_content_seq().min(snapshot)),
    )?;
    map_search(indexes.ensure_search_bounded())
}

fn build_search_engine(
    docs: Arc<BTreeMap<CxId, Constellation>>,
    persisted: Arc<PersistedSearchIndexes>,
    generation: &PersistedSearchGeneration,
) -> ToolResult<SearchEngine> {
    let indexes = SlotIndexMap::new();
    for slot in &generation.slots {
        indexes.register(PersistedSlotIndex {
            persisted: Arc::clone(&persisted),
            docs: Arc::clone(&docs),
            descriptor: slot.clone(),
            base_seq: generation.base_seq,
        })?;
    }
    let mut search = SearchEngine::new(indexes);
    for cx in docs.values() {
        search.put_constellation(cx.clone());
    }
    search.set_assoc_graph(association_graph(&docs)?);
    Ok(search)
}

fn build_empty_search_engine(docs: &BTreeMap<CxId, Constellation>) -> ToolResult<SearchEngine> {
    let mut search = SearchEngine::new(SlotIndexMap::new());
    search.set_assoc_graph(association_graph(docs)?);
    Ok(search)
}

struct PersistedSlotIndex {
    persisted: Arc<PersistedSearchIndexes>,
    docs: Arc<BTreeMap<CxId, Constellation>>,
    descriptor: PersistedSearchSlot,
    base_seq: u64,
}

impl SextantIndex for PersistedSlotIndex {
    fn slot(&self) -> SlotId {
        self.descriptor.slot
    }

    fn shape(&self) -> SlotShape {
        self.descriptor.shape
    }

    fn insert(&mut self, _cx_id: CxId, _vector: SlotVector, _seq: u64) -> calyx_core::Result<()> {
        Err(read_only_error(self.descriptor.slot))
    }

    fn search(
        &self,
        query: &SlotVector,
        k: usize,
        _ef: Option<usize>,
    ) -> calyx_core::Result<Vec<IndexSearchHit>> {
        self.persisted
            .search(self.descriptor.slot, query, k)
            .map_err(CalyxError::from)
    }

    fn rebuild(&mut self) -> calyx_core::Result<()> {
        Err(read_only_error(self.descriptor.slot))
    }

    fn vector(&self, cx_id: CxId) -> Option<SlotVector> {
        self.docs
            .get(&cx_id)
            .and_then(|cx| cx.slots.get(&self.descriptor.slot))
            .cloned()
    }

    fn set_base_seq(&mut self, _seq: u64) {}

    fn stats(&self) -> IndexStats {
        IndexStats {
            slot: self.descriptor.slot,
            shape: self.descriptor.shape,
            len: self.descriptor.len,
            built_at_seq: self.descriptor.built_at_seq,
            base_seq: self.base_seq,
            kind: "persisted",
        }
    }
}

fn read_only_error(slot: SlotId) -> CalyxError {
    CalyxError::stale_derived(format!(
        "MCP navigation slot {slot} is a read-only persisted search generation"
    ))
}

fn association_graph(docs: &BTreeMap<CxId, Constellation>) -> ToolResult<calyx_paths::AssocGraph> {
    let mut builder = calyx_paths::AssocGraph::builder();
    let mut ordered = docs
        .values()
        .map(|cx| (cx.created_at, cx.cx_id))
        .collect::<Vec<_>>();
    ordered.sort();
    for (_, cx_id) in &ordered {
        builder
            .add_node(*cx_id, 1.0)
            .map_err(|err| CalyxError::stale_derived(format!("build association node: {err}")))?;
    }
    for pair in ordered.windows(2) {
        let left = docs.get(&pair[0].1).expect("ordered id in docs");
        let right = docs.get(&pair[1].1).expect("ordered id in docs");
        builder
            .add_edge(left.cx_id, right.cx_id, association_weight(left, right))
            .map_err(|err| CalyxError::stale_derived(format!("build association edge: {err}")))?;
    }
    Ok(builder.build())
}

fn association_weight(left: &Constellation, right: &Constellation) -> f32 {
    let mut sum = 0.0_f32;
    let mut n = 0_usize;
    for (slot, vector) in &left.slots {
        if let (Some(a), Some(b)) = (
            vector.as_dense(),
            right.slots.get(slot).and_then(SlotVector::as_dense),
        ) && let Some(cos) = calyx_core::dense_cosine(a, b)
        {
            sum += cos.clamp(0.0, 1.0);
            n += 1;
        }
    }
    if n == 0 {
        1.0
    } else {
        (sum / n as f32).clamp(0.05, 1.0)
    }
}

fn map_search<T>(result: Result<T, SearchError>) -> ToolResult<T> {
    result.map_err(map_search_error)
}

fn map_search_error(error: SearchError) -> ToolError {
    match error {
        SearchError::Calyx(error) => error.into(),
        SearchError::Usage(message) => ToolError::invalid_params(message),
        SearchError::Io(message) => {
            CalyxError::stale_derived(format!("persisted navigation I/O failure: {message}")).into()
        }
    }
}

fn is_missing_manifest(error: &SearchError) -> bool {
    matches!(
        error,
        SearchError::Calyx(error)
            if error.code == "CALYX_STALE_DERIVED"
                && error.message.contains("manifest missing")
    )
}
