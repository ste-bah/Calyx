use std::collections::BTreeMap;

use calyx_assay::estimate::{EstimatorKind, MiEstimate, TrustTag};
use calyx_assay::store::{AssayCacheKey, AssayStore, AssaySubject};
use calyx_core::{
    AnchorKind, Asymmetry, CalyxError, ConfidenceInterval, Input, Lens, LensId, Modality, Panel,
    QuantPolicy, Result, Signal, SlotShape, SlotVector, VaultId,
};

use super::*;
use crate::runtime::algorithmic::AlgorithmicLens;
use crate::spec::LensRuntime;
use crate::{
    CapabilityCard, CapabilityGateThresholds, CapabilitySignalKind, CostMetrics, CoverageMetrics,
    FrozenLensContract, LensSpec, MetricSource, NormPolicy, SeparationMetrics, SpreadMetrics,
};

#[test]
fn list_panel_uses_stored_slot_bits() {
    let (registry, lens_id) = registry_with_lens();
    let panel = panel_with_slot(lens_id, Some(0.31));
    let listing = list_panel(&panel, &registry);

    assert_eq!(listing[0].bits_about, Some(0.31));
}

#[test]
fn list_panel_with_assay_overlays_scoped_assay_bits() {
    let (registry, lens_id) = registry_with_lens();
    let panel = panel_with_slot(lens_id, Some(0.31));
    let cache_key = assay_key();
    let mut store = AssayStore::default();
    store.put(
        cache_key.clone(),
        AssaySubject::Lens {
            slot: panel.slots[0].slot_id,
        },
        MiEstimate::point(0.47, 72, EstimatorKind::Ksg, TrustTag::Trusted),
        "panel assay bits",
        12,
    );

    let listing = list_panel_with_assay(&panel, &registry, &store, &cache_key);

    assert_eq!(listing[0].bits_about, Some(0.47));
}

#[test]
fn apply_capability_gate_uses_existing_lifecycle_states() {
    let (registry, lens_id) = registry_with_lens();
    let panel = panel_with_slot(lens_id, None);
    let slot_id = panel.slots[0].slot_id;
    let mut controller = SwapController::new(panel);

    let parked = apply_capability_gate(
        &mut controller,
        slot_id,
        &evaluation(lens_id, CapabilityGateDecision::Park),
        20,
    )
    .expect("park from gate");
    let retired = apply_capability_gate(
        &mut controller,
        slot_id,
        &evaluation(lens_id, CapabilityGateDecision::Retire),
        21,
    )
    .expect("retire from gate");

    assert_eq!(registry.health(lens_id).unwrap(), LensHealth::Loaded);
    assert_eq!(parked.state, SlotState::Parked);
    assert_eq!(retired.state, SlotState::Retired);
    assert_eq!(controller.panel().slots[0].state, SlotState::Retired);
}

#[test]
fn list_panel_runtime_probe_marks_loaded_text_lens_failing_when_measure_fails() {
    let mut registry = Registry::new();
    let lens = FailingMeasureLens::new("loaded-but-unmeasurable");
    let spec = loaded_onnx_spec_for(&lens);
    let lens_id = registry
        .register_frozen_with_spec(lens.clone(), lens.contract.clone(), spec)
        .unwrap();
    let panel = panel_with_slot(lens_id, None);

    assert_eq!(registry.health(lens_id).unwrap(), LensHealth::Loaded);

    let listing = list_panel_with_runtime_probe(&panel, &registry);

    assert_eq!(
        listing[0].health,
        LensHealth::Failing {
            code: "CALYX_LENS_UNREACHABLE".to_string(),
            reason: "runtime probe measurement failed".to_string(),
        }
    );
}

fn registry_with_lens() -> (Registry, LensId) {
    let mut registry = Registry::new();
    let lens = AlgorithmicLens::byte_features("panel-assay-list", Modality::Text);
    let lens_id = registry
        .register_frozen(lens.clone(), lens.contract().clone())
        .unwrap();
    (registry, lens_id)
}

#[derive(Clone)]
struct FailingMeasureLens {
    contract: FrozenLensContract,
}

impl FailingMeasureLens {
    fn new(name: &str) -> Self {
        Self {
            contract: test_contract(name),
        }
    }
}

impl Lens for FailingMeasureLens {
    fn id(&self) -> LensId {
        self.contract.lens_id()
    }

    fn shape(&self) -> SlotShape {
        SlotShape::Dense(1)
    }

    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn measure(&self, _input: &Input) -> Result<SlotVector> {
        Err(CalyxError::lens_unreachable(
            "runtime probe measurement failed",
        ))
    }
}

fn loaded_onnx_spec_for(lens: &FailingMeasureLens) -> LensSpec {
    let root =
        std::env::temp_dir().join(format!("calyx-panel-runtime-probe-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join(format!("{}.onnx", lens.contract.name()));
    let tokenizer = root.join(format!("{}-tokenizer.json", lens.contract.name()));
    let config = root.join(format!("{}-config.json", lens.contract.name()));
    std::fs::write(&model, b"model").unwrap();
    std::fs::write(&tokenizer, b"tokenizer").unwrap();
    std::fs::write(&config, b"config").unwrap();
    LensSpec {
        name: lens.contract.name().to_string(),
        runtime: LensRuntime::Onnx {
            model_id: "unit/loaded-but-unmeasurable".to_string(),
            files: vec![model, tokenizer, config],
        },
        output: SlotShape::Dense(1),
        modality: Modality::Text,
        weights_sha256: lens.contract.weights_sha256(),
        corpus_hash: lens.contract.corpus_hash(),
        norm_policy: NormPolicy::None,
        max_batch: None,
        axis: None,
        asymmetry: Asymmetry::None,
        quant_default: QuantPolicy::turboquant_default(),
        truncate_dim: None,
        recall_delta: crate::spec::default_recall_delta(),
        retrieval_only: false,
        excluded_from_dedup: false,
    }
}

fn test_contract(name: &str) -> FrozenLensContract {
    FrozenLensContract::new(
        name,
        test_hash(name, b"weights"),
        test_hash(name, b"corpus"),
        SlotShape::Dense(1),
        Modality::Text,
        crate::LensDType::F32,
        NormPolicy::None,
    )
}

fn test_hash(name: &str, suffix: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update(suffix);
    hasher.finalize().into()
}

fn panel_with_slot(lens_id: LensId, bits: Option<f32>) -> Panel {
    let slot_id = SlotId::new(0);
    let mut bits_about = BTreeMap::new();
    if let Some(bits) = bits {
        bits_about.insert(
            AnchorKind::Reward,
            Signal {
                bits,
                ci: ConfidenceInterval {
                    low: bits - 0.01,
                    high: bits + 0.01,
                },
                n: 64,
                estimator: "unit".to_string(),
                ts: 1,
            },
        );
    }
    Panel {
        version: 1,
        slots: vec![Slot {
            slot_id,
            slot_key: SlotKey::new(slot_id, "panel-assay".to_string()),
            lens_id,
            shape: SlotShape::Dense(4),
            modality: Modality::Text,
            asymmetry: Asymmetry::None,
            quant: QuantPolicy::None,
            resource: Default::default(),
            axis: None,
            retrieval_only: false,
            excluded_from_dedup: false,
            bits_about,
            state: SlotState::Active,
            added_at_panel_version: 1,
        }],
        created_at: 1,
        kernel_ref: None,
        guard_ref: None,
    }
}

fn assay_key() -> AssayCacheKey {
    AssayCacheKey::scoped(1, "panel-unit", vault_id(), AnchorKind::Reward)
}

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
}

fn evaluation(lens_id: LensId, decision: CapabilityGateDecision) -> CapabilityGateEvaluation {
    CapabilityGateEvaluation {
        lens_id,
        decision,
        signal_bits: 0.08,
        signal_grounded: true,
        max_pairwise_corr: 0.1,
        thresholds: CapabilityGateThresholds::default(),
        reason: "unit gate".to_string(),
        card: CapabilityCard {
            lens_id,
            probe_count: 4,
            signal: Some(0.08),
            signal_source: MetricSource::AssayStore,
            signal_kind: CapabilitySignalKind::LearnedEncoder,
            signal_reliability: None,
            proxy_signal: 0.08,
            differentiation: Some(0.07),
            differentiation_source: MetricSource::AssayStore,
            proxy_differentiation: 0.7,
            spread: SpreadMetrics {
                participation_ratio: 2.0,
                normalized_participation_ratio: 0.5,
                stable_rank: 2.0,
                total_variance: 1.0,
                mean_pairwise_distance: 1.0,
            },
            separation: SeparationMetrics {
                score: 0.5,
                silhouette: 0.5,
                mean_pairwise_distance: 1.0,
                labeled_groups: 2,
                used_labels: true,
            },
            cost: CostMetrics {
                total_ms: 1.0,
                ms_per_input: 1.0,
                vram_bytes: 0,
                vram_observed: true,
                ram_bytes: 0,
                batch_ceiling: 1_000,
            },
            coverage: CoverageMetrics {
                requested: 4,
                measured: 4,
                failed: 0,
                rate: 1.0,
            },
            health: LensHealth::Loaded,
            low_spread: false,
        },
    }
}
