use super::*;
use calyx_assay::estimate::{EstimateReliability, EstimatorKind, MiEstimate, TrustTag};
use calyx_assay::store::{AssayCacheKey, AssayStore, AssaySubject};
use calyx_core::{
    AnchorKind, Asymmetry, Lens, Modality, QuantPolicy, Slot, SlotId, SlotKey, SlotShape,
    SlotState, VaultId,
};

use crate::frozen::{FrozenLensContract, LensDType, NormPolicy, sha256_digest};
use crate::runtime::algorithmic::AlgorithmicLens;

#[test]
fn profiles_algorithmic_lens_with_real_numbers() {
    let mut registry = Registry::new();
    let lens = AlgorithmicLens::byte_features("profile-test", Modality::Text);
    let id = registry
        .register_frozen(lens.clone(), lens.contract().clone())
        .unwrap();
    let probes = profile_probes();

    let card = profile_lens(&registry, id, &probes).unwrap();

    println!("{}", serde_json::to_string_pretty(&card).unwrap());
    assert_eq!(card.coverage.requested, probes.len());
    assert_eq!(card.coverage.failed, 0);
    assert!(card.spread.participation_ratio > 0.0);
    assert!(card.spread.normalized_participation_ratio > 0.0);
    assert_eq!(card.signal, None);
    assert_eq!(card.signal_source, MetricSource::AssayPending);
    assert!(card.proxy_signal.is_finite());
    assert_eq!(card.differentiation, None);
    assert_eq!(card.differentiation_source, MetricSource::AssayPending);
    assert!(card.proxy_differentiation.is_finite());
    assert!(card.cost.ms_per_input >= 0.0);
    assert_eq!(card.execution.measurement_passes, 1);
    assert_eq!(card.execution.batch_measure_calls, 1);
    assert_eq!(card.execution.scalar_measure_calls, 0);
}

#[test]
fn offline_dense_vectors_report_zero_runtime_execution_calls() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let labels = vec![Some("a".to_string()), Some("b".to_string())];
    let card = profile_dense_vectors(DenseProfileRequest {
        lens_id: LensId::from_bytes([0x57; 16]),
        probe_count: vectors.len(),
        vectors: &vectors,
        labels: &labels,
        cost: CostMetrics {
            total_ms: 0.0,
            ms_per_input: 0.0,
            vram_bytes: 0,
            vram_observed: false,
            ram_bytes: 16,
            batch_ceiling: u32::MAX,
        },
        signal: None,
        signal_kind: CapabilitySignalKind::DeterministicContentFeature,
        health: LensHealth::Loaded,
        execution: ProfileExecutionStats::default(),
    })
    .unwrap();

    assert_eq!(card.execution.measurement_passes, 0);
    assert_eq!(card.execution.batch_measure_calls, 0);
    assert_eq!(card.execution.scalar_measure_calls, 0);
    assert_eq!(card.execution.pairwise_distance_matrices, 1);
    assert_eq!(card.execution.pairwise_distance_values, 4);
    assert_eq!(card.execution.measured_rows, 2);
    assert_eq!(card.execution.vector_dim, 2);
}

#[test]
fn assay_owned_metrics_serialize_as_null_until_attached() {
    let mut registry = Registry::new();
    let lens = AlgorithmicLens::byte_features("profile-null-assay-fields", Modality::Text);
    let id = registry
        .register_frozen(lens.clone(), lens.contract().clone())
        .unwrap();

    let card = profile_lens(&registry, id, &profile_probes()).unwrap();
    let json = serde_json::to_value(&card).unwrap();

    assert!(json["signal"].is_null());
    assert_eq!(json["signal_source"], "assay_pending");
    assert!(json["proxy_signal"].as_f64().unwrap().is_finite());
    assert!(json["differentiation"].is_null());
    assert_eq!(json["differentiation_source"], "assay_pending");
    assert!(json["proxy_differentiation"].as_f64().unwrap().is_finite());
}

#[test]
fn assay_rows_attach_signal_and_pair_gain_metrics() {
    let mut registry = Registry::new();
    let lens = AlgorithmicLens::byte_features("profile-assay-fields", Modality::Text);
    let id = registry
        .register_frozen(lens.clone(), lens.contract().clone())
        .unwrap();
    let slot = slot_for_lens(id, 0);
    let cache_key = assay_key();
    let mut store = AssayStore::default();
    store.put(
        cache_key.clone(),
        AssaySubject::Lens { slot: slot.slot_id },
        reliable_estimate(0.42, EstimatorKind::Ksg),
        "unit lens signal",
        10,
    );
    store.put(
        cache_key.clone(),
        AssaySubject::Pair {
            a: slot.slot_id,
            b: SlotId::new(9),
        },
        estimate(0.07, EstimatorKind::PairGain),
        "unit pair gain",
        11,
    );

    let card = profile_slot_with_assay(&registry, &slot, &profile_probes(), &store, &cache_key)
        .expect("profile with assay");
    let json = serde_json::to_value(&card).unwrap();

    assert_eq!(card.signal, Some(0.42));
    assert_eq!(card.signal_source, MetricSource::AssayStore);
    let reliability = card.signal_reliability.expect("signal reliability");
    assert_eq!(reliability.seed_count, 5);
    assert!((reliability.seed_sigma - 0.01).abs() <= 1e-6);
    assert!(!reliability.unresolved);
    assert_eq!(card.differentiation, Some(0.07));
    assert_eq!(card.differentiation_source, MetricSource::AssayStore);
    assert_eq!(json["signal_source"], "assay_store");
    assert_eq!(json["differentiation_source"], "assay_store");
}

#[test]
fn collapsed_lens_is_flagged_low_spread() {
    let mut registry = Registry::new();
    let lens = CollapsedLens::new();
    let id = registry
        .register_frozen(lens.clone(), lens.contract.clone())
        .unwrap();

    let card = profile_lens(&registry, id, &profile_probes()).unwrap();

    assert!(card.low_spread);
    assert_eq!(card.spread.participation_ratio, 0.0);
    assert_eq!(card.spread.mean_pairwise_distance, 0.0);
}

#[test]
fn wrong_modality_counts_as_failed_coverage() {
    let mut registry = Registry::new();
    let lens = AlgorithmicLens::byte_features("profile-coverage", Modality::Text);
    let id = registry
        .register_frozen(lens.clone(), lens.contract().clone())
        .unwrap();
    let probes = vec![
        ProfileProbe::new(Input::new(Modality::Text, b"ok".to_vec())),
        ProfileProbe::new(Input::new(Modality::Image, vec![1, 2, 3])),
    ];

    let card = profile_lens(&registry, id, &probes).unwrap();

    assert_eq!(card.coverage.measured, 1);
    assert_eq!(card.coverage.failed, 1);
    assert_eq!(card.coverage.rate, 0.5);
}

#[test]
fn empty_probe_set_fails_closed() {
    let registry = Registry::new();
    let error = profile_lens(&registry, LensId::from_bytes([7; 16]), &[]).unwrap_err();

    assert_eq!(error.code, "CALYX_ASSAY_INSUFFICIENT_SAMPLES");
}

fn profile_probes() -> Vec<ProfileProbe> {
    vec![
        ProfileProbe::labeled(Input::new(Modality::Text, b"alpha words".to_vec()), "words"),
        ProfileProbe::labeled(Input::new(Modality::Text, b"beta phrase".to_vec()), "words"),
        ProfileProbe::labeled(
            Input::new(Modality::Text, b"12345 67890".to_vec()),
            "digits",
        ),
        ProfileProbe::labeled(
            Input::new(Modality::Text, b"98765 43210".to_vec()),
            "digits",
        ),
    ]
}

fn slot_for_lens(lens_id: LensId, slot_id: u16) -> Slot {
    let slot_id = SlotId::new(slot_id);
    Slot {
        slot_id,
        slot_key: SlotKey::new(slot_id, format!("slot-{slot_id}")),
        lens_id,
        shape: SlotShape::Dense(4),
        modality: Modality::Text,
        asymmetry: Asymmetry::None,
        quant: QuantPolicy::None,
        resource: Default::default(),
        axis: None,
        retrieval_only: false,
        excluded_from_dedup: false,
        bits_about: Default::default(),
        state: SlotState::Active,
        added_at_panel_version: 1,
    }
}

fn assay_key() -> AssayCacheKey {
    AssayCacheKey::scoped(1, "profile-unit", vault_id(), AnchorKind::Reward)
}

fn estimate(bits: f32, estimator: EstimatorKind) -> MiEstimate {
    MiEstimate::point(bits, 64, estimator, TrustTag::Trusted)
}

fn reliable_estimate(bits: f32, estimator: EstimatorKind) -> MiEstimate {
    MiEstimate::new(
        bits,
        bits - 0.02,
        bits + 0.02,
        64,
        estimator,
        TrustTag::Trusted,
    )
    .with_reliability(EstimateReliability::new(5, 0.01, false).unwrap())
}

fn vault_id() -> VaultId {
    "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap()
}

#[derive(Clone)]
struct CollapsedLens {
    contract: FrozenLensContract,
}

impl CollapsedLens {
    fn new() -> Self {
        Self {
            contract: collapsed_contract("collapsed-profile-test"),
        }
    }
}

impl Lens for CollapsedLens {
    fn id(&self) -> LensId {
        self.contract.lens_id()
    }

    fn shape(&self) -> SlotShape {
        SlotShape::Dense(4)
    }

    fn modality(&self) -> Modality {
        Modality::Text
    }

    fn measure(&self, _input: &Input) -> Result<SlotVector> {
        Ok(SlotVector::Dense {
            dim: 4,
            data: vec![1.0, 0.0, 0.0, 0.0],
        })
    }
}

fn collapsed_contract(name: &str) -> FrozenLensContract {
    FrozenLensContract::new(
        name,
        sha256_digest(&[name.as_bytes(), b"weights"]),
        sha256_digest(&[name.as_bytes(), b"corpus"]),
        SlotShape::Dense(4),
        Modality::Text,
        LensDType::F32,
        NormPolicy::None,
    )
}
