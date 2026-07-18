use super::*;
use crate::lens_commands::support::hex_from_bytes;
use sha2::{Digest, Sha256};

#[test]
fn rejects_sub_ten_content_templates() {
    let template = SavedPanelTemplate {
        schema_version: OBJECT_VERSION,
        name: "too-small".to_string(),
        version: 1,
        notes: String::new(),
        min_content_lenses: MIN_CONTENT_LENSES,
        lenses: Vec::new(),
        time_controls: default_time_controls(),
        ensemble_card: None,
    };

    let error = template.validate().unwrap_err();
    assert_eq!(error.code(), TEMPLATE_INVALID);
}

#[test]
fn temporal_controls_never_count_toward_a35() {
    for control in default_time_controls() {
        assert!(!control.counts_toward_a35);
    }
}

#[test]
fn templates_without_a37_card_are_not_gate_eligible() {
    let template = SavedPanelTemplate {
        schema_version: OBJECT_VERSION,
        name: "diagnostic".to_string(),
        version: 1,
        notes: String::new(),
        min_content_lenses: MIN_CONTENT_LENSES,
        lenses: Vec::new(),
        time_controls: default_time_controls(),
        ensemble_card: None,
    };

    assert!(!template.a37_gate_eligible());
    assert_eq!(
        template.require_a37_gate().unwrap_err().code(),
        TEMPLATE_A37_GATE_REFUSED
    );
}

#[test]
fn hash_from_bytes_stays_lower_hex() {
    assert_eq!(
        hex_from_bytes(&[0xabu8; 32]),
        "abababababababababababababababababababababababababababababababab"
    );
}

#[test]
fn catalog_lens_ref_hashes_real_artifacts_and_ignores_alias_mutation() {
    let root = std::env::temp_dir().join(format!("calyx-template-snapshot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let entry = real_tei_entry(&root, None);

    let reference = lens_ref_from_catalog(&entry).unwrap();
    let snapshot = reference.immutable_snapshot.as_ref().unwrap();
    std::fs::write(&entry.manifest, b"{\"mutated\":true}").unwrap();
    let verified = reference
        .verified_materialization_spec("fixture-template-id")
        .unwrap();

    assert_eq!(reference.runtime, "tei_http");
    assert_eq!(reference.shape, SlotShape::Dense(8));
    assert_eq!(reference.modality, Modality::Text);
    assert_eq!(reference.lens_name, "fixture-tei");
    assert_eq!(verified, snapshot.spec);
    assert_ne!(
        snapshot.manifest_blake3,
        blake3::hash(b"{\"mutated\":true}").to_hex().to_string()
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn catalog_lens_ref_rejects_stale_catalog_id() {
    let root = std::env::temp_dir().join(format!("calyx-template-stale-id-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let entry = real_tei_entry(&root, Some(LensId::from_bytes([7_u8; 16])));

    let error = lens_ref_from_catalog(&entry).unwrap_err();

    assert_eq!(error.code(), TEMPLATE_INVALID);
    assert!(error.message().contains("manifest"));
    assert!(error.message().contains("resolves to"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn missing_immutable_snapshot_fails_with_template_and_lens_identity() {
    let root = std::env::temp_dir().join(format!(
        "calyx-template-missing-snapshot-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut reference = lens_ref_from_catalog(&real_tei_entry(&root, None)).unwrap();
    reference.immutable_snapshot = None;

    let error = reference
        .verified_materialization_spec("template-missing-snapshot")
        .unwrap_err();

    assert_eq!(error.code(), TEMPLATE_INVALID);
    assert!(error.message().contains("template-missing-snapshot"));
    assert!(error.message().contains("fixture-tei"));
    assert!(error.message().contains("snapshot is missing"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn altered_spec_snapshot_hash_fails_before_artifact_materialization() {
    let root =
        std::env::temp_dir().join(format!("calyx-template-spec-hash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut reference = lens_ref_from_catalog(&real_tei_entry(&root, None)).unwrap();
    reference.immutable_snapshot.as_mut().unwrap().spec_blake3 = "00".repeat(32);

    let error = reference
        .verified_materialization_spec("template-altered-spec")
        .unwrap_err();

    assert_eq!(error.code(), TEMPLATE_INVALID);
    assert!(error.message().contains("spec snapshot hash mismatch"));
    assert!(error.message().contains("expected="));
    assert!(error.message().contains("actual="));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn conflicting_frozen_runtime_contract_fails_with_both_identities() {
    let root = std::env::temp_dir().join(format!(
        "calyx-template-runtime-conflict-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut reference = lens_ref_from_catalog(&real_tei_entry(&root, None)).unwrap();
    let snapshot = reference.immutable_snapshot.as_mut().unwrap();
    snapshot.runtime_contract = FrozenLensContract::tei_http(
        "fixture-tei-conflict",
        "http://127.0.0.1:18081/embed",
        Modality::Text,
        8,
    );
    snapshot.runtime_contract_blake3 = json_blake3(&snapshot.runtime_contract).unwrap();

    let error = reference
        .verified_materialization_spec("template-runtime-conflict")
        .unwrap_err();

    assert_eq!(error.code(), TEMPLATE_INVALID);
    assert!(error.message().contains("derived runtime contract id="));
    assert!(error.message().contains("expected id="));
    assert!(error.message().contains("spec_blake3="));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn artifact_hash_drift_fails_even_when_embedded_snapshots_are_intact() {
    let root = std::env::temp_dir().join(format!(
        "calyx-template-artifact-drift-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let reference = lens_ref_from_catalog(&real_tei_entry(&root, None)).unwrap();
    std::fs::write(root.join("descriptor.json"), b"corrupt").unwrap();

    let error = reference
        .verified_materialization_spec("template-artifact-drift")
        .unwrap_err();

    assert_eq!(error.code(), "CALYX_LENS_FROZEN_VIOLATION");
    assert!(error.message().contains("sha256"));
    assert!(error.message().contains("!= manifest"));
    std::fs::remove_dir_all(root).unwrap();
}

fn real_tei_entry(
    root: &std::path::Path,
    lens_id_override: Option<LensId>,
) -> super::super::LensCatalogEntry {
    let descriptor =
        br#"{"source_hf_id":"fixture/tei","endpoint":"http://127.0.0.1:18080/embed","dim":8}"#;
    let digest = Sha256::digest(descriptor);
    let digest_hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    std::fs::write(root.join("descriptor.json"), descriptor).unwrap();
    let manifest = root.join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::json!({
            "name": "fixture-tei",
            "modality": "text",
            "runtime": "tei",
            "dim": 8,
            "dtype": "f32",
            "weights_sha256": digest_hex,
            "files": [{
                "role": "model",
                "path": "descriptor.json",
                "sha256": digest_hex,
                "bytes": descriptor.len()
            }],
            "pooling": "mean",
            "norm": "unit",
            "source_hf_id": "fixture/tei",
            "endpoint": "http://127.0.0.1:18080/embed",
            "license": "apache-2.0"
        })
        .to_string(),
    )
    .unwrap();
    let spec = calyx_registry::lens_spec_from_manifest_path(&manifest).unwrap();
    super::super::LensCatalogEntry {
        lens_id: lens_id_override
            .unwrap_or_else(|| spec.lens_id())
            .to_string(),
        name: spec.name.clone(),
        modality: "text".to_string(),
        runtime: "tei_http".to_string(),
        dim: 8,
        retrieval_only: false,
        excluded_from_dedup: false,
        weights_sha256: hex_from_bytes(&spec.weights_sha256),
        manifest,
        cost: LensCost::default(),
        placement: Placement::Cpu,
    }
}
