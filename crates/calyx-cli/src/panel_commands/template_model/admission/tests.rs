use std::collections::BTreeSet;
use std::str::FromStr;

use calyx_core::{LensCost, LensId, Modality, Placement, SlotShape};

use super::super::{
    MIN_CONTENT_LENSES, OBJECT_VERSION, SavedPanelTemplate, TemplateLensRef, default_time_controls,
};
use super::*;

fn lens(name: &str, runtime: &str, placement: Placement) -> TemplateLensRef {
    TemplateLensRef {
        slot_key: name.to_string(),
        lens_name: name.to_string(),
        lens_id: LensId::from_str("11111111111111111111111111111111").unwrap(),
        runtime_lens_id: None,
        weights_sha256: "2".repeat(64),
        runtime: runtime.to_string(),
        modality: Modality::Text,
        shape: SlotShape::Dense(8),
        placement,
        cost: LensCost::default(),
        manifest: format!("/lenses/{name}/manifest.json"),
        immutable_snapshot: None,
        counts_toward_a35: true,
    }
}

fn template(lenses: Vec<TemplateLensRef>) -> SavedPanelTemplate {
    SavedPanelTemplate {
        schema_version: OBJECT_VERSION,
        name: "legal-test".to_string(),
        version: 1,
        notes: String::new(),
        min_content_lenses: MIN_CONTENT_LENSES,
        lenses,
        time_controls: default_time_controls(),
        ensemble_card: None,
    }
}

#[test]
fn gpu_only_template_passes_admission() {
    let template = template(vec![
        lens("bge-small", "onnx", Placement::Gpu),
        lens("gte-base", "onnx", Placement::Gpu),
    ]);

    require_gpu_lens_admission_with_allow_list(&template, "save", &BTreeSet::new())
        .expect("GPU-only template admitted");
}

#[test]
fn cpu_lens_is_refused_without_opt_in_and_names_the_lens() {
    let template = template(vec![
        lens("bge-small", "onnx", Placement::Gpu),
        lens("semantic_potion_base_8m", "static_lookup", Placement::Cpu),
    ]);

    let error = require_gpu_lens_admission_with_allow_list(&template, "swap", &BTreeSet::new())
        .unwrap_err();

    assert_eq!(error.code(), TEMPLATE_CPU_LENS_REFUSED);
    assert!(error.message().contains("semantic_potion_base_8m"));
    assert!(error.message().contains("static_lookup"));
    assert!(error.message().contains("GPU deployment policy"));
    assert!(error.remediation().contains(ALLOW_CPU_LENS_ENV));
}

#[test]
fn cpu_lens_with_explicit_opt_in_is_admitted() {
    let template = template(vec![
        lens("bge-small", "onnx", Placement::Gpu),
        lens("semantic_potion_base_8m", "static_lookup", Placement::Cpu),
    ]);
    let allow = parse_allow_list(Some("semantic_potion_base_8m"));

    require_gpu_lens_admission_with_allow_list(&template, "save", &allow)
        .expect("opted-in CPU lens admitted");
}

#[test]
fn opt_in_is_per_lens_not_blanket() {
    let template = template(vec![
        lens("semantic_potion_base_8m", "static_lookup", Placement::Cpu),
        lens("other_cpu_lens", "onnx", Placement::Cpu),
    ]);
    let allow = parse_allow_list(Some("semantic_potion_base_8m"));

    let error = require_gpu_lens_admission_with_allow_list(&template, "save", &allow).unwrap_err();

    assert_eq!(error.code(), TEMPLATE_CPU_LENS_REFUSED);
    assert!(error.message().contains("other_cpu_lens"));
    assert!(!error.message().contains("semantic_potion_base_8m"));
}

#[test]
fn cpu_only_runtime_claiming_gpu_placement_is_refused_even_with_opt_in() {
    let template = template(vec![lens(
        "semantic_potion_base_8m",
        "static_lookup",
        Placement::Gpu,
    )]);
    let allow = parse_allow_list(Some("semantic_potion_base_8m"));

    let error = require_gpu_lens_admission_with_allow_list(&template, "save", &allow).unwrap_err();

    assert_eq!(error.code(), TEMPLATE_CPU_LENS_REFUSED);
    assert!(error.message().contains("CPU-only"));
    assert!(error.message().contains("placement Gpu"));
}

#[test]
fn allow_list_parses_comma_separated_names_with_whitespace() {
    let allow = parse_allow_list(Some(" potion , other_lens ,, "));

    assert_eq!(
        allow,
        BTreeSet::from(["potion".to_string(), "other_lens".to_string()])
    );
    assert!(parse_allow_list(None).is_empty());
    assert!(parse_allow_list(Some("")).is_empty());
}
