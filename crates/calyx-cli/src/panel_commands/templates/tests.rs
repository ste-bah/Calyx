use super::*;

#[test]
fn parses_repeated_lenses() {
    let flags = Flags::parse(&[
        "--name".to_string(),
        "text-deep".to_string(),
        "--lens".to_string(),
        "a".to_string(),
        "--lens".to_string(),
        "b".to_string(),
    ])
    .unwrap();

    assert_eq!(flags.name.as_deref(), Some("text-deep"));
    assert_eq!(flags.lenses, ["a", "b"]);
}

#[test]
fn parses_a37_template_flags() {
    let flags = Flags::parse(&[
        "--template".to_string(),
        "text-deep".to_string(),
        "--assay-card".to_string(),
        "ensemble_card.json".to_string(),
        "--require-a37-gate".to_string(),
        "--a37-admission-card".to_string(),
        "multi_anchor_card.json".to_string(),
    ])
    .unwrap();

    assert_eq!(flags.template.as_deref(), Some("text-deep"));
    assert_eq!(
        flags.assay_card.as_deref(),
        Some(Path::new("ensemble_card.json"))
    );
    assert_eq!(
        flags.a37_admission_card.as_deref(),
        Some(Path::new("multi_anchor_card.json"))
    );
    assert!(flags.require_a37_gate);
}

#[test]
fn modality_parser_matches_catalog_strings() {
    assert_eq!(modality_name(parse_modality("text").unwrap()), "text");
    assert!(parse_modality("temporal").is_err());
}

#[test]
fn resident_backed_swap_address_is_explicit_and_loopback_only() {
    let flags = Flags::parse(&[
        "--template".to_string(),
        "legal-v1".to_string(),
        "--vault".to_string(),
        "legal-pilot".to_string(),
        "--resident-addr".to_string(),
        "127.0.0.1:18401".to_string(),
    ])
    .unwrap();
    assert_eq!(
        flags.resident_addr,
        Some("127.0.0.1:18401".parse().unwrap())
    );

    let error = Flags::parse(&["--resident-addr".to_string(), "192.0.2.1:18401".to_string()])
        .expect_err("remote resident must fail closed");
    assert!(error.message().contains("must be loopback"));

    let error = required_swap_resident(None).expect_err("swap without a resident must fail");
    assert_eq!(error.code(), RESIDENT_REQUIRED_CODE);
    assert!(error.message().contains("second GPU model set"));
}
