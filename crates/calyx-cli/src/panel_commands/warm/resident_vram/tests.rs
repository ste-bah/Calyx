use super::*;

#[test]
fn grouped_vram_counts_one_shared_runtime_and_every_independent_runtime() {
    let bge_m3 = MeasurementGroupKey::from_bytes([0xB3; 32]);
    let total = sum_grouped_vram_bytes([
        (Some(bge_m3), 1_153_516_174),
        (Some(bge_m3), 1_153_516_174),
        (Some(bge_m3), 1_153_516_174),
        (None, 1_127_143_136),
        (None, 548_026_095),
    ])
    .unwrap();

    assert_eq!(total, 2_828_685_405);
}

#[test]
fn grouped_vram_uses_largest_declared_cost_within_one_runtime() {
    let shared = MeasurementGroupKey::from_bytes([0x44; 32]);
    let total =
        sum_grouped_vram_bytes([(Some(shared), 128), (Some(shared), 256), (Some(shared), 64)])
            .unwrap();

    assert_eq!(total, 256);
}

#[test]
fn grouped_vram_overflow_fails_closed() {
    let error = sum_grouped_vram_bytes([(None, u64::MAX), (None, 1)]).unwrap_err();

    assert_eq!(error.code(), WARM_VRAM_BUDGET);
    assert!(error.message().contains("overflowed u64"));
}
