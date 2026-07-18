use super::*;

pub(super) fn assert_close_vec(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{name} mismatch at {idx}: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
        );
    }
}

pub(super) fn assert_close_f64_vec(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_close_f64(&format!("{name}[{idx}]"), actual, expected, tolerance);
    }
}

pub(super) fn assert_close_f64(name: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{name} mismatch: actual={actual} expected={expected} diff={diff} tolerance={tolerance}"
    );
}
