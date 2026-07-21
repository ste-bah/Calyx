use super::*;

#[test]
fn lagged_samples_condition_on_target_immediate_past_for_lag_gt_one() {
    let source = (0..8).map(|t| (t, t as f32)).collect::<Vec<_>>();
    let target = (0..8).map(|t| (t, 100.0 + t as f32)).collect::<Vec<_>>();

    let samples = lagged_samples(&source, &target, 2, 1).unwrap();

    assert_eq!(samples[1].future, vec![103.0]);
    assert_eq!(samples[1].joint_past, vec![1.0, 102.0]);
    assert_eq!(samples[1].own_past, vec![102.0]);
}
