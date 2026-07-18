use super::*;

pub(super) fn provisional_tc(slot_count: usize, n_samples: usize, clock: &dyn Clock) -> TCResult {
    TCResult {
        tc: 0.0,
        n_eff: slot_count as f32,
        ci_95: (0.0, 0.0),
        n_samples,
        slot_count,
        sum_marginal_entropy: 0.0,
        joint_entropy: 0.0,
        provisional: true,
        error_code: Some(CALYX_TC_INSUFFICIENT_SAMPLES.to_string()),
        trust: TrustTag::Provisional,
        computed_at: clock.now(),
    }
}

pub(super) fn provisional_ii(n_samples: usize, clock: &dyn Clock) -> IIResult {
    IIResult {
        ii: 0.0,
        sign: IISign::Unclear,
        ci_95: (0.0, 0.0),
        n_samples,
        provisional: true,
        error_code: Some(CALYX_TC_INSUFFICIENT_SAMPLES.to_string()),
        trust: TrustTag::Provisional,
        computed_at: clock.now(),
    }
}
