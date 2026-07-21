use super::*;

pub(super) fn maxsim_cuda_error(slot: SlotId) -> impl FnOnce(calyx_core::CalyxError) -> CliError {
    move |err| {
        stale(format!(
            "persistent MaxSim CUDA search failed for slot {slot}: {err}; rebuild with CUDA available or unset CALYX_SEARCH_MAXSIM_CUDA_STRICT"
        ))
    }
}

pub(super) fn cuda_scores(
    result: calyx_sextant::index::MaxSimCudaTopK,
    k: usize,
) -> Vec<(CxId, f32)> {
    let mut scored = Vec::with_capacity(result.scores.len());
    for ((hi, lo), score) in result
        .id_hi
        .iter()
        .zip(result.id_lo.iter())
        .zip(result.scores.iter())
    {
        scored.push((cx_id_from_halves(*hi, *lo), *score));
    }
    top_k(scored, k)
}
