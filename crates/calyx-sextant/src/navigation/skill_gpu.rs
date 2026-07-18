use std::collections::BTreeMap;

use calyx_core::{Result, SlotId};

use super::skills::SkillExecutionStats;
use crate::error::{CALYX_SEXTANT_SKILL_CUDA_REQUIRED, sextant_error};

pub(super) const SKILL_CUDA_MIN_POINTS: usize = 256;

type SkillMstEdges = Vec<(usize, usize, f64)>;
type SkillMstResult = Result<(SkillMstEdges, SkillExecutionStats)>;

#[cfg(feature = "cuda")]
pub(super) fn minimum_spanning_tree(
    vectors: &[BTreeMap<SlotId, Vec<f32>>],
    min_samples: usize,
) -> SkillMstResult {
    let slots = flatten_slots(vectors)?;
    let output = context()?
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .minimum_spanning_tree(vectors.len(), &slots, min_samples)
        .map_err(map_forge_error)?;
    let edges = output
        .edges
        .iter()
        .map(|edge| (edge.source, edge.destination, edge.weight))
        .collect();
    Ok((edges, SkillExecutionStats::cuda(output.stats)))
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn large_shape_refuses_cpu_fallback_without_cuda() {
        let vectors = vec![BTreeMap::new(); SKILL_CUDA_MIN_POINTS];
        let error = minimum_spanning_tree(&vectors, 1).expect_err("strict CUDA requirement");
        assert_eq!(error.code, CALYX_SEXTANT_SKILL_CUDA_REQUIRED);
    }
}

#[cfg(feature = "cuda")]
fn context() -> Result<&'static std::sync::Mutex<calyx_forge::CudaSkillContext>> {
    use std::sync::{Mutex, OnceLock};

    debug_assert_eq!(SKILL_CUDA_MIN_POINTS, calyx_forge::SKILL_CUDA_MIN_POINTS);
    static CONTEXT: OnceLock<std::result::Result<Mutex<calyx_forge::CudaSkillContext>, String>> =
        OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            calyx_forge::CudaSkillContext::new(0)
                .map(Mutex::new)
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|detail| {
            sextant_error(
                CALYX_SEXTANT_SKILL_CUDA_REQUIRED,
                format!("strict CUDA skill provider initialization failed: {detail}"),
            )
        })
}

#[cfg(feature = "cuda")]
fn flatten_slots(
    vectors: &[BTreeMap<SlotId, Vec<f32>>],
) -> Result<Vec<calyx_forge::CudaSkillSlot>> {
    use std::collections::BTreeSet;

    use crate::error::CALYX_SEXTANT_DIM_MISMATCH;

    let slot_ids = vectors
        .iter()
        .flat_map(|point| point.keys().copied())
        .collect::<BTreeSet<_>>();
    let mut output = Vec::with_capacity(slot_ids.len());
    for slot_id in slot_ids {
        let mut dim = None;
        let mut point_indices = Vec::new();
        let mut values = Vec::new();
        for (point, slots) in vectors.iter().enumerate() {
            let Some(vector) = slots.get(&slot_id) else {
                continue;
            };
            match dim {
                Some(expected) if expected != vector.len() => {
                    return Err(sextant_error(
                        CALYX_SEXTANT_DIM_MISMATCH,
                        format!(
                            "skill slot {slot_id} dims differ: {expected} vs {}",
                            vector.len()
                        ),
                    ));
                }
                None => dim = Some(vector.len()),
                _ => {}
            }
            point_indices.push(point as u32);
            values.extend_from_slice(vector);
        }
        output.push(calyx_forge::CudaSkillSlot {
            dim: dim.unwrap_or(0),
            point_indices,
            values,
        });
    }
    Ok(output)
}

#[cfg(feature = "cuda")]
fn map_forge_error(error: calyx_forge::ForgeError) -> calyx_core::CalyxError {
    use calyx_forge::ForgeError;

    use crate::error::{CALYX_SEXTANT_SKILL_PAIR_NO_OVERLAP, CALYX_SEXTANT_VECTOR_SHAPE};

    let code = match &error {
        ForgeError::NumericalInvariant { op, .. } if op == "skill.pair_no_overlap" => {
            CALYX_SEXTANT_SKILL_PAIR_NO_OVERLAP
        }
        ForgeError::NumericalInvariant { op, .. }
            if op == "skill.vector_finite" || op == "skill.vector_norm" =>
        {
            CALYX_SEXTANT_VECTOR_SHAPE
        }
        _ => CALYX_SEXTANT_SKILL_CUDA_REQUIRED,
    };
    sextant_error(code, error.to_string())
}

#[cfg(not(feature = "cuda"))]
pub(super) fn minimum_spanning_tree(
    vectors: &[BTreeMap<SlotId, Vec<f32>>],
    _min_samples: usize,
) -> SkillMstResult {
    Err(sextant_error(
        CALYX_SEXTANT_SKILL_CUDA_REQUIRED,
        format!(
            "skill clustering points={} reached CUDA crossover {SKILL_CUDA_MIN_POINTS}, but calyx-sextant was compiled without feature `cuda`; refusing silent O(N^2) CPU fallback",
            vectors.len()
        ),
    ))
}
