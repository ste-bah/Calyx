//! Host-side validation shared by strict CUDA Loom entry points.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::collections::BTreeMap;

use calyx_core::{Result, SlotId};

use crate::error::{
    CALYX_LOOM_DIM_MISMATCH, CALYX_LOOM_NON_FINITE_VECTOR, CALYX_LOOM_SLOT_MISSING,
    CALYX_LOOM_ZERO_NORM_VECTOR, loom_error,
};

pub(crate) fn slot_values(slots: &BTreeMap<SlotId, Vec<f32>>, slot: SlotId) -> Result<&[f32]> {
    slots.get(&slot).map(Vec::as_slice).ok_or_else(|| {
        loom_error(
            CALYX_LOOM_SLOT_MISSING,
            format!("slot {} missing", slot.get()),
        )
    })
}

pub(crate) fn validate_row(values: &[f32], dim: &mut Option<usize>) -> Result<()> {
    match *dim {
        Some(expected) if expected != values.len() => {
            return Err(loom_error(
                CALYX_LOOM_DIM_MISMATCH,
                format!("xterm dims {expected} and {}", values.len()),
            ));
        }
        None if values.is_empty() => {
            return Err(loom_error(
                CALYX_LOOM_DIM_MISMATCH,
                "xterm vectors must be non-empty",
            ));
        }
        None => *dim = Some(values.len()),
        Some(_) => {}
    }
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }
    Err(loom_error(
        CALYX_LOOM_NON_FINITE_VECTOR,
        "xterm vector contains NaN or infinity",
    ))
}

pub(crate) fn validate_agreement_norm(values: &[f32]) -> Result<()> {
    let norm = values
        .iter()
        .fold(0.0_f32, |sum, value| sum + value * value);
    if norm > f32::EPSILON {
        return Ok(());
    }
    Err(loom_error(
        CALYX_LOOM_ZERO_NORM_VECTOR,
        "agreement requires non-zero vectors",
    ))
}
