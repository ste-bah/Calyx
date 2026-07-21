use calyx_core::{CalyxError, Result};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

pub const STRICT_CUDA_ENV: &str = "CALYX_ASSAY_CUDA_STRICT";

pub fn strict_cuda_requested() -> bool {
    std::env::var(STRICT_CUDA_ENV)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_unavailable(op: &str) -> CalyxError {
    CalyxError::forge_device_unavailable(format!(
        "{op} requires calyx-assay feature `cuda` and a working Forge CUDA runtime when {STRICT_CUDA_ENV}=1; strict mode does not fall back to CPU"
    ))
}

pub fn deterministic_permutations(n: usize, permutations: usize, seed: u64) -> Result<Vec<i32>> {
    if n > i32::MAX as usize {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "assay CUDA permutation rows exceed i32 kernel index range: n={n}"
        )));
    }
    let capacity = n
        .checked_mul(permutations)
        .ok_or_else(|| CalyxError::forge_vram_budget("assay CUDA permutation buffer overflow"))?;
    let mut out = Vec::with_capacity(capacity);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut perm: Vec<i32> = (0..n as i32).collect();
    for _ in 0..permutations {
        perm.shuffle(&mut rng);
        out.extend_from_slice(&perm);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn forge_to_calyx(op: &str, err: calyx_forge::ForgeError) -> CalyxError {
    let message = format!("{op} CUDA strict failure: {err}");
    match err {
        calyx_forge::ForgeError::NumericalInvariant { .. } => {
            CalyxError::forge_numerical_invariant(message)
        }
        calyx_forge::ForgeError::DeviceUnavailable { .. } => {
            CalyxError::forge_device_unavailable(message)
        }
        calyx_forge::ForgeError::VramBudget { .. }
        | calyx_forge::ForgeError::LensVramBudget { .. } => CalyxError::forge_vram_budget(message),
        calyx_forge::ForgeError::ShapeMismatch { .. } => {
            CalyxError::assay_insufficient_samples(message)
        }
        _ => CalyxError::forge_device_unavailable(message),
    }
}

#[cfg(feature = "cuda")]
pub fn forge_linear_algebra_to_calyx(op: &str, err: calyx_forge::ForgeError) -> CalyxError {
    let message = format!("{op} CUDA strict linear algebra failure: {err}");
    match err {
        calyx_forge::ForgeError::NumericalInvariant { .. } => {
            CalyxError::assay_degenerate_input(message)
        }
        calyx_forge::ForgeError::ShapeMismatch { .. } => {
            CalyxError::assay_insufficient_samples(message)
        }
        other => forge_to_calyx(op, other),
    }
}
