//! Strict CUDA routing and resident Forge batching for production Loom paths.

use std::collections::BTreeMap;

use calyx_core::{Result, SlotId};
use serde::{Deserialize, Serialize};

use crate::cross_term::{CrossTermKind, CrossTermValue};
use crate::error::{CALYX_LOOM_FORGE_UNAVAILABLE, loom_error};

pub const LOOM_CUDA_STRICT_ENV: &str = "CALYX_LOOM_CUDA_STRICT";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoomCudaStats {
    pub row_count: usize,
    pub dim: usize,
    pub agreement_pairs: usize,
    pub vector_terms: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub peak_device_bytes: usize,
    pub host_to_device_copies: usize,
    pub device_to_host_copies: usize,
    pub kernel_launches: usize,
    pub gemm_calls: usize,
    pub workspace_reused: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaTermRequest {
    pub a: SlotId,
    pub b: SlotId,
    pub kind: CrossTermKind,
}

pub(crate) struct CudaTermBatch {
    pub values: Vec<CrossTermValue>,
    pub stats: LoomCudaStats,
}

pub fn loom_cuda_strict_requested() -> bool {
    std::env::var(LOOM_CUDA_STRICT_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(crate) fn execute_terms(
    slots: &BTreeMap<SlotId, Vec<f32>>,
    requests: &[CudaTermRequest],
) -> Result<CudaTermBatch> {
    #[cfg(feature = "cuda")]
    {
        execute_terms_cuda(slots, requests)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (slots, requests);
        Err(loom_error(
            CALYX_LOOM_FORGE_UNAVAILABLE,
            "strict Loom CUDA execution requires calyx-loom feature cuda",
        ))
    }
}

pub(crate) fn agreement_slices_gpu(pairs: &[(&[f32], &[f32])]) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        agreement_slices_cuda(pairs)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = pairs;
        Err(loom_error(
            CALYX_LOOM_FORGE_UNAVAILABLE,
            "agreement_batch_gpu requires calyx-loom feature cuda",
        ))
    }
}

#[cfg(feature = "cuda")]
fn execute_terms_cuda(
    slots: &BTreeMap<SlotId, Vec<f32>>,
    requests: &[CudaTermRequest],
) -> Result<CudaTermBatch> {
    use std::collections::{BTreeMap as RowMap, BTreeSet};

    use calyx_forge::{CudaLoomBatch, CudaLoomVectorKind, CudaLoomVectorRequest};

    if requests.is_empty() {
        return Ok(CudaTermBatch {
            values: Vec::new(),
            stats: LoomCudaStats::default(),
        });
    }
    let mut row_ids = BTreeSet::new();
    let mut agreement_ids = BTreeSet::new();
    for request in requests {
        row_ids.extend([request.a, request.b]);
        if request.kind == CrossTermKind::Agreement {
            agreement_ids.extend([request.a, request.b]);
        }
    }
    let mut row_map = RowMap::new();
    let mut matrix = Vec::new();
    let mut dim = None;
    for (row, slot) in row_ids.into_iter().enumerate() {
        let values = crate::cuda_validation::slot_values(slots, slot)?;
        crate::cuda_validation::validate_row(values, &mut dim)?;
        if agreement_ids.contains(&slot) {
            crate::cuda_validation::validate_agreement_norm(values)?;
        }
        row_map.insert(slot, row);
        matrix.extend_from_slice(values);
    }
    let dim = dim.expect("non-empty request row set");
    let mut agreements = Vec::new();
    let mut vectors = Vec::new();
    let mut routes = Vec::with_capacity(requests.len());
    for request in requests {
        let left_row = row_map[&request.a];
        let right_row = row_map[&request.b];
        match request.kind {
            CrossTermKind::Agreement => {
                routes.push(OutputRoute::Agreement(agreements.len()));
                agreements.push((left_row, right_row));
            }
            CrossTermKind::Delta | CrossTermKind::Interaction => {
                routes.push(OutputRoute::Vector(vectors.len()));
                vectors.push(CudaLoomVectorRequest {
                    left_row,
                    right_row,
                    kind: if request.kind == CrossTermKind::Delta {
                        CudaLoomVectorKind::Delta
                    } else {
                        CudaLoomVectorKind::Interaction
                    },
                });
            }
            CrossTermKind::Concat => unreachable!("concat remains host-only"),
        }
    }
    let CudaLoomBatch {
        agreements,
        vector_terms,
        stats,
    } = resident_context()?
        .execute(&matrix, row_map.len(), dim, &agreements, &vectors)
        .map_err(map_forge_error)?;
    let mut vector_terms: Vec<_> = vector_terms.into_iter().map(Some).collect();
    let values = routes
        .into_iter()
        .map(|route| match route {
            OutputRoute::Agreement(index) => CrossTermValue::Scalar(agreements[index]),
            OutputRoute::Vector(index) => {
                CrossTermValue::Vector(vector_terms[index].take().expect("unique output route"))
            }
        })
        .collect();
    Ok(CudaTermBatch {
        values,
        stats: stats.into(),
    })
}

#[cfg(feature = "cuda")]
fn agreement_slices_cuda(pairs: &[(&[f32], &[f32])]) -> Result<Vec<f32>> {
    use calyx_forge::CudaLoomBatch;

    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let mut matrix = Vec::new();
    let mut dim = None;
    for (left, right) in pairs {
        crate::cuda_validation::validate_row(left, &mut dim)?;
        crate::cuda_validation::validate_row(right, &mut dim)?;
        crate::cuda_validation::validate_agreement_norm(left)?;
        crate::cuda_validation::validate_agreement_norm(right)?;
        matrix.extend_from_slice(left);
        matrix.extend_from_slice(right);
    }
    let dim = dim.expect("non-empty agreement batch");
    let agreement_pairs: Vec<_> = (0..pairs.len())
        .map(|index| (2 * index, 2 * index + 1))
        .collect();
    let CudaLoomBatch { agreements, .. } = resident_context()?
        .execute(&matrix, pairs.len() * 2, dim, &agreement_pairs, &[])
        .map_err(map_forge_error)?;
    Ok(agreements)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum OutputRoute {
    Agreement(usize),
    Vector(usize),
}

#[cfg(feature = "cuda")]
fn resident_context() -> Result<&'static calyx_forge::CudaLoomContext> {
    use std::sync::OnceLock;

    static CONTEXT: OnceLock<calyx_forge::CudaLoomContext> = OnceLock::new();
    if let Some(context) = CONTEXT.get() {
        return Ok(context);
    }
    let created = calyx_forge::CudaLoomContext::new(0).map_err(map_forge_error)?;
    let _ = CONTEXT.set(created);
    Ok(CONTEXT.get().expect("resident CUDA context initialized"))
}

#[cfg(feature = "cuda")]
fn map_forge_error(error: calyx_forge::ForgeError) -> calyx_core::CalyxError {
    use crate::error::{
        CALYX_LOOM_DIM_MISMATCH, CALYX_LOOM_NON_FINITE_VECTOR, CALYX_LOOM_ZERO_NORM_VECTOR,
    };
    use calyx_forge::ForgeError;

    let rendered = error.to_string();
    let code = match &error {
        ForgeError::ShapeMismatch { .. } => CALYX_LOOM_DIM_MISMATCH,
        ForgeError::NumericalInvariant { detail, .. } if detail.contains("zero-norm") => {
            CALYX_LOOM_ZERO_NORM_VECTOR
        }
        ForgeError::NumericalInvariant { detail, .. }
            if detail.contains("non-finite")
                || detail.contains("NaN")
                || detail.contains("infinity") =>
        {
            CALYX_LOOM_NON_FINITE_VECTOR
        }
        _ => CALYX_LOOM_FORGE_UNAVAILABLE,
    };
    loom_error(code, format!("Forge Loom CUDA batch failed: {rendered}"))
}

#[cfg(feature = "cuda")]
impl From<calyx_forge::CudaLoomStats> for LoomCudaStats {
    fn from(stats: calyx_forge::CudaLoomStats) -> Self {
        Self {
            row_count: stats.row_count,
            dim: stats.dim,
            agreement_pairs: stats.agreement_pairs,
            vector_terms: stats.vector_terms,
            host_to_device_bytes: stats.host_to_device_bytes,
            device_to_host_bytes: stats.device_to_host_bytes,
            peak_device_bytes: stats.peak_device_bytes,
            host_to_device_copies: stats.host_to_device_copies,
            device_to_host_copies: stats.device_to_host_copies,
            kernel_launches: stats.kernel_launches,
            gemm_calls: stats.gemm_calls,
            workspace_reused: stats.workspace_reused,
        }
    }
}
