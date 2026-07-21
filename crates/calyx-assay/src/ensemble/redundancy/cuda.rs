use super::*;

#[cfg(feature = "cuda")]
use crate::ensemble::model::LinearCkaEstimate;

#[cfg(feature = "cuda")]
pub(super) fn ensemble_redundancy_from_lenses_cuda_strict_impl(
    lenses: &[EnsembleLensInput],
    nmi_bins: usize,
) -> Result<EnsembleRedundancyEvidence> {
    let row_count = lenses.first().map(|lens| lens.vectors.len()).unwrap_or(0);
    let plan = linear_cka_tuple_plan(row_count)?;
    let (flat, offsets, dimensions) = flatten_lenses_for_linear_cka_cuda(lenses, row_count)?;
    let tuples = tuple_plan_indices_for_cuda(&plan)?;
    let cka = linear_cka_pair_estimates_cuda_strict_impl(
        &flat,
        &offsets,
        &dimensions,
        row_count,
        &tuples,
        plan.is_exact(),
    )?;
    let nmi_signatures = lenses
        .iter()
        .map(|lens| row_signature(&lens.vectors))
        .collect::<Result<Vec<_>>>()?;
    let mut pairs = Vec::new();
    let mut pair_index = 0usize;
    for a in 0..lenses.len() {
        for b in (a + 1)..lenses.len() {
            let linear_cka = LinearCkaEstimate {
                raw_signed_point: cka.raw_signed_point[pair_index],
                redundancy_point: cka.redundancy_point[pair_index],
                mc_standard_error: cka.mc_standard_error[pair_index],
                mc_gate_upper_estimate: cka.mc_gate_upper_estimate[pair_index],
            };
            let nmi =
                partitioned_histogram_nmi(&nmi_signatures[a], &nmi_signatures[b], nmi_bins)?.nmi;
            pairs.push(EnsemblePairRedundancyEvidence {
                a: lenses[a].name.clone(),
                b: lenses[b].name.clone(),
                slot_a: lenses[a].slot,
                slot_b: lenses[b].slot,
                linear_cka,
                nmi,
            });
            pair_index += 1;
        }
    }
    let evidence = EnsembleRedundancyEvidence {
        method: redundancy_method(&plan),
        pairs,
    };
    validate_evidence(lenses, &evidence)?;
    Ok(evidence)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn ensemble_redundancy_from_lenses_cuda_strict_impl(
    _lenses: &[EnsembleLensInput],
    _nmi_bins: usize,
) -> Result<EnsembleRedundancyEvidence> {
    Err(cuda_unavailable("ensemble redundancy linear CKA"))
}
#[cfg(feature = "cuda")]
fn flatten_lenses_for_linear_cka_cuda(
    lenses: &[EnsembleLensInput],
    row_count: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<i32>)> {
    if lenses.len() < 2 {
        return Err(CalyxError::assay_insufficient_samples(format!(
            "linear CKA CUDA requires at least two lenses; got {}",
            lenses.len()
        )));
    }
    let mut flat = Vec::new();
    let mut offsets = vec![0_i32];
    let mut dimensions = Vec::with_capacity(lenses.len());
    for lens in lenses {
        if lens.vectors.len() != row_count {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "linear CKA CUDA lens {} rows {} != {}",
                lens.name,
                lens.vectors.len(),
                row_count
            )));
        }
        let dim = lens.vectors.first().map(Vec::len).unwrap_or(0);
        if dim == 0 {
            return Err(CalyxError::assay_insufficient_samples(format!(
                "linear CKA CUDA lens {} has empty vectors",
                lens.name
            )));
        }
        dimensions.push(usize_to_i32_for_linear_cka_cuda(
            dim,
            "linear CKA dimension",
        )?);
        for (row_idx, row) in lens.vectors.iter().enumerate() {
            if row.len() != dim {
                return Err(CalyxError::assay_insufficient_samples(format!(
                    "linear CKA CUDA lens {} row {} dim {} != {}",
                    lens.name,
                    row_idx,
                    row.len(),
                    dim
                )));
            }
            for (col_idx, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(CalyxError::forge_numerical_invariant(format!(
                        "linear CKA CUDA lens {} row {} col {} is non-finite: {}",
                        lens.name, row_idx, col_idx, value
                    )));
                }
                flat.push(value);
            }
        }
        offsets.push(usize_to_i32_for_linear_cka_cuda(
            flat.len(),
            "linear CKA lens offset",
        )?);
    }
    Ok((flat, offsets, dimensions))
}

#[cfg(feature = "cuda")]
fn tuple_plan_indices_for_cuda(plan: &LinearCkaTuplePlan) -> Result<Vec<i32>> {
    let mut tuples = Vec::with_capacity(plan.tuples.len() * 4);
    for tuple in &plan.tuples {
        for &row in tuple {
            tuples.push(usize_to_i32_for_linear_cka_cuda(
                row,
                "linear CKA tuple row index",
            )?);
        }
    }
    Ok(tuples)
}

#[cfg(feature = "cuda")]
fn usize_to_i32_for_linear_cka_cuda(value: usize, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        CalyxError::assay_insufficient_samples(format!(
            "{name} exceeds CUDA i32 index range: {value}"
        ))
    })
}

#[cfg(feature = "cuda")]
fn linear_cka_pair_estimates_cuda_strict_impl(
    values: &[f32],
    offsets: &[i32],
    dimensions: &[i32],
    row_count: usize,
    tuples: &[i32],
    exact: bool,
) -> Result<calyx_forge::CudaLinearCkaPairEstimates> {
    let backend = calyx_forge::CudaBackend::new()
        .map_err(|err| crate::cuda_strict::forge_to_calyx("linear CKA", err))?;
    calyx_forge::linear_cka_pair_estimates_host(
        backend.context(),
        values,
        offsets,
        dimensions,
        row_count,
        tuples,
        exact,
    )
    .map_err(|err| crate::cuda_strict::forge_to_calyx("linear CKA", err))
}
