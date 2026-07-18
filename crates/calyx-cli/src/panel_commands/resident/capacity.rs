use super::server::ResidentService;
use super::stream::{DEFAULT_MEASURE_WINDOW_MULTIPLIER, measure_batch_chunked};
use super::*;
use crate::panel_commands::warm::warm_probe_bytes;
use calyx_registry::{LensRuntime, OnnxShapeBucketBudget, onnx_shape_bucket_budget};

const CAPACITY_PROBE_FAILED: &str = "CALYX_PANEL_RESIDENT_CAPACITY_PROBE_FAILED";
// Exercise consecutive full streaming windows. One window proves only the
// first allocation shape; the second proves that retained arena allocations
// leave enough capacity for the next production window.
const CAPACITY_PROBE_WINDOWS: usize = 2;

pub(super) struct CapacityProbeReport {
    pub(super) input_count: usize,
    pub(super) elapsed_ms: u128,
    pub(super) modalities: Vec<Modality>,
    pub(super) onnx_shape_budget: Option<OnnxShapeBucketBudget>,
}

pub(super) fn run(service: &ResidentService) -> CliResult<CapacityProbeReport> {
    let modalities = capacity_probe_modalities(service);
    let onnx_shape_budget = resident_uses_onnx(service)
        .then(|| onnx_shape_bucket_budget(service.max_runtime_batch))
        .transpose()?;
    if let Some(budget) = onnx_shape_budget {
        eprintln!(
            "CALYX_PANEL_RESIDENT_RUNTIME phase=shape_budget_admitted process_id={} configured_shape_limit={} required_shape_count={} sequence_bucket_count={} batch_bucket_count={} max_sequence_tokens={} max_runtime_batch={}",
            std::process::id(),
            budget.configured_shape_limit,
            budget.required_shape_count,
            budget.sequence_bucket_count,
            budget.batch_bucket_count,
            budget.max_sequence_tokens,
            budget.max_runtime_batch,
        );
    }
    let inputs_per_modality = service
        .max_runtime_batch
        .checked_mul(DEFAULT_MEASURE_WINDOW_MULTIPLIER)
        .and_then(|count| count.checked_mul(CAPACITY_PROBE_WINDOWS))
        .ok_or_else(|| capacity_error(service, None, "capacity probe input count overflowed"))?;
    let started = Instant::now();
    for modality in &modalities {
        let bytes = capacity_probe_bytes(*modality)?;
        let inputs = vec![bytes; inputs_per_modality];
        let mut emitted = 0_usize;
        eprintln!(
            "CALYX_PANEL_RESIDENT_RUNTIME phase=capacity_probe_start process_id={} modality={modality:?} inputs={inputs_per_modality} max_runtime_batch={}",
            std::process::id(),
            service.max_runtime_batch,
        );
        measure_batch_chunked(
            service,
            *modality,
            inputs,
            Some(service.max_runtime_batch),
            &mut |rows| {
                emitted = emitted.checked_add(rows.len()).ok_or_else(|| {
                    capacity_error(service, Some(*modality), "emitted row count overflowed")
                })?;
                Ok(())
            },
        )
        .map_err(|error| capacity_cause(service, *modality, error))?;
        if emitted != inputs_per_modality {
            return Err(capacity_error(
                service,
                Some(*modality),
                &format!("capacity probe emitted {emitted} rows for {inputs_per_modality} inputs"),
            ));
        }
        eprintln!(
            "CALYX_PANEL_RESIDENT_RUNTIME phase=capacity_probe_ok process_id={} modality={modality:?} inputs={emitted} max_runtime_batch={} elapsed_ms={}",
            std::process::id(),
            service.max_runtime_batch,
            started.elapsed().as_millis(),
        );
    }
    Ok(CapacityProbeReport {
        input_count: inputs_per_modality
            .checked_mul(modalities.len())
            .ok_or_else(|| {
                capacity_error(service, None, "capacity probe report count overflowed")
            })?,
        elapsed_ms: started.elapsed().as_millis(),
        modalities,
        onnx_shape_budget,
    })
}

fn resident_uses_onnx(service: &ResidentService) -> bool {
    service.state.build.panel.slots.iter().any(|slot| {
        slot.state == SlotState::Active
            && !slot.retrieval_only
            && !slot.excluded_from_dedup
            && service
                .state
                .build
                .registry
                .lens_spec(slot.lens_id)
                .is_some_and(|spec| {
                    matches!(
                        spec.runtime,
                        LensRuntime::Onnx { .. }
                            | LensRuntime::OnnxColbert { .. }
                            | LensRuntime::FastembedSparse { .. }
                            | LensRuntime::FastembedBgem3 { .. }
                            | LensRuntime::FastembedReranker { .. }
                            | LensRuntime::FastembedQwen3 { .. }
                    )
                })
    })
}

fn capacity_probe_modalities(service: &ResidentService) -> Vec<Modality> {
    const ORDER: [Modality; 10] = [
        Modality::Text,
        Modality::Code,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
        Modality::Protein,
        Modality::Dna,
        Modality::Molecule,
        Modality::Structured,
        Modality::Mixed,
    ];
    ORDER
        .into_iter()
        .filter(|modality| {
            service.state.build.panel.slots.iter().any(|slot| {
                slot.state == SlotState::Active
                    && slot.modality == *modality
                    && !slot.retrieval_only
                    && !slot.excluded_from_dedup
                    && service
                        .state
                        .build
                        .registry
                        .lens_spec(slot.lens_id)
                        .is_some_and(|spec| is_local_dynamic_runtime(&spec.runtime))
            })
        })
        .collect()
}

fn is_local_dynamic_runtime(runtime: &LensRuntime) -> bool {
    matches!(
        runtime,
        LensRuntime::CandleLocal { .. }
            | LensRuntime::Onnx { .. }
            | LensRuntime::OnnxColbert { .. }
            | LensRuntime::FastembedSparse { .. }
            | LensRuntime::FastembedBgem3 { .. }
            | LensRuntime::FastembedReranker { .. }
            | LensRuntime::FastembedQwen3 { .. }
    )
}

fn capacity_probe_bytes(modality: Modality) -> CliResult<Vec<u8>> {
    let base = warm_probe_bytes(modality)?;
    let repetitions = match modality {
        Modality::Text | Modality::Code | Modality::Structured | Modality::Mixed => 512,
        Modality::Protein | Modality::Dna => 256,
        _ => 1,
    };
    let mut bytes = Vec::with_capacity(base.len().saturating_mul(repetitions));
    for _ in 0..repetitions {
        bytes.extend_from_slice(&base);
        if matches!(
            modality,
            Modality::Text | Modality::Code | Modality::Structured | Modality::Mixed
        ) {
            bytes.push(b' ');
        }
    }
    Ok(bytes)
}

fn capacity_cause(service: &ResidentService, modality: Modality, error: CliError) -> CliError {
    capacity_error(
        service,
        Some(modality),
        &format!(
            "cause_code={} cause={} cause_remediation={}",
            error.code(),
            error.message(),
            error.remediation()
        ),
    )
}

fn capacity_error(service: &ResidentService, modality: Option<Modality>, detail: &str) -> CliError {
    CliError::from(CalyxError {
        code: CAPACITY_PROBE_FAILED,
        message: format!(
            "resident capacity probe failed template={} modality={modality:?} max_runtime_batch={}: {detail}",
            service.state.template_source, service.max_runtime_batch
        ),
        remediation: "increase the configured runtime arena/device budget or lower --max-runtime-batch, then restart; readiness is withheld until the real full-window probe succeeds",
    })
}

#[cfg(test)]
mod tests;
