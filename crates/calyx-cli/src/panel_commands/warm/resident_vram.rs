use super::load_progress::is_content_slot;
use super::*;
use calyx_core::MeasurementGroupKey;
use std::collections::BTreeMap;

pub(super) fn vault_resident_vram_preflight(
    selector: &str,
    panel: &Panel,
    registry: &Registry,
    max_resident_vram_mib: u64,
    resident_overhead_multiplier_milli: u64,
    progress_log: Option<&WarmProgressLog>,
) -> CliResult<WarmPreflight> {
    let content_lenses = panel
        .slots
        .iter()
        .filter(|slot| is_content_slot(slot))
        .count();
    let declared_bytes = grouped_resident_vram_bytes(panel, registry)?;
    let declared_template_vram_mib = bytes_to_mib_ceil(declared_bytes);
    let estimated_resident_vram_mib =
        estimate_resident_vram_mib(declared_bytes, resident_overhead_multiplier_milli);
    append_preflight_progress(
        progress_log,
        selector,
        content_lenses,
        declared_template_vram_mib,
        estimated_resident_vram_mib,
        max_resident_vram_mib,
        resident_overhead_multiplier_milli,
        None,
    )?;
    if estimated_resident_vram_mib > max_resident_vram_mib {
        return vram_budget_error(
            progress_log,
            selector,
            content_lenses,
            declared_template_vram_mib,
            estimated_resident_vram_mib,
            max_resident_vram_mib,
            resident_overhead_multiplier_milli,
        );
    }
    Ok(WarmPreflight {
        lens_count: content_lenses,
        declared_template_vram_mib,
        estimated_resident_vram_mib,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_preflight_progress(
    progress_log: Option<&WarmProgressLog>,
    selector: &str,
    content_lenses: usize,
    declared_mib: u64,
    estimated_mib: u64,
    max_mib: u64,
    multiplier_milli: u64,
    error: Option<(&str, &str)>,
) -> CliResult {
    let Some(log) = progress_log else {
        return Ok(());
    };
    let mut record = run_progress_record(
        selector,
        if error.is_some() {
            "vram_preflight_error"
        } else {
            "vram_preflight"
        },
    );
    record.lens_count = Some(content_lenses);
    record.declared_template_vram_mib = Some(declared_mib);
    record.estimated_resident_vram_mib = Some(estimated_mib);
    record.max_resident_vram_mib = Some(max_mib);
    record.resident_overhead_multiplier_milli = Some(multiplier_milli);
    if let Some((code, message)) = error {
        record.error_code = Some(code.to_string());
        record.error_message = Some(message.to_string());
        record.remediation = Some(VRAM_REMEDIATION.to_string());
    }
    log.append(&record)
}

const VRAM_REMEDIATION: &str = "prune resident GPU runtimes or raise the explicit \
    --max-resident-vram-mib after verifying physical GPU headroom";

#[allow(clippy::too_many_arguments)]
fn vram_budget_error(
    progress_log: Option<&WarmProgressLog>,
    selector: &str,
    content_lenses: usize,
    declared_mib: u64,
    estimated_mib: u64,
    max_mib: u64,
    multiplier_milli: u64,
) -> CliResult<WarmPreflight> {
    let message = format!(
        "panel resident refuses vault {selector}: grouped_declared_vram_mib={declared_mib} \
         resident_overhead_multiplier={} estimated_resident_vram_mib={estimated_mib} \
         max_resident_vram_mib={max_mib} content_lens_count={content_lenses}; \
         grouped lens projections are counted once by their runtime measurement group",
        format_multiplier_milli(multiplier_milli),
    );
    append_preflight_progress(
        progress_log,
        selector,
        content_lenses,
        declared_mib,
        estimated_mib,
        max_mib,
        multiplier_milli,
        Some((WARM_VRAM_BUDGET, &message)),
    )?;
    Err(CliError::from(CalyxError {
        code: WARM_VRAM_BUDGET,
        message,
        remediation: VRAM_REMEDIATION,
    }))
}

fn grouped_resident_vram_bytes(panel: &Panel, registry: &Registry) -> CliResult<u64> {
    let costs = panel
        .slots
        .iter()
        .filter(|slot| is_content_slot(slot))
        .map(|slot| {
            registry
                .measurement_group_key(slot.lens_id)
                .map(|group| (group, slot.resource.cost.vram_bytes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    sum_grouped_vram_bytes(costs)
}

fn sum_grouped_vram_bytes(
    costs: impl IntoIterator<Item = (Option<MeasurementGroupKey>, u64)>,
) -> CliResult<u64> {
    let mut ungrouped_bytes = 0_u64;
    let mut grouped_bytes = BTreeMap::<MeasurementGroupKey, u64>::new();
    for (group, bytes) in costs {
        match group {
            Some(group) => {
                let current = grouped_bytes.entry(group).or_default();
                *current = (*current).max(bytes);
            }
            None => ungrouped_bytes = checked_vram_add(ungrouped_bytes, bytes)?,
        }
    }
    grouped_bytes
        .into_values()
        .try_fold(ungrouped_bytes, checked_vram_add)
}

fn checked_vram_add(total: u64, bytes: u64) -> CliResult<u64> {
    total.checked_add(bytes).ok_or_else(|| {
        CliError::from(CalyxError {
            code: WARM_VRAM_BUDGET,
            message: format!(
                "resident declared VRAM byte sum overflowed u64: accumulated_bytes={total} next_runtime_bytes={bytes}"
            ),
            remediation: "repair the persisted slot resource costs; declared VRAM costs must sum exactly within u64",
        })
    })
}

#[cfg(test)]
mod tests;
