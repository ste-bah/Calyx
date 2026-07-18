//! GPU deployment-policy admission for panel templates (#1490).
//!
//! Panels are GPU-served by design: the resident service refuses to warm
//! CPU/non-GPU content lenses and search demands its GPU roster from the
//! resident. A CPU-placed content lens must therefore never enter a panel
//! silently — it is refused at template save AND at swap-into-vault unless
//! the operator opts in explicitly (and loudly) by naming the lens in
//! `CALYX_PANEL_ALLOW_CPU_LENS`. Opted-in CPU lenses are excluded from the
//! resident's warm roster and measured in-process at ingest/search time.

use std::collections::BTreeSet;
use std::env;

use calyx_core::Placement;

use super::{SavedPanelTemplate, template_error};
use crate::error::CliResult;

pub(in crate::panel_commands) const TEMPLATE_CPU_LENS_REFUSED: &str =
    "CALYX_PANEL_TEMPLATE_CPU_LENS_REFUSED";

/// Comma-separated list of lens names the operator explicitly allows to enter
/// a panel despite not being GPU-servable.
pub(in crate::panel_commands) const ALLOW_CPU_LENS_ENV: &str = "CALYX_PANEL_ALLOW_CPU_LENS";

/// Runtime kinds that can never be GPU-served regardless of what the catalog
/// placement claims (a static lookup table has no GPU execution path).
const CPU_ONLY_RUNTIMES: &[&str] = &["static_lookup"];

/// Enforce the GPU deployment policy on a template at an admission boundary
/// (`action` is "save" or "swap"), reading the opt-in from the environment.
pub(in crate::panel_commands) fn require_gpu_lens_admission(
    template: &SavedPanelTemplate,
    action: &str,
) -> CliResult {
    let allow = parse_allow_list(env::var(ALLOW_CPU_LENS_ENV).ok().as_deref());
    require_gpu_lens_admission_with_allow_list(template, action, &allow)
}

pub(in crate::panel_commands) fn parse_allow_list(raw: Option<&str>) -> BTreeSet<String> {
    raw.map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

pub(in crate::panel_commands) fn require_gpu_lens_admission_with_allow_list(
    template: &SavedPanelTemplate,
    action: &str,
    allow: &BTreeSet<String>,
) -> CliResult {
    let mut refused = Vec::new();
    for lens in &template.lenses {
        let cpu_only_runtime = CPU_ONLY_RUNTIMES.contains(&lens.runtime.as_str());
        if lens.placement == Placement::Gpu && cpu_only_runtime {
            // A CPU-only runtime claiming GPU placement is a lying manifest,
            // not an opt-in candidate: it would pass admission and then wedge
            // the resident at warm time.
            return Err(template_error(
                TEMPLATE_CPU_LENS_REFUSED,
                format!(
                    "panel template {} {action} refused: lens {} (slot_key {}) declares placement Gpu but runtime {} is CPU-only and can never be GPU-served",
                    template.name, lens.lens_name, lens.slot_key, lens.runtime
                ),
                "fix the lens catalog placement to Cpu for this runtime, then opt in explicitly with CALYX_PANEL_ALLOW_CPU_LENS=<lens-name> or drop the lens from the template",
            ));
        }
        if lens.placement == Placement::Gpu {
            continue;
        }
        if allow.contains(&lens.lens_name) {
            eprintln!(
                "CALYX_PANEL_CPU_LENS_OPT_IN action={action} template={} lens={} slot_key={} runtime={} placement={:?} allowed_by={ALLOW_CPU_LENS_ENV}",
                template.name, lens.lens_name, lens.slot_key, lens.runtime, lens.placement
            );
            continue;
        }
        refused.push(format!(
            "{}:{}:{}:{:?}",
            lens.slot_key, lens.lens_name, lens.runtime, lens.placement
        ));
    }
    if refused.is_empty() {
        return Ok(());
    }
    Err(template_error(
        TEMPLATE_CPU_LENS_REFUSED,
        format!(
            "panel template {} {action} refused under the GPU deployment policy: {} content lens(es) cannot be GPU-served: {}",
            template.name,
            refused.len(),
            refused.join(", ")
        ),
        "drop the CPU-placed lens(es) from the template, or opt in explicitly and loudly with CALYX_PANEL_ALLOW_CPU_LENS=<lens-name>[,<lens-name>]; opted-in CPU lenses are excluded from the GPU resident and measured in-process at ingest/search time",
    ))
}

#[cfg(test)]
mod tests;
