//! Hybrid query measurement for GPU panels: Gpu-placed text slots are measured
//! through the resident service (explicit address or auto-discovered via the
//! `<CALYX_HOME>/resident/discovery.json` record the server writes), Cpu-placed
//! text slots are measured locally - the same split the ingest measure path
//! uses. Shared by the CLI and calyx-mcp so search has ONE resident route.

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::Path;
use std::str::FromStr;

use calyx_core::{
    CalyxError, Input, LensId, Modality, Placement, Slot, SlotId, SlotState, SlotVector,
};
use calyx_registry::VaultPanelState;
use calyx_registry::resident::{measure_batch_at, read_resident_discovery, ready_value_at};

use crate::engine_measure::{indexable, measure_query_vectors_with_slots};
use crate::error::CliResult;

/// Derive the resident-service identity string for a vault path - must match
/// the server's `template_source` (`vault:<canonical-path>` with forward
/// slashes and no Windows extended prefix).
pub fn vault_template_source(path: &Path) -> CliResult<String> {
    let canonical = path.canonicalize().map_err(|error| CalyxError {
        code: "CALYX_SEARCH_RESIDENT_MISMATCH",
        message: format!("canonicalize vault path {}: {error}", path.display()),
        remediation: "pass an existing vault directory",
    })?;
    let raw = canonical.to_str().ok_or_else(|| CalyxError {
        code: "CALYX_SEARCH_RESIDENT_MISMATCH",
        message: format!("canonical path {} is not valid UTF-8", canonical.display()),
        remediation: "use a UTF-8 vault path",
    })?;
    let raw = raw.strip_prefix(r"\\?\").unwrap_or(raw);
    Ok(format!("vault:{}", raw.replace('\\', "/")))
}

/// Measure the query for every active text slot: Gpu-placed slots via the
/// resident service, Cpu-placed slots locally. When GPU slots exist and no
/// resident route is available (explicitly or by discovery), fails closed with
/// CALYX_SEARCH_RESIDENT_REQUIRED - never cold-loads GPU models in-process.
pub fn measure_query_vectors_resident_hybrid(
    state: &VaultPanelState,
    home: &Path,
    vault_path: &Path,
    query: &str,
    explicit_addr: Option<SocketAddr>,
) -> CliResult<Vec<(SlotId, SlotVector)>> {
    let mut gpu_slots: Vec<&Slot> = Vec::new();
    let mut cpu_slots: BTreeSet<SlotId> = BTreeSet::new();
    for slot in active_registered_text_slots(state) {
        if slot.resource.placement == Placement::Gpu {
            gpu_slots.push(slot);
        } else {
            cpu_slots.insert(slot.slot_id);
        }
    }
    let mut out = Vec::new();
    if !gpu_slots.is_empty() {
        let expected_template_source = vault_template_source(vault_path)?;
        let addr = match explicit_addr {
            Some(addr) => addr,
            None => discover_resident_addr(home, &expected_template_source, &gpu_slots)?,
        };
        out.extend(measure_gpu_slots_via_resident(
            state,
            &gpu_slots,
            &expected_template_source,
            query,
            addr,
        )?);
    }
    if !cpu_slots.is_empty() {
        out.extend(measure_query_vectors_with_slots(
            state,
            query,
            Some(&cpu_slots),
        )?);
    }
    Ok(out)
}

/// Resolve a live resident route from the discovery file, validating readiness
/// and vault identity. Every anomaly resolves to the fail-closed
/// RESIDENT_REQUIRED error carrying the reason.
fn discover_resident_addr(
    home: &Path,
    expected_template_source: &str,
    gpu_slots: &[&Slot],
) -> CliResult<SocketAddr> {
    let reason = match read_resident_discovery(home)? {
        Ok(discovery) => match ready_value_at(discovery.bind) {
            Ok(value) => {
                let ready = value.get("ready").and_then(|v| v.as_bool()).unwrap_or(false);
                let template_source = value
                    .get("template_source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if ready && template_source == expected_template_source {
                    return Ok(discovery.bind);
                }
                if !ready {
                    "resident_not_ready"
                } else {
                    "resident_template_source_mismatch"
                }
            }
            Err(_) => "resident_not_reachable",
        },
        Err(reason) => reason,
    };
    Err(resident_required_error(gpu_slots, reason).into())
}

fn resident_required_error(gpu_slots: &[&Slot], reason: &'static str) -> CalyxError {
    let described = gpu_slots
        .iter()
        .map(|slot| {
            format!(
                "slot={} key={} lens={} placement={:?}",
                slot.slot_id.get(),
                slot.slot_key.key(),
                slot.lens_id,
                slot.resource.placement
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    CalyxError {
        code: "CALYX_SEARCH_RESIDENT_REQUIRED",
        message: format!(
            "search refuses cold local query measurement for {} active GPU text lens(es) \
             (resident route: {reason}): {described}",
            gpu_slots.len()
        ),
        remediation: "start `calyx panel resident serve --vault <vault-path>` for this vault; \
                      search auto-discovers it via <CALYX_HOME>/resident/discovery.json",
    }
}

fn measure_gpu_slots_via_resident(
    state: &VaultPanelState,
    gpu_slots: &[&Slot],
    expected_template_source: &str,
    query: &str,
    addr: SocketAddr,
) -> CliResult<Vec<(SlotId, SlotVector)>> {
    let input = Input::new(Modality::Text, query.as_bytes().to_vec());
    let resident =
        measure_batch_at(addr, Modality::Text, &[input], None).map_err(|error| CalyxError {
            code: "CALYX_PANEL_RESIDENT_UNAVAILABLE",
            message: format!(
                "resident search measurement failed addr={addr} code={} message={}",
                error.code, error.message
            ),
            remediation:
                "start `calyx panel resident serve --vault <vault-path>` for the matching \
                 active vault, then retry",
        })?;
    let response = resident.response;
    if !response.ready {
        return Err(CalyxError::lens_unreachable(format!(
            "resident service {addr} returned ready=false for search measurement"
        ))
        .into());
    }
    if response.template_source != expected_template_source {
        return Err(CalyxError {
            code: "CALYX_SEARCH_RESIDENT_MISMATCH",
            message: format!(
                "resident service {addr} served template_source {}, expected {}",
                response.template_source, expected_template_source
            ),
            remediation: "restart the resident service with `calyx panel resident serve \
                          --vault <this-vault-path>` and retry",
        }
        .into());
    }
    if response.modality != Modality::Text || response.input_count != 1 || response.rows.len() != 1
    {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "resident search response mismatch: modality {:?} input_count {} rows {}, \
             expected Text/1/1",
            response.modality,
            response.input_count,
            response.rows.len()
        ))
        .into());
    }
    let row = &response.rows[0];
    if row.input_index != 0 {
        return Err(CalyxError::lens_dim_mismatch(format!(
            "resident search response row index {}, expected 0",
            row.input_index
        ))
        .into());
    }
    let mut out = Vec::new();
    for slot in gpu_slots {
        let returned = row
            .slots
            .iter()
            .find(|returned| {
                returned.measured
                    && returned.slot == slot.slot_id.get()
                    && LensId::from_str(&returned.lens_id)
                        .map(|lens_id| lens_id == slot.lens_id)
                        .unwrap_or(false)
            })
            .ok_or_else(|| {
                CalyxError::lens_unreachable(format!(
                    "resident service {addr} did not return active text slot {} lens {}",
                    slot.slot_id.get(),
                    slot.lens_id
                ))
            })?;
        if returned.modality != Modality::Text || returned.placement != slot.resource.placement {
            return Err(CalyxError::lens_unreachable(format!(
                "resident service {addr} returned slot {} lens {} with modality {:?} \
                 placement {:?}, expected Text/{:?}",
                slot.slot_id.get(),
                slot.lens_id,
                returned.modality,
                returned.placement,
                slot.resource.placement
            ))
            .into());
        }
        let vector = returned.vector.clone().ok_or_else(|| {
            CalyxError::lens_unreachable(format!(
                "resident service {addr} measured slot {} lens {} without a vector",
                slot.slot_id.get(),
                slot.lens_id
            ))
        })?;
        if !indexable(&vector) {
            return Err(CalyxError::stale_derived(format!(
                "resident service {addr} returned non-indexable query vector for slot {} lens {}",
                slot.slot_id.get(),
                slot.lens_id
            ))
            .into());
        }
        let snapshot = registry_snapshot_for_lens(state, slot.lens_id).ok_or_else(|| {
            CalyxError::lens_unreachable(format!(
                "resident search requires persisted registry snapshot contract for lens {}",
                slot.lens_id
            ))
        })?;
        snapshot.contract.verify_vector(slot.lens_id, &vector)?;
        out.push((slot.slot_id, vector));
    }
    Ok(out)
}

fn active_registered_text_slots(state: &VaultPanelState) -> impl Iterator<Item = &Slot> {
    state.panel.slots.iter().filter(|slot| {
        slot.state == SlotState::Active
            && slot.modality == Modality::Text
            && state.registry.contains(slot.lens_id)
    })
}

fn registry_snapshot_for_lens(
    state: &VaultPanelState,
    lens_id: LensId,
) -> Option<&calyx_registry::RegistryLensSnapshot> {
    state
        .registry_snapshot
        .as_ref()?
        .lenses
        .iter()
        .find(|snapshot| snapshot.lens_id == lens_id)
}
