use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::Arc;

use calyx_core::{
    AbsentReason, CalyxError, Input, Lens, LensId, Modality, Placement, Result as CalyxResult,
    SlotShape, SlotVector,
};
use calyx_registry::{FrozenLensContract, LensSpec, Registry, lens_spec_with_frozen_contract};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{SavedPanelTemplate, id_for_loaded};
use crate::error::{CliError, CliResult};
use crate::panel_commands::resident::{
    MeasureBatchResponse, ResidentMeasuredInput, ResidentSlotMeasure, measure_batch_at,
    ready_value_at,
};
use crate::panel_commands::warm::warm_probe_bytes;

const ATTESTATION_SCHEMA: &str = "calyx-panel-resident-swap-attestation-v1";
const ATTESTATION_ERROR: &str = "CALYX_PANEL_RESIDENT_ATTESTATION_FAILED";

#[derive(Clone, Debug, Serialize)]
pub(in crate::panel_commands) struct ResidentSwapAttestation {
    schema: &'static str,
    addr: SocketAddr,
    process_id: u32,
    template_source: String,
    ready_source_of_truth: String,
    probe_repetitions: usize,
    modalities: Vec<Modality>,
    slot_count: usize,
    output_sha256_by_slot: BTreeMap<u16, String>,
}

pub(super) struct AttestedSlot {
    slot: u16,
    key: String,
    pub(super) contract: FrozenLensContract,
    pub(super) spec: LensSpec,
    placement: Placement,
}

#[derive(Clone, Debug)]
struct ExpectedResidentSlot {
    slot: u16,
    key: String,
    lens_id: String,
    modality: Modality,
    placement: Placement,
    role: ExpectedResidentSlotRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedResidentSlotRole {
    Content,
    NonContent,
}

pub(in crate::panel_commands) fn register_template_lenses_from_resident(
    registry: &mut Registry,
    template: &mut SavedPanelTemplate,
    addr: SocketAddr,
) -> CliResult<(usize, ResidentSwapAttestation)> {
    if !addr.ip().is_loopback() {
        return Err(attestation_error(format!(
            "resident-backed template swap requires a loopback address, got {addr}"
        )));
    }
    let template_id = id_for_loaded(template)?;
    let expected = expected_slots(template, &template_id)?;
    let expected_resident = expected_resident_slots(template, &expected)?;
    let expected_template_source = format!("saved:{}:{template_id}", template.name);
    let (attestation, attested_vectors) = attest(
        addr,
        &expected,
        &expected_resident,
        &expected_template_source,
    )?;

    let mut staged_registry = registry.clone();
    let mut staged_template = template.clone();
    let mut added = 0;
    for (index, lens) in staged_template.lenses.iter_mut().enumerate() {
        let expected_slot = &expected[index];
        let expected_runtime_id = expected_slot.contract.lens_id();
        if let Some(recorded) = lens.runtime_lens_id
            && recorded != expected_runtime_id
        {
            return Err(attestation_error(format!(
                "template {template_id} lens {} records runtime id {recorded}, but its verified frozen contract resolves to {expected_runtime_id}",
                lens.lens_name
            )));
        }
        if let Some(existing) = staged_registry.find_lens_by_spec_id(expected_slot.spec.lens_id()) {
            if existing != expected_runtime_id
                || staged_registry.lens_spec(existing) != Some(&expected_slot.spec)
            {
                return Err(attestation_error(format!(
                    "vault registry entry for template lens {} differs from the verified resident-attested spec/contract",
                    lens.lens_name
                )));
            }
            lens.runtime_lens_id = Some(existing);
            continue;
        }
        let proxy = ResidentProxyLens {
            addr,
            slot: expected_slot.slot,
            key: expected_slot.key.clone(),
            contract: expected_slot.contract.clone(),
            placement: expected_slot.placement,
        };
        let registered = staged_registry.register_frozen_arc_with_spec(
            Arc::new(proxy),
            expected_slot.contract.clone(),
            expected_slot.spec.clone(),
        )?;
        if registered != expected_runtime_id {
            return Err(attestation_error(format!(
                "resident-attested registration returned {registered}, expected {expected_runtime_id}"
            )));
        }
        let vector = attested_vectors.get(&expected_slot.slot).ok_or_else(|| {
            attestation_error(format!(
                "resident attestation lost slot {} before registry publication",
                expected_slot.slot
            ))
        })?;
        expected_slot
            .contract
            .verify_vector(registered, vector)
            .map_err(|error| attestation_core_error("verify attested vector", error))?;
        lens.runtime_lens_id = Some(registered);
        added += 1;
    }
    *registry = staged_registry;
    *template = staged_template;
    Ok((added, attestation))
}

fn expected_resident_slots(
    template: &SavedPanelTemplate,
    expected_content: &[AttestedSlot],
) -> CliResult<Vec<ExpectedResidentSlot>> {
    let expected_total = expected_content
        .len()
        .checked_add(template.time_controls.len())
        .ok_or_else(|| attestation_error("template total slot count overflow"))?;
    if expected_total > usize::from(u16::MAX) + 1 {
        return Err(attestation_error(format!(
            "template has {expected_total} total slots, exceeding the 65536-slot u16 identity space"
        )));
    }

    let mut slots = expected_content
        .iter()
        .map(|slot| ExpectedResidentSlot {
            slot: slot.slot,
            key: slot.key.clone(),
            lens_id: slot.contract.lens_id().to_string(),
            modality: slot.spec.modality,
            placement: slot.placement,
            role: ExpectedResidentSlotRole::Content,
        })
        .collect::<Vec<_>>();

    // The target panel is the canonical source of the deterministic temporal
    // sidecar identities. Deriving them here from the same immutable template
    // prevents resident attestation from accepting merely the right count of
    // unknown or reordered non-content slots.
    let target = template.to_target_panel(0);
    if target.slots.len() != expected_total {
        return Err(attestation_error(format!(
            "template target panel produced {} slots, expected {expected_total}",
            target.slots.len()
        )));
    }
    for target_slot in target.slots.iter().skip(expected_content.len()) {
        slots.push(ExpectedResidentSlot {
            slot: target_slot.slot_id.get(),
            key: target_slot.slot_key.key().to_string(),
            lens_id: target_slot.lens_id.to_string(),
            modality: target_slot.modality,
            placement: target_slot.resource.placement,
            role: ExpectedResidentSlotRole::NonContent,
        });
    }
    let unique = slots.iter().map(|slot| slot.slot).collect::<BTreeSet<_>>();
    if unique.len() != expected_total {
        return Err(attestation_error(format!(
            "template target panel contains {} unique slot ids across {expected_total} slots",
            unique.len()
        )));
    }
    Ok(slots)
}

pub(super) fn expected_slots(
    template: &SavedPanelTemplate,
    template_id: &str,
) -> CliResult<Vec<AttestedSlot>> {
    template
        .lenses
        .iter()
        .enumerate()
        .map(|(index, lens)| {
            let slot = u16::try_from(index)
                .map_err(|_| attestation_error("template content slot count exceeds u16"))?;
            let verified_spec = lens.verified_materialization_spec(template_id)?;
            let contract = lens.expected_runtime_contract().cloned().ok_or_else(|| {
                attestation_error(format!(
                    "template {template_id} lens {} has no frozen runtime contract",
                    lens.lens_name
                ))
            })?;
            // Saved artifacts prove where the immutable model came from. The frozen
            // runtime contract proves the exact identity that the commissioned
            // runtime produced. Registry identity must use that same canonical spec,
            // just like the local materialization path, without loading the model in
            // this process.
            let spec = lens_spec_with_frozen_contract(verified_spec, &contract);
            Ok(AttestedSlot {
                slot,
                key: lens.slot_key.clone(),
                contract,
                spec,
                placement: lens.placement,
            })
        })
        .collect()
}

fn attest(
    addr: SocketAddr,
    expected: &[AttestedSlot],
    expected_resident: &[ExpectedResidentSlot],
    expected_template_source: &str,
) -> CliResult<(ResidentSwapAttestation, BTreeMap<u16, SlotVector>)> {
    let ready = ready_value_at(addr)?;
    require_json_bool(&ready, "ready", true)?;
    require_json_str(&ready, "schema", "calyx-panel-resident-readiness-v2")?;
    validate_readiness_identity(
        &ready,
        expected.len(),
        expected_resident.len(),
        expected_template_source,
    )?;
    let ready_process = require_json_u64(&ready, "process_id")? as u32;
    let ready_source = require_json_string(&ready, "source_of_truth")?;
    let ready_template_source = require_json_string(&ready, "template_source")?;

    let mut modalities = expected
        .iter()
        .map(|slot| slot.spec.modality)
        .collect::<Vec<_>>();
    modalities.sort_by_key(|modality| format!("{modality:?}"));
    modalities.dedup();
    let mut vectors = BTreeMap::new();
    let mut output_hashes = BTreeMap::new();
    for modality in &modalities {
        let probe = warm_probe_bytes(*modality)?;
        let inputs = [
            Input::new(*modality, probe.clone()),
            Input::new(*modality, probe),
        ];
        let response = measure_batch_at(addr, *modality, &inputs, None)?.response;
        validate_response_identity(&response, ready_process, &ready_template_source, *modality)?;
        let expected_modality = expected
            .iter()
            .filter(|slot| slot.spec.modality == *modality)
            .collect::<Vec<_>>();
        let first = validate_row(&response.rows[0], expected_resident, *modality)?;
        let second = validate_row(&response.rows[1], expected_resident, *modality)?;
        for slot in expected_modality {
            let left = first.get(&slot.slot).ok_or_else(|| {
                attestation_error(format!("first probe omitted slot {}", slot.slot))
            })?;
            let right = second.get(&slot.slot).ok_or_else(|| {
                attestation_error(format!("second probe omitted slot {}", slot.slot))
            })?;
            let left_bytes = serde_json::to_vec(left).map_err(|error| {
                attestation_error(format!("serialize slot {} probe: {error}", slot.slot))
            })?;
            let right_bytes = serde_json::to_vec(right).map_err(|error| {
                attestation_error(format!("serialize slot {} repeat: {error}", slot.slot))
            })?;
            if left_bytes != right_bytes {
                return Err(attestation_error(format!(
                    "resident slot {} produced non-deterministic bytes for two identical live probes",
                    slot.slot
                )));
            }
            slot.contract
                .verify_vector(slot.contract.lens_id(), left)
                .map_err(|error| attestation_core_error("verify resident vector", error))?;
            output_hashes.insert(slot.slot, sha256_hex(&left_bytes));
            vectors.insert(slot.slot, left.clone());
        }
    }
    if vectors.len() != expected.len() {
        return Err(attestation_error(format!(
            "resident attested {} unique slots, expected {}",
            vectors.len(),
            expected.len()
        )));
    }
    Ok((
        ResidentSwapAttestation {
            schema: ATTESTATION_SCHEMA,
            addr,
            process_id: ready_process,
            template_source: ready_template_source,
            ready_source_of_truth: ready_source,
            probe_repetitions: 2,
            modalities,
            slot_count: vectors.len(),
            output_sha256_by_slot: output_hashes,
        },
        vectors,
    ))
}

fn validate_readiness_identity(
    ready: &serde_json::Value,
    expected_content_lenses: usize,
    expected_total_slots: usize,
    expected_template_source: &str,
) -> CliResult {
    let actual_total_slots = require_json_u64(ready, "slot_count")? as usize;
    let actual_content_lenses = require_json_u64(ready, "content_lens_count")? as usize;
    let actual_registry_lenses = require_json_u64(ready, "registry_lens_count")? as usize;
    let actual_warmed_lenses = require_json_u64(ready, "warmed_lens_count")? as usize;
    let actual_gpu_content_lenses = require_json_u64(ready, "gpu_content_lens_count")? as usize;
    let actual_cpu_content_lenses = require_json_u64(ready, "cpu_content_lens_count")? as usize;
    let actual_template_source = require_json_string(ready, "template_source")?;

    if actual_total_slots != expected_total_slots
        || actual_content_lenses != expected_content_lenses
        || actual_registry_lenses != expected_content_lenses
        || actual_warmed_lenses != expected_content_lenses
        || actual_gpu_content_lenses != expected_content_lenses
        || actual_cpu_content_lenses != 0
        || actual_template_source != expected_template_source
    {
        return Err(attestation_error(format!(
            "resident readiness identity/cardinality mismatch: total_slots={actual_total_slots} \
             expected_total_slots={expected_total_slots} content_lenses={actual_content_lenses} \
             expected_content_lenses={expected_content_lenses} registry_lenses={actual_registry_lenses} \
             warmed_lenses={actual_warmed_lenses} gpu_content_lenses={actual_gpu_content_lenses} \
             cpu_content_lenses={actual_cpu_content_lenses} template_source={actual_template_source:?} \
             expected_template_source={expected_template_source:?}"
        )));
    }
    Ok(())
}

fn validate_response_identity(
    response: &MeasureBatchResponse,
    process_id: u32,
    template_source: &str,
    modality: Modality,
) -> CliResult {
    if !response.ready
        || response.process_id != process_id
        || response.template_source != template_source
        || response.modality != modality
        || response.input_count != 2
        || response.rows.len() != 2
    {
        return Err(attestation_error(format!(
            "resident live-probe identity/count mismatch for {modality:?}: process={} source={} input_count={} rows={} ready={}",
            response.process_id,
            response.template_source,
            response.input_count,
            response.rows.len(),
            response.ready
        )));
    }
    Ok(())
}

fn validate_row(
    row: &ResidentMeasuredInput,
    expected: &[ExpectedResidentSlot],
    probe_modality: Modality,
) -> CliResult<BTreeMap<u16, SlotVector>> {
    let expected_measured = expected
        .iter()
        .filter(|slot| {
            slot.role == ExpectedResidentSlotRole::Content && slot.modality == probe_modality
        })
        .count();
    let expected_absent = expected.len().saturating_sub(expected_measured);
    if row.measured_slot_count != expected_measured
        || row.absent_slot_count != expected_absent
        || row.slots.len() != expected.len()
    {
        return Err(attestation_error(format!(
            "resident probe row {} for {probe_modality:?} reports measured={} absent={} slots={}; expected measured={expected_measured} absent={expected_absent} total={}",
            row.input_index,
            row.measured_slot_count,
            row.absent_slot_count,
            row.slots.len(),
            expected.len()
        )));
    }
    let expected_by_slot = expected
        .iter()
        .map(|slot| (slot.slot, slot))
        .collect::<BTreeMap<_, _>>();
    let mut vectors = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for actual in &row.slots {
        let target = expected_by_slot.get(&actual.slot).ok_or_else(|| {
            attestation_error(format!(
                "resident returned unexpected slot {} for probe row {}",
                actual.slot, row.input_index
            ))
        })?;
        if !seen.insert(actual.slot) {
            return Err(attestation_error(format!(
                "resident returned duplicate slot {}",
                actual.slot
            )));
        }
        let should_measure =
            target.role == ExpectedResidentSlotRole::Content && target.modality == probe_modality;
        if should_measure {
            validate_measured_slot(actual, target)?;
            let vector = actual.vector.as_ref().ok_or_else(|| {
                attestation_error(format!("resident slot {} has no live vector", actual.slot))
            })?;
            vectors.insert(actual.slot, vector.clone());
        } else {
            let expected_reason = if target.modality == probe_modality {
                AbsentReason::LensUnavailable
            } else {
                AbsentReason::NotApplicable
            };
            validate_absent_slot(actual, target, &expected_reason)?;
        }
    }
    if seen.len() != expected.len() || vectors.len() != expected_measured {
        return Err(attestation_error(format!(
            "resident probe row {} validated {} unique slots and {} vectors; expected {} slots and {expected_measured} vectors",
            row.input_index,
            seen.len(),
            vectors.len(),
            expected.len()
        )));
    }
    Ok(vectors)
}

fn validate_measured_slot(
    actual: &ResidentSlotMeasure,
    expected: &ExpectedResidentSlot,
) -> CliResult {
    if actual.key != expected.key
        || actual.lens_id != expected.lens_id
        || actual.modality != expected.modality
        || actual.placement != expected.placement
        || !actual.measured
        || actual.absent_reason.is_some()
    {
        return Err(attestation_error(format!(
            "resident slot {} identity mismatch: key={} lens_id={} modality={:?} placement={:?} measured={} absent={:?}; expected key={} lens_id={} modality={:?} placement={:?}",
            actual.slot,
            actual.key,
            actual.lens_id,
            actual.modality,
            actual.placement,
            actual.measured,
            actual.absent_reason,
            expected.key,
            expected.lens_id,
            expected.modality,
            expected.placement
        )));
    }
    Ok(())
}

fn validate_absent_slot(
    actual: &ResidentSlotMeasure,
    expected: &ExpectedResidentSlot,
    expected_reason: &AbsentReason,
) -> CliResult {
    if actual.key != expected.key
        || actual.lens_id != expected.lens_id
        || actual.modality != expected.modality
        || actual.placement != expected.placement
        || actual.measured
        || actual.vector.is_some()
        || actual.absent_reason.as_ref() != Some(expected_reason)
    {
        return Err(attestation_error(format!(
            "resident absent slot {} identity/state mismatch: key={} lens_id={} modality={:?} placement={:?} measured={} vector_present={} absent={:?}; expected key={} lens_id={} modality={:?} placement={:?} measured=false vector_present=false absent={expected_reason:?}",
            actual.slot,
            actual.key,
            actual.lens_id,
            actual.modality,
            actual.placement,
            actual.measured,
            actual.vector.is_some(),
            actual.absent_reason,
            expected.key,
            expected.lens_id,
            expected.modality,
            expected.placement
        )));
    }
    Ok(())
}

struct ResidentProxyLens {
    addr: SocketAddr,
    slot: u16,
    key: String,
    contract: FrozenLensContract,
    placement: Placement,
}

impl Lens for ResidentProxyLens {
    fn id(&self) -> LensId {
        self.contract.lens_id()
    }

    fn shape(&self) -> SlotShape {
        self.contract.shape()
    }

    fn modality(&self) -> Modality {
        self.contract.modality()
    }

    fn measure(&self, input: &Input) -> CalyxResult<SlotVector> {
        let mut rows = self.measure_batch(std::slice::from_ref(input))?;
        rows.pop().ok_or_else(|| {
            CalyxError::lens_unreachable("resident proxy returned no measurement row")
        })
    }

    fn measure_batch(&self, inputs: &[Input]) -> CalyxResult<Vec<SlotVector>> {
        let measured = measure_batch_at(self.addr, self.modality(), inputs, None)
            .map_err(|error| proxy_error(self.addr, &error))?
            .response;
        if !measured.ready || measured.rows.len() != inputs.len() {
            return Err(CalyxError::lens_unreachable(format!(
                "resident proxy {} returned ready={} rows={} for {} inputs",
                self.addr,
                measured.ready,
                measured.rows.len(),
                inputs.len()
            )));
        }
        measured
            .rows
            .iter()
            .map(|row| {
                let actual = row
                    .slots
                    .iter()
                    .find(|slot| slot.slot == self.slot)
                    .ok_or_else(|| {
                        CalyxError::lens_unreachable(format!(
                            "resident proxy {} omitted slot {}",
                            self.addr, self.slot
                        ))
                    })?;
                if actual.key != self.key
                    || actual.lens_id != self.id().to_string()
                    || actual.modality != self.modality()
                    || actual.placement != self.placement
                    || !actual.measured
                    || actual.absent_reason.is_some()
                {
                    return Err(CalyxError::lens_unreachable(format!(
                        "resident proxy {} slot {} identity changed during registration",
                        self.addr, self.slot
                    )));
                }
                let vector = actual.vector.clone().ok_or_else(|| {
                    CalyxError::lens_unreachable(format!(
                        "resident proxy {} slot {} returned no vector",
                        self.addr, self.slot
                    ))
                })?;
                self.contract.verify_vector(self.id(), &vector)?;
                Ok(vector)
            })
            .collect()
    }
}

fn proxy_error(addr: SocketAddr, error: &CliError) -> CalyxError {
    CalyxError::lens_unreachable(format!(
        "resident proxy {addr} failed: {}: {} (remediation: {})",
        error.code(),
        error.message(),
        error.remediation()
    ))
}

fn require_json_bool(value: &serde_json::Value, field: &str, expected: bool) -> CliResult {
    match value.get(field).and_then(serde_json::Value::as_bool) {
        Some(actual) if actual == expected => Ok(()),
        actual => Err(attestation_error(format!(
            "resident readiness field {field} is {actual:?}, expected {expected}"
        ))),
    }
}

fn require_json_str(value: &serde_json::Value, field: &str, expected: &str) -> CliResult {
    match value.get(field).and_then(serde_json::Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        actual => Err(attestation_error(format!(
            "resident readiness field {field} is {actual:?}, expected {expected:?}"
        ))),
    }
}

fn require_json_u64(value: &serde_json::Value, field: &str) -> CliResult<u64> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| attestation_error(format!("resident readiness lacks u64 field {field}")))
}

fn require_json_string(value: &serde_json::Value, field: &str) -> CliResult<String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| attestation_error(format!("resident readiness lacks string field {field}")))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn attestation_core_error(stage: &str, error: CalyxError) -> CliError {
    attestation_error(format!(
        "{stage} failed: {}: {} (remediation: {})",
        error.code, error.message, error.remediation
    ))
}

fn attestation_error(message: impl Into<String>) -> CliError {
    CliError::from(CalyxError {
        code: ATTESTATION_ERROR,
        message: message.into(),
        remediation: "start an exact healthy loopback resident for the desired frozen template and retry with the same --resident-addr",
    })
}

#[cfg(test)]
mod tests {
    use calyx_core::{AbsentReason, Modality, Placement, SlotVector};

    use super::{
        ATTESTATION_ERROR, ExpectedResidentSlot, ExpectedResidentSlotRole, ResidentMeasuredInput,
        ResidentSlotMeasure, validate_readiness_identity, validate_row,
    };

    fn legal_v1_live_readiness() -> serde_json::Value {
        serde_json::json!({
            "slot_count": 13,
            "content_lens_count": 10,
            "registry_lens_count": 10,
            "warmed_lens_count": 10,
            "gpu_content_lens_count": 10,
            "cpu_content_lens_count": 0,
            "template_source": "saved:legal-v1:91c7fe817bdb29b0404e606a9a04a029f177e40c8ce077074aba969016c5b17f"
        })
    }

    #[test]
    fn legal_v1_live_readiness_accepts_time_control_slots_separately() {
        validate_readiness_identity(
            &legal_v1_live_readiness(),
            10,
            13,
            "saved:legal-v1:91c7fe817bdb29b0404e606a9a04a029f177e40c8ce077074aba969016c5b17f",
        )
        .expect("the observed legal-v1 resident has 10 content and 13 total slots");
    }

    #[test]
    fn readiness_rejects_content_total_and_source_drift_with_typed_detail() {
        for (field, value) in [
            ("slot_count", serde_json::json!(12)),
            ("content_lens_count", serde_json::json!(9)),
            ("template_source", serde_json::json!("saved:legal-v1:stale")),
        ] {
            let mut readiness = legal_v1_live_readiness();
            readiness[field] = value;
            let error = validate_readiness_identity(
                &readiness,
                10,
                13,
                "saved:legal-v1:91c7fe817bdb29b0404e606a9a04a029f177e40c8ce077074aba969016c5b17f",
            )
            .expect_err("readiness identity drift must fail closed");
            assert_eq!(error.code(), ATTESTATION_ERROR);
            assert!(error.message().contains("total_slots="));
            assert!(error.message().contains("content_lenses="));
            assert!(error.message().contains("template_source="));
        }
    }

    #[test]
    fn legal_v1_live_row_accepts_ten_content_and_three_absent_time_controls() {
        let expected = legal_v1_expected_slots();
        let row = legal_v1_live_row();

        let vectors = validate_row(&row, &expected, Modality::Text)
            .expect("the observed legal-v1 row has ten measured content slots and three exact temporal absences");

        assert_eq!(vectors.len(), 10);
        assert_eq!(
            vectors.keys().copied().collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legal_v1_live_row_rejects_temporal_identity_state_and_reason_drift() {
        let mutations: [fn(&mut ResidentMeasuredInput); 3] = [
            |row: &mut ResidentMeasuredInput| row.slots[10].lens_id = "wrong-time-id".to_string(),
            |row: &mut ResidentMeasuredInput| {
                row.slots[11].vector = Some(SlotVector::Dense {
                    dim: 1,
                    data: vec![0.0],
                })
            },
            |row: &mut ResidentMeasuredInput| {
                row.slots[12].absent_reason = Some(AbsentReason::LensUnavailable)
            },
        ];
        for mutate in mutations {
            let mut row = legal_v1_live_row();
            mutate(&mut row);
            let error = validate_row(&row, &legal_v1_expected_slots(), Modality::Text)
                .expect_err("temporal sidecar drift must fail closed");
            assert_eq!(error.code(), ATTESTATION_ERROR);
            assert!(error.message().contains("absent slot"));
        }
    }

    #[test]
    fn row_validation_accounts_for_content_slots_from_other_modalities() {
        let expected = vec![
            expected_slot(0, "text", Modality::Text, ExpectedResidentSlotRole::Content),
            expected_slot(1, "code", Modality::Code, ExpectedResidentSlotRole::Content),
            expected_slot(
                2,
                "time",
                Modality::Structured,
                ExpectedResidentSlotRole::NonContent,
            ),
        ];
        let row = ResidentMeasuredInput {
            input_index: 0,
            input_len: 4,
            measured_slot_count: 1,
            absent_slot_count: 2,
            slots: vec![
                measured_slot(&expected[0]),
                absent_slot(&expected[1], AbsentReason::NotApplicable),
                absent_slot(&expected[2], AbsentReason::NotApplicable),
            ],
        };

        let vectors = validate_row(&row, &expected, Modality::Text)
            .expect("off-modality content and temporal slots must be exact explicit absences");
        assert_eq!(vectors.len(), 1);
        assert!(vectors.contains_key(&0));
    }

    fn legal_v1_expected_slots() -> Vec<ExpectedResidentSlot> {
        let mut slots = (0..10)
            .map(|slot| {
                expected_slot(
                    slot,
                    &format!("content-{slot}"),
                    Modality::Text,
                    ExpectedResidentSlotRole::Content,
                )
            })
            .collect::<Vec<_>>();
        for (slot, key) in [
            (10, "E2_recency"),
            (11, "E3_periodic"),
            (12, "E4_positional"),
        ] {
            slots.push(expected_slot(
                slot,
                key,
                Modality::Structured,
                ExpectedResidentSlotRole::NonContent,
            ));
        }
        slots
    }

    fn legal_v1_live_row() -> ResidentMeasuredInput {
        let expected = legal_v1_expected_slots();
        ResidentMeasuredInput {
            input_index: 0,
            input_len: 18,
            measured_slot_count: 10,
            absent_slot_count: 3,
            slots: expected
                .iter()
                .map(|slot| match slot.role {
                    ExpectedResidentSlotRole::Content => measured_slot(slot),
                    ExpectedResidentSlotRole::NonContent => {
                        absent_slot(slot, AbsentReason::NotApplicable)
                    }
                })
                .collect(),
        }
    }

    fn expected_slot(
        slot: u16,
        key: &str,
        modality: Modality,
        role: ExpectedResidentSlotRole,
    ) -> ExpectedResidentSlot {
        ExpectedResidentSlot {
            slot,
            key: key.to_string(),
            lens_id: format!("lens-{slot}"),
            modality,
            placement: match role {
                ExpectedResidentSlotRole::Content => Placement::Gpu,
                ExpectedResidentSlotRole::NonContent => Placement::Cpu,
            },
            role,
        }
    }

    fn measured_slot(expected: &ExpectedResidentSlot) -> ResidentSlotMeasure {
        ResidentSlotMeasure {
            slot: expected.slot,
            key: expected.key.clone(),
            lens_id: expected.lens_id.clone(),
            modality: expected.modality,
            placement: expected.placement,
            measured: true,
            vector: Some(SlotVector::Dense {
                dim: 1,
                data: vec![expected.slot as f32],
            }),
            absent_reason: None,
        }
    }

    fn absent_slot(expected: &ExpectedResidentSlot, reason: AbsentReason) -> ResidentSlotMeasure {
        ResidentSlotMeasure {
            slot: expected.slot,
            key: expected.key.clone(),
            lens_id: expected.lens_id.clone(),
            modality: expected.modality,
            placement: expected.placement,
            measured: false,
            vector: None,
            absent_reason: Some(reason),
        }
    }
}
