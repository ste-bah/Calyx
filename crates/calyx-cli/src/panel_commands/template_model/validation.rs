use std::collections::BTreeSet;

use calyx_core::CalyxError;

use crate::error::{CliError, CliResult};

use super::{
    A37_ADMISSION_VERSION, MIN_CONTENT_LENSES, OBJECT_VERSION, SavedPanelTemplate,
    TEMPLATE_A37_GATE_REFUSED, TEMPLATE_INVALID, TemplateA37Admission, id_for_loaded,
};

impl SavedPanelTemplate {
    pub(in crate::panel_commands) fn content_lens_count(&self) -> usize {
        self.lenses
            .iter()
            .filter(|lens| lens.counts_toward_a35)
            .count()
    }

    pub(in crate::panel_commands) fn validate(&self) -> CliResult {
        let template_id = id_for_loaded(self)?;
        self.validate_with_id(&template_id)
    }

    pub(in crate::panel_commands) fn validate_with_id(&self, template_id: &str) -> CliResult {
        if self.schema_version != OBJECT_VERSION {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "template {template_id} uses legacy object schema {}; current schema is {OBJECT_VERSION}",
                    self.schema_version
                ),
                "run `calyx panel template refresh --template <name-or-id> --home <dir>` to explicitly re-resolve and snapshot every lens",
            ));
        }
        if self.name.trim().is_empty() || self.name.contains(['/', '\\']) {
            return Err(template_error(
                TEMPLATE_INVALID,
                "panel template name must be non-empty and path-safe",
                "choose a stable template name such as text-deep",
            ));
        }
        if self.content_lens_count() < MIN_CONTENT_LENSES {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "panel template {} has {} content lenses; minimum is {MIN_CONTENT_LENSES}",
                    self.name,
                    self.content_lens_count()
                ),
                "add real frozen content lenses until the template has at least ten",
            ));
        }
        validate_lenses(self, template_id, true)?;
        validate_time_controls(self)
    }

    pub(in crate::panel_commands) fn validate_refresh_source(
        &self,
        template_id: &str,
    ) -> CliResult {
        if self.schema_version == OBJECT_VERSION {
            return self.validate_with_id(template_id);
        }
        if self.schema_version != 1 {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "template {template_id} has unsupported legacy object schema {}",
                    self.schema_version
                ),
                "migrate the template with a Calyx binary that supports that source schema",
            ));
        }
        if self.name.trim().is_empty() || self.name.contains(['/', '\\']) {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!("legacy template {template_id} has an unsafe name"),
                "repair the immutable source object from a verified backup before migration",
            ));
        }
        if self.content_lens_count() < MIN_CONTENT_LENSES {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "legacy template {template_id} has {} content lenses; minimum is {MIN_CONTENT_LENSES}",
                    self.content_lens_count()
                ),
                "repair the immutable source object from a verified backup before migration",
            ));
        }
        validate_lenses(self, template_id, false)?;
        validate_time_controls(self)
    }

    pub(in crate::panel_commands) fn a37_admission(&self) -> TemplateA37Admission {
        self.ensemble_card
            .as_ref()
            .map(|card| card.a37_admission.clone())
            .unwrap_or_default()
    }

    pub(in crate::panel_commands) fn a37_gate_eligible(&self) -> bool {
        self.a37_admission().gate_eligible
    }

    pub(in crate::panel_commands) fn require_a37_gate(&self) -> CliResult {
        let admission = self.a37_admission();
        if admission.gate_eligible {
            return Ok(());
        }
        Err(template_error(
            TEMPLATE_A37_GATE_REFUSED,
            format!(
                "template {} is not A37 gate eligible: {}",
                self.name, admission.verdict
            ),
            "profile the template with an Assay EnsembleCard whose A37 status is gate_passed",
        ))
    }
}

impl Default for TemplateA37Admission {
    fn default() -> Self {
        Self {
            schema_version: A37_ADMISSION_VERSION,
            source: "missing_assay_ensemble_card".to_string(),
            gate_eligible: false,
            status: "missing_a37_ensemble_card".to_string(),
            verdict: "A37 gate not evaluated; template has no Assay EnsembleCard".to_string(),
            content_lens_count: 0,
            temporal_sidecar_count: 0,
            temporal_counts_toward_content_floor: false,
            association_family_count: 0,
            n_eff: None,
            mean_pairwise_corr: None,
            mean_pairwise_nmi: None,
            sum_unique_pid_bits: None,
        }
    }
}

pub(in crate::panel_commands) fn template_error(
    code: &'static str,
    message: impl Into<String>,
    remediation: &'static str,
) -> CliError {
    CliError::from(CalyxError {
        code,
        message: message.into(),
        remediation,
    })
}

fn validate_lenses(
    template: &SavedPanelTemplate,
    template_id: &str,
    require_snapshot: bool,
) -> CliResult {
    let mut ids = BTreeSet::new();
    let mut runtime_ids = BTreeSet::new();
    for lens in &template.lenses {
        if !ids.insert(lens.lens_id) {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!("template {} repeats lens {}", template.name, lens.lens_id),
                "remove duplicate lens ids from the template",
            ));
        }
        if let Some(runtime_lens_id) = lens.runtime_lens_id
            && !runtime_ids.insert(runtime_lens_id)
        {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!(
                    "template {} repeats runtime lens {}",
                    template.name, runtime_lens_id
                ),
                "remove duplicate runtime lens ids from the template",
            ));
        }
        validate_weight_hash(&lens.weights_sha256)?;
        match &lens.immutable_snapshot {
            Some(snapshot) => snapshot.validate_summary(template_id, lens)?,
            None if require_snapshot => {
                return Err(template_error(
                    TEMPLATE_INVALID,
                    format!(
                        "template {template_id} lens {} is missing its immutable manifest/spec snapshot",
                        lens.lens_name
                    ),
                    "run `calyx panel template refresh --template <name-or-id> --home <dir>` to explicitly migrate the legacy template",
                ));
            }
            None => {}
        }
        if !lens.counts_toward_a35 {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!("template {} has a non-counting content lens", template.name),
                "store non-content time controls in time_controls, not lenses",
            ));
        }
    }
    Ok(())
}

fn validate_time_controls(template: &SavedPanelTemplate) -> CliResult {
    for control in &template.time_controls {
        if control.counts_toward_a35 {
            return Err(template_error(
                TEMPLATE_INVALID,
                format!("time control {} counts toward A35", control.slot_key),
                "temporal/time capture is a control sidecar and must not count as an embedder",
            ));
        }
    }
    Ok(())
}

fn validate_weight_hash(value: &str) -> CliResult {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Ok(());
    }
    Err(template_error(
        TEMPLATE_INVALID,
        format!("weights_sha256 must be 64 hex chars, got {value}"),
        "rebuild the template from frozen lens manifests",
    ))
}
