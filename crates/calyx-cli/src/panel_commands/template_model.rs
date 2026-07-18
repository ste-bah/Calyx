//! Persisted panel-template model and its fail-closed validation contracts.

mod codec;
mod lens;
mod panel;
mod types;
mod validation;

pub(in crate::panel_commands) mod admission;

#[cfg(test)]
mod tests;

#[cfg(test)]
use calyx_core::{LensCost, LensId, Modality, Placement, SlotShape};
#[cfg(test)]
use calyx_registry::FrozenLensContract;
#[cfg(test)]
use lens::json_blake3;

pub(super) use codec::{id_for_loaded, object_bytes};
pub(super) use lens::lens_ref_from_catalog;
pub(super) use panel::default_time_controls;
pub(super) use types::{
    CapabilityCardRef, PanelTemplateCatalog, PanelTemplateIndexEntry, PanelTemplateVersionRef,
    SavedPanelTemplate, TemplateA37Admission, TemplateA37CardRef, TemplateDraft,
    TemplateEnsembleCard, TemplateLensRef, TemplateLensSnapshot, TemplateTimeControl,
};
pub(super) use validation::template_error;

pub(super) const MIN_CONTENT_LENSES: usize = 10;
pub(super) const CATALOG_VERSION: u16 = 1;
pub(super) const OBJECT_VERSION: u16 = 2;
pub(super) const LENS_SNAPSHOT_VERSION: u16 = 1;
pub(super) const CARD_VERSION: u16 = 1;
pub(super) const A37_ADMISSION_VERSION: u16 = 1;
pub(super) const TEMPLATE_INVALID: &str = "CALYX_PANEL_TEMPLATE_INVALID";
pub(super) const TEMPLATE_NOT_FOUND: &str = "CALYX_PANEL_TEMPLATE_NOT_FOUND";
pub(super) const TEMPLATE_A37_GATE_REFUSED: &str = "CALYX_PANEL_TEMPLATE_A37_GATE_REFUSED";
