use calyx_core::{Constellation, Result, SlotId};

use super::encode::{self, WriteRow};
use crate::cf::{ColumnFamily, anchor_key, base_key, slot_key};

pub(super) struct PreparedConstellationEncoding {
    slot_hashes: Vec<(SlotId, [u8; 32])>,
    slot_rows: Vec<(SlotId, Vec<u8>)>,
}

impl PreparedConstellationEncoding {
    pub(super) fn new(constellation: &Constellation) -> Result<Self> {
        let mut slot_hashes = Vec::with_capacity(constellation.slots.len());
        let mut slot_rows = Vec::with_capacity(constellation.slots.len());
        for (slot, vector) in &constellation.slots {
            let bytes = encode::encode_slot_vector(vector)?;
            slot_hashes.push((*slot, encode::hash_slot_bytes(&bytes)));
            slot_rows.push((*slot, bytes));
        }
        Ok(Self {
            slot_hashes,
            slot_rows,
        })
    }

    pub(super) fn encode_base(&self, constellation: &Constellation) -> Result<Vec<u8>> {
        encode::encode_constellation_base_with_slot_hashes(constellation, &self.slot_hashes)
    }
}

pub(super) fn stage_validated_constellation_rows(
    rows: &mut Vec<WriteRow>,
    constellation: &Constellation,
    prepared: PreparedConstellationEncoding,
) -> Result<()> {
    let id = constellation.cx_id;
    rows.push(WriteRow {
        cf: ColumnFamily::Base,
        key: base_key(id),
        value: prepared.encode_base(constellation)?,
    });
    for (slot, bytes) in prepared.slot_rows {
        rows.push(WriteRow {
            cf: ColumnFamily::slot(slot),
            key: slot_key(id),
            value: bytes,
        });
    }
    for anchor in &constellation.anchors {
        rows.push(WriteRow {
            cf: ColumnFamily::Anchors,
            key: anchor_key(id, &anchor.kind),
            value: encode::encode_anchor(anchor)?,
        });
    }
    Ok(())
}
