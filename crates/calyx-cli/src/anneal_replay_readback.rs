use calyx_anneal::decode_replay_rows;
use calyx_aster::cf::ColumnFamily;
use calyx_aster::mvcc::is_tombstone_value;
use calyx_aster::sst::SstReader;
use serde_json::json;
use std::path::Path;

use crate::cf_read::{hex_bytes, latest_cf_rows, list_sst_files};
use crate::error::CliError;

pub fn replay_status(vault: &Path) -> crate::error::CliResult {
    let cf = ColumnFamily::AnnealReplay;
    let mut physical_rows = Vec::new();
    for file in list_sst_files(&vault.join("cf").join(cf.name()))? {
        let reader = SstReader::open(&file)?;
        for row in reader.iter()? {
            let readback = json!({
                "file": file.display().to_string(),
                "key_hex": hex_bytes(&row.key),
                "value_hex": hex_bytes(&row.value),
                "value_len": row.value.len(),
                "tombstone": is_tombstone_value(&row.value),
            });
            physical_rows.push(readback);
        }
    }
    let live_rows = latest_cf_rows(vault, cf)?
        .into_iter()
        .filter(|(_, value)| !is_tombstone_value(value))
        .collect::<Vec<_>>();
    let (capacity, len, top_surprises, rows) = match decode_replay_rows(&live_rows)? {
        Some(snapshot) => {
            let mut entries = snapshot.entries;
            entries.sort_by(|left, right| right.cmp(left));
            let top_surprises = entries
                .iter()
                .take(5)
                .map(|entry| entry.surprise)
                .collect::<Vec<_>>();
            let rows = json!({
                "keys_hex": live_rows.iter().map(|(key, _)| hex_bytes(key)).collect::<Vec<_>>(),
                "entries": entries,
            });
            (json!(snapshot.capacity), entries.len(), top_surprises, rows)
        }
        None => (
            json!(null),
            0,
            Vec::new(),
            json!({"keys_hex": [], "entries": []}),
        ),
    };
    let readback = json!({
        "cf": cf.name(),
        "vault": vault.display().to_string(),
        "len": len,
        "capacity": capacity,
        "top_surprises": top_surprises,
        "physical_row_count": physical_rows.len(),
        "physical_rows": physical_rows,
        "rows": rows,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&readback).map_err(|error| {
            CliError::runtime(format!("serialize anneal replay readback: {error}"))
        })?
    );
    Ok(())
}
