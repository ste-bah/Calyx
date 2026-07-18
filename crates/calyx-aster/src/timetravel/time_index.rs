//! The `time_index` column family: a wall-clock → MVCC-seqno map.
//!
//! Each committed group-commit writes one entry whose **key is the data** —
//! `big_endian_u64(millis_utc) || big_endian_u64(seqno)` — and whose value is a
//! single sentinel byte. Big-endian ordering means a predecessor seek at
//! `millis = t, seqno = u64::MAX` lands the `floor(t)` entry, so resolving a
//! timestamp to the greatest seqno `≤ t` is one bounded seek with no WAL replay
//! (PRD `17 §8`). The index is the sole source of truth for the time→seqno
//! mapping. Raw callers cannot write this reserved CF; only the atomic commit
//! path may derive its rows.

use calyx_core::{CalyxError, Clock, Result, Seq};

use crate::cf::ColumnFamily;
use crate::vault::AsterVault;

/// Sentinel value stored under every time-index key (the key carries the data).
pub(crate) const SENTINEL: &[u8] = &[0u8];

const KEY_LEN: usize = 16;

/// Encodes a `(millis, seqno)` time-index key.
pub(crate) fn encode_key(millis: u64, seqno: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(KEY_LEN);
    key.extend_from_slice(&millis.to_be_bytes());
    key.extend_from_slice(&seqno.to_be_bytes());
    key
}

/// Decodes a `(millis, seqno)` time-index key, failing closed on a malformed
/// key rather than returning a silently wrong seqno.
pub(crate) fn decode_key(key: &[u8]) -> Result<(u64, u64)> {
    if key.len() != KEY_LEN {
        return Err(CalyxError::aster_corrupt_shard(format!(
            "time_index key must be {KEY_LEN} bytes, found {}",
            key.len()
        )));
    }
    let millis = u64::from_be_bytes(key[..8].try_into().expect("8-byte millis"));
    let seqno = u64::from_be_bytes(key[8..].try_into().expect("8-byte seqno"));
    Ok((millis, seqno))
}

/// The `(cf, key, value)` triple to append to a group-commit batch for `seqno`
/// committed at `millis`.
pub(crate) fn entry_row(millis: u64, seqno: Seq) -> (ColumnFamily, Vec<u8>, Vec<u8>) {
    (
        ColumnFamily::TimeIndex,
        encode_key(millis, seqno),
        SENTINEL.to_vec(),
    )
}

/// One decoded time-index entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeIndexEntry {
    pub millis: u64,
    pub seqno: Seq,
}

/// Inclusive seek target for the greatest `(millis, seqno)` at `millis <= t`.
fn floor_target(t_millis: u64) -> Vec<u8> {
    encode_key(t_millis, u64::MAX)
}

/// Resolves `t_millis` to the greatest seqno committed at or before it, reading
/// the index at the vault's latest sequence. Returns `CALYX_TIMETRAVEL_NO_DATA`
/// when the vault has no entry at or before `t` (an explicit empty result, never
/// a silent stale seqno).
pub(crate) fn resolve<C: Clock>(vault: &AsterVault<C>, t_millis: u64) -> Result<Seq> {
    let latest = vault.latest_seq();
    let Some((key, _)) = vault.predecessor_cf_at(
        latest,
        ColumnFamily::TimeIndex,
        &encode_key(0, 0),
        &floor_target(t_millis),
    )?
    else {
        return Err(no_data(format!(
            "no time-index entry at or before t={t_millis}ms"
        )));
    };
    decode_key(&key).map(|(_, seqno)| seqno)
}

/// Reads every time-index entry visible at the vault's latest sequence, in
/// `(millis, seqno)` order. Used for FSV readback of the source of truth.
pub fn read_all<C: Clock>(vault: &AsterVault<C>) -> Result<Vec<TimeIndexEntry>> {
    let latest = vault.latest_seq();
    vault
        .scan_cf_at(latest, ColumnFamily::TimeIndex)?
        .into_iter()
        .map(|(key, _)| {
            let (millis, seqno) = decode_key(&key)?;
            Ok(TimeIndexEntry { millis, seqno })
        })
        .collect()
}

pub(crate) fn no_data(message: impl Into<String>) -> CalyxError {
    CalyxError {
        code: "CALYX_TIMETRAVEL_NO_DATA",
        message: message.into(),
        remediation: "query at or after the first write, or check the vault has any committed data",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_round_trips_and_orders_by_millis_then_seqno() {
        let a = encode_key(1000, 5);
        let b = encode_key(1000, 6);
        let c = encode_key(2000, 1);
        assert!(a < b, "same millis orders by seqno");
        assert!(b < c, "later millis sorts after earlier");
        assert_eq!(decode_key(&a).unwrap(), (1000, 5));
    }

    #[test]
    fn decode_rejects_short_key() {
        assert_eq!(
            decode_key(&[0u8; 8]).unwrap_err().code,
            "CALYX_ASTER_CORRUPT_SHARD"
        );
    }

    #[test]
    fn floor_target_selects_same_millis_max_seqno() {
        let target = floor_target(1500);
        assert_eq!(target, encode_key(1500, u64::MAX));
        assert!(encode_key(1500, 9) <= target);
        assert!(encode_key(1501, 0) > target);
    }

    #[test]
    fn floor_target_at_u64_max_does_not_overflow() {
        assert_eq!(floor_target(u64::MAX), vec![0xff; KEY_LEN]);
    }
}
