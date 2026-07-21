use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

use calyx_core::{CxId, SlotId};
use calyx_sextant::index::{
    CUVS_RESIDENT_FLAT_MAX_BATCH, CUVS_RESIDENT_FLAT_MAX_K, CuvsResidentFlatIndex, IndexSearchHit,
    ranked,
};

use super::Index;
use crate::error::CliResult;
use crate::persisted::pinned;
use crate::persisted::{SearchIndexEntry, stale};

const CACHE_MIB_ENV: &str = "CALYX_SEARCH_FLAT_CUDA_CACHE_MIB";
const CACHE_ENTRIES_ENV: &str = "CALYX_SEARCH_FLAT_CUDA_CACHE_ENTRIES";
const DEFAULT_CACHE_MIB: u64 = 2048;
const DEFAULT_CACHE_ENTRIES: usize = 256;

type CacheKey = (String, u16, String);

struct Entry {
    index: Arc<Mutex<CuvsResidentFlatIndex>>,
    bytes: u64,
    tick: u64,
}

struct Cache {
    entries: BTreeMap<CacheKey, Entry>,
    max_bytes: u64,
    max_entries: usize,
    resident_bytes: u64,
    tick: u64,
}

static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();

pub(super) fn search(
    vault_dir: &std::path::Path,
    entry: &SearchIndexEntry,
    slot: SlotId,
    host: &Arc<Index>,
    query: &[f32],
    k: usize,
    candidates: Option<&BTreeSet<CxId>>,
) -> CliResult<Vec<IndexSearchHit>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let key = (
        pinned::canonical_vault_dir(vault_dir)?,
        slot.get(),
        entry.require_sha256(slot)?.to_string(),
    );
    let resident = acquire(key, host)?;
    let allowed = candidates.map(|candidates| {
        host.rows
            .iter()
            .enumerate()
            .filter(|(_, (cx_id, _))| candidates.contains(cx_id))
            .map(|(idx, _)| u32::try_from(idx).expect("flat rows fit u32"))
            .collect::<Vec<_>>()
    });
    if allowed.as_ref().is_some_and(Vec::is_empty) {
        return Ok(Vec::new());
    }
    let local = resident
        .lock()
        .map_err(|_| stale("resident flat CUDA index lock poisoned"))?
        .search(query, 1, k.min(host.rows.len()), allowed.as_deref())?
        .pop()
        .expect("one flat query row");
    let scored = local
        .into_iter()
        .filter_map(|(id, distance)| {
            host.rows
                .get(id as usize)
                .map(|(cx_id, _)| (*cx_id, 1.0 - distance))
        })
        .collect();
    Ok(ranked(scored))
}

fn acquire(key: CacheKey, host: &Index) -> CliResult<Arc<Mutex<CuvsResidentFlatIndex>>> {
    let estimated = estimated_bytes(host.header.len, host.header.dim as usize);
    let mut cache = cache()
        .lock()
        .map_err(|_| stale("resident flat CUDA cache lock poisoned"))?;
    cache.tick = cache.tick.wrapping_add(1);
    let tick = cache.tick;
    invalidate_generation(&mut cache, &key);
    if let Some(entry) = cache.entries.get_mut(&key) {
        entry.tick = tick;
        return Ok(entry.index.clone());
    }
    reserve(&mut cache, estimated)?;
    let values = host
        .rows
        .iter()
        .flat_map(|(_, values)| values.iter().copied())
        .collect::<Vec<_>>();
    let index = CuvsResidentFlatIndex::new(host.header.len, host.header.dim as usize, &values)?;
    let bytes = index.resident_bytes();
    reserve(&mut cache, bytes)?;
    let index = Arc::new(Mutex::new(index));
    cache.resident_bytes = cache.resident_bytes.saturating_add(bytes);
    cache.entries.insert(
        key,
        Entry {
            index: index.clone(),
            bytes,
            tick,
        },
    );
    Ok(index)
}

fn invalidate_generation(cache: &mut Cache, key: &CacheKey) {
    let stale_keys = cache
        .entries
        .iter()
        .filter_map(|(candidate, entry)| {
            (candidate.0 == key.0
                && candidate.1 == key.1
                && candidate.2 != key.2
                && Arc::strong_count(&entry.index) == 1)
                .then_some(candidate.clone())
        })
        .collect::<Vec<_>>();
    for stale_key in stale_keys {
        if let Some(entry) = cache.entries.remove(&stale_key) {
            cache.resident_bytes = cache.resident_bytes.saturating_sub(entry.bytes);
        }
    }
}

fn reserve(cache: &mut Cache, additional: u64) -> CliResult {
    while cache.resident_bytes.saturating_add(additional) > cache.max_bytes
        || cache.entries.len() >= cache.max_entries
    {
        let victim = cache
            .entries
            .iter()
            .filter(|(_, entry)| Arc::strong_count(&entry.index) == 1)
            .min_by_key(|(_, entry)| entry.tick)
            .map(|(key, _)| key.clone());
        let Some(victim) = victim else {
            return Err(calyx_sextant::sextant_error(
                calyx_sextant::CALYX_SEXTANT_GPU_CACHE_EXHAUSTED,
                format!(
                    "resident flat cache cannot reserve {additional} bytes: resident={} cap={} entries={}/{}",
                    cache.resident_bytes,
                    cache.max_bytes,
                    cache.entries.len(),
                    cache.max_entries
                ),
            )
            .into());
        };
        if let Some(entry) = cache.entries.remove(&victim) {
            cache.resident_bytes = cache.resident_bytes.saturating_sub(entry.bytes);
        }
    }
    Ok(())
}

fn estimated_bytes(rows: usize, dim: usize) -> u64 {
    let dataset = rows.saturating_mul(dim).saturating_mul(size_of::<f32>());
    let query = CUVS_RESIDENT_FLAT_MAX_BATCH
        .saturating_mul(dim)
        .saturating_mul(size_of::<f32>());
    let output = CUVS_RESIDENT_FLAT_MAX_BATCH
        .saturating_mul(CUVS_RESIDENT_FLAT_MAX_K)
        .saturating_mul(size_of::<i64>() + size_of::<f32>());
    let filter = rows.div_ceil(32).saturating_mul(size_of::<u32>());
    u64::try_from(
        dataset
            .saturating_add(query)
            .saturating_add(output)
            .saturating_add(filter),
    )
    .unwrap_or(u64::MAX)
}

fn cache() -> &'static Mutex<Cache> {
    CACHE.get_or_init(|| {
        Mutex::new(Cache {
            entries: BTreeMap::new(),
            max_bytes: env_u64(CACHE_MIB_ENV, DEFAULT_CACHE_MIB).saturating_mul(1024 * 1024),
            max_entries: env_usize(CACHE_ENTRIES_ENV, DEFAULT_CACHE_ENTRIES),
            resident_bytes: 0,
            tick: 0,
        })
    })
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}
