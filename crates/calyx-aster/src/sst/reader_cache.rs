//! Process-wide bounded shared `SstReader` cache — the LSM "table cache".
//!
//! `SstReader::open` computes a whole-body CRC32 over the mapped file on every
//! open (see [`super::read_header`]). While every SST is small that cost is
//! invisible, but once compaction collapses a column family to one large SST,
//! any read loop that re-opens the same file per item re-checksums the whole
//! file each time and pins a core at 100%.
//!
//! This cache keeps a bounded set of already-opened [`SstReader`]s keyed by the
//! canonical file path and hands out cheap `Arc` clones. SSTs are immutable by
//! the LSM invariant (unique names from flush ordinals, durable seqs, and
//! compaction outputs; never rewritten in place), so a cached reader stays
//! valid for the life of the file. The per-hit `len` + `modified` revalidation
//! is belt-and-braces: if a path were ever reused for different bytes, the
//! mismatch forces a fresh open (and a fresh checksum) rather than serving
//! stale bytes.
//!
//! Windows nuance: an active memory mapping keeps `DeleteFile` failing, so a
//! cached reader can block reclaim of its file. Callers therefore invalidate
//! via [`invalidate_reader`] immediately before `fs::remove_file`; this drops
//! the cache's `Arc`, shrinking the window in which a delete can fail to the
//! set of reads still holding their own clone in flight (the same fail-closed
//! behaviour as before this cache existed — deletion never silently races a
//! live mapping).

use super::SstReader;
use calyx_core::{CalyxError, Result};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::SystemTime;

/// Upper bound on concurrently cached readers. Each entry holds one `Mmap`
/// handle plus decoded index/bloom, so the cap bounds both file descriptors and
/// resident metadata. 256 comfortably covers a fanned-out level while staying
/// small enough to evict promptly under pathological path churn.
const MAX_CACHED_READERS: usize = 256;

/// A cached reader plus the file identity it was validated against.
struct Entry {
    reader: Arc<SstReader>,
    len: u64,
    /// `None` when the platform cannot report a modification time; two `None`s
    /// compare equal and fall back to length-only revalidation.
    modified: Option<SystemTime>,
    last_used: u64,
}

fn cache() -> &'static Mutex<HashMap<PathBuf, Entry>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Monotonic LRU stamp. Hand-rolled so the cache needs no extra dependency.
fn next_tick() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(0);
    CLOCK.fetch_add(1, Ordering::Relaxed)
}

/// Locks the cache, recovering from a poisoned mutex: entries are just `Arc`s
/// and revalidated metadata, so a panic mid-mutation leaves no torn invariant.
fn lock() -> MutexGuard<'static, HashMap<PathBuf, Entry>> {
    cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Canonical, absolute cache key. Fails closed if an existing file cannot be
/// canonicalized so callers never silently degrade to an uncached path form
/// that would miss both the cache and later invalidation.
fn canonical_key(path: &Path) -> Result<PathBuf> {
    fs::canonicalize(path).map_err(|error| metadata_error("canonicalize SST", path, error))
}

fn metadata_error(context: &str, path: &Path, error: io::Error) -> CalyxError {
    CalyxError::disk_pressure(format!("{context} {}: {error}", path.display()))
}

/// Returns a shared reader for `path`, opening (and checksumming) it at most
/// once per distinct on-disk version. On a cache hit the file's current `len`
/// and `modified` are revalidated against the cached identity; a match bumps
/// the LRU stamp and returns a cheap `Arc` clone, a mismatch reopens fresh.
pub fn shared_reader(path: &Path) -> Result<Arc<SstReader>> {
    let key = canonical_key(path)?;
    let meta = fs::metadata(&key).map_err(|error| metadata_error("stat SST", &key, error))?;
    let len = meta.len();
    let modified = meta.modified().ok();

    {
        let mut cache = lock();
        if let Some(entry) = cache.get_mut(&key) {
            if entry.len == len && entry.modified == modified {
                entry.last_used = next_tick();
                return Ok(entry.reader.clone());
            }
            // Identity drifted (path reused for new bytes): drop and reopen.
            cache.remove(&key);
        }
    }

    // Open outside the lock so the whole-file checksum never serializes other
    // readers. A benign race may open the same file twice; both readers are
    // valid over the immutable file and the last insert simply wins.
    let reader = Arc::new(SstReader::open(&key)?);

    let mut cache = lock();
    cache.insert(
        key,
        Entry {
            reader: reader.clone(),
            len,
            modified,
            last_used: next_tick(),
        },
    );
    evict_over_cap(&mut cache);
    Ok(reader)
}

/// Drops the cached reader for `path` (if any) so a subsequent `fs::remove_file`
/// is not blocked by this cache's live mapping. Best-effort and infallible:
/// callers invoke it just before deletion while the file still exists, so the
/// canonical key matches the stored entry; if the file is already gone the
/// canonical form is unavailable and the raw path is tried as a fallback.
pub fn invalidate_reader(path: &Path) {
    let mut cache = lock();
    match fs::canonicalize(path) {
        Ok(key) => {
            cache.remove(&key);
        }
        Err(_) => {
            cache.remove(path);
        }
    }
}

/// Evicts least-recently-used entries until the cache is within its cap.
fn evict_over_cap(cache: &mut HashMap<PathBuf, Entry>) {
    while cache.len() > MAX_CACHED_READERS {
        let victim = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        match victim {
            Some(key) => {
                cache.remove(&key);
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sst::write_sst;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn test_dir(name: &str) -> PathBuf {
        let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "calyx-sst-reader-cache-{name}-{}-{id}",
            std::process::id()
        ));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn shared_reader_returns_same_instance_until_invalidated() {
        let dir = test_dir("hit");
        let path = dir.join("a.sst");
        write_sst(&path, [(b"k".as_slice(), b"v".as_slice())]).unwrap();

        let first = shared_reader(&path).unwrap();
        let second = shared_reader(&path).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache hit must return the same shared reader"
        );
        assert_eq!(first.get(b"k").unwrap().as_deref(), Some(b"v".as_slice()));

        invalidate_reader(&path);
        let third = shared_reader(&path).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &third),
            "invalidation must force a fresh open"
        );

        drop((first, second, third));
        invalidate_reader(&path);
        fs::remove_dir_all(dir).unwrap();
    }

    /// Unix-only: Windows cannot replace a file while the cache's live mapping
    /// exists, which is exactly why reclaim sites call [`invalidate_reader`]
    /// before deletion there.
    #[cfg(unix)]
    #[test]
    fn shared_reader_reopens_when_file_identity_drifts() {
        let dir = test_dir("drift");
        let path = dir.join("b.sst");
        write_sst(&path, [(b"k".as_slice(), b"old".as_slice())]).unwrap();
        let first = shared_reader(&path).unwrap();

        // Rewrite the path with different bytes (violating the immutability
        // invariant on purpose): revalidation must serve the new content.
        fs::remove_file(&path).ok();
        write_sst(&path, [(b"k".as_slice(), b"new-longer-value".as_slice())]).unwrap();

        let second = shared_reader(&path).unwrap();
        assert_eq!(
            second.get(b"k").unwrap().as_deref(),
            Some(b"new-longer-value".as_slice()),
            "identity drift must force a fresh open with the new bytes"
        );

        drop((first, second));
        invalidate_reader(&path);
        fs::remove_dir_all(dir).unwrap();
    }
}
