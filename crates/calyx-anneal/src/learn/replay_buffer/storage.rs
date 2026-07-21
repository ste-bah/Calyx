use calyx_aster::cf::ColumnFamily;
use calyx_aster::mvcc::tombstone_value;
use calyx_aster::vault::AsterVault;
use calyx_core::{Clock, Result};

use super::cf_unavailable;

/// One atomic replay-CF mutation. Deletes are persisted as Aster MVCC
/// tombstones, so a crash exposes either the old generation or the new one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayWrite {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

pub trait ReplayStorage: Send + Sync {
    /// Returns the complete live AnnealReplay keyspace at one consistent read.
    fn scan_rows(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Atomically commits every supplied operation in order.
    fn commit(&self, writes: &[ReplayWrite]) -> Result<()>;
}

pub struct AsterReplayStorage<'a, C>
where
    C: Clock,
{
    vault: &'a AsterVault<C>,
}

impl<'a, C> AsterReplayStorage<'a, C>
where
    C: Clock,
{
    pub const fn new(vault: &'a AsterVault<C>) -> Self {
        Self { vault }
    }
}

impl<C> ReplayStorage for AsterReplayStorage<'_, C>
where
    C: Clock,
{
    fn scan_rows(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.vault
            .scan_cf_at(self.vault.latest_seq(), ColumnFamily::AnnealReplay)
            .map_err(|error| cf_unavailable("scan anneal_replay CF", error))
    }

    fn commit(&self, writes: &[ReplayWrite]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        let rows = writes.iter().map(|write| match write {
            ReplayWrite::Put { key, value } => {
                (ColumnFamily::AnnealReplay, key.clone(), value.clone())
            }
            ReplayWrite::Delete { key } => {
                (ColumnFamily::AnnealReplay, key.clone(), tombstone_value())
            }
        });
        self.vault
            .write_cf_batch(rows)
            .map(|_| ())
            .map_err(|error| cf_unavailable("commit anneal_replay CF batch", error))
    }
}
