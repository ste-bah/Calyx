use super::{AsterVault, DEFAULT_LEASE_MS};
use crate::mvcc::{Freshness, Snapshot, VersionedCfStore};
use calyx_core::{Clock, Seq};

pub(crate) struct ScopedSnapshot<'a> {
    rows: &'a VersionedCfStore,
    snapshot: Snapshot,
}

impl ScopedSnapshot<'_> {
    pub(crate) const fn snapshot(&self) -> Snapshot {
        self.snapshot
    }
}

impl Drop for ScopedSnapshot<'_> {
    fn drop(&mut self) {
        self.rows.release_lease(self.snapshot.lease().id());
    }
}

impl<C> AsterVault<C>
where
    C: Clock,
{
    pub(crate) fn snapshot_handle(&self, seq: Seq) -> ScopedSnapshot<'_> {
        let snapshot =
            self.rows
                .pin_snapshot_at(seq, Freshness::FreshDerived, &self.clock, DEFAULT_LEASE_MS);
        ScopedSnapshot {
            rows: &self.rows,
            snapshot,
        }
    }

    pub(crate) fn with_scoped_snapshot<T>(
        &self,
        seq: Seq,
        read: impl FnOnce(Snapshot) -> calyx_core::Result<T>,
    ) -> calyx_core::Result<T> {
        let snapshot = self.snapshot_handle(seq);
        read(snapshot.snapshot())
    }
}
