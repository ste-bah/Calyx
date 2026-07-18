use super::*;

impl VersionedCfStore {
    /// Streams visible rows for one CF at the pinned sequence in bounded pages.
    pub fn scan_cf_pages_at<F, E>(
        &self,
        snapshot: Snapshot,
        cf: ColumnFamily,
        limit: usize,
        clock: &dyn Clock,
        on_page: F,
    ) -> std::result::Result<(), E>
    where
        F: FnMut(Vec<(Vec<u8>, Vec<u8>)>) -> std::result::Result<(), E>,
        E: From<calyx_core::CalyxError>,
    {
        self.scan_cf_range_pages_at(snapshot, cf, &KeyRange::all(), limit, clock, on_page)
    }

    /// Streams visible rows in bounded pages without reopening SST readers per page.
    pub fn scan_cf_range_pages_at<F, E>(
        &self,
        snapshot: Snapshot,
        cf: ColumnFamily,
        range: &KeyRange,
        limit: usize,
        clock: &dyn Clock,
        mut on_page: F,
    ) -> std::result::Result<(), E>
    where
        F: FnMut(Vec<(Vec<u8>, Vec<u8>)>) -> std::result::Result<(), E>,
        E: From<calyx_core::CalyxError>,
    {
        self.ensure_snapshot_live(snapshot, clock)
            .map_err(E::from)?;
        if limit == 0 {
            return Ok(());
        }
        if self.router_latest_readback.load(Ordering::Acquire) {
            // Values in embedding CFs can be multiple MiB each.  The former
            // router fast path cloned every visible table value into one
            // overlay Vec before emitting its first page, so `limit` bounded
            // callback size but not memory.  Resolve the (small) visible key
            // set once, then point-read only one page of values at a time at
            // the exact same pinned snapshot.
            let keys = self
                .scan_cf_range_keys_at(snapshot, cf, range, clock)
                .map_err(E::from)?;
            for page_keys in keys.chunks(limit) {
                let reads = page_keys
                    .iter()
                    .cloned()
                    .map(|key| CfRead::new(cf, key))
                    .collect::<Vec<_>>();
                let values = self.read_batch(snapshot, &reads, clock).map_err(E::from)?;
                let mut page = Vec::with_capacity(page_keys.len());
                for (key, value) in page_keys.iter().cloned().zip(values) {
                    let value = value.ok_or_else(|| {
                        E::from(calyx_core::CalyxError::aster_corrupt_shard(format!(
                            "visible {} key {} disappeared during pinned paged read",
                            cf.name(),
                            super::read::hex_prefix(&key)
                        )))
                    })?;
                    page.push((key, value));
                }
                on_page(page)?;
            }
            return Ok(());
        }
        let mut after_key = None::<Vec<u8>>;
        loop {
            let page = self
                .scan_cf_range_page_at(snapshot, cf, range, after_key.as_deref(), limit, clock)
                .map_err(E::from)?;
            let Some(last_key) = page.last().map(|(key, _)| key.clone()) else {
                break;
            };
            after_key = Some(last_key);
            on_page(page)?;
        }
        Ok(())
    }
}
