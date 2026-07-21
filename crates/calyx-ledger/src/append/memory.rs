use super::*;

/// In-memory row store for deterministic tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryLedgerStore {
    rows: BTreeMap<u64, Vec<u8>>,
    anchor: Option<LedgerHeadAnchor>,
}

impl MemoryLedgerStore {
    pub fn insert_raw(&mut self, seq: u64, bytes: Vec<u8>) {
        self.rows.insert(seq, bytes);
    }

    pub fn remove_raw(&mut self, seq: u64) {
        self.rows.remove(&seq);
    }
}

impl LedgerCfStore for MemoryLedgerStore {
    fn scan(&self) -> Result<Vec<LedgerRow>> {
        Ok(self
            .rows
            .iter()
            .map(|(seq, bytes)| LedgerRow {
                seq: *seq,
                bytes: bytes.clone(),
            })
            .collect())
    }

    fn read_seq(&self, seq: u64) -> Result<Option<LedgerRow>> {
        Ok(self.rows.get(&seq).map(|bytes| LedgerRow {
            seq,
            bytes: bytes.clone(),
        }))
    }

    fn put_new(&mut self, seq: u64, bytes: &[u8]) -> Result<()> {
        if self.rows.contains_key(&seq) {
            return Err(append_only_violation(format!(
                "ledger seq {seq} already exists"
            )));
        }
        self.rows.insert(seq, bytes.to_vec());
        Ok(())
    }

    fn head_anchor(&self) -> Result<Option<LedgerHeadAnchor>> {
        Ok(self.anchor.clone())
    }

    fn put_head_anchor(&mut self, anchor: &LedgerHeadAnchor) -> Result<()> {
        if let Some(current) = &self.anchor
            && anchor.height < current.height
        {
            return Err(append_only_violation(format!(
                "ledger head anchor regressed from {} to {}",
                current.height, anchor.height
            )));
        }
        self.anchor = Some(anchor.clone());
        Ok(())
    }
}
