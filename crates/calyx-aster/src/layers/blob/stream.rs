use super::*;

pub struct BlobChunkStream<'a, C: Clock> {
    pub(super) vault: &'a AsterVault<C>,
    pub(super) chunk_prefix: Vec<u8>,
    pub(super) chunk_count: u32,
    pub(super) next_idx: u32,
}

impl<C: Clock> Iterator for BlobChunkStream<'_, C> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx >= self.chunk_count {
            return None;
        }
        let idx = self.next_idx;
        self.next_idx += 1;
        let mut key = self.chunk_prefix.clone();
        key.extend_from_slice(&idx.to_be_bytes());
        match self
            .vault
            .read_cf_at(self.vault.latest_seq(), ColumnFamily::Blob, &key)
        {
            Ok(Some(bytes)) => Some(Ok(bytes)),
            Ok(None) => Some(Err(corrupt(format!(
                "blob manifest claims {} chunks but chunk {idx} is missing",
                self.chunk_count
            )))),
            Err(error) => Some(Err(error)),
        }
    }
}
