use calyx_core::{CalyxError, Result};

const MAGIC: &[u8; 4] = b"CXA1";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 16;
pub(crate) const COLUMN_TRANSPOSE_CUDA_MIN_ELEMENTS: usize = 262_144;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColumnEncodeStats {
    pub backend: &'static str,
    pub kernel_launches: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub peak_pinned_staging_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrowChunkView<'a> {
    raw: &'a [u8],
    rows: Vec<f32>,
    n_rows: usize,
    dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrowColumnView<'a> {
    raw: &'a [u8],
    n_rows: usize,
    dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ArrowColumnValues<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ArrowChunkView<'a> {
    pub fn row(&self, index: usize) -> Result<&[f32]> {
        if index >= self.n_rows {
            return Err(CalyxError::aster_corrupt_shard(
                "arrow row index out of bounds",
            ));
        }
        let start = index * self.dim;
        Ok(&self.rows[start..start + self.dim])
    }

    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub const fn raw_bytes(&self) -> &'a [u8] {
        self.raw
    }
}

impl<'a> ArrowColumnView<'a> {
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub const fn raw_bytes(&self) -> &'a [u8] {
        self.raw
    }

    pub fn column_bytes(&self, column: usize) -> Result<&'a [u8]> {
        if column >= self.dim {
            return Err(CalyxError::aster_corrupt_shard(
                "arrow column index out of bounds",
            ));
        }
        let start = HEADER_LEN + column * self.n_rows * 4;
        let end = start + self.n_rows * 4;
        Ok(&self.raw[start..end])
    }

    pub fn column_values(&self, column: usize) -> Result<ArrowColumnValues<'a>> {
        Ok(ArrowColumnValues {
            bytes: self.column_bytes(column)?,
            offset: 0,
        })
    }

    pub fn value(&self, column: usize, row: usize) -> Result<f32> {
        if row >= self.n_rows {
            return Err(CalyxError::aster_corrupt_shard(
                "arrow row index out of bounds",
            ));
        }
        let bytes = self.column_bytes(column)?;
        let start = row * 4;
        Ok(f32::from_le_bytes(
            bytes[start..start + 4].try_into().expect("f32"),
        ))
    }
}

impl Iterator for ArrowColumnValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.bytes.len() {
            return None;
        }
        let value = f32::from_le_bytes(
            self.bytes[self.offset..self.offset + 4]
                .try_into()
                .expect("f32"),
        );
        self.offset += 4;
        Some(value)
    }
}

pub fn encode_column_chunk(rows: &[&[f32]]) -> Result<Vec<u8>> {
    let (dim, value_count, encoded_len) = validate_encode_shape(rows)?;
    encode_columns_cpu(rows, dim, value_count, encoded_len)
}

pub(crate) fn encode_column_chunk_accelerated(
    rows: &[&[f32]],
) -> Result<(Vec<u8>, ColumnEncodeStats)> {
    let (dim, value_count, encoded_len) = validate_encode_shape(rows)?;
    if value_count < COLUMN_TRANSPOSE_CUDA_MIN_ELEMENTS {
        return Ok((
            encode_columns_cpu(rows, dim, value_count, encoded_len)?,
            cpu_encode_stats(),
        ));
    }
    encode_columns_cuda(rows, dim, encoded_len)
}

fn validate_encode_shape(rows: &[&[f32]]) -> Result<(usize, usize, usize)> {
    let dim = rows
        .first()
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk has no rows"))?
        .len();
    if dim == 0 {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk dim must be > 0",
        ));
    }
    if rows.iter().any(|row| row.len() != dim) {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk row dims differ",
        ));
    }
    u32::try_from(rows.len())
        .map_err(|_| CalyxError::aster_corrupt_shard("arrow chunk rows exceed u32"))?;
    u32::try_from(dim)
        .map_err(|_| CalyxError::aster_corrupt_shard("arrow chunk dim exceeds u32"))?;
    let value_count = rows
        .len()
        .checked_mul(dim)
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk value count overflow"))?;
    let payload_len = value_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk payload length overflow"))?;
    let encoded_len = HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk encoded length overflow"))?;
    Ok((dim, value_count, encoded_len))
}

fn encode_columns_cpu(
    rows: &[&[f32]],
    dim: usize,
    value_count: usize,
    encoded_len: usize,
) -> Result<Vec<u8>> {
    let mut out = allocate_encoded(encoded_len)?;
    write_header(&mut out, rows.len(), dim);
    for column in 0..dim {
        for row in rows {
            out.extend_from_slice(&row[column].to_le_bytes());
        }
    }
    debug_assert_eq!(out.len(), HEADER_LEN + value_count * size_of::<f32>());
    Ok(out)
}

fn allocate_encoded(capacity: usize) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| CalyxError::disk_pressure("arrow chunk allocation failed"))?;
    Ok(output)
}

fn write_header(output: &mut Vec<u8>, rows: usize, dim: usize) {
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.extend_from_slice(&(rows as u32).to_le_bytes());
    output.extend_from_slice(&(dim as u32).to_le_bytes());
}

fn cpu_encode_stats() -> ColumnEncodeStats {
    ColumnEncodeStats {
        backend: "cpu",
        kernel_launches: 0,
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        peak_pinned_staging_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
fn encode_columns_cuda(
    rows: &[&[f32]],
    dim: usize,
    encoded_len: usize,
) -> Result<(Vec<u8>, ColumnEncodeStats)> {
    let (columns, stats) = crate::cuda_olap::with_context(|context| context.transpose_rows(rows))
        .map_err(forge_error)?;
    let mut output = allocate_encoded(encoded_len)?;
    write_header(&mut output, rows.len(), dim);
    if cfg!(target_endian = "little") {
        let payload = unsafe {
            std::slice::from_raw_parts(
                columns.as_ptr().cast::<u8>(),
                columns.len() * size_of::<f32>(),
            )
        };
        output.extend_from_slice(payload);
    } else {
        for value in columns {
            output.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok((
        output,
        ColumnEncodeStats {
            backend: "cuda",
            kernel_launches: stats.kernel_launches,
            host_to_device_bytes: stats.host_to_device_bytes,
            device_to_host_bytes: stats.device_to_host_bytes,
            peak_pinned_staging_bytes: stats.peak_pinned_staging_bytes,
        },
    ))
}

#[cfg(not(feature = "cuda"))]
fn encode_columns_cuda(
    _rows: &[&[f32]],
    _dim: usize,
    _encoded_len: usize,
) -> Result<(Vec<u8>, ColumnEncodeStats)> {
    Err(CalyxError {
        code: "CALYX_FORGE_DEVICE_UNAVAILABLE",
        message: format!(
            "dense column transpose at or above {COLUMN_TRANSPOSE_CUDA_MIN_ELEMENTS} elements requires the CUDA Aster feature"
        ),
        remediation: "build Aster with feature cuda or keep the materialization below the measured CPU/GPU crossover",
    })
}

#[cfg(feature = "cuda")]
fn forge_error(error: calyx_forge::ForgeError) -> CalyxError {
    CalyxError {
        code: error.code(),
        message: error.to_string(),
        remediation: "fix the dense column shape or restore CUDA availability; large transpose never falls back to CPU",
    }
}

pub fn decode_column_shape(bytes: &[u8]) -> Result<ArrowColumnView<'_>> {
    if bytes.len() < HEADER_LEN {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk header missing",
        ));
    }
    if &bytes[0..4] != MAGIC {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk magic mismatch",
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("version"));
    if version != VERSION {
        return Err(CalyxError::aster_corrupt_shard(
            "unsupported arrow chunk version",
        ));
    }
    let n_rows = u32::from_le_bytes(bytes[8..12].try_into().expect("rows")) as usize;
    let dim = u32::from_le_bytes(bytes[12..16].try_into().expect("dim")) as usize;
    if n_rows == 0 || dim == 0 {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk shape must be non-zero",
        ));
    }
    let value_count = n_rows
        .checked_mul(dim)
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk shape overflow"))?;
    let payload_len = value_count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk payload overflow"))?;
    let expected = HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| CalyxError::aster_corrupt_shard("arrow chunk length overflow"))?;
    if bytes.len() != expected {
        return Err(CalyxError::aster_corrupt_shard(
            "arrow chunk byte length mismatch",
        ));
    }
    Ok(ArrowColumnView {
        raw: bytes,
        n_rows,
        dim,
    })
}

pub fn decode_column_chunk(bytes: &[u8]) -> Result<ArrowChunkView<'_>> {
    let column_view = decode_column_shape(bytes)?;
    let n_rows = column_view.n_rows();
    let dim = column_view.dim();
    let value_count = n_rows * dim;
    let payload = &bytes[HEADER_LEN..];
    let mut rows = vec![0.0_f32; value_count];
    for column in 0..dim {
        for row in 0..n_rows {
            let offset = (column * n_rows + row) * 4;
            let value = f32::from_le_bytes(payload[offset..offset + 4].try_into().expect("f32"));
            rows[row * dim + column] = value;
        }
    }
    Ok(ArrowChunkView {
        raw: bytes,
        rows,
        n_rows,
        dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_chunk_roundtrips_and_exposes_magic() {
        let rows = [vec![1.0, 2.0, 3.5, 4.25], vec![5.0, 6.0, 7.0, 8.0]];
        let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
        let bytes = encode_column_chunk(&refs).expect("encode");
        let decoded = decode_column_chunk(&bytes).expect("decode");

        assert_eq!(&bytes[0..4], b"CXA1");
        assert_eq!(&bytes[4..8], &1_u32.to_le_bytes());
        let payload = bytes[HEADER_LEN..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32")))
            .collect::<Vec<_>>();
        assert_eq!(payload, vec![1.0, 5.0, 2.0, 6.0, 3.5, 7.0, 4.25, 8.0]);
        assert_eq!(decoded.n_rows(), 2);
        assert_eq!(decoded.dim(), 4);
        assert_eq!(decoded.row(0).unwrap(), rows[0].as_slice());
        assert_eq!(decoded.raw_bytes(), bytes.as_slice());
        let columns = decode_column_shape(&bytes).expect("column shape");
        let column_one = columns.column_values(1).unwrap().collect::<Vec<_>>();
        assert_eq!(column_one, vec![2.0, 6.0]);
        assert_eq!(columns.value(2, 1).unwrap(), 7.0);
    }

    #[test]
    fn fail_closed_edges() {
        assert!(encode_column_chunk(&[]).is_err());
        assert!(encode_column_chunk(&[&[]]).is_err());
        assert!(encode_column_chunk(&[&[1.0][..], &[1.0, 2.0][..]]).is_err());
        assert!(decode_column_chunk(b"").is_err());
        let mut bad = encode_column_chunk(&[&[1.0][..]]).unwrap();
        bad[0] = 0;
        assert!(decode_column_chunk(&bad).is_err());
        let truncated = &bad[..bad.len() - 1];
        assert!(decode_column_chunk(truncated).is_err());
    }

    #[test]
    fn accelerated_small_shape_keeps_cpu_canonical_bytes() {
        let rows = [vec![1.0, 2.0], vec![3.0, 4.0]];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let canonical = encode_column_chunk(&refs).expect("CPU encode");
        let (accelerated, stats) =
            encode_column_chunk_accelerated(&refs).expect("accelerated dispatch");
        assert_eq!(accelerated, canonical);
        assert_eq!(stats.backend, "cpu");
        assert_eq!(stats.kernel_launches, 0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn large_transpose_without_cuda_fails_closed() {
        let rows = vec![vec![1.0_f32; 64]; COLUMN_TRANSPOSE_CUDA_MIN_ELEMENTS / 64];
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let error = encode_column_chunk_accelerated(&refs)
            .expect_err("large transpose must not run on CPU");
        assert_eq!(error.code, "CALYX_FORGE_DEVICE_UNAVAILABLE");
    }

    proptest! {
        #[test]
        fn chunks_roundtrip_bit_exact(n in 1usize..16, dim in 1usize..32, values in proptest::collection::vec(any::<u32>(), 1..512)) {
            let mut rows = Vec::new();
            let mut cursor = 0;
            for _ in 0..n {
                let mut row = Vec::new();
                for _ in 0..dim {
                    row.push(f32::from_bits(values[cursor % values.len()]));
                    cursor += 1;
                }
                rows.push(row);
            }
            let refs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
            let bytes = encode_column_chunk(&refs).expect("encode");
            let decoded = decode_column_chunk(&bytes).expect("decode");
        for (index, row) in rows.iter().enumerate() {
            let got = decoded.row(index).unwrap();
            let got_bits: Vec<_> = got.iter().map(|value| value.to_bits()).collect();
            let want_bits: Vec<_> = row.iter().map(|value| value.to_bits()).collect();
            prop_assert_eq!(got_bits, want_bits);
        }
        }
    }
}
