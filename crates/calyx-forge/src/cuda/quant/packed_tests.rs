use super::{
    BINARY_CUDA_MIN_ELEMENTS, CudaQuantContext, INT8_CUDA_MIN_ELEMENTS, QuantDispatch,
    binary_dispatch, int8_dispatch,
};
use crate::cuda::{init_cuda, test_lock};
use crate::quant::{BinaryCodec, Quantizer, ScalarInt8Codec, new_seed};
use crate::{ForgeError, Result};

fn fixture(rows: usize, dim: usize, salt: u32) -> Vec<f32> {
    let mut state = salt;
    (0..rows * dim)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 0.5 - 0.25;
            random + (index % 7) as f32 * 0.00390625
        })
        .collect()
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32, label: &str) {
    assert_eq!(left.len(), right.len(), "{label} length");
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "{label}[{index}] left={left} right={right} tolerance={tolerance}"
        );
    }
}

#[test]
fn packed_dispatch_boundaries_are_explicit_and_overflow_safe() {
    for (dispatch, threshold) in [
        (
            binary_dispatch as fn(usize, usize) -> _,
            BINARY_CUDA_MIN_ELEMENTS,
        ),
        (
            int8_dispatch as fn(usize, usize) -> _,
            INT8_CUDA_MIN_ELEMENTS,
        ),
    ] {
        assert_eq!(dispatch(1, 1), QuantDispatch::Cpu);
        assert_eq!(dispatch(1, threshold), QuantDispatch::Cuda);
        assert_eq!(dispatch(usize::MAX, 2), QuantDispatch::Cpu);
    }
}

#[test]
fn binary_cuda_encoding_and_decode_match_cpu_at_edges() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    for dim in [1, 7, 8, 9, 63, 64, 65, 768, 1_536, 2_048] {
        let codec = BinaryCodec::new(new_seed(dim, b"issue-1767-binary-parity"))?;
        let mut input = fixture(3, dim, 0xB100 ^ dim as u32);
        input[..dim].fill(0.0);
        let gpu = quant.encode_binary(&codec, &input)?;
        let actual = gpu.read_encoded()?;
        let expected = input
            .chunks_exact(dim)
            .map(|row| codec.encode(row))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(gpu.rows(), 3);
        assert_eq!(gpu.dim(), dim);
        assert_eq!(gpu.encoded_bytes_per_row(), dim.div_ceil(8));
        assert_eq!(actual.len(), expected.len());
        for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.bytes, expected.bytes,
                "binary bytes dim={dim} row={row}"
            );
            assert_eq!(actual.scale.to_bits(), expected.scale.to_bits());
            assert_eq!(actual.seed_id, expected.seed_id);
            if dim % 8 != 0 {
                let padding_mask = !((1_u16 << (dim % 8)) - 1) as u8;
                assert_eq!(actual.bytes.last().copied().unwrap_or(0) & padding_mask, 0);
            }
        }
        let expected_decoded = expected
            .iter()
            .map(|row| codec.decode(row))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_close(&gpu.decode()?, &expected_decoded, 2e-6, "binary decode");
    }
    Ok(())
}

#[test]
fn int8_cuda_encoding_and_decode_match_cpu_at_edges() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    for dim in [1, 7, 8, 9, 63, 64, 65, 768, 1_536, 2_048] {
        let codec = ScalarInt8Codec::new(dim);
        let mut input = fixture(3, dim, 0x8100 ^ dim as u32);
        input[..dim].fill(0.0);
        if dim == 8 {
            input[dim..2 * dim].copy_from_slice(&[0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.0]);
        }
        let gpu = quant.encode_int8(&codec, &input)?;
        let actual = gpu.read_encoded()?;
        let expected = input
            .chunks_exact(dim)
            .map(|row| codec.encode(row))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(gpu.rows(), 3);
        assert_eq!(gpu.dim(), dim);
        assert_eq!(gpu.encoded_bytes_per_row(), dim);
        for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.bytes, expected.bytes,
                "int8 bytes dim={dim} row={row}"
            );
            assert_eq!(actual.scale.to_bits(), expected.scale.to_bits());
            assert_eq!(actual.seed_id, expected.seed_id);
        }
        let expected_decoded = expected
            .iter()
            .map(|row| codec.decode(row))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(gpu.decode()?, expected_decoded, "int8 decode dim={dim}");
    }
    Ok(())
}

#[test]
fn binary_scores_stay_resident_and_expose_exact_mismatches() -> Result<()> {
    let _guard = test_lock();
    let dim = 65;
    let rows = 2_049;
    let codec = BinaryCodec::new(new_seed(dim, b"issue-1767-binary-score"))?;
    let query = (0..dim)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let mut corpus = fixture(rows, dim, 0x1767);
    corpus[..dim].copy_from_slice(&query);
    corpus[dim..2 * dim].copy_from_slice(&query);
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let gpu_query = quant.encode_binary(&codec, &query)?;
    let gpu_corpus = quant.encode_binary(&codec, &corpus)?;
    let cpu_query = codec.encode(&query)?;
    let cpu_rows = corpus
        .chunks_exact(dim)
        .map(|row| codec.encode(row))
        .collect::<Result<Vec<_>>>()?;

    quant.reset_stats();
    let scores = gpu_corpus.score(&gpu_query)?;
    let resident = quant.stats();
    assert_eq!(resident.kernel_launches, 1);
    assert_eq!(resident.scored_candidates, rows as u64);
    assert_eq!(resident.d2h_bytes, 0);
    let actual = scores.read()?;
    let expected = cpu_rows
        .iter()
        .map(|row| codec.dot_estimate(&cpu_query, row))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(actual, expected);
    let mismatches = scores.read_mismatch_counts()?;
    for (row, (&mismatches, &score)) in mismatches.iter().zip(&actual).enumerate() {
        assert_eq!(
            score,
            1.0 - 2.0 * mismatches as f32 / dim as f32,
            "row {row}"
        );
    }

    quant.reset_stats();
    let top = scores.topk(8)?;
    assert_eq!(top[0].0, 0);
    assert_eq!(top[1].0, 1);
    let stats = quant.stats();
    assert_eq!(stats.kernel_launches, 1);
    assert!(stats.d2h_bytes < rows as u64 * size_of::<f32>() as u64);
    Ok(())
}

#[test]
fn int8_scores_stay_resident_and_match_cpu_dot() -> Result<()> {
    let _guard = test_lock();
    let dim = 65;
    let rows = 2_049;
    let codec = ScalarInt8Codec::new(dim);
    let query = (0..dim)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let mut corpus = fixture(rows, dim, 0x8176);
    corpus[..dim].copy_from_slice(&query);
    corpus[dim..2 * dim].copy_from_slice(&query);
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let gpu_query = quant.encode_int8(&codec, &query)?;
    let gpu_corpus = quant.encode_int8(&codec, &corpus)?;
    let cpu_query = codec.encode(&query)?;
    let cpu_rows = corpus
        .chunks_exact(dim)
        .map(|row| codec.encode(row))
        .collect::<Result<Vec<_>>>()?;

    quant.reset_stats();
    let scores = gpu_corpus.score(&gpu_query)?;
    assert_eq!(quant.stats().d2h_bytes, 0);
    let actual = scores.read()?;
    let expected = cpu_rows
        .iter()
        .map(|row| codec.dot_estimate(&cpu_query, row))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(actual, expected);
    quant.reset_stats();
    let top = scores.topk(8)?;
    assert_eq!(top[0].0, 0);
    assert_eq!(top[1].0, 1);
    let stats = quant.stats();
    assert_eq!(stats.kernel_launches, 1);
    assert!(stats.d2h_bytes < rows as u64 * size_of::<f32>() as u64);
    Ok(())
}

#[test]
fn packed_cuda_rejects_empty_bad_values_oversize_and_bad_pairs() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let binary = BinaryCodec::new(new_seed(8, b"issue-1767-errors"))?;
    let int8 = ScalarInt8Codec::new(8);
    assert!(matches!(
        quant.encode_binary(&binary, &[]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        quant.encode_int8(&int8, &[0.0; 7]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        quant.encode_int8(&ScalarInt8Codec::new(0), &[]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let mut nonfinite = [0.0; 8];
    nonfinite[3] = f32::INFINITY;
    assert!(matches!(
        quant.encode_binary(&binary, &nonfinite),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    assert!(matches!(
        quant.encode_int8(&int8, &nonfinite),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    let oversized = BinaryCodec::new(new_seed(4_097, b"issue-1767-oversized"))?;
    assert!(matches!(
        quant.encode_binary(&oversized, &vec![0.0; 4_097]),
        Err(ForgeError::ShapeMismatch { .. })
    ));

    let binary_corpus = quant.encode_binary(&binary, &[0.0; 8])?;
    let binary_multi = quant.encode_binary(&binary, &[0.0; 16])?;
    assert!(binary_corpus.score(&binary_multi).is_err());
    let other_binary = BinaryCodec::new(new_seed(8, b"issue-1767-other-seed"))?;
    let other_binary = quant.encode_binary(&other_binary, &[0.0; 8])?;
    assert!(binary_corpus.score(&other_binary).is_err());
    let int8_corpus = quant.encode_int8(&int8, &[0.0; 8])?;
    let int8_multi = quant.encode_int8(&int8, &[0.0; 16])?;
    assert!(int8_corpus.score(&int8_multi).is_err());
    let other_dim = quant.encode_int8(&ScalarInt8Codec::new(7), &[0.0; 7])?;
    assert!(int8_corpus.score(&other_dim).is_err());
    Ok(())
}
