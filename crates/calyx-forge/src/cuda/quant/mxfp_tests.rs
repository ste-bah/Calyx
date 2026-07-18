use super::{
    CudaMxFpBatch, CudaQuantContext, MXFP4_CUDA_MIN_ELEMENTS, MXFP8_CUDA_MIN_ELEMENTS,
    QuantDispatch, mxfp4_dispatch, mxfp8_dispatch,
};
use crate::cuda::{init_cuda, test_lock};
use crate::quant::{AssayQuantSafety, MxFp4Codec, QuantLevel, QuantizedVec, Quantizer};
use crate::{ForgeError, Result};

fn passing_safety() -> AssayQuantSafety {
    AssayQuantSafety {
        baseline_bits: 1.0,
        quantized_bits: 0.97,
        cosine: 0.995,
        far_delta: 0.005,
    }
}

fn fixture(rows: usize, dim: usize, salt: u32) -> Vec<f32> {
    let mut state = salt;
    (0..rows * dim)
        .map(|index| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let random = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0;
            random + (index % 7) as f32 * 0.00390625
        })
        .collect()
}

fn add_special_values(input: &mut [f32], dim: usize) {
    let values = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0 / 1024.0,
        -1.0 / 1024.0,
        1.0625,
        -1.0625,
        1.5 / 7.0,
        -1.5 / 7.0,
        f32::MAX,
        -f32::MAX,
    ];
    for (slot, value) in input[dim..2 * dim].iter_mut().zip(values) {
        *slot = value;
    }
}

fn cpu_fp4(codec: &MxFp4Codec, input: &[f32], dim: usize) -> Result<Vec<QuantizedVec>> {
    let safety = passing_safety();
    input
        .chunks_exact(dim)
        .map(|row| codec.encode_assay_checked("slot:issue-1768", row, &safety))
        .collect()
}

fn cpu_fp8(codec: &MxFp4Codec, input: &[f32], dim: usize) -> Result<Vec<QuantizedVec>> {
    input
        .chunks_exact(dim)
        .map(|row| codec.encode_mxfp8(row))
        .collect()
}

fn decoded_rows(codec: &MxFp4Codec, rows: &[QuantizedVec]) -> Result<Vec<f32>> {
    Ok(rows
        .iter()
        .map(|row| codec.decode(row))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect())
}

#[test]
fn mxfp_dispatch_boundaries_are_explicit_and_overflow_safe() {
    for (dispatch, threshold) in [
        (
            mxfp4_dispatch as fn(usize, usize) -> QuantDispatch,
            MXFP4_CUDA_MIN_ELEMENTS,
        ),
        (
            mxfp8_dispatch as fn(usize, usize) -> QuantDispatch,
            MXFP8_CUDA_MIN_ELEMENTS,
        ),
    ] {
        assert_eq!(dispatch(1, 1), QuantDispatch::Cpu);
        assert_eq!(dispatch(1, threshold), QuantDispatch::Cuda);
        assert_eq!(dispatch(usize::MAX, 2), QuantDispatch::Cpu);
    }
}

#[test]
fn mxfp_cuda_encode_decode_matches_cpu_bytes_and_special_values() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let safety = passing_safety();
    for dim in [1, 31, 32, 33, 65, 768, 1_536, 2_048] {
        let codec = MxFp4Codec::new(dim);
        let mut input = fixture(3, dim, 0x1768 ^ dim as u32);
        input[..dim].fill(0.0);
        add_special_values(&mut input, dim);
        let gpu4 = quant.encode_mxfp4(&codec, "slot:issue-1768", &safety, &input)?;
        let gpu8 = quant.encode_mxfp8(&codec, &input)?;
        let cpu4 = cpu_fp4(&codec, &input, dim)?;
        let cpu8 = cpu_fp8(&codec, &input, dim)?;
        assert_eq!(gpu4.rows(), 3);
        assert_eq!(gpu4.level(), QuantLevel::Bits4Fp);
        assert_eq!(gpu4.encoded_bytes_per_row(), dim.div_ceil(32) * 17);
        assert_eq!(gpu8.level(), QuantLevel::Bits8Fp);
        assert_eq!(gpu8.encoded_bytes_per_row(), dim.div_ceil(32) * 33);
        assert_eq!(gpu4.read_encoded()?, cpu4, "MXFP4 bytes dim={dim}");
        assert_eq!(gpu8.read_encoded()?, cpu8, "MXFP8 bytes dim={dim}");
        assert_eq!(
            gpu4.decode()?,
            decoded_rows(&codec, &cpu4)?,
            "MXFP4 decode dim={dim}"
        );
        assert_eq!(
            gpu8.decode()?,
            decoded_rows(&codec, &cpu8)?,
            "MXFP8 decode dim={dim}"
        );
    }
    Ok(())
}

#[test]
fn mxfp_packed_upload_roundtrips_and_rejects_malformed_payloads() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let dim = 33;
    let codec = MxFp4Codec::new(dim);
    let input = fixture(2, dim, 0xB10C);
    let fp4 = cpu_fp4(&codec, &input, dim)?;
    let fp8 = cpu_fp8(&codec, &input, dim)?;
    let uploaded4 = quant.upload_mxfp(&codec, &fp4)?;
    let uploaded8 = quant.upload_mxfp(&codec, &fp8)?;
    assert_eq!(uploaded4.read_encoded()?, fp4);
    assert_eq!(uploaded8.read_encoded()?, fp8);
    assert_eq!(uploaded4.decode()?, decoded_rows(&codec, &fp4)?);
    assert_eq!(uploaded8.decode()?, decoded_rows(&codec, &fp8)?);

    let mut reserved = fp4[0].clone();
    reserved.bytes[0] = (reserved.bytes[0] & 0xf0) | 0x0f;
    let reserved_gpu = quant.upload_mxfp(&codec, &[reserved.clone()])?;
    assert_eq!(reserved_gpu.decode()?, codec.decode(&reserved)?);
    assert_eq!(reserved_gpu.decode()?[0], 0.0);

    let mut malformed = fp4[0].clone();
    malformed.bytes.pop();
    assert_quant_error(quant.upload_mxfp(&codec, &[malformed]));
    let mut metadata = fp4[0].clone();
    metadata.scale = 1.0;
    assert_quant_error(quant.upload_mxfp(&codec, &[metadata]));
    let mut scale = fp4[0].clone();
    scale.bytes[16] = u8::MAX;
    assert_quant_error(quant.upload_mxfp(&codec, &[scale]));
    let mut padding4 = fp4[0].clone();
    padding4.bytes[17] &= 0x0f;
    assert_quant_error(quant.upload_mxfp(&codec, &[padding4]));
    let mut padding8 = fp8[0].clone();
    padding8.bytes[34] = 1;
    assert_quant_error(quant.upload_mxfp(&codec, &[padding8]));
    assert!(matches!(
        quant.upload_mxfp(&codec, &[fp4[0].clone(), fp8[0].clone()]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    Ok(())
}

fn assert_quant_error(result: Result<CudaMxFpBatch>) {
    assert!(matches!(result, Err(ForgeError::QuantError { .. })));
}

#[test]
fn mxfp_all_dot_combinations_match_cpu_rank_and_stay_resident() -> Result<()> {
    let _guard = test_lock();
    let dim = 65;
    let rows = 2_049;
    let codec = MxFp4Codec::new(dim);
    let safety = passing_safety();
    let query = (0..dim)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
        .collect::<Vec<_>>();
    let mut corpus = fixture(rows, dim, 0xC0DE);
    corpus.iter_mut().for_each(|value| *value *= 0.25);
    corpus[..dim].copy_from_slice(&query);
    corpus[dim..2 * dim].copy_from_slice(&query);
    let cpu_query4 = codec.encode_assay_checked("slot:query", &query, &safety)?;
    let cpu_query8 = codec.encode_mxfp8(&query)?;
    let cpu_corpus4 = cpu_fp4(&codec, &corpus, dim)?;
    let cpu_corpus8 = cpu_fp8(&codec, &corpus, dim)?;
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let gpu_query4 = quant.encode_mxfp4(&codec, "slot:query", &safety, &query)?;
    let gpu_query8 = quant.encode_mxfp8(&codec, &query)?;
    let gpu_corpus4 = quant.encode_mxfp4(&codec, "slot:corpus", &safety, &corpus)?;
    let gpu_corpus8 = quant.encode_mxfp8(&codec, &corpus)?;

    assert_score_combo(
        &quant,
        &codec,
        &gpu_corpus4,
        &gpu_query4,
        &cpu_corpus4,
        &cpu_query4,
    )?;
    assert_score_combo(
        &quant,
        &codec,
        &gpu_corpus8,
        &gpu_query8,
        &cpu_corpus8,
        &cpu_query8,
    )?;
    assert_score_combo(
        &quant,
        &codec,
        &gpu_corpus8,
        &gpu_query4,
        &cpu_corpus8,
        &cpu_query4,
    )?;
    assert_score_combo(
        &quant,
        &codec,
        &gpu_corpus4,
        &gpu_query8,
        &cpu_corpus4,
        &cpu_query8,
    )?;
    Ok(())
}

fn assert_score_combo(
    quant: &CudaQuantContext,
    codec: &MxFp4Codec,
    gpu_corpus: &CudaMxFpBatch,
    gpu_query: &CudaMxFpBatch,
    cpu_corpus: &[QuantizedVec],
    cpu_query: &QuantizedVec,
) -> Result<()> {
    quant.reset_stats();
    let scores = gpu_corpus.score(gpu_query)?;
    assert_eq!(quant.stats().d2h_bytes, 0);
    let actual = scores.read()?;
    let expected = cpu_corpus
        .iter()
        .map(|row| codec.dot_estimate(cpu_query, row))
        .collect::<Result<Vec<_>>>()?;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let tolerance = 2e-6 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "MXFP score {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
    quant.reset_stats();
    let gpu_top = scores.topk(8)?;
    let cpu_top = host_topk(&expected, 8);
    assert_eq!(
        gpu_top.iter().map(|pair| pair.0).collect::<Vec<_>>(),
        cpu_top.iter().map(|pair| pair.0).collect::<Vec<_>>()
    );
    assert_eq!(gpu_top[0].0, 0);
    assert_eq!(gpu_top[1].0, 1);
    assert!(quant.stats().d2h_bytes < actual.len() as u64 * size_of::<f32>() as u64);
    Ok(())
}

fn host_topk(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs = scores.iter().copied().enumerate().collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    pairs.truncate(k.min(pairs.len()));
    pairs
}

#[test]
fn mxfp_cuda_rejects_bad_shapes_values_safety_and_pairs() -> Result<()> {
    let _guard = test_lock();
    let quant = CudaQuantContext::new(init_cuda(0, false)?);
    let codec = MxFp4Codec::new(8);
    let safety = passing_safety();
    assert!(matches!(
        quant.encode_mxfp8(&codec, &[]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        quant.encode_mxfp8(&codec, &[0.0; 7]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let mut nonfinite = [0.0; 8];
    nonfinite[3] = f32::NAN;
    assert!(matches!(
        quant.encode_mxfp8(&codec, &nonfinite),
        Err(ForgeError::NumericalInvariant { .. })
    ));
    let bad_safety = AssayQuantSafety {
        cosine: 0.5,
        ..safety.clone()
    };
    assert!(matches!(
        quant.encode_mxfp4(&codec, "slot:unsafe", &bad_safety, &[0.0; 8]),
        Err(ForgeError::QuantIntelligenceLoss { .. })
    ));
    let oversized = MxFp4Codec::new(4_097);
    assert!(matches!(
        quant.encode_mxfp8(&oversized, &vec![0.0; 4_097]),
        Err(ForgeError::ShapeMismatch { .. })
    ));
    let corpus = quant.encode_mxfp4(&codec, "slot:corpus", &safety, &[0.0; 8])?;
    let multi = quant.encode_mxfp8(&codec, &[0.0; 16])?;
    assert!(corpus.score(&multi).is_err());
    let other = quant.encode_mxfp8(&MxFp4Codec::new(7), &[0.0; 7])?;
    assert!(corpus.score(&other).is_err());
    Ok(())
}
