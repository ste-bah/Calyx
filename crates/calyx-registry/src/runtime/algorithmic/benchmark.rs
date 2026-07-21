use std::hint::black_box;
use std::time::Instant;

use calyx_core::content_address;
use serde::Serialize;

use super::batch::{
    BYTE_FEATURES_CUDA_MIN_INPUT_BYTES, SPARSE_KEYWORDS_CUDA_MIN_TOKENS, TOKEN_HASH_CUDA_MIN_WORDS,
};
use super::cpu::{
    byte_features, byte_features_from_raw, hash_part, token_vector, token_vectors_from_words,
};

const SAMPLES: usize = 7;

#[derive(Debug, Serialize)]
struct CrossoverRow {
    encoder: &'static str,
    work_items: usize,
    input_bytes: usize,
    cpu_us: f64,
    cuda_us: f64,
    speedup: f64,
}

#[derive(Debug, Serialize)]
struct CrossoverReport {
    samples: usize,
    byte_threshold: usize,
    sparse_threshold: usize,
    token_word_threshold: usize,
    rows: Vec<CrossoverRow>,
}

#[test]
#[ignore = "manual release-mode CUDA crossover FSV"]
fn algorithmic_cuda_crossover_probe() {
    let context = calyx_forge::CudaAlgorithmicContext::new(0).expect("CUDA context");
    let mut report = CrossoverReport {
        samples: SAMPLES,
        byte_threshold: BYTE_FEATURES_CUDA_MIN_INPUT_BYTES,
        sparse_threshold: SPARSE_KEYWORDS_CUDA_MIN_TOKENS,
        token_word_threshold: TOKEN_HASH_CUDA_MIN_WORDS,
        rows: Vec::new(),
    };
    benchmark_bytes(&context, &mut report.rows);
    benchmark_sparse(&context, &mut report.rows);
    benchmark_tokens(&context, &mut report.rows);
    println!(
        "ALGORITHMIC_CUDA_CROSSOVER_JSON={}",
        serde_json::to_string(&report).expect("serialize crossover report")
    );
}

fn benchmark_bytes(context: &calyx_forge::CudaAlgorithmicContext, report: &mut Vec<CrossoverRow>) {
    for total in [4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304] {
        let rows = byte_rows(total, 64);
        let refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let ragged = calyx_forge::CudaByteRaggedBatch::from_slices(&refs).unwrap();
        let expected = rows
            .iter()
            .map(|row| byte_features(row))
            .collect::<Vec<_>>();
        let (raw, _) = context.byte_features_raw(&ragged).unwrap();
        let actual = raw
            .into_iter()
            .map(|row| byte_features_from_raw(row.values))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        let cpu_us = median_us(|| {
            black_box(
                rows.iter()
                    .map(|row| byte_features(row))
                    .collect::<Vec<_>>(),
            );
        });
        let cuda_us = median_us(|| {
            let (raw, _) = context.byte_features_raw(&ragged).unwrap();
            black_box(
                raw.into_iter()
                    .map(|row| byte_features_from_raw(row.values))
                    .collect::<Vec<_>>(),
            );
        });
        report.push(row("byte_features", total, total, cpu_us, cuda_us));
    }
}

fn benchmark_sparse(context: &calyx_forge::CudaAlgorithmicContext, report: &mut Vec<CrossoverRow>) {
    for count in [128, 512, 2_048, 8_192, 32_768] {
        let tokens = (0..count)
            .map(|index| format!("keyword-{index:08x}").into_bytes())
            .collect::<Vec<_>>();
        let refs = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let ragged = calyx_forge::CudaByteRaggedBatch::from_slices(&refs).unwrap();
        let expected = sparse_cpu_hashes(&tokens);
        assert_eq!(context.sparse_keyword_hashes(&ragged).unwrap().0, expected);

        let cpu_us = median_us(|| {
            black_box(sparse_cpu_hashes(&tokens));
        });
        let cuda_us = median_us(|| {
            black_box(context.sparse_keyword_hashes(&ragged).unwrap().0);
        });
        report.push(row(
            "sparse_keywords",
            count,
            ragged.input_bytes(),
            cpu_us,
            cuda_us,
        ));
    }
}

fn benchmark_tokens(context: &calyx_forge::CudaAlgorithmicContext, report: &mut Vec<CrossoverRow>) {
    const DIM: u32 = 128;
    for count in [4, 16, 64, 256, 1_024] {
        let tokens = (0..count)
            .map(|index| format!("token-{index:08x}").into_bytes())
            .collect::<Vec<_>>();
        let refs = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let ragged = calyx_forge::CudaByteRaggedBatch::from_slices(&refs).unwrap();
        let expected = tokens
            .iter()
            .map(|token| token_vector(token, DIM))
            .collect::<Vec<_>>();
        let raw = context.token_hash_words(&ragged, DIM).unwrap().0;
        assert_eq!(token_vectors_from_words(&raw, DIM), expected);

        let cpu_us = median_us(|| {
            black_box(
                tokens
                    .iter()
                    .map(|token| token_vector(token, DIM))
                    .collect::<Vec<_>>(),
            );
        });
        let cuda_us = median_us(|| {
            let raw = context.token_hash_words(&ragged, DIM).unwrap().0;
            black_box(
                raw.chunks_exact(DIM as usize)
                    .map(|words| words.iter().copied().map(hash_part).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            );
        });
        report.push(row(
            "token_hash",
            count * DIM as usize,
            ragged.input_bytes(),
            cpu_us,
            cuda_us,
        ));
    }
}

fn sparse_cpu_hashes(tokens: &[Vec<u8>]) -> Vec<u32> {
    tokens
        .iter()
        .map(|token| {
            let digest = content_address([token.as_slice()]);
            u32::from_be_bytes(digest[..4].try_into().expect("digest word"))
        })
        .collect()
}

fn byte_rows(total: usize, row_count: usize) -> Vec<Vec<u8>> {
    let width = total / row_count;
    (0..row_count)
        .map(|row| {
            (0..width)
                .map(|col| ((row * 17 + col * 31) & 0xff) as u8)
                .collect()
        })
        .collect()
}

fn median_us(mut operation: impl FnMut()) -> f64 {
    operation();
    let mut elapsed = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            operation();
            start.elapsed().as_secs_f64() * 1_000_000.0
        })
        .collect::<Vec<_>>();
    elapsed.sort_by(f64::total_cmp);
    elapsed[SAMPLES / 2]
}

fn row(
    encoder: &'static str,
    work_items: usize,
    input_bytes: usize,
    cpu_us: f64,
    cuda_us: f64,
) -> CrossoverRow {
    CrossoverRow {
        encoder,
        work_items,
        input_bytes,
        cpu_us,
        cuda_us,
        speedup: cpu_us / cuda_us,
    }
}
