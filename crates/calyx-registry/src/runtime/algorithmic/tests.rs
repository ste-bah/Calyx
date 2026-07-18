use super::*;

#[test]
fn byte_features_are_bit_deterministic() {
    let lens = AlgorithmicLens::byte_features("byte-fsv", Modality::Text);
    let input = Input::new(Modality::Text, b"Calyx PH17: 2+2=4\n".to_vec());

    let first = lens.measure(&input).unwrap();
    let second = lens.measure(&input).unwrap();

    assert_eq!(first, second);
}

#[test]
fn empty_input_emits_real_dense_vector() {
    let lens = AlgorithmicLens::byte_features("byte-empty", Modality::Text);
    let input = Input::new(Modality::Text, Vec::new());
    let vector = lens.measure(&input).unwrap();
    let bytes = serde_json::to_vec(&vector).unwrap();

    println!(
        "ALGORITHMIC_EMPTY_BYTES={}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        vector,
        SlotVector::Dense {
            dim: BYTE_FEATURE_DIM,
            data: {
                let mut data = vec![0.0; BYTE_FEATURE_DIM as usize];
                data[0] = 1.0;
                data
            }
        }
    );
}

#[test]
fn scalar_feature_is_centered_for_cosine_assay() {
    let lens = AlgorithmicLens::scalar("scalar-fsv", Modality::Structured);
    let low = Input::new(Modality::Structured, b"!!!!!!!!!!!!!!!!".to_vec());
    let high = Input::new(Modality::Structured, b"zzzzzzzzzzzzzzzz".to_vec());

    let low = lens.measure(&low).unwrap();
    let high = lens.measure(&high).unwrap();

    assert!(matches!(low, SlotVector::Dense { data, .. } if data[0] < 0.0));
    assert!(matches!(high, SlotVector::Dense { data, .. } if data[0] > 0.0));
}

#[test]
fn algorithmic_fsv_determinism_probe() {
    let lens = AlgorithmicLens::byte_features("byte-fsv", Modality::Text);
    let input = Input::new(Modality::Text, b"Calyx registry manual FSV".to_vec());
    let first = lens.measure(&input).unwrap();
    let second = lens.measure(&input).unwrap();
    let first_bytes = serde_json::to_vec(&first).unwrap();
    let second_bytes = serde_json::to_vec(&second).unwrap();

    println!("ALGORITHMIC_FSV_DIGEST={}", digest_hex(&first_bytes));
    println!(
        "ALGORITHMIC_FSV_BYTES={}",
        String::from_utf8_lossy(&first_bytes)
    );
    assert_eq!(first_bytes, second_bytes);
}

#[test]
fn small_batches_persist_cpu_provider_evidence() {
    let lens = AlgorithmicLens::byte_features("byte-small-batch", Modality::Text);
    let inputs = vec![
        Input::new(Modality::Text, Vec::new()),
        Input::new(Modality::Text, vec![0, 0xff, b'\n', b'{', b'9']),
        Input::new(Modality::Text, vec![b'a'; 65]),
    ];
    let expected = inputs
        .iter()
        .map(|input| lens.measure_cpu(input).unwrap())
        .collect::<Vec<_>>();

    let actual = lens.measure_batch(&inputs).unwrap();
    let stats = lens.last_batch_stats().expect("batch stats persisted");

    assert_eq!(actual, expected);
    assert_eq!(stats.provider, AlgorithmicBatchProvider::Cpu);
    assert_eq!(stats.rows, inputs.len() as u64);
    assert_eq!(stats.input_bytes, 70);
    assert_eq!(stats.host_to_device_bytes, 0);
    assert_eq!(stats.kernel_launches, 0);
    let encoded = serde_json::to_value(stats).expect("stats serialize");
    assert_eq!(encoded["provider"], "cpu");
}

#[test]
fn malformed_utf8_and_hash_boundaries_remain_cpu_bit_exact() {
    let sparse = AlgorithmicLens::sparse_keywords("sparse-boundary", Modality::Text, 65_537);
    let token = AlgorithmicLens::token_hash("token-boundary", Modality::Text, 17);
    let inputs = vec![
        Input::new(Modality::Text, Vec::new()),
        Input::new(Modality::Text, vec![0xff, b' ', 0xfe, b'\t', b'x']),
        Input::new(Modality::Text, vec![b'a'; 63]),
        Input::new(Modality::Text, vec![b'b'; 64]),
        Input::new(Modality::Text, vec![b'c'; 65]),
    ];

    for lens in [&sparse, &token] {
        let expected = inputs
            .iter()
            .map(|input| lens.measure_cpu(input).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(lens.measure_batch(&inputs).unwrap(), expected);
        assert_eq!(
            lens.last_batch_stats().unwrap().provider,
            AlgorithmicBatchProvider::Cpu
        );
    }
}

#[test]
fn batches_below_each_measured_crossover_stay_on_cpu() {
    let byte = AlgorithmicLens::byte_features("byte-below-crossover", Modality::Text);
    byte.measure_batch(&[Input::new(
        Modality::Text,
        vec![b'x'; BYTE_FEATURES_CUDA_MIN_INPUT_BYTES - 1],
    )])
    .unwrap();
    assert_eq!(
        byte.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cpu
    );

    let sparse = AlgorithmicLens::sparse_keywords("sparse-below-crossover", Modality::Text, 4096);
    let sparse_input = (0..SPARSE_KEYWORDS_CUDA_MIN_TOKENS - 1)
        .map(|index| format!("k{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    sparse
        .measure_batch(&[Input::new(Modality::Text, sparse_input.into_bytes())])
        .unwrap();
    assert_eq!(
        sparse.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cpu
    );

    let token = AlgorithmicLens::token_hash(
        "token-below-crossover",
        Modality::Text,
        (TOKEN_HASH_CUDA_MIN_WORDS - 1) as u32,
    );
    token
        .measure_batch(&[Input::new(Modality::Text, b"token".to_vec())])
        .unwrap();
    assert_eq!(
        token.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cpu
    );
}

#[cfg(feature = "cuda")]
#[test]
fn batches_at_each_measured_crossover_route_to_cuda() {
    let byte = AlgorithmicLens::byte_features("byte-at-crossover", Modality::Text);
    byte.measure_batch(&[Input::new(
        Modality::Text,
        vec![b'x'; BYTE_FEATURES_CUDA_MIN_INPUT_BYTES],
    )])
    .unwrap();
    assert_eq!(
        byte.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cuda
    );

    let sparse = AlgorithmicLens::sparse_keywords("sparse-at-crossover", Modality::Text, 4096);
    let sparse_input = (0..SPARSE_KEYWORDS_CUDA_MIN_TOKENS)
        .map(|index| format!("k{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    sparse
        .measure_batch(&[Input::new(Modality::Text, sparse_input.into_bytes())])
        .unwrap();
    assert_eq!(
        sparse.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cuda
    );

    let token = AlgorithmicLens::token_hash(
        "token-at-crossover",
        Modality::Text,
        TOKEN_HASH_CUDA_MIN_WORDS as u32,
    );
    token
        .measure_batch(&[Input::new(Modality::Text, b"token".to_vec())])
        .unwrap();
    assert_eq!(
        token.last_batch_stats().unwrap().provider,
        AlgorithmicBatchProvider::Cuda
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_byte_batch_is_bit_exact_with_one_launch() {
    let lens = AlgorithmicLens::byte_features("byte-cuda-batch", Modality::Text);
    let mut inputs = (0..128)
        .map(|row| {
            let mut bytes = (0..1024)
                .map(|col| ((row * 17 + col * 31) & 0xff) as u8)
                .collect::<Vec<_>>();
            bytes.extend_from_slice(&[0, 0xff, b'\n', b'\\', b'{']);
            Input::new(Modality::Text, bytes)
        })
        .collect::<Vec<_>>();
    inputs.extend([
        Input::new(Modality::Text, Vec::new()),
        Input::new(Modality::Text, vec![0xff]),
        Input::new(Modality::Text, vec![b'x'; 64]),
        Input::new(Modality::Text, vec![b'y'; 65]),
    ]);
    let expected = inputs
        .iter()
        .map(|input| lens.measure_cpu(input).unwrap())
        .collect::<Vec<_>>();

    let actual = lens.measure_batch(&inputs).unwrap();
    let stats = lens.last_batch_stats().unwrap();
    println!(
        "ALGORITHMIC_CUDA_BYTE_STATS={}",
        serde_json::to_string(&stats).unwrap()
    );

    assert_eq!(actual, expected);
    assert_eq!(stats.provider, AlgorithmicBatchProvider::Cuda);
    assert_eq!(stats.kernel_launches, 1);
    assert!(stats.host_to_device_bytes >= stats.input_bytes);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sparse_batch_matches_blake3_for_ragged_non_utf8_tokens() {
    let lens = AlgorithmicLens::sparse_keywords("sparse-cuda-batch", Modality::Text, 65_537);
    let inputs = (0..8)
        .map(|row| {
            let mut bytes = Vec::new();
            if row == 0 {
                for boundary in [1, 56, 57, 1_016] {
                    if !bytes.is_empty() {
                        bytes.push(b' ');
                    }
                    bytes.extend(std::iter::repeat_n(b'q', boundary));
                }
            }
            for term in 0..320 {
                if !bytes.is_empty() {
                    bytes.push(if term % 2 == 0 { b' ' } else { b'\n' });
                }
                bytes.extend_from_slice(format!("r{row}-term-{term:04}").as_bytes());
            }
            if row == 0 {
                bytes.extend_from_slice(&[b' ', 0xff, 0xfe]);
            }
            Input::new(Modality::Text, bytes)
        })
        .collect::<Vec<_>>();
    let expected = inputs
        .iter()
        .map(|input| lens.measure_cpu(input).unwrap())
        .collect::<Vec<_>>();

    let actual = lens.measure_batch(&inputs).unwrap();
    let stats = lens.last_batch_stats().unwrap();
    println!(
        "ALGORITHMIC_CUDA_SPARSE_STATS={}",
        serde_json::to_string(&stats).unwrap()
    );

    assert_eq!(actual, expected);
    assert_eq!(stats.provider, AlgorithmicBatchProvider::Cuda);
    assert_eq!(stats.kernel_launches, 1);
    assert!(stats.work_items >= SPARSE_KEYWORDS_CUDA_MIN_TOKENS as u64);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_token_hash_batch_matches_blake3_across_digest_boundaries() {
    let lens = AlgorithmicLens::token_hash("token-cuda-batch", Modality::Text, 129);
    let inputs = (0..4)
        .map(|row| {
            let mut terms = (0..32)
                .map(|term| format!("row-{row}-token-{term:02}"))
                .collect::<Vec<_>>();
            if row == 0 {
                for (index, boundary) in [1, 28, 29, 30, 989].into_iter().enumerate() {
                    terms[index] = "q".repeat(boundary);
                }
            }
            let text = terms.join(" \t");
            Input::new(Modality::Text, text.into_bytes())
        })
        .collect::<Vec<_>>();
    let expected = inputs
        .iter()
        .map(|input| lens.measure_cpu(input).unwrap())
        .collect::<Vec<_>>();

    let actual = lens.measure_batch(&inputs).unwrap();
    let stats = lens.last_batch_stats().unwrap();
    println!(
        "ALGORITHMIC_CUDA_TOKEN_STATS={}",
        serde_json::to_string(&stats).unwrap()
    );

    assert_eq!(actual, expected);
    assert_eq!(stats.provider, AlgorithmicBatchProvider::Cuda);
    assert_eq!(stats.kernel_launches, 1);
    assert!(stats.work_items >= TOKEN_HASH_CUDA_MIN_WORDS.div_ceil(8) as u64);
}

#[cfg(feature = "cuda")]
#[test]
fn overlong_single_chunk_token_uses_documented_cpu_path() {
    let lens = AlgorithmicLens::token_hash("token-long", Modality::Text, 8192);
    let inputs = vec![Input::new(Modality::Text, vec![b'z'; 990])];
    let expected = vec![lens.measure_cpu(&inputs[0]).unwrap()];

    assert_eq!(lens.measure_batch(&inputs).unwrap(), expected);
    let stats = lens.last_batch_stats().unwrap();
    assert_eq!(stats.provider, AlgorithmicBatchProvider::Cpu);
    assert!(stats.reason.contains("single-chunk"));
}

fn digest_hex(bytes: &[u8]) -> String {
    calyx_core::content_address([bytes])
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
