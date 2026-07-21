use calyx_core::{AbsentReason, SlotVector, SparseEntry};

use super::encode::{
    EncodedMultiSlotVector, EncodedSlotVectorShape, WriteRow, decode_slot_vector,
    decode_write_batch, encode_slot_vector, encode_write_batch, inspect_slot_vector,
};
use crate::cf::ColumnFamily;

#[test]
fn slot_vector_codec_round_trips_every_shape_and_consumes_all_bytes() {
    let vectors = [
        SlotVector::Dense {
            dim: 2,
            data: vec![1.25, -2.5],
        },
        SlotVector::Absent {
            reason: AbsentReason::Error("lens failed explicitly".to_string()),
        },
        SlotVector::Sparse {
            dim: 16,
            entries: vec![SparseEntry { idx: 3, val: 0.75 }],
        },
        SlotVector::Multi {
            token_dim: 2,
            tokens: vec![vec![1.0, 0.0], vec![0.25, 0.75]],
        },
    ];

    let mut evidence = Vec::new();
    for vector in vectors {
        let bytes = encode_slot_vector(&vector).expect("encode valid slot vector");
        let shape = inspect_slot_vector(&bytes).expect("inspect valid slot vector");
        let decoded = decode_slot_vector(&bytes).expect("decode valid slot vector");
        assert_eq!(decoded, vector);
        evidence.push(format!("{shape:?}:{}", bytes.len()));
    }

    assert!(matches!(
        inspect_slot_vector(
            &encode_slot_vector(&SlotVector::Multi {
                token_dim: 2,
                tokens: vec![vec![1.0, 0.0]],
            })
            .unwrap()
        )
        .unwrap(),
        EncodedSlotVectorShape::Multi {
            token_dim: 2,
            token_count: 1
        }
    ));
    println!("ISSUE1604_SLOT_CODEC_HAPPY {evidence:?}");
}

#[test]
fn encoded_multi_view_streams_exact_components_without_decoding() {
    let vector = SlotVector::Multi {
        token_dim: 2,
        tokens: vec![vec![1.25, -2.5], vec![3.75, 4.5]],
    };
    let encoded = encode_slot_vector(&vector).unwrap();
    let view = EncodedMultiSlotVector::new(&encoded).unwrap();

    assert_eq!(view.token_dim(), 2);
    assert_eq!(view.token_count(), 2);
    assert_eq!(
        view.components().collect::<Vec<_>>(),
        vec![1.25, -2.5, 3.75, 4.5]
    );
}

#[test]
fn encoded_multi_view_rejects_non_multi_and_malformed_payloads() {
    let dense = encode_slot_vector(&SlotVector::Dense {
        dim: 1,
        data: vec![1.0],
    })
    .unwrap();
    let wrong_shape = EncodedMultiSlotVector::new(&dense).unwrap_err();
    let truncated = EncodedMultiSlotVector::new(&[3, 0, 0, 0, 2, 0, 0, 0, 1]).unwrap_err();

    assert!(wrong_shape.message.contains("not a multi-vector"));
    assert!(truncated.message.contains("byte length mismatch"));
}

#[test]
fn slot_vector_codec_rejects_unbounded_counts_before_allocation() {
    let cases = [
        ("dense", encoded_header(0, u32::MAX, None)),
        ("sparse", encoded_header(2, 16, Some(u32::MAX))),
        ("multi", encoded_header(3, u32::MAX, Some(u32::MAX))),
    ];

    for (shape, bytes) in cases {
        let error = decode_slot_vector(&bytes).expect_err("unbounded header must fail");
        assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
        assert!(
            error.message.contains("byte") || error.message.contains("overflow"),
            "unexpected {shape} error: {}",
            error.message
        );
        println!(
            "ISSUE1604_SLOT_CODEC_UNBOUNDED shape={shape} before_bytes={} after_error_code={} detail={}",
            bytes.len(),
            error.code,
            error.message
        );
    }
}

#[test]
fn slot_vector_codec_rejects_truncation_trailing_bytes_and_invalid_utf8() {
    let valid = encode_slot_vector(&SlotVector::Dense {
        dim: 1,
        data: vec![3.5],
    })
    .unwrap();
    let mut trailing = valid.clone();
    trailing.push(0);
    let truncated = &valid[..valid.len() - 1];
    let invalid_utf8_absent = [1, 5, 0, 0, 0, 1, 0xff];

    for (case, bytes) in [
        ("truncated", truncated),
        ("trailing", trailing.as_slice()),
        ("invalid_utf8", invalid_utf8_absent.as_slice()),
    ] {
        let error = decode_slot_vector(bytes).expect_err("malformed bytes must fail");
        assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
        println!(
            "ISSUE1604_SLOT_CODEC_EDGE case={case} before_bytes={} after_error_code={} detail={}",
            bytes.len(),
            error.code,
            error.message
        );
    }
}

#[test]
fn slot_vector_codec_rejects_schema_invalid_physical_payloads() {
    let mut nan_dense = encoded_header(0, 1, None);
    nan_dense.extend_from_slice(&f32::NAN.to_bits().to_be_bytes());
    let mut sparse_out_of_range = encoded_header(2, 1, Some(1));
    sparse_out_of_range.extend_from_slice(&1_u32.to_be_bytes());
    sparse_out_of_range.extend_from_slice(&1_f32.to_bits().to_be_bytes());
    let zero_dim_multi = encoded_header(3, 0, Some(1));
    let empty_multi = encoded_header(3, 2, Some(0));

    for (case, bytes) in [
        ("nan_dense", nan_dense),
        ("sparse_out_of_range", sparse_out_of_range),
        ("zero_dim_multi", zero_dim_multi),
        ("empty_multi", empty_multi),
    ] {
        let error = decode_slot_vector(&bytes).expect_err("invalid schema must fail");
        assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
        println!(
            "ISSUE1604_SLOT_CODEC_SCHEMA case={case} before_bytes={} after_error_code={} detail={}",
            bytes.len(),
            error.code,
            error.message
        );
    }

    let invalid = SlotVector::Dense {
        dim: 1,
        data: vec![f32::INFINITY],
    };
    let error = encode_slot_vector(&invalid).expect_err("invalid vector must not encode");
    assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
}

fn encoded_header(tag: u8, first: u32, second: Option<u32>) -> Vec<u8> {
    let mut bytes = vec![tag];
    bytes.extend_from_slice(&first.to_be_bytes());
    if let Some(second) = second {
        bytes.extend_from_slice(&second.to_be_bytes());
    }
    bytes
}

#[test]
fn write_batch_codec_tracks_physical_rows_and_rejects_trailing_bytes() {
    let rows = [WriteRow {
        cf: ColumnFamily::Base,
        key: b"known-key".to_vec(),
        value: b"known-value".to_vec(),
    }];
    let encoded = encode_write_batch(&rows).unwrap();
    assert_eq!(decode_write_batch(&encoded).unwrap(), rows);

    let mut trailing = encoded;
    trailing.push(0xff);
    let error = decode_write_batch(&trailing).unwrap_err();

    assert_eq!(error.code, "CALYX_ASTER_CORRUPT_SHARD");
    assert!(error.message.contains("trailing bytes"));
}
