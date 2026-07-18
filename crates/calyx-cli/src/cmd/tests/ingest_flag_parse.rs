use super::*;
use calyx_core::Modality;

#[test]
fn parse_ingest_allow_cold_gpu_workers_flag() {
    let parsed = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--allow-cold-gpu-workers",
    ]))
    .unwrap();
    let Subcommand::Ingest(args) = parsed else {
        panic!("expected ingest subcommand");
    };
    assert!(args.allow_cold_gpu_workers);
    assert!(args.resident_addr.is_none());
}

#[test]
fn parse_ingest_text_command() {
    let parsed = parse(&tokens(["ingest", "mydb", "--text", "hello"])).unwrap();
    assert_eq!(
        parsed,
        Subcommand::Ingest(IngestArgs {
            vault: "mydb".to_string(),
            text: Some("hello".to_string()),
            batch: None,
            file: None,
            modality: None,
            idempotent: true,
            output: IngestOutput::Summary,
            resident_addr: None,
            allow_cold_gpu_workers: false,
            session_id: None,
            precondition: IngestPrecondition::default(),
        })
    );
}

#[test]
fn parse_ingest_video_file_command() {
    let parsed = parse(&tokens([
        "ingest",
        "media",
        "--file",
        "clip.webm",
        "--modality",
        "video",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        Subcommand::Ingest(IngestArgs {
            vault: "media".to_string(),
            text: None,
            batch: None,
            file: Some("clip.webm".into()),
            modality: Some(Modality::Video),
            idempotent: true,
            output: IngestOutput::Summary,
            resident_addr: None,
            allow_cold_gpu_workers: false,
            session_id: None,
            precondition: IngestPrecondition::default(),
        })
    );
}

#[test]
fn parse_ingest_rows_output_command() {
    let parsed = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--output",
        "rows",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        Subcommand::Ingest(IngestArgs {
            vault: "mydb".to_string(),
            text: None,
            batch: Some("batch.jsonl".into()),
            file: None,
            modality: None,
            idempotent: true,
            output: IngestOutput::Rows,
            resident_addr: None,
            allow_cold_gpu_workers: false,
            session_id: None,
            precondition: IngestPrecondition::default(),
        })
    );
}

#[test]
fn parse_ingest_resident_addr_command() {
    let parsed = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--resident-addr",
        "127.0.0.1:8787",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        Subcommand::Ingest(IngestArgs {
            vault: "mydb".to_string(),
            text: None,
            batch: Some("batch.jsonl".into()),
            file: None,
            modality: None,
            idempotent: true,
            output: IngestOutput::Summary,
            resident_addr: Some("127.0.0.1:8787".parse().unwrap()),
            allow_cold_gpu_workers: false,
            session_id: None,
            precondition: IngestPrecondition::default(),
        })
    );
}

#[test]
fn parse_ingest_rejects_non_loopback_resident_addr() {
    let err = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--resident-addr",
        "10.0.0.10:8787",
    ]))
    .unwrap_err();

    assert_eq!(err.code(), "CALYX_INGEST_RESIDENT_ADDR_REFUSED");
}

#[test]
fn parse_ingest_rejects_unknown_output_mode() {
    let err = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--output",
        "verbose",
    ]))
    .unwrap_err();

    assert_eq!(err.code(), "CALYX_CLI_USAGE_ERROR");
    assert!(err.message().contains("summary or rows"));
}

#[test]
fn parse_ingest_exact_vault_state_precondition() {
    let parsed = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--expect-durable-seq",
        "17",
        "--expect-manifest-seq",
        "9",
        "--expect-base-count",
        "1000",
    ]))
    .unwrap();
    let Subcommand::Ingest(args) = parsed else {
        panic!("expected ingest subcommand");
    };
    assert_eq!(
        args.precondition,
        IngestPrecondition {
            expected_durable_seq: Some(17),
            expected_manifest_seq: Some(9),
            expected_base_count: Some(1000),
        }
    );
}

#[test]
fn parse_ingest_precondition_rejects_non_batch_duplicate_and_invalid_integer() {
    let non_batch = parse(&tokens([
        "ingest",
        "mydb",
        "--text",
        "hello",
        "--expect-base-count",
        "0",
    ]))
    .unwrap_err();
    assert_eq!(non_batch.code(), "CALYX_CLI_USAGE_ERROR");
    assert!(non_batch.message().contains("only valid with --batch"));

    let duplicate = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--expect-base-count",
        "0",
        "--expect-base-count",
        "1",
    ]))
    .unwrap_err();
    assert_eq!(duplicate.code(), "CALYX_CLI_USAGE_ERROR");
    assert!(duplicate.message().contains("duplicate"));

    let invalid = parse(&tokens([
        "ingest",
        "mydb",
        "--batch",
        "batch.jsonl",
        "--expect-durable-seq",
        "not-a-sequence",
    ]))
    .unwrap_err();
    assert_eq!(invalid.code(), "CALYX_CLI_USAGE_ERROR");
    assert!(invalid.message().contains("--expect-durable-seq"));
}
