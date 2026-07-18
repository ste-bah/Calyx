use super::super::*;
use super::tokens;

#[test]
fn parse_ingest_status_command() {
    let parsed = parse(&tokens([
        "ingest-status",
        "mydb",
        "--session",
        "issue1065-session",
    ]))
    .unwrap();
    assert_eq!(
        parsed,
        Subcommand::IngestStatus(IngestStatusArgs {
            vault: "mydb".to_string(),
            session_id: "issue1065-session".to_string(),
        })
    );
}

#[test]
fn parse_ingest_session_id_requires_batch() {
    let err = parse(&tokens([
        "ingest",
        "mydb",
        "--text",
        "hello",
        "--session-id",
        "not-for-text",
    ]))
    .unwrap_err();
    assert_eq!(err.code(), "CALYX_CLI_USAGE_ERROR");
    assert!(err.message().contains("--batch"));
}

#[test]
fn parse_ingest_session_id_rejects_ledger_token_boundary() {
    let maximum =
        "generic-session-".to_string() + &"a".repeat(calyx_ledger::MAX_UNCLASSIFIED_TOKEN_LEN - 16);
    let accepted = parse(&[
        "ingest".to_string(),
        "mydb".to_string(),
        "--batch".to_string(),
        "real.jsonl".to_string(),
        "--session-id".to_string(),
        maximum,
    ])
    .unwrap();
    assert!(matches!(accepted, Subcommand::Ingest(_)));

    let rejected = "a".repeat(calyx_ledger::MAX_UNCLASSIFIED_TOKEN_LEN - 1) + "-x";
    let err = parse(&[
        "ingest".to_string(),
        "mydb".to_string(),
        "--batch".to_string(),
        "real.jsonl".to_string(),
        "--session-id".to_string(),
        rejected,
    ])
    .unwrap_err();
    assert_eq!(err.code(), "CALYX_INGEST_SESSION_INVALID");
    assert!(err.message().contains("use at most 39 characters"));

    let recognized_hex_id = "a".repeat(64);
    let accepted = parse(&[
        "ingest".to_string(),
        "mydb".to_string(),
        "--batch".to_string(),
        "real.jsonl".to_string(),
        "--session-id".to_string(),
        recognized_hex_id,
    ])
    .unwrap();
    assert!(matches!(accepted, Subcommand::Ingest(_)));
}
