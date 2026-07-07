//! Minimal resident-service client calls, shared by CLI, search, and MCP.
//! Moved from calyx-cli with CalyxError error types; behavior unchanged.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use calyx_core::{CalyxError, Input, Modality, Result};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::codec::{decode_binary, encode_binary, read_frame, write_frame};
use super::protocol::{
    MEASURE_BATCH_SCHEMA, MeasureBatchAtResponse, MeasureBatchResponse,
    MeasureBatchSummaryResponse, RESIDENT_BINARY_PROTOCOL_VERSION,
    ResidentMeasureBatchBinaryRequest, ResidentMeasureBatchStreamFrame,
    ResidentMeasureBatchStreamHeader, ResidentMeasuredInput,
};
use super::{
    CLIENT_TIMEOUT_REMEDIATION, RESIDENT_BINARY_MAGIC, client_timeout_secs, ensure_loopback,
    io_client_error,
};

/// Programmatic readiness probe used by resident-route discovery: one JSON
/// `ready` round-trip returning the raw readiness value.
pub fn ready_value_at(addr: SocketAddr) -> Result<Value> {
    send_request(addr, json!({ "op": "ready" }))
}

pub fn send_request(addr: SocketAddr, request: Value) -> Result<Value> {
    ensure_loopback(addr)?;
    let mut stream = TcpStream::connect(addr)
        .map_err(|error| io_client_error(addr, "connect", error))?;
    let timeout = Some(Duration::from_secs(client_timeout_secs()?));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| io_client_error(addr, "configure", error))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| io_client_error(addr, "configure", error))?;
    serde_json::to_writer(&mut stream, &request).map_err(|error| CalyxError {
        code: "CALYX_PANEL_RESIDENT_UNAVAILABLE",
        message: format!("write resident request to {addr}: {error}"),
        remediation: CLIENT_TIMEOUT_REMEDIATION,
    })?;
    stream
        .write_all(b"\n")
        .and_then(|_| stream.flush())
        .map_err(|error| io_client_error(addr, "write to", error))?;
    let mut response = String::new();
    BufReader::new(stream)
        .read_line(&mut response)
        .map_err(|error| io_client_error(addr, "read from", error))?;
    serde_json::from_str(&response).map_err(|error| CalyxError {
        code: "CALYX_PANEL_RESIDENT_UNAVAILABLE",
        message: format!("parse resident response from {addr}: {error}"),
        remediation: CLIENT_TIMEOUT_REMEDIATION,
    })
}

pub fn measure_batch_at(
    addr: SocketAddr,
    modality: Modality,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> Result<MeasureBatchAtResponse> {
    let (mut stream, request_len) =
        open_measure_batch_stream(addr, modality, inputs, runtime_batch_limit)?;
    read_measure_batch_stream(&mut stream, inputs.len(), request_len)
}

pub fn measure_batch_summary_at(
    addr: SocketAddr,
    modality: Modality,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> Result<MeasureBatchSummaryResponse> {
    let (mut stream, request_len) =
        open_measure_batch_stream(addr, modality, inputs, runtime_batch_limit)?;
    read_measure_batch_summary_stream(&mut stream, inputs.len(), request_len)
}

fn open_measure_batch_stream(
    addr: SocketAddr,
    modality: Modality,
    inputs: &[Input],
    runtime_batch_limit: Option<usize>,
) -> Result<(TcpStream, usize)> {
    ensure_loopback(addr)?;
    let mut stream = TcpStream::connect(addr)
        .map_err(|error| io_client_error(addr, "connect", error))?;
    let timeout = Some(Duration::from_secs(client_timeout_secs()?));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| io_client_error(addr, "configure", error))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| io_client_error(addr, "configure", error))?;
    stream
        .write_all(RESIDENT_BINARY_MAGIC)
        .map_err(|error| io_client_error(addr, "write to", error))?;
    let request_bytes = encode_binary(&ResidentMeasureBatchBinaryRequest {
        protocol_version: RESIDENT_BINARY_PROTOCOL_VERSION,
        modality,
        inputs: inputs
            .iter()
            .map(|input| input.bytes.clone())
            .collect::<Vec<_>>(),
        runtime_batch_limit,
    })?;
    write_frame(&mut stream, &request_bytes)?;
    stream
        .flush()
        .map_err(|error| io_client_error(addr, "write to", error))?;
    Ok((stream, request_bytes.len()))
}

/// Consume the streamed measure_batch frames: Header, then one Row frame per
/// input, then End. Any Err frame, out-of-order frame, truncated stream, or
/// row/count mismatch fails closed — a partial stream never yields rows.
fn read_measure_batch_stream(
    stream: &mut TcpStream,
    expected_inputs: usize,
    request_bytes: usize,
) -> Result<MeasureBatchAtResponse> {
    let mut response_bytes = 0usize;
    let mut next_frame = |stream: &mut TcpStream| -> Result<ResidentMeasureBatchStreamFrame> {
        let frame = read_frame(stream)?;
        response_bytes += frame.len();
        decode_binary::<ResidentMeasureBatchStreamFrame>(&frame)
    };
    let header = match next_frame(stream)? {
        ResidentMeasureBatchStreamFrame::Header(header) => header,
        ResidentMeasureBatchStreamFrame::Err {
            code,
            message,
            remediation,
        } => return Err(remote_stream_error(&code, &message, &remediation)),
        other => return Err(unexpected_stream_frame("Header", &other)),
    };
    validate_measure_batch_header(&header)?;
    let mut rows: Vec<ResidentMeasuredInput> = Vec::with_capacity(header.input_count);
    let end = loop {
        match next_frame(stream)? {
            ResidentMeasureBatchStreamFrame::Row(row) => {
                if row.input_index != rows.len() {
                    return Err(stream_order_error(format!(
                        "resident measure_batch row frame carries input_index {} but {} rows were received",
                        row.input_index,
                        rows.len()
                    )));
                }
                rows.push(*row);
            }
            ResidentMeasureBatchStreamFrame::End(end) => break end,
            ResidentMeasureBatchStreamFrame::Err {
                code,
                message,
                remediation,
            } => return Err(remote_stream_error(&code, &message, &remediation)),
            other => return Err(unexpected_stream_frame("Row or End", &other)),
        }
    };
    if end.row_count != rows.len() || rows.len() != expected_inputs {
        return Err(stream_order_error(format!(
            "resident measure_batch stream ended with {} rows (end frame says {}) for {} inputs",
            rows.len(),
            end.row_count,
            expected_inputs
        )));
    }
    Ok(MeasureBatchAtResponse {
        response: MeasureBatchResponse {
            schema: header.schema,
            ready: header.ready,
            process_id: header.process_id,
            template_source: header.template_source,
            modality: header.modality,
            input_count: header.input_count,
            elapsed_ms: end.elapsed_ms,
            runtime_batch_limit: header.runtime_batch_limit,
            rows,
        },
        request_bytes,
        response_bytes,
    })
}

fn read_measure_batch_summary_stream(
    stream: &mut TcpStream,
    expected_inputs: usize,
    request_bytes: usize,
) -> Result<MeasureBatchSummaryResponse> {
    let mut response_bytes = 0usize;
    let frame = read_frame(stream)?;
    response_bytes += frame.len();
    let header = match decode_binary::<ResidentMeasureBatchStreamFrame>(&frame)? {
        ResidentMeasureBatchStreamFrame::Header(header) => header,
        ResidentMeasureBatchStreamFrame::Err {
            code,
            message,
            remediation,
        } => return Err(remote_stream_error(&code, &message, &remediation)),
        other => return Err(unexpected_stream_frame("Header", &other)),
    };
    validate_measure_batch_header(&header)?;
    let mut hasher = Sha256::new();
    let mut row_count = 0usize;
    let mut measured_slot_counts = Vec::new();
    let mut absent_slot_counts = Vec::new();
    let end = loop {
        let frame = read_frame(stream)?;
        response_bytes += frame.len();
        match decode_binary::<ResidentMeasureBatchStreamFrame>(&frame)? {
            ResidentMeasureBatchStreamFrame::Row(row) => {
                if row.input_index != row_count {
                    return Err(stream_order_error(format!(
                        "resident measure_batch row frame carries input_index {} but {row_count} rows were received",
                        row.input_index
                    )));
                }
                hasher.update(&frame);
                push_unique(&mut measured_slot_counts, row.measured_slot_count);
                push_unique(&mut absent_slot_counts, row.absent_slot_count);
                row_count += 1;
            }
            ResidentMeasureBatchStreamFrame::End(end) => break end,
            ResidentMeasureBatchStreamFrame::Err {
                code,
                message,
                remediation,
            } => return Err(remote_stream_error(&code, &message, &remediation)),
            other => return Err(unexpected_stream_frame("Row or End", &other)),
        }
    };
    if end.row_count != row_count || row_count != expected_inputs {
        return Err(stream_order_error(format!(
            "resident measure_batch stream ended with {row_count} rows (end frame says {}) for {expected_inputs} inputs",
            end.row_count
        )));
    }
    Ok(MeasureBatchSummaryResponse {
        schema: header.schema,
        ready: header.ready,
        process_id: header.process_id,
        template_source: header.template_source,
        modality: header.modality,
        input_count: header.input_count,
        elapsed_ms: end.elapsed_ms,
        runtime_batch_limit: header.runtime_batch_limit,
        row_count,
        measured_slot_counts,
        absent_slot_counts,
        response_rows_sha256: hex_digest(&hasher.finalize()),
        request_bytes,
        response_bytes,
    })
}

fn validate_measure_batch_header(header: &ResidentMeasureBatchStreamHeader) -> Result<()> {
    if header.protocol_version != RESIDENT_BINARY_PROTOCOL_VERSION {
        return Err(CalyxError {
            code: "CALYX_PANEL_RESIDENT_PROTOCOL_MISMATCH",
            message: format!(
                "resident measure_batch binary protocol {}, expected {}",
                header.protocol_version, RESIDENT_BINARY_PROTOCOL_VERSION
            ),
            remediation: "restart the resident service from the same Calyx build as the CLI",
        });
    }
    if header.schema != MEASURE_BATCH_SCHEMA {
        return Err(CalyxError {
            code: "CALYX_PANEL_RESIDENT_SCHEMA_MISMATCH",
            message: format!(
                "resident measure_batch schema {}, expected {}",
                header.schema, MEASURE_BATCH_SCHEMA
            ),
            remediation: "restart the resident service from the same Calyx build as the CLI",
        });
    }
    Ok(())
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
        values.sort_unstable();
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn stream_order_error(message: String) -> CalyxError {
    CalyxError {
        code: "CALYX_PANEL_RESIDENT_STREAM_ORDER",
        message,
        remediation: "restart the resident service from the same Calyx build as the CLI",
    }
}

fn remote_stream_error(code: &str, message: &str, remediation: &str) -> CalyxError {
    CalyxError {
        code: resident_remote_error_code(code),
        message: format!("{code}: {message}; remediation={remediation}"),
        remediation: CLIENT_TIMEOUT_REMEDIATION,
    }
}

fn unexpected_stream_frame(expected: &str, got: &ResidentMeasureBatchStreamFrame) -> CalyxError {
    let kind = match got {
        ResidentMeasureBatchStreamFrame::Header(_) => "Header",
        ResidentMeasureBatchStreamFrame::Row(_) => "Row",
        ResidentMeasureBatchStreamFrame::End(_) => "End",
        ResidentMeasureBatchStreamFrame::Err { .. } => "Err",
    };
    CalyxError {
        code: "CALYX_PANEL_RESIDENT_STREAM_ORDER",
        message: format!(
            "resident measure_batch stream sent a {kind} frame where {expected} was expected"
        ),
        remediation: "restart the resident service from the same Calyx build as the CLI",
    }
}

fn resident_remote_error_code(remote_code: &str) -> &'static str {
    match remote_code {
        "CALYX_PANEL_RESIDENT_BAD_REQUEST" => "CALYX_PANEL_RESIDENT_BAD_REQUEST",
        "CALYX_PANEL_RESIDENT_INPUT_HEX_INVALID" => "CALYX_PANEL_RESIDENT_INPUT_HEX_INVALID",
        "CALYX_PANEL_RESIDENT_UNAVAILABLE" => "CALYX_PANEL_RESIDENT_UNAVAILABLE",
        "CALYX_PANEL_RESIDENT_SCHEMA_MISMATCH" => "CALYX_PANEL_RESIDENT_SCHEMA_MISMATCH",
        "CALYX_PANEL_RESIDENT_BINARY_ENCODE" => "CALYX_PANEL_RESIDENT_BINARY_ENCODE",
        "CALYX_PANEL_RESIDENT_BINARY_DECODE" => "CALYX_PANEL_RESIDENT_BINARY_DECODE",
        "CALYX_PANEL_RESIDENT_BINARY_FRAME" => "CALYX_PANEL_RESIDENT_BINARY_FRAME",
        "CALYX_PANEL_RESIDENT_PROTOCOL_MISMATCH" => "CALYX_PANEL_RESIDENT_PROTOCOL_MISMATCH",
        _ => "CALYX_PANEL_RESIDENT_ERROR",
    }
}
