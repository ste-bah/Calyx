//! Client side of the panel resident-service wire protocol, shared by every
//! consumer that measures through a running `calyx panel resident serve`
//! process: the CLI (ingest route, search, client commands), calyx-mcp, and
//! any future embedder host. The server implementation stays in calyx-cli;
//! this module owns the single source of truth for the frame codec, the
//! protocol types, the discovery file, and the minimal client calls, so the
//! wire format cannot drift between consumers.

use std::io;
use std::net::{IpAddr, SocketAddr};

use calyx_core::{CalyxError, Result};

pub mod client;
pub mod codec;
pub mod discovery;
pub mod protocol;

pub use client::{measure_batch_at, measure_batch_summary_at, ready_value_at, send_request};
pub use discovery::{
    RESIDENT_DISCOVERY_SCHEMA, ResidentDiscovery, read_resident_discovery,
    remove_resident_discovery, resident_discovery_path, unix_now_ms, write_resident_discovery,
};
pub use protocol::{
    MEASURE_BATCH_SCHEMA, MEASURE_SCHEMA, MeasureBatchAtResponse, MeasureBatchResponse,
    MeasureBatchSummaryResponse, MeasureResponse, READY_SCHEMA, RESIDENT_BINARY_PROTOCOL_VERSION,
    ReadyResponse, ResidentMeasureBatchBinaryRequest, ResidentMeasureBatchStreamEnd,
    ResidentMeasureBatchStreamFrame, ResidentMeasureBatchStreamHeader, ResidentMeasuredInput,
    ResidentRequest, ResidentSlotMeasure, hex_decode, hex_encode,
};

pub const RESIDENT_BINARY_MAGIC: &[u8] = b"CALYX_PANEL_RESIDENT_BIN1\n";
pub const MAX_RESIDENT_SERVICE_FRAME_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Resident client socket read/write timeout. SO_RCVTIMEO is per-syscall, so
/// this bounds one blocking read — and the frame-header wait IS the GPU
/// measure wait, so it must cover the slowest single batch. Matches the cold
/// lens-worker default (DEFAULT_LENS_WORKER_TIMEOUT_SECS = 300).
pub const DEFAULT_CLIENT_TIMEOUT_SECS: u64 = 300;
pub const CLIENT_TIMEOUT_ENV: &str = "CALYX_RESIDENT_CLIENT_TIMEOUT_SECS";
pub const CLIENT_TIMEOUT_REMEDIATION: &str =
    "start `calyx panel resident serve` on the requested loopback address; for measurements \
     slower than the client timeout, raise CALYX_RESIDENT_CLIENT_TIMEOUT_SECS";

/// Resolve the resident client timeout, overridable via
/// CALYX_RESIDENT_CLIENT_TIMEOUT_SECS. Fails loud on an unparseable or zero
/// value — never falls back silently.
pub fn client_timeout_secs() -> Result<u64> {
    let Some(value) = std::env::var_os(CLIENT_TIMEOUT_ENV) else {
        return Ok(DEFAULT_CLIENT_TIMEOUT_SECS);
    };
    let value = value.to_str().ok_or_else(|| CalyxError {
        code: "CALYX_PANEL_RESIDENT_BAD_REQUEST",
        message: format!("{CLIENT_TIMEOUT_ENV} is not valid unicode"),
        remediation: "set a positive integer number of seconds",
    })?;
    let secs = value.trim().parse::<u64>().map_err(|error| CalyxError {
        code: "CALYX_PANEL_RESIDENT_BAD_REQUEST",
        message: format!("{CLIENT_TIMEOUT_ENV}={value} is not a valid timeout in seconds: {error}"),
        remediation: "set a positive integer number of seconds",
    })?;
    if secs == 0 {
        return Err(CalyxError {
            code: "CALYX_PANEL_RESIDENT_BAD_REQUEST",
            message: format!("{CLIENT_TIMEOUT_ENV} must be greater than zero"),
            remediation: "set a positive integer number of seconds",
        });
    }
    Ok(secs)
}

/// Resident services are loopback-only by doctrine; every client call and the
/// server bind path enforce this.
pub fn ensure_loopback(addr: SocketAddr) -> Result<()> {
    match addr.ip() {
        IpAddr::V4(ip) if ip.is_loopback() => Ok(()),
        IpAddr::V6(ip) if ip.is_loopback() => Ok(()),
        _ => Err(CalyxError {
            code: "CALYX_PANEL_RESIDENT_BIND_REFUSED",
            message: format!("resident service address {addr} is not loopback"),
            remediation: "bind resident services only to 127.0.0.1 or [::1]",
        }),
    }
}

fn io_client_error(addr: SocketAddr, what: &str, error: io::Error) -> CalyxError {
    CalyxError {
        code: "CALYX_PANEL_RESIDENT_UNAVAILABLE",
        message: format!("{what} resident service {addr}: {error}"),
        remediation: CLIENT_TIMEOUT_REMEDIATION,
    }
}
