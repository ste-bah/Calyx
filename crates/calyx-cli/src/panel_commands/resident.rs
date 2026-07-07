use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bincode::config;
use calyx_core::{AbsentReason, CalyxError, Input, LensId, Modality, SlotState, SlotVector};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

mod flags;

// Wire protocol, frame codec, discovery file, and the minimal client calls
// live in calyx-registry::resident (single source of truth shared with
// calyx-mcp and calyx-search); the server implementation stays here.
pub(crate) use calyx_registry::resident::{codec, discovery, protocol};

pub(crate) use discovery::{ResidentDiscovery, read_resident_discovery, resident_discovery_path};

use flags::{
    ServeFlags, calyx_home, ensure_loopback, parse_addr, parse_client_flags, parse_serve_flags,
};
use protocol::{
    ClientMeasureInput, MEASURE_BATCH_SCHEMA, MEASURE_SCHEMA, MeasureResponse, READY_SCHEMA,
    RESIDENT_BINARY_PROTOCOL_VERSION, ReadyResponse, ResidentMeasureBatchBinaryRequest,
    ResidentMeasureBatchStreamEnd, ResidentMeasureBatchStreamFrame,
    ResidentMeasureBatchStreamHeader, ResidentRequest, hex_decode,
};
pub(crate) use protocol::{
    MeasureBatchAtResponse, MeasureBatchResponse, MeasureBatchSummaryResponse,
    ResidentMeasuredInput, ResidentSlotMeasure,
};

use super::warm::resident_support::{
    ResidentWarmOptions, ResidentWarmState, load_resident_warm_state,
};
use crate::error::{CliError, CliResult};
use crate::output::print_json;

const DEFAULT_BIND: &str = "127.0.0.1:8787";
const DEFAULT_MAX_RESIDENT_VRAM_MIB: u64 = 22 * 1024;
const DEFAULT_RESIDENT_OVERHEAD_MULTIPLIER_MILLI: u64 = 2100;
const DEFAULT_MAX_LOAD_SECS: u64 = 60;
use calyx_registry::resident::{
    CLIENT_TIMEOUT_REMEDIATION, MAX_RESIDENT_SERVICE_FRAME_BYTES, RESIDENT_BINARY_MAGIC,
    client_timeout_secs,
};

mod client;
mod dispatch;
mod parallel;
mod server;
mod stream;

#[cfg(test)]
mod tests;

pub(crate) use calyx_registry::resident::{measure_batch_at, ready_value_at};
pub(crate) fn run(args: &[String]) -> CliResult {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(CliError::usage(
            "calyx panel resident requires serve, ready, measure, measure-batch, or stop",
        ));
    };
    match command {
        "serve" => server::serve(&args[1..]),
        "ready" => client::client_command(&args[1..], "ready"),
        "measure" => client::client_command(&args[1..], "measure"),
        "measure-batch" => client::client_command(&args[1..], "measure-batch"),
        "stop" => client::client_command(&args[1..], "shutdown"),
        other => Err(CliError::usage(format!(
            "unknown panel resident subcommand {other}; expected serve, ready, measure, measure-batch, or stop"
        ))),
    }
}

fn write_json_file(path: PathBuf, value: &impl Serialize) -> CliResult {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| CliError::runtime(format!("serialize {}: {error}", path.display())))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn cli_error_value(error: &CliError) -> Value {
    error_value(error.code(), error.message(), error.remediation())
}

fn error_value(
    code: impl Into<String>,
    message: impl Into<String>,
    remediation: impl Into<String>,
) -> Value {
    json!({
        "ok": false,
        "code": code.into(),
        "message": message.into(),
        "remediation": remediation.into(),
    })
}
