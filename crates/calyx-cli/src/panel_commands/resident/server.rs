use super::discovery::{
    RESIDENT_DISCOVERY_SCHEMA, ResidentDiscovery, remove_resident_discovery, unix_now_ms,
    write_resident_discovery,
};
use super::dispatch::{dispatch_request, readiness};
use super::stream::serve_binary_measure_batch;
use super::*;

pub(crate) struct ResidentService {
    pub(crate) state: ResidentWarmState,
    pub(crate) bind: SocketAddr,
    pub(crate) started: Instant,
    pub(crate) max_runtime_batch: usize,
    pub(crate) capacity_probe_input_count: usize,
    pub(crate) capacity_probe_ms: u128,
    pub(crate) capacity_probe_modalities: Vec<Modality>,
    pub(crate) onnx_shape_budget: Option<calyx_registry::OnnxShapeBucketBudget>,
}

pub(crate) fn serve(args: &[String]) -> CliResult {
    let mut flags = parse_serve_flags(args)?;
    let bind = flags.bind.unwrap_or(parse_addr(DEFAULT_BIND)?);
    ensure_loopback(bind)?;
    let home = resolve_home(&mut flags)?;
    if flags.template.is_some() == flags.vault.is_some() {
        return Err(CliError::usage(
            "calyx panel resident serve requires exactly one of --template <name-or-id> or --vault <vault>",
        ));
    }
    let listener = TcpListener::bind(bind)?;
    let local_addr = listener.local_addr()?;
    // Canonicalize the vault source before warm state consumes the flags so
    // discovery-file consumers can compare vault identity path-for-path.
    let discovery_vault = match flags.vault.as_deref() {
        Some(vault) => Some(vault.canonicalize().map_err(|error| {
            CliError::io(format!(
                "canonicalize resident --vault {}: {error}",
                vault.display()
            ))
        })?),
        None => None,
    };
    let discovery_template = flags.template.clone();
    let max_runtime_batch = flags.max_runtime_batch.unwrap_or(DEFAULT_MAX_RUNTIME_BATCH);
    let state = load_resident_warm_state(warm_options(home.clone(), flags))?;
    let mut service = ResidentService {
        state,
        bind: local_addr,
        started: Instant::now(),
        max_runtime_batch,
        capacity_probe_input_count: 0,
        capacity_probe_ms: 0,
        capacity_probe_modalities: Vec::new(),
        onnx_shape_budget: None,
    };
    let capacity = super::capacity::run(&service)?;
    service.capacity_probe_input_count = capacity.input_count;
    service.capacity_probe_ms = capacity.elapsed_ms;
    service.capacity_probe_modalities = capacity.modalities;
    service.onnx_shape_budget = capacity.onnx_shape_budget;
    let service = Arc::new(service);
    let ready = readiness(&service);
    if let Some(path) = service.state.ready_out.clone() {
        write_json_file(path, &ready)?;
    }
    let discovery_path = write_resident_discovery(
        &home,
        &ResidentDiscovery {
            schema: RESIDENT_DISCOVERY_SCHEMA.to_string(),
            bind: local_addr,
            process_id: std::process::id(),
            vault: discovery_vault,
            template: discovery_template,
            written_at_unix_ms: unix_now_ms(),
        },
    )?;
    eprintln!(
        "CALYX_PANEL_RESIDENT_RUNTIME phase=discovery_written path={} bind={local_addr}",
        discovery_path.display()
    );
    print_json(&ready)?;
    let served = serve_loop(listener, service);
    // Best-effort removal of this process's own record on graceful shutdown;
    // a crashed service leaves the file behind and ingest discovery detects
    // that via the live readiness probe.
    let removed = remove_resident_discovery(&home, std::process::id());
    served?;
    removed
}

fn resolve_home(flags: &mut ServeFlags) -> CliResult<PathBuf> {
    resolve_home_with(flags.home.take(), calyx_home)
}

pub(crate) fn resolve_home_with(
    provided: Option<PathBuf>,
    fallback: impl FnOnce() -> CliResult<PathBuf>,
) -> CliResult<PathBuf> {
    match provided {
        Some(home) => Ok(home),
        None => fallback(),
    }
}

fn warm_options(home: PathBuf, flags: ServeFlags) -> ResidentWarmOptions {
    ResidentWarmOptions {
        home,
        template: flags.template,
        vault: flags.vault,
        slots: flags.slots,
        modality: flags.modality,
        ready_out: flags.ready_out,
        max_resident_vram_mib: flags
            .max_resident_vram_mib
            .unwrap_or(DEFAULT_MAX_RESIDENT_VRAM_MIB),
        resident_overhead_multiplier_milli: flags
            .resident_overhead_multiplier_milli
            .unwrap_or(DEFAULT_RESIDENT_OVERHEAD_MULTIPLIER_MILLI),
        max_load_secs: flags.max_load_secs.unwrap_or(DEFAULT_MAX_LOAD_SECS),
        load_parallelism: flags.load_parallelism,
        progress_out: flags.progress_out,
    }
}

fn serve_loop(listener: TcpListener, service: Arc<ResidentService>) -> CliResult {
    let running = Arc::new(AtomicBool::new(true));
    serve_loop_with(listener, Arc::clone(&running), |stream, running| {
        handle_client(stream, Arc::clone(&service), running)
    })
}

pub(super) fn serve_loop_with(
    listener: TcpListener,
    running: Arc<AtomicBool>,
    mut handle: impl FnMut(TcpStream, Arc<AtomicBool>) -> CliResult,
) -> CliResult {
    while running.load(Ordering::SeqCst) {
        let (stream, peer) = match listener.accept() {
            Ok(accepted) => accepted,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                eprintln!(
                    "CALYX_PANEL_RESIDENT_RUNTIME {}",
                    json!({
                        "event_code": "CALYX_PANEL_RESIDENT_ACCEPT_INTERRUPTED",
                        "phase": "accept_retry",
                        "error_message": error.to_string(),
                        "remediation": "no operator action required; the listener is retrying",
                    })
                );
                continue;
            }
            Err(error) => {
                return Err(CliError::io(format!(
                    "accept resident client on {}: {error}",
                    listener.local_addr().map_or_else(
                        |addr_error| format!("unknown ({addr_error})"),
                        |addr| addr.to_string()
                    )
                )));
            }
        };
        if !peer.ip().is_loopback() {
            eprintln!(
                "CALYX_PANEL_RESIDENT_RUNTIME {}",
                json!({
                    "event_code": "CALYX_PANEL_RESIDENT_NON_LOOPBACK_CLIENT",
                    "phase": "client_rejected",
                    "peer": peer.to_string(),
                    "remediation": "connect only through the configured loopback address",
                })
            );
            let _ = stream.shutdown(Shutdown::Both);
            continue;
        }
        if let Err(error) = handle(stream, Arc::clone(&running)) {
            eprintln!(
                "CALYX_PANEL_RESIDENT_RUNTIME {}",
                json!({
                    "event_code": "CALYX_PANEL_RESIDENT_CLIENT_ERROR",
                    "phase": "client_error",
                    "peer": peer.to_string(),
                    "error_code": error.code(),
                    "error_message": error.message(),
                    "cause_remediation": error.remediation(),
                    "remediation": "inspect and repair only the named client; use the resident JSON-line or binary protocol and read the complete response; the resident remains active",
                })
            );
        }
    }
    Ok(())
}

fn handle_client(
    mut stream: TcpStream,
    service: Arc<ResidentService>,
    running: Arc<AtomicBool>,
) -> CliResult {
    let timeout = Some(Duration::from_secs(CLIENT_TIMEOUT_SECS));
    stream
        .set_read_timeout(timeout)
        .map_err(|error| CliError::io(format!("set resident client read timeout: {error}")))?;
    stream
        .set_write_timeout(timeout)
        .map_err(|error| CliError::io(format!("set resident client write timeout: {error}")))?;
    let reader_stream = stream
        .try_clone()
        .map_err(|error| CliError::io(format!("clone resident client stream: {error}")))?;
    let mut reader = BufReader::new(reader_stream);
    let mut first_line = Vec::new();
    reader
        .read_until(b'\n', &mut first_line)
        .map_err(|error| CliError::io(format!("read resident client request line: {error}")))?;
    if first_line == RESIDENT_BINARY_MAGIC {
        serve_binary_measure_batch(&mut reader, &mut stream, &service)?;
        stream
            .flush()
            .map_err(|error| CliError::io(format!("flush resident binary response: {error}")))?;
        let _ = stream.shutdown(Shutdown::Both);
        return Ok(());
    }

    let response = match String::from_utf8(first_line) {
        Ok(line) => match serde_json::from_str::<ResidentRequest>(&line) {
            Ok(request) => dispatch_request(request, &service, &running),
            Err(error) => error_value(
                "CALYX_PANEL_RESIDENT_BAD_REQUEST",
                format!("decode resident request JSON line: {error}"),
                "send one JSON object per connection with op=ready, measure, or shutdown",
            ),
        },
        Err(error) => error_value(
            "CALYX_PANEL_RESIDENT_BAD_REQUEST",
            format!("resident request was neither binary magic nor valid UTF-8 JSON: {error}"),
            "send one JSON object per connection or the resident binary magic line",
        ),
    };
    serde_json::to_writer(&mut stream, &response)
        .map_err(|error| CliError::runtime(format!("write resident response JSON: {error}")))?;
    stream
        .write_all(b"\n")
        .map_err(|error| CliError::io(format!("write resident response terminator: {error}")))?;
    stream
        .flush()
        .map_err(|error| CliError::io(format!("flush resident JSON response: {error}")))?;
    let _ = stream.shutdown(Shutdown::Both);
    Ok(())
}
