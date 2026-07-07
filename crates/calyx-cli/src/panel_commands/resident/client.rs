//! CLI plumbing for `calyx panel resident ready|measure|measure-batch|stop`.
//! The wire client itself lives in calyx_registry::resident::client (shared
//! with calyx-search and calyx-mcp); this module only parses flags, converts
//! CLI inputs, and renders responses.

use super::*;
use calyx_registry::resident::{measure_batch_summary_at, send_request};

pub(crate) fn client_command(args: &[String], op: &str) -> CliResult {
    let flags = parse_client_flags(args, op)?;
    if op == "measure-batch" {
        let modality = flags.modality.expect("parsed modality");
        let inputs = flags
            .inputs
            .into_iter()
            .map(|input| client_input_to_core(input, modality))
            .collect::<CliResult<Vec<_>>>()?;
        if flags.summary_only {
            let response =
                measure_batch_summary_at(flags.addr, modality, &inputs, flags.runtime_batch_limit)?;
            if let Some(path) = flags.out {
                write_json_file(path, &response)?;
            }
            return print_json(&response);
        }
        let response = measure_batch_at(flags.addr, modality, &inputs, flags.runtime_batch_limit)?;
        if let Some(path) = flags.out {
            write_json_file(path, &response.response)?;
        }
        return print_json(&response.response);
    }
    let mut request = json!({ "op": op });
    if op == "measure" {
        request["modality"] = serde_json::to_value(flags.modality.expect("parsed modality"))
            .map_err(|error| {
                CliError::runtime(format!("serialize resident measure modality: {error}"))
            })?;
        match flags.input.expect("parsed input") {
            protocol::ClientMeasureInput::Utf8(input) => request["input"] = json!(input),
            protocol::ClientMeasureInput::Hex(input_hex) => {
                request["input_hex"] = json!(input_hex)
            }
        }
    }
    let response = send_request(flags.addr, request)?;
    if let Some(path) = flags.out {
        write_json_file(path, &response)?;
    }
    print_json(&response)
}

fn client_input_to_core(
    input: protocol::ClientMeasureInput,
    modality: Modality,
) -> CliResult<Input> {
    let bytes = match input {
        protocol::ClientMeasureInput::Utf8(input) => input.into_bytes(),
        protocol::ClientMeasureInput::Hex(input_hex) => {
            protocol::hex_decode(&input_hex).map_err(CliError::usage)?
        }
    };
    Ok(Input {
        modality,
        bytes,
        pointer: None,
    })
}
