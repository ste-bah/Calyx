//! Bounded, policy-respecting CourtListener locator verification.
//!
//! This command uses the supported legal-search API. It deliberately never probes
//! or automates the public opinion HTML surface, and it treats every challenge,
//! throttle, malformed response, and locator mismatch as a typed refusal.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use calyx_core::CalyxError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::durable_write::write_bytes_atomic_new;
use crate::error::{CliError, CliResult};
use crate::output::print_json;

const DEFAULT_BASE_URL: &str = "https://www.courtlistener.com";
const MIN_POLICY_INTERVAL_MS: u64 = 12_000;
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

#[derive(Clone, Debug)]
struct Args {
    input: PathBuf,
    out: PathBuf,
    token_env: Option<String>,
    min_interval_ms: u64,
    timeout_ms: u64,
    base_url: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocatorExpectation {
    cluster_id: u64,
    expected_url: String,
}

#[derive(Clone, Debug, Serialize)]
struct Observation {
    ordinal: usize,
    cluster_id: u64,
    endpoint: String,
    expected_url: String,
    waited_before_ms: u64,
    http_status: Option<u16>,
    retry_after: Option<String>,
    waf_action: Option<String>,
    challenge: bool,
    title: Option<String>,
    canonical: Option<String>,
    og_url: Option<String>,
    canonical_source: &'static str,
    og_url_source: &'static str,
    verdict: &'static str,
    detail: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    source_of_truth: &'static str,
    input_path: String,
    input_sha256: String,
    request_count: usize,
    accepted_count: usize,
    refused_count: usize,
    min_interval_ms: u64,
    timeout_ms: u64,
    authentication: &'static str,
    base_url: String,
    policy: &'static str,
    constellation_policy: &'static str,
    observations: Vec<Observation>,
}

pub(crate) fn try_run(args: &[String]) -> Option<CliResult> {
    if args.first().map(String::as_str) != Some("courtlistener-locator-audit") {
        return None;
    }
    if matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) {
        return Some(crate::usage::print_command_usage(
            "courtlistener-locator-audit",
        ));
    }
    Some(parse(&args[1..]).and_then(run))
}

fn parse(args: &[String]) -> CliResult<Args> {
    let mut input = None;
    let mut out = None;
    let mut token_env = None;
    let mut min_interval_ms = MIN_POLICY_INTERVAL_MS;
    let mut timeout_ms = DEFAULT_TIMEOUT_MS;
    let mut base_url = DEFAULT_BASE_URL.to_string();
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| CliError::usage(format!("{flag} requires a value")))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--token-env" => token_env = Some(value.clone()),
            "--min-interval-ms" => min_interval_ms = parse_positive(flag, value)?,
            "--timeout-ms" => timeout_ms = parse_positive(flag, value)?,
            "--base-url" => base_url = value.trim_end_matches('/').to_string(),
            other => {
                return Err(CliError::usage(format!(
                    "unexpected courtlistener-locator-audit flag {other}"
                )));
            }
        }
        index += 2;
    }
    if min_interval_ms < MIN_POLICY_INTERVAL_MS {
        return Err(contract_error(
            "CALYX_COURTLISTENER_RATE_POLICY",
            format!(
                "--min-interval-ms {min_interval_ms} is below the supported 5 requests/minute floor"
            ),
            "use at least 12000 milliseconds or obtain and document a higher contractual API tier",
        ));
    }
    validate_base_url(&base_url)?;
    Ok(Args {
        input: input.ok_or_else(|| CliError::usage("--input <jsonl> is required"))?,
        out: out.ok_or_else(|| CliError::usage("--out <json> is required"))?,
        token_env,
        min_interval_ms,
        timeout_ms,
        base_url,
    })
}

fn parse_positive(flag: &str, value: &str) -> CliResult<u64> {
    value
        .parse::<u64>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .ok_or_else(|| CliError::usage(format!("{flag} requires an integer greater than zero")))
}

fn validate_base_url(value: &str) -> CliResult {
    if value == DEFAULT_BASE_URL
        || value == "http://127.0.0.1"
        || value
            .strip_prefix("http://127.0.0.1:")
            .is_some_and(|port| port.parse::<u16>().is_ok())
    {
        return Ok(());
    }
    Err(contract_error(
        "CALYX_COURTLISTENER_ENDPOINT_REFUSED",
        format!("unsupported CourtListener audit base URL {value}"),
        "use the exact supported CourtListener origin; loopback is allowed only for a bounded manual refusal drill",
    ))
}

fn run(args: Args) -> CliResult {
    let (input_sha256, rows) = read_input(&args.input)?;
    let token = args
        .token_env
        .as_deref()
        .map(|name| {
            std::env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    contract_error(
                        "CALYX_COURTLISTENER_AUTH_MISSING",
                        format!("token environment variable {name} is unset or empty"),
                        "set the named variable to a CourtListener API token; never place a token on the command line",
                    )
                })
        })
        .transpose()?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_millis(args.timeout_ms)))
        .build()
        .into();
    let mut observations = Vec::with_capacity(rows.len());
    let mut previous_started = None;
    for (ordinal, row) in rows.iter().enumerate() {
        let waited_before_ms = pace(previous_started, args.min_interval_ms);
        let started = Instant::now();
        let observation = request_one(
            &agent,
            &args.base_url,
            token.as_deref(),
            row,
            ordinal + 1,
            waited_before_ms,
        );
        previous_started = Some(started);
        observations.push(observation);
    }
    let refused_count = observations
        .iter()
        .filter(|observation| observation.verdict == "refuse")
        .count();
    let report = Report {
        schema: "courtlistener_locator_audit_v1",
        status: if refused_count == 0 {
            "complete"
        } else {
            "refused"
        },
        source_of_truth: "CourtListener REST API v4 legal-search result absolute_url, cluster_id, and caseName fields",
        input_path: args.input.display().to_string(),
        input_sha256,
        request_count: rows.len(),
        accepted_count: rows.len() - refused_count,
        refused_count,
        min_interval_ms: args.min_interval_ms,
        timeout_ms: args.timeout_ms,
        authentication: if token.is_some() {
            "token"
        } else {
            "anonymous"
        },
        base_url: args.base_url,
        policy: "supported legal-search API only; never automate, scrape, solve, or bypass the public-page WAF; 202/challenge/429 are refusals",
        constellation_policy: "locator verification neither measures nor transforms constellation slots; every typed lens lane remains separate",
        observations,
    };
    let mut bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| CliError::runtime(format!("serialize locator audit: {error}")))?;
    bytes.push(b'\n');
    write_bytes_atomic_new(&args.out, &bytes, "CourtListener locator audit report")?;
    print_json(&report)?;
    if refused_count > 0 {
        return Err(contract_error(
            "CALYX_COURTLISTENER_LOCATOR_REFUSED",
            format!(
                "{} of {} CourtListener locator observations refused; inspect {}",
                refused_count,
                rows.len(),
                args.out.display()
            ),
            "respect Retry-After/backoff, correct the expected locator, or retry later; never treat challenge output as verification",
        ));
    }
    Ok(())
}

fn pace(previous_started: Option<Instant>, min_interval_ms: u64) -> u64 {
    let Some(previous_started) = previous_started else {
        return 0;
    };
    let interval = Duration::from_millis(min_interval_ms);
    let elapsed = previous_started.elapsed();
    if elapsed >= interval {
        return 0;
    }
    let wait = interval - elapsed;
    thread::sleep(wait);
    u64::try_from(wait.as_millis()).unwrap_or(u64::MAX)
}

fn request_one(
    agent: &ureq::Agent,
    base_url: &str,
    token: Option<&str>,
    row: &LocatorExpectation,
    ordinal: usize,
    waited_before_ms: u64,
) -> Observation {
    let endpoint = format!(
        "{base_url}/api/rest/v4/search/?type=o&q=id%3A{}",
        row.cluster_id
    );
    let mut request = agent.get(&endpoint).header("Accept", "application/json");
    if let Some(token) = token {
        request = request.header("Authorization", &format!("Token {token}"));
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(error) => {
            return empty_observation(
                ordinal,
                row,
                endpoint,
                waited_before_ms,
                format!("transport failure: {error}"),
            );
        }
    };
    let status = response.status().as_u16();
    let retry_after = header(&response, "retry-after");
    let waf_action = header(&response, "x-amzn-waf-action");
    let header_challenge = waf_action
        .as_deref()
        .is_some_and(|value| value.eq_ignore_ascii_case("challenge"));
    let mut response = response;
    let body = match response.body_mut().read_to_string() {
        Ok(body) => body,
        Err(error) => {
            return Observation {
                ordinal,
                cluster_id: row.cluster_id,
                endpoint,
                expected_url: row.expected_url.clone(),
                waited_before_ms,
                http_status: Some(status),
                retry_after,
                waf_action,
                challenge: header_challenge || status == 202,
                title: None,
                canonical: None,
                og_url: None,
                canonical_source: "unavailable",
                og_url_source: "not_exposed_by_legal_search_api",
                verdict: "refuse",
                detail: format!("read response body failed: {error}"),
            };
        }
    };
    let body_challenge = body.to_ascii_lowercase().contains("human verification");
    let challenge = status == 202 || header_challenge || body_challenge;
    let parsed = serde_json::from_str::<Value>(&body).ok();
    let results = parsed
        .as_ref()
        .and_then(|value| value.get("results"))
        .and_then(Value::as_array);
    let result = results.and_then(|values| (values.len() == 1).then(|| &values[0]));
    let returned_cluster_id = result
        .and_then(|value| value.get("cluster_id"))
        .and_then(Value::as_u64);
    let title = result
        .and_then(|value| value.get("caseName"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let absolute_url = result
        .and_then(|value| value.get("absolute_url"))
        .and_then(Value::as_str)
        .map(|path| normalize_locator(base_url, path));
    let success = status == 200
        && !challenge
        && returned_cluster_id == Some(row.cluster_id)
        && absolute_url.as_deref() == Some(&row.expected_url);
    let detail = if challenge {
        "WAF/human-verification challenge response".to_string()
    } else if status == 429 {
        "CourtListener rate limit response".to_string()
    } else if status != 200 {
        format!("unexpected HTTP status {status}")
    } else if parsed.is_none() {
        "response is not valid JSON".to_string()
    } else if results.is_none() {
        "search response has no results array".to_string()
    } else if result.is_none() {
        format!(
            "exact-id search returned {} results instead of one",
            results.map_or(0, Vec::len)
        )
    } else if returned_cluster_id != Some(row.cluster_id) {
        format!(
            "search result cluster_id {:?} does not match expected {}",
            returned_cluster_id, row.cluster_id
        )
    } else if absolute_url.is_none() {
        "search result has no absolute_url".to_string()
    } else if !success {
        format!(
            "cluster absolute_url {:?} does not match expected locator",
            absolute_url
        )
    } else {
        "exact supported-API locator match".to_string()
    };
    Observation {
        ordinal,
        cluster_id: row.cluster_id,
        endpoint,
        expected_url: row.expected_url.clone(),
        waited_before_ms,
        http_status: Some(status),
        retry_after,
        waf_action,
        challenge,
        title,
        canonical: absolute_url,
        og_url: None,
        canonical_source: "legal_search_api.results[0].absolute_url",
        og_url_source: "not_exposed_by_legal_search_api",
        verdict: if success { "accept" } else { "refuse" },
        detail,
    }
}

fn header(response: &ureq::http::Response<ureq::Body>, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn normalize_locator(base_url: &str, absolute_url: &str) -> String {
    if absolute_url.starts_with("http://") || absolute_url.starts_with("https://") {
        absolute_url.to_string()
    } else {
        format!("{base_url}/{}", absolute_url.trim_start_matches('/'))
    }
}

fn empty_observation(
    ordinal: usize,
    row: &LocatorExpectation,
    endpoint: String,
    waited_before_ms: u64,
    detail: String,
) -> Observation {
    Observation {
        ordinal,
        cluster_id: row.cluster_id,
        endpoint,
        expected_url: row.expected_url.clone(),
        waited_before_ms,
        http_status: None,
        retry_after: None,
        waf_action: None,
        challenge: false,
        title: None,
        canonical: None,
        og_url: None,
        canonical_source: "unavailable",
        og_url_source: "not_exposed_by_legal_search_api",
        verdict: "refuse",
        detail,
    }
}

fn read_input(path: &Path) -> CliResult<(String, Vec<LocatorExpectation>)> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        CliError::io(format!("inspect locator input {}: {error}", path.display()))
    })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(contract_error(
            "CALYX_COURTLISTENER_INPUT_INVALID",
            format!("locator input {} is not a plain file", path.display()),
            "provide an immutable plain JSONL input file",
        ));
    }
    let bytes = fs::read(path)
        .map_err(|error| CliError::io(format!("read locator input {}: {error}", path.display())))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut rows = Vec::new();
    for (offset, line) in BufReader::new(bytes.as_slice()).lines().enumerate() {
        let line_number = offset + 1;
        let line = line.map_err(|error| {
            CliError::io(format!("read locator input line {line_number}: {error}"))
        })?;
        if line.trim().is_empty() {
            return Err(contract_error(
                "CALYX_COURTLISTENER_INPUT_INVALID",
                format!("locator input line {line_number} is blank"),
                "remove blank lines and provide one exact cluster expectation per JSON line",
            ));
        }
        let row: LocatorExpectation = serde_json::from_str(&line).map_err(|error| {
            contract_error(
                "CALYX_COURTLISTENER_INPUT_INVALID",
                format!("parse locator input line {line_number}: {error}"),
                "provide {\"cluster_id\":<u64>,\"expected_url\":\"https://www.courtlistener.com/opinion/.../\"}",
            )
        })?;
        let prefix = format!("{DEFAULT_BASE_URL}/opinion/{}/", row.cluster_id);
        if !row.expected_url.starts_with(&prefix) || !row.expected_url.ends_with('/') {
            return Err(contract_error(
                "CALYX_COURTLISTENER_INPUT_INVALID",
                format!(
                    "locator input line {line_number} expected_url is not the cluster {} canonical form",
                    row.cluster_id
                ),
                "use the exact https CourtListener cluster-plus-slug URL with a trailing slash",
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(contract_error(
            "CALYX_COURTLISTENER_INPUT_INVALID",
            "locator input has no rows",
            "provide at least one exact cluster locator expectation",
        ));
    }
    Ok((sha256, rows))
}

fn contract_error(
    code: &'static str,
    message: impl Into<String>,
    remediation: &'static str,
) -> CliError {
    CliError::from(CalyxError {
        code,
        message: message.into(),
        remediation,
    })
}
