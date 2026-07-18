use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

#[derive(Clone, Copy)]
pub struct TeiEndpoint {
    pub port: u16,
    pub path: &'static str,
    pub body: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct HealthReadback {
    pub port: u16,
    pub path: &'static str,
    pub status: u16,
    pub ok: bool,
    pub latency_ms: f64,
    pub response_bytes: usize,
    pub error: Option<String>,
    pub status_line: Option<String>,
    pub headers_preview: String,
    pub body_preview: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TeiRequestReadback {
    pub port: u16,
    pub path: &'static str,
    pub status: u16,
    pub latency_ms: f64,
    pub ok: bool,
    pub response_bytes: usize,
    pub error: Option<String>,
    pub status_line: Option<String>,
    pub body_preview: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TeiLoadReadback {
    pub requested: usize,
    pub successes: usize,
    pub failures: usize,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub requests: Vec<TeiRequestReadback>,
}

struct HttpReadback {
    status: u16,
    latency_ms: f64,
    response_bytes: usize,
    error: Option<String>,
    status_line: Option<String>,
    headers_preview: String,
    body_preview: String,
}

pub fn check_tei_health(ports: &[u16]) -> Vec<HealthReadback> {
    ports
        .iter()
        .map(|port| {
            let response = http_request(*port, "GET", "/health", "");
            HealthReadback {
                port: *port,
                path: "/health",
                status: response.status,
                ok: response.error.is_none() && response.status == 200,
                latency_ms: response.latency_ms,
                response_bytes: response.response_bytes,
                error: response.error,
                status_line: response.status_line,
                headers_preview: response.headers_preview,
                body_preview: response.body_preview,
            }
        })
        .collect()
}

pub fn background_tei_load(n: usize, endpoints: &[TeiEndpoint]) -> TeiLoadReadback {
    let requests = Arc::new(Mutex::new(Vec::with_capacity(n)));
    thread::scope(|scope| {
        for idx in 0..n {
            let endpoint = endpoints[idx % endpoints.len()];
            let out = Arc::clone(&requests);
            scope.spawn(move || {
                let response = http_request(endpoint.port, "POST", endpoint.path, endpoint.body);
                out.lock().unwrap().push(TeiRequestReadback {
                    port: endpoint.port,
                    path: endpoint.path,
                    status: response.status,
                    latency_ms: response.latency_ms,
                    ok: response.error.is_none() && (200..300).contains(&response.status),
                    response_bytes: response.response_bytes,
                    error: response.error,
                    status_line: response.status_line,
                    body_preview: response.body_preview,
                });
            });
        }
    });
    let mut requests = requests.lock().unwrap().clone();
    requests.sort_by(|a, b| a.latency_ms.total_cmp(&b.latency_ms));
    let successes = requests.iter().filter(|item| item.ok).count();
    let failures = requests.len().saturating_sub(successes);
    let latencies = requests
        .iter()
        .map(|item| item.latency_ms)
        .collect::<Vec<_>>();
    TeiLoadReadback {
        requested: n,
        successes,
        failures,
        p50_ms: percentile(&latencies, 0.50),
        p99_ms: percentile(&latencies, 0.99),
        requests,
    }
}

fn http_request(port: u16, method: &str, path: &str, body: &str) -> HttpReadback {
    let started = Instant::now();
    let result = read_http_response(port, method, path, body);
    let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
    match result {
        Ok(bytes) => {
            let parsed = parse_response(&bytes);
            HttpReadback {
                status: parsed.status.unwrap_or(0),
                latency_ms,
                response_bytes: bytes.len(),
                error: parsed
                    .status
                    .is_none()
                    .then(|| "missing or invalid HTTP status line".to_string()),
                status_line: parsed.status_line,
                headers_preview: parsed.headers_preview,
                body_preview: parsed.body_preview,
            }
        }
        Err(error) => HttpReadback {
            status: 0,
            latency_ms,
            response_bytes: 0,
            error: Some(error.to_string()),
            status_line: None,
            headers_preview: String::new(),
            body_preview: String::new(),
        },
    }
}

fn read_http_response(port: u16, method: &str, path: &str, body: &str) -> std::io::Result<Vec<u8>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes())?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(response)
}

struct ParsedResponse {
    status: Option<u16>,
    status_line: Option<String>,
    headers_preview: String,
    body_preview: String,
}

fn parse_response(bytes: &[u8]) -> ParsedResponse {
    let response = String::from_utf8_lossy(bytes);
    let status_line = response.lines().next().map(ToString::to_string);
    let status = status_line
        .as_deref()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse().ok());
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .or_else(|| response.split_once("\n\n"))
        .unwrap_or((response.as_ref(), ""));
    ParsedResponse {
        status,
        status_line,
        headers_preview: preview(headers),
        body_preview: preview(body),
    }
}

fn preview(text: &str) -> String {
    text.chars().take(512).collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}
