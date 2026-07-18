use super::*;
use crate::cache::{cached_json_response, store_and_respond};
use http_body_util::BodyExt;

fn bytes(s: &str) -> Bytes {
    Bytes::copy_from_slice(s.as_bytes())
}

#[test]
fn hit_returns_byte_identical_body() {
    let cache = ResponseCache::new(Duration::from_secs(60), 16);
    let body = bytes(r#"{"id":"a","found":true}"#);
    cache.put("k".to_string(), body.clone());
    let (got, age) = cache.get("k").expect("fresh entry must hit");
    assert_eq!(got, body, "hit must replay the exact stored bytes");
    assert!(age < Duration::from_secs(60), "fresh entry age < ttl");
}

#[test]
fn absent_key_misses() {
    let cache = ResponseCache::new(Duration::from_secs(60), 16);
    assert!(cache.get("never-stored").is_none());
}

#[test]
fn entry_expires_after_ttl_and_is_dropped() {
    let cache = ResponseCache::new(Duration::from_millis(40), 16);
    cache.put("k".to_string(), bytes("v"));
    assert!(cache.get("k").is_some(), "before TTL: HIT");
    std::thread::sleep(Duration::from_millis(70));
    assert!(cache.get("k").is_none(), "after TTL: MISS (expired)");
    // The expired entry must have been evicted, not merely hidden.
    assert!(
        !cache.entries.lock().unwrap().contains_key("k"),
        "expired entry must be dropped on read"
    );
}

#[test]
fn zero_ttl_disables_caching() {
    let cache = ResponseCache::new(Duration::ZERO, 16);
    println!("CACHE_EDGE_ZERO_TTL_BEFORE entries=0");
    cache.put("k".to_string(), bytes("v"));
    assert!(cache.get("k").is_none(), "TTL=0 never serves a hit");
    assert!(
        cache.entries.lock().unwrap().is_empty(),
        "TTL=0 never stores an entry"
    );
    println!("CACHE_EDGE_ZERO_TTL_AFTER entries=0 hit=false");
}

#[test]
fn capacity_is_a_hard_bound_evicting_oldest() {
    let cache = ResponseCache::new(Duration::from_secs(60), 2);
    cache.put("a".to_string(), bytes("1"));
    std::thread::sleep(Duration::from_millis(5));
    cache.put("b".to_string(), bytes("2"));
    std::thread::sleep(Duration::from_millis(5));
    cache.put("c".to_string(), bytes("3")); // exceeds capacity 2
    let len = cache.entries.lock().unwrap().len();
    assert_eq!(len, 2, "len never exceeds capacity");
    assert!(cache.get("a").is_none(), "oldest-stored key 'a' evicted");
    assert!(cache.get("b").is_some(), "'b' retained");
    assert!(cache.get("c").is_some(), "'c' retained");
}

#[test]
fn age_reflects_time_since_store() {
    let cache = ResponseCache::new(Duration::from_secs(60), 16);
    cache.put("k".to_string(), bytes("v"));
    std::thread::sleep(Duration::from_millis(30));
    let (_, age) = cache.get("k").expect("hit");
    assert!(
        age >= Duration::from_millis(25),
        "age tracks elapsed: {age:?}"
    );
}

#[tokio::test]
async fn response_body_reuses_shared_bytes_for_empty_small_and_large_payloads() {
    for (case, source) in [
        ("empty", Bytes::new()),
        ("small", Bytes::from_static(br#"{"ok":true}"#)),
        ("large_2mib", Bytes::from(vec![b'x'; 2 * 1024 * 1024])),
    ] {
        let source_ptr = source.as_ptr();
        let source_len = source.len();
        println!("CACHE_SHARED_BEFORE case={case} len={source_len} ptr={source_ptr:p}");
        let response = cached_json_response(source, "HIT", Duration::from_secs(7));
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(response.headers()["x-cache"], "HIT");
        assert_eq!(response.headers()[header::AGE], "7");

        let frame = response.into_body().frame().await;
        if source_len == 0 {
            assert!(frame.is_none(), "empty body has no data frame");
            println!("CACHE_SHARED_AFTER case={case} len=0 frame=none");
            continue;
        }
        let frame = frame
            .expect("one body frame")
            .expect("infallible cached body");
        let delivered = frame.into_data().expect("cached body is a data frame");
        assert_eq!(delivered.len(), source_len);
        assert_eq!(
            delivered.as_ptr(),
            source_ptr,
            "Body::from(Bytes) must retain the shared allocation"
        );
        println!(
            "CACHE_SHARED_AFTER case={case} len={} ptr={:p} same_allocation=true",
            delivered.len(),
            delivered.as_ptr()
        );
    }
}

#[tokio::test]
async fn response_owns_bytes_after_cache_eviction() {
    let cache = ResponseCache::new(Duration::from_secs(60), 1);
    let original = Bytes::from(vec![0x5a; 1024 * 1024]);
    cache.put("original".to_string(), original.clone());
    let (hit, age) = cache.get("original").expect("original cache hit");
    let response = cached_json_response(hit, "HIT", age);
    println!(
        "CACHE_EVICTION_BEFORE original_present=true response_len={}",
        original.len()
    );

    cache.put(
        "replacement".to_string(),
        Bytes::from_static(b"replacement"),
    );
    assert!(cache.get("original").is_none(), "source entry was evicted");

    let delivered = response
        .into_body()
        .collect()
        .await
        .expect("collect cached response")
        .to_bytes();
    assert_eq!(
        delivered, original,
        "response retains independent shared ownership"
    );
    println!(
        "CACHE_EVICTION_AFTER original_present=false replacement_present=true response_len={}",
        delivered.len()
    );
}

#[tokio::test]
async fn over_capacity_vec_conversion_moves_the_allocation_without_copying() {
    let mut source = Vec::with_capacity(4 * 1024 * 1024);
    source.extend_from_slice(br#"{"bounded":"payload"}"#);
    let source_ptr = source.as_ptr();
    let source_len = source.len();
    let shared = Bytes::from(source);
    println!("CACHE_OVERCAP_BEFORE len={source_len} capacity=4194304 ptr={source_ptr:p}");
    assert_eq!(shared.as_ptr(), source_ptr, "Vec allocation is transferred");

    let delivered = cached_json_response(shared, "MISS", Duration::ZERO)
        .into_body()
        .collect()
        .await
        .expect("collect response")
        .to_bytes();
    assert_eq!(delivered.len(), source_len);
    assert_eq!(
        delivered.as_ptr(),
        source_ptr,
        "response still shares the allocation"
    );
    println!(
        "CACHE_OVERCAP_AFTER len={} ptr={:p} same_allocation=true",
        delivered.len(),
        delivered.as_ptr()
    );
}

#[tokio::test]
async fn serialized_miss_cache_entry_and_response_share_one_allocation() {
    let cache = ResponseCache::new(Duration::from_secs(60), 4);
    let value = json!({"id":"shared-miss","payload":"x".repeat(1024 * 1024)});
    println!("CACHE_MISS_BEFORE entries=0 payload_bytes=1048576");
    let response = store_and_respond(&cache, "shared-miss".to_string(), &value);
    let (cached, _) = cache.get("shared-miss").expect("serialized bytes cached");
    let cached_ptr = cached.as_ptr();
    let delivered = response
        .into_body()
        .collect()
        .await
        .expect("collect miss response")
        .to_bytes();
    assert_eq!(delivered, cached);
    assert_eq!(delivered.as_ptr(), cached_ptr);
    println!(
        "CACHE_MISS_AFTER entries=1 response_len={} same_allocation=true ptr={cached_ptr:p}",
        delivered.len()
    );
}

#[test]
#[ignore = "manual response-cache construction scaling FSV"]
fn cached_response_construction_scaling_fsv() {
    for size in [1024, 1024 * 1024, 16 * 1024 * 1024] {
        let cache = ResponseCache::new(Duration::from_secs(60), 1);
        cache.put("k".to_string(), Bytes::from(vec![0x41; size]));
        for hits in [1_u32, 100, 10_000] {
            let started = Instant::now();
            for _ in 0..hits {
                let (body, age) = cache.get("k").expect("cache hit");
                let response = cached_json_response(body, "HIT", age);
                assert_eq!(response.status(), StatusCode::OK);
            }
            println!(
                "CACHE_SCALE size={} hits={} elapsed_us={} retained_entries=1",
                size,
                hits,
                started.elapsed().as_micros()
            );
        }
    }
}

// --- /metrics exposition rendering (#1249 G11, #597) ----------------
// PURE render path exercised with synthetic snapshots: a healthy origin,
// a tampered (broken-chain) ledger, and a total scan failure. Each asserts
// the EXACT series a Prometheus scrape would parse — no mocks, no I/O.

fn metric_value(body: &str, name: &str) -> Option<f64> {
    body.lines()
        .find(|l| !l.starts_with('#') && l.split(' ').next() == Some(name))
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|v| v.parse::<f64>().ok())
}

#[test]
fn render_metrics_healthy_origin_all_green() {
    let body = render_metrics(&MetricsSnapshot {
        vault_ready: 1,
        gpu_ready: 1,
        faithfulness_ready: 1,
        panel_version: 7,
        scan_ok: 1,
        ledger_rows: 126,
        chain_intact: 1,
        chain_broken_seq: -1,
        scrape_duration_seconds: 0.012_345,
    });
    // Content-shape: a TYPE line precedes every sample (Prometheus requires
    // the TYPE before the first sample for a name).
    assert!(body.contains("# TYPE calyx_origin_healthy gauge"));
    assert_eq!(metric_value(&body, "calyx_up"), Some(1.0));
    assert_eq!(metric_value(&body, "calyx_origin_healthy"), Some(1.0));
    assert_eq!(metric_value(&body, "calyx_ledger_rows"), Some(126.0));
    assert_eq!(metric_value(&body, "calyx_ledger_chain_intact"), Some(1.0));
    assert_eq!(
        metric_value(&body, "calyx_ledger_chain_broken_seq"),
        Some(-1.0)
    );
    assert_eq!(metric_value(&body, "calyx_panel_version"), Some(7.0));
}

#[test]
fn render_metrics_broken_chain_flips_healthy_and_exposes_seq() {
    // Tamper edge: chain broken at seq 42 → not intact, not healthy, the
    // broken seq is surfaced so an alert can name the failing entry.
    let body = render_metrics(&MetricsSnapshot {
        vault_ready: 1,
        gpu_ready: 1,
        faithfulness_ready: 1,
        panel_version: 7,
        scan_ok: 1,
        ledger_rows: 100,
        chain_intact: 0,
        chain_broken_seq: 42,
        scrape_duration_seconds: 0.001,
    });
    assert_eq!(metric_value(&body, "calyx_ledger_chain_intact"), Some(0.0));
    assert_eq!(
        metric_value(&body, "calyx_ledger_chain_broken_seq"),
        Some(42.0)
    );
    assert_eq!(
        metric_value(&body, "calyx_origin_healthy"),
        Some(0.0),
        "a broken chain must drop the health roll-up even with gpu/vault up"
    );
}

#[test]
fn render_metrics_scan_failure_is_unhealthy_with_zero_rows() {
    // Edge: ledger unreadable → scan_ok 0, rows 0, not intact, not healthy.
    let body = render_metrics(&MetricsSnapshot {
        vault_ready: 0,
        gpu_ready: 0,
        faithfulness_ready: 0,
        panel_version: 0,
        scan_ok: 0,
        ledger_rows: 0,
        chain_intact: 0,
        chain_broken_seq: -1,
        scrape_duration_seconds: 0.0,
    });
    assert_eq!(metric_value(&body, "calyx_ledger_scan_ok"), Some(0.0));
    assert_eq!(metric_value(&body, "calyx_ledger_rows"), Some(0.0));
    assert_eq!(metric_value(&body, "calyx_origin_healthy"), Some(0.0));
    // calyx_up is still 1: the process answered the scrape.
    assert_eq!(metric_value(&body, "calyx_up"), Some(1.0));
}

// --- per-route HTTP RED metrics (#597) -------------------------------
// Synthetic requests with KNOWN inputs → assert the exact counter and
// histogram series a Prometheus scrape would parse.

#[test]
fn http_metrics_counts_requests_by_method_route_code() {
    let m = HttpMetrics::new();
    // 2 OK + 1 error on the same route, plus a different route once.
    m.record("POST", "/v1/measure", 200, 0.02);
    m.record("POST", "/v1/measure", 200, 0.2);
    m.record("POST", "/v1/measure", 500, 1.5);
    m.record("GET", "/v1/health", 200, 0.001);
    let body = m.render();
    assert!(body.contains(
        "calyx_http_requests_total{method=\"POST\",route=\"/v1/measure\",code=\"200\"} 2"
    ));
    assert!(body.contains(
        "calyx_http_requests_total{method=\"POST\",route=\"/v1/measure\",code=\"500\"} 1"
    ));
    assert!(
        body.contains(
            "calyx_http_requests_total{method=\"GET\",route=\"/v1/health\",code=\"200\"} 1"
        )
    );
    assert!(body.contains("# TYPE calyx_http_request_duration_seconds histogram"));
}

#[test]
fn http_histogram_buckets_are_cumulative_and_inf_equals_count() {
    let m = HttpMetrics::new();
    // latencies: 0.02 (<=0.025), 0.2 (<=0.25), 1.5 (<=2.5)
    for (s, lat) in [(200u16, 0.02f64), (200, 0.2), (500, 1.5)] {
        m.record("POST", "/v1/measure", s, lat);
    }
    let body = m.render();
    // le=0.025 covers only the 0.02 obs → 1
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"POST\",route=\"/v1/measure\",le=\"0.025\"} 1"
    ));
    // le=0.25 covers 0.02 and 0.2 → 2 (cumulative)
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"POST\",route=\"/v1/measure\",le=\"0.25\"} 2"
    ));
    // le=2.5 covers all three → 3
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"POST\",route=\"/v1/measure\",le=\"2.5\"} 3"
    ));
    // +Inf == _count == 3, _sum == 1.72
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"POST\",route=\"/v1/measure\",le=\"+Inf\"} 3"
    ));
    assert!(body.contains(
        "calyx_http_request_duration_seconds_count{method=\"POST\",route=\"/v1/measure\"} 3"
    ));
    assert!(body.contains(
        "calyx_http_request_duration_seconds_sum{method=\"POST\",route=\"/v1/measure\"} 1.720000"
    ));
}

#[test]
fn http_metrics_empty_renders_headers_only_no_samples() {
    // Edge: zero requests → TYPE/HELP present, no sample lines.
    let body = HttpMetrics::new().render();
    assert!(body.contains("# TYPE calyx_http_requests_total counter"));
    assert!(
        !body.contains("calyx_http_requests_total{"),
        "no sample lines when nothing recorded"
    );
}

#[test]
fn http_histogram_slow_request_only_in_inf_bucket() {
    // Edge: a 12s request exceeds every finite bound — it must NOT appear in
    // le=10 but must be in +Inf and _count.
    let m = HttpMetrics::new();
    m.record("GET", "/v1/kernel", 504, 12.0);
    let body = m.render();
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"GET\",route=\"/v1/kernel\",le=\"10\"} 0"
    ));
    assert!(body.contains(
        "calyx_http_request_duration_seconds_bucket{method=\"GET\",route=\"/v1/kernel\",le=\"+Inf\"} 1"
    ));
}

#[test]
fn parse_env_u64_defaults_when_unset_and_fails_loud_when_garbage() {
    // Unset → default (use a name no test sets).
    assert_eq!(
        parse_env_u64("CALYX_WEB_API_CACHE_TTL_SECS_UNSET_XYZ", 30).unwrap(),
        30
    );
    // Present-but-garbage → loud error, never silent default.
    // SAFETY: single-threaded test; var removed immediately after assert.
    unsafe { std::env::set_var("CALYX_WEB_API_TEST_BAD_INT", "not-a-number") };
    let err = parse_env_u64("CALYX_WEB_API_TEST_BAD_INT", 7).unwrap_err();
    unsafe { std::env::remove_var("CALYX_WEB_API_TEST_BAD_INT") };
    assert!(
        err.contains("non-negative integer"),
        "loud parse error: {err}"
    );
}
