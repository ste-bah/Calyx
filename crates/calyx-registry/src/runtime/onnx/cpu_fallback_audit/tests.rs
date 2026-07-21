//! Unit tests for the node-placement audit: synthetic ORT traces, the
//! heavy-op / fraction verdicts, and the #1487 toleration semantics.

use super::*;

fn kernel_event(name: &str, op_name: &str, provider: &str) -> Value {
    serde_json::json!({
        "cat": "Node",
        "name": name,
        "dur": 42,
        "ph": "X",
        "args": {"op_name": op_name, "provider": provider}
    })
}

fn counts_from(entries: &[(&str, &str, usize)]) -> ProviderNodeCounts {
    let mut counts = ProviderNodeCounts::default();
    for (provider, op, repeat) in entries {
        for _ in 0..*repeat {
            counts.add(provider, op);
        }
    }
    counts
}

#[test]
fn parses_kernel_time_nodes_per_provider_with_op_names() {
    let trace = serde_json::json!([
        {"cat": "Session", "name": "model_loading", "args": {}},
        kernel_event("MatMul_kernel_time", "MatMul", "CUDAExecutionProvider"),
        kernel_event("Add_kernel_time", "Add", "CUDAExecutionProvider"),
        kernel_event("QLinearMatMul_kernel_time", "QLinearMatMul", "CPUExecutionProvider"),
        kernel_event("Shape_kernel_time", "Shape", "CPUExecutionProvider"),
        // fence events must not be counted as separate compute nodes.
        {"cat": "Node", "name": "MatMul_fence_before", "args": {"provider": "CUDAExecutionProvider"}},
        {"cat": "Node", "name": "MatMul_fence_after", "args": {"provider": "CUDAExecutionProvider"}},
    ])
    .to_string();
    let counts = parse_profiling_nodes(&trace).unwrap();
    assert_eq!(counts.total(), 4);
    assert_eq!(counts.cpu_nodes(), 2);
    let audit = evaluate_placement(&counts, true, 0.10);
    assert_eq!(audit.heavy_cpu_ops, "QLinearMatMul:1");
    assert_eq!(audit.tolerated_cpu_ops, "Shape:1");
}

#[test]
fn parses_trace_events_wrapper_and_falls_back_without_kernel_suffix() {
    let trace = serde_json::json!({
        "traceEvents": [
            {"cat": "Node", "name": "MatMul", "args": {"op_name": "MatMul", "provider": "CUDAExecutionProvider"}},
            {"cat": "Node", "name": "QGemm", "args": {"op_name": "QGemm", "provider": "CPUExecutionProvider"}},
        ]
    })
    .to_string();
    let counts = parse_profiling_nodes(&trace).unwrap();
    assert_eq!(counts.total(), 2);
    assert_eq!(counts.cpu_nodes(), 1);
    let audit = evaluate_placement(&counts, true, 0.99);
    assert_eq!(audit.heavy_cpu_ops, "QGemm:1");
    assert!(audit.violation());
}

#[test]
fn malformed_trace_fails_closed() {
    assert_eq!(
        parse_profiling_nodes("not json").unwrap_err().code,
        "CALYX_ONNX_PROFILE_PARSE"
    );
    assert_eq!(
        parse_profiling_nodes("{\"nope\": 1}").unwrap_err().code,
        "CALYX_ONNX_PROFILE_PARSE"
    );
}

#[test]
fn gpu_policy_over_threshold_fails_loud() {
    // A predominantly-CPU graph of trivial ops: 20 CPU nodes, 4 CUDA
    // nodes → 83% CPU trips the fraction gate even without heavy ops.
    let counts = counts_from(&[
        ("CPUExecutionProvider", "Cast", 20),
        ("CUDAExecutionProvider", "MatMul", 4),
    ]);
    let audit = evaluate_placement(&counts, true, 0.10);
    assert_eq!(audit.total_nodes, 24);
    assert_eq!(audit.cpu_nodes, 20);
    assert!(audit.over_threshold);
    assert_eq!(audit.heavy_cpu_op_count, 0);
    assert!(audit.violation());

    let err = enforce_audit("frac-test", &audit, true, AuditMode::Fail).unwrap_err();
    assert_eq!(err.code, CPU_FALLBACK_CODE);
    assert!(err.message.contains("83"), "{}", err.message);
    assert!(err.message.contains("Cast:20"), "{}", err.message);
}

#[test]
fn heavy_op_on_cpu_fails_even_under_the_fraction_budget() {
    // 2 heavy CPU nodes out of 202 = under 1% CPU, but the real math is
    // off-device: refuse regardless of the fraction (#1487 semantics).
    let counts = counts_from(&[
        ("CUDAExecutionProvider", "MatMul", 200),
        ("CPUExecutionProvider", "MatMul", 1),
        ("CPUExecutionProvider", "Attention", 1),
    ]);
    let audit = evaluate_placement(&counts, true, 0.10);
    assert!(!audit.over_threshold);
    assert_eq!(audit.heavy_cpu_op_count, 2);
    assert!(audit.violation());

    let err = enforce_audit("heavy-test", &audit, true, AuditMode::Fail).unwrap_err();
    assert_eq!(err.code, CPU_FALLBACK_CODE);
    assert!(err.message.contains("Attention:1"), "{}", err.message);
    assert!(err.message.contains("MatMul:1"), "{}", err.message);
}

#[test]
fn trivial_cpu_ops_under_budget_are_tolerated_and_listed() {
    // The real-world BERT/XLM-R shape: heavy compute on CUDA, a handful of
    // Shape/Gather/Cast bookkeeping nodes on CPU. Must pass (#1487).
    let counts = counts_from(&[
        ("CUDAExecutionProvider", "MatMul", 96),
        ("CUDAExecutionProvider", "LayerNormalization", 25),
        ("CPUExecutionProvider", "Shape", 4),
        ("CPUExecutionProvider", "Gather", 3),
        ("CPUExecutionProvider", "Cast", 2),
    ]);
    let audit = evaluate_placement(&counts, true, 0.10);
    assert!(!audit.violation());
    assert_eq!(audit.heavy_cpu_ops, "none");
    assert_eq!(audit.tolerated_cpu_ops, "Cast:2,Gather:3,Shape:4");
    enforce_audit("tolerate-test", &audit, true, AuditMode::Fail).unwrap();
}

#[test]
fn warn_mode_never_errors_even_on_violation() {
    let counts = counts_from(&[("CPUExecutionProvider", "MatMul", 10)]);
    let audit = evaluate_placement(&counts, true, 0.10);
    assert!(audit.violation());
    enforce_audit("warn-test", &audit, true, AuditMode::Warn).unwrap();
}

#[test]
fn gpu_session_fully_on_device_passes() {
    let mut counts = counts_from(&[("CUDAExecutionProvider", "MatMul", 120)]);
    // Two legit CPU nodes (e.g. a Cast) stay under the 10% budget.
    counts.add("CPUExecutionProvider", "Cast");
    counts.add("CPUExecutionProvider", "Cast");
    let audit = evaluate_placement(&counts, true, 0.10);
    assert!(!audit.violation());
    assert!(audit.cpu_fraction < 0.10);
    enforce_audit("clean-test", &audit, true, AuditMode::Fail).unwrap();
}

#[test]
fn cpu_policy_is_never_flagged() {
    let counts = counts_from(&[("CPUExecutionProvider", "MatMul", 50)]);
    // gpu_policy=false: an all-CPU session under CPU policy is correct.
    let audit = evaluate_placement(&counts, false, 0.10);
    assert!(!audit.violation());
    enforce_audit("cpu-test", &audit, false, AuditMode::Fail).unwrap();
}

#[test]
fn mode_and_fraction_env_parsing() {
    assert!(configured_max_cpu_fraction().unwrap() > 0.0);
    assert!(!AuditMode::Off.enabled());
    assert!(AuditMode::Fail.enabled());
}

#[test]
fn heavy_op_set_is_sorted_and_matches_exactly() {
    let mut sorted = HEAVY_COMPUTE_OPS.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, HEAVY_COMPUTE_OPS, "keep HEAVY_COMPUTE_OPS sorted");
    assert!(is_heavy_compute_op("MatMul"));
    assert!(is_heavy_compute_op("QLinearMatMul"));
    assert!(!is_heavy_compute_op("Shape"));
    assert!(!is_heavy_compute_op("Gather"));
    assert!(!is_heavy_compute_op("matmul"), "matching is exact");
}
