//! #1143 — CUDA BFC arena environment knobs and the shape-diversity gate.
//!
//! `std::env` is process-global, so every test that reads or writes the
//! `CALYX_ONNX_*` arena knobs serializes on one lock.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::fastembed_runtime::execution_providers;
use super::super::{
    OnnxProviderPolicy, arena, cpu_fallback_audit, io_binding, onnx_shape_bucket_budget, session,
};

static ARENA_ENV_LOCK: Mutex<()> = Mutex::new(());

fn arena_env_lock() -> MutexGuard<'static, ()> {
    ARENA_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn temp_root(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("calyx-{name}-{nanos}"));
    std::fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn execution_provider_policy_is_cuda_fail_loud() {
    let _lock = arena_env_lock();
    let providers = execution_providers(OnnxProviderPolicy::CudaFailLoud).unwrap();

    assert_eq!(providers.len(), 1);
    let provider = format!("{:?}", providers[0]);
    assert!(provider.contains("CUDA"));
    assert!(provider.contains("error_on_failure: true"));
}

#[test]
fn resident_shape_budget_rejects_limit_below_bucket_domain() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_MAX_DISTINCT_SHAPES", "29") };

    let error = onnx_shape_bucket_budget(4).unwrap_err();

    assert_eq!(error.code, "CALYX_ONNX_SHAPE_LIMIT_BELOW_BUCKET_DOMAIN");
    assert!(error.message.contains("required=30"));
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };
}

#[test]
fn resident_shape_budget_accepts_exact_bucket_domain() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_MAX_DISTINCT_SHAPES", "30") };

    let budget = onnx_shape_bucket_budget(4).unwrap();

    assert_eq!(budget.configured_shape_limit, 30);
    assert_eq!(budget.required_shape_count, 30);
    assert_eq!(budget.sequence_bucket_count, 10);
    assert_eq!(budget.batch_bucket_count, 3);
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };
}

#[test]
fn resident_shape_budget_accepts_default_above_bucket_domain() {
    let _lock = arena_env_lock();
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };

    let budget = onnx_shape_bucket_budget(4).unwrap();

    assert_eq!(budget.configured_shape_limit, 64);
    assert_eq!(budget.required_shape_count, 30);
}

#[test]
fn execution_provider_policy_can_be_explicit_cpu() {
    let _lock = arena_env_lock();
    let providers = execution_providers(OnnxProviderPolicy::CpuExplicit).unwrap();

    assert_eq!(providers.len(), 1);
    let provider = format!("{:?}", providers[0]);
    assert!(provider.contains("CPU"));
    assert!(!provider.contains("CUDA"));
}

#[test]
fn cuda_fail_loud_relaxes_ort_knob_and_mandates_placement_audit() {
    let _lock = arena_env_lock();
    unsafe { std::env::remove_var("CALYX_ONNX_DISABLE_CPU_EP_FALLBACK") };
    unsafe { std::env::remove_var("CALYX_ONNX_CPU_FALLBACK_AUDIT") };

    // #1487: the strict ORT knob is no longer the CudaFailLoud default — real
    // transformer exports always have trivial CPU-assigned nodes, so
    // zero-tolerance refused every real onnx-custom lens at Initialize.
    assert!(
        !session::cpu_ep_fallback_disabled_for_policy(OnnxProviderPolicy::CudaFailLoud).unwrap()
    );
    assert!(
        !session::cpu_ep_fallback_disabled_for_policy(OnnxProviderPolicy::CpuExplicit).unwrap()
    );
    // Without the knob, the placement audit is mandatory fail for GPU policy;
    // the environment default (off) cannot weaken it.
    assert_eq!(
        cpu_fallback_audit::effective_audit_mode(true, false).unwrap(),
        cpu_fallback_audit::AuditMode::Fail
    );
    // With the strict knob active, ORT enforces zero tolerance at Initialize;
    // the audit is not additionally mandated.
    assert_eq!(
        cpu_fallback_audit::effective_audit_mode(true, true).unwrap(),
        cpu_fallback_audit::AuditMode::Off
    );
    // CPU-policy sessions keep the configured (default off) mode.
    assert_eq!(
        cpu_fallback_audit::effective_audit_mode(false, false).unwrap(),
        cpu_fallback_audit::AuditMode::Off
    );

    // warn cannot downgrade the mandatory GPU-policy audit, but still applies
    // to sessions the mandate does not cover.
    unsafe { std::env::set_var("CALYX_ONNX_CPU_FALLBACK_AUDIT", "warn") };
    assert_eq!(
        cpu_fallback_audit::effective_audit_mode(true, false).unwrap(),
        cpu_fallback_audit::AuditMode::Fail
    );
    assert_eq!(
        cpu_fallback_audit::effective_audit_mode(false, false).unwrap(),
        cpu_fallback_audit::AuditMode::Warn
    );
    unsafe { std::env::remove_var("CALYX_ONNX_CPU_FALLBACK_AUDIT") };

    // The zero-tolerance opt-in still works for CudaFailLoud and is still
    // rejected for explicit-CPU sessions.
    unsafe { std::env::set_var("CALYX_ONNX_DISABLE_CPU_EP_FALLBACK", "1") };
    let error =
        session::cpu_ep_fallback_disabled_for_policy(OnnxProviderPolicy::CpuExplicit).unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_CPU_EP_FALLBACK_POLICY_INVALID");
    assert!(
        session::cpu_ep_fallback_disabled_for_policy(OnnxProviderPolicy::CudaFailLoud).unwrap()
    );
    unsafe { std::env::remove_var("CALYX_ONNX_DISABLE_CPU_EP_FALLBACK") };
    if let Some(root) = calyx_fsv::fsv_root("CALYX_FSV_ROOT") {
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("onnx-cuda-fail-loud-cpu-fallback-readback.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "source_of_truth": "cpu_ep_fallback_disabled_for_policy + effective_audit_mode policy decisions",
                "cuda_fail_loud_default_disable_cpu_ep_fallback": false,
                "cuda_fail_loud_mandatory_audit_mode": "fail",
                "cuda_fail_loud_env_warn_cannot_downgrade": true,
                "strict_env_opt_in_disable_cpu_ep_fallback": true,
                "cpu_explicit_default_disable_cpu_ep_fallback": false,
                "cpu_explicit_env_error_code": error.code,
                "node_placement_verification": "mandatory_profiling_placement_audit",
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn cuda_graph_env_enables_cuda_provider_option() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_CUDA_GRAPHS", "1") };
    assert!(session::configured_cuda_graphs().unwrap());
    let providers = execution_providers(OnnxProviderPolicy::CudaFailLoud).unwrap();
    unsafe { std::env::remove_var("CALYX_ONNX_CUDA_GRAPHS") };

    let provider = format!("{:?}", providers[0]);
    assert!(provider.contains("CUDA"));
}

#[test]
fn cuda_graph_env_fails_closed_on_garbage() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_CUDA_GRAPHS", "maybe") };
    let error = execution_providers(OnnxProviderPolicy::CudaFailLoud).unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_CUDA_GRAPHS_INVALID");
    unsafe { std::env::remove_var("CALYX_ONNX_CUDA_GRAPHS") };
}

#[test]
fn cuda_graphs_require_gpu_policy_and_io_binding() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_CUDA_GRAPHS", "1") };
    let error = io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CpuExplicit, "cpu-graph-test")
        .unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_CUDA_GRAPHS_CPU_POLICY");

    unsafe { std::env::set_var("CALYX_ONNX_IO_BINDING", "0") };
    let error = io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "graph-no-bind")
        .unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_CUDA_GRAPHS_IO_BINDING");
    unsafe { std::env::remove_var("CALYX_ONNX_IO_BINDING") };

    unsafe { std::env::set_var("CALYX_ONNX_ARENA_SHRINK", "always") };
    let error =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "graph-shrink").unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_CUDA_GRAPHS_ARENA_SHRINK");
    unsafe { std::env::remove_var("CALYX_ONNX_ARENA_SHRINK") };
    unsafe { std::env::remove_var("CALYX_ONNX_CUDA_GRAPHS") };
}

#[test]
fn green_context_env_fails_closed_on_bad_config() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_GREEN_CONTEXT_SMS", "nope") };
    let error =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "green-bad").unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_GREEN_CONTEXT_SMS_INVALID");

    unsafe { std::env::set_var("CALYX_ONNX_GREEN_CONTEXT_SMS", "8") };
    let error =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CpuExplicit, "green-cpu").unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_GREEN_CONTEXT_CPU_POLICY");

    unsafe { std::env::set_var("CALYX_ONNX_CUDA_GRAPHS", "1") };
    let error =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "green-graph").unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_GREEN_CONTEXT_CUDA_GRAPHS");

    unsafe { std::env::remove_var("CALYX_ONNX_CUDA_GRAPHS") };
    unsafe { std::env::remove_var("CALYX_ONNX_GREEN_CONTEXT_SMS") };
}

#[test]
fn gpu_mem_limit_env_applies_and_fails_closed_on_garbage() {
    let _lock = arena_env_lock();
    // SAFETY: single-threaded within the arena env lock; restored before unlock.
    unsafe { std::env::set_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB", "4096") };
    assert_eq!(
        arena::configured_gpu_mem_limit().unwrap(),
        Some(4096 * 1024 * 1024)
    );
    assert!(execution_providers(OnnxProviderPolicy::CudaFailLoud).is_ok());

    unsafe { std::env::set_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB", "not-a-number") };
    let error = execution_providers(OnnxProviderPolicy::CudaFailLoud).unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_GPU_MEM_LIMIT_INVALID");

    unsafe { std::env::set_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB", "0") };
    let error = execution_providers(OnnxProviderPolicy::CudaFailLoud).unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_GPU_MEM_LIMIT_INVALID");

    unsafe { std::env::remove_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB") };
    assert_eq!(arena::configured_gpu_mem_limit().unwrap(), None);
    assert!(execution_providers(OnnxProviderPolicy::CudaFailLoud).is_ok());
}

#[test]
fn gpu_mem_limit_preflight_refuses_artifacts_larger_than_cap() {
    let _lock = arena_env_lock();
    let root = temp_root("onnx-arena-preflight");
    let model = root.join("model.onnx");
    let external = root.join("model.onnx.data");
    std::fs::write(&model, vec![0_u8; 700 * 1024]).unwrap();
    std::fs::write(&external, vec![0_u8; 500 * 1024]).unwrap();
    unsafe { std::env::set_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB", "1") };

    let error = arena::preflight_gpu_mem_limit_for_artifacts(
        "preflight-test",
        OnnxProviderPolicy::CudaFailLoud,
        [model.as_path(), external.as_path()],
    )
    .unwrap_err();
    assert_eq!(error.code, "CALYX_LENS_CONFIG_INVALID");
    assert!(error.message.contains("refused before ONNX session init"));
    assert!(error.message.contains("CALYX_ONNX_GPU_MEM_LIMIT_MIB=1 MiB"));

    unsafe { std::env::remove_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB") };
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn gpu_mem_limit_preflight_skips_cpu_policy() {
    let _lock = arena_env_lock();
    let root = temp_root("onnx-arena-cpu-preflight");
    let model = root.join("model.onnx");
    std::fs::write(&model, vec![0_u8; 2 * 1024 * 1024]).unwrap();
    unsafe { std::env::set_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB", "1") };

    arena::preflight_gpu_mem_limit_for_artifacts(
        "cpu-preflight-test",
        OnnxProviderPolicy::CpuExplicit,
        [model.as_path()],
    )
    .unwrap();

    unsafe { std::env::remove_var("CALYX_ONNX_GPU_MEM_LIMIT_MIB") };
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn arena_shrink_env_fails_closed_on_garbage() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_ARENA_SHRINK", "sometimes") };
    let error = io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "shrink-env-test")
        .unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_ARENA_SHRINK_INVALID");
    unsafe { std::env::remove_var("CALYX_ONNX_ARENA_SHRINK") };
}

#[test]
fn arena_shrink_defaults_to_every_run_for_long_lived_gpu_sessions() {
    let _lock = arena_env_lock();
    unsafe { std::env::remove_var("CALYX_ONNX_ARENA_SHRINK") };

    assert_eq!(
        arena::configured_arena_shrink().unwrap(),
        arena::ArenaShrinkPolicy::Always
    );
}

#[test]
fn arena_shrink_keeps_new_shape_as_an_explicit_diagnostic_policy() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_ARENA_SHRINK", "new-shape") };

    assert_eq!(
        arena::configured_arena_shrink().unwrap(),
        arena::ArenaShrinkPolicy::NewShape
    );

    unsafe { std::env::remove_var("CALYX_ONNX_ARENA_SHRINK") };
}

#[test]
fn max_distinct_shapes_env_fails_closed_on_garbage() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_MAX_DISTINCT_SHAPES", "-3") };
    let error =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "shape-limit-env-test")
            .unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_SHAPE_LIMIT_INVALID");
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };
}

#[test]
fn gpu_shape_diversity_fails_loud_past_the_cap() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_MAX_DISTINCT_SHAPES", "4") };
    let mut plan =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CudaFailLoud, "shape-cap-test").unwrap();
    for batch in 1..=4_usize {
        assert!(plan.enforce_shape_contract((batch, 128)).unwrap());
        // Repeats of a seen shape never trip the cap.
        assert!(!plan.enforce_shape_contract((batch, 128)).unwrap());
    }
    let error = plan.enforce_shape_contract((5, 128)).unwrap_err();
    assert_eq!(error.code, "CALYX_ONNX_SHAPE_DIVERSITY");
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };
}

#[test]
fn cpu_sessions_do_not_gate_shape_diversity() {
    let _lock = arena_env_lock();
    unsafe { std::env::set_var("CALYX_ONNX_MAX_DISTINCT_SHAPES", "2") };
    let mut plan =
        io_binding::OnnxRunPlan::new(OnnxProviderPolicy::CpuExplicit, "cpu-shape-test").unwrap();
    for batch in 1..=8_usize {
        plan.enforce_shape_contract((batch, 64)).unwrap();
    }
    unsafe { std::env::remove_var("CALYX_ONNX_MAX_DISTINCT_SHAPES") };
}
