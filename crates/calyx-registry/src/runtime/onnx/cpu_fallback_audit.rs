//! Per-provider node-placement audit for GPU-policy ONNX sessions (#1142,
//! #1487).
//!
//! The CUDA execution provider has no kernels for int8-quantized ops
//! (`QLinearMatMul`, `QGemm`, `MatMulInteger`, `DynamicQuantizeLinear`,
//! `ConvInteger`, …). ORT silently places those nodes on the implicit CPU EP,
//! so a session that reports `provider=CudaFailLoud` can execute most of its
//! compute on the CPU with a device↔host copy per node — measured at
//! 130–250 ms/input, unusable for bulk encode. The `session_ready` telemetry
//! said "gpu" while execution was CPU-bound (#1142), and `#1136`'s I/O binding
//! cannot fix it — it addresses the copy path of GPU-executable graphs.
//!
//! #1287 answered that with ORT `session.disable_cpu_ep_fallback=1` for every
//! `CudaFailLoud` session, but that knob is all-or-nothing: real BERT/XLM-R
//! ONNX exports always leave a handful of trivial shape-bookkeeping nodes
//! (`Shape`/`Gather`/`Concat` over int64 scalars) on the CPU EP, so
//! zero-tolerance refused every real onnx-custom lens at Initialize (#1487).
//! The `CudaFailLoud` semantic is therefore enforced in two verifiable parts:
//! CUDA EP registration fails loud at session build (`error_on_failure`), and
//! this placement audit — **mandatory** for `CudaFailLoud`, always in `fail`
//! mode — refuses the session at its first inference if the model did not
//! substantially execute on GPU: any heavy compute op on CPU, or a CPU
//! compute-node fraction over the configured budget. Trivial CPU-assigned ops
//! are tolerated and listed once in the audit log line, never silently.
//!
//! This audit parses the ORT profiling trace after the first real run (the
//! vendored ORT 1.26 C API exposes no per-node EP assignment query, so the
//! profiling trace — which records `provider` and `op_name` per executed
//! kernel — is the strongest introspection available), counts compute nodes
//! per execution provider, emits the per-provider counts in the readback
//! telemetry (so any fallback is *visible*, not inferred), and — in `fail`
//! mode — refuses a GPU-policy session that violates placement. Every capture
//! failure (profiling unavailable, unreadable, unparseable) is a structured
//! error: the audit never silently skips, and a session whose audit failed is
//! poisoned for all later runs. The pure parsing and policy functions are
//! exercised directly by unit tests with synthetic traces; the GPU run that
//! populates a real trace is exercised by the lens runtimes on device.
//!
//! Environment knobs:
//! - `CALYX_ONNX_CPU_FALLBACK_AUDIT` — `off` (default) | `warn` | `fail`.
//!   For `CudaFailLoud` sessions (without the strict ORT knob below) the
//!   effective mode is always `fail`; the env cannot weaken the policy, and an
//!   attempted downgrade is logged. For other sessions the env applies as
//!   configured. When the effective mode is not `off`, ORT profiling is
//!   enabled at session build and the audit runs once after the first
//!   successful inference.
//! - `CALYX_ONNX_MAX_CPU_NODE_FRACTION` — CPU compute-node fraction a GPU-policy
//!   session may reach before `fail` refuses it (default 0.10, range [0,1]).
//!   Heavy compute ops on CPU are refused regardless of this fraction.
//! - `CALYX_ONNX_DISABLE_CPU_EP_FALLBACK=1` (see `session.rs`) — zero-tolerance
//!   opt-in: sets the ORT build-time knob that refuses *any* CPU-assigned node
//!   at Initialize. Strictly stronger than this audit, so the audit is not
//!   additionally mandated when it is set.
//! - `CALYX_ONNX_GPU_STRICT_GATES_ADVISORY=1` (ste-bah fork opt-out) — downgrades
//!   the mandatory `CudaFailLoud` fail to honor the configured mode (`warn`), so
//!   a validated lens set with intrinsic partial-CPU placement loads while the
//!   audit still logs every verdict loudly. Also lifts the FastEmbed host-owned
//!   output block (#1570). See `gpu_strict_gates_advisory` / `effective_audit_mode`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use calyx_core::{CalyxError, Result};
use serde_json::Value;

pub(super) const CPU_FALLBACK_AUDIT_ENV: &str = "CALYX_ONNX_CPU_FALLBACK_AUDIT";
pub(super) const MAX_CPU_NODE_FRACTION_ENV: &str = "CALYX_ONNX_MAX_CPU_NODE_FRACTION";
pub(super) const CPU_FALLBACK_CODE: &str = "CALYX_ONNX_QUANT_CPU_FALLBACK";

const DEFAULT_MAX_CPU_FRACTION: f64 = 0.10;

/// Op classes that constitute the substantial compute of transformer / CNN /
/// recurrent ONNX graphs (dense and quantized forms, plus the contrib fusions
/// ORT's graph optimizer emits for them). Real transformer exports always
/// leave a few trivial bookkeeping nodes (`Shape`, `Gather`, `Cast`, `Concat`
/// over int64 scalars) on the CPU EP — ORT has no CUDA kernels for those and
/// their cost is negligible, which is why zero CPU nodes is unattainable
/// (#1487). But if any op in *this* set lands on CPU, the model's real math is
/// running off-device — exactly the silent perf cliff #1287/#1142 exist to
/// prevent — so a GPU-policy session refuses it regardless of the CPU-node
/// fraction. Kept sorted (asserted in tests); matching is exact on the ORT
/// `op_name`.
pub(super) const HEAVY_COMPUTE_OPS: &[&str] = &[
    "Attention",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "DynamicQuantizeLinear",
    "Einsum",
    "EmbedLayerNormalization",
    "FusedGemm",
    "FusedMatMul",
    "GRU",
    "Gemm",
    "GroupQueryAttention",
    "LSTM",
    "LayerNormalization",
    "MatMul",
    "MatMulInteger",
    "MatMulNBits",
    "MultiHeadAttention",
    "QAttention",
    "QGemm",
    "QLinearConv",
    "QLinearMatMul",
    "SimplifiedLayerNormalization",
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
];

fn is_heavy_compute_op(op_name: &str) -> bool {
    HEAVY_COMPUTE_OPS.contains(&op_name)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AuditMode {
    Off,
    Warn,
    Fail,
}

impl AuditMode {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Warn => "warn",
            Self::Fail => "fail",
        }
    }

    pub(super) const fn enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

pub(super) fn configured_audit_mode() -> Result<AuditMode> {
    let Ok(raw) = std::env::var(CPU_FALLBACK_AUDIT_ENV) else {
        return Ok(AuditMode::Off);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "0" | "false" => Ok(AuditMode::Off),
        "warn" => Ok(AuditMode::Warn),
        "fail" | "1" | "true" => Ok(AuditMode::Fail),
        other => Err(CalyxError {
            code: "CALYX_ONNX_CPU_FALLBACK_AUDIT_INVALID",
            message: format!("{CPU_FALLBACK_AUDIT_ENV}={other} is not off, warn, or fail"),
            remediation: "set CALYX_ONNX_CPU_FALLBACK_AUDIT to off, warn, or fail (default off)",
        }),
    }
}

/// FORK DIVERGENCE (ste-bah, import #6): setting
/// `CALYX_ONNX_GPU_STRICT_GATES_ADVISORY=1` downgrades import #6's new strict
/// GPU gates from fatal to advisory for a validated lens set. It covers both
/// (a) this mandatory `CudaFailLoud` placement audit — it then honors the
/// configured `CALYX_ONNX_CPU_FALLBACK_AUDIT` mode (we run `warn`) instead of
/// force-failing — and (b) the FastEmbed host-owned-output block (#1570) that
/// otherwise refuses every FastEmbed sparse/reranker/onnx lens under
/// `CudaFailLoud`. Our production book vaults (tolkien-full etc.) run lenses —
/// scincl fp32 (~28% CPU nodes), int8 bge-small (int8 heavy ops), colbert,
/// splade (FastEmbed sparse) — whose real ONNX graphs legitimately place a
/// substantial fraction of nodes, including some heavy ops with no CUDA kernel,
/// on the CPU EP, and which are verified in reality to produce correct vectors
/// at acceptable latency across months of generation. Upstream's fatal verdict
/// is a false positive for them. The audit still RUNS and LOGS every lens's
/// placement loudly under `warn` (never silent — the perf-cliff visibility
/// #1287/#1487 exist for is preserved); only the fatal abort is downgraded.
/// Unset (the default) keeps upstream's policy for every other deployment.
pub(super) const GPU_STRICT_GATES_ADVISORY_ENV: &str = "CALYX_ONNX_GPU_STRICT_GATES_ADVISORY";

pub(super) fn gpu_strict_gates_advisory() -> bool {
    std::env::var(GPU_STRICT_GATES_ADVISORY_ENV)
        .ok()
        .is_some_and(|raw| matches!(raw.trim().to_ascii_lowercase().as_str(), "1" | "true" | "warn"))
}

/// The audit mode a session actually runs under. For a GPU-policy session
/// whose build did not set the strict ORT `disable_cpu_ep_fallback` knob, the
/// placement audit is the *only* verification that the model executes on the
/// GPU, so it is mandatory and always `fail` — `CudaFailLoud` would otherwise
/// be a provider-list claim, the silent-fallback hole #1287 closed (#1487).
/// The environment can therefore only strengthen other sessions, never weaken
/// this one; an attempted downgrade is logged, not honored — except under the
/// explicit `CALYX_ONNX_GPU_STRICT_GATES_ADVISORY` fork opt-out above.
pub(super) fn effective_audit_mode(
    gpu_policy: bool,
    ort_cpu_ep_fallback_disabled: bool,
) -> Result<AuditMode> {
    let configured = configured_audit_mode()?;
    if gpu_policy && !ort_cpu_ep_fallback_disabled && configured != AuditMode::Fail {
        if gpu_strict_gates_advisory() {
            eprintln!(
                "CALYX_ONNX_RUNTIME phase=cpu_fallback_audit_mode configured={} effective={} reason=gpu_strict_gates_advisory_fork_optout",
                configured.as_str(),
                configured.as_str()
            );
            return Ok(configured);
        }
        if configured == AuditMode::Warn {
            eprintln!(
                "CALYX_ONNX_RUNTIME phase=cpu_fallback_audit_mode configured=warn effective=fail reason=cuda_fail_loud_mandatory_placement_audit"
            );
        }
        return Ok(AuditMode::Fail);
    }
    Ok(configured)
}

pub(super) fn configured_max_cpu_fraction() -> Result<f64> {
    let Ok(raw) = std::env::var(MAX_CPU_NODE_FRACTION_ENV) else {
        return Ok(DEFAULT_MAX_CPU_FRACTION);
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(DEFAULT_MAX_CPU_FRACTION);
    }
    raw.parse::<f64>()
        .ok()
        .filter(|fraction| fraction.is_finite() && (0.0..=1.0).contains(fraction))
        .ok_or_else(|| CalyxError {
            code: "CALYX_ONNX_MAX_CPU_NODE_FRACTION_INVALID",
            message: format!("{MAX_CPU_NODE_FRACTION_ENV}={raw} is not a fraction in [0, 1]"),
            remediation: "set CALYX_ONNX_MAX_CPU_NODE_FRACTION to a value in [0, 1] (default 0.10), or unset it",
        })
}

/// A unique, writable profiling trace path for a session. ORT appends its own
/// timestamp and `.json` suffix and returns the final path from `end_profiling`.
pub(super) fn profiling_file_path(label: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let slug: String = label
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    std::env::temp_dir().join(format!(
        "calyx_onnx_profile_{}_{}_{seq}",
        std::process::id(),
        slug
    ))
}

/// Compute-node counts keyed by ORT execution-provider name, plus the op
/// types of the CPU-assigned nodes (the evidence a placement verdict names).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ProviderNodeCounts {
    counts: BTreeMap<String, usize>,
    cpu_ops: BTreeMap<String, usize>,
}

impl ProviderNodeCounts {
    fn add(&mut self, provider: &str, op_name: &str) {
        *self.counts.entry(provider.to_string()).or_default() += 1;
        if is_cpu_provider(provider) {
            let op = if op_name.trim().is_empty() {
                "unknown"
            } else {
                op_name
            };
            *self.cpu_ops.entry(op.to_string()).or_default() += 1;
        }
    }

    fn total(&self) -> usize {
        self.counts.values().copied().sum()
    }

    fn cpu_nodes(&self) -> usize {
        self.counts
            .iter()
            .filter(|(provider, _)| is_cpu_provider(provider))
            .map(|(_, count)| *count)
            .sum()
    }

    fn render(&self) -> String {
        render_op_counts(&self.counts)
    }

    /// Split the CPU-assigned ops into (heavy compute, tolerable trivial).
    fn cpu_op_partition(&self) -> (BTreeMap<String, usize>, BTreeMap<String, usize>) {
        let mut heavy = BTreeMap::new();
        let mut trivial = BTreeMap::new();
        for (op, count) in &self.cpu_ops {
            if is_heavy_compute_op(op) {
                heavy.insert(op.clone(), *count);
            } else {
                trivial.insert(op.clone(), *count);
            }
        }
        (heavy, trivial)
    }
}

fn render_op_counts(counts: &BTreeMap<String, usize>) -> String {
    if counts.is_empty() {
        return "none".to_string();
    }
    counts
        .iter()
        .map(|(key, count)| format!("{key}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn is_cpu_provider(provider: &str) -> bool {
    provider.to_ascii_uppercase().contains("CPU")
}

/// Parse an ORT profiling trace into per-provider compute-node counts.
///
/// ORT emits three events per node (`_fence_before`, `_kernel_time`,
/// `_fence_after`); the `_kernel_time` record is the actual compute and carries
/// `args.provider` plus `args.op_name`. We count those. Some ORT builds omit
/// the suffix, so if no `_kernel_time` records are present we fall back to
/// every `cat=="Node"` event carrying a provider.
pub(super) fn parse_profiling_nodes(trace_json: &str) -> Result<ProviderNodeCounts> {
    let value: Value = serde_json::from_str(trace_json).map_err(|err| CalyxError {
        code: "CALYX_ONNX_PROFILE_PARSE",
        message: format!("ONNX profiling trace is not valid JSON: {err}"),
        remediation: "the ORT profiling trace is malformed; rerun with CALYX_ONNX_CPU_FALLBACK_AUDIT and check ORT version",
    })?;
    let events = match &value {
        Value::Array(events) => events.as_slice(),
        Value::Object(map) => match map.get("traceEvents") {
            Some(Value::Array(events)) => events.as_slice(),
            _ => {
                return Err(CalyxError {
                    code: "CALYX_ONNX_PROFILE_PARSE",
                    message: "ONNX profiling trace object has no traceEvents array".to_string(),
                    remediation: "expected an ORT profiling trace (JSON array or {traceEvents:[...]})",
                });
            }
        },
        _ => {
            return Err(CalyxError {
                code: "CALYX_ONNX_PROFILE_PARSE",
                message: "ONNX profiling trace is neither an array nor a traceEvents object"
                    .to_string(),
                remediation: "expected an ORT profiling trace (JSON array or {traceEvents:[...]})",
            });
        }
    };

    let mut kernel = ProviderNodeCounts::default();
    let mut any_node = ProviderNodeCounts::default();
    for event in events {
        let Some(obj) = event.as_object() else {
            continue;
        };
        if obj.get("cat").and_then(Value::as_str) != Some("Node") {
            continue;
        }
        let Some(args) = obj.get("args").and_then(Value::as_object) else {
            continue;
        };
        let Some(provider) = args
            .get("provider")
            .and_then(Value::as_str)
            .filter(|provider| !provider.trim().is_empty())
        else {
            continue;
        };
        let op_name = args.get("op_name").and_then(Value::as_str).unwrap_or("");
        any_node.add(provider, op_name);
        if obj
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|name| name.ends_with("_kernel_time"))
        {
            kernel.add(provider, op_name);
        }
    }
    Ok(if kernel.total() > 0 { kernel } else { any_node })
}

/// The verdict of a placement audit — the numbers that also go to telemetry.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CpuFallbackAudit {
    pub(super) total_nodes: usize,
    pub(super) cpu_nodes: usize,
    pub(super) cpu_fraction: f64,
    pub(super) max_cpu_fraction: f64,
    /// CPU-node fraction breached the configured budget (GPU policy only).
    pub(super) over_threshold: bool,
    /// Heavy compute ops (see [`HEAVY_COMPUTE_OPS`]) assigned to CPU under a
    /// GPU policy, rendered `Op:count,...`; `"none"` when placement is clean.
    pub(super) heavy_cpu_ops: String,
    pub(super) heavy_cpu_op_count: usize,
    /// Trivial CPU-assigned ops that were tolerated, rendered `Op:count,...`.
    pub(super) tolerated_cpu_ops: String,
    pub(super) per_provider: String,
}

impl CpuFallbackAudit {
    /// A GPU-policy session with this placement must be refused in `fail`
    /// mode: heavy compute off-device, or too much of the graph on CPU.
    pub(super) const fn violation(&self) -> bool {
        self.over_threshold || self.heavy_cpu_op_count > 0
    }
}

pub(super) fn evaluate_placement(
    counts: &ProviderNodeCounts,
    gpu_policy: bool,
    max_cpu_fraction: f64,
) -> CpuFallbackAudit {
    let total_nodes = counts.total();
    let cpu_nodes = counts.cpu_nodes();
    let cpu_fraction = if total_nodes == 0 {
        0.0
    } else {
        cpu_nodes as f64 / total_nodes as f64
    };
    // Strictly greater than the budget so an exact-threshold panel passes, and
    // a session with no measured nodes never trips (nothing to judge).
    let over_threshold = gpu_policy && total_nodes > 0 && cpu_fraction > max_cpu_fraction;
    let (heavy, trivial) = counts.cpu_op_partition();
    let heavy_cpu_op_count = if gpu_policy {
        heavy.values().copied().sum()
    } else {
        0
    };
    CpuFallbackAudit {
        total_nodes,
        cpu_nodes,
        cpu_fraction,
        max_cpu_fraction,
        over_threshold,
        heavy_cpu_ops: render_op_counts(&heavy),
        heavy_cpu_op_count,
        tolerated_cpu_ops: render_op_counts(&trivial),
        per_provider: counts.render(),
    }
}

/// Emit the audit telemetry line and — in `fail` mode — refuse a GPU-policy
/// session whose placement is a violation. The success path logs the tolerated
/// trivial CPU ops once per session so toleration is visible, never silent.
pub(super) fn enforce_audit(
    label: &str,
    audit: &CpuFallbackAudit,
    gpu_policy: bool,
    mode: AuditMode,
) -> Result<()> {
    let verdict = if audit.violation() { "violation" } else { "ok" };
    eprintln!(
        "CALYX_ONNX_RUNTIME phase=cpu_fallback_audit label={label} mode={} gpu_policy={gpu_policy} total_nodes={} cpu_nodes={} cpu_fraction={:.4} max_cpu_fraction={:.4} providers={} heavy_cpu_ops={} tolerated_cpu_ops={} verdict={verdict}",
        mode.as_str(),
        audit.total_nodes,
        audit.cpu_nodes,
        audit.cpu_fraction,
        audit.max_cpu_fraction,
        audit.per_provider,
        audit.heavy_cpu_ops,
        audit.tolerated_cpu_ops,
    );
    if mode != AuditMode::Fail || !audit.violation() {
        return Ok(());
    }
    if audit.heavy_cpu_op_count > 0 {
        return Err(CalyxError {
            code: CPU_FALLBACK_CODE,
            message: format!(
                "{label} claims a GPU execution provider but ORT assigned heavy compute ops to the CPU EP: {} ({}/{} compute nodes = {:.1}% on CPU, providers={}) — MatMul/Gemm/Attention-class nodes on CPU mean the model is not substantially executing on GPU, typically because the graph is int8-quantized (no CUDA kernels) or the op's dtype has no CUDA implementation",
                audit.heavy_cpu_ops,
                audit.cpu_nodes,
                audit.total_nodes,
                audit.cpu_fraction * 100.0,
                audit.per_provider,
            ),
            remediation: "use the fp16/fp32 ONNX variant of this lens on GPU (int8 graphs have no CUDA kernels), or run this lens under CPU policy; CALYX_ONNX_DISABLE_CPU_EP_FALLBACK=1 remains the zero-tolerance opt-in",
        });
    }
    Err(CalyxError {
        code: CPU_FALLBACK_CODE,
        message: format!(
            "{label} claims a GPU execution provider but ran {}/{} compute nodes ({:.1}%) on CPU (providers={}, cpu_ops={}), exceeding {MAX_CPU_NODE_FRACTION_ENV}={:.4} — each CPU-assigned node costs a device<->host copy each way",
            audit.cpu_nodes,
            audit.total_nodes,
            audit.cpu_fraction * 100.0,
            audit.per_provider,
            audit.tolerated_cpu_ops,
            audit.max_cpu_fraction,
        ),
        remediation: "prefer the fp16/fp32 ONNX variant of this lens for bulk encode (assay corpus-build / stream-fbin / ingest); keep int8 graphs for resident low-VRAM serving under CPU policy, or raise CALYX_ONNX_MAX_CPU_NODE_FRACTION only if this mixed placement is expected",
    })
}

/// Parse the trace, evaluate placement, emit telemetry, and — in `fail` mode —
/// refuse a GPU-policy session with a placement violation.
pub(super) fn audit_from_trace(
    label: &str,
    trace_json: &str,
    gpu_policy: bool,
    mode: AuditMode,
    max_cpu_fraction: f64,
) -> Result<CpuFallbackAudit> {
    let counts = parse_profiling_nodes(trace_json)?;
    let audit = evaluate_placement(&counts, gpu_policy, max_cpu_fraction);
    enforce_audit(label, &audit, gpu_policy, mode)?;
    Ok(audit)
}

#[cfg(test)]
mod tests;
