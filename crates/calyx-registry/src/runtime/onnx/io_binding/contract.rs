use calyx_core::{CalyxError, Result};
use ort::session::Session;

use super::OnnxRunPlan;
use crate::runtime::onnx::arena::MAX_DISTINCT_SHAPES_ENV;
use crate::runtime::onnx::config_invalid;
use crate::runtime::onnx::cpu_fallback_audit::{CPU_FALLBACK_CODE, audit_from_trace};
use crate::runtime::onnx::session::REQUIRE_STATIC_BINDING_ENV;

impl OnnxRunPlan {
    /// Once per session, after the first successful run, read the ORT profiling
    /// trace and audit per-provider node placement (#1142, #1487). Mandatory
    /// (`fail` mode) for GPU-policy sessions built without the strict ORT
    /// `disable_cpu_ep_fallback` knob; otherwise a no-op unless the operator
    /// enabled CALYX_ONNX_CPU_FALLBACK_AUDIT. Marked audited even on error so
    /// a failing gate does not re-run profiling every batch — and any failure
    /// (placement violation *or* an unavailable capture mechanism) poisons the
    /// plan so later runs are refused instead of silently proceeding.
    pub(super) fn audit_placement_once(&mut self, session: &mut Session) -> Result<()> {
        if !self.audit_mode.enabled() || self.audited {
            return Ok(());
        }
        self.audited = true;
        let result = self.run_placement_audit(session);
        if result.is_err() {
            self.audit_poisoned = true;
        }
        result
    }

    fn run_placement_audit(&mut self, session: &mut Session) -> Result<()> {
        let trace_path = session.end_profiling().map_err(|err| {
            config_invalid(format!(
                "ONNX end_profiling failed for {}: {err}",
                self.label
            ))
        })?;
        let trace = std::fs::read_to_string(&trace_path).map_err(|err| {
            config_invalid(format!(
                "read ONNX profiling trace {trace_path} failed for {}: {err}",
                self.label
            ))
        })?;
        audit_from_trace(
            &self.label,
            &trace,
            self.gpu_policy,
            self.audit_mode,
            self.max_cpu_fraction,
        )
        .map(|_| ())
    }

    pub(super) fn refuse_if_audit_poisoned(&self) -> Result<()> {
        if !self.audit_poisoned {
            return Ok(());
        }
        Err(CalyxError {
            code: CPU_FALLBACK_CODE,
            message: format!(
                "{} was refused by the node-placement audit after its first run; the session stays refused for every later run",
                self.label
            ),
            remediation: "see the first CALYX_ONNX_RUNTIME phase=cpu_fallback_audit failure for this label; fix the model placement (fp16/fp32 export) or run the lens under CPU policy",
        })
    }

    /// Records the run shape; returns whether it is first-seen. GPU sessions
    /// fail loud when distinct-shape diversity exceeds the configured cap —
    /// the ORT CUDA BFC arena retains per-shape allocations forever, so
    /// unbounded diversity is a slow-motion device OOM (#1143), and the
    /// batch/seq bucketing upstream keeps legitimate streams far below the
    /// cap.
    pub(in crate::runtime::onnx) fn enforce_shape_contract(
        &mut self,
        shape: (usize, usize),
    ) -> Result<bool> {
        let new_shape = self.seen_shapes.insert(shape);
        if new_shape {
            eprintln!(
                "CALYX_ONNX_RUNTIME phase=io_binding_shape label={} batch={} seq={} io_binding={} distinct_shapes={}",
                self.label,
                shape.0,
                shape.1,
                self.io_binding,
                self.seen_shapes.len()
            );
            if self.gpu_policy && self.seen_shapes.len() > self.max_distinct_shapes {
                return Err(CalyxError {
                    code: "CALYX_ONNX_SHAPE_DIVERSITY",
                    message: format!(
                        "{} has run {} distinct (batch, seq) shapes, exceeding {MAX_DISTINCT_SHAPES_ENV}={} — unbounded shape diversity grows the CUDA BFC arena until device OOM (new shape batch={} seq={})",
                        self.label,
                        self.seen_shapes.len(),
                        self.max_distinct_shapes,
                        shape.0,
                        shape.1
                    ),
                    remediation: "batch and sequence bucketing should cap distinct shapes; find the caller that bypasses bucketed batching, or raise CALYX_ONNX_MAX_DISTINCT_SHAPES only if the workload legitimately needs more shape classes",
                });
            }
        }
        if !self.require_static {
            return Ok(new_shape);
        }
        match self.bound_shape {
            None => {
                self.bound_shape = Some(shape);
                Ok(new_shape)
            }
            Some(bound) if bound == shape => Ok(new_shape),
            Some(bound) => Err(CalyxError {
                code: "CALYX_ONNX_STATIC_BINDING_SHAPE",
                message: format!(
                    "{} requires the captured static binding shape batch={} seq={} but received batch={} seq={} under {REQUIRE_STATIC_BINDING_ENV}=1",
                    self.label, bound.0, bound.1, shape.0, shape.1
                ),
                remediation: "bucket inputs to the captured shape (fixed batch and sequence length) or unset CALYX_ONNX_REQUIRE_STATIC_BINDING to allow per-shape rebinding",
            }),
        }
    }
}
