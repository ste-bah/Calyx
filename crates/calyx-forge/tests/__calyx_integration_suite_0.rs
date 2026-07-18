//! Generated integration-test harness. Regenerate with
//! `python scripts/consolidate_integration_tests.py`.

#[allow(
    dead_code,
    reason = "shared integration support is used selectively by each harness"
)]
#[path = "cuda_parity_support.rs"]
mod __calyx_shared_cuda_parity_support_rs;

#[path = "autotune_tests.rs"]
mod autotune_tests;
#[path = "compression_report.rs"]
mod compression_report;
#[path = "cpu_kernels.rs"]
mod cpu_kernels;
#[path = "cuda_parity.rs"]
mod cuda_parity;
#[path = "grouped_gemm_mode_fsv.rs"]
mod grouped_gemm_mode_fsv;
#[path = "grouped_gemm_tests.rs"]
mod grouped_gemm_tests;
#[path = "issue935_compression_report_bytes_fsv.rs"]
mod issue935_compression_report_bytes_fsv;
#[path = "ph57_admission_fsv.rs"]
mod ph57_admission_fsv;
#[path = "ph57_oom_guard_fsv.rs"]
mod ph57_oom_guard_fsv;
#[path = "ph57_vram_fsv.rs"]
mod ph57_vram_fsv;
#[path = "ph57_yield_policy_fsv.rs"]
mod ph57_yield_policy_fsv;
#[path = "turboquant_tests.rs"]
mod turboquant_tests;
