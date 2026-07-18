use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub(super) struct Telemetry {
    pub(super) batches: AtomicU64,
    pub(super) queries: AtomicU64,
    pub(super) cagra_kernel_launches: AtomicU64,
    pub(super) exact_filter_kernel_launches: AtomicU64,
    pub(super) partitioned_exact_kernel_launches: AtomicU64,
    pub(super) partitioned_merge_kernel_launches: AtomicU64,
    pub(super) partitioned_prepare_us: AtomicU64,
    pub(super) partitioned_execute_us: AtomicU64,
    pub(super) partitioned_scratch_bytes: AtomicU64,
    pub(super) partitioned_i8_dataset_loads: AtomicU64,
    pub(super) partitioned_f32_dataset_loads: AtomicU64,
    pub(super) query_uploads: AtomicU64,
    pub(super) filter_uploads: AtomicU64,
    pub(super) h2d_bytes: AtomicU64,
    pub(super) d2h_bytes: AtomicU64,
    pub(super) final_readback_pairs: AtomicU64,
    pub(super) failures: AtomicU64,
}

pub(super) struct TelemetrySnapshot {
    pub(super) batches: u64,
    pub(super) queries: u64,
    pub(super) cagra_kernel_launches: u64,
    pub(super) exact_filter_kernel_launches: u64,
    pub(super) partitioned_exact_kernel_launches: u64,
    pub(super) partitioned_merge_kernel_launches: u64,
    pub(super) partitioned_prepare_us: u64,
    pub(super) partitioned_execute_us: u64,
    pub(super) partitioned_scratch_bytes: u64,
    pub(super) partitioned_i8_dataset_loads: u64,
    pub(super) partitioned_f32_dataset_loads: u64,
    pub(super) query_uploads: u64,
    pub(super) filter_uploads: u64,
    pub(super) h2d_bytes: u64,
    pub(super) d2h_bytes: u64,
    pub(super) final_readback_pairs: u64,
    pub(super) failures: u64,
}

pub(super) static TELEMETRY: Telemetry = Telemetry {
    batches: AtomicU64::new(0),
    queries: AtomicU64::new(0),
    cagra_kernel_launches: AtomicU64::new(0),
    exact_filter_kernel_launches: AtomicU64::new(0),
    partitioned_exact_kernel_launches: AtomicU64::new(0),
    partitioned_merge_kernel_launches: AtomicU64::new(0),
    partitioned_prepare_us: AtomicU64::new(0),
    partitioned_execute_us: AtomicU64::new(0),
    partitioned_scratch_bytes: AtomicU64::new(0),
    partitioned_i8_dataset_loads: AtomicU64::new(0),
    partitioned_f32_dataset_loads: AtomicU64::new(0),
    query_uploads: AtomicU64::new(0),
    filter_uploads: AtomicU64::new(0),
    h2d_bytes: AtomicU64::new(0),
    d2h_bytes: AtomicU64::new(0),
    final_readback_pairs: AtomicU64::new(0),
    failures: AtomicU64::new(0),
};

pub(super) fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

pub(super) fn telemetry_snapshot() -> TelemetrySnapshot {
    TelemetrySnapshot {
        batches: TELEMETRY.batches.load(Ordering::Relaxed),
        queries: TELEMETRY.queries.load(Ordering::Relaxed),
        cagra_kernel_launches: TELEMETRY.cagra_kernel_launches.load(Ordering::Relaxed),
        exact_filter_kernel_launches: TELEMETRY
            .exact_filter_kernel_launches
            .load(Ordering::Relaxed),
        partitioned_exact_kernel_launches: TELEMETRY
            .partitioned_exact_kernel_launches
            .load(Ordering::Relaxed),
        partitioned_merge_kernel_launches: TELEMETRY
            .partitioned_merge_kernel_launches
            .load(Ordering::Relaxed),
        partitioned_prepare_us: TELEMETRY.partitioned_prepare_us.load(Ordering::Relaxed),
        partitioned_execute_us: TELEMETRY.partitioned_execute_us.load(Ordering::Relaxed),
        partitioned_scratch_bytes: TELEMETRY.partitioned_scratch_bytes.load(Ordering::Relaxed),
        partitioned_i8_dataset_loads: TELEMETRY
            .partitioned_i8_dataset_loads
            .load(Ordering::Relaxed),
        partitioned_f32_dataset_loads: TELEMETRY
            .partitioned_f32_dataset_loads
            .load(Ordering::Relaxed),
        query_uploads: TELEMETRY.query_uploads.load(Ordering::Relaxed),
        filter_uploads: TELEMETRY.filter_uploads.load(Ordering::Relaxed),
        h2d_bytes: TELEMETRY.h2d_bytes.load(Ordering::Relaxed),
        d2h_bytes: TELEMETRY.d2h_bytes.load(Ordering::Relaxed),
        final_readback_pairs: TELEMETRY.final_readback_pairs.load(Ordering::Relaxed),
        failures: TELEMETRY.failures.load(Ordering::Relaxed),
    }
}
