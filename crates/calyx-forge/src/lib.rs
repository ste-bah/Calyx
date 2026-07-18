//! Forge math runtime skeleton for CPU, CUDA, and quantized kernels.

/// True when this build compiled the CUDA kernel backend (`cuda` feature).
/// Exported for build-info capability readback (#1130): deploy gates assert
/// this resolved value, not a top-level feature spelling.
pub const CUDA_COMPILED: bool = cfg!(feature = "cuda");

pub mod autotune;
mod backend;
pub mod compression_report;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
mod error;
#[path = "cuda/mxfp4.rs"]
pub mod mxfp4;
#[path = "cuda/mxfp8.rs"]
pub mod mxfp8;
pub mod quant;
pub mod vram;

pub use autotune::{
    AbHook, AutotuneCache, AutotuneKey, BenchCudaContext, BenchResult, EPSILON, Explorer,
    ExplorerPolicy, MIN_PROMOTE_MARGIN, MIN_PROMOTE_TRIALS, PROMOTION_LEDGER_SCHEMA_VERSION,
    PromotionAction, PromotionEvent, autotune, decode_promotion_ledger_payload, log_promotion,
    microbench, next_candidate, promote_if_winner, promotion_ledger_events,
    promotion_ledger_subject, record_trial, rollback_promotion, should_promote,
    should_use_challenger,
};
pub use backend::{
    Backend, BackendKind, BestConfig, CUDA_EXACT_TOPK_MAX_K, DeviceInfo,
    FORGE_DEFERRED_BACKEND_OPS, FORGE_SHIPPED_BACKEND_OPS, Result,
};
pub use compression_report::{
    COMPRESSION_REPORT_SCHEMA_VERSION, CompressionReport, CompressionReportInput,
    CompressionSlotMeasurement, CompressionSlotReport, CompressionTotals, IntelligenceDeltaReport,
    KernelCompressionMeasurement, KernelCompressionReport, compression_report,
};
pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::{
    ALGORITHMIC_SPARSE_MAX_TOKEN_BYTES, ALGORITHMIC_TOKEN_HASH_MAX_TOKEN_BYTES, AbsentSlotSentinel,
    BINARY_CUDA_MIN_ELEMENTS, CUDA_ENERGY_MAX_DEVICE_BYTES, CUDA_ENERGY_PINNED_CHUNK_BYTES,
    CUDA_GRANGER_STATUS_INVALID_LAG, CUDA_GRANGER_STATUS_NONFINITE, CUDA_GRANGER_STATUS_OK,
    CUDA_GRANGER_STATUS_RANK_DEFICIENT, CUDA_OLAP_PINNED_CHUNK_BYTES, CUDA_SKILL_MAX_DEVICE_BYTES,
    CudaAlgorithmicContext, CudaAlgorithmicStats, CudaAutocorrelationSums, CudaBackend,
    CudaBinaryBatch, CudaBinaryScores, CudaByteFeatureRaw, CudaByteRaggedBatch,
    CudaCategoricalCounts, CudaCcmPredictions, CudaContext, CudaCopulaTerms,
    CudaCorrelationPrecision, CudaCrossCorrelationBatch, CudaDcorResult, CudaDependenceStats,
    CudaEnergyContext, CudaEnergyDescent, CudaEnergyStats, CudaGrangerLagBatch,
    CudaGrangerLagSummary, CudaGreenContextStream, CudaHawkesFit, CudaHistogramCounts,
    CudaHsicResult, CudaInt8Batch, CudaKsgContinuousCounts, CudaLinearCkaPairEstimates,
    CudaLogisticConfig, CudaLogisticDataset, CudaLogisticSplits, CudaLogisticSummaries,
    CudaLoomBatch, CudaLoomContext, CudaLoomStats, CudaLoomVectorKind, CudaLoomVectorRequest,
    CudaMicPair, CudaMicScan, CudaMixedKsgCounts, CudaMmdChangePointResult, CudaMmdResult,
    CudaMxFpBatch, CudaOlapAggregate, CudaOlapContext, CudaOlapGroup, CudaOlapScan, CudaOlapStats,
    CudaPeriodogramBatch, CudaQuantContext, CudaQuantScores, CudaQuantStats, CudaRankPair,
    CudaSkillContext, CudaSkillEdge, CudaSkillMst, CudaSkillSlot, CudaSkillStats,
    CudaTurboQuantBatch, CudaTurboQuantRow, CudaTurboQuantScores, ENERGY_CUDA_MIN_ELEMENTS,
    GemmProblem, GroupedGemmExecutionMode, GroupedGemmPlan, INT8_CUDA_MIN_ELEMENTS,
    MXFP4_CUDA_MIN_ELEMENTS, MXFP8_CUDA_MIN_ELEMENTS, OLAP_CUDA_MIN_ROWS, ProfilePairwiseCudaStats,
    QuantDispatch, RaggedBatch, SKILL_CUDA_MIN_POINTS, TRANSPOSE_CUDA_MIN_ELEMENTS,
    TURBOQUANT_CUDA_MIN_ELEMENTS, autocorrelation_sums_host, binary_dispatch,
    build_grouped_gemm_plan, build_ragged_batch, build_ragged_batch_from_slabs,
    categorical_counts_host, ccm_simplex_predictions_host, copula_terms_host,
    correlation_precision_host, cross_correlation_batch_host, dcor_1d_host, entropy_radii_host,
    execute_grouped_gemm, execute_grouped_gemm_strict, extract_ragged_results, gaussian_mmd_host,
    granger_lag_summaries_host, hawkes_em_host, histogram_counts_host, hsic_1d_host, init_cuda,
    int8_dispatch, ksg_continuous_counts_host, linear_cka_pair_estimates_host,
    logistic_summaries_host, mic_pair_host, mixed_ksg_counts_host, mmd_change_point_host,
    mxfp4_dispatch, mxfp8_dispatch, olap_sum_tolerance, pairwise_euclidean_gram_tiled_host,
    periodogram_batch_host, query_device_info, rank_pair_host, read_grouped_gemm_output,
    try_extract_ragged_results, turboquant_dispatch,
};
pub use error::ForgeError;
pub use mxfp4::{
    MXFP4_BLOCK_SIZE, MXFP4_PACKED_BYTES, MxFp4Block, decode_mxfp4, decode_mxfp4_block, e8m0_scale,
    encode_mxfp4, encode_mxfp4_block,
};
pub use mxfp8::{
    MXFP8_BLOCK_BYTES, MXFP8_BLOCK_SIZE, MxFp8Block, decode_mxfp8, decode_mxfp8_block,
    encode_mxfp8, encode_mxfp8_block,
};
pub use quant::{
    AssayQuantSafety, BinaryCodec, CURRENT_SEED_VERSION, MxFp4Codec, PreparedQuant, QjlResidual,
    QuantLevel, QuantizedVec, Quantizer, RotationSeed, ScalarInt8Codec, SeedId, TurboQuantCodec,
    apply_inverse_rotation, apply_rotation, apply_rotation_batch, binary_prefilter,
    dot_estimate_unbiased, dot_qjl_correction, encode_qjl_residual, hamming_dot_estimate, new_seed,
    seed_id_hex,
};
#[cfg(feature = "cuda")]
pub use vram::CudaStream;
#[cfg(feature = "cuda")]
pub use vram::CudaVramProbe;
#[cfg(feature = "cuda")]
pub use vram::RawCudaMalloc;
pub use vram::{
    ANNEAL_VRAM_BUDGET_ENV, AdmissionController, AdmissionOutput, AdmitDecision, BlockDeallocator,
    BlockId, BlockKind, Category, CudaAllocError, CudaMalloc, DEFAULT_ANNEAL_THROTTLE_SLEEP,
    DEFAULT_ANNEAL_VRAM_CAP_BYTES, DEFAULT_OOM_MAX_RETRIES, DEFAULT_POWER_BACKOFF_THRESHOLD_W,
    DEFAULT_SOFT_CAP_BYTES, DevicePtr, GpuBlockRegistry, GpuBlockStats,
    LENS_VRAM_BUDGET_REMEDIATION, LensAdmission, LensAdmissionPlacement, LensAdmissionRequest,
    NvmlPowerProbe, OomGuard, OomGuardStats, PowerProbe, QueuedDispatch, RESERVED_HEADROOM_BYTES,
    VRAM_BUDGET_ENV, VRAM_BUDGET_REMEDIATION, VramBudgeter, VramGuard, VramProbe, VramStats,
    YieldPolicy, YieldStats, admit_lens,
};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_metadata_is_present() {
        assert_eq!(env!("CARGO_PKG_NAME"), "calyx-forge");
    }
}
