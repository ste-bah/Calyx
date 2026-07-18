pub mod algorithmic;
pub mod assay;
pub mod context;
pub mod distance;
#[cfg(test)]
mod distance_tests;
pub mod energy;
pub mod gemm;
pub mod green_context;
pub mod grouped_gemm;
#[cfg(test)]
mod grouped_gemm_tests;
pub mod kernels;
pub mod loom;
pub mod olap;
pub mod postprocess;
#[cfg(test)]
mod postprocess_tests;
pub mod profile;
pub mod quant;
pub mod ragged_gemm;
pub mod resident;
#[cfg(test)]
mod resident_tests;
pub mod skill;
pub mod topk;
#[cfg(test)]
mod topk_tests;
mod validate;

use crate::{Backend, DeviceInfo, Result};

pub use crate::mxfp4;
pub use algorithmic::{
    ALGORITHMIC_SPARSE_MAX_TOKEN_BYTES, ALGORITHMIC_TOKEN_HASH_MAX_TOKEN_BYTES,
    CudaAlgorithmicContext, CudaAlgorithmicStats, CudaByteFeatureRaw, CudaByteRaggedBatch,
};
pub use assay::{
    CUDA_GRANGER_STATUS_INVALID_LAG, CUDA_GRANGER_STATUS_NONFINITE, CUDA_GRANGER_STATUS_OK,
    CUDA_GRANGER_STATUS_RANK_DEFICIENT, CudaAutocorrelationSums, CudaCategoricalCounts,
    CudaCcmPredictions, CudaCopulaTerms, CudaCorrelationPrecision, CudaCrossCorrelationBatch,
    CudaDcorResult, CudaDependenceStats, CudaGrangerLagBatch, CudaGrangerLagSummary, CudaHawkesFit,
    CudaHistogramCounts, CudaHsicResult, CudaKsgContinuousCounts, CudaLinearCkaPairEstimates,
    CudaLogisticConfig, CudaLogisticDataset, CudaLogisticSplits, CudaLogisticSummaries,
    CudaMicPair, CudaMicScan, CudaMixedKsgCounts, CudaMmdChangePointResult, CudaMmdResult,
    CudaPeriodogramBatch, CudaRankPair, autocorrelation_sums_host, categorical_counts_host,
    ccm_simplex_predictions_host, copula_terms_host, correlation_precision_host,
    cross_correlation_batch_host, dcor_1d_host, entropy_radii_host, gaussian_mmd_host,
    granger_lag_summaries_host, hawkes_em_host, histogram_counts_host, hsic_1d_host,
    ksg_continuous_counts_host, linear_cka_pair_estimates_host, logistic_summaries_host,
    mic_pair_host, mixed_ksg_counts_host, mmd_change_point_host, periodogram_batch_host,
    rank_pair_host,
};
pub use context::{CudaContext, init_cuda, query_device_info};
pub use distance::{cosine_batch_gpu, dot_batch_gpu, l2_batch_gpu, normalize_rows_gpu};
pub use energy::{
    CUDA_ENERGY_MAX_DEVICE_BYTES, CUDA_ENERGY_PINNED_CHUNK_BYTES, CudaEnergyContext,
    CudaEnergyDescent, CudaEnergyStats, ENERGY_CUDA_MIN_ELEMENTS,
};
pub use gemm::{
    bench_gemm_cublas, bench_gemm_reference_cublas, gemm_cublas, gemm_mxfp4_fp32_accum,
    gemm_mxfp8_fp32_accum, probe_allocation,
};
pub use green_context::CudaGreenContextStream;
pub use grouped_gemm::{
    AbsentSlotSentinel, GemmProblem, GroupedGemmExecutionMode, GroupedGemmPlan,
    build_grouped_gemm_plan, execute_grouped_gemm, execute_grouped_gemm_strict,
    read_grouped_gemm_output,
};
pub use loom::{
    CudaLoomBatch, CudaLoomContext, CudaLoomStats, CudaLoomVectorKind, CudaLoomVectorRequest,
};
pub use olap::{
    CUDA_OLAP_PINNED_CHUNK_BYTES, CudaOlapAggregate, CudaOlapContext, CudaOlapGroup, CudaOlapScan,
    CudaOlapStats, OLAP_CUDA_MIN_ROWS, TRANSPOSE_CUDA_MIN_ELEMENTS, olap_sum_tolerance,
};
pub use postprocess::{
    CudaDenseTokenPostprocess, CudaMultiRows, CudaPostprocessPooling, CudaSparseRows,
    bgem3_colbert_tokens_from_external_f32, bgem3_sparse_from_external_f32,
    colbert_tokens_from_external_f32, dense_2d_from_external_f32, dense_tokens_from_external_f32,
    sparse_positive_from_external_f32,
};
pub use profile::{ProfilePairwiseCudaStats, pairwise_euclidean_gram_tiled_host};
pub use quant::{
    BINARY_CUDA_MIN_ELEMENTS, CudaBinaryBatch, CudaBinaryScores, CudaInt8Batch, CudaMxFpBatch,
    CudaQuantContext, CudaQuantScores, CudaQuantStats, CudaTurboQuantBatch, CudaTurboQuantRow,
    CudaTurboQuantScores, INT8_CUDA_MIN_ELEMENTS, MXFP4_CUDA_MIN_ELEMENTS, MXFP8_CUDA_MIN_ELEMENTS,
    QuantDispatch, TURBOQUANT_CUDA_MIN_ELEMENTS, binary_dispatch, int8_dispatch, mxfp4_dispatch,
    mxfp8_dispatch, turboquant_dispatch,
};
pub use ragged_gemm::{
    RaggedBatch, build_ragged_batch, build_ragged_batch_from_slabs, extract_ragged_results,
    try_extract_ragged_results,
};
pub use resident::{DeviceCandidateBlock, cosine_resident_host, upload_candidate_block};
pub use skill::{
    CUDA_SKILL_MAX_DEVICE_BYTES, CudaSkillContext, CudaSkillEdge, CudaSkillMst, CudaSkillSlot,
    CudaSkillStats, SKILL_CUDA_MIN_POINTS,
};
pub use topk::topk_gpu;

#[derive(Clone, Debug)]
pub struct CudaBackend {
    ctx: CudaContext,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        init_cuda(0, false).map(|ctx| Self { ctx })
    }

    pub fn with_context(ctx: CudaContext) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    pub fn grouped_gemm(&self, plan: &mut GroupedGemmPlan) -> Result<()> {
        grouped_gemm::execute_grouped_gemm(&self.ctx, plan)
    }

    pub fn grouped_gemm_strict(&self, plan: &mut GroupedGemmPlan) -> Result<()> {
        grouped_gemm::execute_grouped_gemm_strict(&self.ctx, plan)
    }
}

impl Backend for CudaBackend {
    fn gemm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) -> Result<()> {
        gemm::gemm_host(&self.ctx, a, b, m, k, n, out)
    }

    fn cosine(&self, a: &[f32], b: &[f32], dim: usize, out: &mut [f32]) -> Result<()> {
        distance::cosine_host(&self.ctx, a, b, dim, out)
    }

    fn dot(&self, a: &[f32], b: &[f32], dim: usize, out: &mut [f32]) -> Result<()> {
        distance::dot_host(&self.ctx, a, b, dim, out)
    }

    fn l2(&self, a: &[f32], b: &[f32], dim: usize, out: &mut [f32]) -> Result<()> {
        distance::l2_host(&self.ctx, a, b, dim, out)
    }

    fn normalize(&self, vecs: &mut [f32], dim: usize) -> Result<()> {
        distance::normalize_host(&self.ctx, vecs, dim)
    }

    fn topk(&self, scores: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        topk::topk_host(&self.ctx, scores, k)
    }

    fn device_info(&self) -> DeviceInfo {
        query_device_info(&self.ctx)
    }
}

#[cfg(test)]
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_TEST_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}
