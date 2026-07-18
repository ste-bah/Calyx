#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kCentroidDims = 32;
constexpr int kCentroidLanes = 8;
constexpr int kCentroidRows = 256;

constexpr uint32_t kMemberNonfinite = 1u << 0;
constexpr uint32_t kMemberZeroNorm = 1u << 1;
constexpr uint32_t kQueryNonfinite = 1u << 2;
constexpr uint32_t kQueryZeroNorm = 1u << 3;
constexpr uint32_t kCentroidNonfinite = 1u << 4;
constexpr uint32_t kCentroidZeroNorm = 1u << 5;
constexpr uint32_t kStateNonfinite = 1u << 6;

__device__ __forceinline__ bool energy_active(const uint32_t *control, const uint32_t *status) {
    return control[0] != 0u && status[0] == 0u;
}

__device__ __forceinline__ float block_sum(float value, float *scratch) {
    const int lane = static_cast<int>(threadIdx.x);
    scratch[lane] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

}  // namespace

extern "C" __global__ __launch_bounds__(kThreads) void energy_member_inv_norms_f32(
    const float *members,
    int member_count,
    int dim,
    int require_nonzero,
    float *inverse_norms,
    uint32_t *status) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= member_count) {
        return;
    }
    __shared__ float sums[kThreads];
    __shared__ uint32_t invalid;
    if (threadIdx.x == 0) {
        invalid = 0u;
    }
    __syncthreads();

    float sum = 0.0f;
    const float *member = members + static_cast<size_t>(row) * static_cast<size_t>(dim);
    for (int col = static_cast<int>(threadIdx.x); col < dim; col += blockDim.x) {
        const float value = member[col];
        if (!isfinite(value)) {
            atomicOr(&invalid, 1u);
        } else {
            sum += value * value;
        }
    }
    const float norm_sq = block_sum(sum, sums);
    if (threadIdx.x == 0) {
        if (invalid != 0u || !isfinite(norm_sq)) {
            atomicOr(status, kMemberNonfinite);
            inverse_norms[row] = 0.0f;
        } else if (require_nonzero != 0 && !(norm_sq > 0.0f)) {
            atomicOr(status, kMemberZeroNorm);
            inverse_norms[row] = 0.0f;
        } else {
            inverse_norms[row] = norm_sq > 0.0f ? rsqrtf(norm_sq) : 0.0f;
        }
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void energy_query_inv_norm_f32(
    const float *query,
    int dim,
    float *inverse_norm,
    const uint32_t *control,
    uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    __shared__ float sums[kThreads];
    __shared__ uint32_t invalid;
    if (threadIdx.x == 0) {
        invalid = 0u;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int col = static_cast<int>(threadIdx.x); col < dim; col += blockDim.x) {
        const float value = query[col];
        if (!isfinite(value)) {
            atomicOr(&invalid, 1u);
        } else {
            sum += value * value;
        }
    }
    const float norm_sq = block_sum(sum, sums);
    if (threadIdx.x == 0) {
        if (invalid != 0u || !isfinite(norm_sq)) {
            atomicOr(status, kQueryNonfinite);
        } else if (!(norm_sq > 0.0f)) {
            atomicOr(status, kQueryZeroNorm);
        } else {
            inverse_norm[0] = rsqrtf(norm_sq);
        }
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void energy_cosine_scaled_f32(
    const float *query,
    const float *members,
    const float *query_inverse_norm,
    const float *member_inverse_norms,
    int member_count,
    int dim,
    float beta,
    float *scores,
    const uint32_t *control,
    uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (row >= member_count) {
        return;
    }
    const float *member = members + static_cast<size_t>(row) * static_cast<size_t>(dim);
    float dot = 0.0f;
    for (int col = 0; col < dim; ++col) {
        dot += query[col] * member[col];
    }
    const float score = beta * dot * query_inverse_norm[0] * member_inverse_norms[row];
    if (!isfinite(score)) {
        atomicOr(status, kStateNonfinite);
    } else {
        scores[row] = score;
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void energy_softmax_state_f32(
    const float *scores,
    int member_count,
    float beta,
    int step,
    float eps,
    float *weights,
    float *metadata,
    uint32_t *control,
    uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    __shared__ float scratch[kThreads];
    float local_max = -INFINITY;
    for (int row = static_cast<int>(threadIdx.x); row < member_count; row += blockDim.x) {
        local_max = fmaxf(local_max, scores[row]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = scratch[0];
    float local_sum = 0.0f;
    for (int row = static_cast<int>(threadIdx.x); row < member_count; row += blockDim.x) {
        const float weight = expf(scores[row] - maximum);
        weights[row] = weight;
        local_sum += weight;
    }
    const float sum = block_sum(local_sum, scratch);
    if (!(sum > 0.0f) || !isfinite(sum) || !isfinite(maximum)) {
        if (threadIdx.x == 0) {
            atomicOr(status, kStateNonfinite);
        }
        return;
    }
    for (int row = static_cast<int>(threadIdx.x); row < member_count; row += blockDim.x) {
        weights[row] /= sum;
    }
    if (threadIdx.x == 0) {
        const float next = beta == 0.0f ? -logf(static_cast<float>(member_count))
                                        : -(maximum + logf(sum));
        if (!isfinite(next)) {
            atomicOr(status, kStateNonfinite);
            return;
        }
        if (step == 0) {
            metadata[0] = next;
            return;
        }
        const float previous = metadata[0];
        metadata[0] = next;
        control[1] = static_cast<uint32_t>(step);
        if (member_count == 1 || fabsf(next - previous) < eps) {
            control[0] = 0u;
            control[2] = 1u;
        }
    }
}

extern "C" __global__ __launch_bounds__(kCentroidDims * kCentroidLanes)
void energy_centroid_partials_f32(
    const float *members,
    const float *weights,
    int member_count,
    int dim,
    float *partials,
    const uint32_t *control,
    const uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    const int col = static_cast<int>(blockIdx.x) * kCentroidDims + static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);
    const int row_begin = static_cast<int>(blockIdx.y) * kCentroidRows;
    const int row_end = min(row_begin + kCentroidRows, member_count);
    __shared__ float sums[kCentroidLanes][kCentroidDims];
    float sum = 0.0f;
    if (col < dim) {
        for (int row = row_begin + lane; row < row_end; row += kCentroidLanes) {
            sum += weights[row] * members[static_cast<size_t>(row) * dim + col];
        }
    }
    sums[lane][threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kCentroidLanes / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane][threadIdx.x] += sums[lane + stride][threadIdx.x];
        }
        __syncthreads();
    }
    if (lane == 0 && col < dim) {
        partials[static_cast<size_t>(blockIdx.y) * dim + col] = sums[0][threadIdx.x];
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void energy_centroid_finalize_f32(
    const float *partials,
    int tile_count,
    int dim,
    float *query,
    const uint32_t *control,
    uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    const int col = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (col >= dim) {
        return;
    }
    float sum = 0.0f;
    for (int tile = 0; tile < tile_count; ++tile) {
        sum += partials[static_cast<size_t>(tile) * dim + col];
    }
    if (!isfinite(sum)) {
        atomicOr(status, kCentroidNonfinite);
    } else {
        query[col] = sum;
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void energy_normalize_query_f32(
    float *query,
    int dim,
    const uint32_t *control,
    uint32_t *status) {
    if (!energy_active(control, status)) {
        return;
    }
    __shared__ float sums[kThreads];
    __shared__ uint32_t invalid;
    if (threadIdx.x == 0) {
        invalid = 0u;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int col = static_cast<int>(threadIdx.x); col < dim; col += blockDim.x) {
        const float value = query[col];
        if (!isfinite(value)) {
            atomicOr(&invalid, 1u);
        } else {
            sum += value * value;
        }
    }
    const float norm_sq = block_sum(sum, sums);
    if (invalid != 0u || !isfinite(norm_sq)) {
        if (threadIdx.x == 0) {
            atomicOr(status, kCentroidNonfinite);
        }
        return;
    }
    if (!(norm_sq > 0.0f)) {
        if (threadIdx.x == 0) {
            atomicOr(status, kCentroidZeroNorm);
        }
        return;
    }
    const float inverse = rsqrtf(norm_sq);
    for (int col = static_cast<int>(threadIdx.x); col < dim; col += blockDim.x) {
        query[col] *= inverse;
    }
}
