#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define LOOM_THREADS 256
#define LOOM_FLAG_NONFINITE 1u
#define LOOM_FLAG_ZERO_NORM 2u
#define LOOM_FLAG_INVALID 4u

extern "C" __global__ __launch_bounds__(LOOM_THREADS) void loom_normalize_rows_f32(
    const float *input,
    int row_count,
    int dim,
    float *normalized,
    unsigned int *flags) {
    __shared__ float sums[LOOM_THREADS];
    __shared__ unsigned int bad[LOOM_THREADS];
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    float sum = 0.0f;
    unsigned int local_bad =
        row >= row_count || row_count <= 0 || dim <= 0 ? LOOM_FLAG_INVALID : 0u;

    if (local_bad == 0u) {
        const unsigned long long base = (unsigned long long)row * (unsigned long long)dim;
        for (int col = tid; col < dim; col += blockDim.x) {
            const float value = input[base + col];
            if (!isfinite(value)) {
                local_bad |= LOOM_FLAG_NONFINITE;
            }
            normalized[base + col] = value;
            sum += value * value;
        }
    }
    sums[tid] = sum;
    bad[tid] = local_bad;
    __syncthreads();
    for (int stride = LOOM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            bad[tid] |= bad[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && !(isfinite(sums[0]))) {
        bad[0] |= LOOM_FLAG_NONFINITE;
    }
    __syncthreads();
    if (bad[0] != 0u) {
        if (tid == 0) {
            atomicOr(flags, bad[0]);
        }
        return;
    }
    if (sums[0] <= FLT_EPSILON) {
        return;
    }
    const float inverse_norm = 1.0f / sqrtf(sums[0]);
    const unsigned long long base = (unsigned long long)row * (unsigned long long)dim;
    for (int col = tid; col < dim; col += blockDim.x) {
        normalized[base + col] *= inverse_norm;
    }
}

extern "C" __global__ __launch_bounds__(LOOM_THREADS) void loom_extract_pairs_f32(
    const float *gram,
    const unsigned int *left_rows,
    const unsigned int *right_rows,
    int pair_count,
    int row_count,
    float *agreements,
    unsigned int *flags) {
    const int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) {
        return;
    }
    const unsigned int left = left_rows[pair];
    const unsigned int right = right_rows[pair];
    if (pair_count <= 0 || row_count <= 0 || left >= (unsigned int)row_count ||
        right >= (unsigned int)row_count || left > right) {
        atomicOr(flags, LOOM_FLAG_INVALID);
        return;
    }
    // cuBLAS writes column-major; the Gram matrix is symmetric, so this is also
    // the canonical row-major (left, right) score.
    const float score = gram[(unsigned long long)left +
                             (unsigned long long)right * (unsigned long long)row_count];
    if (!isfinite(score)) {
        atomicOr(flags, LOOM_FLAG_NONFINITE);
        return;
    }
    agreements[pair] = score;
}

extern "C" __global__ __launch_bounds__(LOOM_THREADS) void loom_cross_terms_f32(
    const float *matrix,
    const unsigned int *left_rows,
    const unsigned int *right_rows,
    const unsigned int *kinds,
    int request_count,
    int row_count,
    int dim,
    float *output,
    unsigned int *flags) {
    const int request = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (request >= request_count) {
        return;
    }
    const unsigned int left = left_rows[request];
    const unsigned int right = right_rows[request];
    const unsigned int kind = kinds[request];
    if (request_count <= 0 || row_count <= 0 || dim <= 0 ||
        left >= (unsigned int)row_count || right >= (unsigned int)row_count ||
        left > right || kind > 1u) {
        if (tid == 0) {
            atomicOr(flags, LOOM_FLAG_INVALID);
        }
        return;
    }
    const unsigned long long left_base = (unsigned long long)left * (unsigned long long)dim;
    const unsigned long long right_base = (unsigned long long)right * (unsigned long long)dim;
    const unsigned long long out_base =
        (unsigned long long)request * (unsigned long long)dim;
    for (int col = tid; col < dim; col += blockDim.x) {
        const float a = matrix[left_base + col];
        const float b = matrix[right_base + col];
        const float value = kind == 0u ? a - b : a * b;
        if (!(isfinite(a) && isfinite(b) && isfinite(value))) {
            atomicOr(flags, LOOM_FLAG_NONFINITE);
            continue;
        }
        output[out_base + col] = value;
    }
}
