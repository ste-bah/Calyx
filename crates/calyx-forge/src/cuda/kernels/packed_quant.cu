#include <math.h>
#include <stdint.h>

#define PQ_BLOCK 256

__device__ void pq_block_hadamard(float *values, int dim) {
    const int tid = threadIdx.x;
    int offset = 0;
    while (offset < dim) {
        const int remaining = dim - offset;
        int block_len = 1;
        while ((block_len << 1) <= remaining) {
            block_len <<= 1;
        }
        for (int width = 1; width < block_len; width <<= 1) {
            const int butterflies = block_len >> 1;
            for (int pair = tid; pair < butterflies; pair += blockDim.x) {
                const int base = offset + (pair / width) * (width << 1) + (pair % width);
                const float left = values[base];
                const float right = values[base + width];
                values[base] = left + right;
                values[base + width] = left - right;
            }
            __syncthreads();
        }
        const float factor = 1.0f / sqrtf(static_cast<float>(block_len));
        for (int index = tid; index < block_len; index += blockDim.x) {
            values[offset + index] *= factor;
        }
        __syncthreads();
        offset += block_len;
    }
}

extern "C" __global__ __launch_bounds__(PQ_BLOCK) void pq_binary_encode_f32(
    const float *input,
    const float *diagonal,
    int dim,
    int rows,
    unsigned char *encoded,
    int *bad_flags) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    if (tid == 0) {
        bad_flags[row] = 0;
    }
    __syncthreads();
    const int base = row * dim;
    for (int index = tid; index < dim; index += blockDim.x) {
        float value = input[base + index];
        if (!isfinite(value)) {
            atomicOr(&bad_flags[row], 1);
            value = 0.0f;
        }
        values[index] = value * diagonal[index];
    }
    __syncthreads();
    pq_block_hadamard(values, dim);

    const int stride = (dim + 7) >> 3;
    unsigned char *row_out = encoded + row * stride;
    for (int byte = tid; byte < stride; byte += blockDim.x) {
        unsigned int packed = 0;
        const int start = byte << 3;
        for (int bit = 0; bit < 8 && start + bit < dim; ++bit) {
            if (values[start + bit] > 0.0f) {
                packed |= 1u << bit;
            }
        }
        row_out[byte] = static_cast<unsigned char>(packed);
    }
}

extern "C" __global__ __launch_bounds__(PQ_BLOCK) void pq_binary_decode_f32(
    const unsigned char *encoded,
    const float *diagonal,
    int dim,
    int rows,
    float amplitude,
    float *output) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    const int stride = (dim + 7) >> 3;
    const unsigned char *row_in = encoded + row * stride;
    for (int index = tid; index < dim; index += blockDim.x) {
        const bool positive = ((row_in[index >> 3] >> (index & 7)) & 1u) != 0u;
        values[index] = positive ? amplitude : -amplitude;
    }
    __syncthreads();
    pq_block_hadamard(values, dim);
    for (int index = tid; index < dim; index += blockDim.x) {
        output[row * dim + index] = values[index] * diagonal[index];
    }
}

extern "C" __global__ void pq_binary_score(
    const unsigned char *query,
    const unsigned char *candidates,
    int dim,
    int rows,
    int *mismatch_counts,
    float *scores) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    const int stride = (dim + 7) >> 3;
    const int full_bytes = dim >> 3;
    const unsigned char *candidate = candidates + row * stride;
    int mismatches = 0;
    for (int index = 0; index < full_bytes; ++index) {
        mismatches += __popc(static_cast<unsigned int>(query[index] ^ candidate[index]));
    }
    const int tail = dim & 7;
    if (tail != 0) {
        const unsigned int mask = (1u << tail) - 1u;
        mismatches += __popc(static_cast<unsigned int>(query[full_bytes] ^ candidate[full_bytes]) & mask);
    }
    mismatch_counts[row] = mismatches;
    scores[row] = 1.0f - 2.0f * static_cast<float>(mismatches) / static_cast<float>(dim);
}

extern "C" __global__ __launch_bounds__(PQ_BLOCK) void pq_int8_encode_f32(
    const float *input,
    int dim,
    int rows,
    unsigned char *encoded,
    float *scales,
    int *bad_flags) {
    __shared__ float row_scale;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    const int base = row * dim;
    if (tid == 0) {
        bad_flags[row] = 0;
        float max_abs = 0.0f;
        for (int index = 0; index < dim; ++index) {
            const float value = input[base + index];
            if (!isfinite(value)) {
                bad_flags[row] |= 1;
            }
            max_abs = fmaxf(max_abs, fabsf(value));
        }
        row_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
        scales[row] = row_scale;
        if (!isfinite(row_scale)) {
            bad_flags[row] |= 2;
        }
    }
    __syncthreads();
    for (int index = tid; index < dim; index += blockDim.x) {
        int code = 0;
        if (row_scale != 0.0f) {
            code = __float2int_rn(input[base + index] / row_scale);
            code = max(-127, min(127, code));
        }
        encoded[base + index] = static_cast<unsigned char>(static_cast<signed char>(code));
    }
}

extern "C" __global__ void pq_int8_decode_f32(
    const unsigned char *encoded,
    const float *scales,
    int dim,
    int rows,
    float *output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = dim * rows;
    if (index >= count) {
        return;
    }
    const int raw = static_cast<int>(encoded[index]);
    const int code = raw < 128 ? raw : raw - 256;
    output[index] = static_cast<float>(code) * scales[index / dim];
}

extern "C" __global__ void pq_int8_score(
    const unsigned char *query,
    const float *query_scale,
    const unsigned char *candidates,
    const float *candidate_scales,
    int dim,
    int rows,
    float *scores) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    int64_t code_dot = 0;
    const int base = row * dim;
    for (int index = 0; index < dim; ++index) {
        const int left_raw = static_cast<int>(query[index]);
        const int right_raw = static_cast<int>(candidates[base + index]);
        const int left = left_raw < 128 ? left_raw : left_raw - 256;
        const int right = right_raw < 128 ? right_raw : right_raw - 256;
        code_dot += static_cast<int64_t>(left) * static_cast<int64_t>(right);
    }
    scores[row] = static_cast<float>(code_dot) * query_scale[0] * candidate_scales[row];
}
