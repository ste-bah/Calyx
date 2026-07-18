#include <math.h>
#include <stdint.h>

#define TQ_BLOCK 256
#define TQ_QJL_TAG_V2 0x02u

__device__ __forceinline__ int tq_lane_width(int level, int index) {
    const bool high = (index & 1) == 0;
    return level == 35 ? (high ? 3 : 2) : (high ? 2 : 1);
}

__device__ __forceinline__ int tq_centroid_offset(int width) {
    return width == 1 ? 0 : (width == 2 ? 2 : 6);
}

__device__ __forceinline__ int tq_threshold_offset(int width) {
    return width == 1 ? 0 : (width == 2 ? 1 : 4);
}

__device__ __forceinline__ unsigned char tq_quantize_code(
    float value,
    int width,
    const float *thresholds) {
    const int count = (1 << width) - 1;
    const int offset = tq_threshold_offset(width);
    int code = 0;
    while (code < count && value > thresholds[offset + code]) {
        ++code;
    }
    return static_cast<unsigned char>(code);
}

extern "C" __global__ __launch_bounds__(TQ_BLOCK) void tq_rotate_fwht_f32(
    const float *input,
    const float *diagonal,
    int diagonal_stride,
    int input_dim,
    int rot_width,
    int rows,
    float *output,
    int *bad_flags) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || rot_width <= 0) {
        return;
    }
    if (tid == 0) {
        bad_flags[row] = 0;
    }
    __syncthreads();

    for (int index = tid; index < rot_width; index += blockDim.x) {
        float value = 0.0f;
        if (index < input_dim) {
            value = input[row * input_dim + index];
            if (!isfinite(value)) {
                atomicOr(&bad_flags[row], 1);
                value = 0.0f;
            }
        }
        values[index] = value * diagonal[row * diagonal_stride + index];
    }
    __syncthreads();

    for (int width = 1; width < rot_width; width <<= 1) {
        const int butterflies = rot_width >> 1;
        for (int pair = tid; pair < butterflies; pair += blockDim.x) {
            const int base = (pair / width) * (width << 1) + (pair % width);
            const float left = values[base];
            const float right = values[base + width];
            values[base] = left + right;
            values[base + width] = left - right;
        }
        __syncthreads();
    }

    const float factor = 1.0f / sqrtf(static_cast<float>(rot_width));
    for (int index = tid; index < rot_width; index += blockDim.x) {
        output[row * rot_width + index] = values[index] * factor;
    }
}

extern "C" __global__ __launch_bounds__(TQ_BLOCK) void tq_quantize_rows_f32(
    const float *rotated,
    const float *thresholds,
    const float *centroids,
    int rot_width,
    int rows,
    int level,
    float *scales,
    unsigned char *codes,
    float *decoded,
    int *bad_flags) {
    __shared__ float row_scale;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || rot_width <= 0) {
        return;
    }
    if (tid == 0) {
        double norm_sq = 0.0;
        const int base = row * rot_width;
        for (int index = 0; index < rot_width; ++index) {
            const double value = static_cast<double>(rotated[base + index]);
            norm_sq += value * value;
        }
        row_scale = static_cast<float>(sqrt(norm_sq));
        scales[row] = row_scale;
        if (!isfinite(row_scale)) {
            bad_flags[row] |= 2;
        }
    }
    __syncthreads();

    const float root = sqrtf(static_cast<float>(rot_width));
    const float unit = row_scale == 0.0f ? 0.0f : root / row_scale;
    const float decode_scale = row_scale == 0.0f ? 0.0f : row_scale / root;
    const int base = row * rot_width;
    for (int index = tid; index < rot_width; index += blockDim.x) {
        const int width = tq_lane_width(level, index);
        const unsigned char code = row_scale == 0.0f
            ? 0
            : tq_quantize_code(rotated[base + index] * unit, width, thresholds);
        codes[base + index] = code;
        decoded[base + index] = row_scale == 0.0f
            ? 0.0f
            : centroids[tq_centroid_offset(width) + code] * decode_scale;
    }
}

extern "C" __global__ void tq_pack_scalar_v4(
    const unsigned char *codes,
    int rot_width,
    int rows,
    int level,
    int encoded_stride,
    unsigned char *encoded) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    unsigned char *out = encoded + row * encoded_stride;
    int bit_offset = 0;
    const int base = row * rot_width;
    for (int index = 0; index < rot_width; ++index) {
        const int width = tq_lane_width(level, index);
        const unsigned int code = codes[base + index];
        for (int bit = 0; bit < width; ++bit) {
            if (((code >> bit) & 1u) != 0u) {
                const int absolute = bit_offset + bit;
                out[absolute >> 3] |= static_cast<unsigned char>(1u << (absolute & 7));
            }
        }
        bit_offset += width;
    }
}

extern "C" __global__ __launch_bounds__(TQ_BLOCK) void tq_residual_rows_f32(
    const float *rotated,
    const float *decoded,
    int rot_width,
    int rows,
    float *residual,
    float *residual_norms,
    int *bad_flags) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int base = row * rot_width;
    for (int index = tid; index < rot_width; index += blockDim.x) {
        residual[base + index] = rotated[base + index] - decoded[base + index];
    }
    __syncthreads();
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (int index = 0; index < rot_width; ++index) {
            const float value = residual[base + index];
            norm_sq += value * value;
        }
        const float norm = sqrtf(norm_sq);
        residual_norms[row] = norm;
        if (!isfinite(norm)) {
            bad_flags[row] |= 4;
        }
    }
}

extern "C" __global__ void tq_pack_qjl_v2(
    const float *qjl_rotated,
    const float *residual_norms,
    const unsigned char *rademacher_seed,
    int seed_stride,
    int rot_width,
    int rows,
    int scalar_len,
    int encoded_stride,
    unsigned char *qjl_bits,
    unsigned char *encoded) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int bits_len = (rot_width + 7) >> 3;
    unsigned char *row_bits = qjl_bits + row * bits_len;
    const int base = row * rot_width;
    for (int index = 0; index < rot_width; ++index) {
        if (qjl_rotated[base + index] > 0.0f) {
            row_bits[index >> 3] |= static_cast<unsigned char>(1u << (index & 7));
        }
    }
    unsigned char *section = encoded + row * encoded_stride + scalar_len;
    section[0] = TQ_QJL_TAG_V2;
    for (int index = 0; index < 32; ++index) {
        section[1 + index] = rademacher_seed[row * seed_stride + index];
    }
    union {
        float value;
        unsigned char bytes[4];
    } norm;
    norm.value = residual_norms[row];
    for (int index = 0; index < 4; ++index) {
        section[33 + index] = norm.bytes[index];
    }
    for (int index = 0; index < bits_len; ++index) {
        section[37 + index] = row_bits[index];
    }
}

extern "C" __global__ __launch_bounds__(TQ_BLOCK) void tq_decode_inverse_fwht_f32(
    const unsigned char *codes,
    const float *scales,
    const float *centroids,
    const float *diagonal,
    int dim,
    int rot_width,
    int rows,
    int level,
    float *output) {
    extern __shared__ float values[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || rot_width <= 0) {
        return;
    }
    const float scale = scales[row];
    const float unit = scale == 0.0f ? 0.0f : scale / sqrtf(static_cast<float>(rot_width));
    const int base = row * rot_width;
    for (int index = tid; index < rot_width; index += blockDim.x) {
        const int width = tq_lane_width(level, index);
        const unsigned char code = codes[base + index];
        values[index] = scale == 0.0f
            ? 0.0f
            : centroids[tq_centroid_offset(width) + code] * unit;
    }
    __syncthreads();
    for (int width = 1; width < rot_width; width <<= 1) {
        const int butterflies = rot_width >> 1;
        for (int pair = tid; pair < butterflies; pair += blockDim.x) {
            const int offset = (pair / width) * (width << 1) + (pair % width);
            const float left = values[offset];
            const float right = values[offset + width];
            values[offset] = left + right;
            values[offset + width] = left - right;
        }
        __syncthreads();
    }
    const float factor = 1.0f / sqrtf(static_cast<float>(rot_width));
    for (int index = tid; index < dim; index += blockDim.x) {
        output[row * dim + index] = values[index] * factor * diagonal[index];
    }
}

extern "C" __global__ void tq_score_prepared_v4(
    const unsigned char *query_codes,
    const unsigned char *query_signs,
    const float *query_scale,
    const float *query_residual_norm,
    const unsigned char *candidate_codes,
    const unsigned char *candidate_signs,
    const float *candidate_scales,
    const float *candidate_residual_norms,
    const float *centroids,
    int rot_width,
    int candidates,
    int level,
    float *scores) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidates || rot_width <= 0) {
        return;
    }
    const int code_base = candidate * rot_width;
    const int bits_len = (rot_width + 7) >> 3;
    const int sign_base = candidate * bits_len;
    bool equal_codes = true;
    bool complement_codes = true;
    float centroid_dot = 0.0f;
    for (int index = 0; index < rot_width; ++index) {
        const int width = tq_lane_width(level, index);
        const unsigned char left = query_codes[index];
        const unsigned char right = candidate_codes[code_base + index];
        equal_codes &= left == right;
        complement_codes &= static_cast<int>(left) + static_cast<int>(right) == ((1 << width) - 1);
        centroid_dot += centroids[tq_centroid_offset(width) + left]
            * centroids[tq_centroid_offset(width) + right];
    }

    bool equal_signs = true;
    bool complement_signs = true;
    unsigned int mismatches = 0;
    for (int index = 0; index < bits_len; ++index) {
        unsigned int mask = 0xffu;
        if (index + 1 == bits_len && (rot_width & 7) != 0) {
            mask = (1u << (rot_width & 7)) - 1u;
        }
        const unsigned int bits =
            (static_cast<unsigned int>(query_signs[index])
                ^ static_cast<unsigned int>(candidate_signs[sign_base + index]))
            & mask;
        equal_signs &= bits == 0u;
        complement_signs &= bits == mask;
        mismatches += __popc(bits);
    }

    const float left_scale = query_scale[0];
    const float right_scale = candidate_scales[candidate];
    const float left_norm = query_residual_norm[0];
    const float right_norm = candidate_residual_norms[candidate];
    const bool matching_endpoint = __float_as_uint(left_scale) == __float_as_uint(right_scale)
        && __float_as_uint(left_norm) == __float_as_uint(right_norm);
    if (matching_endpoint && equal_codes && equal_signs) {
        scores[candidate] = left_scale * right_scale;
        return;
    }
    if (matching_endpoint && complement_codes && complement_signs) {
        scores[candidate] = -(left_scale * right_scale);
        return;
    }

    float scalar = 0.0f;
    if (left_scale != 0.0f && right_scale != 0.0f) {
        scalar = left_scale * right_scale * centroid_dot / static_cast<float>(rot_width);
    }
    const float mean =
        (static_cast<float>(rot_width) - 2.0f * static_cast<float>(mismatches))
        / static_cast<float>(rot_width);
    const float correction = left_norm * right_norm * sinf(1.57079632679489661923f * mean);
    scores[candidate] = scalar + correction;
}
