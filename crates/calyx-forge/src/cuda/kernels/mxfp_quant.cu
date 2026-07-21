#include <math.h>
#include <stdint.h>

#define MQ_BLOCK 256
#define MQ_QUANT_BLOCK 32
#define MQ_FP4_BLOCK_BYTES 17
#define MQ_FP8_BLOCK_BYTES 33

__device__ int mq_floor_exponent(float value) {
    const unsigned int bits = __float_as_uint(value) & 0x7fffffffu;
    const int stored = static_cast<int>((bits >> 23) & 0xffu);
    return stored == 0 ? -127 : stored - 127;
}

__device__ unsigned char mq_scale_byte(float abs_max) {
    if (abs_max == 0.0f) {
        return 0;
    }
    const int exponent = max(-127, min(127, mq_floor_exponent(abs_max)));
    return static_cast<unsigned char>(exponent + 127);
}

__device__ float mq_scale(unsigned char scale_e8m0) {
    return ldexpf(1.0f, static_cast<int>(scale_e8m0) - 127);
}

__device__ unsigned char mq_fp4_encode(float value, float scale) {
    if (value == 0.0f) {
        return 7;
    }
    float normalized = __fdiv_rn(value, scale);
    normalized = fmaxf(-1.0f, fminf(1.0f, normalized));
    int code = static_cast<int>(roundf(__fmul_rn(normalized, 7.0f)));
    if (code == 0) {
        code = value > 0.0f ? 1 : -1;
    }
    return static_cast<unsigned char>(code + 7);
}

__device__ float mq_fp4_decode(unsigned char code, float scale) {
    if (code == 15) {
        return 0.0f;
    }
    const int signed_code = min(static_cast<int>(code), 14) - 7;
    return __fdiv_rn(__fmul_rn(static_cast<float>(signed_code), scale), 7.0f);
}

__device__ float mq_fp8_decode_positive(unsigned char code) {
    const int exponent = static_cast<int>((code & 0x78u) >> 3);
    const int mantissa = static_cast<int>(code & 0x07u);
    if (exponent == 0) {
        return mantissa == 0 ? 0.0f : ldexpf(static_cast<float>(mantissa), -9);
    }
    return ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
}

__device__ float mq_fp8_decode(unsigned char code) {
    const float magnitude = mq_fp8_decode_positive(code & 0x7fu);
    return (code & 0x80u) == 0 ? magnitude : -magnitude;
}

__device__ unsigned char mq_fp8_encode(float value, float scale) {
    const float normalized = __fdiv_rn(value, scale);
    if (normalized == 0.0f) {
        return 0;
    }
    const unsigned char sign = normalized < 0.0f ? 0x80u : 0u;
    const float magnitude = fabsf(normalized);
    if (magnitude >= 480.0f) {
        return static_cast<unsigned char>(0x7fu | sign);
    }
    int lower;
    if (magnitude < 0.015625f) {
        lower = static_cast<int>(floorf(__fmul_rn(magnitude, 512.0f)));
        lower = max(0, min(7, lower));
    } else {
        const int exponent = mq_floor_exponent(magnitude);
        const int stored = max(1, min(15, exponent + 7));
        const float base = ldexpf(1.0f, exponent);
        const float position = __fmul_rn(__fdiv_rn(magnitude, base) - 1.0f, 8.0f);
        const int mantissa = max(0, min(7, static_cast<int>(floorf(position))));
        lower = (stored << 3) | mantissa;
    }
    const int upper = min(127, lower + 1);
    const float lower_value = mq_fp8_decode_positive(static_cast<unsigned char>(lower));
    const float upper_value = mq_fp8_decode_positive(static_cast<unsigned char>(upper));
    const int selected = magnitude - lower_value <= upper_value - magnitude ? lower : upper;
    return selected == 0 ? 0 : static_cast<unsigned char>(selected) | sign;
}

extern "C" __global__ __launch_bounds__(MQ_BLOCK) void mq_mxfp4_encode_f32(
    const float *input,
    int dim,
    int rows,
    unsigned char *encoded,
    int *bad_flags) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    if (tid == 0) {
        bad_flags[row] = 0;
    }
    __syncthreads();
    const int blocks = (dim + MQ_QUANT_BLOCK - 1) / MQ_QUANT_BLOCK;
    const int input_base = row * dim;
    const int output_base = row * blocks * MQ_FP4_BLOCK_BYTES;
    for (int block = tid; block < blocks; block += blockDim.x) {
        float values[MQ_QUANT_BLOCK];
        float abs_max = 0.0f;
        for (int index = 0; index < MQ_QUANT_BLOCK; ++index) {
            const int column = block * MQ_QUANT_BLOCK + index;
            float value = column < dim ? input[input_base + column] : 0.0f;
            if (!isfinite(value)) {
                atomicOr(&bad_flags[row], 1);
                value = 0.0f;
            }
            values[index] = value;
            abs_max = fmaxf(abs_max, fabsf(value));
        }
        const unsigned char scale_byte = mq_scale_byte(abs_max);
        const float scale = mq_scale(scale_byte);
        unsigned char *output = encoded + output_base + block * MQ_FP4_BLOCK_BYTES;
        for (int byte = 0; byte < 16; ++byte) {
            const unsigned char low = mq_fp4_encode(values[byte * 2], scale);
            const unsigned char high = mq_fp4_encode(values[byte * 2 + 1], scale);
            output[byte] = static_cast<unsigned char>(low | (high << 4));
        }
        output[16] = scale_byte;
    }
}

extern "C" __global__ __launch_bounds__(MQ_BLOCK) void mq_mxfp8_encode_f32(
    const float *input,
    int dim,
    int rows,
    unsigned char *encoded,
    int *bad_flags) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    if (tid == 0) {
        bad_flags[row] = 0;
    }
    __syncthreads();
    const int blocks = (dim + MQ_QUANT_BLOCK - 1) / MQ_QUANT_BLOCK;
    const int input_base = row * dim;
    const int output_base = row * blocks * MQ_FP8_BLOCK_BYTES;
    for (int block = tid; block < blocks; block += blockDim.x) {
        float values[MQ_QUANT_BLOCK];
        float abs_max = 0.0f;
        for (int index = 0; index < MQ_QUANT_BLOCK; ++index) {
            const int column = block * MQ_QUANT_BLOCK + index;
            float value = column < dim ? input[input_base + column] : 0.0f;
            if (!isfinite(value)) {
                atomicOr(&bad_flags[row], 1);
                value = 0.0f;
            }
            values[index] = value;
            abs_max = fmaxf(abs_max, fabsf(value));
        }
        const unsigned char scale_byte = mq_scale_byte(abs_max);
        const float scale = mq_scale(scale_byte);
        unsigned char *output = encoded + output_base + block * MQ_FP8_BLOCK_BYTES;
        for (int index = 0; index < MQ_QUANT_BLOCK; ++index) {
            output[index] = mq_fp8_encode(values[index], scale);
        }
        output[MQ_QUANT_BLOCK] = scale_byte;
    }
}

__device__ float mq_decode_value(
    const unsigned char *row,
    int level,
    int index) {
    const int block = index / MQ_QUANT_BLOCK;
    const int offset = index % MQ_QUANT_BLOCK;
    if (level == 4) {
        const unsigned char *payload = row + block * MQ_FP4_BLOCK_BYTES;
        const unsigned char packed = payload[offset / 2];
        const unsigned char code = offset % 2 == 0 ? packed & 0x0fu : packed >> 4;
        return mq_fp4_decode(code, mq_scale(payload[16]));
    }
    const unsigned char *payload = row + block * MQ_FP8_BLOCK_BYTES;
    return __fmul_rn(mq_fp8_decode(payload[offset]), mq_scale(payload[MQ_QUANT_BLOCK]));
}

extern "C" __global__ void mq_mxfp_decode_f32(
    const unsigned char *encoded,
    int dim,
    int rows,
    int level,
    float *output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = dim * rows;
    if (index >= count || dim <= 0) {
        return;
    }
    const int row = index / dim;
    const int stride = ((dim + MQ_QUANT_BLOCK - 1) / MQ_QUANT_BLOCK)
        * (level == 4 ? MQ_FP4_BLOCK_BYTES : MQ_FP8_BLOCK_BYTES);
    output[index] = mq_decode_value(encoded + row * stride, level, index % dim);
}

extern "C" __global__ void mq_mxfp_score(
    const unsigned char *query,
    int query_level,
    const unsigned char *candidates,
    int candidate_level,
    int dim,
    int rows,
    float *scores) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || dim <= 0) {
        return;
    }
    const int blocks = (dim + MQ_QUANT_BLOCK - 1) / MQ_QUANT_BLOCK;
    const int candidate_stride = blocks
        * (candidate_level == 4 ? MQ_FP4_BLOCK_BYTES : MQ_FP8_BLOCK_BYTES);
    const unsigned char *candidate = candidates + row * candidate_stride;
    float dot = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const int count = min(MQ_QUANT_BLOCK, dim - block * MQ_QUANT_BLOCK);
        float block_dot = 0.0f;
        for (int offset = 0; offset < count; ++offset) {
            const int index = block * MQ_QUANT_BLOCK + offset;
            const float left = mq_decode_value(query, query_level, index);
            const float right = mq_decode_value(candidate, candidate_level, index);
            block_dot = __fadd_rn(block_dot, __fmul_rn(left, right));
        }
        dot = __fadd_rn(dot, block_dot);
    }
    scores[row] = dot;
}
