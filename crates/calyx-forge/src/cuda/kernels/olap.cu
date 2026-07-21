#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t kValueNonfinite = 1U;
constexpr uint32_t kGroupNonfinite = 2U;
constexpr uint32_t kGroupCap = 4U;
constexpr uint32_t kDictionaryFull = 8U;
constexpr int kReduceThreads = 256;

__device__ __forceinline__ uint32_t ordered_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (bits & 0x80000000U) ? 0xffffffffU : 0x80000000U;
    return bits ^ mask;
}

__device__ __forceinline__ uint32_t hash_key(uint32_t key) {
    key ^= key >> 16;
    key *= 0x7feb352dU;
    key ^= key >> 15;
    key *= 0x846ca68bU;
    return key ^ (key >> 16);
}

}  // namespace

extern "C" __global__ void olap_reduce_f32(
    const float* values,
    uint32_t rows,
    unsigned long long* block_counts,
    double* block_sums,
    uint32_t* block_mins,
    uint32_t* block_maxs,
    uint32_t* status) {
    __shared__ unsigned long long counts[kReduceThreads];
    __shared__ double sums[kReduceThreads];
    __shared__ uint32_t mins[kReduceThreads];
    __shared__ uint32_t maxs[kReduceThreads];

    unsigned long long count = 0;
    double sum = 0.0;
    uint32_t min_value = 0xffffffffU;
    uint32_t max_value = 0U;
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += blockDim.x * gridDim.x) {
        const float value = values[row];
        if (!isfinite(value)) {
            atomicOr(status, kValueNonfinite);
            continue;
        }
        const uint32_t ordered = ordered_float(value);
        ++count;
        sum += static_cast<double>(value);
        min_value = min(min_value, ordered);
        max_value = max(max_value, ordered);
    }

    counts[threadIdx.x] = count;
    sums[threadIdx.x] = sum;
    mins[threadIdx.x] = min_value;
    maxs[threadIdx.x] = max_value;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
            sums[threadIdx.x] += sums[threadIdx.x + stride];
            mins[threadIdx.x] = min(mins[threadIdx.x], mins[threadIdx.x + stride]);
            maxs[threadIdx.x] = max(maxs[threadIdx.x], maxs[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = counts[0];
        block_sums[blockIdx.x] = sums[0];
        block_mins[blockIdx.x] = mins[0];
        block_maxs[blockIdx.x] = maxs[0];
    }
}

extern "C" __global__ void olap_group_init(
    unsigned long long* slots,
    unsigned long long* counts,
    double* sums,
    uint32_t* mins,
    uint32_t* maxs,
    uint32_t capacity) {
    for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < capacity;
         index += blockDim.x * gridDim.x) {
        slots[index] = 0ULL;
        counts[index] = 0ULL;
        sums[index] = 0.0;
        mins[index] = 0xffffffffU;
        maxs[index] = 0U;
    }
}

extern "C" __global__ void olap_group_reduce_f32(
    const float* values,
    const float* group_keys,
    uint32_t rows,
    uint32_t max_groups,
    unsigned long long* slots,
    unsigned long long* counts,
    double* sums,
    uint32_t* mins,
    uint32_t* maxs,
    uint32_t capacity,
    uint32_t* unique_count,
    uint32_t* status) {
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < rows;
         row += blockDim.x * gridDim.x) {
        const float value = values[row];
        const float group = group_keys[row];
        if (!isfinite(group)) {
            atomicOr(status, kGroupNonfinite);
            continue;
        }
        if (!isfinite(value)) {
            continue;
        }

        const uint32_t key = __float_as_uint(group);
        const unsigned long long encoded = (1ULL << 32) | key;
        const uint32_t mask = capacity - 1U;
        uint32_t slot = hash_key(key) & mask;
        bool found = false;
        for (uint32_t probe = 0; probe < capacity; ++probe) {
            const unsigned long long previous = atomicCAS(&slots[slot], 0ULL, encoded);
            if (previous == 0ULL) {
                const uint32_t count = atomicAdd(unique_count, 1U) + 1U;
                if (count > max_groups) {
                    atomicOr(status, kGroupCap);
                    break;
                }
                found = true;
                break;
            }
            if (previous == encoded) {
                found = true;
                break;
            }
            slot = (slot + 1U) & mask;
        }
        if (!found) {
            if ((*status & kGroupCap) == 0U) {
                atomicOr(status, kDictionaryFull);
            }
            continue;
        }

        atomicAdd(&counts[slot], 1ULL);
        atomicAdd(&sums[slot], static_cast<double>(value));
        const uint32_t ordered = ordered_float(value);
        atomicMin(&mins[slot], ordered);
        atomicMax(&maxs[slot], ordered);
    }
}

extern "C" __global__ void olap_transpose_f32(
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t columns) {
    __shared__ float tile[32][33];
    const uint32_t input_column = blockIdx.x * 32U + threadIdx.x;
    const uint32_t input_row = blockIdx.y * 32U + threadIdx.y;

    for (uint32_t offset = 0; offset < 32U; offset += 8U) {
        if (input_column < columns && input_row + offset < rows) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * columns + input_column];
        }
    }
    __syncthreads();

    const uint32_t output_row = blockIdx.x * 32U + threadIdx.y;
    const uint32_t output_column = blockIdx.y * 32U + threadIdx.x;
    for (uint32_t offset = 0; offset < 32U; offset += 8U) {
        if (output_row + offset < columns && output_column < rows) {
            output[(output_row + offset) * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}
