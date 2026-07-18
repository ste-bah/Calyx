#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxPoints = 2048;

constexpr uint32_t kNonfinite = 1u << 0;
constexpr uint32_t kZeroNorm = 1u << 1;
constexpr uint32_t kNoOverlap = 1u << 2;
constexpr uint32_t kPrimInvariant = 1u << 3;

__device__ __forceinline__ bool better_pair(double left_value, int left_index,
                                             double right_value, int right_index) {
    return left_value < right_value ||
           (left_value == right_value && left_index < right_index);
}

}  // namespace

extern "C" __global__ __launch_bounds__(kThreads) void skill_pairwise_fused_cosine_f64(
    const float *values,
    const int64_t *offsets,
    const int32_t *slot_dims,
    int points,
    int slots,
    double *distances,
    uint32_t *status) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t cells = static_cast<int64_t>(points) * points;
    if (linear >= cells) {
        return;
    }
    const int row = static_cast<int>(linear / points);
    const int col = static_cast<int>(linear - static_cast<int64_t>(row) * points);
    if (row == col) {
        distances[linear] = 0.0;
        return;
    }
    if (row > col) {
        return;
    }

    double cosine_sum = 0.0;
    int shared = 0;
    for (int slot = 0; slot < slots; ++slot) {
        const int64_t left_offset = offsets[static_cast<int64_t>(row) * slots + slot];
        const int64_t right_offset = offsets[static_cast<int64_t>(col) * slots + slot];
        if (left_offset < 0 || right_offset < 0) {
            continue;
        }
        const int dim = slot_dims[slot];
        double dot = 0.0;
        double left_norm = 0.0;
        double right_norm = 0.0;
        bool finite = true;
        for (int feature = 0; feature < dim; ++feature) {
            const float left = values[left_offset + feature];
            const float right = values[right_offset + feature];
            finite = finite && isfinite(left) && isfinite(right);
            const double left64 = static_cast<double>(left);
            const double right64 = static_cast<double>(right);
            dot += left64 * right64;
            left_norm += left64 * left64;
            right_norm += right64 * right64;
        }
        if (!finite || !isfinite(dot) || !isfinite(left_norm) || !isfinite(right_norm)) {
            atomicOr(status, kNonfinite);
            continue;
        }
        const double denominator = sqrt(left_norm) * sqrt(right_norm);
        if (!(denominator > 2.2204460492503131e-16)) {
            atomicOr(status, kZeroNorm);
            continue;
        }
        const float cosine = static_cast<float>(dot / denominator);
        cosine_sum += static_cast<double>(cosine);
        ++shared;
    }
    if (shared == 0) {
        atomicOr(status, kNoOverlap);
        return;
    }
    double distance = 1.0 - cosine_sum / static_cast<double>(shared);
    distance = fmin(2.0, fmax(0.0, distance));
    distances[static_cast<int64_t>(row) * points + col] = distance;
    distances[static_cast<int64_t>(col) * points + row] = distance;
}

extern "C" __global__ __launch_bounds__(kThreads) void skill_core_distance_sort_f64(
    const double *distances,
    int points,
    int sort_length,
    int neighbor_rank,
    double *core,
    const uint32_t *status) {
    if (status[0] != 0u) {
        return;
    }
    const int row = static_cast<int>(blockIdx.x);
    if (row >= points) {
        return;
    }
    __shared__ double ordered[kMaxPoints];
    for (int col = static_cast<int>(threadIdx.x); col < sort_length; col += blockDim.x) {
        ordered[col] = col < points && col != row
                           ? distances[static_cast<int64_t>(row) * points + col]
                           : INFINITY;
    }
    __syncthreads();
    for (int width = 2; width <= sort_length; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int index = static_cast<int>(threadIdx.x); index < sort_length;
                 index += blockDim.x) {
                const int other = index ^ stride;
                if (other > index) {
                    const bool ascending = (index & width) == 0;
                    const double left = ordered[index];
                    const double right = ordered[other];
                    if ((ascending && right < left) || (!ascending && left < right)) {
                        ordered[index] = right;
                        ordered[other] = left;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        core[row] = ordered[neighbor_rank - 1];
    }
}

extern "C" __global__ __launch_bounds__(kThreads) void skill_prim_mst_f64(
    const double *distances,
    const double *core,
    int points,
    uint32_t *edge_sources,
    uint32_t *edge_destinations,
    double *edge_weights,
    uint32_t *status) {
    if (status[0] != 0u) {
        return;
    }
    __shared__ double keys[kMaxPoints];
    __shared__ int parents[kMaxPoints];
    __shared__ uint8_t in_tree[kMaxPoints];
    __shared__ double best_values[kThreads];
    __shared__ int best_indices[kThreads];
    __shared__ int selected;

    for (int point = static_cast<int>(threadIdx.x); point < points; point += blockDim.x) {
        keys[point] = point == 0 ? 0.0 : INFINITY;
        parents[point] = -1;
        in_tree[point] = 0u;
    }
    __syncthreads();

    for (int iteration = 0; iteration < points; ++iteration) {
        double local_value = INFINITY;
        int local_index = points;
        for (int point = static_cast<int>(threadIdx.x); point < points; point += blockDim.x) {
            if (in_tree[point] == 0u && better_pair(keys[point], point, local_value, local_index)) {
                local_value = keys[point];
                local_index = point;
            }
        }
        best_values[threadIdx.x] = local_value;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (static_cast<int>(threadIdx.x) < stride &&
                better_pair(best_values[threadIdx.x + stride], best_indices[threadIdx.x + stride],
                            best_values[threadIdx.x], best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = best_values[threadIdx.x + stride];
                best_indices[threadIdx.x] = best_indices[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected = best_indices[0];
            if (selected >= points || !isfinite(best_values[0])) {
                atomicOr(status, kPrimInvariant);
            } else {
                in_tree[selected] = 1u;
                if (iteration > 0) {
                    const int parent = parents[selected];
                    if (parent < 0) {
                        atomicOr(status, kPrimInvariant);
                    } else {
                        edge_sources[iteration - 1] = static_cast<uint32_t>(min(parent, selected));
                        edge_destinations[iteration - 1] =
                            static_cast<uint32_t>(max(parent, selected));
                        edge_weights[iteration - 1] = keys[selected];
                    }
                }
            }
        }
        __syncthreads();
        if (status[0] != 0u) {
            return;
        }
        const int source = selected;
        for (int point = static_cast<int>(threadIdx.x); point < points; point += blockDim.x) {
            if (in_tree[point] == 0u) {
                double weight = distances[static_cast<int64_t>(source) * points + point];
                weight = fmax(weight, fmax(core[source], core[point]));
                if (weight < keys[point]) {
                    keys[point] = weight;
                    parents[point] = source;
                }
            }
        }
        __syncthreads();
    }
}
