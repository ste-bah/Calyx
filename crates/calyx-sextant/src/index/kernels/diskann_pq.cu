#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

namespace {

constexpr int kAccumulationAxisTile = 8;

__device__ __forceinline__ uint8_t nearest_centroid(
    const float* row,
    const float* codebook,
    int centroids,
    int subdim) {
  int best = 0;
  float best_distance = FLT_MAX;
  for (int centroid = 0; centroid < centroids; ++centroid) {
    const float* candidate = codebook + centroid * subdim;
    float distance = 0.0f;
    for (int axis = 0; axis < subdim; ++axis) {
      const float delta = row[axis] - candidate[axis];
      distance += delta * delta;
    }
    // Strict comparison preserves the CPU path's lowest-id tie break.
    if (distance < best_distance) {
      best_distance = distance;
      best = centroid;
    }
  }
  return static_cast<uint8_t>(best);
}

}  // namespace

extern "C" __global__ void diskann_pq_assign_nearest(
    const float* rows,
    int row_count,
    int dim,
    int subvectors,
    int centroids,
    int subdim,
    const float* codebook,
    uint8_t* labels) {
  const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t item_count = static_cast<size_t>(row_count) * subvectors;
  if (item >= item_count) return;

  const int row_id = static_cast<int>(item / subvectors);
  const int subvector = static_cast<int>(item % subvectors);
  const float* row = rows + static_cast<size_t>(row_id) * dim + subvector * subdim;
  const float* subspace_codebook =
      codebook + static_cast<size_t>(subvector) * centroids * subdim;
  labels[item] = nearest_centroid(row, subspace_codebook, centroids, subdim);
}

extern "C" __global__ void diskann_pq_accumulate_tiled(
    const float* rows,
    const uint8_t* labels,
    int row_count,
    int dim,
    int subvectors,
    int centroids,
    int subdim,
    int axis_tiles,
    float* sums,
    uint32_t* counts) {
  extern __shared__ unsigned char shared_bytes[];
  float* local_sums = reinterpret_cast<float*>(shared_bytes);
  uint32_t* local_counts = reinterpret_cast<uint32_t*>(
      local_sums + static_cast<size_t>(centroids) * kAccumulationAxisTile);

  const int group = static_cast<int>(blockIdx.y);
  const int subvector = group / axis_tiles;
  const int axis_tile = group % axis_tiles;
  const int axis_start = axis_tile * kAccumulationAxisTile;
  const int sum_cells = centroids * kAccumulationAxisTile;
  for (int cell = threadIdx.x; cell < sum_cells; cell += blockDim.x) {
    local_sums[cell] = 0.0f;
  }
  for (int centroid = threadIdx.x; centroid < centroids; centroid += blockDim.x) {
    local_counts[centroid] = 0;
  }
  __syncthreads();

  const int row_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_id < row_count && subvector < subvectors) {
    const uint8_t label = labels[static_cast<size_t>(row_id) * subvectors + subvector];
    const float* row = rows + static_cast<size_t>(row_id) * dim +
                       subvector * subdim + axis_start;
    const int remaining = subdim - axis_start;
    const int width = remaining < kAccumulationAxisTile ? remaining : kAccumulationAxisTile;
    float* destination = local_sums + static_cast<int>(label) * kAccumulationAxisTile;
    for (int axis = 0; axis < width; ++axis) {
      atomicAdd(destination + axis, row[axis]);
    }
    if (axis_tile == 0) atomicAdd(local_counts + label, 1u);
  }
  __syncthreads();

  for (int cell = threadIdx.x; cell < sum_cells; cell += blockDim.x) {
    const int centroid = cell / kAccumulationAxisTile;
    const int axis = axis_start + cell % kAccumulationAxisTile;
    if (axis < subdim) {
      const size_t destination =
          (static_cast<size_t>(subvector) * centroids + centroid) * subdim + axis;
      const float value = local_sums[cell];
      if (value != 0.0f) atomicAdd(sums + destination, value);
    }
  }
  if (axis_tile == 0) {
    for (int centroid = threadIdx.x; centroid < centroids; centroid += blockDim.x) {
      const uint32_t value = local_counts[centroid];
      if (value != 0) {
        atomicAdd(counts + static_cast<size_t>(subvector) * centroids + centroid, value);
      }
    }
  }
}

extern "C" __global__ void diskann_pq_finalize_centroids(
    const float* sums,
    const uint32_t* counts,
    int centroids,
    int subdim,
    size_t codebook_cells,
    float* codebook) {
  const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell >= codebook_cells) return;
  const size_t cluster = cell / subdim;
  const uint32_t count = counts[cluster];
  if (count != 0) codebook[cell] = sums[cell] / static_cast<float>(count);
}
