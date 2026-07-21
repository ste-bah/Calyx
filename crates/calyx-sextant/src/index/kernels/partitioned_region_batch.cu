#include <math.h>
#include <stdint.h>

#define CALYX_REGION_THREADS 256
#define CALYX_REGION_WARPS (CALYX_REGION_THREADS / 32)
#define CALYX_REGION_SMALL_K 32

__device__ __forceinline__ bool region_pair_less(
    float left_distance,
    uint64_t left_id,
    float right_distance,
    uint64_t right_id) {
  return left_distance < right_distance ||
      (left_distance == right_distance && left_id < right_id);
}

__device__ void insert_local(
    float distance,
    uint64_t id,
    int k,
    int* count,
    float* distances,
    uint64_t* ids) {
  int insert = *count;
  if (*count == k) {
    insert = k - 1;
    if (!region_pair_less(distance, id, distances[insert], ids[insert])) return;
  } else {
    ++*count;
  }
  while (insert > 0 &&
         region_pair_less(distance, id, distances[insert - 1], ids[insert - 1])) {
    distances[insert] = distances[insert - 1];
    ids[insert] = ids[insert - 1];
    --insert;
  }
  distances[insert] = distance;
  ids[insert] = id;
}

__device__ float warp_row_distance(
    const void* dataset,
    int dataset_dtype,
    int row,
    int dim,
    int64_t row_stride,
    int64_t column_stride,
    const float* query,
    int metric,
    float query_norm) {
  const int lane = (int)threadIdx.x & 31;
  float dot = 0.0f;
  float row_norm = 0.0f;
  float l2 = 0.0f;
  for (int col = lane; col < dim; col += 32) {
    const float left = query[col];
    const int64_t offset = row * row_stride + col * column_stride;
    const float right = dataset_dtype == 1
        ? (float)reinterpret_cast<const int8_t*>(dataset)[offset]
        : reinterpret_cast<const float*>(dataset)[offset];
    if (metric == 0) {
      const float delta = left - right;
      l2 += delta * delta;
    } else {
      dot += left * right;
      row_norm += right * right;
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    l2 += __shfl_down_sync(0xffffffff, l2, offset);
    dot += __shfl_down_sync(0xffffffff, dot, offset);
    row_norm += __shfl_down_sync(0xffffffff, row_norm, offset);
  }
  if (metric == 0) return l2;
  return query_norm == 0.0f || row_norm == 0.0f
      ? 1.0f
      : fmaxf(0.0f, 1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
}

extern "C" __global__ void partitioned_region_exact_small(
    const uint64_t* dataset_addresses,
    const int* dataset_dtypes,
    const uint64_t* global_id_addresses,
    const int* region_rows,
    const int64_t* row_strides,
    const int64_t* column_strides,
    int region_count,
    const float* query,
    int dim,
    int metric,
    int k,
    uint64_t* region_ids,
    float* region_distances) {
  const int region = (int)blockIdx.x;
  const int thread = (int)threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  if (region >= region_count || thread >= CALYX_REGION_THREADS ||
      k <= 0 || k > CALYX_REGION_SMALL_K) return;
  const void* dataset = reinterpret_cast<const void*>(dataset_addresses[region]);
  const uint64_t* global_ids =
      reinterpret_cast<const uint64_t*>(global_id_addresses[region]);
  __shared__ float query_norm;
  if (thread == 0) {
    query_norm = 0.0f;
    if (metric != 0) {
      for (int col = 0; col < dim; ++col) {
        query_norm += query[col] * query[col];
      }
    }
  }
  __shared__ float shared_distances[
      CALYX_REGION_WARPS * CALYX_REGION_SMALL_K];
  __shared__ uint64_t shared_ids[
      CALYX_REGION_WARPS * CALYX_REGION_SMALL_K];
  for (int rank = thread;
       rank < CALYX_REGION_WARPS * CALYX_REGION_SMALL_K;
       rank += CALYX_REGION_THREADS) {
    shared_distances[rank] = INFINITY;
    shared_ids[rank] = UINT64_MAX;
  }
  __syncthreads();
  int local_count = 0;
  for (int row = warp; row < region_rows[region]; row += CALYX_REGION_WARPS) {
    const float distance = warp_row_distance(
        dataset,
        dataset_dtypes[region],
        row,
        dim,
        row_strides[region],
        column_strides[region],
        query,
        metric,
        query_norm);
    if (lane == 0) {
      insert_local(
          distance,
          global_ids[row],
          k,
          &local_count,
          shared_distances + warp * CALYX_REGION_SMALL_K,
          shared_ids + warp * CALYX_REGION_SMALL_K);
    }
  }
  __syncthreads();
  if (thread != 0) return;
  const int output = region * k;
  int count = 0;
  for (int rank = 0; rank < k; ++rank) {
    region_distances[output + rank] = INFINITY;
    region_ids[output + rank] = UINT64_MAX;
  }
  for (int thread = 0; thread < CALYX_REGION_WARPS; ++thread) {
    for (int rank = 0; rank < k; ++rank) {
      const int candidate = thread * CALYX_REGION_SMALL_K + rank;
      if (shared_ids[candidate] == UINT64_MAX) break;
      insert_local(
          shared_distances[candidate],
          shared_ids[candidate],
          k,
          &count,
          region_distances + output,
          region_ids + output);
    }
  }
}

extern "C" __global__ void partitioned_region_exact_large(
    const uint64_t* dataset_addresses,
    const int* dataset_dtypes,
    const uint64_t* global_id_addresses,
    const int* region_rows,
    const int64_t* row_strides,
    const int64_t* column_strides,
    int region_count,
    const float* query,
    int dim,
    int metric,
    int k,
    uint64_t* region_ids,
    float* region_distances) {
  const int region = (int)blockIdx.x;
  if (region >= region_count || threadIdx.x != 0 || k <= 0) return;
  const void* dataset = reinterpret_cast<const void*>(dataset_addresses[region]);
  const uint64_t* global_ids =
      reinterpret_cast<const uint64_t*>(global_id_addresses[region]);
  float query_norm = 0.0f;
  if (metric != 0) {
    for (int col = 0; col < dim; ++col) query_norm += query[col] * query[col];
  }
  const int output = region * k;
  int count = 0;
  for (int rank = 0; rank < k; ++rank) {
    region_distances[output + rank] = INFINITY;
    region_ids[output + rank] = UINT64_MAX;
  }
  for (int row = 0; row < region_rows[region]; ++row) {
    float distance = 0.0f;
    float dot = 0.0f;
    float row_norm = 0.0f;
    for (int col = 0; col < dim; ++col) {
      const float left = query[col];
      const int64_t offset =
          row * row_strides[region] + col * column_strides[region];
      const float right = dataset_dtypes[region] == 1
          ? (float)reinterpret_cast<const int8_t*>(dataset)[offset]
          : reinterpret_cast<const float*>(dataset)[offset];
      if (metric == 0) {
        const float delta = left - right;
        distance += delta * delta;
      } else {
        dot += left * right;
        row_norm += right * right;
      }
    }
    if (metric != 0) {
      distance = query_norm == 0.0f || row_norm == 0.0f
          ? 1.0f
          : fmaxf(
                0.0f,
                1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
    }
    insert_local(
        distance,
        global_ids[row],
        k,
        &count,
        region_distances + output,
        region_ids + output);
  }
}

extern "C" __global__ void partitioned_region_merge_topk(
    const uint64_t* region_ids,
    const float* region_distances,
    int region_count,
    int k,
    uint64_t* output_ids,
    float* output_distances) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || k <= 0) return;
  int count = 0;
  for (int rank = 0; rank < k; ++rank) {
    output_ids[rank] = UINT64_MAX;
    output_distances[rank] = INFINITY;
  }
  for (int candidate = 0; candidate < region_count * k; ++candidate) {
    const uint64_t id = region_ids[candidate];
    if (id == UINT64_MAX) continue;
    bool duplicate = false;
    for (int rank = 0; rank < count; ++rank) {
      if (output_ids[rank] == id) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      insert_local(
          region_distances[candidate],
          id,
          k,
          &count,
          output_distances,
          output_ids);
    }
  }
}
