#include <stdint.h>
#include <math.h>

#define CALYX_CHUNKED_EXACT_MAX_K 1024
#define CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK 256
#define CALYX_BOUNDARY_REPAIR_COLS_PER_TILE 32

__device__ __forceinline__ bool pair_less(
    float left_distance,
    int64_t left_id,
    float right_distance,
    int64_t right_id) {
  return left_distance < right_distance ||
         (left_distance == right_distance && left_id < right_id);
}

__device__ __forceinline__ uint32_t rotate_left(uint32_t value, int shift) {
  return (value << shift) | (value >> (32 - shift));
}

__device__ __forceinline__ uint32_t pcg32(uint64_t* state) {
  *state = *state * 6364136223846793005ULL + 11634580027462260723ULL;
  const uint32_t xorshifted = (uint32_t)((((*state) >> 18) ^ (*state)) >> 27);
  const uint32_t rotation = (uint32_t)((*state) >> 59);
  return (xorshifted >> rotation) |
         (xorshifted << ((0U - rotation) & 31U));
}

__device__ __forceinline__ void chacha_quarter(
    uint32_t* a,
    uint32_t* b,
    uint32_t* c,
    uint32_t* d) {
  *a += *b;
  *d = rotate_left(*d ^ *a, 16);
  *c += *d;
  *b = rotate_left(*b ^ *c, 12);
  *a += *b;
  *d = rotate_left(*d ^ *a, 8);
  *c += *d;
  *b = rotate_left(*b ^ *c, 7);
}

__device__ __forceinline__ void chacha8_block(
    const uint32_t* key,
    uint64_t counter,
    uint32_t* output) {
  uint32_t state[16] = {
      0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
      key[0], key[1], key[2], key[3],
      key[4], key[5], key[6], key[7],
      (uint32_t)counter, (uint32_t)(counter >> 32), 0U, 0U};
  uint32_t working[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) working[i] = state[i];
#pragma unroll
  for (int round = 0; round < 4; ++round) {
    chacha_quarter(&working[0], &working[4], &working[8], &working[12]);
    chacha_quarter(&working[1], &working[5], &working[9], &working[13]);
    chacha_quarter(&working[2], &working[6], &working[10], &working[14]);
    chacha_quarter(&working[3], &working[7], &working[11], &working[15]);
    chacha_quarter(&working[0], &working[5], &working[10], &working[15]);
    chacha_quarter(&working[1], &working[6], &working[11], &working[12]);
    chacha_quarter(&working[2], &working[7], &working[8], &working[13]);
    chacha_quarter(&working[3], &working[4], &working[9], &working[14]);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) output[i] = working[i] + state[i];
}

extern "C" __global__ void convert_i8_chunk_to_f32(
    const int8_t* source,
    int rows,
    int dim,
    int normalize,
    float* destination) {
  const int64_t index =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (!normalize) {
    const int64_t values = (int64_t)rows * (int64_t)dim;
    if (index < values) destination[index] = (float)source[index];
    return;
  }
  if (index >= rows) return;
  const int64_t base = index * dim;
  float norm = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const float value = (float)source[base + col];
    destination[base + col] = value;
    norm += value * value;
  }
  norm = sqrtf(norm);
  if (norm > 0.0f) {
    for (int col = 0; col < dim; ++col) destination[base + col] /= norm;
  }
}

extern "C" __global__ void generate_synthetic_chunk(
    uint64_t seed,
    uint64_t start,
    int rows,
    int dim,
    float* destination) {
  const int local_row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (local_row >= rows) return;
  const uint64_t row = start + (uint64_t)local_row;
  uint64_t pcg_state = seed ^ (row * 0x9E3779B97F4A7C15ULL);
  uint32_t key[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) key[i] = pcg32(&pcg_state);

  float norm = 0.0f;
  const int64_t base = (int64_t)local_row * dim;
  for (int col_base = 0; col_base < dim; col_base += 16) {
    uint32_t random_words[16];
    chacha8_block(key, (uint64_t)(col_base / 16), random_words);
    const int block_cols = dim - col_base < 16 ? dim - col_base : 16;
    for (int offset = 0; offset < block_cols; ++offset) {
      const int col = col_base + offset;
      const uint32_t mantissa = random_words[offset] >> 9;
      const float unit = __uint_as_float(0x3f800000U | mantissa) - 1.0f;
      const uint64_t ramp = (row + (uint64_t)col) % (uint64_t)dim;
      float value = (unit * 2.0f - 1.0f) + (float)ramp * 0.001f;
      if ((uint64_t)col == row % (uint64_t)dim) value += 4.0f;
      destination[base + col] = value;
      norm += value * value;
    }
  }
  norm = sqrtf(norm);
  if (norm > 0.0f) {
    for (int col = 0; col < dim; ++col) destination[base + col] /= norm;
  }
}

extern "C" __global__ void merge_chunked_exact_topk(
    int64_t* chunk_ids,
    float* chunk_distances,
    int64_t* global_ids,
    float* global_distances,
    int query_count,
    int candidate_stride,
    int chunk_k,
    int global_k,
    int old_count,
    int output_count,
    int64_t chunk_base) {
  const int query = (int)blockIdx.x;
  if (query >= query_count || threadIdx.x != 0) return;

  __shared__ int64_t old_ids[CALYX_CHUNKED_EXACT_MAX_K];
  __shared__ float old_distances[CALYX_CHUNKED_EXACT_MAX_K];

  const int chunk_offset = query * candidate_stride;
  const int global_offset = query * global_k;

  // cuVS orders by distance. Canonicalize every selected chunk by global row
  // id as well so equal-distance ranks match the CPU oracle exactly.
  for (int i = 1; i < chunk_k; ++i) {
    const int64_t id = chunk_ids[chunk_offset + i];
    const float distance = chunk_distances[chunk_offset + i];
    int j = i;
    while (j > 0 && pair_less(
        distance,
        id,
        chunk_distances[chunk_offset + j - 1],
        chunk_ids[chunk_offset + j - 1])) {
      chunk_ids[chunk_offset + j] = chunk_ids[chunk_offset + j - 1];
      chunk_distances[chunk_offset + j] = chunk_distances[chunk_offset + j - 1];
      --j;
    }
    chunk_ids[chunk_offset + j] = id;
    chunk_distances[chunk_offset + j] = distance;
  }

  for (int i = 0; i < old_count; ++i) {
    old_ids[i] = global_ids[global_offset + i];
    old_distances[i] = global_distances[global_offset + i];
  }

  int old_pos = 0;
  int chunk_pos = 0;
  for (int out = 0; out < output_count; ++out) {
    const bool use_old = old_pos < old_count &&
        (chunk_pos >= chunk_k || pair_less(
            old_distances[old_pos],
            old_ids[old_pos],
            chunk_distances[chunk_offset + chunk_pos],
            chunk_base + chunk_ids[chunk_offset + chunk_pos]));
    if (use_old) {
      global_ids[global_offset + out] = old_ids[old_pos];
      global_distances[global_offset + out] = old_distances[old_pos];
      ++old_pos;
    } else {
      global_ids[global_offset + out] =
          chunk_base + chunk_ids[chunk_offset + chunk_pos];
      global_distances[global_offset + out] =
          chunk_distances[chunk_offset + chunk_pos];
      ++chunk_pos;
    }
  }
}

extern "C" __global__ void repair_zero_cosine_queries(
    const float* queries,
    int dim,
    int query_count,
    int chunk_k,
    int64_t* chunk_ids,
    float* chunk_distances) {
  const int query = (int)blockIdx.x;
  if (query >= query_count || threadIdx.x != 0) return;
  float norm = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const float value = queries[query * dim + col];
    norm += value * value;
  }
  if (norm != 0.0f) return;
  for (int rank = 0; rank < chunk_k; ++rank) {
    chunk_ids[query * chunk_k + rank] = rank;
    chunk_distances[query * chunk_k + rank] = 1.0f;
  }
}

extern "C" __global__ void compute_chunked_exact_repair_distances(
    const float* corpus,
    int rows,
    int dim,
    const float* queries,
    int query_count,
    int query_start,
    int batch_count,
    int candidate_k,
    int output_k,
    int metric,
    const float* chunk_distances,
    float* repair_distances) {
  const int local_query = (int)blockIdx.y;
  const int query = query_start + local_query;
  if (local_query >= batch_count || query >= query_count ||
      candidate_k <= output_k) return;
  const int output = query * candidate_k;
  if (chunk_distances[output + output_k - 1] !=
      chunk_distances[output + output_k]) return;

  // The padded transpose keeps global row loads coalesced and shared-memory
  // reads bank-conflict free while preserving scalar column accumulation order.
  __shared__ float corpus_tile[
      CALYX_BOUNDARY_REPAIR_COLS_PER_TILE *
      (CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK + 1)];
  const int local_row = (int)threadIdx.x;
  const int row = (int)blockIdx.x * CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK +
                  local_row;
  float dot = 0.0f;
  float query_norm = 0.0f;
  float row_norm = 0.0f;
  float squared_l2 = 0.0f;

  for (int col_base = 0; col_base < dim;
       col_base += CALYX_BOUNDARY_REPAIR_COLS_PER_TILE) {
    for (int tile_index = (int)threadIdx.x;
         tile_index < CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK *
                          CALYX_BOUNDARY_REPAIR_COLS_PER_TILE;
         tile_index += (int)blockDim.x) {
      const int tile_row = tile_index / CALYX_BOUNDARY_REPAIR_COLS_PER_TILE;
      const int tile_col = tile_index % CALYX_BOUNDARY_REPAIR_COLS_PER_TILE;
      const int corpus_row =
          (int)blockIdx.x * CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK + tile_row;
      const int col = col_base + tile_col;
      corpus_tile[tile_col * (CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK + 1) +
                  tile_row] = corpus_row < rows && col < dim
          ? corpus[corpus_row * dim + col]
          : 0.0f;
    }
    __syncthreads();

    if (local_row < CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK && row < rows) {
      const int remaining_cols = dim - col_base;
      const int tile_cols = remaining_cols < CALYX_BOUNDARY_REPAIR_COLS_PER_TILE
          ? remaining_cols
          : CALYX_BOUNDARY_REPAIR_COLS_PER_TILE;
      for (int tile_col = 0; tile_col < tile_cols; ++tile_col) {
        const float left = queries[query * dim + col_base + tile_col];
        const float right = corpus_tile[
            tile_col * (CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK + 1) + local_row];
        if (metric == 0) {
          dot += left * right;
          query_norm += left * left;
          row_norm += right * right;
        } else {
          const float delta = left - right;
          squared_l2 += delta * delta;
        }
      }
    }
    __syncthreads();
  }

  if (local_row < CALYX_BOUNDARY_REPAIR_ROWS_PER_BLOCK && row < rows) {
    float distance = squared_l2;
    if (metric == 0) {
      distance = query_norm == 0.0f || row_norm == 0.0f
          ? 1.0f
          : fmaxf(0.0f, 1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
    }
    repair_distances[local_query * rows + row] = distance;
  }
}

extern "C" __global__ void repair_chunked_exact_boundary_ties(
    int rows,
    int query_count,
    int query_start,
    int batch_count,
    int candidate_k,
    int output_k,
    int64_t* chunk_ids,
    float* chunk_distances,
    const float* repair_distances) {
  const int local_query = (int)blockIdx.x;
  const int query = query_start + local_query;
  if (local_query >= batch_count || query >= query_count || threadIdx.x != 0 ||
      candidate_k <= output_k) return;
  const int output = query * candidate_k;
  if (chunk_distances[output + output_k - 1] !=
      chunk_distances[output + output_k]) return;

  int count = 0;
  for (int row = 0; row < rows; ++row) {
    const float distance = repair_distances[local_query * rows + row];
    int insert = count;
    if (count == output_k) {
      insert = output_k - 1;
      if (!pair_less(
          distance,
          row,
          chunk_distances[output + insert],
          chunk_ids[output + insert])) {
        continue;
      }
    } else {
      ++count;
    }
    while (insert > 0 && pair_less(
        distance,
        row,
        chunk_distances[output + insert - 1],
        chunk_ids[output + insert - 1])) {
      chunk_ids[output + insert] = chunk_ids[output + insert - 1];
      chunk_distances[output + insert] = chunk_distances[output + insert - 1];
      --insert;
    }
    chunk_ids[output + insert] = row;
    chunk_distances[output + insert] = distance;
  }
}

extern "C" __global__ void exact_cosine_chunk_with_zero_rows(
    const float* corpus,
    int rows,
    int dim,
    const float* queries,
    int query_count,
    int chunk_k,
    int64_t* chunk_ids,
    float* chunk_distances) {
  const int query = (int)blockIdx.x;
  if (query >= query_count || threadIdx.x != 0) return;
  const int output = query * chunk_k;
  float query_norm = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const float value = queries[query * dim + col];
    query_norm += value * value;
  }
  if (query_norm == 0.0f) {
    for (int rank = 0; rank < chunk_k; ++rank) {
      chunk_ids[output + rank] = rank;
      chunk_distances[output + rank] = 1.0f;
    }
    return;
  }

  int count = 0;
  for (int row = 0; row < rows; ++row) {
    float dot = 0.0f;
    float row_norm = 0.0f;
    for (int col = 0; col < dim; ++col) {
      const float left = queries[query * dim + col];
      const float right = corpus[row * dim + col];
      dot += left * right;
      row_norm += right * right;
    }
    const float distance = row_norm == 0.0f
        ? 1.0f
        : fmaxf(0.0f, 1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
    int insert = count;
    if (count == chunk_k) {
      insert = chunk_k - 1;
      if (!pair_less(
          distance,
          row,
          chunk_distances[output + insert],
          chunk_ids[output + insert])) {
        continue;
      }
    } else {
      ++count;
    }
    while (insert > 0 && pair_less(
        distance,
        row,
        chunk_distances[output + insert - 1],
        chunk_ids[output + insert - 1])) {
      chunk_ids[output + insert] = chunk_ids[output + insert - 1];
      chunk_distances[output + insert] = chunk_distances[output + insert - 1];
      --insert;
    }
    chunk_ids[output + insert] = row;
    chunk_distances[output + insert] = distance;
  }
}

// Resident flat-index exact path. It is used for filtered requests and for
// corpora containing zero rows, where cuVS cosine cannot preserve Calyx's
// zero-vector and deterministic tie conventions. The corpus, query matrix,
// bitset, and top-k outputs all remain on device.
extern "C" __global__ void exact_cosine_resident_filtered(
    const float* corpus,
    int rows,
    int dim,
    int64_t row_stride,
    int64_t column_stride,
    const float* queries,
    int query_count,
    const unsigned int* allowed,
    int metric,
    int output_k,
    int64_t* output_ids,
    float* output_distances) {
  const int query = (int)blockIdx.x;
  if (query >= query_count || threadIdx.x != 0) return;
  const int output = query * output_k;
  float query_norm = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const float value = queries[query * dim + col];
    query_norm += value * value;
  }
  int count = 0;
  for (int row = 0; row < rows; ++row) {
    if ((allowed[row >> 5] & (1u << (row & 31))) == 0) continue;
    float dot = 0.0f;
    float row_norm = 0.0f;
    for (int col = 0; col < dim; ++col) {
      const float left = queries[query * dim + col];
      const float right = corpus[row * row_stride + col * column_stride];
      dot += left * right;
      row_norm += right * right;
    }
    float distance = 0.0f;
    if (metric == 2) {
      for (int col = 0; col < dim; ++col) {
        const float delta = queries[query * dim + col]
            - corpus[row * row_stride + col * column_stride];
        distance += delta * delta;
      }
    } else {
      distance = query_norm == 0.0f || row_norm == 0.0f
          ? 1.0f
          : fmaxf(0.0f, 1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
      // CAGRA's UnitL2 path reports expanded L2, which is 2*cosine distance.
      // Preserve that contract so the shared host validator applies its 0.5 map.
      if (metric == 1) distance *= 2.0f;
    }
    int insert = count;
    if (count == output_k) {
      insert = output_k - 1;
      if (!pair_less(
          distance,
          row,
          output_distances[output + insert],
          output_ids[output + insert])) {
        continue;
      }
    } else {
      ++count;
    }
    while (insert > 0 && pair_less(
        distance,
        row,
        output_distances[output + insert - 1],
        output_ids[output + insert - 1])) {
      output_ids[output + insert] = output_ids[output + insert - 1];
      output_distances[output + insert] = output_distances[output + insert - 1];
      --insert;
    }
    output_ids[output + insert] = row;
    output_distances[output + insert] = distance;
  }
}

// Parallel resident exact path: one CUDA thread owns one corpus row and keeps
// the scalar column accumulation order used by the CPU oracle. A second
// deterministic kernel selects top-k by (distance, row id).
extern "C" __global__ void exact_cosine_resident_distances(
    const float* corpus,
    int rows,
    int dim,
    int64_t row_stride,
    int64_t column_stride,
    const float* queries,
    int query_count,
    const unsigned int* allowed,
    int metric,
    float* row_distances) {
  const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const int query = (int)blockIdx.y;
  if (row >= rows || query >= query_count) return;
  if ((allowed[row >> 5] & (1u << (row & 31))) == 0) {
    row_distances[query * rows + row] = INFINITY;
    return;
  }
  float dot = 0.0f;
  float query_norm = 0.0f;
  float row_norm = 0.0f;
  float squared_l2 = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const float left = queries[query * dim + col];
    const float right = corpus[row * row_stride + col * column_stride];
    dot += left * right;
    query_norm += left * left;
    row_norm += right * right;
    const float delta = left - right;
    squared_l2 += delta * delta;
  }
  float distance = squared_l2;
  if (metric != 2) {
    distance = query_norm == 0.0f || row_norm == 0.0f
        ? 1.0f
        : fmaxf(0.0f, 1.0f - dot / (sqrtf(query_norm) * sqrtf(row_norm)));
    if (metric == 1) distance *= 2.0f;
  }
  row_distances[query * rows + row] = distance;
}

extern "C" __global__ void exact_cosine_resident_topk(
    const float* row_distances,
    int rows,
    int query_count,
    int output_k,
    int64_t* output_ids,
    float* output_distances) {
  const int query = (int)blockIdx.x;
  if (query >= query_count || threadIdx.x != 0) return;
  const int output = query * output_k;
  int count = 0;
  for (int row = 0; row < rows; ++row) {
    const float distance = row_distances[query * rows + row];
    if (!isfinite(distance)) continue;
    int insert = count;
    if (count == output_k) {
      insert = output_k - 1;
      if (!pair_less(distance, row, output_distances[output + insert],
                     output_ids[output + insert])) continue;
    } else {
      ++count;
    }
    while (insert > 0 && pair_less(
        distance, row, output_distances[output + insert - 1],
        output_ids[output + insert - 1])) {
      output_ids[output + insert] = output_ids[output + insert - 1];
      output_distances[output + insert] = output_distances[output + insert - 1];
      --insert;
    }
    output_ids[output + insert] = row;
    output_distances[output + insert] = distance;
  }
}
