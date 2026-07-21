#include <stdint.h>
#include <math.h>

#define CALYX_SPARSE_BM25_MAX_K 1024
#define CALYX_SPARSE_BM25_THREADS 256

__device__ __forceinline__ bool sparse_pair_better(
    float left_score,
    uint32_t left_doc,
    float right_score,
    uint32_t right_doc) {
  return left_score > right_score ||
         (left_score == right_score && left_doc < right_doc);
}

__device__ __forceinline__ int find_doc_tf(
    const uint32_t* posting_doc_ordinals,
    uint32_t begin,
    uint32_t end,
    uint32_t doc) {
  uint32_t lo = begin;
  uint32_t hi = end;
  while (lo < hi) {
    const uint32_t mid = lo + ((hi - lo) >> 1);
    const uint32_t value = posting_doc_ordinals[mid];
    if (value < doc) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < end && posting_doc_ordinals[lo] == doc) ? (int)lo : -1;
}

extern "C" __global__ void sparse_bm25_score_docs(
    const uint32_t* term_offsets,
    const uint32_t* posting_doc_ordinals,
    const float* posting_tfs,
    const float* doc_lengths,
    const uint8_t* candidate_mask,
    const uint32_t* query_term_ordinals,
    const float* query_weights,
    int total_docs,
    int query_terms,
    float avg_doc_len,
    float k1,
    float b,
    float* scores) {
  const uint32_t doc =
      (uint32_t)blockIdx.x * (uint32_t)blockDim.x + (uint32_t)threadIdx.x;
  if (doc >= (uint32_t)total_docs) return;
  if (candidate_mask[doc] == 0) {
    scores[doc] = 0.0f;
    return;
  }

  float score = 0.0f;
  const float doc_len = doc_lengths[doc];
  for (int query = 0; query < query_terms; ++query) {
    const uint32_t term = query_term_ordinals[query];
    const uint32_t begin = term_offsets[term];
    const uint32_t end = term_offsets[term + 1U];
    const int posting_index = find_doc_tf(posting_doc_ordinals, begin, end, doc);
    if (posting_index < 0) continue;

    const float tf = posting_tfs[posting_index];
    const uint32_t df = end - begin;
    if (!isfinite(tf) || tf <= 0.0f || df == 0U ||
        !isfinite(doc_len) || doc_len < 0.0f ||
        !isfinite(avg_doc_len) || avg_doc_len < 0.0f ||
        total_docs <= 0) {
      continue;
    }
    const float len_norm = avg_doc_len <= 0.0f ? 1.0f : doc_len / avg_doc_len;
    const float denom = tf + k1 * (1.0f - b + b * len_norm);
    const float idf =
        logf((((float)total_docs - (float)df + 0.5f) / ((float)df + 0.5f)) +
             1.0f);
    score += idf * (tf * (k1 + 1.0f)) / denom * query_weights[query];
  }
  scores[doc] = isfinite(score) ? score : 0.0f;
}

extern "C" __global__ void sparse_bm25_topk(
    const float* scores,
    int total_docs,
    int k,
    uint32_t* out_doc_ordinals,
    float* out_scores,
    uint32_t* out_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (k > CALYX_SPARSE_BM25_MAX_K) k = CALYX_SPARSE_BM25_MAX_K;

  for (int i = 0; i < k; ++i) {
    out_doc_ordinals[i] = UINT32_MAX;
    out_scores[i] = -INFINITY;
  }

  uint32_t seen = 0;
  uint32_t kept = 0;
  for (uint32_t doc = 0; doc < (uint32_t)total_docs; ++doc) {
    const float score = scores[doc];
    if (!(score > 0.0f) || !isfinite(score)) continue;
    ++seen;

    int insert_at = -1;
    const uint32_t limit = kept < (uint32_t)k ? kept : (uint32_t)k;
    for (uint32_t pos = 0; pos < limit; ++pos) {
      if (sparse_pair_better(score, doc, out_scores[pos], out_doc_ordinals[pos])) {
        insert_at = (int)pos;
        break;
      }
    }
    if (insert_at < 0 && kept < (uint32_t)k) insert_at = (int)kept;
    if (insert_at < 0 || insert_at >= k) continue;

    const uint32_t shift_end = kept < (uint32_t)k ? kept : (uint32_t)(k - 1);
    for (int pos = (int)shift_end; pos > insert_at; --pos) {
      out_doc_ordinals[pos] = out_doc_ordinals[pos - 1];
      out_scores[pos] = out_scores[pos - 1];
    }
    out_doc_ordinals[insert_at] = doc;
    out_scores[insert_at] = score;
    if (kept < (uint32_t)k) ++kept;
  }
  out_count[0] = seen < (uint32_t)k ? seen : (uint32_t)k;
}
