#include <stdint.h>

namespace {

__device__ __constant__ uint32_t B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};
__device__ __constant__ uint8_t B3_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
};
constexpr uint32_t B3_CHUNK_START = 1u;
constexpr uint32_t B3_CHUNK_END = 2u;
constexpr uint32_t B3_ROOT = 8u;
constexpr uint32_t BYTE_FEATURE_STRIDE = 15u;
constexpr uint32_t THREADS = 256u;
constexpr uint32_t TOKEN_DOMAIN_LEN = 31u;
__device__ __constant__ char TOKEN_DOMAIN[TOKEN_DOMAIN_LEN + 1] =
    "calyx-algorithmic-token-hash-v1";

__device__ __forceinline__ uint32_t rotr32(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32u - count));
}

__device__ __forceinline__ void b3_g(
    uint32_t state[16], uint32_t a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t x, uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = rotr32(state[d] ^ state[a], 16u);
    state[c] += state[d];
    state[b] = rotr32(state[b] ^ state[c], 12u);
    state[a] = state[a] + state[b] + y;
    state[d] = rotr32(state[d] ^ state[a], 8u);
    state[c] += state[d];
    state[b] = rotr32(state[b] ^ state[c], 7u);
}

__device__ __forceinline__ void b3_round(uint32_t state[16], const uint32_t m[16]) {
    b3_g(state, 0, 4, 8, 12, m[0], m[1]);
    b3_g(state, 1, 5, 9, 13, m[2], m[3]);
    b3_g(state, 2, 6, 10, 14, m[4], m[5]);
    b3_g(state, 3, 7, 11, 15, m[6], m[7]);
    b3_g(state, 0, 5, 10, 15, m[8], m[9]);
    b3_g(state, 1, 6, 11, 12, m[10], m[11]);
    b3_g(state, 2, 7, 8, 13, m[12], m[13]);
    b3_g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ __forceinline__ void b3_permute(uint32_t m[16]) {
    uint32_t permuted[16];
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        permuted[i] = m[B3_PERMUTATION[i]];
    }
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        m[i] = permuted[i];
    }
}

__device__ __forceinline__ void b3_compress(
    const uint32_t cv[8], const uint32_t block[16], uint32_t block_len,
    uint32_t flags, uint32_t out[16]) {
    uint32_t state[16];
    uint32_t message[16];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        state[i] = cv[i];
        message[i] = block[i];
        message[i + 8] = block[i + 8];
    }
    state[8] = B3_IV[0];
    state[9] = B3_IV[1];
    state[10] = B3_IV[2];
    state[11] = B3_IV[3];
    state[12] = 0;
    state[13] = 0;
    state[14] = block_len;
    state[15] = flags;
#pragma unroll
    for (uint32_t round = 0; round < 7; ++round) {
        b3_round(state, message);
        if (round != 6) {
            b3_permute(message);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        out[i] = state[i] ^ state[i + 8];
        out[i + 8] = state[i + 8] ^ cv[i];
    }
}

// kind 0 is the content-address message: u64 big-endian length || token.
// kind 1 is the token-vector message: fixed domain || token || u32 counter.
__device__ __forceinline__ uint8_t logical_byte(
    const uint8_t *bytes, uint32_t start, uint32_t token_len, uint32_t kind,
    uint32_t counter, uint32_t index) {
    if (kind == 0) {
        if (index < 8) {
            uint32_t shift = (7u - index) * 8u;
            return static_cast<uint8_t>(static_cast<uint64_t>(token_len) >> shift);
        }
        return bytes[start + index - 8u];
    }
    if (index < TOKEN_DOMAIN_LEN) {
        return static_cast<uint8_t>(TOKEN_DOMAIN[index]);
    }
    index -= TOKEN_DOMAIN_LEN;
    if (index < token_len) {
        return bytes[start + index];
    }
    index -= token_len;
    return static_cast<uint8_t>(counter >> ((3u - index) * 8u));
}

__device__ __forceinline__ void logical_block(
    const uint8_t *bytes, uint32_t start, uint32_t token_len, uint32_t kind,
    uint32_t counter, uint32_t message_offset, uint32_t block_len,
    uint32_t block[16]) {
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        block[i] = 0;
    }
    for (uint32_t i = 0; i < block_len; ++i) {
        uint32_t word = i >> 2;
        uint32_t shift = (i & 3u) * 8u;
        block[word] |= static_cast<uint32_t>(logical_byte(
            bytes, start, token_len, kind, counter, message_offset + i)) << shift;
    }
}

__device__ __forceinline__ void blake3_single_chunk(
    const uint8_t *bytes, uint32_t start, uint32_t token_len, uint32_t kind,
    uint32_t counter, uint32_t out[16]) {
    uint32_t message_len = kind == 0
        ? token_len + 8u
        : TOKEN_DOMAIN_LEN + token_len + 4u;
    uint32_t cv[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        cv[i] = B3_IV[i];
    }
    uint32_t offset = 0;
    bool first = true;
    while (message_len - offset > 64u) {
        uint32_t block[16];
        uint32_t compressed[16];
        logical_block(bytes, start, token_len, kind, counter, offset, 64u, block);
        b3_compress(cv, block, 64u, first ? B3_CHUNK_START : 0u, compressed);
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            cv[i] = compressed[i];
        }
        offset += 64u;
        first = false;
    }
    uint32_t last_len = message_len - offset;
    uint32_t last[16];
    logical_block(bytes, start, token_len, kind, counter, offset, last_len, last);
    uint32_t flags = B3_CHUNK_END | B3_ROOT;
    if (first) {
        flags |= B3_CHUNK_START;
    }
    b3_compress(cv, last, last_len, flags, out);
}

__device__ __forceinline__ uint32_t big_endian_word(uint32_t word) {
    return ((word & 0x000000ffu) << 24u) |
           ((word & 0x0000ff00u) << 8u) |
           ((word & 0x00ff0000u) >> 8u) |
           ((word & 0xff000000u) >> 24u);
}

__device__ __forceinline__ bool ascii_punctuation(uint8_t byte) {
    return (byte >= 33u && byte <= 47u) ||
           (byte >= 58u && byte <= 64u) ||
           (byte >= 91u && byte <= 96u) ||
           (byte >= 123u && byte <= 126u);
}

}  // namespace

extern "C" __global__ void algorithmic_byte_features(
    const uint8_t *bytes, const uint32_t *offsets, uint32_t rows,
    uint64_t *output) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    uint32_t start = offsets[row];
    uint32_t end = offsets[row + 1u];
    uint64_t counts[BYTE_FEATURE_STRIDE] = {};
    counts[0] = static_cast<uint64_t>(end - start);
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t pos = start; pos < end; ++pos) {
        uint8_t byte = bytes[pos];
        hash = (hash ^ static_cast<uint64_t>(byte)) * 0x100000001b3ull;
        counts[1] += byte < 128u;
        counts[2] += byte == 9u || byte == 10u || byte == 12u || byte == 13u || byte == 32u;
        counts[3] += (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z');
        counts[4] += byte >= '0' && byte <= '9';
        counts[5] += ascii_punctuation(byte);
        counts[6] += byte >= 'A' && byte <= 'Z';
        counts[7] += byte >= 'a' && byte <= 'z';
        counts[8] += byte <= 31u || byte == 127u;
        counts[9] += byte == 0u;
        counts[10] += byte == '/' || byte == '\\';
        counts[11] += byte == '{' || byte == '}' || byte == '(' || byte == ')' ||
                      byte == '[' || byte == ']';
        counts[12] += byte == '\n' || byte == '\r';
        counts[13] += byte;
    }
    counts[14] = hash;
#pragma unroll
    for (uint32_t i = 0; i < BYTE_FEATURE_STRIDE; ++i) {
        output[static_cast<uint64_t>(row) * BYTE_FEATURE_STRIDE + i] = counts[i];
    }
}

extern "C" __global__ void algorithmic_sparse_hashes(
    const uint8_t *bytes, const uint32_t *offsets, uint32_t tokens,
    uint32_t *output) {
    uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= tokens) {
        return;
    }
    uint32_t start = offsets[token];
    uint32_t token_len = offsets[token + 1u] - start;
    uint32_t digest[16];
    blake3_single_chunk(bytes, start, token_len, 0u, 0u, digest);
    output[token] = big_endian_word(digest[0]);
}

extern "C" __global__ void algorithmic_token_hash_words(
    const uint8_t *bytes, const uint32_t *offsets, uint32_t tokens,
    uint32_t token_dim, uint32_t groups, uint32_t *output) {
    uint32_t job = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t jobs = static_cast<uint64_t>(tokens) * groups;
    if (static_cast<uint64_t>(job) >= jobs) {
        return;
    }
    uint32_t token = job / groups;
    uint32_t counter = job - token * groups;
    uint32_t start = offsets[token];
    uint32_t token_len = offsets[token + 1u] - start;
    uint32_t digest[16];
    blake3_single_chunk(bytes, start, token_len, 1u, counter, digest);
    uint32_t base = token * token_dim + counter * 8u;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        if (base + i < (token + 1u) * token_dim) {
            output[base + i] = big_endian_word(digest[i]);
        }
    }
}
