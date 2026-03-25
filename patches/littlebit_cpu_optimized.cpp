#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace littlebit_cpu_ops {

namespace {

// ============================================================================
// Scalar fallbacks
// ============================================================================

inline int32_t dot_int8xsign_word_scalar(const int8_t * input, uint32_t bits, int64_t cols_this_word) {
    int32_t acc = 0;
    for (int64_t bit = 0; bit < cols_this_word; ++bit) {
        const int32_t value = static_cast<int32_t>(input[bit]);
        const bool is_neg = ((bits >> bit) & 1U) != 0;
        acc += is_neg ? -value : value;
    }
    return acc;
}

inline int32_t dot_int8xsign_row_scalar(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int32_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        acc += dot_int8xsign_word_scalar(
            input_row + word_idx * 32,
            static_cast<uint32_t>(packed_weight_row[word_idx]),
            32
        );
    }

    if (tail_cols > 0) {
        acc += dot_int8xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

inline int64_t dot_int32xsign_word_scalar(const int32_t * input, uint32_t bits, int64_t cols_this_word) {
    int64_t acc = 0;
    for (int64_t bit = 0; bit < cols_this_word; ++bit) {
        const int64_t value = static_cast<int64_t>(input[bit]);
        const bool is_neg = ((bits >> bit) & 1U) != 0;
        acc += is_neg ? -value : value;
    }
    return acc;
}

inline int64_t dot_int32xsign_row_scalar(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int64_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        const uint32_t bits = static_cast<uint32_t>(packed_weight_row[word_idx]);
        acc += dot_int32xsign_word_scalar(input_row + word_idx * 32, bits, 32);
    }

    if (tail_cols > 0) {
        acc += dot_int32xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// ============================================================================
// AVX2 optimized paths
// ============================================================================

#ifdef __AVX2__

// --- Core: expand 32 packed sign bits to 32 sign bytes {-1, +1} ---
// Uses bit-manipulation instead of LUT. Inspired by llama.cpp's approach.
// bit=1 → -1 (negative), bit=0 → +1 (positive)
inline __m256i bits_to_sign_bytes(__m256i vbits, const __m256i & byte_sel, const __m256i & bit_mask) {
    // Shuffle to place the correct source byte for each of 32 lanes
    vbits = _mm256_shuffle_epi8(vbits, byte_sel);
    // Isolate one bit per lane
    __m256i is_set = _mm256_cmpeq_epi8(
        _mm256_and_si256(vbits, bit_mask), bit_mask);
    // is_set = 0xFF where bit=1, 0x00 where bit=0
    // We want: bit=1 → -1, bit=0 → +1
    // -1 = 0xFF in int8, so: result = is_set | 1 gives -1 or +1
    return _mm256_or_si256(is_set, _mm256_set1_epi8(1));
}

inline __m256i make_bits_broadcast(uint32_t packed_bits) {
    return _mm256_set1_epi32(static_cast<int32_t>(packed_bits));
}

// Pre-computed constants for bits_to_sign_bytes
static const __m256i k_byte_sel = _mm256_setr_epi8(
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
static const __m256i k_bit_mask = _mm256_setr_epi8(
    1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128,
    1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128);

inline int32_t hsum_epi32(__m256i value) {
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1)
    );
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

// --- Optimized int8 × sign dot product for 32 elements ---
// Uses _mm256_sign_epi8: sign_epi8(a, b) = a * sign(b)
// Then sums via maddubs + madd pattern (fastest on x86)
inline int32_t dot_int8xsign_word32_avx2(const int8_t * input, uint32_t packed_bits) {
    const __m256i x_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
    const __m256i signs = bits_to_sign_bytes(
        make_bits_broadcast(packed_bits), k_byte_sel, k_bit_mask);

    // Apply sign: effectively x * sign(weight)
    const __m256i signed_x = _mm256_sign_epi8(x_bytes, signs);

    // Sum all 32 int8 values: widen to int16 pairs, then to int32
    // maddubs treats first arg as unsigned, so we use abs values + manual sign
    // Instead, use the madd_epi16 with ones trick:
    // First widen int8→int16 by adding pairs, then int16→int32
    const __m256i ones = _mm256_set1_epi8(1);
    // maddubs(ones_unsigned, signed_x) = sum of pairs as int16
    const __m256i sum16 = _mm256_maddubs_epi16(
        _mm256_abs_epi8(signed_x),
        _mm256_sign_epi8(ones, signed_x));  // keep the sign
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i sum32 = _mm256_madd_epi16(sum16, ones16);
    return hsum_epi32(sum32);
}

// --- 4-word unrolled: process 128 elements at once ---
inline int32_t dot_int8xsign_4words_avx2(
    const int8_t * input,
    const int32_t * packed_words) {
    __m256i acc32 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);

    for (int w = 0; w < 4; ++w) {
        const __m256i x_bytes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(input + w * 32));
        const __m256i signs = bits_to_sign_bytes(
            make_bits_broadcast(static_cast<uint32_t>(packed_words[w])),
            k_byte_sel, k_bit_mask);
        const __m256i signed_x = _mm256_sign_epi8(x_bytes, signs);

        const __m256i sum16 = _mm256_maddubs_epi16(
            _mm256_abs_epi8(signed_x),
            _mm256_sign_epi8(ones8, signed_x));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(sum16, ones16));
    }

    return hsum_epi32(acc32);
}

inline int32_t dot_int8xsign_row_avx2(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int32_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    // Process 4 words (128 elements) at a time
    int64_t word_idx = 0;
    const int64_t full_4words = (full_words / 4) * 4;
    for (; word_idx < full_4words; word_idx += 4) {
        acc += dot_int8xsign_4words_avx2(
            input_row + word_idx * 32,
            packed_weight_row + word_idx);
    }

    // Remaining full words (1-3)
    for (; word_idx < full_words; ++word_idx) {
        acc += dot_int8xsign_word32_avx2(
            input_row + word_idx * 32,
            static_cast<uint32_t>(packed_weight_row[word_idx])
        );
    }

    if (tail_cols > 0) {
        acc += dot_int8xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// --- int32 × sign dot (for stage 2) ---
inline int64_t dot_int32xsign_word8_avx2(const int32_t * input, const __m256i & signs_i32) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
    const __m256i prod = _mm256_mullo_epi32(x, signs_i32);
    return static_cast<int64_t>(hsum_epi32(prod));
}

inline int64_t dot_int32xsign_row_avx2(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int64_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        const uint32_t bits = static_cast<uint32_t>(packed_weight_row[word_idx]);
        const __m256i vbits = make_bits_broadcast(bits);
        const __m256i signs = bits_to_sign_bytes(vbits, k_byte_sel, k_bit_mask);

        // Process 8 int32 values at a time (need to widen sign bytes to int32)
        const int64_t base = word_idx * 32;
        for (int sub = 0; sub < 4; ++sub) {
            // Extract 8 sign bytes, widen to int32
            __m128i sign_slice;
            if (sub == 0) sign_slice = _mm256_castsi256_si128(signs);
            else if (sub == 1) sign_slice = _mm_srli_si128(_mm256_castsi256_si128(signs), 8);
            else if (sub == 2) sign_slice = _mm256_extracti128_si256(signs, 1);
            else sign_slice = _mm_srli_si128(_mm256_extracti128_si256(signs, 1), 8);

            // Only take lower 8 bytes → widen to 8 int32
            const __m256i signs_i32 = _mm256_cvtepi8_epi32(sign_slice);
            const __m256i x = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(input_row + base + sub * 8));
            const __m256i prod = _mm256_mullo_epi32(x, signs_i32);
            acc += static_cast<int64_t>(hsum_epi32(prod));
        }
    }

    if (tail_cols > 0) {
        acc += dot_int32xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// --- Quantization helpers (AVX2) ---

inline float reduce_max_lanes(__m256 max_vec) {
    alignas(32) float lanes[8];
    _mm256_storeu_ps(lanes, max_vec);
    float max_abs = 0.0f;
    for (float lane : lanes) {
        max_abs = std::max(max_abs, lane);
    }
    return max_abs;
}

inline float max_abs_row_avx2(const float * input_row, int64_t cols) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 max_vec = _mm256_setzero_ps();
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 value = _mm256_loadu_ps(input_row + col);
        max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(value, abs_mask));
    }

    float max_abs = reduce_max_lanes(max_vec);
    for (; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col]));
    }
    return max_abs;
}

inline float max_abs_mul_row_avx2(const float * input_row, const float * mul_row, int64_t cols) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 max_vec = _mm256_setzero_ps();
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 mul_vec = _mm256_loadu_ps(mul_row + col);
        const __m256 value = _mm256_mul_ps(input_vec, mul_vec);
        max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(value, abs_mask));
    }

    float max_abs = reduce_max_lanes(max_vec);
    for (; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col] * mul_row[col]));
    }
    return max_abs;
}

inline void store_quantized_8(__m256i ints, int8_t * quantized_row) {
    const __m128i ints_lo = _mm256_castsi256_si128(ints);
    const __m128i ints_hi = _mm256_extracti128_si256(ints, 1);
    const __m128i packed_16 = _mm_packs_epi32(ints_lo, ints_hi);
    const __m128i packed_8 = _mm_packs_epi16(packed_16, packed_16);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(quantized_row), packed_8);
}

inline void quantize_row_avx2(
    const float * input_row,
    int64_t cols,
    float scale,
    int8_t * quantized_row) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(-128.0f);
    const __m256 max_vec = _mm256_set1_ps(127.0f);

    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 scaled = _mm256_mul_ps(input_vec, scale_vec);
        const __m256 clamped = _mm256_max_ps(min_vec, _mm256_min_ps(max_vec, scaled));
        const __m256i rounded = _mm256_cvtps_epi32(clamped);
        store_quantized_8(rounded, quantized_row + col);
    }

    for (; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
}

inline void quantize_mul_row_avx2(
    const float * input_row,
    const float * mul_row,
    int64_t cols,
    float scale,
    int8_t * quantized_row) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(-128.0f);
    const __m256 max_vec = _mm256_set1_ps(127.0f);

    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 mul_vec = _mm256_loadu_ps(mul_row + col);
        const __m256 product = _mm256_mul_ps(input_vec, mul_vec);
        const __m256 scaled = _mm256_mul_ps(product, scale_vec);
        const __m256 clamped = _mm256_max_ps(min_vec, _mm256_min_ps(max_vec, scaled));
        const __m256i rounded = _mm256_cvtps_epi32(clamped);
        store_quantized_8(rounded, quantized_row + col);
    }

    for (; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * mul_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
}
#endif  // __AVX2__

// ============================================================================
// Dispatch wrappers
// ============================================================================

inline float quantize_mul_row_to_int8(
    const float * input_row,
    const float * mul_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
#ifdef __AVX2__
    max_abs = max_abs_mul_row_avx2(input_row, mul_row, cols);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float value = input_row[col] * mul_row[col];
        max_abs = std::max(max_abs, std::abs(value));
    }
#endif

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
#ifdef __AVX2__
    quantize_mul_row_avx2(input_row, mul_row, cols, scale, quantized_row);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * mul_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
#endif
    return scale;
}

inline float quantize_row_to_int8(
    const float * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
#ifdef __AVX2__
    max_abs = max_abs_row_avx2(input_row, cols);
#else
    for (int64_t col = 0; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col]));
    }
#endif

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
#ifdef __AVX2__
    quantize_row_avx2(input_row, cols, scale, quantized_row);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
#endif
    return scale;
}

template <typename scalar_t>
inline float quantize_input_row_to_int8(
    const scalar_t * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        const float value = static_cast<float>(input_row[col]);
        max_abs = std::max(max_abs, std::abs(value));
    }

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(static_cast<float>(input_row[col]) * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
    return scale;
}

template <>
inline float quantize_input_row_to_int8<float>(
    const float * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    return quantize_row_to_int8(input_row, cols, quantized_row);
}

inline int32_t dot_int8xsign_row(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
#ifdef __AVX2__
    return dot_int8xsign_row_avx2(input_row, packed_weight_row, n_cols);
#else
    return dot_int8xsign_row_scalar(input_row, packed_weight_row, n_cols);
#endif
}

inline int64_t dot_int32xsign_row(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
#ifdef __AVX2__
    return dot_int32xsign_row_avx2(input_row, packed_weight_row, n_cols);
#else
    return dot_int32xsign_row_scalar(input_row, packed_weight_row, n_cols);
#endif
}

template <typename Fn>
inline void parallel_for_maybe_serial(int64_t begin, int64_t end, int64_t serial_limit, Fn && fn) {
    if (end <= begin) {
        return;
    }
    if ((end - begin) <= serial_limit || at::get_num_threads() == 1) {
        fn(begin, end);
        return;
    }
    at::parallel_for(begin, end, 0, std::forward<Fn>(fn));
}

}  // namespace

void check_cpu_tensor(const at::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_float32_tensor(const at::Tensor & tensor, const char * name) {
    check_cpu_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
}

std::tuple<at::Tensor, at::Tensor> quantize_per_row_int8_cpu(const at::Tensor & input) {
    check_cpu_tensor(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(input.is_floating_point(), "input must be floating point");

    const auto rows = input.size(0);
    const auto cols = input.size(1);

    auto q = at::empty({rows, cols}, input.options().dtype(at::kChar));
    auto scale = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "littlebit_cpu_quantize_per_row_int8",
        [&] {
            const auto * input_ptr = input.data_ptr<scalar_t>();
            auto * q_ptr = q.data_ptr<int8_t>();
            auto * scale_ptr = scale.data_ptr<float>();

            at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    const auto * row_ptr = input_ptr + row * cols;
                    auto * q_row_ptr = q_ptr + row * cols;
                    scale_ptr[row] = quantize_input_row_to_int8(row_ptr, cols, q_row_ptr);
                }
            });
        });

    return std::make_tuple(q, scale);
}

at::Tensor gemv_int8xsign_cpu(
    const at::Tensor & input,
    const at::Tensor & packed_weight,
    int64_t n_cols) {
    check_cpu_tensor(input, "input");
    check_cpu_tensor(packed_weight, "packed_weight");

    TORCH_CHECK(input.scalar_type() == at::kChar, "input must be int8");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must have shape [N, ceil(K/32)]");
    TORCH_CHECK(input.size(1) == n_cols, "input K must match n_cols");

    const int64_t expected_words = (n_cols + 31) / 32;
    TORCH_CHECK(
        packed_weight.size(1) == expected_words,
        "packed_weight second dimension must equal ceil(n_cols / 32)"
    );

    const auto m = input.size(0);
    const auto n = packed_weight.size(0);
    const auto words_per_row = packed_weight.size(1);

    auto output = at::zeros({m, n}, input.options().dtype(at::kInt));

    const auto * input_ptr = input.data_ptr<int8_t>();
    const auto * weight_ptr = packed_weight.data_ptr<int32_t>();
    auto * out_ptr = output.data_ptr<int32_t>();

    at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t sample = index / n;
            const int64_t out_row = index % n;

            const auto * input_row = input_ptr + sample * n_cols;
            const auto * weight_row = weight_ptr + out_row * words_per_row;

            out_ptr[sample * n + out_row] = dot_int8xsign_row(
                input_row, weight_row, n_cols);
        }
    });

    return output;
}

at::Tensor littlebit_linear_cpu(
    const at::Tensor & input,
    const at::Tensor & v2,
    const at::Tensor & v_sign,
    int64_t v_cols,
    const at::Tensor & mid,
    const at::Tensor & u_sign,
    int64_t u_cols,
    const at::Tensor & u1) {
    check_float32_tensor(input, "input");
    check_float32_tensor(v2, "v2");
    check_cpu_tensor(v_sign, "v_sign");
    check_float32_tensor(mid, "mid");
    check_cpu_tensor(u_sign, "u_sign");
    check_float32_tensor(u1, "u1");

    TORCH_CHECK(v_sign.scalar_type() == at::kInt, "v_sign must be int32");
    TORCH_CHECK(u_sign.scalar_type() == at::kInt, "u_sign must be int32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(v2.dim() == 2 && v2.size(0) == 1, "v2 must have shape [1, K]");
    TORCH_CHECK(mid.dim() == 2 && mid.size(0) == 1, "mid must have shape [1, R]");
    TORCH_CHECK(u1.dim() == 2 && u1.size(0) == 1, "u1 must have shape [1, N]");
    TORCH_CHECK(input.size(1) == v_cols, "input K must match v_cols");
    TORCH_CHECK(v2.size(1) == v_cols, "v2 second dimension must match v_cols");
    TORCH_CHECK(v_sign.dim() == 2, "v_sign must have shape [R, ceil(K/32)]");
    TORCH_CHECK(u_sign.dim() == 2, "u_sign must have shape [N, ceil(R/32)]");
    TORCH_CHECK(v_sign.size(0) == mid.size(1), "mid rank must match v_sign rows");
    TORCH_CHECK(u_sign.size(0) == u1.size(1), "u1 out_features must match u_sign rows");
    TORCH_CHECK(mid.size(1) == u_cols, "mid rank must match u_cols");
    TORCH_CHECK(
        v_sign.size(1) == (v_cols + 31) / 32,
        "v_sign second dimension must equal ceil(v_cols / 32)"
    );
    TORCH_CHECK(
        u_sign.size(1) == (u_cols + 31) / 32,
        "u_sign second dimension must equal ceil(u_cols / 32)"
    );

    const auto rows = input.size(0);
    const auto rank = v_sign.size(0);
    const auto out_features = u_sign.size(0);
    const auto v_words = v_sign.size(1);
    const auto u_words = u_sign.size(1);

    auto output = at::empty({rows, out_features}, input.options().dtype(at::kFloat));

    const auto * input_ptr = input.data_ptr<float>();
    const auto * v2_ptr = v2.data_ptr<float>();
    const auto * v_sign_ptr = v_sign.data_ptr<int32_t>();
    const auto * mid_ptr = mid.data_ptr<float>();
    const auto * u_sign_ptr = u_sign.data_ptr<int32_t>();
    const auto * u1_ptr = u1.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

    struct LittleBitScratch {
        std::vector<int8_t> q1;
        std::vector<float> stage1;
        std::vector<int8_t> q2;
    };
    thread_local LittleBitScratch scratch;
    scratch.q1.resize(v_cols);
    scratch.stage1.resize(rank);
    scratch.q2.resize(rank);

    auto * q1_ptr = scratch.q1.data();
    auto * stage1_ptr = scratch.stage1.data();
    auto * q2_ptr = scratch.q2.data();
    for (int64_t row = 0; row < rows; ++row) {
        const auto * input_row = input_ptr + row * v_cols;
        auto * out_row = out_ptr + row * out_features;

        const float scale1 = quantize_mul_row_to_int8(input_row, v2_ptr, v_cols, q1_ptr);
        const float inv_scale1 = 1.0f / scale1;

        // Stage 1: V projection — rank is typically small (32), run serial
        parallel_for_maybe_serial(0, rank, 256, [&](int64_t begin, int64_t end) {
            for (int64_t rank_idx = begin; rank_idx < end; ++rank_idx) {
                const int32_t acc = dot_int8xsign_row(
                    q1_ptr,
                    v_sign_ptr + rank_idx * v_words,
                    v_cols
                );
                stage1_ptr[rank_idx] = static_cast<float>(acc) * inv_scale1 * mid_ptr[rank_idx];
            }
        });

        const float scale2 = quantize_row_to_int8(stage1_ptr, rank, q2_ptr);
        const float inv_scale2 = 1.0f / scale2;

        // Stage 2: U projection — out_features=4096, parallelize aggressively
        parallel_for_maybe_serial(0, out_features, 256, [&](int64_t begin, int64_t end) {
            for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
                const int32_t acc = dot_int8xsign_row(
                    q2_ptr,
                    u_sign_ptr + out_idx * u_words,
                    u_cols
                );
                out_row[out_idx] = static_cast<float>(acc) * inv_scale2 * u1_ptr[out_idx];
            }
        });
    }

    return output;
}

// ============================================================================
// BitNet-style i2 (2-bit packed binary) kernel
// ============================================================================

// Repack int32-packed sign bits to byte-packed 2-bit format (BitNet format)
// Input: int32[ceil(K/32)] where each bit = 1 sign (bit=1 → -1, bit=0 → +1)
// Output: uint8[ceil(K/4)] where each byte holds 4 × 2-bit values:
//   byte = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3
//   with encoding: +1 → 2, -1 → 0
at::Tensor repack_signs_to_i2_cpu(
    const at::Tensor & packed_signs,
    int64_t n_cols) {
    check_cpu_tensor(packed_signs, "packed_signs");
    TORCH_CHECK(packed_signs.scalar_type() == at::kInt, "packed_signs must be int32");

    // Each row: ceil(K/32) int32 words → K signs → ceil(K/4) bytes in i2 format
    const auto n_rows = packed_signs.size(0);
    const int64_t i2_bytes_per_row = (n_cols + 3) / 4;
    const int64_t words_per_row = packed_signs.size(1);

    auto output = at::zeros({n_rows, i2_bytes_per_row}, packed_signs.options().dtype(at::kByte));

    const auto * src_ptr = packed_signs.data_ptr<int32_t>();
    auto * dst_ptr = output.data_ptr<uint8_t>();

    at::parallel_for(0, n_rows, 0, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const int32_t * src_row = src_ptr + row * words_per_row;
            uint8_t * dst_row = dst_ptr + row * i2_bytes_per_row;

            for (int64_t col = 0; col < n_cols; ++col) {
                const int64_t word_idx = col / 32;
                const int64_t bit_idx = col % 32;
                const bool is_neg = ((static_cast<uint32_t>(src_row[word_idx]) >> bit_idx) & 1U) != 0;

                // Encoding: +1 → 2, -1 → 0
                const uint8_t i2_val = is_neg ? 0 : 2;

                // Pack 4 i2 values per byte: groups of 32 elements like BitNet
                // BitNet layout: within each block of 128 elements,
                //   positions [0..31] go to shift 6 (group 0)
                //   positions [32..63] go to shift 4 (group 1)
                //   positions [64..95] go to shift 2 (group 2)
                //   positions [96..127] go to shift 0 (group 3)
                // Byte index within block = position % 32
                // Block index = position / 128
                const int64_t block_idx = col / 128;
                const int64_t pos_in_block = col % 128;
                const int64_t group_idx = pos_in_block / 32;  // 0-3
                const int64_t group_pos = pos_in_block % 32;  // 0-31
                const int64_t byte_offset = block_idx * 32 + group_pos;

                if (byte_offset < i2_bytes_per_row) {
                    const int shift = 6 - 2 * static_cast<int>(group_idx);
                    dst_row[byte_offset] |= (i2_val << shift);
                }
            }
        }
    });

    return output;
}

#ifdef __AVX2__
// BitNet MAD-style dot product: int8 input × 2-bit packed binary weights
// Returns raw maddubs accumulation (needs sum_all correction afterwards)
inline int32_t dot_int8_x_i2_block_avx2(
    const int8_t * input,   // 128 int8 values
    const uint8_t * weight  // 32 bytes = 128 × 2-bit values
) {
    const __m256i mask = _mm256_set1_epi8(0x03);

    // Load 32 bytes of packed weights → 128 2-bit values
    __m256i xq8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weight));
    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

    xq8_3 = _mm256_and_si256(xq8_3, mask);
    xq8_2 = _mm256_and_si256(xq8_2, mask);
    xq8_1 = _mm256_and_si256(xq8_1, mask);
    xq8_0 = _mm256_and_si256(xq8_0, mask);

    // Load 4 × 32 = 128 int8 input values
    __m256i yq8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
    __m256i yq8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 32));
    __m256i yq8_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 64));
    __m256i yq8_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + 96));

    // maddubs: unsigned weight × signed input → int16 pair sums
    xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
    xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
    xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
    xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

    // Sum all into int16
    __m256i accu16 = _mm256_add_epi16(_mm256_add_epi16(xq8_0, xq8_1),
                                       _mm256_add_epi16(xq8_2, xq8_3));

    // Widen int16 → int32
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i accu32 = _mm256_madd_epi16(accu16, ones16);

    return hsum_epi32(accu32);
}

// Sum all int8 values in a row (precomputed once per quantization)
inline int32_t sum_int8_row_avx2(const int8_t * data, int64_t n) {
    int32_t sum = 0;
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc32 = _mm256_setzero_si256();

    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));
        // Sum pairs: maddubs(ones_unsigned, signed_v) = pairwise sums as int16
        const __m256i sum16 = _mm256_maddubs_epi16(
            _mm256_abs_epi8(v), _mm256_sign_epi8(ones8, v));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(sum16, ones16));
    }
    sum = hsum_epi32(acc32);
    for (; i < n; ++i) sum += static_cast<int32_t>(data[i]);
    return sum;
}

// Full row dot product: int8 × i2 packed weights with sum_all correction
// true_dot = maddubs_result - sum_all(input)
// Because encoding is +1→2, -1→0:
//   maddubs gives 2*sum(input where sign=+1)
//   true_dot = 2*sum(input_pos) - sum_all
//            = maddubs_result - sum_all
inline int32_t dot_int8_x_i2_row_avx2(
    const int8_t * input_row,
    const uint8_t * weight_i2_row,
    int64_t n_cols,
    int32_t sum_all) {
    const int64_t n_blocks = n_cols / 128;
    const int64_t tail = n_cols % 128;

    int32_t maddubs_sum = 0;
    for (int64_t b = 0; b < n_blocks; ++b) {
        maddubs_sum += dot_int8_x_i2_block_avx2(
            input_row + b * 128,
            weight_i2_row + b * 32);
    }

    // Tail: scalar fallback for remaining elements
    if (tail > 0) {
        const int8_t * input_tail = input_row + n_blocks * 128;
        const uint8_t * weight_tail = weight_i2_row + n_blocks * 32;
        for (int64_t c = 0; c < tail; ++c) {
            const int64_t group_idx = c / 32;
            const int64_t group_pos = c % 32;
            if (group_pos < 32 && (n_blocks * 32 + group_pos) < ((n_cols + 3) / 4)) {
                const int shift = 6 - 2 * static_cast<int>(group_idx);
                const uint8_t i2_val = (weight_tail[group_pos] >> shift) & 0x03;
                // i2_val is 0 or 2: weight = i2_val - 1 gives -1 or +1
                const int32_t sign_val = static_cast<int32_t>(i2_val) - 1;
                maddubs_sum += static_cast<int32_t>(input_tail[c]) * (sign_val + 1);
                // Actually simpler: maddubs would give i2_val * input
                // We accumulate the same way
            }
        }
    }

    return maddubs_sum - sum_all;
}
#endif  // __AVX2__

at::Tensor littlebit_linear_i2_cpu(
    const at::Tensor & input,
    const at::Tensor & v2,
    const at::Tensor & v_sign_i2,
    int64_t v_cols,
    const at::Tensor & mid,
    const at::Tensor & u_sign_i2,
    int64_t u_cols,
    const at::Tensor & u1) {
    check_float32_tensor(input, "input");
    check_float32_tensor(v2, "v2");
    check_cpu_tensor(v_sign_i2, "v_sign_i2");
    check_float32_tensor(mid, "mid");
    check_cpu_tensor(u_sign_i2, "u_sign_i2");
    check_float32_tensor(u1, "u1");

    TORCH_CHECK(v_sign_i2.scalar_type() == at::kByte, "v_sign_i2 must be uint8");
    TORCH_CHECK(u_sign_i2.scalar_type() == at::kByte, "u_sign_i2 must be uint8");

    const auto rows = input.size(0);
    const auto rank = v_sign_i2.size(0);
    const auto out_features = u_sign_i2.size(0);

    auto output = at::empty({rows, out_features}, input.options().dtype(at::kFloat));

    const auto * input_ptr = input.data_ptr<float>();
    const auto * v2_ptr = v2.data_ptr<float>();
    const auto * v_i2_ptr = v_sign_i2.data_ptr<uint8_t>();
    const auto * mid_ptr = mid.data_ptr<float>();
    const auto * u_i2_ptr = u_sign_i2.data_ptr<uint8_t>();
    const auto * u1_ptr = u1.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

    const auto v_i2_stride = v_sign_i2.size(1);
    const auto u_i2_stride = u_sign_i2.size(1);

    struct I2Scratch {
        std::vector<int8_t> q1;
        std::vector<float> stage1;
        std::vector<int8_t> q2;
    };
    thread_local I2Scratch scratch;
    scratch.q1.resize(v_cols);
    scratch.stage1.resize(rank);
    scratch.q2.resize(rank);

    for (int64_t row = 0; row < rows; ++row) {
        const auto * input_row = input_ptr + row * v_cols;

        // Quantize input × v2 → int8
        const float scale1 = quantize_mul_row_to_int8(input_row, v2_ptr, v_cols, scratch.q1.data());
        const float inv_scale1 = 1.0f / scale1;

#ifdef __AVX2__
        // Precompute sum_all for the quantized input (for i2 correction)
        const int32_t sum_all_q1 = sum_int8_row_avx2(scratch.q1.data(), v_cols);

        // Stage 1: V projection using i2 BitNet kernel
        parallel_for_maybe_serial(0, rank, 256, [&](int64_t begin, int64_t end) {
            for (int64_t rank_idx = begin; rank_idx < end; ++rank_idx) {
                const int32_t acc = dot_int8_x_i2_row_avx2(
                    scratch.q1.data(),
                    v_i2_ptr + rank_idx * v_i2_stride,
                    v_cols,
                    sum_all_q1);
                scratch.stage1.data()[rank_idx] = static_cast<float>(acc) * inv_scale1 * mid_ptr[rank_idx];
            }
        });

        // Quantize stage1 → int8
        const float scale2 = quantize_row_to_int8(scratch.stage1.data(), rank, scratch.q2.data());
        const float inv_scale2 = 1.0f / scale2;

        const int32_t sum_all_q2 = sum_int8_row_avx2(scratch.q2.data(), rank);

        // Stage 2: U projection using i2 BitNet kernel
        parallel_for_maybe_serial(0, out_features, 256, [&](int64_t begin, int64_t end) {
            for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
                const int32_t acc = dot_int8_x_i2_row_avx2(
                    scratch.q2.data(),
                    u_i2_ptr + out_idx * u_i2_stride,
                    rank,
                    sum_all_q2);
                out_ptr[row * out_features + out_idx] = static_cast<float>(acc) * inv_scale2 * u1_ptr[out_idx];
            }
        });
#else
        // Scalar fallback — same as original littlebit_linear_cpu
        // (omitted for brevity, would use existing scalar path)
        TORCH_CHECK(false, "littlebit_linear_i2 requires AVX2");
#endif
    }

    return output;
}

// ============================================================================
// New ops: embedding, rms_norm, dense_gemv, silu_mul
// ============================================================================

at::Tensor embedding_lookup_cpu(
    const at::Tensor & weight,
    const at::Tensor & indices) {
    // weight: [vocab, hidden], indices: [N] or [1, N] -> output: [N, hidden]
    check_float32_tensor(weight, "weight");
    check_cpu_tensor(indices, "indices");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [vocab, hidden]");

    auto flat_indices = indices.reshape({-1});
    const auto n_tokens = flat_indices.size(0);
    const auto hidden = weight.size(1);

    auto output = at::empty({n_tokens, hidden}, weight.options());
    const auto * weight_ptr = weight.data_ptr<float>();
    const auto * idx_ptr = flat_indices.data_ptr<int64_t>();
    auto * out_ptr = output.data_ptr<float>();

    for (int64_t i = 0; i < n_tokens; ++i) {
        const int64_t idx = idx_ptr[i];
        std::memcpy(out_ptr + i * hidden, weight_ptr + idx * hidden,
                    hidden * sizeof(float));
    }
    return output;
}

at::Tensor rms_norm_cpu(
    const at::Tensor & input,
    const at::Tensor & weight,
    double eps) {
    check_float32_tensor(input, "input");
    check_float32_tensor(weight, "weight");
    // input: [1, hidden] or [hidden], weight: [hidden]
    auto x = input.reshape({-1}).contiguous();
    auto w = weight.reshape({-1}).contiguous();
    const auto cols = x.size(0);
    TORCH_CHECK(w.size(0) == cols, "weight size must match input hidden dim");

    auto output = at::empty({1, cols}, x.options());
    const auto * x_ptr = x.data_ptr<float>();
    const auto * w_ptr = w.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

    // Compute sum of squares
    float sum_sq = 0.0f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 v = _mm256_loadu_ps(x_ptr + col);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int i = 0; i < 8; ++i) sum_sq += tmp[i];
    for (; col < cols; ++col) sum_sq += x_ptr[col] * x_ptr[col];
#else
    for (int64_t col = 0; col < cols; ++col) sum_sq += x_ptr[col] * x_ptr[col];
#endif

    const float rms_scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + static_cast<float>(eps));

    // Apply: out = x * rms_scale * weight
#ifdef __AVX2__
    const __m256 scale_vec = _mm256_set1_ps(rms_scale);
    col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 xv = _mm256_loadu_ps(x_ptr + col);
        const __m256 wv = _mm256_loadu_ps(w_ptr + col);
        _mm256_storeu_ps(out_ptr + col, _mm256_mul_ps(_mm256_mul_ps(xv, scale_vec), wv));
    }
    for (; col < cols; ++col) {
        out_ptr[col] = x_ptr[col] * rms_scale * w_ptr[col];
    }
#else
    for (int64_t col = 0; col < cols; ++col) {
        out_ptr[col] = x_ptr[col] * rms_scale * w_ptr[col];
    }
#endif
    return output;
}

at::Tensor dense_gemv_f32_cpu(
    const at::Tensor & input,
    const at::Tensor & weight) {
    // input: [1, K], weight: [N, K] -> output: [1, N]
    // Optimized GEMV for lm_head: parallel over output rows, AVX2 FMA inner loop
    check_float32_tensor(input, "input");
    check_float32_tensor(weight, "weight");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(input.size(1) == weight.size(1), "input K must match weight K");

    const auto K = weight.size(1);
    const auto N = weight.size(0);
    auto output = at::empty({1, N}, input.options());

    const auto * in_ptr = input.data_ptr<float>();
    const auto * w_ptr = weight.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

    // Parallel over output rows (N ≈ 128K for lm_head)
    at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const auto * w_row = w_ptr + n * K;
            float dot = 0.0f;
#ifdef __AVX2__
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int64_t k = 0;
            // 4-way unrolled: 32 floats per iteration
            for (; k + 32 <= K; k += 32) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + k + 8),
                                       _mm256_loadu_ps(w_row + k + 8), acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + k + 16),
                                       _mm256_loadu_ps(w_row + k + 16), acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + k + 24),
                                       _mm256_loadu_ps(w_row + k + 24), acc3);
            }
            // Remaining 8-wide
            for (; k + 8 <= K; k += 8) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + k),
                                       _mm256_loadu_ps(w_row + k), acc0);
            }
            // Reduce 4 accumulators
            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            alignas(32) float lanes[8];
            _mm256_storeu_ps(lanes, acc0);
            for (int i = 0; i < 8; ++i) dot += lanes[i];
            // Scalar tail
            for (; k < K; ++k) dot += in_ptr[k] * w_row[k];
#else
            for (int64_t k = 0; k < K; ++k) dot += in_ptr[k] * w_row[k];
#endif
            out_ptr[n] = dot;
        }
    });

    return output;
}

// Fast exp approximation (Schraudolph's method) for AVX2
// exp(x) ≈ reinterpret(int(x * 2^23/ln2 + 127*2^23))
// Accurate to ~0.1% for |x| < 88
#ifdef __AVX2__
static const __m256 k_exp_a = _mm256_set1_ps(12102203.0f);  // 2^23 / ln(2)
static const __m256 k_exp_b = _mm256_set1_ps(1065353216.0f); // 127 * 2^23
static const __m256 k_exp_clamp_lo = _mm256_set1_ps(-88.0f);
static const __m256 k_exp_clamp_hi = _mm256_set1_ps(88.0f);

inline __m256 fast_exp_avx2(__m256 x) {
    x = _mm256_max_ps(k_exp_clamp_lo, _mm256_min_ps(k_exp_clamp_hi, x));
    __m256i fi = _mm256_cvtps_epi32(_mm256_add_ps(k_exp_b, _mm256_mul_ps(k_exp_a, x)));
    return _mm256_castsi256_ps(fi);
}
#endif

at::Tensor silu_mul_cpu(
    const at::Tensor & gate,
    const at::Tensor & up) {
    // Fused: SiLU(gate) * up = gate * sigmoid(gate) * up
    check_float32_tensor(gate, "gate");
    check_float32_tensor(up, "up");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");

    auto output = at::empty_like(gate);
    const auto numel = gate.numel();
    const auto * g_ptr = gate.data_ptr<float>();
    const auto * u_ptr = up.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

#ifdef __AVX2__
    const __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= numel; i += 8) {
        const __m256 g = _mm256_loadu_ps(g_ptr + i);
        const __m256 u = _mm256_loadu_ps(u_ptr + i);
        // sigmoid(g) = 1 / (1 + exp(-g))  using fast_exp
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 exp_neg = fast_exp_avx2(neg_g);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        _mm256_storeu_ps(out_ptr + i, _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u));
    }
    for (; i < numel; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-g_ptr[i]));
        out_ptr[i] = g_ptr[i] * sig * u_ptr[i];
    }
#else
    for (int64_t i = 0; i < numel; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-g_ptr[i]));
        out_ptr[i] = g_ptr[i] * sig * u_ptr[i];
    }
#endif
    return output;
}

at::Tensor fused_attention_cpu(
    const at::Tensor & q_flat,
    const at::Tensor & k_cache,
    const at::Tensor & v_cache,
    const at::Tensor & k_new_flat,
    const at::Tensor & v_new_flat,
    int64_t position,
    int64_t num_kv_heads,
    int64_t kv_repeat,
    int64_t head_dim,
    double attn_scale) {
    // q_flat: [1, num_heads * head_dim] where num_heads = num_kv_heads * kv_repeat
    // k_cache: [num_kv_heads, max_seq_len, head_dim]
    // v_cache: [num_kv_heads, max_seq_len, head_dim]
    // k_new_flat: [1, num_kv_heads * head_dim]
    // v_new_flat: [1, num_kv_heads * head_dim]
    // Returns: [1, num_heads * head_dim]

    check_float32_tensor(q_flat, "q_flat");
    check_float32_tensor(k_cache, "k_cache");
    check_float32_tensor(v_cache, "v_cache");
    check_float32_tensor(k_new_flat, "k_new_flat");
    check_float32_tensor(v_new_flat, "v_new_flat");

    const int64_t num_heads = num_kv_heads * kv_repeat;
    const int64_t max_seq = k_cache.size(1);
    const int64_t seq_len = position + 1;  // including current position
    const float scale = static_cast<float>(attn_scale);

    // Write new K, V to cache
    auto * k_cache_ptr = k_cache.data_ptr<float>();
    auto * v_cache_ptr = v_cache.data_ptr<float>();
    const auto * k_new_ptr = k_new_flat.data_ptr<float>();
    const auto * v_new_ptr = v_new_flat.data_ptr<float>();

    for (int64_t h = 0; h < num_kv_heads; ++h) {
        std::memcpy(
            k_cache_ptr + h * max_seq * head_dim + position * head_dim,
            k_new_ptr + h * head_dim,
            head_dim * sizeof(float));
        std::memcpy(
            v_cache_ptr + h * max_seq * head_dim + position * head_dim,
            v_new_ptr + h * head_dim,
            head_dim * sizeof(float));
    }

    // Output
    auto output = at::empty({1, num_heads * head_dim}, q_flat.options());
    auto * out_ptr = output.data_ptr<float>();
    const auto * q_ptr = q_flat.data_ptr<float>();

    // For each KV head group
    for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
        const float * k_base = k_cache_ptr + kv_h * max_seq * head_dim;
        const float * v_base = v_cache_ptr + kv_h * max_seq * head_dim;

        // Process each query head in this KV group
        for (int64_t r = 0; r < kv_repeat; ++r) {
            const int64_t q_head = kv_h * kv_repeat + r;
            const float * q_head_ptr = q_ptr + q_head * head_dim;
            float * out_head_ptr = out_ptr + q_head * head_dim;

            // Compute attention scores: Q @ K^T for each position
            // scores[t] = sum_d(q[d] * k[t][d]) * scale
            thread_local std::vector<float> scores_buf;
            scores_buf.resize(seq_len);
            float * scores = scores_buf.data();

            for (int64_t t = 0; t < seq_len; ++t) {
                const float * k_t = k_base + t * head_dim;
                float dot = 0.0f;
#ifdef __AVX2__
                __m256 acc = _mm256_setzero_ps();
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(q_head_ptr + d),
                        _mm256_loadu_ps(k_t + d), acc);
                }
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                for (int i = 0; i < 8; ++i) dot += tmp[i];
                for (; d < head_dim; ++d) dot += q_head_ptr[d] * k_t[d];
#else
                for (int64_t d = 0; d < head_dim; ++d)
                    dot += q_head_ptr[d] * k_t[d];
#endif
                scores[t] = dot * scale;
            }

            // Softmax
            float max_score = scores[0];
            for (int64_t t = 1; t < seq_len; ++t)
                max_score = std::max(max_score, scores[t]);

            float sum_exp = 0.0f;
            for (int64_t t = 0; t < seq_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int64_t t = 0; t < seq_len; ++t)
                scores[t] *= inv_sum;

            // Context: probs @ V
            // out[d] = sum_t(probs[t] * v[t][d])
            std::memset(out_head_ptr, 0, head_dim * sizeof(float));
            for (int64_t t = 0; t < seq_len; ++t) {
                const float prob = scores[t];
                const float * v_t = v_base + t * head_dim;
#ifdef __AVX2__
                const __m256 prob_vec = _mm256_set1_ps(prob);
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 out_v = _mm256_loadu_ps(out_head_ptr + d);
                    out_v = _mm256_fmadd_ps(prob_vec, _mm256_loadu_ps(v_t + d), out_v);
                    _mm256_storeu_ps(out_head_ptr + d, out_v);
                }
                for (; d < head_dim; ++d)
                    out_head_ptr[d] += prob * v_t[d];
#else
                for (int64_t d = 0; d < head_dim; ++d)
                    out_head_ptr[d] += prob * v_t[d];
#endif
            }
        }
    }

    return output;
}

// ============================================================================
// Monolithic Full Forward Pass (all 32 layers in one C++ call)
// ============================================================================

// Per-layer tensor layout in the flat tensor list:
// [0] input_layernorm_weight  (float32 [hidden_size])
// [1] post_attention_layernorm_weight (float32 [hidden_size])
// Then 7 projections (q, k, v, o, gate, up, down), each with 5 tensors:
// [2+p*5+0] v2         (float32 [1, v_cols])
// [2+p*5+1] v_sign_i2  (uint8 [rank, i2_bytes_per_row])
// [2+p*5+2] mid        (float32 [1, rank])
// [2+p*5+3] u_sign_i2  (uint8 [out_features, i2_bytes_per_row])
// [2+p*5+4] u1         (float32 [1, out_features])
// Total per layer: 2 + 7*5 = 37 tensors
// Plus v_cols and u_cols passed as int list (7*2 = 14 per layer)

// Inline LB linear (i2 kernel) without tensor allocation
inline void lb_linear_i2_inline(
    const float * input,      // [1, v_cols]
    int64_t v_cols,
    const float * v2,         // [v_cols]
    const uint8_t * v_sign_i2,// [rank, i2_stride]
    int64_t v_i2_stride,
    const float * mid,        // [rank]
    int64_t rank,
    const uint8_t * u_sign_i2,// [out_features, u_i2_stride]
    int64_t u_i2_stride,
    const float * u1,         // [out_features]
    int64_t out_features,
    int64_t u_cols,           // = rank
    float * output,           // [out_features]
    int8_t * q1_buf,          // scratch [v_cols]
    float * stage1_buf,       // scratch [rank]
    int8_t * q2_buf           // scratch [rank]
) {
    // Stage 1: quantize(input * v2) × v_sign_i2 → stage1 × mid
    const float scale1 = quantize_mul_row_to_int8(input, v2, v_cols, q1_buf);
    const float inv_scale1 = 1.0f / scale1;

#ifdef __AVX2__
    const int32_t sum_all_q1 = sum_int8_row_avx2(q1_buf, v_cols);
    for (int64_t r = 0; r < rank; ++r) {
        const int32_t acc = dot_int8_x_i2_row_avx2(
            q1_buf, v_sign_i2 + r * v_i2_stride, v_cols, sum_all_q1);
        stage1_buf[r] = static_cast<float>(acc) * inv_scale1 * mid[r];
    }
#else
    TORCH_CHECK(false, "full_forward requires AVX2");
#endif

    // Stage 2: quantize(stage1) × u_sign_i2 → output × u1
    // ** OPTIMIZED: parallel_for over output features (gate/up have 14336 rows) **
    const float scale2 = quantize_row_to_int8(stage1_buf, rank, q2_buf);
    const float inv_scale2 = 1.0f / scale2;

#ifdef __AVX2__
    const int32_t sum_all_q2 = sum_int8_row_avx2(q2_buf, rank);
    at::parallel_for(0, out_features, 0, [&](int64_t begin, int64_t end) {
        for (int64_t o = begin; o < end; ++o) {
            const int32_t acc = dot_int8_x_i2_row_avx2(
                q2_buf, u_sign_i2 + o * u_i2_stride, u_cols, sum_all_q2);
            output[o] = static_cast<float>(acc) * inv_scale2 * u1[o];
        }
    });
#endif
}

// Inline RMSNorm without tensor allocation
inline void rms_norm_inline(
    const float * input, const float * weight, float eps,
    int64_t hidden_size, float * output) {
#ifdef __AVX2__
    __m256 sum_sq = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= hidden_size; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum_sq);
    float ss = 0.0f;
    for (int j = 0; j < 8; ++j) ss += tmp[j];
    for (; i < hidden_size; ++i) ss += input[i] * input[i];

    const float rsqrt = 1.0f / std::sqrt(ss / static_cast<float>(hidden_size) + eps);

    i = 0;
    const __m256 vrsqrt = _mm256_set1_ps(rsqrt);
    for (; i + 8 <= hidden_size; i += 8) {
        _mm256_storeu_ps(output + i,
            _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(input + i), vrsqrt),
                _mm256_loadu_ps(weight + i)));
    }
    for (; i < hidden_size; ++i)
        output[i] = input[i] * rsqrt * weight[i];
#else
    float ss = 0.0f;
    for (int64_t i = 0; i < hidden_size; ++i) ss += input[i] * input[i];
    const float rsqrt = 1.0f / std::sqrt(ss / static_cast<float>(hidden_size) + eps);
    for (int64_t i = 0; i < hidden_size; ++i) output[i] = input[i] * rsqrt * weight[i];
#endif
}




// Inline SiLU-mul without tensor allocation
// ** OPTIMIZED: Uses fast AVX2 exp approximation instead of scalar std::exp **
inline void silu_mul_inline(const float * gate, const float * up,
                             int64_t n, float * output) {
#ifdef __AVX2__
    const __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        // sigmoid(g) = 1 / (1 + exp(-g))  using fast_exp
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        __m256 exp_neg = fast_exp_avx2(neg_g);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        // SiLU(g) * up = g * sigmoid(g) * up
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u));
    }
    for (; i < n; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-gate[i]));
        output[i] = gate[i] * sig * up[i];
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-gate[i]));
        output[i] = gate[i] * sig * up[i];
    }
#endif
}

// Inline fused attention without tensor allocation
inline void fused_attention_inline(
    const float * q, const float * k_new, const float * v_new,
    float * k_cache, float * v_cache,
    int64_t position, int64_t num_kv_heads, int64_t kv_repeat,
    int64_t head_dim, int64_t max_seq, float attn_scale,
    float * output   // [num_heads * head_dim]
) {
    const int64_t seq_len = position + 1;

    // Write new K, V to cache
    for (int64_t h = 0; h < num_kv_heads; ++h) {
        std::memcpy(k_cache + h * max_seq * head_dim + position * head_dim,
                    k_new + h * head_dim, head_dim * sizeof(float));
        std::memcpy(v_cache + h * max_seq * head_dim + position * head_dim,
                    v_new + h * head_dim, head_dim * sizeof(float));
    }

    // ** OPTIMIZED: Parallelize attention across all query heads **
    const int64_t num_heads = num_kv_heads * kv_repeat;
    at::parallel_for(0, num_heads, 0, [&](int64_t head_begin, int64_t head_end) {
        // Each thread gets its own scores buffer
        std::vector<float> local_scores(seq_len);

        for (int64_t q_head = head_begin; q_head < head_end; ++q_head) {
            const int64_t kv_h = q_head / kv_repeat;
            const float * k_base = k_cache + kv_h * max_seq * head_dim;
            const float * v_base = v_cache + kv_h * max_seq * head_dim;
            const float * q_ptr = q + q_head * head_dim;
            float * out_ptr = output + q_head * head_dim;

            // Q × K^T
            for (int64_t t = 0; t < seq_len; ++t) {
                const float * k_t = k_base + t * head_dim;
                float dot = 0.0f;
#ifdef __AVX2__
                __m256 acc = _mm256_setzero_ps();
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_ptr + d),
                                          _mm256_loadu_ps(k_t + d), acc);
                alignas(32) float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                for (int j = 0; j < 8; ++j) dot += tmp[j];
                for (; d < head_dim; ++d) dot += q_ptr[d] * k_t[d];
#else
                for (int64_t d = 0; d < head_dim; ++d) dot += q_ptr[d] * k_t[d];
#endif
                local_scores[t] = dot * attn_scale;
            }

            // Softmax
            float max_s = local_scores[0];
            for (int64_t t = 1; t < seq_len; ++t) max_s = std::max(max_s, local_scores[t]);
            float sum_exp = 0.0f;
            for (int64_t t = 0; t < seq_len; ++t) {
                local_scores[t] = std::exp(local_scores[t] - max_s);
                sum_exp += local_scores[t];
            }
            const float inv_sum = 1.0f / sum_exp;
            for (int64_t t = 0; t < seq_len; ++t) local_scores[t] *= inv_sum;

            // Attn × V
            std::memset(out_ptr, 0, head_dim * sizeof(float));
            for (int64_t t = 0; t < seq_len; ++t) {
                const float prob = local_scores[t];
                const float * v_t = v_base + t * head_dim;
#ifdef __AVX2__
                const __m256 pv = _mm256_set1_ps(prob);
                int64_t d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 o = _mm256_loadu_ps(out_ptr + d);
                    o = _mm256_fmadd_ps(pv, _mm256_loadu_ps(v_t + d), o);
                    _mm256_storeu_ps(out_ptr + d, o);
                }
                for (; d < head_dim; ++d) out_ptr[d] += prob * v_t[d];
#else
                for (int64_t d = 0; d < head_dim; ++d) out_ptr[d] += prob * v_t[d];
#endif
            }
        }
    });
}

at::Tensor full_forward_cpu(
    int64_t token_id,
    const at::Tensor & embed_weight,     // [vocab_size, hidden_size]
    const at::Tensor & final_norm_weight, // [hidden_size]
    at::TensorList layer_tensors,         // 37 tensors per layer × num_layers
    at::TensorList kv_caches,             // 2 tensors per layer (k_cache, v_cache)
    const std::vector<int64_t> & layer_dims,  // [v_cols, rank, u_cols_q, out_q, u_cols_k, out_k, ...] per proj
    int64_t num_layers,
    int64_t hidden_size,
    int64_t num_kv_heads,
    int64_t kv_repeat,
    int64_t head_dim,
    int64_t max_seq_len,
    int64_t position,
    double attn_scale,
    double rms_eps) {

    // Embedding lookup
    const float * embed_ptr = embed_weight.data_ptr<float>();
    const float * final_norm_ptr = final_norm_weight.data_ptr<float>();

    // Pre-allocate scratch buffers (reused across all layers)
    const int64_t max_v_cols = hidden_size;  // typically 4096
    const int64_t max_rank = 256;            // safe upper bound
    const int64_t max_out = hidden_size * 4; // for gate/up: intermediate_size

    // Determine actual max dimensions from layer_dims
    // layer_dims layout: per projection (7 per layer), each has 4 ints:
    //   [v_cols, rank, u_cols, out_features]
    // Total: 7 * 4 = 28 ints per layer

    struct ForwardScratch {
        std::vector<float> buf0;       // double-buffer 0 [hidden_size]
        std::vector<float> buf1;       // double-buffer 1 [hidden_size]
        std::vector<float> normed;     // [hidden_size]
        std::vector<float> proj_out;   // max(out_features) across all projections
        std::vector<float> q_out, k_out, v_out, o_out;
        std::vector<float> gate_out, up_out, down_out;
        std::vector<float> mlp_hidden; // [intermediate_size]
        std::vector<float> attn_out;   // [num_heads * head_dim]
        std::vector<int8_t> q1_buf;
        std::vector<float> stage1_buf;
        std::vector<int8_t> q2_buf;
    };
    thread_local ForwardScratch scratch;

    // Find max dimensions from first layer
    const int64_t n_proj_dims = 28;  // 7 projections × 4 dims each
    int64_t max_v = 0, max_r = 0, max_o = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(layer_dims.size()); ++i) {
        int64_t pos_in_layer = i % n_proj_dims;
        if (pos_in_layer % 4 == 0) max_v = std::max(max_v, layer_dims[i]);      // v_cols
        else if (pos_in_layer % 4 == 1) max_r = std::max(max_r, layer_dims[i]); // rank
        else if (pos_in_layer % 4 == 3) max_o = std::max(max_o, layer_dims[i]); // out_features
    }

    // ** OPTIMIZED: double-buffer instead of separate x + residual (eliminates memcpy) **
    scratch.buf0.resize(hidden_size);
    scratch.buf1.resize(hidden_size);
    scratch.normed.resize(hidden_size);
    scratch.q_out.resize(max_o);
    scratch.k_out.resize(max_o);
    scratch.v_out.resize(max_o);
    scratch.o_out.resize(max_o);
    scratch.gate_out.resize(max_o);
    scratch.up_out.resize(max_o);
    scratch.down_out.resize(max_o);
    scratch.mlp_hidden.resize(max_o);
    scratch.attn_out.resize(hidden_size);
    scratch.q1_buf.resize(max_v);
    scratch.stage1_buf.resize(max_r);
    scratch.q2_buf.resize(max_r);

    // Embedding: buf0 = embed_weight[token_id]
    float * x_ptr = scratch.buf0.data();
    float * res_ptr = scratch.buf1.data();
    std::memcpy(x_ptr, embed_ptr + token_id * hidden_size,
                hidden_size * sizeof(float));

    const float eps = static_cast<float>(rms_eps);
    const float scale = static_cast<float>(attn_scale);

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const int64_t tbase = layer * 37;  // tensor offset
        const int64_t dbase = layer * 28;  // dims offset

        const float * ln1_w = layer_tensors[tbase + 0].data_ptr<float>();
        const float * ln2_w = layer_tensors[tbase + 1].data_ptr<float>();

        // ** OPTIMIZED: pointer swap instead of memcpy for residual **
        // residual = x (swap pointers, zero-copy)
        std::swap(x_ptr, res_ptr);
        // Now res_ptr holds old x (= residual), x_ptr is free for reuse

        // RMSNorm (reads from res_ptr which is the old x)
        rms_norm_inline(res_ptr, ln1_w, eps, hidden_size, scratch.normed.data());

        // 7 projections: q=0, k=1, v=2, o=3, gate=4, up=5, down=6
        auto do_proj = [&](int64_t p, const float * input_data, float * out_data) {
            const int64_t to = tbase + 2 + p * 5;
            const int64_t do_ = dbase + p * 4;

            const int64_t v_cols = layer_dims[do_ + 0];
            const int64_t rank = layer_dims[do_ + 1];
            const int64_t u_cols = layer_dims[do_ + 2];
            const int64_t out_f = layer_dims[do_ + 3];

            const float * v2 = layer_tensors[to + 0].data_ptr<float>();
            const uint8_t * v_i2 = layer_tensors[to + 1].data_ptr<uint8_t>();
            const int64_t v_i2_stride = layer_tensors[to + 1].size(1);
            const float * mid_p = layer_tensors[to + 2].data_ptr<float>();
            const uint8_t * u_i2 = layer_tensors[to + 3].data_ptr<uint8_t>();
            const int64_t u_i2_stride = layer_tensors[to + 3].size(1);
            const float * u1 = layer_tensors[to + 4].data_ptr<float>();

            lb_linear_i2_inline(
                input_data, v_cols, v2,
                v_i2, v_i2_stride, mid_p, rank,
                u_i2, u_i2_stride, u1, out_f, u_cols,
                out_data,
                scratch.q1_buf.data(), scratch.stage1_buf.data(), scratch.q2_buf.data());
        };

        // Q, K, V projections
        do_proj(0, scratch.normed.data(), scratch.q_out.data());
        do_proj(1, scratch.normed.data(), scratch.k_out.data());
        do_proj(2, scratch.normed.data(), scratch.v_out.data());

        // Fused attention
        float * k_cache = kv_caches[layer * 2 + 0].data_ptr<float>();
        float * v_cache = kv_caches[layer * 2 + 1].data_ptr<float>();

        fused_attention_inline(
            scratch.q_out.data(), scratch.k_out.data(), scratch.v_out.data(),
            k_cache, v_cache,
            position, num_kv_heads, kv_repeat, head_dim,
            max_seq_len, scale, scratch.attn_out.data());

        // O projection: attn_out → o_out
        do_proj(3, scratch.attn_out.data(), scratch.o_out.data());

        // Residual add: x = residual + o_out
#ifdef __AVX2__
        for (int64_t i = 0; i + 8 <= hidden_size; i += 8) {
            _mm256_storeu_ps(x_ptr + i,
                _mm256_add_ps(_mm256_loadu_ps(res_ptr + i),
                              _mm256_loadu_ps(scratch.o_out.data() + i)));
        }
        for (int64_t i = (hidden_size / 8) * 8; i < hidden_size; ++i)
            x_ptr[i] = res_ptr[i] + scratch.o_out[i];
#else
        for (int64_t i = 0; i < hidden_size; ++i)
            x_ptr[i] = res_ptr[i] + scratch.o_out[i];
#endif

        // MLP: swap for second residual (x_ptr has attn_residual result)
        std::swap(x_ptr, res_ptr);
        // res_ptr = post-attn x, x_ptr = free

        // Post-attention RMSNorm
        rms_norm_inline(res_ptr, ln2_w, eps, hidden_size, scratch.normed.data());

        // Gate and Up projections
        do_proj(4, scratch.normed.data(), scratch.gate_out.data());
        do_proj(5, scratch.normed.data(), scratch.up_out.data());

        // SiLU-mul
        const int64_t intermediate_size = layer_dims[dbase + 4 * 4 + 3]; // gate out_features
        silu_mul_inline(scratch.gate_out.data(), scratch.up_out.data(),
                        intermediate_size, scratch.mlp_hidden.data());

        // Down projection
        do_proj(6, scratch.mlp_hidden.data(), scratch.down_out.data());

        // Residual add: x = residual + down_out
#ifdef __AVX2__
        for (int64_t i = 0; i + 8 <= hidden_size; i += 8) {
            _mm256_storeu_ps(x_ptr + i,
                _mm256_add_ps(_mm256_loadu_ps(res_ptr + i),
                              _mm256_loadu_ps(scratch.down_out.data() + i)));
        }
        for (int64_t i = (hidden_size / 8) * 8; i < hidden_size; ++i)
            x_ptr[i] = res_ptr[i] + scratch.down_out[i];
#else
        for (int64_t i = 0; i < hidden_size; ++i)
            x_ptr[i] = res_ptr[i] + scratch.down_out[i];
#endif
    }

    // Final RMSNorm — output directly to result tensor
    auto output = at::empty({1, hidden_size}, embed_weight.options());
    rms_norm_inline(x_ptr, final_norm_ptr, eps, hidden_size, output.data_ptr<float>());
    return output;
}

// ============================================================================
// Q4_0 quantized lm_head (llama.cpp compatible format)
// ============================================================================

// Q4_0 block: 32 values, 1 fp16 scale + 16 nibble bytes = 18 bytes/block
#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;           // fp16 delta (scale)
    uint8_t qs[16];       // 32 x 4-bit nibbles packed in pairs
};
struct block_q8_0 {
    uint16_t d;           // fp16 delta (scale)
    int8_t qs[32];        // 32 x int8
};
#pragma pack(pop)

static constexpr int QK4_0 = 32;
static constexpr int QK8_0 = 32;

// FP16 <-> FP32 conversion helpers
static inline float fp16_to_fp32(uint16_t h) {
    // Use compiler/hardware conversion if available
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (expo == 0) {
        if (mant == 0) { float r; uint32_t v = sign; std::memcpy(&r, &v, 4); return r; }
        // subnormal
        while (!(mant & 0x400)) { mant <<= 1; expo--; }
        expo++; mant &= ~0x400;
    } else if (expo == 31) {
        uint32_t v = sign | 0x7F800000u | (mant << 13);
        float r; std::memcpy(&r, &v, 4); return r;
    }
    uint32_t v = sign | ((expo + 112) << 23) | (mant << 13);
    float r; std::memcpy(&r, &v, 4); return r;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t v; std::memcpy(&v, &f, 4);
    uint32_t sign = (v >> 16) & 0x8000;
    int32_t expo = ((v >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (v >> 13) & 0x3FF;
    if (expo <= 0) return static_cast<uint16_t>(sign);
    if (expo >= 31) return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (expo << 10) | mant);
}

// Quantize a single row of FP32 to Q4_0 blocks
static void quantize_row_fp32_to_q4_0(const float * x, block_q4_0 * y, int64_t k) {
    const int64_t nb = k / QK4_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f, max_val = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            if (std::fabs(x[i * QK4_0 + j]) > amax) {
                amax = std::fabs(x[i * QK4_0 + j]);
                max_val = x[i * QK4_0 + j];
            }
        }
        const float d = max_val / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = x[i * QK4_0 + j] * id;
            const float x1 = x[i * QK4_0 + QK4_0 / 2 + j] * id;
            const uint8_t xi0 = std::min(15, static_cast<int>(x0 + 8.5f));
            const uint8_t xi1 = std::min(15, static_cast<int>(x1 + 8.5f));
            y[i].qs[j] = xi0 | (xi1 << 4);
        }
    }
}

// Quantize a single row of FP32 to Q8_0 blocks
static void quantize_row_fp32_to_q8_0(const float * x, block_q8_0 * y, int64_t k) {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j)
            amax = std::max(amax, std::fabs(x[i * QK8_0 + j]));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j)
            y[i].qs[j] = static_cast<int8_t>(std::roundf(x[i * QK8_0 + j] * id));
    }
}

// Q4_0 x Q8_0 dot product for one row (llama.cpp AVX2 pattern)
static float vec_dot_q4_0_q8_0(const block_q4_0 * x, const block_q8_0 * y, int64_t nb) {
    float sumf = 0.0f;
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        const __m256 d_v = _mm256_set1_ps(d);

        // Unpack Q4 nibbles to bytes: load 16 bytes -> 32 values in [0,15]
        const __m128i q4bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x[i].qs));
        const __m256i q4_lo = _mm256_and_si256(
            _mm256_insertf128_si256(_mm256_castsi128_si256(q4bits), q4bits, 1),
            _mm256_set1_epi8(0x0F));
        const __m256i q4_hi = _mm256_and_si256(
            _mm256_insertf128_si256(
                _mm256_castsi128_si256(_mm_srli_epi16(q4bits, 4)),
                _mm_srli_epi16(q4bits, 4), 1),
            _mm256_set1_epi8(0x0F));
        // Interleave: low nibbles in low 128, high nibbles in high 128
        const __m256i qx = _mm256_permute2x128_si256(q4_lo, q4_hi, 0x20);
        // Offset into [-8, +7]
        const __m256i off = _mm256_set1_epi8(8);
        const __m256i qx_signed = _mm256_sub_epi8(qx, off);

        // Load Q8 values
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[i].qs));

        // Dot product: need signed*signed handling via maddubs
        // maddubs requires first operand unsigned. Use abs(qy) and sign(qx, qy) trick
        const __m256i abs_qy = _mm256_abs_epi8(qy);
        const __m256i sign_qx = _mm256_sign_epi8(qx_signed, qy);
        const __m256i prod16 = _mm256_maddubs_epi16(abs_qy, sign_qx);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
        acc = _mm256_fmadd_ps(d_v, _mm256_cvtepi32_ps(prod32), acc);
    }
    // horizontal sum
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, 1));
    sumf = _mm_cvtss_f32(r1);
#else
    for (int64_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * fp16_to_fp32(y[i].d);
        int32_t sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >> 4) - 8;
            sumi += v0 * y[i].qs[j] + v1 * y[i].qs[j + QK4_0 / 2];
        }
        sumf += d * static_cast<float>(sumi);
    }
#endif
    return sumf;
}

// Quantize lm_head weight from FP32 to Q4_0 packed tensor
// Returns: (q4_packed [packed_bytes], original_shape_info_unused)
at::Tensor quantize_lm_head_q4_cpu(const at::Tensor & weight) {
    check_float32_tensor(weight, "weight");
    const int64_t vocab = weight.size(0);
    const int64_t hidden = weight.size(1);
    TORCH_CHECK(hidden % QK4_0 == 0, "hidden_size must be multiple of 32");

    const int64_t blocks_per_row = hidden / QK4_0;
    const int64_t bytes_per_row = blocks_per_row * sizeof(block_q4_0);
    const int64_t total_bytes = vocab * bytes_per_row;

    // Store as raw byte tensor
    auto q_packed = at::empty({total_bytes}, weight.options().dtype(at::kByte));
    const float * w_ptr = weight.data_ptr<float>();
    uint8_t * q_ptr = q_packed.data_ptr<uint8_t>();

    at::parallel_for(0, vocab, 0, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            quantize_row_fp32_to_q4_0(
                w_ptr + row * hidden,
                reinterpret_cast<block_q4_0 *>(q_ptr + row * bytes_per_row),
                hidden);
        }
    });

    return q_packed;
}

// generate_token_cpu: full_forward + Q4 lm_head + argmax in ONE C++ call
// Returns: token_id as scalar tensor
int64_t generate_token_cpu(
    int64_t token_id,
    const at::Tensor & embed_weight,
    const at::Tensor & final_norm_weight,
    at::TensorList layer_tensors,
    at::TensorList kv_caches,
    at::IntArrayRef layer_dims,
    const at::Tensor & lm_head_q4,    // Q4_0 packed weight bytes
    int64_t vocab_size,
    int64_t num_layers,
    int64_t hidden_size,
    int64_t num_kv_heads,
    int64_t kv_repeat,
    int64_t head_dim,
    int64_t max_seq_len,
    int64_t position,
    double attn_scale,
    double rms_eps
) {
    // ===== Step 1: full_forward (reuse existing logic) =====
    std::vector<int64_t> layer_dims_vec(layer_dims.begin(), layer_dims.end());
    at::Tensor hidden = full_forward_cpu(
        token_id, embed_weight, final_norm_weight,
        layer_tensors, kv_caches, layer_dims_vec,
        num_layers, hidden_size, num_kv_heads, kv_repeat,
        head_dim, max_seq_len, position, attn_scale, rms_eps);

    const float * hidden_ptr = hidden.data_ptr<float>();

    // ===== Step 2: Q4 lm_head GEMV =====
    TORCH_CHECK(hidden_size % QK4_0 == 0, "hidden_size must be multiple of 32");
    const int64_t blocks_per_row = hidden_size / QK4_0;
    const int64_t bytes_per_row = blocks_per_row * sizeof(block_q4_0);
    const uint8_t * q4_ptr = lm_head_q4.data_ptr<uint8_t>();

    // Quantize hidden state to Q8_0
    thread_local std::vector<block_q8_0> input_q8;
    input_q8.resize(blocks_per_row);
    quantize_row_fp32_to_q8_0(hidden_ptr, input_q8.data(), hidden_size);

    // Parallel GEMV across vocab rows
    thread_local std::vector<float> logits_buf;
    logits_buf.resize(vocab_size);

    at::parallel_for(0, vocab_size, 0, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const block_q4_0 * w_row = reinterpret_cast<const block_q4_0 *>(
                q4_ptr + row * bytes_per_row);
            logits_buf[row] = vec_dot_q4_0_q8_0(w_row, input_q8.data(), blocks_per_row);
        }
    });

    // ===== Step 3: argmax =====
    int64_t best_id = 0;
    float best_val = logits_buf[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits_buf[i] > best_val) {
            best_val = logits_buf[i];
            best_id = i;
        }
    }

    return best_id;
}

}  // namespace littlebit_cpu_ops

TORCH_LIBRARY(littlebit_cpu_ops, m) {
    m.def("quantize_per_row_int8(Tensor input) -> (Tensor, Tensor)");
    m.def("gemv_int8xsign(Tensor input, Tensor packed_weight, int n_cols) -> Tensor");
    m.def(
        "littlebit_linear("
        "Tensor input, Tensor v2, Tensor v_sign, int v_cols, "
        "Tensor mid, Tensor u_sign, int u_cols, Tensor u1"
        ") -> Tensor"
    );
    m.def("embedding_lookup(Tensor weight, Tensor indices) -> Tensor");
    m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
    m.def("dense_gemv_f32(Tensor input, Tensor weight) -> Tensor");
    m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
    m.def(
        "fused_attention("
        "Tensor q, Tensor k_cache, Tensor v_cache, "
        "Tensor k_new, Tensor v_new, "
        "int position, int num_kv_heads, int kv_repeat, "
        "int head_dim, float attn_scale"
        ") -> Tensor"
    );
    m.def("repack_signs_to_i2(Tensor packed_signs, int n_cols) -> Tensor");
    m.def(
        "littlebit_linear_i2("
        "Tensor input, Tensor v2, Tensor v_sign_i2, int v_cols, "
        "Tensor mid, Tensor u_sign_i2, int u_cols, Tensor u1"
        ") -> Tensor"
    );
    m.def(
        "full_forward("
        "int token_id, Tensor embed_weight, Tensor final_norm_weight, "
        "Tensor[] layer_tensors, Tensor[] kv_caches, "
        "int[] layer_dims, int num_layers, int hidden_size, "
        "int num_kv_heads, int kv_repeat, int head_dim, "
        "int max_seq_len, int position, float attn_scale, float rms_eps"
        ") -> Tensor"
    );
    m.def("quantize_lm_head_q4(Tensor weight) -> Tensor");
    m.def(
        "generate_token("
        "int token_id, Tensor embed_weight, Tensor final_norm_weight, "
        "Tensor[] layer_tensors, Tensor[] kv_caches, "
        "int[] layer_dims, Tensor lm_head_q4, int vocab_size, "
        "int num_layers, int hidden_size, "
        "int num_kv_heads, int kv_repeat, int head_dim, "
        "int max_seq_len, int position, float attn_scale, float rms_eps"
        ") -> int"
    );
}

TORCH_LIBRARY_IMPL(littlebit_cpu_ops, CPU, m) {
    m.impl("quantize_per_row_int8", littlebit_cpu_ops::quantize_per_row_int8_cpu);
    m.impl("gemv_int8xsign", littlebit_cpu_ops::gemv_int8xsign_cpu);
    m.impl("littlebit_linear", littlebit_cpu_ops::littlebit_linear_cpu);
    m.impl("embedding_lookup", littlebit_cpu_ops::embedding_lookup_cpu);
    m.impl("rms_norm", littlebit_cpu_ops::rms_norm_cpu);
    m.impl("dense_gemv_f32", littlebit_cpu_ops::dense_gemv_f32_cpu);
    m.impl("silu_mul", littlebit_cpu_ops::silu_mul_cpu);
    m.impl("fused_attention", littlebit_cpu_ops::fused_attention_cpu);
    m.impl("repack_signs_to_i2", littlebit_cpu_ops::repack_signs_to_i2_cpu);
    m.impl("littlebit_linear_i2", littlebit_cpu_ops::littlebit_linear_i2_cpu);
    m.impl("full_forward", littlebit_cpu_ops::full_forward_cpu);
    m.impl("quantize_lm_head_q4", littlebit_cpu_ops::quantize_lm_head_q4_cpu);
    m.impl("generate_token", littlebit_cpu_ops::generate_token_cpu);
}
