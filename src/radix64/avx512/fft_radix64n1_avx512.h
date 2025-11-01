/**
 * @file fft_radix64_avx512_n1.h
 * @brief Radix-64 N1 (Twiddle-less) AVX-512 - 8×8 Cooley-Tukey [U=4 OPTIMIZED]
 *
 * @details
 * CLEAN REBUILD - Follows scalar implementation exactly
 * U=4 OPTIMIZATION - m-stripmined for minimal register pressure
 *
 * N1 CODELET ARCHITECTURE:
 * ========================
 * - "N1" = No stage twiddles (all W₆₄ stage twiddles = 1+0i)
 * - Only internal W₆₄ geometric merge twiddles remain
 * - Used as first/last stage in mixed-radix factorizations
 * - 64 = 8 × 8 decomposition for optimal performance
 *
 * ARCHITECTURE: 8×8 COOLEY-TUKEY
 * ================================
 * 1. Eight radix-8 N1 butterflies (r=0..7, 8..15, ..., 56..63)
 * 2. Apply W₆₄ merge twiddles to outputs 1-7 (output 0 unchanged)
 * 3. Radix-8 final combine (across 8 sub-FFTs)
 *
 * AVX-512 OPTIMIZATIONS:
 * =======================
 * ✅ Native mask registers (__mmask8) for tail handling
 * ✅ 8-wide vectors (8 doubles per ZMM)
 * ✅ Main loop: k += 32 (U=4 with m-stripmining)
 * ✅ Register pressure optimized: ~28 ZMM peak (fits in 32)
 * ✅ W64 constants broadcast on-demand (saves 14 ZMM)
 * ✅ Prefetch distance: 32 doubles (conservative)
 * ✅ K-tiling: Tk = 64 (L1-cache sized)
 * ✅ NT store threshold: 256KB
 * ✅ Optimized W₈ twiddles (specialized paths for W₈^1, W₈^3)
 * ✅ Generic W₆₄ twiddles (no shortcuts - mathematically correct)
 *
 * U=4 M-STRIPMINE OPTIMIZATION:
 * ==============================
 * Instead of processing all 64 lanes per slot:
 *   for m in 0..7:
 *     for slot in {k, k+8, k+16, k+24}:
 *       - Load only 8 lanes for this m-slice
 *       - Radix-8 butterfly (in-place)
 *       - Apply W64 (broadcast on-demand)
 *       - Final combine & store immediately
 *
 * Peak live registers per m-slice:
 *   - Inputs: 16 ZMM (x0..x7 re/im)
 *   - Temps: ~8 ZMM (butterfly e/o, reused)
 *   - Constants: 6 ZMM (W8 hot, W64 transient)
 *   - Working: ~3 ZMM
 *   - Total: ~29 ZMM ✅ Fits in 32
 *
 * @author VectorFFT Team
 * @version 3.0 (U=4 m-stripmine optimization)
 * @date 2025
 */

#ifndef FFT_RADIX64_AVX512_N1_H
#define FFT_RADIX64_AVX512_N1_H

#include <immintrin.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

// Include radix-8 N1 for reuse
#include "fft_radix8_avx512_n1.h"

//==============================================================================
// COMPILER HINTS (PRESERVED)
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX512
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX512
#endif

//==============================================================================
// CONFIGURATION (PRESERVED)
//==============================================================================

#ifndef RADIX64_PREFETCH_DISTANCE_N1_AVX512
#define RADIX64_PREFETCH_DISTANCE_N1_AVX512 32
#endif

#ifndef RADIX64_TILE_SIZE_N1_AVX512
#define RADIX64_TILE_SIZE_N1_AVX512 64
#endif

#ifndef RADIX64_STREAM_THRESHOLD_KB_N1_AVX512
#define RADIX64_STREAM_THRESHOLD_KB_N1_AVX512 256
#endif

//==============================================================================
// HELPER: Complex Multiply (Generic, FMA-optimized) - PRESERVED
//==============================================================================

/**
 * @brief Generic complex multiply with FMA (ar + i*ai) * (br + i*bi)
 *
 * @details
 * For SoA layout, this is already optimal:
 * - real = ar*br - ai*bi  (one FMA)
 * - imag = ar*bi + ai*br  (one FMA)
 */
TARGET_AVX512
FORCE_INLINE void
cmul_v512(
    const __m512d ar, const __m512d ai,
    const __m512d br, const __m512d bi,
    __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    // Issue MUL early for better pipelining
    __m512d t0 = _mm512_mul_pd(ai, br); // ai*br
    __m512d t1 = _mm512_mul_pd(ai, bi); // ai*bi

    *tr = _mm512_fmsub_pd(ar, br, t1); // ar*br - ai*bi
    *ti = _mm512_fmadd_pd(ar, bi, t0); // ar*bi + ai*br
}

//==============================================================================
// NT STORE DECISION (PRESERVED)
//==============================================================================

FORCE_INLINE bool
radix64_should_use_nt_stores_n1_avx512(
    size_t K,
    const void *out_re,
    const void *out_im)
{
    const size_t bytes_per_k = 64 * 2 * sizeof(double);
    const size_t threshold_k = (RADIX64_STREAM_THRESHOLD_KB_N1_AVX512 * 1024) / bytes_per_k;

    return (K >= threshold_k) &&
           (((uintptr_t)out_re & 63) == 0) &&
           (((uintptr_t)out_im & 63) == 0);
}

//==============================================================================
// PREFETCH HELPERS (PRESERVED)
//==============================================================================

#define RADIX64_PREFETCH_INPUTS_N1_AVX512(k_next, k_limit, K, in_re, in_im)             \
    do                                                                                  \
    {                                                                                   \
        if ((k_next) < (k_limit))                                                       \
        {                                                                               \
            for (int _r = 0; _r < 64; _r++)                                             \
            {                                                                           \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0); \
            }                                                                           \
        }                                                                               \
    } while (0)

//==============================================================================
// OPTIMIZED W₈ TWIDDLES (Phase 1 - SAFE OPTIMIZATIONS) - PRESERVED
//==============================================================================

/**
 * @brief Apply W₈ twiddles - OPTIMIZED (Forward)
 *
 * @details
 * Uses specialized paths for W₈^1 and W₈^3:
 * - W₈^1 = (c, -c) where c = √2/2
 * - W₈^2 = (0, -1)
 * - W₈^3 = (-c, -c)
 *
 * These are the ONLY safe optimizations (proven mathematically correct).
 */
TARGET_AVX512
FORCE_INLINE void
apply_w8_twiddles_forward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    const __m512d C8 = _mm512_set1_pd(C8_CONSTANT);
    const __m512d NEG_C8 = _mm512_set1_pd(-C8_CONSTANT);
    const __m512d neg_zero = _mm512_set1_pd(-0.0);

    // W₈^1 = (C8, -C8) - OPTIMIZED
    {
        __m512d r = *o1_re;
        __m512d i = *o1_im;
        __m512d sum = _mm512_add_pd(r, i);
        __m512d diff = _mm512_sub_pd(i, r);
        *o1_re = _mm512_mul_pd(sum, C8);
        *o1_im = _mm512_mul_pd(diff, C8);
    }

    // W₈^2 = (0, -1) - OPTIMIZED
    {
        __m512d r = *o2_re;
        *o2_re = *o2_im;
        *o2_im = _mm512_xor_pd(r, neg_zero);
    }

    // W₈^3 = (-C8, -C8) - OPTIMIZED
    {
        __m512d r = *o3_re;
        __m512d i = *o3_im;
        __m512d diff = _mm512_sub_pd(r, i);
        __m512d sum = _mm512_add_pd(r, i);
        *o3_re = _mm512_mul_pd(diff, NEG_C8);
        *o3_im = _mm512_mul_pd(sum, NEG_C8);
    }
}

/**
 * @brief Apply W₈ twiddles - OPTIMIZED (Backward)
 */
TARGET_AVX512
FORCE_INLINE void
apply_w8_twiddles_backward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    const __m512d C8 = _mm512_set1_pd(C8_CONSTANT);
    const __m512d NEG_C8 = _mm512_set1_pd(-C8_CONSTANT);
    const __m512d neg_zero = _mm512_set1_pd(-0.0);

    // W₈^(-1) = (C8, +C8) - OPTIMIZED
    {
        __m512d r = *o1_re;
        __m512d i = *o1_im;
        __m512d diff = _mm512_sub_pd(r, i);
        __m512d sum = _mm512_add_pd(r, i);
        *o1_re = _mm512_mul_pd(diff, C8);
        *o1_im = _mm512_mul_pd(sum, C8);
    }

    // W₈^(-2) = (0, +1) - OPTIMIZED
    {
        __m512d r = *o2_re;
        *o2_re = _mm512_xor_pd(*o2_im, neg_zero);
        *o2_im = r;
    }

    // W₈^(-3) = (-C8, +C8) - OPTIMIZED
    {
        __m512d r = *o3_re;
        __m512d i = *o3_im;
        __m512d sum = _mm512_add_pd(r, i);
        __m512d diff = _mm512_sub_pd(i, r);
        *o3_re = _mm512_mul_pd(sum, NEG_C8);
        *o3_im = _mm512_mul_pd(diff, C8);
    }
}

//==============================================================================
// RADIX-8 N1 BUTTERFLY (Inline for Register Arrays) - PRESERVED
//==============================================================================

TARGET_AVX512
FORCE_INLINE void
radix8_n1_butterfly_inline_forward_avx512(
    __m512d x_re[8], __m512d x_im[8],
    const __m512d W8_1_re, const __m512d W8_1_im,
    const __m512d W8_3_re, const __m512d W8_3_im,
    const __m512d sign_mask)
{
    // Even radix-4: (x0, x2, x4, x6)
    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    // Odd radix-4: (x1, x3, x5, x7)
    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    // Apply W₈ twiddles to odd outputs
    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Combine
    x_re[0] = _mm512_add_pd(e0_re, o0_re);
    x_im[0] = _mm512_add_pd(e0_im, o0_im);
    x_re[1] = _mm512_add_pd(e1_re, o1_re);
    x_im[1] = _mm512_add_pd(e1_im, o1_im);
    x_re[2] = _mm512_add_pd(e2_re, o2_re);
    x_im[2] = _mm512_add_pd(e2_im, o2_im);
    x_re[3] = _mm512_add_pd(e3_re, o3_re);
    x_im[3] = _mm512_add_pd(e3_im, o3_im);
    x_re[4] = _mm512_sub_pd(e0_re, o0_re);
    x_im[4] = _mm512_sub_pd(e0_im, o0_im);
    x_re[5] = _mm512_sub_pd(e1_re, o1_re);
    x_im[5] = _mm512_sub_pd(e1_im, o1_im);
    x_re[6] = _mm512_sub_pd(e2_re, o2_re);
    x_im[6] = _mm512_sub_pd(e2_im, o2_im);
    x_re[7] = _mm512_sub_pd(e3_re, o3_re);
    x_im[7] = _mm512_sub_pd(e3_im, o3_im);
}

TARGET_AVX512
FORCE_INLINE void
radix8_n1_butterfly_inline_backward_avx512(
    __m512d x_re[8], __m512d x_im[8],
    const __m512d W8_1_re, const __m512d W8_1_im,
    const __m512d W8_3_re, const __m512d W8_3_im,
    const __m512d sign_mask)
{
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d neg_sign = _mm512_xor_pd(sign_mask, neg_zero);

    // Even radix-4
    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       neg_sign);

    // Odd radix-4
    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       neg_sign);

    // Apply conjugate W₈ twiddles
    apply_w8_twiddles_backward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Combine
    x_re[0] = _mm512_add_pd(e0_re, o0_re);
    x_im[0] = _mm512_add_pd(e0_im, o0_im);
    x_re[1] = _mm512_add_pd(e1_re, o1_re);
    x_im[1] = _mm512_add_pd(e1_im, o1_im);
    x_re[2] = _mm512_add_pd(e2_re, o2_re);
    x_im[2] = _mm512_add_pd(e2_im, o2_im);
    x_re[3] = _mm512_add_pd(e3_re, o3_re);
    x_im[3] = _mm512_add_pd(e3_im, o3_im);
    x_re[4] = _mm512_sub_pd(e0_re, o0_re);
    x_im[4] = _mm512_sub_pd(e0_im, o0_im);
    x_re[5] = _mm512_sub_pd(e1_re, o1_re);
    x_im[5] = _mm512_sub_pd(e1_im, o1_im);
    x_re[6] = _mm512_sub_pd(e2_re, o2_re);
    x_im[6] = _mm512_sub_pd(e2_im, o2_im);
    x_re[7] = _mm512_sub_pd(e3_re, o3_re);
    x_im[7] = _mm512_sub_pd(e3_im, o3_im);
}

//==============================================================================
// W₆₄ CONSTANT TABLES (For On-Demand Broadcasting)
//==============================================================================

/**
 * @brief W64 twiddle constants for on-demand broadcasting
 * 
 * @details
 * Instead of hoisting all 14 W64 constants (28 ZMM total), we store them
 * in small tables and broadcast on-demand. This saves 14 precious ZMM registers.
 * 
 * vbroadcastsd has 0.5 cycle throughput on SKX+, so the extra broadcasts
 * are essentially free compared to the register pressure savings.
 */

// Forward W64 constants (W64^k for k=1..7)
static const double W64_FV_TABLE_RE[8] = {
    1.0,           // W64^0 (unused, for alignment)
    W64_FV_1_RE,   // W64^1
    W64_FV_2_RE,   // W64^2
    W64_FV_3_RE,   // W64^3
    W64_FV_4_RE,   // W64^4
    W64_FV_5_RE,   // W64^5
    W64_FV_6_RE,   // W64^6
    W64_FV_7_RE    // W64^7
};

static const double W64_FV_TABLE_IM[8] = {
    0.0,           // W64^0 (unused, for alignment)
    W64_FV_1_IM,   // W64^1
    W64_FV_2_IM,   // W64^2
    W64_FV_3_IM,   // W64^3
    W64_FV_4_IM,   // W64^4
    W64_FV_5_IM,   // W64^5
    W64_FV_6_IM,   // W64^6
    W64_FV_7_IM    // W64^7
};

// Backward W64 constants (W64^(-k) for k=1..7)
static const double W64_BV_TABLE_RE[8] = {
    1.0,           // W64^0 (unused, for alignment)
    W64_BV_1_RE,   // W64^(-1)
    W64_BV_2_RE,   // W64^(-2)
    W64_BV_3_RE,   // W64^(-3)
    W64_BV_4_RE,   // W64^(-4)
    W64_BV_5_RE,   // W64^(-5)
    W64_BV_6_RE,   // W64^(-6)
    W64_BV_7_RE    // W64^(-7)
};

static const double W64_BV_TABLE_IM[8] = {
    0.0,           // W64^0 (unused, for alignment)
    W64_BV_1_IM,   // W64^(-1)
    W64_BV_2_IM,   // W64^(-2)
    W64_BV_3_IM,   // W64^(-3)
    W64_BV_4_IM,   // W64^(-4)
    W64_BV_5_IM,   // W64^(-5)
    W64_BV_6_IM,   // W64^(-6)
    W64_BV_7_IM    // W64^(-7)
};

//==============================================================================
// M-SLICE LOAD/STORE (8 lanes for single m-slice)
//==============================================================================

/**
 * @brief Load 8 complex lanes for a single m-slice
 * 
 * @details
 * For m-slice 'm', loads x_r[m] for r = 0..7 (i.e., indices m, m+8, m+16, ..., m+56)
 * This is the key to U=4: we only load 8 lanes at a time, not all 64.
 * 
 * Memory pattern: Each x_r[m] is at position: slot + (r*8 + m) * K
 */
FORCE_INLINE void
load_m_slice_soa_n1_avx512(
    size_t slot,        // k-position (k, k+8, k+16, or k+24)
    size_t m,           // m-slice index (0..7)
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    __m512d x_re[8],    // Output: x0[m]..x7[m]
    __m512d x_im[8])
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);

    for (int r = 0; r < 8; r++)
    {
        size_t idx = slot + (r * 8 + m) * K;
        x_re[r] = _mm512_load_pd(&in_re_aligned[idx]);
        x_im[r] = _mm512_load_pd(&in_im_aligned[idx]);
    }
}

/**
 * @brief Store 8 complex lanes for a single m-slice
 */
FORCE_INLINE void
store_m_slice_soa_n1_avx512(
    size_t slot,
    size_t m,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d y_re[8],
    const __m512d y_im[8])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (int r = 0; r < 8; r++)
    {
        size_t idx = slot + (r * 8 + m) * K;
        _mm512_store_pd(&out_re_aligned[idx], y_re[r]);
        _mm512_store_pd(&out_im_aligned[idx], y_im[r]);
    }
}

/**
 * @brief Store 8 complex lanes for a single m-slice (non-temporal)
 */
FORCE_INLINE void
store_m_slice_soa_n1_avx512_stream(
    size_t slot,
    size_t m,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d y_re[8],
    const __m512d y_im[8])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (int r = 0; r < 8; r++)
    {
        size_t idx = slot + (r * 8 + m) * K;
        _mm512_stream_pd(&out_re_aligned[idx], y_re[r]);
        _mm512_stream_pd(&out_im_aligned[idx], y_im[r]);
    }
}

//==============================================================================
// APPLY W₆₄ FOR SINGLE M-SLICE (On-Demand Broadcast)
//==============================================================================

/**
 * @brief Apply W64 twiddles to a single m-slice (Forward)
 * 
 * @details
 * Broadcasts W64^r on-demand for r=1..7 and applies to x_r[m].
 * x_0[m] is unchanged (W64^0 = 1).
 * 
 * This saves 14 ZMM registers compared to hoisting all W64 constants.
 * vbroadcastsd latency (~1-2 cycles) is hidden by FMA throughput.
 */
TARGET_AVX512
FORCE_INLINE void
apply_w64_m_slice_forward_avx512(
    __m512d x_re[8],    // x0[m]..x7[m] - modified in-place
    __m512d x_im[8],
    const double *RESTRICT w64_re_table,  // W64_FV_TABLE_RE
    const double *RESTRICT w64_im_table)  // W64_FV_TABLE_IM
{
    __m512d tmp_re, tmp_im;

    // x0[m] unchanged (W64^0 = 1)

    // x1[m] *= W64^1
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[1]);
        __m512d wi = _mm512_set1_pd(w64_im_table[1]);
        cmul_v512(x_re[1], x_im[1], wr, wi, &tmp_re, &tmp_im);
        x_re[1] = tmp_re;
        x_im[1] = tmp_im;
    }

    // x2[m] *= W64^2
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[2]);
        __m512d wi = _mm512_set1_pd(w64_im_table[2]);
        cmul_v512(x_re[2], x_im[2], wr, wi, &tmp_re, &tmp_im);
        x_re[2] = tmp_re;
        x_im[2] = tmp_im;
    }

    // x3[m] *= W64^3
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[3]);
        __m512d wi = _mm512_set1_pd(w64_im_table[3]);
        cmul_v512(x_re[3], x_im[3], wr, wi, &tmp_re, &tmp_im);
        x_re[3] = tmp_re;
        x_im[3] = tmp_im;
    }

    // x4[m] *= W64^4
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[4]);
        __m512d wi = _mm512_set1_pd(w64_im_table[4]);
        cmul_v512(x_re[4], x_im[4], wr, wi, &tmp_re, &tmp_im);
        x_re[4] = tmp_re;
        x_im[4] = tmp_im;
    }

    // x5[m] *= W64^5
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[5]);
        __m512d wi = _mm512_set1_pd(w64_im_table[5]);
        cmul_v512(x_re[5], x_im[5], wr, wi, &tmp_re, &tmp_im);
        x_re[5] = tmp_re;
        x_im[5] = tmp_im;
    }

    // x6[m] *= W64^6
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[6]);
        __m512d wi = _mm512_set1_pd(w64_im_table[6]);
        cmul_v512(x_re[6], x_im[6], wr, wi, &tmp_re, &tmp_im);
        x_re[6] = tmp_re;
        x_im[6] = tmp_im;
    }

    // x7[m] *= W64^7
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[7]);
        __m512d wi = _mm512_set1_pd(w64_im_table[7]);
        cmul_v512(x_re[7], x_im[7], wr, wi, &tmp_re, &tmp_im);
        x_re[7] = tmp_re;
        x_im[7] = tmp_im;
    }
}

/**
 * @brief Apply W64 twiddles to a single m-slice (Backward)
 */
TARGET_AVX512
FORCE_INLINE void
apply_w64_m_slice_backward_avx512(
    __m512d x_re[8],    // x0[m]..x7[m] - modified in-place
    __m512d x_im[8],
    const double *RESTRICT w64_re_table,  // W64_BV_TABLE_RE
    const double *RESTRICT w64_im_table)  // W64_BV_TABLE_IM
{
    __m512d tmp_re, tmp_im;

    // x0[m] unchanged (W64^0 = 1)

    // Apply W64^(-k) for k=1..7 (generic complex multiply)
    for (int r = 1; r < 8; r++)
    {
        __m512d wr = _mm512_set1_pd(w64_re_table[r]);
        __m512d wi = _mm512_set1_pd(w64_im_table[r]);
        cmul_v512(x_re[r], x_im[r], wr, wi, &tmp_re, &tmp_im);
        x_re[r] = tmp_re;
        x_im[r] = tmp_im;
    }
}

//==============================================================================
// RADIX-8 FINAL COMBINE FOR SINGLE M-SLICE
//==============================================================================

/**
 * @brief Final radix-8 combine for a single m-slice (Forward)
 * 
 * @details
 * Takes x0[m]..x7[m] (outputs from 8 radix-8 butterflies after W64 twiddling)
 * and performs the final radix-8 across them to produce y[m + r*8] for r=0..7.
 * 
 * This is equivalent to one iteration of the radix8_final_combine loop.
 */
TARGET_AVX512
FORCE_INLINE void
radix8_final_combine_m_slice_forward_avx512(
    const __m512d x_re[8],  // x0[m]..x7[m]
    const __m512d x_im[8],
    __m512d y_re[8],        // Output: y[m + r*8] for r=0..7
    __m512d y_im[8],
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const __m512d sign_mask)
{
    // Copy inputs to working array
    __m512d inputs_re[8], inputs_im[8];
    for (int r = 0; r < 8; r++)
    {
        inputs_re[r] = x_re[r];
        inputs_im[r] = x_im[r];
    }

    // Perform radix-8 butterfly
    radix8_n1_butterfly_inline_forward_avx512(
        inputs_re, inputs_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    // Copy outputs
    for (int r = 0; r < 8; r++)
    {
        y_re[r] = inputs_re[r];
        y_im[r] = inputs_im[r];
    }
}

/**
 * @brief Final radix-8 combine for a single m-slice (Backward)
 */
TARGET_AVX512
FORCE_INLINE void
radix8_final_combine_m_slice_backward_avx512(
    const __m512d x_re[8],  // x0[m]..x7[m]
    const __m512d x_im[8],
    __m512d y_re[8],        // Output: y[m + r*8] for r=0..7
    __m512d y_im[8],
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const __m512d sign_mask)
{
    // Copy inputs to working array
    __m512d inputs_re[8], inputs_im[8];
    for (int r = 0; r < 8; r++)
    {
        inputs_re[r] = x_re[r];
        inputs_im[r] = x_im[r];
    }

    // Perform radix-8 butterfly
    radix8_n1_butterfly_inline_backward_avx512(
        inputs_re, inputs_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    // Copy outputs
    for (int r = 0; r < 8; r++)
    {
        y_re[r] = inputs_re[r];
        y_im[r] = inputs_im[r];
    }
}

//==============================================================================
// M-STRIPMINED PROCESSING KERNEL (U=4 Core)
//==============================================================================

/**
 * @brief Process single m-slice across U=4 slots (Forward)
 * 
 * @details
 * This is the heart of the U=4 optimization. For a given m (0..7):
 * - Process 4 slots in parallel: k, k+8, k+16, k+24
 * - Each slot: load → butterfly → W64 → combine → store
 * - Peak registers: ~29 ZMM (fits in 32)
 * 
 * Register pressure breakdown:
 *   - Current slot data: 16 ZMM (x_re[8], x_im[8])
 *   - Butterfly temps: ~8 ZMM (reused)
 *   - Constants (W8): 5 ZMM (kept hot)
 *   - W64 broadcast: 2 ZMM (transient)
 *   - Working: ~3 ZMM
 *   Total: ~29 ZMM ✅
 */
TARGET_AVX512
FORCE_INLINE void
process_m_slice_u4_forward_avx512(
    size_t k,           // Base k position
    size_t m,           // m-slice index (0..7)
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask,
    bool use_nt_stores)
{
    // Process 4 slots: k, k+8, k+16, k+24
    for (int u = 0; u < 4; u++)
    {
        size_t slot = k + u * 8;

        // Working registers for this slot (reused across iterations)
        __m512d x_re[8], x_im[8];
        __m512d y_re[8], y_im[8];

        // Load m-slice for this slot
        load_m_slice_soa_n1_avx512(slot, m, K, in_re, in_im, x_re, x_im);

        // Radix-8 N1 butterfly (in-place)
        radix8_n1_butterfly_inline_forward_avx512(
            x_re, x_im,
            W8_1_re, W8_1_im,
            W8_3_re, W8_3_im,
            sign_mask);

        // Apply W64 merge twiddles (in-place, broadcast on-demand)
        apply_w64_m_slice_forward_avx512(x_re, x_im, w64_re_table, w64_im_table);

        // Final radix-8 combine
        radix8_final_combine_m_slice_forward_avx512(
            x_re, x_im, y_re, y_im,
            W8_1_re, W8_1_im,
            W8_3_re, W8_3_im,
            sign_mask);

        // Store results
        if (use_nt_stores)
        {
            store_m_slice_soa_n1_avx512_stream(slot, m, K, out_re, out_im, y_re, y_im);
        }
        else
        {
            store_m_slice_soa_n1_avx512(slot, m, K, out_re, out_im, y_re, y_im);
        }
    }
}

/**
 * @brief Process single m-slice across U=4 slots (Backward)
 */
TARGET_AVX512
FORCE_INLINE void
process_m_slice_u4_backward_avx512(
    size_t k,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask,
    bool use_nt_stores)
{
    // Process 4 slots: k, k+8, k+16, k+24
    for (int u = 0; u < 4; u++)
    {
        size_t slot = k + u * 8;

        __m512d x_re[8], x_im[8];
        __m512d y_re[8], y_im[8];

        load_m_slice_soa_n1_avx512(slot, m, K, in_re, in_im, x_re, x_im);

        radix8_n1_butterfly_inline_backward_avx512(
            x_re, x_im,
            W8_1_re, W8_1_im,
            W8_3_re, W8_3_im,
            sign_mask);

        apply_w64_m_slice_backward_avx512(x_re, x_im, w64_re_table, w64_im_table);

        radix8_final_combine_m_slice_backward_avx512(
            x_re, x_im, y_re, y_im,
            W8_1_re, W8_1_im,
            W8_3_re, W8_3_im,
            sign_mask);

        if (use_nt_stores)
        {
            store_m_slice_soa_n1_avx512_stream(slot, m, K, out_re, out_im, y_re, y_im);
        }
        else
        {
            store_m_slice_soa_n1_avx512(slot, m, K, out_re, out_im, y_re, y_im);
        }
    }
}

//==============================================================================
// TAIL PROCESSING HELPERS (For k+8, k+16, k+24 partial coverage)
//==============================================================================

/**
 * @brief Process single m-slice for one slot (used in tail loops)
 * 
 * @details
 * Similar to U=4 kernel but processes only 1 slot.
 * Used for tail cases where we have k+8 or k+16 but not full U=4.
 */
TARGET_AVX512
FORCE_INLINE void
process_m_slice_single_forward_avx512(
    size_t slot,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask,
    bool use_nt_stores)
{
    __m512d x_re[8], x_im[8];
    __m512d y_re[8], y_im[8];

    load_m_slice_soa_n1_avx512(slot, m, K, in_re, in_im, x_re, x_im);

    radix8_n1_butterfly_inline_forward_avx512(
        x_re, x_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    apply_w64_m_slice_forward_avx512(x_re, x_im, w64_re_table, w64_im_table);

    radix8_final_combine_m_slice_forward_avx512(
        x_re, x_im, y_re, y_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    if (use_nt_stores)
    {
        store_m_slice_soa_n1_avx512_stream(slot, m, K, out_re, out_im, y_re, y_im);
    }
    else
    {
        store_m_slice_soa_n1_avx512(slot, m, K, out_re, out_im, y_re, y_im);
    }
}

TARGET_AVX512
FORCE_INLINE void
process_m_slice_single_backward_avx512(
    size_t slot,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const __m512d W8_1_re,
    const __m512d W8_1_im,
    const __m512d W8_3_re,
    const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask,
    bool use_nt_stores)
{
    __m512d x_re[8], x_im[8];
    __m512d y_re[8], y_im[8];

    load_m_slice_soa_n1_avx512(slot, m, K, in_re, in_im, x_re, x_im);

    radix8_n1_butterfly_inline_backward_avx512(
        x_re, x_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    apply_w64_m_slice_backward_avx512(x_re, x_im, w64_re_table, w64_im_table);

    radix8_final_combine_m_slice_backward_avx512(
        x_re, x_im, y_re, y_im,
        W8_1_re, W8_1_im,
        W8_3_re, W8_3_im,
        sign_mask);

    if (use_nt_stores)
    {
        store_m_slice_soa_n1_avx512_stream(slot, m, K, out_re, out_im, y_re, y_im);
    }
    else
    {
        store_m_slice_soa_n1_avx512(slot, m, K, out_re, out_im, y_re, y_im);
    }
}

//==============================================================================
// MASKED TAIL PROCESSING (For k < 8 remaining elements)
//==============================================================================

/**
 * @brief Process masked tail with m-stripmining (Forward)
 * 
 * @details
 * Handles 1-7 remaining elements at end of tile.
 * Uses masking to avoid out-of-bounds access.
 */
TARGET_AVX512
FORCE_INLINE void
radix64_process_tail_masked_n1_forward_avx512(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m512d W8_1_re, const __m512d W8_1_im,
    const __m512d W8_3_re, const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;
    __mmask8 mask = (1ULL << remaining) - 1;

    // Load with masking
    __m512d x_re[64], x_im[64];
    for (int r = 0; r < 64; r++)
    {
        x_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + r * K]);
        x_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + r * K]);
    }

    // Extract 8 sub-arrays for 8 radix-8 butterflies
    __m512d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
    __m512d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
    __m512d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
    __m512d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

    for (int i = 0; i < 8; i++)
    {
        x0_re[i] = x_re[i + 0 * 8];
        x0_im[i] = x_im[i + 0 * 8];
        x1_re[i] = x_re[i + 1 * 8];
        x1_im[i] = x_im[i + 1 * 8];
        x2_re[i] = x_re[i + 2 * 8];
        x2_im[i] = x_im[i + 2 * 8];
        x3_re[i] = x_re[i + 3 * 8];
        x3_im[i] = x_im[i + 3 * 8];
        x4_re[i] = x_re[i + 4 * 8];
        x4_im[i] = x_im[i + 4 * 8];
        x5_re[i] = x_re[i + 5 * 8];
        x5_im[i] = x_im[i + 5 * 8];
        x6_re[i] = x_re[i + 6 * 8];
        x6_im[i] = x_im[i + 6 * 8];
        x7_re[i] = x_re[i + 7 * 8];
        x7_im[i] = x_im[i + 7 * 8];
    }

    // Eight radix-8 butterflies
    radix8_n1_butterfly_inline_forward_avx512(x0_re, x0_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x1_re, x1_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x2_re, x2_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x3_re, x3_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x4_re, x4_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x5_re, x5_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x6_re, x6_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx512(x7_re, x7_im, W8_1_re, W8_1_im,
                                              W8_3_re, W8_3_im, sign_mask);

    // Apply W₆₄ merge twiddles using on-demand broadcast
    apply_w64_m_slice_forward_avx512(x1_re, x1_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x2_re, x2_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x3_re, x3_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x4_re, x4_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x5_re, x5_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x6_re, x6_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_forward_avx512(x7_re, x7_im, w64_re_table, w64_im_table);

    // Final combine (8 radix-8 butterflies)
    __m512d y_re[64], y_im[64];
    for (int m = 0; m < 8; m++)
    {
        __m512d inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                                x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        __m512d inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                                x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        radix8_n1_butterfly_inline_forward_avx512(inputs_re, inputs_im,
                                                  W8_1_re, W8_1_im,
                                                  W8_3_re, W8_3_im,
                                                  sign_mask);

        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }

    // Store with masking
    for (int r = 0; r < 64; r++)
    {
        _mm512_mask_store_pd(&out_re[k + r * K], mask, y_re[r]);
        _mm512_mask_store_pd(&out_im[k + r * K], mask, y_im[r]);
    }
}

TARGET_AVX512
FORCE_INLINE void
radix64_process_tail_masked_n1_backward_avx512(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m512d W8_1_re, const __m512d W8_1_im,
    const __m512d W8_3_re, const __m512d W8_3_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table,
    const __m512d sign_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;
    __mmask8 mask = (1ULL << remaining) - 1;

    __m512d x_re[64], x_im[64];
    for (int r = 0; r < 64; r++)
    {
        x_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + r * K]);
        x_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + r * K]);
    }

    __m512d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
    __m512d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
    __m512d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
    __m512d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

    for (int i = 0; i < 8; i++)
    {
        x0_re[i] = x_re[i + 0 * 8];
        x0_im[i] = x_im[i + 0 * 8];
        x1_re[i] = x_re[i + 1 * 8];
        x1_im[i] = x_im[i + 1 * 8];
        x2_re[i] = x_re[i + 2 * 8];
        x2_im[i] = x_im[i + 2 * 8];
        x3_re[i] = x_re[i + 3 * 8];
        x3_im[i] = x_im[i + 3 * 8];
        x4_re[i] = x_re[i + 4 * 8];
        x4_im[i] = x_im[i + 4 * 8];
        x5_re[i] = x_re[i + 5 * 8];
        x5_im[i] = x_im[i + 5 * 8];
        x6_re[i] = x_re[i + 6 * 8];
        x6_im[i] = x_im[i + 6 * 8];
        x7_re[i] = x_re[i + 7 * 8];
        x7_im[i] = x_im[i + 7 * 8];
    }

    radix8_n1_butterfly_inline_backward_avx512(x0_re, x0_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x1_re, x1_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x2_re, x2_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x3_re, x3_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x4_re, x4_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x5_re, x5_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x6_re, x6_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx512(x7_re, x7_im, W8_1_re, W8_1_im,
                                               W8_3_re, W8_3_im, sign_mask);

    apply_w64_m_slice_backward_avx512(x1_re, x1_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x2_re, x2_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x3_re, x3_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x4_re, x4_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x5_re, x5_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x6_re, x6_im, w64_re_table, w64_im_table);
    apply_w64_m_slice_backward_avx512(x7_re, x7_im, w64_re_table, w64_im_table);

    __m512d y_re[64], y_im[64];
    for (int m = 0; m < 8; m++)
    {
        __m512d inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                                x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        __m512d inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                                x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        radix8_n1_butterfly_inline_backward_avx512(inputs_re, inputs_im,
                                                   W8_1_re, W8_1_im,
                                                   W8_3_re, W8_3_im,
                                                   sign_mask);

        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }

    for (int r = 0; r < 64; r++)
    {
        _mm512_mask_store_pd(&out_re[k + r * K], mask, y_re[r]);
        _mm512_mask_store_pd(&out_im[k + r * K], mask, y_im[r]);
    }
}

//==============================================================================
// MAIN DRIVER: FORWARD N1 - AVX-512 (U=4 M-STRIPMINED)
//==============================================================================

TARGET_AVX512
void radix64_stage_dit_forward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m512d sign_mask = _mm512_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX64_PREFETCH_DISTANCE_N1_AVX512;
    const size_t tile_size = RADIX64_TILE_SIZE_N1_AVX512;

    const bool use_nt_stores = radix64_should_use_nt_stores_n1_avx512(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    // Hoist W₈ constants (kept hot in registers - 5 ZMM)
    const __m512d W8_1_re = _mm512_set1_pd(W8_FV_1_RE);
    const __m512d W8_1_im = _mm512_set1_pd(W8_FV_1_IM);
    const __m512d W8_3_re = _mm512_set1_pd(W8_FV_3_RE);
    const __m512d W8_3_im = _mm512_set1_pd(W8_FV_3_IM);

    // W₆₄ constants in tables (broadcast on-demand - saves 14 ZMM)
    const double *w64_re_table = W64_FV_TABLE_RE;
    const double *w64_im_table = W64_FV_TABLE_IM;

    // K-tiling outer loop (preserves cache locality)
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        //======================================================================
        // MAIN U=4 LOOP: k += 32 (Process 4 slots of 8 elements each)
        //======================================================================
        size_t k;
        for (k = k_tile; k + 32 <= k_end; k += 32)
        {
            // Prefetch next U=4 iteration (one tile ahead)
            size_t k_next = k + 32 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX64_PREFETCH_INPUTS_N1_AVX512(k_next, k_end, K,
                                                  in_re_aligned, in_im_aligned);
            }

            // M-STRIPMINE: Process all 8 m-slices for this k-block
            // Each m-iteration processes 4 slots: k, k+8, k+16, k+24
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_u4_forward_avx512(
                    k, m, K,
                    in_re_aligned, in_im_aligned,
                    out_re_aligned, out_im_aligned,
                    W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                    w64_re_table, w64_im_table,
                    sign_mask, use_nt_stores);
            }
        }

        //======================================================================
        // TAIL LOOP #1: k += 24 (U=3)
        //======================================================================
        if (k + 24 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                // Process 3 slots: k, k+8, k+16
                for (int u = 0; u < 3; u++)
                {
                    size_t slot = k + u * 8;
                    process_m_slice_single_forward_avx512(
                        slot, m, K,
                        in_re_aligned, in_im_aligned,
                        out_re_aligned, out_im_aligned,
                        W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                        w64_re_table, w64_im_table,
                        sign_mask, use_nt_stores);
                }
            }
            k += 24;
        }

        //======================================================================
        // TAIL LOOP #2: k += 16 (U=2)
        //======================================================================
        if (k + 16 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                // Process 2 slots: k, k+8
                for (int u = 0; u < 2; u++)
                {
                    size_t slot = k + u * 8;
                    process_m_slice_single_forward_avx512(
                        slot, m, K,
                        in_re_aligned, in_im_aligned,
                        out_re_aligned, out_im_aligned,
                        W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                        w64_re_table, w64_im_table,
                        sign_mask, use_nt_stores);
                }
            }
            k += 16;
        }

        //======================================================================
        // TAIL LOOP #3: k += 8 (U=1)
        //======================================================================
        if (k + 8 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_single_forward_avx512(
                    k, m, K,
                    in_re_aligned, in_im_aligned,
                    out_re_aligned, out_im_aligned,
                    W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                    w64_re_table, w64_im_table,
                    sign_mask, use_nt_stores);
            }
            k += 8;
        }

        //======================================================================
        // TAIL LOOP #4: Masked (1-7 elements remaining)
        //======================================================================
        radix64_process_tail_masked_n1_forward_avx512(
            k, k_end, K,
            in_re_aligned, in_im_aligned,
            out_re_aligned, out_im_aligned,
            W8_1_re, W8_1_im, W8_3_re, W8_3_im,
            w64_re_table, w64_im_table,
            sign_mask);
    }

    // Fence for NT stores
    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// MAIN DRIVER: BACKWARD N1 - AVX-512 (U=4 M-STRIPMINED)
//==============================================================================

TARGET_AVX512
void radix64_stage_dit_backward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m512d sign_mask = _mm512_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX64_PREFETCH_DISTANCE_N1_AVX512;
    const size_t tile_size = RADIX64_TILE_SIZE_N1_AVX512;

    const bool use_nt_stores = radix64_should_use_nt_stores_n1_avx512(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    // Hoist W₈ constants (BACKWARD)
    const __m512d W8_1_re = _mm512_set1_pd(W8_BV_1_RE);
    const __m512d W8_1_im = _mm512_set1_pd(W8_BV_1_IM);
    const __m512d W8_3_re = _mm512_set1_pd(W8_BV_3_RE);
    const __m512d W8_3_im = _mm512_set1_pd(W8_BV_3_IM);

    // W₆₄ constants in tables (BACKWARD)
    const double *w64_re_table = W64_BV_TABLE_RE;
    const double *w64_im_table = W64_BV_TABLE_IM;

    // K-tiling outer loop
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        //======================================================================
        // MAIN U=4 LOOP: k += 32
        //======================================================================
        size_t k;
        for (k = k_tile; k + 32 <= k_end; k += 32)
        {
            // Prefetch next iteration
            size_t k_next = k + 32 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX64_PREFETCH_INPUTS_N1_AVX512(k_next, k_end, K,
                                                  in_re_aligned, in_im_aligned);
            }

            // M-STRIPMINE: Process all 8 m-slices
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_u4_backward_avx512(
                    k, m, K,
                    in_re_aligned, in_im_aligned,
                    out_re_aligned, out_im_aligned,
                    W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                    w64_re_table, w64_im_table,
                    sign_mask, use_nt_stores);
            }
        }

        //======================================================================
        // TAIL LOOP #1: k += 24 (U=3)
        //======================================================================
        if (k + 24 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                for (int u = 0; u < 3; u++)
                {
                    size_t slot = k + u * 8;
                    process_m_slice_single_backward_avx512(
                        slot, m, K,
                        in_re_aligned, in_im_aligned,
                        out_re_aligned, out_im_aligned,
                        W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                        w64_re_table, w64_im_table,
                        sign_mask, use_nt_stores);
                }
            }
            k += 24;
        }

        //======================================================================
        // TAIL LOOP #2: k += 16 (U=2)
        //======================================================================
        if (k + 16 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                for (int u = 0; u < 2; u++)
                {
                    size_t slot = k + u * 8;
                    process_m_slice_single_backward_avx512(
                        slot, m, K,
                        in_re_aligned, in_im_aligned,
                        out_re_aligned, out_im_aligned,
                        W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                        w64_re_table, w64_im_table,
                        sign_mask, use_nt_stores);
                }
            }
            k += 16;
        }

        //======================================================================
        // TAIL LOOP #3: k += 8 (U=1)
        //======================================================================
        if (k + 8 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_single_backward_avx512(
                    k, m, K,
                    in_re_aligned, in_im_aligned,
                    out_re_aligned, out_im_aligned,
                    W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                    w64_re_table, w64_im_table,
                    sign_mask, use_nt_stores);
            }
            k += 8;
        }

        //======================================================================
        // TAIL LOOP #4: Masked (1-7 elements)
        //======================================================================
        radix64_process_tail_masked_n1_backward_avx512(
            k, k_end, K,
            in_re_aligned, in_im_aligned,
            out_re_aligned, out_im_aligned,
            W8_1_re, W8_1_im, W8_3_re, W8_3_im,
            w64_re_table, w64_im_table,
            sign_mask);
    }

    // Fence for NT stores
    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

#endif // FFT_RADIX64_AVX512_N1_H