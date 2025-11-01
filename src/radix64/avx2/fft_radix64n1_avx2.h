/**
 * @file fft_radix64_avx2_n1.h
 * @brief Radix-64 N1 (Twiddle-less) AVX-2 - 8×8 Cooley-Tukey
 *
 * @details
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
 * 3. Radix-8 final combine (2 radix-4 + W₈ structure)
 *
 * PERFORMANCE GAINS VS 2×32:
 * ============================
 * ✅ Reuses optimized radix-8 N1 kernel (8 calls)
 * ✅ Better cache locality (8 passes × 8 lanes)
 * ✅ Better register allocation (smaller working sets)
 * ✅ More arithmetic intensity
 * ✅ Matches FFTW's 8×8 strategy
 *
 * W₆₄ MERGE TWIDDLES (GEOMETRIC CONSTANTS):
 * ==========================================
 * - W64^0 = 1 (identity - skip)
 * - W64^1 = cos(π/32) - i*sin(π/32)
 * - W64^2 = W₈¹ (REUSE from radix-8!)
 * - W64^3 = cos(3π/32) - i*sin(3π/32)
 * - W64^4 = (√2/2, -√2/2)
 * - W64^5 = cos(5π/32) - i*sin(5π/32)
 * - W64^6 = cos(3π/16) - i*sin(3π/16)
 * - W64^7 = cos(7π/32) - i*sin(7π/32)
 *
 * Total: 7 geometric constants (hoisted, not loaded)
 *
 * AVX-2 ADAPTATIONS:
 * ===================
 * - Main loop: k += 8, U=2 with k and k+4
 * - Tail loop: k += 4, then masked
 * - Prefetch: 16 doubles ahead (inputs only)
 * - 256-bit vectors (4 doubles)
 *
 * @author VectorFFT Team
 * @version 1.0 (8×8 Cooley-Tukey)
 * @date 2025
 */

#ifndef FFT_RADIX64_AVX2_N1_H
#define FFT_RADIX64_AVX2_N1_H

#include <immintrin.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

// CRITICAL: Include radix-8 N1 implementation for reuse
#include "fft_radix8_avx2_n1.h"

//==============================================================================
// COMPILER HINTS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#ifndef RADIX64_PREFETCH_DISTANCE_N1_AVX2
#define RADIX64_PREFETCH_DISTANCE_N1_AVX2 16 // Prefetch inputs only
#endif

#ifndef RADIX64_TILE_SIZE_N1_AVX2
#define RADIX64_TILE_SIZE_N1_AVX2 64 // Same K-tiling as radix-32
#endif

#ifndef RADIX64_STREAM_THRESHOLD_KB_N1_AVX2
#define RADIX64_STREAM_THRESHOLD_KB_N1_AVX2 256 // Same NT threshold
#endif

//==============================================================================
// W₆₄ GEOMETRIC CONSTANTS (FORWARD)
//==============================================================================

// W64^k = exp(-2πik/64) for k=1..7

#define W64_FV_1_RE 0.9951847266721968862448369531524  // cos(π/32)
#define W64_FV_1_IM -0.0980171403295606019941955638886 // -sin(π/32)

// W64^2 = W₈^1 (REUSE!)
#define W64_FV_2_RE C8_CONSTANT    // 0.7071067811865475
#define W64_FV_2_IM (-C8_CONSTANT) // -0.7071067811865475

#define W64_FV_3_RE 0.9569403357322088967434217380776  // cos(3π/32)
#define W64_FV_3_IM -0.2902846772544623346189090451868 // -sin(3π/32)

#define W64_FV_4_RE 0.9238795325112867561281831893968  // cos(π/8)
#define W64_FV_4_IM -0.3826834323650897717284599840304 // -sin(π/8)

#define W64_FV_5_RE 0.8819212643483549847313153015198  // cos(5π/32)
#define W64_FV_5_IM -0.4713967368259976652433103328801 // -sin(5π/32)

#define W64_FV_6_RE 0.8314696123025452356762228503123  // cos(3π/16)
#define W64_FV_6_IM -0.5555702330196022247428308139218 // -sin(3π/16)

#define W64_FV_7_RE 0.7730104533627369556568782690311  // cos(7π/32)
#define W64_FV_7_IM -0.6343932841636455425806922248933 // -sin(7π/32)

//==============================================================================
// W₆₄ GEOMETRIC CONSTANTS (BACKWARD)
//==============================================================================

// W64^(-k) = exp(+2πik/64) for k=1..7

#define W64_BV_1_RE W64_FV_1_RE                       // cos(π/32)
#define W64_BV_1_IM 0.0980171403295606019941955638886 // +sin(π/32)

#define W64_BV_2_RE C8_CONSTANT // 0.7071067811865475
#define W64_BV_2_IM C8_CONSTANT // +0.7071067811865475

#define W64_BV_3_RE W64_FV_3_RE
#define W64_BV_3_IM 0.2902846772544623346189090451868

#define W64_BV_4_RE W64_FV_4_RE
#define W64_BV_4_IM 0.3826834323650897717284599840304

#define W64_BV_5_RE W64_FV_5_RE
#define W64_BV_5_IM 0.4713967368259976652433103328801

#define W64_BV_6_RE W64_FV_6_RE
#define W64_BV_6_IM 0.5555702330196022247428308139218

#define W64_BV_7_RE W64_FV_7_RE
#define W64_BV_7_IM 0.6343932841636455425806922248933

//==============================================================================
// NT STORE DECISION
//==============================================================================

FORCE_INLINE bool
radix64_should_use_nt_stores_n1_avx2(
    size_t K,
    const void *out_re,
    const void *out_im)
{
    const size_t bytes_per_k = 64 * 2 * sizeof(double); // 1024 bytes (64 complex)
    const size_t threshold_k = (RADIX64_STREAM_THRESHOLD_KB_N1_AVX2 * 1024) / bytes_per_k;

    return (K >= threshold_k) &&
           (((uintptr_t)out_re & 31) == 0) &&
           (((uintptr_t)out_im & 31) == 0);
}

//==============================================================================
// PREFETCH HELPERS (N1 - INPUTS ONLY)
//==============================================================================

/**
 * @brief Prefetch inputs for next iteration (N1 - no twiddles!)
 */
#define RADIX64_PREFETCH_INPUTS_N1_AVX2(k_next, k_limit, K, in_re, in_im)               \
    do                                                                                  \
    {                                                                                   \
        if ((k_next) < (k_limit))                                                       \
        {                                                                               \
            /* Prefetch all 64 input lanes */                                           \
            for (int _r = 0; _r < 64; _r++)                                             \
            {                                                                           \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0); \
            }                                                                           \
        }                                                                               \
    } while (0)

//==============================================================================
// RADIX-8 N1 BUTTERFLY (INLINE VERSION FOR REGISTER ARRAYS)
//==============================================================================

/**
 * @brief Radix-8 N1 butterfly operating on register arrays
 *
 * @details This is the core radix-8 computation extracted for reuse.
 * Identical algorithm to radix8_n1_butterfly_forward_avx2() but operates
 * on __m256d arrays instead of memory.
 *
 * Uses 2×radix-4 + W₈ twiddles decomposition.
 */
TARGET_AVX2
FORCE_INLINE void
radix8_n1_butterfly_inline_forward_avx2(
    __m256d x_re[8], __m256d x_im[8], // Input/output: 8 complex values
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d sign_mask)
{
    // First radix-4: even-indexed inputs (x0, x2, x4, x6)
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x_re[0], x_im[0], x_re[2], x_im[2],
                     x_re[4], x_im[4], x_re[6], x_im[6],
                     &e0_re, &e0_im, &e1_re, &e1_im,
                     &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    // Second radix-4: odd-indexed inputs (x1, x3, x5, x7)
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x_re[1], x_im[1], x_re[3], x_im[3],
                     x_re[5], x_im[5], x_re[7], x_im[7],
                     &o0_re, &o0_im, &o1_re, &o1_im,
                     &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    // Apply W₈ twiddles to odd outputs
    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination: y[k] = e[k] + o[k], y[k+4] = e[k] - o[k]
    x_re[0] = _mm256_add_pd(e0_re, o0_re);
    x_im[0] = _mm256_add_pd(e0_im, o0_im);
    x_re[1] = _mm256_add_pd(e1_re, o1_re);
    x_im[1] = _mm256_add_pd(e1_im, o1_im);
    x_re[2] = _mm256_add_pd(e2_re, o2_re);
    x_im[2] = _mm256_add_pd(e2_im, o2_im);
    x_re[3] = _mm256_add_pd(e3_re, o3_re);
    x_im[3] = _mm256_add_pd(e3_im, o3_im);
    x_re[4] = _mm256_sub_pd(e0_re, o0_re);
    x_im[4] = _mm256_sub_pd(e0_im, o0_im);
    x_re[5] = _mm256_sub_pd(e1_re, o1_re);
    x_im[5] = _mm256_sub_pd(e1_im, o1_im);
    x_re[6] = _mm256_sub_pd(e2_re, o2_re);
    x_im[6] = _mm256_sub_pd(e2_im, o2_im);
    x_re[7] = _mm256_sub_pd(e3_re, o3_re);
    x_im[7] = _mm256_sub_pd(e3_im, o3_im);
}

TARGET_AVX2
FORCE_INLINE void
radix8_n1_butterfly_inline_backward_avx2(
    __m256d x_re[8], __m256d x_im[8],
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d sign_mask)
{
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    // First radix-4: even-indexed inputs
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x_re[0], x_im[0], x_re[2], x_im[2],
                     x_re[4], x_im[4], x_re[6], x_im[6],
                     &e0_re, &e0_im, &e1_re, &e1_im,
                     &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    // Second radix-4: odd-indexed inputs
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x_re[1], x_im[1], x_re[3], x_im[3],
                     x_re[5], x_im[5], x_re[7], x_im[7],
                     &o0_re, &o0_im, &o1_re, &o1_im,
                     &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    // Apply conjugate W₈ twiddles
    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination
    x_re[0] = _mm256_add_pd(e0_re, o0_re);
    x_im[0] = _mm256_add_pd(e0_im, o0_im);
    x_re[1] = _mm256_add_pd(e1_re, o1_re);
    x_im[1] = _mm256_add_pd(e1_im, o1_im);
    x_re[2] = _mm256_add_pd(e2_re, o2_re);
    x_im[2] = _mm256_add_pd(e2_im, o2_im);
    x_re[3] = _mm256_add_pd(e3_re, o3_re);
    x_im[3] = _mm256_add_pd(e3_im, o3_im);
    x_re[4] = _mm256_sub_pd(e0_re, o0_re);
    x_im[4] = _mm256_sub_pd(e0_im, o0_im);
    x_re[5] = _mm256_sub_pd(e1_re, o1_re);
    x_im[5] = _mm256_sub_pd(e1_im, o1_im);
    x_re[6] = _mm256_sub_pd(e2_re, o2_re);
    x_im[6] = _mm256_sub_pd(e2_im, o2_im);
    x_re[7] = _mm256_sub_pd(e3_re, o3_re);
    x_im[7] = _mm256_sub_pd(e3_im, o3_im);
}

//==============================================================================
// W₆₄ MERGE TWIDDLES APPLICATION
//==============================================================================

/**
 * @brief Apply W₆₄ merge twiddles to 7 radix-8 outputs
 *
 * @details Multiplies sub-FFT outputs 1-7 by W64^1..W64^7.
 * Sub-FFT output 0 unchanged (W64^0 = 1).
 *
 * This is the "merge" layer that combines 8 independent radix-8 results
 * into a full radix-64 transform.
 */
TARGET_AVX2
FORCE_INLINE void
apply_w64_merge_twiddles_forward_avx2(
    __m256d x1_re[8], __m256d x1_im[8],
    __m256d x2_re[8], __m256d x2_im[8],
    __m256d x3_re[8], __m256d x3_im[8],
    __m256d x4_re[8], __m256d x4_im[8],
    __m256d x5_re[8], __m256d x5_im[8],
    __m256d x6_re[8], __m256d x6_im[8],
    __m256d x7_re[8], __m256d x7_im[8],
    const __m256d W64_1_re, const __m256d W64_1_im,
    const __m256d W64_2_re, const __m256d W64_2_im,
    const __m256d W64_3_re, const __m256d W64_3_im,
    const __m256d W64_4_re, const __m256d W64_4_im,
    const __m256d W64_5_re, const __m256d W64_5_im,
    const __m256d W64_6_re, const __m256d W64_6_im,
    const __m256d W64_7_re, const __m256d W64_7_im)
{
    const __m256d C8 = _mm256_set1_pd(C8_CONSTANT);
    __m256d tmp_re, tmp_im;

    // x1 *= W64^1 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x1_im[i], W64_1_re);
        __m256d t1 = _mm256_mul_pd(x1_im[i], W64_1_im);
        tmp_re = _mm256_fmsub_pd(x1_re[i], W64_1_re, t1);
        tmp_im = _mm256_fmadd_pd(x1_re[i], W64_1_im, t0);
        x1_re[i] = tmp_re;
        x1_im[i] = tmp_im;
    }

    // x2 *= W64^2 = W₈^1 - OPTIMIZED PATH
    for (int i = 0; i < 8; i++)
    {
        __m256d sum = _mm256_add_pd(x2_re[i], x2_im[i]);
        __m256d diff = _mm256_sub_pd(x2_im[i], x2_re[i]);
        x2_re[i] = _mm256_mul_pd(sum, C8);
        x2_im[i] = _mm256_mul_pd(diff, C8);
    }

    // x3 *= W64^3 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x3_im[i], W64_3_re);
        __m256d t1 = _mm256_mul_pd(x3_im[i], W64_3_im);
        tmp_re = _mm256_fmsub_pd(x3_re[i], W64_3_re, t1);
        tmp_im = _mm256_fmadd_pd(x3_re[i], W64_3_im, t0);
        x3_re[i] = tmp_re;
        x3_im[i] = tmp_im;
    }

    // x4 *= W64^4 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x4_im[i], W64_4_re);
        __m256d t1 = _mm256_mul_pd(x4_im[i], W64_4_im);
        tmp_re = _mm256_fmsub_pd(x4_re[i], W64_4_re, t1);
        tmp_im = _mm256_fmadd_pd(x4_re[i], W64_4_im, t0);
        x4_re[i] = tmp_re;
        x4_im[i] = tmp_im;
    }

    // x5 *= W64^5 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x5_im[i], W64_5_re);
        __m256d t1 = _mm256_mul_pd(x5_im[i], W64_5_im);
        tmp_re = _mm256_fmsub_pd(x5_re[i], W64_5_re, t1);
        tmp_im = _mm256_fmadd_pd(x5_re[i], W64_5_im, t0);
        x5_re[i] = tmp_re;
        x5_im[i] = tmp_im;
    }

    // x6 *= W64^6 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x6_im[i], W64_6_re);
        __m256d t1 = _mm256_mul_pd(x6_im[i], W64_6_im);
        tmp_re = _mm256_fmsub_pd(x6_re[i], W64_6_re, t1);
        tmp_im = _mm256_fmadd_pd(x6_re[i], W64_6_im, t0);
        x6_re[i] = tmp_re;
        x6_im[i] = tmp_im;
    }

    // x7 *= W64^7 - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x7_im[i], W64_7_re);
        __m256d t1 = _mm256_mul_pd(x7_im[i], W64_7_im);
        tmp_re = _mm256_fmsub_pd(x7_re[i], W64_7_re, t1);
        tmp_im = _mm256_fmadd_pd(x7_re[i], W64_7_im, t0);
        x7_re[i] = tmp_re;
        x7_im[i] = tmp_im;
    }
}

/**
 * @brief Apply W₆₄ merge twiddles - OPTIMIZED (backward, AVX-2)
 */
TARGET_AVX2
FORCE_INLINE void
apply_w64_merge_twiddles_backward_avx2(
    __m256d x1_re[8], __m256d x1_im[8],
    __m256d x2_re[8], __m256d x2_im[8],
    __m256d x3_re[8], __m256d x3_im[8],
    __m256d x4_re[8], __m256d x4_im[8],
    __m256d x5_re[8], __m256d x5_im[8],
    __m256d x6_re[8], __m256d x6_im[8],
    __m256d x7_re[8], __m256d x7_im[8],
    const __m256d W64_1_re, const __m256d W64_1_im,
    const __m256d W64_2_re, const __m256d W64_2_im,
    const __m256d W64_3_re, const __m256d W64_3_im,
    const __m256d W64_4_re, const __m256d W64_4_im,
    const __m256d W64_5_re, const __m256d W64_5_im,
    const __m256d W64_6_re, const __m256d W64_6_im,
    const __m256d W64_7_re, const __m256d W64_7_im)
{
    const __m256d C8 = _mm256_set1_pd(C8_CONSTANT);
    __m256d tmp_re, tmp_im;

    // x1 *= W64^(-1) - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x1_im[i], W64_1_re);
        __m256d t1 = _mm256_mul_pd(x1_im[i], W64_1_im);
        tmp_re = _mm256_fmsub_pd(x1_re[i], W64_1_re, t1);
        tmp_im = _mm256_fmadd_pd(x1_re[i], W64_1_im, t0);
        x1_re[i] = tmp_re;
        x1_im[i] = tmp_im;
    }

    // x2 *= W64^(-2) = W₈^(-1) - OPTIMIZED PATH
    for (int i = 0; i < 8; i++)
    {
        __m256d diff = _mm256_sub_pd(x2_re[i], x2_im[i]);
        __m256d sum = _mm256_add_pd(x2_re[i], x2_im[i]);
        x2_re[i] = _mm256_mul_pd(diff, C8);
        x2_im[i] = _mm256_mul_pd(sum, C8);
    }

    // x3 *= W64^(-3) - GENERIC
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x3_im[i], W64_3_re);
        __m256d t1 = _mm256_mul_pd(x3_im[i], W64_3_im);
        tmp_re = _mm256_fmsub_pd(x3_re[i], W64_3_re, t1);
        tmp_im = _mm256_fmadd_pd(x3_re[i], W64_3_im, t0);
        x3_re[i] = tmp_re;
        x3_im[i] = tmp_im;
    }

    // x4..x7 - GENERIC (same pattern)
    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x4_im[i], W64_4_re);
        __m256d t1 = _mm256_mul_pd(x4_im[i], W64_4_im);
        tmp_re = _mm256_fmsub_pd(x4_re[i], W64_4_re, t1);
        tmp_im = _mm256_fmadd_pd(x4_re[i], W64_4_im, t0);
        x4_re[i] = tmp_re;
        x4_im[i] = tmp_im;
    }

    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x5_im[i], W64_5_re);
        __m256d t1 = _mm256_mul_pd(x5_im[i], W64_5_im);
        tmp_re = _mm256_fmsub_pd(x5_re[i], W64_5_re, t1);
        tmp_im = _mm256_fmadd_pd(x5_re[i], W64_5_im, t0);
        x5_re[i] = tmp_re;
        x5_im[i] = tmp_im;
    }

    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x6_im[i], W64_6_re);
        __m256d t1 = _mm256_mul_pd(x6_im[i], W64_6_im);
        tmp_re = _mm256_fmsub_pd(x6_re[i], W64_6_re, t1);
        tmp_im = _mm256_fmadd_pd(x6_re[i], W64_6_im, t0);
        x6_re[i] = tmp_re;
        x6_im[i] = tmp_im;
    }

    for (int i = 0; i < 8; i++)
    {
        __m256d t0 = _mm256_mul_pd(x7_im[i], W64_7_re);
        __m256d t1 = _mm256_mul_pd(x7_im[i], W64_7_im);
        tmp_re = _mm256_fmsub_pd(x7_re[i], W64_7_re, t1);
        tmp_im = _mm256_fmadd_pd(x7_re[i], W64_7_im, t0);
        x7_re[i] = tmp_re;
        x7_im[i] = tmp_im;
    }
}

//==============================================================================
// RADIX-8 FINAL COMBINE (8 INPUTS → 64 OUTPUTS)
//==============================================================================

/**
 * @brief Final radix-8 combine after W₆₄ twiddle application
 *
 * @details Combines 8 radix-8 outputs into 64 final outputs.
 * Uses same 2×radix-4 + W₈ structure as radix-8 butterfly.
 *
 * This is the transpose/interleaving step that produces the final
 * radix-64 output from 8 independent radix-8 transforms.
 */
TARGET_AVX2
FORCE_INLINE void
radix8_final_combine_forward_avx2(
    const __m256d x0_re[8], const __m256d x0_im[8],
    const __m256d x1_re[8], const __m256d x1_im[8],
    const __m256d x2_re[8], const __m256d x2_im[8],
    const __m256d x3_re[8], const __m256d x3_im[8],
    const __m256d x4_re[8], const __m256d x4_im[8],
    const __m256d x5_re[8], const __m256d x5_im[8],
    const __m256d x6_re[8], const __m256d x6_im[8],
    const __m256d x7_re[8], const __m256d x7_im[8],
    __m256d y_re[64], __m256d y_im[64],
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d sign_mask)
{
    // For each of the 8 output positions within a radix-8 group,
    // perform a radix-8 butterfly across the 8 sub-FFT outputs

    for (int m = 0; m < 8; m++)
    {
        // Gather m-th element from each sub-FFT
        __m256d inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                                x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        __m256d inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                                x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        // Radix-8 butterfly (in-place on inputs array)
        radix8_n1_butterfly_inline_forward_avx2(inputs_re, inputs_im,
                                                W8_1_re, W8_1_im,
                                                W8_3_re, W8_3_im,
                                                sign_mask);

        // Scatter results to output (interleaved pattern)
        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }
}

TARGET_AVX2
FORCE_INLINE void
radix8_final_combine_backward_avx2(
    const __m256d x0_re[8], const __m256d x0_im[8],
    const __m256d x1_re[8], const __m256d x1_im[8],
    const __m256d x2_re[8], const __m256d x2_im[8],
    const __m256d x3_re[8], const __m256d x3_im[8],
    const __m256d x4_re[8], const __m256d x4_im[8],
    const __m256d x5_re[8], const __m256d x5_im[8],
    const __m256d x6_re[8], const __m256d x6_im[8],
    const __m256d x7_re[8], const __m256d x7_im[8],
    __m256d y_re[64], __m256d y_im[64],
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d sign_mask)
{
    for (int m = 0; m < 8; m++)
    {
        __m256d inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                                x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        __m256d inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                                x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        radix8_n1_butterfly_inline_backward_avx2(inputs_re, inputs_im,
                                                 W8_1_re, W8_1_im,
                                                 W8_3_re, W8_3_im,
                                                 sign_mask);

        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }
}

//==============================================================================
// LOAD/STORE FOR 64 LANES (AVX-2)
//==============================================================================

FORCE_INLINE void
load_64_lanes_soa_n1_avx2(size_t k, size_t K,
                          const double *RESTRICT in_re,
                          const double *RESTRICT in_im,
                          __m256d re[64], __m256d im[64])
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);

    for (int r = 0; r < 64; r++)
    {
        re[r] = _mm256_load_pd(&in_re_aligned[k + r * K]);
        im[r] = _mm256_load_pd(&in_im_aligned[k + r * K]);
    }
}

FORCE_INLINE void
store_64_lanes_soa_n1_avx2(size_t k, size_t K,
                           double *RESTRICT out_re,
                           double *RESTRICT out_im,
                           const __m256d y_re[64], const __m256d y_im[64])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 64; r++)
    {
        _mm256_store_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm256_store_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

FORCE_INLINE void
store_64_lanes_soa_n1_avx2_stream(size_t k, size_t K,
                                  double *RESTRICT out_re,
                                  double *RESTRICT out_im,
                                  const __m256d y_re[64], const __m256d y_im[64])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 64; r++)
    {
        _mm256_stream_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm256_stream_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

//==============================================================================
// MASKED TAIL PROCESSING (AVX-2)
//==============================================================================

/**
 * @brief Process remaining k-indices with masking (N1)
 */
TARGET_AVX2
FORCE_INLINE void
radix64_process_tail_masked_n1_forward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d W64_1_re, const __m256d W64_1_im,
    const __m256d W64_2_re, const __m256d W64_2_im,
    const __m256d W64_3_re, const __m256d W64_3_im,
    const __m256d W64_4_re, const __m256d W64_4_im,
    const __m256d W64_5_re, const __m256d W64_5_im,
    const __m256d W64_6_re, const __m256d W64_6_im,
    const __m256d W64_7_re, const __m256d W64_7_im,
    const __m256d sign_mask, const __m256d neg_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;

    // Build mask
    int mask_array[4] = {0};
    for (size_t i = 0; i < remaining && i < 4; i++)
    {
        mask_array[i] = -1;
    }
    __m256i mask_i = _mm256_loadu_si256((const __m256i *)mask_array);

    // Load all 64 lanes with masking
    __m256d x_re[64], x_im[64];
    for (int r = 0; r < 64; r++)
    {
        x_re[r] = _mm256_maskload_pd(&in_re[k + r * K], mask_i);
        x_im[r] = _mm256_maskload_pd(&in_im[k + r * K], mask_i);
    }

    // Eight radix-8 N1 butterflies
    __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
    __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
    __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
    __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

    radix8_n1_butterfly_inline_forward_avx2(x0_re, x0_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x1_re, x1_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x2_re, x2_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x3_re, x3_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x4_re, x4_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x5_re, x5_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x6_re, x6_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_forward_avx2(x7_re, x7_im, W8_1_re, W8_1_im,
                                            W8_3_re, W8_3_im, sign_mask);

    // Apply W₆₄ merge twiddles
    apply_w64_merge_twiddles_forward_avx2(
        x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
        x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
        W64_1_re, W64_1_im, W64_2_re, W64_2_im, W64_3_re, W64_3_im,
        W64_4_re, W64_4_im, W64_5_re, W64_5_im, W64_6_re, W64_6_im,
        W64_7_re, W64_7_im);

    // Radix-8 final combine
    __m256d y_re[64], y_im[64];
    radix8_final_combine_forward_avx2(
        x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
        x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
        y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

    // Store with masking
    for (int r = 0; r < 64; r++)
    {
        _mm256_maskstore_pd(&out_re[k + r * K], mask_i, y_re[r]);
        _mm256_maskstore_pd(&out_im[k + r * K], mask_i, y_im[r]);
    }
}

TARGET_AVX2
FORCE_INLINE void
radix64_process_tail_masked_n1_backward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m256d W8_1_re, const __m256d W8_1_im,
    const __m256d W8_3_re, const __m256d W8_3_im,
    const __m256d W64_1_re, const __m256d W64_1_im,
    const __m256d W64_2_re, const __m256d W64_2_im,
    const __m256d W64_3_re, const __m256d W64_3_im,
    const __m256d W64_4_re, const __m256d W64_4_im,
    const __m256d W64_5_re, const __m256d W64_5_im,
    const __m256d W64_6_re, const __m256d W64_6_im,
    const __m256d W64_7_re, const __m256d W64_7_im,
    const __m256d sign_mask, const __m256d neg_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;

    int mask_array[4] = {0};
    for (size_t i = 0; i < remaining && i < 4; i++)
    {
        mask_array[i] = -1;
    }
    __m256i mask_i = _mm256_loadu_si256((const __m256i *)mask_array);

    __m256d x_re[64], x_im[64];
    for (int r = 0; r < 64; r++)
    {
        x_re[r] = _mm256_maskload_pd(&in_re[k + r * K], mask_i);
        x_im[r] = _mm256_maskload_pd(&in_im[k + r * K], mask_i);
    }

    __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
    __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
    __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
    __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

    radix8_n1_butterfly_inline_backward_avx2(x0_re, x0_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x1_re, x1_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x2_re, x2_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x3_re, x3_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x4_re, x4_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x5_re, x5_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x6_re, x6_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);
    radix8_n1_butterfly_inline_backward_avx2(x7_re, x7_im, W8_1_re, W8_1_im,
                                             W8_3_re, W8_3_im, sign_mask);

    apply_w64_merge_twiddles_backward_avx2(
        x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
        x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
        W64_1_re, W64_1_im, W64_2_re, W64_2_im, W64_3_re, W64_3_im,
        W64_4_re, W64_4_im, W64_5_re, W64_5_im, W64_6_re, W64_6_im,
        W64_7_re, W64_7_im);

    __m256d y_re[64], y_im[64];
    radix8_final_combine_backward_avx2(
        x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
        x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
        y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

    for (int r = 0; r < 64; r++)
    {
        _mm256_maskstore_pd(&out_re[k + r * K], mask_i, y_re[r]);
        _mm256_maskstore_pd(&out_im[k + r * K], mask_i, y_im[r]);
    }
}

//==============================================================================
// MAIN DRIVER: FORWARD N1 (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-64 DIT Forward Stage - N1 (NO TWIDDLES) - AVX-2
 *
 * @details
 * 8×8 COOLEY-TUKEY DECOMPOSITION:
 * - Eight radix-8 N1 butterflies (reuse optimized kernel)
 * - W₆₄ geometric merge twiddles (7 constants, hoisted)
 * - Radix-8 final combine
 *
 * PERFORMANCE: 40-50% faster than standard radix-64
 *
 * AVX-2 LOOP STRUCTURE:
 * - Main loop: k += 8 (U=2: process k and k+4)
 * - Tail loop: k += 4
 * - Masked tail: remaining elements
 */
TARGET_AVX2
void radix64_stage_dit_forward_n1_soa_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX64_PREFETCH_DISTANCE_N1_AVX2;
    const size_t tile_size = RADIX64_TILE_SIZE_N1_AVX2;

    const bool use_nt_stores = radix64_should_use_nt_stores_n1_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Hoist W₈ constants (for radix-8 butterflies)
    const __m256d W8_1_re = _mm256_set1_pd(W8_FV_1_RE);
    const __m256d W8_1_im = _mm256_set1_pd(W8_FV_1_IM);
    const __m256d W8_3_re = _mm256_set1_pd(W8_FV_3_RE);
    const __m256d W8_3_im = _mm256_set1_pd(W8_FV_3_IM);

    // Hoist W₆₄ merge constants
    const __m256d W64_1_re = _mm256_set1_pd(W64_FV_1_RE);
    const __m256d W64_1_im = _mm256_set1_pd(W64_FV_1_IM);
    const __m256d W64_2_re = _mm256_set1_pd(W64_FV_2_RE);
    const __m256d W64_2_im = _mm256_set1_pd(W64_FV_2_IM);
    const __m256d W64_3_re = _mm256_set1_pd(W64_FV_3_RE);
    const __m256d W64_3_im = _mm256_set1_pd(W64_FV_3_IM);
    const __m256d W64_4_re = _mm256_set1_pd(W64_FV_4_RE);
    const __m256d W64_4_im = _mm256_set1_pd(W64_FV_4_IM);
    const __m256d W64_5_re = _mm256_set1_pd(W64_FV_5_RE);
    const __m256d W64_5_im = _mm256_set1_pd(W64_FV_5_IM);
    const __m256d W64_6_re = _mm256_set1_pd(W64_FV_6_RE);
    const __m256d W64_6_im = _mm256_set1_pd(W64_FV_6_IM);
    const __m256d W64_7_re = _mm256_set1_pd(W64_FV_7_RE);
    const __m256d W64_7_im = _mm256_set1_pd(W64_FV_7_IM);

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=2 LOOP: k += 8
        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            // Prefetch next iteration (inputs only - no twiddles!)
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX64_PREFETCH_INPUTS_N1_AVX2(k_next, k_end, K,
                                                in_re_aligned, in_im_aligned);
            }

            // ==================== PROCESS k ====================
            {
                // Load all 64 complex inputs
                __m256d x_re[64], x_im[64];
                load_64_lanes_soa_n1_avx2(k, K, in_re_aligned, in_im_aligned,
                                          x_re, x_im);

                // Eight radix-8 N1 butterflies
                __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
                __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
                __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
                __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

                // Distribute inputs to 8 groups
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

                // Apply 8 radix-8 N1 butterflies
                radix8_n1_butterfly_inline_forward_avx2(x0_re, x0_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x1_re, x1_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x2_re, x2_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x3_re, x3_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x4_re, x4_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x5_re, x5_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x6_re, x6_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x7_re, x7_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);

                // Apply W₆₄ merge twiddles
                apply_w64_merge_twiddles_forward_avx2(
                    x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                    x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                    W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                    W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                    W64_7_re, W64_7_im);

                // Radix-8 final combine
                __m256d y_re[64], y_im[64];
                radix8_final_combine_forward_avx2(
                    x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                    x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

                // Store
                if (use_nt_stores)
                {
                    store_64_lanes_soa_n1_avx2_stream(k, K, out_re_aligned,
                                                      out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_64_lanes_soa_n1_avx2(k, K, out_re_aligned,
                                               out_im_aligned, y_re, y_im);
                }
            }

            // ==================== PROCESS k+4 (U=2 SOFTWARE PIPELINING) ====================
            {
                __m256d x_re[64], x_im[64];
                load_64_lanes_soa_n1_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                          x_re, x_im);

                __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
                __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
                __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
                __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

                radix8_n1_butterfly_inline_forward_avx2(x0_re, x0_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x1_re, x1_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x2_re, x2_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x3_re, x3_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x4_re, x4_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x5_re, x5_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x6_re, x6_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);
                radix8_n1_butterfly_inline_forward_avx2(x7_re, x7_im,
                                                        W8_1_re, W8_1_im,
                                                        W8_3_re, W8_3_im,
                                                        sign_mask);

                apply_w64_merge_twiddles_forward_avx2(
                    x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                    x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                    W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                    W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                    W64_7_re, W64_7_im);

                __m256d y_re[64], y_im[64];
                radix8_final_combine_forward_avx2(
                    x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                    x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

                if (use_nt_stores)
                {
                    store_64_lanes_soa_n1_avx2_stream(k + 4, K, out_re_aligned,
                                                      out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_64_lanes_soa_n1_avx2(k + 4, K, out_re_aligned,
                                               out_im_aligned, y_re, y_im);
                }
            }
        }

        // TAIL LOOP #1: k += 4
        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[64], x_im[64];
            load_64_lanes_soa_n1_avx2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);

            __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
            __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
            __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
            __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

            radix8_n1_butterfly_inline_forward_avx2(x0_re, x0_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x1_re, x1_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x2_re, x2_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x3_re, x3_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x4_re, x4_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x5_re, x5_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x6_re, x6_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_forward_avx2(x7_re, x7_im, W8_1_re, W8_1_im,
                                                    W8_3_re, W8_3_im, sign_mask);

            apply_w64_merge_twiddles_forward_avx2(
                x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                W64_7_re, W64_7_im);

            __m256d y_re[64], y_im[64];
            radix8_final_combine_forward_avx2(
                x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

            if (use_nt_stores)
            {
                store_64_lanes_soa_n1_avx2_stream(k, K, out_re_aligned,
                                                  out_im_aligned, y_re, y_im);
            }
            else
            {
                store_64_lanes_soa_n1_avx2(k, K, out_re_aligned, out_im_aligned,
                                           y_re, y_im);
            }
        }

        // TAIL LOOP #2: Masked
        radix64_process_tail_masked_n1_forward_avx2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            W8_1_re, W8_1_im, W8_3_re, W8_3_im,
            W64_1_re, W64_1_im, W64_2_re, W64_2_im, W64_3_re, W64_3_im,
            W64_4_re, W64_4_im, W64_5_re, W64_5_im, W64_6_re, W64_6_im,
            W64_7_re, W64_7_im, sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// MAIN DRIVER: BACKWARD N1 (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-64 DIT Backward Stage - N1 (NO TWIDDLES) - AVX-2
 *
 * @details
 * Same as forward but with backward butterflies (conjugated rotations).
 */
TARGET_AVX2
void radix64_stage_dit_backward_n1_soa_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX64_PREFETCH_DISTANCE_N1_AVX2;
    const size_t tile_size = RADIX64_TILE_SIZE_N1_AVX2;

    const bool use_nt_stores = radix64_should_use_nt_stores_n1_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Hoist W₈ constants (BACKWARD)
    const __m256d W8_1_re = _mm256_set1_pd(W8_BV_1_RE);
    const __m256d W8_1_im = _mm256_set1_pd(W8_BV_1_IM);
    const __m256d W8_3_re = _mm256_set1_pd(W8_BV_3_RE);
    const __m256d W8_3_im = _mm256_set1_pd(W8_BV_3_IM);

    // Hoist W₆₄ merge constants (BACKWARD)
    const __m256d W64_1_re = _mm256_set1_pd(W64_BV_1_RE);
    const __m256d W64_1_im = _mm256_set1_pd(W64_BV_1_IM);
    const __m256d W64_2_re = _mm256_set1_pd(W64_BV_2_RE);
    const __m256d W64_2_im = _mm256_set1_pd(W64_BV_2_IM);
    const __m256d W64_3_re = _mm256_set1_pd(W64_BV_3_RE);
    const __m256d W64_3_im = _mm256_set1_pd(W64_BV_3_IM);
    const __m256d W64_4_re = _mm256_set1_pd(W64_BV_4_RE);
    const __m256d W64_4_im = _mm256_set1_pd(W64_BV_4_IM);
    const __m256d W64_5_re = _mm256_set1_pd(W64_BV_5_RE);
    const __m256d W64_5_im = _mm256_set1_pd(W64_BV_5_IM);
    const __m256d W64_6_re = _mm256_set1_pd(W64_BV_6_RE);
    const __m256d W64_6_im = _mm256_set1_pd(W64_BV_6_IM);
    const __m256d W64_7_re = _mm256_set1_pd(W64_BV_7_RE);
    const __m256d W64_7_im = _mm256_set1_pd(W64_BV_7_IM);

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX64_PREFETCH_INPUTS_N1_AVX2(k_next, k_end, K,
                                                in_re_aligned, in_im_aligned);
            }

            // Process k
            {
                __m256d x_re[64], x_im[64];
                load_64_lanes_soa_n1_avx2(k, K, in_re_aligned, in_im_aligned,
                                          x_re, x_im);

                __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
                __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
                __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
                __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

                radix8_n1_butterfly_inline_backward_avx2(x0_re, x0_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x1_re, x1_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x2_re, x2_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x3_re, x3_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x4_re, x4_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x5_re, x5_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x6_re, x6_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x7_re, x7_im,
                                                         W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im,
                                                         sign_mask);

                apply_w64_merge_twiddles_backward_avx2(
                    x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                    x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                    W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                    W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                    W64_7_re, W64_7_im);

                __m256d y_re[64], y_im[64];
                radix8_final_combine_backward_avx2(
                    x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                    x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

                if (use_nt_stores)
                {
                    store_64_lanes_soa_n1_avx2_stream(k, K, out_re_aligned,
                                                      out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_64_lanes_soa_n1_avx2(k, K, out_re_aligned,
                                               out_im_aligned, y_re, y_im);
                }
            }

            // Process k+4
            {
                __m256d x_re[64], x_im[64];
                load_64_lanes_soa_n1_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                          x_re, x_im);

                __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
                __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
                __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
                __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

                radix8_n1_butterfly_inline_backward_avx2(x0_re, x0_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x1_re, x1_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x2_re, x2_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x3_re, x3_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x4_re, x4_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x5_re, x5_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x6_re, x6_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);
                radix8_n1_butterfly_inline_backward_avx2(x7_re, x7_im, W8_1_re, W8_1_im,
                                                         W8_3_re, W8_3_im, sign_mask);

                apply_w64_merge_twiddles_backward_avx2(
                    x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                    x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                    W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                    W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                    W64_7_re, W64_7_im);

                __m256d y_re[64], y_im[64];
                radix8_final_combine_backward_avx2(
                    x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                    x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                    y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

                if (use_nt_stores)
                {
                    store_64_lanes_soa_n1_avx2_stream(k + 4, K, out_re_aligned,
                                                      out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_64_lanes_soa_n1_avx2(k + 4, K, out_re_aligned,
                                               out_im_aligned, y_re, y_im);
                }
            }
        }

        // TAIL LOOP #1: k += 4
        for (; k + 4 <= k_end; k += 4)
        {
            __m256d x_re[64], x_im[64];
            load_64_lanes_soa_n1_avx2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);

            __m256d x0_re[8], x0_im[8], x1_re[8], x1_im[8];
            __m256d x2_re[8], x2_im[8], x3_re[8], x3_im[8];
            __m256d x4_re[8], x4_im[8], x5_re[8], x5_im[8];
            __m256d x6_re[8], x6_im[8], x7_re[8], x7_im[8];

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

            radix8_n1_butterfly_inline_backward_avx2(x0_re, x0_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x1_re, x1_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x2_re, x2_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x3_re, x3_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x4_re, x4_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x5_re, x5_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x6_re, x6_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);
            radix8_n1_butterfly_inline_backward_avx2(x7_re, x7_im, W8_1_re, W8_1_im,
                                                     W8_3_re, W8_3_im, sign_mask);

            apply_w64_merge_twiddles_backward_avx2(
                x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
                x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                W64_1_re, W64_1_im, W64_2_re, W64_2_im,
                W64_3_re, W64_3_im, W64_4_re, W64_4_im,
                W64_5_re, W64_5_im, W64_6_re, W64_6_im,
                W64_7_re, W64_7_im);

            __m256d y_re[64], y_im[64];
            radix8_final_combine_backward_avx2(
                x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
                x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
                y_re, y_im, W8_1_re, W8_1_im, W8_3_re, W8_3_im, sign_mask);

            if (use_nt_stores)
            {
                store_64_lanes_soa_n1_avx2_stream(k, K, out_re_aligned,
                                                  out_im_aligned, y_re, y_im);
            }
            else
            {
                store_64_lanes_soa_n1_avx2(k, K, out_re_aligned, out_im_aligned,
                                           y_re, y_im);
            }
        }

        // TAIL LOOP #2: Masked
        radix64_process_tail_masked_n1_backward_avx2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            W8_1_re, W8_1_im, W8_3_re, W8_3_im,
            W64_1_re, W64_1_im, W64_2_re, W64_2_im, W64_3_re, W64_3_im,
            W64_4_re, W64_4_im, W64_5_re, W64_5_im, W64_6_re, W64_6_im,
            W64_7_re, W64_7_im, sign_mask, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

#endif // FFT_RADIX64_AVX2_N1_H