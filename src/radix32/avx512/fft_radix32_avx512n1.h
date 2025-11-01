/**
 * @file fft_radix32_avx512_n1.h
 * @brief Radix-32 N1 (No Twiddles) AVX-512 Implementation - Forward/Backward
 *
 * @details
 * CRITICAL OPTIMIZATION FOR FIRST STAGE (ALL TWIDDLES = 1+0i)
 * ============================================================
 * 
 * This is the twiddle-less variant for radix-32 first-stage transforms.
 * When all stage twiddles and merge twiddles equal 1+0i, we can skip
 * ALL complex multiplications and achieve 40-60% speedup.
 * 
 * ARCHITECTURE: 2×16 COOLEY-TUKEY (SIMPLIFIED)
 * =============================================
 * 1. Even half (r=0..15): Radix-16 butterfly (NO twiddles)
 * 2. Odd half (r=16..31): Radix-16 butterfly (NO twiddles)
 * 3. Radix-2 combine: even ± odd (NO merge twiddles)
 * 
 * PERFORMANCE GAINS:
 * ==================
 * - Skip 30 cmuls (15 per radix-16 × 2)
 * - Skip 16 cmuls (merge twiddles)
 * - Total: 46 complex multiplications eliminated
 * - Simpler code: no recurrence, no BLOCKED modes
 * - Expected speedup: 40-60% vs. standard version
 * 
 * AVX-512 ADAPTATIONS:
 * ====================
 * - Main loop: k += 16, U=2 with k and k+8
 * - Tail loop: k += 8, then masked
 * - No twiddle prefetch (they don't exist!)
 * - Simpler prefetch pattern (inputs only)
 * - 8-wide vectors (512-bit)
 * 
 * @author VectorFFT Team
 * @version 1.0 (N1 - No Twiddles)
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX512_N1_H
#define FFT_RADIX32_AVX512_N1_H

#include <immintrin.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// CRITICAL: Include radix-16 N1 implementation
#include "fft_radix16_avx512_n1.h"

//==============================================================================
// COMPILER HINTS
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
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX512
#endif

//==============================================================================
// CONFIGURATION (N1 - SIMPLIFIED)
//==============================================================================

#ifndef RADIX32_PREFETCH_DISTANCE_N1_AVX512
#define RADIX32_PREFETCH_DISTANCE_N1_AVX512 32 // Prefetch inputs only
#endif

#ifndef RADIX32_TILE_SIZE_N1_AVX512
#define RADIX32_TILE_SIZE_N1_AVX512 64 // Same K-tiling
#endif

#ifndef RADIX32_STREAM_THRESHOLD_KB_N1_AVX512
#define RADIX32_STREAM_THRESHOLD_KB_N1_AVX512 256 // Same NT threshold
#endif

//==============================================================================
// NT STORE DECISION (N1)
//==============================================================================

FORCE_INLINE bool
radix32_should_use_nt_stores_n1_avx512(
    size_t K,
    const void *out_re,
    const void *out_im)
{
    const size_t bytes_per_k = 32 * 2 * sizeof(double); // 512 bytes
    const size_t threshold_k = (RADIX32_STREAM_THRESHOLD_KB_N1_AVX512 * 1024) / bytes_per_k;

    return (K >= threshold_k) &&
           (((uintptr_t)out_re & 63) == 0) &&
           (((uintptr_t)out_im & 63) == 0);
}

//==============================================================================
// PREFETCH HELPERS (N1 - INPUTS ONLY)
//==============================================================================

/**
 * @brief Prefetch inputs for next iteration (N1 - no twiddles!)
 */
#define RADIX32_PREFETCH_INPUTS_N1_AVX512(k_next, k_limit, K, in_re, in_im)    \
    do                                                                          \
    {                                                                           \
        if ((k_next) < (k_limit))                                               \
        {                                                                       \
            /* Prefetch even half (r=0..15) */                                  \
            for (int _r = 0; _r < 16; _r++)                                     \
            {                                                                   \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0); \
            }                                                                   \
            /* Prefetch odd half (r=16..31) */                                  \
            for (int _r = 16; _r < 32; _r++)                                    \
            {                                                                   \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0); \
            }                                                                   \
        }                                                                       \
    } while (0)

//==============================================================================
// LOAD/STORE FOR 32 LANES (AVX-512)
//==============================================================================

/**
 * @brief Load 16 complex values (AVX-512)
 * Used for both even and odd halves
 */
FORCE_INLINE void
load_16_lanes_soa_n1_avx512(size_t k, size_t K,
                            const double *RESTRICT in_re,
                            const double *RESTRICT in_im,
                            __m512d re[16], __m512d im[16])
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);

    for (int r = 0; r < 16; r++)
    {
        re[r] = _mm512_load_pd(&in_re_aligned[k + r * K]);
        im[r] = _mm512_load_pd(&in_im_aligned[k + r * K]);
    }
}

/**
 * @brief Store 32 complex values (AVX-512)
 */
FORCE_INLINE void
store_32_lanes_soa_n1_avx512(size_t k, size_t K,
                             double *RESTRICT out_re,
                             double *RESTRICT out_im,
                             const __m512d y_re[32], const __m512d y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (int r = 0; r < 32; r++)
    {
        _mm512_store_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm512_store_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

/**
 * @brief Store with non-temporal hint (AVX-512)
 */
FORCE_INLINE void
store_32_lanes_soa_n1_avx512_stream(size_t k, size_t K,
                                    double *RESTRICT out_re,
                                    double *RESTRICT out_im,
                                    const __m512d y_re[32], const __m512d y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (int r = 0; r < 32; r++)
    {
        _mm512_stream_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm512_stream_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

//==============================================================================
// RADIX-2 BUTTERFLY (NATIVE SOA, AVX-512)
//==============================================================================

/**
 * @brief Radix-2 butterfly for combining even/odd halves (AVX-512)
 *
 * @details After radix-16 sub-FFTs (NO twiddles applied):
 *   out[m]    = even[m] + odd[m]
 *   out[m+16] = even[m] - odd[m]
 */
FORCE_INLINE void
radix2_butterfly_combine_soa_n1_avx512(
    const __m512d even_re[16], const __m512d even_im[16],
    const __m512d odd_re[16], const __m512d odd_im[16],
    __m512d out_re[32], __m512d out_im[32])
{
    // First half: out[0..15] = even + odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m] = _mm512_add_pd(even_re[m], odd_re[m]);
        out_im[m] = _mm512_add_pd(even_im[m], odd_im[m]);
    }

    // Second half: out[16..31] = even - odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m + 16] = _mm512_sub_pd(even_re[m], odd_re[m]);
        out_im[m + 16] = _mm512_sub_pd(even_im[m], odd_im[m]);
    }
}

//==============================================================================
// MASKED TAIL PROCESSING (AVX-512)
//==============================================================================

/**
 * @brief Process remaining k-indices with masking (N1)
 */
TARGET_AVX512
FORCE_INLINE void
radix32_process_tail_masked_n1_forward_avx512(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m512d neg_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;
    const __m512d rot_sign_mask = _mm512_set1_pd(-0.0);

    // Build mask for remaining elements
    __mmask8 mask = (__mmask8)((1U << remaining) - 1U);

    // Load even half with masking
    __m512d even_re[16], even_im[16];
    for (int r = 0; r < 16; r++)
    {
        even_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + r * K]);
        even_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + r * K]);
    }

    // Radix-16 butterfly (NO twiddles!)
    radix16_complete_butterfly_forward_soa_avx512(even_re, even_im,
                                                  rot_sign_mask, neg_mask);

    // Load odd half with masking
    __m512d odd_re[16], odd_im[16];
    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + (r + 16) * K]);
        odd_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + (r + 16) * K]);
    }

    // Radix-16 butterfly (NO twiddles!)
    radix16_complete_butterfly_forward_soa_avx512(odd_re, odd_im,
                                                  rot_sign_mask, neg_mask);

    // NO MERGE TWIDDLES (all = 1)

    // Radix-2 combine
    __m512d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                           y_re, y_im);

    // Store with masking
    for (int r = 0; r < 32; r++)
    {
        _mm512_mask_store_pd(&out_re[k + r * K], mask, y_re[r]);
        _mm512_mask_store_pd(&out_im[k + r * K], mask, y_im[r]);
    }
}

TARGET_AVX512
FORCE_INLINE void
radix32_process_tail_masked_n1_backward_avx512(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const __m512d neg_mask)
{
    if (k >= k_end)
        return;

    const size_t remaining = k_end - k;
    const __m512d rot_sign_mask = _mm512_set1_pd(-0.0);

    __mmask8 mask = (__mmask8)((1U << remaining) - 1U);

    __m512d even_re[16], even_im[16];
    for (int r = 0; r < 16; r++)
    {
        even_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + r * K]);
        even_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + r * K]);
    }

    radix16_complete_butterfly_backward_soa_avx512(even_re, even_im,
                                                   rot_sign_mask, neg_mask);

    __m512d odd_re[16], odd_im[16];
    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm512_maskz_load_pd(mask, &in_re[k + (r + 16) * K]);
        odd_im[r] = _mm512_maskz_load_pd(mask, &in_im[k + (r + 16) * K]);
    }

    radix16_complete_butterfly_backward_soa_avx512(odd_re, odd_im,
                                                   rot_sign_mask, neg_mask);

    __m512d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                           y_re, y_im);

    for (int r = 0; r < 32; r++)
    {
        _mm512_mask_store_pd(&out_re[k + r * K], mask, y_re[r]);
        _mm512_mask_store_pd(&out_im[k + r * K], mask, y_im[r]);
    }
}

//==============================================================================
// MAIN DRIVER: FORWARD N1 (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - N1 (NO TWIDDLES) - AVX-512
 *
 * @details
 * SIMPLIFIED ARCHITECTURE (ALL TWIDDLES = 1+0i):
 * - Even half: radix-16 butterfly (NO stage twiddles)
 * - Odd half: radix-16 butterfly (NO stage twiddles)
 * - Radix-2 combine (NO merge twiddles)
 * 
 * PERFORMANCE: 40-60% faster than standard version
 * 
 * AVX-512 LOOP STRUCTURE:
 * - Main loop: k += 16 (U=2: process k and k+8)
 * - Tail loop: k += 8
 * - Masked tail: remaining elements
 */
TARGET_AVX512
void radix32_stage_dit_forward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m512d rot_sign_mask = _mm512_set1_pd(-0.0);
    const __m512d neg_mask = _mm512_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_N1_AVX512;
    const size_t tile_size = RADIX32_TILE_SIZE_N1_AVX512;

    const bool use_nt_stores = radix32_should_use_nt_stores_n1_avx512(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=2 LOOP: k += 16
        size_t k;
        for (k = k_tile; k + 16 <= k_end; k += 16)
        {
            // Prefetch next iteration (inputs only - no twiddles!)
            size_t k_next = k + 16 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX32_PREFETCH_INPUTS_N1_AVX512(k_next, k_end, K,
                                                  in_re_aligned, in_im_aligned);
            }

            // ==================== PROCESS k ====================
            {
                // Even half: radix-16 butterfly (NO twiddles)
                __m512d even_re[16], even_im[16];
                load_16_lanes_soa_n1_avx512(k, K, in_re_aligned, in_im_aligned,
                                            even_re, even_im);

                radix16_complete_butterfly_forward_soa_avx512(even_re, even_im,
                                                              rot_sign_mask, neg_mask);

                // Odd half: radix-16 butterfly (NO twiddles)
                __m512d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm512_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm512_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                radix16_complete_butterfly_forward_soa_avx512(odd_re, odd_im,
                                                              rot_sign_mask, neg_mask);

                // NO MERGE TWIDDLES (all = 1)

                // Radix-2 combine
                __m512d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                       y_re, y_im);

                // Store
                if (use_nt_stores)
                {
                    store_32_lanes_soa_n1_avx512_stream(k, K, out_re_aligned, out_im_aligned,
                                                        y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_n1_avx512(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
                }
            }

            // ==================== PROCESS k+8 ====================
            {
                __m512d even_re[16], even_im[16];
                load_16_lanes_soa_n1_avx512(k + 8, K, in_re_aligned, in_im_aligned,
                                            even_re, even_im);

                radix16_complete_butterfly_forward_soa_avx512(even_re, even_im,
                                                              rot_sign_mask, neg_mask);

                __m512d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm512_load_pd(&in_re_aligned[k + 8 + (r + 16) * K]);
                    odd_im[r] = _mm512_load_pd(&in_im_aligned[k + 8 + (r + 16) * K]);
                }

                radix16_complete_butterfly_forward_soa_avx512(odd_re, odd_im,
                                                              rot_sign_mask, neg_mask);

                __m512d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                       y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_n1_avx512_stream(k + 8, K, out_re_aligned,
                                                        out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_n1_avx512(k + 8, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
                }
            }
        }

        // TAIL LOOP #1: k += 8
        for (; k + 8 <= k_end; k += 8)
        {
            __m512d even_re[16], even_im[16];
            load_16_lanes_soa_n1_avx512(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            radix16_complete_butterfly_forward_soa_avx512(even_re, even_im,
                                                          rot_sign_mask, neg_mask);

            __m512d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm512_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm512_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            radix16_complete_butterfly_forward_soa_avx512(odd_re, odd_im,
                                                          rot_sign_mask, neg_mask);

            __m512d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                   y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_n1_avx512_stream(k, K, out_re_aligned, out_im_aligned,
                                                    y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_n1_avx512(k, K, out_re_aligned, out_im_aligned,
                                             y_re, y_im);
            }
        }

        // TAIL LOOP #2: Masked
        radix32_process_tail_masked_n1_forward_avx512(k, k_end, K,
                                                      in_re_aligned, in_im_aligned,
                                                      out_re_aligned, out_im_aligned,
                                                      neg_mask);
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
 * @brief Radix-32 DIT Backward Stage - N1 (NO TWIDDLES) - AVX-512
 *
 * @details
 * Same as forward but with backward butterflies (conjugated rotations).
 */
TARGET_AVX512
void radix32_stage_dit_backward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const __m512d rot_sign_mask = _mm512_set1_pd(-0.0);
    const __m512d neg_mask = _mm512_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_N1_AVX512;
    const size_t tile_size = RADIX32_TILE_SIZE_N1_AVX512;

    const bool use_nt_stores = radix32_should_use_nt_stores_n1_avx512(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 16 <= k_end; k += 16)
        {
            size_t k_next = k + 16 + prefetch_dist;
            if (k_next < k_end)
            {
                RADIX32_PREFETCH_INPUTS_N1_AVX512(k_next, k_end, K,
                                                  in_re_aligned, in_im_aligned);
            }

            {
                __m512d even_re[16], even_im[16];
                load_16_lanes_soa_n1_avx512(k, K, in_re_aligned, in_im_aligned,
                                            even_re, even_im);

                radix16_complete_butterfly_backward_soa_avx512(even_re, even_im,
                                                               rot_sign_mask, neg_mask);

                __m512d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm512_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm512_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                radix16_complete_butterfly_backward_soa_avx512(odd_re, odd_im,
                                                               rot_sign_mask, neg_mask);

                __m512d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                       y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_n1_avx512_stream(k, K, out_re_aligned, out_im_aligned,
                                                        y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_n1_avx512(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
                }
            }

            {
                __m512d even_re[16], even_im[16];
                load_16_lanes_soa_n1_avx512(k + 8, K, in_re_aligned, in_im_aligned,
                                            even_re, even_im);

                radix16_complete_butterfly_backward_soa_avx512(even_re, even_im,
                                                               rot_sign_mask, neg_mask);

                __m512d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm512_load_pd(&in_re_aligned[k + 8 + (r + 16) * K]);
                    odd_im[r] = _mm512_load_pd(&in_im_aligned[k + 8 + (r + 16) * K]);
                }

                radix16_complete_butterfly_backward_soa_avx512(odd_re, odd_im,
                                                               rot_sign_mask, neg_mask);

                __m512d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                       y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_n1_avx512_stream(k + 8, K, out_re_aligned,
                                                        out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_n1_avx512(k + 8, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
                }
            }
        }

        for (; k + 8 <= k_end; k += 8)
        {
            __m512d even_re[16], even_im[16];
            load_16_lanes_soa_n1_avx512(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            radix16_complete_butterfly_backward_soa_avx512(even_re, even_im,
                                                           rot_sign_mask, neg_mask);

            __m512d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm512_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm512_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            radix16_complete_butterfly_backward_soa_avx512(odd_re, odd_im,
                                                           rot_sign_mask, neg_mask);

            __m512d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_n1_avx512(even_re, even_im, odd_re, odd_im,
                                                   y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_n1_avx512_stream(k, K, out_re_aligned, out_im_aligned,
                                                    y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_n1_avx512(k, K, out_re_aligned, out_im_aligned,
                                             y_re, y_im);
            }
        }

        radix32_process_tail_masked_n1_backward_avx512(k, k_end, K,
                                                       in_re_aligned, in_im_aligned,
                                                       out_re_aligned, out_im_aligned,
                                                       neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

#endif // FFT_RADIX32_AVX512_N1_H