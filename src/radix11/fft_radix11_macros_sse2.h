/**
 * @file fft_radix11_macros_true_soa_sse2_scalar.h
 * @brief SSE2 and Scalar Radix-11 Butterfly Macros - TRUE SoA
 *
 * @details
 * This header implements SSE2 and scalar versions of radix-11 FFT butterflies
 * using direct geometric decomposition, operating entirely in Structure-of-Arrays
 * (SoA) format without any split/join operations.
 *
 * @author FFT Optimization Team
 * @version 3.0 (TRUE END-TO-END SoA)
 * @date 2025
 */

#ifndef FFT_RADIX11_MACROS_TRUE_SOA_SSE2_SCALAR_H
#define FFT_RADIX11_MACROS_TRUE_SOA_SSE2_SCALAR_H

#include "fft_radix11.h"
#include "simd_math.h"

// Include common configuration and constants from main header
#include "fft_radix11_macros_true_soa.h"

//==============================================================================
// SSE2 IMPLEMENTATION
//==============================================================================

#ifdef __SSE2__

//==============================================================================
// CONSTANT BROADCASTING - Once per butterfly (CRITICAL OPTIMIZATION)
//==============================================================================

/**
 * @brief Pre-broadcast geometric constants for radix-11 (SSE2)
 *
 * OPTIMIZATION: Broadcast all 10 constants ONCE per butterfly instead of
 * 30+ times (5 cosines × 6 macros + 5 sines × 6 macros).
 *
 * PERFORMANCE IMPACT:
 *   Old: 30+ broadcasts per butterfly = ~30 cycles wasted
 *   New: 10 broadcasts once = ~10 cycles total
 *   Savings: 20 cycles per butterfly
 *
 * SSE2 processes 1 butterfly at once (128-bit registers).
 * Uses 128-bit registers (_mm operations).
 *
 * USAGE:
 *   radix11_consts_sse2 K = broadcast_radix11_consts_sse2();
 *   // Pass K to all SSE2 macros
 */
typedef struct
{
    __m128d c1, c2, c3, c4, c5; // Cosine constants (128-bit)
    __m128d s1, s2, s3, s4, s5; // Sine constants (128-bit)
} radix11_consts_sse2;

static inline __attribute__((always_inline))
radix11_consts_sse2
broadcast_radix11_consts_sse2(void)
{
    return (radix11_consts_sse2){
        .c1 = _mm_set1_pd(C11_1),
        .c2 = _mm_set1_pd(C11_2),
        .c3 = _mm_set1_pd(C11_3),
        .c4 = _mm_set1_pd(C11_4),
        .c5 = _mm_set1_pd(C11_5),
        .s1 = _mm_set1_pd(S11_1),
        .s2 = _mm_set1_pd(S11_2),
        .s3 = _mm_set1_pd(S11_3),
        .s4 = _mm_set1_pd(S11_4),
        .s5 = _mm_set1_pd(S11_5)};
}

//==============================================================================
// PREFETCHING
//==============================================================================

/**
 * @brief Prefetch 11 lanes from SoA buffers ahead of time
 *
 * @details
 * Prefetch input data and stage twiddles to L1/L2 cache.
 * Uses _MM_HINT_T0 for temporal locality (data will be reused).
 */
#define PREFETCH_11_LANES_R11_SSE2_SOA(k, K, in_re, in_im, stage_tw, sub_len)       \
    do                                                                              \
    {                                                                               \
        if ((k) + R11_PREFETCH_DISTANCE < (K))                                      \
        {                                                                           \
            int pk = (k) + R11_PREFETCH_DISTANCE;                                   \
            /* Prefetch input data for all 11 lanes */                              \
            _mm_prefetch((const char *)&in_re[pk + 0 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 0 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 1 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 1 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 2 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 2 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 3 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 3 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 4 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 4 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 5 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 5 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 6 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 6 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 7 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 7 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 8 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 8 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 9 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_im[pk + 9 * K], _MM_HINT_T0);            \
            _mm_prefetch((const char *)&in_re[pk + 10 * K], _MM_HINT_T0);           \
            _mm_prefetch((const char *)&in_im[pk + 10 * K], _MM_HINT_T0);           \
            /* Prefetch stage twiddles for all 10 lanes (skip lane 0 = DC) */       \
            if ((sub_len) > 1)                                                      \
            {                                                                       \
                _mm_prefetch((const char *)&stage_tw->re[0 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[0 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[1 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[1 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[2 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[2 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[3 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[3 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[4 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[4 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[5 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[5 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[6 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[6 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[7 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[7 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[8 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[8 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->re[9 * K + pk], _MM_HINT_T0); \
                _mm_prefetch((const char *)&stage_tw->im[9 * K + pk], _MM_HINT_T0); \
            }                                                                       \
        }                                                                           \
    } while (0)

//==============================================================================
// LOAD/STORE OPERATIONS - NATIVE SoA
//==============================================================================

/**
 * @brief Load 11 lanes from SoA buffers - SSE2
 *
 * @details
 * Loads 11 complex numbers from separate real and imaginary arrays.
 * Each load fetches 2 consecutive doubles (1 complex pair).
 *
 * Memory layout:
 *   in_re: [r0, r1, ...] (real parts)
 *   in_im: [i0, i1, ...] (imaginary parts)
 *
 * Register output (interleaved):
 *   x0: [r0, i0]
 *   x1: [r1, i1]
 *   ... (11 total registers for 11 complex lanes)
 *
 * BEFORE: Used _mm_loadu_pd which loaded 2 doubles, only used first one
 * AFTER:  Uses _mm_load_sd to load exactly 1 double per lane
 *
 * Benefit: No wasted loads, no risk of reading past array bounds
 */
#define LOAD_11_LANES_SSE2_NATIVE_SOA(k, K, in_re, in_im,                          \
                                      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) \
    do                                                                             \
    {                                                                              \
        __m128d re, im;                                                            \
        re = _mm_load_sd(&in_re[(k) + 0 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 0 * (K)]);                                   \
        x0 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 1 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 1 * (K)]);                                   \
        x1 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 2 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 2 * (K)]);                                   \
        x2 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 3 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 3 * (K)]);                                   \
        x3 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 4 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 4 * (K)]);                                   \
        x4 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 5 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 5 * (K)]);                                   \
        x5 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 6 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 6 * (K)]);                                   \
        x6 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 7 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 7 * (K)]);                                   \
        x7 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 8 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 8 * (K)]);                                   \
        x8 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 9 * (K)]);                                   \
        im = _mm_load_sd(&in_im[(k) + 9 * (K)]);                                   \
        x9 = _mm_unpacklo_pd(re, im);                                              \
        re = _mm_load_sd(&in_re[(k) + 10 * (K)]);                                  \
        im = _mm_load_sd(&in_im[(k) + 10 * (K)]);                                  \
        x10 = _mm_unpacklo_pd(re, im);                                             \
    } while (0)

/**
 * @brief Store 11 lanes to SoA buffers - SSE2 (normal stores)
 *
 * BEFORE: Used _mm_storeu_pd which wrote 2 doubles, corrupting adjacent data
 * AFTER:  Uses _mm_store_sd to write exactly 1 double per lane
 *
 * This was causing silent data corruption where each store would overwrite
 * the next element in the array!
 */
#define STORE_11_LANES_SSE2_NATIVE_SOA(k, K, out_re, out_im,                        \
                                       y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) \
    do                                                                              \
    {                                                                               \
        __m128d re, im;                                                             \
        re = _mm_shuffle_pd(y0, y0, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 0 * (K)], re);                                   \
        im = _mm_shuffle_pd(y0, y0, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 0 * (K)], im);                                   \
        re = _mm_shuffle_pd(y1, y1, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 1 * (K)], re);                                   \
        im = _mm_shuffle_pd(y1, y1, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 1 * (K)], im);                                   \
        re = _mm_shuffle_pd(y2, y2, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 2 * (K)], re);                                   \
        im = _mm_shuffle_pd(y2, y2, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 2 * (K)], im);                                   \
        re = _mm_shuffle_pd(y3, y3, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 3 * (K)], re);                                   \
        im = _mm_shuffle_pd(y3, y3, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 3 * (K)], im);                                   \
        re = _mm_shuffle_pd(y4, y4, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 4 * (K)], re);                                   \
        im = _mm_shuffle_pd(y4, y4, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 4 * (K)], im);                                   \
        re = _mm_shuffle_pd(y5, y5, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 5 * (K)], re);                                   \
        im = _mm_shuffle_pd(y5, y5, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 5 * (K)], im);                                   \
        re = _mm_shuffle_pd(y6, y6, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 6 * (K)], re);                                   \
        im = _mm_shuffle_pd(y6, y6, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 6 * (K)], im);                                   \
        re = _mm_shuffle_pd(y7, y7, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 7 * (K)], re);                                   \
        im = _mm_shuffle_pd(y7, y7, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 7 * (K)], im);                                   \
        re = _mm_shuffle_pd(y8, y8, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 8 * (K)], re);                                   \
        im = _mm_shuffle_pd(y8, y8, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 8 * (K)], im);                                   \
        re = _mm_shuffle_pd(y9, y9, 0x0);                                           \
        _mm_store_sd(&out_re[(k) + 9 * (K)], re);                                   \
        im = _mm_shuffle_pd(y9, y9, 0x3);                                           \
        _mm_store_sd(&out_im[(k) + 9 * (K)], im);                                   \
        re = _mm_shuffle_pd(y10, y10, 0x0);                                         \
        _mm_store_sd(&out_re[(k) + 10 * (K)], re);                                  \
        im = _mm_shuffle_pd(y10, y10, 0x3);                                         \
        _mm_store_sd(&out_im[(k) + 10 * (K)], im);                                  \
    } while (0)

/**
 * @brief Store 11 lanes to SoA buffers - SSE2 (streaming/non-temporal stores)
 * BEFORE: Used _mm_stream_pd which wrote 2 doubles, corrupting adjacent data
 * AFTER:  Uses _mm_stream_sd to write exactly 1 double per lane
 *
 * Note: _mm_stream_sd may not be available on all SSE2 implementations.
 * If it's not available, you can use _mm_store_sd + _mm_sfence instead.
 */
#define STORE_11_LANES_SSE2_STREAM_NATIVE_SOA(k, K, out_re, out_im,                        \
                                              y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) \
    do                                                                                     \
    {                                                                                      \
        __m128d re, im;                                                                    \
        re = _mm_shuffle_pd(y0, y0, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 0 * (K)], re);                                         \
        im = _mm_shuffle_pd(y0, y0, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 0 * (K)], im);                                         \
        re = _mm_shuffle_pd(y1, y1, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 1 * (K)], re);                                         \
        im = _mm_shuffle_pd(y1, y1, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 1 * (K)], im);                                         \
        re = _mm_shuffle_pd(y2, y2, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 2 * (K)], re);                                         \
        im = _mm_shuffle_pd(y2, y2, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 2 * (K)], im);                                         \
        re = _mm_shuffle_pd(y3, y3, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 3 * (K)], re);                                         \
        im = _mm_shuffle_pd(y3, y3, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 3 * (K)], im);                                         \
        re = _mm_shuffle_pd(y4, y4, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 4 * (K)], re);                                         \
        im = _mm_shuffle_pd(y4, y4, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 4 * (K)], im);                                         \
        re = _mm_shuffle_pd(y5, y5, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 5 * (K)], re);                                         \
        im = _mm_shuffle_pd(y5, y5, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 5 * (K)], im);                                         \
        re = _mm_shuffle_pd(y6, y6, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 6 * (K)], re);                                         \
        im = _mm_shuffle_pd(y6, y6, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 6 * (K)], im);                                         \
        re = _mm_shuffle_pd(y7, y7, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 7 * (K)], re);                                         \
        im = _mm_shuffle_pd(y7, y7, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 7 * (K)], im);                                         \
        re = _mm_shuffle_pd(y8, y8, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 8 * (K)], re);                                         \
        im = _mm_shuffle_pd(y8, y8, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 8 * (K)], im);                                         \
        re = _mm_shuffle_pd(y9, y9, 0x0);                                                  \
        _mm_stream_sd(&out_re[(k) + 9 * (K)], re);                                         \
        im = _mm_shuffle_pd(y9, y9, 0x3);                                                  \
        _mm_stream_sd(&out_im[(k) + 9 * (K)], im);                                         \
        re = _mm_shuffle_pd(y10, y10, 0x0);                                                \
        _mm_stream_sd(&out_re[(k) + 10 * (K)], re);                                        \
        im = _mm_shuffle_pd(y10, y10, 0x3);                                                \
        _mm_stream_sd(&out_im[(k) + 10 * (K)], im);                                        \
    } while (0)

/* ============================================================================
 * Alternative: STREAMING STORE with regular stores + fence
 * ============================================================================
 *
 * If _mm_stream_sd is not available, use this version instead.
 * It uses regular stores followed by a fence for write-combining behavior.
 */
#define STORE_11_LANES_SSE2_STREAM_FALLBACK_NATIVE_SOA(k, K, out_re, out_im,                        \
                                                       y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) \
    do                                                                                              \
    {                                                                                               \
        __m128d re, im;                                                                             \
        re = _mm_shuffle_pd(y0, y0, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 0 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y0, y0, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 0 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y1, y1, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 1 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y1, y1, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 1 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y2, y2, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 2 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y2, y2, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 2 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y3, y3, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 3 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y3, y3, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 3 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y4, y4, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 4 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y4, y4, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 4 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y5, y5, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 5 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y5, y5, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 5 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y6, y6, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 6 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y6, y6, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 6 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y7, y7, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 7 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y7, y7, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 7 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y8, y8, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 8 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y8, y8, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 8 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y9, y9, 0x0);                                                           \
        _mm_store_sd(&out_re[(k) + 9 * (K)], re);                                                   \
        im = _mm_shuffle_pd(y9, y9, 0x3);                                                           \
        _mm_store_sd(&out_im[(k) + 9 * (K)], im);                                                   \
        re = _mm_shuffle_pd(y10, y10, 0x0);                                                         \
        _mm_store_sd(&out_re[(k) + 10 * (K)], re);                                                  \
        im = _mm_shuffle_pd(y10, y10, 0x3);                                                         \
        _mm_store_sd(&out_im[(k) + 10 * (K)], im);                                                  \
        _mm_sfence();                                                                               \
    } while (0)

//==============================================================================
// STAGE TWIDDLE APPLICATION - NATIVE SoA
//==============================================================================

/**
 * @brief Apply stage twiddles to 10 complex values (skip x0) - SSE2 Native SoA
 *
 * @details
 * For sub_len > 1, multiply each x_r by stage twiddle W_k^r.
 * Stage twiddles are stored in SoA format: separate re[] and im[] arrays.
 *
 * Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
 * SSE2 doesn't have FMA, so use separate mul and add/sub operations.
 */
#define APPLY_STAGE_TWIDDLES_R11_SSE2_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,       \
                                                 x6, x7, x8, x9, x10,            \
                                                 stage_tw, sub_len)              \
    do                                                                           \
    {                                                                            \
        if ((sub_len) > 1)                                                       \
        {                                                                        \
            __m128d w_re, w_im, x_re, x_im, tmp_re, tmp_im;                      \
            /* Lane 1 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[0 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[0 * K + k]);                        \
            x_re = _mm_shuffle_pd(x1, x1, 0x0);                                  \
            x_im = _mm_shuffle_pd(x1, x1, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x1 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 2 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[1 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[1 * K + k]);                        \
            x_re = _mm_shuffle_pd(x2, x2, 0x0);                                  \
            x_im = _mm_shuffle_pd(x2, x2, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x2 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 3 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[2 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[2 * K + k]);                        \
            x_re = _mm_shuffle_pd(x3, x3, 0x0);                                  \
            x_im = _mm_shuffle_pd(x3, x3, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x3 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 4 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[3 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[3 * K + k]);                        \
            x_re = _mm_shuffle_pd(x4, x4, 0x0);                                  \
            x_im = _mm_shuffle_pd(x4, x4, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x4 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 5 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[4 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[4 * K + k]);                        \
            x_re = _mm_shuffle_pd(x5, x5, 0x0);                                  \
            x_im = _mm_shuffle_pd(x5, x5, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x5 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 6 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[5 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[5 * K + k]);                        \
            x_re = _mm_shuffle_pd(x6, x6, 0x0);                                  \
            x_im = _mm_shuffle_pd(x6, x6, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x6 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 7 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[6 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[6 * K + k]);                        \
            x_re = _mm_shuffle_pd(x7, x7, 0x0);                                  \
            x_im = _mm_shuffle_pd(x7, x7, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x7 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 8 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[7 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[7 * K + k]);                        \
            x_re = _mm_shuffle_pd(x8, x8, 0x0);                                  \
            x_im = _mm_shuffle_pd(x8, x8, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x8 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 9 */                                                         \
            w_re = _mm_load_sd(&stage_tw->re[8 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[8 * K + k]);                        \
            x_re = _mm_shuffle_pd(x9, x9, 0x0);                                  \
            x_im = _mm_shuffle_pd(x9, x9, 0x3);                                  \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x9 = _mm_unpacklo_pd(tmp_re, tmp_im);                                \
            /* Lane 10 */                                                        \
            w_re = _mm_load_sd(&stage_tw->re[9 * K + k]);                        \
            w_im = _mm_load_sd(&stage_tw->im[9 * K + k]);                        \
            x_re = _mm_shuffle_pd(x10, x10, 0x0);                                \
            x_im = _mm_shuffle_pd(x10, x10, 0x3);                                \
            tmp_re = _mm_sub_pd(_mm_mul_pd(x_re, w_re), _mm_mul_pd(x_im, w_im)); \
            tmp_im = _mm_add_pd(_mm_mul_pd(x_re, w_im), _mm_mul_pd(x_im, w_re)); \
            x10 = _mm_unpacklo_pd(tmp_re, tmp_im);                               \
        }                                                                        \
    } while (0)

//==============================================================================
// BUTTERFLY CORE - SSE2
//==============================================================================

/**
 * @brief Butterfly core for SSE2 (1 butterfly at once)
 *
 * Forms 5 symmetric pairs and computes y0 (DC component):
 *   t0 = b + k
 *   t1 = c + j
 *   t2 = d + i
 *   t3 = e + h
 *   t4 = f + g
 *   s0 = b - k
 *   s1 = c - j
 *   s2 = d - i
 *   s3 = e - h
 *   s4 = f - g
 *   y0 = a + (t0 + t1 + t2 + t3 + t4)
 */
#define RADIX11_BUTTERFLY_CORE_SSE2(a, b, c, d, e, f, g, h, i, j, xk,           \
                                    t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0) \
    do                                                                          \
    {                                                                           \
        t0 = _mm_add_pd(b, xk);                                                 \
        t1 = _mm_add_pd(c, j);                                                  \
        t2 = _mm_add_pd(d, i);                                                  \
        t3 = _mm_add_pd(e, h);                                                  \
        t4 = _mm_add_pd(f, g);                                                  \
        s0 = _mm_sub_pd(b, xk);                                                 \
        s1 = _mm_sub_pd(c, j);                                                  \
        s2 = _mm_sub_pd(d, i);                                                  \
        s3 = _mm_sub_pd(e, h);                                                  \
        s4 = _mm_sub_pd(f, g);                                                  \
        __m128d sum_t = _mm_add_pd(_mm_add_pd(t0, t1),                          \
                                   _mm_add_pd(_mm_add_pd(t2, t3), t4));         \
        y0 = _mm_add_pd(a, sum_t);                                              \
    } while (0)

//==============================================================================
// ALL 5 REAL PAIRS (SSE2)
//==============================================================================

/**
 * @brief Real part of pair 1 - SSE2
 *
 * real_out = a + c1*t0 + c2*t1 + c3*t2 + c4*t3 + c5*t4
 * SSE2 doesn't have FMA, so chain mul and add operations
 */
#define RADIX11_REAL_PAIR1_SSE2(a, t0, t1, t2, t3, t4, K, real_out)           \
    do                                                                        \
    {                                                                         \
        __m128d tmp = _mm_add_pd(_mm_mul_pd(K.c1, t0), _mm_mul_pd(K.c2, t1)); \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c3, t2));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c4, t3));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c5, t4));                          \
        real_out = _mm_add_pd(a, tmp);                                        \
    } while (0)

#define RADIX11_REAL_PAIR2_SSE2(a, t0, t1, t2, t3, t4, K, real_out)           \
    do                                                                        \
    {                                                                         \
        __m128d tmp = _mm_add_pd(_mm_mul_pd(K.c2, t0), _mm_mul_pd(K.c4, t1)); \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c5, t2));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c3, t3));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c1, t4));                          \
        real_out = _mm_add_pd(a, tmp);                                        \
    } while (0)

#define RADIX11_REAL_PAIR3_SSE2(a, t0, t1, t2, t3, t4, K, real_out)           \
    do                                                                        \
    {                                                                         \
        __m128d tmp = _mm_add_pd(_mm_mul_pd(K.c3, t0), _mm_mul_pd(K.c5, t1)); \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c2, t2));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c1, t3));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c4, t4));                          \
        real_out = _mm_add_pd(a, tmp);                                        \
    } while (0)

#define RADIX11_REAL_PAIR4_SSE2(a, t0, t1, t2, t3, t4, K, real_out)           \
    do                                                                        \
    {                                                                         \
        __m128d tmp = _mm_add_pd(_mm_mul_pd(K.c4, t0), _mm_mul_pd(K.c3, t1)); \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c1, t2));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c5, t3));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c2, t4));                          \
        real_out = _mm_add_pd(a, tmp);                                        \
    } while (0)

#define RADIX11_REAL_PAIR5_SSE2(a, t0, t1, t2, t3, t4, K, real_out)           \
    do                                                                        \
    {                                                                         \
        __m128d tmp = _mm_add_pd(_mm_mul_pd(K.c5, t0), _mm_mul_pd(K.c1, t1)); \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c4, t2));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c2, t3));                          \
        tmp = _mm_add_pd(tmp, _mm_mul_pd(K.c3, t4));                          \
        real_out = _mm_add_pd(a, tmp);                                        \
    } while (0)

//==============================================================================
// ALL 5 IMAGINARY PAIRS - FORWARD (SSE2)
//==============================================================================

/**
 * @brief Imaginary part of pair 1 - Forward Transform - SSE2
 *
 * rot_out = -i * (s1*s0 + s2*s1 + s3*s2 + s4*s3 + s5*s4)
 */
#define RADIX11_IMAG_PAIR1_FV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s1, s0), _mm_mul_pd(K.s2, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s4));                         \
        rot_out = rot_neg_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR2_FV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s2, s0), _mm_mul_pd(K.s4, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s4));                         \
        rot_out = rot_neg_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR3_FV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s3, s0), _mm_mul_pd(K.s5, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s4));                         \
        rot_out = rot_neg_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR4_FV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s4, s0), _mm_mul_pd(K.s3, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s4));                         \
        rot_out = rot_neg_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR5_FV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s5, s0), _mm_mul_pd(K.s1, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s4));                         \
        rot_out = rot_neg_i_sse2(base);                                        \
    } while (0)

//==============================================================================
// ALL 5 IMAGINARY PAIRS - INVERSE (SSE2)
//==============================================================================

/**
 * @brief Imaginary part of pair 1 - Inverse Transform - SSE2
 *
 * rot_out = +i * (s1*s0 + s2*s1 + s3*s2 + s4*s3 + s5*s4)
 */
#define RADIX11_IMAG_PAIR1_BV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s1, s0), _mm_mul_pd(K.s2, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s4));                         \
        rot_out = rot_pos_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR2_BV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s2, s0), _mm_mul_pd(K.s4, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s4));                         \
        rot_out = rot_pos_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR3_BV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s3, s0), _mm_mul_pd(K.s5, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s4));                         \
        rot_out = rot_pos_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR4_BV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s4, s0), _mm_mul_pd(K.s3, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s1, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s5, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s4));                         \
        rot_out = rot_pos_i_sse2(base);                                        \
    } while (0)

#define RADIX11_IMAG_PAIR5_BV_SSE2(s0, s1, s2, s3, s4, K, rot_out)             \
    do                                                                         \
    {                                                                          \
        __m128d base = _mm_add_pd(_mm_mul_pd(K.s5, s0), _mm_mul_pd(K.s1, s1)); \
        base = _mm_add_pd(base, _mm_mul_pd(K.s4, s2));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s2, s3));                         \
        base = _mm_add_pd(base, _mm_mul_pd(K.s3, s4));                         \
        rot_out = rot_pos_i_sse2(base);                                        \
    } while (0)

//==============================================================================
// PAIR ASSEMBLY (SSE2)
//==============================================================================

/**
 * @brief Assemble output pairs (SSE2)
 *
 * Combines real and imaginary parts:
 *   y_m = real + rot
 *   y_11m = real - rot  (conjugate symmetry)
 */
#define RADIX11_ASSEMBLE_PAIR_SSE2(real, rot, y_m, y_11m) \
    do                                                    \
    {                                                     \
        y_m = _mm_add_pd(real, rot);                      \
        y_11m = _mm_sub_pd(real, rot);                    \
    } while (0)

//==============================================================================
// TOP-LEVEL BUTTERFLY MACROS - NATIVE SoA
//==============================================================================

/**
 * @brief Forward radix-11 butterfly - SSE2 - Native SoA (normal stores)
 */
#define RADIX11_BUTTERFLY_FV_SSE2_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                             out_re, out_im, sub_len)                \
    do                                                                               \
    {                                                                                \
        radix11_consts_sse2 KC = broadcast_radix11_consts_sse2();                    \
        __m128d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;                         \
        LOAD_11_LANES_SSE2_NATIVE_SOA(k, K, in_re, in_im,                            \
                                      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);  \
        APPLY_STAGE_TWIDDLES_R11_SSE2_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,           \
                                                 x6, x7, x8, x9, x10,                \
                                                 stage_tw, sub_len);                 \
        __m128d t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0;                          \
        RADIX11_BUTTERFLY_CORE_SSE2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,     \
                                    t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0);     \
        __m128d real1, real2, real3, real4, real5;                                   \
        RADIX11_REAL_PAIR1_SSE2(x0, t0, t1, t2, t3, t4, KC, real1);                  \
        RADIX11_REAL_PAIR2_SSE2(x0, t0, t1, t2, t3, t4, KC, real2);                  \
        RADIX11_REAL_PAIR3_SSE2(x0, t0, t1, t2, t3, t4, KC, real3);                  \
        RADIX11_REAL_PAIR4_SSE2(x0, t0, t1, t2, t3, t4, KC, real4);                  \
        RADIX11_REAL_PAIR5_SSE2(x0, t0, t1, t2, t3, t4, KC, real5);                  \
        __m128d rot1, rot2, rot3, rot4, rot5;                                        \
        RADIX11_IMAG_PAIR1_FV_SSE2(s0, s1, s2, s3, s4, KC, rot1);                    \
        RADIX11_IMAG_PAIR2_FV_SSE2(s0, s1, s2, s3, s4, KC, rot2);                    \
        RADIX11_IMAG_PAIR3_FV_SSE2(s0, s1, s2, s3, s4, KC, rot3);                    \
        RADIX11_IMAG_PAIR4_FV_SSE2(s0, s1, s2, s3, s4, KC, rot4);                    \
        RADIX11_IMAG_PAIR5_FV_SSE2(s0, s1, s2, s3, s4, KC, rot5);                    \
        __m128d y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real1, rot1, y1, y10);                            \
        RADIX11_ASSEMBLE_PAIR_SSE2(real2, rot2, y2, y9);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real3, rot3, y3, y8);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real4, rot4, y4, y7);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real5, rot5, y5, y6);                             \
        STORE_11_LANES_SSE2_NATIVE_SOA(k, K, out_re, out_im,                         \
                                       y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10); \
    } while (0)

/**
 * @brief Forward radix-11 butterfly - SSE2 - Native SoA (streaming stores)
 */
#define RADIX11_BUTTERFLY_FV_SSE2_STREAM_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                                    out_re, out_im, sub_len)                \
    do                                                                                      \
    {                                                                                       \
        radix11_consts_sse2 KC = broadcast_radix11_consts_sse2();                           \
        __m128d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;                                \
        LOAD_11_LANES_SSE2_NATIVE_SOA(k, K, in_re, in_im,                                   \
                                      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);         \
        APPLY_STAGE_TWIDDLES_R11_SSE2_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,                  \
                                                 x6, x7, x8, x9, x10,                       \
                                                 stage_tw, sub_len);                        \
        __m128d t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0;                                 \
        RADIX11_BUTTERFLY_CORE_SSE2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,            \
                                    t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0);            \
        __m128d real1, real2, real3, real4, real5;                                          \
        RADIX11_REAL_PAIR1_SSE2(x0, t0, t1, t2, t3, t4, KC, real1);                         \
        RADIX11_REAL_PAIR2_SSE2(x0, t0, t1, t2, t3, t4, KC, real2);                         \
        RADIX11_REAL_PAIR3_SSE2(x0, t0, t1, t2, t3, t4, KC, real3);                         \
        RADIX11_REAL_PAIR4_SSE2(x0, t0, t1, t2, t3, t4, KC, real4);                         \
        RADIX11_REAL_PAIR5_SSE2(x0, t0, t1, t2, t3, t4, KC, real5);                         \
        __m128d rot1, rot2, rot3, rot4, rot5;                                               \
        RADIX11_IMAG_PAIR1_FV_SSE2(s0, s1, s2, s3, s4, KC, rot1);                           \
        RADIX11_IMAG_PAIR2_FV_SSE2(s0, s1, s2, s3, s4, KC, rot2);                           \
        RADIX11_IMAG_PAIR3_FV_SSE2(s0, s1, s2, s3, s4, KC, rot3);                           \
        RADIX11_IMAG_PAIR4_FV_SSE2(s0, s1, s2, s3, s4, KC, rot4);                           \
        RADIX11_IMAG_PAIR5_FV_SSE2(s0, s1, s2, s3, s4, KC, rot5);                           \
        __m128d y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real1, rot1, y1, y10);                                   \
        RADIX11_ASSEMBLE_PAIR_SSE2(real2, rot2, y2, y9);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real3, rot3, y3, y8);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real4, rot4, y4, y7);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real5, rot5, y5, y6);                                    \
        STORE_11_LANES_SSE2_STREAM_NATIVE_SOA(k, K, out_re, out_im,                         \
                                              y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10); \
    } while (0)

/**
 * @brief Backward radix-11 butterfly - SSE2 - Native SoA (normal stores)
 */
#define RADIX11_BUTTERFLY_BV_SSE2_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                             out_re, out_im, sub_len)                \
    do                                                                               \
    {                                                                                \
        radix11_consts_sse2 KC = broadcast_radix11_consts_sse2();                    \
        __m128d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;                         \
        LOAD_11_LANES_SSE2_NATIVE_SOA(k, K, in_re, in_im,                            \
                                      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);  \
        APPLY_STAGE_TWIDDLES_R11_SSE2_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,           \
                                                 x6, x7, x8, x9, x10,                \
                                                 stage_tw, sub_len);                 \
        __m128d t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0;                          \
        RADIX11_BUTTERFLY_CORE_SSE2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,     \
                                    t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0);     \
        __m128d real1, real2, real3, real4, real5;                                   \
        RADIX11_REAL_PAIR1_SSE2(x0, t0, t1, t2, t3, t4, KC, real1);                  \
        RADIX11_REAL_PAIR2_SSE2(x0, t0, t1, t2, t3, t4, KC, real2);                  \
        RADIX11_REAL_PAIR3_SSE2(x0, t0, t1, t2, t3, t4, KC, real3);                  \
        RADIX11_REAL_PAIR4_SSE2(x0, t0, t1, t2, t3, t4, KC, real4);                  \
        RADIX11_REAL_PAIR5_SSE2(x0, t0, t1, t2, t3, t4, KC, real5);                  \
        __m128d rot1, rot2, rot3, rot4, rot5;                                        \
        RADIX11_IMAG_PAIR1_BV_SSE2(s0, s1, s2, s3, s4, KC, rot1);                    \
        RADIX11_IMAG_PAIR2_BV_SSE2(s0, s1, s2, s3, s4, KC, rot2);                    \
        RADIX11_IMAG_PAIR3_BV_SSE2(s0, s1, s2, s3, s4, KC, rot3);                    \
        RADIX11_IMAG_PAIR4_BV_SSE2(s0, s1, s2, s3, s4, KC, rot4);                    \
        RADIX11_IMAG_PAIR5_BV_SSE2(s0, s1, s2, s3, s4, KC, rot5);                    \
        __m128d y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real1, rot1, y1, y10);                            \
        RADIX11_ASSEMBLE_PAIR_SSE2(real2, rot2, y2, y9);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real3, rot3, y3, y8);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real4, rot4, y4, y7);                             \
        RADIX11_ASSEMBLE_PAIR_SSE2(real5, rot5, y5, y6);                             \
        STORE_11_LANES_SSE2_NATIVE_SOA(k, K, out_re, out_im,                         \
                                       y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10); \
    } while (0)

/**
 * @brief Backward radix-11 butterfly - SSE2 - Native SoA (streaming stores)
 */
#define RADIX11_BUTTERFLY_BV_SSE2_STREAM_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                                    out_re, out_im, sub_len)                \
    do                                                                                      \
    {                                                                                       \
        radix11_consts_sse2 KC = broadcast_radix11_consts_sse2();                           \
        __m128d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;                                \
        LOAD_11_LANES_SSE2_NATIVE_SOA(k, K, in_re, in_im,                                   \
                                      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10);         \
        APPLY_STAGE_TWIDDLES_R11_SSE2_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,                  \
                                                 x6, x7, x8, x9, x10,                       \
                                                 stage_tw, sub_len);                        \
        __m128d t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0;                                 \
        RADIX11_BUTTERFLY_CORE_SSE2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,            \
                                    t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0);            \
        __m128d real1, real2, real3, real4, real5;                                          \
        RADIX11_REAL_PAIR1_SSE2(x0, t0, t1, t2, t3, t4, KC, real1);                         \
        RADIX11_REAL_PAIR2_SSE2(x0, t0, t1, t2, t3, t4, KC, real2);                         \
        RADIX11_REAL_PAIR3_SSE2(x0, t0, t1, t2, t3, t4, KC, real3);                         \
        RADIX11_REAL_PAIR4_SSE2(x0, t0, t1, t2, t3, t4, KC, real4);                         \
        RADIX11_REAL_PAIR5_SSE2(x0, t0, t1, t2, t3, t4, KC, real5);                         \
        __m128d rot1, rot2, rot3, rot4, rot5;                                               \
        RADIX11_IMAG_PAIR1_BV_SSE2(s0, s1, s2, s3, s4, KC, rot1);                           \
        RADIX11_IMAG_PAIR2_BV_SSE2(s0, s1, s2, s3, s4, KC, rot2);                           \
        RADIX11_IMAG_PAIR3_BV_SSE2(s0, s1, s2, s3, s4, KC, rot3);                           \
        RADIX11_IMAG_PAIR4_BV_SSE2(s0, s1, s2, s3, s4, KC, rot4);                           \
        RADIX11_IMAG_PAIR5_BV_SSE2(s0, s1, s2, s3, s4, KC, rot5);                           \
        __m128d y1, y2, y3, y4, y5, y6, y7, y8, y9, y10;                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real1, rot1, y1, y10);                                   \
        RADIX11_ASSEMBLE_PAIR_SSE2(real2, rot2, y2, y9);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real3, rot3, y3, y8);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real4, rot4, y4, y7);                                    \
        RADIX11_ASSEMBLE_PAIR_SSE2(real5, rot5, y5, y6);                                    \
        STORE_11_LANES_SSE2_STREAM_NATIVE_SOA(k, K, out_re, out_im,                         \
                                              y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10); \
    } while (0)

#endif // __SSE2__

//==============================================================================
// SCALAR FALLBACK - NATIVE SoA
//==============================================================================

/**
 * @brief Scalar radix-11 butterfly - NATIVE SoA
 * @details For cleanup iterations and non-SIMD builds
 */
#define RADIX11_BUTTERFLY_SCALAR_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                            out_re, out_im, sub_len)                \
    do                                                                              \
    {                                                                               \
        double x_re[11], x_im[11];                                                  \
        for (int _i = 0; _i < 11; _i++)                                             \
        {                                                                           \
            x_re[_i] = in_re[k + _i * K];                                           \
            x_im[_i] = in_im[k + _i * K];                                           \
        }                                                                           \
        if ((sub_len) > 1)                                                          \
        {                                                                           \
            for (int _r = 1; _r < 11; _r++)                                         \
            {                                                                       \
                double w_re = stage_tw->re[(_r - 1) * K + k];                       \
                double w_im = stage_tw->im[(_r - 1) * K + k];                       \
                double tmp_re = x_re[_r] * w_re - x_im[_r] * w_im;                  \
                double tmp_im = x_re[_r] * w_im + x_im[_r] * w_re;                  \
                x_re[_r] = tmp_re;                                                  \
                x_im[_r] = tmp_im;                                                  \
            }                                                                       \
        }                                                                           \
        /* Form 5 symmetric pairs */                                                \
        double t0_re = x_re[1] + x_re[10];                                          \
        double t0_im = x_im[1] + x_im[10];                                          \
        double t1_re = x_re[2] + x_re[9];                                           \
        double t1_im = x_im[2] + x_im[9];                                           \
        double t2_re = x_re[3] + x_re[8];                                           \
        double t2_im = x_im[3] + x_im[8];                                           \
        double t3_re = x_re[4] + x_re[7];                                           \
        double t3_im = x_im[4] + x_im[7];                                           \
        double t4_re = x_re[5] + x_re[6];                                           \
        double t4_im = x_im[5] + x_im[6];                                           \
        double s0_re = x_re[1] - x_re[10];                                          \
        double s0_im = x_im[1] - x_im[10];                                          \
        double s1_re = x_re[2] - x_re[9];                                           \
        double s1_im = x_im[2] - x_im[9];                                           \
        double s2_re = x_re[3] - x_re[8];                                           \
        double s2_im = x_im[3] - x_im[8];                                           \
        double s3_re = x_re[4] - x_re[7];                                           \
        double s3_im = x_im[4] - x_im[7];                                           \
        double s4_re = x_re[5] - x_re[6];                                           \
        double s4_im = x_im[5] - x_im[6];                                           \
        /* DC component */                                                          \
        double y0_re = x_re[0] + t0_re + t1_re + t2_re + t3_re + t4_re;             \
        double y0_im = x_im[0] + t0_im + t1_im + t2_im + t3_im + t4_im;             \
        /* Real parts of 5 pairs */                                                 \
        double real1_re = x_re[0] + C11_1 * t0_re + C11_2 * t1_re + C11_3 * t2_re + \
                          C11_4 * t3_re + C11_5 * t4_re;                            \
        double real1_im = x_im[0] + C11_1 * t0_im + C11_2 * t1_im + C11_3 * t2_im + \
                          C11_4 * t3_im + C11_5 * t4_im;                            \
        double real2_re = x_re[0] + C11_2 * t0_re + C11_4 * t1_re + C11_5 * t2_re + \
                          C11_3 * t3_re + C11_1 * t4_re;                            \
        double real2_im = x_im[0] + C11_2 * t0_im + C11_4 * t1_im + C11_5 * t2_im + \
                          C11_3 * t3_im + C11_1 * t4_im;                            \
        double real3_re = x_re[0] + C11_3 * t0_re + C11_5 * t1_re + C11_2 * t2_re + \
                          C11_1 * t3_re + C11_4 * t4_re;                            \
        double real3_im = x_im[0] + C11_3 * t0_im + C11_5 * t1_im + C11_2 * t2_im + \
                          C11_1 * t3_im + C11_4 * t4_im;                            \
        double real4_re = x_re[0] + C11_4 * t0_re + C11_3 * t1_re + C11_1 * t2_re + \
                          C11_5 * t3_re + C11_2 * t4_re;                            \
        double real4_im = x_im[0] + C11_4 * t0_im + C11_3 * t1_im + C11_1 * t2_im + \
                          C11_5 * t3_im + C11_2 * t4_im;                            \
        double real5_re = x_re[0] + C11_5 * t0_re + C11_1 * t1_re + C11_4 * t2_re + \
                          C11_2 * t3_re + C11_3 * t4_re;                            \
        double real5_im = x_im[0] + C11_5 * t0_im + C11_1 * t1_im + C11_4 * t2_im + \
                          C11_2 * t3_im + C11_3 * t4_im;                            \
        /* Imaginary parts (multiplied by -i for forward) */                        \
        double base1_re = S11_1 * s0_re + S11_2 * s1_re + S11_3 * s2_re +           \
                          S11_4 * s3_re + S11_5 * s4_re;                            \
        double base1_im = S11_1 * s0_im + S11_2 * s1_im + S11_3 * s2_im +           \
                          S11_4 * s3_im + S11_5 * s4_im;                            \
        double rot1_re = base1_im;                                                  \
        double rot1_im = -base1_re;                                                 \
        double base2_re = S11_2 * s0_re + S11_4 * s1_re + S11_5 * s2_re +           \
                          S11_3 * s3_re + S11_1 * s4_re;                            \
        double base2_im = S11_2 * s0_im + S11_4 * s1_im + S11_5 * s2_im +           \
                          S11_3 * s3_im + S11_1 * s4_im;                            \
        double rot2_re = base2_im;                                                  \
        double rot2_im = -base2_re;                                                 \
        double base3_re = S11_3 * s0_re + S11_5 * s1_re + S11_2 * s2_re +           \
                          S11_1 * s3_re + S11_4 * s4_re;                            \
        double base3_im = S11_3 * s0_im + S11_5 * s1_im + S11_2 * s2_im +           \
                          S11_1 * s3_im + S11_4 * s4_im;                            \
        double rot3_re = base3_im;                                                  \
        double rot3_im = -base3_re;                                                 \
        double base4_re = S11_4 * s0_re + S11_3 * s1_re + S11_1 * s2_re +           \
                          S11_5 * s3_re + S11_2 * s4_re;                            \
        double base4_im = S11_4 * s0_im + S11_3 * s1_im + S11_1 * s2_im +           \
                          S11_5 * s3_im + S11_2 * s4_im;                            \
        double rot4_re = base4_im;                                                  \
        double rot4_im = -base4_re;                                                 \
        double base5_re = S11_5 * s0_re + S11_1 * s1_re + S11_4 * s2_re +           \
                          S11_2 * s3_re + S11_3 * s4_re;                            \
        double base5_im = S11_5 * s0_im + S11_1 * s1_im + S11_4 * s2_im +           \
                          S11_2 * s3_im + S11_3 * s4_im;                            \
        double rot5_re = base5_im;                                                  \
        double rot5_im = -base5_re;                                                 \
        /* Assemble outputs */                                                      \
        double y_re[11], y_im[11];                                                  \
        y_re[0] = y0_re;                                                            \
        y_im[0] = y0_im;                                                            \
        y_re[1] = real1_re + rot1_re;                                               \
        y_im[1] = real1_im + rot1_im;                                               \
        y_re[10] = real1_re - rot1_re;                                              \
        y_im[10] = real1_im - rot1_im;                                              \
        y_re[2] = real2_re + rot2_re;                                               \
        y_im[2] = real2_im + rot2_im;                                               \
        y_re[9] = real2_re - rot2_re;                                               \
        y_im[9] = real2_im - rot2_im;                                               \
        y_re[3] = real3_re + rot3_re;                                               \
        y_im[3] = real3_im + rot3_im;                                               \
        y_re[8] = real3_re - rot3_re;                                               \
        y_im[8] = real3_im - rot3_im;                                               \
        y_re[4] = real4_re + rot4_re;                                               \
        y_im[4] = real4_im + rot4_im;                                               \
        y_re[7] = real4_re - rot4_re;                                               \
        y_im[7] = real4_im - rot4_im;                                               \
        y_re[5] = real5_re + rot5_re;                                               \
        y_im[5] = real5_im + rot5_im;                                               \
        y_re[6] = real5_re - rot5_re;                                               \
        y_im[6] = real5_im - rot5_im;                                               \
        /* Store */                                                                 \
        for (int _i = 0; _i < 11; _i++)                                             \
        {                                                                           \
            out_re[k + _i * K] = y_re[_i];                                          \
            out_im[k + _i * K] = y_im[_i];                                          \
        }                                                                           \
    } while (0)

#endif // FFT_RADIX11_MACROS_TRUE_SOA_SSE2_SCALAR_H