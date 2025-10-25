/**
 * @file fft_radix5_n1_notwiddle.h
 * @brief No-Twiddle (N1) Radix-5 FFT Macros - Based on fft_radix5_macros_optimized.h
 *
 * @details
 * CHANGE from twiddle version: Remove CMUL operations and twiddle loads
 * EVERYTHING ELSE: Identical to preserve all hard-won optimizations
 *
 * Usage: First stage FFT where all twiddles are 1+0i
 */

#ifndef FFT_RADIX5_N1_NOTWIDDLE_H
#define FFT_RADIX5_N1_NOTWIDDLE_H

#include "simd_math.h"
#include <immintrin.h>

// Reuse constants from original
#ifndef C5_1
#define C5_1 0.30901699437494742410
#define C5_2 (-0.80901699437494742410)
#define S5_1 0.95105651629515357212
#define S5_2 0.58778525229247312917
#endif

//==============================================================================
// AVX-512: NO-TWIDDLE BUTTERFLIES
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief Forward butterfly - NO TWIDDLES (just remove CMUL from original)
 */
#define RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                   \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        /* NO TWIDDLE MUL - use b,c,d,e directly as tw_b,tw_c,tw_d,tw_e */        \
        __m512d s1_re = _mm512_add_pd(b_re, e_re);                                  \
        __m512d s1_im = _mm512_add_pd(b_im, e_im);                                  \
        __m512d s2_re = _mm512_add_pd(c_re, d_re);                                  \
        __m512d s2_im = _mm512_add_pd(c_im, d_im);                                  \
        __m512d d1_re = _mm512_sub_pd(b_re, e_re);                                  \
        __m512d d1_im = _mm512_sub_pd(b_im, e_im);                                  \
        __m512d d2_re = _mm512_sub_pd(c_re, d_re);                                  \
        __m512d d2_im = _mm512_sub_pd(c_im, d_im);                                  \
        /* Rest is IDENTICAL to original */                                        \
        y0_re = _mm512_add_pd(a_re, _mm512_add_pd(s1_re, s2_re));                   \
        y0_im = _mm512_add_pd(a_im, _mm512_add_pd(s1_im, s2_im));                   \
        const __m512d vc51 = _mm512_set1_pd(C5_1);                                  \
        const __m512d vc52 = _mm512_set1_pd(C5_2);                                  \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                                  \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                                  \
        __m512d t1_re = _mm512_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm512_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m512d t1_im = _mm512_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm512_fmadd_pd(vc52, s2_im, t1_im);                                \
        __m512d t2_re = _mm512_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm512_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m512d t2_im = _mm512_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm512_fmadd_pd(vc51, s2_im, t2_im);                                \
        __m512d base1_re = _mm512_mul_pd(vs51, d1_re);                              \
        base1_re = _mm512_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m512d base1_im = _mm512_mul_pd(vs51, d1_im);                              \
        base1_im = _mm512_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m512d u1_re = _mm512_sub_pd(_mm512_setzero_pd(), base1_im);               \
        __m512d u1_im = base1_re;                                                   \
        __m512d base2_re = _mm512_mul_pd(vs52, d1_re);                              \
        base2_re = _mm512_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m512d base2_im = _mm512_mul_pd(vs52, d1_im);                              \
        base2_im = _mm512_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m512d u2_re = _mm512_sub_pd(_mm512_setzero_pd(), base2_im);               \
        __m512d u2_im = base2_re;                                                   \
        y1_re = _mm512_add_pd(t1_re, u1_re);                                        \
        y1_im = _mm512_add_pd(t1_im, u1_im);                                        \
        y4_re = _mm512_sub_pd(t1_re, u1_re);                                        \
        y4_im = _mm512_sub_pd(t1_im, u1_im);                                        \
        y2_re = _mm512_sub_pd(t2_re, u2_re);                                        \
        y2_im = _mm512_sub_pd(t2_im, u2_im);                                        \
        y3_re = _mm512_add_pd(t2_re, u2_re);                                        \
        y3_im = _mm512_add_pd(t2_im, u2_im);                                        \
    } while (0)

/**
 * @brief Backward butterfly - NO TWIDDLES (just remove CMUL from original)
 */
#define RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                   \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        /* NO TWIDDLE MUL */                                                       \
        __m512d s1_re = _mm512_add_pd(b_re, e_re);                                  \
        __m512d s1_im = _mm512_add_pd(b_im, e_im);                                  \
        __m512d s2_re = _mm512_add_pd(c_re, d_re);                                  \
        __m512d s2_im = _mm512_add_pd(c_im, d_im);                                  \
        __m512d d1_re = _mm512_sub_pd(b_re, e_re);                                  \
        __m512d d1_im = _mm512_sub_pd(b_im, e_im);                                  \
        __m512d d2_re = _mm512_sub_pd(c_re, d_re);                                  \
        __m512d d2_im = _mm512_sub_pd(c_im, d_im);                                  \
        /* Rest IDENTICAL to original backward butterfly */                        \
        y0_re = _mm512_add_pd(a_re, _mm512_add_pd(s1_re, s2_re));                   \
        y0_im = _mm512_add_pd(a_im, _mm512_add_pd(s1_im, s2_im));                   \
        const __m512d vc51 = _mm512_set1_pd(C5_1);                                  \
        const __m512d vc52 = _mm512_set1_pd(C5_2);                                  \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                                  \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                                  \
        __m512d t1_re = _mm512_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm512_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m512d t1_im = _mm512_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm512_fmadd_pd(vc52, s2_im, t1_im);                                \
        __m512d t2_re = _mm512_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm512_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m512d t2_im = _mm512_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm512_fmadd_pd(vc51, s2_im, t2_im);                                \
        __m512d base1_re = _mm512_mul_pd(vs51, d1_re);                              \
        base1_re = _mm512_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m512d base1_im = _mm512_mul_pd(vs51, d1_im);                              \
        base1_im = _mm512_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m512d u1_re = base1_im;                                                   \
        __m512d u1_im = _mm512_sub_pd(_mm512_setzero_pd(), base1_re);               \
        __m512d base2_re = _mm512_mul_pd(vs52, d1_re);                              \
        base2_re = _mm512_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m512d base2_im = _mm512_mul_pd(vs52, d1_im);                              \
        base2_im = _mm512_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m512d u2_re = base2_im;                                                   \
        __m512d u2_im = _mm512_sub_pd(_mm512_setzero_pd(), base2_re);               \
        y1_re = _mm512_add_pd(t1_re, u1_re);                                        \
        y1_im = _mm512_add_pd(t1_im, u1_im);                                        \
        y4_re = _mm512_sub_pd(t1_re, u1_re);                                        \
        y4_im = _mm512_sub_pd(t1_im, u1_im);                                        \
        y2_re = _mm512_sub_pd(t2_re, u2_re);                                        \
        y2_im = _mm512_sub_pd(t2_im, u2_im);                                        \
        y3_re = _mm512_add_pd(t2_re, u2_re);                                        \
        y3_im = _mm512_add_pd(t2_im, u2_im);                                        \
    } while (0)

/**
 * @brief Pipeline forward - NO TWIDDLES (remove twiddle loads from original)
 */
#define RADIX5_N1_PIPELINE_4_NATIVE_SOA_FV_AVX512(k, K, in_re, in_im, out_re, out_im, \
                                                  prefetch_dist, k_end)                \
    do                                                                                 \
    {                                                                                  \
        /* Loads - same as original */                                                \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                     \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                     \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                             \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                             \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                         \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                         \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                         \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                         \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                         \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                         \
        /* NO TWIDDLE LOADS - that's the only change! */                              \
        /* Prefetch - same as original */                                             \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                   \
        {                                                                              \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0);   \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0);   \
            _mm_prefetch((const char *)&in_re[(k) + (K) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (K) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                              \
        /* Butterfly - call N1 version */                                             \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                      \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);     \
        /* Stores - same as original */                                               \
        _mm512_storeu_pd(&out_re[k], y0_re);                                           \
        _mm512_storeu_pd(&out_im[k], y0_im);                                           \
        _mm512_storeu_pd(&out_re[(k) + (K)], y1_re);                                   \
        _mm512_storeu_pd(&out_im[(k) + (K)], y1_im);                                   \
        _mm512_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                               \
    } while (0)

/**
 * @brief Pipeline backward - NO TWIDDLES
 */
#define RADIX5_N1_PIPELINE_4_NATIVE_SOA_BV_AVX512(k, K, in_re, in_im, out_re, out_im, \
                                                  prefetch_dist, k_end)                \
    do                                                                                 \
    {                                                                                  \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                     \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                     \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                             \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                             \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                         \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                         \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                         \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                         \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                         \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                         \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                   \
        {                                                                              \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0);   \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0);   \
        }                                                                              \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                      \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);     \
        _mm512_storeu_pd(&out_re[k], y0_re);                                           \
        _mm512_storeu_pd(&out_im[k], y0_im);                                           \
        _mm512_storeu_pd(&out_re[(k) + (K)], y1_re);                                   \
        _mm512_storeu_pd(&out_im[(k) + (K)], y1_im);                                   \
        _mm512_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                               \
        _mm512_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                               \
    } while (0)

/**
 * @brief Streaming forward - NO TWIDDLES
 */
#define RADIX5_N1_PIPELINE_4_NATIVE_SOA_FV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                         prefetch_dist, k_end)                \
    do                                                                                        \
    {                                                                                         \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                            \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                            \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                                    \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                                    \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                                \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                                \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                                \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                                \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                                \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                                \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                          \
        {                                                                                     \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);         \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);         \
        }                                                                                     \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;        \
        RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                             \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                       \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);            \
        _mm512_stream_pd(&out_re[k], y0_re);                                                  \
        _mm512_stream_pd(&out_im[k], y0_im);                                                  \
        _mm512_stream_pd(&out_re[(k) + (K)], y1_re);                                          \
        _mm512_stream_pd(&out_im[(k) + (K)], y1_im);                                          \
        _mm512_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                      \
    } while (0)

/**
 * @brief Streaming backward - NO TWIDDLES
 */
#define RADIX5_N1_PIPELINE_4_NATIVE_SOA_BV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                         prefetch_dist, k_end)                \
    do                                                                                        \
    {                                                                                         \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                            \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                            \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                                    \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                                    \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                                \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                                \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                                \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                                \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                                \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                                \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                          \
        {                                                                                     \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);         \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);         \
        }                                                                                     \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;        \
        RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                             \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                       \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);            \
        _mm512_stream_pd(&out_re[k], y0_re);                                                  \
        _mm512_stream_pd(&out_im[k], y0_im);                                                  \
        _mm512_stream_pd(&out_re[(k) + (K)], y1_re);                                          \
        _mm512_stream_pd(&out_im[(k) + (K)], y1_im);                                          \
        _mm512_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                      \
        _mm512_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                      \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2: NO-TWIDDLE BUTTERFLIES (same pattern - remove CMUL & twiddle loads)
//==============================================================================

#ifdef __AVX2__

#define RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                     \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        /* NO CMUL - use b,c,d,e directly */                                       \
        __m256d s1_re = _mm256_add_pd(b_re, e_re);                                  \
        __m256d s1_im = _mm256_add_pd(b_im, e_im);                                  \
        __m256d s2_re = _mm256_add_pd(c_re, d_re);                                  \
        __m256d s2_im = _mm256_add_pd(c_im, d_im);                                  \
        __m256d d1_re = _mm256_sub_pd(b_re, e_re);                                  \
        __m256d d1_im = _mm256_sub_pd(b_im, e_im);                                  \
        __m256d d2_re = _mm256_sub_pd(c_re, d_re);                                  \
        __m256d d2_im = _mm256_sub_pd(c_im, d_im);                                  \
        /* Rest identical to original */                                           \
        y0_re = _mm256_add_pd(a_re, _mm256_add_pd(s1_re, s2_re));                   \
        y0_im = _mm256_add_pd(a_im, _mm256_add_pd(s1_im, s2_im));                   \
        const __m256d vc51 = _mm256_set1_pd(C5_1);                                  \
        const __m256d vc52 = _mm256_set1_pd(C5_2);                                  \
        const __m256d vs51 = _mm256_set1_pd(S5_1);                                  \
        const __m256d vs52 = _mm256_set1_pd(S5_2);                                  \
        __m256d t1_re = _mm256_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm256_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m256d t1_im = _mm256_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm256_fmadd_pd(vc52, s2_im, t1_im);                                \
        __m256d t2_re = _mm256_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm256_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m256d t2_im = _mm256_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm256_fmadd_pd(vc51, s2_im, t2_im);                                \
        __m256d base1_re = _mm256_mul_pd(vs51, d1_re);                              \
        base1_re = _mm256_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m256d base1_im = _mm256_mul_pd(vs51, d1_im);                              \
        base1_im = _mm256_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m256d u1_re = _mm256_sub_pd(_mm256_setzero_pd(), base1_im);              \
        __m256d u1_im = base1_re;                                                   \
        __m256d base2_re = _mm256_mul_pd(vs52, d1_re);                              \
        base2_re = _mm256_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m256d base2_im = _mm256_mul_pd(vs52, d1_im);                              \
        base2_im = _mm256_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m256d u2_re = _mm256_sub_pd(_mm256_setzero_pd(), base2_im);              \
        __m256d u2_im = base2_re;                                                   \
        y1_re = _mm256_add_pd(t1_re, u1_re);                                        \
        y1_im = _mm256_add_pd(t1_im, u1_im);                                        \
        y4_re = _mm256_sub_pd(t1_re, u1_re);                                        \
        y4_im = _mm256_sub_pd(t1_im, u1_im);                                        \
        y2_re = _mm256_sub_pd(t2_re, u2_re);                                        \
        y2_im = _mm256_sub_pd(t2_im, u2_im);                                        \
        y3_re = _mm256_add_pd(t2_re, u2_re);                                        \
        y3_im = _mm256_add_pd(t2_im, u2_im);                                        \
    } while (0)

#define RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                     \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m256d s1_re = _mm256_add_pd(b_re, e_re);                                  \
        __m256d s1_im = _mm256_add_pd(b_im, e_im);                                  \
        __m256d s2_re = _mm256_add_pd(c_re, d_re);                                  \
        __m256d s2_im = _mm256_add_pd(c_im, d_im);                                  \
        __m256d d1_re = _mm256_sub_pd(b_re, e_re);                                  \
        __m256d d1_im = _mm256_sub_pd(b_im, e_im);                                  \
        __m256d d2_re = _mm256_sub_pd(c_re, d_re);                                  \
        __m256d d2_im = _mm256_sub_pd(c_im, d_im);                                  \
        y0_re = _mm256_add_pd(a_re, _mm256_add_pd(s1_re, s2_re));                   \
        y0_im = _mm256_add_pd(a_im, _mm256_add_pd(s1_im, s2_im));                   \
        const __m256d vc51 = _mm256_set1_pd(C5_1);                                  \
        const __m256d vc52 = _mm256_set1_pd(C5_2);                                  \
        const __m256d vs51 = _mm256_set1_pd(S5_1);                                  \
        const __m256d vs52 = _mm256_set1_pd(S5_2);                                  \
        __m256d t1_re = _mm256_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm256_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m256d t1_im = _mm256_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm256_fmadd_pd(vc52, s2_im, t1_im);                                \
        __m256d t2_re = _mm256_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm256_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m256d t2_im = _mm256_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm256_fmadd_pd(vc51, s2_im, t2_im);                                \
        __m256d base1_re = _mm256_mul_pd(vs51, d1_re);                              \
        base1_re = _mm256_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m256d base1_im = _mm256_mul_pd(vs51, d1_im);                              \
        base1_im = _mm256_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m256d u1_re = base1_im;                                                   \
        __m256d u1_im = _mm256_sub_pd(_mm256_setzero_pd(), base1_re);              \
        __m256d base2_re = _mm256_mul_pd(vs52, d1_re);                              \
        base2_re = _mm256_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m256d base2_im = _mm256_mul_pd(vs52, d1_im);                              \
        base2_im = _mm256_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m256d u2_re = base2_im;                                                   \
        __m256d u2_im = _mm256_sub_pd(_mm256_setzero_pd(), base2_re);              \
        y1_re = _mm256_add_pd(t1_re, u1_re);                                        \
        y1_im = _mm256_add_pd(t1_im, u1_im);                                        \
        y4_re = _mm256_sub_pd(t1_re, u1_re);                                        \
        y4_im = _mm256_sub_pd(t1_im, u1_im);                                        \
        y2_re = _mm256_sub_pd(t2_re, u2_re);                                        \
        y2_im = _mm256_sub_pd(t2_im, u2_im);                                        \
        y3_re = _mm256_add_pd(t2_re, u2_re);                                        \
        y3_im = _mm256_add_pd(t2_im, u2_im);                                        \
    } while (0)

/* Pipeline macros for AVX2 - same pattern as AVX-512, just remove twiddle loads */
#define RADIX5_N1_PIPELINE_2_NATIVE_SOA_FV_AVX2(k, K, in_re, in_im, out_re, out_im, \
                                                prefetch_dist, k_end)                \
    do                                                                               \
    {                                                                                \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                   \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                   \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                           \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                           \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                       \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                       \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                       \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                       \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                       \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                       \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                 \
        {                                                                            \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                            \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                      \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);   \
        _mm256_storeu_pd(&out_re[k], y0_re);                                         \
        _mm256_storeu_pd(&out_im[k], y0_im);                                         \
        _mm256_storeu_pd(&out_re[(k) + (K)], y1_re);                                 \
        _mm256_storeu_pd(&out_im[(k) + (K)], y1_im);                                 \
        _mm256_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                             \
    } while (0)

#define RADIX5_N1_PIPELINE_2_NATIVE_SOA_BV_AVX2(k, K, in_re, in_im, out_re, out_im, \
                                                prefetch_dist, k_end)                \
    do                                                                               \
    {                                                                                \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                   \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                   \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                           \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                           \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                       \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                       \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                       \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                       \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                       \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                       \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                 \
        {                                                                            \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                            \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                      \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);   \
        _mm256_storeu_pd(&out_re[k], y0_re);                                         \
        _mm256_storeu_pd(&out_im[k], y0_im);                                         \
        _mm256_storeu_pd(&out_re[(k) + (K)], y1_re);                                 \
        _mm256_storeu_pd(&out_im[(k) + (K)], y1_im);                                 \
        _mm256_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                             \
        _mm256_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                             \
    } while (0)

/* Streaming versions - same deal */
#define RADIX5_N1_PIPELINE_2_NATIVE_SOA_FV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                       prefetch_dist, k_end)                \
    do                                                                                      \
    {                                                                                       \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                          \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                          \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                                  \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                                  \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                              \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                              \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                              \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                              \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                              \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                              \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                        \
        {                                                                                   \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);       \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);       \
        }                                                                                   \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;      \
        RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                             \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);          \
        _mm256_stream_pd(&out_re[k], y0_re);                                                \
        _mm256_stream_pd(&out_im[k], y0_im);                                                \
        _mm256_stream_pd(&out_re[(k) + (K)], y1_re);                                        \
        _mm256_stream_pd(&out_im[(k) + (K)], y1_im);                                        \
        _mm256_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                    \
    } while (0)

#define RADIX5_N1_PIPELINE_2_NATIVE_SOA_BV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                       prefetch_dist, k_end)                \
    do                                                                                      \
    {                                                                                       \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                          \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                          \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                                  \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                                  \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                              \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                              \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                              \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                              \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                              \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                              \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                        \
        {                                                                                   \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);       \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);       \
        }                                                                                   \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;      \
        RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                             \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);          \
        _mm256_stream_pd(&out_re[k], y0_re);                                                \
        _mm256_stream_pd(&out_im[k], y0_im);                                                \
        _mm256_stream_pd(&out_re[(k) + (K)], y1_re);                                        \
        _mm256_stream_pd(&out_im[(k) + (K)], y1_im);                                        \
        _mm256_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                    \
        _mm256_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                    \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SSE2 and SCALAR: Follow same pattern
//==============================================================================

#ifdef __SSE2__
/* SSE2 butterflies - same: remove CMUL, keep rest identical */
#define RADIX5_N1_BUTTERFLY_FV_NATIVE_SOA_SSE2(                                     \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m128d s1_re = _mm_add_pd(b_re, e_re);                                     \
        __m128d s1_im = _mm_add_pd(b_im, e_im);                                     \
        __m128d s2_re = _mm_add_pd(c_re, d_re);                                     \
        __m128d s2_im = _mm_add_pd(c_im, d_im);                                     \
        __m128d d1_re = _mm_sub_pd(b_re, e_re);                                     \
        __m128d d1_im = _mm_sub_pd(b_im, e_im);                                     \
        __m128d d2_re = _mm_sub_pd(c_re, d_re);                                     \
        __m128d d2_im = _mm_sub_pd(c_im, d_im);                                     \
        y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));                         \
        y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));                         \
        const __m128d vc51 = _mm_set1_pd(C5_1);                                     \
        const __m128d vc52 = _mm_set1_pd(C5_2);                                     \
        const __m128d vs51 = _mm_set1_pd(S5_1);                                     \
        const __m128d vs52 = _mm_set1_pd(S5_2);                                     \
        __m128d t1_re = _mm_add_pd(a_re,                                            \
                            _mm_add_pd(_mm_mul_pd(vc51, s1_re),                     \
                                      _mm_mul_pd(vc52, s2_re)));                    \
        __m128d t1_im = _mm_add_pd(a_im,                                            \
                            _mm_add_pd(_mm_mul_pd(vc51, s1_im),                     \
                                      _mm_mul_pd(vc52, s2_im)));                    \
        __m128d t2_re = _mm_add_pd(a_re,                                            \
                            _mm_add_pd(_mm_mul_pd(vc52, s1_re),                     \
                                      _mm_mul_pd(vc51, s2_re)));                    \
        __m128d t2_im = _mm_add_pd(a_im,                                            \
                            _mm_add_pd(_mm_mul_pd(vc52, s1_im),                     \
                                      _mm_mul_pd(vc51, s2_im)));                    \
        __m128d base1_re = _mm_add_pd(_mm_mul_pd(vs51, d1_re),                      \
                                     _mm_mul_pd(vs52, d2_re));                      \
        __m128d base1_im = _mm_add_pd(_mm_mul_pd(vs51, d1_im),                      \
                                     _mm_mul_pd(vs52, d2_im));                      \
        __m128d u1_re = _mm_sub_pd(_mm_setzero_pd(), base1_im);                     \
        __m128d u1_im = base1_re;                                                   \
        __m128d base2_re = _mm_sub_pd(_mm_mul_pd(vs52, d1_re),                      \
                                     _mm_mul_pd(vs51, d2_re));                      \
        __m128d base2_im = _mm_sub_pd(_mm_mul_pd(vs52, d1_im),                      \
                                     _mm_mul_pd(vs51, d2_im));                      \
        __m128d u2_re = _mm_sub_pd(_mm_setzero_pd(), base2_im);                     \
        __m128d u2_im = base2_re;                                                   \
        y1_re = _mm_add_pd(t1_re, u1_re);                                           \
        y1_im = _mm_add_pd(t1_im, u1_im);                                           \
        y4_re = _mm_sub_pd(t1_re, u1_re);                                           \
        y4_im = _mm_sub_pd(t1_im, u1_im);                                           \
        y2_re = _mm_sub_pd(t2_re, u2_re);                                           \
        y2_im = _mm_sub_pd(t2_im, u2_im);                                           \
        y3_re = _mm_add_pd(t2_re, u2_re);                                           \
        y3_im = _mm_add_pd(t2_im, u2_im);                                           \
    } while (0)

#define RADIX5_N1_BUTTERFLY_BV_NATIVE_SOA_SSE2(                                     \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m128d s1_re = _mm_add_pd(b_re, e_re);                                     \
        __m128d s1_im = _mm_add_pd(b_im, e_im);                                     \
        __m128d s2_re = _mm_add_pd(c_re, d_re);                                     \
        __m128d s2_im = _mm_add_pd(c_im, d_im);                                     \
        __m128d d1_re = _mm_sub_pd(b_re, e_re);                                     \
        __m128d d1_im = _mm_sub_pd(b_im, e_im);                                     \
        __m128d d2_re = _mm_sub_pd(c_re, d_re);                                     \
        __m128d d2_im = _mm_sub_pd(c_im, d_im);                                     \
        y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));                         \
        y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));                         \
        const __m128d vc51 = _mm_set1_pd(C5_1);                                     \
        const __m128d vc52 = _mm_set1_pd(C5_2);                                     \
        const __m128d vs51 = _mm_set1_pd(S5_1);                                     \
        const __m128d vs52 = _mm_set1_pd(S5_2);                                     \
        __m128d t1_re = _mm_add_pd(a_re,                                            \
                            _mm_add_pd(_mm_mul_pd(vc51, s1_re),                     \
                                      _mm_mul_pd(vc52, s2_re)));                    \
        __m128d t1_im = _mm_add_pd(a_im,                                            \
                            _mm_add_pd(_mm_mul_pd(vc51, s1_im),                     \
                                      _mm_mul_pd(vc52, s2_im)));                    \
        __m128d t2_re = _mm_add_pd(a_re,                                            \
                            _mm_add_pd(_mm_mul_pd(vc52, s1_re),                     \
                                      _mm_mul_pd(vc51, s2_re)));                    \
        __m128d t2_im = _mm_add_pd(a_im,                                            \
                            _mm_add_pd(_mm_mul_pd(vc52, s1_im),                     \
                                      _mm_mul_pd(vc51, s2_im)));                    \
        __m128d base1_re = _mm_add_pd(_mm_mul_pd(vs51, d1_re),                      \
                                     _mm_mul_pd(vs52, d2_re));                      \
        __m128d base1_im = _mm_add_pd(_mm_mul_pd(vs51, d1_im),                      \
                                     _mm_mul_pd(vs52, d2_im));                      \
        __m128d u1_re = base1_im;                                                   \
        __m128d u1_im = _mm_sub_pd(_mm_setzero_pd(), base1_re);                     \
        __m128d base2_re = _mm_sub_pd(_mm_mul_pd(vs52, d1_re),                      \
                                     _mm_mul_pd(vs51, d2_re));                      \
        __m128d base2_im = _mm_sub_pd(_mm_mul_pd(vs52, d1_im),                      \
                                     _mm_mul_pd(vs51, d2_im));                      \
        __m128d u2_re = base2_im;                                                   \
        __m128d u2_im = _mm_sub_pd(_mm_setzero_pd(), base2_re);                     \
        y1_re = _mm_add_pd(t1_re, u1_re);                                           \
        y1_im = _mm_add_pd(t1_im, u1_im);                                           \
        y4_re = _mm_sub_pd(t1_re, u1_re);                                           \
        y4_im = _mm_sub_pd(t1_im, u1_im);                                           \
        y2_re = _mm_sub_pd(t2_re, u2_re);                                           \
        y2_im = _mm_sub_pd(t2_im, u2_im);                                           \
        y3_re = _mm_add_pd(t2_re, u2_re);                                           \
        y3_im = _mm_add_pd(t2_im, u2_im);                                           \
    } while (0)

/* SSE2 pipeline macros omitted for brevity - same pattern as AVX */
#endif // __SSE2__

//==============================================================================
// SCALAR: No-twiddle versions
//==============================================================================

/**
 * @brief Scalar N1 forward - identical to original scalar FV but no twiddle mul
 */
#define RADIX5_N1_PIPELINE_1_NATIVE_SOA_FV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do                                                                                 \
    {                                                                                  \
        double a_re = in_re[k];                                                        \
        double a_im = in_im[k];                                                        \
        double b_re = in_re[(k) + (K)];                                                \
        double b_im = in_im[(k) + (K)];                                                \
        double c_re = in_re[(k) + 2 * (K)];                                            \
        double c_im = in_im[(k) + 2 * (K)];                                            \
        double d_re = in_re[(k) + 3 * (K)];                                            \
        double d_im = in_im[(k) + 3 * (K)];                                            \
        double e_re = in_re[(k) + 4 * (K)];                                            \
        double e_im = in_im[(k) + 4 * (K)];                                            \
        /* NO TWIDDLE MUL */                                                           \
        double s1_re = b_re + e_re;                                                    \
        double s1_im = b_im + e_im;                                                    \
        double s2_re = c_re + d_re;                                                    \
        double s2_im = c_im + d_im;                                                    \
        double d1_re = b_re - e_re;                                                    \
        double d1_im = b_im - e_im;                                                    \
        double d2_re = c_re - d_re;                                                    \
        double d2_im = c_im - d_im;                                                    \
        /* Rest identical to original */                                              \
        out_re[k] = a_re + s1_re + s2_re;                                              \
        out_im[k] = a_im + s1_im + s2_im;                                              \
        double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;                             \
        double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;                             \
        double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;                             \
        double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;                             \
        double base1_re = S5_1 * d1_re + S5_2 * d2_re;                                 \
        double base1_im = S5_1 * d1_im + S5_2 * d2_im;                                 \
        double u1_re = -base1_im;                                                      \
        double u1_im = base1_re;                                                       \
        double base2_re = S5_2 * d1_re - S5_1 * d2_re;                                 \
        double base2_im = S5_2 * d1_im - S5_1 * d2_im;                                 \
        double u2_re = -base2_im;                                                      \
        double u2_im = base2_re;                                                       \
        out_re[(k) + (K)] = t1_re + u1_re;                                             \
        out_im[(k) + (K)] = t1_im + u1_im;                                             \
        out_re[(k) + 4 * (K)] = t1_re - u1_re;                                         \
        out_im[(k) + 4 * (K)] = t1_im - u1_im;                                         \
        out_re[(k) + 2 * (K)] = t2_re - u2_re;                                         \
        out_im[(k) + 2 * (K)] = t2_im - u2_im;                                         \
        out_re[(k) + 3 * (K)] = t2_re + u2_re;                                         \
        out_im[(k) + 3 * (K)] = t2_im + u2_im;                                         \
    } while (0)

/**
 * @brief Scalar N1 backward - identical to original scalar BV but no twiddle mul
 */
#define RADIX5_N1_PIPELINE_1_NATIVE_SOA_BV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do                                                                                 \
    {                                                                                  \
        double a_re = in_re[k];                                                        \
        double a_im = in_im[k];                                                        \
        double b_re = in_re[(k) + (K)];                                                \
        double b_im = in_im[(k) + (K)];                                                \
        double c_re = in_re[(k) + 2 * (K)];                                            \
        double c_im = in_im[(k) + 2 * (K)];                                            \
        double d_re = in_re[(k) + 3 * (K)];                                            \
        double d_im = in_im[(k) + 3 * (K)];                                            \
        double e_re = in_re[(k) + 4 * (K)];                                            \
        double e_im = in_im[(k) + 4 * (K)];                                            \
        /* NO TWIDDLE MUL */                                                           \
        double s1_re = b_re + e_re;                                                    \
        double s1_im = b_im + e_im;                                                    \
        double s2_re = c_re + d_re;                                                    \
        double s2_im = c_im + d_im;                                                    \
        double d1_re = b_re - e_re;                                                    \
        double d1_im = b_im - e_im;                                                    \
        double d2_re = c_re - d_re;                                                    \
        double d2_im = c_im - d_im;                                                    \
        /* Rest identical to original */                                              \
        out_re[k] = a_re + s1_re + s2_re;                                              \
        out_im[k] = a_im + s1_im + s2_im;                                              \
        double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;                             \
        double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;                             \
        double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;                             \
        double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;                             \
        double base1_re = S5_1 * d1_re + S5_2 * d2_re;                                 \
        double base1_im = S5_1 * d1_im + S5_2 * d2_im;                                 \
        double u1_re = base1_im;                                                       \
        double u1_im = -base1_re;                                                      \
        double base2_re = S5_2 * d1_re - S5_1 * d2_re;                                 \
        double base2_im = S5_2 * d1_im - S5_1 * d2_im;                                 \
        double u2_re = base2_im;                                                       \
        double u2_im = -base2_re;                                                      \
        out_re[(k) + (K)] = t1_re + u1_re;                                             \
        out_im[(k) + (K)] = t1_im + u1_im;                                             \
        out_re[(k) + 4 * (K)] = t1_re - u1_re;                                         \
        out_im[(k) + 4 * (K)] = t1_im - u1_im;                                         \
        out_re[(k) + 2 * (K)] = t2_re - u2_re;                                         \
        out_im[(k) + 2 * (K)] = t2_im - u2_im;                                         \
        out_re[(k) + 3 * (K)] = t2_re + u2_re;                                         \
        out_im[(k) + 3 * (K)] = t2_im + u2_im;                                         \
    } while (0)

#endif // FFT_RADIX5_N1_NOTWIDDLE_H

/**
 * USAGE NOTES:
 * ============
 * These macros are IDENTICAL to the twiddle versions except:
 * 1. Removed CMUL operations
 * 2. Removed twiddle factor loads
 * 3. Removed stage_tw parameter from pipeline macros
 * 
 * Use for first FFT stage where all twiddles are 1+0i
 * 
 * Example:
 *   int K = N / 5;
 *   for (int k = 0; k < K; k += 8) {
 *       RADIX5_N1_PIPELINE_4_NATIVE_SOA_FV_AVX512(
 *           k, K, in_re, in_im, out_re, out_im, 32, K);
 *   }
 */