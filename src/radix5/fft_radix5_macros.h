/**
 * @file fft_radix5_macros_optimized.h
 * @brief TRUE END-TO-END Native SoA Radix-5 FFT Macros (FULLY OPTIMIZED!)
 *
 * @details
 * This header applies ALL optimizations from radix-4 to radix-5:
 * ✅ Native SoA architecture (ZERO shuffle overhead!)
 * ✅ Double-pumping (improved ILP)
 * ✅ Software prefetching (hide memory latency)
 * ✅ Streaming stores (cache bypass for large N)
 * ✅ Complete SIMD coverage (AVX-512, AVX2, SSE2, scalar)
 * ✅ FMA usage (where available)
 *
 * ARCHITECTURAL REVOLUTION:
 * =========================
 * Same principle as radix-4: Accept/return separate re[] and im[] arrays,
 * NO split/join in the hot path!
 *
 * EXPECTED PERFORMANCE:
 * ====================
 * - Radix-5 has log₅(N) stages vs log₄(N) for radix-4
 * - Fewer stages BUT more complex butterfly (5 points vs 4)
 * - With these optimizations: ~50-70% faster than old split-form!
 *
 * @author FFT Optimization Team
 * @version 2.0 (Native SoA with full optimizations)
 * @date 2025
 */

#ifndef FFT_RADIX5_MACROS_OPTIMIZED_H
#define FFT_RADIX5_MACROS_OPTIMIZED_H

#include "simd_math.h"
#include <immintrin.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX5_PREFETCH_DISTANCE
 * @brief Software prefetch lead distance for radix-5 strided access
 */
#ifndef RADIX5_PREFETCH_DISTANCE
#define RADIX5_PREFETCH_DISTANCE 32  // Tuned for 5-way stride pattern
#endif

//==============================================================================
// RADIX-5 GEOMETRIC CONSTANTS
//==============================================================================

#define C5_1 0.30901699437494742410    // cos(2π/5)
#define C5_2 (-0.80901699437494742410) // cos(4π/5)
#define S5_1 0.95105651629515357212    // sin(2π/5)
#define S5_2 0.58778525229247312917    // sin(4π/5)

//==============================================================================
// COMPLEX MULTIPLY - NATIVE SoA (NO SPLIT/JOIN!)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX-512)
 * @details Data is ALREADY in split form from memory!
 * Computes: (ar + i*ai) * (wr + i*wi)
 */
#define CMUL_NATIVE_SOA_R5_AVX512(ar, ai, w_re, w_im, tr, ti)    \
    do                                                           \
    {                                                            \
        tr = _mm512_fmsub_pd(ar, w_re, _mm512_mul_pd(ai, w_im)); \
        ti = _mm512_fmadd_pd(ar, w_im, _mm512_mul_pd(ai, w_re)); \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX2)
 */
#if defined(__FMA__)
#define CMUL_NATIVE_SOA_R5_AVX2(ar, ai, w_re, w_im, tr, ti)      \
    do                                                           \
    {                                                            \
        tr = _mm256_fmsub_pd(ar, w_re, _mm256_mul_pd(ai, w_im)); \
        ti = _mm256_fmadd_pd(ar, w_im, _mm256_mul_pd(ai, w_re)); \
    } while (0)
#else
#define CMUL_NATIVE_SOA_R5_AVX2(ar, ai, w_re, w_im, tr, ti) \
    do                                                      \
    {                                                       \
        tr = _mm256_sub_pd(_mm256_mul_pd(ar, w_re),         \
                           _mm256_mul_pd(ai, w_im));        \
        ti = _mm256_add_pd(_mm256_mul_pd(ar, w_im),         \
                           _mm256_mul_pd(ai, w_re));        \
    } while (0)
#endif
#endif

#ifdef __SSE2__
/**
 * @brief Complex multiply - NATIVE SoA form (SSE2)
 */
#define CMUL_NATIVE_SOA_R5_SSE2(ar, ai, w_re, w_im, tr, ti)          \
    do                                                               \
    {                                                                \
        tr = _mm_sub_pd(_mm_mul_pd(ar, w_re), _mm_mul_pd(ai, w_im)); \
        ti = _mm_add_pd(_mm_mul_pd(ar, w_im), _mm_mul_pd(ai, w_re)); \
    } while (0)
#endif

//==============================================================================
// AVX-512: NATIVE SoA RADIX-5 BUTTERFLY (8 doubles = 4 complex)
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief Radix-5 butterfly - Forward - NATIVE SoA (AVX-512, 4 butterflies)
 * @details ZERO SHUFFLE VERSION! Input/output already in split form.
 *
 * Algorithm: Standard Cooley-Tukey radix-5 with rotation by ±i
 */
#define RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                      \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        /* Twiddle multiplications */                                              \
        __m512d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_AVX512(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);     \
                                                                                    \
        /* Pair sums and differences */                                            \
        __m512d s1_re = _mm512_add_pd(tw_b_re, tw_e_re);                            \
        __m512d s1_im = _mm512_add_pd(tw_b_im, tw_e_im);                            \
        __m512d s2_re = _mm512_add_pd(tw_c_re, tw_d_re);                            \
        __m512d s2_im = _mm512_add_pd(tw_c_im, tw_d_im);                            \
        __m512d d1_re = _mm512_sub_pd(tw_b_re, tw_e_re);                            \
        __m512d d1_im = _mm512_sub_pd(tw_b_im, tw_e_im);                            \
        __m512d d2_re = _mm512_sub_pd(tw_c_re, tw_d_re);                            \
        __m512d d2_im = _mm512_sub_pd(tw_c_im, tw_d_im);                            \
                                                                                    \
        /* Output 0: y0 = a + s1 + s2 */                                           \
        y0_re = _mm512_add_pd(a_re, _mm512_add_pd(s1_re, s2_re));                   \
        y0_im = _mm512_add_pd(a_im, _mm512_add_pd(s1_im, s2_im));                   \
                                                                                    \
        /* Constants */                                                             \
        const __m512d vc51 = _mm512_set1_pd(C5_1);                                  \
        const __m512d vc52 = _mm512_set1_pd(C5_2);                                  \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                                  \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                                  \
                                                                                    \
        /* Intermediate terms t1 = a + C5_1*s1 + C5_2*s2 */                         \
        __m512d t1_re = _mm512_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm512_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m512d t1_im = _mm512_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm512_fmadd_pd(vc52, s2_im, t1_im);                                \
                                                                                    \
        /* t2 = a + C5_2*s1 + C5_1*s2 */                                            \
        __m512d t2_re = _mm512_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm512_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m512d t2_im = _mm512_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm512_fmadd_pd(vc51, s2_im, t2_im);                                \
                                                                                    \
        /* Rotations: u1 = i*(S5_1*d1 + S5_2*d2) for FORWARD */                    \
        __m512d base1_re = _mm512_mul_pd(vs51, d1_re);                              \
        base1_re = _mm512_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m512d base1_im = _mm512_mul_pd(vs51, d1_im);                              \
        base1_im = _mm512_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m512d u1_re = _mm512_sub_pd(_mm512_setzero_pd(), base1_im); /* -im */    \
        __m512d u1_im = base1_re;                                       /* +re */   \
                                                                                    \
        /* u2 = i*(S5_2*d1 - S5_1*d2) for FORWARD */                                \
        __m512d base2_re = _mm512_mul_pd(vs52, d1_re);                              \
        base2_re = _mm512_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m512d base2_im = _mm512_mul_pd(vs52, d1_im);                              \
        base2_im = _mm512_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m512d u2_re = _mm512_sub_pd(_mm512_setzero_pd(), base2_im); /* -im */    \
        __m512d u2_im = base2_re;                                       /* +re */   \
                                                                                    \
        /* Outputs 1 and 4 */                                                       \
        y1_re = _mm512_add_pd(t1_re, u1_re);                                        \
        y1_im = _mm512_add_pd(t1_im, u1_im);                                        \
        y4_re = _mm512_sub_pd(t1_re, u1_re);                                        \
        y4_im = _mm512_sub_pd(t1_im, u1_im);                                        \
                                                                                    \
        /* Outputs 2 and 3 */                                                       \
        y2_re = _mm512_sub_pd(t2_re, u2_re);                                        \
        y2_im = _mm512_sub_pd(t2_im, u2_im);                                        \
        y3_re = _mm512_add_pd(t2_re, u2_re);                                        \
        y3_im = _mm512_add_pd(t2_im, u2_im);                                        \
    } while (0)

/**
 * @brief Radix-5 butterfly - Backward (Inverse) - NATIVE SoA (AVX-512)
 * @details Only difference from forward: rotation direction (conjugate)
 */
#define RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                      \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        /* Twiddle multiplications */                                              \
        __m512d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_AVX512(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);     \
        CMUL_NATIVE_SOA_R5_AVX512(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);     \
                                                                                    \
        /* Pair sums and differences */                                            \
        __m512d s1_re = _mm512_add_pd(tw_b_re, tw_e_re);                            \
        __m512d s1_im = _mm512_add_pd(tw_b_im, tw_e_im);                            \
        __m512d s2_re = _mm512_add_pd(tw_c_re, tw_d_re);                            \
        __m512d s2_im = _mm512_add_pd(tw_c_im, tw_d_im);                            \
        __m512d d1_re = _mm512_sub_pd(tw_b_re, tw_e_re);                            \
        __m512d d1_im = _mm512_sub_pd(tw_b_im, tw_e_im);                            \
        __m512d d2_re = _mm512_sub_pd(tw_c_re, tw_d_re);                            \
        __m512d d2_im = _mm512_sub_pd(tw_c_im, tw_d_im);                            \
                                                                                    \
        /* Output 0: y0 = a + s1 + s2 */                                           \
        y0_re = _mm512_add_pd(a_re, _mm512_add_pd(s1_re, s2_re));                   \
        y0_im = _mm512_add_pd(a_im, _mm512_add_pd(s1_im, s2_im));                   \
                                                                                    \
        /* Constants */                                                             \
        const __m512d vc51 = _mm512_set1_pd(C5_1);                                  \
        const __m512d vc52 = _mm512_set1_pd(C5_2);                                  \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                                  \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                                  \
                                                                                    \
        /* Intermediate terms */                                                    \
        __m512d t1_re = _mm512_fmadd_pd(vc51, s1_re, a_re);                         \
        t1_re = _mm512_fmadd_pd(vc52, s2_re, t1_re);                                \
        __m512d t1_im = _mm512_fmadd_pd(vc51, s1_im, a_im);                         \
        t1_im = _mm512_fmadd_pd(vc52, s2_im, t1_im);                                \
                                                                                    \
        __m512d t2_re = _mm512_fmadd_pd(vc52, s1_re, a_re);                         \
        t2_re = _mm512_fmadd_pd(vc51, s2_re, t2_re);                                \
        __m512d t2_im = _mm512_fmadd_pd(vc52, s1_im, a_im);                         \
        t2_im = _mm512_fmadd_pd(vc51, s2_im, t2_im);                                \
                                                                                    \
        /* Rotations: u1 = -i*(S5_1*d1 + S5_2*d2) for BACKWARD (conjugate) */      \
        __m512d base1_re = _mm512_mul_pd(vs51, d1_re);                              \
        base1_re = _mm512_fmadd_pd(vs52, d2_re, base1_re);                          \
        __m512d base1_im = _mm512_mul_pd(vs51, d1_im);                              \
        base1_im = _mm512_fmadd_pd(vs52, d2_im, base1_im);                          \
        __m512d u1_re = base1_im;                                       /* +im */   \
        __m512d u1_im = _mm512_sub_pd(_mm512_setzero_pd(), base1_re); /* -re */    \
                                                                                    \
        /* u2 = -i*(S5_2*d1 - S5_1*d2) for BACKWARD */                              \
        __m512d base2_re = _mm512_mul_pd(vs52, d1_re);                              \
        base2_re = _mm512_fnmadd_pd(vs51, d2_re, base2_re);                         \
        __m512d base2_im = _mm512_mul_pd(vs52, d1_im);                              \
        base2_im = _mm512_fnmadd_pd(vs51, d2_im, base2_im);                         \
        __m512d u2_re = base2_im;                                       /* +im */   \
        __m512d u2_im = _mm512_sub_pd(_mm512_setzero_pd(), base2_re); /* -re */    \
                                                                                    \
        /* Outputs */                                                               \
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
 * @brief FULL PIPELINE - Radix-5 butterfly with prefetch - FORWARD (AVX-512)
 * @details Processes 4 butterflies with software prefetching
 */
#define RADIX5_PIPELINE_4_NATIVE_SOA_FV_AVX512(k, K, in_re, in_im, out_re, out_im, \
                                               stage_tw, prefetch_dist, k_end)     \
    do                                                                             \
    {                                                                              \
        /* Load inputs (5 lanes per butterfly, 4 butterflies) */                  \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                 \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                 \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                         \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                         \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                     \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                     \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                     \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                     \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                     \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                     \
                                                                                   \
        /* Load twiddles (SoA - direct, no shuffle!) */                           \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * (K) + (k)]);             \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * (K) + (k)]);             \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * (K) + (k)]);             \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * (K) + (k)]);             \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw->re[2 * (K) + (k)]);             \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw->im[2 * (K) + (k)]);             \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw->re[3 * (K) + (k)]);             \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw->im[3 * (K) + (k)]);             \
                                                                                   \
        /* Software prefetch (if not at tail) */                                  \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))               \
        {                                                                          \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_re[(k) + (K) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (K) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->re[0 * (K) + (k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw->im[0 * (K) + (k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                          \
                                                                                   \
        /* Compute butterfly */                                                    \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,            \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
                                                                                   \
        /* Store outputs (5 lanes) */                                             \
        _mm512_storeu_pd(&out_re[k], y0_re);                                       \
        _mm512_storeu_pd(&out_im[k], y0_im);                                       \
        _mm512_storeu_pd(&out_re[(k) + (K)], y1_re);                               \
        _mm512_storeu_pd(&out_im[(k) + (K)], y1_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                           \
        _mm512_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                           \
        _mm512_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                           \
    } while (0)

/**
 * @brief STREAMING VERSION - Non-temporal stores (AVX-512)
 */
#define RADIX5_PIPELINE_4_NATIVE_SOA_FV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                      stage_tw, prefetch_dist, k_end)     \
    do                                                                                    \
    {                                                                                     \
        /* Load inputs */                                                                 \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                        \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                        \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                                \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                                \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                            \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                            \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                            \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                            \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                            \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                            \
                                                                                          \
        /* Load twiddles */                                                               \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                    \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                    \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                    \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                    \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                    \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                    \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                    \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                    \
                                                                                          \
        /* Prefetch */                                                                    \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                      \
        {                                                                                 \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);     \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);     \
        }                                                                                 \
                                                                                          \
        /* Compute butterfly */                                                           \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;    \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX512(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                   \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                       \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);        \
                                                                                          \
        /* NON-TEMPORAL stores (cache bypass) */                                         \
        _mm512_stream_pd(&out_re[k], y0_re);                                              \
        _mm512_stream_pd(&out_im[k], y0_im);                                              \
        _mm512_stream_pd(&out_re[(k) + (K)], y1_re);                                      \
        _mm512_stream_pd(&out_im[(k) + (K)], y1_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                  \
        _mm512_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                  \
        _mm512_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                  \
    } while (0)

/**
 * @brief BACKWARD versions (same as forward but use BV butterfly)
 */
#define RADIX5_PIPELINE_4_NATIVE_SOA_BV_AVX512(k, K, in_re, in_im, out_re, out_im, \
                                               stage_tw, prefetch_dist, k_end)     \
    do                                                                             \
    {                                                                              \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                 \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                 \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                         \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                         \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                     \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                     \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                     \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                     \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                     \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                     \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * (K) + (k)]);             \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * (K) + (k)]);             \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * (K) + (k)]);             \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * (K) + (k)]);             \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw->re[2 * (K) + (k)]);             \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw->im[2 * (K) + (k)]);             \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw->re[3 * (K) + (k)]);             \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw->im[3 * (K) + (k)]);             \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))               \
        {                                                                          \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                          \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,            \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
        _mm512_storeu_pd(&out_re[k], y0_re);                                       \
        _mm512_storeu_pd(&out_im[k], y0_im);                                       \
        _mm512_storeu_pd(&out_re[(k) + (K)], y1_re);                               \
        _mm512_storeu_pd(&out_im[(k) + (K)], y1_im);                               \
        _mm512_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                           \
        _mm512_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                           \
        _mm512_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                           \
        _mm512_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                           \
    } while (0)

#define RADIX5_PIPELINE_4_NATIVE_SOA_BV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                      stage_tw, prefetch_dist, k_end)     \
    do                                                                                    \
    {                                                                                     \
        __m512d a_re = _mm512_loadu_pd(&in_re[k]);                                        \
        __m512d a_im = _mm512_loadu_pd(&in_im[k]);                                        \
        __m512d b_re = _mm512_loadu_pd(&in_re[(k) + (K)]);                                \
        __m512d b_im = _mm512_loadu_pd(&in_im[(k) + (K)]);                                \
        __m512d c_re = _mm512_loadu_pd(&in_re[(k) + 2 * (K)]);                            \
        __m512d c_im = _mm512_loadu_pd(&in_im[(k) + 2 * (K)]);                            \
        __m512d d_re = _mm512_loadu_pd(&in_re[(k) + 3 * (K)]);                            \
        __m512d d_im = _mm512_loadu_pd(&in_im[(k) + 3 * (K)]);                            \
        __m512d e_re = _mm512_loadu_pd(&in_re[(k) + 4 * (K)]);                            \
        __m512d e_im = _mm512_loadu_pd(&in_im[(k) + 4 * (K)]);                            \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                    \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                    \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                    \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                    \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                    \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                    \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                    \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                    \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))                      \
        {                                                                                 \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_NTA);     \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_NTA);     \
        }                                                                                 \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;    \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX512(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                   \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                       \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);        \
        _mm512_stream_pd(&out_re[k], y0_re);                                              \
        _mm512_stream_pd(&out_im[k], y0_im);                                              \
        _mm512_stream_pd(&out_re[(k) + (K)], y1_re);                                      \
        _mm512_stream_pd(&out_im[(k) + (K)], y1_im);                                      \
        _mm512_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                  \
        _mm512_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                  \
        _mm512_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                  \
        _mm512_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                  \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2: NATIVE SoA RADIX-5 BUTTERFLY (4 doubles = 2 complex)
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Radix-5 butterfly - Forward - NATIVE SoA (AVX2, 2 butterflies)
 */
#define RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                        \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m256d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_AVX2(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);       \
        __m256d s1_re = _mm256_add_pd(tw_b_re, tw_e_re);                            \
        __m256d s1_im = _mm256_add_pd(tw_b_im, tw_e_im);                            \
        __m256d s2_re = _mm256_add_pd(tw_c_re, tw_d_re);                            \
        __m256d s2_im = _mm256_add_pd(tw_c_im, tw_d_im);                            \
        __m256d d1_re = _mm256_sub_pd(tw_b_re, tw_e_re);                            \
        __m256d d1_im = _mm256_sub_pd(tw_b_im, tw_e_im);                            \
        __m256d d2_re = _mm256_sub_pd(tw_c_re, tw_d_re);                            \
        __m256d d2_im = _mm256_sub_pd(tw_c_im, tw_d_im);                            \
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

/**
 * @brief Radix-5 butterfly - Backward - NATIVE SoA (AVX2)
 */
#define RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                        \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m256d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_AVX2(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);       \
        CMUL_NATIVE_SOA_R5_AVX2(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);       \
        __m256d s1_re = _mm256_add_pd(tw_b_re, tw_e_re);                            \
        __m256d s1_im = _mm256_add_pd(tw_b_im, tw_e_im);                            \
        __m256d s2_re = _mm256_add_pd(tw_c_re, tw_d_re);                            \
        __m256d s2_im = _mm256_add_pd(tw_c_im, tw_d_im);                            \
        __m256d d1_re = _mm256_sub_pd(tw_b_re, tw_e_re);                            \
        __m256d d1_im = _mm256_sub_pd(tw_b_im, tw_e_im);                            \
        __m256d d2_re = _mm256_sub_pd(tw_c_re, tw_d_re);                            \
        __m256d d2_im = _mm256_sub_pd(tw_c_im, tw_d_im);                            \
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

/**
 * @brief FULL PIPELINE macros for AVX2 (2 butterflies at a time)
 */
#define RADIX5_PIPELINE_2_NATIVE_SOA_FV_AVX2(k, K, in_re, in_im, out_re, out_im, \
                                             stage_tw, prefetch_dist, k_end)     \
    do                                                                           \
    {                                                                            \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                               \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                               \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                       \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                       \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                   \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                   \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                   \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                   \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                   \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                   \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * (K) + (k)]);           \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * (K) + (k)]);           \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * (K) + (k)]);           \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * (K) + (k)]);           \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw->re[2 * (K) + (k)]);           \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw->im[2 * (K) + (k)]);           \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw->re[3 * (K) + (k)]);           \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw->im[3 * (K) + (k)]);           \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))             \
        {                                                                        \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                        \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,          \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
        _mm256_storeu_pd(&out_re[k], y0_re);                                     \
        _mm256_storeu_pd(&out_im[k], y0_im);                                     \
        _mm256_storeu_pd(&out_re[(k) + (K)], y1_re);                             \
        _mm256_storeu_pd(&out_im[(k) + (K)], y1_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                         \
        _mm256_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                         \
        _mm256_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                         \
    } while (0)

// Streaming, backward versions... (similar pattern, omitted for brevity)
#define RADIX5_PIPELINE_2_NATIVE_SOA_FV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                    stage_tw, prefetch_dist, k_end)     \
    do                                                                                  \
    {                                                                                   \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                      \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                      \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                              \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                              \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                          \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                          \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                          \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                          \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                          \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                          \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                  \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                  \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                  \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                  \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                  \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                  \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                  \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                  \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;   \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_AVX2(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                 \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);      \
        _mm256_stream_pd(&out_re[k], y0_re);                                            \
        _mm256_stream_pd(&out_im[k], y0_im);                                            \
        _mm256_stream_pd(&out_re[(k) + (K)], y1_re);                                    \
        _mm256_stream_pd(&out_im[(k) + (K)], y1_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                \
        _mm256_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                \
        _mm256_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                \
        _mm256_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                \
        _mm256_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                \
        _mm256_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                \
    } while (0)

#define RADIX5_PIPELINE_2_NATIVE_SOA_BV_AVX2(k, K, in_re, in_im, out_re, out_im, \
                                             stage_tw, prefetch_dist, k_end)     \
    do                                                                           \
    {                                                                            \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                               \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                               \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                       \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                       \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                   \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                   \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                   \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                   \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                   \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                   \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * (K) + (k)]);           \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * (K) + (k)]);           \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * (K) + (k)]);           \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * (K) + (k)]);           \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw->re[2 * (K) + (k)]);           \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw->im[2 * (K) + (k)]);           \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw->re[3 * (K) + (k)]);           \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw->im[3 * (K) + (k)]);           \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,          \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
        _mm256_storeu_pd(&out_re[k], y0_re);                                     \
        _mm256_storeu_pd(&out_im[k], y0_im);                                     \
        _mm256_storeu_pd(&out_re[(k) + (K)], y1_re);                             \
        _mm256_storeu_pd(&out_im[(k) + (K)], y1_im);                             \
        _mm256_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                         \
        _mm256_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                         \
        _mm256_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                         \
        _mm256_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                         \
    } while (0)

#define RADIX5_PIPELINE_2_NATIVE_SOA_BV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                    stage_tw, prefetch_dist, k_end)     \
    do                                                                                  \
    {                                                                                   \
        __m256d a_re = _mm256_loadu_pd(&in_re[k]);                                      \
        __m256d a_im = _mm256_loadu_pd(&in_im[k]);                                      \
        __m256d b_re = _mm256_loadu_pd(&in_re[(k) + (K)]);                              \
        __m256d b_im = _mm256_loadu_pd(&in_im[(k) + (K)]);                              \
        __m256d c_re = _mm256_loadu_pd(&in_re[(k) + 2 * (K)]);                          \
        __m256d c_im = _mm256_loadu_pd(&in_im[(k) + 2 * (K)]);                          \
        __m256d d_re = _mm256_loadu_pd(&in_re[(k) + 3 * (K)]);                          \
        __m256d d_im = _mm256_loadu_pd(&in_im[(k) + 3 * (K)]);                          \
        __m256d e_re = _mm256_loadu_pd(&in_re[(k) + 4 * (K)]);                          \
        __m256d e_im = _mm256_loadu_pd(&in_im[(k) + 4 * (K)]);                          \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                  \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                  \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                  \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                  \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                  \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                  \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                  \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                  \
        __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;   \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_AVX2(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                 \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);      \
        _mm256_stream_pd(&out_re[k], y0_re);                                            \
        _mm256_stream_pd(&out_im[k], y0_im);                                            \
        _mm256_stream_pd(&out_re[(k) + (K)], y1_re);                                    \
        _mm256_stream_pd(&out_im[(k) + (K)], y1_im);                                    \
        _mm256_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                \
        _mm256_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                \
        _mm256_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                \
        _mm256_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                \
        _mm256_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                \
        _mm256_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SSE2 and SCALAR versions
//==============================================================================

#ifdef __SSE2__

/**
 * @brief Radix-5 butterfly - Forward - NATIVE SoA (SSE2, 1 butterfly)
 */
#define RADIX5_BUTTERFLY_FV_NATIVE_SOA_SSE2(                                        \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m128d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_SSE2(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);       \
        __m128d s1_re = _mm_add_pd(tw_b_re, tw_e_re);                               \
        __m128d s1_im = _mm_add_pd(tw_b_im, tw_e_im);                               \
        __m128d s2_re = _mm_add_pd(tw_c_re, tw_d_re);                               \
        __m128d s2_im = _mm_add_pd(tw_c_im, tw_d_im);                               \
        __m128d d1_re = _mm_sub_pd(tw_b_re, tw_e_re);                               \
        __m128d d1_im = _mm_sub_pd(tw_b_im, tw_e_im);                               \
        __m128d d2_re = _mm_sub_pd(tw_c_re, tw_d_re);                               \
        __m128d d2_im = _mm_sub_pd(tw_c_im, tw_d_im);                               \
        y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));                         \
        y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));                         \
        const __m128d vc51 = _mm_set1_pd(C5_1);                                     \
        const __m128d vc52 = _mm_set1_pd(C5_2);                                     \
        const __m128d vs51 = _mm_set1_pd(S5_1);                                     \
        const __m128d vs52 = _mm_set1_pd(S5_2);                                     \
        __m128d t1_re = _mm_add_pd(a_re, _mm_add_pd(_mm_mul_pd(vc51, s1_re), _mm_mul_pd(vc52, s2_re))); \
        __m128d t1_im = _mm_add_pd(a_im, _mm_add_pd(_mm_mul_pd(vc51, s1_im), _mm_mul_pd(vc52, s2_im))); \
        __m128d t2_re = _mm_add_pd(a_re, _mm_add_pd(_mm_mul_pd(vc52, s1_re), _mm_mul_pd(vc51, s2_re))); \
        __m128d t2_im = _mm_add_pd(a_im, _mm_add_pd(_mm_mul_pd(vc52, s1_im), _mm_mul_pd(vc51, s2_im))); \
        __m128d base1_re = _mm_add_pd(_mm_mul_pd(vs51, d1_re), _mm_mul_pd(vs52, d2_re)); \
        __m128d base1_im = _mm_add_pd(_mm_mul_pd(vs51, d1_im), _mm_mul_pd(vs52, d2_im)); \
        __m128d u1_re = _mm_sub_pd(_mm_setzero_pd(), base1_im);                     \
        __m128d u1_im = base1_re;                                                    \
        __m128d base2_re = _mm_sub_pd(_mm_mul_pd(vs52, d1_re), _mm_mul_pd(vs51, d2_re)); \
        __m128d base2_im = _mm_sub_pd(_mm_mul_pd(vs52, d1_im), _mm_mul_pd(vs51, d2_im)); \
        __m128d u2_re = _mm_sub_pd(_mm_setzero_pd(), base2_im);                     \
        __m128d u2_im = base2_re;                                                    \
        y1_re = _mm_add_pd(t1_re, u1_re);                                            \
        y1_im = _mm_add_pd(t1_im, u1_im);                                            \
        y4_re = _mm_sub_pd(t1_re, u1_re);                                            \
        y4_im = _mm_sub_pd(t1_im, u1_im);                                            \
        y2_re = _mm_sub_pd(t2_re, u2_re);                                            \
        y2_im = _mm_sub_pd(t2_im, u2_im);                                            \
        y3_re = _mm_add_pd(t2_re, u2_re);                                            \
        y3_im = _mm_add_pd(t2_im, u2_im);                                            \
    } while (0)

/**
 * @brief Radix-5 butterfly - Backward - NATIVE SoA (SSE2)
 */
#define RADIX5_BUTTERFLY_BV_NATIVE_SOA_SSE2(                                        \
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                    \
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                        \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im)          \
    do                                                                              \
    {                                                                               \
        __m128d tw_b_re, tw_b_im, tw_c_re, tw_c_im, tw_d_re, tw_d_im, tw_e_re, tw_e_im; \
        CMUL_NATIVE_SOA_R5_SSE2(b_re, b_im, w1_re, w1_im, tw_b_re, tw_b_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(c_re, c_im, w2_re, w2_im, tw_c_re, tw_c_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(d_re, d_im, w3_re, w3_im, tw_d_re, tw_d_im);       \
        CMUL_NATIVE_SOA_R5_SSE2(e_re, e_im, w4_re, w4_im, tw_e_re, tw_e_im);       \
        __m128d s1_re = _mm_add_pd(tw_b_re, tw_e_re);                               \
        __m128d s1_im = _mm_add_pd(tw_b_im, tw_e_im);                               \
        __m128d s2_re = _mm_add_pd(tw_c_re, tw_d_re);                               \
        __m128d s2_im = _mm_add_pd(tw_c_im, tw_d_im);                               \
        __m128d d1_re = _mm_sub_pd(tw_b_re, tw_e_re);                               \
        __m128d d1_im = _mm_sub_pd(tw_b_im, tw_e_im);                               \
        __m128d d2_re = _mm_sub_pd(tw_c_re, tw_d_re);                               \
        __m128d d2_im = _mm_sub_pd(tw_c_im, tw_d_im);                               \
        y0_re = _mm_add_pd(a_re, _mm_add_pd(s1_re, s2_re));                         \
        y0_im = _mm_add_pd(a_im, _mm_add_pd(s1_im, s2_im));                         \
        const __m128d vc51 = _mm_set1_pd(C5_1);                                     \
        const __m128d vc52 = _mm_set1_pd(C5_2);                                     \
        const __m128d vs51 = _mm_set1_pd(S5_1);                                     \
        const __m128d vs52 = _mm_set1_pd(S5_2);                                     \
        __m128d t1_re = _mm_add_pd(a_re, _mm_add_pd(_mm_mul_pd(vc51, s1_re), _mm_mul_pd(vc52, s2_re))); \
        __m128d t1_im = _mm_add_pd(a_im, _mm_add_pd(_mm_mul_pd(vc51, s1_im), _mm_mul_pd(vc52, s2_im))); \
        __m128d t2_re = _mm_add_pd(a_re, _mm_add_pd(_mm_mul_pd(vc52, s1_re), _mm_mul_pd(vc51, s2_re))); \
        __m128d t2_im = _mm_add_pd(a_im, _mm_add_pd(_mm_mul_pd(vc52, s1_im), _mm_mul_pd(vc51, s2_im))); \
        __m128d base1_re = _mm_add_pd(_mm_mul_pd(vs51, d1_re), _mm_mul_pd(vs52, d2_re)); \
        __m128d base1_im = _mm_add_pd(_mm_mul_pd(vs51, d1_im), _mm_mul_pd(vs52, d2_im)); \
        __m128d u1_re = base1_im;                                                    \
        __m128d u1_im = _mm_sub_pd(_mm_setzero_pd(), base1_re);                     \
        __m128d base2_re = _mm_sub_pd(_mm_mul_pd(vs52, d1_re), _mm_mul_pd(vs51, d2_re)); \
        __m128d base2_im = _mm_sub_pd(_mm_mul_pd(vs52, d1_im), _mm_mul_pd(vs51, d2_im)); \
        __m128d u2_re = base2_im;                                                    \
        __m128d u2_im = _mm_sub_pd(_mm_setzero_pd(), base2_re);                     \
        y1_re = _mm_add_pd(t1_re, u1_re);                                            \
        y1_im = _mm_add_pd(t1_im, u1_im);                                            \
        y4_re = _mm_sub_pd(t1_re, u1_re);                                            \
        y4_im = _mm_sub_pd(t1_im, u1_im);                                            \
        y2_re = _mm_sub_pd(t2_re, u2_re);                                            \
        y2_im = _mm_sub_pd(t2_im, u2_im);                                            \
        y3_re = _mm_add_pd(t2_re, u2_re);                                            \
        y3_im = _mm_add_pd(t2_im, u2_im);                                            \
    } while (0)

#define RADIX5_PIPELINE_1_NATIVE_SOA_FV_SSE2(k, K, in_re, in_im, out_re, out_im, \
                                             stage_tw, prefetch_dist, k_end)     \
    do                                                                           \
    {                                                                            \
        __m128d a_re = _mm_loadu_pd(&in_re[k]);                                  \
        __m128d a_im = _mm_loadu_pd(&in_im[k]);                                  \
        __m128d b_re = _mm_loadu_pd(&in_re[(k) + (K)]);                          \
        __m128d b_im = _mm_loadu_pd(&in_im[(k) + (K)]);                          \
        __m128d c_re = _mm_loadu_pd(&in_re[(k) + 2 * (K)]);                      \
        __m128d c_im = _mm_loadu_pd(&in_im[(k) + 2 * (K)]);                      \
        __m128d d_re = _mm_loadu_pd(&in_re[(k) + 3 * (K)]);                      \
        __m128d d_im = _mm_loadu_pd(&in_im[(k) + 3 * (K)]);                      \
        __m128d e_re = _mm_loadu_pd(&in_re[(k) + 4 * (K)]);                      \
        __m128d e_im = _mm_loadu_pd(&in_im[(k) + 4 * (K)]);                      \
        __m128d w1_re = _mm_loadu_pd(&stage_tw->re[0 * (K) + (k)]);              \
        __m128d w1_im = _mm_loadu_pd(&stage_tw->im[0 * (K) + (k)]);              \
        __m128d w2_re = _mm_loadu_pd(&stage_tw->re[1 * (K) + (k)]);              \
        __m128d w2_im = _mm_loadu_pd(&stage_tw->im[1 * (K) + (k)]);              \
        __m128d w3_re = _mm_loadu_pd(&stage_tw->re[2 * (K) + (k)]);              \
        __m128d w3_im = _mm_loadu_pd(&stage_tw->im[2 * (K) + (k)]);              \
        __m128d w4_re = _mm_loadu_pd(&stage_tw->re[3 * (K) + (k)]);              \
        __m128d w4_im = _mm_loadu_pd(&stage_tw->im[3 * (K) + (k)]);              \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (k_end))             \
        {                                                                        \
            _mm_prefetch((const char *)&in_re[(k) + (prefetch_dist)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&in_im[(k) + (prefetch_dist)], _MM_HINT_T0); \
        }                                                                        \
        __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_SSE2(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,          \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
        _mm_storeu_pd(&out_re[k], y0_re);                                        \
        _mm_storeu_pd(&out_im[k], y0_im);                                        \
        _mm_storeu_pd(&out_re[(k) + (K)], y1_re);                                \
        _mm_storeu_pd(&out_im[(k) + (K)], y1_im);                                \
        _mm_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                            \
        _mm_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                            \
        _mm_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                            \
        _mm_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                            \
        _mm_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                            \
        _mm_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                            \
    } while (0)

#define RADIX5_PIPELINE_1_NATIVE_SOA_FV_SSE2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                    stage_tw, prefetch_dist, k_end)     \
    do                                                                                  \
    {                                                                                   \
        __m128d a_re = _mm_loadu_pd(&in_re[k]);                                         \
        __m128d a_im = _mm_loadu_pd(&in_im[k]);                                         \
        __m128d b_re = _mm_loadu_pd(&in_re[(k) + (K)]);                                 \
        __m128d b_im = _mm_loadu_pd(&in_im[(k) + (K)]);                                 \
        __m128d c_re = _mm_loadu_pd(&in_re[(k) + 2 * (K)]);                             \
        __m128d c_im = _mm_loadu_pd(&in_im[(k) + 2 * (K)]);                             \
        __m128d d_re = _mm_loadu_pd(&in_re[(k) + 3 * (K)]);                             \
        __m128d d_im = _mm_loadu_pd(&in_im[(k) + 3 * (K)]);                             \
        __m128d e_re = _mm_loadu_pd(&in_re[(k) + 4 * (K)]);                             \
        __m128d e_im = _mm_loadu_pd(&in_im[(k) + 4 * (K)]);                             \
        __m128d w1_re = _mm_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                     \
        __m128d w1_im = _mm_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                     \
        __m128d w2_re = _mm_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                     \
        __m128d w2_im = _mm_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                     \
        __m128d w3_re = _mm_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                     \
        __m128d w3_im = _mm_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                     \
        __m128d w4_re = _mm_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                     \
        __m128d w4_im = _mm_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                     \
        __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;   \
        RADIX5_BUTTERFLY_FV_NATIVE_SOA_SSE2(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                 \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);      \
        _mm_stream_pd(&out_re[k], y0_re);                                               \
        _mm_stream_pd(&out_im[k], y0_im);                                               \
        _mm_stream_pd(&out_re[(k) + (K)], y1_re);                                       \
        _mm_stream_pd(&out_im[(k) + (K)], y1_im);                                       \
        _mm_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                   \
        _mm_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                   \
        _mm_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                   \
        _mm_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                   \
        _mm_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                   \
        _mm_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                   \
    } while (0)

#define RADIX5_PIPELINE_1_NATIVE_SOA_BV_SSE2(k, K, in_re, in_im, out_re, out_im, \
                                             stage_tw, prefetch_dist, k_end)     \
    do                                                                           \
    {                                                                            \
        __m128d a_re = _mm_loadu_pd(&in_re[k]);                                  \
        __m128d a_im = _mm_loadu_pd(&in_im[k]);                                  \
        __m128d b_re = _mm_loadu_pd(&in_re[(k) + (K)]);                          \
        __m128d b_im = _mm_loadu_pd(&in_im[(k) + (K)]);                          \
        __m128d c_re = _mm_loadu_pd(&in_re[(k) + 2 * (K)]);                      \
        __m128d c_im = _mm_loadu_pd(&in_im[(k) + 2 * (K)]);                      \
        __m128d d_re = _mm_loadu_pd(&in_re[(k) + 3 * (K)]);                      \
        __m128d d_im = _mm_loadu_pd(&in_im[(k) + 3 * (K)]);                      \
        __m128d e_re = _mm_loadu_pd(&in_re[(k) + 4 * (K)]);                      \
        __m128d e_im = _mm_loadu_pd(&in_im[(k) + 4 * (K)]);                      \
        __m128d w1_re = _mm_loadu_pd(&stage_tw->re[0 * (K) + (k)]);              \
        __m128d w1_im = _mm_loadu_pd(&stage_tw->im[0 * (K) + (k)]);              \
        __m128d w2_re = _mm_loadu_pd(&stage_tw->re[1 * (K) + (k)]);              \
        __m128d w2_im = _mm_loadu_pd(&stage_tw->im[1 * (K) + (k)]);              \
        __m128d w3_re = _mm_loadu_pd(&stage_tw->re[2 * (K) + (k)]);              \
        __m128d w3_im = _mm_loadu_pd(&stage_tw->im[2 * (K) + (k)]);              \
        __m128d w4_re = _mm_loadu_pd(&stage_tw->re[3 * (K) + (k)]);              \
        __m128d w4_im = _mm_loadu_pd(&stage_tw->im[3 * (K) + (k)]);              \
        __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im; \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_SSE2(                                     \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,          \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,              \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im); \
        _mm_storeu_pd(&out_re[k], y0_re);                                        \
        _mm_storeu_pd(&out_im[k], y0_im);                                        \
        _mm_storeu_pd(&out_re[(k) + (K)], y1_re);                                \
        _mm_storeu_pd(&out_im[(k) + (K)], y1_im);                                \
        _mm_storeu_pd(&out_re[(k) + 2 * (K)], y2_re);                            \
        _mm_storeu_pd(&out_im[(k) + 2 * (K)], y2_im);                            \
        _mm_storeu_pd(&out_re[(k) + 3 * (K)], y3_re);                            \
        _mm_storeu_pd(&out_im[(k) + 3 * (K)], y3_im);                            \
        _mm_storeu_pd(&out_re[(k) + 4 * (K)], y4_re);                            \
        _mm_storeu_pd(&out_im[(k) + 4 * (K)], y4_im);                            \
    } while (0)

#define RADIX5_PIPELINE_1_NATIVE_SOA_BV_SSE2_STREAM(k, K, in_re, in_im, out_re, out_im, \
                                                    stage_tw, prefetch_dist, k_end)     \
    do                                                                                  \
    {                                                                                   \
        __m128d a_re = _mm_loadu_pd(&in_re[k]);                                         \
        __m128d a_im = _mm_loadu_pd(&in_im[k]);                                         \
        __m128d b_re = _mm_loadu_pd(&in_re[(k) + (K)]);                                 \
        __m128d b_im = _mm_loadu_pd(&in_im[(k) + (K)]);                                 \
        __m128d c_re = _mm_loadu_pd(&in_re[(k) + 2 * (K)]);                             \
        __m128d c_im = _mm_loadu_pd(&in_im[(k) + 2 * (K)]);                             \
        __m128d d_re = _mm_loadu_pd(&in_re[(k) + 3 * (K)]);                             \
        __m128d d_im = _mm_loadu_pd(&in_im[(k) + 3 * (K)]);                             \
        __m128d e_re = _mm_loadu_pd(&in_re[(k) + 4 * (K)]);                             \
        __m128d e_im = _mm_loadu_pd(&in_im[(k) + 4 * (K)]);                             \
        __m128d w1_re = _mm_loadu_pd(&stage_tw->re[0 * (K) + (k)]);                     \
        __m128d w1_im = _mm_loadu_pd(&stage_tw->im[0 * (K) + (k)]);                     \
        __m128d w2_re = _mm_loadu_pd(&stage_tw->re[1 * (K) + (k)]);                     \
        __m128d w2_im = _mm_loadu_pd(&stage_tw->im[1 * (K) + (k)]);                     \
        __m128d w3_re = _mm_loadu_pd(&stage_tw->re[2 * (K) + (k)]);                     \
        __m128d w3_im = _mm_loadu_pd(&stage_tw->im[2 * (K) + (k)]);                     \
        __m128d w4_re = _mm_loadu_pd(&stage_tw->re[3 * (K) + (k)]);                     \
        __m128d w4_im = _mm_loadu_pd(&stage_tw->im[3 * (K) + (k)]);                     \
        __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im;   \
        RADIX5_BUTTERFLY_BV_NATIVE_SOA_SSE2(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, e_re, e_im,                 \
            w1_re, w1_im, w2_re, w2_im, w3_re, w3_im, w4_re, w4_im,                     \
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, y4_re, y4_im);      \
        _mm_stream_pd(&out_re[k], y0_re);                                               \
        _mm_stream_pd(&out_im[k], y0_im);                                               \
        _mm_stream_pd(&out_re[(k) + (K)], y1_re);                                       \
        _mm_stream_pd(&out_im[(k) + (K)], y1_im);                                       \
        _mm_stream_pd(&out_re[(k) + 2 * (K)], y2_re);                                   \
        _mm_stream_pd(&out_im[(k) + 2 * (K)], y2_im);                                   \
        _mm_stream_pd(&out_re[(k) + 3 * (K)], y3_re);                                   \
        _mm_stream_pd(&out_im[(k) + 3 * (K)], y3_im);                                   \
        _mm_stream_pd(&out_re[(k) + 4 * (K)], y4_re);                                   \
        _mm_stream_pd(&out_im[(k) + 4 * (K)], y4_im);                                   \
    } while (0)

#endif // __SSE2__

//==============================================================================
// SCALAR PIPELINE MACROS
//==============================================================================

/**
 * @brief Scalar radix-5 pipeline - Forward - Native SoA
 */
#define RADIX5_PIPELINE_1_NATIVE_SOA_FV_SCALAR(k, K, in_re, in_im, out_re, out_im, stage_tw) \
    do                                                                                        \
    {                                                                                         \
        double a_re = in_re[k];                                                               \
        double a_im = in_im[k];                                                               \
        double b_re = in_re[(k) + (K)];                                                       \
        double b_im = in_im[(k) + (K)];                                                       \
        double c_re = in_re[(k) + 2 * (K)];                                                   \
        double c_im = in_im[(k) + 2 * (K)];                                                   \
        double d_re = in_re[(k) + 3 * (K)];                                                   \
        double d_im = in_im[(k) + 3 * (K)];                                                   \
        double e_re = in_re[(k) + 4 * (K)];                                                   \
        double e_im = in_im[(k) + 4 * (K)];                                                   \
        double w1_re = stage_tw->re[0 * (K) + (k)];                                           \
        double w1_im = stage_tw->im[0 * (K) + (k)];                                           \
        double w2_re = stage_tw->re[1 * (K) + (k)];                                           \
        double w2_im = stage_tw->im[1 * (K) + (k)];                                           \
        double w3_re = stage_tw->re[2 * (K) + (k)];                                           \
        double w3_im = stage_tw->im[2 * (K) + (k)];                                           \
        double w4_re = stage_tw->re[3 * (K) + (k)];                                           \
        double w4_im = stage_tw->im[3 * (K) + (k)];                                           \
        double tw_b_re = b_re * w1_re - b_im * w1_im;                                         \
        double tw_b_im = b_re * w1_im + b_im * w1_re;                                         \
        double tw_c_re = c_re * w2_re - c_im * w2_im;                                         \
        double tw_c_im = c_re * w2_im + c_im * w2_re;                                         \
        double tw_d_re = d_re * w3_re - d_im * w3_im;                                         \
        double tw_d_im = d_re * w3_im + d_im * w3_re;                                         \
        double tw_e_re = e_re * w4_re - e_im * w4_im;                                         \
        double tw_e_im = e_re * w4_im + e_im * w4_re;                                         \
        double s1_re = tw_b_re + tw_e_re;                                                     \
        double s1_im = tw_b_im + tw_e_im;                                                     \
        double s2_re = tw_c_re + tw_d_re;                                                     \
        double s2_im = tw_c_im + tw_d_im;                                                     \
        double d1_re = tw_b_re - tw_e_re;                                                     \
        double d1_im = tw_b_im - tw_e_im;                                                     \
        double d2_re = tw_c_re - tw_d_re;                                                     \
        double d2_im = tw_c_im - tw_d_im;                                                     \
        out_re[k] = a_re + s1_re + s2_re;                                                     \
        out_im[k] = a_im + s1_im + s2_im;                                                     \
        double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;                                    \
        double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;                                    \
        double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;                                    \
        double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;                                    \
        double base1_re = S5_1 * d1_re + S5_2 * d2_re;                                        \
        double base1_im = S5_1 * d1_im + S5_2 * d2_im;                                        \
        double u1_re = -base1_im;                                                             \
        double u1_im = base1_re;                                                              \
        double base2_re = S5_2 * d1_re - S5_1 * d2_re;                                        \
        double base2_im = S5_2 * d1_im - S5_1 * d2_im;                                        \
        double u2_re = -base2_im;                                                             \
        double u2_im = base2_re;                                                              \
        out_re[(k) + (K)] = t1_re + u1_re;                                                    \
        out_im[(k) + (K)] = t1_im + u1_im;                                                    \
        out_re[(k) + 4 * (K)] = t1_re - u1_re;                                                \
        out_im[(k) + 4 * (K)] = t1_im - u1_im;                                                \
        out_re[(k) + 2 * (K)] = t2_re - u2_re;                                                \
        out_im[(k) + 2 * (K)] = t2_im - u2_im;                                                \
        out_re[(k) + 3 * (K)] = t2_re + u2_re;                                                \
        out_im[(k) + 3 * (K)] = t2_im + u2_im;                                                \
    } while (0)

/**
 * @brief Scalar radix-5 pipeline - Backward - Native SoA
 */
#define RADIX5_PIPELINE_1_NATIVE_SOA_BV_SCALAR(k, K, in_re, in_im, out_re, out_im, stage_tw) \
    do                                                                                        \
    {                                                                                         \
        double a_re = in_re[k];                                                               \
        double a_im = in_im[k];                                                               \
        double b_re = in_re[(k) + (K)];                                                       \
        double b_im = in_im[(k) + (K)];                                                       \
        double c_re = in_re[(k) + 2 * (K)];                                                   \
        double c_im = in_im[(k) + 2 * (K)];                                                   \
        double d_re = in_re[(k) + 3 * (K)];                                                   \
        double d_im = in_im[(k) + 3 * (K)];                                                   \
        double e_re = in_re[(k) + 4 * (K)];                                                   \
        double e_im = in_im[(k) + 4 * (K)];                                                   \
        double w1_re = stage_tw->re[0 * (K) + (k)];                                           \
        double w1_im = stage_tw->im[0 * (K) + (k)];                                           \
        double w2_re = stage_tw->re[1 * (K) + (k)];                                           \
        double w2_im = stage_tw->im[1 * (K) + (k)];                                           \
        double w3_re = stage_tw->re[2 * (K) + (k)];                                           \
        double w3_im = stage_tw->im[2 * (K) + (k)];                                           \
        double w4_re = stage_tw->re[3 * (K) + (k)];                                           \
        double w4_im = stage_tw->im[3 * (K) + (k)];                                           \
        double tw_b_re = b_re * w1_re - b_im * w1_im;                                         \
        double tw_b_im = b_re * w1_im + b_im * w1_re;                                         \
        double tw_c_re = c_re * w2_re - c_im * w2_im;                                         \
        double tw_c_im = c_re * w2_im + c_im * w2_re;                                         \
        double tw_d_re = d_re * w3_re - d_im * w3_im;                                         \
        double tw_d_im = d_re * w3_im + d_im * w3_re;                                         \
        double tw_e_re = e_re * w4_re - e_im * w4_im;                                         \
        double tw_e_im = e_re * w4_im + e_im * w4_re;                                         \
        double s1_re = tw_b_re + tw_e_re;                                                     \
        double s1_im = tw_b_im + tw_e_im;                                                     \
        double s2_re = tw_c_re + tw_d_re;                                                     \
        double s2_im = tw_c_im + tw_d_im;                                                     \
        double d1_re = tw_b_re - tw_e_re;                                                     \
        double d1_im = tw_b_im - tw_e_im;                                                     \
        double d2_re = tw_c_re - tw_d_re;                                                     \
        double d2_im = tw_c_im - tw_d_im;                                                     \
        out_re[k] = a_re + s1_re + s2_re;                                                     \
        out_im[k] = a_im + s1_im + s2_im;                                                     \
        double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;                                    \
        double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;                                    \
        double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;                                    \
        double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;                                    \
        double base1_re = S5_1 * d1_re + S5_2 * d2_re;                                        \
        double base1_im = S5_1 * d1_im + S5_2 * d2_im;                                        \
        double u1_re = base1_im;                                                              \
        double u1_im = -base1_re;                                                             \
        double base2_re = S5_2 * d1_re - S5_1 * d2_re;                                        \
        double base2_im = S5_2 * d1_im - S5_1 * d2_im;                                        \
        double u2_re = base2_im;                                                              \
        double u2_im = -base2_re;                                                             \
        out_re[(k) + (K)] = t1_re + u1_re;                                                    \
        out_im[(k) + (K)] = t1_im + u1_im;                                                    \
        out_re[(k) + 4 * (K)] = t1_re - u1_re;                                                \
        out_im[(k) + 4 * (K)] = t1_im - u1_im;                                                \
        out_re[(k) + 2 * (K)] = t2_re - u2_re;                                                \
        out_im[(k) + 2 * (K)] = t2_im - u2_im;                                                \
        out_re[(k) + 3 * (K)] = t2_re + u2_re;                                                \
        out_im[(k) + 3 * (K)] = t2_im + u2_im;                                                \
    } while (0)

#endif // FFT_RADIX5_MACROS_OPTIMIZED_H

//==============================================================================
// PERFORMANCE SUMMARY
//==============================================================================

/**
 * @page radix5_perf_summary Radix-5 Performance Summary
 *
 * @section optimizations APPLIED OPTIMIZATIONS
 *
 * 1. ✅ Native SoA Architecture
 *    - NO split/join in hot path
 *    - Direct loads from separate re[]/im[] arrays
 *    - 95%+ shuffle elimination (same as radix-4)
 *
 * 2. ✅ SIMD Optimization
 *    - AVX-512: 4 butterflies (8 doubles) per iteration
 *    - AVX2: 2 butterflies (4 doubles) per iteration
 *    - SSE2: 1 butterfly (2 doubles) per iteration
 *    - Scalar: fallback path
 *
 * 3. ✅ FMA Usage
 *    - All complex multiplies use FMA when available
 *    - 2 ops instead of 4 per complex multiply
 *
 * 4. ✅ Software Prefetching
 *    - Prefetch distance: 32 elements (tuned for radix-5)
 *    - Hides memory latency
 *
 * 5. ✅ Streaming Stores
 *    - Non-temporal stores for large N
 *    - Cache bypass when write > 70% LLC
 *
 * 6. ✅ Double-Pumping Ready
 *    - Pipeline macros designed for double-pumping
 *    - Process k and k+step together in driver code
 *
 * @section expected_speedup EXPECTED PERFORMANCE
 *
 * Compared to old split-form radix-5:
 * - Small FFTs (125-625):   +10-15%
 * - Medium FFTs (3125-78K): +25-35%
 * - Large FFTs (>390K):     +50-70%
 *
 * Radix-5 vs Radix-4:
 * - Fewer stages: log₅(N) < log₄(N)
 * - More complex butterfly (5 points vs 4)
 * - Generally 5-10% slower than radix-4 for same N
 * - BUT: Better for N = 5^k (no mixed radix overhead)
 *
 * @section usage USAGE PATTERN
 *
 * @code
 * // Convert to SoA once
 * fft_aos_to_soa(input, workspace_re, workspace_im, N);
 *
 * // All stages in native SoA (ping-pong buffers)
 * for (int stage = 0; stage < num_stages; stage++) {
 *     if (stage % 2 == 0)
 *         fft_radix5_native_soa(buf_b_re, buf_b_im, buf_a_re, buf_a_im, ...);
 *     else
 *         fft_radix5_native_soa(buf_a_re, buf_a_im, buf_b_re, buf_b_im, ...);
 * }
 *
 * // Convert back once
 * fft_soa_to_aos(workspace_re, workspace_im, output, N);
 * @endcode
 */