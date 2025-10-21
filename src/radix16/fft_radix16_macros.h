//==============================================================================
// fft_radix16_macros.h - Optimized Shared Macros for Radix-16 Butterflies
//==============================================================================
//
// ALGORITHM: 2-stage radix-4 decomposition with optimizations
//   1. Apply input twiddles W_N^(j*k) for j=1..15
//   2. First radix-4 stage (4 groups of 4)
//   3. Apply W_4 intermediate twiddles (optimized: ±i and -1 as XOR)
//   4. Second radix-4 stage (in-place to reduce register pressure)
//
// OPTIMIZATIONS:
//   - Hoisted W_4 constants outside loops
//   - W_4^1 = ±i as permute + XOR (not full multiply)
//   - W_4^2 = -1 as XOR only (not full multiply)
//   - In-place 2nd stage (reduces register pressure)
//   - Single-level prefetching
//   - Alignment hints support
//   - Hoisted sign masks for W_4 intermediate twiddles
//   - Fully unrolled twiddle application (15 iterations)
//   - SoA twiddle layout for zero-shuffle hot path
//   - Fixed AoS↔SoA conversion (movedup + permute)
//   - Reduced register pressure (≤24 zmm live)
//   - Software pipelined twiddle application
//

#ifndef FFT_RADIX16_MACROS_H
#define FFT_RADIX16_MACROS_H

#include "simd_math.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

#define STREAM_THRESHOLD_R16 1024

//==============================================================================
// W_4 CONSTANTS
//==============================================================================

#define W4_FV_0_RE 1.0
#define W4_FV_0_IM 0.0
#define W4_FV_1_RE 0.0
#define W4_FV_1_IM (-1.0)
#define W4_FV_2_RE (-1.0)
#define W4_FV_2_IM 0.0
#define W4_FV_3_RE 0.0
#define W4_FV_3_IM 1.0

#define W4_BV_0_RE 1.0
#define W4_BV_0_IM 0.0
#define W4_BV_1_RE 0.0
#define W4_BV_1_IM 1.0
#define W4_BV_2_RE (-1.0)
#define W4_BV_2_IM 0.0
#define W4_BV_3_RE 0.0
#define W4_BV_3_IM (-1.0)

//==============================================================================
// AVX-512 SUPPORT - FULLY OPTIMIZED
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// CONFIGURATION & TUNING PARAMETERS
//==============================================================================

#define K_TILE_R16_AVX512 1024
#define PREFETCH_DISTANCE_AVX512 16

//==============================================================================
// AoS ↔ SoA CONVERSION (FIXED)
//==============================================================================

// CORRECT: movedup for real, permute for imaginary
#define AOS_TO_SOA_AVX512(aos, re, im)       \
    do                                       \
    {                                        \
        (re) = _mm512_movedup_pd(aos);       \
        (im) = _mm512_permute_pd(aos, 0xFF); \
    } while (0)

#define SOA_TO_AOS_AVX512(re, im, aos) \
    (aos) = _mm512_unpacklo_pd(re, im)

//==============================================================================
// SOA COMPLEX ARITHMETIC
//==============================================================================

#if defined(__FMA__)
#define CMUL_FMA_SOA_AVX512(out_re, out_im, a_re, a_im, w_re, w_im)        \
    do                                                                     \
    {                                                                      \
        (out_re) = _mm512_fmsub_pd(a_re, w_re, _mm512_mul_pd(a_im, w_im)); \
        (out_im) = _mm512_fmadd_pd(a_re, w_im, _mm512_mul_pd(a_im, w_re)); \
    } while (0)
#else
#define CMUL_FMA_SOA_AVX512(out_re, out_im, a_re, a_im, w_re, w_im)                     \
    do                                                                                  \
    {                                                                                   \
        (out_re) = _mm512_sub_pd(_mm512_mul_pd(a_re, w_re), _mm512_mul_pd(a_im, w_im)); \
        (out_im) = _mm512_add_pd(_mm512_mul_pd(a_re, w_im), _mm512_mul_pd(a_im, w_re)); \
    } while (0)
#endif

//==============================================================================
// LEGACY AOS COMPLEX MULTIPLY (for backward compatibility, FIXED)
//==============================================================================

#if defined(__FMA__)
#define CMUL_FMA_AOS_AVX512(out, a, w)                               \
    do                                                               \
    {                                                                \
        __m512d ar = _mm512_movedup_pd(a);                           \
        __m512d ai = _mm512_permute_pd(a, 0xFF);                     \
        __m512d wr = _mm512_movedup_pd(w);                           \
        __m512d wi = _mm512_permute_pd(w, 0xFF);                     \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
        (out) = _mm512_unpacklo_pd(re, im);                          \
    } while (0)
#else
#define CMUL_FMA_AOS_AVX512(out, a, w)                                            \
    do                                                                            \
    {                                                                             \
        __m512d ar = _mm512_movedup_pd(a);                                        \
        __m512d ai = _mm512_permute_pd(a, 0xFF);                                  \
        __m512d wr = _mm512_movedup_pd(w);                                        \
        __m512d wi = _mm512_permute_pd(w, 0xFF);                                  \
        __m512d re = _mm512_sub_pd(_mm512_mul_pd(ar, wr), _mm512_mul_pd(ai, wi)); \
        __m512d im = _mm512_add_pd(_mm512_mul_pd(ar, wi), _mm512_mul_pd(ai, wr)); \
        (out) = _mm512_unpacklo_pd(re, im);                                       \
    } while (0)
#endif

//==============================================================================
// SOA RADIX-4 BUTTERFLY
//==============================================================================

#define RADIX4_BUTTERFLY_SOA_AVX512(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                        \
                                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, rot_sign_mask) \
    do                                                                                                     \
    {                                                                                                      \
        __m512d sumBD_re = _mm512_add_pd(b_re, d_re);                                                      \
        __m512d sumBD_im = _mm512_add_pd(b_im, d_im);                                                      \
        __m512d difBD_re = _mm512_sub_pd(b_re, d_re);                                                      \
        __m512d difBD_im = _mm512_sub_pd(b_im, d_im);                                                      \
        __m512d sumAC_re = _mm512_add_pd(a_re, c_re);                                                      \
        __m512d sumAC_im = _mm512_add_pd(a_im, c_im);                                                      \
        __m512d difAC_re = _mm512_sub_pd(a_re, c_re);                                                      \
        __m512d difAC_im = _mm512_sub_pd(a_im, c_im);                                                      \
        y0_re = _mm512_add_pd(sumAC_re, sumBD_re);                                                         \
        y0_im = _mm512_add_pd(sumAC_im, sumBD_im);                                                         \
        y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);                                                         \
        y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);                                                         \
        __m512d rot_re = _mm512_xor_pd(difBD_im, rot_sign_mask);                                           \
        __m512d rot_im = _mm512_xor_pd(_mm512_sub_pd(_mm512_setzero_pd(), difBD_re), rot_sign_mask);       \
        y1_re = _mm512_sub_pd(difAC_re, rot_re);                                                           \
        y1_im = _mm512_sub_pd(difAC_im, rot_im);                                                           \
        y3_re = _mm512_add_pd(difAC_re, rot_re);                                                           \
        y3_im = _mm512_add_pd(difAC_im, rot_im);                                                           \
    } while (0)

//==============================================================================
// LEGACY AOS RADIX-4 BUTTERFLY (for backward compatibility)
//==============================================================================

#define RADIX4_BUTTERFLY_AVX512(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do                                                                \
    {                                                                 \
        __m512d sumBD = _mm512_add_pd(b, d);                          \
        __m512d difBD = _mm512_sub_pd(b, d);                          \
        __m512d sumAC = _mm512_add_pd(a, c);                          \
        __m512d difAC = _mm512_sub_pd(a, c);                          \
        y0 = _mm512_add_pd(sumAC, sumBD);                             \
        y2 = _mm512_sub_pd(sumAC, sumBD);                             \
        __m512d dif_bd_swp = _mm512_permute_pd(difBD, 0b01010101);    \
        __m512d dif_bd_rot = _mm512_xor_pd(dif_bd_swp, rot_mask);     \
        y1 = _mm512_sub_pd(difAC, dif_bd_rot);                        \
        y3 = _mm512_add_pd(difAC, dif_bd_rot);                        \
    } while (0)

//==============================================================================
// SOA W_4 INTERMEDIATE TWIDDLES
//==============================================================================

#define APPLY_W4_INTERMEDIATE_FV_SOA_AVX512(y_re, y_im, neg_mask) \
    do                                                            \
    {                                                             \
        /* m=1: W_4^{j} for j=1,2,3 = {-i, -1, +i} */             \
        {                                                         \
            __m512d tmp_re = y_re[5];                             \
            y_re[5] = y_im[5];                                    \
            y_im[5] = _mm512_xor_pd(tmp_re, neg_mask);            \
            y_re[6] = _mm512_xor_pd(y_re[6], neg_mask);           \
            y_im[6] = _mm512_xor_pd(y_im[6], neg_mask);           \
            tmp_re = y_re[7];                                     \
            y_re[7] = _mm512_xor_pd(y_im[7], neg_mask);           \
            y_im[7] = tmp_re;                                     \
        }                                                         \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, +1, -1} */            \
        {                                                         \
            y_re[9] = _mm512_xor_pd(y_re[9], neg_mask);           \
            y_im[9] = _mm512_xor_pd(y_im[9], neg_mask);           \
            y_re[11] = _mm512_xor_pd(y_re[11], neg_mask);         \
            y_im[11] = _mm512_xor_pd(y_im[11], neg_mask);         \
        }                                                         \
        /* m=3: W_4^{3j} for j=1,2,3 = {+i, -1, -i} */            \
        {                                                         \
            __m512d tmp_re = y_re[13];                            \
            y_re[13] = _mm512_xor_pd(y_im[13], neg_mask);         \
            y_im[13] = tmp_re;                                    \
            y_re[14] = _mm512_xor_pd(y_re[14], neg_mask);         \
            y_im[14] = _mm512_xor_pd(y_im[14], neg_mask);         \
            tmp_re = y_re[15];                                    \
            y_re[15] = y_im[15];                                  \
            y_im[15] = _mm512_xor_pd(tmp_re, neg_mask);           \
        }                                                         \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_SOA_AVX512(y_re, y_im, neg_mask) \
    do                                                            \
    {                                                             \
        /* m=1: W_4^{j} for j=1,2,3 = {+i, -1, -i} */             \
        {                                                         \
            __m512d tmp_re = y_re[5];                             \
            y_re[5] = _mm512_xor_pd(y_im[5], neg_mask);           \
            y_im[5] = tmp_re;                                     \
            y_re[6] = _mm512_xor_pd(y_re[6], neg_mask);           \
            y_im[6] = _mm512_xor_pd(y_im[6], neg_mask);           \
            tmp_re = y_re[7];                                     \
            y_re[7] = y_im[7];                                    \
            y_im[7] = _mm512_xor_pd(tmp_re, neg_mask);            \
        }                                                         \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, +1, -1} */            \
        {                                                         \
            y_re[9] = _mm512_xor_pd(y_re[9], neg_mask);           \
            y_im[9] = _mm512_xor_pd(y_im[9], neg_mask);           \
            y_re[11] = _mm512_xor_pd(y_re[11], neg_mask);         \
            y_im[11] = _mm512_xor_pd(y_im[11], neg_mask);         \
        }                                                         \
        /* m=3: W_4^{3j} for j=1,2,3 = {-i, -1, +i} */            \
        {                                                         \
            __m512d tmp_re = y_re[13];                            \
            y_re[13] = y_im[13];                                  \
            y_im[13] = _mm512_xor_pd(tmp_re, neg_mask);           \
            y_re[14] = _mm512_xor_pd(y_re[14], neg_mask);         \
            y_im[14] = _mm512_xor_pd(y_im[14], neg_mask);         \
            tmp_re = y_re[15];                                    \
            y_re[15] = _mm512_xor_pd(y_im[15], neg_mask);         \
            y_im[15] = tmp_re;                                    \
        }                                                         \
    } while (0)

//==============================================================================
// LEGACY AOS W_4 INTERMEDIATE TWIDDLES (for backward compatibility)
//==============================================================================

#define APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask)    \
    do                                                          \
    {                                                           \
        {                                                       \
            __m512d y5_swp = _mm512_permute_pd(y[5], 0x55);     \
            y[5] = _mm512_xor_pd(y5_swp, W4_SIGN_MASK_FV_5);    \
            y[6] = _mm512_xor_pd(y[6], neg_mask);               \
            __m512d y7_swp = _mm512_permute_pd(y[7], 0x55);     \
            y[7] = _mm512_xor_pd(y7_swp, W4_SIGN_MASK_FV_7);    \
        }                                                       \
        {                                                       \
            y[9] = _mm512_xor_pd(y[9], neg_mask);               \
            y[11] = _mm512_xor_pd(y[11], neg_mask);             \
        }                                                       \
        {                                                       \
            __m512d y13_swp = _mm512_permute_pd(y[13], 0x55);   \
            y[13] = _mm512_xor_pd(y13_swp, W4_SIGN_MASK_FV_13); \
            y[14] = _mm512_xor_pd(y[14], neg_mask);             \
            __m512d y15_swp = _mm512_permute_pd(y[15], 0x55);   \
            y[15] = _mm512_xor_pd(y15_swp, W4_SIGN_MASK_FV_15); \
        }                                                       \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask)    \
    do                                                          \
    {                                                           \
        {                                                       \
            __m512d y5_swp = _mm512_permute_pd(y[5], 0x55);     \
            y[5] = _mm512_xor_pd(y5_swp, W4_SIGN_MASK_BV_5);    \
            y[6] = _mm512_xor_pd(y[6], neg_mask);               \
            __m512d y7_swp = _mm512_permute_pd(y[7], 0x55);     \
            y[7] = _mm512_xor_pd(y7_swp, W4_SIGN_MASK_BV_7);    \
        }                                                       \
        {                                                       \
            y[9] = _mm512_xor_pd(y[9], neg_mask);               \
            y[11] = _mm512_xor_pd(y[11], neg_mask);             \
        }                                                       \
        {                                                       \
            __m512d y13_swp = _mm512_permute_pd(y[13], 0x55);   \
            y[13] = _mm512_xor_pd(y13_swp, W4_SIGN_MASK_BV_13); \
            y[14] = _mm512_xor_pd(y[14], neg_mask);             \
            __m512d y15_swp = _mm512_permute_pd(y[15], 0x55);   \
            y[15] = _mm512_xor_pd(y15_swp, W4_SIGN_MASK_BV_15); \
        }                                                       \
    } while (0)

//==============================================================================
// LOAD & STORE HELPERS
//==============================================================================

#define LOAD_16_LANES_SOA_AVX512(kk, K, sub_outputs, x_re, x_im)        \
    do                                                                  \
    {                                                                   \
        for (int lane = 0; lane < 16; lane++)                           \
        {                                                               \
            __m512d aos = load4_aos(&sub_outputs[(kk) + lane * K],      \
                                    &sub_outputs[(kk) + 1 + lane * K],  \
                                    &sub_outputs[(kk) + 2 + lane * K],  \
                                    &sub_outputs[(kk) + 3 + lane * K]); \
            AOS_TO_SOA_AVX512(aos, x_re[lane], x_im[lane]);             \
        }                                                               \
    } while (0)

#define STORE_16_LANES_SOA_AVX512(kk, K, output_buffer, y_re, y_im)      \
    do                                                                   \
    {                                                                    \
        for (int m = 0; m < 4; m++)                                      \
        {                                                                \
            __m512d aos0, aos4, aos8, aos12;                             \
            SOA_TO_AOS_AVX512(y_re[m], y_im[m], aos0);                   \
            SOA_TO_AOS_AVX512(y_re[m + 4], y_im[m + 4], aos4);           \
            SOA_TO_AOS_AVX512(y_re[m + 8], y_im[m + 8], aos8);           \
            SOA_TO_AOS_AVX512(y_re[m + 12], y_im[m + 12], aos12);        \
            STOREU_PD512(&output_buffer[(kk) + m * K].re, aos0);         \
            STOREU_PD512(&output_buffer[(kk) + (m + 4) * K].re, aos4);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 8) * K].re, aos8);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 12) * K].re, aos12); \
        }                                                                \
    } while (0)

#define STORE_16_LANES_SOA_AVX512_STREAM(kk, K, output_buffer, y_re, y_im)   \
    do                                                                       \
    {                                                                        \
        for (int m = 0; m < 4; m++)                                          \
        {                                                                    \
            __m512d aos0, aos4, aos8, aos12;                                 \
            SOA_TO_AOS_AVX512(y_re[m], y_im[m], aos0);                       \
            SOA_TO_AOS_AVX512(y_re[m + 4], y_im[m + 4], aos4);               \
            SOA_TO_AOS_AVX512(y_re[m + 8], y_im[m + 8], aos8);               \
            SOA_TO_AOS_AVX512(y_re[m + 12], y_im[m + 12], aos12);            \
            _mm512_stream_pd(&output_buffer[(kk) + m * K].re, aos0);         \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, aos4);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, aos8);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, aos12); \
        }                                                                    \
    } while (0)

// Legacy AoS versions (for backward compatibility)
#define LOAD_16_LANES_AVX512(kk, K, sub_outputs, x)                 \
    do                                                              \
    {                                                               \
        for (int lane = 0; lane < 16; lane++)                       \
        {                                                           \
            x[lane] = load4_aos(&sub_outputs[(kk) + lane * K],      \
                                &sub_outputs[(kk) + 1 + lane * K],  \
                                &sub_outputs[(kk) + 2 + lane * K],  \
                                &sub_outputs[(kk) + 3 + lane * K]); \
        }                                                           \
    } while (0)

#define STORE_16_LANES_AVX512(kk, K, output_buffer, y)                       \
    do                                                                       \
    {                                                                        \
        for (int m = 0; m < 4; m++)                                          \
        {                                                                    \
            STOREU_PD512(&output_buffer[(kk) + m * K].re, y[m]);             \
            STOREU_PD512(&output_buffer[(kk) + (m + 4) * K].re, y[m + 4]);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 8) * K].re, y[m + 8]);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 12) * K].re, y[m + 12]); \
        }                                                                    \
    } while (0)

#define STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, y)                    \
    do                                                                           \
    {                                                                            \
        for (int m = 0; m < 4; m++)                                              \
        {                                                                        \
            _mm512_stream_pd(&output_buffer[(kk) + m * K].re, y[m]);             \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, y[m + 4]);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, y[m + 8]);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, y[m + 12]); \
        }                                                                        \
    } while (0)

//==============================================================================
// PREFETCHING
//==============================================================================

#define PREFETCH_16_LANES_AVX512(k, K, distance, sub_outputs, hint)                          \
    do                                                                                       \
    {                                                                                        \
        if ((k) + (distance) < K)                                                            \
        {                                                                                    \
            for (int lane = 0; lane < 16; lane++)                                            \
            {                                                                                \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + lane * K], hint); \
            }                                                                                \
        }                                                                                    \
    } while (0)

#define PREFETCH_STAGE_TW_AVX512(kk, distance, stage_tw_re, stage_tw_im, sub_len)                 \
    do                                                                                            \
    {                                                                                             \
        _mm_prefetch((const char *)&stage_tw_re[0 * (sub_len) + (kk) + (distance)], _MM_HINT_T0); \
        _mm_prefetch((const char *)&stage_tw_im[0 * (sub_len) + (kk) + (distance)], _MM_HINT_T0); \
    } while (0)

//==============================================================================
// SOA TWIDDLE APPLICATION - OPTIMIZED (Direct SoA load, software pipelined)
//==============================================================================

// Unrolled by 3 for software pipelining to hide latency
#define APPLY_STAGE_TWIDDLES_R16_SOA_AVX512(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len) \
    do                                                                                         \
    {                                                                                          \
        /* r=1,2,3 - first triplet */                                                          \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw_re[0 * (sub_len) + (kk)]);                   \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw_im[0 * (sub_len) + (kk)]);                   \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw_re[1 * (sub_len) + (kk)]);                   \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw_im[1 * (sub_len) + (kk)]);                   \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw_re[2 * (sub_len) + (kk)]);                   \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw_im[2 * (sub_len) + (kk)]);                   \
        CMUL_FMA_SOA_AVX512(x_re[1], x_im[1], x_re[1], x_im[1], w1_re, w1_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[2], x_im[2], x_re[2], x_im[2], w2_re, w2_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[3], x_im[3], x_re[3], x_im[3], w3_re, w3_im);                 \
        /* r=4,5,6 */                                                                          \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw_re[3 * (sub_len) + (kk)]);                   \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw_im[3 * (sub_len) + (kk)]);                   \
        __m512d w5_re = _mm512_loadu_pd(&stage_tw_re[4 * (sub_len) + (kk)]);                   \
        __m512d w5_im = _mm512_loadu_pd(&stage_tw_im[4 * (sub_len) + (kk)]);                   \
        __m512d w6_re = _mm512_loadu_pd(&stage_tw_re[5 * (sub_len) + (kk)]);                   \
        __m512d w6_im = _mm512_loadu_pd(&stage_tw_im[5 * (sub_len) + (kk)]);                   \
        CMUL_FMA_SOA_AVX512(x_re[4], x_im[4], x_re[4], x_im[4], w4_re, w4_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[5], x_im[5], x_re[5], x_im[5], w5_re, w5_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[6], x_im[6], x_re[6], x_im[6], w6_re, w6_im);                 \
        /* r=7,8,9 */                                                                          \
        __m512d w7_re = _mm512_loadu_pd(&stage_tw_re[6 * (sub_len) + (kk)]);                   \
        __m512d w7_im = _mm512_loadu_pd(&stage_tw_im[6 * (sub_len) + (kk)]);                   \
        __m512d w8_re = _mm512_loadu_pd(&stage_tw_re[7 * (sub_len) + (kk)]);                   \
        __m512d w8_im = _mm512_loadu_pd(&stage_tw_im[7 * (sub_len) + (kk)]);                   \
        __m512d w9_re = _mm512_loadu_pd(&stage_tw_re[8 * (sub_len) + (kk)]);                   \
        __m512d w9_im = _mm512_loadu_pd(&stage_tw_im[8 * (sub_len) + (kk)]);                   \
        CMUL_FMA_SOA_AVX512(x_re[7], x_im[7], x_re[7], x_im[7], w7_re, w7_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[8], x_im[8], x_re[8], x_im[8], w8_re, w8_im);                 \
        CMUL_FMA_SOA_AVX512(x_re[9], x_im[9], x_re[9], x_im[9], w9_re, w9_im);                 \
        /* r=10,11,12 */                                                                       \
        __m512d w10_re = _mm512_loadu_pd(&stage_tw_re[9 * (sub_len) + (kk)]);                  \
        __m512d w10_im = _mm512_loadu_pd(&stage_tw_im[9 * (sub_len) + (kk)]);                  \
        __m512d w11_re = _mm512_loadu_pd(&stage_tw_re[10 * (sub_len) + (kk)]);                 \
        __m512d w11_im = _mm512_loadu_pd(&stage_tw_im[10 * (sub_len) + (kk)]);                 \
        __m512d w12_re = _mm512_loadu_pd(&stage_tw_re[11 * (sub_len) + (kk)]);                 \
        __m512d w12_im = _mm512_loadu_pd(&stage_tw_im[11 * (sub_len) + (kk)]);                 \
        CMUL_FMA_SOA_AVX512(x_re[10], x_im[10], x_re[10], x_im[10], w10_re, w10_im);           \
        CMUL_FMA_SOA_AVX512(x_re[11], x_im[11], x_re[11], x_im[11], w11_re, w11_im);           \
        CMUL_FMA_SOA_AVX512(x_re[12], x_im[12], x_re[12], x_im[12], w12_re, w12_im);           \
        /* r=13,14,15 */                                                                       \
        __m512d w13_re = _mm512_loadu_pd(&stage_tw_re[12 * (sub_len) + (kk)]);                 \
        __m512d w13_im = _mm512_loadu_pd(&stage_tw_im[12 * (sub_len) + (kk)]);                 \
        __m512d w14_re = _mm512_loadu_pd(&stage_tw_re[13 * (sub_len) + (kk)]);                 \
        __m512d w14_im = _mm512_loadu_pd(&stage_tw_im[13 * (sub_len) + (kk)]);                 \
        __m512d w15_re = _mm512_loadu_pd(&stage_tw_re[14 * (sub_len) + (kk)]);                 \
        __m512d w15_im = _mm512_loadu_pd(&stage_tw_im[14 * (sub_len) + (kk)]);                 \
        CMUL_FMA_SOA_AVX512(x_re[13], x_im[13], x_re[13], x_im[13], w13_re, w13_im);           \
        CMUL_FMA_SOA_AVX512(x_re[14], x_im[14], x_re[14], x_im[14], w14_re, w14_im);           \
        CMUL_FMA_SOA_AVX512(x_re[15], x_im[15], x_re[15], x_im[15], w15_re, w15_im);           \
    } while (0)

//==============================================================================
// LEGACY AOS TWIDDLE APPLICATION (for backward compatibility)
//==============================================================================

#define APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw, sub_len)           \
    do                                                                      \
    {                                                                       \
        __m512d w1 = _mm512_loadu_pd(&stage_tw[0 * (sub_len) + (kk)].re);   \
        __m512d w2 = _mm512_loadu_pd(&stage_tw[1 * (sub_len) + (kk)].re);   \
        __m512d w3 = _mm512_loadu_pd(&stage_tw[2 * (sub_len) + (kk)].re);   \
        __m512d w4 = _mm512_loadu_pd(&stage_tw[3 * (sub_len) + (kk)].re);   \
        __m512d w5 = _mm512_loadu_pd(&stage_tw[4 * (sub_len) + (kk)].re);   \
        __m512d w6 = _mm512_loadu_pd(&stage_tw[5 * (sub_len) + (kk)].re);   \
        __m512d w7 = _mm512_loadu_pd(&stage_tw[6 * (sub_len) + (kk)].re);   \
        __m512d w8 = _mm512_loadu_pd(&stage_tw[7 * (sub_len) + (kk)].re);   \
        __m512d w9 = _mm512_loadu_pd(&stage_tw[8 * (sub_len) + (kk)].re);   \
        __m512d w10 = _mm512_loadu_pd(&stage_tw[9 * (sub_len) + (kk)].re);  \
        __m512d w11 = _mm512_loadu_pd(&stage_tw[10 * (sub_len) + (kk)].re); \
        __m512d w12 = _mm512_loadu_pd(&stage_tw[11 * (sub_len) + (kk)].re); \
        __m512d w13 = _mm512_loadu_pd(&stage_tw[12 * (sub_len) + (kk)].re); \
        __m512d w14 = _mm512_loadu_pd(&stage_tw[13 * (sub_len) + (kk)].re); \
        __m512d w15 = _mm512_loadu_pd(&stage_tw[14 * (sub_len) + (kk)].re); \
        CMUL_FMA_AOS_AVX512(x[1], x[1], w1);                                \
        CMUL_FMA_AOS_AVX512(x[2], x[2], w2);                                \
        CMUL_FMA_AOS_AVX512(x[3], x[3], w3);                                \
        CMUL_FMA_AOS_AVX512(x[4], x[4], w4);                                \
        CMUL_FMA_AOS_AVX512(x[5], x[5], w5);                                \
        CMUL_FMA_AOS_AVX512(x[6], x[6], w6);                                \
        CMUL_FMA_AOS_AVX512(x[7], x[7], w7);                                \
        CMUL_FMA_AOS_AVX512(x[8], x[8], w8);                                \
        CMUL_FMA_AOS_AVX512(x[9], x[9], w9);                                \
        CMUL_FMA_AOS_AVX512(x[10], x[10], w10);                             \
        CMUL_FMA_AOS_AVX512(x[11], x[11], w11);                             \
        CMUL_FMA_AOS_AVX512(x[12], x[12], w12);                             \
        CMUL_FMA_AOS_AVX512(x[13], x[13], w13);                             \
        CMUL_FMA_AOS_AVX512(x[14], x[14], w14);                             \
        CMUL_FMA_AOS_AVX512(x[15], x[15], w15);                             \
    } while (0)

//==============================================================================
// OPTIMIZED SOA BUTTERFLY PIPELINE (REDUCED REGISTER PRESSURE)
// Strategy: Load → Twiddle → 1st radix-4 → W4 twiddles → 2nd radix-4 → Store
// Peak live: ~20-24 zmm (avoiding spills)
//==============================================================================

#define RADIX16_BUTTERFLY_FV_SOA_AVX512(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                            \
    {                                                                                                                             \
        /* Load input data */                                                                                                     \
        __m512d x_re[16], x_im[16];                                                                                               \
        LOAD_16_LANES_SOA_AVX512(kk, K, sub_outputs, x_re, x_im);                                                                 \
        /* Apply input twiddles (x[0] unchanged) */                                                                               \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX512(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                   \
        /* First radix-4 stage - compute into temp, then copy back to x */                                                        \
        __m512d t_re[16], t_im[16];                                                                                               \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                     \
                                    t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                     \
                                    t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                   \
                                    t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                   \
                                    t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);    \
        /* Apply W₄ intermediate twiddles in-place on t */                                                                        \
        APPLY_W4_INTERMEDIATE_FV_SOA_AVX512(t_re, t_im, neg_mask);                                                                \
        /* Second radix-4 stage - compute directly to output using x as temp */                                                   \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                     \
                                    x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                     \
                                    x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                   \
                                    x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                   \
                                    x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);        \
        /* Store results */                                                                                                       \
        STORE_16_LANES_SOA_AVX512(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

#define RADIX16_BUTTERFLY_BV_SOA_AVX512(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                            \
    {                                                                                                                             \
        __m512d x_re[16], x_im[16];                                                                                               \
        LOAD_16_LANES_SOA_AVX512(kk, K, sub_outputs, x_re, x_im);                                                                 \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX512(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                   \
        __m512d t_re[16], t_im[16];                                                                                               \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                     \
                                    t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                     \
                                    t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                   \
                                    t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                   \
                                    t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);    \
        APPLY_W4_INTERMEDIATE_BV_SOA_AVX512(t_re, t_im, neg_mask);                                                                \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                     \
                                    x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                     \
                                    x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                   \
                                    x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                   \
                                    x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);        \
        STORE_16_LANES_SOA_AVX512(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

// Streaming versions
#define RADIX16_BUTTERFLY_FV_SOA_AVX512_STREAM(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                                   \
    {                                                                                                                                    \
        __m512d x_re[16], x_im[16];                                                                                                      \
        LOAD_16_LANES_SOA_AVX512(kk, K, sub_outputs, x_re, x_im);                                                                        \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX512(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                          \
        __m512d t_re[16], t_im[16];                                                                                                      \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                            \
                                    t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                            \
                                    t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                          \
                                    t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                          \
                                    t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);           \
        APPLY_W4_INTERMEDIATE_FV_SOA_AVX512(t_re, t_im, neg_mask);                                                                       \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                            \
                                    x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                            \
                                    x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                          \
                                    x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                          \
                                    x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);               \
        STORE_16_LANES_SOA_AVX512_STREAM(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

#define RADIX16_BUTTERFLY_BV_SOA_AVX512_STREAM(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                                   \
    {                                                                                                                                    \
        __m512d x_re[16], x_im[16];                                                                                                      \
        LOAD_16_LANES_SOA_AVX512(kk, K, sub_outputs, x_re, x_im);                                                                        \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX512(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                          \
        __m512d t_re[16], t_im[16];                                                                                                      \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                            \
                                    t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                            \
                                    t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                          \
                                    t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX512(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                          \
                                    t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);           \
        APPLY_W4_INTERMEDIATE_BV_SOA_AVX512(t_re, t_im, neg_mask);                                                                       \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                            \
                                    x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                            \
                                    x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                          \
                                    x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX512(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                          \
                                    x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);               \
        STORE_16_LANES_SOA_AVX512_STREAM(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

//==============================================================================
// LEGACY AOS PIPELINE MACROS (for backward compatibility)
//==============================================================================

#define RADIX16_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                         \
    {                                                                                                          \
        __m512d x[16];                                                                                         \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                           \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw, sub_len);                                             \
        __m512d y[16];                                                                                         \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                    \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                    \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);                 \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);               \
        APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask);                                                  \
        __m512d temp[4];                                                                                       \
        for (int m = 0; m < 4; m++)                                                                            \
        {                                                                                                      \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                       \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                             \
            y[m] = temp[0];                                                                                    \
            y[m + 4] = temp[1];                                                                                \
            y[m + 8] = temp[2];                                                                                \
            y[m + 12] = temp[3];                                                                               \
        }                                                                                                      \
        STORE_16_LANES_AVX512(kk, K, output_buffer, y);                                                        \
    } while (0)

#define RADIX16_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                         \
    {                                                                                                          \
        __m512d x[16];                                                                                         \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                           \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw, sub_len);                                             \
        __m512d y[16];                                                                                         \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                    \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                    \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);                 \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);               \
        APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask);                                                  \
        __m512d temp[4];                                                                                       \
        for (int m = 0; m < 4; m++)                                                                            \
        {                                                                                                      \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                       \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                             \
            y[m] = temp[0];                                                                                    \
            y[m + 4] = temp[1];                                                                                \
            y[m + 8] = temp[2];                                                                                \
            y[m + 12] = temp[3];                                                                               \
        }                                                                                                      \
        STORE_16_LANES_AVX512(kk, K, output_buffer, y);                                                        \
    } while (0)

#define RADIX16_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                \
    {                                                                                                                 \
        __m512d x[16];                                                                                                \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                                  \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw, sub_len);                                                    \
        __m512d y[16];                                                                                                \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                           \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                           \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);                        \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);                      \
        APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask);                                                         \
        __m512d temp[4];                                                                                              \
        for (int m = 0; m < 4; m++)                                                                                   \
        {                                                                                                             \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                              \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                                    \
            y[m] = temp[0];                                                                                           \
            y[m + 4] = temp[1];                                                                                       \
            y[m + 8] = temp[2];                                                                                       \
            y[m + 12] = temp[3];                                                                                      \
        }                                                                                                             \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, y);                                                        \
    } while (0)

#define RADIX16_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                \
    {                                                                                                                 \
        __m512d x[16];                                                                                                \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                                  \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw, sub_len);                                                    \
        __m512d y[16];                                                                                                \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                           \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                           \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);                        \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);                      \
        APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask);                                                         \
        __m512d temp[4];                                                                                              \
        for (int m = 0; m < 4; m++)                                                                                   \
        {                                                                                                             \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                              \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                                    \
            y[m] = temp[0];                                                                                           \
            y[m + 4] = temp[1];                                                                                       \
            y[m + 8] = temp[2];                                                                                       \
            y[m + 12] = temp[3];                                                                                      \
        }                                                                                                             \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, y);                                                        \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

// Hoisted sign masks for W_4 intermediate twiddles (avoiding ODR issues)
static const __m256d W4_SIGN_MASK_FV_5_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_FV_7_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_FV_13_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_FV_15_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_BV_5_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_BV_7_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_BV_13_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_BV_15_AVX2 = {0.0, -0.0, 0.0, -0.0};

//==============================================================================
// CONFIGURATION
//==============================================================================

#define K_TILE_R16_AVX2 512 // Smaller than AVX-512 due to 2-way processing

//==============================================================================
// AoS ↔ SoA CONVERSION (AVX2) - FIXED
//==============================================================================

#define AOS_TO_SOA_AVX2(aos, re, im)        \
    do                                      \
    {                                       \
        (re) = _mm256_movedup_pd(aos);      \
        (im) = _mm256_permute_pd(aos, 0xF); \
    } while (0)

#define SOA_TO_AOS_AVX2(re, im, aos) \
    (aos) = _mm256_unpacklo_pd(re, im)

//==============================================================================
// SOA COMPLEX ARITHMETIC (AVX2)
//==============================================================================

#if defined(__FMA__)
#define CMUL_FMA_SOA_AVX2(out_re, out_im, a_re, a_im, w_re, w_im)          \
    do                                                                     \
    {                                                                      \
        (out_re) = _mm256_fmsub_pd(a_re, w_re, _mm256_mul_pd(a_im, w_im)); \
        (out_im) = _mm256_fmadd_pd(a_re, w_im, _mm256_mul_pd(a_im, w_re)); \
    } while (0)
#else
#define CMUL_FMA_SOA_AVX2(out_re, out_im, a_re, a_im, w_re, w_im)                       \
    do                                                                                  \
    {                                                                                   \
        (out_re) = _mm256_sub_pd(_mm256_mul_pd(a_re, w_re), _mm256_mul_pd(a_im, w_im)); \
        (out_im) = _mm256_add_pd(_mm256_mul_pd(a_re, w_im), _mm256_mul_pd(a_im, w_re)); \
    } while (0)
#endif

//==============================================================================
// LEGACY AOS COMPLEX MULTIPLY (for backward compatibility, FIXED)
//==============================================================================

#if defined(__FMA__)
#define CMUL_FMA_AOS(out, a, w)                                      \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_movedup_pd(a);                           \
        __m256d ai = _mm256_permute_pd(a, 0xF);                      \
        __m256d wr = _mm256_movedup_pd(w);                           \
        __m256d wi = _mm256_permute_pd(w, 0xF);                      \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)
#else
#define CMUL_FMA_AOS(out, a, w)                                                   \
    do                                                                            \
    {                                                                             \
        __m256d ar = _mm256_movedup_pd(a);                                        \
        __m256d ai = _mm256_permute_pd(a, 0xF);                                   \
        __m256d wr = _mm256_movedup_pd(w);                                        \
        __m256d wi = _mm256_permute_pd(w, 0xF);                                   \
        __m256d re = _mm256_sub_pd(_mm256_mul_pd(ar, wr), _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_add_pd(_mm256_mul_pd(ar, wi), _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                                       \
    } while (0)
#endif

//==============================================================================
// SOA RADIX-4 BUTTERFLY (AVX2)
//==============================================================================

#define RADIX4_BUTTERFLY_SOA_AVX2(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                        \
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, rot_sign_mask) \
    do                                                                                                   \
    {                                                                                                    \
        __m256d sumBD_re = _mm256_add_pd(b_re, d_re);                                                    \
        __m256d sumBD_im = _mm256_add_pd(b_im, d_im);                                                    \
        __m256d difBD_re = _mm256_sub_pd(b_re, d_re);                                                    \
        __m256d difBD_im = _mm256_sub_pd(b_im, d_im);                                                    \
        __m256d sumAC_re = _mm256_add_pd(a_re, c_re);                                                    \
        __m256d sumAC_im = _mm256_add_pd(a_im, c_im);                                                    \
        __m256d difAC_re = _mm256_sub_pd(a_re, c_re);                                                    \
        __m256d difAC_im = _mm256_sub_pd(a_im, c_im);                                                    \
        y0_re = _mm256_add_pd(sumAC_re, sumBD_re);                                                       \
        y0_im = _mm256_add_pd(sumAC_im, sumBD_im);                                                       \
        y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);                                                       \
        y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);                                                       \
        __m256d rot_re = _mm256_xor_pd(difBD_im, rot_sign_mask);                                         \
        __m256d rot_im = _mm256_xor_pd(_mm256_sub_pd(_mm256_setzero_pd(), difBD_re), rot_sign_mask);     \
        y1_re = _mm256_sub_pd(difAC_re, rot_re);                                                         \
        y1_im = _mm256_sub_pd(difAC_im, rot_im);                                                         \
        y3_re = _mm256_add_pd(difAC_re, rot_re);                                                         \
        y3_im = _mm256_add_pd(difAC_im, rot_im);                                                         \
    } while (0)

//==============================================================================
// LEGACY AOS RADIX-4 BUTTERFLY (for backward compatibility)
//==============================================================================

#define RADIX4_BUTTERFLY_AVX2(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do                                                              \
    {                                                               \
        __m256d sumBD = _mm256_add_pd(b, d);                        \
        __m256d difBD = _mm256_sub_pd(b, d);                        \
        __m256d sumAC = _mm256_add_pd(a, c);                        \
        __m256d difAC = _mm256_sub_pd(a, c);                        \
        y0 = _mm256_add_pd(sumAC, sumBD);                           \
        y2 = _mm256_sub_pd(sumAC, sumBD);                           \
        __m256d dif_bd_swp = _mm256_permute_pd(difBD, 0b0101);      \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask);   \
        y1 = _mm256_sub_pd(difAC, dif_bd_rot);                      \
        y3 = _mm256_add_pd(difAC, dif_bd_rot);                      \
    } while (0)

//==============================================================================
// SOA W_4 INTERMEDIATE TWIDDLES (AVX2)
//==============================================================================

#define APPLY_W4_INTERMEDIATE_FV_SOA_AVX2(y_re, y_im, neg_mask) \
    do                                                          \
    {                                                           \
        /* m=1: W_4^{j} for j=1,2,3 = {-i, -1, +i} */           \
        {                                                       \
            __m256d tmp_re = y_re[5];                           \
            y_re[5] = y_im[5];                                  \
            y_im[5] = _mm256_xor_pd(tmp_re, neg_mask);          \
            y_re[6] = _mm256_xor_pd(y_re[6], neg_mask);         \
            y_im[6] = _mm256_xor_pd(y_im[6], neg_mask);         \
            tmp_re = y_re[7];                                   \
            y_re[7] = _mm256_xor_pd(y_im[7], neg_mask);         \
            y_im[7] = tmp_re;                                   \
        }                                                       \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, +1, -1} */          \
        {                                                       \
            y_re[9] = _mm256_xor_pd(y_re[9], neg_mask);         \
            y_im[9] = _mm256_xor_pd(y_im[9], neg_mask);         \
            y_re[11] = _mm256_xor_pd(y_re[11], neg_mask);       \
            y_im[11] = _mm256_xor_pd(y_im[11], neg_mask);       \
        }                                                       \
        /* m=3: W_4^{3j} for j=1,2,3 = {+i, -1, -i} */          \
        {                                                       \
            __m256d tmp_re = y_re[13];                          \
            y_re[13] = _mm256_xor_pd(y_im[13], neg_mask);       \
            y_im[13] = tmp_re;                                  \
            y_re[14] = _mm256_xor_pd(y_re[14], neg_mask);       \
            y_im[14] = _mm256_xor_pd(y_im[14], neg_mask);       \
            tmp_re = y_re[15];                                  \
            y_re[15] = y_im[15];                                \
            y_im[15] = _mm256_xor_pd(tmp_re, neg_mask);         \
        }                                                       \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_SOA_AVX2(y_re, y_im, neg_mask) \
    do                                                          \
    {                                                           \
        /* m=1: W_4^{j} for j=1,2,3 = {+i, -1, -i} */           \
        {                                                       \
            __m256d tmp_re = y_re[5];                           \
            y_re[5] = _mm256_xor_pd(y_im[5], neg_mask);         \
            y_im[5] = tmp_re;                                   \
            y_re[6] = _mm256_xor_pd(y_re[6], neg_mask);         \
            y_im[6] = _mm256_xor_pd(y_im[6], neg_mask);         \
            tmp_re = y_re[7];                                   \
            y_re[7] = y_im[7];                                  \
            y_im[7] = _mm256_xor_pd(tmp_re, neg_mask);          \
        }                                                       \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, +1, -1} */          \
        {                                                       \
            y_re[9] = _mm256_xor_pd(y_re[9], neg_mask);         \
            y_im[9] = _mm256_xor_pd(y_im[9], neg_mask);         \
            y_re[11] = _mm256_xor_pd(y_re[11], neg_mask);       \
            y_im[11] = _mm256_xor_pd(y_im[11], neg_mask);       \
        }                                                       \
        /* m=3: W_4^{3j} for j=1,2,3 = {-i, -1, +i} */          \
        {                                                       \
            __m256d tmp_re = y_re[13];                          \
            y_re[13] = y_im[13];                                \
            y_im[13] = _mm256_xor_pd(tmp_re, neg_mask);         \
            y_re[14] = _mm256_xor_pd(y_re[14], neg_mask);       \
            y_im[14] = _mm256_xor_pd(y_im[14], neg_mask);       \
            tmp_re = y_re[15];                                  \
            y_re[15] = _mm256_xor_pd(y_im[15], neg_mask);       \
            y_im[15] = tmp_re;                                  \
        }                                                       \
    } while (0)

//==============================================================================
// LEGACY AOS W_4 INTERMEDIATE TWIDDLES (for backward compatibility)
//==============================================================================

#define APPLY_W4_INTERMEDIATE_FV_AVX2_HOISTED(y, neg_mask)           \
    do                                                               \
    {                                                                \
        {                                                            \
            __m256d y5_swp = _mm256_permute_pd(y[5], 0x5);           \
            y[5] = _mm256_xor_pd(y5_swp, W4_SIGN_MASK_FV_5_AVX2);    \
            y[6] = _mm256_xor_pd(y[6], neg_mask);                    \
            __m256d y7_swp = _mm256_permute_pd(y[7], 0x5);           \
            y[7] = _mm256_xor_pd(y7_swp, W4_SIGN_MASK_FV_7_AVX2);    \
        }                                                            \
        {                                                            \
            y[9] = _mm256_xor_pd(y[9], neg_mask);                    \
            y[11] = _mm256_xor_pd(y[11], neg_mask);                  \
        }                                                            \
        {                                                            \
            __m256d y13_swp = _mm256_permute_pd(y[13], 0x5);         \
            y[13] = _mm256_xor_pd(y13_swp, W4_SIGN_MASK_FV_13_AVX2); \
            y[14] = _mm256_xor_pd(y[14], neg_mask);                  \
            __m256d y15_swp = _mm256_permute_pd(y[15], 0x5);         \
            y[15] = _mm256_xor_pd(y15_swp, W4_SIGN_MASK_FV_15_AVX2); \
        }                                                            \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_AVX2_HOISTED(y, neg_mask)           \
    do                                                               \
    {                                                                \
        {                                                            \
            __m256d y5_swp = _mm256_permute_pd(y[5], 0x5);           \
            y[5] = _mm256_xor_pd(y5_swp, W4_SIGN_MASK_BV_5_AVX2);    \
            y[6] = _mm256_xor_pd(y[6], neg_mask);                    \
            __m256d y7_swp = _mm256_permute_pd(y[7], 0x5);           \
            y[7] = _mm256_xor_pd(y7_swp, W4_SIGN_MASK_BV_7_AVX2);    \
        }                                                            \
        {                                                            \
            y[9] = _mm256_xor_pd(y[9], neg_mask);                    \
            y[11] = _mm256_xor_pd(y[11], neg_mask);                  \
        }                                                            \
        {                                                            \
            __m256d y13_swp = _mm256_permute_pd(y[13], 0x5);         \
            y[13] = _mm256_xor_pd(y13_swp, W4_SIGN_MASK_BV_13_AVX2); \
            y[14] = _mm256_xor_pd(y[14], neg_mask);                  \
            __m256d y15_swp = _mm256_permute_pd(y[15], 0x5);         \
            y[15] = _mm256_xor_pd(y15_swp, W4_SIGN_MASK_BV_15_AVX2); \
        }                                                            \
    } while (0)

//==============================================================================
// LOAD & STORE HELPERS (AVX2)
//==============================================================================

#define LOAD_16_LANES_SOA_AVX2(kk, K, sub_outputs, x_re, x_im)          \
    do                                                                  \
    {                                                                   \
        for (int lane = 0; lane < 16; lane++)                           \
        {                                                               \
            __m256d aos = load2_aos(&sub_outputs[(kk) + lane * K],      \
                                    &sub_outputs[(kk) + 1 + lane * K]); \
            AOS_TO_SOA_AVX2(aos, x_re[lane], x_im[lane]);               \
        }                                                               \
    } while (0)

#define STORE_16_LANES_SOA_AVX2(kk, K, output_buffer, y_re, y_im)     \
    do                                                                \
    {                                                                 \
        for (int m = 0; m < 4; m++)                                   \
        {                                                             \
            __m256d aos0, aos4, aos8, aos12;                          \
            SOA_TO_AOS_AVX2(y_re[m], y_im[m], aos0);                  \
            SOA_TO_AOS_AVX2(y_re[m + 4], y_im[m + 4], aos4);          \
            SOA_TO_AOS_AVX2(y_re[m + 8], y_im[m + 8], aos8);          \
            SOA_TO_AOS_AVX2(y_re[m + 12], y_im[m + 12], aos12);       \
            STOREU_PD(&output_buffer[(kk) + m * K].re, aos0);         \
            STOREU_PD(&output_buffer[(kk) + (m + 4) * K].re, aos4);   \
            STOREU_PD(&output_buffer[(kk) + (m + 8) * K].re, aos8);   \
            STOREU_PD(&output_buffer[(kk) + (m + 12) * K].re, aos12); \
        }                                                             \
    } while (0)

#define STORE_16_LANES_SOA_AVX2_STREAM(kk, K, output_buffer, y_re, y_im)     \
    do                                                                       \
    {                                                                        \
        for (int m = 0; m < 4; m++)                                          \
        {                                                                    \
            __m256d aos0, aos4, aos8, aos12;                                 \
            SOA_TO_AOS_AVX2(y_re[m], y_im[m], aos0);                         \
            SOA_TO_AOS_AVX2(y_re[m + 4], y_im[m + 4], aos4);                 \
            SOA_TO_AOS_AVX2(y_re[m + 8], y_im[m + 8], aos8);                 \
            SOA_TO_AOS_AVX2(y_re[m + 12], y_im[m + 12], aos12);              \
            _mm256_stream_pd(&output_buffer[(kk) + m * K].re, aos0);         \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, aos4);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, aos8);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, aos12); \
        }                                                                    \
    } while (0)

// Legacy AoS versions
#define LOAD_16_LANES_AVX2(kk, K, sub_outputs, x)                                                  \
    do                                                                                             \
    {                                                                                              \
        for (int lane = 0; lane < 16; lane++)                                                      \
        {                                                                                          \
            x[lane] = load2_aos(&sub_outputs[(kk) + lane * K], &sub_outputs[(kk) + 1 + lane * K]); \
        }                                                                                          \
    } while (0)

#define STORE_16_LANES_AVX2(kk, K, output_buffer, y)                      \
    do                                                                    \
    {                                                                     \
        for (int m = 0; m < 4; m++)                                       \
        {                                                                 \
            STOREU_PD(&output_buffer[(kk) + m * K].re, y[m]);             \
            STOREU_PD(&output_buffer[(kk) + (m + 4) * K].re, y[m + 4]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 8) * K].re, y[m + 8]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 12) * K].re, y[m + 12]); \
        }                                                                 \
    } while (0)

#define STORE_16_LANES_AVX2_STREAM(kk, K, output_buffer, y)                      \
    do                                                                           \
    {                                                                            \
        for (int m = 0; m < 4; m++)                                              \
        {                                                                        \
            _mm256_stream_pd(&output_buffer[(kk) + m * K].re, y[m]);             \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, y[m + 4]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, y[m + 8]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, y[m + 12]); \
        }                                                                        \
    } while (0)

//==============================================================================
// PREFETCHING (AVX2)
//==============================================================================

#define PREFETCH_16_LANES(k, K, distance, sub_outputs, hint)                                 \
    do                                                                                       \
    {                                                                                        \
        if ((k) + (distance) < K)                                                            \
        {                                                                                    \
            for (int lane = 0; lane < 16; lane++)                                            \
            {                                                                                \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + lane * K], hint); \
            }                                                                                \
        }                                                                                    \
    } while (0)

#define PREFETCH_STAGE_TW_AVX2(kk, distance, stage_tw_re, stage_tw_im, sub_len)                   \
    do                                                                                            \
    {                                                                                             \
        _mm_prefetch((const char *)&stage_tw_re[0 * (sub_len) + (kk) + (distance)], _MM_HINT_T0); \
        _mm_prefetch((const char *)&stage_tw_im[0 * (sub_len) + (kk) + (distance)], _MM_HINT_T0); \
    } while (0)

//==============================================================================
// SOA TWIDDLE APPLICATION - OPTIMIZED (AVX2)
//==============================================================================

#define APPLY_STAGE_TWIDDLES_R16_SOA_AVX2(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len) \
    do                                                                                       \
    {                                                                                        \
        /* Unrolled by 3 for software pipelining */                                          \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw_re[0 * (sub_len) + (kk)]);                 \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw_im[0 * (sub_len) + (kk)]);                 \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw_re[1 * (sub_len) + (kk)]);                 \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw_im[1 * (sub_len) + (kk)]);                 \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw_re[2 * (sub_len) + (kk)]);                 \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw_im[2 * (sub_len) + (kk)]);                 \
        CMUL_FMA_SOA_AVX2(x_re[1], x_im[1], x_re[1], x_im[1], w1_re, w1_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[2], x_im[2], x_re[2], x_im[2], w2_re, w2_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[3], x_im[3], x_re[3], x_im[3], w3_re, w3_im);                 \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw_re[3 * (sub_len) + (kk)]);                 \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw_im[3 * (sub_len) + (kk)]);                 \
        __m256d w5_re = _mm256_loadu_pd(&stage_tw_re[4 * (sub_len) + (kk)]);                 \
        __m256d w5_im = _mm256_loadu_pd(&stage_tw_im[4 * (sub_len) + (kk)]);                 \
        __m256d w6_re = _mm256_loadu_pd(&stage_tw_re[5 * (sub_len) + (kk)]);                 \
        __m256d w6_im = _mm256_loadu_pd(&stage_tw_im[5 * (sub_len) + (kk)]);                 \
        CMUL_FMA_SOA_AVX2(x_re[4], x_im[4], x_re[4], x_im[4], w4_re, w4_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[5], x_im[5], x_re[5], x_im[5], w5_re, w5_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[6], x_im[6], x_re[6], x_im[6], w6_re, w6_im);                 \
        __m256d w7_re = _mm256_loadu_pd(&stage_tw_re[6 * (sub_len) + (kk)]);                 \
        __m256d w7_im = _mm256_loadu_pd(&stage_tw_im[6 * (sub_len) + (kk)]);                 \
        __m256d w8_re = _mm256_loadu_pd(&stage_tw_re[7 * (sub_len) + (kk)]);                 \
        __m256d w8_im = _mm256_loadu_pd(&stage_tw_im[7 * (sub_len) + (kk)]);                 \
        __m256d w9_re = _mm256_loadu_pd(&stage_tw_re[8 * (sub_len) + (kk)]);                 \
        __m256d w9_im = _mm256_loadu_pd(&stage_tw_im[8 * (sub_len) + (kk)]);                 \
        CMUL_FMA_SOA_AVX2(x_re[7], x_im[7], x_re[7], x_im[7], w7_re, w7_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[8], x_im[8], x_re[8], x_im[8], w8_re, w8_im);                 \
        CMUL_FMA_SOA_AVX2(x_re[9], x_im[9], x_re[9], x_im[9], w9_re, w9_im);                 \
        __m256d w10_re = _mm256_loadu_pd(&stage_tw_re[9 * (sub_len) + (kk)]);                \
        __m256d w10_im = _mm256_loadu_pd(&stage_tw_im[9 * (sub_len) + (kk)]);                \
        __m256d w11_re = _mm256_loadu_pd(&stage_tw_re[10 * (sub_len) + (kk)]);               \
        __m256d w11_im = _mm256_loadu_pd(&stage_tw_im[10 * (sub_len) + (kk)]);               \
        __m256d w12_re = _mm256_loadu_pd(&stage_tw_re[11 * (sub_len) + (kk)]);               \
        __m256d w12_im = _mm256_loadu_pd(&stage_tw_im[11 * (sub_len) + (kk)]);               \
        CMUL_FMA_SOA_AVX2(x_re[10], x_im[10], x_re[10], x_im[10], w10_re, w10_im);           \
        CMUL_FMA_SOA_AVX2(x_re[11], x_im[11], x_re[11], x_im[11], w11_re, w11_im);           \
        CMUL_FMA_SOA_AVX2(x_re[12], x_im[12], x_re[12], x_im[12], w12_re, w12_im);           \
        __m256d w13_re = _mm256_loadu_pd(&stage_tw_re[12 * (sub_len) + (kk)]);               \
        __m256d w13_im = _mm256_loadu_pd(&stage_tw_im[12 * (sub_len) + (kk)]);               \
        __m256d w14_re = _mm256_loadu_pd(&stage_tw_re[13 * (sub_len) + (kk)]);               \
        __m256d w14_im = _mm256_loadu_pd(&stage_tw_im[13 * (sub_len) + (kk)]);               \
        __m256d w15_re = _mm256_loadu_pd(&stage_tw_re[14 * (sub_len) + (kk)]);               \
        __m256d w15_im = _mm256_loadu_pd(&stage_tw_im[14 * (sub_len) + (kk)]);               \
        CMUL_FMA_SOA_AVX2(x_re[13], x_im[13], x_re[13], x_im[13], w13_re, w13_im);           \
        CMUL_FMA_SOA_AVX2(x_re[14], x_im[14], x_re[14], x_im[14], w14_re, w14_im);           \
        CMUL_FMA_SOA_AVX2(x_re[15], x_im[15], x_re[15], x_im[15], w15_re, w15_im);           \
    } while (0)

//==============================================================================
// LEGACY AOS TWIDDLE APPLICATION (for backward compatibility)
//==============================================================================

#define APPLY_STAGE_TWIDDLES_R16_AVX2(kk, x, stage_tw, sub_len)             \
    do                                                                      \
    {                                                                       \
        __m256d w1 = _mm256_loadu_pd(&stage_tw[0 * (sub_len) + (kk)].re);   \
        __m256d w2 = _mm256_loadu_pd(&stage_tw[1 * (sub_len) + (kk)].re);   \
        __m256d w3 = _mm256_loadu_pd(&stage_tw[2 * (sub_len) + (kk)].re);   \
        __m256d w4 = _mm256_loadu_pd(&stage_tw[3 * (sub_len) + (kk)].re);   \
        __m256d w5 = _mm256_loadu_pd(&stage_tw[4 * (sub_len) + (kk)].re);   \
        __m256d w6 = _mm256_loadu_pd(&stage_tw[5 * (sub_len) + (kk)].re);   \
        __m256d w7 = _mm256_loadu_pd(&stage_tw[6 * (sub_len) + (kk)].re);   \
        __m256d w8 = _mm256_loadu_pd(&stage_tw[7 * (sub_len) + (kk)].re);   \
        __m256d w9 = _mm256_loadu_pd(&stage_tw[8 * (sub_len) + (kk)].re);   \
        __m256d w10 = _mm256_loadu_pd(&stage_tw[9 * (sub_len) + (kk)].re);  \
        __m256d w11 = _mm256_loadu_pd(&stage_tw[10 * (sub_len) + (kk)].re); \
        __m256d w12 = _mm256_loadu_pd(&stage_tw[11 * (sub_len) + (kk)].re); \
        __m256d w13 = _mm256_loadu_pd(&stage_tw[12 * (sub_len) + (kk)].re); \
        __m256d w14 = _mm256_loadu_pd(&stage_tw[13 * (sub_len) + (kk)].re); \
        __m256d w15 = _mm256_loadu_pd(&stage_tw[14 * (sub_len) + (kk)].re); \
        CMUL_FMA_AOS(x[1], x[1], w1);                                       \
        CMUL_FMA_AOS(x[2], x[2], w2);                                       \
        CMUL_FMA_AOS(x[3], x[3], w3);                                       \
        CMUL_FMA_AOS(x[4], x[4], w4);                                       \
        CMUL_FMA_AOS(x[5], x[5], w5);                                       \
        CMUL_FMA_AOS(x[6], x[6], w6);                                       \
        CMUL_FMA_AOS(x[7], x[7], w7);                                       \
        CMUL_FMA_AOS(x[8], x[8], w8);                                       \
        CMUL_FMA_AOS(x[9], x[9], w9);                                       \
        CMUL_FMA_AOS(x[10], x[10], w10);                                    \
        CMUL_FMA_AOS(x[11], x[11], w11);                                    \
        CMUL_FMA_AOS(x[12], x[12], w12);                                    \
        CMUL_FMA_AOS(x[13], x[13], w13);                                    \
        CMUL_FMA_AOS(x[14], x[14], w14);                                    \
        CMUL_FMA_AOS(x[15], x[15], w15);                                    \
    } while (0)

//==============================================================================
// OPTIMIZED SOA BUTTERFLY PIPELINES (AVX2) - REDUCED REGISTER PRESSURE
//==============================================================================

#define RADIX16_BUTTERFLY_FV_SOA_AVX2(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                          \
    {                                                                                                                           \
        __m256d x_re[16], x_im[16];                                                                                             \
        LOAD_16_LANES_SOA_AVX2(kk, K, sub_outputs, x_re, x_im);                                                                 \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX2(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                   \
        __m256d t_re[16], t_im[16];                                                                                             \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                     \
                                  t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                     \
                                  t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                   \
                                  t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                   \
                                  t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);    \
        APPLY_W4_INTERMEDIATE_FV_SOA_AVX2(t_re, t_im, neg_mask);                                                                \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                     \
                                  x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                     \
                                  x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                   \
                                  x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                   \
                                  x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);        \
        STORE_16_LANES_SOA_AVX2(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

#define RADIX16_BUTTERFLY_BV_SOA_AVX2_STREAM(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                                 \
    {                                                                                                                                  \
        __m256d x_re[16], x_im[16];                                                                                                    \
        LOAD_16_LANES_SOA_AVX2(kk, K, sub_outputs, x_re, x_im);                                                                        \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX2(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                          \
        __m256d t_re[16], t_im[16];                                                                                                    \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                            \
                                  t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                            \
                                  t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                          \
                                  t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                          \
                                  t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);           \
        APPLY_W4_INTERMEDIATE_BV_SOA_AVX2(t_re, t_im, neg_mask);                                                                       \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                            \
                                  x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                            \
                                  x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                          \
                                  x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                          \
                                  x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);               \
        STORE_16_LANES_SOA_AVX2_STREAM(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

// STREAMING FORWARD (FV)
#define RADIX16_BUTTERFLY_FV_SOA_AVX2_STREAM(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                                 \
    {                                                                                                                                  \
        __m256d x_re[16], x_im[16];                                                                                                    \
        LOAD_16_LANES_SOA_AVX2(kk, K, sub_outputs, x_re, x_im);                                                                        \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX2(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                          \
        __m256d t_re[16], t_im[16];                                                                                                    \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                            \
                                  t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                            \
                                  t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);                   \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                          \
                                  t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                          \
                                  t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);           \
        APPLY_W4_INTERMEDIATE_FV_SOA_AVX2(t_re, t_im, neg_mask);                                                                       \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                            \
                                  x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                            \
                                  x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);                 \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                          \
                                  x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);               \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                          \
                                  x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);               \
        STORE_16_LANES_SOA_AVX2_STREAM(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

#define RADIX16_BUTTERFLY_BV_SOA_AVX2(kk, K, sub_outputs, stage_tw_re, stage_tw_im, sub_len, output_buffer, rot_mask, neg_mask) \
    do                                                                                                                          \
    {                                                                                                                           \
        __m256d x_re[16], x_im[16];                                                                                             \
        LOAD_16_LANES_SOA_AVX2(kk, K, sub_outputs, x_re, x_im);                                                                 \
        APPLY_STAGE_TWIDDLES_R16_SOA_AVX2(kk, x_re, x_im, stage_tw_re, stage_tw_im, sub_len);                                   \
        __m256d t_re[16], t_im[16];                                                                                             \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12],                     \
                                  t_re[0], t_im[0], t_re[1], t_im[1], t_re[2], t_im[2], t_re[3], t_im[3], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13],                     \
                                  t_re[4], t_im[4], t_re[5], t_im[5], t_re[6], t_im[6], t_re[7], t_im[7], rot_mask);            \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14],                   \
                                  t_re[8], t_im[8], t_re[9], t_im[9], t_re[10], t_im[10], t_re[11], t_im[11], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX2(x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15],                   \
                                  t_re[12], t_im[12], t_re[13], t_im[13], t_re[14], t_im[14], t_re[15], t_im[15], rot_mask);    \
        APPLY_W4_INTERMEDIATE_BV_SOA_AVX2(t_re, t_im, neg_mask);                                                                \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[0], t_im[0], t_re[4], t_im[4], t_re[8], t_im[8], t_re[12], t_im[12],                     \
                                  x_re[0], x_im[0], x_re[4], x_im[4], x_re[8], x_im[8], x_re[12], x_im[12], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[1], t_im[1], t_re[5], t_im[5], t_re[9], t_im[9], t_re[13], t_im[13],                     \
                                  x_re[1], x_im[1], x_re[5], x_im[5], x_re[9], x_im[9], x_re[13], x_im[13], rot_mask);          \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[2], t_im[2], t_re[6], t_im[6], t_re[10], t_im[10], t_re[14], t_im[14],                   \
                                  x_re[2], x_im[2], x_re[6], x_im[6], x_re[10], x_im[10], x_re[14], x_im[14], rot_mask);        \
        RADIX4_BUTTERFLY_SOA_AVX2(t_re[3], t_im[3], t_re[7], t_im[7], t_re[11], t_im[11], t_re[15], t_im[15],                   \
                                  x_re[3], x_im[3], x_re[7], x_im[7], x_re[11], x_im[11], x_re[15], x_im[15], rot_mask);        \
        STORE_16_LANES_SOA_AVX2(kk, K, output_buffer, x_re, x_im);                                                              \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT (Fallback)
//==============================================================================

#define RADIX4_BUTTERFLY_SCALAR(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                   \
                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, rot_sign) \
    do                                                                                            \
    {                                                                                             \
        double sumBD_re = b_re + d_re, sumBD_im = b_im + d_im;                                    \
        double difBD_re = b_re - d_re, difBD_im = b_im - d_im;                                    \
        double sumAC_re = a_re + c_re, sumAC_im = a_im + c_im;                                    \
        double difAC_re = a_re - c_re, difAC_im = a_im - c_im;                                    \
        y0_re = sumAC_re + sumBD_re;                                                              \
        y0_im = sumAC_im + sumBD_im;                                                              \
        y2_re = sumAC_re - sumBD_re;                                                              \
        y2_im = sumAC_im - sumBD_im;                                                              \
        double rot_re = (rot_sign) * difBD_im, rot_im = (rot_sign) * (-difBD_re);                 \
        y1_re = difAC_re - rot_re;                                                                \
        y1_im = difAC_im - rot_im;                                                                \
        y3_re = difAC_re + rot_re;                                                                \
        y3_im = difAC_im + rot_im;                                                                \
    } while (0)

#define APPLY_W4_INTERMEDIATE_FV_SCALAR(y)     \
    do                                         \
    {                                          \
        {                                      \
            double r = y[5].re, i = y[5].im;   \
            y[5].re = i;                       \
            y[5].im = -r;                      \
            y[6].re = -y[6].re;                \
            y[6].im = -y[6].im;                \
            r = y[7].re;                       \
            i = y[7].im;                       \
            y[7].re = -i;                      \
            y[7].im = r;                       \
        }                                      \
        {                                      \
            y[9].re = -y[9].re;                \
            y[9].im = -y[9].im;                \
            y[11].re = -y[11].re;              \
            y[11].im = -y[11].im;              \
        }                                      \
        {                                      \
            double r = y[13].re, i = y[13].im; \
            y[13].re = -i;                     \
            y[13].im = r;                      \
            y[14].re = -y[14].re;              \
            y[14].im = -y[14].im;              \
            r = y[15].re;                      \
            i = y[15].im;                      \
            y[15].re = i;                      \
            y[15].im = -r;                     \
        }                                      \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_SCALAR(y)     \
    do                                         \
    {                                          \
        {                                      \
            double r = y[5].re, i = y[5].im;   \
            y[5].re = -i;                      \
            y[5].im = r;                       \
            y[6].re = -y[6].re;                \
            y[6].im = -y[6].im;                \
            r = y[7].re;                       \
            i = y[7].im;                       \
            y[7].re = i;                       \
            y[7].im = -r;                      \
        }                                      \
        {                                      \
            y[9].re = -y[9].re;                \
            y[9].im = -y[9].im;                \
            y[11].re = -y[11].re;              \
            y[11].im = -y[11].im;              \
        }                                      \
        {                                      \
            double r = y[13].re, i = y[13].im; \
            y[13].re = i;                      \
            y[13].im = -r;                     \
            y[14].re = -y[14].re;              \
            y[14].im = -y[14].im;              \
            r = y[15].re;                      \
            i = y[15].im;                      \
            y[15].re = -i;                     \
            y[15].im = r;                      \
        }                                      \
    } while (0)

#define APPLY_STAGE_TWIDDLES_R16_SCALAR(k, x, stage_tw, sub_len)      \
    do                                                                \
    {                                                                 \
        for (int r = 1; r <= 15; r++)                                 \
        {                                                             \
            const fft_data *w = &stage_tw[(r - 1) * (sub_len) + (k)]; \
            fft_data a = x[r];                                        \
            x[r].re = a.re * w->re - a.im * w->im;                    \
            x[r].im = a.re * w->im + a.im * w->re;                    \
        }                                                             \
    } while (0)

#define APPLY_STAGE_TWIDDLES_R16_SCALAR_SOA(k, x, tw, K)     \
    do                                                       \
    {                                                        \
        for (int r = 1; r <= 15; ++r)                        \
        {                                                    \
            const double wr = (tw)->re[(r - 1) * (K) + (k)]; \
            const double wi = (tw)->im[(r - 1) * (K) + (k)]; \
            const double ar = x[r].re, ai = x[r].im;         \
            x[r].re = ar * wr - ai * wi;                     \
            x[r].im = ar * wi + ai * wr;                     \
        }                                                    \
    } while (0)

#endif // FFT_RADIX16_MACROS