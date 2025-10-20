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
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

// Hoisted sign masks for W_4 intermediate twiddles - Forward
static const __m512d W4_SIGN_MASK_FV_5 = {-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0};
static const __m512d W4_SIGN_MASK_FV_7 = {0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};
static const __m512d W4_SIGN_MASK_FV_13 = {0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};
static const __m512d W4_SIGN_MASK_FV_15 = {-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0};

// Hoisted sign masks for W_4 intermediate twiddles - Backward
static const __m512d W4_SIGN_MASK_BV_5 = {0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};
static const __m512d W4_SIGN_MASK_BV_7 = {-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0};
static const __m512d W4_SIGN_MASK_BV_13 = {-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0};
static const __m512d W4_SIGN_MASK_BV_15 = {0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};

#define CMUL_FMA_AOS_AVX512(out, a, w)                               \
    do                                                               \
    {                                                                \
        __m512d ar = _mm512_unpacklo_pd(a, a);                       \
        __m512d ai = _mm512_unpackhi_pd(a, a);                       \
        __m512d wr = _mm512_unpacklo_pd(w, w);                       \
        __m512d wi = _mm512_unpackhi_pd(w, w);                       \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
        (out) = _mm512_unpacklo_pd(re, im);                          \
    } while (0)

#define RADIX4_BUTTERFLY_AVX512(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do                                                                \
    {                                                                 \
        __m512d sumBD = _mm512_add_pd(b, d);                          \
        __m512d difBD = _mm512_sub_pd(b, d);                          \
        __m512d sumAC = _mm512_add_pd(a, c);                          \
        __m512d difAC = _mm512_sub_pd(a, c);                          \
                                                                      \
        y0 = _mm512_add_pd(sumAC, sumBD);                             \
        y2 = _mm512_sub_pd(sumAC, sumBD);                             \
                                                                      \
        __m512d dif_bd_swp = _mm512_permute_pd(difBD, 0b01010101);    \
        __m512d dif_bd_rot = _mm512_xor_pd(dif_bd_swp, rot_mask);     \
                                                                      \
        y1 = _mm512_sub_pd(difAC, dif_bd_rot);                        \
        y3 = _mm512_add_pd(difAC, dif_bd_rot);                        \
    } while (0)

//==============================================================================
// OPTIMIZED W_4 INTERMEDIATE TWIDDLES - AVX-512 (with hoisted masks)
//==============================================================================

#define APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask)  \
    do                                                        \
    {                                                         \
        /* m=1: W_4^j for j=1,2,3 = {-i, -1, +i} */           \
        {                                                     \
            __m512d y5_swp = _mm512_permute_pd(y[5], 0x55);   \
            y[5] = _mm512_xor_pd(y5_swp, W4_SIGN_MASK_FV_5);  \
            y[6] = _mm512_xor_pd(y[6], neg_mask);             \
            __m512d y7_swp = _mm512_permute_pd(y[7], 0x55);   \
            y[7] = _mm512_xor_pd(y7_swp, W4_SIGN_MASK_FV_7);  \
        }                                                     \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, 1, -1} */         \
        {                                                     \
            y[9] = _mm512_xor_pd(y[9], neg_mask);             \
            y[11] = _mm512_xor_pd(y[11], neg_mask);           \
        }                                                     \
        /* m=3: W_4^{3j} for j=1,2,3 = {+i, -1, -i} */        \
        {                                                     \
            __m512d y13_swp = _mm512_permute_pd(y[13], 0x55); \
            y[13] = _mm512_xor_pd(y13_swp, W4_SIGN_MASK_FV_13); \
            y[14] = _mm512_xor_pd(y[14], neg_mask);           \
            __m512d y15_swp = _mm512_permute_pd(y[15], 0x55); \
            y[15] = _mm512_xor_pd(y15_swp, W4_SIGN_MASK_FV_15); \
        }                                                     \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask)  \
    do                                                        \
    {                                                         \
        /* m=1: W_4^j for j=1,2,3 = {+i, -1, -i} */           \
        {                                                     \
            __m512d y5_swp = _mm512_permute_pd(y[5], 0x55);   \
            y[5] = _mm512_xor_pd(y5_swp, W4_SIGN_MASK_BV_5);  \
            y[6] = _mm512_xor_pd(y[6], neg_mask);             \
            __m512d y7_swp = _mm512_permute_pd(y[7], 0x55);   \
            y[7] = _mm512_xor_pd(y7_swp, W4_SIGN_MASK_BV_7);  \
        }                                                     \
        /* m=2: W_4^{2j} for j=1,2,3 = {-1, 1, -1} */         \
        {                                                     \
            y[9] = _mm512_xor_pd(y[9], neg_mask);             \
            y[11] = _mm512_xor_pd(y[11], neg_mask);           \
        }                                                     \
        /* m=3: W_4^{3j} for j=1,2,3 = {-i, -1, +i} */        \
        {                                                     \
            __m512d y13_swp = _mm512_permute_pd(y[13], 0x55); \
            y[13] = _mm512_xor_pd(y13_swp, W4_SIGN_MASK_BV_13); \
            y[14] = _mm512_xor_pd(y[14], neg_mask);           \
            __m512d y15_swp = _mm512_permute_pd(y[15], 0x55); \
            y[15] = _mm512_xor_pd(y15_swp, W4_SIGN_MASK_BV_15); \
        }                                                     \
    } while (0)

//==============================================================================
// FULLY UNROLLED TWIDDLE APPLICATION - AVX-512
//==============================================================================

#define APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw)               \
    do                                                                 \
    {                                                                  \
        __m512d w1 = load4_aos(&stage_tw[(kk) * 15 + 0],               \
                               &stage_tw[(kk + 1) * 15 + 0],           \
                               &stage_tw[(kk + 2) * 15 + 0],           \
                               &stage_tw[(kk + 3) * 15 + 0]);          \
        __m512d w2 = load4_aos(&stage_tw[(kk) * 15 + 1],               \
                               &stage_tw[(kk + 1) * 15 + 1],           \
                               &stage_tw[(kk + 2) * 15 + 1],           \
                               &stage_tw[(kk + 3) * 15 + 1]);          \
        __m512d w3 = load4_aos(&stage_tw[(kk) * 15 + 2],               \
                               &stage_tw[(kk + 1) * 15 + 2],           \
                               &stage_tw[(kk + 2) * 15 + 2],           \
                               &stage_tw[(kk + 3) * 15 + 2]);          \
        __m512d w4 = load4_aos(&stage_tw[(kk) * 15 + 3],               \
                               &stage_tw[(kk + 1) * 15 + 3],           \
                               &stage_tw[(kk + 2) * 15 + 3],           \
                               &stage_tw[(kk + 3) * 15 + 3]);          \
        __m512d w5 = load4_aos(&stage_tw[(kk) * 15 + 4],               \
                               &stage_tw[(kk + 1) * 15 + 4],           \
                               &stage_tw[(kk + 2) * 15 + 4],           \
                               &stage_tw[(kk + 3) * 15 + 4]);          \
        __m512d w6 = load4_aos(&stage_tw[(kk) * 15 + 5],               \
                               &stage_tw[(kk + 1) * 15 + 5],           \
                               &stage_tw[(kk + 2) * 15 + 5],           \
                               &stage_tw[(kk + 3) * 15 + 5]);          \
        __m512d w7 = load4_aos(&stage_tw[(kk) * 15 + 6],               \
                               &stage_tw[(kk + 1) * 15 + 6],           \
                               &stage_tw[(kk + 2) * 15 + 6],           \
                               &stage_tw[(kk + 3) * 15 + 6]);          \
        __m512d w8 = load4_aos(&stage_tw[(kk) * 15 + 7],               \
                               &stage_tw[(kk + 1) * 15 + 7],           \
                               &stage_tw[(kk + 2) * 15 + 7],           \
                               &stage_tw[(kk + 3) * 15 + 7]);          \
        __m512d w9 = load4_aos(&stage_tw[(kk) * 15 + 8],               \
                               &stage_tw[(kk + 1) * 15 + 8],           \
                               &stage_tw[(kk + 2) * 15 + 8],           \
                               &stage_tw[(kk + 3) * 15 + 8]);          \
        __m512d w10 = load4_aos(&stage_tw[(kk) * 15 + 9],              \
                                &stage_tw[(kk + 1) * 15 + 9],          \
                                &stage_tw[(kk + 2) * 15 + 9],          \
                                &stage_tw[(kk + 3) * 15 + 9]);         \
        __m512d w11 = load4_aos(&stage_tw[(kk) * 15 + 10],             \
                                &stage_tw[(kk + 1) * 15 + 10],         \
                                &stage_tw[(kk + 2) * 15 + 10],         \
                                &stage_tw[(kk + 3) * 15 + 10]);        \
        __m512d w12 = load4_aos(&stage_tw[(kk) * 15 + 11],             \
                                &stage_tw[(kk + 1) * 15 + 11],         \
                                &stage_tw[(kk + 2) * 15 + 11],         \
                                &stage_tw[(kk + 3) * 15 + 11]);        \
        __m512d w13 = load4_aos(&stage_tw[(kk) * 15 + 12],             \
                                &stage_tw[(kk + 1) * 15 + 12],         \
                                &stage_tw[(kk + 2) * 15 + 12],         \
                                &stage_tw[(kk + 3) * 15 + 12]);        \
        __m512d w14 = load4_aos(&stage_tw[(kk) * 15 + 13],             \
                                &stage_tw[(kk + 1) * 15 + 13],         \
                                &stage_tw[(kk + 2) * 15 + 13],         \
                                &stage_tw[(kk + 3) * 15 + 13]);        \
        __m512d w15 = load4_aos(&stage_tw[(kk) * 15 + 14],             \
                                &stage_tw[(kk + 1) * 15 + 14],         \
                                &stage_tw[(kk + 2) * 15 + 14],         \
                                &stage_tw[(kk + 3) * 15 + 14]);        \
                                                                       \
        CMUL_FMA_AOS_AVX512(x[1], x[1], w1);                            \
        CMUL_FMA_AOS_AVX512(x[2], x[2], w2);                            \
        CMUL_FMA_AOS_AVX512(x[3], x[3], w3);                            \
        CMUL_FMA_AOS_AVX512(x[4], x[4], w4);                            \
        CMUL_FMA_AOS_AVX512(x[5], x[5], w5);                            \
        CMUL_FMA_AOS_AVX512(x[6], x[6], w6);                            \
        CMUL_FMA_AOS_AVX512(x[7], x[7], w7);                            \
        CMUL_FMA_AOS_AVX512(x[8], x[8], w8);                            \
        CMUL_FMA_AOS_AVX512(x[9], x[9], w9);                            \
        CMUL_FMA_AOS_AVX512(x[10], x[10], w10);                         \
        CMUL_FMA_AOS_AVX512(x[11], x[11], w11);                         \
        CMUL_FMA_AOS_AVX512(x[12], x[12], w12);                         \
        CMUL_FMA_AOS_AVX512(x[13], x[13], w13);                         \
        CMUL_FMA_AOS_AVX512(x[14], x[14], w14);                         \
        CMUL_FMA_AOS_AVX512(x[15], x[15], w15);                         \
    } while (0)

#define PREFETCH_STAGE_TW_AVX512(kk, distance, stage_tw) \
    _mm_prefetch((const char*)&stage_tw[((kk) + (distance)) * 15], _MM_HINT_T0)

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

#define PREFETCH_L1_AVX512 16

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

#define RADIX16_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask) \
    do                                                                                                \
    {                                                                                                 \
        __m512d x[16];                                                                                \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                  \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                             \
                                                                                                      \
        __m512d y[16];                                                                                \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);           \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);           \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);        \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);      \
                                                                                                      \
        APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask);                                         \
                                                                                                      \
        __m512d temp[4];                                                                              \
        for (int m = 0; m < 4; m++)                                                                   \
        {                                                                                             \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                              \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                    \
            y[m] = temp[0];                                                                           \
            y[m + 4] = temp[1];                                                                       \
            y[m + 8] = temp[2];                                                                       \
            y[m + 12] = temp[3];                                                                      \
        }                                                                                             \
                                                                                                      \
        STORE_16_LANES_AVX512(kk, K, output_buffer, y);                                               \
    } while (0)

#define RADIX16_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask) \
    do                                                                                                \
    {                                                                                                 \
        __m512d x[16];                                                                                \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                  \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                             \
                                                                                                      \
        __m512d y[16];                                                                                \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);           \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);           \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);        \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);      \
                                                                                                      \
        APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask);                                         \
                                                                                                      \
        __m512d temp[4];                                                                              \
        for (int m = 0; m < 4; m++)                                                                   \
        {                                                                                             \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                              \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                    \
            y[m] = temp[0];                                                                           \
            y[m + 4] = temp[1];                                                                       \
            y[m + 8] = temp[2];                                                                       \
            y[m + 12] = temp[3];                                                                      \
        }                                                                                             \
                                                                                                      \
        STORE_16_LANES_AVX512(kk, K, output_buffer, y);                                               \
    } while (0)

#define RADIX16_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask) \
    do                                                                                                       \
    {                                                                                                        \
        __m512d x[16];                                                                                       \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                         \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                                    \
                                                                                                             \
        __m512d y[16];                                                                                       \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                  \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                  \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);               \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);             \
                                                                                                             \
        APPLY_W4_INTERMEDIATE_FV_AVX512_HOISTED(y, neg_mask);                                                \
                                                                                                             \
        __m512d temp[4];                                                                                     \
        for (int m = 0; m < 4; m++)                                                                          \
        {                                                                                                    \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                     \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                           \
            y[m] = temp[0];                                                                                  \
            y[m + 4] = temp[1];                                                                              \
            y[m + 8] = temp[2];                                                                              \
            y[m + 12] = temp[3];                                                                             \
        }                                                                                                    \
                                                                                                             \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, y);                                               \
    } while (0)

#define RADIX16_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, neg_mask) \
    do                                                                                                       \
    {                                                                                                        \
        __m512d x[16];                                                                                       \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                         \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                                    \
                                                                                                             \
        __m512d y[16];                                                                                       \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask);                  \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask);                  \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask);               \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask);             \
                                                                                                             \
        APPLY_W4_INTERMEDIATE_BV_AVX512_HOISTED(y, neg_mask);                                                \
                                                                                                             \
        __m512d temp[4];                                                                                     \
        for (int m = 0; m < 4; m++)                                                                          \
        {                                                                                                    \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                                     \
                                    temp[0], temp[1], temp[2], temp[3], rot_mask);                           \
            y[m] = temp[0];                                                                                  \
            y[m + 4] = temp[1];                                                                              \
            y[m + 8] = temp[2];                                                                              \
            y[m + 12] = temp[3];                                                                             \
        }                                                                                                    \
                                                                                                             \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, y);                                               \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

// Hoisted sign masks for W_4 intermediate twiddles - Forward
static const __m256d W4_SIGN_MASK_FV_5_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_FV_7_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_FV_13_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_FV_15_AVX2 = {-0.0, 0.0, -0.0, 0.0};

// Hoisted sign masks for W_4 intermediate twiddles - Backward
static const __m256d W4_SIGN_MASK_BV_5_AVX2 = {0.0, -0.0, 0.0, -0.0};
static const __m256d W4_SIGN_MASK_BV_7_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_BV_13_AVX2 = {-0.0, 0.0, -0.0, 0.0};
static const __m256d W4_SIGN_MASK_BV_15_AVX2 = {0.0, -0.0, 0.0, -0.0};

#define CMUL_FMA_AOS(out, a, w)                                      \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_unpacklo_pd(a, a);                       \
        __m256d ai = _mm256_unpackhi_pd(a, a);                       \
        __m256d wr = _mm256_unpacklo_pd(w, w);                       \
        __m256d wi = _mm256_unpackhi_pd(w, w);                       \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)

#define RADIX4_BUTTERFLY_AVX2(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do                                                              \
    {                                                               \
        __m256d sumBD = _mm256_add_pd(b, d);                        \
        __m256d difBD = _mm256_sub_pd(b, d);                        \
        __m256d sumAC = _mm256_add_pd(a, c);                        \
        __m256d difAC = _mm256_sub_pd(a, c);                        \
                                                                    \
        y0 = _mm256_add_pd(sumAC, sumBD);                           \
        y2 = _mm256_sub_pd(sumAC, sumBD);                           \
                                                                    \
        __m256d dif_bd_swp = _mm256_permute_pd(difBD, 0b0101);      \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask);   \
                                                                    \
        y1 = _mm256_sub_pd(difAC, dif_bd_rot);                      \
        y3 = _mm256_add_pd(difAC, dif_bd_rot);                      \
    } while (0)

#define APPLY_W4_INTERMEDIATE_FV_AVX2_HOISTED(y, neg_mask)        \
    do                                                            \
    {                                                             \
        {                                                         \
            __m256d y5_swp = _mm256_permute_pd(y[5], 0x5);        \
            y[5] = _mm256_xor_pd(y5_swp, W4_SIGN_MASK_FV_5_AVX2); \
            y[6] = _mm256_xor_pd(y[6], neg_mask);                 \
            __m256d y7_swp = _mm256_permute_pd(y[7], 0x5);        \
            y[7] = _mm256_xor_pd(y7_swp, W4_SIGN_MASK_FV_7_AVX2); \
        }                                                         \
        {                                                         \
            y[9] = _mm256_xor_pd(y[9], neg_mask);                 \
            y[11] = _mm256_xor_pd(y[11], neg_mask);               \
        }                                                         \
        {                                                         \
            __m256d y13_swp = _mm256_permute_pd(y[13], 0x5);      \
            y[13] = _mm256_xor_pd(y13_swp, W4_SIGN_MASK_FV_13_AVX2); \
            y[14] = _mm256_xor_pd(y[14], neg_mask);               \
            __m256d y15_swp = _mm256_permute_pd(y[15], 0x5);      \
            y[15] = _mm256_xor_pd(y15_swp, W4_SIGN_MASK_FV_15_AVX2); \
        }                                                         \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_AVX2_HOISTED(y, neg_mask)        \
    do                                                            \
    {                                                             \
        {                                                         \
            __m256d y5_swp = _mm256_permute_pd(y[5], 0x5);        \
            y[5] = _mm256_xor_pd(y5_swp, W4_SIGN_MASK_BV_5_AVX2); \
            y[6] = _mm256_xor_pd(y[6], neg_mask);                 \
            __m256d y7_swp = _mm256_permute_pd(y[7], 0x5);        \
            y[7] = _mm256_xor_pd(y7_swp, W4_SIGN_MASK_BV_7_AVX2); \
        }                                                         \
        {                                                         \
            y[9] = _mm256_xor_pd(y[9], neg_mask);                 \
            y[11] = _mm256_xor_pd(y[11], neg_mask);               \
        }                                                         \
        {                                                         \
            __m256d y13_swp = _mm256_permute_pd(y[13], 0x5);      \
            y[13] = _mm256_xor_pd(y13_swp, W4_SIGN_MASK_BV_13_AVX2); \
            y[14] = _mm256_xor_pd(y[14], neg_mask);               \
            __m256d y15_swp = _mm256_permute_pd(y[15], 0x5);      \
            y[15] = _mm256_xor_pd(y15_swp, W4_SIGN_MASK_BV_15_AVX2); \
        }                                                         \
    } while (0)

#define APPLY_STAGE_TWIDDLES_R16_AVX2(kk, x, stage_tw)                     \
    do                                                                     \
    {                                                                      \
        __m256d w1 = load2_aos(&stage_tw[(kk) * 15 + 0],                   \
                               &stage_tw[(kk + 1) * 15 + 0]);              \
        __m256d w2 = load2_aos(&stage_tw[(kk) * 15 + 1],                   \
                               &stage_tw[(kk + 1) * 15 + 1]);              \
        __m256d w3 = load2_aos(&stage_tw[(kk) * 15 + 2],                   \
                               &stage_tw[(kk + 1) * 15 + 2]);              \
        __m256d w4 = load2_aos(&stage_tw[(kk) * 15 + 3],                   \
                               &stage_tw[(kk + 1) * 15 + 3]);              \
        __m256d w5 = load2_aos(&stage_tw[(kk) * 15 + 4],                   \
                               &stage_tw[(kk + 1) * 15 + 4]);              \
        __m256d w6 = load2_aos(&stage_tw[(kk) * 15 + 5],                   \
                               &stage_tw[(kk + 1) * 15 + 5]);              \
        __m256d w7 = load2_aos(&stage_tw[(kk) * 15 + 6],                   \
                               &stage_tw[(kk + 1) * 15 + 6]);              \
        __m256d w8 = load2_aos(&stage_tw[(kk) * 15 + 7],                   \
                               &stage_tw[(kk + 1) * 15 + 7]);              \
        __m256d w9 = load2_aos(&stage_tw[(kk) * 15 + 8],                   \
                               &stage_tw[(kk + 1) * 15 + 8]);              \
        __m256d w10 = load2_aos(&stage_tw[(kk) * 15 + 9],                  \
                                &stage_tw[(kk + 1) * 15 + 9]);             \
        __m256d w11 = load2_aos(&stage_tw[(kk) * 15 + 10],                 \
                                &stage_tw[(kk + 1) * 15 + 10]);            \
        __m256d w12 = load2_aos(&stage_tw[(kk) * 15 + 11],                 \
                                &stage_tw[(kk + 1) * 15 + 11]);            \
        __m256d w13 = load2_aos(&stage_tw[(kk) * 15 + 12],                 \
                                &stage_tw[(kk + 1) * 15 + 12]);            \
        __m256d w14 = load2_aos(&stage_tw[(kk) * 15 + 13],                 \
                                &stage_tw[(kk + 1) * 15 + 13]);            \
        __m256d w15 = load2_aos(&stage_tw[(kk) * 15 + 14],                 \
                                &stage_tw[(kk + 1) * 15 + 14]);            \
                                                                           \
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

#define PREFETCH_STAGE_TW_AVX2(kk, distance, stage_tw) \
    _mm_prefetch((const char*)&stage_tw[((kk) + (distance)) * 15], _MM_HINT_T0)

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

#define PREFETCH_L1 8

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

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

#define RADIX4_BUTTERFLY_SCALAR(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                   \
                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, rot_sign) \
    do                                                                                            \
    {                                                                                             \
        double sumBD_re = b_re + d_re;                                                            \
        double sumBD_im = b_im + d_im;                                                            \
        double difBD_re = b_re - d_re;                                                            \
        double difBD_im = b_im - d_im;                                                            \
        double sumAC_re = a_re + c_re;                                                            \
        double sumAC_im = a_im + c_im;                                                            \
        double difAC_re = a_re - c_re;                                                            \
        double difAC_im = a_im - c_im;                                                            \
                                                                                                  \
        y0_re = sumAC_re + sumBD_re;                                                              \
        y0_im = sumAC_im + sumBD_im;                                                              \
        y2_re = sumAC_re - sumBD_re;                                                              \
        y2_im = sumAC_im - sumBD_im;                                                              \
                                                                                                  \
        double rot_re = (rot_sign) * difBD_im;                                                    \
        double rot_im = (rot_sign) * (-difBD_re);                                                 \
                                                                                                  \
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

#define APPLY_STAGE_TWIDDLES_R16_SCALAR(k, x, stage_tw)                \
    do                                                                 \
    {                                                                  \
        const fft_data *w_ptr = &stage_tw[(k) * 15];                   \
        for (int j = 1; j <= 15; j++)                                  \
        {                                                              \
            fft_data a = x[j];                                         \
            x[j].re = a.re * w_ptr[j - 1].re - a.im * w_ptr[j - 1].im; \
            x[j].im = a.re * w_ptr[j - 1].im + a.im * w_ptr[j - 1].re; \
        }                                                              \
    } while (0)

#endif // FFT_RADIX16_MACROS_H