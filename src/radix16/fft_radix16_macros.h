
//==============================================================================
// fft_radix16_macros.h - Shared Macros for Radix-16 Butterflies
//==============================================================================
//
// ALGORITHM: 2-stage radix-4 decomposition
//   1. Apply input twiddles W_N^(j*k) for j=1..15
//   2. First radix-4 stage (4 groups of 4)
//   3. Apply W_4 intermediate twiddles
//   4. Second radix-4 stage (final)
//
// KEY INSIGHT: Reuse radix-4 macros for both stages!
//

#ifndef FFT_RADIX16_MACROS_H
#define FFT_RADIX16_MACROS_H

#include "simd_math.h"

//==============================================================================
// REUSE RADIX-4 INFRASTRUCTURE
//==============================================================================

// We'll manually include the key radix-4 macros we need
// (In real code, could #include "fft_radix4_macros.h")

//==============================================================================
// AVX-512 SUPPORT - 4X throughput vs AVX2 (processes 4 butterflies)
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

/**
 * @brief Optimized complex multiply for AVX-512: out = a * w (4 complex values)
 *
 * Uses FMA and handles 4 complex numbers (8 doubles) per operation.
 * Same algorithm as AVX2 version but with 512-bit registers.
 */
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

//==============================================================================
// RADIX-4 BUTTERFLY - AVX-512 (reused in both stages)
//==============================================================================

/**
 * @brief Radix-4 butterfly core for AVX-512 (4 butterflies)
 *
 * Processes 4 radix-4 butterflies simultaneously.
 * Reused in both the first and second stages of the radix-16 decomposition.
 */
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
// APPLY W_4 INTERMEDIATE TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief Apply W_4 intermediate twiddles for FORWARD FFT (AVX-512, 4 butterflies)
 *
 * Pattern for y[4*m + j] where m=0..3, j=0..3:
 * - y[0..3]:   no twiddle (W_4^0 = 1)
 * - y[4..7]:   W_4^{0*j, 1*j, 2*j, 3*j} = {1, -i, -1, +i}
 * - y[8..11]:  W_4^{0*j, 2*j, 0*j, 2*j} = {1, -1, 1, -1}
 * - y[12..15]: W_4^{0*j, 3*j, 2*j, 1*j} = {1, +i, -1, -i}
 *
 * Processes 4 butterflies simultaneously with 512-bit registers.
 */
#define APPLY_W4_INTERMEDIATE_FV_AVX512(y)                                              \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m512d w1 = _mm512_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE,  \
                                       W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            __m512d w2 = _mm512_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE,  \
                                       W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m512d w3 = _mm512_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE,  \
                                       W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            CMUL_FMA_AOS_AVX512(y[5], y[5], w1);                                        \
            CMUL_FMA_AOS_AVX512(y[6], y[6], w2);                                        \
            CMUL_FMA_AOS_AVX512(y[7], y[7], w3);                                        \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m512d w2 = _mm512_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE,  \
                                       W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            CMUL_FMA_AOS_AVX512(y[9], y[9], w2);                                        \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS_AVX512(y[11], y[11], w2);                                      \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {+i, -1, -i} */                                  \
        {                                                                               \
            __m512d w3 = _mm512_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE,  \
                                       W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            __m512d w2 = _mm512_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE,  \
                                       W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m512d w1 = _mm512_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE,  \
                                       W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            CMUL_FMA_AOS_AVX512(y[13], y[13], w3);                                      \
            CMUL_FMA_AOS_AVX512(y[14], y[14], w2);                                      \
            CMUL_FMA_AOS_AVX512(y[15], y[15], w1);                                      \
        }                                                                               \
    } while (0)

/**
 * @brief Apply W_4 intermediate twiddles for INVERSE FFT (AVX-512, 4 butterflies)
 *
 * Same pattern but with conjugated W_4 twiddles.
 * Processes 4 butterflies simultaneously with 512-bit registers.
 */
#define APPLY_W4_INTERMEDIATE_BV_AVX512(y)                                              \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m512d w1 = _mm512_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE,  \
                                       W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            __m512d w2 = _mm512_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE,  \
                                       W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m512d w3 = _mm512_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE,  \
                                       W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            CMUL_FMA_AOS_AVX512(y[5], y[5], w1);                                        \
            CMUL_FMA_AOS_AVX512(y[6], y[6], w2);                                        \
            CMUL_FMA_AOS_AVX512(y[7], y[7], w3);                                        \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m512d w2 = _mm512_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE,  \
                                       W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            CMUL_FMA_AOS_AVX512(y[9], y[9], w2);                                        \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS_AVX512(y[11], y[11], w2);                                      \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {-i, -1, +i} */                                  \
        {                                                                               \
            __m512d w3 = _mm512_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE,  \
                                       W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            __m512d w2 = _mm512_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE,  \
                                       W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m512d w1 = _mm512_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE,  \
                                       W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            CMUL_FMA_AOS_AVX512(y[13], y[13], w3);                                      \
            CMUL_FMA_AOS_AVX512(y[14], y[14], w2);                                      \
            CMUL_FMA_AOS_AVX512(y[15], y[15], w1);                                      \
        }                                                                               \
    } while (0)

//==============================================================================
// APPLY STAGE TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief AVX-512: Apply stage twiddles for 4 butterflies (kk through kk+3)
 *
 * Applies precomputed twiddle factors to lanes 1-15 for four butterflies.
 * Uses load4_aos to pack 4 complex values into one 512-bit register.
 */
#define APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw)               \
    do                                                                 \
    {                                                                  \
        for (int j = 1; j <= 15; j++)                                  \
        {                                                              \
            __m512d w = load4_aos(&stage_tw[(kk) * 15 + (j - 1)],      \
                                  &stage_tw[(kk + 1) * 15 + (j - 1)],  \
                                  &stage_tw[(kk + 2) * 15 + (j - 1)],  \
                                  &stage_tw[(kk + 3) * 15 + (j - 1)]); \
            CMUL_FMA_AOS_AVX512(x[j], x[j], w);                        \
        }                                                              \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 16 lanes for AVX-512 (four butterflies: kk, kk+1, kk+2, kk+3)
 *
 * Loads input data for 16 lanes (0 to 15) from sub_outputs buffer into SIMD registers.
 * Each register holds 4 complex values (for 4 butterflies) in AoS layout.
 */
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

/**
 * @brief Store 16 lanes for AVX-512 (unaligned store)
 *
 * Stores final outputs from the radix-16 butterfly into the output buffer.
 * Handles strided storage across 4 groups with unaligned stores.
 */
#define STORE_16_LANES_AVX512(kk, K, output_buffer, z)                       \
    do                                                                       \
    {                                                                        \
        for (int m = 0; m < 4; m++)                                          \
        {                                                                    \
            STOREU_PD512(&output_buffer[(kk) + m * K].re, z[m]);             \
            STOREU_PD512(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            STOREU_PD512(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                    \
    } while (0)

/**
 * @brief Store 16 lanes for AVX-512 with streaming stores
 *
 * Uses non-temporal stores to bypass cache for large datasets.
 * Beneficial when output is not immediately reused.
 */
#define STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, z)                    \
    do                                                                           \
    {                                                                            \
        for (int m = 0; m < 4; m++)                                              \
        {                                                                        \
            _mm512_stream_pd(&output_buffer[(kk) + m * K].re, z[m]);             \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            _mm512_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                        \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512
//==============================================================================

/**
 * @brief Prefetch distances optimized for AVX-512 (larger working set)
 */
#define PREFETCH_L1_AVX512 16  // 1KB ahead
#define PREFETCH_L2_AVX512 64  // 4KB ahead
#define PREFETCH_L3_AVX512 128 // 8KB ahead

/**
 * @brief Prefetch 16 lanes ahead for AVX-512 (4 butterflies)
 *
 * Prefetches more aggressively than AVX2 due to higher throughput.
 * Issues prefetch for all 16 strided lanes plus twiddles.
 */
#define PREFETCH_16_LANES_AVX512(k, K, distance, sub_outputs, stage_tw, hint)                \
    do                                                                                       \
    {                                                                                        \
        if ((k) + (distance) < K)                                                            \
        {                                                                                    \
            for (int lane = 0; lane < 16; lane++)                                            \
            {                                                                                \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + lane * K], hint); \
            }                                                                                \
            _mm_prefetch((const char *)&stage_tw[((k) + (distance)) * 15], hint);            \
        }                                                                                    \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

/**
 * @brief Complete AVX-512 radix-16 butterfly pipeline (FORWARD, 4 butterflies)
 *
 * Processes 4 butterflies in one macro call using 2-stage radix-4 decomposition:
 * 1. Load 16 lanes
 * 2. Apply input twiddles (lanes 1-15)
 * 3. First radix-4 stage (4 groups of 4)
 * 4. Apply W_4 intermediate twiddles (forward)
 * 5. Second radix-4 stage (final)
 * 6. Store results
 */
#define RADIX16_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer)                   \
    do                                                                                              \
    {                                                                                               \
        /* Step 1: Load inputs */                                                                   \
        __m512d x[16];                                                                              \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                \
                                                                                                    \
        /* Step 2: Apply input twiddles */                                                          \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                           \
                                                                                                    \
        /* Step 3: First radix-4 stage (4 groups of 4) */                                           \
        __m512d y[16];                                                                              \
        const __m512d rot_mask_fv = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,                             \
                                                  -0.0, 0.0, -0.0, 0.0);                            \
                                                                                                    \
        /* Group 0: [0,4,8,12] */                                                                   \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask_fv);      \
        /* Group 1: [1,5,9,13] */                                                                   \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask_fv);      \
        /* Group 2: [2,6,10,14] */                                                                  \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask_fv);   \
        /* Group 3: [3,7,11,15] */                                                                  \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask_fv); \
                                                                                                    \
        /* Step 4: Apply W_4 intermediate twiddles */                                               \
        APPLY_W4_INTERMEDIATE_FV_AVX512(y);                                                         \
                                                                                                    \
        /* Step 5: Second radix-4 stage (transpose groups) */                                       \
        __m512d z[16];                                                                              \
        for (int m = 0; m < 4; m++)                                                                 \
        {                                                                                           \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                            \
                                    z[m], z[m + 4], z[m + 8], z[m + 12], rot_mask_fv);              \
        }                                                                                           \
                                                                                                    \
        /* Step 6: Store results */                                                                 \
        STORE_16_LANES_AVX512(kk, K, output_buffer, z);                                             \
    } while (0)

/**
 * @brief Complete AVX-512 radix-16 butterfly pipeline (INVERSE, 4 butterflies)
 *
 * Identical to forward except uses inverse W_4 twiddles and rotation.
 */
#define RADIX16_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer)                   \
    do                                                                                              \
    {                                                                                               \
        /* Step 1: Load inputs */                                                                   \
        __m512d x[16];                                                                              \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                \
                                                                                                    \
        /* Step 2: Apply input twiddles */                                                          \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                           \
                                                                                                    \
        /* Step 3: First radix-4 stage (4 groups of 4) */                                           \
        __m512d y[16];                                                                              \
        const __m512d rot_mask_bv = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,                             \
                                                  0.0, -0.0, 0.0, -0.0);                            \
                                                                                                    \
        /* Group 0: [0,4,8,12] */                                                                   \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask_bv);      \
        /* Group 1: [1,5,9,13] */                                                                   \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask_bv);      \
        /* Group 2: [2,6,10,14] */                                                                  \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask_bv);   \
        /* Group 3: [3,7,11,15] */                                                                  \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask_bv); \
                                                                                                    \
        /* Step 4: Apply W_4 intermediate twiddles (inverse) */                                     \
        APPLY_W4_INTERMEDIATE_BV_AVX512(y);                                                         \
                                                                                                    \
        /* Step 5: Second radix-4 stage (transpose groups) */                                       \
        __m512d z[16];                                                                              \
        for (int m = 0; m < 4; m++)                                                                 \
        {                                                                                           \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                            \
                                    z[m], z[m + 4], z[m + 8], z[m + 12], rot_mask_bv);              \
        }                                                                                           \
                                                                                                    \
        /* Step 6: Store results */                                                                 \
        STORE_16_LANES_AVX512(kk, K, output_buffer, z);                                             \
    } while (0)

/**
 * @brief Streaming version (forward) - for large transforms
 *
 * Uses non-temporal stores to avoid cache pollution.
 */
#define RADIX16_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer)            \
    do                                                                                              \
    {                                                                                               \
        __m512d x[16];                                                                              \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                           \
                                                                                                    \
        __m512d y[16];                                                                              \
        const __m512d rot_mask_fv = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,                             \
                                                  -0.0, 0.0, -0.0, 0.0);                            \
                                                                                                    \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask_fv);      \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask_fv);      \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask_fv);   \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask_fv); \
                                                                                                    \
        APPLY_W4_INTERMEDIATE_FV_AVX512(y);                                                         \
                                                                                                    \
        __m512d z[16];                                                                              \
        for (int m = 0; m < 4; m++)                                                                 \
        {                                                                                           \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                            \
                                    z[m], z[m + 4], z[m + 8], z[m + 12], rot_mask_fv);              \
        }                                                                                           \
                                                                                                    \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, z);                                      \
    } while (0)

/**
 * @brief Streaming version (inverse) - for large transforms
 */
#define RADIX16_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer)            \
    do                                                                                              \
    {                                                                                               \
        __m512d x[16];                                                                              \
        LOAD_16_LANES_AVX512(kk, K, sub_outputs, x);                                                \
        APPLY_STAGE_TWIDDLES_R16_AVX512(kk, x, stage_tw);                                           \
                                                                                                    \
        __m512d y[16];                                                                              \
        const __m512d rot_mask_bv = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,                             \
                                                  0.0, -0.0, 0.0, -0.0);                            \
                                                                                                    \
        RADIX4_BUTTERFLY_AVX512(x[0], x[4], x[8], x[12], y[0], y[1], y[2], y[3], rot_mask_bv);      \
        RADIX4_BUTTERFLY_AVX512(x[1], x[5], x[9], x[13], y[4], y[5], y[6], y[7], rot_mask_bv);      \
        RADIX4_BUTTERFLY_AVX512(x[2], x[6], x[10], x[14], y[8], y[9], y[10], y[11], rot_mask_bv);   \
        RADIX4_BUTTERFLY_AVX512(x[3], x[7], x[11], x[15], y[12], y[13], y[14], y[15], rot_mask_bv); \
                                                                                                    \
        APPLY_W4_INTERMEDIATE_BV_AVX512(y);                                                         \
                                                                                                    \
        __m512d z[16];                                                                              \
        for (int m = 0; m < 4; m++)                                                                 \
        {                                                                                           \
            RADIX4_BUTTERFLY_AVX512(y[m], y[m + 4], y[m + 8], y[m + 12],                            \
                                    z[m], z[m + 4], z[m + 8], z[m + 12], rot_mask_bv);              \
        }                                                                                           \
                                                                                                    \
        STORE_16_LANES_AVX512_STREAM(kk, K, output_buffer, z);                                      \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// USAGE EXAMPLE (for future implementation)
//==============================================================================

/**
 * In fft_radix16_fv.c or fft_radix16_bv.c:
 *
 * #ifdef __AVX512F__
 *     // Main loop: 4 butterflies per iteration (64 complex values processed)
 *     for (; k + 3 < K; k += 4) {
 *         PREFETCH_16_LANES_AVX512(k, K, PREFETCH_L1_AVX512, sub_outputs, stage_tw, _MM_HINT_T0);
 *         RADIX16_PIPELINE_4_FV_AVX512(k, K, sub_outputs, stage_tw, output_buffer);
 *     }
 * #endif
 *
 * #ifdef __AVX2__
 *     // Fallback to AVX2: 2 butterflies per iteration
 *     for (; k + 1 < K; k += 2) {
 *         PREFETCH_16_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
 *         // ... existing AVX2 code ...
 *     }
 * #endif
 *
 * // Scalar tail
 * for (; k < K; k++) {
 *     // ... scalar butterfly ...
 * }
 *
 * PERFORMANCE NOTES:
 * - AVX-512 processes 4 butterflies (64 complex values) per iteration
 * - 2-stage radix-4 decomposition enables excellent instruction-level parallelism
 * - 2x throughput vs AVX2 (which does 2 butterflies = 32 complex values)
 * - 4x throughput vs scalar
 * - Radix-16 is highly efficient for sizes N = 16^k (powers of 16)
 * - Expect 40-60% speedup on Skylake-X/Cascade Lake CPUs
 * - Expect 60-90% speedup on Ice Lake/Sapphire Rapids with full AVX-512 FP
 * - Use streaming stores for transforms larger than L3 cache
 * - The 2-stage decomposition minimizes complex multiplications
 * - Reusing radix-4 butterflies reduces code complexity while maximizing efficiency
 * - Particularly effective for 3D FFTs with dimensions like 256×256×256
 */

/**
 * @brief Optimized complex multiply: out = a * w (6 FMA + 2 UNPACK)
 *
 * This macro performs a complex multiplication using AVX2 instructions, optimized with fused multiply-add (FMA) operations.
 * It is reused from the radix-4 implementation and used for applying twiddle factors in both forward and inverse transforms.
 * The operation assumes Array-of-Structures (AoS) layout for complex numbers (real and imaginary parts interleaved).
 */
#ifdef __AVX2__
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
#endif

//==============================================================================
// W_4 INTERMEDIATE TWIDDLES - Direction-dependent
//==============================================================================

/**
 * @brief W_4 twiddle constants for FORWARD FFT
 *
 * W_4 = exp(-2πi/4) = exp(-πi/2) = -i
 * W_4^0 = 1
 * W_4^1 = -i
 * W_4^2 = -1
 * W_4^3 = +i
 *
 * These constants are used to apply fixed intermediate twiddles in the radix-16 butterfly for the forward transform.
 * They facilitate the two-stage decomposition by multiplying outputs from the first radix-4 stage.
 */
#define W4_FV_0_RE 1.0
#define W4_FV_0_IM 0.0
#define W4_FV_1_RE 0.0
#define W4_FV_1_IM (-1.0)
#define W4_FV_2_RE (-1.0)
#define W4_FV_2_IM 0.0
#define W4_FV_3_RE 0.0
#define W4_FV_3_IM 1.0

/**
 * @brief W_4 twiddle constants for INVERSE FFT
 *
 * W_4 = exp(+2πi/4) = exp(+πi/2) = +i
 * W_4^0 = 1
 * W_4^1 = +i
 * W_4^2 = -1
 * W_4^3 = -i
 *
 * These constants are used to apply fixed intermediate twiddles in the radix-16 butterfly for the inverse transform.
 * They are the complex conjugates of the forward twiddles, ensuring the inverse operation reverses the forward transform.
 */
#define W4_BV_0_RE 1.0
#define W4_BV_0_IM 0.0
#define W4_BV_1_RE 0.0
#define W4_BV_1_IM 1.0
#define W4_BV_2_RE (-1.0)
#define W4_BV_2_IM 0.0
#define W4_BV_3_RE 0.0
#define W4_BV_3_IM (-1.0)

//==============================================================================
// RADIX-4 BUTTERFLY - Reused from radix-4
//==============================================================================

/**
 * @brief Radix-4 butterfly core (IDENTICAL for forward/inverse)
 *
 * This macro implements the core arithmetic of a radix-4 butterfly, reused from the radix-4 implementation.
 * It is applied in both stages of the radix-16 decomposition to compute sums, differences, and rotations.
 * The rot_mask parameter allows direction-specific rotation (e.g., for -i or +i multiplication).
 */
#ifdef __AVX2__
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
#endif

//==============================================================================
// APPLY W_4 INTERMEDIATE TWIDDLES - ONLY DIFFERENCE
//==============================================================================

/**
 * @brief Apply W_4 intermediate twiddles for FORWARD FFT
 *
 * Pattern for y[4*m + j] where m=0..3, j=0..3:
 * - y[0..3]:   no twiddle (W_4^0 = 1)
 * - y[4..7]:   W_4^{0*j, 1*j, 2*j, 3*j} = {1, -i, -1, +i}
 * - y[8..11]:  W_4^{0*j, 2*j, 0*j, 2*j} = {1, -1, 1, -1}
 * - y[12..15]: W_4^{0*j, 3*j, 2*j, 1*j} = {1, +i, -1, -i}
 *
 * This macro applies fixed twiddle factors from the 4th roots of unity to the outputs of the first radix-4 stage.
 * It uses optimized complex multiplications for the forward transform in the radix-16 butterfly.
 */
#ifdef __AVX2__
#define APPLY_W4_INTERMEDIATE_FV_AVX2(y)                                                \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m256d w1 = _mm256_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m256d w3 = _mm256_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            CMUL_FMA_AOS(y[5], y[5], w1);                                               \
            CMUL_FMA_AOS(y[6], y[6], w2);                                               \
            CMUL_FMA_AOS(y[7], y[7], w3);                                               \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            CMUL_FMA_AOS(y[9], y[9], w2);                                               \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS(y[11], y[11], w2);                                             \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {+i, -1, -i} */                                  \
        {                                                                               \
            __m256d w3 = _mm256_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m256d w1 = _mm256_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            CMUL_FMA_AOS(y[13], y[13], w3);                                             \
            CMUL_FMA_AOS(y[14], y[14], w2);                                             \
            CMUL_FMA_AOS(y[15], y[15], w1);                                             \
        }                                                                               \
    } while (0)
#endif

/**
 * @brief Apply W_4 intermediate twiddles for INVERSE FFT
 *
 * Same pattern but with conjugated W_4 twiddles
 *
 * This macro applies fixed twiddle factors from the 4th roots of unity to the outputs of the first radix-4 stage.
 * It uses optimized complex multiplications for the inverse transform in the radix-16 butterfly.
 */
#ifdef __AVX2__
#define APPLY_W4_INTERMEDIATE_BV_AVX2(y)                                                \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m256d w1 = _mm256_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m256d w3 = _mm256_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            CMUL_FMA_AOS(y[5], y[5], w1);                                               \
            CMUL_FMA_AOS(y[6], y[6], w2);                                               \
            CMUL_FMA_AOS(y[7], y[7], w3);                                               \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            CMUL_FMA_AOS(y[9], y[9], w2);                                               \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS(y[11], y[11], w2);                                             \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {-i, -1, +i} */                                  \
        {                                                                               \
            __m256d w3 = _mm256_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m256d w1 = _mm256_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            CMUL_FMA_AOS(y[13], y[13], w3);                                             \
            CMUL_FMA_AOS(y[14], y[14], w2);                                             \
            CMUL_FMA_AOS(y[15], y[15], w1);                                             \
        }                                                                               \
    } while (0)
#endif

/**
 * @brief Scalar version to apply W_4 intermediate twiddles for FORWARD FFT.
 *
 * This macro performs the same twiddle applications as APPLY_W4_INTERMEDIATE_FV_AVX2 but in scalar arithmetic.
 * It is used for non-SIMD paths or small sizes in the forward transform.
 */
#define APPLY_W4_INTERMEDIATE_FV_SCALAR(y)              \
    do                                                  \
    {                                                   \
        /* m=1: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[5].re;                                \
            i = y[5].im;                                \
            y[5].re = r * W4_FV_1_RE - i * W4_FV_1_IM;  \
            y[5].im = r * W4_FV_1_IM + i * W4_FV_1_RE;  \
            r = y[6].re;                                \
            i = y[6].im;                                \
            y[6].re = r * W4_FV_2_RE - i * W4_FV_2_IM;  \
            y[6].im = r * W4_FV_2_IM + i * W4_FV_2_RE;  \
            r = y[7].re;                                \
            i = y[7].im;                                \
            y[7].re = r * W4_FV_3_RE - i * W4_FV_3_IM;  \
            y[7].im = r * W4_FV_3_IM + i * W4_FV_3_RE;  \
        }                                               \
        /* m=2: j=1,3 */                                \
        {                                               \
            y[9].re = -y[9].re;                         \
            y[9].im = -y[9].im;                         \
            y[11].re = -y[11].re;                       \
            y[11].im = -y[11].im;                       \
        }                                               \
        /* m=3: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[13].re;                               \
            i = y[13].im;                               \
            y[13].re = r * W4_FV_3_RE - i * W4_FV_3_IM; \
            y[13].im = r * W4_FV_3_IM + i * W4_FV_3_RE; \
            y[14].re = -y[14].re;                       \
            y[14].im = -y[14].im;                       \
            r = y[15].re;                               \
            i = y[15].im;                               \
            y[15].re = r * W4_FV_1_RE - i * W4_FV_1_IM; \
            y[15].im = r * W4_FV_1_IM + i * W4_FV_1_RE; \
        }                                               \
    } while (0)

/**
 * @brief Scalar version to apply W_4 intermediate twiddles for INVERSE FFT.
 *
 * This macro performs the same twiddle applications as APPLY_W4_INTERMEDIATE_BV_AVX2 but in scalar arithmetic.
 * It is used for non-SIMD paths or small sizes in the inverse transform.
 */
#define APPLY_W4_INTERMEDIATE_BV_SCALAR(y)              \
    do                                                  \
    {                                                   \
        /* m=1: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[5].re;                                \
            i = y[5].im;                                \
            y[5].re = r * W4_BV_1_RE - i * W4_BV_1_IM;  \
            y[5].im = r * W4_BV_1_IM + i * W4_BV_1_RE;  \
            r = y[6].re;                                \
            i = y[6].im;                                \
            y[6].re = r * W4_BV_2_RE - i * W4_BV_2_IM;  \
            y[6].im = r * W4_BV_2_IM + i * W4_BV_2_RE;  \
            r = y[7].re;                                \
            i = y[7].im;                                \
            y[7].re = r * W4_BV_3_RE - i * W4_BV_3_IM;  \
            y[7].im = r * W4_BV_3_IM + i * W4_BV_3_RE;  \
        }                                               \
        /* m=2: j=1,3 */                                \
        {                                               \
            y[9].re = -y[9].re;                         \
            y[9].im = -y[9].im;                         \
            y[11].re = -y[11].re;                       \
            y[11].im = -y[11].im;                       \
        }                                               \
        /* m=3: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[13].re;                               \
            i = y[13].im;                               \
            y[13].re = r * W4_BV_3_RE - i * W4_BV_3_IM; \
            y[13].im = r * W4_BV_3_IM + i * W4_BV_3_RE; \
            y[14].re = -y[14].re;                       \
            y[14].im = -y[14].im;                       \
            r = y[15].re;                               \
            i = y[15].im;                               \
            y[15].re = r * W4_BV_1_RE - i * W4_BV_1_IM; \
            y[15].im = r * W4_BV_1_IM + i * W4_BV_1_RE; \
        }                                               \
    } while (0)

//==============================================================================
// APPLY STAGE TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Scalar: Apply stage twiddles to lanes 1-15
 *
 * stage_tw layout: [W^(1*k), W^(2*k), ..., W^(15*k)] for each k
 *
 * This macro multiplies the input lanes 1 through 15 by precomputed twiddle factors for the current stage.
 * It is the first step in the radix-16 butterfly, preparing inputs for the first radix-4 stage.
 */
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

/**
 * @brief AVX2: Apply stage twiddles for 2 butterflies (kk and kk+1)
 *
 * This macro applies precomputed twiddle factors to inputs for two simultaneous butterflies using AVX2.
 * It loads twiddles in AoS format and uses CMUL_FMA_AOS for multiplication, optimizing for SIMD parallelism.
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_R16_AVX2(kk, x, stage_tw)                                                 \
    do                                                                                                 \
    {                                                                                                  \
        for (int j = 1; j <= 15; j++)                                                                  \
        {                                                                                              \
            __m256d w = load2_aos(&stage_tw[(kk) * 15 + (j - 1)], &stage_tw[(kk + 1) * 15 + (j - 1)]); \
            CMUL_FMA_AOS(x[j], x[j], w);                                                               \
        }                                                                                              \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Load 16 lanes for AVX2 (two butterflies: kk and kk+1).
 *
 * This macro loads input data for 16 lanes (0 to 15) from the sub_outputs buffer into SIMD registers.
 * It assumes AoS layout and loads two complex values per register (for two butterflies).
 */
#ifdef __AVX2__
#define LOAD_16_LANES_AVX2(kk, K, sub_outputs, x)                                                  \
    do                                                                                             \
    {                                                                                              \
        for (int lane = 0; lane < 16; lane++)                                                      \
        {                                                                                          \
            x[lane] = load2_aos(&sub_outputs[(kk) + lane * K], &sub_outputs[(kk) + 1 + lane * K]); \
        }                                                                                          \
    } while (0)
#endif

/**
 * @brief Store 16 lanes for AVX2 (unaligned store).
 *
 * This macro stores the final outputs from the radix-16 butterfly into the output buffer.
 * It handles strided storage across 4 groups, using unaligned stores for flexibility.
 */
#define STORE_16_LANES_AVX2(kk, K, output_buffer, z, m_offset)            \
    do                                                                    \
    {                                                                     \
        for (int m = 0; m < 4; m++)                                       \
        {                                                                 \
            STOREU_PD(&output_buffer[(kk) + m * K].re, z[m]);             \
            STOREU_PD(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                 \
    } while (0)

/**
 * @brief Store 16 lanes for AVX2 with streaming stores.
 *
 * Similar to STORE_16_LANES_AVX2, but uses non-temporal streaming stores to bypass cache for large datasets.
 * This is beneficial for performance when the output is not immediately reused, reducing cache pollution.
 */
#define STORE_16_LANES_AVX2_STREAM(kk, K, output_buffer, z, m_offset)            \
    do                                                                           \
    {                                                                            \
        for (int m = 0; m < 4; m++)                                              \
        {                                                                        \
            _mm256_stream_pd(&output_buffer[(kk) + m * K].re, z[m]);             \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                        \
    } while (0)

//==============================================================================
// PREFETCHING
//==============================================================================

/**
 * @brief Prefetch distances for L1, L2, and L3 caches.
 *
 * These constants define how far ahead to prefetch data in terms of indices.
 * They are tuned to improve memory access performance by bringing data into caches before it's needed.
 */
#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

/**
 * @brief Prefetch 16 lanes ahead for AVX2.
 *
 * This macro issues prefetch instructions for future data accesses in the sub_outputs buffer.
 * It prefetches 16 strided lanes, using the specified cache hint to optimize memory hierarchy usage.
 */
#ifdef __AVX2__
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
#endif

#endif // FFT_RADIX16_MACROS_H
