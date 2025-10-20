//==============================================================================
// fft_radix8_macros.h - Optimized Shared Macros for Radix-8 Butterflies
//==============================================================================
//
// ALGORITHM: Split-radix 2×(4,4) decomposition with fused operations
//   1. Apply input twiddles W_N^(j*k) to lanes 1-7
//   2. Two parallel radix-4 butterflies (even [0,2,4,6], odd [1,3,5,7])
//   3. Apply W_8 geometric twiddles to odd outputs (FUSED with radix-4)
//   4. Final radix-2 combination
//
// OPTIMIZATIONS:
//   - Hoisted W_8 constants outside loops
//   - Fused radix-4 + W_8 twiddle application
//   - Optimized W_8^2 = (0, ±1) handling
//   - Single-level prefetching (no cache pollution)
//   - Full AVX-512 support (4 butterflies/iteration)
//   - Alignment hints for better codegen
//

#ifndef FFT_RADIX8_MACROS_H
#define FFT_RADIX8_MACROS_H

#include "simd_math.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @brief Streaming store threshold
 *
 * For K >= STREAM_THRESHOLD, use non-temporal stores to avoid cache pollution.
 * Tuned for modern CPUs with large L3 caches.
 */
#define STREAM_THRESHOLD 2048

//==============================================================================
// W_8 CONSTANTS - Direction-dependent
//==============================================================================

/**
 * @brief High-precision constant for sqrt(2)/2
 */
#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

/**
 * @brief W_8 twiddle constants for FORWARD FFT
 *
 * W_8 = exp(-2πi/8) = exp(-πi/4)
 * W_8^1 = (√2/2, -√2/2)
 * W_8^2 = (0, -1)
 * W_8^3 = (-√2/2, -√2/2)
 */
#define W8_FV_1_RE C8_CONSTANT
#define W8_FV_1_IM (-C8_CONSTANT)
#define W8_FV_2_RE 0.0
#define W8_FV_2_IM (-1.0)
#define W8_FV_3_RE (-C8_CONSTANT)
#define W8_FV_3_IM (-C8_CONSTANT)

/**
 * @brief W_8 twiddle constants for INVERSE FFT
 *
 * W_8 = exp(+2πi/8) = exp(+πi/4)
 * W_8^1 = (√2/2, +√2/2)
 * W_8^2 = (0, +1)
 * W_8^3 = (-√2/2, +√2/2)
 */
#define W8_BV_1_RE C8_CONSTANT
#define W8_BV_1_IM C8_CONSTANT
#define W8_BV_2_RE 0.0
#define W8_BV_2_IM 1.0
#define W8_BV_3_RE (-C8_CONSTANT)
#define W8_BV_3_IM C8_CONSTANT

//==============================================================================
// AVX-512 SUPPORT - 4X throughput vs AVX2
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

/**
 * @brief Optimized complex multiply for AVX-512: out = a * w (4 complex values)
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
// RADIX-4 SUB-BUTTERFLY - AVX-512
//==============================================================================

/**
 * @brief Radix-4 butterfly core for AVX-512 (4 butterflies)
 */
#define RADIX4_CORE_AVX512(a, b, c, d, y0, y1, y2, y3, rot_mask)    \
    do                                                              \
    {                                                               \
        __m512d sum_bd = _mm512_add_pd(b, d);                       \
        __m512d dif_bd = _mm512_sub_pd(b, d);                       \
        __m512d sum_ac = _mm512_add_pd(a, c);                       \
        __m512d dif_ac = _mm512_sub_pd(a, c);                       \
                                                                    \
        y0 = _mm512_add_pd(sum_ac, sum_bd);                         \
        y2 = _mm512_sub_pd(sum_ac, sum_bd);                         \
                                                                    \
        __m512d dif_bd_swp = _mm512_permute_pd(dif_bd, 0b01010101); \
        __m512d dif_bd_rot = _mm512_xor_pd(dif_bd_swp, rot_mask);   \
                                                                    \
        y1 = _mm512_sub_pd(dif_ac, dif_bd_rot);                     \
        y3 = _mm512_add_pd(dif_ac, dif_bd_rot);                     \
    } while (0)

//==============================================================================
// FUSED RADIX-4 + W_8 TWIDDLE APPLICATION - AVX-512 (NEW!)
//==============================================================================

/**
 * @brief Fused odd radix-4 butterfly with W_8 twiddle application (FORWARD)
 *
 * Combines radix-4 butterfly on odd lanes [1,3,5,7] with immediate W_8
 * twiddle multiplication. Reduces register pressure and improves scheduling.
 *
 * @param x1,x3,x5,x7 Odd input lanes
 * @param o0,o1,o2,o3 Output registers
 * @param rot_mask Forward rotation mask (-i)
 * @param vw81_re,vw81_im W_8^1 constants (hoisted)
 * @param vw83_re,vw83_im W_8^3 constants (hoisted)
 */
#define RADIX4_ODD_WITH_W8_FV_AVX512(x1, x3, x5, x7, o0, o1, o2, o3, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                             \
    {                                                                                                              \
        /* Standard radix-4 arithmetic */                                                                          \
        __m512d sum_bd = _mm512_add_pd(x3, x7);                                                                    \
        __m512d dif_bd = _mm512_sub_pd(x3, x7);                                                                    \
        __m512d sum_ac = _mm512_add_pd(x1, x5);                                                                    \
        __m512d dif_ac = _mm512_sub_pd(x1, x5);                                                                    \
                                                                                                                   \
        o0 = _mm512_add_pd(sum_ac, sum_bd); /* No twiddle on o[0] */                                               \
        __m512d o2_pre = _mm512_sub_pd(sum_ac, sum_bd);                                                            \
                                                                                                                   \
        __m512d dif_bd_swp = _mm512_permute_pd(dif_bd, 0b01010101);                                                \
        __m512d dif_bd_rot = _mm512_xor_pd(dif_bd_swp, rot_mask);                                                  \
                                                                                                                   \
        __m512d o1_pre = _mm512_sub_pd(dif_ac, dif_bd_rot);                                                        \
        __m512d o3_pre = _mm512_add_pd(dif_ac, dif_bd_rot);                                                        \
                                                                                                                   \
        /* Fuse W_8 twiddles immediately */                                                                        \
        /* o1 *= W_8^1 = (√2/2, -√2/2) */                                                                          \
        {                                                                                                          \
            __m512d o1_re = _mm512_movedup_pd(o1_pre);                                                             \
            __m512d o1_im = _mm512_permute_pd(o1_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o1_re, vw81_re, _mm512_mul_pd(o1_im, vw81_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o1_re, vw81_im, _mm512_mul_pd(o1_im, vw81_re));                       \
            o1 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
                                                                                                                   \
        /* o2 *= W_8^2 = (0, -1) - just swap and sign flip */                                                      \
        o2 = _mm512_permute_pd(o2_pre, 0x55);                                                                      \
        o2 = _mm512_xor_pd(o2, _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0));                         \
                                                                                                                   \
        /* o3 *= W_8^3 = (-√2/2, -√2/2) */                                                                         \
        {                                                                                                          \
            __m512d o3_re = _mm512_movedup_pd(o3_pre);                                                             \
            __m512d o3_im = _mm512_permute_pd(o3_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o3_re, vw83_re, _mm512_mul_pd(o3_im, vw83_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o3_re, vw83_im, _mm512_mul_pd(o3_im, vw83_re));                       \
            o3 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
    } while (0)

/**
 * @brief Fused odd radix-4 butterfly with W_8 twiddle application (INVERSE)
 */
#define RADIX4_ODD_WITH_W8_BV_AVX512(x1, x3, x5, x7, o0, o1, o2, o3, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                             \
    {                                                                                                              \
        /* Standard radix-4 arithmetic */                                                                          \
        __m512d sum_bd = _mm512_add_pd(x3, x7);                                                                    \
        __m512d dif_bd = _mm512_sub_pd(x3, x7);                                                                    \
        __m512d sum_ac = _mm512_add_pd(x1, x5);                                                                    \
        __m512d dif_ac = _mm512_sub_pd(x1, x5);                                                                    \
                                                                                                                   \
        o0 = _mm512_add_pd(sum_ac, sum_bd);                                                                        \
        __m512d o2_pre = _mm512_sub_pd(sum_ac, sum_bd);                                                            \
                                                                                                                   \
        __m512d dif_bd_swp = _mm512_permute_pd(dif_bd, 0b01010101);                                                \
        __m512d dif_bd_rot = _mm512_xor_pd(dif_bd_swp, rot_mask);                                                  \
                                                                                                                   \
        __m512d o1_pre = _mm512_sub_pd(dif_ac, dif_bd_rot);                                                        \
        __m512d o3_pre = _mm512_add_pd(dif_ac, dif_bd_rot);                                                        \
                                                                                                                   \
        /* Fuse W_8 twiddles (inverse sign) */                                                                     \
        /* o1 *= W_8^1 = (√2/2, +√2/2) */                                                                          \
        {                                                                                                          \
            __m512d o1_re = _mm512_movedup_pd(o1_pre);                                                             \
            __m512d o1_im = _mm512_permute_pd(o1_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o1_re, vw81_re, _mm512_mul_pd(o1_im, vw81_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o1_re, vw81_im, _mm512_mul_pd(o1_im, vw81_re));                       \
            o1 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
                                                                                                                   \
        /* o2 *= W_8^2 = (0, +1) */                                                                                \
        o2 = _mm512_permute_pd(o2_pre, 0x55);                                                                      \
        o2 = _mm512_xor_pd(o2, _mm512_set_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0));                         \
                                                                                                                   \
        /* o3 *= W_8^3 = (-√2/2, +√2/2) */                                                                         \
        {                                                                                                          \
            __m512d o3_re = _mm512_movedup_pd(o3_pre);                                                             \
            __m512d o3_im = _mm512_permute_pd(o3_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o3_re, vw83_re, _mm512_mul_pd(o3_im, vw83_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o3_re, vw83_im, _mm512_mul_pd(o3_im, vw83_re));                       \
            o3 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
    } while (0)

//==============================================================================
// FINAL RADIX-2 COMBINATION - AVX-512
//==============================================================================

/**
 * @brief Final radix-2 combination with normal stores
 */
#define FINAL_RADIX2_AVX512(e, o, output_buffer, k, K)             \
    do                                                             \
    {                                                              \
        for (int m = 0; m < 4; m++)                                \
        {                                                          \
            __m512d sum = _mm512_add_pd(e[m], o[m]);               \
            __m512d dif = _mm512_sub_pd(e[m], o[m]);               \
            STOREU_PD512(&output_buffer[k + m * K].re, sum);       \
            STOREU_PD512(&output_buffer[k + (m + 4) * K].re, dif); \
        }                                                          \
    } while (0)

/**
 * @brief Final radix-2 combination with streaming stores
 */
#define FINAL_RADIX2_AVX512_STREAM(e, o, output_buffer, k, K)          \
    do                                                                 \
    {                                                                  \
        for (int m = 0; m < 4; m++)                                    \
        {                                                              \
            __m512d sum = _mm512_add_pd(e[m], o[m]);                   \
            __m512d dif = _mm512_sub_pd(e[m], o[m]);                   \
            _mm512_stream_pd(&output_buffer[k + m * K].re, sum);       \
            _mm512_stream_pd(&output_buffer[k + (m + 4) * K].re, dif); \
        }                                                              \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief Apply stage twiddles for 4 butterflies
 */
#define APPLY_STAGE_TWIDDLES_AVX512(kk, x, stage_tw)                  \
    do                                                                \
    {                                                                 \
        for (int j = 1; j <= 7; j++)                                  \
        {                                                             \
            __m512d w = load4_aos(&stage_tw[(kk) * 7 + (j - 1)],      \
                                  &stage_tw[(kk + 1) * 7 + (j - 1)],  \
                                  &stage_tw[(kk + 2) * 7 + (j - 1)],  \
                                  &stage_tw[(kk + 3) * 7 + (j - 1)]); \
            CMUL_FMA_AOS_AVX512(x[j], x[j], w);                       \
        }                                                             \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 8 lanes for 4 butterflies
 */
#define LOAD_8_LANES_AVX512(kk, K, sub_outputs, x)            \
    do                                                        \
    {                                                         \
        x[0] = load4_aos(&sub_outputs[kk],                    \
                         &sub_outputs[(kk) + 1],              \
                         &sub_outputs[(kk) + 2],              \
                         &sub_outputs[(kk) + 3]);             \
        for (int j = 1; j <= 7; j++)                          \
        {                                                     \
            x[j] = load4_aos(&sub_outputs[(kk) + j * K],      \
                             &sub_outputs[(kk) + 1 + j * K],  \
                             &sub_outputs[(kk) + 2 + j * K],  \
                             &sub_outputs[(kk) + 3 + j * K]); \
        }                                                     \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512 (Single level, optimized)
//==============================================================================

/**
 * @brief Prefetch distance for AVX-512 L1 cache
 */
#define PREFETCH_L1_AVX512 16

/**
 * @brief Optimized single-level prefetch for AVX-512
 */
#define PREFETCH_8_LANES_AVX512(k, K, distance, sub_outputs, hint)                        \
    do                                                                                    \
    {                                                                                     \
        if ((k) + (distance) < K)                                                         \
        {                                                                                 \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);             \
            for (int j = 1; j < 8; j++)                                                   \
            {                                                                             \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + j * K], hint); \
            }                                                                             \
        }                                                                                 \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512 (OPTIMIZED!)
//==============================================================================

/**
 * @brief Complete optimized radix-8 butterfly (FORWARD, 4 butterflies)
 *
 * Uses fused radix-4 + W_8 twiddle application for better performance.
 */
#define RADIX8_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                         \
    {                                                                                                                          \
        __m512d x[8];                                                                                                          \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                            \
        APPLY_STAGE_TWIDDLES_AVX512(kk, x, stage_tw);                                                                          \
                                                                                                                               \
        __m512d e[4], o[4];                                                                                                    \
                                                                                                                               \
        /* Even radix-4: [0,2,4,6] - no twiddles */                                                                            \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                          \
                                                                                                                               \
        /* Odd radix-4 + W_8 twiddles (FUSED) */                                                                               \
        RADIX4_ODD_WITH_W8_FV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                           \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                            \
                                                                                                                               \
        FINAL_RADIX2_AVX512(e, o, output_buffer, kk, K);                                                                       \
    } while (0)

/**
 * @brief Complete optimized radix-8 butterfly (INVERSE, 4 butterflies)
 */
#define RADIX8_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                         \
    {                                                                                                                          \
        __m512d x[8];                                                                                                          \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                            \
        APPLY_STAGE_TWIDDLES_AVX512(kk, x, stage_tw);                                                                          \
                                                                                                                               \
        __m512d e[4], o[4];                                                                                                    \
                                                                                                                               \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                          \
        RADIX4_ODD_WITH_W8_BV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                           \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                            \
                                                                                                                               \
        FINAL_RADIX2_AVX512(e, o, output_buffer, kk, K);                                                                       \
    } while (0)

/**
 * @brief Streaming version (forward)
 */
#define RADIX8_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                \
    {                                                                                                                                 \
        __m512d x[8];                                                                                                                 \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                   \
        APPLY_STAGE_TWIDDLES_AVX512(kk, x, stage_tw);                                                                                 \
                                                                                                                                      \
        __m512d e[4], o[4];                                                                                                           \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                 \
        RADIX4_ODD_WITH_W8_FV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                  \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                   \
                                                                                                                                      \
        FINAL_RADIX2_AVX512_STREAM(e, o, output_buffer, kk, K);                                                                       \
    } while (0)

/**
 * @brief Streaming version (inverse)
 */
#define RADIX8_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                \
    {                                                                                                                                 \
        __m512d x[8];                                                                                                                 \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                   \
        APPLY_STAGE_TWIDDLES_AVX512(kk, x, stage_tw);                                                                                 \
                                                                                                                                      \
        __m512d e[4], o[4];                                                                                                           \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                 \
        RADIX4_ODD_WITH_W8_BV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                  \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                   \
                                                                                                                                      \
        FINAL_RADIX2_AVX512_STREAM(e, o, output_buffer, kk, K);                                                                       \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Complex multiply for AVX2 with FMA
 */
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

/**
 * @brief Radix-4 core for AVX2
 */
#define RADIX4_CORE_AVX2(a, b, c, d, y0, y1, y2, y3, rot_mask)    \
    do                                                            \
    {                                                             \
        __m256d sum_bd = _mm256_add_pd(b, d);                     \
        __m256d dif_bd = _mm256_sub_pd(b, d);                     \
        __m256d sum_ac = _mm256_add_pd(a, c);                     \
        __m256d dif_ac = _mm256_sub_pd(a, c);                     \
                                                                  \
        y0 = _mm256_add_pd(sum_ac, sum_bd);                       \
        y2 = _mm256_sub_pd(sum_ac, sum_bd);                       \
                                                                  \
        __m256d dif_bd_swp = _mm256_permute_pd(dif_bd, 0b0101);   \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask); \
                                                                  \
        y1 = _mm256_sub_pd(dif_ac, dif_bd_rot);                   \
        y3 = _mm256_add_pd(dif_ac, dif_bd_rot);                   \
    } while (0)

//==============================================================================
// FUSED RADIX-4 + W_8 - AVX2 (NEW!)
//==============================================================================

/**
 * @brief Fused odd radix-4 + W_8 twiddles (FORWARD, AVX2)
 */
#define RADIX4_ODD_WITH_W8_FV_AVX2(x1, x3, x5, x7, o0, o1, o2, o3, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                           \
    {                                                                                                            \
        __m256d sum_bd = _mm256_add_pd(x3, x7);                                                                  \
        __m256d dif_bd = _mm256_sub_pd(x3, x7);                                                                  \
        __m256d sum_ac = _mm256_add_pd(x1, x5);                                                                  \
        __m256d dif_ac = _mm256_sub_pd(x1, x5);                                                                  \
                                                                                                                 \
        o0 = _mm256_add_pd(sum_ac, sum_bd);                                                                      \
        __m256d o2_pre = _mm256_sub_pd(sum_ac, sum_bd);                                                          \
                                                                                                                 \
        __m256d dif_bd_swp = _mm256_permute_pd(dif_bd, 0b0101);                                                  \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask);                                                \
                                                                                                                 \
        __m256d o1_pre = _mm256_sub_pd(dif_ac, dif_bd_rot);                                                      \
        __m256d o3_pre = _mm256_add_pd(dif_ac, dif_bd_rot);                                                      \
                                                                                                                 \
        /* Fuse W_8 twiddles */                                                                                  \
        {                                                                                                        \
            __m256d o1_re = _mm256_movedup_pd(o1_pre);                                                           \
            __m256d o1_im = _mm256_permute_pd(o1_pre, 0xF);                                                      \
            __m256d new_re = _mm256_fmsub_pd(o1_re, vw81_re, _mm256_mul_pd(o1_im, vw81_im));                     \
            __m256d new_im = _mm256_fmadd_pd(o1_re, vw81_im, _mm256_mul_pd(o1_im, vw81_re));                     \
            o1 = _mm256_unpacklo_pd(new_re, new_im);                                                             \
        }                                                                                                        \
                                                                                                                 \
        o2 = _mm256_permute_pd(o2_pre, 0x5);                                                                     \
        o2 = _mm256_xor_pd(o2, _mm256_set_pd(-0.0, 0.0, -0.0, 0.0));                                             \
                                                                                                                 \
        {                                                                                                        \
            __m256d o3_re = _mm256_movedup_pd(o3_pre);                                                           \
            __m256d o3_im = _mm256_permute_pd(o3_pre, 0xF);                                                      \
            __m256d new_re = _mm256_fmsub_pd(o3_re, vw83_re, _mm256_mul_pd(o3_im, vw83_im));                     \
            __m256d new_im = _mm256_fmadd_pd(o3_re, vw83_im, _mm256_mul_pd(o3_im, vw83_re));                     \
            o3 = _mm256_unpacklo_pd(new_re, new_im);                                                             \
        }                                                                                                        \
    } while (0)

/**
 * @brief Fused odd radix-4 + W_8 twiddles (INVERSE, AVX2)
 */
#define RADIX4_ODD_WITH_W8_BV_AVX2(x1, x3, x5, x7, o0, o1, o2, o3, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                           \
    {                                                                                                            \
        __m256d sum_bd = _mm256_add_pd(x3, x7);                                                                  \
        __m256d dif_bd = _mm256_sub_pd(x3, x7);                                                                  \
        __m256d sum_ac = _mm256_add_pd(x1, x5);                                                                  \
        __m256d dif_ac = _mm256_sub_pd(x1, x5);                                                                  \
                                                                                                                 \
        o0 = _mm256_add_pd(sum_ac, sum_bd);                                                                      \
        __m256d o2_pre = _mm256_sub_pd(sum_ac, sum_bd);                                                          \
                                                                                                                 \
        __m256d dif_bd_swp = _mm256_permute_pd(dif_bd, 0b0101);                                                  \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask);                                                \
                                                                                                                 \
        __m256d o1_pre = _mm256_sub_pd(dif_ac, dif_bd_rot);                                                      \
        __m256d o3_pre = _mm256_add_pd(dif_ac, dif_bd_rot);                                                      \
                                                                                                                 \
        /* Fuse W_8 twiddles (inverse) */                                                                        \
        {                                                                                                        \
            __m256d o1_re = _mm256_movedup_pd(o1_pre);                                                           \
            __m256d o1_im = _mm256_permute_pd(o1_pre, 0xF);                                                      \
            __m256d new_re = _mm256_fmsub_pd(o1_re, vw81_re, _mm256_mul_pd(o1_im, vw81_im));                     \
            __m256d new_im = _mm256_fmadd_pd(o1_re, vw81_im, _mm256_mul_pd(o1_im, vw81_re));                     \
            o1 = _mm256_unpacklo_pd(new_re, new_im);                                                             \
        }                                                                                                        \
                                                                                                                 \
        o2 = _mm256_permute_pd(o2_pre, 0x5);                                                                     \
        o2 = _mm256_xor_pd(o2, _mm256_set_pd(0.0, -0.0, 0.0, -0.0));                                             \
                                                                                                                 \
        {                                                                                                        \
            __m256d o3_re = _mm256_movedup_pd(o3_pre);                                                           \
            __m256d o3_im = _mm256_permute_pd(o3_pre, 0xF);                                                      \
            __m256d new_re = _mm256_fmsub_pd(o3_re, vw83_re, _mm256_mul_pd(o3_im, vw83_im));                     \
            __m256d new_im = _mm256_fmadd_pd(o3_re, vw83_im, _mm256_mul_pd(o3_im, vw83_re));                     \
            o3 = _mm256_unpacklo_pd(new_re, new_im);                                                             \
        }                                                                                                        \
    } while (0)

/**
 * @brief Final radix-2 for AVX2
 */
#define FINAL_RADIX2_AVX2(e, o, output_buffer, k, K)            \
    do                                                          \
    {                                                           \
        for (int m = 0; m < 4; m++)                             \
        {                                                       \
            __m256d sum = _mm256_add_pd(e[m], o[m]);            \
            __m256d dif = _mm256_sub_pd(e[m], o[m]);            \
            STOREU_PD(&output_buffer[k + m * K].re, sum);       \
            STOREU_PD(&output_buffer[k + (m + 4) * K].re, dif); \
        }                                                       \
    } while (0)

/**
 * @brief Final radix-2 for AVX2 with streaming stores
 */
#define FINAL_RADIX2_AVX2_STREAM(e, o, output_buffer, k, K)            \
    do                                                                 \
    {                                                                  \
        for (int m = 0; m < 4; m++)                                    \
        {                                                              \
            __m256d sum = _mm256_add_pd(e[m], o[m]);                   \
            __m256d dif = _mm256_sub_pd(e[m], o[m]);                   \
            _mm256_stream_pd(&output_buffer[k + m * K].re, sum);       \
            _mm256_stream_pd(&output_buffer[k + (m + 4) * K].re, dif); \
        }                                                              \
    } while (0)

/**
 * @brief Apply stage twiddles (AVX2)
 */
#define APPLY_STAGE_TWIDDLES_AVX2(kk, x, stage_tw)                                                   \
    do                                                                                               \
    {                                                                                                \
        for (int j = 1; j <= 7; j++)                                                                 \
        {                                                                                            \
            __m256d w = load2_aos(&stage_tw[(kk) * 7 + (j - 1)], &stage_tw[(kk + 1) * 7 + (j - 1)]); \
            CMUL_FMA_AOS(x[j], x[j], w);                                                             \
        }                                                                                            \
    } while (0)

/**
 * @brief Load 8 lanes (AVX2)
 */
#define LOAD_8_LANES_AVX2(kk, K, sub_outputs, x)                                          \
    do                                                                                    \
    {                                                                                     \
        x[0] = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                       \
        for (int j = 1; j <= 7; j++)                                                      \
        {                                                                                 \
            x[j] = load2_aos(&sub_outputs[(kk) + j * K], &sub_outputs[(kk) + 1 + j * K]); \
        }                                                                                 \
    } while (0)

/**
 * @brief Prefetch distance for AVX2 L1 cache
 */
#define PREFETCH_L1 8

/**
 * @brief Optimized single-level prefetch for AVX2
 */
#define PREFETCH_8_LANES(k, K, distance, sub_outputs, hint)                               \
    do                                                                                    \
    {                                                                                     \
        if ((k) + (distance) < K)                                                         \
        {                                                                                 \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);             \
            for (int j = 1; j < 8; j++)                                                   \
            {                                                                             \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + j * K], hint); \
            }                                                                             \
        }                                                                                 \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

/**
 * @brief Radix-4 core for scalar implementation
 */
#define RADIX4_CORE_SCALAR(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,         \
                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, \
                           rot_sign)                                               \
    do                                                                             \
    {                                                                              \
        double sum_bd_re = b_re + d_re;                                            \
        double sum_bd_im = b_im + d_im;                                            \
        double dif_bd_re = b_re - d_re;                                            \
        double dif_bd_im = b_im - d_im;                                            \
        double sum_ac_re = a_re + c_re;                                            \
        double sum_ac_im = a_im + c_im;                                            \
        double dif_ac_re = a_re - c_re;                                            \
        double dif_ac_im = a_im - c_im;                                            \
                                                                                   \
        y0_re = sum_ac_re + sum_bd_re;                                             \
        y0_im = sum_ac_im + sum_bd_im;                                             \
        y2_re = sum_ac_re - sum_bd_re;                                             \
        y2_im = sum_ac_im - sum_bd_im;                                             \
                                                                                   \
        double rot_re = (rot_sign) * dif_bd_im;                                    \
        double rot_im = (rot_sign) * (-dif_bd_re);                                 \
                                                                                   \
        y1_re = dif_ac_re - rot_re;                                                \
        y1_im = dif_ac_im - rot_im;                                                \
        y3_re = dif_ac_re + rot_re;                                                \
        y3_im = dif_ac_im + rot_im;                                                \
    } while (0)

/**
 * @brief Apply W_8 twiddles for forward scalar
 */
#define APPLY_W8_TWIDDLES_FV_SCALAR(o)                 \
    do                                                 \
    {                                                  \
        {                                              \
            double r = o[1].re, i = o[1].im;           \
            o[1].re = r * W8_FV_1_RE - i * W8_FV_1_IM; \
            o[1].im = r * W8_FV_1_IM + i * W8_FV_1_RE; \
        }                                              \
        {                                              \
            double r = o[2].re, i = o[2].im;           \
            o[2].re = -i * W8_FV_2_IM;                 \
            o[2].im = r * W8_FV_2_IM;                  \
        }                                              \
        {                                              \
            double r = o[3].re, i = o[3].im;           \
            o[3].re = r * W8_FV_3_RE - i * W8_FV_3_IM; \
            o[3].im = r * W8_FV_3_IM + i * W8_FV_3_RE; \
        }                                              \
    } while (0)

/**
 * @brief Apply W_8 twiddles for inverse scalar
 */
#define APPLY_W8_TWIDDLES_BV_SCALAR(o)                 \
    do                                                 \
    {                                                  \
        {                                              \
            double r = o[1].re, i = o[1].im;           \
            o[1].re = r * W8_BV_1_RE - i * W8_BV_1_IM; \
            o[1].im = r * W8_BV_1_IM + i * W8_BV_1_RE; \
        }                                              \
        {                                              \
            double r = o[2].re, i = o[2].im;           \
            o[2].re = -i * W8_BV_2_IM;                 \
            o[2].im = r * W8_BV_2_IM;                  \
        }                                              \
        {                                              \
            double r = o[3].re, i = o[3].im;           \
            o[3].re = r * W8_BV_3_RE - i * W8_BV_3_IM; \
            o[3].im = r * W8_BV_3_IM + i * W8_BV_3_RE; \
        }                                              \
    } while (0)

/**
 * @brief Final radix-2 for scalar
 */
#define FINAL_RADIX2_SCALAR(e, o, output_buffer, k, K)             \
    do                                                             \
    {                                                              \
        for (int m = 0; m < 4; m++)                                \
        {                                                          \
            output_buffer[k + m * K].re = e[m].re + o[m].re;       \
            output_buffer[k + m * K].im = e[m].im + o[m].im;       \
            output_buffer[k + (m + 4) * K].re = e[m].re - o[m].re; \
            output_buffer[k + (m + 4) * K].im = e[m].im - o[m].im; \
        }                                                          \
    } while (0)

/**
 * @brief Apply stage twiddles for scalar
 */
#define APPLY_STAGE_TWIDDLES_SCALAR(k, x, stage_tw)                    \
    do                                                                 \
    {                                                                  \
        const fft_data *w_ptr = &stage_tw[(k) * 7];                    \
        for (int j = 1; j <= 7; j++)                                   \
        {                                                              \
            fft_data a = x[j];                                         \
            x[j].re = a.re * w_ptr[j - 1].re - a.im * w_ptr[j - 1].im; \
            x[j].im = a.re * w_ptr[j - 1].im + a.im * w_ptr[j - 1].re; \
        }                                                              \
    } while (0)

#endif // FFT_RADIX8_MACROS_H