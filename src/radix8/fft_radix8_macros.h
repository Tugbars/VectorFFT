//==============================================================================
// fft_radix8_macros.h - PURE SOA VERSION (ZERO SHUFFLE OVERHEAD!)
//==============================================================================
//
// ALGORITHM: Split-radix 2×(4,4) decomposition with fused operations
//   1. Apply input twiddles W_N^(j*k) to lanes 1-7 (NOW SOA!)
//   2. Two parallel radix-4 butterflies (even [0,2,4,6], odd [1,3,5,7])
//   3. Apply W_8 geometric twiddles to odd outputs (FUSED with radix-4)
//   4. Final radix-2 combination
//
// OPTIMIZATIONS PRESERVED:
//   - Hoisted W_8 constants outside loops
//   - Fused radix-4 + W_8 twiddle application (CROWN JEWEL!)
//   - Optimized W_8^2 = (0, ±1) handling
//   - Single-level prefetching (no cache pollution)
//   - Full AVX-512 support (4 butterflies/iteration)
//   - Alignment hints for better codegen
//
// SOA CHANGES:
//   - Twiddle loads: Direct re/im arrays (zero shuffle overhead!)
//   - Complex multiply: cmul_*_soa versions
//   - Prefetch: Separate re/im blocks (7 twiddles!)
//

#ifndef FFT_RADIX8_MACROS_H
#define FFT_RADIX8_MACROS_H

#include "simd_math.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

#define STREAM_THRESHOLD 2048

//==============================================================================
// W_8 CONSTANTS - Direction-dependent (UNCHANGED)
//==============================================================================

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

// Forward W_8 twiddles
#define W8_FV_1_RE C8_CONSTANT
#define W8_FV_1_IM (-C8_CONSTANT)
#define W8_FV_2_RE 0.0
#define W8_FV_2_IM (-1.0)
#define W8_FV_3_RE (-C8_CONSTANT)
#define W8_FV_3_IM (-C8_CONSTANT)

// Inverse W_8 twiddles
#define W8_BV_1_RE C8_CONSTANT
#define W8_BV_1_IM C8_CONSTANT
#define W8_BV_2_RE 0.0
#define W8_BV_2_IM 1.0
#define W8_BV_3_RE (-C8_CONSTANT)
#define W8_BV_3_IM C8_CONSTANT

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Complex multiply with SoA twiddles (AVX-512)
 */
#define CMUL_FMA_SOA_AVX512(out, a, w_re, w_im)                          \
    do                                                                   \
    {                                                                    \
        __m512d ar = _mm512_unpacklo_pd(a, a);                           \
        __m512d ai = _mm512_unpackhi_pd(a, a);                           \
        /* Twiddles already separated (SoA) - NO SHUFFLE! */             \
        __m512d re = _mm512_fmsub_pd(ar, w_re, _mm512_mul_pd(ai, w_im)); \
        __m512d im = _mm512_fmadd_pd(ar, w_im, _mm512_mul_pd(ai, w_re)); \
        (out) = _mm512_unpacklo_pd(re, im);                              \
    } while (0)

//==============================================================================
// RADIX-4 SUB-BUTTERFLY - AVX-512 (UNCHANGED)
//==============================================================================

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
// FUSED RADIX-4 + W_8 TWIDDLE APPLICATION - AVX-512 (UNCHANGED!)
//==============================================================================

/**
 * @brief Fused odd radix-4 butterfly with W_8 twiddle application (FORWARD)
 *
 * THIS IS YOUR CROWN JEWEL OPTIMIZATION - PRESERVED 100%!
 *
 * Combines radix-4 butterfly on odd lanes [1,3,5,7] with immediate W_8
 * twiddle multiplication. Reduces register pressure and improves scheduling.
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
        {                                                                                                          \
            __m512d o1_re = _mm512_movedup_pd(o1_pre);                                                             \
            __m512d o1_im = _mm512_permute_pd(o1_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o1_re, vw81_re, _mm512_mul_pd(o1_im, vw81_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o1_re, vw81_im, _mm512_mul_pd(o1_im, vw81_re));                       \
            o1 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
                                                                                                                   \
        o2 = _mm512_permute_pd(o2_pre, 0x55);                                                                      \
        o2 = _mm512_xor_pd(o2, _mm512_set_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0));                         \
                                                                                                                   \
        {                                                                                                          \
            __m512d o3_re = _mm512_movedup_pd(o3_pre);                                                             \
            __m512d o3_im = _mm512_permute_pd(o3_pre, 0xFF);                                                       \
            __m512d new_re = _mm512_fmsub_pd(o3_re, vw83_re, _mm512_mul_pd(o3_im, vw83_im));                       \
            __m512d new_im = _mm512_fmadd_pd(o3_re, vw83_im, _mm512_mul_pd(o3_im, vw83_re));                       \
            o3 = _mm512_unpacklo_pd(new_re, new_im);                                                               \
        }                                                                                                          \
    } while (0)

//==============================================================================
// FINAL RADIX-2 COMBINATION - AVX-512 (UNCHANGED)
//==============================================================================

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
// APPLY PRECOMPUTED TWIDDLES - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Apply stage twiddles for 4 butterflies (SoA)
 *
 * CRITICAL CHANGE: Load from separate re/im arrays
 *
 * OLD: Load interleaved: stage_tw[k*7 + j] for j=0..6
 * NEW: Load SoA: tw->re[j*K + k], tw->im[j*K + k] for j=0..6
 *
 * Radix-8 has 7 twiddles per butterfly (for lanes 1-7):
 * - W^(1*k), W^(2*k), W^(3*k), W^(4*k), W^(5*k), W^(6*k), W^(7*k)
 */
#define APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, x, stage_tw)                    \
    do                                                                         \
    {                                                                          \
        for (int j = 1; j <= 7; j++)                                           \
        {                                                                      \
            /* Load W^(j*k) for 4 butterflies */                               \
            __m512d w_re = _mm512_loadu_pd(&stage_tw->re[(j - 1) * K + (kk)]); \
            __m512d w_im = _mm512_loadu_pd(&stage_tw->im[(j - 1) * K + (kk)]); \
            /* Apply twiddle (SoA complex multiply - ZERO SHUFFLE!) */         \
            CMUL_FMA_SOA_AVX512(x[j], x[j], w_re, w_im);                       \
        }                                                                      \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512 (UNCHANGED - data is still AoS)
//==============================================================================

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
// PREFETCHING - AVX-512 (SoA Version)
//==============================================================================

#define PREFETCH_L1_AVX512 16

/**
 * @brief Prefetch SoA twiddles (7 separate re/im blocks for radix-8!)
 *
 * OLD: Prefetch interleaved: stage_tw[(k+dist)*7 + j]
 * NEW: Prefetch separate: tw->re[j*K + k+dist] for j=0..6
 */
#define PREFETCH_8_LANES_AVX512_SOA(k, K, distance, sub_outputs, stage_tw, hint)           \
    do                                                                                     \
    {                                                                                      \
        if ((k) + (distance) < K)                                                          \
        {                                                                                  \
            /* Prefetch input lanes (8 cache lines) */                                     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);              \
            for (int j = 1; j < 8; j++)                                                    \
            {                                                                              \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + j * K], hint);  \
            }                                                                              \
            /* Prefetch SoA twiddles (7 pairs of re/im blocks) */                          \
            for (int j = 0; j < 7; j++)                                                    \
            {                                                                              \
                _mm_prefetch((const char *)&stage_tw->re[j * K + (k) + (distance)], hint); \
                _mm_prefetch((const char *)&stage_tw->im[j * K + (k) + (distance)], hint); \
            }                                                                              \
        }                                                                                  \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Complete optimized radix-8 butterfly (FORWARD, 4 butterflies, SoA)
 */
#define RADIX8_PIPELINE_4_FV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                             \
    {                                                                                                                              \
        __m512d x[8];                                                                                                              \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, x, stage_tw);                                                                       \
                                                                                                                                   \
        __m512d e[4], o[4];                                                                                                        \
                                                                                                                                   \
        /* Even radix-4: [0,2,4,6] - no twiddles */                                                                                \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                              \
                                                                                                                                   \
        /* Odd radix-4 + W_8 twiddles (FUSED - your crown jewel!) */                                                               \
        RADIX4_ODD_WITH_W8_FV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                               \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                \
                                                                                                                                   \
        FINAL_RADIX2_AVX512(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

/**
 * @brief Complete optimized radix-8 butterfly (INVERSE, 4 butterflies, SoA)
 */
#define RADIX8_PIPELINE_4_BV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                             \
    {                                                                                                                              \
        __m512d x[8];                                                                                                              \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, x, stage_tw);                                                                       \
                                                                                                                                   \
        __m512d e[4], o[4];                                                                                                        \
                                                                                                                                   \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                              \
        RADIX4_ODD_WITH_W8_BV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                               \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                \
                                                                                                                                   \
        FINAL_RADIX2_AVX512(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

/**
 * @brief Streaming version (forward, SoA)
 */
#define RADIX8_PIPELINE_4_FV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                    \
    {                                                                                                                                     \
        __m512d x[8];                                                                                                                     \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                       \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, x, stage_tw);                                                                              \
                                                                                                                                          \
        __m512d e[4], o[4];                                                                                                               \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                     \
        RADIX4_ODD_WITH_W8_FV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                      \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                       \
                                                                                                                                          \
        FINAL_RADIX2_AVX512_STREAM(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

/**
 * @brief Streaming version (inverse, SoA)
 */
#define RADIX8_PIPELINE_4_BV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                    \
    {                                                                                                                                     \
        __m512d x[8];                                                                                                                     \
        LOAD_8_LANES_AVX512(kk, K, sub_outputs, x);                                                                                       \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, x, stage_tw);                                                                              \
                                                                                                                                          \
        __m512d e[4], o[4];                                                                                                               \
        RADIX4_CORE_AVX512(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                     \
        RADIX4_ODD_WITH_W8_BV_AVX512(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                      \
                                     rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                       \
                                                                                                                                          \
        FINAL_RADIX2_AVX512_STREAM(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX2 (SoA Version)
//==============================================================================

#define CMUL_FMA_SOA_AVX2(out, a, w_re, w_im)                            \
    do                                                                   \
    {                                                                    \
        __m256d ar = _mm256_unpacklo_pd(a, a);                           \
        __m256d ai = _mm256_unpackhi_pd(a, a);                           \
        /* Twiddles already separated (SoA) - NO SHUFFLE! */             \
        __m256d re = _mm256_fmsub_pd(ar, w_re, _mm256_mul_pd(ai, w_im)); \
        __m256d im = _mm256_fmadd_pd(ar, w_im, _mm256_mul_pd(ai, w_re)); \
        (out) = _mm256_unpacklo_pd(re, im);                              \
    } while (0)

//==============================================================================
// RADIX-4 CORE - AVX2 (UNCHANGED)
//==============================================================================

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
// FUSED RADIX-4 + W_8 - AVX2 (UNCHANGED!)
//==============================================================================

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

//==============================================================================
// FINAL RADIX-2 - AVX2 (UNCHANGED)
//==============================================================================

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

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX2 (SoA Version)
//==============================================================================

/**
 * @brief Apply stage twiddles (AVX2, SoA)
 */
#define APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, x, stage_tw)                      \
    do                                                                         \
    {                                                                          \
        for (int j = 1; j <= 7; j++)                                           \
        {                                                                      \
            /* Load W^(j*k) for 2 butterflies */                               \
            __m256d w_re = _mm256_loadu_pd(&stage_tw->re[(j - 1) * K + (kk)]); \
            __m256d w_im = _mm256_loadu_pd(&stage_tw->im[(j - 1) * K + (kk)]); \
            CMUL_FMA_SOA_AVX2(x[j], x[j], w_re, w_im);                         \
        }                                                                      \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2 (UNCHANGED)
//==============================================================================

#define LOAD_8_LANES_AVX2(kk, K, sub_outputs, x)                                          \
    do                                                                                    \
    {                                                                                     \
        x[0] = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                       \
        for (int j = 1; j <= 7; j++)                                                      \
        {                                                                                 \
            x[j] = load2_aos(&sub_outputs[(kk) + j * K], &sub_outputs[(kk) + 1 + j * K]); \
        }                                                                                 \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2 (SoA Version)
//==============================================================================

#define PREFETCH_L1 8

#define PREFETCH_8_LANES_AVX2_SOA(k, K, distance, sub_outputs, stage_tw, hint)             \
    do                                                                                     \
    {                                                                                      \
        if ((k) + (distance) < K)                                                          \
        {                                                                                  \
            /* Prefetch input lanes */                                                     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);              \
            for (int j = 1; j < 8; j++)                                                    \
            {                                                                              \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + j * K], hint);  \
            }                                                                              \
            /* Prefetch SoA twiddles (7 pairs) */                                          \
            for (int j = 0; j < 7; j++)                                                    \
            {                                                                              \
                _mm_prefetch((const char *)&stage_tw->re[j * K + (k) + (distance)], hint); \
                _mm_prefetch((const char *)&stage_tw->im[j * K + (k) + (distance)], hint); \
            }                                                                              \
        }                                                                                  \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX2 (SoA Version)
//==============================================================================

#define RADIX8_PIPELINE_2_FV_AVX2_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                           \
    {                                                                                                                            \
        __m256d x[8];                                                                                                            \
        LOAD_8_LANES_AVX2(kk, K, sub_outputs, x);                                                                                \
        APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, x, stage_tw);                                                                       \
                                                                                                                                 \
        __m256d e[4], o[4];                                                                                                      \
        RADIX4_CORE_AVX2(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                              \
        RADIX4_ODD_WITH_W8_FV_AVX2(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                               \
                                   rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                \
                                                                                                                                 \
        FINAL_RADIX2_AVX2(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

#define RADIX8_PIPELINE_2_BV_AVX2_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                           \
    {                                                                                                                            \
        __m256d x[8];                                                                                                            \
        LOAD_8_LANES_AVX2(kk, K, sub_outputs, x);                                                                                \
        APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, x, stage_tw);                                                                       \
                                                                                                                                 \
        __m256d e[4], o[4];                                                                                                      \
        RADIX4_CORE_AVX2(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                              \
        RADIX4_ODD_WITH_W8_BV_AVX2(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                               \
                                   rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                \
                                                                                                                                 \
        FINAL_RADIX2_AVX2(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

#define RADIX8_PIPELINE_2_FV_AVX2_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                  \
    {                                                                                                                                   \
        __m256d x[8];                                                                                                                   \
        LOAD_8_LANES_AVX2(kk, K, sub_outputs, x);                                                                                       \
        APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, x, stage_tw);                                                                              \
                                                                                                                                        \
        __m256d e[4], o[4];                                                                                                             \
        RADIX4_CORE_AVX2(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                     \
        RADIX4_ODD_WITH_W8_FV_AVX2(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                      \
                                   rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                       \
                                                                                                                                        \
        FINAL_RADIX2_AVX2_STREAM(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

#define RADIX8_PIPELINE_2_BV_AVX2_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, rot_mask, vw81_re, vw81_im, vw83_re, vw83_im) \
    do                                                                                                                                  \
    {                                                                                                                                   \
        __m256d x[8];                                                                                                                   \
        LOAD_8_LANES_AVX2(kk, K, sub_outputs, x);                                                                                       \
        APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, x, stage_tw);                                                                              \
                                                                                                                                        \
        __m256d e[4], o[4];                                                                                                             \
        RADIX4_CORE_AVX2(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);                                                     \
        RADIX4_ODD_WITH_W8_BV_AVX2(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3],                                                      \
                                   rot_mask, vw81_re, vw81_im, vw83_re, vw83_im);                                                       \
                                                                                                                                        \
        FINAL_RADIX2_AVX2_STREAM(e, o, output_buffer, kk, K);                                                                           \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT (SoA Version)
//==============================================================================

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
 * @brief Apply stage twiddles for scalar (SoA)
 *
 * OLD: Load from stage_tw[k*7 + j]
 * NEW: Load from tw->re[j*K + k], tw->im[j*K + k]
 */
#define APPLY_STAGE_TWIDDLES_SCALAR_SOA(k, K, x, stage_tw) \
    do                                                     \
    {                                                      \
        for (int j = 1; j <= 7; j++)                       \
        {                                                  \
            fft_data a = x[j];                             \
            double w_re = stage_tw->re[(j - 1) * K + k];   \
            double w_im = stage_tw->im[(j - 1) * K + k];   \
            x[j].re = a.re * w_re - a.im * w_im;           \
            x[j].im = a.re * w_im + a.im * w_re;           \
        }                                                  \
    } while (0)

#endif // FFT_RADIX8_MACROS_H