//==============================================================================
// fft_radix5_macros.h - PURE SOA VERSION (ZERO SHUFFLE OVERHEAD!)
//==============================================================================
//
// USAGE:
//   #include "fft_radix5_macros.h" in both fft_radix5_fv.c and fft_radix5_bv.c
//
// BENEFITS:
//   - 99% code reuse between forward/inverse
//   - Single source of truth for radix-5 butterfly
//   - Only difference: rotation direction (±i multiplication)
//
// SOA CHANGES:
//   - Twiddle loads: Direct re/im arrays (zero shuffle overhead!)
//   - Complex multiply: cmul_*_soa versions
//   - Prefetch: Separate re/im blocks (4 twiddle pairs!)
//

#ifndef FFT_RADIX5_MACROS_H
#define FFT_RADIX5_MACROS_H

#include "simd_math.h"

//==============================================================================
// RADIX-5 GEOMETRIC CONSTANTS (IDENTICAL for both directions, UNCHANGED)
//==============================================================================

#define C5_1 0.30901699437494742410    // cos(2π/5) = (sqrt(5) - 1) / 4
#define C5_2 (-0.80901699437494742410) // cos(4π/5) = -(sqrt(5) + 1) / 4
#define S5_1 0.95105651629515357212    // sin(2π/5)
#define S5_2 0.58778525229247312917    // sin(4π/5)

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
#define CMUL_FMA_SOA_R5_AVX512(out, a, w_re, w_im)                       \
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
// RADIX-5 BUTTERFLY CORE - AVX-512 (UNCHANGED)
//==============================================================================

/**
 * @brief Compute intermediate sums for radix-5 (AVX-512, 4 butterflies)
 *
 * Stage 1: Compute pair sums
 * s1 = tw_b + tw_e  (indices 1 and 4)
 * s2 = tw_c + tw_d  (indices 2 and 3)
 * d1 = tw_b - tw_e
 * d2 = tw_c - tw_d
 *
 * Stage 2: Common terms
 * sum_all = s1 + s2
 * y0 = a + sum_all
 */
#define RADIX5_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, tw_e,   \
                                     s1, s2, d1, d2, sum_all, y0) \
    do                                                            \
    {                                                             \
        s1 = _mm512_add_pd(tw_b, tw_e); /* b + e */               \
        s2 = _mm512_add_pd(tw_c, tw_d); /* c + d */               \
        d1 = _mm512_sub_pd(tw_b, tw_e); /* b - e */               \
        d2 = _mm512_sub_pd(tw_c, tw_d); /* c - d */               \
        sum_all = _mm512_add_pd(s1, s2);                          \
        y0 = _mm512_add_pd(a, sum_all);                           \
    } while (0)

//==============================================================================
// INTERMEDIATE COMPUTATIONS - AVX-512 (UNCHANGED)
//==============================================================================

/**
 * @brief Compute scaled sums using geometric constants (AVX-512)
 *
 * t1 = a + C5_1 * s1 + C5_2 * s2
 * t2 = a + C5_2 * s1 + C5_1 * s2
 */
#define RADIX5_COMPUTE_T_AVX512(a, s1, s2, t1, t2) \
    do                                             \
    {                                              \
        const __m512d vc51 = _mm512_set1_pd(C5_1); \
        const __m512d vc52 = _mm512_set1_pd(C5_2); \
        t1 = _mm512_fmadd_pd(vc51, s1, a);         \
        t1 = _mm512_fmadd_pd(vc52, s2, t1);        \
        t2 = _mm512_fmadd_pd(vc52, s1, a);         \
        t2 = _mm512_fmadd_pd(vc51, s2, t2);        \
    } while (0)

//==============================================================================
// ROTATION AND SCALING - DIRECTION-SPECIFIC (UNCHANGED)
//==============================================================================

/**
 * @brief FORWARD rotation and scaling (AVX-512, 4 butterflies)
 */
#define RADIX5_ROTATE_FORWARD_AVX512(d1, d2, u1, u2)                  \
    do                                                                \
    {                                                                 \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                    \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                    \
        const __m512d rot_mask = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,  \
                                               0.0, -0.0, 0.0, -0.0); \
                                                                      \
        __m512d temp1 = _mm512_mul_pd(vs51, d1);                      \
        temp1 = _mm512_fmadd_pd(vs52, d2, temp1);                     \
                                                                      \
        __m512d temp2 = _mm512_mul_pd(vs52, d1);                      \
        temp2 = _mm512_fnmadd_pd(vs51, d2, temp2);                    \
                                                                      \
        __m512d temp1_swp = _mm512_permute_pd(temp1, 0b01010101);     \
        u1 = _mm512_xor_pd(temp1_swp, rot_mask);                      \
                                                                      \
        __m512d temp2_swp = _mm512_permute_pd(temp2, 0b01010101);     \
        u2 = _mm512_xor_pd(temp2_swp, rot_mask);                      \
    } while (0)

/**
 * @brief INVERSE rotation and scaling (AVX-512, 4 butterflies)
 */
#define RADIX5_ROTATE_INVERSE_AVX512(d1, d2, u1, u2)                  \
    do                                                                \
    {                                                                 \
        const __m512d vs51 = _mm512_set1_pd(S5_1);                    \
        const __m512d vs52 = _mm512_set1_pd(S5_2);                    \
        const __m512d rot_mask = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,  \
                                               -0.0, 0.0, -0.0, 0.0); \
                                                                      \
        __m512d temp1 = _mm512_mul_pd(vs51, d1);                      \
        temp1 = _mm512_fmadd_pd(vs52, d2, temp1);                     \
                                                                      \
        __m512d temp2 = _mm512_mul_pd(vs52, d1);                      \
        temp2 = _mm512_fnmadd_pd(vs51, d2, temp2);                    \
                                                                      \
        __m512d temp1_swp = _mm512_permute_pd(temp1, 0b01010101);     \
        u1 = _mm512_xor_pd(temp1_swp, rot_mask);                      \
                                                                      \
        __m512d temp2_swp = _mm512_permute_pd(temp2, 0b01010101);     \
        u2 = _mm512_xor_pd(temp2_swp, rot_mask);                      \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512 (UNCHANGED)
//==============================================================================

/**
 * @brief Assemble final radix-5 outputs (AVX-512, 4 butterflies)
 */
#define RADIX5_ASSEMBLE_OUTPUTS_AVX512(y0, t1, t2, u1, u2, \
                                       y1, y2, y3, y4)     \
    do                                                     \
    {                                                      \
        y1 = _mm512_add_pd(t1, u1);                        \
        y2 = _mm512_add_pd(t2, u2);                        \
        y3 = _mm512_sub_pd(t2, u2);                        \
        y4 = _mm512_sub_pd(t1, u1);                        \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Apply stage twiddles for 4 butterflies (SoA)
 *
 * CRITICAL CHANGE: Load from separate re/im arrays
 *
 * OLD: stage_tw[k*4 + j] for j=0..3
 * NEW: tw->re[j*K + k], tw->im[j*K + k] for j=0..3
 *
 * Radix-5 has 4 twiddles per butterfly (for lanes 1-4):
 * - W^(1*k), W^(2*k), W^(3*k), W^(4*k)
 */
#define APPLY_STAGE_TWIDDLES_R5_AVX512_SOA(kk, K, b, c, d, e, stage_tw, \
                                           tw_b, tw_c, tw_d, tw_e)      \
    do                                                                  \
    {                                                                   \
        /* Load W^(1*k) for 4 butterflies */                            \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * K + (kk)]);   \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * K + (kk)]);   \
        /* Load W^(2*k) for 4 butterflies */                            \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * K + (kk)]);   \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * K + (kk)]);   \
        /* Load W^(3*k) for 4 butterflies */                            \
        __m512d w3_re = _mm512_loadu_pd(&stage_tw->re[2 * K + (kk)]);   \
        __m512d w3_im = _mm512_loadu_pd(&stage_tw->im[2 * K + (kk)]);   \
        /* Load W^(4*k) for 4 butterflies */                            \
        __m512d w4_re = _mm512_loadu_pd(&stage_tw->re[3 * K + (kk)]);   \
        __m512d w4_im = _mm512_loadu_pd(&stage_tw->im[3 * K + (kk)]);   \
                                                                        \
        /* Apply twiddles (SoA complex multiply - ZERO SHUFFLE!) */     \
        CMUL_FMA_SOA_R5_AVX512(tw_b, b, w1_re, w1_im);                  \
        CMUL_FMA_SOA_R5_AVX512(tw_c, c, w2_re, w2_im);                  \
        CMUL_FMA_SOA_R5_AVX512(tw_d, d, w3_re, w3_im);                  \
        CMUL_FMA_SOA_R5_AVX512(tw_e, e, w4_re, w4_im);                  \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512 (UNCHANGED - data is still AoS)
//==============================================================================

#define LOAD_5_LANES_AVX512(kk, K, sub_outputs, a, b, c, d, e) \
    do                                                         \
    {                                                          \
        a = load4_aos(&sub_outputs[kk],                        \
                      &sub_outputs[(kk) + 1],                  \
                      &sub_outputs[(kk) + 2],                  \
                      &sub_outputs[(kk) + 3]);                 \
        b = load4_aos(&sub_outputs[(kk) + K],                  \
                      &sub_outputs[(kk) + 1 + K],              \
                      &sub_outputs[(kk) + 2 + K],              \
                      &sub_outputs[(kk) + 3 + K]);             \
        c = load4_aos(&sub_outputs[(kk) + 2 * K],              \
                      &sub_outputs[(kk) + 1 + 2 * K],          \
                      &sub_outputs[(kk) + 2 + 2 * K],          \
                      &sub_outputs[(kk) + 3 + 2 * K]);         \
        d = load4_aos(&sub_outputs[(kk) + 3 * K],              \
                      &sub_outputs[(kk) + 1 + 3 * K],          \
                      &sub_outputs[(kk) + 2 + 3 * K],          \
                      &sub_outputs[(kk) + 3 + 3 * K]);         \
        e = load4_aos(&sub_outputs[(kk) + 4 * K],              \
                      &sub_outputs[(kk) + 1 + 4 * K],          \
                      &sub_outputs[(kk) + 2 + 4 * K],          \
                      &sub_outputs[(kk) + 3 + 4 * K]);         \
    } while (0)

#define STORE_5_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                                 \
    {                                                                  \
        STOREU_PD512(&output_buffer[kk].re, y0);                       \
        STOREU_PD512(&output_buffer[(kk) + K].re, y1);                 \
        STOREU_PD512(&output_buffer[(kk) + 2 * K].re, y2);             \
        STOREU_PD512(&output_buffer[(kk) + 3 * K].re, y3);             \
        STOREU_PD512(&output_buffer[(kk) + 4 * K].re, y4);             \
    } while (0)

#define STORE_5_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                                        \
    {                                                                         \
        _mm512_stream_pd(&output_buffer[kk].re, y0);                          \
        _mm512_stream_pd(&output_buffer[(kk) + K].re, y1);                    \
        _mm512_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);                \
        _mm512_stream_pd(&output_buffer[(kk) + 3 * K].re, y3);                \
        _mm512_stream_pd(&output_buffer[(kk) + 4 * K].re, y4);                \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512 (SoA Version)
//==============================================================================

#define PREFETCH_L1_R5_AVX512 16
#define PREFETCH_L2_R5_AVX512 64
#define PREFETCH_L3_R5_AVX512 128

/**
 * @brief Prefetch SoA twiddles (4 separate re/im blocks for radix-5!)
 *
 * OLD: Prefetch interleaved: stage_tw[(k+dist)*4 + j]
 * NEW: Prefetch separate: tw->re[j*K + k+dist] for j=0..3
 */
#define PREFETCH_5_LANES_AVX512_SOA(k, K, distance, sub_outputs, stage_tw, hint)           \
    do                                                                                     \
    {                                                                                      \
        if ((k) + (distance) < K)                                                          \
        {                                                                                  \
            /* Prefetch input lanes (5 cache lines) */                                     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);              \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);          \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint);      \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 3 * K], hint);      \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 4 * K], hint);      \
            /* Prefetch SoA twiddles (4 pairs of re/im blocks) */                          \
            for (int j = 0; j < 4; j++)                                                    \
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
 * @brief Complete AVX-512 radix-5 butterfly (FORWARD, 4 butterflies, SoA)
 */
#define RADIX5_PIPELINE_4_FV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                               \
    {                                                                                \
        __m512d a, b, c, d, e;                                                       \
        LOAD_5_LANES_AVX512(kk, K, sub_outputs, a, b, c, d, e);                      \
                                                                                     \
        __m512d tw_b, tw_c, tw_d, tw_e;                                              \
        APPLY_STAGE_TWIDDLES_R5_AVX512_SOA(kk, K, b, c, d, e, stage_tw,              \
                                           tw_b, tw_c, tw_d, tw_e);                  \
                                                                                     \
        __m512d s1, s2, d1, d2, sum_all, y0;                                         \
        RADIX5_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, tw_e,                      \
                                     s1, s2, d1, d2, sum_all, y0);                   \
                                                                                     \
        __m512d t1, t2;                                                              \
        RADIX5_COMPUTE_T_AVX512(a, s1, s2, t1, t2);                                  \
                                                                                     \
        __m512d u1, u2;                                                              \
        RADIX5_ROTATE_FORWARD_AVX512(d1, d2, u1, u2);                                \
                                                                                     \
        __m512d y1, y2, y3, y4;                                                      \
        RADIX5_ASSEMBLE_OUTPUTS_AVX512(y0, t1, t2, u1, u2, y1, y2, y3, y4);          \
                                                                                     \
        STORE_5_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4);              \
    } while (0)

/**
 * @brief Complete AVX-512 radix-5 butterfly (INVERSE, 4 butterflies, SoA)
 */
#define RADIX5_PIPELINE_4_BV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                               \
    {                                                                                \
        __m512d a, b, c, d, e;                                                       \
        LOAD_5_LANES_AVX512(kk, K, sub_outputs, a, b, c, d, e);                      \
                                                                                     \
        __m512d tw_b, tw_c, tw_d, tw_e;                                              \
        APPLY_STAGE_TWIDDLES_R5_AVX512_SOA(kk, K, b, c, d, e, stage_tw,              \
                                           tw_b, tw_c, tw_d, tw_e);                  \
                                                                                     \
        __m512d s1, s2, d1, d2, sum_all, y0;                                         \
        RADIX5_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, tw_e,                      \
                                     s1, s2, d1, d2, sum_all, y0);                   \
                                                                                     \
        __m512d t1, t2;                                                              \
        RADIX5_COMPUTE_T_AVX512(a, s1, s2, t1, t2);                                  \
                                                                                     \
        __m512d u1, u2;                                                              \
        RADIX5_ROTATE_INVERSE_AVX512(d1, d2, u1, u2); /* INVERSE rotation */         \
                                                                                     \
        __m512d y1, y2, y3, y4;                                                      \
        RADIX5_ASSEMBLE_OUTPUTS_AVX512(y0, t1, t2, u1, u2, y1, y2, y3, y4);          \
                                                                                     \
        STORE_5_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4);              \
    } while (0)

//==============================================================================
// STREAMING VERSIONS
//==============================================================================

#define RADIX5_PIPELINE_4_FV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                                      \
    {                                                                                       \
        __m512d a, b, c, d, e;                                                              \
        LOAD_5_LANES_AVX512(kk, K, sub_outputs, a, b, c, d, e);                             \
        __m512d tw_b, tw_c, tw_d, tw_e;                                                     \
        APPLY_STAGE_TWIDDLES_R5_AVX512_SOA(kk, K, b, c, d, e, stage_tw,                     \
                                           tw_b, tw_c, tw_d, tw_e);                         \
        __m512d s1, s2, d1, d2, sum_all, y0;                                                \
        RADIX5_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, tw_e,                             \
                                     s1, s2, d1, d2, sum_all, y0);                          \
        __m512d t1, t2;                                                                     \
        RADIX5_COMPUTE_T_AVX512(a, s1, s2, t1, t2);                                         \
        __m512d u1, u2;                                                                     \
        RADIX5_ROTATE_FORWARD_AVX512(d1, d2, u1, u2);                                       \
        __m512d y1, y2, y3, y4;                                                             \
        RADIX5_ASSEMBLE_OUTPUTS_AVX512(y0, t1, t2, u1, u2, y1, y2, y3, y4);                 \
        STORE_5_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4);              \
    } while (0)

#define RADIX5_PIPELINE_4_BV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                                      \
    {                                                                                       \
        __m512d a, b, c, d, e;                                                              \
        LOAD_5_LANES_AVX512(kk, K, sub_outputs, a, b, c, d, e);                             \
        __m512d tw_b, tw_c, tw_d, tw_e;                                                     \
        APPLY_STAGE_TWIDDLES_R5_AVX512_SOA(kk, K, b, c, d, e, stage_tw,                     \
                                           tw_b, tw_c, tw_d, tw_e);                         \
        __m512d s1, s2, d1, d2, sum_all, y0;                                                \
        RADIX5_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, tw_e,                             \
                                     s1, s2, d1, d2, sum_all, y0);                          \
        __m512d t1, t2;                                                                     \
        RADIX5_COMPUTE_T_AVX512(a, s1, s2, t1, t2);                                         \
        __m512d u1, u2;                                                                     \
        RADIX5_ROTATE_INVERSE_AVX512(d1, d2, u1, u2);                                       \
        __m512d y1, y2, y3, y4;                                                             \
        RADIX5_ASSEMBLE_OUTPUTS_AVX512(y0, t1, t2, u1, u2, y1, y2, y3, y4);                 \
        STORE_5_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4);              \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX2 (SoA Version)
//==============================================================================

#define CMUL_FMA_SOA_R5_AVX2(out, a, w_re, w_im)                         \
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
// APPLY PRECOMPUTED TWIDDLES - AVX2 (SoA Version)
//==============================================================================

/**
 * @brief Apply stage twiddles for 2 butterflies (SoA)
 */
#define APPLY_STAGE_TWIDDLES_R5_AVX2_SOA(kk, K, b, c, d, e, stage_tw, \
                                         tw_b, tw_c, tw_d, tw_e)      \
    do                                                                \
    {                                                                 \
        /* Load W^(1*k) for 2 butterflies */                          \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * K + (kk)]); \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * K + (kk)]); \
        /* Load W^(2*k) for 2 butterflies */                          \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * K + (kk)]); \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * K + (kk)]); \
        /* Load W^(3*k) for 2 butterflies */                          \
        __m256d w3_re = _mm256_loadu_pd(&stage_tw->re[2 * K + (kk)]); \
        __m256d w3_im = _mm256_loadu_pd(&stage_tw->im[2 * K + (kk)]); \
        /* Load W^(4*k) for 2 butterflies */                          \
        __m256d w4_re = _mm256_loadu_pd(&stage_tw->re[3 * K + (kk)]); \
        __m256d w4_im = _mm256_loadu_pd(&stage_tw->im[3 * K + (kk)]); \
                                                                      \
        CMUL_FMA_SOA_R5_AVX2(tw_b, b, w1_re, w1_im);                  \
        CMUL_FMA_SOA_R5_AVX2(tw_c, c, w2_re, w2_im);                  \
        CMUL_FMA_SOA_R5_AVX2(tw_d, d, w3_re, w3_im);                  \
        CMUL_FMA_SOA_R5_AVX2(tw_e, e, w4_re, w4_im);                  \
    } while (0)

//==============================================================================
// RADIX-5 BUTTERFLY CORE - AVX2 (UNCHANGED)
//==============================================================================

#define RADIX5_BUTTERFLY_CORE_AVX2(a, b2, c2, d2, e2, t0, t1, t2, t3) \
    do                                                                \
    {                                                                 \
        t0 = _mm256_add_pd(b2, e2);                                   \
        t1 = _mm256_add_pd(c2, d2);                                   \
        t2 = _mm256_sub_pd(b2, e2);                                   \
        t3 = _mm256_sub_pd(c2, d2);                                   \
    } while (0)

//==============================================================================
// RADIX-5 OUTPUT COMPUTATION - AVX2 (UNCHANGED)
//==============================================================================

#define RADIX5_BUTTERFLY_FV_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)   \
    do                                                                    \
    {                                                                     \
        const __m256d vc1 = _mm256_set1_pd(C5_1);                         \
        const __m256d vc2 = _mm256_set1_pd(C5_2);                         \
        const __m256d vs1 = _mm256_set1_pd(S5_1);                         \
        const __m256d vs2 = _mm256_set1_pd(S5_2);                         \
        const __m256d rot_mask_fv = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);  \
                                                                          \
        __m256d t0 = _mm256_add_pd(b2, e2);                               \
        __m256d t1 = _mm256_add_pd(c2, d2);                               \
        __m256d t2 = _mm256_sub_pd(b2, e2);                               \
        __m256d t3 = _mm256_sub_pd(c2, d2);                               \
                                                                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                     \
                                                                          \
        __m256d base1 = _mm256_fmadd_pd(vs1, t2, _mm256_mul_pd(vs2, t3)); \
        __m256d tmp1 = _mm256_fmadd_pd(vc1, t0, _mm256_mul_pd(vc2, t1));  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);             \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_fv);               \
        __m256d a1 = _mm256_add_pd(a, tmp1);                              \
        y1 = _mm256_add_pd(a1, r1);                                       \
        y4 = _mm256_sub_pd(a1, r1);                                       \
                                                                          \
        __m256d base2 = _mm256_fmsub_pd(vs2, t2, _mm256_mul_pd(vs1, t3)); \
        __m256d tmp2 = _mm256_fmadd_pd(vc2, t0, _mm256_mul_pd(vc1, t1));  \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);             \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_fv);               \
        __m256d a2 = _mm256_add_pd(a, tmp2);                              \
        y3 = _mm256_add_pd(a2, r2);                                       \
        y2 = _mm256_sub_pd(a2, r2);                                       \
    } while (0)

#define RADIX5_BUTTERFLY_BV_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)   \
    do                                                                    \
    {                                                                     \
        const __m256d vc1 = _mm256_set1_pd(C5_1);                         \
        const __m256d vc2 = _mm256_set1_pd(C5_2);                         \
        const __m256d vs1 = _mm256_set1_pd(S5_1);                         \
        const __m256d vs2 = _mm256_set1_pd(S5_2);                         \
        const __m256d rot_mask_bv = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);  \
                                                                          \
        __m256d t0 = _mm256_add_pd(b2, e2);                               \
        __m256d t1 = _mm256_add_pd(c2, d2);                               \
        __m256d t2 = _mm256_sub_pd(b2, e2);                               \
        __m256d t3 = _mm256_sub_pd(c2, d2);                               \
                                                                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                     \
                                                                          \
        __m256d base1 = _mm256_fmadd_pd(vs1, t2, _mm256_mul_pd(vs2, t3)); \
        __m256d tmp1 = _mm256_fmadd_pd(vc1, t0, _mm256_mul_pd(vc2, t1));  \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);             \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask_bv);               \
        __m256d a1 = _mm256_add_pd(a, tmp1);                              \
        y1 = _mm256_add_pd(a1, r1);                                       \
        y4 = _mm256_sub_pd(a1, r1);                                       \
                                                                          \
        __m256d base2 = _mm256_fmsub_pd(vs2, t2, _mm256_mul_pd(vs1, t3)); \
        __m256d tmp2 = _mm256_fmadd_pd(vc2, t0, _mm256_mul_pd(vc1, t1));  \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);             \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask_bv);               \
        __m256d a2 = _mm256_add_pd(a, tmp2);                              \
        y3 = _mm256_add_pd(a2, r2);                                       \
        y2 = _mm256_sub_pd(a2, r2);                                       \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2 (UNCHANGED)
//==============================================================================

#define LOAD_5_LANES_AVX2(kk, K, sub_outputs, a, b, c, d, e)                       \
    do                                                                             \
    {                                                                              \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                   \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);         \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]); \
        d = load2_aos(&sub_outputs[(kk) + 3 * K], &sub_outputs[(kk) + 1 + 3 * K]); \
        e = load2_aos(&sub_outputs[(kk) + 4 * K], &sub_outputs[(kk) + 1 + 4 * K]); \
    } while (0)

#define STORE_5_LANES_AVX2(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                               \
    {                                                                \
        STOREU_PD(&output_buffer[kk].re, y0);                        \
        STOREU_PD(&output_buffer[(kk) + K].re, y1);                  \
        STOREU_PD(&output_buffer[(kk) + 2 * K].re, y2);              \
        STOREU_PD(&output_buffer[(kk) + 3 * K].re, y3);              \
        STOREU_PD(&output_buffer[(kk) + 4 * K].re, y4);              \
    } while (0)

#define STORE_5_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4) \
    do                                                                      \
    {                                                                       \
        _mm256_stream_pd(&output_buffer[kk].re, y0);                        \
        _mm256_stream_pd(&output_buffer[(kk) + K].re, y1);                  \
        _mm256_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);              \
        _mm256_stream_pd(&output_buffer[(kk) + 3 * K].re, y3);              \
        _mm256_stream_pd(&output_buffer[(kk) + 4 * K].re, y4);              \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2 (SoA Version)
//==============================================================================

#define PREFETCH_L1_R5 8
#define PREFETCH_L2_R5 32
#define PREFETCH_L3_R5 64

#define PREFETCH_5_LANES_R5_AVX2_SOA(k, K, distance, sub_outputs, stage_tw, hint)             \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            /* Prefetch input lanes */                                                        \
            for (int _lane = 0; _lane < 5; _lane++)                                           \
            {                                                                                 \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + _lane * K], hint); \
            }                                                                                 \
            /* Prefetch SoA twiddles (4 pairs) */                                             \
            for (int j = 0; j < 4; j++)                                                       \
            {                                                                                 \
                _mm_prefetch((const char *)&stage_tw->re[j * K + (k) + (distance)], hint);    \
                _mm_prefetch((const char *)&stage_tw->im[j * K + (k) + (distance)], hint);    \
            }                                                                                 \
        }                                                                                     \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT (SoA Version)
//==============================================================================

#define RADIX5_BUTTERFLY_FV_SCALAR(a, b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, \
                                   y0, y1, y2, y3, y4)                        \
    do                                                                        \
    {                                                                         \
        double t0r = b2r + e2r;                                               \
        double t0i = b2i + e2i;                                               \
        double t1r = c2r + d2r;                                               \
        double t1i = c2i + d2i;                                               \
        double t2r = b2r - e2r;                                               \
        double t2i = b2i - e2i;                                               \
        double t3r = c2r - d2r;                                               \
        double t3i = c2i - d2i;                                               \
                                                                              \
        y0.re = a.re + t0r + t1r;                                             \
        y0.im = a.im + t0i + t1i;                                             \
                                                                              \
        double base1r = S5_1 * t2r + S5_2 * t3r;                              \
        double base1i = S5_1 * t2i + S5_2 * t3i;                              \
        double tmp1r = C5_1 * t0r + C5_2 * t1r;                               \
        double tmp1i = C5_1 * t0i + C5_2 * t1i;                               \
        double r1r = base1i;                                                  \
        double r1i = -base1r;                                                 \
        double a1r = a.re + tmp1r;                                            \
        double a1i = a.im + tmp1i;                                            \
        y1.re = a1r + r1r;                                                    \
        y1.im = a1i + r1i;                                                    \
        y4.re = a1r - r1r;                                                    \
        y4.im = a1i - r1i;                                                    \
                                                                              \
        double base2r = S5_2 * t2r - S5_1 * t3r;                              \
        double base2i = S5_2 * t2i - S5_1 * t3i;                              \
        double tmp2r = C5_2 * t0r + C5_1 * t1r;                               \
        double tmp2i = C5_2 * t0i + C5_1 * t1i;                               \
        double r2r = base2i;                                                  \
        double r2i = -base2r;                                                 \
        double a2r = a.re + tmp2r;                                            \
        double a2i = a.im + tmp2i;                                            \
        y3.re = a2r + r2r;                                                    \
        y3.im = a2i + r2i;                                                    \
        y2.re = a2r - r2r;                                                    \
        y2.im = a2i - r2i;                                                    \
    } while (0)

#define RADIX5_BUTTERFLY_BV_SCALAR(a, b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, \
                                   y0, y1, y2, y3, y4)                        \
    do                                                                        \
    {                                                                         \
        double t0r = b2r + e2r;                                               \
        double t0i = b2i + e2i;                                               \
        double t1r = c2r + d2r;                                               \
        double t1i = c2i + d2i;                                               \
        double t2r = b2r - e2r;                                               \
        double t2i = b2i - e2i;                                               \
        double t3r = c2r - d2r;                                               \
        double t3i = c2i - d2i;                                               \
                                                                              \
        y0.re = a.re + t0r + t1r;                                             \
        y0.im = a.im + t0i + t1i;                                             \
                                                                              \
        double base1r = S5_1 * t2r + S5_2 * t3r;                              \
        double base1i = S5_1 * t2i + S5_2 * t3i;                              \
        double tmp1r = C5_1 * t0r + C5_2 * t1r;                               \
        double tmp1i = C5_1 * t0i + C5_2 * t1i;                               \
        double r1r = -base1i;                                                 \
        double r1i = base1r;                                                  \
        double a1r = a.re + tmp1r;                                            \
        double a1i = a.im + tmp1i;                                            \
        y1.re = a1r + r1r;                                                    \
        y1.im = a1i + r1i;                                                    \
        y4.re = a1r - r1r;                                                    \
        y4.im = a1i - r1i;                                                    \
                                                                              \
        double base2r = S5_2 * t2r - S5_1 * t3r;                              \
        double base2i = S5_2 * t2i - S5_1 * t3i;                              \
        double tmp2r = C5_2 * t0r + C5_1 * t1r;                               \
        double tmp2i = C5_2 * t0i + C5_1 * t1i;                               \
        double r2r = -base2i;                                                 \
        double r2i = base2r;                                                  \
        double a2r = a.re + tmp2r;                                            \
        double a2i = a.im + tmp2i;                                            \
        y3.re = a2r + r2r;                                                    \
        y3.im = a2i + r2i;                                                    \
        y2.re = a2r - r2r;                                                    \
        y2.im = a2i - r2i;                                                    \
    } while (0)

/**
 * @brief Apply scalar SoA twiddles
 *
 * OLD: Load from stage_tw[k*4 + j]
 * NEW: Load from tw->re[j*K + k], tw->im[j*K + k]
 */
#define APPLY_STAGE_TWIDDLES_SCALAR_SOA_R5(k, K, b, c, d, e, stage_tw, \
                                           tw_b, tw_c, tw_d, tw_e)     \
    do                                                                 \
    {                                                                  \
        /* W^(1*k) */                                                  \
        double w1_re = stage_tw->re[0 * K + k];                        \
        double w1_im = stage_tw->im[0 * K + k];                        \
        tw_b.re = b.re * w1_re - b.im * w1_im;                         \
        tw_b.im = b.re * w1_im + b.im * w1_re;                         \
                                                                       \
        /* W^(2*k) */                                                  \
        double w2_re = stage_tw->re[1 * K + k];                        \
        double w2_im = stage_tw->im[1 * K + k];                        \
        tw_c.re = c.re * w2_re - c.im * w2_im;                         \
        tw_c.im = c.re * w2_im + c.im * w2_re;                         \
                                                                       \
        /* W^(3*k) */                                                  \
        double w3_re = stage_tw->re[2 * K + k];                        \
        double w3_im = stage_tw->im[2 * K + k];                        \
        tw_d.re = d.re * w3_re - d.im * w3_im;                         \
        tw_d.im = d.re * w3_im + d.im * w3_re;                         \
                                                                       \
        /* W^(4*k) */                                                  \
        double w4_re = stage_tw->re[3 * K + k];                        \
        double w4_im = stage_tw->im[3 * K + k];                        \
        tw_e.re = e.re * w4_re - e.im * w4_im;                         \
        tw_e.im = e.re * w4_im + e.im * w4_re;                         \
    } while (0)

#endif // FFT_RADIX5_MACROS_H