//==============================================================================
// fft_radix4_macros.h - Shared Macros for Radix-4 Butterflies (OPTIMIZED)
//==============================================================================
//
// DESIGN:
// - Small helpers: Could use inline functions (but macros work well here)
// - Large SIMD blocks: Macros (performance)
// - Direction stays in function names (_fv vs _bv)
//

#ifndef FFT_RADIX4_MACROS_H
#define FFT_RADIX4_MACROS_H

#include "fft_radix4.h"
#include "simd_math.h"

//==============================================================================
// STREAMING THRESHOLD
//==============================================================================

#define STREAM_THRESHOLD 8192  // Use non-temporal stores for K >= 8192

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

#define CMUL_FMA_AOS_AVX512(out, a, w)                                    \
    do                                                                    \
    {                                                                     \
        __m512d ar = _mm512_unpacklo_pd(a, a);                            \
        __m512d ai = _mm512_unpackhi_pd(a, a);                            \
        __m512d wr = _mm512_unpacklo_pd(w, w);                            \
        __m512d wi = _mm512_unpackhi_pd(w, w);                            \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));     \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));     \
        (out) = _mm512_unpacklo_pd(re, im);                               \
    } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY CORE - AVX-512
//==============================================================================

#define RADIX4_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d, sumBD, difBD, sumAC, difAC) \
    do                                                                                \
    {                                                                                 \
        sumBD = _mm512_add_pd(tw_b, tw_d);                                            \
        difBD = _mm512_sub_pd(tw_b, tw_d);                                            \
        sumAC = _mm512_add_pd(a, tw_c);                                               \
        difAC = _mm512_sub_pd(a, tw_c);                                               \
    } while (0)

//==============================================================================
// ROTATION - AVX-512
//==============================================================================

#define RADIX4_ROTATE_FORWARD_AVX512(difBD, rot)                                        \
    do                                                                                  \
    {                                                                                   \
        __m512d rot_mask_fv = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0); \
        rot = _mm512_xor_pd(_mm512_permute_pd(difBD, 0b01010101), rot_mask_fv);         \
    } while (0)

#define RADIX4_ROTATE_INVERSE_AVX512(difBD, rot)                                        \
    do                                                                                  \
    {                                                                                   \
        __m512d rot_mask_bv = _mm512_set_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0); \
        rot = _mm512_xor_pd(_mm512_permute_pd(difBD, 0b01010101), rot_mask_bv);         \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512
//==============================================================================

#define RADIX4_ASSEMBLE_OUTPUTS_AVX512(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do                                                                           \
    {                                                                            \
        y0 = _mm512_add_pd(sumAC, sumBD);                                        \
        y2 = _mm512_sub_pd(sumAC, sumBD);                                        \
        y1 = _mm512_sub_pd(difAC, rot);                                          \
        y3 = _mm512_add_pd(difAC, rot);                                          \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX-512
//==============================================================================

#define APPLY_STAGE_TWIDDLES_AVX512(kk, b_vec, c_vec, d_vec, stage_tw,           \
                                    tw_b, tw_c, tw_d)                            \
    do                                                                           \
    {                                                                            \
        __m512d w1 = load4_aos(&stage_tw[(kk) * 3 + 0],                          \
                               &stage_tw[(kk + 1) * 3 + 0],                      \
                               &stage_tw[(kk + 2) * 3 + 0],                      \
                               &stage_tw[(kk + 3) * 3 + 0]);                     \
        __m512d w2 = load4_aos(&stage_tw[(kk) * 3 + 1],                          \
                               &stage_tw[(kk + 1) * 3 + 1],                      \
                               &stage_tw[(kk + 2) * 3 + 1],                      \
                               &stage_tw[(kk + 3) * 3 + 1]);                     \
        __m512d w3 = load4_aos(&stage_tw[(kk) * 3 + 2],                          \
                               &stage_tw[(kk + 1) * 3 + 2],                      \
                               &stage_tw[(kk + 2) * 3 + 2],                      \
                               &stage_tw[(kk + 3) * 3 + 2]);                     \
                                                                                 \
        CMUL_FMA_AOS_AVX512(tw_b, b_vec, w1);                                     \
        CMUL_FMA_AOS_AVX512(tw_c, c_vec, w2);                                     \
        CMUL_FMA_AOS_AVX512(tw_d, d_vec, w3);                                     \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

#define LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d)                           \
    do                                                                                \
    {                                                                                 \
        a = load4_aos(&sub_outputs[kk],                                               \
                      &sub_outputs[(kk) + 1],                                         \
                      &sub_outputs[(kk) + 2],                                         \
                      &sub_outputs[(kk) + 3]);                                        \
        b = load4_aos(&sub_outputs[(kk) + K],                                         \
                      &sub_outputs[(kk) + 1 + K],                                     \
                      &sub_outputs[(kk) + 2 + K],                                     \
                      &sub_outputs[(kk) + 3 + K]);                                    \
        c = load4_aos(&sub_outputs[(kk) + 2 * K],                                     \
                      &sub_outputs[(kk) + 1 + 2 * K],                                 \
                      &sub_outputs[(kk) + 2 + 2 * K],                                 \
                      &sub_outputs[(kk) + 3 + 2 * K]);                                \
        d = load4_aos(&sub_outputs[(kk) + 3 * K],                                     \
                      &sub_outputs[(kk) + 1 + 3 * K],                                 \
                      &sub_outputs[(kk) + 2 + 3 * K],                                 \
                      &sub_outputs[(kk) + 3 + 3 * K]);                                \
    } while (0)

#define STORE_4_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3)      \
    do                                                                  \
    {                                                                   \
        STOREU_PD512(&output_buffer[kk].re, y0);                        \
        STOREU_PD512(&output_buffer[(kk) + K].re, y1);                  \
        STOREU_PD512(&output_buffer[(kk) + 2 * K].re, y2);              \
        STOREU_PD512(&output_buffer[(kk) + 3 * K].re, y3);              \
    } while (0)

#define STORE_4_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3) \
    do                                                                    \
    {                                                                     \
        _mm512_stream_pd(&output_buffer[kk].re, y0);                      \
        _mm512_stream_pd(&output_buffer[(kk) + K].re, y1);                \
        _mm512_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);            \
        _mm512_stream_pd(&output_buffer[(kk) + 3 * K].re, y3);            \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512
//==============================================================================

#define PREFETCH_L1_AVX512 16
#define PREFETCH_TWIDDLE_AVX512 16

#define PREFETCH_4_LANES_AVX512(k, K, distance, sub_outputs, hint)                    \
    do                                                                                \
    {                                                                                 \
        if ((k) + (distance) < K)                                                     \
        {                                                                             \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);         \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 3 * K], hint); \
        }                                                                             \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

#define RADIX4_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                           \
    {                                                                            \
        __m512d a, b, c, d;                                                      \
        LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d);                     \
                                                                                 \
        __m512d tw_b, tw_c, tw_d;                                                \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);    \
                                                                                 \
        __m512d sumBD, difBD, sumAC, difAC;                                      \
        RADIX4_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d,                        \
                                     sumBD, difBD, sumAC, difAC);                \
                                                                                 \
        __m512d rot;                                                             \
        RADIX4_ROTATE_FORWARD_AVX512(difBD, rot);                                \
                                                                                 \
        __m512d y0, y1, y2, y3;                                                  \
        RADIX4_ASSEMBLE_OUTPUTS_AVX512(sumAC, sumBD, difAC, rot,                 \
                                       y0, y1, y2, y3);                          \
                                                                                 \
        STORE_4_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                           \
    {                                                                            \
        __m512d a, b, c, d;                                                      \
        LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d);                     \
                                                                                 \
        __m512d tw_b, tw_c, tw_d;                                                \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);    \
                                                                                 \
        __m512d sumBD, difBD, sumAC, difAC;                                      \
        RADIX4_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d,                        \
                                     sumBD, difBD, sumAC, difAC);                \
                                                                                 \
        __m512d rot;                                                             \
        RADIX4_ROTATE_INVERSE_AVX512(difBD, rot);                                \
                                                                                 \
        __m512d y0, y1, y2, y3;                                                  \
        RADIX4_ASSEMBLE_OUTPUTS_AVX512(sumAC, sumBD, difAC, rot,                 \
                                       y0, y1, y2, y3);                          \
                                                                                 \
        STORE_4_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                                  \
    {                                                                                   \
        __m512d a, b, c, d;                                                             \
        LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d);                            \
                                                                                        \
        __m512d tw_b, tw_c, tw_d;                                                       \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);           \
                                                                                        \
        __m512d sumBD, difBD, sumAC, difAC;                                             \
        RADIX4_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d,                               \
                                     sumBD, difBD, sumAC, difAC);                       \
                                                                                        \
        __m512d rot;                                                                    \
        RADIX4_ROTATE_FORWARD_AVX512(difBD, rot);                                       \
                                                                                        \
        __m512d y0, y1, y2, y3;                                                         \
        RADIX4_ASSEMBLE_OUTPUTS_AVX512(sumAC, sumBD, difAC, rot,                        \
                                       y0, y1, y2, y3);                                 \
                                                                                        \
        STORE_4_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                                  \
    {                                                                                   \
        __m512d a, b, c, d;                                                             \
        LOAD_4_LANES_AVX512(kk, K, sub_outputs, a, b, c, d);                            \
                                                                                        \
        __m512d tw_b, tw_c, tw_d;                                                       \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);           \
                                                                                        \
        __m512d sumBD, difBD, sumAC, difAC;                                             \
        RADIX4_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, tw_d,                               \
                                     sumBD, difBD, sumAC, difAC);                       \
                                                                                        \
        __m512d rot;                                                                    \
        RADIX4_ROTATE_INVERSE_AVX512(difBD, rot);                                       \
                                                                                        \
        __m512d y0, y1, y2, y3;                                                         \
        RADIX4_ASSEMBLE_OUTPUTS_AVX512(sumAC, sumBD, difAC, rot,                        \
                                       y0, y1, y2, y3);                                 \
                                                                                        \
        STORE_4_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX2
//==============================================================================

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

//==============================================================================
// RADIX-4 BUTTERFLY CORE - AVX2
//==============================================================================

#define RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d, sumBD, difBD, sumAC, difAC) \
    do                                                                              \
    {                                                                               \
        sumBD = _mm256_add_pd(tw_b, tw_d);                                          \
        difBD = _mm256_sub_pd(tw_b, tw_d);                                          \
        sumAC = _mm256_add_pd(a, tw_c);                                             \
        difAC = _mm256_sub_pd(a, tw_c);                                             \
    } while (0)

//==============================================================================
// ROTATION - AVX2
//==============================================================================

#define RADIX4_ROTATE_FORWARD_AVX2(difBD, rot)                              \
    do                                                                      \
    {                                                                       \
        __m256d rot_mask_fv = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);          \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_fv); \
    } while (0)

#define RADIX4_ROTATE_INVERSE_AVX2(difBD, rot)                              \
    do                                                                      \
    {                                                                       \
        __m256d rot_mask_bv = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);          \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_bv); \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX2
//==============================================================================

#define RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do                                                                         \
    {                                                                          \
        y0 = _mm256_add_pd(sumAC, sumBD);                                      \
        y2 = _mm256_sub_pd(sumAC, sumBD);                                      \
        y1 = _mm256_sub_pd(difAC, rot);                                        \
        y3 = _mm256_add_pd(difAC, rot);                                        \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX2
//==============================================================================

#define APPLY_STAGE_TWIDDLES_AVX2(kk, b_vec, c_vec, d_vec, stage_tw,                  \
                                  tw_b, tw_c, tw_d)                                   \
    do                                                                                \
    {                                                                                 \
        __m256d w1 = load2_aos(&stage_tw[(kk) * 3 + 0], &stage_tw[(kk + 1) * 3 + 0]); \
        __m256d w2 = load2_aos(&stage_tw[(kk) * 3 + 1], &stage_tw[(kk + 1) * 3 + 1]); \
        __m256d w3 = load2_aos(&stage_tw[(kk) * 3 + 2], &stage_tw[(kk + 1) * 3 + 2]); \
                                                                                      \
        CMUL_FMA_AOS(tw_b, b_vec, w1);                                                \
        CMUL_FMA_AOS(tw_c, c_vec, w2);                                                \
        CMUL_FMA_AOS(tw_d, d_vec, w3);                                                \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2
//==============================================================================

#define LOAD_4_LANES_AVX2(kk, K, sub_outputs, a, b, c, d)                          \
    do                                                                             \
    {                                                                              \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                   \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);         \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]); \
        d = load2_aos(&sub_outputs[(kk) + 3 * K], &sub_outputs[(kk) + 1 + 3 * K]); \
    } while (0)

#define STORE_4_LANES_AVX2(kk, K, output_buffer, y0, y1, y2, y3) \
    do                                                           \
    {                                                            \
        STOREU_PD(&output_buffer[kk].re, y0);                    \
        STOREU_PD(&output_buffer[(kk) + K].re, y1);              \
        STOREU_PD(&output_buffer[(kk) + 2 * K].re, y2);          \
        STOREU_PD(&output_buffer[(kk) + 3 * K].re, y3);          \
    } while (0)

#define STORE_4_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2, y3) \
    do                                                                  \
    {                                                                   \
        _mm256_stream_pd(&output_buffer[kk].re, y0);                    \
        _mm256_stream_pd(&output_buffer[(kk) + K].re, y1);              \
        _mm256_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);          \
        _mm256_stream_pd(&output_buffer[(kk) + 3 * K].re, y3);          \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2
//==============================================================================

#define PREFETCH_L1 8
#define PREFETCH_TWIDDLE 8

#define PREFETCH_4_LANES_AVX2(k, K, distance, sub_outputs, hint)                      \
    do                                                                                \
    {                                                                                 \
        if ((k) + (distance) < K)                                                     \
        {                                                                             \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);         \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 3 * K], hint); \
        }                                                                             \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX2
//==============================================================================

#define RADIX4_PIPELINE_2_FV_AVX2(k, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                      \
        __m256d a, b, c, d;                                                   \
        LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);                     \
                                                                              \
        __m256d tw_b, tw_c, tw_d;                                             \
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);    \
                                                                              \
        __m256d sumBD, difBD, sumAC, difAC;                                   \
        RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d,                       \
                                   sumBD, difBD, sumAC, difAC);               \
                                                                              \
        __m256d rot;                                                          \
        RADIX4_ROTATE_FORWARD_AVX2(difBD, rot);                               \
                                                                              \
        __m256d y0, y1, y2, y3;                                               \
        RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot,                \
                                     y0, y1, y2, y3);                         \
                                                                              \
        STORE_4_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_2_BV_AVX2(k, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                      \
        __m256d a, b, c, d;                                                   \
        LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);                     \
                                                                              \
        __m256d tw_b, tw_c, tw_d;                                             \
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);    \
                                                                              \
        __m256d sumBD, difBD, sumAC, difAC;                                   \
        RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d,                       \
                                   sumBD, difBD, sumAC, difAC);               \
                                                                              \
        __m256d rot;                                                          \
        RADIX4_ROTATE_INVERSE_AVX2(difBD, rot);                               \
                                                                              \
        __m256d y0, y1, y2, y3;                                               \
        RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot,                \
                                     y0, y1, y2, y3);                         \
                                                                              \
        STORE_4_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_2_FV_AVX2_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                             \
        __m256d a, b, c, d;                                                          \
        LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);                            \
                                                                                     \
        __m256d tw_b, tw_c, tw_d;                                                    \
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);           \
                                                                                     \
        __m256d sumBD, difBD, sumAC, difAC;                                          \
        RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d,                              \
                                   sumBD, difBD, sumAC, difAC);                      \
                                                                                     \
        __m256d rot;                                                                 \
        RADIX4_ROTATE_FORWARD_AVX2(difBD, rot);                                      \
                                                                                     \
        __m256d y0, y1, y2, y3;                                                      \
        RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot,                       \
                                     y0, y1, y2, y3);                                \
                                                                                     \
        STORE_4_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#define RADIX4_PIPELINE_2_BV_AVX2_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                             \
        __m256d a, b, c, d;                                                          \
        LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);                            \
                                                                                     \
        __m256d tw_b, tw_c, tw_d;                                                    \
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);           \
                                                                                     \
        __m256d sumBD, difBD, sumAC, difAC;                                          \
        RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d,                              \
                                   sumBD, difBD, sumAC, difAC);                      \
                                                                                     \
        __m256d rot;                                                                 \
        RADIX4_ROTATE_INVERSE_AVX2(difBD, rot);                                      \
                                                                                     \
        __m256d y0, y1, y2, y3;                                                      \
        RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot,                       \
                                     y0, y1, y2, y3);                                \
                                                                                     \
        STORE_4_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3);              \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

#define RADIX4_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, tw_d,                    \
                                     sumBD_re, sumBD_im, difBD_re, difBD_im, \
                                     sumAC_re, sumAC_im, difAC_re, difAC_im) \
    do                                                                       \
    {                                                                        \
        sumBD_re = tw_b.re + tw_d.re;                                        \
        sumBD_im = tw_b.im + tw_d.im;                                        \
        difBD_re = tw_b.re - tw_d.re;                                        \
        difBD_im = tw_b.im - tw_d.im;                                        \
        sumAC_re = a.re + tw_c.re;                                           \
        sumAC_im = a.im + tw_c.im;                                           \
        difAC_re = a.re - tw_c.re;                                           \
        difAC_im = a.im - tw_c.im;                                           \
    } while (0)

#define RADIX4_ROTATE_FORWARD_SCALAR(difBD_re, difBD_im, rot_re, rot_im) \
    do                                                                   \
    {                                                                    \
        rot_re = difBD_im;                                               \
        rot_im = -difBD_re;                                              \
    } while (0)

#define RADIX4_ROTATE_INVERSE_SCALAR(difBD_re, difBD_im, rot_re, rot_im) \
    do                                                                   \
    {                                                                    \
        rot_re = -difBD_im;                                              \
        rot_im = difBD_re;                                               \
    } while (0)

#define RADIX4_ASSEMBLE_OUTPUTS_SCALAR(sumAC_re, sumAC_im, sumBD_re, sumBD_im, \
                                       difAC_re, difAC_im, rot_re, rot_im,     \
                                       y0, y1, y2, y3)                         \
    do                                                                         \
    {                                                                          \
        y0.re = sumAC_re + sumBD_re;                                           \
        y0.im = sumAC_im + sumBD_im;                                           \
        y2.re = sumAC_re - sumBD_re;                                           \
        y2.im = sumAC_im - sumBD_im;                                           \
        y1.re = difAC_re - rot_re;                                             \
        y1.im = difAC_im - rot_im;                                             \
        y3.re = difAC_re + rot_re;                                             \
        y3.im = difAC_im + rot_im;                                             \
    } while (0)

#define APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, d, stage_tw, tw_b, tw_c, tw_d) \
    do                                                                      \
    {                                                                       \
        const fft_data *w_ptr = &stage_tw[(k) * 3];                         \
                                                                            \
        tw_b.re = b.re * w_ptr[0].re - b.im * w_ptr[0].im;                  \
        tw_b.im = b.re * w_ptr[0].im + b.im * w_ptr[0].re;                  \
                                                                            \
        tw_c.re = c.re * w_ptr[1].re - c.im * w_ptr[1].im;                  \
        tw_c.im = c.re * w_ptr[1].im + c.im * w_ptr[1].re;                  \
                                                                            \
        tw_d.re = d.re * w_ptr[2].re - d.im * w_ptr[2].im;                  \
        tw_d.im = d.re * w_ptr[2].im + d.im * w_ptr[2].re;                  \
    } while (0)

#define RADIX4_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                         \
    {                                                                          \
        fft_data a = sub_outputs[k];                                           \
        fft_data b = sub_outputs[k + K];                                       \
        fft_data c = sub_outputs[k + 2 * K];                                   \
        fft_data d = sub_outputs[k + 3 * K];                                   \
                                                                               \
        fft_data tw_b, tw_c, tw_d;                                             \
        APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);   \
                                                                               \
        double sumBD_re, sumBD_im, difBD_re, difBD_im;                         \
        double sumAC_re, sumAC_im, difAC_re, difAC_im;                         \
        RADIX4_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, tw_d,                      \
                                     sumBD_re, sumBD_im, difBD_re, difBD_im,   \
                                     sumAC_re, sumAC_im, difAC_re, difAC_im);  \
                                                                               \
        double rot_re, rot_im;                                                 \
        RADIX4_ROTATE_FORWARD_SCALAR(difBD_re, difBD_im, rot_re, rot_im);      \
                                                                               \
        fft_data y0, y1, y2, y3;                                               \
        RADIX4_ASSEMBLE_OUTPUTS_SCALAR(sumAC_re, sumAC_im, sumBD_re, sumBD_im, \
                                       difAC_re, difAC_im, rot_re, rot_im,     \
                                       y0, y1, y2, y3);                        \
                                                                               \
        output_buffer[k] = y0;                                                 \
        output_buffer[k + K] = y1;                                             \
        output_buffer[k + 2 * K] = y2;                                         \
        output_buffer[k + 3 * K] = y3;                                         \
    } while (0)

#define RADIX4_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer) \
    do                                                                         \
    {                                                                          \
        fft_data a = sub_outputs[k];                                           \
        fft_data b = sub_outputs[k + K];                                       \
        fft_data c = sub_outputs[k + 2 * K];                                   \
        fft_data d = sub_outputs[k + 3 * K];                                   \
                                                                               \
        fft_data tw_b, tw_c, tw_d;                                             \
        APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);   \
                                                                               \
        double sumBD_re, sumBD_im, difBD_re, difBD_im;                         \
        double sumAC_re, sumAC_im, difAC_re, difAC_im;                         \
        RADIX4_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, tw_d,                      \
                                     sumBD_re, sumBD_im, difBD_re, difBD_im,   \
                                     sumAC_re, sumAC_im, difAC_re, difAC_im);  \
                                                                               \
        double rot_re, rot_im;                                                 \
        RADIX4_ROTATE_INVERSE_SCALAR(difBD_re, difBD_im, rot_re, rot_im);      \
                                                                               \
        fft_data y0, y1, y2, y3;                                               \
        RADIX4_ASSEMBLE_OUTPUTS_SCALAR(sumAC_re, sumAC_im, sumBD_re, sumBD_im, \
                                       difAC_re, difAC_im, rot_re, rot_im,     \
                                       y0, y1, y2, y3);                        \
                                                                               \
        output_buffer[k] = y0;                                                 \
        output_buffer[k + K] = y1;                                             \
        output_buffer[k + 2 * K] = y2;                                         \
        output_buffer[k + 3 * K] = y3;                                         \
    } while (0)

#endif // FFT_RADIX4_MACROS_H
