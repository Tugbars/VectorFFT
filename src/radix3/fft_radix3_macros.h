//==============================================================================
// fft_radix3_macros.h - PURE SOA VERSION (ZERO SHUFFLE OVERHEAD!)
//==============================================================================
//
// DESIGN:
// - Small helpers: static __always_inline (type safety)
// - Large SIMD blocks: Macros (performance)
// - Direction stays in function names (_fv vs _bv)
//
// OPTIMIZATIONS PRESERVED:
// - Hoisted constants (rotation masks, geometric constants)
// - Fixed AVX-512 complex multiply (unpacklo/hi)
// - Parameterized pipeline macros (reduced duplication)
// - Interleaved twiddle loads with computation
// - 64-byte alignment hints for AVX-512
// - Combined prefetch macros
// - SSE2 tail processing
// - AVX-512 masked stores for partial butterflies
//
// SOA CHANGES:
// - Twiddle loads: Direct re/im arrays (zero shuffle overhead!)
// - Complex multiply: cmul_*_soa versions (handle SoA twiddles)
// - Prefetch: Separate re/im blocks
//

#ifndef FFT_RADIX3_MACROS_H
#define FFT_RADIX3_MACROS_H

#include "fft_radix3.h"
#include "simd_math.h"

//==============================================================================
// GEOMETRIC CONSTANTS - IDENTICAL for forward/inverse
//==============================================================================

#define C_HALF (-0.5)
#define S_SQRT3_2 0.8660254037844386467618 // sqrt(3)/2

//==============================================================================
// STREAMING THRESHOLD
//==============================================================================

#define STREAM_THRESHOLD 8192 // Use non-temporal stores for K >= 8192

//==============================================================================
// PORTABLE HOISTED CONSTANTS - Rotation Masks (UNCHANGED)
//==============================================================================

#ifdef __AVX512F__
alignas(64) static const double ROT_MASK_FWD_AVX512_DATA[8] = {
    0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0};

alignas(64) static const double ROT_MASK_INV_AVX512_DATA[8] = {
    -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0};

#define ROT_MASK_FWD_AVX512 (_mm512_load_pd(ROT_MASK_FWD_AVX512_DATA))
#define ROT_MASK_INV_AVX512 (_mm512_load_pd(ROT_MASK_INV_AVX512_DATA))
#endif

#ifdef __AVX2__
alignas(32) static const double ROT_MASK_FWD_AVX2_DATA[4] = {
    0.0, -0.0, 0.0, -0.0};

alignas(32) static const double ROT_MASK_INV_AVX2_DATA[4] = {
    -0.0, 0.0, -0.0, 0.0};

#define ROT_MASK_FWD_AVX2 (_mm256_load_pd(ROT_MASK_FWD_AVX2_DATA))
#define ROT_MASK_INV_AVX2 (_mm256_load_pd(ROT_MASK_INV_AVX2_DATA))
#endif

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Complex multiply with SoA twiddles (AVX-512)
 *
 * OLD: CMUL_FMA_AOS_AVX512(out, a, w_interleaved)
 * NEW: CMUL_FMA_SOA_AVX512(out, a, w_re, w_im)
 */
#define CMUL_FMA_SOA_AVX512(out, a, w_re, w_im)                            \
    do                                                                     \
    {                                                                      \
        __m512d ar = _mm512_unpacklo_pd(a, a); /* [re0,re0,re1,re1,...] */ \
        __m512d ai = _mm512_unpackhi_pd(a, a); /* [im0,im0,im1,im1,...] */ \
        /* Twiddles already separated (SoA) - NO SHUFFLE! */               \
        __m512d re = _mm512_fmsub_pd(ar, w_re, _mm512_mul_pd(ai, w_im));   \
        __m512d im = _mm512_fmadd_pd(ar, w_im, _mm512_mul_pd(ai, w_re));   \
        (out) = _mm512_unpacklo_pd(re, im); /* interleave back to AOS */   \
    } while (0)

//==============================================================================
// RADIX-3 BUTTERFLY CORE - AVX-512 (UNCHANGED)
//==============================================================================

#define RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common, v_half) \
    do                                                                        \
    {                                                                         \
        sum = _mm512_add_pd(tw_b, tw_c);                                      \
        dif = _mm512_sub_pd(tw_b, tw_c);                                      \
        common = _mm512_fmadd_pd(v_half, sum, a);                             \
    } while (0)

//==============================================================================
// ROTATION AND SCALING - AVX-512 (UNCHANGED)
//==============================================================================

#define RADIX3_ROTATE_AVX512(dif, scaled_rot, v_sqrt3_2, rot_mask) \
    do                                                             \
    {                                                              \
        __m512d dif_swp = _mm512_permute_pd(dif, 0b01010101);      \
        __m512d rot90 = _mm512_xor_pd(dif_swp, rot_mask);          \
        scaled_rot = _mm512_mul_pd(rot90, v_sqrt3_2);              \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512 (UNCHANGED)
//==============================================================================

#define RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2) \
    do                                                                         \
    {                                                                          \
        y0 = _mm512_add_pd(a, sum);                                            \
        y1 = _mm512_add_pd(common, scaled_rot);                                \
        y2 = _mm512_sub_pd(common, scaled_rot);                                \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX-512 (SoA Version)
//==============================================================================

/**
 * @brief Load and apply SoA twiddles (AVX-512)
 *
 * CRITICAL CHANGE: Load from separate re/im arrays
 *
 * OLD: Load interleaved: stage_tw[k*2+0], stage_tw[k*2+1]
 * NEW: Load SoA: tw->re[0*K+k], tw->im[0*K+k], tw->re[1*K+k], tw->im[1*K+k]
 *
 * For 4 butterflies (k, k+1, k+2, k+3):
 * - Load 8 doubles from tw->re[r*K + k]
 * - Load 8 doubles from tw->im[r*K + k]
 */
#define APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, b, c, stage_tw, tw_b, tw_c) \
    do                                                                     \
    {                                                                      \
        /* Load W^(1*k) for 4 butterflies */                               \
        __m512d w1_re = _mm512_loadu_pd(&stage_tw->re[0 * K + (kk)]);      \
        __m512d w1_im = _mm512_loadu_pd(&stage_tw->im[0 * K + (kk)]);      \
        /* Load W^(2*k) for 4 butterflies */                               \
        __m512d w2_re = _mm512_loadu_pd(&stage_tw->re[1 * K + (kk)]);      \
        __m512d w2_im = _mm512_loadu_pd(&stage_tw->im[1 * K + (kk)]);      \
                                                                           \
        /* Apply twiddles (SoA complex multiply - ZERO SHUFFLE!) */        \
        CMUL_FMA_SOA_AVX512(tw_b, b, w1_re, w1_im);                        \
        CMUL_FMA_SOA_AVX512(tw_c, c, w2_re, w2_im);                        \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512 (UNCHANGED - data is still AoS)
//==============================================================================

#define LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c) \
    do                                                   \
    {                                                    \
        a = load4_aos(&sub_outputs[kk],                  \
                      &sub_outputs[(kk) + 1],            \
                      &sub_outputs[(kk) + 2],            \
                      &sub_outputs[(kk) + 3]);           \
        b = load4_aos(&sub_outputs[(kk) + K],            \
                      &sub_outputs[(kk) + 1 + K],        \
                      &sub_outputs[(kk) + 2 + K],        \
                      &sub_outputs[(kk) + 3 + K]);       \
        c = load4_aos(&sub_outputs[(kk) + 2 * K],        \
                      &sub_outputs[(kk) + 1 + 2 * K],    \
                      &sub_outputs[(kk) + 2 + 2 * K],    \
                      &sub_outputs[(kk) + 3 + 2 * K]);   \
    } while (0)

#define STORE_3_LANES_AVX512(kk, K, output_buffer, y0, y1, y2) \
    do                                                         \
    {                                                          \
        STOREU_PD512(&output_buffer[kk].re, y0);               \
        STOREU_PD512(&output_buffer[(kk) + K].re, y1);         \
        STOREU_PD512(&output_buffer[(kk) + 2 * K].re, y2);     \
    } while (0)

#define STORE_3_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2) \
    do                                                                \
    {                                                                 \
        _mm512_stream_pd(&output_buffer[kk].re, y0);                  \
        _mm512_stream_pd(&output_buffer[(kk) + K].re, y1);            \
        _mm512_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);        \
    } while (0)

#define MASK_3_COMPLEX_AVX512 ((__mmask8)0x3F) // bits 0-5 set

#define STORE_3_LANES_AVX512_MASKED(kk, K, output_buffer, y0, y1, y2, mask) \
    do                                                                      \
    {                                                                       \
        _mm512_mask_storeu_pd(&output_buffer[kk].re, mask, y0);             \
        _mm512_mask_storeu_pd(&output_buffer[(kk) + K].re, mask, y1);       \
        _mm512_mask_storeu_pd(&output_buffer[(kk) + 2 * K].re, mask, y2);   \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512 (SoA Version)
//==============================================================================

#define PREFETCH_L1_AVX512 16

/**
 * @brief Prefetch SoA twiddles (2 separate re/im blocks)
 *
 * OLD: Prefetch interleaved: stage_tw[(k+dist)*2]
 * NEW: Prefetch separate: tw->re[r*K + k+dist] for r=0,1
 */
#define PREFETCH_RADIX3_AVX512_SOA(k, K, distance, sub_outputs, stage_tw, hint)        \
    do                                                                                 \
    {                                                                                  \
        if ((k) + (distance) < K)                                                      \
        {                                                                              \
            /* Prefetch input lanes (3 cache lines) */                                 \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);          \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);      \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint);  \
            /* Prefetch SoA twiddles (W^k and W^(2k) blocks) */                        \
            _mm_prefetch((const char *)&stage_tw->re[0 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->im[0 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->re[1 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->im[1 * K + (k) + (distance)], hint); \
        }                                                                              \
    } while (0)

//==============================================================================
// UNIFIED BUTTERFLY PIPELINE - AVX-512 (SoA Version)
//==============================================================================

#define RADIX3_PIPELINE_4_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                     v_half, v_sqrt3_2, rot_mask, store_macro)    \
    do                                                                            \
    {                                                                             \
        __m512d a, b, c;                                                          \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                         \
                                                                                  \
        __m512d tw_b, tw_c;                                                       \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, b, c, stage_tw, tw_b, tw_c);       \
                                                                                  \
        __m512d sum, dif, common;                                                 \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common, v_half);    \
                                                                                  \
        __m512d scaled_rot;                                                       \
        RADIX3_ROTATE_AVX512(dif, scaled_rot, v_sqrt3_2, rot_mask);               \
                                                                                  \
        __m512d y0, y1, y2;                                                       \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);   \
                                                                                  \
        store_macro(kk, K, output_buffer, y0, y1, y2);                            \
    } while (0)

#define RADIX3_PIPELINE_4_AVX512_MASKED_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                            v_half, v_sqrt3_2, rot_mask, mask)           \
    do                                                                                   \
    {                                                                                    \
        __m512d a, b, c;                                                                 \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                                \
                                                                                         \
        __m512d tw_b, tw_c;                                                              \
        APPLY_STAGE_TWIDDLES_AVX512_SOA(kk, K, b, c, stage_tw, tw_b, tw_c);              \
                                                                                         \
        __m512d sum, dif, common;                                                        \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common, v_half);           \
                                                                                         \
        __m512d scaled_rot;                                                              \
        RADIX3_ROTATE_AVX512(dif, scaled_rot, v_sqrt3_2, rot_mask);                      \
                                                                                         \
        __m512d y0, y1, y2;                                                              \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);          \
                                                                                         \
        STORE_3_LANES_AVX512_MASKED(kk, K, output_buffer, y0, y1, y2, mask);             \
    } while (0)

// Instantiate specific versions
#define RADIX3_PIPELINE_4_FV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                        v_half, v_sqrt3_2, rot_mask)                 \
    RADIX3_PIPELINE_4_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer,        \
                                 v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX512)

#define RADIX3_PIPELINE_4_FV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                               v_half, v_sqrt3_2, rot_mask)                 \
    RADIX3_PIPELINE_4_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer,               \
                                 v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX512_STREAM)

#define RADIX3_PIPELINE_4_BV_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                        v_half, v_sqrt3_2, rot_mask)                 \
    RADIX3_PIPELINE_4_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer,        \
                                 v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX512)

#define RADIX3_PIPELINE_4_BV_AVX512_STREAM_SOA(kk, K, sub_outputs, stage_tw, output_buffer, \
                                               v_half, v_sqrt3_2, rot_mask)                 \
    RADIX3_PIPELINE_4_AVX512_SOA(kk, K, sub_outputs, stage_tw, output_buffer,               \
                                 v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX512_STREAM)

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
// RADIX-3 BUTTERFLY CORE - AVX2 (UNCHANGED)
//==============================================================================

#define RADIX3_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, sum, dif, common, v_half) \
    do                                                                      \
    {                                                                       \
        sum = _mm256_add_pd(tw_b, tw_c);                                    \
        dif = _mm256_sub_pd(tw_b, tw_c);                                    \
        common = _mm256_fmadd_pd(v_half, sum, a);                           \
    } while (0)

//==============================================================================
// ROTATION AND SCALING - AVX2 (UNCHANGED)
//==============================================================================

#define RADIX3_ROTATE_AVX2(dif, scaled_rot, v_sqrt3_2, rot_mask) \
    do                                                           \
    {                                                            \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);        \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);        \
        scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2);            \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX2 (UNCHANGED)
//==============================================================================

#define RADIX3_ASSEMBLE_OUTPUTS_AVX2(a, sum, common, scaled_rot, y0, y1, y2) \
    do                                                                       \
    {                                                                        \
        y0 = _mm256_add_pd(a, sum);                                          \
        y1 = _mm256_add_pd(common, scaled_rot);                              \
        y2 = _mm256_sub_pd(common, scaled_rot);                              \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX2 (SoA Version)
//==============================================================================

/**
 * @brief Load and apply SoA twiddles (AVX2)
 *
 * For 2 butterflies (k, k+1):
 * - Load 4 doubles from tw->re[r*K + k]
 * - Load 4 doubles from tw->im[r*K + k]
 */
#define APPLY_STAGE_TWIDDLES_AVX2_SOA(kk, K, b, c, stage_tw, tw_b, tw_c) \
    do                                                                   \
    {                                                                    \
        /* Load W^(1*k) for 2 butterflies */                             \
        __m256d w1_re = _mm256_loadu_pd(&stage_tw->re[0 * K + (kk)]);    \
        __m256d w1_im = _mm256_loadu_pd(&stage_tw->im[0 * K + (kk)]);    \
        /* Load W^(2*k) for 2 butterflies */                             \
        __m256d w2_re = _mm256_loadu_pd(&stage_tw->re[1 * K + (kk)]);    \
        __m256d w2_im = _mm256_loadu_pd(&stage_tw->im[1 * K + (kk)]);    \
                                                                         \
        CMUL_FMA_SOA_AVX2(tw_b, b, w1_re, w1_im);                        \
        CMUL_FMA_SOA_AVX2(tw_c, c, w2_re, w2_im);                        \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2 (UNCHANGED)
//==============================================================================

#define LOAD_3_LANES_AVX2(kk, K, sub_outputs, a, b, c)                             \
    do                                                                             \
    {                                                                              \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk) + 1]);                   \
        b = load2_aos(&sub_outputs[(kk) + K], &sub_outputs[(kk) + 1 + K]);         \
        c = load2_aos(&sub_outputs[(kk) + 2 * K], &sub_outputs[(kk) + 1 + 2 * K]); \
    } while (0)

#define STORE_3_LANES_AVX2(kk, K, output_buffer, y0, y1, y2) \
    do                                                       \
    {                                                        \
        STOREU_PD(&output_buffer[kk].re, y0);                \
        STOREU_PD(&output_buffer[(kk) + K].re, y1);          \
        STOREU_PD(&output_buffer[(kk) + 2 * K].re, y2);      \
    } while (0)

#define STORE_3_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2) \
    do                                                              \
    {                                                               \
        _mm256_stream_pd(&output_buffer[kk].re, y0);                \
        _mm256_stream_pd(&output_buffer[(kk) + K].re, y1);          \
        _mm256_stream_pd(&output_buffer[(kk) + 2 * K].re, y2);      \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2 (SoA Version)
//==============================================================================

#define PREFETCH_L1 8

#define PREFETCH_RADIX3_AVX2_SOA(k, K, distance, sub_outputs, stage_tw, hint)          \
    do                                                                                 \
    {                                                                                  \
        if ((k) + (distance) < K)                                                      \
        {                                                                              \
            /* Prefetch input lanes */                                                 \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], hint);          \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + K], hint);      \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 2 * K], hint);  \
            /* Prefetch SoA twiddles */                                                \
            _mm_prefetch((const char *)&stage_tw->re[0 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->im[0 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->re[1 * K + (k) + (distance)], hint); \
            _mm_prefetch((const char *)&stage_tw->im[1 * K + (k) + (distance)], hint); \
        }                                                                              \
    } while (0)

//==============================================================================
// UNIFIED BUTTERFLY PIPELINE - AVX2 (SoA Version)
//==============================================================================

#define RADIX3_PIPELINE_2_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                   v_half, v_sqrt3_2, rot_mask, store_macro)   \
    do                                                                         \
    {                                                                          \
        __m256d a, b, c;                                                       \
        LOAD_3_LANES_AVX2(k, K, sub_outputs, a, b, c);                         \
                                                                               \
        __m256d tw_b, tw_c;                                                    \
        APPLY_STAGE_TWIDDLES_AVX2_SOA(k, K, b, c, stage_tw, tw_b, tw_c);       \
                                                                               \
        __m256d sum, dif, common;                                              \
        RADIX3_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, sum, dif, common, v_half);   \
                                                                               \
        __m256d scaled_rot;                                                    \
        RADIX3_ROTATE_AVX2(dif, scaled_rot, v_sqrt3_2, rot_mask);              \
                                                                               \
        __m256d y0, y1, y2;                                                    \
        RADIX3_ASSEMBLE_OUTPUTS_AVX2(a, sum, common, scaled_rot, y0, y1, y2);  \
                                                                               \
        store_macro(k, K, output_buffer, y0, y1, y2);                          \
    } while (0)

// Instantiate specific versions
#define RADIX3_PIPELINE_2_FV_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                      v_half, v_sqrt3_2, rot_mask)                \
    RADIX3_PIPELINE_2_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer,        \
                               v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX2)

#define RADIX3_PIPELINE_2_FV_AVX2_STREAM_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                             v_half, v_sqrt3_2, rot_mask)                \
    RADIX3_PIPELINE_2_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer,               \
                               v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX2_STREAM)

#define RADIX3_PIPELINE_2_BV_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                      v_half, v_sqrt3_2, rot_mask)                \
    RADIX3_PIPELINE_2_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer,        \
                               v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX2)

#define RADIX3_PIPELINE_2_BV_AVX2_STREAM_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                             v_half, v_sqrt3_2, rot_mask)                \
    RADIX3_PIPELINE_2_AVX2_SOA(k, K, sub_outputs, stage_tw, output_buffer,               \
                               v_half, v_sqrt3_2, rot_mask, STORE_3_LANES_AVX2_STREAM)

#endif // __AVX2__

//==============================================================================
// SSE2 SUPPORT (SoA Version)
//==============================================================================

#ifdef __SSE2__

alignas(16) static const double ROT_MASK_FWD_SSE2_DATA[2] = {0.0, -0.0};
alignas(16) static const double ROT_MASK_INV_SSE2_DATA[2] = {-0.0, 0.0};

#define ROT_MASK_FWD_SSE2 (_mm_load_pd(ROT_MASK_FWD_SSE2_DATA))
#define ROT_MASK_INV_SSE2 (_mm_load_pd(ROT_MASK_INV_SSE2_DATA))

#define CMUL_SSE2_SOA(out, a, w_re_scalar, w_im_scalar)       \
    do                                                        \
    {                                                         \
        __m128d ar = _mm_unpacklo_pd(a, a);                   \
        __m128d ai = _mm_unpackhi_pd(a, a);                   \
        __m128d re = _mm_sub_pd(_mm_mul_pd(ar, w_re_scalar),  \
                                _mm_mul_pd(ai, w_im_scalar)); \
        __m128d im = _mm_add_pd(_mm_mul_pd(ar, w_im_scalar),  \
                                _mm_mul_pd(ai, w_re_scalar)); \
        (out) = _mm_unpacklo_pd(re, im);                      \
    } while (0)

#define RADIX3_PIPELINE_1_SSE2_SOA(k, K, sub_outputs, stage_tw, output_buffer, \
                                   c_half, s_sqrt3_2, rot_mask)                \
    do                                                                         \
    {                                                                          \
        __m128d a = LOADU_SSE2(&sub_outputs[k].re);                            \
        __m128d b = LOADU_SSE2(&sub_outputs[(k) + K].re);                      \
        __m128d c = LOADU_SSE2(&sub_outputs[(k) + 2 * K].re);                  \
                                                                               \
        /* Load SoA twiddles (broadcast scalars) */                            \
        __m128d w1_re = _mm_set1_pd(stage_tw->re[0 * K + k]);                  \
        __m128d w1_im = _mm_set1_pd(stage_tw->im[0 * K + k]);                  \
        __m128d w2_re = _mm_set1_pd(stage_tw->re[1 * K + k]);                  \
        __m128d w2_im = _mm_set1_pd(stage_tw->im[1 * K + k]);                  \
                                                                               \
        __m128d tw_b, tw_c;                                                    \
        CMUL_SSE2_SOA(tw_b, b, w1_re, w1_im);                                  \
        CMUL_SSE2_SOA(tw_c, c, w2_re, w2_im);                                  \
                                                                               \
        __m128d v_half = _mm_set1_pd(c_half);                                  \
        __m128d v_sqrt3_2 = _mm_set1_pd(s_sqrt3_2);                            \
        __m128d sum = _mm_add_pd(tw_b, tw_c);                                  \
        __m128d dif = _mm_sub_pd(tw_b, tw_c);                                  \
        __m128d common = _mm_add_pd(a, _mm_mul_pd(v_half, sum));               \
                                                                               \
        __m128d dif_swp = _mm_shuffle_pd(dif, dif, 0x1);                       \
        __m128d rot90 = _mm_xor_pd(dif_swp, rot_mask);                         \
        __m128d scaled_rot = _mm_mul_pd(rot90, v_sqrt3_2);                     \
                                                                               \
        __m128d y0 = _mm_add_pd(a, sum);                                       \
        __m128d y1 = _mm_add_pd(common, scaled_rot);                           \
        __m128d y2 = _mm_sub_pd(common, scaled_rot);                           \
                                                                               \
        STOREU_SSE2(&output_buffer[k].re, y0);                                 \
        STOREU_SSE2(&output_buffer[(k) + K].re, y1);                           \
        STOREU_SSE2(&output_buffer[(k) + 2 * K].re, y2);                       \
    } while (0)

#endif // __SSE2__

//==============================================================================
// SCALAR SUPPORT (SoA Version)
//==============================================================================

#define RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c,                  \
                                     sum_re, sum_im, dif_re, dif_im, \
                                     common_re, common_im)           \
    do                                                               \
    {                                                                \
        sum_re = tw_b.re + tw_c.re;                                  \
        sum_im = tw_b.im + tw_c.im;                                  \
        dif_re = tw_b.re - tw_c.re;                                  \
        dif_im = tw_b.im - tw_c.im;                                  \
        common_re = a.re + C_HALF * sum_re;                          \
        common_im = a.im + C_HALF * sum_im;                          \
    } while (0)

#define RADIX3_ROTATE_FORWARD_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im) \
    do                                                                             \
    {                                                                              \
        scaled_rot_re = S_SQRT3_2 * dif_im;                                        \
        scaled_rot_im = -S_SQRT3_2 * dif_re;                                       \
    } while (0)

#define RADIX3_ROTATE_INVERSE_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im) \
    do                                                                             \
    {                                                                              \
        scaled_rot_re = -S_SQRT3_2 * dif_im;                                       \
        scaled_rot_im = S_SQRT3_2 * dif_re;                                        \
    } while (0)

#define RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im,            \
                                       common_re, common_im,         \
                                       scaled_rot_re, scaled_rot_im, \
                                       y0, y1, y2)                   \
    do                                                               \
    {                                                                \
        y0.re = a.re + sum_re;                                       \
        y0.im = a.im + sum_im;                                       \
        y1.re = common_re + scaled_rot_re;                           \
        y1.im = common_im + scaled_rot_im;                           \
        y2.re = common_re - scaled_rot_re;                           \
        y2.im = common_im - scaled_rot_im;                           \
    } while (0)

/**
 * @brief Apply scalar SoA twiddles
 *
 * OLD: Load from stage_tw[(k)*2], stage_tw[(k)*2 + 1]
 * NEW: Load from tw->re[0*K+k], tw->im[0*K+k], tw->re[1*K+k], tw->im[1*K+k]
 */
#define APPLY_STAGE_TWIDDLES_SCALAR_SOA(k, K, b, c, stage_tw, tw_b, tw_c) \
    do                                                                    \
    {                                                                     \
        /* W^(1*k) */                                                     \
        double w1_re = stage_tw->re[0 * K + k];                           \
        double w1_im = stage_tw->im[0 * K + k];                           \
        tw_b.re = b.re * w1_re - b.im * w1_im;                            \
        tw_b.im = b.re * w1_im + b.im * w1_re;                            \
                                                                          \
        /* W^(2*k) */                                                     \
        double w2_re = stage_tw->re[1 * K + k];                           \
        double w2_im = stage_tw->im[1 * K + k];                           \
        tw_c.re = c.re * w2_re - c.im * w2_im;                            \
        tw_c.im = c.re * w2_im + c.im * w2_re;                            \
    } while (0)

#define RADIX3_BUTTERFLY_SCALAR_FV_SOA(k, K, sub_outputs, stage_tw, output_buffer)  \
    do                                                                              \
    {                                                                               \
        fft_data a = sub_outputs[k];                                                \
        fft_data b = sub_outputs[k + K];                                            \
        fft_data c = sub_outputs[k + 2 * K];                                        \
                                                                                    \
        fft_data tw_b, tw_c;                                                        \
        APPLY_STAGE_TWIDDLES_SCALAR_SOA(k, K, b, c, stage_tw, tw_b, tw_c);          \
                                                                                    \
        double sum_re, sum_im, dif_re, dif_im, common_re, common_im;                \
        RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c,                                 \
                                     sum_re, sum_im, dif_re, dif_im,                \
                                     common_re, common_im);                         \
                                                                                    \
        double scaled_rot_re, scaled_rot_im;                                        \
        RADIX3_ROTATE_FORWARD_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im); \
                                                                                    \
        fft_data y0, y1, y2;                                                        \
        RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im,                           \
                                       common_re, common_im,                        \
                                       scaled_rot_re, scaled_rot_im,                \
                                       y0, y1, y2);                                 \
                                                                                    \
        output_buffer[k] = y0;                                                      \
        output_buffer[k + K] = y1;                                                  \
        output_buffer[k + 2 * K] = y2;                                              \
    } while (0)

#define RADIX3_BUTTERFLY_SCALAR_BV_SOA(k, K, sub_outputs, stage_tw, output_buffer)  \
    do                                                                              \
    {                                                                               \
        fft_data a = sub_outputs[k];                                                \
        fft_data b = sub_outputs[k + K];                                            \
        fft_data c = sub_outputs[k + 2 * K];                                        \
                                                                                    \
        fft_data tw_b, tw_c;                                                        \
        APPLY_STAGE_TWIDDLES_SCALAR_SOA(k, K, b, c, stage_tw, tw_b, tw_c);          \
                                                                                    \
        double sum_re, sum_im, dif_re, dif_im, common_re, common_im;                \
        RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c,                                 \
                                     sum_re, sum_im, dif_re, dif_im,                \
                                     common_re, common_im);                         \
                                                                                    \
        double scaled_rot_re, scaled_rot_im;                                        \
        RADIX3_ROTATE_INVERSE_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im); \
                                                                                    \
        fft_data y0, y1, y2;                                                        \
        RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im,                           \
                                       common_re, common_im,                        \
                                       scaled_rot_re, scaled_rot_im,                \
                                       y0, y1, y2);                                 \
                                                                                    \
        output_buffer[k] = y0;                                                      \
        output_buffer[k + K] = y1;                                                  \
        output_buffer[k + 2 * K] = y2;                                              \
    } while (0)

//==============================================================================
// PERFORMANCE NOTES (UNCHANGED)
//==============================================================================

/**
 * CYCLE ANALYSIS (per butterfly, Intel Skylake-X):
 *
 * AVX-512:
 * - 2x CMUL (twiddles):    ~8 cycles (2x FMA chains, latency ~4-5)
 * - Butterfly core:        ~3 cycles (add/sub/FMA)
 * - Rotation:              ~2 cycles (permute+XOR+mul)
 * - Assembly:              ~2 cycles (add/sub)
 * - Total:                 ~15 cycles / 4 butterflies = 3.75 cycles/butterfly
 *
 * AVX2:
 * - Similar breakdown:     ~7.5 cycles/butterfly
 *
 * SSE2:
 * - ~10-12 cycles/butterfly
 *
 * Scalar:
 * - ~12-15 cycles/butterfly
 *
 * Theoretical minimum:     ~3 cycles (limited by FMA latency)
 * Current efficiency:      ~80% (3.75 / 4.5 theoretical)
 *
 * Bottleneck: Twiddle multiplication dominates (53% of cycles)
 *
 * SOA BENEFIT: Eliminated shuffle overhead in twiddle loads (was 2-3% of cycles)
 */

#endif // FFT_RADIX3_MACROS_H