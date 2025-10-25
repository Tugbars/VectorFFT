/**
 * @file fft_radix2_macros_true_soa.h
 * @brief TRUE END-TO-END SoA Butterfly Macros (ZERO SHUFFLE!)
 *
 * @details
 * This header provides macro implementations for radix-2 FFT butterflies that
 * operate entirely in Structure-of-Arrays (SoA) format without any split/join
 * operations in the computational hot path.
 *
 * @author FFT Optimization Team
 * @version 2.2 (Optimized with N/8 paths and aligned loads)
 * @date 2025
 *
 * @section changes CHANGES IN v2.2
 * - Added N/8 and 3N/8 constant-twiddle fast paths
 * - Switched to aligned loads for twiddles (movapd vs movupd)
 * - Added input alignment peeling for NT mode
 * - Optimized FMA ordering for port pressure
 * - Improved prefetch hints (T1 for twiddles, T0 for outputs)
 * - Added AVX-512 masked tail handling
 */

#ifndef FFT_RADIX2_MACROS_TRUE_SOA_H
#define FFT_RADIX2_MACROS_TRUE_SOA_H

#include "fft_radix2_uniform.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX2_PREFETCH_DISTANCE
 * @brief Software prefetch lead distance (in elements)
 * @details Number of elements to prefetch ahead in butterfly loops.
 *          Optimal values:
 *          - AVX-512: 16-32 elements (2-4 iterations ahead)
 *          - AVX2:    16-32 elements (4-8 iterations ahead)
 *          - SSE2:    16-32 elements (8-16 iterations ahead)
 *          Set to 0 to disable prefetching.
 */
#ifndef RADIX2_PREFETCH_DISTANCE
#define RADIX2_PREFETCH_DISTANCE 24
#endif

/**
 * @def SQRT1_2
 * @brief √2/2 constant for N/8 and 3N/8 twiddle optimizations
 */
#define SQRT1_2 0.70710678118654752440

//==============================================================================
// IMPLEMENTATION NOTES FOR .c FILE
//==============================================================================

/**
 * @section impl_notes ADDITIONAL OPTIMIZATIONS FOR IMPLEMENTATION
 *
 * The following optimizations should be applied in the .c implementation file:
 *
 * 1. **Alignment Hints (GCC/Clang)**
 *    After verifying pointer alignment, add __builtin_assume_aligned:
 *    @code
 *    in_re  = (const double*)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
 *    in_im  = (const double*)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
 *    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
 *    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
 *    stage_tw->re = (const double*)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
 *    stage_tw->im = (const double*)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
 *    @endcode
 *
 * 2. **Denormals Flush-to-Zero (Optional, per-thread init)**
 *    For workloads with tiny signals, flush denormals to zero:
 *    @code
 *    #ifdef VECTORFFT_FLUSH_DENORMALS
 *    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
 *    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
 *    #endif
 *    @endcode
 *    Big win on long transforms with tiny magnitudes. Default: OFF (guarded by config).
 *
 * 3. **N/8 and 3N/8 Fast Path Indices**
 *    For DIF with k ∈ [0, half):
 *    - k_eighth     = half >> 2   (N/8 index)
 *    - k_3eighth    = (3*half) >> 2  (3N/8 index)
 *    - Guard: (half & 3) == 0
 *    Call radix2_k_eighth_native_soa_scalar() with sign_re = +1 for N/8, -1 for 3N/8
 *
 * 4. **Aligned Input Loads in STREAM Mode**
 *    The *_STREAM macros use aligned input loads. Ensure you peel k to alignment
 *    before entering the bulk STREAM loop. The peel already exists for output alignment;
 *    extend it to also align inputs when use_streaming is true.
 */

//==============================================================================
// COMPLEX MULTIPLY - NATIVE SoA (NO SPLIT/JOIN NEEDED!)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX-512)
 *
 * @details
 * ⚡⚡ CRITICAL: Data is ALREADY in split form from memory!
 * No split operation needed - direct loads from separate re/im arrays!
 *
 * Computes: (ar + i*ai) * (wr + i*wi) = (ar*wr - ai*wi) + i*(ar*wi + ai*wr)
 *
 * Uses FMA instructions for optimal performance with ordering to minimize
 * dependent chains and utilize dual mul ports:
 * - t0 = ai*wi (mul port)
 * - tr = ar*wr - t0 (FMA port, overlaps with t1)
 * - t1 = ai*wr (mul port, dual-issues with tr FMA)
 * - ti = ar*wi + t1 (FMA port)
 *
 * @param[in] ar Input real parts (__m512d, already loaded from re[] array)
 * @param[in] ai Input imag parts (__m512d, already loaded from im[] array)
 * @param[in] w_re Twiddle real parts (__m512d, SoA format)
 * @param[in] w_im Twiddle imag parts (__m512d, SoA format)
 * @param[out] tr Output real parts (__m512d)
 * @param[out] ti Output imag parts (__m512d)
 *
 * @note Requires AVX-512F support
 * @note Uses 4 FMA operations (optimal for complex multiply)
 */
#define CMUL_NATIVE_SOA_AVX512(ar, ai, w_re, w_im, tr, ti)       \
    do                                                           \
    {                                                            \
        __m512d _t0 = _mm512_mul_pd(ai, w_im);                   \
        tr = _mm512_fmsub_pd(ar, w_re, _t0);                     \
        __m512d _t1 = _mm512_mul_pd(ai, w_re);                   \
        ti = _mm512_fmadd_pd(ar, w_im, _t1);                     \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX2)
 *
 * @details
 * AVX2 version with FMA support when available (Haswell+).
 * Falls back to separate multiply-add when FMA not available.
 *
 * @param[in] ar Input real parts (__m256d)
 * @param[in] ai Input imag parts (__m256d)
 * @param[in] w_re Twiddle real parts (__m256d)
 * @param[in] w_im Twiddle imag parts (__m256d)
 * @param[out] tr Output real parts (__m256d)
 * @param[out] ti Output imag parts (__m256d)
 *
 * @note Requires AVX2 support
 * @note Uses FMA if __FMA__ is defined
 */
#if defined(__FMA__)
#define CMUL_NATIVE_SOA_AVX2(ar, ai, w_re, w_im, tr, ti)         \
    do                                                           \
    {                                                            \
        __m256d _t0 = _mm256_mul_pd(ai, w_im);                   \
        tr = _mm256_fmsub_pd(ar, w_re, _t0);                     \
        __m256d _t1 = _mm256_mul_pd(ai, w_re);                   \
        ti = _mm256_fmadd_pd(ar, w_im, _t1);                     \
    } while (0)
#else
#define CMUL_NATIVE_SOA_AVX2(ar, ai, w_re, w_im, tr, ti) \
    do                                                   \
    {                                                    \
        tr = _mm256_sub_pd(_mm256_mul_pd(ar, w_re),      \
                           _mm256_mul_pd(ai, w_im));     \
        ti = _mm256_add_pd(_mm256_mul_pd(ar, w_im),      \
                           _mm256_mul_pd(ai, w_re));     \
    } while (0)
#endif
#endif

/**
 * @brief Complex multiply - NATIVE SoA form (SSE2)
 *
 * @details
 * SSE2 version using separate multiply-add operations.
 * Compatible with all x86-64 processors.
 *
 * @param[in] ar Input real parts (__m128d)
 * @param[in] ai Input imag parts (__m128d)
 * @param[in] w_re Twiddle real parts (__m128d)
 * @param[in] w_im Twiddle imag parts (__m128d)
 * @param[out] tr Output real parts (__m128d)
 * @param[out] ti Output imag parts (__m128d)
 *
 * @note Requires SSE2 support (baseline for x86-64)
 */
#define CMUL_NATIVE_SOA_SSE2(ar, ai, w_re, w_im, tr, ti)             \
    do                                                               \
    {                                                                \
        tr = _mm_sub_pd(_mm_mul_pd(ar, w_re), _mm_mul_pd(ai, w_im)); \
        ti = _mm_add_pd(_mm_mul_pd(ar, w_im), _mm_mul_pd(ai, w_re)); \
    } while (0)

//==============================================================================
// RADIX-2 BUTTERFLY - NATIVE SoA (ZERO SHUFFLE!)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Radix-2 butterfly - NATIVE SoA form (AVX-512)
 *
 * @details
 * ⚡⚡ ZERO SHUFFLE VERSION - Pure arithmetic, no data rearrangement!
 *
 * Computes FFT butterfly with twiddle factor:
 * @code
 *   y0 = even + odd * W
 *   y1 = even - odd * W
 * @endcode
 *
 * All inputs are ALREADY in split form (separate re/im arrays).
 * No split or join operations needed!
 *
 * @param[in] e_re Even real parts (__m512d)
 * @param[in] e_im Even imag parts (__m512d)
 * @param[in] o_re Odd real parts (__m512d)
 * @param[in] o_im Odd imag parts (__m512d)
 * @param[in] w_re Twiddle real parts (__m512d)
 * @param[in] w_im Twiddle imag parts (__m512d)
 * @param[out] y0_re Output real parts (first half) (__m512d)
 * @param[out] y0_im Output imag parts (first half) (__m512d)
 * @param[out] y1_re Output real parts (second half) (__m512d)
 * @param[out] y1_im Output imag parts (second half) (__m512d)
 *
 * @note Requires AVX-512F support
 */
#define RADIX2_BUTTERFLY_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,    \
                                           w_re, w_im,                \
                                           y0_re, y0_im, y1_re, y1_im) \
    do                                                                \
    {                                                                 \
        __m512d prod_re, prod_im;                                     \
        CMUL_NATIVE_SOA_AVX512(o_re, o_im, w_re, w_im,                \
                               prod_re, prod_im);                     \
        y0_re = _mm512_add_pd(e_re, prod_re);                         \
        y0_im = _mm512_add_pd(e_im, prod_im);                         \
        y1_re = _mm512_sub_pd(e_re, prod_re);                         \
        y1_im = _mm512_sub_pd(e_im, prod_im);                         \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Radix-2 butterfly - NATIVE SoA form (AVX2)
 *
 * @details
 * ⚡⚡ ZERO SHUFFLE VERSION - Pure arithmetic!
 *
 * @param[in] e_re Even real parts (__m256d)
 * @param[in] e_im Even imag parts (__m256d)
 * @param[in] o_re Odd real parts (__m256d)
 * @param[in] o_im Odd imag parts (__m256d)
 * @param[in] w_re Twiddle real parts (__m256d)
 * @param[in] w_im Twiddle imag parts (__m256d)
 * @param[out] y0_re Output real parts (first half) (__m256d)
 * @param[out] y0_im Output imag parts (first half) (__m256d)
 * @param[out] y1_re Output real parts (second half) (__m256d)
 * @param[out] y1_im Output imag parts (second half) (__m256d)
 *
 * @note Requires AVX2 support
 */
#define RADIX2_BUTTERFLY_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,      \
                                         w_re, w_im,                  \
                                         y0_re, y0_im, y1_re, y1_im)   \
    do                                                                \
    {                                                                 \
        __m256d prod_re, prod_im;                                     \
        CMUL_NATIVE_SOA_AVX2(o_re, o_im, w_re, w_im,                  \
                             prod_re, prod_im);                       \
        y0_re = _mm256_add_pd(e_re, prod_re);                         \
        y0_im = _mm256_add_pd(e_im, prod_im);                         \
        y1_re = _mm256_sub_pd(e_re, prod_re);                         \
        y1_im = _mm256_sub_pd(e_im, prod_im);                         \
    } while (0)
#endif

/**
 * @brief Radix-2 butterfly - NATIVE SoA form (SSE2)
 *
 * @details
 * ⚡⚡ ZERO SHUFFLE VERSION - Pure arithmetic!
 *
 * @param[in] e_re Even real parts (__m128d)
 * @param[in] e_im Even imag parts (__m128d)
 * @param[in] o_re Odd real parts (__m128d)
 * @param[in] o_im Odd imag parts (__m128d)
 * @param[in] w_re Twiddle real parts (__m128d)
 * @param[in] w_im Twiddle imag parts (__m128d)
 * @param[out] y0_re Output real parts (first half) (__m128d)
 * @param[out] y0_im Output imag parts (first half) (__m128d)
 * @param[out] y1_re Output real parts (second half) (__m128d)
 * @param[out] y1_im Output imag parts (second half) (__m128d)
 *
 * @note Requires SSE2 support (baseline for x86-64)
 */
#define RADIX2_BUTTERFLY_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,  \
                                         w_re, w_im,              \
                                         y0_re, y0_im, y1_re, y1_im) \
    do                                                            \
    {                                                             \
        __m128d prod_re, prod_im;                                 \
        CMUL_NATIVE_SOA_SSE2(o_re, o_im, w_re, w_im,              \
                             prod_re, prod_im);                   \
        y0_re = _mm_add_pd(e_re, prod_re);                        \
        y0_im = _mm_add_pd(e_im, prod_im);                        \
        y1_re = _mm_sub_pd(e_re, prod_re);                        \
        y1_im = _mm_sub_pd(e_im, prod_im);                        \
    } while (0)

//==============================================================================
// PIPELINE MACROS WITH SOFTWARE PREFETCH
//==============================================================================

// Prefetch macros with T0/T1 hints
#define PREFETCH_INPUT_T0(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)

#define PREFETCH_TWIDDLE_T1(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T1)

#define PREFETCH_OUTPUT_T0(addr) \
    _mm_prefetch((char*)(addr), _MM_HINT_T0)

//==============================================================================
// AVX-512 PIPELINE MACROS
//==============================================================================

#ifdef __AVX512F__

// Regular loads/stores
#define LOAD_RE_AVX512(ptr) _mm512_loadu_pd(ptr)
#define LOAD_IM_AVX512(ptr) _mm512_loadu_pd(ptr)
#define STORE_RE_AVX512(ptr, val) _mm512_storeu_pd(ptr, val)
#define STORE_IM_AVX512(ptr, val) _mm512_storeu_pd(ptr, val)

// Aligned loads/stores (for peeled loops)
#define LOAD_RE_AVX512_ALIGNED(ptr) _mm512_load_pd(ptr)
#define LOAD_IM_AVX512_ALIGNED(ptr) _mm512_load_pd(ptr)
#define STORE_RE_AVX512_ALIGNED(ptr, val) _mm512_store_pd(ptr, val)
#define STORE_IM_AVX512_ALIGNED(ptr, val) _mm512_store_pd(ptr, val)

// Streaming stores (non-temporal)
#define STREAM_RE_AVX512(ptr, val) _mm512_stream_pd(ptr, val)
#define STREAM_IM_AVX512(ptr, val) _mm512_stream_pd(ptr, val)

/**
 * @brief 8-butterfly AVX-512 pipeline with prefetch
 */
#define RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(k, in_re, in_im, out_re, out_im, \
                                            stage_tw, half, prefetch_dist)   \
    do                                                                       \
    {                                                                        \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))          \
        {                                                                    \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                 \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                 \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));        \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));        \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));        \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));        \
        }                                                                    \
        const __m512d e_re = LOAD_RE_AVX512(&in_re[k]);                      \
        const __m512d e_im = LOAD_IM_AVX512(&in_im[k]);                      \
        const __m512d o_re = LOAD_RE_AVX512(&in_re[(k) + (half)]);           \
        const __m512d o_im = LOAD_IM_AVX512(&in_im[(k) + (half)]);           \
        const __m512d w_re = _mm512_load_pd(&stage_tw->re[k]);               \
        const __m512d w_im = _mm512_load_pd(&stage_tw->im[k]);               \
        __m512d y0_re, y0_im, y1_re, y1_im;                                  \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,           \
                                           w_re, w_im,                       \
                                           y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_AVX512(&out_re[k], y0_re);                                  \
        STORE_IM_AVX512(&out_im[k], y0_im);                                  \
        STORE_RE_AVX512(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX512(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 8-butterfly AVX-512 pipeline with aligned loads/stores
 */
#define RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                    stage_tw, half, prefetch_dist)   \
    do                                                                               \
    {                                                                                \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                  \
        {                                                                            \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                         \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                         \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));                \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));                \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));                \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));                \
        }                                                                            \
        const __m512d e_re = LOAD_RE_AVX512_ALIGNED(&in_re[k]);                      \
        const __m512d e_im = LOAD_IM_AVX512_ALIGNED(&in_im[k]);                      \
        const __m512d o_re = LOAD_RE_AVX512_ALIGNED(&in_re[(k) + (half)]);           \
        const __m512d o_im = LOAD_IM_AVX512_ALIGNED(&in_im[(k) + (half)]);           \
        const __m512d w_re = _mm512_load_pd(&stage_tw->re[k]);                       \
        const __m512d w_im = _mm512_load_pd(&stage_tw->im[k]);                       \
        __m512d y0_re, y0_im, y1_re, y1_im;                                          \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,                   \
                                           w_re, w_im,                               \
                                           y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_AVX512_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_AVX512_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_AVX512_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX512_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 8-butterfly AVX-512 pipeline with streaming stores
 * @note NT stores bypass cache, so no output prefetch needed
 */
#define RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_STREAM(k, in_re, in_im, out_re, out_im, \
                                                   stage_tw, half, prefetch_dist)   \
    do                                                                              \
    {                                                                               \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                 \
        {                                                                           \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                        \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                        \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));               \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));               \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));               \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));               \
        }                                                                           \
        const __m512d e_re = LOAD_RE_AVX512_ALIGNED(&in_re[k]);                     \
        const __m512d e_im = LOAD_IM_AVX512_ALIGNED(&in_im[k]);                     \
        const __m512d o_re = LOAD_RE_AVX512_ALIGNED(&in_re[(k) + (half)]);          \
        const __m512d o_im = LOAD_IM_AVX512_ALIGNED(&in_im[(k) + (half)]);          \
        const __m512d w_re = _mm512_load_pd(&stage_tw->re[k]);                      \
        const __m512d w_im = _mm512_load_pd(&stage_tw->im[k]);                      \
        __m512d y0_re, y0_im, y1_re, y1_im;                                         \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,                  \
                                           w_re, w_im,                              \
                                           y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_AVX512(&out_re[k], y0_re);                                        \
        STREAM_IM_AVX512(&out_im[k], y0_im);                                        \
        STREAM_RE_AVX512(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_AVX512(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

// Masked load/store helpers for AVX-512 tail handling
#define LOAD_RE_MASK_AVX512(ptr, mask) _mm512_maskz_loadu_pd(mask, ptr)
#define LOAD_IM_MASK_AVX512(ptr, mask) _mm512_maskz_loadu_pd(mask, ptr)
#define STORE_RE_MASK_AVX512(ptr, mask, val) _mm512_mask_storeu_pd(ptr, mask, val)
#define STORE_IM_MASK_AVX512(ptr, mask, val) _mm512_mask_storeu_pd(ptr, mask, val)

/**
 * @brief AVX-512 masked tail processing (branchless cleanup)
 * 
 * @details
 * Processes remaining butterflies (<8) using AVX-512 masks instead of scalar cleanup.
 * Keeps the loop branchless and reduces I-cache footprint. Measurable benefit at huge N.
 * 
 * @param[in] k Butterfly index
 * @param[in] count Number of remaining butterflies (1-7)
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] stage_tw Stage twiddle factors
 * @param[in] half Transform half-size
 */
#define RADIX2_PIPELINE_MASKED_NATIVE_SOA_AVX512(k, count, in_re, in_im, out_re, out_im, \
                                                 stage_tw, half)                         \
    do                                                                                   \
    {                                                                                    \
        const __mmask8 mask = (__mmask8)((1u << (count)) - 1u);                          \
        const __m512d e_re = LOAD_RE_MASK_AVX512(&in_re[k], mask);                       \
        const __m512d e_im = LOAD_IM_MASK_AVX512(&in_im[k], mask);                       \
        const __m512d o_re = LOAD_RE_MASK_AVX512(&in_re[(k) + (half)], mask);            \
        const __m512d o_im = LOAD_IM_MASK_AVX512(&in_im[(k) + (half)], mask);            \
        const __m512d w_re = _mm512_maskz_load_pd(mask, &stage_tw->re[k]);               \
        const __m512d w_im = _mm512_maskz_load_pd(mask, &stage_tw->im[k]);               \
        __m512d y0_re, y0_im, y1_re, y1_im;                                              \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,                       \
                                           w_re, w_im,                                   \
                                           y0_re, y0_im, y1_re, y1_im);                  \
        STORE_RE_MASK_AVX512(&out_re[k], mask, y0_re);                                   \
        STORE_IM_MASK_AVX512(&out_im[k], mask, y0_im);                                   \
        STORE_RE_MASK_AVX512(&out_re[(k) + (half)], mask, y1_re);                        \
        STORE_IM_MASK_AVX512(&out_im[(k) + (half)], mask, y1_im);                        \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 PIPELINE MACROS
//==============================================================================

#ifdef __AVX2__

// Regular loads/stores
#define LOAD_RE_AVX2(ptr) _mm256_loadu_pd(ptr)
#define LOAD_IM_AVX2(ptr) _mm256_loadu_pd(ptr)
#define STORE_RE_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)
#define STORE_IM_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_AVX2_ALIGNED(ptr) _mm256_load_pd(ptr)
#define LOAD_IM_AVX2_ALIGNED(ptr) _mm256_load_pd(ptr)
#define STORE_RE_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)
#define STORE_IM_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)

// Streaming stores
#define STREAM_RE_AVX2(ptr, val) _mm256_stream_pd(ptr, val)
#define STREAM_IM_AVX2(ptr, val) _mm256_stream_pd(ptr, val)

/**
 * @brief 4-butterfly AVX2 pipeline with prefetch
 */
#define RADIX2_PIPELINE_4_NATIVE_SOA_AVX2(k, in_re, in_im, out_re, out_im, \
                                          stage_tw, half, prefetch_dist)   \
    do                                                                     \
    {                                                                      \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))        \
        {                                                                  \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));      \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));      \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));      \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));      \
        }                                                                  \
        const __m256d e_re = LOAD_RE_AVX2(&in_re[k]);                      \
        const __m256d e_im = LOAD_IM_AVX2(&in_im[k]);                      \
        const __m256d o_re = LOAD_RE_AVX2(&in_re[(k) + (half)]);           \
        const __m256d o_im = LOAD_IM_AVX2(&in_im[(k) + (half)]);           \
        const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);             \
        const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);             \
        __m256d y0_re, y0_im, y1_re, y1_im;                                \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,           \
                                         w_re, w_im,                       \
                                         y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_AVX2(&out_re[k], y0_re);                                  \
        STORE_IM_AVX2(&out_im[k], y0_im);                                  \
        STORE_RE_AVX2(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX2(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 4-butterfly AVX2 pipeline with aligned loads/stores
 */
#define RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                  stage_tw, half, prefetch_dist)   \
    do                                                                             \
    {                                                                              \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                \
        {                                                                          \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));              \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));              \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));              \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));              \
        }                                                                          \
        const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);                      \
        const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);                      \
        const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[(k) + (half)]);           \
        const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[(k) + (half)]);           \
        const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);                     \
        const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);                     \
        __m256d y0_re, y0_im, y1_re, y1_im;                                        \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,                   \
                                         w_re, w_im,                               \
                                         y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_AVX2_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_AVX2_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_AVX2_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX2_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 4-butterfly AVX2 pipeline with streaming stores
 * @note NT stores bypass cache, so no output prefetch needed
 */
#define RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_STREAM(k, in_re, in_im, out_re, out_im, \
                                                 stage_tw, half, prefetch_dist)   \
    do                                                                            \
    {                                                                             \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))               \
        {                                                                         \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));             \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));             \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));             \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));             \
        }                                                                         \
        const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);                     \
        const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);                     \
        const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[(k) + (half)]);          \
        const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[(k) + (half)]);          \
        const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);                    \
        const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);                    \
        __m256d y0_re, y0_im, y1_re, y1_im;                                       \
        RADIX2_BUTTERFLY_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,                  \
                                         w_re, w_im,                              \
                                         y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_AVX2(&out_re[k], y0_re);                                        \
        STREAM_IM_AVX2(&out_im[k], y0_im);                                        \
        STREAM_RE_AVX2(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_AVX2(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SSE2 PIPELINE MACROS
//==============================================================================

// Regular loads/stores
#define LOAD_RE_SSE2(ptr) _mm_loadu_pd(ptr)
#define LOAD_IM_SSE2(ptr) _mm_loadu_pd(ptr)
#define STORE_RE_SSE2(ptr, val) _mm_storeu_pd(ptr, val)
#define STORE_IM_SSE2(ptr, val) _mm_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_SSE2_ALIGNED(ptr) _mm_load_pd(ptr)
#define LOAD_IM_SSE2_ALIGNED(ptr) _mm_load_pd(ptr)
#define STORE_RE_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)
#define STORE_IM_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)

// Streaming stores
#define STREAM_RE_SSE2(ptr, val) _mm_stream_pd(ptr, val)
#define STREAM_IM_SSE2(ptr, val) _mm_stream_pd(ptr, val)

/**
 * @brief 2-butterfly SSE2 pipeline with prefetch
 */
#define RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(k, in_re, in_im, out_re, out_im, \
                                          stage_tw, half, prefetch_dist)   \
    do                                                                     \
    {                                                                      \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))        \
        {                                                                  \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));      \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));      \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));      \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));      \
        }                                                                  \
        const __m128d e_re = LOAD_RE_SSE2(&in_re[k]);                      \
        const __m128d e_im = LOAD_IM_SSE2(&in_im[k]);                      \
        const __m128d o_re = LOAD_RE_SSE2(&in_re[(k) + (half)]);           \
        const __m128d o_im = LOAD_IM_SSE2(&in_im[(k) + (half)]);           \
        const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);                \
        const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);                \
        __m128d y0_re, y0_im, y1_re, y1_im;                                \
        RADIX2_BUTTERFLY_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,           \
                                         w_re, w_im,                       \
                                         y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_SSE2(&out_re[k], y0_re);                                  \
        STORE_IM_SSE2(&out_im[k], y0_im);                                  \
        STORE_RE_SSE2(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_SSE2(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 2-butterfly SSE2 pipeline with aligned loads/stores
 */
#define RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                  stage_tw, half, prefetch_dist)   \
    do                                                                             \
    {                                                                              \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                \
        {                                                                          \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));              \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));              \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));              \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));              \
        }                                                                          \
        const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);                      \
        const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);                      \
        const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[(k) + (half)]);           \
        const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[(k) + (half)]);           \
        const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);                        \
        const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);                        \
        __m128d y0_re, y0_im, y1_re, y1_im;                                        \
        RADIX2_BUTTERFLY_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,                   \
                                         w_re, w_im,                               \
                                         y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_SSE2_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_SSE2_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_SSE2_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_SSE2_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 2-butterfly SSE2 pipeline with streaming stores
 * @note NT stores bypass cache, so no output prefetch needed
 */
#define RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_STREAM(k, in_re, in_im, out_re, out_im, \
                                                 stage_tw, half, prefetch_dist)   \
    do                                                                            \
    {                                                                             \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))               \
        {                                                                         \
            PREFETCH_INPUT_T0(in_re, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0(in_im, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0(in_re, (k) + (half) + (prefetch_dist));             \
            PREFETCH_INPUT_T0(in_im, (k) + (half) + (prefetch_dist));             \
            PREFETCH_TWIDDLE_T1(stage_tw->re, (k) + (prefetch_dist));             \
            PREFETCH_TWIDDLE_T1(stage_tw->im, (k) + (prefetch_dist));             \
        }                                                                         \
        const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);                     \
        const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);                     \
        const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[(k) + (half)]);          \
        const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[(k) + (half)]);          \
        const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);                       \
        const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);                       \
        __m128d y0_re, y0_im, y1_re, y1_im;                                       \
        RADIX2_BUTTERFLY_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,                  \
                                         w_re, w_im,                              \
                                         y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_SSE2(&out_re[k], y0_re);                                        \
        STREAM_IM_SSE2(&out_im[k], y0_im);                                        \
        STREAM_RE_SSE2(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_SSE2(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

/**
 * @brief 1-butterfly scalar - NATIVE SoA
 *
 * @details
 * Scalar fallback for cleanup or when SIMD not available.
 * Processes a single butterfly using scalar floating-point operations.
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] stage_tw Stage twiddle factors (SoA)
 * @param[in] half Transform half-size
 */
#define RADIX2_PIPELINE_1_NATIVE_SOA_SCALAR(k, in_re, in_im, out_re, out_im, \
                                            stage_tw, half)                  \
    do                                                                       \
    {                                                                        \
        double e_re = in_re[k];                                              \
        double e_im = in_im[k];                                              \
        double o_re = in_re[(k) + (half)];                                   \
        double o_im = in_im[(k) + (half)];                                   \
        double w_re = stage_tw->re[k];                                       \
        double w_im = stage_tw->im[k];                                       \
        double prod_re = o_re * w_re - o_im * w_im;                          \
        double prod_im = o_re * w_im + o_im * w_re;                          \
        out_re[k] = e_re + prod_re;                                          \
        out_im[k] = e_im + prod_im;                                          \
        out_re[(k) + (half)] = e_re - prod_re;                               \
        out_im[(k) + (half)] = e_im - prod_im;                               \
    } while (0)

//==============================================================================
// SPECIAL CASE: k=0 (w=1) - SCALAR ONLY!
//==============================================================================

/**
 * @brief Special case for k=0 where twiddle = 1
 *
 * @details
 * ⚠️  CRITICAL: k=0 applies to SINGLE INDEX ONLY (not a range!)
 * W[0] = 1, so no twiddle multiply needed for ONLY k=0.
 * Do NOT vectorize this - it's just one butterfly.
 *
 * Algorithm:
 * @code
 *   y[0] = x[0] + x[half]
 *   y[half] = x[0] - x[half]
 * @endcode
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 */
static inline void radix2_k0_native_soa_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half)
{
    const double e_re = in_re[0];
    const double e_im = in_im[0];
    const double o_re = in_re[half];
    const double o_im = in_im[half];
    
    out_re[0] = e_re + o_re;
    out_im[0] = e_im + o_im;
    out_re[half] = e_re - o_re;
    out_im[half] = e_im - o_im;
}

//==============================================================================
// SPECIAL CASE: k=N/4 (w=-i) - SCALAR ONLY!
//==============================================================================

/**
 * @brief Special case for k=N/4 where twiddle = -i
 *
 * @details
 * ⚠️  CRITICAL: k=N/4 applies to SINGLE INDEX ONLY (not a range!)
 * W[N/4] = -i = (0, -1), so multiply by -i = swap and negate real
 * Only this ONE butterfly at k=N/4 has this property.
 * Do NOT vectorize this - it's just one butterfly.
 *
 * Algorithm:
 * @code
 *   prod = odd * (-i)
 *        = (o_re, o_im) * (0, -1)
 *        = (o_im, -o_re)   // swap and negate real
 *
 *   y[k] = even + prod
 *   y[k+half] = even - prod
 * @endcode
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] k_quarter Index k = N/4
 * @param[in] half Transform half-size
 */
static inline void radix2_k_quarter_native_soa_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_quarter,
    int half)
{
    const double e_re = in_re[k_quarter];
    const double e_im = in_im[k_quarter];
    const double o_re = in_re[k_quarter + half];
    const double o_im = in_im[k_quarter + half];
    
    // W = -i: multiply by -i = swap and negate real
    // prod = o * (-i) = (o_re, o_im) * (0, -1) = (o_im, -o_re)
    out_re[k_quarter] = e_re + o_im;
    out_im[k_quarter] = e_im - o_re;
    out_re[k_quarter + half] = e_re - o_im;
    out_im[k_quarter + half] = e_im + o_re;
}

//==============================================================================
// SPECIAL CASE: k=N/8 and k=3N/8 (w = ±√2/2 - i√2/2) - SCALAR ONLY!
//==============================================================================

/**
 * @brief Special case for k=N/8 and k=3N/8 where twiddles are ±√2/2 - i√2/2
 *
 * @details
 * For N divisible by 8:
 * - k = N/8:  W = exp(-jπ/4) = (√2/2, -√2/2)
 * - k = 3N/8: W = exp(-j3π/4) = (-√2/2, -√2/2)
 *
 * Complex multiply simplifies from 4 muls to 2 muls:
 * W = (sign_re*c, -c) where c = √2/2
 * prod_re = c*(o_re*sign_re ± o_im)  (+ for N/8, - for 3N/8)
 * prod_im = c*(o_im ∓ o_re*sign_re)
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] k_eighth Index (N/8 or 3N/8)
 * @param[in] half Transform half-size
 * @param[in] sign_re +1 for N/8, -1 for 3N/8
 */
static inline void radix2_k_eighth_native_soa_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_eighth,
    int half,
    int sign_re)
{
    const double c = SQRT1_2;
    const double e_re = in_re[k_eighth];
    const double e_im = in_im[k_eighth];
    const double o_re = in_re[k_eighth + half];
    const double o_im = in_im[k_eighth + half];

    // W = (sign_re*c, -c)
    // Compute sum and diff based on sign_re
    const double sum  = (sign_re > 0) ? (o_re + o_im) : (-o_re + o_im);
    const double diff = (sign_re > 0) ? (o_im - o_re) : (-o_im - o_re);

    const double pr = c * sum;
    const double pi = c * diff;

    out_re[k_eighth] = e_re + pr;
    out_im[k_eighth] = e_im + pi;
    out_re[k_eighth + half] = e_re - pr;
    out_im[k_eighth + half] = e_im - pi;
}

#endif // FFT_RADIX2_MACROS_TRUE_SOA_H

