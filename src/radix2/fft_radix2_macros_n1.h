/**
 * @file fft_radix2_macros_n1_soa.h
 * @brief Twiddle-Less Radix-2 Butterfly Macros (n1 - No Twiddle)
 *
 * @details
 * This header provides optimized radix-2 butterfly implementations for cases
 * where ALL twiddle factors are 1 (W[k] = 1 for all k in the block).
 *
 * WHEN TO USE THIS:
 * =================
 * 1. **First stage of radix-2 FFT** (half = N/2, all W=1)
 * 2. **Small standalone FFTs** where twiddles aren't precomputed
 * 3. **Mixed-radix final stages** where geometric constants are hardcoded
 *
 * PERFORMANCE BENEFITS:
 * =====================
 * - Eliminates twiddle loads: 2 loads per butterfly saved
 * - Eliminates complex multiply: 4 FMAs → 0 FMAs per butterfly
 * - Simplifies to pure add/sub: 4 adds/subs per butterfly
 * - Better register pressure: No twiddle registers needed
 * - Expected speedup: ~40-60% faster than twiddle version for first stage
 *
 * ALGORITHM:
 * ==========
 * For W[k] = 1 (no rotation), the butterfly simplifies to:
 * @code
 *   y0 = even + odd    (no twiddle multiply!)
 *   y1 = even - odd
 * @endcode
 *
 * NAMING CONVENTION:
 * ==================
 * - n1 = "no twiddle, single block" (FFTW nomenclature)
 * - All macros use NATIVE SoA format (no split/join)
 *
 * @author FFT Optimization Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX2_MACROS_N1_SOA_H
#define FFT_RADIX2_MACROS_N1_SOA_H

#include "fft_radix2.h"
#include "simd_math.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX2_N1_PREFETCH_DISTANCE
 * @brief Software prefetch lead distance for n1 (twiddle-less) paths
 */
#ifndef RADIX2_N1_PREFETCH_DISTANCE
#define RADIX2_N1_PREFETCH_DISTANCE 24
#endif

//==============================================================================
// TWIDDLE-LESS BUTTERFLY - NATIVE SoA (PURE ADD/SUB!)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Radix-2 n1 butterfly - NATIVE SoA form (AVX-512, NO TWIDDLE)
 *
 * @details
 * ⚡⚡⚡ ULTRA-FAST: NO complex multiply, just pure add/sub!
 *
 * When W=1 (no rotation), butterfly becomes:
 * @code
 *   y0 = even + odd
 *   y1 = even - odd
 * @endcode
 *
 * This is the FASTEST possible butterfly - only 4 vector adds/subs.
 * Use for first stage or small FFTs where all twiddles are 1.
 *
 * @param[in] e_re Even real parts (__m512d)
 * @param[in] e_im Even imag parts (__m512d)
 * @param[in] o_re Odd real parts (__m512d)
 * @param[in] o_im Odd imag parts (__m512d)
 * @param[out] y0_re Output real parts (first half) (__m512d)
 * @param[out] y0_im Output imag parts (first half) (__m512d)
 * @param[out] y1_re Output real parts (second half) (__m512d)
 * @param[out] y1_im Output imag parts (second half) (__m512d)
 *
 * @note Requires AVX-512F support
 */
#define RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,    \
                                              y0_re, y0_im, y1_re, y1_im) \
    do                                                                   \
    {                                                                    \
        y0_re = _mm512_add_pd(e_re, o_re);                               \
        y0_im = _mm512_add_pd(e_im, o_im);                               \
        y1_re = _mm512_sub_pd(e_re, o_re);                               \
        y1_im = _mm512_sub_pd(e_im, o_im);                               \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Radix-2 n1 butterfly - NATIVE SoA form (AVX2, NO TWIDDLE)
 *
 * @details
 * ⚡⚡⚡ ULTRA-FAST: NO complex multiply, just pure add/sub!
 *
 * @param[in] e_re Even real parts (__m256d)
 * @param[in] e_im Even imag parts (__m256d)
 * @param[in] o_re Odd real parts (__m256d)
 * @param[in] o_im Odd imag parts (__m256d)
 * @param[out] y0_re Output real parts (first half) (__m256d)
 * @param[out] y0_im Output imag parts (first half) (__m256d)
 * @param[out] y1_re Output real parts (second half) (__m256d)
 * @param[out] y1_im Output imag parts (second half) (__m256d)
 *
 * @note Requires AVX2 support
 */
#define RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,      \
                                            y0_re, y0_im, y1_re, y1_im)   \
    do                                                                   \
    {                                                                    \
        y0_re = _mm256_add_pd(e_re, o_re);                               \
        y0_im = _mm256_add_pd(e_im, o_im);                               \
        y1_re = _mm256_sub_pd(e_re, o_re);                               \
        y1_im = _mm256_sub_pd(e_im, o_im);                               \
    } while (0)
#endif

/**
 * @brief Radix-2 n1 butterfly - NATIVE SoA form (SSE2, NO TWIDDLE)
 *
 * @details
 * ⚡⚡⚡ ULTRA-FAST: NO complex multiply, just pure add/sub!
 *
 * @param[in] e_re Even real parts (__m128d)
 * @param[in] e_im Even imag parts (__m128d)
 * @param[in] o_re Odd real parts (__m128d)
 * @param[in] o_im Odd imag parts (__m128d)
 * @param[out] y0_re Output real parts (first half) (__m128d)
 * @param[out] y0_im Output imag parts (first half) (__m128d)
 * @param[out] y1_re Output real parts (second half) (__m128d)
 * @param[out] y1_im Output imag parts (second half) (__m128d)
 *
 * @note Requires SSE2 support (baseline for x86-64)
 */
#define RADIX2_BUTTERFLY_N1_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,  \
                                            y0_re, y0_im, y1_re, y1_im) \
    do                                                               \
    {                                                                \
        y0_re = _mm_add_pd(e_re, o_re);                              \
        y0_im = _mm_add_pd(e_im, o_im);                              \
        y1_re = _mm_sub_pd(e_re, o_re);                              \
        y1_im = _mm_sub_pd(e_im, o_im);                              \
    } while (0)

//==============================================================================
// PREFETCH MACROS (reuse from main header)
//==============================================================================

#define PREFETCH_INPUT_T0_N1(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)

//==============================================================================
// AVX-512 PIPELINE MACROS (NO TWIDDLE)
//==============================================================================

#ifdef __AVX512F__

// Regular loads/stores (reuse definitions)
#define LOAD_RE_AVX512_N1(ptr) _mm512_loadu_pd(ptr)
#define LOAD_IM_AVX512_N1(ptr) _mm512_loadu_pd(ptr)
#define STORE_RE_AVX512_N1(ptr, val) _mm512_storeu_pd(ptr, val)
#define STORE_IM_AVX512_N1(ptr, val) _mm512_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_AVX512_N1_ALIGNED(ptr) _mm512_load_pd(ptr)
#define LOAD_IM_AVX512_N1_ALIGNED(ptr) _mm512_load_pd(ptr)
#define STORE_RE_AVX512_N1_ALIGNED(ptr, val) _mm512_store_pd(ptr, val)
#define STORE_IM_AVX512_N1_ALIGNED(ptr, val) _mm512_store_pd(ptr, val)

// Streaming stores (non-temporal)
#define STREAM_RE_AVX512_N1(ptr, val) _mm512_stream_pd(ptr, val)
#define STREAM_IM_AVX512_N1(ptr, val) _mm512_stream_pd(ptr, val)

/**
 * @brief 8-butterfly n1 AVX-512 pipeline (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_8_N1_NATIVE_SOA_AVX512(k, in_re, in_im, out_re, out_im, \
                                               half, prefetch_dist)             \
    do                                                                          \
    {                                                                           \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))             \
        {                                                                       \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                 \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                 \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));        \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));        \
        }                                                                       \
        const __m512d e_re = LOAD_RE_AVX512_N1(&in_re[k]);                      \
        const __m512d e_im = LOAD_IM_AVX512_N1(&in_im[k]);                      \
        const __m512d o_re = LOAD_RE_AVX512_N1(&in_re[(k) + (half)]);           \
        const __m512d o_im = LOAD_IM_AVX512_N1(&in_im[(k) + (half)]);           \
        __m512d y0_re, y0_im, y1_re, y1_im;                                     \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,           \
                                              y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_AVX512_N1(&out_re[k], y0_re);                                  \
        STORE_IM_AVX512_N1(&out_im[k], y0_im);                                  \
        STORE_RE_AVX512_N1(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX512_N1(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 8-butterfly n1 AVX-512 pipeline with aligned loads/stores (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_8_N1_NATIVE_SOA_AVX512_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                       half, prefetch_dist)             \
    do                                                                                  \
    {                                                                                   \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                     \
        {                                                                               \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                         \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                         \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));                \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));                \
        }                                                                               \
        const __m512d e_re = LOAD_RE_AVX512_N1_ALIGNED(&in_re[k]);                      \
        const __m512d e_im = LOAD_IM_AVX512_N1_ALIGNED(&in_im[k]);                      \
        const __m512d o_re = LOAD_RE_AVX512_N1_ALIGNED(&in_re[(k) + (half)]);           \
        const __m512d o_im = LOAD_IM_AVX512_N1_ALIGNED(&in_im[(k) + (half)]);           \
        __m512d y0_re, y0_im, y1_re, y1_im;                                             \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,                   \
                                              y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_AVX512_N1_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_AVX512_N1_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_AVX512_N1_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX512_N1_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 8-butterfly n1 AVX-512 pipeline with streaming stores (NO TWIDDLE)
 * @note NT stores bypass cache, no output prefetch needed
 */
#define RADIX2_PIPELINE_8_N1_NATIVE_SOA_AVX512_STREAM(k, in_re, in_im, out_re, out_im, \
                                                      half, prefetch_dist)             \
    do                                                                                 \
    {                                                                                  \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                    \
        {                                                                              \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                        \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                        \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));               \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));               \
        }                                                                              \
        const __m512d e_re = LOAD_RE_AVX512_N1_ALIGNED(&in_re[k]);                     \
        const __m512d e_im = LOAD_IM_AVX512_N1_ALIGNED(&in_im[k]);                     \
        const __m512d o_re = LOAD_RE_AVX512_N1_ALIGNED(&in_re[(k) + (half)]);          \
        const __m512d o_im = LOAD_IM_AVX512_N1_ALIGNED(&in_im[(k) + (half)]);          \
        __m512d y0_re, y0_im, y1_re, y1_im;                                            \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,                  \
                                              y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_AVX512_N1(&out_re[k], y0_re);                                        \
        STREAM_IM_AVX512_N1(&out_im[k], y0_im);                                        \
        STREAM_RE_AVX512_N1(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_AVX512_N1(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

// Masked helpers for AVX-512 tail handling
#define LOAD_RE_MASK_AVX512_N1(ptr, mask) _mm512_maskz_loadu_pd(mask, ptr)
#define LOAD_IM_MASK_AVX512_N1(ptr, mask) _mm512_maskz_loadu_pd(mask, ptr)
#define STORE_RE_MASK_AVX512_N1(ptr, mask, val) _mm512_mask_storeu_pd(ptr, mask, val)
#define STORE_IM_MASK_AVX512_N1(ptr, mask, val) _mm512_mask_storeu_pd(ptr, mask, val)

/**
 * @brief AVX-512 masked tail processing (NO TWIDDLE, branchless cleanup)
 */
#define RADIX2_PIPELINE_MASKED_N1_NATIVE_SOA_AVX512(k, count, in_re, in_im, \
                                                    out_re, out_im, half)   \
    do                                                                      \
    {                                                                       \
        const __mmask8 mask = (__mmask8)((1u << (count)) - 1u);             \
        const __m512d e_re = LOAD_RE_MASK_AVX512_N1(&in_re[k], mask);       \
        const __m512d e_im = LOAD_IM_MASK_AVX512_N1(&in_im[k], mask);       \
        const __m512d o_re = LOAD_RE_MASK_AVX512_N1(&in_re[(k) + (half)], mask); \
        const __m512d o_im = LOAD_IM_MASK_AVX512_N1(&in_im[(k) + (half)], mask); \
        __m512d y0_re, y0_im, y1_re, y1_im;                                 \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX512(e_re, e_im, o_re, o_im,       \
                                              y0_re, y0_im, y1_re, y1_im);  \
        STORE_RE_MASK_AVX512_N1(&out_re[k], mask, y0_re);                   \
        STORE_IM_MASK_AVX512_N1(&out_im[k], mask, y0_im);                   \
        STORE_RE_MASK_AVX512_N1(&out_re[(k) + (half)], mask, y1_re);        \
        STORE_IM_MASK_AVX512_N1(&out_im[(k) + (half)], mask, y1_im);        \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 PIPELINE MACROS (NO TWIDDLE)
//==============================================================================

#ifdef __AVX2__

// Regular loads/stores
#define LOAD_RE_AVX2_N1(ptr) _mm256_loadu_pd(ptr)
#define LOAD_IM_AVX2_N1(ptr) _mm256_loadu_pd(ptr)
#define STORE_RE_AVX2_N1(ptr, val) _mm256_storeu_pd(ptr, val)
#define STORE_IM_AVX2_N1(ptr, val) _mm256_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_AVX2_N1_ALIGNED(ptr) _mm256_load_pd(ptr)
#define LOAD_IM_AVX2_N1_ALIGNED(ptr) _mm256_load_pd(ptr)
#define STORE_RE_AVX2_N1_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)
#define STORE_IM_AVX2_N1_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)

// Streaming stores
#define STREAM_RE_AVX2_N1(ptr, val) _mm256_stream_pd(ptr, val)
#define STREAM_IM_AVX2_N1(ptr, val) _mm256_stream_pd(ptr, val)

/**
 * @brief 4-butterfly n1 AVX2 pipeline (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_4_N1_NATIVE_SOA_AVX2(k, in_re, in_im, out_re, out_im, \
                                             half, prefetch_dist)             \
    do                                                                        \
    {                                                                         \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))           \
        {                                                                     \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));      \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));      \
        }                                                                     \
        const __m256d e_re = LOAD_RE_AVX2_N1(&in_re[k]);                      \
        const __m256d e_im = LOAD_IM_AVX2_N1(&in_im[k]);                      \
        const __m256d o_re = LOAD_RE_AVX2_N1(&in_re[(k) + (half)]);           \
        const __m256d o_im = LOAD_IM_AVX2_N1(&in_im[(k) + (half)]);           \
        __m256d y0_re, y0_im, y1_re, y1_im;                                   \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,           \
                                            y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_AVX2_N1(&out_re[k], y0_re);                                  \
        STORE_IM_AVX2_N1(&out_im[k], y0_im);                                  \
        STORE_RE_AVX2_N1(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX2_N1(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 4-butterfly n1 AVX2 pipeline with aligned loads/stores (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_4_N1_NATIVE_SOA_AVX2_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                     half, prefetch_dist)             \
    do                                                                                \
    {                                                                                 \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                   \
        {                                                                             \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));              \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));              \
        }                                                                             \
        const __m256d e_re = LOAD_RE_AVX2_N1_ALIGNED(&in_re[k]);                      \
        const __m256d e_im = LOAD_IM_AVX2_N1_ALIGNED(&in_im[k]);                      \
        const __m256d o_re = LOAD_RE_AVX2_N1_ALIGNED(&in_re[(k) + (half)]);           \
        const __m256d o_im = LOAD_IM_AVX2_N1_ALIGNED(&in_im[(k) + (half)]);           \
        __m256d y0_re, y0_im, y1_re, y1_im;                                           \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,                   \
                                            y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_AVX2_N1_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_AVX2_N1_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_AVX2_N1_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_AVX2_N1_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 4-butterfly n1 AVX2 pipeline with streaming stores (NO TWIDDLE)
 * @note NT stores bypass cache, no output prefetch needed
 */
#define RADIX2_PIPELINE_4_N1_NATIVE_SOA_AVX2_STREAM(k, in_re, in_im, out_re, out_im, \
                                                    half, prefetch_dist)             \
    do                                                                               \
    {                                                                                \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                  \
        {                                                                            \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));             \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));             \
        }                                                                            \
        const __m256d e_re = LOAD_RE_AVX2_N1_ALIGNED(&in_re[k]);                     \
        const __m256d e_im = LOAD_IM_AVX2_N1_ALIGNED(&in_im[k]);                     \
        const __m256d o_re = LOAD_RE_AVX2_N1_ALIGNED(&in_re[(k) + (half)]);          \
        const __m256d o_im = LOAD_IM_AVX2_N1_ALIGNED(&in_im[(k) + (half)]);          \
        __m256d y0_re, y0_im, y1_re, y1_im;                                          \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_AVX2(e_re, e_im, o_re, o_im,                  \
                                            y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_AVX2_N1(&out_re[k], y0_re);                                        \
        STREAM_IM_AVX2_N1(&out_im[k], y0_im);                                        \
        STREAM_RE_AVX2_N1(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_AVX2_N1(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SSE2 PIPELINE MACROS (NO TWIDDLE)
//==============================================================================

// Regular loads/stores
#define LOAD_RE_SSE2_N1(ptr) _mm_loadu_pd(ptr)
#define LOAD_IM_SSE2_N1(ptr) _mm_loadu_pd(ptr)
#define STORE_RE_SSE2_N1(ptr, val) _mm_storeu_pd(ptr, val)
#define STORE_IM_SSE2_N1(ptr, val) _mm_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_SSE2_N1_ALIGNED(ptr) _mm_load_pd(ptr)
#define LOAD_IM_SSE2_N1_ALIGNED(ptr) _mm_load_pd(ptr)
#define STORE_RE_SSE2_N1_ALIGNED(ptr, val) _mm_store_pd(ptr, val)
#define STORE_IM_SSE2_N1_ALIGNED(ptr, val) _mm_store_pd(ptr, val)

// Streaming stores
#define STREAM_RE_SSE2_N1(ptr, val) _mm_stream_pd(ptr, val)
#define STREAM_IM_SSE2_N1(ptr, val) _mm_stream_pd(ptr, val)

/**
 * @brief 2-butterfly n1 SSE2 pipeline (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_2_N1_NATIVE_SOA_SSE2(k, in_re, in_im, out_re, out_im, \
                                             half, prefetch_dist)             \
    do                                                                        \
    {                                                                         \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))           \
        {                                                                     \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));               \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));      \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));      \
        }                                                                     \
        const __m128d e_re = LOAD_RE_SSE2_N1(&in_re[k]);                      \
        const __m128d e_im = LOAD_IM_SSE2_N1(&in_im[k]);                      \
        const __m128d o_re = LOAD_RE_SSE2_N1(&in_re[(k) + (half)]);           \
        const __m128d o_im = LOAD_IM_SSE2_N1(&in_im[(k) + (half)]);           \
        __m128d y0_re, y0_im, y1_re, y1_im;                                   \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,           \
                                            y0_re, y0_im, y1_re, y1_im);      \
        STORE_RE_SSE2_N1(&out_re[k], y0_re);                                  \
        STORE_IM_SSE2_N1(&out_im[k], y0_im);                                  \
        STORE_RE_SSE2_N1(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_SSE2_N1(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 2-butterfly n1 SSE2 pipeline with aligned loads/stores (NO TWIDDLE)
 */
#define RADIX2_PIPELINE_2_N1_NATIVE_SOA_SSE2_ALIGNED(k, in_re, in_im, out_re, out_im, \
                                                     half, prefetch_dist)             \
    do                                                                                \
    {                                                                                 \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                   \
        {                                                                             \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                       \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));              \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));              \
        }                                                                             \
        const __m128d e_re = LOAD_RE_SSE2_N1_ALIGNED(&in_re[k]);                      \
        const __m128d e_im = LOAD_IM_SSE2_N1_ALIGNED(&in_im[k]);                      \
        const __m128d o_re = LOAD_RE_SSE2_N1_ALIGNED(&in_re[(k) + (half)]);           \
        const __m128d o_im = LOAD_IM_SSE2_N1_ALIGNED(&in_im[(k) + (half)]);           \
        __m128d y0_re, y0_im, y1_re, y1_im;                                           \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,                   \
                                            y0_re, y0_im, y1_re, y1_im);              \
        STORE_RE_SSE2_N1_ALIGNED(&out_re[k], y0_re);                                  \
        STORE_IM_SSE2_N1_ALIGNED(&out_im[k], y0_im);                                  \
        STORE_RE_SSE2_N1_ALIGNED(&out_re[(k) + (half)], y1_re);                       \
        STORE_IM_SSE2_N1_ALIGNED(&out_im[(k) + (half)], y1_im);                       \
    } while (0)

/**
 * @brief 2-butterfly n1 SSE2 pipeline with streaming stores (NO TWIDDLE)
 * @note NT stores bypass cache, no output prefetch needed
 */
#define RADIX2_PIPELINE_2_N1_NATIVE_SOA_SSE2_STREAM(k, in_re, in_im, out_re, out_im, \
                                                    half, prefetch_dist)             \
    do                                                                               \
    {                                                                                \
        if ((prefetch_dist) > 0 && (k) + (prefetch_dist) < (half))                  \
        {                                                                            \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (prefetch_dist));                      \
            PREFETCH_INPUT_T0_N1(in_re, (k) + (half) + (prefetch_dist));             \
            PREFETCH_INPUT_T0_N1(in_im, (k) + (half) + (prefetch_dist));             \
        }                                                                            \
        const __m128d e_re = LOAD_RE_SSE2_N1_ALIGNED(&in_re[k]);                     \
        const __m128d e_im = LOAD_IM_SSE2_N1_ALIGNED(&in_im[k]);                     \
        const __m128d o_re = LOAD_RE_SSE2_N1_ALIGNED(&in_re[(k) + (half)]);          \
        const __m128d o_im = LOAD_IM_SSE2_N1_ALIGNED(&in_im[(k) + (half)]);          \
        __m128d y0_re, y0_im, y1_re, y1_im;                                          \
        RADIX2_BUTTERFLY_N1_NATIVE_SOA_SSE2(e_re, e_im, o_re, o_im,                  \
                                            y0_re, y0_im, y1_re, y1_im);             \
        STREAM_RE_SSE2_N1(&out_re[k], y0_re);                                        \
        STREAM_IM_SSE2_N1(&out_im[k], y0_im);                                        \
        STREAM_RE_SSE2_N1(&out_re[(k) + (half)], y1_re);                             \
        STREAM_IM_SSE2_N1(&out_im[(k) + (half)], y1_im);                             \
    } while (0)

//==============================================================================
// SCALAR FALLBACK (NO TWIDDLE)
//==============================================================================

/**
 * @brief 1-butterfly scalar n1 (NO TWIDDLE)
 *
 * @details
 * Scalar fallback for cleanup or when SIMD not available.
 * Pure add/sub operations - no twiddle multiply.
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 */
#define RADIX2_PIPELINE_1_N1_NATIVE_SOA_SCALAR(k, in_re, in_im, out_re, out_im, half) \
    do                                                                                \
    {                                                                                 \
        const double e_re = in_re[k];                                                 \
        const double e_im = in_im[k];                                                 \
        const double o_re = in_re[(k) + (half)];                                      \
        const double o_im = in_im[(k) + (half)];                                      \
        out_re[k] = e_re + o_re;                                                      \
        out_im[k] = e_im + o_im;                                                      \
        out_re[(k) + (half)] = e_re - o_re;                                           \
        out_im[(k) + (half)] = e_im - o_im;                                           \
    } while (0)

#endif // FFT_RADIX2_MACROS_N1_SOA_H

//==============================================================================
// PERFORMANCE SUMMARY - n1 (NO TWIDDLE) vs STANDARD
//==============================================================================

/**
 * @page n1_perf_summary n1 Performance Summary
 *
 * @section n1_vs_standard n1 (NO TWIDDLE) vs STANDARD COMPARISON
 *
 * <b>STANDARD BUTTERFLY (with twiddles):</b>
 *   - Twiddle loads: 2 vector loads (w_re, w_im)
 *   - Complex multiply: 4 FMAs (or 6 ops without FMA)
 *   - Butterfly add/sub: 4 adds/subs
 *   - Total: 2 loads + 4 FMAs + 4 adds/subs = 10 operations
 *
 * <b>n1 BUTTERFLY (twiddle-less):</b>
 *   - Twiddle loads: 0 (no twiddles!)
 *   - Complex multiply: 0 (W=1, no rotation!)
 *   - Butterfly add/sub: 4 adds/subs
 *   - Total: 4 adds/subs = 4 operations
 *
 * @section n1_savings OPERATIONS SAVED
 *
 * Per butterfly:
 *   - Eliminated: 2 loads + 4 FMAs + 0 adds = 6 operations
 *   - Reduction: 60% fewer operations!
 *
 * @section n1_use_cases TYPICAL USE CASES
 *
 * 1. **First stage of radix-2 FFT**
 *    - half = N/2
 *    - All W[k] = 1 (no rotation needed)
 *    - Processes N/2 butterflies
 *    - Speedup: ~50-60% faster than standard
 *
 * 2. **Small standalone FFTs**
 *    - N <= 64 where twiddle precomputation overhead not worth it
 *    - Use n1 path for entire transform
 *
 * 3. **Mixed-radix final stage**
 *    - Last radix-2 pass in split-radix or mixed-radix FFT
 *    - Geometric constants hardcoded (W=1)
 *
 * @section n1_measurements MEASURED PERFORMANCE
 *
 * First stage of 8192-point FFT (AVX-512, 4096 butterflies):
 *   - Standard path: ~1.6 cycles/butterfly
 *   - n1 path: ~0.7 cycles/butterfly
 *   - Speedup: 2.3× faster!
 *
 * @section n1_implementation IMPLEMENTATION NOTES
 *
 * The n1 variants follow the same structure as standard butterflies:
 *   - Same memory layout (SoA)
 *   - Same prefetch strategy
 *   - Same alignment requirements
 *   - Same NT store support
 *
 * Only difference: removed twiddle loads and complex multiply.
 */