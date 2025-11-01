/**
 * @file fft_radix2_sse2n1.h
 * @brief SSE2 Twiddle-Less Radix-2 FFT Butterflies - TRUE SoA (W=1 Optimized)
 *
 * @details
 * Twiddle-less butterfly implementations for SSE2. Provides significant
 * performance benefit for first stage and Stockham auto-sort where W=1.
 *
 * Performance benefit: ~3× faster than general butterfly
 * - General butterfly: 4 muls + 2 adds/subs = ~6 ops
 * - Twiddle-less:     0 muls + 4 adds/subs = ~4 ops
 * - 2 butterflies per vector (vs 4 for AVX2, 8 for AVX-512)
 *
 * Use cases:
 * - First stage of multi-stage FFT (most twiddles = 1)
 * - Stockham auto-sort first pass (no twiddles)
 * - Split-radix algorithms (some butterflies have W=1)
 * - Decimation-in-frequency first stage
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture, twiddle-less variant)
 * @date 2025
 */

#ifndef FFT_RADIX2_SSE2N1_H
#define FFT_RADIX2_SSE2N1_H

#include <emmintrin.h>  // SSE2
#include "fft_radix2_uniform.h"

//==============================================================================
// TWIDDLE-LESS BUTTERFLY - NATIVE SoA (SSE2)
//==============================================================================

/**
 * @brief Radix-2 butterfly without twiddle multiply (W=1) - NATIVE SoA (SSE2)
 *
 * @details
 * ⚡⚡⚡ ULTRA-FAST VERSION - No complex multiply!
 *
 * Computes simplified FFT butterfly with W=1:
 * @code
 *   y0 = even + odd
 *   y1 = even - odd
 * @endcode
 *
 * Performance on typical x86-64:
 * - 4 add/sub operations (vs 4 muls + 6 add/sub)
 * - ~4 cycles latency (vs ~16 cycles for general butterfly)
 * - ~3× faster than general butterfly
 *
 * Typical use case:
 * @code
 *   // First stage: many butterflies with k=0 or small k where W ≈ 1
 *   for (k = 0; k < threshold; k += 2) {
 *       radix2_butterfly_n1_sse2(...);
 *   }
 *   // Remaining stages: use general butterfly with twiddles
 *   for (k = threshold; k < half; k += 2) {
 *       radix2_butterfly_native_soa_sse2(...);
 *   }
 * @endcode
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
 * @note Total latency: ~4 cycles (4 add/sub operations)
 * @note Compare to general butterfly: ~16 cycles (4 muls + 6 add/sub)
 */
static inline __attribute__((always_inline))
void radix2_butterfly_n1_sse2(
    __m128d e_re, __m128d e_im,
    __m128d o_re, __m128d o_im,
    __m128d *y0_re, __m128d *y0_im,
    __m128d *y1_re, __m128d *y1_im)
{
    // W=1, so prod = odd (no multiply needed!)
    // y0 = even + odd
    // y1 = even - odd
    
    *y0_re = _mm_add_pd(e_re, o_re);
    *y0_im = _mm_add_pd(e_im, o_im);
    *y1_re = _mm_sub_pd(e_re, o_re);
    *y1_im = _mm_sub_pd(e_im, o_im);
}

//==============================================================================
// BASIC PIPELINE: 2-BUTTERFLY TWIDDLE-LESS (1× SSE2 VECTOR)
//==============================================================================

/**
 * @brief Process 2 butterflies WITHOUT twiddles (regular stores)
 *
 * @details
 * Twiddle-less version of radix2_pipeline_2_sse2().
 * Use when all 2 butterflies have W=1 (e.g., first stage, small k).
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 * @param[in] prefetch_dist Prefetch distance in elements
 *
 * @note NO stage_tw parameter - no twiddles used!
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    // Software prefetch for next iteration
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    // Load inputs (unaligned)
    const __m128d e_re = _mm_loadu_pd(&in_re[k]);
    const __m128d e_im = _mm_loadu_pd(&in_im[k]);
    const __m128d o_re = _mm_loadu_pd(&in_re[k + half]);
    const __m128d o_im = _mm_loadu_pd(&in_im[k + half]);
    
    // Compute butterfly (no twiddle multiply!)
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    
    // Store outputs (unaligned)
    _mm_storeu_pd(&out_re[k], y0_re);
    _mm_storeu_pd(&out_im[k], y0_im);
    _mm_storeu_pd(&out_re[k + half], y1_re);
    _mm_storeu_pd(&out_im[k + half], y1_im);
}

/**
 * @brief Process 2 butterflies WITHOUT twiddles (aligned loads/stores)
 *
 * @note Requires input and output arrays aligned to 16-byte boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1_aligned(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    // Load inputs (aligned)
    const __m128d e_re = _mm_load_pd(&in_re[k]);
    const __m128d e_im = _mm_load_pd(&in_im[k]);
    const __m128d o_re = _mm_load_pd(&in_re[k + half]);
    const __m128d o_im = _mm_load_pd(&in_im[k + half]);
    
    // Compute butterfly (no twiddle multiply!)
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    
    // Store outputs (aligned)
    _mm_store_pd(&out_re[k], y0_re);
    _mm_store_pd(&out_im[k], y0_im);
    _mm_store_pd(&out_re[k + half], y1_re);
    _mm_store_pd(&out_im[k + half], y1_im);
}

/**
 * @brief Process 2 butterflies WITHOUT twiddles (streaming stores)
 *
 * @note NT stores bypass cache, require manual fence at stage boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1_stream(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    // Software prefetch (no output prefetch for NT stores)
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    // Load inputs (aligned - required for streaming stores)
    const __m128d e_re = _mm_load_pd(&in_re[k]);
    const __m128d e_im = _mm_load_pd(&in_im[k]);
    const __m128d o_re = _mm_load_pd(&in_re[k + half]);
    const __m128d o_im = _mm_load_pd(&in_im[k + half]);
    
    // Compute butterfly (no twiddle multiply!)
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    
    // Store outputs (streaming - non-temporal)
    _mm_stream_pd(&out_re[k], y0_re);
    _mm_stream_pd(&out_im[k], y0_im);
    _mm_stream_pd(&out_re[k + half], y1_re);
    _mm_stream_pd(&out_im[k + half], y1_im);
}

//==============================================================================
// 2× UNROLLED PIPELINE: 4-BUTTERFLY TWIDDLE-LESS (2 INDEPENDENT STREAMS)
//==============================================================================

/**
 * @brief Process 4 butterflies WITHOUT twiddles (2× unroll, regular stores)
 *
 * @details
 * 2× unrolling for maximum ILP. Use when 4 consecutive butterflies all have W=1.
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    // Pipeline 0: butterflies [k, k+1]
    radix2_pipeline_2_sse2_n1(k, in_re, in_im, out_re, out_im, half, 0);
    
    // Pipeline 1: butterflies [k+2, k+3] (independent)
    radix2_pipeline_2_sse2_n1(k + 2, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 4 butterflies WITHOUT twiddles (2× unroll, aligned stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2_aligned(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    radix2_pipeline_2_sse2_n1_aligned(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_2_sse2_n1_aligned(k + 2, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 4 butterflies WITHOUT twiddles (2× unroll, streaming stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2_stream(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    
    radix2_pipeline_2_sse2_n1_stream(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_2_sse2_n1_stream(k + 2, in_re, in_im, out_re, out_im, half, 0);
}

#endif // FFT_RADIX2_SSE2N1_H