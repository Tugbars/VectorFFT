/**
 * @file fft_radix2_sse2.h
 * @brief SSE2 Optimized Radix-2 FFT Butterflies - TRUE SoA (ZERO SHUFFLE!)
 *
 * @details
 * SSE2 implementations with all hard-won optimizations:
 * - Native SoA with zero shuffle operations
 * - Separate multiply-add/sub (no FMA on SSE2)
 * - Software prefetch with T0/T1 hints
 * - Aligned and streaming store variants
 * - 2× unrolling for improved ILP
 * - Baseline x86-64 support (all processors have SSE2)
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture)
 * @date 2025
 */

#ifndef FFT_RADIX2_SSE2_H
#define FFT_RADIX2_SSE2_H

#include <emmintrin.h>  // SSE2
#include "fft_radix2_uniform.h"

//==============================================================================
// SSE2 CONFIGURATION
//==============================================================================

/// Vector width: 2 doubles per SSE2 register
#define SSE2_VECTOR_WIDTH 2

/// Required alignment for SSE2 (16 bytes = 128 bits)
#define SSE2_ALIGNMENT 16

/// Optimal prefetch distance for SSE2
/// 16 elements = 8 iterations ahead (2 butterflies per iteration)
#ifndef SSE2_PREFETCH_DISTANCE
#define SSE2_PREFETCH_DISTANCE 16
#endif

//==============================================================================
// SSE2 LOAD/STORE PRIMITIVES
//==============================================================================

// Regular loads/stores (unaligned)
#define LOAD_RE_SSE2(ptr) _mm_loadu_pd(ptr)
#define LOAD_IM_SSE2(ptr) _mm_loadu_pd(ptr)
#define STORE_RE_SSE2(ptr, val) _mm_storeu_pd(ptr, val)
#define STORE_IM_SSE2(ptr, val) _mm_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_SSE2_ALIGNED(ptr) _mm_load_pd(ptr)
#define LOAD_IM_SSE2_ALIGNED(ptr) _mm_load_pd(ptr)
#define STORE_RE_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)
#define STORE_IM_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)

// Streaming stores (non-temporal)
#define STREAM_RE_SSE2(ptr, val) _mm_stream_pd(ptr, val)
#define STREAM_IM_SSE2(ptr, val) _mm_stream_pd(ptr, val)

//==============================================================================
// SSE2 PREFETCH PRIMITIVES
//==============================================================================

#define PREFETCH_INPUT_T0_SSE2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)

#define PREFETCH_TWIDDLE_T1_SSE2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T1)

//==============================================================================
// COMPLEX MULTIPLY - NATIVE SoA (SSE2)
//==============================================================================

/**
 * @brief Complex multiply - NATIVE SoA form (SSE2)
 *
 * @details
 * SSE2 baseline version using separate multiply and add/sub operations.
 * No FMA available on SSE2, so we use the traditional approach.
 *
 * Computes: (ar + i*ai) * (wr + i*wi) = (ar*wr - ai*wi) + i*(ar*wi + ai*wr)
 *
 * Operation sequence:
 * - t0 = ar * wr
 * - t1 = ai * wi
 * - t2 = ar * wi
 * - t3 = ai * wr
 * - tr = t0 - t1
 * - ti = t2 + t3
 *
 * @param[in] ar Input real parts (__m128d)
 * @param[in] ai Input imag parts (__m128d)
 * @param[in] w_re Twiddle real parts (__m128d)
 * @param[in] w_im Twiddle imag parts (__m128d)
 * @param[out] tr Output real parts (__m128d)
 * @param[out] ti Output imag parts (__m128d)
 *
 * @note Requires SSE2 support (baseline for x86-64)
 * @note Latency: ~12-14 cycles (4 muls + 2 add/sub)
 */
static inline __attribute__((always_inline))
void cmul_native_soa_sse2(
    __m128d ar, __m128d ai,
    __m128d w_re, __m128d w_im,
    __m128d *tr, __m128d *ti)
{
    *tr = _mm_sub_pd(_mm_mul_pd(ar, w_re), _mm_mul_pd(ai, w_im));
    *ti = _mm_add_pd(_mm_mul_pd(ar, w_im), _mm_mul_pd(ai, w_re));
}

//==============================================================================
// RADIX-2 BUTTERFLY - NATIVE SoA (SSE2)
//==============================================================================

/**
 * @brief Radix-2 butterfly - NATIVE SoA form (SSE2)
 *
 * @details
 * ⚡⚡ ZERO SHUFFLE VERSION - Pure arithmetic!
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
 * @note Total latency: ~16-18 cycles (cmul + 4 add/sub)
 */
static inline __attribute__((always_inline))
void radix2_butterfly_native_soa_sse2(
    __m128d e_re, __m128d e_im,
    __m128d o_re, __m128d o_im,
    __m128d w_re, __m128d w_im,
    __m128d *y0_re, __m128d *y0_im,
    __m128d *y1_re, __m128d *y1_im)
{
    __m128d prod_re, prod_im;
    cmul_native_soa_sse2(o_re, o_im, w_re, w_im, &prod_re, &prod_im);
    
    *y0_re = _mm_add_pd(e_re, prod_re);
    *y0_im = _mm_add_pd(e_im, prod_im);
    *y1_re = _mm_sub_pd(e_re, prod_re);
    *y1_im = _mm_sub_pd(e_im, prod_im);
}

//==============================================================================
// BASIC PIPELINE: 2-BUTTERFLY (1× SSE2 VECTOR)
//==============================================================================

/**
 * @brief Process 2 butterflies with prefetch (regular stores)
 *
 * @details
 * Basic building block for SSE2 radix-2 FFT. Processes one SSE2 vector worth
 * of butterflies with software prefetch hints for next iteration.
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] stage_tw Stage twiddle factors (SoA)
 * @param[in] half Transform half-size
 * @param[in] prefetch_dist Prefetch distance in elements
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    // Software prefetch for next iteration
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    // Load inputs (unaligned)
    const __m128d e_re = LOAD_RE_SSE2(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2(&in_im[k + half]);
    
    // Load twiddles (aligned)
    const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);
    
    // Compute butterfly
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    // Store outputs (unaligned)
    STORE_RE_SSE2(&out_re[k], y0_re);
    STORE_IM_SSE2(&out_im[k], y0_im);
    STORE_RE_SSE2(&out_re[k + half], y1_re);
    STORE_IM_SSE2(&out_im[k + half], y1_im);
}

/**
 * @brief Process 2 butterflies with aligned loads/stores
 *
 * @note Requires input and output arrays aligned to 16-byte boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_aligned(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[k + half]);
    
    const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);
    
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    STORE_RE_SSE2_ALIGNED(&out_re[k], y0_re);
    STORE_IM_SSE2_ALIGNED(&out_im[k], y0_im);
    STORE_RE_SSE2_ALIGNED(&out_re[k + half], y1_re);
    STORE_IM_SSE2_ALIGNED(&out_im[k + half], y1_im);
}

/**
 * @brief Process 2 butterflies with streaming stores
 *
 * @note NT stores bypass cache, require manual fence at stage boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_stream(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    // Software prefetch (no output prefetch for NT stores)
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[k + half]);
    
    const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);
    
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    STREAM_RE_SSE2(&out_re[k], y0_re);
    STREAM_IM_SSE2(&out_im[k], y0_im);
    STREAM_RE_SSE2(&out_re[k + half], y1_re);
    STREAM_IM_SSE2(&out_im[k + half], y1_im);
}

//==============================================================================
// 2× UNROLLED PIPELINE: 4-BUTTERFLY (2 INDEPENDENT STREAMS)
//==============================================================================

/**
 * @brief Process 4 butterflies (2× unroll) - regular stores
 *
 * @details
 * 2× unrolling creates two independent instruction streams for better
 * utilization of execution ports. This is optimal for SSE2.
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    // Pipeline 0: butterflies [k, k+1]
    radix2_pipeline_2_sse2(k, in_re, in_im, out_re, out_im,
                           stage_tw, half, 0);
    
    // Pipeline 1: butterflies [k+2, k+3] (independent)
    radix2_pipeline_2_sse2(k + 2, in_re, in_im, out_re, out_im,
                           stage_tw, half, 0);
}

/**
 * @brief Process 4 butterflies (2× unroll) - aligned stores
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2_aligned(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    radix2_pipeline_2_sse2_aligned(k, in_re, in_im, out_re, out_im,
                                   stage_tw, half, 0);
    radix2_pipeline_2_sse2_aligned(k + 2, in_re, in_im, out_re, out_im,
                                   stage_tw, half, 0);
}

/**
 * @brief Process 4 butterflies (2× unroll) - streaming stores
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2_stream(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    
    radix2_pipeline_2_sse2_stream(k, in_re, in_im, out_re, out_im,
                                  stage_tw, half, 0);
    radix2_pipeline_2_sse2_stream(k + 2, in_re, in_im, out_re, out_im,
                                  stage_tw, half, 0);
}

#endif // FFT_RADIX2_SSE2_H
