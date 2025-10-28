/**
 * @file fft_radix2_avx2.h
 * @brief AVX2 Optimized Radix-2 FFT Butterflies - TRUE SoA (ZERO SHUFFLE!)
 *
 * @details
 * AVX2 implementations with all hard-won optimizations:
 * - Native SoA with zero shuffle operations
 * - FMA-optimized complex multiply (Haswell+)
 * - Software prefetch with T0/T1 hints
 * - Aligned and streaming store variants
 * - 2× unrolling for improved ILP
 * - Fallback to non-FMA for pre-Haswell (rare but supported)
 *
 * @author FFT Optimization Team
 * @version 3.0 (Separated architecture)
 * @date 2025
 */

#ifndef FFT_RADIX2_AVX2_H
#define FFT_RADIX2_AVX2_H

#ifdef __AVX2__

#include <immintrin.h>
#include "fft_radix2_uniform.h"

//==============================================================================
// AVX2 CONFIGURATION
//==============================================================================

/// Vector width: 4 doubles per AVX2 register
#define AVX2_VECTOR_WIDTH 4

/// Required alignment for AVX2 (32 bytes = 256 bits)
#define AVX2_ALIGNMENT 32

/// Optimal prefetch distance for AVX2
/// 24 elements = 6 iterations ahead (4 butterflies per iteration)
#ifndef AVX2_PREFETCH_DISTANCE
#define AVX2_PREFETCH_DISTANCE 24
#endif

//==============================================================================
// AVX2 LOAD/STORE PRIMITIVES
//==============================================================================

// Regular loads/stores (unaligned)
#define LOAD_RE_AVX2(ptr) _mm256_loadu_pd(ptr)
#define LOAD_IM_AVX2(ptr) _mm256_loadu_pd(ptr)
#define STORE_RE_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)
#define STORE_IM_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)

// Aligned loads/stores
#define LOAD_RE_AVX2_ALIGNED(ptr) _mm256_load_pd(ptr)
#define LOAD_IM_AVX2_ALIGNED(ptr) _mm256_load_pd(ptr)
#define STORE_RE_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)
#define STORE_IM_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)

// Streaming stores (non-temporal)
#define STREAM_RE_AVX2(ptr, val) _mm256_stream_pd(ptr, val)
#define STREAM_IM_AVX2(ptr, val) _mm256_stream_pd(ptr, val)

//==============================================================================
// AVX2 PREFETCH PRIMITIVES
//==============================================================================

#define PREFETCH_INPUT_T0_AVX2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)

#define PREFETCH_TWIDDLE_T1_AVX2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T1)

//==============================================================================
// COMPLEX MULTIPLY - NATIVE SoA (AVX2)
//==============================================================================

#if defined(__FMA__)
/**
 * @brief Complex multiply - NATIVE SoA form (AVX2 with FMA)
 *
 * @details
 * Haswell+ version with FMA support. Uses fused multiply-add for optimal
 * performance and numerical accuracy.
 *
 * Computes: (ar + i*ai) * (wr + i*wi) = (ar*wr - ai*wi) + i*(ar*wi + ai*wr)
 *
 * FMA ordering minimizes dependency chains:
 * - t0 = ai*wi
 * - tr = ar*wr - t0  (fmsub)
 * - t1 = ai*wr
 * - ti = ar*wi + t1  (fmadd)
 *
 * @param[in] ar Input real parts (__m256d)
 * @param[in] ai Input imag parts (__m256d)
 * @param[in] w_re Twiddle real parts (__m256d)
 * @param[in] w_im Twiddle imag parts (__m256d)
 * @param[out] tr Output real parts (__m256d)
 * @param[out] ti Output imag parts (__m256d)
 *
 * @note Requires AVX2 and FMA support (Haswell+)
 * @note Latency: ~8 cycles on Haswell/Skylake
 */
static inline __attribute__((always_inline))
void cmul_native_soa_avx2(
    __m256d ar, __m256d ai,
    __m256d w_re, __m256d w_im,
    __m256d *tr, __m256d *ti)
{
    __m256d t0 = _mm256_mul_pd(ai, w_im);
    *tr = _mm256_fmsub_pd(ar, w_re, t0);
    __m256d t1 = _mm256_mul_pd(ai, w_re);
    *ti = _mm256_fmadd_pd(ar, w_im, t1);
}

#else // No FMA support (pre-Haswell)

/**
 * @brief Complex multiply - NATIVE SoA form (AVX2 without FMA)
 *
 * @details
 * Fallback for pre-Haswell processors without FMA.
 * Uses separate multiply and add/sub operations.
 *
 * @note Performance: ~20% slower than FMA version
 * @note Rare case - most AVX2 processors have FMA
 */
static inline __attribute__((always_inline))
void cmul_native_soa_avx2(
    __m256d ar, __m256d ai,
    __m256d w_re, __m256d w_im,
    __m256d *tr, __m256d *ti)
{
    *tr = _mm256_sub_pd(_mm256_mul_pd(ar, w_re),
                        _mm256_mul_pd(ai, w_im));
    *ti = _mm256_add_pd(_mm256_mul_pd(ar, w_im),
                        _mm256_mul_pd(ai, w_re));
}

#endif // __FMA__

//==============================================================================
// RADIX-2 BUTTERFLY - NATIVE SoA (AVX2)
//==============================================================================

/**
 * @brief Radix-2 butterfly - NATIVE SoA form (AVX2)
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
static inline __attribute__((always_inline))
void radix2_butterfly_native_soa_avx2(
    __m256d e_re, __m256d e_im,
    __m256d o_re, __m256d o_im,
    __m256d w_re, __m256d w_im,
    __m256d *y0_re, __m256d *y0_im,
    __m256d *y1_re, __m256d *y1_im)
{
    __m256d prod_re, prod_im;
    cmul_native_soa_avx2(o_re, o_im, w_re, w_im, &prod_re, &prod_im);
    
    *y0_re = _mm256_add_pd(e_re, prod_re);
    *y0_im = _mm256_add_pd(e_im, prod_im);
    *y1_re = _mm256_sub_pd(e_re, prod_re);
    *y1_im = _mm256_sub_pd(e_im, prod_im);
}

//==============================================================================
// BASIC PIPELINE: 4-BUTTERFLY (1× AVX2 VECTOR)
//==============================================================================

/**
 * @brief Process 4 butterflies with prefetch (regular stores)
 *
 * @details
 * Basic building block for AVX2 radix-2 FFT. Processes one AVX2 vector worth
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
void radix2_pipeline_4_avx2(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    // Load inputs (unaligned)
    const __m256d e_re = LOAD_RE_AVX2(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2(&in_im[k + half]);
    
    // Load twiddles (aligned)
    const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);
    
    // Compute butterfly
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    // Store outputs (unaligned)
    STORE_RE_AVX2(&out_re[k], y0_re);
    STORE_IM_AVX2(&out_im[k], y0_im);
    STORE_RE_AVX2(&out_re[k + half], y1_re);
    STORE_IM_AVX2(&out_im[k + half], y1_im);
}

/**
 * @brief Process 4 butterflies with aligned loads/stores
 *
 * @note Requires input and output arrays aligned to 32-byte boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_aligned(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[k + half]);
    
    const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);
    
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    STORE_RE_AVX2_ALIGNED(&out_re[k], y0_re);
    STORE_IM_AVX2_ALIGNED(&out_im[k], y0_im);
    STORE_RE_AVX2_ALIGNED(&out_re[k + half], y1_re);
    STORE_IM_AVX2_ALIGNED(&out_im[k + half], y1_im);
}

/**
 * @brief Process 4 butterflies with streaming stores
 *
 * @note NT stores bypass cache, require manual fence at stage boundaries
 */
static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_stream(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[k + half]);
    
    const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);
    
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im,
                                     w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    
    STREAM_RE_AVX2(&out_re[k], y0_re);
    STREAM_IM_AVX2(&out_im[k], y0_im);
    STREAM_RE_AVX2(&out_re[k + half], y1_re);
    STREAM_IM_AVX2(&out_im[k + half], y1_im);
}

//==============================================================================
// 2× UNROLLED PIPELINE: 8-BUTTERFLY (2 INDEPENDENT STREAMS)
//==============================================================================

/**
 * @brief Process 8 butterflies (2× unroll) - regular stores
 *
 * @details
 * 2× unrolling creates two independent instruction streams for better
 * utilization of execution ports. This is optimal for most AVX2 processors.
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    // Pipeline 0: butterflies [k, k+3]
    radix2_pipeline_4_avx2(k, in_re, in_im, out_re, out_im,
                           stage_tw, half, 0);
    
    // Pipeline 1: butterflies [k+4, k+7] (independent)
    radix2_pipeline_4_avx2(k + 4, in_re, in_im, out_re, out_im,
                           stage_tw, half, 0);
}

/**
 * @brief Process 8 butterflies (2× unroll) - aligned stores
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2_aligned(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    radix2_pipeline_4_avx2_aligned(k, in_re, in_im, out_re, out_im,
                                   stage_tw, half, 0);
    radix2_pipeline_4_avx2_aligned(k + 4, in_re, in_im, out_re, out_im,
                                   stage_tw, half, 0);
}

/**
 * @brief Process 8 butterflies (2× unroll) - streaming stores
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2_stream(
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
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    
    radix2_pipeline_4_avx2_stream(k, in_re, in_im, out_re, out_im,
                                  stage_tw, half, 0);
    radix2_pipeline_4_avx2_stream(k + 4, in_re, in_im, out_re, out_im,
                                  stage_tw, half, 0);
}

#endif // __AVX2__

#endif // FFT_RADIX2_AVX2_H
