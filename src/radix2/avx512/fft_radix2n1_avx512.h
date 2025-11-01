/**
 * @file fft_radix2_avx512_twiddleless.h
 * @brief AVX-512 Twiddle-Less Radix-2 FFT Butterflies - TRUE SoA (W=1 Optimized)
 *
 * @details
 * Twiddle-less butterfly implementations for AVX-512. Massive performance win
 * on high-end Intel (Skylake-X, Ice Lake, Sapphire Rapids) for first stage
 * and Stockham auto-sort.
 *
 * Performance benefit: ~3× faster than general butterfly
 * - 8 butterflies per vector (vs 4 for AVX2, 2 for SSE2)
 * - Dual 512-bit FMA ports fully utilized
 * - 4× unrolling for maximum ILP
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture, twiddle-less variant)
 * @date 2025
 */

#ifndef FFT_RADIX2_AVX512_TWIDDLELESS_H
#define FFT_RADIX2_AVX512_TWIDDLELESS_H

#ifdef __AVX512F__

#include <immintrin.h>
#include "fft_radix2_uniform.h"

//==============================================================================
// TWIDDLE-LESS BUTTERFLY - NATIVE SoA (AVX-512)
//==============================================================================

/**
 * @brief Radix-2 butterfly without twiddle multiply (W=1) - NATIVE SoA (AVX-512)
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
 * Performance on Skylake-X/Ice Lake:
 * - 4 add/sub operations (vs 4 muls + 6 add/sub)
 * - ~4 cycles latency (vs ~12 cycles for general butterfly)
 * - Can sustain 2 butterflies/cycle with dual FMA ports
 * - ~3× faster than general butterfly
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
 * @note Total latency: ~4 cycles (4 add/sub on ports 0/5)
 */
static inline __attribute__((always_inline))
void radix2_butterfly_n1_avx512(
    __m512d e_re, __m512d e_im,
    __m512d o_re, __m512d o_im,
    __m512d *y0_re, __m512d *y0_im,
    __m512d *y1_re, __m512d *y1_im)
{
    // W=1, so prod = odd (no multiply needed!)
    *y0_re = _mm512_add_pd(e_re, o_re);
    *y0_im = _mm512_add_pd(e_im, o_im);
    *y1_re = _mm512_sub_pd(e_re, o_re);
    *y1_im = _mm512_sub_pd(e_im, o_im);
}

//==============================================================================
// BASIC PIPELINE: 8-BUTTERFLY TWIDDLE-LESS (1× AVX-512 VECTOR)
//==============================================================================

/**
 * @brief Process 8 butterflies WITHOUT twiddles (regular stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx512_n1(
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
    
    const __m512d e_re = _mm512_loadu_pd(&in_re[k]);
    const __m512d e_im = _mm512_loadu_pd(&in_im[k]);
    const __m512d o_re = _mm512_loadu_pd(&in_re[k + half]);
    const __m512d o_im = _mm512_loadu_pd(&in_im[k + half]);
    
    __m512d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx512(e_re, e_im, o_re, o_im,
                                        &y0_re, &y0_im, &y1_re, &y1_im);
    
    _mm512_storeu_pd(&out_re[k], y0_re);
    _mm512_storeu_pd(&out_im[k], y0_im);
    _mm512_storeu_pd(&out_re[k + half], y1_re);
    _mm512_storeu_pd(&out_im[k + half], y1_im);
}

/**
 * @brief Process 8 butterflies WITHOUT twiddles (aligned loads/stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx512_n1_aligned(
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
    
    const __m512d e_re = _mm512_load_pd(&in_re[k]);
    const __m512d e_im = _mm512_load_pd(&in_im[k]);
    const __m512d o_re = _mm512_load_pd(&in_re[k + half]);
    const __m512d o_im = _mm512_load_pd(&in_im[k + half]);
    
    __m512d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx512(e_re, e_im, o_re, o_im,
                                        &y0_re, &y0_im, &y1_re, &y1_im);
    
    _mm512_store_pd(&out_re[k], y0_re);
    _mm512_store_pd(&out_im[k], y0_im);
    _mm512_store_pd(&out_re[k + half], y1_re);
    _mm512_store_pd(&out_im[k + half], y1_im);
}

/**
 * @brief Process 8 butterflies WITHOUT twiddles (streaming stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_8_avx512_n1_stream(
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
    
    const __m512d e_re = _mm512_load_pd(&in_re[k]);
    const __m512d e_im = _mm512_load_pd(&in_im[k]);
    const __m512d o_re = _mm512_load_pd(&in_re[k + half]);
    const __m512d o_im = _mm512_load_pd(&in_im[k + half]);
    
    __m512d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx512(e_re, e_im, o_re, o_im,
                                        &y0_re, &y0_im, &y1_re, &y1_im);
    
    _mm512_stream_pd(&out_re[k], y0_re);
    _mm512_stream_pd(&out_im[k], y0_im);
    _mm512_stream_pd(&out_re[k + half], y1_re);
    _mm512_stream_pd(&out_im[k + half], y1_im);
}

//==============================================================================
// 2× UNROLLED PIPELINE: 16-BUTTERFLY TWIDDLE-LESS
//==============================================================================

/**
 * @brief Process 16 butterflies WITHOUT twiddles (2× unroll, regular stores)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_16_avx512_n1_unroll2(
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
    
    radix2_pipeline_8_avx512_n1(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1(k + 8, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 16 butterflies WITHOUT twiddles (2× unroll, aligned)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_16_avx512_n1_unroll2_aligned(
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
    
    radix2_pipeline_8_avx512_n1_aligned(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_aligned(k + 8, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 16 butterflies WITHOUT twiddles (2× unroll, streaming)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_16_avx512_n1_unroll2_stream(
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
    
    radix2_pipeline_8_avx512_n1_stream(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_stream(k + 8, in_re, in_im, out_re, out_im, half, 0);
}

//==============================================================================
// 4× UNROLLED PIPELINE: 32-BUTTERFLY TWIDDLE-LESS (MAX ILP!)
//==============================================================================

/**
 * @brief Process 32 butterflies WITHOUT twiddles (4× unroll) - NEW!
 *
 *
 * 4× unrolling with twiddle-less butterflies combines two major optimizations:
 * 1. No complex multiply (3× speedup)
 * 2. Maximum ILP from 4 independent streams (1.5× speedup)
 * Total: ~4.5× faster than single-stream general butterfly!
 *
 * Ideal use case: First stage of large FFT (N >= 2^16)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_32_avx512_n1_unroll4(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int prefetch_dist)
{
    // Extended prefetch for 4× unroll
    if (prefetch_dist > 0 && k + prefetch_dist < half)
    {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + prefetch_dist + 16], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist + 16], _MM_HINT_T0);
    }
    
    // Pipeline 0-3: four independent streams
    radix2_pipeline_8_avx512_n1(k,      in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1(k + 8,  in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1(k + 16, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1(k + 24, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 32 butterflies WITHOUT twiddles (4× unroll, aligned)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_32_avx512_n1_unroll4_aligned(
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
        _mm_prefetch((char*)&in_re[k + prefetch_dist + 16], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist + 16], _MM_HINT_T0);
    }
    
    radix2_pipeline_8_avx512_n1_aligned(k,      in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_aligned(k + 8,  in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_aligned(k + 16, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_aligned(k + 24, in_re, in_im, out_re, out_im, half, 0);
}

/**
 * @brief Process 32 butterflies WITHOUT twiddles (4× unroll, streaming)
 */
static inline __attribute__((always_inline))
void radix2_pipeline_32_avx512_n1_unroll4_stream(
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
        _mm_prefetch((char*)&in_re[k + prefetch_dist + 16], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist + 16], _MM_HINT_T0);
    }
    
    radix2_pipeline_8_avx512_n1_stream(k,      in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_stream(k + 8,  in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_stream(k + 16, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_8_avx512_n1_stream(k + 24, in_re, in_im, out_re, out_im, half, 0);
}

#endif // __AVX512F__

#endif // FFT_RADIX2_AVX512_TWIDDLELESS_H