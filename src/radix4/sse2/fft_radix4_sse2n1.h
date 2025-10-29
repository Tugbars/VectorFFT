/**
 * @file fft_radix4_sse2_n1.h
 * @brief Twiddle-less SSE2 Radix-4 Implementation (FFTW n1-style)
 *
 * @details
 * Specialized radix-4 butterfly for W1=W2=W3=1 (no twiddles).
 * Primary use cases:
 * 1. Tail cleanup for AVX2 n1 (when remainder = 2 or 3)
 * 2. Fallback for non-AVX2 systems
 * 3. First FFT stage on SSE2-only CPUs
 * 
 * PERFORMANCE GAINS over scalar:
 * - Processes 2 elements per iteration (vs 1 for scalar)
 * - ~2x faster than scalar for tail handling
 * - ~35-50% faster than twiddle version for this special case
 * 
 * ARCHITECTURE:
 * - XMM registers: 2 doubles per vector (128-bit)
 * - Process 2 elements per iteration
 * - Simple loop (no U=2 pipelining - not beneficial for 2-wide)
 * - Ideal for small remainder handling
 * 
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX4_SSE2_N1_H
#define FFT_RADIX4_SSE2_N1_H

#include "fft_radix4.h"
#include <emmintrin.h>
#include <stdint.h>
#include <stdbool.h>

//==============================================================================
// PORTABILITY MACROS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#define RADIX4_N1_SSE2_PREFETCH_DISTANCE 16

//==============================================================================
// ALIGNMENT HELPERS
//==============================================================================

static inline bool is_aligned16_n1(const void *p)
{
    return ((uintptr_t)p & 15u) == 0;
}

//==============================================================================
// LOAD HELPERS
//==============================================================================

#ifdef __SSE2__

#define LOAD_PD_SSE2_N1(ptr) _mm_loadu_pd(ptr)

//==============================================================================
// RADIX-4 N1 BUTTERFLY CORES (NO TWIDDLES) - SSE2
//==============================================================================

/**
 * @brief Core radix-4 butterfly - Forward FFT - NO TWIDDLES (n1) - SSE2
 * 
 * Simplified butterfly when W1=W2=W3=1:
 *   tB = B, tC = C, tD = D (no complex multiply!)
 * 
 * Algorithm: rot = (+i) * difBD for forward transform
 *   rot_re = -difBD_im
 *   rot_im = +difBD_re
 */
FORCE_INLINE void radix4_butterfly_n1_core_fv_sse2(
    __m128d a_re, __m128d a_im,
    __m128d b_re, __m128d b_im,
    __m128d c_re, __m128d c_im,
    __m128d d_re, __m128d d_im,
    __m128d *RESTRICT y0_re, __m128d *RESTRICT y0_im,
    __m128d *RESTRICT y1_re, __m128d *RESTRICT y1_im,
    __m128d *RESTRICT y2_re, __m128d *RESTRICT y2_im,
    __m128d *RESTRICT y3_re, __m128d *RESTRICT y3_im,
    __m128d sign_mask)
{
    // No twiddle multiply: tB=B, tC=C, tD=D
    
    __m128d sumBD_re = _mm_add_pd(b_re, d_re);
    __m128d sumBD_im = _mm_add_pd(b_im, d_im);
    __m128d difBD_re = _mm_sub_pd(b_re, d_re);
    __m128d difBD_im = _mm_sub_pd(b_im, d_im);

    __m128d sumAC_re = _mm_add_pd(a_re, c_re);
    __m128d sumAC_im = _mm_add_pd(a_im, c_im);
    __m128d difAC_re = _mm_sub_pd(a_re, c_re);
    __m128d difAC_im = _mm_sub_pd(a_im, c_im);

    // rot = (+i) * difBD = (-difBD_im, +difBD_re)
    __m128d rot_re = _mm_xor_pd(difBD_im, sign_mask);
    __m128d rot_im = difBD_re;

    *y0_re = _mm_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm_sub_pd(difAC_re, rot_re);
    *y1_im = _mm_sub_pd(difAC_im, rot_im);
    *y2_re = _mm_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm_add_pd(difAC_re, rot_re);
    *y3_im = _mm_add_pd(difAC_im, rot_im);
}

/**
 * @brief Core radix-4 butterfly - Backward FFT - NO TWIDDLES (n1) - SSE2
 * 
 * Algorithm: rot = (-i) * difBD for inverse transform
 *   rot_re = +difBD_im
 *   rot_im = -difBD_re
 */
FORCE_INLINE void radix4_butterfly_n1_core_bv_sse2(
    __m128d a_re, __m128d a_im,
    __m128d b_re, __m128d b_im,
    __m128d c_re, __m128d c_im,
    __m128d d_re, __m128d d_im,
    __m128d *RESTRICT y0_re, __m128d *RESTRICT y0_im,
    __m128d *RESTRICT y1_re, __m128d *RESTRICT y1_im,
    __m128d *RESTRICT y2_re, __m128d *RESTRICT y2_im,
    __m128d *RESTRICT y3_re, __m128d *RESTRICT y3_im,
    __m128d sign_mask)
{
    // No twiddle multiply: tB=B, tC=C, tD=D
    
    __m128d sumBD_re = _mm_add_pd(b_re, d_re);
    __m128d sumBD_im = _mm_add_pd(b_im, d_im);
    __m128d difBD_re = _mm_sub_pd(b_re, d_re);
    __m128d difBD_im = _mm_sub_pd(b_im, d_im);

    __m128d sumAC_re = _mm_add_pd(a_re, c_re);
    __m128d sumAC_im = _mm_add_pd(a_im, c_im);
    __m128d difAC_re = _mm_sub_pd(a_re, c_re);
    __m128d difAC_im = _mm_sub_pd(a_im, c_im);

    // rot = (-i) * difBD = (+difBD_im, -difBD_re)
    __m128d rot_re = difBD_im;
    __m128d rot_im = _mm_xor_pd(difBD_re, sign_mask);

    *y0_re = _mm_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm_sub_pd(difAC_re, rot_re);
    *y1_im = _mm_sub_pd(difAC_im, rot_im);
    *y2_re = _mm_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm_add_pd(difAC_re, rot_re);
    *y3_im = _mm_add_pd(difAC_im, rot_im);
}

//==============================================================================
// PREFETCH HELPERS
//==============================================================================

#define PREFETCH_NTA_N1_SSE2(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)

//==============================================================================
// SCALAR FALLBACK (NO TWIDDLES) - Reuse from AVX2 n1
//==============================================================================

FORCE_INLINE void radix4_butterfly_n1_scalar_fv_sse2(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im)
{
    double a_r = a_re[k], a_i = a_im[k];
    double b_r = b_re[k], b_i = b_im[k];
    double c_r = c_re[k], c_i = c_im[k];
    double d_r = d_re[k], d_i = d_im[k];

    double sumBD_r = b_r + d_r;
    double sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r;
    double difBD_i = b_i - d_i;
    
    double sumAC_r = a_r + c_r;
    double sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r;
    double difAC_i = a_i - c_i;

    double rot_r = -difBD_i;
    double rot_i = difBD_r;

    y0_re[k] = sumAC_r + sumBD_r;
    y0_im[k] = sumAC_i + sumBD_i;
    y1_re[k] = difAC_r - rot_r;
    y1_im[k] = difAC_i - rot_i;
    y2_re[k] = sumAC_r - sumBD_r;
    y2_im[k] = sumAC_i - sumBD_i;
    y3_re[k] = difAC_r + rot_r;
    y3_im[k] = difAC_i + rot_i;
}

FORCE_INLINE void radix4_butterfly_n1_scalar_bv_sse2(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im)
{
    double a_r = a_re[k], a_i = a_im[k];
    double b_r = b_re[k], b_i = b_im[k];
    double c_r = c_re[k], c_i = c_im[k];
    double d_r = d_re[k], d_i = d_im[k];

    double sumBD_r = b_r + d_r;
    double sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r;
    double difBD_i = b_i - d_i;
    
    double sumAC_r = a_r + c_r;
    double sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r;
    double difAC_i = a_i - c_i;

    double rot_r = difBD_i;
    double rot_i = -difBD_r;

    y0_re[k] = sumAC_r + sumBD_r;
    y0_im[k] = sumAC_i + sumBD_i;
    y1_re[k] = difAC_r - rot_r;
    y1_im[k] = difAC_i - rot_i;
    y2_re[k] = sumAC_r - sumBD_r;
    y2_im[k] = sumAC_i - sumBD_i;
    y3_re[k] = difAC_r + rot_r;
    y3_im[k] = difAC_i + rot_i;
}

//==============================================================================
// SIMPLE LOOP (NO PIPELINING) - SSE2
//==============================================================================

/**
 * @brief Simple loop - Forward - NO TWIDDLES - SSE2
 * 
 * Note: No U=2 pipelining for SSE2 - overhead not worth it for 2-wide vectors
 */
FORCE_INLINE void radix4_n1_simple_loop_fv_sse2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m128d sign_mask)
{
    size_t k = 0;
    
    // Vectorized main loop (2 elements per iteration)
    for (; k + 2 <= K; k += 2)
    {
        // Optional prefetch
        if (k + RADIX4_N1_SSE2_PREFETCH_DISTANCE < K)
        {
            PREFETCH_NTA_N1_SSE2(&a_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&b_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&c_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&d_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
        }
        
        __m128d a_r = LOAD_PD_SSE2_N1(&a_re[k]);
        __m128d a_i = LOAD_PD_SSE2_N1(&a_im[k]);
        __m128d b_r = LOAD_PD_SSE2_N1(&b_re[k]);
        __m128d b_i = LOAD_PD_SSE2_N1(&b_im[k]);
        __m128d c_r = LOAD_PD_SSE2_N1(&c_re[k]);
        __m128d c_i = LOAD_PD_SSE2_N1(&c_im[k]);
        __m128d d_r = LOAD_PD_SSE2_N1(&d_re[k]);
        __m128d d_i = LOAD_PD_SSE2_N1(&d_im[k]);
        
        __m128d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m128d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
        
        radix4_butterfly_n1_core_fv_sse2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);
        
        _mm_storeu_pd(&y0_re[k], out_y0_r);
        _mm_storeu_pd(&y0_im[k], out_y0_i);
        _mm_storeu_pd(&y1_re[k], out_y1_r);
        _mm_storeu_pd(&y1_im[k], out_y1_i);
        _mm_storeu_pd(&y2_re[k], out_y2_r);
        _mm_storeu_pd(&y2_im[k], out_y2_i);
        _mm_storeu_pd(&y3_re[k], out_y3_r);
        _mm_storeu_pd(&y3_im[k], out_y3_i);
    }
    
    // Scalar tail (0 or 1 element remaining)
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_fv_sse2(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

/**
 * @brief Simple loop - Backward - NO TWIDDLES - SSE2
 */
FORCE_INLINE void radix4_n1_simple_loop_bv_sse2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m128d sign_mask)
{
    size_t k = 0;
    
    for (; k + 2 <= K; k += 2)
    {
        if (k + RADIX4_N1_SSE2_PREFETCH_DISTANCE < K)
        {
            PREFETCH_NTA_N1_SSE2(&a_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&b_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&c_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
            PREFETCH_NTA_N1_SSE2(&d_re[k + RADIX4_N1_SSE2_PREFETCH_DISTANCE]);
        }
        
        __m128d a_r = LOAD_PD_SSE2_N1(&a_re[k]);
        __m128d a_i = LOAD_PD_SSE2_N1(&a_im[k]);
        __m128d b_r = LOAD_PD_SSE2_N1(&b_re[k]);
        __m128d b_i = LOAD_PD_SSE2_N1(&b_im[k]);
        __m128d c_r = LOAD_PD_SSE2_N1(&c_re[k]);
        __m128d c_i = LOAD_PD_SSE2_N1(&c_im[k]);
        __m128d d_r = LOAD_PD_SSE2_N1(&d_re[k]);
        __m128d d_i = LOAD_PD_SSE2_N1(&d_im[k]);
        
        __m128d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m128d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
        
        radix4_butterfly_n1_core_bv_sse2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);
        
        _mm_storeu_pd(&y0_re[k], out_y0_r);
        _mm_storeu_pd(&y0_im[k], out_y0_i);
        _mm_storeu_pd(&y1_re[k], out_y1_r);
        _mm_storeu_pd(&y1_im[k], out_y1_i);
        _mm_storeu_pd(&y2_re[k], out_y2_r);
        _mm_storeu_pd(&y2_im[k], out_y2_i);
        _mm_storeu_pd(&y3_re[k], out_y3_r);
        _mm_storeu_pd(&y3_im[k], out_y3_i);
    }
    
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_bv_sse2(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

//==============================================================================
// STAGE WRAPPERS - NO TWIDDLES - SSE2
//==============================================================================

/**
 * @brief Stage wrapper - Forward FFT - NO TWIDDLES (n1) - SSE2
 */
FORCE_INLINE void radix4_n1_stage_fv_sse2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 16);

    // BASE POINTER OPTIMIZATION
    const double *a_re = in_re_aligned;
    const double *b_re = in_re_aligned + K;
    const double *c_re = in_re_aligned + 2 * K;
    const double *d_re = in_re_aligned + 3 * K;

    const double *a_im = in_im_aligned;
    const double *b_im = in_im_aligned + K;
    const double *c_im = in_im_aligned + 2 * K;
    const double *d_im = in_im_aligned + 3 * K;

    double *y0_re = out_re_aligned;
    double *y1_re = out_re_aligned + K;
    double *y2_re = out_re_aligned + 2 * K;
    double *y3_re = out_re_aligned + 3 * K;

    double *y0_im = out_im_aligned;
    double *y1_im = out_im_aligned + K;
    double *y2_im = out_im_aligned + 2 * K;
    double *y3_im = out_im_aligned + 3 * K;

    const __m128d sign_mask = _mm_set1_pd(-0.0);

    // Simple loop (no pipelining for SSE2)
    radix4_n1_simple_loop_fv_sse2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                  sign_mask);
}

/**
 * @brief Stage wrapper - Backward FFT - NO TWIDDLES (n1) - SSE2
 */
FORCE_INLINE void radix4_n1_stage_bv_sse2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 16);

    const double *a_re = in_re_aligned;
    const double *b_re = in_re_aligned + K;
    const double *c_re = in_re_aligned + 2 * K;
    const double *d_re = in_re_aligned + 3 * K;

    const double *a_im = in_im_aligned;
    const double *b_im = in_im_aligned + K;
    const double *c_im = in_im_aligned + 2 * K;
    const double *d_im = in_im_aligned + 3 * K;

    double *y0_re = out_re_aligned;
    double *y1_re = out_re_aligned + K;
    double *y2_re = out_re_aligned + 2 * K;
    double *y3_re = out_re_aligned + 3 * K;

    double *y0_im = out_im_aligned;
    double *y1_im = out_im_aligned + K;
    double *y2_im = out_im_aligned + 2 * K;
    double *y3_im = out_im_aligned + 3 * K;

    const __m128d sign_mask = _mm_set1_pd(-0.0);

    radix4_n1_simple_loop_bv_sse2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                  sign_mask);
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Main entry point - Forward radix-4 stage - NO TWIDDLES (n1) - SSE2
 */
FORCE_INLINE void fft_radix4_n1_forward_stage_sse2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    radix4_n1_stage_fv_sse2(N, K, in_re, in_im, out_re, out_im);
}

/**
 * @brief Main entry point - Backward radix-4 stage - NO TWIDDLES (n1) - SSE2
 */
FORCE_INLINE void fft_radix4_n1_backward_stage_sse2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    radix4_n1_stage_bv_sse2(N, K, in_re, in_im, out_re, out_im);
}

#endif // __SSE2__

#endif // FFT_RADIX4_SSE2_N1_H
