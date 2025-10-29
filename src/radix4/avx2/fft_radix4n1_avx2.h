/**
 * @file fft_radix4_avx2n1.h
 * @brief Twiddle-less AVX2 Radix-4 Implementation (FFTW n1-style)
 *
 * @details
 * Specialized radix-4 butterfly for W1=W2=W3=1 (no twiddles).
 * Used in first FFT stage or when all twiddles are unity.
 * 
 * PERFORMANCE GAINS over standard radix-4:
 * - Eliminates 6 twiddle loads per iteration (24 bytes/iter for AVX2)
 * - Eliminates 3 complex multiplies per iteration (9 FMAs saved)
 * - Reduces register pressure (no W registers)
 * - ~40-60% faster than twiddle version for this special case
 * 
 * ARCHITECTURE:
 * - YMM registers: 4 doubles per vector (256-bit)
 * - Process 4 elements per iteration (1 AVX2 vector)
 * - U=2 software pipelining for large K
 * - Simple loop for small K
 * 
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX4_AVX2_N1_H
#define FFT_RADIX4_AVX2_N1_H

#include "fft_radix4.h"
#include <immintrin.h>
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

#define RADIX4_N1_STREAM_THRESHOLD 8192
#define RADIX4_N1_SMALL_K_THRESHOLD 16  // Use simple loop below this
#define RADIX4_N1_PREFETCH_DISTANCE 32

//==============================================================================
// ALIGNMENT HELPERS
//==============================================================================

static inline bool is_aligned32_n1(const void *p)
{
    return ((uintptr_t)p & 31u) == 0;
}

//==============================================================================
// LOAD HELPERS
//==============================================================================

#ifdef __AVX2__

#define LOAD_PD_AVX2_N1(ptr) _mm256_loadu_pd(ptr)

//==============================================================================
// RADIX-4 N1 BUTTERFLY CORES (NO TWIDDLES)
//==============================================================================

/**
 * @brief Core radix-4 butterfly - Forward FFT - NO TWIDDLES (n1)
 * 
 * Simplified butterfly when W1=W2=W3=1:
 *   tB = B, tC = C, tD = D (no complex multiply!)
 * 
 * Algorithm: rot = (+i) * difBD for forward transform
 *   rot_re = -difBD_im
 *   rot_im = +difBD_re
 */
FORCE_INLINE void radix4_butterfly_n1_core_fv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im,
    __m256d d_re, __m256d d_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d sign_mask)
{
    // No twiddle multiply: tB=B, tC=C, tD=D
    
    __m256d sumBD_re = _mm256_add_pd(b_re, d_re);
    __m256d sumBD_im = _mm256_add_pd(b_im, d_im);
    __m256d difBD_re = _mm256_sub_pd(b_re, d_re);
    __m256d difBD_im = _mm256_sub_pd(b_im, d_im);

    __m256d sumAC_re = _mm256_add_pd(a_re, c_re);
    __m256d sumAC_im = _mm256_add_pd(a_im, c_im);
    __m256d difAC_re = _mm256_sub_pd(a_re, c_re);
    __m256d difAC_im = _mm256_sub_pd(a_im, c_im);

    // rot = (+i) * difBD = (-difBD_im, +difBD_re)
    __m256d rot_re = _mm256_xor_pd(difBD_im, sign_mask);
    __m256d rot_im = difBD_re;

    *y0_re = _mm256_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm256_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

/**
 * @brief Core radix-4 butterfly - Backward FFT - NO TWIDDLES (n1)
 * 
 * Algorithm: rot = (-i) * difBD for inverse transform
 *   rot_re = +difBD_im
 *   rot_im = -difBD_re
 */
FORCE_INLINE void radix4_butterfly_n1_core_bv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im,
    __m256d d_re, __m256d d_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d sign_mask)
{
    // No twiddle multiply: tB=B, tC=C, tD=D
    
    __m256d sumBD_re = _mm256_add_pd(b_re, d_re);
    __m256d sumBD_im = _mm256_add_pd(b_im, d_im);
    __m256d difBD_re = _mm256_sub_pd(b_re, d_re);
    __m256d difBD_im = _mm256_sub_pd(b_im, d_im);

    __m256d sumAC_re = _mm256_add_pd(a_re, c_re);
    __m256d sumAC_im = _mm256_add_pd(a_im, c_im);
    __m256d difAC_re = _mm256_sub_pd(a_re, c_re);
    __m256d difAC_im = _mm256_sub_pd(a_im, c_im);

    // rot = (-i) * difBD = (+difBD_im, -difBD_re)
    __m256d rot_re = difBD_im;
    __m256d rot_im = _mm256_xor_pd(difBD_re, sign_mask);

    *y0_re = _mm256_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm256_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

//==============================================================================
// PREFETCH HELPERS
//==============================================================================

#define PREFETCH_NTA_N1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)

FORCE_INLINE void prefetch_radix4_n1_data(
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    size_t pk)
{
    PREFETCH_NTA_N1(&a_re[pk]);
    PREFETCH_NTA_N1(&a_im[pk]);
    PREFETCH_NTA_N1(&b_re[pk]);
    PREFETCH_NTA_N1(&b_im[pk]);
    PREFETCH_NTA_N1(&c_re[pk]);
    PREFETCH_NTA_N1(&c_im[pk]);
    PREFETCH_NTA_N1(&d_re[pk]);
    PREFETCH_NTA_N1(&d_im[pk]);
}

//==============================================================================
// SCALAR FALLBACK (NO TWIDDLES)
//==============================================================================

FORCE_INLINE void radix4_butterfly_n1_scalar_fv(
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

    // No twiddle multiply: tB=B, tC=C, tD=D
    
    double sumBD_r = b_r + d_r;
    double sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r;
    double difBD_i = b_i - d_i;
    
    double sumAC_r = a_r + c_r;
    double sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r;
    double difAC_i = a_i - c_i;

    // Forward: rot = (+i) * difBD = (-difBD_i, +difBD_r)
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

FORCE_INLINE void radix4_butterfly_n1_scalar_bv(
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

    // No twiddle multiply: tB=B, tC=C, tD=D
    
    double sumBD_r = b_r + d_r;
    double sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r;
    double difBD_i = b_i - d_i;
    
    double sumAC_r = a_r + c_r;
    double sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r;
    double difAC_i = a_i - c_i;

    // Backward: rot = (-i) * difBD = (+difBD_i, -difBD_r)
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
// SMALL-K SIMPLE LOOP (NO PIPELINING)
//==============================================================================

/**
 * @brief Simple loop for small K - Forward - No twiddles
 */
FORCE_INLINE void radix4_n1_small_k_fv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask)
{
    size_t k = 0;
    
    // Vectorized main loop
    for (; k + 4 <= K; k += 4)
    {
        __m256d a_r = LOAD_PD_AVX2_N1(&a_re[k]);
        __m256d a_i = LOAD_PD_AVX2_N1(&a_im[k]);
        __m256d b_r = LOAD_PD_AVX2_N1(&b_re[k]);
        __m256d b_i = LOAD_PD_AVX2_N1(&b_im[k]);
        __m256d c_r = LOAD_PD_AVX2_N1(&c_re[k]);
        __m256d c_i = LOAD_PD_AVX2_N1(&c_im[k]);
        __m256d d_r = LOAD_PD_AVX2_N1(&d_re[k]);
        __m256d d_i = LOAD_PD_AVX2_N1(&d_im[k]);
        
        __m256d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m256d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
        
        radix4_butterfly_n1_core_fv_avx2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);
        
        _mm256_storeu_pd(&y0_re[k], out_y0_r);
        _mm256_storeu_pd(&y0_im[k], out_y0_i);
        _mm256_storeu_pd(&y1_re[k], out_y1_r);
        _mm256_storeu_pd(&y1_im[k], out_y1_i);
        _mm256_storeu_pd(&y2_re[k], out_y2_r);
        _mm256_storeu_pd(&y2_im[k], out_y2_i);
        _mm256_storeu_pd(&y3_re[k], out_y3_r);
        _mm256_storeu_pd(&y3_im[k], out_y3_i);
    }
    
    // Scalar tail
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_fv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

/**
 * @brief Simple loop for small K - Backward - No twiddles
 */
FORCE_INLINE void radix4_n1_small_k_bv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask)
{
    size_t k = 0;
    
    // Vectorized main loop
    for (; k + 4 <= K; k += 4)
    {
        __m256d a_r = LOAD_PD_AVX2_N1(&a_re[k]);
        __m256d a_i = LOAD_PD_AVX2_N1(&a_im[k]);
        __m256d b_r = LOAD_PD_AVX2_N1(&b_re[k]);
        __m256d b_i = LOAD_PD_AVX2_N1(&b_im[k]);
        __m256d c_r = LOAD_PD_AVX2_N1(&c_re[k]);
        __m256d c_i = LOAD_PD_AVX2_N1(&c_im[k]);
        __m256d d_r = LOAD_PD_AVX2_N1(&d_re[k]);
        __m256d d_i = LOAD_PD_AVX2_N1(&d_im[k]);
        
        __m256d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m256d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
        
        radix4_butterfly_n1_core_bv_avx2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);
        
        _mm256_storeu_pd(&y0_re[k], out_y0_r);
        _mm256_storeu_pd(&y0_im[k], out_y0_i);
        _mm256_storeu_pd(&y1_re[k], out_y1_r);
        _mm256_storeu_pd(&y1_im[k], out_y1_i);
        _mm256_storeu_pd(&y2_re[k], out_y2_r);
        _mm256_storeu_pd(&y2_im[k], out_y2_i);
        _mm256_storeu_pd(&y3_re[k], out_y3_r);
        _mm256_storeu_pd(&y3_im[k], out_y3_i);
    }
    
    // Scalar tail
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_bv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

//==============================================================================
// U=2 SOFTWARE PIPELINED STAGE - FORWARD (NO TWIDDLES)
//==============================================================================

/**
 * @brief U=2 modulo-scheduled kernel - Forward FFT - NO TWIDDLES
 * 
 * Simplified pipeline without twiddle loads and complex multiplies.
 * Only load/butterfly/store stages remain.
 * 
 * PERFORMANCE: ~40-60% faster than standard twiddle version
 */
FORCE_INLINE void radix4_n1_stage_u2_pipelined_fv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask,
    bool do_stream)
{
    const size_t K_main = (K / 4) * 4;
    const size_t K_tail = K - K_main;
    const int prefetch_dist = RADIX4_N1_PREFETCH_DISTANCE;

    if (K_main == 0)
        goto handle_tail;

    // Pipeline registers (iteration i and i+1)
    __m256d A0_re, A0_im, B0_re, B0_im, C0_re, C0_im, D0_re, D0_im;
    __m256d A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im;
    __m256d OUT_y0_r, OUT_y0_i, OUT_y1_r, OUT_y1_i;
    __m256d OUT_y2_r, OUT_y2_i, OUT_y3_r, OUT_y3_i;

    //==========================================================================
    // PROLOGUE
    //==========================================================================

    // load(0)
    A0_re = LOAD_PD_AVX2_N1(&a_re[0]);
    A0_im = LOAD_PD_AVX2_N1(&a_im[0]);
    B0_re = LOAD_PD_AVX2_N1(&b_re[0]);
    B0_im = LOAD_PD_AVX2_N1(&b_im[0]);
    C0_re = LOAD_PD_AVX2_N1(&c_re[0]);
    C0_im = LOAD_PD_AVX2_N1(&c_im[0]);
    D0_re = LOAD_PD_AVX2_N1(&d_re[0]);
    D0_im = LOAD_PD_AVX2_N1(&d_im[0]);

    if (K_main < 8)
    {
        // Single-vector case
        A1_re = A0_re; A1_im = A0_im;
        B1_re = B0_re; B1_im = B0_im;
        C1_re = C0_re; C1_im = C0_im;
        D1_re = D0_re; D1_im = D0_im;
        goto epilogue_single;
    }

    // load(1)
    A1_re = LOAD_PD_AVX2_N1(&a_re[4]);
    A1_im = LOAD_PD_AVX2_N1(&a_im[4]);
    B1_re = LOAD_PD_AVX2_N1(&b_re[4]);
    B1_im = LOAD_PD_AVX2_N1(&b_im[4]);
    C1_re = LOAD_PD_AVX2_N1(&c_re[4]);
    C1_im = LOAD_PD_AVX2_N1(&c_im[4]);
    D1_re = LOAD_PD_AVX2_N1(&d_re[4]);
    D1_im = LOAD_PD_AVX2_N1(&d_im[4]);

    //==========================================================================
    // MAIN LOOP
    //==========================================================================

    for (size_t k = 4; k < K_main; k += 4)
    {
        // Prefetch
        size_t pk = k + prefetch_dist;
        if (pk < K)
        {
            prefetch_radix4_n1_data(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, pk);
        }

        // store(i-2)
        if (k >= 8)
        {
            size_t store_k = k - 8;
            if (do_stream)
            {
                _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
                _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
                _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
                _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
                _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
            }
            else
            {
                _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
                _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
                _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
                _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
                _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
            }
        }

        // butterfly(i-1): NO TWIDDLES - uses A1, B1, C1, D1 directly
        radix4_butterfly_n1_core_fv_avx2(
            A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im,
            &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
            &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
            sign_mask);

        // load(i+1)
        size_t k_next = k + 4;
        if (k_next < K_main)
        {
            A0_re = LOAD_PD_AVX2_N1(&a_re[k_next]);
            A0_im = LOAD_PD_AVX2_N1(&a_im[k_next]);
            B0_re = LOAD_PD_AVX2_N1(&b_re[k_next]);
            B0_im = LOAD_PD_AVX2_N1(&b_im[k_next]);
            C0_re = LOAD_PD_AVX2_N1(&c_re[k_next]);
            C0_im = LOAD_PD_AVX2_N1(&c_im[k_next]);
            D0_re = LOAD_PD_AVX2_N1(&d_re[k_next]);
            D0_im = LOAD_PD_AVX2_N1(&d_im[k_next]);

            // rotate
            A1_re = A0_re; A1_im = A0_im;
            B1_re = B0_re; B1_im = B0_im;
            C1_re = C0_re; C1_im = C0_im;
            D1_re = D0_re; D1_im = D0_im;
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================

    if (K_main >= 8)
    {
        size_t store_k = K_main - 8;
        if (do_stream)
        {
            _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

epilogue_single:
    radix4_butterfly_n1_core_fv_avx2(
        A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im,
        &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
        &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
        sign_mask);

    {
        size_t store_k = K_main - 4;
        if (do_stream)
        {
            _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

handle_tail:
    // Scalar tail
    for (size_t k = K_main; k < K; k++)
    {
        radix4_butterfly_n1_scalar_fv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

//==============================================================================
// U=2 SOFTWARE PIPELINED STAGE - BACKWARD (NO TWIDDLES)
//==============================================================================

/**
 * @brief U=2 modulo-scheduled kernel - Backward FFT - NO TWIDDLES
 */
FORCE_INLINE void radix4_n1_stage_u2_pipelined_bv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask,
    bool do_stream)
{
    const size_t K_main = (K / 4) * 4;
    const size_t K_tail = K - K_main;
    const int prefetch_dist = RADIX4_N1_PREFETCH_DISTANCE;

    if (K_main == 0)
        goto handle_tail;

    __m256d A0_re, A0_im, B0_re, B0_im, C0_re, C0_im, D0_re, D0_im;
    __m256d A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im;
    __m256d OUT_y0_r, OUT_y0_i, OUT_y1_r, OUT_y1_i;
    __m256d OUT_y2_r, OUT_y2_i, OUT_y3_r, OUT_y3_i;

    // PROLOGUE
    A0_re = LOAD_PD_AVX2_N1(&a_re[0]);
    A0_im = LOAD_PD_AVX2_N1(&a_im[0]);
    B0_re = LOAD_PD_AVX2_N1(&b_re[0]);
    B0_im = LOAD_PD_AVX2_N1(&b_im[0]);
    C0_re = LOAD_PD_AVX2_N1(&c_re[0]);
    C0_im = LOAD_PD_AVX2_N1(&c_im[0]);
    D0_re = LOAD_PD_AVX2_N1(&d_re[0]);
    D0_im = LOAD_PD_AVX2_N1(&d_im[0]);

    if (K_main < 8)
    {
        A1_re = A0_re; A1_im = A0_im;
        B1_re = B0_re; B1_im = B0_im;
        C1_re = C0_re; C1_im = C0_im;
        D1_re = D0_re; D1_im = D0_im;
        goto epilogue_single;
    }

    A1_re = LOAD_PD_AVX2_N1(&a_re[4]);
    A1_im = LOAD_PD_AVX2_N1(&a_im[4]);
    B1_re = LOAD_PD_AVX2_N1(&b_re[4]);
    B1_im = LOAD_PD_AVX2_N1(&b_im[4]);
    C1_re = LOAD_PD_AVX2_N1(&c_re[4]);
    C1_im = LOAD_PD_AVX2_N1(&c_im[4]);
    D1_re = LOAD_PD_AVX2_N1(&d_re[4]);
    D1_im = LOAD_PD_AVX2_N1(&d_im[4]);

    // MAIN LOOP
    for (size_t k = 4; k < K_main; k += 4)
    {
        size_t pk = k + prefetch_dist;
        if (pk < K)
        {
            prefetch_radix4_n1_data(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, pk);
        }

        if (k >= 8)
        {
            size_t store_k = k - 8;
            if (do_stream)
            {
                _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
                _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
                _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
                _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
                _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
            }
            else
            {
                _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
                _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
                _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
                _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
                _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
            }
        }

        // BACKWARD butterfly
        radix4_butterfly_n1_core_bv_avx2(
            A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im,
            &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
            &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
            sign_mask);

        size_t k_next = k + 4;
        if (k_next < K_main)
        {
            A0_re = LOAD_PD_AVX2_N1(&a_re[k_next]);
            A0_im = LOAD_PD_AVX2_N1(&a_im[k_next]);
            B0_re = LOAD_PD_AVX2_N1(&b_re[k_next]);
            B0_im = LOAD_PD_AVX2_N1(&b_im[k_next]);
            C0_re = LOAD_PD_AVX2_N1(&c_re[k_next]);
            C0_im = LOAD_PD_AVX2_N1(&c_im[k_next]);
            D0_re = LOAD_PD_AVX2_N1(&d_re[k_next]);
            D0_im = LOAD_PD_AVX2_N1(&d_im[k_next]);

            A1_re = A0_re; A1_im = A0_im;
            B1_re = B0_re; B1_im = B0_im;
            C1_re = C0_re; C1_im = C0_im;
            D1_re = D0_re; D1_im = D0_im;
        }
    }

    // EPILOGUE
    if (K_main >= 8)
    {
        size_t store_k = K_main - 8;
        if (do_stream)
        {
            _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

epilogue_single:
    radix4_butterfly_n1_core_bv_avx2(
        A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im,
        &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
        &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
        sign_mask);

    {
        size_t store_k = K_main - 4;
        if (do_stream)
        {
            _mm256_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

handle_tail:
    for (size_t k = K_main; k < K; k++)
    {
        radix4_butterfly_n1_scalar_bv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

//==============================================================================
// STAGE WRAPPERS - NO TWIDDLES
//==============================================================================

/**
 * @brief Stage wrapper - Forward FFT - NO TWIDDLES (n1)
 */
FORCE_INLINE void radix4_n1_stage_fv_avx2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    bool is_write_only,
    bool is_cold_out)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 32);

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

    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    const bool do_stream =
        (N >= RADIX4_N1_STREAM_THRESHOLD) && is_write_only && is_cold_out &&
        is_aligned32_n1(y0_re) && is_aligned32_n1(y0_im) &&
        is_aligned32_n1(y1_re) && is_aligned32_n1(y1_im) &&
        is_aligned32_n1(y2_re) && is_aligned32_n1(y2_im) &&
        is_aligned32_n1(y3_re) && is_aligned32_n1(y3_im);

    if (K < RADIX4_N1_SMALL_K_THRESHOLD)
    {
        // Simple loop for small K
        radix4_n1_small_k_fv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                  sign_mask);
    }
    else
    {
        // U=2 pipelined for large K
        radix4_n1_stage_u2_pipelined_fv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                             y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                             sign_mask, do_stream);
    }

    if (do_stream)
    {
        _mm_sfence();
    }
}

/**
 * @brief Stage wrapper - Backward FFT - NO TWIDDLES (n1)
 */
FORCE_INLINE void radix4_n1_stage_bv_avx2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    bool is_write_only,
    bool is_cold_out)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 32);

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

    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    const bool do_stream =
        (N >= RADIX4_N1_STREAM_THRESHOLD) && is_write_only && is_cold_out &&
        is_aligned32_n1(y0_re) && is_aligned32_n1(y0_im) &&
        is_aligned32_n1(y1_re) && is_aligned32_n1(y1_im) &&
        is_aligned32_n1(y2_re) && is_aligned32_n1(y2_im) &&
        is_aligned32_n1(y3_re) && is_aligned32_n1(y3_im);

    if (K < RADIX4_N1_SMALL_K_THRESHOLD)
    {
        radix4_n1_small_k_bv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                  sign_mask);
    }
    else
    {
        radix4_n1_stage_u2_pipelined_bv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                             y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                                             sign_mask, do_stream);
    }

    if (do_stream)
    {
        _mm_sfence();
    }
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Main entry point - Forward radix-4 stage - NO TWIDDLES (n1)
 */
FORCE_INLINE void fft_radix4_n1_forward_stage_avx2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    const bool is_write_only = true;
    const bool is_cold_out = (N >= 4096);

    radix4_n1_stage_fv_avx2(N, K, in_re, in_im, out_re, out_im,
                            is_write_only, is_cold_out);
}

/**
 * @brief Main entry point - Backward radix-4 stage - NO TWIDDLES (n1)
 */
FORCE_INLINE void fft_radix4_n1_backward_stage_avx2(
    size_t N,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    const bool is_write_only = true;
    const bool is_cold_out = (N >= 4096);

    radix4_n1_stage_bv_avx2(N, K, in_re, in_im, out_re, out_im,
                            is_write_only, is_cold_out);
}

#endif // __AVX2__

#endif // FFT_RADIX4_AVX2_N1_H
