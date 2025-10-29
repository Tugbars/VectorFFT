/**
 * @file fft_radix4_bv.c
 * @brief TRUE END-TO-END SoA Radix-4 FFT Implementation - Backward
 * 
 * @details
 * This module implements native SoA radix-4 backward FFT with:
 * - Standard twiddle version (fft_radix4_bv)
 * - Twiddle-less n1 version (fft_radix4_bv_n1) for first stage
 * 
 * @author VectorFFT Team
 * @version 2.2 (with N1 support)
 * @date 2025
 */

#include "fft_radix4_uniform.h"

// Standard twiddle implementations
#ifdef __AVX512F__
    #include "fft_radix4_avx512_corrected_part1.h"
    #include "fft_radix4_avx512_corrected_part2.h"
#endif

#ifdef __AVX2__
    #include "fft_radix4_avx2_u2_pipelined.h"
#endif

#ifdef __SSE2__
    #include "fft_radix4_sse2_u2_pipelined.h"
#endif

#include "fft_radix4_scalar_optimized.h"

// N1 (twiddle-less) implementations
#ifdef __AVX512F__
    #include "fft_radix4_avx512_n1.h"
#endif

#ifdef __AVX2__
    #include "fft_radix4_avx2_n1.h"
#endif

#ifdef __SSE2__
    #include "fft_radix4_sse2_n1.h"
#endif

#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#define CACHE_LINE_BYTES 64

#if defined(__AVX512F__)
    #define REQUIRED_ALIGNMENT 64
    #define VECTOR_WIDTH 8
#elif defined(__AVX2__) || defined(__AVX__)
    #define REQUIRED_ALIGNMENT 32
    #define VECTOR_WIDTH 4
#elif defined(__SSE2__)
    #define REQUIRED_ALIGNMENT 16
    #define VECTOR_WIDTH 2
#else
    #define REQUIRED_ALIGNMENT 8
    #define VECTOR_WIDTH 1
#endif

#ifndef LLC_BYTES
    #define LLC_BYTES (8 * 1024 * 1024)
#endif

#define NT_THRESHOLD 0.7
#define NT_MIN_K 4096

//==============================================================================
// HELPER: Environment Variable Parsing
//==============================================================================

static inline int check_nt_env_override(void)
{
    static int cached_value = -2;
    
    if (cached_value == -2)
    {
        const char *env = getenv("FFT_NT");
        if (env == NULL)
        {
            cached_value = -1;
        }
        else if (env[0] == '0')
        {
            cached_value = 0;
        }
        else if (env[0] == '1')
        {
            cached_value = 1;
        }
        else
        {
            cached_value = -1;
        }
    }
    
    return cached_value;
}

//==============================================================================
// HELPER: Process Range (Native SoA) - WITH TWIDDLES - BACKWARD
//==============================================================================

static void radix4_process_range_native_soa_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K,
    int N,
    int k_start,
    int k_end,
    int use_streaming)
{
    const double *restrict a_re = in_re;
    const double *restrict b_re = in_re + K;
    const double *restrict c_re = in_re + 2 * K;
    const double *restrict d_re = in_re + 3 * K;
    
    const double *restrict a_im = in_im;
    const double *restrict b_im = in_im + K;
    const double *restrict c_im = in_im + 2 * K;
    const double *restrict d_im = in_im + 3 * K;
    
    double *restrict y0_re = out_re;
    double *restrict y1_re = out_re + K;
    double *restrict y2_re = out_re + 2 * K;
    double *restrict y3_re = out_re + 3 * K;
    
    double *restrict y0_im = out_im;
    double *restrict y1_im = out_im + K;
    double *restrict y2_im = out_im + 2 * K;
    double *restrict y3_im = out_im + 3 * K;
    
    const double *restrict tw_re = stage_tw->re;
    const double *restrict tw_im = stage_tw->im;
    
    int range_K = k_end - k_start;
    
    a_re += k_start; a_im += k_start;
    b_re += k_start; b_im += k_start;
    c_re += k_start; c_im += k_start;
    d_re += k_start; d_im += k_start;
    
    y0_re += k_start; y0_im += k_start;
    y1_re += k_start; y1_im += k_start;
    y2_re += k_start; y2_im += k_start;
    y3_re += k_start; y3_im += k_start;
    
#ifdef __AVX512F__
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_bv_avx512(N, range_K, a_re, a_im,
                                       out_re + k_start, out_im + k_start,
                                       &tw_local, is_write_only, is_cold_out);
        return;
    }
#endif
    
#ifdef __AVX2__
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_bv_avx2(N, range_K, a_re, a_im,
                                     out_re + k_start, out_im + k_start,
                                     &tw_local, is_write_only, is_cold_out);
        return;
    }
#endif
    
#ifdef __SSE2__
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_bv_sse2(N, range_K, a_re, a_im,
                                     out_re + k_start, out_im + k_start,
                                     &tw_local, is_write_only, is_cold_out);
        return;
    }
#endif
    
    {
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_bv_scalar(N, range_K, a_re, a_im,
                                       out_re + k_start, out_im + k_start,
                                       &tw_local);
    }
}

//==============================================================================
// HELPER: Process Range (Native SoA) - NO TWIDDLES (N1) - BACKWARD
//==============================================================================

static void radix4_process_range_native_soa_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K,
    int N,
    int k_start,
    int k_end)
{
    const double *restrict a_re = in_re;
    const double *restrict b_re = in_re + K;
    const double *restrict c_re = in_re + 2 * K;
    const double *restrict d_re = in_re + 3 * K;
    
    const double *restrict a_im = in_im;
    const double *restrict b_im = in_im + K;
    const double *restrict c_im = in_im + 2 * K;
    const double *restrict d_im = in_im + 3 * K;
    
    double *restrict y0_re = out_re;
    double *restrict y1_re = out_re + K;
    double *restrict y2_re = out_re + 2 * K;
    double *restrict y3_re = out_re + 3 * K;
    
    double *restrict y0_im = out_im;
    double *restrict y1_im = out_im + K;
    double *restrict y2_im = out_im + 2 * K;
    double *restrict y3_im = out_im + 3 * K;
    
    int range_K = k_end - k_start;
    
    a_re += k_start; a_im += k_start;
    b_re += k_start; b_im += k_start;
    c_re += k_start; c_im += k_start;
    d_re += k_start; d_im += k_start;
    
    y0_re += k_start; y0_im += k_start;
    y1_re += k_start; y1_im += k_start;
    y2_re += k_start; y2_im += k_start;
    y3_re += k_start; y3_im += k_start;
    
    int k = 0;
    
#ifdef __AVX512F__
    for (; k + 8 <= range_K; k += 8)
    {
        fft_radix4_n1_backward_stage_avx512(N, 8,
                                            a_re + k, a_im + k,
                                            y0_re + k, y0_im + k);
    }
#endif
    
#ifdef __AVX2__
    if (k + 4 <= range_K)
    {
        fft_radix4_n1_backward_stage_avx2(N, 4,
                                          a_re + k, a_im + k,
                                          y0_re + k, y0_im + k);
        k += 4;
    }
#endif
    
#ifdef __SSE2__
    if (k + 2 <= range_K)
    {
        fft_radix4_n1_backward_stage_sse2(N, 2,
                                          a_re + k, a_im + k,
                                          y0_re + k, y0_im + k);
        k += 2;
    }
#endif
    
    for (; k < range_K; k++)
    {
        radix4_butterfly_n1_scalar_bv_avx512(k,
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-4 DIF Butterfly - NATIVE SoA - BACKWARD (WITH TWIDDLES)
//==============================================================================

void fft_radix4_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    const int N = 4 * K;
    
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double*)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double*)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
#endif
    
    int nt_env_override = check_nt_env_override();
    
    const size_t write_footprint = 4ull * K * sizeof(double);
    const int is_out_of_place = (in_re != out_re) && (in_im != out_im);
    
    int use_streaming = 0;
    
    if (nt_env_override == 0)
    {
        use_streaming = 0;
    }
    else if (nt_env_override == 1)
    {
        use_streaming = is_out_of_place;
    }
    else
    {
        use_streaming = is_out_of_place &&
                       (K >= NT_MIN_K) &&
                       (write_footprint > (size_t)(NT_THRESHOLD * LLC_BYTES));
    }
    
    if (use_streaming)
    {
        uintptr_t r0 = (uintptr_t)&out_re[0];
        uintptr_t i0 = (uintptr_t)&out_im[0];
        
        if ((r0 % REQUIRED_ALIGNMENT) != 0 || (i0 % REQUIRED_ALIGNMENT) != 0)
        {
            use_streaming = 0;
        }
    }
    
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->re must be properly aligned for SIMD");
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->im must be properly aligned for SIMD");
    
    radix4_process_range_native_soa_bv(out_re, out_im, in_re, in_im,
                                       stage_tw, K, N, 0, K,
                                       use_streaming);
}

//==============================================================================
// MAIN FUNCTION: Radix-4 DIF Butterfly - NATIVE SoA - BACKWARD (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-4 DIF butterfly - NATIVE SoA - Backward FFT - NO TWIDDLES (n1)
 * 
 * @details
 * Twiddle-less variant for first radix-4 stage (inverse) or when all W1=W2=W3=1.
 * 40-60% faster than standard version.
 * 
 * @param[out] out_re Output real array (N elements, N=4K)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] K Transform quarter-size (N/4)
 */
void fft_radix4_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    const int N = 4 * K;
    
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif
    
    radix4_process_range_native_soa_bv_n1(out_re, out_im, in_re, in_im,
                                          K, N, 0, K);
}
