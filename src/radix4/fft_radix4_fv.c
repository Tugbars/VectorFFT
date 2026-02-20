/**
 * @file fft_radix4_fv.c
 * @brief TRUE END-TO-END SoA Radix-4 FFT Implementation - Forward
 * 
 * Dispatch: AVX-512 > AVX2 > Scalar
 * 
 * @author VectorFFT Team
 * @version 2.3 (AVX512 + AVX2 + Scalar)
 * @date 2025
 */

#include "fft_radix4_uniform.h"
#include "avx2/fft_radix4_avx2.h"
#include "scalar/fft_radix4_scalar.h"

#ifdef __AVX512F__
    #include "avx512/fft_radix4_avx512n1.h"
    #include "avx512/fft_radix4_avx512.h"
#endif

#include "avx2/fft_radix4_avx2n1.h"

#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#define CACHE_LINE_BYTES 64
#define REQUIRED_ALIGNMENT 32
#define VECTOR_WIDTH 4

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
            cached_value = -1;
        else if (env[0] == '0')
            cached_value = 0;
        else if (env[0] == '1')
            cached_value = 1;
        else
            cached_value = -1;
    }
    
    return cached_value;
}

//==============================================================================
// HELPER: Process Range (Native SoA) - WITH TWIDDLES - FORWARD
//==============================================================================

static void radix4_process_range_native_soa_fv(
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
    const double *restrict a_re = in_re + k_start;
    const double *restrict a_im = in_im + k_start;
    
    int range_K = k_end - k_start;
    (void)K; (void)use_streaming;

#ifdef __AVX512F__
    {
        fft_twiddles_soa tw_local;
        tw_local.re = stage_tw->re + k_start;
        tw_local.im = stage_tw->im + k_start;
        
        fft_radix4_tw_forward_stage_avx512((size_t)N, (size_t)range_K,
                                           a_re, a_im,
                                           out_re + k_start, out_im + k_start,
                                           &tw_local);
        return;
    }
#elif defined(__AVX2__)
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = stage_tw->re + k_start;
        tw_local.im = stage_tw->im + k_start;
        
        radix4_stage_baseptr_fv_avx2(N, range_K, a_re, a_im,
                                     out_re + k_start, out_im + k_start,
                                     &tw_local, is_write_only, is_cold_out);
        return;
    }
#endif
    
    /* Scalar fallback */
    {
        fft_twiddles_soa tw_local;
        tw_local.re = stage_tw->re + k_start;
        tw_local.im = stage_tw->im + k_start;
        
        radix4_stage_baseptr_fv_scalar(N, range_K, a_re, a_im,
                                       out_re + k_start, out_im + k_start,
                                       &tw_local);
    }
}

//==============================================================================
// HELPER: Process Range (Native SoA) - NO TWIDDLES (N1) - FORWARD
//==============================================================================

static void radix4_process_range_native_soa_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K,
    int N,
    int k_start,
    int k_end)
{
    (void)k_start; (void)k_end;
    
#ifdef __AVX512F__
    fft_radix4_n1_forward_stage_avx512((size_t)N, (size_t)K,
                                       in_re, in_im, out_re, out_im);
#elif defined(__AVX2__)
    fft_radix4_n1_forward_stage_avx2((size_t)N, (size_t)K,
                                     in_re, in_im, out_re, out_im);
#else
    const double *a_re = in_re,        *a_im = in_im;
    const double *b_re = in_re + K,    *b_im = in_im + K;
    const double *c_re = in_re + 2*K,  *c_im = in_im + 2*K;
    const double *d_re = in_re + 3*K,  *d_im = in_im + 3*K;
    double *y0_re = out_re,        *y0_im = out_im;
    double *y1_re = out_re + K,    *y1_im = out_im + K;
    double *y2_re = out_re + 2*K,  *y2_im = out_im + 2*K;
    double *y3_re = out_re + 3*K,  *y3_im = out_im + 3*K;
    for (int k = 0; k < K; k++)
        radix4_butterfly_n1_scalar_fv((size_t)k,
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
#endif
}

//==============================================================================
// MAIN: FORWARD (WITH TWIDDLES)
//==============================================================================

void fft_radix4_fv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw, int K)
{
    const int N = 4 * K;
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    const double *tw_re_a = (const double*)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    const double *tw_im_a = (const double*)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
    (void)tw_re_a; (void)tw_im_a;
#endif
    int nt_env = check_nt_env_override();
    const size_t wfp = 4ull * K * sizeof(double);
    const int oop = (in_re != out_re) && (in_im != out_im);
    int use_nt = 0;
    if (nt_env == 0) use_nt = 0;
    else if (nt_env == 1) use_nt = oop;
    else use_nt = oop && (K >= NT_MIN_K) && (wfp > (size_t)(NT_THRESHOLD * LLC_BYTES));
    if (use_nt) {
        if (((uintptr_t)out_re % REQUIRED_ALIGNMENT) || ((uintptr_t)out_im % REQUIRED_ALIGNMENT))
            use_nt = 0;
    }
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0);
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0);
    radix4_process_range_native_soa_fv(out_re, out_im, in_re, in_im,
                                       stage_tw, K, N, 0, K, use_nt);
}

//==============================================================================
// MAIN: FORWARD (NO TWIDDLES)
//==============================================================================

void fft_radix4_fv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im, int K)
{
    const int N = 4 * K;
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif
    radix4_process_range_native_soa_fv_n1(out_re, out_im, in_re, in_im, K, N, 0, K);
}