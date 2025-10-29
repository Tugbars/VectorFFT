/**
 * @file fft_radix8_bv.c
 * @brief TRUE END-TO-END SoA Radix-8 FFT Implementation - Backward
 *
 * @details
 * This module implements native SoA radix-8 backward FFT with:
 * - Hybrid blocked twiddle system (BLOCKED4 for K≤256, BLOCKED2 for K>256)
 * - Standard twiddle version (fft_radix8_bv)
 * - Twiddle-less n1 version (fft_radix8_bv_n1) for first stage
 * - Adaptive NT stores for large transforms
 * - U=2 software pipelining
 *
 * @author VectorFFT Team
 * @version 3.1 (Hybrid Blocked System)
 * @date 2025
 */

#include "fft_radix8_uniform.h"

// Hybrid blocked twiddle implementations
#ifdef __AVX512F__
#include "fft_radix8_avx512_blocked_hybrid_fixed.h"
#endif

#ifdef __AVX2__
#include "fft_radix8_avx2_blocked_hybrid_fixed.h"
#endif

#ifdef __SSE2__
#include "fft_radix8_sse2_blocked_hybrid_fixed.h"
#endif

#include "fft_radix8_scalar_blocked_hybrid.h"

// N1 (twiddle-less) implementations
#ifdef __AVX512F__
#include "fft_radix8_avx512_n1.h"
#endif

#ifdef __AVX2__
#include "fft_radix8_avx2_n1.h"
#endif

#ifdef __SSE2__
#include "fft_radix8_sse2_n1.h"
#endif

#include "fft_radix8_scalar_n1.h"

#include <immintrin.h>
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
// HELPER: Process Range (Native SoA) - BLOCKED4 - BACKWARD
//==============================================================================

static void radix8_process_range_blocked4_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix8_stage_twiddles_blocked4_t *restrict stage_tw,
    int K,
    int N)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double *)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double *)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
#endif

#ifdef __AVX512F__
    radix8_stage_blocked4_backward_avx512(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __AVX2__
    radix8_stage_blocked4_backward_avx2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __SSE2__
    radix8_stage_blocked4_backward_sse2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

    radix8_stage_blocked4_backward_scalar(K, in_re, in_im, out_re, out_im, stage_tw);
}

//==============================================================================
// HELPER: Process Range (Native SoA) - BLOCKED2 - BACKWARD
//==============================================================================

static void radix8_process_range_blocked2_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix8_stage_twiddles_blocked2_t *restrict stage_tw,
    int K,
    int N)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double *)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double *)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
#endif

#ifdef __AVX512F__
    radix8_stage_blocked2_backward_avx512(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __AVX2__
    radix8_stage_blocked2_backward_avx2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __SSE2__
    radix8_stage_blocked2_backward_sse2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

    radix8_stage_blocked2_backward_scalar(K, in_re, in_im, out_re, out_im, stage_tw);
}

//==============================================================================
// HELPER: Process Range (Native SoA) - NO TWIDDLES (N1) - BACKWARD
//==============================================================================

static void radix8_process_range_n1_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K,
    int N)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif

#ifdef __AVX512F__
    radix8_stage_n1_backward_avx512(K, in_re, in_im, out_re, out_im);
    return;
#endif

#ifdef __AVX2__
    radix8_stage_n1_backward_avx2(K, in_re, in_im, out_re, out_im);
    return;
#endif

#ifdef __SSE2__
    radix8_stage_n1_backward_sse2(K, in_re, in_im, out_re, out_im);
    return;
#endif

    radix8_stage_n1_backward_scalar(K, in_re, in_im, out_re, out_im);
}

//==============================================================================
// MAIN FUNCTION: Radix-8 DIF Butterfly - NATIVE SoA - BACKWARD (WITH TWIDDLES)
//==============================================================================

/**
 * @brief Radix-8 DIF butterfly - Native SoA - Backward FFT - WITH TWIDDLES
 *
 * @details
 * Hybrid blocked twiddle system:
 * - BLOCKED4 for K ≤ 256 (43% bandwidth savings)
 * - BLOCKED2 for K > 256 (71% bandwidth savings)
 *
 * Applies twiddle factors and performs radix-8 butterfly with all optimizations:
 * - U=2 software pipelining
 * - Adaptive NT stores
 * - Prefetch tuning
 * - Hoisted constants
 *
 * CRITICAL: Planner must ensure proper alignment and mode selection.
 *
 * @param[out] out_re Output real array (N elements, N=8K, SoA layout)
 * @param[out] out_im Output imaginary array (N elements, SoA layout)
 * @param[in] in_re Input real array (N elements, SoA layout)
 * @param[in] in_im Input imaginary array (N elements, SoA layout)
 * @param[in] stage_tw_blocked4 BLOCKED4 twiddle factors (for K ≤ 256)
 * @param[in] stage_tw_blocked2 BLOCKED2 twiddle factors (for K > 256)
 * @param[in] K Transform eighth-size (N/8)
 */
void fft_radix8_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix8_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
    const radix8_stage_twiddles_blocked2_t *restrict stage_tw_blocked2,
    int K)
{
    const int N = 8 * K;

    // Choose twiddle mode based on K
    radix8_twiddle_mode_t mode = radix8_choose_twiddle_mode(K);

    if (mode == RADIX8_TW_BLOCKED4)
    {
        // K ≤ 256: Use BLOCKED4 (4 twiddle blocks loaded, 3 sign-flipped)
        radix8_process_range_blocked4_bv(out_re, out_im, in_re, in_im,
                                         stage_tw_blocked4, K, N);
    }
    else
    {
        // K > 256: Use BLOCKED2 (2 twiddle blocks loaded, 2 derived, 3 sign-flipped)
        radix8_process_range_blocked2_bv(out_re, out_im, in_re, in_im,
                                         stage_tw_blocked2, K, N);
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-8 DIF Butterfly - NATIVE SoA - BACKWARD (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-8 DIF butterfly - Native SoA - Backward FFT - NO TWIDDLES (n1)
 *
 * @details
 * Twiddle-less variant for first radix-8 stage (inverse) or when all W1=...=W7=1.
 * 50-70% faster than standard version due to:
 * - No twiddle loads
 * - No complex multiplications
 * - Reduced register pressure
 *
 * Use this for the very first stage of a backward mixed-radix FFT where
 * twiddle factors are unity.
 *
 * CRITICAL: Planner must ensure proper alignment.
 *
 * @param[out] out_re Output real array (N elements, N=8K, SoA layout)
 * @param[out] out_im Output imaginary array (N elements, SoA layout)
 * @param[in] in_re Input real array (N elements, SoA layout)
 * @param[in] in_im Input imaginary array (N elements, SoA layout)
 * @param[in] K Transform eighth-size (N/8)
 */
void fft_radix8_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    const int N = 8 * K;

    radix8_process_range_n1_bv(out_re, out_im, in_re, in_im, K, N);
}