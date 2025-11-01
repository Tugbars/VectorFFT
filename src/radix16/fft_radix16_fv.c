/**
 * @file fft_radix16_fv_optimized.c
 * @brief Radix-16 Forward FFT - Native SoA with Multi-SIMD Support
 */

#include "fft_radix16_uniform_optimized.h"

// SIMD implementations
#ifdef __AVX512F__
#include "fft_radix16_avx512_native_soa_optimized.h"
#endif

#ifdef __AVX2__
#include "fft_radix16_avx2_native_soa_optimized.h"
#endif

#ifdef __SSE2__
#include "fft_radix16_sse2_native_soa_optimized.h"
#endif

#include "fft_radix16_scalar_native_soa_optimized.h"

// N1 implementations
#ifdef __AVX512F__
#include "fft_radix16_avx512_native_soa_n1.h"
#endif

#ifdef __AVX2__
#include "fft_radix16_avx2_native_soa_n1.h"
#endif

#ifdef __SSE2__
#include "fft_radix16_sse2_native_soa_n1.h"
#endif

#include "fft_radix16_scalar_native_soa_n1.h"

#include <stdint.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#if defined(__AVX512F__)
#define REQUIRED_ALIGNMENT 64
#elif defined(__AVX2__) || defined(__AVX__)
#define REQUIRED_ALIGNMENT 32
#elif defined(__SSE2__)
#define REQUIRED_ALIGNMENT 16
#else
#define REQUIRED_ALIGNMENT 8
#endif

//==============================================================================
// HELPER: Process Range - BLOCKED8 - FORWARD
//==============================================================================

static void radix16_process_range_blocked8_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw,
    int K)
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
    radix16_stage_blocked8_forward_avx512(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __AVX2__
    radix16_stage_blocked8_forward_avx2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __SSE2__
    radix16_stage_blocked8_forward_sse2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

    radix16_stage_blocked8_forward_scalar(K, in_re, in_im, out_re, out_im, stage_tw);
}

//==============================================================================
// HELPER: Process Range - BLOCKED4 - FORWARD
//==============================================================================

static void radix16_process_range_blocked4_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw,
    int K)
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
    radix16_stage_blocked4_forward_avx512(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __AVX2__
    radix16_stage_blocked4_forward_avx2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

#ifdef __SSE2__
    radix16_stage_blocked4_forward_sse2(K, in_re, in_im, out_re, out_im, stage_tw);
    return;
#endif

    radix16_stage_blocked4_forward_scalar(K, in_re, in_im, out_re, out_im, stage_tw);
}

//==============================================================================
// HELPER: Process Range - N1 - FORWARD
//==============================================================================

static void radix16_process_range_n1_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif

#ifdef __AVX512F__
    radix16_stage_n1_forward_avx512(K, in_re, in_im, out_re, out_im);
    return;
#endif

#ifdef __AVX2__
    radix16_stage_n1_forward_avx2(K, in_re, in_im, out_re, out_im);
    return;
#endif

#ifdef __SSE2__
    radix16_stage_n1_forward_sse2(K, in_re, in_im, out_re, out_im);
    return;
#endif

    radix16_stage_n1_forward_scalar(K, in_re, in_im, out_re, out_im);
}

//==============================================================================
// MAIN API - FORWARD TRANSFORM (WITH TWIDDLES)
//==============================================================================

void fft_radix16_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw_blocked8,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
    int K)
{
    // Choose twiddle mode based on K
    radix16_twiddle_mode_t mode = radix16_choose_twiddle_mode(K);

    if (mode == RADIX16_TW_BLOCKED4)
    {
        radix16_process_range_blocked4_fv(out_re, out_im, in_re, in_im,
                                          stage_tw_blocked4, K);
    }
    else
    {
        radix16_process_range_blocked8_fv(out_re, out_im, in_re, in_im,
                                          stage_tw_blocked8, K);
    }
}

//==============================================================================
// MAIN API - FORWARD TRANSFORM (NO TWIDDLES - N1)
//==============================================================================

void fft_radix16_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    radix16_process_range_n1_fv(out_re, out_im, in_re, in_im, K);
}