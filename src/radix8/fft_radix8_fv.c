/**
 * @file fft_radix8_fv.c
 * @brief Radix-8 Forward FFT Stage Driver — ISA Dispatch
 *
 * @details
 * Native SoA radix-8 DIF forward butterfly.
 * Dispatch chain: AVX-512 → AVX2 → scalar (FMA).
 *
 * Hybrid blocked twiddle system:
 *   BLOCKED4 (K ≤ 256): load W1..W4, derive W5=W1·W4, W6=W2·W4, W7=W3·W4
 *   BLOCKED2 (K > 256): load W1,W2,  derive rest via complex mul/square
 *
 * Each ISA backend provides:
 *   - U=2 software pipelining
 *   - Adaptive NT stores (AVX2/AVX-512)
 *   - Adaptive prefetch NTA/T0 (AVX2/AVX-512)
 *   - Fast W8 micro-kernels (add/sub, no cmul)
 *
 * @version 4.0 (Sign-flip fix, separated N1)
 * @date 2025
 */

#include "fft_radix8.h"

/*============================================================================
 * ISA-SPECIFIC STAGE HEADERS
 *============================================================================*/

/* Blocked hybrid (B4/B2) implementations */
#ifdef __AVX512F__
#include "fft_radix8_avx512_blocked_hybrid_fixed.h"
#endif

#ifdef __AVX2__
#include "fft_radix8_avx2_blocked_hybrid_fixed.h"
#endif

#include "fft_radix8_scalar_blocked_hybrid_xe_optimized.h"

/* N1 (twiddle-less) implementations */
#ifdef __AVX512F__
#include "fft_radix8_avx512_n1.h"
#endif

#ifdef __AVX2__
#include "fft_radix8_avx2_n1.h"
#endif

#include "fft_radix8_scalar_n1.h"

/*============================================================================
 * ISA CONFIGURATION & MINIMUM K REQUIREMENTS
 *
 * Each ISA backend requires K to be a multiple of the vector width
 * AND large enough for U=2 software pipelining (2× vector width).
 *============================================================================*/

#if defined(__AVX512F__)
#define AVX512_MIN_K 16   /* 8-wide × U=2, K must be multiple of 8 */
#endif

#if defined(__AVX2__)
#define AVX2_MIN_K   8    /* 4-wide × U=2, K must be multiple of 4 */
#endif

/*============================================================================
 * DISPATCH: BLOCKED4 FORWARD
 *============================================================================*/

static void dispatch_blocked4_forward(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    const radix8_stage_twiddles_blocked4_t *tw,
    int K)
{
#if defined(__AVX512F__)
    if (K >= AVX512_MIN_K && (K & 7) == 0) {
        radix8_stage_blocked4_forward_avx512((size_t)K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
#if defined(__AVX2__)
    if (K >= AVX2_MIN_K && (K & 3) == 0) {
        radix8_stage_blocked4_forward_avx2((size_t)K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
    radix8_stage_blocked4_forward_scalar((size_t)K, in_re, in_im, out_re, out_im, tw);
}

/*============================================================================
 * DISPATCH: BLOCKED2 FORWARD
 *============================================================================*/

static void dispatch_blocked2_forward(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    const radix8_stage_twiddles_blocked2_t *tw,
    int K)
{
#if defined(__AVX512F__)
    if (K >= AVX512_MIN_K && (K & 7) == 0) {
        radix8_stage_blocked2_forward_avx512((size_t)K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
#if defined(__AVX2__)
    if (K >= AVX2_MIN_K && (K & 3) == 0) {
        radix8_stage_blocked2_forward_avx2((size_t)K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
    radix8_stage_blocked2_forward_scalar((size_t)K, in_re, in_im, out_re, out_im, tw);
}

/*============================================================================
 * DISPATCH: N1 FORWARD
 *============================================================================*/

static void dispatch_n1_forward(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    int K)
{
#if defined(__AVX512F__)
    if (K >= AVX512_MIN_K && (K & 7) == 0) {
        radix8_stage_n1_forward_avx512((size_t)K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
#if defined(__AVX2__)
    if (K >= AVX2_MIN_K && (K & 3) == 0) {
        radix8_stage_n1_forward_avx2((size_t)K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    radix8_stage_n1_forward_scalar((size_t)K, in_re, in_im, out_re, out_im);
}

/*============================================================================
 * PUBLIC: fft_radix8_fv — Forward with twiddles
 *============================================================================*/

void fft_radix8_fv(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    const radix8_stage_twiddles_blocked4_t *tw4,
    const radix8_stage_twiddles_blocked2_t *tw2,
    int K)
{
    if (K <= RADIX8_BLOCKED4_THRESHOLD)
        dispatch_blocked4_forward(out_re, out_im, in_re, in_im, tw4, K);
    else
        dispatch_blocked2_forward(out_re, out_im, in_re, in_im, tw2, K);
}

/*============================================================================
 * PUBLIC: fft_radix8_fv_n1 — Forward twiddle-less
 *============================================================================*/

void fft_radix8_fv_n1(
    double *out_re,
    double *out_im,
    const double *in_re,
    const double *in_im,
    int K)
{
    dispatch_n1_forward(out_re, out_im, in_re, in_im, K);
}