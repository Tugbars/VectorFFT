/**
 * @file fft_radix3_bv.c
 * @brief Backward Radix-3 Stage Driver — ISA Dispatch
 *
 * Mirrors fft_radix3_fv.c with backward butterfly kernels.
 *
 * @version 1.0
 * @date 2025
 */

#include "fft_radix3.h"

#if defined(__AVX512F__)
#include "avx512/fft_radix3_avx512.h"
#include "avx512/fft_radix3_avx512_n1.h"
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include "avx2/fft_radix3_avx2.h"
#include "avx2/fft_radix3_avx2_n1.h"
#endif

#include "scalar/fft_radix3_scalar.h"
#include "scalar/fft_radix3_scalar_n1.h"

#define AVX512_MIN_K  8
#define AVX2_MIN_K    4

/*============================================================================
 * BACKWARD — TWIDDLED
 *============================================================================*/

void fft_radix3_bv(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im,
    const radix3_stage_twiddles_t *tw)
{
#if defined(__AVX512F__)
    if (K >= AVX512_MIN_K) {
        radix3_stage_backward_avx512(K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if (K >= AVX2_MIN_K) {
        radix3_stage_backward_avx2(K, in_re, in_im, out_re, out_im, tw);
        return;
    }
#endif
    radix3_stage_backward_scalar(K, in_re, in_im, out_re, out_im, tw);
}

/*============================================================================
 * BACKWARD — N1 (twiddle-less)
 *============================================================================*/

void fft_radix3_bv_n1(
    size_t K,
    const double *in_re,
    const double *in_im,
    double       *out_re,
    double       *out_im)
{
#if defined(__AVX512F__)
    if (K >= AVX512_MIN_K) {
        radix3_stage_n1_backward_avx512(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if (K >= AVX2_MIN_K) {
        radix3_stage_n1_backward_avx2(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    radix3_stage_n1_backward_scalar(K, in_re, in_im, out_re, out_im);
}