/**
 * @file fft_radix8_dif_dispatch.h
 * @brief Radix-8 DIF cross-ISA dispatch — twiddle AFTER butterfly
 *
 * Include AFTER fft_radix8_dispatch.h (reuses ISA detection).
 *
 * Entry points:
 *   radix8_tw_dif_forward  / radix8_tw_dif_backward
 */

#ifndef FFT_RADIX8_DIF_DISPATCH_H
#define FFT_RADIX8_DIF_DISPATCH_H

/* ── DIF codelet includes ── */
#include "scalar/fft_radix8_scalar_dif_tw.h"

#ifdef __AVX2__
#include "avx2/fft_radix8_avx2_dif_tw.h"
#endif

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix8_avx512_dif_tw.h"
#endif

/* ── DIF strided dispatch ── */

static inline void radix8_tw_dif_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix8_tw_dif_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                          tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix8_tw_dif_kernel_fwd_avx2(in_re, in_im, out_re, out_im,
                                        tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dif_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                      tw_re, tw_im, K);
}

static inline void radix8_tw_dif_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix8_tw_dif_kernel_bwd_avx512(in_re, in_im, out_re, out_im,
                                          tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix8_tw_dif_kernel_bwd_avx2(in_re, in_im, out_re, out_im,
                                        tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dif_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                      tw_re, tw_im, K);
}

#endif /* FFT_RADIX8_DIF_DISPATCH_H */
