/**
 * @file fft_radix7_dif_dispatch.h
 * @brief Radix-7 DIF cross-ISA dispatch — bwd only (genfft)
 * Include AFTER fft_radix7_dispatch.h.
 */
#ifndef FFT_RADIX7_DIF_DISPATCH_H
#define FFT_RADIX7_DIF_DISPATCH_H

#include "scalar/fft_radix7_scalar_dif_tw.h"
#ifdef __AVX2__
#include "avx2/fft_radix7_avx2_dif_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix7_avx512_dif_tw.h"
#endif

static inline void radix7_tw_dif_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix7_tw_dif_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix7_tw_dif_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix7_tw_dif_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

#endif /* FFT_RADIX7_DIF_DISPATCH_H */