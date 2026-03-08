/**
 * @file fft_radix25_dif_dispatch.h
 * @brief Radix-25 DIF cross-ISA dispatch — twiddle AFTER butterfly
 * Include AFTER fft_radix25_dispatch.h.
 */
#ifndef FFT_RADIX25_DIF_DISPATCH_H
#define FFT_RADIX25_DIF_DISPATCH_H

#include "scalar/fft_radix25_scalar_dif_tw.h"
#ifdef __AVX2__
#include "avx2/fft_radix25_avx2_dif_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix25_avx512_dif_tw.h"
#endif

static inline void radix25_tw_dif_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix25_tw_flat_dif_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix25_tw_flat_dif_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix25_tw_flat_dif_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

static inline void radix25_tw_dif_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix25_tw_flat_dif_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix25_tw_flat_dif_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix25_tw_flat_dif_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

#endif /* FFT_RADIX25_DIF_DISPATCH_H */