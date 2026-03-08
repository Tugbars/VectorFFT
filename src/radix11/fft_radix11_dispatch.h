/**
 * @file fft_radix11_dispatch.h
 * @brief Radix-11 dispatch — genfft kernels + pack+walk drivers
 */
#ifndef FFT_RADIX11_DISPATCH_H
#define FFT_RADIX11_DISPATCH_H

#include "fft_radix11_genfft.h"

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix11_avx512_tw_pack_walk.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix11_avx2_tw_pack_walk.h"
#endif

/* Strided dispatch — picks best available ISA */
static inline void r11_dispatch_fwd(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix11_genfft_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix11_genfft_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix11_genfft_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void r11_dispatch_bwd(
    const double * R11_RESTRICT in_re, const double * R11_RESTRICT in_im,
    double * R11_RESTRICT out_re, double * R11_RESTRICT out_im,
    size_t K)
{
    r11_dispatch_fwd(in_im, in_re, out_im, out_re, K);
}

#endif /* FFT_RADIX11_DISPATCH_H */