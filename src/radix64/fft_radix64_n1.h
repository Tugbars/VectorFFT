/**
 * @file fft_radix64_n1.h
 * @brief Unified DFT-64 N1 codelet — auto-dispatches across ISAs
 *
 * Public API:
 *   fft_radix64_n1_forward(K, in_re, in_im, out_re, out_im)
 *   fft_radix64_n1_backward(K, in_re, in_im, out_re, out_im)
 *
 * Dispatch:
 *   K >= 8 and K % 8 == 0 and AVX-512 → AVX-512
 *   K >= 4 and K % 4 == 0             → AVX2
 *   otherwise                          → scalar
 */

#ifndef FFT_RADIX64_N1_H
#define FFT_RADIX64_N1_H

#include <stddef.h>

#include "fft_radix64_n1_gen_driver.h"

static inline void
fft_radix64_n1_forward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) {
        radix64_n1_forward_avx512(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    if (K >= 4 && (K & 3) == 0) {
        radix64_n1_forward_avx2(K, in_re, in_im, out_re, out_im);
        return;
    }
    radix64_n1_forward_scalar(K, in_re, in_im, out_re, out_im);
}

static inline void
fft_radix64_n1_backward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) {
        radix64_n1_backward_avx512(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    if (K >= 4 && (K & 3) == 0) {
        radix64_n1_backward_avx2(K, in_re, in_im, out_re, out_im);
        return;
    }
    radix64_n1_backward_scalar(K, in_re, in_im, out_re, out_im);
}

#endif /* FFT_RADIX64_N1_H */
