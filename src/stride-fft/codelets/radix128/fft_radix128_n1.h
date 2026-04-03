/**
 * @file fft_radix128_n1.h
 * @brief Unified DFT-128 N1 codelet — auto-dispatches across ISAs
 *
 * Public API:
 *   fft_radix128_n1_forward(K, in_re, in_im, out_re, out_im)
 *   fft_radix128_n1_backward(K, in_re, in_im, out_re, out_im)
 *
 * Dispatch strategy:
 *   K >= 8 and K % 8 == 0 and AVX-512 available → AVX-512 kernel
 *   K >= 4 and K % 4 == 0                       → AVX2 kernel
 *   otherwise                                    → scalar kernel
 *
 * Scalar handles tail iterations (K not divisible by SIMD width) plus
 * the main loop for very small K (1, 2, 3).
 *
 * For K with mixed alignment (e.g. K=12: 8 via AVX-512 + 4 via AVX2),
 * the dispatch picks the widest ISA that divides K cleanly. The planner
 * is expected to choose K that aligns to the target ISA. If K doesn't
 * divide evenly, we fall through to the next narrower ISA.
 *
 * All kernels are twiddle-less (N1 codelets) — the caller handles
 * inter-pass twiddle factors.
 */

#ifndef FFT_RADIX128_N1_H
#define FFT_RADIX128_N1_H

#include <stddef.h>

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif

/* Pull in the driver which includes all ISA-specific generated kernels
 * and provides per-ISA entry points with aligned/unaligned dispatch. */
#include "fft_radix128_n1_gen_driver.h"

/* ======================================================================
 * Forward DFT-128 — unified dispatch
 * ====================================================================== */

static inline void
fft_radix128_n1_forward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) {
        radix128_n1_forward_avx512(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    if (K >= 4 && (K & 3) == 0) {
        radix128_n1_forward_avx2(K, in_re, in_im, out_re, out_im);
        return;
    }
    radix128_n1_forward_scalar(K, in_re, in_im, out_re, out_im);
}

/* ======================================================================
 * Backward DFT-128 — unified dispatch
 * ====================================================================== */

static inline void
fft_radix128_n1_backward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) {
        radix128_n1_backward_avx512(K, in_re, in_im, out_re, out_im);
        return;
    }
#endif
    if (K >= 4 && (K & 3) == 0) {
        radix128_n1_backward_avx2(K, in_re, in_im, out_re, out_im);
        return;
    }
    radix128_n1_backward_scalar(K, in_re, in_im, out_re, out_im);
}

#endif /* FFT_RADIX128_N1_H */
