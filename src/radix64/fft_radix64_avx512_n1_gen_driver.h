/**
 * @file fft_radix64_avx512_n1_gen_driver.h
 * @brief Hybrid driver: generated kernel (fwd+bwd) for small K, ArchA for large K
 *
 * v3: Native backward kernel - no conjugate trick, no temp buffers.
 *
 * Generated header provides two functions per include:
 *   radix64_n1_dit_kernel_fwd_avx512()
 *   radix64_n1_dit_kernel_bwd_avx512()
 *
 * We multi-include to create aligned/unaligned variants of each.
 */

#ifndef FFT_RADIX64_AVX512_N1_GEN_DRIVER_H
#define FFT_RADIX64_AVX512_N1_GEN_DRIVER_H

#ifndef RADIX64_GEN_CROSSOVER_K
#define RADIX64_GEN_CROSSOVER_K 256
#endif

/* -- Variant 1: unaligned/unaligned -- */
#undef  R64G_LD
#undef  R64G_ST
#define R64G_LD(p)    _mm512_loadu_pd(p)
#define R64G_ST(p,v)  _mm512_storeu_pd((p),(v))
#define radix64_n1_dit_kernel_fwd_avx512  radix64_n1_dit_kernel_fwd_avx512_uu
#define radix64_n1_dit_kernel_bwd_avx512  radix64_n1_dit_kernel_bwd_avx512_uu
#include "fft_radix64_avx512_n1_gen.h"
#undef  radix64_n1_dit_kernel_fwd_avx512
#undef  radix64_n1_dit_kernel_bwd_avx512

/* -- Variant 2: aligned/aligned -- */
#undef  R64G_LD
#undef  R64G_ST
#undef  FFT_RADIX64_AVX512_N1_GEN_H
#define R64G_LD(p)    _mm512_load_pd(p)
#define R64G_ST(p,v)  _mm512_store_pd((p),(v))
#define radix64_n1_dit_kernel_fwd_avx512  radix64_n1_dit_kernel_fwd_avx512_aa
#define radix64_n1_dit_kernel_bwd_avx512  radix64_n1_dit_kernel_bwd_avx512_aa
#include "fft_radix64_avx512_n1_gen.h"
#undef  radix64_n1_dit_kernel_fwd_avx512
#undef  radix64_n1_dit_kernel_bwd_avx512

/* Restore defaults */
#undef  R64G_LD
#undef  R64G_ST
#define R64G_LD(p)    _mm512_loadu_pd(p)
#define R64G_ST(p,v)  _mm512_storeu_pd((p),(v))

/*============================================================================
 * FORWARD - hybrid dispatch
 *============================================================================*/
TARGET_AVX512
static void
radix64_n1_forward_avx512_gen(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 8 && "K must be >= 8");

    if (K <= RADIX64_GEN_CROSSOVER_K) {
        const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                         (uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;
        if (al)
            radix64_n1_dit_kernel_fwd_avx512_aa(in_re, in_im, out_re, out_im, K);
        else
            radix64_n1_dit_kernel_fwd_avx512_uu(in_re, in_im, out_re, out_im, K);
    } else {
        radix64_n1_forward_avx512(K, in_re, in_im, out_re, out_im);
    }
}

/*============================================================================
 * BACKWARD - native generated kernel, no conjugate trick
 *============================================================================*/
TARGET_AVX512
static void
radix64_n1_backward_avx512_gen(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 8 && "K must be >= 8");

    if (K <= RADIX64_GEN_CROSSOVER_K) {
        const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                         (uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;
        if (al)
            radix64_n1_dit_kernel_bwd_avx512_aa(in_re, in_im, out_re, out_im, K);
        else
            radix64_n1_dit_kernel_bwd_avx512_uu(in_re, in_im, out_re, out_im, K);
    } else {
        radix64_n1_backward_avx512(K, in_re, in_im, out_re, out_im);
    }
}

#endif /* FFT_RADIX64_AVX512_N1_GEN_DRIVER_H */
