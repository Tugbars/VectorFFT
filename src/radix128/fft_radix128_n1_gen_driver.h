/**
 * @file fft_radix128_n1_gen_driver.h
 * @brief DFT-128 N1 driver — aligned/unaligned dispatch for all ISAs
 *
 * Includes the generated scalar, AVX2, and AVX-512 kernels, creating
 * aligned (_aa) and unaligned (_uu) variants for the SIMD paths.
 * Scalar has no alignment variants (always element-at-a-time).
 *
 * Provides per-ISA entry points:
 *   radix128_n1_forward_scalar(K, in_re, in_im, out_re, out_im)
 *   radix128_n1_backward_scalar(...)
 *   radix128_n1_forward_avx2(K, ...)
 *   radix128_n1_backward_avx2(...)
 *   radix128_n1_forward_avx512(K, ...)
 *   radix128_n1_backward_avx512(...)
 *
 * K constraints:
 *   scalar:  K >= 1 (any)
 *   avx2:    K >= 4, multiple of 4
 *   avx512:  K >= 8, multiple of 8
 */

#ifndef FFT_RADIX128_N1_GEN_DRIVER_H
#define FFT_RADIX128_N1_GEN_DRIVER_H

#include <stdint.h>  /* uintptr_t */
#include <assert.h>

/* ======================================================================
 * SCALAR — no alignment variants needed
 * ====================================================================== */

#include "fft_radix128_scalar_n1_gen.h"

static void
radix128_n1_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert(K >= 1);
    radix128_n1_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static void
radix128_n1_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert(K >= 1);
    radix128_n1_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ======================================================================
 * AVX2 — unaligned + aligned variants via multi-include
 * ====================================================================== */

/* -- Variant 1: unaligned -- */
#undef  R128A_LD
#undef  R128A_ST
#define R128A_LD(p)    _mm256_loadu_pd(p)
#define R128A_ST(p,v)  _mm256_storeu_pd((p),(v))
#define radix128_n1_dit_kernel_fwd_avx2  radix128_n1_dit_kernel_fwd_avx2_uu
#define radix128_n1_dit_kernel_bwd_avx2  radix128_n1_dit_kernel_bwd_avx2_uu
#include "fft_radix128_avx2_n1_gen.h"
#undef  radix128_n1_dit_kernel_fwd_avx2
#undef  radix128_n1_dit_kernel_bwd_avx2

/* -- Variant 2: aligned -- */
#undef  R128A_LD
#undef  R128A_ST
#undef  FFT_RADIX128_AVX2_N1_GEN_H
#define R128A_LD(p)    _mm256_load_pd(p)
#define R128A_ST(p,v)  _mm256_store_pd((p),(v))
#define radix128_n1_dit_kernel_fwd_avx2  radix128_n1_dit_kernel_fwd_avx2_aa
#define radix128_n1_dit_kernel_bwd_avx2  radix128_n1_dit_kernel_bwd_avx2_aa
#include "fft_radix128_avx2_n1_gen.h"
#undef  radix128_n1_dit_kernel_fwd_avx2
#undef  radix128_n1_dit_kernel_bwd_avx2

/* Restore defaults */
#undef  R128A_LD
#undef  R128A_ST
#define R128A_LD(p)    _mm256_loadu_pd(p)
#define R128A_ST(p,v)  _mm256_storeu_pd((p),(v))

__attribute__((target("avx2,fma")))
static void
radix128_n1_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 3) == 0 && "K must be multiple of 4 for AVX2");
    assert(K >= 4);

    const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                     (uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;
    if (al)
        radix128_n1_dit_kernel_fwd_avx2_aa(in_re, in_im, out_re, out_im, K);
    else
        radix128_n1_dit_kernel_fwd_avx2_uu(in_re, in_im, out_re, out_im, K);
}

__attribute__((target("avx2,fma")))
static void
radix128_n1_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 3) == 0 && "K must be multiple of 4 for AVX2");
    assert(K >= 4);

    const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                     (uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;
    if (al)
        radix128_n1_dit_kernel_bwd_avx2_aa(in_re, in_im, out_re, out_im, K);
    else
        radix128_n1_dit_kernel_bwd_avx2_uu(in_re, in_im, out_re, out_im, K);
}

/* ======================================================================
 * AVX-512 — unaligned + aligned variants via multi-include
 * ====================================================================== */

#ifdef __AVX512F__

/* -- Variant 1: unaligned -- */
#undef  R128G_LD
#undef  R128G_ST
#define R128G_LD(p)    _mm512_loadu_pd(p)
#define R128G_ST(p,v)  _mm512_storeu_pd((p),(v))
#define radix128_n1_dit_kernel_fwd_avx512  radix128_n1_dit_kernel_fwd_avx512_uu
#define radix128_n1_dit_kernel_bwd_avx512  radix128_n1_dit_kernel_bwd_avx512_uu
#include "fft_radix128_avx512_n1_gen.h"
#undef  radix128_n1_dit_kernel_fwd_avx512
#undef  radix128_n1_dit_kernel_bwd_avx512

/* -- Variant 2: aligned -- */
#undef  R128G_LD
#undef  R128G_ST
#undef  FFT_RADIX128_AVX512_N1_GEN_H
#define R128G_LD(p)    _mm512_load_pd(p)
#define R128G_ST(p,v)  _mm512_store_pd((p),(v))
#define radix128_n1_dit_kernel_fwd_avx512  radix128_n1_dit_kernel_fwd_avx512_aa
#define radix128_n1_dit_kernel_bwd_avx512  radix128_n1_dit_kernel_bwd_avx512_aa
#include "fft_radix128_avx512_n1_gen.h"
#undef  radix128_n1_dit_kernel_fwd_avx512
#undef  radix128_n1_dit_kernel_bwd_avx512

/* Restore defaults */
#undef  R128G_LD
#undef  R128G_ST
#define R128G_LD(p)    _mm512_loadu_pd(p)
#define R128G_ST(p,v)  _mm512_storeu_pd((p),(v))

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix128_n1_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 8);

    const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                     (uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;
    if (al)
        radix128_n1_dit_kernel_fwd_avx512_aa(in_re, in_im, out_re, out_im, K);
    else
        radix128_n1_dit_kernel_fwd_avx512_uu(in_re, in_im, out_re, out_im, K);
}

__attribute__((target("avx512f,avx512dq,fma")))
static void
radix128_n1_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 8);

    const int al = (((uintptr_t)in_re | (uintptr_t)in_im |
                     (uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;
    if (al)
        radix128_n1_dit_kernel_bwd_avx512_aa(in_re, in_im, out_re, out_im, K);
    else
        radix128_n1_dit_kernel_bwd_avx512_uu(in_re, in_im, out_re, out_im, K);
}

#endif /* __AVX512F__ */

#endif /* FFT_RADIX128_N1_GEN_DRIVER_H */
