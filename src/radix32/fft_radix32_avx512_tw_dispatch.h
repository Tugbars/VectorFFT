/**
 * @file fft_radix32_avx512_tw_dispatch.h
 * @brief Twiddled DFT-32 AVX-512 dispatch: flat + ladder + NT stores
 *
 * Multi-include trick: includes fft_radix32_avx512_tw_ladder.h twice:
 *   1st: temporal stores → radix32_tw_ladder_dit_kernel_*
 *   2nd: NT stores       → radix32_tw_ladder_dit_kernel_*_nt
 *
 * Dispatch:
 *   K < 16          → flat U=1 (temporal)
 *   16 ≤ K < 128    → flat U=1 (temporal, twiddle table fits L1)
 *   128 ≤ K < THRESH→ ladder U=1 (temporal)
 *   K ≥ THRESH      → ladder U=1 NT + sfence
 *
 * NT_THRESH tunable — default 2048 based on twiddled codelet crossover data.
 */

#ifndef FFT_RADIX32_AVX512_TW_DISPATCH_H
#define FFT_RADIX32_AVX512_TW_DISPATCH_H

#include <immintrin.h>
#include <stddef.h>

#ifndef R32TW_NT_THRESH
#define R32TW_NT_THRESH 2048
#endif

/* ═══════════════════════════════════════════════════════════════
 * FIRST INCLUDE: temporal stores (default)
 * ═══════════════════════════════════════════════════════════════ */
#undef  R32L_LD
#undef  R32L_ST
#define R32L_LD(p)   _mm512_load_pd(p)
#define R32L_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_tw_ladder.h"

/* ═══════════════════════════════════════════════════════════════
 * SECOND INCLUDE: NT stores → *_nt function variants
 *
 * Undef include guard, redefine ST macro, rename functions via
 * preprocessor token paste.
 * ═══════════════════════════════════════════════════════════════ */
#undef FFT_RADIX32_AVX512_TW_LADDER_H
#undef  R32L_LD
#undef  R32L_ST
#define R32L_LD(p)   _mm512_load_pd(p)
#define R32L_ST(p,v) _mm512_stream_pd((p),(v))

/* Rename all 6 functions to *_nt variants */
#define radix32_tw_flat_dit_kernel_fwd_avx512     radix32_tw_flat_dit_kernel_fwd_avx512_nt
#define radix32_tw_flat_dit_kernel_bwd_avx512     radix32_tw_flat_dit_kernel_bwd_avx512_nt
#define radix32_tw_ladder_dit_kernel_fwd_avx512_u1  radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt
#define radix32_tw_ladder_dit_kernel_bwd_avx512_u1  radix32_tw_ladder_dit_kernel_bwd_avx512_u1_nt
#define radix32_tw_ladder_dit_kernel_fwd_avx512_u2  radix32_tw_ladder_dit_kernel_fwd_avx512_u2_nt
#define radix32_tw_ladder_dit_kernel_bwd_avx512_u2  radix32_tw_ladder_dit_kernel_bwd_avx512_u2_nt

#include "fft_radix32_avx512_tw_ladder.h"

/* Undo renames */
#undef radix32_tw_flat_dit_kernel_fwd_avx512
#undef radix32_tw_flat_dit_kernel_bwd_avx512
#undef radix32_tw_ladder_dit_kernel_fwd_avx512_u1
#undef radix32_tw_ladder_dit_kernel_bwd_avx512_u1
#undef radix32_tw_ladder_dit_kernel_fwd_avx512_u2
#undef radix32_tw_ladder_dit_kernel_bwd_avx512_u2

/* ═══════════════════════════════════════════════════════════════
 * UNIFIED DISPATCH — forward
 *
 * Flat table:   tw_re[31*K], tw_im[31*K]
 * Ladder table: base_tw_re[5*K], base_tw_im[5*K]
 *
 * Caller must provide BOTH tables — dispatch selects internally.
 * (For production, the planner would only allocate the needed one.)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_dispatch_fwd_avx512(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    const double * __restrict__ base_tw_re,
    const double * __restrict__ base_tw_im)
{
    if (K < 128) {
        /* Flat: twiddle table ≤ 62*64*8 = 32KB, fits L1 */
        radix32_tw_flat_dit_kernel_fwd_avx512(
            in_re, in_im, out_re, out_im, flat_tw_re, flat_tw_im, K);
    } else if (K < R32TW_NT_THRESH) {
        /* Ladder temporal: 5 loads/k-step, table fits L1 */
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1(
            in_re, in_im, out_re, out_im, base_tw_re, base_tw_im, K);
    } else {
        /* Ladder NT: bypass cache for output writes */
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(
            in_re, in_im, out_re, out_im, base_tw_re, base_tw_im, K);
        _mm_sfence();
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_dispatch_bwd_avx512(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    const double * __restrict__ base_tw_re,
    const double * __restrict__ base_tw_im)
{
    if (K < 128) {
        radix32_tw_flat_dit_kernel_bwd_avx512(
            in_re, in_im, out_re, out_im, flat_tw_re, flat_tw_im, K);
    } else if (K < R32TW_NT_THRESH) {
        radix32_tw_ladder_dit_kernel_bwd_avx512_u1(
            in_re, in_im, out_re, out_im, base_tw_re, base_tw_im, K);
    } else {
        radix32_tw_ladder_dit_kernel_bwd_avx512_u1_nt(
            in_re, in_im, out_re, out_im, base_tw_re, base_tw_im, K);
        _mm_sfence();
    }
}

#endif /* FFT_RADIX32_AVX512_TW_DISPATCH_H */
