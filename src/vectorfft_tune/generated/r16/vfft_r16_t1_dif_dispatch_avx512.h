/* vfft_r16_t1_dif_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=16 / dispatcher=t1_dif / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R16_T1_DIF_DISPATCH_AVX512_H
#define VFFT_R16_T1_DIF_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix16_avx512.h"

static inline void vfft_r16_t1_dif_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..∞]: ct_t1_dif
     */
    {
        radix16_t1_dif_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r16_t1_dif_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..∞]: ct_t1_dif
     */
    {
        radix16_t1_dif_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R16_T1_DIF_DISPATCH_AVX512_H */
