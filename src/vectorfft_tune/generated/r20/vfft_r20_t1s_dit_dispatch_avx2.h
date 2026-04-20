/* vfft_r20_t1s_dit_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=20 / dispatcher=t1s_dit / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R20_T1S_DIT_DISPATCH_AVX2_H
#define VFFT_R20_T1S_DIT_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix20_avx2.h"

static inline void vfft_r20_t1s_dit_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..∞]: ct_t1s_dit
     */
    {
        radix20_t1s_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r20_t1s_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..∞]: ct_t1s_dit
     */
    {
        radix20_t1s_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R20_T1S_DIT_DISPATCH_AVX2_H */
