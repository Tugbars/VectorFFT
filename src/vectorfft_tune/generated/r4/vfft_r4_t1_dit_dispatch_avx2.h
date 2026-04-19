/* vfft_r4_t1_dit_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=4 / dispatcher=t1_dit / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R4_T1_DIT_DISPATCH_AVX2_H
#define VFFT_R4_T1_DIT_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix4_avx2.h"

static inline void vfft_r4_t1_dit_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..∞]: ct_t1_dit_log1
     */
    {
        radix4_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r4_t1_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..64] pow2 ios: ct_t1_dit
     *   me∈[64..64] padded ios: ct_t1_dit_log1
     *   me∈[128..512]: ct_t1_dit_log1
     *   me∈[1024..1024] pow2 ios: ct_t1_dit_log1
     *   me∈[1024..1024] padded ios: ct_t1_dit
     *   me∈[2048..∞]: ct_t1_dit_log1
     */
    if (me <= 64) {
        if (ios == me) {
            radix4_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 128 && me <= 512) {
        radix4_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 1024) {
        if (ios == me) {
            radix4_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 2048) {
        radix4_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R4_T1_DIT_DISPATCH_AVX2_H */
