/* vfft_r8_t1_dif_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=8 / dispatcher=t1_dif / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R8_T1_DIF_DISPATCH_AVX2_H
#define VFFT_R8_T1_DIF_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix8_avx2.h"

static inline void vfft_r8_t1_dif_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..256] pow2 ios: ct_t1_dif
     *   me∈[64..256] padded ios: ct_t1_dif_prefetch
     *   me∈[512..1024]: ct_t1_dif
     *   me∈[2048..∞] pow2 ios: ct_t1_dif
     *   me∈[2048..∞] padded ios: ct_t1_dif_prefetch
     */
    if (me <= 256) {
        if (ios == me) {
            radix8_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dif_prefetch_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512 && me <= 1024) {
        radix8_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix8_t1_dif_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dif_prefetch_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r8_t1_dif_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..64]: ct_t1_dif
     *   me∈[128..128] pow2 ios: ct_t1_dif
     *   me∈[128..128] padded ios: ct_t1_dif_prefetch
     *   me∈[256..256]: ct_t1_dif
     *   me∈[512..512]: ct_t1_dif_prefetch
     *   me∈[1024..∞]: ct_t1_dif
     */
    if (me <= 64) {
        radix8_t1_dif_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 128 && me <= 128) {
        if (ios == me) {
            radix8_t1_dif_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dif_prefetch_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 256 && me <= 256) {
        radix8_t1_dif_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 512 && me <= 512) {
        radix8_t1_dif_prefetch_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024) {
        radix8_t1_dif_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R8_T1_DIF_DISPATCH_AVX2_H */
