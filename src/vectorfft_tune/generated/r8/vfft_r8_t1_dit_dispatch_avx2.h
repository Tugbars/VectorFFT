/* vfft_r8_t1_dit_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=8 / dispatcher=t1_dit / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R8_T1_DIT_DISPATCH_AVX2_H
#define VFFT_R8_T1_DIT_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix8_avx2.h"

static inline void vfft_r8_t1_dit_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..255]: ct_t1_dit
     *   me∈[256..383] pow2 ios: ct_t1_dit_log1
     *   me∈[256..383] padded ios: ct_t1_dit
     *   me∈[384..511] pow2 ios: ct_t1_dit
     *   me∈[384..511] padded ios: ct_t1_dit_log1
     *   me∈[512..767]: ct_t1_dit_prefetch
     *   me∈[768..1535]: ct_t1_dit
     *   me∈[1536..2047]: ct_t1_dit_log1
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_prefetch
     *   me∈[2048..∞] padded ios: ct_t1_dit_log1
     */
    if (me <= 255) {
        radix8_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 383) {
        if (ios == me) {
            radix8_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 384 && me <= 511) {
        if (ios == me) {
            radix8_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512 && me <= 767) {
        radix8_t1_dit_prefetch_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 768 && me <= 1535) {
        radix8_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1536 && me <= 2047) {
        radix8_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix8_t1_dit_prefetch_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r8_t1_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..255]: ct_t1_dit
     *   me∈[256..383]: ct_t1_dit_log1
     *   me∈[384..511]: ct_t1_dit
     *   me∈[512..767] pow2 ios: ct_t1_dit_prefetch
     *   me∈[512..767] padded ios: ct_t1_dit
     *   me∈[768..1023] pow2 ios: ct_t1_dit_log1
     *   me∈[768..1023] padded ios: ct_t1_dit
     *   me∈[1024..2047]: ct_t1_dit
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_prefetch
     *   me∈[2048..∞] padded ios: ct_t1_dit_log1
     */
    if (me <= 255) {
        radix8_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 383) {
        radix8_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 384 && me <= 511) {
        radix8_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 512 && me <= 767) {
        if (ios == me) {
            radix8_t1_dit_prefetch_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 768 && me <= 1023) {
        if (ios == me) {
            radix8_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 1024 && me <= 2047) {
        radix8_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix8_t1_dit_prefetch_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

#endif /* VFFT_R8_T1_DIT_DISPATCH_AVX2_H */
