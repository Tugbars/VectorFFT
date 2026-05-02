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
     *   me∈[64..127]: ct_t1_dit
     *   me∈[128..191] pow2 ios: ct_t1_dit_u2
     *   me∈[128..191] padded ios: ct_t1_dit
     *   me∈[192..255]: ct_t1_dit
     *   me∈[256..383] pow2 ios: ct_t1_dit_u2
     *   me∈[256..383] padded ios: ct_t1_dit
     *   me∈[384..511]: ct_t1_dit
     *   me∈[512..1023]: ct_t1_dit_log1
     *   me∈[1024..1535] pow2 ios: ct_t1_dit
     *   me∈[1024..1535] padded ios: ct_t1_dit_log1
     *   me∈[1536..2047]: ct_t1_dit_log1
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_log1
     *   me∈[2048..∞] padded ios: ct_t1_dit
     */
    if (me <= 127) {
        radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 128 && me <= 191) {
        if (ios == me) {
            radix4_t1_dit_u2_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 192 && me <= 255) {
        radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 383) {
        if (ios == me) {
            radix4_t1_dit_u2_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 384 && me <= 511) {
        radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 512 && me <= 1023) {
        radix4_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 1535) {
        if (ios == me) {
            radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 1536 && me <= 2047) {
        radix4_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix4_t1_dit_log1_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix4_t1_dit_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r4_t1_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..511]: ct_t1_dit
     *   me∈[512..∞]: ct_t1_dit_log1
     */
    if (me <= 511) {
        radix4_t1_dit_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 512) {
        radix4_t1_dit_log1_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R4_T1_DIT_DISPATCH_AVX2_H */
