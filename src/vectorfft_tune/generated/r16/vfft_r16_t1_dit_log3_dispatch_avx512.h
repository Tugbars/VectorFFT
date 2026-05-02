/* vfft_r16_t1_dit_log3_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=16 / dispatcher=t1_dit_log3 / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R16_T1_DIT_LOG3_DISPATCH_AVX512_H
#define VFFT_R16_T1_DIT_LOG3_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix16_avx512.h"

static inline void vfft_r16_t1_dit_log3_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..95]: ct_t1_dit_log3_isub2
     *   me∈[96..191] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[96..191] padded ios: ct_t1_dit_log3
     *   me∈[192..383] pow2 ios: ct_t1_dit_log3
     *   me∈[192..383] padded ios: ct_t1_dit_log3_isub2
     *   me∈[384..511] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[384..511] padded ios: ct_t1_dit_log3
     *   me∈[512..767] pow2 ios: ct_t1_dit_log3
     *   me∈[512..767] padded ios: ct_t1_dit_log3_isub2
     *   me∈[768..1023]: ct_t1_dit_log3
     *   me∈[1024..2047]: ct_t1_dit_log3_isub2
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[2048..∞] padded ios: ct_t1_dit_log3
     */
    if (me <= 95) {
        radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 96 && me <= 191) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 192 && me <= 383) {
        if (ios == me) {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 384 && me <= 511) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512 && me <= 767) {
        if (ios == me) {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 768 && me <= 1023) {
        radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 2047) {
        radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r16_t1_dit_log3_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..127]: ct_t1_dit_log3_isub2
     *   me∈[128..255] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[128..255] padded ios: ct_t1_dit_log3
     *   me∈[256..383]: ct_t1_dit_log3
     *   me∈[384..767]: ct_t1_dit_log3_isub2
     *   me∈[768..1023]: ct_t1_dit_log3
     *   me∈[1024..1535]: ct_t1_dit_log3_isub2
     *   me∈[1536..∞] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[1536..∞] padded ios: ct_t1_dit_log3
     */
    if (me <= 127) {
        radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 128 && me <= 255) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 256 && me <= 383) {
        radix16_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 384 && me <= 767) {
        radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 768 && me <= 1023) {
        radix16_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 1535) {
        radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1536) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

#endif /* VFFT_R16_T1_DIT_LOG3_DISPATCH_AVX512_H */
