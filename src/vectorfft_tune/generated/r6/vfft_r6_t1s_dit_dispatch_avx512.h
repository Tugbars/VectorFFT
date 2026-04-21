/* vfft_r6_t1s_dit_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=6 / dispatcher=t1s_dit / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R6_T1S_DIT_DISPATCH_AVX512_H
#define VFFT_R6_T1S_DIT_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix6_avx512.h"

static inline void vfft_r6_t1s_dit_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..8]: ct_t1s_dit
     *   me∈[16..16] pow2 ios: ct_t1s_dit_u2
     *   me∈[16..16] padded ios: ct_t1s_dit
     *   me∈[32..32] pow2 ios: ct_t1s_dit
     *   me∈[32..32] padded ios: ct_t1s_dit_u2
     *   me∈[64..64] pow2 ios: ct_t1s_dit_u2
     *   me∈[64..64] padded ios: ct_t1s_dit
     *   me∈[128..128]: ct_t1s_dit
     *   me∈[256..∞] pow2 ios: ct_t1s_dit_u2
     *   me∈[256..∞] padded ios: ct_t1s_dit
     */
    if (me <= 8) {
        radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 16 && me <= 16) {
        if (ios == me) {
            radix6_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 32 && me <= 32) {
        if (ios == me) {
            radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 64 && me <= 64) {
        if (ios == me) {
            radix6_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 128 && me <= 128) {
        radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256) {
        if (ios == me) {
            radix6_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r6_t1s_dit_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..8]: ct_t1s_dit
     *   me∈[16..16] pow2 ios: ct_t1s_dit_u2
     *   me∈[16..16] padded ios: ct_t1s_dit
     *   me∈[32..32] pow2 ios: ct_t1s_dit
     *   me∈[32..32] padded ios: ct_t1s_dit_u2
     *   me∈[64..64]: ct_t1s_dit
     *   me∈[128..128] pow2 ios: ct_t1s_dit_u2
     *   me∈[128..128] padded ios: ct_t1s_dit
     *   me∈[256..∞] pow2 ios: ct_t1s_dit
     *   me∈[256..∞] padded ios: ct_t1s_dit_u2
     */
    if (me <= 8) {
        radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 16 && me <= 16) {
        if (ios == me) {
            radix6_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 32 && me <= 32) {
        if (ios == me) {
            radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 64 && me <= 64) {
        radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 128 && me <= 128) {
        if (ios == me) {
            radix6_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 256) {
        if (ios == me) {
            radix6_t1s_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix6_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

#endif /* VFFT_R6_T1S_DIT_DISPATCH_AVX512_H */
