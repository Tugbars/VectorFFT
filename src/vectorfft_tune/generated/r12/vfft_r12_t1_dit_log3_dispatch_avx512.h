/* vfft_r12_t1_dit_log3_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=12 / dispatcher=t1_dit_log3 / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R12_T1_DIT_LOG3_DISPATCH_AVX512_H
#define VFFT_R12_T1_DIT_LOG3_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix12_avx512.h"

static inline void vfft_r12_t1_dit_log3_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..8]: ct_t1_dit_log3
     *   me∈[16..16] pow2 ios: ct_t1_dit_log3_u2b
     *   me∈[16..16] padded ios: ct_t1_dit_log3_u2a
     *   me∈[32..32]: ct_t1_dit_log3_u2b
     *   me∈[64..64] pow2 ios: ct_t1_dit_log3_u2b
     *   me∈[64..64] padded ios: ct_t1_dit_log3_u2a
     *   me∈[128..∞]: ct_t1_dit_log3_u2a
     */
    if (me <= 8) {
        radix12_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 16 && me <= 16) {
        if (ios == me) {
            radix12_t1_dit_log3_u2b_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix12_t1_dit_log3_u2a_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 32 && me <= 32) {
        radix12_t1_dit_log3_u2b_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 64 && me <= 64) {
        if (ios == me) {
            radix12_t1_dit_log3_u2b_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix12_t1_dit_log3_u2a_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 128) {
        radix12_t1_dit_log3_u2a_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r12_t1_dit_log3_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[8..8]: ct_t1_dit_log3
     *   me∈[16..64]: ct_t1_dit_log3_u2b
     *   me∈[128..128]: ct_t1_dit_log3_u2a
     *   me∈[256..∞] pow2 ios: ct_t1_dit_log3
     *   me∈[256..∞] padded ios: ct_t1_dit_log3_u2b
     */
    if (me <= 8) {
        radix12_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 16 && me <= 64) {
        radix12_t1_dit_log3_u2b_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 128 && me <= 128) {
        radix12_t1_dit_log3_u2a_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256) {
        if (ios == me) {
            radix12_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix12_t1_dit_log3_u2b_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

#endif /* VFFT_R12_T1_DIT_LOG3_DISPATCH_AVX512_H */
