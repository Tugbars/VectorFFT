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
     *   me∈[64..128]: ct_t1_dit_log_half
     *   me∈[256..256] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[256..256] padded ios: ct_t1_dit_log3
     *   me∈[512..512] pow2 ios: ct_t1_dit_log3
     *   me∈[512..512] padded ios: ct_t1_dit_log3_isub2
     *   me∈[1024..∞]: ct_t1_dit_log3
     */
    if (me <= 128) {
        radix16_t1_dit_log_half_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 256) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512 && me <= 512) {
        if (ios == me) {
            radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_isub2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 1024) {
        radix16_t1_dit_log3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r16_t1_dit_log3_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..64] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[64..64] padded ios: ct_t1_dit_log_half
     *   me∈[128..256] pow2 ios: ct_t1_dit_log_half
     *   me∈[128..256] padded ios: ct_t1_dit_log3_isub2
     *   me∈[512..1024]: ct_t1_dit_log3
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_log3_isub2
     *   me∈[2048..∞] padded ios: ct_t1_dit_log3
     */
    if (me <= 64) {
        if (ios == me) {
            radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log_half_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 128 && me <= 256) {
        if (ios == me) {
            radix16_t1_dit_log_half_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_dit_log3_isub2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512 && me <= 1024) {
        radix16_t1_dit_log3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
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
