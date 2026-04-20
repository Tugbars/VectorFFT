/* vfft_r8_t1_dit_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=8 / dispatcher=t1_dit / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R8_T1_DIT_DISPATCH_AVX512_H
#define VFFT_R8_T1_DIT_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix8_avx512.h"

static inline void vfft_r8_t1_dit_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..128]: ct_t1_dit
     *   me∈[256..512]: ct_t1_dit_log1
     *   me∈[1024..1024] pow2 ios: ct_t1_dit_u2
     *   me∈[1024..1024] padded ios: ct_t1_dit
     *   me∈[2048..∞] pow2 ios: ct_t1_dit
     *   me∈[2048..∞] padded ios: ct_t1_dit_log1
     */
    if (me <= 128) {
        radix8_t1_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 512) {
        radix8_t1_dit_log1_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 1024) {
        if (ios == me) {
            radix8_t1_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix8_t1_dit_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_log1_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r8_t1_dit_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..128]: ct_t1_dit
     *   me∈[256..1024]: ct_t1_dit_log1
     *   me∈[2048..∞] pow2 ios: ct_t1_dit_log1
     *   me∈[2048..∞] padded ios: ct_t1_dit_u2
     */
    if (me <= 128) {
        radix8_t1_dit_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 1024) {
        radix8_t1_dit_log1_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix8_t1_dit_log1_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix8_t1_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

#endif /* VFFT_R8_T1_DIT_DISPATCH_AVX512_H */
