/* vfft_r3_t1s_dit_dispatch_avx512.h
 *
 * Auto-generated codelet dispatcher for R=3 / dispatcher=t1s_dit / isa=avx512.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R3_T1S_DIT_DISPATCH_AVX512_H
#define VFFT_R3_T1S_DIT_DISPATCH_AVX512_H

#include <stddef.h>
#include "fft_radix3_avx512.h"

static inline void vfft_r3_t1s_dit_dispatch_fwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[24..24] pow2 ios: ct_t1s_dit_u3
     *   me∈[24..24] padded ios: ct_t1s_dit_u1
     *   me∈[48..48]: ct_t1s_dit_u2
     *   me∈[96..192]: ct_t1s_dit_u3
     *   me∈[384..384] pow2 ios: ct_t1s_dit_u3
     *   me∈[384..384] padded ios: ct_t1s_dit_u2
     *   me∈[768..768] pow2 ios: ct_t1s_dit_u2
     *   me∈[768..768] padded ios: ct_t1s_dit_u3
     *   me∈[1536..1536] pow2 ios: ct_t1s_dit_u3
     *   me∈[1536..1536] padded ios: ct_t1s_dit_u2
     *   me∈[3072..∞]: ct_t1s_dit_u1
     */
    if (me <= 24) {
        if (ios == me) {
            radix3_t1s_dit_u3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u1_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 48 && me <= 48) {
        radix3_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 96 && me <= 192) {
        radix3_t1s_dit_u3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 384 && me <= 384) {
        if (ios == me) {
            radix3_t1s_dit_u3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 768 && me <= 768) {
        if (ios == me) {
            radix3_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 1536 && me <= 1536) {
        if (ios == me) {
            radix3_t1s_dit_u3_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u2_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 3072) {
        radix3_t1s_dit_u1_fwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r3_t1s_dit_dispatch_bwd_avx512(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[24..24] pow2 ios: ct_t1s_dit_u1
     *   me∈[24..24] padded ios: ct_t1s_dit_u3
     *   me∈[48..48] pow2 ios: ct_t1s_dit_u2
     *   me∈[48..48] padded ios: ct_t1s_dit_u3
     *   me∈[96..96]: ct_t1s_dit_u2
     *   me∈[192..192]: ct_t1s_dit_u3
     *   me∈[384..384] pow2 ios: ct_t1s_dit_u3
     *   me∈[384..384] padded ios: ct_t1s_dit_u2
     *   me∈[768..768]: ct_t1s_dit_u3
     *   me∈[1536..∞]: ct_t1s_dit_u1
     */
    if (me <= 24) {
        if (ios == me) {
            radix3_t1s_dit_u1_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 48 && me <= 48) {
        if (ios == me) {
            radix3_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 96 && me <= 96) {
        radix3_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 192 && me <= 192) {
        radix3_t1s_dit_u3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 384 && me <= 384) {
        if (ios == me) {
            radix3_t1s_dit_u3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix3_t1s_dit_u2_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 768 && me <= 768) {
        radix3_t1s_dit_u3_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1536) {
        radix3_t1s_dit_u1_bwd_avx512(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R3_T1S_DIT_DISPATCH_AVX512_H */
