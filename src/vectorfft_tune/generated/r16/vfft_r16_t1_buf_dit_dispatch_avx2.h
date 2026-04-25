/* vfft_r16_t1_buf_dit_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=16 / dispatcher=t1_buf_dit / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R16_T1_BUF_DIT_DISPATCH_AVX2_H
#define VFFT_R16_T1_BUF_DIT_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix16_avx2.h"

static inline void vfft_r16_t1_buf_dit_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..1023]: ct_t1_buf_dit_tile64_temporal
     *   me∈[1024..1535] pow2 ios: ct_t1_buf_dit_tile128_temporal
     *   me∈[1024..1535] padded ios: ct_t1_buf_dit_tile64_temporal
     *   me∈[1536..2047]: ct_t1_buf_dit_tile64_temporal
     *   me∈[2048..∞] pow2 ios: ct_t1_buf_dit_tile128_temporal
     *   me∈[2048..∞] padded ios: ct_t1_buf_dit_tile64_temporal
     */
    if (me <= 1023) {
        radix16_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 1535) {
        if (ios == me) {
            radix16_t1_buf_dit_tile128_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 1536 && me <= 2047) {
        radix16_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 2048) {
        if (ios == me) {
            radix16_t1_buf_dit_tile128_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
}

static inline void vfft_r16_t1_buf_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..191]: ct_t1_buf_dit_tile64_temporal
     *   me∈[192..255]: ct_t1_buf_dit_tile128_temporal
     *   me∈[256..1023]: ct_t1_buf_dit_tile64_temporal
     *   me∈[1024..2047] pow2 ios: ct_t1_buf_dit_tile128_temporal
     *   me∈[1024..2047] padded ios: ct_t1_buf_dit_tile64_temporal
     *   me∈[2048..∞]: ct_t1_buf_dit_tile64_temporal
     */
    if (me <= 191) {
        radix16_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 192 && me <= 255) {
        radix16_t1_buf_dit_tile128_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 1023) {
        radix16_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 1024 && me <= 2047) {
        if (ios == me) {
            radix16_t1_buf_dit_tile128_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix16_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 2048) {
        radix16_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R16_T1_BUF_DIT_DISPATCH_AVX2_H */
