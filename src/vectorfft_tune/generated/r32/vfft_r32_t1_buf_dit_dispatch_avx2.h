/* vfft_r32_t1_buf_dit_dispatch_avx2.h
 *
 * Auto-generated codelet dispatcher for R=32 / dispatcher=t1_buf_dit / isa=avx2.
 * Derived from a bench run on this host. The dispatcher picks the
 * fastest variant per (me, ios) based on measured ns/call.
 *
 * To retune: re-run common/bench.py.
 */
#ifndef VFFT_R32_T1_BUF_DIT_DISPATCH_AVX2_H
#define VFFT_R32_T1_BUF_DIT_DISPATCH_AVX2_H

#include <stddef.h>
#include "fft_radix32_avx2.h"

static inline void vfft_r32_t1_buf_dit_dispatch_fwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..191]: ct_t1_buf_dit_tile64_temporal
     *   me∈[192..255]: ct_t1_buf_dit_tile128_temporal
     *   me∈[256..∞]: ct_t1_buf_dit_tile64_temporal
     */
    if (me <= 191) {
        radix32_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 192 && me <= 255) {
        radix32_t1_buf_dit_tile128_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256) {
        radix32_t1_buf_dit_tile64_temporal_fwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

static inline void vfft_r32_t1_buf_dit_dispatch_bwd_avx2(
    double * __restrict__ rio_re, double * __restrict__ rio_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t ios, size_t me)
{
    /* dispatch rules (per bench):
     *   me∈[64..191]: ct_t1_buf_dit_tile64_temporal
     *   me∈[192..255]: ct_t1_buf_dit_tile128_temporal
     *   me∈[256..383]: ct_t1_buf_dit_tile64_temporal
     *   me∈[384..511] pow2 ios: ct_t1_buf_dit_tile64_temporal
     *   me∈[384..511] padded ios: ct_t1_buf_dit_tile128_temporal
     *   me∈[512..∞]: ct_t1_buf_dit_tile64_temporal
     */
    if (me <= 191) {
        radix32_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 192 && me <= 255) {
        radix32_t1_buf_dit_tile128_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 256 && me <= 383) {
        radix32_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
    else if (me >= 384 && me <= 511) {
        if (ios == me) {
            radix32_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        } else {
            radix32_t1_buf_dit_tile128_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
            return;
        }
    }
    else if (me >= 512) {
        radix32_t1_buf_dit_tile64_temporal_bwd_avx2(rio_re, rio_im, W_re, W_im, ios, me);
        return;
    }
}

#endif /* VFFT_R32_T1_BUF_DIT_DISPATCH_AVX2_H */
