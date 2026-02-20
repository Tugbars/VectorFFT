/**
 * @file fft_radix2_sse2n1.h
 * @brief SSE2 Twiddle-Less Radix-2 FFT Butterflies (W=1 Optimized)
 * @author Tugbars
 * @version 3.0
 * @date 2025
 */

#ifndef FFT_RADIX2_SSE2N1_H
#define FFT_RADIX2_SSE2N1_H

#include <emmintrin.h>
#include "fft_radix2_uniform.h"

static inline __attribute__((always_inline))
void radix2_butterfly_n1_sse2(
    __m128d e_re, __m128d e_im, __m128d o_re, __m128d o_im,
    __m128d *y0_re, __m128d *y0_im, __m128d *y1_re, __m128d *y1_im)
{
    *y0_re = _mm_add_pd(e_re, o_re); *y0_im = _mm_add_pd(e_im, o_im);
    *y1_re = _mm_sub_pd(e_re, o_re); *y1_im = _mm_sub_pd(e_im, o_im);
}

/* ── 2-bf N1: unaligned ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    const __m128d e_re = _mm_loadu_pd(&in_re[k]);
    const __m128d e_im = _mm_loadu_pd(&in_im[k]);
    const __m128d o_re = _mm_loadu_pd(&in_re[k + half]);
    const __m128d o_im = _mm_loadu_pd(&in_im[k + half]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm_storeu_pd(&out_re[k], y0_re); _mm_storeu_pd(&out_im[k], y0_im);
    _mm_storeu_pd(&out_re[k+half], y1_re); _mm_storeu_pd(&out_im[k+half], y1_im);
}

/* ── 2-bf N1: aligned ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    const __m128d e_re = _mm_load_pd(&in_re[k]);
    const __m128d e_im = _mm_load_pd(&in_im[k]);
    const __m128d o_re = _mm_load_pd(&in_re[k + half]);
    const __m128d o_im = _mm_load_pd(&in_im[k + half]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm_store_pd(&out_re[k], y0_re); _mm_store_pd(&out_im[k], y0_im);
    _mm_store_pd(&out_re[k+half], y1_re); _mm_store_pd(&out_im[k+half], y1_im);
}

/* ── 2-bf N1: streaming ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_n1_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    const __m128d e_re = _mm_load_pd(&in_re[k]);
    const __m128d e_im = _mm_load_pd(&in_im[k]);
    const __m128d o_re = _mm_load_pd(&in_re[k + half]);
    const __m128d o_im = _mm_load_pd(&in_im[k + half]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_sse2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm_stream_pd(&out_re[k], y0_re); _mm_stream_pd(&out_im[k], y0_im);
    _mm_stream_pd(&out_re[k+half], y1_re); _mm_stream_pd(&out_im[k+half], y1_im);
}

/* ── 4-bf N1 (2× unroll) ── */

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    radix2_pipeline_2_sse2_n1(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_2_sse2_n1(k+2, in_re, in_im, out_re, out_im, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    radix2_pipeline_2_sse2_n1_aligned(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_2_sse2_n1_aligned(k+2, in_re, in_im, out_re, out_im, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_n1_unroll2_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        _mm_prefetch((char*)&in_re[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_re[k + half + prefetch_dist], _MM_HINT_T0);
        _mm_prefetch((char*)&in_im[k + half + prefetch_dist], _MM_HINT_T0);
    }
    radix2_pipeline_2_sse2_n1_stream(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_2_sse2_n1_stream(k+2, in_re, in_im, out_re, out_im, half, 0);
}

#endif /* FFT_RADIX2_SSE2N1_H */
