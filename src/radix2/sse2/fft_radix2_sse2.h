/**
 * @file fft_radix2_sse2.h
 * @brief SSE2 Optimized Radix-2 FFT Butterflies - TRUE SoA (ZERO SHUFFLE!)
 *
 * FIX: radix2_pipeline_2_sse2 twiddle loads changed from _mm_load_pd (aligned)
 *      to _mm_loadu_pd (unaligned). k may be odd after special-case indices.
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture)
 * @date 2025
 */

#ifndef FFT_RADIX2_SSE2_H
#define FFT_RADIX2_SSE2_H

#include <emmintrin.h>
#include "fft_radix2_uniform.h"

#define SSE2_VECTOR_WIDTH 2
#define SSE2_ALIGNMENT 16

#ifndef SSE2_PREFETCH_DISTANCE
#define SSE2_PREFETCH_DISTANCE 16
#endif

/* ── load/store primitives ── */

#define LOAD_RE_SSE2(ptr)          _mm_loadu_pd(ptr)
#define LOAD_IM_SSE2(ptr)          _mm_loadu_pd(ptr)
#define STORE_RE_SSE2(ptr, val)    _mm_storeu_pd(ptr, val)
#define STORE_IM_SSE2(ptr, val)    _mm_storeu_pd(ptr, val)

#define LOAD_RE_SSE2_ALIGNED(ptr)  _mm_load_pd(ptr)
#define LOAD_IM_SSE2_ALIGNED(ptr)  _mm_load_pd(ptr)
#define STORE_RE_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)
#define STORE_IM_SSE2_ALIGNED(ptr, val) _mm_store_pd(ptr, val)

#define STREAM_RE_SSE2(ptr, val)   _mm_stream_pd(ptr, val)
#define STREAM_IM_SSE2(ptr, val)   _mm_stream_pd(ptr, val)

/* ── prefetch ── */

#define PREFETCH_INPUT_T0_SSE2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)
#define PREFETCH_TWIDDLE_T1_SSE2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T1)

/* ── complex multiply (no FMA on SSE2) ── */

static inline __attribute__((always_inline))
void cmul_native_soa_sse2(
    __m128d ar, __m128d ai, __m128d w_re, __m128d w_im,
    __m128d *tr, __m128d *ti)
{
    *tr = _mm_sub_pd(_mm_mul_pd(ar, w_re), _mm_mul_pd(ai, w_im));
    *ti = _mm_add_pd(_mm_mul_pd(ar, w_im), _mm_mul_pd(ai, w_re));
}

/* ── butterfly ── */

static inline __attribute__((always_inline))
void radix2_butterfly_native_soa_sse2(
    __m128d e_re, __m128d e_im, __m128d o_re, __m128d o_im,
    __m128d w_re, __m128d w_im,
    __m128d *y0_re, __m128d *y0_im, __m128d *y1_re, __m128d *y1_im)
{
    __m128d prod_re, prod_im;
    cmul_native_soa_sse2(o_re, o_im, w_re, w_im, &prod_re, &prod_im);
    *y0_re = _mm_add_pd(e_re, prod_re); *y0_im = _mm_add_pd(e_im, prod_im);
    *y1_re = _mm_sub_pd(e_re, prod_re); *y1_im = _mm_sub_pd(e_im, prod_im);
}

/* ── 2-bf: unaligned (FIX: twiddle loads also unaligned) ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    const __m128d e_re = LOAD_RE_SSE2(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2(&in_im[k + half]);
    /* FIX: _mm_loadu_pd — k may be odd after special-case indices */
    const __m128d w_re = _mm_loadu_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_loadu_pd(&stage_tw->im[k]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STORE_RE_SSE2(&out_re[k], y0_re); STORE_IM_SSE2(&out_im[k], y0_im);
    STORE_RE_SSE2(&out_re[k+half], y1_re); STORE_IM_SSE2(&out_im[k+half], y1_im);
}

/* ── 2-bf: aligned ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[k + half]);
    const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STORE_RE_SSE2_ALIGNED(&out_re[k], y0_re); STORE_IM_SSE2_ALIGNED(&out_im[k], y0_im);
    STORE_RE_SSE2_ALIGNED(&out_re[k+half], y1_re); STORE_IM_SSE2_ALIGNED(&out_im[k+half], y1_im);
}

/* ── 2-bf: streaming ── */

static inline __attribute__((always_inline))
void radix2_pipeline_2_sse2_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_SSE2(stage_tw->im, k + prefetch_dist);
    }
    const __m128d e_re = LOAD_RE_SSE2_ALIGNED(&in_re[k]);
    const __m128d e_im = LOAD_IM_SSE2_ALIGNED(&in_im[k]);
    const __m128d o_re = LOAD_RE_SSE2_ALIGNED(&in_re[k + half]);
    const __m128d o_im = LOAD_IM_SSE2_ALIGNED(&in_im[k + half]);
    const __m128d w_re = _mm_load_pd(&stage_tw->re[k]);
    const __m128d w_im = _mm_load_pd(&stage_tw->im[k]);
    __m128d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_sse2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STREAM_RE_SSE2(&out_re[k], y0_re); STREAM_IM_SSE2(&out_im[k], y0_im);
    STREAM_RE_SSE2(&out_re[k+half], y1_re); STREAM_IM_SSE2(&out_im[k+half], y1_im);
}

/* ── 4-bf (2× unroll) ── */

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_2_sse2(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_2_sse2(k+2, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_2_sse2_aligned(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_2_sse2_aligned(k+2, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_sse2_unroll2_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_SSE2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_SSE2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_2_sse2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_2_sse2_stream(k+2, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

#endif /* FFT_RADIX2_SSE2_H */
