/**
 * @file fft_radix2_avx2n1.h
 * @brief AVX2 Twiddle-Less (N1) Radix-2 FFT Butterflies
 *
 * Naming convention: "_n1" suffix matches scalar (scalarn1) and orchestrator (.c)
 * conventions. The original "_twiddleless" naming is preserved as aliases below.
 */
#ifndef FFT_RADIX2_AVX2N1_H
#define FFT_RADIX2_AVX2N1_H
#ifdef __AVX2__

#include <immintrin.h>
#include "fft_radix2_uniform.h"

/* ── Twiddle-less butterfly core ── */

static inline __attribute__((always_inline))
void radix2_butterfly_n1_avx2(
    __m256d e_re, __m256d e_im, __m256d o_re, __m256d o_im,
    __m256d *y0_re, __m256d *y0_im, __m256d *y1_re, __m256d *y1_im)
{
    *y0_re = _mm256_add_pd(e_re, o_re);
    *y0_im = _mm256_add_pd(e_im, o_im);
    *y1_re = _mm256_sub_pd(e_re, o_re);
    *y1_im = _mm256_sub_pd(e_im, o_im);
}

/* ── 4-butterfly N1 pipelines ── */

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_n1(
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
    const __m256d e_re = _mm256_loadu_pd(&in_re[k]);
    const __m256d e_im = _mm256_loadu_pd(&in_im[k]);
    const __m256d o_re = _mm256_loadu_pd(&in_re[k + half]);
    const __m256d o_im = _mm256_loadu_pd(&in_im[k + half]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm256_storeu_pd(&out_re[k], y0_re);
    _mm256_storeu_pd(&out_im[k], y0_im);
    _mm256_storeu_pd(&out_re[k + half], y1_re);
    _mm256_storeu_pd(&out_im[k + half], y1_im);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_n1_aligned(
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
    const __m256d e_re = _mm256_load_pd(&in_re[k]);
    const __m256d e_im = _mm256_load_pd(&in_im[k]);
    const __m256d o_re = _mm256_load_pd(&in_re[k + half]);
    const __m256d o_im = _mm256_load_pd(&in_im[k + half]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm256_store_pd(&out_re[k], y0_re);
    _mm256_store_pd(&out_im[k], y0_im);
    _mm256_store_pd(&out_re[k + half], y1_re);
    _mm256_store_pd(&out_im[k + half], y1_im);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_n1_stream(
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
    const __m256d e_re = _mm256_load_pd(&in_re[k]);
    const __m256d e_im = _mm256_load_pd(&in_im[k]);
    const __m256d o_re = _mm256_load_pd(&in_re[k + half]);
    const __m256d o_im = _mm256_load_pd(&in_im[k + half]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_n1_avx2(e_re, e_im, o_re, o_im,
                             &y0_re, &y0_im, &y1_re, &y1_im);
    _mm256_stream_pd(&out_re[k], y0_re);
    _mm256_stream_pd(&out_im[k], y0_im);
    _mm256_stream_pd(&out_re[k + half], y1_re);
    _mm256_stream_pd(&out_im[k + half], y1_im);
}

/* ── 8-butterfly N1 (2× unroll) pipelines ── */

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_n1_unroll2(
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
    radix2_pipeline_4_avx2_n1(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_4_avx2_n1(k + 4, in_re, in_im, out_re, out_im, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_n1_unroll2_aligned(
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
    radix2_pipeline_4_avx2_n1_aligned(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_4_avx2_n1_aligned(k + 4, in_re, in_im, out_re, out_im, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_n1_unroll2_stream(
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
    radix2_pipeline_4_avx2_n1_stream(k, in_re, in_im, out_re, out_im, half, 0);
    radix2_pipeline_4_avx2_n1_stream(k + 4, in_re, in_im, out_re, out_im, half, 0);
}

/* ── Backward-compatible aliases for _twiddleless naming ── */
#define radix2_butterfly_twiddleless_avx2         radix2_butterfly_n1_avx2
#define radix2_pipeline_4_avx2_twiddleless         radix2_pipeline_4_avx2_n1
#define radix2_pipeline_4_avx2_twiddleless_aligned radix2_pipeline_4_avx2_n1_aligned
#define radix2_pipeline_4_avx2_twiddleless_stream  radix2_pipeline_4_avx2_n1_stream
#define radix2_pipeline_8_avx2_twiddleless_unroll2         radix2_pipeline_8_avx2_n1_unroll2
#define radix2_pipeline_8_avx2_twiddleless_unroll2_aligned radix2_pipeline_8_avx2_n1_unroll2_aligned
#define radix2_pipeline_8_avx2_twiddleless_unroll2_stream  radix2_pipeline_8_avx2_n1_unroll2_stream

#endif /* __AVX2__ */
#endif /* FFT_RADIX2_AVX2N1_H */
