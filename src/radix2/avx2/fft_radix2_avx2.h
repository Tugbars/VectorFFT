#ifndef FFT_RADIX2_AVX2_H
#define FFT_RADIX2_AVX2_H
#ifdef __AVX2__
#include <immintrin.h>
#include "fft_radix2_uniform.h"

#define AVX2_VECTOR_WIDTH 4
#define AVX2_ALIGNMENT 32
#ifndef AVX2_PREFETCH_DISTANCE
#define AVX2_PREFETCH_DISTANCE 24
#endif

#define LOAD_RE_AVX2(ptr)          _mm256_loadu_pd(ptr)
#define LOAD_IM_AVX2(ptr)          _mm256_loadu_pd(ptr)
#define STORE_RE_AVX2(ptr, val)    _mm256_storeu_pd(ptr, val)
#define STORE_IM_AVX2(ptr, val)    _mm256_storeu_pd(ptr, val)
#define LOAD_RE_AVX2_ALIGNED(ptr)  _mm256_load_pd(ptr)
#define LOAD_IM_AVX2_ALIGNED(ptr)  _mm256_load_pd(ptr)
#define STORE_RE_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)
#define STORE_IM_AVX2_ALIGNED(ptr, val) _mm256_store_pd(ptr, val)
#define STREAM_RE_AVX2(ptr, val)   _mm256_stream_pd(ptr, val)
#define STREAM_IM_AVX2(ptr, val)   _mm256_stream_pd(ptr, val)

#define PREFETCH_INPUT_T0_AVX2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T0)
#define PREFETCH_TWIDDLE_T1_AVX2(addr, dist) \
    _mm_prefetch((char*)&(addr)[(dist)], _MM_HINT_T1)

#if defined(__FMA__)
static inline __attribute__((always_inline))
void cmul_native_soa_avx2(
    __m256d ar, __m256d ai, __m256d w_re, __m256d w_im,
    __m256d *tr, __m256d *ti)
{
    __m256d t0 = _mm256_mul_pd(ai, w_im);
    *tr = _mm256_fmsub_pd(ar, w_re, t0);
    __m256d t1 = _mm256_mul_pd(ai, w_re);
    *ti = _mm256_fmadd_pd(ar, w_im, t1);
}
#else
static inline __attribute__((always_inline))
void cmul_native_soa_avx2(
    __m256d ar, __m256d ai, __m256d w_re, __m256d w_im,
    __m256d *tr, __m256d *ti)
{
    *tr = _mm256_sub_pd(_mm256_mul_pd(ar, w_re), _mm256_mul_pd(ai, w_im));
    *ti = _mm256_add_pd(_mm256_mul_pd(ar, w_im), _mm256_mul_pd(ai, w_re));
}
#endif

static inline __attribute__((always_inline))
void radix2_butterfly_native_soa_avx2(
    __m256d e_re, __m256d e_im, __m256d o_re, __m256d o_im,
    __m256d w_re, __m256d w_im,
    __m256d *y0_re, __m256d *y0_im, __m256d *y1_re, __m256d *y1_im)
{
    __m256d prod_re, prod_im;
    cmul_native_soa_avx2(o_re, o_im, w_re, w_im, &prod_re, &prod_im);
    *y0_re = _mm256_add_pd(e_re, prod_re); *y0_im = _mm256_add_pd(e_im, prod_im);
    *y1_re = _mm256_sub_pd(e_re, prod_re); *y1_im = _mm256_sub_pd(e_im, prod_im);
}

/* ── 4-butterfly pipelines ── */

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    const __m256d e_re = LOAD_RE_AVX2(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2(&in_im[k + half]);
    const __m256d w_re = _mm256_loadu_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_loadu_pd(&stage_tw->im[k]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STORE_RE_AVX2(&out_re[k], y0_re); STORE_IM_AVX2(&out_im[k], y0_im);
    STORE_RE_AVX2(&out_re[k+half], y1_re); STORE_IM_AVX2(&out_im[k+half], y1_im);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[k + half]);
    const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STORE_RE_AVX2_ALIGNED(&out_re[k], y0_re); STORE_IM_AVX2_ALIGNED(&out_im[k], y0_im);
    STORE_RE_AVX2_ALIGNED(&out_re[k+half], y1_re); STORE_IM_AVX2_ALIGNED(&out_im[k+half], y1_im);
}

static inline __attribute__((always_inline))
void radix2_pipeline_4_avx2_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->re, k + prefetch_dist);
        PREFETCH_TWIDDLE_T1_AVX2(stage_tw->im, k + prefetch_dist);
    }
    const __m256d e_re = LOAD_RE_AVX2_ALIGNED(&in_re[k]);
    const __m256d e_im = LOAD_IM_AVX2_ALIGNED(&in_im[k]);
    const __m256d o_re = LOAD_RE_AVX2_ALIGNED(&in_re[k + half]);
    const __m256d o_im = LOAD_IM_AVX2_ALIGNED(&in_im[k + half]);
    const __m256d w_re = _mm256_load_pd(&stage_tw->re[k]);
    const __m256d w_im = _mm256_load_pd(&stage_tw->im[k]);
    __m256d y0_re, y0_im, y1_re, y1_im;
    radix2_butterfly_native_soa_avx2(e_re, e_im, o_re, o_im, w_re, w_im,
                                     &y0_re, &y0_im, &y1_re, &y1_im);
    STREAM_RE_AVX2(&out_re[k], y0_re); STREAM_IM_AVX2(&out_im[k], y0_im);
    STREAM_RE_AVX2(&out_re[k+half], y1_re); STREAM_IM_AVX2(&out_im[k+half], y1_im);
}

/* ── 8-butterfly (2× unroll) pipelines ── */

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_4_avx2(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_4_avx2(k+4, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2_aligned(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_4_avx2_aligned(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_4_avx2_aligned(k+4, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

static inline __attribute__((always_inline))
void radix2_pipeline_8_avx2_unroll2_stream(
    int k, const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw, int half, int prefetch_dist)
{
    if (prefetch_dist > 0 && k + prefetch_dist < half) {
        PREFETCH_INPUT_T0_AVX2(in_re, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_re, k + half + prefetch_dist);
        PREFETCH_INPUT_T0_AVX2(in_im, k + half + prefetch_dist);
    }
    radix2_pipeline_4_avx2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, 0);
    radix2_pipeline_4_avx2_stream(k+4, in_re, in_im, out_re, out_im, stage_tw, half, 0);
}

#endif /* __AVX2__ */
#endif /* FFT_RADIX2_AVX2_H */
