/**
 * @file fft_radix4_avx2_n1.h
 * @brief Twiddle-less AVX2 Radix-4 Implementation (FFTW n1-style)
 */

#ifndef FFT_RADIX4_AVX2_N1_H
#define FFT_RADIX4_AVX2_N1_H

#include "fft_radix4.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

#define RADIX4_N1_STREAM_THRESHOLD 8192
#define RADIX4_N1_SMALL_K_THRESHOLD 16
#define RADIX4_N1_PREFETCH_DISTANCE 32

static inline bool is_aligned32_n1(const void *p)
{
    return ((uintptr_t)p & 31u) == 0;
}

#ifdef __AVX2__

#define LOAD_PD_AVX2_N1(ptr) _mm256_loadu_pd(ptr)

FORCE_INLINE void radix4_butterfly_n1_core_fv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im,
    __m256d d_re, __m256d d_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d sign_mask)
{
    __m256d sumBD_re = _mm256_add_pd(b_re, d_re);
    __m256d sumBD_im = _mm256_add_pd(b_im, d_im);
    __m256d difBD_re = _mm256_sub_pd(b_re, d_re);
    __m256d difBD_im = _mm256_sub_pd(b_im, d_im);

    __m256d sumAC_re = _mm256_add_pd(a_re, c_re);
    __m256d sumAC_im = _mm256_add_pd(a_im, c_im);
    __m256d difAC_re = _mm256_sub_pd(a_re, c_re);
    __m256d difAC_im = _mm256_sub_pd(a_im, c_im);

    // rot = (+i) * difBD = (-difBD_im, +difBD_re)
    __m256d rot_re = _mm256_xor_pd(difBD_im, sign_mask);
    __m256d rot_im = difBD_re;

    *y0_re = _mm256_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm256_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

FORCE_INLINE void radix4_butterfly_n1_core_bv_avx2(
    __m256d a_re, __m256d a_im,
    __m256d b_re, __m256d b_im,
    __m256d c_re, __m256d c_im,
    __m256d d_re, __m256d d_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d sign_mask)
{
    __m256d sumBD_re = _mm256_add_pd(b_re, d_re);
    __m256d sumBD_im = _mm256_add_pd(b_im, d_im);
    __m256d difBD_re = _mm256_sub_pd(b_re, d_re);
    __m256d difBD_im = _mm256_sub_pd(b_im, d_im);

    __m256d sumAC_re = _mm256_add_pd(a_re, c_re);
    __m256d sumAC_im = _mm256_add_pd(a_im, c_im);
    __m256d difAC_re = _mm256_sub_pd(a_re, c_re);
    __m256d difAC_im = _mm256_sub_pd(a_im, c_im);

    // rot = (-i) * difBD = (+difBD_im, -difBD_re)
    __m256d rot_re = difBD_im;
    __m256d rot_im = _mm256_xor_pd(difBD_re, sign_mask);

    *y0_re = _mm256_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm256_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm256_sub_pd(difAC_re, rot_re);
    *y1_im = _mm256_sub_pd(difAC_im, rot_im);
    *y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm256_add_pd(difAC_re, rot_re);
    *y3_im = _mm256_add_pd(difAC_im, rot_im);
}

#define PREFETCH_NTA_N1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)

FORCE_INLINE void prefetch_radix4_n1_data(
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    size_t pk)
{
    PREFETCH_NTA_N1(&a_re[pk]);
    PREFETCH_NTA_N1(&a_im[pk]);
    PREFETCH_NTA_N1(&b_re[pk]);
    PREFETCH_NTA_N1(&b_im[pk]);
    PREFETCH_NTA_N1(&c_re[pk]);
    PREFETCH_NTA_N1(&c_im[pk]);
    PREFETCH_NTA_N1(&d_re[pk]);
    PREFETCH_NTA_N1(&d_im[pk]);
}

FORCE_INLINE void radix4_butterfly_n1_scalar_fv(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im)
{
    double a_r = a_re[k], a_i = a_im[k];
    double b_r = b_re[k], b_i = b_im[k];
    double c_r = c_re[k], c_i = c_im[k];
    double d_r = d_re[k], d_i = d_im[k];

    double sumBD_r = b_r + d_r, sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r, difBD_i = b_i - d_i;
    double sumAC_r = a_r + c_r, sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r, difAC_i = a_i - c_i;

    double rot_r = -difBD_i, rot_i = difBD_r;

    y0_re[k] = sumAC_r + sumBD_r; y0_im[k] = sumAC_i + sumBD_i;
    y1_re[k] = difAC_r - rot_r;   y1_im[k] = difAC_i - rot_i;
    y2_re[k] = sumAC_r - sumBD_r; y2_im[k] = sumAC_i - sumBD_i;
    y3_re[k] = difAC_r + rot_r;   y3_im[k] = difAC_i + rot_i;
}

FORCE_INLINE void radix4_butterfly_n1_scalar_bv(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im)
{
    double a_r = a_re[k], a_i = a_im[k];
    double b_r = b_re[k], b_i = b_im[k];
    double c_r = c_re[k], c_i = c_im[k];
    double d_r = d_re[k], d_i = d_im[k];

    double sumBD_r = b_r + d_r, sumBD_i = b_i + d_i;
    double difBD_r = b_r - d_r, difBD_i = b_i - d_i;
    double sumAC_r = a_r + c_r, sumAC_i = a_i + c_i;
    double difAC_r = a_r - c_r, difAC_i = a_i - c_i;

    double rot_r = difBD_i, rot_i = -difBD_r;

    y0_re[k] = sumAC_r + sumBD_r; y0_im[k] = sumAC_i + sumBD_i;
    y1_re[k] = difAC_r - rot_r;   y1_im[k] = difAC_i - rot_i;
    y2_re[k] = sumAC_r - sumBD_r; y2_im[k] = sumAC_i - sumBD_i;
    y3_re[k] = difAC_r + rot_r;   y3_im[k] = difAC_i + rot_i;
}

FORCE_INLINE void radix4_n1_small_k_fv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask)
{
    size_t k = 0;
    for (; k + 4 <= K; k += 4)
    {
        __m256d a_r = LOAD_PD_AVX2_N1(&a_re[k]);
        __m256d a_i = LOAD_PD_AVX2_N1(&a_im[k]);
        __m256d b_r = LOAD_PD_AVX2_N1(&b_re[k]);
        __m256d b_i = LOAD_PD_AVX2_N1(&b_im[k]);
        __m256d c_r = LOAD_PD_AVX2_N1(&c_re[k]);
        __m256d c_i = LOAD_PD_AVX2_N1(&c_im[k]);
        __m256d d_r = LOAD_PD_AVX2_N1(&d_re[k]);
        __m256d d_i = LOAD_PD_AVX2_N1(&d_im[k]);

        __m256d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m256d out_y2_r, out_y2_i, out_y3_r, out_y3_i;

        radix4_butterfly_n1_core_fv_avx2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);

        _mm256_storeu_pd(&y0_re[k], out_y0_r); _mm256_storeu_pd(&y0_im[k], out_y0_i);
        _mm256_storeu_pd(&y1_re[k], out_y1_r); _mm256_storeu_pd(&y1_im[k], out_y1_i);
        _mm256_storeu_pd(&y2_re[k], out_y2_r); _mm256_storeu_pd(&y2_im[k], out_y2_i);
        _mm256_storeu_pd(&y3_re[k], out_y3_r); _mm256_storeu_pd(&y3_im[k], out_y3_i);
    }
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_fv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

FORCE_INLINE void radix4_n1_small_k_bv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask)
{
    size_t k = 0;
    for (; k + 4 <= K; k += 4)
    {
        __m256d a_r = LOAD_PD_AVX2_N1(&a_re[k]);
        __m256d a_i = LOAD_PD_AVX2_N1(&a_im[k]);
        __m256d b_r = LOAD_PD_AVX2_N1(&b_re[k]);
        __m256d b_i = LOAD_PD_AVX2_N1(&b_im[k]);
        __m256d c_r = LOAD_PD_AVX2_N1(&c_re[k]);
        __m256d c_i = LOAD_PD_AVX2_N1(&c_im[k]);
        __m256d d_r = LOAD_PD_AVX2_N1(&d_re[k]);
        __m256d d_i = LOAD_PD_AVX2_N1(&d_im[k]);

        __m256d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
        __m256d out_y2_r, out_y2_i, out_y3_r, out_y3_i;

        radix4_butterfly_n1_core_bv_avx2(a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                                         &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                         &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                         sign_mask);

        _mm256_storeu_pd(&y0_re[k], out_y0_r); _mm256_storeu_pd(&y0_im[k], out_y0_i);
        _mm256_storeu_pd(&y1_re[k], out_y1_r); _mm256_storeu_pd(&y1_im[k], out_y1_i);
        _mm256_storeu_pd(&y2_re[k], out_y2_r); _mm256_storeu_pd(&y2_im[k], out_y2_i);
        _mm256_storeu_pd(&y3_re[k], out_y3_r); _mm256_storeu_pd(&y3_im[k], out_y3_i);
    }
    for (; k < K; k++)
    {
        radix4_butterfly_n1_scalar_bv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

FORCE_INLINE void radix4_n1_stage_u2_pipelined_fv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask,
    bool do_stream)
{
    const size_t K_main = (K / 4) * 4;
    const int prefetch_dist = RADIX4_N1_PREFETCH_DISTANCE;

    if (K_main == 0)
        goto handle_tail;

    {
        __m256d CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im;
        __m256d CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im;
        __m256d OUT_y0_r, OUT_y0_i, OUT_y1_r, OUT_y1_i;
        __m256d OUT_y2_r, OUT_y2_i, OUT_y3_r, OUT_y3_i;

        /* PROLOGUE: load(0), butterfly(0) */
        CUR_a_re = LOAD_PD_AVX2_N1(&a_re[0]); CUR_a_im = LOAD_PD_AVX2_N1(&a_im[0]);
        CUR_b_re = LOAD_PD_AVX2_N1(&b_re[0]); CUR_b_im = LOAD_PD_AVX2_N1(&b_im[0]);
        CUR_c_re = LOAD_PD_AVX2_N1(&c_re[0]); CUR_c_im = LOAD_PD_AVX2_N1(&c_im[0]);
        CUR_d_re = LOAD_PD_AVX2_N1(&d_re[0]); CUR_d_im = LOAD_PD_AVX2_N1(&d_im[0]);

        radix4_butterfly_n1_core_fv_avx2(
            CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im,
            CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im,
            &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
            &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i, sign_mask);

        /* MAIN LOOP: store(k-4), load(k), butterfly(k) */
        for (size_t k = 4; k < K_main; k += 4)
        {
            size_t pk = k + prefetch_dist;
            if (pk < K)
                prefetch_radix4_n1_data(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, pk);

            {
                size_t store_k = k - 4;
                if (do_stream) {
                    _mm256_stream_pd(&y0_re[store_k], OUT_y0_r); _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                    _mm256_stream_pd(&y1_re[store_k], OUT_y1_r); _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                    _mm256_stream_pd(&y2_re[store_k], OUT_y2_r); _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                    _mm256_stream_pd(&y3_re[store_k], OUT_y3_r); _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
                } else {
                    _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r); _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                    _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r); _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                    _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r); _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                    _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r); _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
                }
            }

            CUR_a_re = LOAD_PD_AVX2_N1(&a_re[k]); CUR_a_im = LOAD_PD_AVX2_N1(&a_im[k]);
            CUR_b_re = LOAD_PD_AVX2_N1(&b_re[k]); CUR_b_im = LOAD_PD_AVX2_N1(&b_im[k]);
            CUR_c_re = LOAD_PD_AVX2_N1(&c_re[k]); CUR_c_im = LOAD_PD_AVX2_N1(&c_im[k]);
            CUR_d_re = LOAD_PD_AVX2_N1(&d_re[k]); CUR_d_im = LOAD_PD_AVX2_N1(&d_im[k]);

            radix4_butterfly_n1_core_fv_avx2(
                CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im,
                CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im,
                &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
                &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i, sign_mask);
        }

        /* EPILOGUE: store last butterfly result */
        {
            size_t store_k = K_main - 4;
            if (do_stream) {
                _mm256_stream_pd(&y0_re[store_k], OUT_y0_r); _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_stream_pd(&y1_re[store_k], OUT_y1_r); _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_stream_pd(&y2_re[store_k], OUT_y2_r); _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_stream_pd(&y3_re[store_k], OUT_y3_r); _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
            } else {
                _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r); _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r); _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r); _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r); _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
            }
        }
    }

handle_tail:
    for (size_t k = K_main; k < K; k++)
    {
        radix4_butterfly_n1_scalar_fv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}

FORCE_INLINE void radix4_n1_stage_u2_pipelined_bv_avx2(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    __m256d sign_mask,
    bool do_stream)
{
    const size_t K_main = (K / 4) * 4;
    const int prefetch_dist = RADIX4_N1_PREFETCH_DISTANCE;

    if (K_main == 0)
        goto handle_tail;

    {
        __m256d CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im;
        __m256d CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im;
        __m256d OUT_y0_r, OUT_y0_i, OUT_y1_r, OUT_y1_i;
        __m256d OUT_y2_r, OUT_y2_i, OUT_y3_r, OUT_y3_i;

        /* PROLOGUE: load(0), butterfly(0) */
        CUR_a_re = LOAD_PD_AVX2_N1(&a_re[0]); CUR_a_im = LOAD_PD_AVX2_N1(&a_im[0]);
        CUR_b_re = LOAD_PD_AVX2_N1(&b_re[0]); CUR_b_im = LOAD_PD_AVX2_N1(&b_im[0]);
        CUR_c_re = LOAD_PD_AVX2_N1(&c_re[0]); CUR_c_im = LOAD_PD_AVX2_N1(&c_im[0]);
        CUR_d_re = LOAD_PD_AVX2_N1(&d_re[0]); CUR_d_im = LOAD_PD_AVX2_N1(&d_im[0]);

        radix4_butterfly_n1_core_bv_avx2(
            CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im,
            CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im,
            &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
            &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i, sign_mask);

        /* MAIN LOOP: store(k-4), load(k), butterfly(k) */
        for (size_t k = 4; k < K_main; k += 4)
        {
            size_t pk = k + prefetch_dist;
            if (pk < K)
                prefetch_radix4_n1_data(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, pk);

            {
                size_t store_k = k - 4;
                if (do_stream) {
                    _mm256_stream_pd(&y0_re[store_k], OUT_y0_r); _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                    _mm256_stream_pd(&y1_re[store_k], OUT_y1_r); _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                    _mm256_stream_pd(&y2_re[store_k], OUT_y2_r); _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                    _mm256_stream_pd(&y3_re[store_k], OUT_y3_r); _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
                } else {
                    _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r); _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                    _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r); _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                    _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r); _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                    _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r); _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
                }
            }

            CUR_a_re = LOAD_PD_AVX2_N1(&a_re[k]); CUR_a_im = LOAD_PD_AVX2_N1(&a_im[k]);
            CUR_b_re = LOAD_PD_AVX2_N1(&b_re[k]); CUR_b_im = LOAD_PD_AVX2_N1(&b_im[k]);
            CUR_c_re = LOAD_PD_AVX2_N1(&c_re[k]); CUR_c_im = LOAD_PD_AVX2_N1(&c_im[k]);
            CUR_d_re = LOAD_PD_AVX2_N1(&d_re[k]); CUR_d_im = LOAD_PD_AVX2_N1(&d_im[k]);

            radix4_butterfly_n1_core_bv_avx2(
                CUR_a_re, CUR_a_im, CUR_b_re, CUR_b_im,
                CUR_c_re, CUR_c_im, CUR_d_re, CUR_d_im,
                &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
                &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i, sign_mask);
        }

        /* EPILOGUE: store last butterfly result */
        {
            size_t store_k = K_main - 4;
            if (do_stream) {
                _mm256_stream_pd(&y0_re[store_k], OUT_y0_r); _mm256_stream_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_stream_pd(&y1_re[store_k], OUT_y1_r); _mm256_stream_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_stream_pd(&y2_re[store_k], OUT_y2_r); _mm256_stream_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_stream_pd(&y3_re[store_k], OUT_y3_r); _mm256_stream_pd(&y3_im[store_k], OUT_y3_i);
            } else {
                _mm256_storeu_pd(&y0_re[store_k], OUT_y0_r); _mm256_storeu_pd(&y0_im[store_k], OUT_y0_i);
                _mm256_storeu_pd(&y1_re[store_k], OUT_y1_r); _mm256_storeu_pd(&y1_im[store_k], OUT_y1_i);
                _mm256_storeu_pd(&y2_re[store_k], OUT_y2_r); _mm256_storeu_pd(&y2_im[store_k], OUT_y2_i);
                _mm256_storeu_pd(&y3_re[store_k], OUT_y3_r); _mm256_storeu_pd(&y3_im[store_k], OUT_y3_i);
            }
        }
    }

handle_tail:
    for (size_t k = K_main; k < K; k++)
    {
        radix4_butterfly_n1_scalar_bv(k, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                      y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im);
    }
}


FORCE_INLINE void radix4_n1_stage_fv_avx2(
    size_t N, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    bool is_write_only, bool is_cold_out)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 32);

    const double *a_re = in_re_aligned;
    const double *b_re = in_re_aligned + K;
    const double *c_re = in_re_aligned + 2 * K;
    const double *d_re = in_re_aligned + 3 * K;
    const double *a_im = in_im_aligned;
    const double *b_im = in_im_aligned + K;
    const double *c_im = in_im_aligned + 2 * K;
    const double *d_im = in_im_aligned + 3 * K;

    double *y0_re = out_re_aligned, *y1_re = out_re_aligned + K;
    double *y2_re = out_re_aligned + 2*K, *y3_re = out_re_aligned + 3*K;
    double *y0_im = out_im_aligned, *y1_im = out_im_aligned + K;
    double *y2_im = out_im_aligned + 2*K, *y3_im = out_im_aligned + 3*K;

    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const bool do_stream = (N >= RADIX4_N1_STREAM_THRESHOLD) && is_write_only && is_cold_out &&
        is_aligned32_n1(y0_re) && is_aligned32_n1(y0_im) && is_aligned32_n1(y1_re) && is_aligned32_n1(y1_im) &&
        is_aligned32_n1(y2_re) && is_aligned32_n1(y2_im) && is_aligned32_n1(y3_re) && is_aligned32_n1(y3_im);

    if (K < RADIX4_N1_SMALL_K_THRESHOLD)
        radix4_n1_small_k_fv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask);
    else
        radix4_n1_stage_u2_pipelined_fv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                             y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask, do_stream);
    if (do_stream) _mm_sfence();
}

FORCE_INLINE void radix4_n1_stage_bv_avx2(
    size_t N, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    bool is_write_only, bool is_cold_out)
{
    const double *in_re_aligned = (const double *)ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = (const double *)ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = (double *)ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = (double *)ASSUME_ALIGNED(out_im, 32);

    const double *a_re = in_re_aligned;
    const double *b_re = in_re_aligned + K;
    const double *c_re = in_re_aligned + 2 * K;
    const double *d_re = in_re_aligned + 3 * K;
    const double *a_im = in_im_aligned;
    const double *b_im = in_im_aligned + K;
    const double *c_im = in_im_aligned + 2 * K;
    const double *d_im = in_im_aligned + 3 * K;

    double *y0_re = out_re_aligned, *y1_re = out_re_aligned + K;
    double *y2_re = out_re_aligned + 2*K, *y3_re = out_re_aligned + 3*K;
    double *y0_im = out_im_aligned, *y1_im = out_im_aligned + K;
    double *y2_im = out_im_aligned + 2*K, *y3_im = out_im_aligned + 3*K;

    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const bool do_stream = (N >= RADIX4_N1_STREAM_THRESHOLD) && is_write_only && is_cold_out &&
        is_aligned32_n1(y0_re) && is_aligned32_n1(y0_im) && is_aligned32_n1(y1_re) && is_aligned32_n1(y1_im) &&
        is_aligned32_n1(y2_re) && is_aligned32_n1(y2_im) && is_aligned32_n1(y3_re) && is_aligned32_n1(y3_im);

    if (K < RADIX4_N1_SMALL_K_THRESHOLD)
        radix4_n1_small_k_bv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask);
    else
        radix4_n1_stage_u2_pipelined_bv_avx2(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                             y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, sign_mask, do_stream);
    if (do_stream) _mm_sfence();
}

FORCE_INLINE void fft_radix4_n1_forward_stage_avx2(
    size_t N, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    radix4_n1_stage_fv_avx2(N, K, in_re, in_im, out_re, out_im, true, (N >= 4096));
}

FORCE_INLINE void fft_radix4_n1_backward_stage_avx2(
    size_t N, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    radix4_n1_stage_bv_avx2(N, K, in_re, in_im, out_re, out_im, true, (N >= 4096));
}

#endif // __AVX2__

#endif // FFT_RADIX4_AVX2_N1_H
