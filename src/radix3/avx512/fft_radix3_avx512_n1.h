/**
 * @file fft_radix3_avx512_n1.h
 * @brief AVX-512 Radix-3 N1 (Twiddle-less) Stage Kernels
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_AVX512_N1_H
#define FFT_RADIX3_AVX512_N1_H

#include "fft_radix3_avx512.h"

#if defined(__AVX512F__)

#define R3ZN1_PF_DIST  32

/*============================================================================
 * FORWARD N1
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3Z_INLINE void
radix3_stage_n1_forward_avx512(
    size_t K,
    const double *R3Z_RESTRICT in_re,
    const double *R3Z_RESTRICT in_im,
    double       *R3Z_RESTRICT out_re,
    double       *R3Z_RESTRICT out_im)
{
    const double *R3Z_RESTRICT a_re = (const double *)R3Z_ASSUME_ALIGNED(in_re, 64);
    const double *R3Z_RESTRICT a_im = (const double *)R3Z_ASSUME_ALIGNED(in_im, 64);
    const double *R3Z_RESTRICT b_re = in_re + K;
    const double *R3Z_RESTRICT b_im = in_im + K;
    const double *R3Z_RESTRICT c_re = in_re + 2*K;
    const double *R3Z_RESTRICT c_im = in_im + 2*K;

    double *R3Z_RESTRICT o0r = (double *)R3Z_ASSUME_ALIGNED(out_re, 64);
    double *R3Z_RESTRICT o0i = (double *)R3Z_ASSUME_ALIGNED(out_im, 64);
    double *R3Z_RESTRICT o1r = out_re + K;
    double *R3Z_RESTRICT o1i = out_im + K;
    double *R3Z_RESTRICT o2r = out_re + 2*K;
    double *R3Z_RESTRICT o2i = out_im + 2*K;

    const __m512d vhalf = _mm512_set1_pd(R3Z_C_HALF);
    const __m512d vsq3  = _mm512_set1_pd(R3Z_C_SQRT3_2);
    const __m512d vzero = _mm512_setzero_pd();

    const size_t k_vec = K & ~(size_t)7;

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k < k_vec; k += 8) {

        if (k + R3ZN1_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
        }

        __m512d ar = _mm512_loadu_pd(&a_re[k]);
        __m512d ai = _mm512_loadu_pd(&a_im[k]);
        __m512d br = _mm512_loadu_pd(&b_re[k]);
        __m512d bi = _mm512_loadu_pd(&b_im[k]);
        __m512d cr = _mm512_loadu_pd(&c_re[k]);
        __m512d ci = _mm512_loadu_pd(&c_im[k]);

        __m512d sum_r = _mm512_add_pd(br, cr);
        __m512d sum_i = _mm512_add_pd(bi, ci);
        __m512d dif_r = _mm512_sub_pd(br, cr);
        __m512d dif_i = _mm512_sub_pd(bi, ci);

        __m512d rot_r = _mm512_mul_pd(vsq3, dif_i);
        __m512d com_r = _mm512_fmadd_pd(vhalf, sum_r, ar);
        __m512d com_i = _mm512_fmadd_pd(vhalf, sum_i, ai);
        __m512d rot_i = _mm512_fnmadd_pd(vsq3, dif_r, vzero);

        _mm512_storeu_pd(&o0r[k], _mm512_add_pd(ar, sum_r));
        _mm512_storeu_pd(&o0i[k], _mm512_add_pd(ai, sum_i));
        _mm512_storeu_pd(&o1r[k], _mm512_add_pd(com_r, rot_r));
        _mm512_storeu_pd(&o1i[k], _mm512_add_pd(com_i, rot_i));
        _mm512_storeu_pd(&o2r[k], _mm512_sub_pd(com_r, rot_r));
        _mm512_storeu_pd(&o2i[k], _mm512_sub_pd(com_i, rot_i));
    }

    {
        const double half    = R3Z_C_HALF;
        const double sqrt3_2 = R3Z_C_SQRT3_2;
        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];
            double sr = br_+cr_, si = bi_+ci_;
            double dr = br_-cr_, di = bi_-ci_;
            double rr =  sqrt3_2*di, ri = -sqrt3_2*dr;
            double cm_r = ar_ + half*sr, cm_i = ai_ + half*si;
            o0r[k] = ar_+sr;   o0i[k] = ai_+si;
            o1r[k] = cm_r+rr;  o1i[k] = cm_i+ri;
            o2r[k] = cm_r-rr;  o2i[k] = cm_i-ri;
        }
    }
}

/*============================================================================
 * BACKWARD N1
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3Z_INLINE void
radix3_stage_n1_backward_avx512(
    size_t K,
    const double *R3Z_RESTRICT in_re,
    const double *R3Z_RESTRICT in_im,
    double       *R3Z_RESTRICT out_re,
    double       *R3Z_RESTRICT out_im)
{
    const double *R3Z_RESTRICT a_re = (const double *)R3Z_ASSUME_ALIGNED(in_re, 64);
    const double *R3Z_RESTRICT a_im = (const double *)R3Z_ASSUME_ALIGNED(in_im, 64);
    const double *R3Z_RESTRICT b_re = in_re + K;
    const double *R3Z_RESTRICT b_im = in_im + K;
    const double *R3Z_RESTRICT c_re = in_re + 2*K;
    const double *R3Z_RESTRICT c_im = in_im + 2*K;

    double *R3Z_RESTRICT o0r = (double *)R3Z_ASSUME_ALIGNED(out_re, 64);
    double *R3Z_RESTRICT o0i = (double *)R3Z_ASSUME_ALIGNED(out_im, 64);
    double *R3Z_RESTRICT o1r = out_re + K;
    double *R3Z_RESTRICT o1i = out_im + K;
    double *R3Z_RESTRICT o2r = out_re + 2*K;
    double *R3Z_RESTRICT o2i = out_im + 2*K;

    const __m512d vhalf = _mm512_set1_pd(R3Z_C_HALF);
    const __m512d vsq3  = _mm512_set1_pd(R3Z_C_SQRT3_2);
    const __m512d vzero = _mm512_setzero_pd();

    const size_t k_vec = K & ~(size_t)7;

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k < k_vec; k += 8) {

        if (k + R3ZN1_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3ZN1_PF_DIST], _MM_HINT_NTA);
        }

        __m512d ar = _mm512_loadu_pd(&a_re[k]);
        __m512d ai = _mm512_loadu_pd(&a_im[k]);
        __m512d br = _mm512_loadu_pd(&b_re[k]);
        __m512d bi = _mm512_loadu_pd(&b_im[k]);
        __m512d cr = _mm512_loadu_pd(&c_re[k]);
        __m512d ci = _mm512_loadu_pd(&c_im[k]);

        __m512d sum_r = _mm512_add_pd(br, cr);
        __m512d sum_i = _mm512_add_pd(bi, ci);
        __m512d dif_r = _mm512_sub_pd(br, cr);
        __m512d dif_i = _mm512_sub_pd(bi, ci);

        __m512d rot_r = _mm512_fnmadd_pd(vsq3, dif_i, vzero);
        __m512d com_r = _mm512_fmadd_pd(vhalf, sum_r, ar);
        __m512d com_i = _mm512_fmadd_pd(vhalf, sum_i, ai);
        __m512d rot_i = _mm512_mul_pd(vsq3, dif_r);

        _mm512_storeu_pd(&o0r[k], _mm512_add_pd(ar, sum_r));
        _mm512_storeu_pd(&o0i[k], _mm512_add_pd(ai, sum_i));
        _mm512_storeu_pd(&o1r[k], _mm512_add_pd(com_r, rot_r));
        _mm512_storeu_pd(&o1i[k], _mm512_add_pd(com_i, rot_i));
        _mm512_storeu_pd(&o2r[k], _mm512_sub_pd(com_r, rot_r));
        _mm512_storeu_pd(&o2i[k], _mm512_sub_pd(com_i, rot_i));
    }

    {
        const double half    = R3Z_C_HALF;
        const double sqrt3_2 = R3Z_C_SQRT3_2;
        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];
            double sr = br_+cr_, si = bi_+ci_;
            double dr = br_-cr_, di = bi_-ci_;
            double rr = -sqrt3_2*di, ri = sqrt3_2*dr;
            double cm_r = ar_ + half*sr, cm_i = ai_ + half*si;
            o0r[k] = ar_+sr;   o0i[k] = ai_+si;
            o1r[k] = cm_r+rr;  o1i[k] = cm_i+ri;
            o2r[k] = cm_r-rr;  o2i[k] = cm_i-ri;
        }
    }
}

#endif /* __AVX512F__ */
#endif /* FFT_RADIX3_AVX512_N1_H */
