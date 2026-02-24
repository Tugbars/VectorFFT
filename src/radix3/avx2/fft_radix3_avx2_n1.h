/**
 * @file fft_radix3_avx2_n1.h
 * @brief AVX2 Radix-3 N1 (Twiddle-less) Stage Kernels
 *
 * For stages where all twiddle factors are unity.
 * No complex multiply — pure butterfly, ~40% fewer FLOPs than twiddled.
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_AVX2_N1_H
#define FFT_RADIX3_AVX2_N1_H

#include "fft_radix3_avx2.h"    /* picks up types, macros, constants */

#if defined(__AVX2__) && defined(__FMA__)

#define R3A2N1_PF_DIST  16

/*============================================================================
 * FORWARD N1
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3A2_INLINE void
radix3_stage_n1_forward_avx2(
    size_t K,
    const double *R3A2_RESTRICT in_re,
    const double *R3A2_RESTRICT in_im,
    double       *R3A2_RESTRICT out_re,
    double       *R3A2_RESTRICT out_im)
{
    const double *R3A2_RESTRICT a_re = in_re;
    const double *R3A2_RESTRICT a_im = in_im;
    const double *R3A2_RESTRICT b_re = in_re + K;
    const double *R3A2_RESTRICT b_im = in_im + K;
    const double *R3A2_RESTRICT c_re = in_re + 2*K;
    const double *R3A2_RESTRICT c_im = in_im + 2*K;

    double *R3A2_RESTRICT o0r = out_re;
    double *R3A2_RESTRICT o0i = out_im;
    double *R3A2_RESTRICT o1r = out_re + K;
    double *R3A2_RESTRICT o1i = out_im + K;
    double *R3A2_RESTRICT o2r = out_re + 2*K;
    double *R3A2_RESTRICT o2i = out_im + 2*K;

    const __m256d vhalf = _mm256_set1_pd(R3A2_C_HALF);
    const __m256d vsq3  = _mm256_set1_pd(R3A2_C_SQRT3_2);
    const __m256d vzero = _mm256_setzero_pd();

    const size_t k_vec = K & ~(size_t)3;

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k < k_vec; k += 4) {

        if (k + R3A2N1_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
        }

        __m256d ar = _mm256_loadu_pd(&a_re[k]);
        __m256d ai = _mm256_loadu_pd(&a_im[k]);
        __m256d br = _mm256_loadu_pd(&b_re[k]);
        __m256d bi = _mm256_loadu_pd(&b_im[k]);
        __m256d cr = _mm256_loadu_pd(&c_re[k]);
        __m256d ci = _mm256_loadu_pd(&c_im[k]);

        /* No twiddle multiply — pure butterfly */
        __m256d sum_r = _mm256_add_pd(br, cr);
        __m256d sum_i = _mm256_add_pd(bi, ci);
        __m256d dif_r = _mm256_sub_pd(br, cr);
        __m256d dif_i = _mm256_sub_pd(bi, ci);

        __m256d rot_r = _mm256_mul_pd(vsq3, dif_i);
        __m256d com_r = _mm256_fmadd_pd(vhalf, sum_r, ar);
        __m256d com_i = _mm256_fmadd_pd(vhalf, sum_i, ai);
        __m256d rot_i = _mm256_fnmadd_pd(vsq3, dif_r, vzero);

        _mm256_storeu_pd(&o0r[k], _mm256_add_pd(ar, sum_r));
        _mm256_storeu_pd(&o0i[k], _mm256_add_pd(ai, sum_i));
        _mm256_storeu_pd(&o1r[k], _mm256_add_pd(com_r, rot_r));
        _mm256_storeu_pd(&o1i[k], _mm256_add_pd(com_i, rot_i));
        _mm256_storeu_pd(&o2r[k], _mm256_sub_pd(com_r, rot_r));
        _mm256_storeu_pd(&o2i[k], _mm256_sub_pd(com_i, rot_i));
    }

    /* Scalar tail */
    {
        const double half    = R3A2_C_HALF;
        const double sqrt3_2 = R3A2_C_SQRT3_2;
        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];
            double sr = br_ + cr_, si = bi_ + ci_;
            double dr = br_ - cr_, di = bi_ - ci_;
            double rr =  sqrt3_2 * di;
            double ri = -sqrt3_2 * dr;
            double cm_r = ar_ + half * sr;
            double cm_i = ai_ + half * si;
            o0r[k] = ar_ + sr;  o0i[k] = ai_ + si;
            o1r[k] = cm_r + rr;  o1i[k] = cm_i + ri;
            o2r[k] = cm_r - rr;  o2i[k] = cm_i - ri;
        }
    }
}

/*============================================================================
 * BACKWARD N1
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3A2_INLINE void
radix3_stage_n1_backward_avx2(
    size_t K,
    const double *R3A2_RESTRICT in_re,
    const double *R3A2_RESTRICT in_im,
    double       *R3A2_RESTRICT out_re,
    double       *R3A2_RESTRICT out_im)
{
    const double *R3A2_RESTRICT a_re = in_re;
    const double *R3A2_RESTRICT a_im = in_im;
    const double *R3A2_RESTRICT b_re = in_re + K;
    const double *R3A2_RESTRICT b_im = in_im + K;
    const double *R3A2_RESTRICT c_re = in_re + 2*K;
    const double *R3A2_RESTRICT c_im = in_im + 2*K;

    double *R3A2_RESTRICT o0r = out_re;
    double *R3A2_RESTRICT o0i = out_im;
    double *R3A2_RESTRICT o1r = out_re + K;
    double *R3A2_RESTRICT o1i = out_im + K;
    double *R3A2_RESTRICT o2r = out_re + 2*K;
    double *R3A2_RESTRICT o2i = out_im + 2*K;

    const __m256d vhalf = _mm256_set1_pd(R3A2_C_HALF);
    const __m256d vsq3  = _mm256_set1_pd(R3A2_C_SQRT3_2);
    const __m256d vzero = _mm256_setzero_pd();

    const size_t k_vec = K & ~(size_t)3;

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k < k_vec; k += 4) {

        if (k + R3A2N1_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3A2N1_PF_DIST], _MM_HINT_NTA);
        }

        __m256d ar = _mm256_loadu_pd(&a_re[k]);
        __m256d ai = _mm256_loadu_pd(&a_im[k]);
        __m256d br = _mm256_loadu_pd(&b_re[k]);
        __m256d bi = _mm256_loadu_pd(&b_im[k]);
        __m256d cr = _mm256_loadu_pd(&c_re[k]);
        __m256d ci = _mm256_loadu_pd(&c_im[k]);

        __m256d sum_r = _mm256_add_pd(br, cr);
        __m256d sum_i = _mm256_add_pd(bi, ci);
        __m256d dif_r = _mm256_sub_pd(br, cr);
        __m256d dif_i = _mm256_sub_pd(bi, ci);

        __m256d rot_r = _mm256_fnmadd_pd(vsq3, dif_i, vzero);  /* backward */
        __m256d com_r = _mm256_fmadd_pd(vhalf, sum_r, ar);
        __m256d com_i = _mm256_fmadd_pd(vhalf, sum_i, ai);
        __m256d rot_i = _mm256_mul_pd(vsq3, dif_r);

        _mm256_storeu_pd(&o0r[k], _mm256_add_pd(ar, sum_r));
        _mm256_storeu_pd(&o0i[k], _mm256_add_pd(ai, sum_i));
        _mm256_storeu_pd(&o1r[k], _mm256_add_pd(com_r, rot_r));
        _mm256_storeu_pd(&o1i[k], _mm256_add_pd(com_i, rot_i));
        _mm256_storeu_pd(&o2r[k], _mm256_sub_pd(com_r, rot_r));
        _mm256_storeu_pd(&o2i[k], _mm256_sub_pd(com_i, rot_i));
    }

    {
        const double half    = R3A2_C_HALF;
        const double sqrt3_2 = R3A2_C_SQRT3_2;
        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];
            double sr = br_ + cr_, si = bi_ + ci_;
            double dr = br_ - cr_, di = bi_ - ci_;
            double rr = -sqrt3_2 * di;
            double ri =  sqrt3_2 * dr;
            double cm_r = ar_ + half * sr;
            double cm_i = ai_ + half * si;
            o0r[k] = ar_ + sr;  o0i[k] = ai_ + si;
            o1r[k] = cm_r + rr;  o1i[k] = cm_i + ri;
            o2r[k] = cm_r - rr;  o2i[k] = cm_i - ri;
        }
    }
}

#endif /* __AVX2__ && __FMA__ */
#endif /* FFT_RADIX3_AVX2_N1_H */