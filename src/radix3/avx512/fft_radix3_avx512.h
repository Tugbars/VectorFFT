/**
 * @file fft_radix3_avx512.h
 * @brief AVX-512 Radix-3 DIF Stage Kernels (FMA)
 *
 * 8-wide (512-bit) radix-3 butterfly, SoA split-complex.
 * Same architecture as AVX2 variant with wider vectors and
 * 32-register file eliminating all spill pressure.
 *
 * REQUIREMENTS: K ≥ 8, K multiple of 8 (caller provides fallback).
 *               64-byte aligned data and twiddle arrays.
 *
 * REGISTER BUDGET (peak 11 / 32 ZMM — no pressure):
 *   3 constants + 2 a + 2 tB + 2 tC + 2 temporaries
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_AVX512_H
#define FFT_RADIX3_AVX512_H

#if defined(__AVX512F__)

#include <immintrin.h>
#include <stddef.h>

/*============================================================================
 * COMPILER PORTABILITY
 *============================================================================*/
#ifdef _MSC_VER
  #define R3Z_INLINE  static __forceinline
  #define R3Z_RESTRICT __restrict
  #define R3Z_ASSUME_ALIGNED(p, a) (p)
#elif defined(__GNUC__) || defined(__clang__)
  #define R3Z_INLINE  static inline __attribute__((always_inline))
  #define R3Z_RESTRICT __restrict__
  #define R3Z_ASSUME_ALIGNED(p, a) __builtin_assume_aligned(p, a)
#else
  #define R3Z_INLINE  static inline
  #define R3Z_RESTRICT
  #define R3Z_ASSUME_ALIGNED(p, a) (p)
#endif

/*============================================================================
 * TWIDDLE TYPES (guarded)
 *============================================================================*/
#ifndef RADIX3_TWIDDLE_TYPES_DEFINED
#define RADIX3_TWIDDLE_TYPES_DEFINED

typedef struct {
    const double *R3Z_RESTRICT re;
    const double *R3Z_RESTRICT im;
} radix3_stage_twiddles_t;

#endif /* RADIX3_TWIDDLE_TYPES_DEFINED */

/*============================================================================
 * CONSTANTS
 *============================================================================*/
#define R3Z_C_HALF     (-0.5)
#define R3Z_C_SQRT3_2  0.86602540378443864676372317075293618347140262690519

/* Prefetch: 32 elements ≈ 4 iterations ahead (256 bytes per stream) */
#define R3Z_PF_DIST  32

/*============================================================================
 * FORWARD STAGE
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3Z_INLINE void
radix3_stage_forward_avx512(
    size_t K,
    const double *R3Z_RESTRICT in_re,
    const double *R3Z_RESTRICT in_im,
    double       *R3Z_RESTRICT out_re,
    double       *R3Z_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3Z_RESTRICT tw)
{
    const double *R3Z_RESTRICT a_re = in_re;
    const double *R3Z_RESTRICT a_im = in_im;
    const double *R3Z_RESTRICT b_re = in_re + K;
    const double *R3Z_RESTRICT b_im = in_im + K;
    const double *R3Z_RESTRICT c_re = in_re + 2*K;
    const double *R3Z_RESTRICT c_im = in_im + 2*K;

    double *R3Z_RESTRICT o0r = out_re;
    double *R3Z_RESTRICT o0i = out_im;
    double *R3Z_RESTRICT o1r = out_re + K;
    double *R3Z_RESTRICT o1i = out_im + K;
    double *R3Z_RESTRICT o2r = out_re + 2*K;
    double *R3Z_RESTRICT o2i = out_im + 2*K;

    const double *R3Z_RESTRICT w1r = (const double *)R3Z_ASSUME_ALIGNED(tw->re, 64);
    const double *R3Z_RESTRICT w1i = (const double *)R3Z_ASSUME_ALIGNED(tw->im, 64);
    const double *R3Z_RESTRICT w2r = tw->re + K;
    const double *R3Z_RESTRICT w2i = tw->im + K;

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

        if (k + R3Z_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&w1r[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w1i[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2r[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2i[k + R3Z_PF_DIST],  _MM_HINT_T0);
        }

        __m512d ar = _mm512_loadu_pd(&a_re[k]);
        __m512d ai = _mm512_loadu_pd(&a_im[k]);
        __m512d br = _mm512_loadu_pd(&b_re[k]);
        __m512d bi = _mm512_loadu_pd(&b_im[k]);
        __m512d cr = _mm512_loadu_pd(&c_re[k]);
        __m512d ci = _mm512_loadu_pd(&c_im[k]);

        __m512d tw1r = _mm512_loadu_pd(&w1r[k]);
        __m512d tw1i = _mm512_loadu_pd(&w1i[k]);
        __m512d tw2r = _mm512_loadu_pd(&w2r[k]);
        __m512d tw2i = _mm512_loadu_pd(&w2i[k]);

        /* tB = b · W1 */
        __m512d tBr = _mm512_fnmadd_pd(bi, tw1i, _mm512_mul_pd(br, tw1r));
        __m512d tBi = _mm512_fmadd_pd(br, tw1i, _mm512_mul_pd(bi, tw1r));

        /* tC = c · W2 */
        __m512d tCr = _mm512_fnmadd_pd(ci, tw2i, _mm512_mul_pd(cr, tw2r));
        __m512d tCi = _mm512_fmadd_pd(cr, tw2i, _mm512_mul_pd(ci, tw2r));

        __m512d sum_r = _mm512_add_pd(tBr, tCr);
        __m512d sum_i = _mm512_add_pd(tBi, tCi);
        __m512d dif_r = _mm512_sub_pd(tBr, tCr);
        __m512d dif_i = _mm512_sub_pd(tBi, tCi);

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

    /* Scalar tail (1-7 remaining) */
    {
        const double half    = R3Z_C_HALF;
        const double sqrt3_2 = R3Z_C_SQRT3_2;
        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];
            double tBr_ = br_*w1r[k] - bi_*w1i[k];
            double tBi_ = br_*w1i[k] + bi_*w1r[k];
            double tCr_ = cr_*w2r[k] - ci_*w2i[k];
            double tCi_ = cr_*w2i[k] + ci_*w2r[k];
            double sr = tBr_+tCr_, si = tBi_+tCi_;
            double dr = tBr_-tCr_, di = tBi_-tCi_;
            double rr =  sqrt3_2*di, ri = -sqrt3_2*dr;
            double cm_r = ar_ + half*sr, cm_i = ai_ + half*si;
            o0r[k] = ar_+sr;   o0i[k] = ai_+si;
            o1r[k] = cm_r+rr;  o1i[k] = cm_i+ri;
            o2r[k] = cm_r-rr;  o2i[k] = cm_i-ri;
        }
    }
}

/*============================================================================
 * BACKWARD STAGE
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3Z_INLINE void
radix3_stage_backward_avx512(
    size_t K,
    const double *R3Z_RESTRICT in_re,
    const double *R3Z_RESTRICT in_im,
    double       *R3Z_RESTRICT out_re,
    double       *R3Z_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3Z_RESTRICT tw)
{
    const double *R3Z_RESTRICT a_re = in_re;
    const double *R3Z_RESTRICT a_im = in_im;
    const double *R3Z_RESTRICT b_re = in_re + K;
    const double *R3Z_RESTRICT b_im = in_im + K;
    const double *R3Z_RESTRICT c_re = in_re + 2*K;
    const double *R3Z_RESTRICT c_im = in_im + 2*K;

    double *R3Z_RESTRICT o0r = out_re;
    double *R3Z_RESTRICT o0i = out_im;
    double *R3Z_RESTRICT o1r = out_re + K;
    double *R3Z_RESTRICT o1i = out_im + K;
    double *R3Z_RESTRICT o2r = out_re + 2*K;
    double *R3Z_RESTRICT o2i = out_im + 2*K;

    const double *R3Z_RESTRICT w1r = (const double *)R3Z_ASSUME_ALIGNED(tw->re, 64);
    const double *R3Z_RESTRICT w1i = (const double *)R3Z_ASSUME_ALIGNED(tw->im, 64);
    const double *R3Z_RESTRICT w2r = tw->re + K;
    const double *R3Z_RESTRICT w2i = tw->im + K;

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

        if (k + R3Z_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3Z_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&w1r[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w1i[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2r[k + R3Z_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2i[k + R3Z_PF_DIST],  _MM_HINT_T0);
        }

        __m512d ar = _mm512_loadu_pd(&a_re[k]);
        __m512d ai = _mm512_loadu_pd(&a_im[k]);
        __m512d br = _mm512_loadu_pd(&b_re[k]);
        __m512d bi = _mm512_loadu_pd(&b_im[k]);
        __m512d cr = _mm512_loadu_pd(&c_re[k]);
        __m512d ci = _mm512_loadu_pd(&c_im[k]);

        __m512d tw1r = _mm512_loadu_pd(&w1r[k]);
        __m512d tw1i = _mm512_loadu_pd(&w1i[k]);
        __m512d tw2r = _mm512_loadu_pd(&w2r[k]);
        __m512d tw2i = _mm512_loadu_pd(&w2i[k]);

        __m512d tBr = _mm512_fnmadd_pd(bi, tw1i, _mm512_mul_pd(br, tw1r));
        __m512d tBi = _mm512_fmadd_pd(br, tw1i, _mm512_mul_pd(bi, tw1r));
        __m512d tCr = _mm512_fnmadd_pd(ci, tw2i, _mm512_mul_pd(cr, tw2r));
        __m512d tCi = _mm512_fmadd_pd(cr, tw2i, _mm512_mul_pd(ci, tw2r));

        __m512d sum_r = _mm512_add_pd(tBr, tCr);
        __m512d sum_i = _mm512_add_pd(tBi, tCi);
        __m512d dif_r = _mm512_sub_pd(tBr, tCr);
        __m512d dif_i = _mm512_sub_pd(tBi, tCi);

        __m512d rot_r = _mm512_fnmadd_pd(vsq3, dif_i, vzero);  /* backward */
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
            double tBr_ = br_*w1r[k] - bi_*w1i[k];
            double tBi_ = br_*w1i[k] + bi_*w1r[k];
            double tCr_ = cr_*w2r[k] - ci_*w2i[k];
            double tCi_ = cr_*w2i[k] + ci_*w2r[k];
            double sr = tBr_+tCr_, si = tBi_+tCi_;
            double dr = tBr_-tCr_, di = tBi_-tCi_;
            double rr = -sqrt3_2*di, ri = sqrt3_2*dr;
            double cm_r = ar_ + half*sr, cm_i = ai_ + half*si;
            o0r[k] = ar_+sr;   o0i[k] = ai_+si;
            o1r[k] = cm_r+rr;  o1i[k] = cm_i+ri;
            o2r[k] = cm_r-rr;  o2i[k] = cm_i-ri;
        }
    }
}

#endif /* __AVX512F__ */
#endif /* FFT_RADIX3_AVX512_H */