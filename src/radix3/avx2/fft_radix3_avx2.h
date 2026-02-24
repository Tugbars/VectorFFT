/**
 * @file fft_radix3_avx2.h
 * @brief AVX2 Radix-3 DIF Stage Kernels (FMA)
 *
 * SoA split-complex radix-3 butterfly, 4-wide (256-bit).
 * Forward and backward transforms with full twiddle multiply.
 *
 * TWIDDLE LAYOUT (SoA contiguous):
 *   tw->re[0..K-1]   = W1_re(k)     tw->im[0..K-1]   = W1_im(k)
 *   tw->re[K..2K-1]  = W2_re(k)     tw->im[K..2K-1]  = W2_im(k)
 *
 * DATA LAYOUT:
 *   in[0..K-1]   = row 0 (a)
 *   in[K..2K-1]  = row 1 (b)
 *   in[2K..3K-1] = row 2 (c)
 *
 * BUTTERFLY (forward DIF, ω = e^{-2πi/3}):
 *   tB = b·W1,  tC = c·W2           (FMA complex multiply)
 *   sum = tB+tC,  dif = tB-tC
 *   rot_re =  (√3/2)·dif_im         (mul)
 *   rot_im = -(√3/2)·dif_re         (fnmadd vs zero)
 *   common = a + (-½)·sum            (fmadd, half = -0.5)
 *   y0 = a + sum,  y1 = common+rot,  y2 = common-rot
 *
 * BACKWARD: rot signs flipped.
 *
 * REQUIREMENTS: K ≥ 4, K multiple of 4 (caller provides scalar tail).
 *               32-byte aligned data and twiddle arrays.
 *
 * REGISTER BUDGET (peak 11 / 16 YMM):
 *   3 constants (half, sqrt3_2, zero) hoisted outside loop
 *   2 a_re/im  (live until y0 store)
 *   2 tB_re/im (from cmul, live until dif)
 *   2 tC_re/im (from cmul, live until dif)
 *   2 sum/dif temporaries (sequential reuse)
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_AVX2_H
#define FFT_RADIX3_AVX2_H

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <stddef.h>

/*============================================================================
 * COMPILER PORTABILITY
 *============================================================================*/
#ifdef _MSC_VER
  #define R3A2_INLINE  static __forceinline
  #define R3A2_RESTRICT __restrict
  #define R3A2_ASSUME_ALIGNED(p, a) (p)
#elif defined(__GNUC__) || defined(__clang__)
  #define R3A2_INLINE  static inline __attribute__((always_inline))
  #define R3A2_RESTRICT __restrict__
  #define R3A2_ASSUME_ALIGNED(p, a) __builtin_assume_aligned(p, a)
#else
  #define R3A2_INLINE  static inline
  #define R3A2_RESTRICT
  #define R3A2_ASSUME_ALIGNED(p, a) (p)
#endif

/*============================================================================
 * TWIDDLE TYPES (guarded — shared across ISA headers)
 *============================================================================*/
#ifndef RADIX3_TWIDDLE_TYPES_DEFINED
#define RADIX3_TWIDDLE_TYPES_DEFINED

typedef struct {
    const double *R3A2_RESTRICT re;   /**< [2K]: W1_re[K] ∥ W2_re[K] */
    const double *R3A2_RESTRICT im;   /**< [2K]: W1_im[K] ∥ W2_im[K] */
} radix3_stage_twiddles_t;

#endif /* RADIX3_TWIDDLE_TYPES_DEFINED */

/*============================================================================
 * CONSTANTS
 *============================================================================*/
#define R3A2_C_HALF     (-0.5)
#define R3A2_C_SQRT3_2  0.86602540378443864676372317075293618347140262690519

/* Prefetch: 16 elements ≈ 4 iterations ahead (128 bytes per stream) */
#define R3A2_PF_DIST  16

/*============================================================================
 * FORWARD STAGE
 *============================================================================*/

/**
 * @brief AVX2 radix-3 forward stage with full twiddles.
 *
 * Processes K butterflies (K must be ≥ 4 and multiple of 4).
 * Caller handles scalar tail for K not a multiple of 4.
 */
/* no-unroll: GCC attribute, Clang/ICX use pragma inside loop */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3A2_INLINE void
radix3_stage_forward_avx2(
    size_t K,
    const double *R3A2_RESTRICT in_re,
    const double *R3A2_RESTRICT in_im,
    double       *R3A2_RESTRICT out_re,
    double       *R3A2_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3A2_RESTRICT tw)
{
    /* ── Hoist row pointers (AGU optimisation) ── */
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

    /* ── Twiddle pointers (SoA contiguous) ── */
    const double *R3A2_RESTRICT w1r = (const double *)R3A2_ASSUME_ALIGNED(tw->re, 32);
    const double *R3A2_RESTRICT w1i = (const double *)R3A2_ASSUME_ALIGNED(tw->im, 32);
    const double *R3A2_RESTRICT w2r = tw->re + K;
    const double *R3A2_RESTRICT w2i = tw->im + K;

    /* ── Hoisted constants (3 of 16 YMM) ── */
    const __m256d vhalf = _mm256_set1_pd(R3A2_C_HALF);
    const __m256d vsq3  = _mm256_set1_pd(R3A2_C_SQRT3_2);
    const __m256d vzero = _mm256_setzero_pd();

    const size_t k_vec = K & ~(size_t)3;   /* round down to multiple of 4 */

    /* ── Main vectorised loop ── */
/* Prevent unroll: GCC uses pragma GCC, Clang/ICX uses pragma clang */
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k < k_vec; k += 4) {

        /* Prefetch: T0 for twiddles (reused), NTA for input (streaming) */
        if (k + R3A2_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&w1r[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w1i[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2r[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2i[k + R3A2_PF_DIST],  _MM_HINT_T0);
        }

        /* ── Load inputs ── */
        __m256d ar = _mm256_loadu_pd(&a_re[k]);
        __m256d ai = _mm256_loadu_pd(&a_im[k]);
        __m256d br = _mm256_loadu_pd(&b_re[k]);
        __m256d bi = _mm256_loadu_pd(&b_im[k]);
        __m256d cr = _mm256_loadu_pd(&c_re[k]);
        __m256d ci = _mm256_loadu_pd(&c_im[k]);

        /* ── Load twiddles (sequential access — SoA contiguous) ── */
        __m256d tw1r = _mm256_loadu_pd(&w1r[k]);
        __m256d tw1i = _mm256_loadu_pd(&w1i[k]);
        __m256d tw2r = _mm256_loadu_pd(&w2r[k]);
        __m256d tw2i = _mm256_loadu_pd(&w2i[k]);

        /* ── Complex multiply: tB = b · W1 ── */
        __m256d tBr = _mm256_fnmadd_pd(bi, tw1i, _mm256_mul_pd(br, tw1r));
        __m256d tBi = _mm256_fmadd_pd(br, tw1i, _mm256_mul_pd(bi, tw1r));

        /* ── Complex multiply: tC = c · W2 ── */
        __m256d tCr = _mm256_fnmadd_pd(ci, tw2i, _mm256_mul_pd(cr, tw2r));
        __m256d tCi = _mm256_fmadd_pd(cr, tw2i, _mm256_mul_pd(ci, tw2r));

        /* ── Butterfly ── */
        __m256d sum_r = _mm256_add_pd(tBr, tCr);
        __m256d sum_i = _mm256_add_pd(tBi, tCi);
        __m256d dif_r = _mm256_sub_pd(tBr, tCr);
        __m256d dif_i = _mm256_sub_pd(tBi, tCi);

        /* micro-schedule: start rot_re before common (independent chain) */
        __m256d rot_r = _mm256_mul_pd(vsq3, dif_i);              /* +√3/2 · dif_im */
        __m256d com_r = _mm256_fmadd_pd(vhalf, sum_r, ar);       /* a + (-½)·sum   */
        __m256d com_i = _mm256_fmadd_pd(vhalf, sum_i, ai);
        __m256d rot_i = _mm256_fnmadd_pd(vsq3, dif_r, vzero);    /* -√3/2 · dif_re */

        /* y0 = a + sum  (store immediately → free a registers) */
        _mm256_storeu_pd(&o0r[k], _mm256_add_pd(ar, sum_r));
        _mm256_storeu_pd(&o0i[k], _mm256_add_pd(ai, sum_i));

        /* y1 = common + rot */
        _mm256_storeu_pd(&o1r[k], _mm256_add_pd(com_r, rot_r));
        _mm256_storeu_pd(&o1i[k], _mm256_add_pd(com_i, rot_i));

        /* y2 = common - rot */
        _mm256_storeu_pd(&o2r[k], _mm256_sub_pd(com_r, rot_r));
        _mm256_storeu_pd(&o2i[k], _mm256_sub_pd(com_i, rot_i));
    }

    /* ── Scalar tail (1-3 remaining elements) ── */
    {
        const double half    = R3A2_C_HALF;
        const double sqrt3_2 = R3A2_C_SQRT3_2;

        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];

            double tBr_ = br_ * w1r[k] - bi_ * w1i[k];
            double tBi_ = br_ * w1i[k] + bi_ * w1r[k];
            double tCr_ = cr_ * w2r[k] - ci_ * w2i[k];
            double tCi_ = cr_ * w2i[k] + ci_ * w2r[k];

            double sr = tBr_ + tCr_, si = tBi_ + tCi_;
            double dr = tBr_ - tCr_, di = tBi_ - tCi_;

            double rr =  sqrt3_2 * di;
            double ri = -sqrt3_2 * dr;
            double cr2 = ar_ + half * sr;
            double ci2 = ai_ + half * si;

            o0r[k] = ar_ + sr;  o0i[k] = ai_ + si;
            o1r[k] = cr2 + rr;  o1i[k] = ci2 + ri;
            o2r[k] = cr2 - rr;  o2i[k] = ci2 - ri;
        }
    }
}

/*============================================================================
 * BACKWARD STAGE
 *============================================================================*/

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
R3A2_INLINE void
radix3_stage_backward_avx2(
    size_t K,
    const double *R3A2_RESTRICT in_re,
    const double *R3A2_RESTRICT in_im,
    double       *R3A2_RESTRICT out_re,
    double       *R3A2_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3A2_RESTRICT tw)
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

    const double *R3A2_RESTRICT w1r = (const double *)R3A2_ASSUME_ALIGNED(tw->re, 32);
    const double *R3A2_RESTRICT w1i = (const double *)R3A2_ASSUME_ALIGNED(tw->im, 32);
    const double *R3A2_RESTRICT w2r = tw->re + K;
    const double *R3A2_RESTRICT w2i = tw->im + K;

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

        if (k + R3A2_PF_DIST < K) {
            _mm_prefetch((const char *)&a_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&a_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&b_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_re[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&c_im[k + R3A2_PF_DIST], _MM_HINT_NTA);
            _mm_prefetch((const char *)&w1r[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w1i[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2r[k + R3A2_PF_DIST],  _MM_HINT_T0);
            _mm_prefetch((const char *)&w2i[k + R3A2_PF_DIST],  _MM_HINT_T0);
        }

        __m256d ar = _mm256_loadu_pd(&a_re[k]);
        __m256d ai = _mm256_loadu_pd(&a_im[k]);
        __m256d br = _mm256_loadu_pd(&b_re[k]);
        __m256d bi = _mm256_loadu_pd(&b_im[k]);
        __m256d cr = _mm256_loadu_pd(&c_re[k]);
        __m256d ci = _mm256_loadu_pd(&c_im[k]);

        __m256d tw1r = _mm256_loadu_pd(&w1r[k]);
        __m256d tw1i = _mm256_loadu_pd(&w1i[k]);
        __m256d tw2r = _mm256_loadu_pd(&w2r[k]);
        __m256d tw2i = _mm256_loadu_pd(&w2i[k]);

        __m256d tBr = _mm256_fnmadd_pd(bi, tw1i, _mm256_mul_pd(br, tw1r));
        __m256d tBi = _mm256_fmadd_pd(br, tw1i, _mm256_mul_pd(bi, tw1r));

        __m256d tCr = _mm256_fnmadd_pd(ci, tw2i, _mm256_mul_pd(cr, tw2r));
        __m256d tCi = _mm256_fmadd_pd(cr, tw2i, _mm256_mul_pd(ci, tw2r));

        __m256d sum_r = _mm256_add_pd(tBr, tCr);
        __m256d sum_i = _mm256_add_pd(tBi, tCi);
        __m256d dif_r = _mm256_sub_pd(tBr, tCr);
        __m256d dif_i = _mm256_sub_pd(tBi, tCi);

        /* BACKWARD: rot signs flipped vs forward */
        __m256d rot_r = _mm256_fnmadd_pd(vsq3, dif_i, vzero);   /* -√3/2 · dif_im */
        __m256d com_r = _mm256_fmadd_pd(vhalf, sum_r, ar);
        __m256d com_i = _mm256_fmadd_pd(vhalf, sum_i, ai);
        __m256d rot_i = _mm256_mul_pd(vsq3, dif_r);             /* +√3/2 · dif_re */

        _mm256_storeu_pd(&o0r[k], _mm256_add_pd(ar, sum_r));
        _mm256_storeu_pd(&o0i[k], _mm256_add_pd(ai, sum_i));

        _mm256_storeu_pd(&o1r[k], _mm256_add_pd(com_r, rot_r));
        _mm256_storeu_pd(&o1i[k], _mm256_add_pd(com_i, rot_i));

        _mm256_storeu_pd(&o2r[k], _mm256_sub_pd(com_r, rot_r));
        _mm256_storeu_pd(&o2i[k], _mm256_sub_pd(com_i, rot_i));
    }

    /* ── Scalar tail ── */
    {
        const double half    = R3A2_C_HALF;
        const double sqrt3_2 = R3A2_C_SQRT3_2;

        for (size_t k = k_vec; k < K; k++) {
            double ar_ = a_re[k], ai_ = a_im[k];
            double br_ = b_re[k], bi_ = b_im[k];
            double cr_ = c_re[k], ci_ = c_im[k];

            double tBr_ = br_ * w1r[k] - bi_ * w1i[k];
            double tBi_ = br_ * w1i[k] + bi_ * w1r[k];
            double tCr_ = cr_ * w2r[k] - ci_ * w2i[k];
            double tCi_ = cr_ * w2i[k] + ci_ * w2r[k];

            double sr = tBr_ + tCr_, si = tBi_ + tCi_;
            double dr = tBr_ - tCr_, di = tBi_ - tCi_;

            double rr = -sqrt3_2 * di;          /* backward */
            double ri =  sqrt3_2 * dr;
            double cr2 = ar_ + half * sr;
            double ci2 = ai_ + half * si;

            o0r[k] = ar_ + sr;  o0i[k] = ai_ + si;
            o1r[k] = cr2 + rr;  o1i[k] = ci2 + ri;
            o2r[k] = cr2 - rr;  o2i[k] = ci2 - ri;
        }
    }
}

#endif /* __AVX2__ && __FMA__ */
#endif /* FFT_RADIX3_AVX2_H */