/**
 * @file fft_twiddle_scalar.h
 * @brief Scalar External Twiddle Application — Element-wise Complex Multiply
 *
 * @details
 * Pure C fallback for fft_twiddle_avx512.h / fft_twiddle_avx2.h.
 * Same API, same semantics. No SIMD dependency.
 *
 * Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 * 4 MUL + 2 ADD per element. U=4 manual unrolling for ILP.
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_TWIDDLE_SCALAR_H
#define FFT_TWIDDLE_SCALAR_H

#include <stddef.h>
#include <assert.h>
#include <string.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define TWDS_INLINE      static __forceinline
  #define TWDS_RESTRICT     __restrict
  #define TWDS_NOINLINE     static __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
  #define TWDS_INLINE      static inline __attribute__((always_inline))
  #define TWDS_RESTRICT     __restrict__
  #define TWDS_NOINLINE     static __attribute__((noinline))
#else
  #define TWDS_INLINE      static inline
  #define TWDS_RESTRICT
  #define TWDS_NOINLINE     static
#endif

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  #define TWDS_IVDEP _Pragma("ivdep")
#elif defined(__GNUC__)
  #define TWDS_IVDEP _Pragma("GCC ivdep")
#else
  #define TWDS_IVDEP
#endif

/* ============================================================================
 * PUBLIC API: IN-PLACE TWIDDLE APPLICATION
 * ========================================================================= */

TWDS_NOINLINE void fft_twiddle_apply_scalar(
    size_t count,
    double *TWDS_RESTRICT data_re,
    double *TWDS_RESTRICT data_im,
    const double *TWDS_RESTRICT tw_re,
    const double *TWDS_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;

    /* U=4 for OoO ILP */
    const size_t count_u4 = (count / 4) * 4;
    TWDS_IVDEP
    for (; i < count_u4; i += 4)
    {
        double a0 = data_re[i],   b0 = data_im[i];
        double a1 = data_re[i+1], b1 = data_im[i+1];
        double a2 = data_re[i+2], b2 = data_im[i+2];
        double a3 = data_re[i+3], b3 = data_im[i+3];

        double c0 = tw_re[i],   d0 = tw_im[i];
        double c1 = tw_re[i+1], d1 = tw_im[i+1];
        double c2 = tw_re[i+2], d2 = tw_im[i+2];
        double c3 = tw_re[i+3], d3 = tw_im[i+3];

        data_re[i]   = a0*c0 - b0*d0;  data_im[i]   = a0*d0 + b0*c0;
        data_re[i+1] = a1*c1 - b1*d1;  data_im[i+1] = a1*d1 + b1*c1;
        data_re[i+2] = a2*c2 - b2*d2;  data_im[i+2] = a2*d2 + b2*c2;
        data_re[i+3] = a3*c3 - b3*d3;  data_im[i+3] = a3*d3 + b3*c3;
    }

    for (; i < count; i++)
    {
        double a = data_re[i], b = data_im[i];
        double c = tw_re[i],   d = tw_im[i];
        data_re[i] = a * c - b * d;
        data_im[i] = a * d + b * c;
    }
}

/* ============================================================================
 * OUT-OF-PLACE VARIANT
 * ========================================================================= */

TWDS_NOINLINE void fft_twiddle_apply_scalar_oop(
    size_t count,
    const double *TWDS_RESTRICT data_re,
    const double *TWDS_RESTRICT data_im,
    double *TWDS_RESTRICT out_re,
    double *TWDS_RESTRICT out_im,
    const double *TWDS_RESTRICT tw_re,
    const double *TWDS_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;
    const size_t count_u4 = (count / 4) * 4;
    TWDS_IVDEP
    for (; i < count_u4; i += 4)
    {
        for (int u = 0; u < 4; u++)
        {
            double a = data_re[i+u], b = data_im[i+u];
            double c = tw_re[i+u],   d = tw_im[i+u];
            out_re[i+u] = a * c - b * d;
            out_im[i+u] = a * d + b * c;
        }
    }
    for (; i < count; i++)
    {
        double a = data_re[i], b = data_im[i];
        double c = tw_re[i],   d = tw_im[i];
        out_re[i] = a * c - b * d;
        out_im[i] = a * d + b * c;
    }
}

/* ============================================================================
 * CONVENIENCE: SKIP ROW 0
 * ========================================================================= */

TWDS_INLINE void fft_twiddle_apply_scalar_skip_row0(
    size_t radix,
    size_t K,
    double *TWDS_RESTRICT data_re,
    double *TWDS_RESTRICT data_im,
    const double *TWDS_RESTRICT tw_re,
    const double *TWDS_RESTRICT tw_im)
{
    assert(radix >= 2);
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_scalar(count,
        data_re + offset, data_im + offset,
        tw_re + offset,   tw_im + offset);
}

TWDS_INLINE void fft_twiddle_apply_scalar_skip_row0_oop(
    size_t radix,
    size_t K,
    const double *TWDS_RESTRICT data_re,
    const double *TWDS_RESTRICT data_im,
    double *TWDS_RESTRICT out_re,
    double *TWDS_RESTRICT out_im,
    const double *TWDS_RESTRICT tw_re,
    const double *TWDS_RESTRICT tw_im)
{
    assert(radix >= 2);
    memcpy(out_re, data_re, K * sizeof(double));
    memcpy(out_im, data_im, K * sizeof(double));
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_scalar_oop(count,
        data_re + offset, data_im + offset,
        out_re + offset,  out_im + offset,
        tw_re + offset,   tw_im + offset);
}

#endif /* FFT_TWIDDLE_SCALAR_H */
