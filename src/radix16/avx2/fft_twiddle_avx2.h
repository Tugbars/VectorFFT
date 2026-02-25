/**
 * @file fft_twiddle_avx2.h
 * @brief AVX2+FMA External Twiddle Application — Element-wise Complex Multiply
 *
 * @details
 * AVX2 counterpart of fft_twiddle_avx512.h. Same API, same semantics.
 *
 * Each YMM register holds 4 doubles, so one SIMD "column" processes
 * 4 independent complex elements in parallel.
 *
 * Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 * FMA pattern (same as AVX-512):
 *   tmp  = data_re · tw_re
 *   re'  = fnmadd(data_im, tw_im, tmp)
 *   tmp2 = data_im · tw_re
 *   im'  = fmadd(data_re, tw_im, tmp2)
 *
 * Throughput: 2 MUL + 2 FMA = 4 FP ops per 4 elements.
 *
 * Unrolling: U=4 (16 elements per iteration).
 *   Register budget: 4×(data_re + data_im + tw_re + tw_im) = 16 YMM (tight)
 *   Temporaries reuse input registers after loads complete.
 *
 * Tail: scalar cleanup for count % 4.
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_TWIDDLE_AVX2_H
#define FFT_TWIDDLE_AVX2_H

#include <immintrin.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define TWDY_INLINE      static __forceinline
  #define TWDY_RESTRICT     __restrict
  #define TWDY_NOINLINE     static __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
  #define TWDY_INLINE      static inline __attribute__((always_inline))
  #define TWDY_RESTRICT     __restrict__
  #define TWDY_NOINLINE     static __attribute__((noinline))
#else
  #define TWDY_INLINE      static inline
  #define TWDY_RESTRICT
  #define TWDY_NOINLINE     static
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TWDY_PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
  #define TWDY_PREFETCH_T0(addr)  ((void)0)
#endif

/* ============================================================================
 * CORE: 4-WIDE COMPLEX MULTIPLY (one YMM-width)
 * ========================================================================= */

TWDY_INLINE void twdy_cmul_4(
    double *TWDY_RESTRICT data_re,
    double *TWDY_RESTRICT data_im,
    const double *TWDY_RESTRICT tw_re,
    const double *TWDY_RESTRICT tw_im)
{
    __m256d dr = _mm256_loadu_pd(data_re);
    __m256d di = _mm256_loadu_pd(data_im);
    __m256d wr = _mm256_loadu_pd(tw_re);
    __m256d wi = _mm256_loadu_pd(tw_im);

    /* (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
    __m256d ac = _mm256_mul_pd(dr, wr);
    __m256d bc = _mm256_mul_pd(di, wr);
    __m256d re = _mm256_fnmadd_pd(di, wi, ac);   /* ac - bd */
    __m256d im = _mm256_fmadd_pd(dr, wi, bc);    /* ad + bc */

    _mm256_storeu_pd(data_re, re);
    _mm256_storeu_pd(data_im, im);
}

/* ============================================================================
 * PUBLIC API: IN-PLACE TWIDDLE APPLICATION
 *
 * data_re[i] + j·data_im[i]  *=  tw_re[i] + j·tw_im[i]
 * for i = 0 .. count-1
 *
 * Main loop: U=4 (16 elements per iteration).
 * Tail: full 4-wide batches + scalar cleanup.
 * ========================================================================= */

TWDY_NOINLINE void fft_twiddle_apply_avx2(
    size_t count,
    double *TWDY_RESTRICT data_re,
    double *TWDY_RESTRICT data_im,
    const double *TWDY_RESTRICT tw_re,
    const double *TWDY_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;

    /* ---- Main loop: U=4, 16 elements per iteration ---- */
    const size_t count_u4 = (count / 16) * 16;
    for (; i < count_u4; i += 16)
    {
        TWDY_PREFETCH_T0(tw_re + i + 16);
        TWDY_PREFETCH_T0(tw_im + i + 16);

        twdy_cmul_4(data_re + i,      data_im + i,
                     tw_re + i,        tw_im + i);
        twdy_cmul_4(data_re + i + 4,  data_im + i + 4,
                     tw_re + i + 4,    tw_im + i + 4);
        twdy_cmul_4(data_re + i + 8,  data_im + i + 8,
                     tw_re + i + 8,    tw_im + i + 8);
        twdy_cmul_4(data_re + i + 12, data_im + i + 12,
                     tw_re + i + 12,   tw_im + i + 12);
    }

    /* ---- Tail: full 4-wide batches ---- */
    for (; i + 4 <= count; i += 4)
    {
        twdy_cmul_4(data_re + i, data_im + i,
                     tw_re + i,  tw_im + i);
    }

    /* ---- Scalar cleanup: 0..3 remaining ---- */
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

TWDY_NOINLINE void fft_twiddle_apply_avx2_oop(
    size_t count,
    const double *TWDY_RESTRICT data_re,
    const double *TWDY_RESTRICT data_im,
    double *TWDY_RESTRICT out_re,
    double *TWDY_RESTRICT out_im,
    const double *TWDY_RESTRICT tw_re,
    const double *TWDY_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;
    const size_t count_u4 = (count / 16) * 16;

    for (; i < count_u4; i += 16)
    {
        TWDY_PREFETCH_T0(tw_re + i + 16);
        TWDY_PREFETCH_T0(tw_im + i + 16);

        for (int u = 0; u < 4; u++)
        {
            size_t off = i + (size_t)u * 4;
            __m256d dr = _mm256_loadu_pd(data_re + off);
            __m256d di = _mm256_loadu_pd(data_im + off);
            __m256d wr = _mm256_loadu_pd(tw_re + off);
            __m256d wi = _mm256_loadu_pd(tw_im + off);

            __m256d ac = _mm256_mul_pd(dr, wr);
            __m256d bc = _mm256_mul_pd(di, wr);
            _mm256_storeu_pd(out_re + off, _mm256_fnmadd_pd(di, wi, ac));
            _mm256_storeu_pd(out_im + off, _mm256_fmadd_pd(dr, wi, bc));
        }
    }

    for (; i + 4 <= count; i += 4)
    {
        __m256d dr = _mm256_loadu_pd(data_re + i);
        __m256d di = _mm256_loadu_pd(data_im + i);
        __m256d wr = _mm256_loadu_pd(tw_re + i);
        __m256d wi = _mm256_loadu_pd(tw_im + i);

        __m256d ac = _mm256_mul_pd(dr, wr);
        __m256d bc = _mm256_mul_pd(di, wr);
        _mm256_storeu_pd(out_re + i, _mm256_fnmadd_pd(di, wi, ac));
        _mm256_storeu_pd(out_im + i, _mm256_fmadd_pd(dr, wi, bc));
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

TWDY_INLINE void fft_twiddle_apply_avx2_skip_row0(
    size_t radix,
    size_t K,
    double *TWDY_RESTRICT data_re,
    double *TWDY_RESTRICT data_im,
    const double *TWDY_RESTRICT tw_re,
    const double *TWDY_RESTRICT tw_im)
{
    assert(radix >= 2);
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_avx2(count,
        data_re + offset, data_im + offset,
        tw_re + offset,   tw_im + offset);
}

TWDY_INLINE void fft_twiddle_apply_avx2_skip_row0_oop(
    size_t radix,
    size_t K,
    const double *TWDY_RESTRICT data_re,
    const double *TWDY_RESTRICT data_im,
    double *TWDY_RESTRICT out_re,
    double *TWDY_RESTRICT out_im,
    const double *TWDY_RESTRICT tw_re,
    const double *TWDY_RESTRICT tw_im)
{
    assert(radix >= 2);

    /* Copy row 0 unchanged */
    memcpy(out_re, data_re, K * sizeof(double));
    memcpy(out_im, data_im, K * sizeof(double));

    /* Twiddle rows 1..R-1 */
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_avx2_oop(count,
        data_re + offset, data_im + offset,
        out_re + offset,  out_im + offset,
        tw_re + offset,   tw_im + offset);
}

#endif /* FFT_TWIDDLE_AVX2_H */
