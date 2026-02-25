/**
 * @file fft_twiddle_avx512.h
 * @brief AVX-512 External Twiddle Application — Element-wise Complex Multiply
 *
 * @details
 * Applies precomputed twiddle factors to FFT data via in-place element-wise
 * complex multiplication:  data[i] *= twiddle[i]  for i = 0..count-1
 *
 * This is the inter-stage twiddle in a mixed-radix FFT:
 *   After radix-R butterfly:  Y[r·K + k] *= W_N^{r·k}
 *   where N = full FFT size, r = row (0..R-1), k = column (0..K-1)
 *
 * The function is radix-agnostic. It just multiplies `count` complex elements.
 * Row-0 skip (where W_N^0 = 1) is the caller's responsibility — use the
 * _skip_row0 wrapper which adjusts pointers and count automatically.
 *
 * Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 * FMA pattern:
 *   tmp  = data_re · tw_re
 *   re'  = fnmadd(data_im, tw_im, tmp)    // -(data_im·tw_im) + tmp
 *   tmp2 = data_im · tw_re
 *   im'  = fmadd(data_re, tw_im, tmp2)    // data_re·tw_im + tmp2
 *
 * Throughput: 2 MUL + 2 FMA = 4 FP ops per complex element.
 * At 2 FMA/cycle (port 0+1), theoretical: 2 cycles per element.
 *
 * Unrolling: U=4 (32 elements per iteration).
 *   Register budget: 4×(data_re + data_im + tw_re + tw_im) = 16 ZMM
 *   + 4 tmp + 4 tmp2 = 8 ZMM temporaries (reused in-place)
 *   Peak: ~20/32 ZMM. 12 free.
 *
 * Tail: opmask (__mmask8) for count % 8.
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_TWIDDLE_AVX512_H
#define FFT_TWIDDLE_AVX512_H

#include <immintrin.h>
#include <stddef.h>
#include <assert.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define TWDZ_INLINE      static __forceinline
  #define TWDZ_RESTRICT     __restrict
  #define TWDZ_NOINLINE     static __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
  #define TWDZ_INLINE      static inline __attribute__((always_inline))
  #define TWDZ_RESTRICT     __restrict__
  #define TWDZ_NOINLINE     static __attribute__((noinline))
#else
  #define TWDZ_INLINE      static inline
  #define TWDZ_RESTRICT
  #define TWDZ_NOINLINE     static
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define TWDZ_PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
  #define TWDZ_PREFETCH_T0(addr)  ((void)0)
#endif

/* ============================================================================
 * CORE: 8-WIDE COMPLEX MULTIPLY (one ZMM-width)
 *
 * In-place: data *= twiddle
 * ========================================================================= */

TWDZ_INLINE void twdz_cmul_8(
    double *TWDZ_RESTRICT data_re,
    double *TWDZ_RESTRICT data_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im)
{
    __m512d dr = _mm512_loadu_pd(data_re);
    __m512d di = _mm512_loadu_pd(data_im);
    __m512d wr = _mm512_loadu_pd(tw_re);
    __m512d wi = _mm512_loadu_pd(tw_im);

    /* (a+bi)(c+di) = (ac-bd) + (ad+bc)i */
    __m512d ac = _mm512_mul_pd(dr, wr);         /* a·c */
    __m512d bc = _mm512_mul_pd(di, wr);         /* b·c */
    __m512d re = _mm512_fnmadd_pd(di, wi, ac);  /* ac - bd */
    __m512d im = _mm512_fmadd_pd(dr, wi, bc);   /* ad + bc */

    _mm512_storeu_pd(data_re, re);
    _mm512_storeu_pd(data_im, im);
}

/* Masked version for tail */
TWDZ_INLINE void twdz_cmul_8_masked(
    double *TWDZ_RESTRICT data_re,
    double *TWDZ_RESTRICT data_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im,
    __mmask8 mask)
{
    __m512d dr = _mm512_maskz_loadu_pd(mask, data_re);
    __m512d di = _mm512_maskz_loadu_pd(mask, data_im);
    __m512d wr = _mm512_maskz_loadu_pd(mask, tw_re);
    __m512d wi = _mm512_maskz_loadu_pd(mask, tw_im);

    __m512d ac = _mm512_mul_pd(dr, wr);
    __m512d bc = _mm512_mul_pd(di, wr);
    __m512d re = _mm512_fnmadd_pd(di, wi, ac);
    __m512d im = _mm512_fmadd_pd(dr, wi, bc);

    _mm512_mask_storeu_pd(data_re, mask, re);
    _mm512_mask_storeu_pd(data_im, mask, im);
}

/* ============================================================================
 * PUBLIC API: IN-PLACE TWIDDLE APPLICATION
 *
 * data_re[i] + j·data_im[i]  *=  tw_re[i] + j·tw_im[i]
 * for i = 0 .. count-1
 *
 * Main loop: U=4 (32 elements per iteration) for maximum ILP.
 * Tail: full 8-wide batches + masked final batch.
 *
 * Prefetch: next iteration's twiddle data to L1. The data arrays are
 * written back so they stay hot, but twiddle tables may be cold on
 * first access per block.
 * ========================================================================= */

TWDZ_NOINLINE void fft_twiddle_apply_avx512(
    size_t count,
    double *TWDZ_RESTRICT data_re,
    double *TWDZ_RESTRICT data_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;

    /* ---- Main loop: U=4, 32 elements per iteration ---- */
    const size_t count_u4 = (count / 32) * 32;
    for (; i < count_u4; i += 32)
    {
        /* Prefetch next iteration's twiddle data */
        TWDZ_PREFETCH_T0(tw_re + i + 32);
        TWDZ_PREFETCH_T0(tw_im + i + 32);

        twdz_cmul_8(data_re + i,      data_im + i,
                     tw_re + i,        tw_im + i);
        twdz_cmul_8(data_re + i + 8,  data_im + i + 8,
                     tw_re + i + 8,    tw_im + i + 8);
        twdz_cmul_8(data_re + i + 16, data_im + i + 16,
                     tw_re + i + 16,   tw_im + i + 16);
        twdz_cmul_8(data_re + i + 24, data_im + i + 24,
                     tw_re + i + 24,   tw_im + i + 24);
    }

    /* ---- Tail: full 8-wide batches ---- */
    for (; i + 8 <= count; i += 8)
    {
        twdz_cmul_8(data_re + i, data_im + i,
                     tw_re + i,  tw_im + i);
    }

    /* ---- Final partial batch ---- */
    if (i < count)
    {
        const __mmask8 mask = (__mmask8)((1u << (count - i)) - 1u);
        twdz_cmul_8_masked(data_re + i, data_im + i,
                           tw_re + i,   tw_im + i, mask);
    }
}

/* ============================================================================
 * OUT-OF-PLACE VARIANT
 *
 * out[i] = data[i] * twiddle[i]
 *
 * Useful when butterfly output goes to a separate buffer and twiddle
 * application feeds the next stage's input.
 * ========================================================================= */

TWDZ_NOINLINE void fft_twiddle_apply_avx512_oop(
    size_t count,
    const double *TWDZ_RESTRICT data_re,
    const double *TWDZ_RESTRICT data_im,
    double *TWDZ_RESTRICT out_re,
    double *TWDZ_RESTRICT out_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im)
{
    if (count == 0) return;

    size_t i = 0;
    const size_t count_u4 = (count / 32) * 32;

    for (; i < count_u4; i += 32)
    {
        TWDZ_PREFETCH_T0(tw_re + i + 32);
        TWDZ_PREFETCH_T0(tw_im + i + 32);

        for (int u = 0; u < 4; u++)
        {
            size_t off = i + (size_t)u * 8;
            __m512d dr = _mm512_loadu_pd(data_re + off);
            __m512d di = _mm512_loadu_pd(data_im + off);
            __m512d wr = _mm512_loadu_pd(tw_re + off);
            __m512d wi = _mm512_loadu_pd(tw_im + off);

            __m512d ac = _mm512_mul_pd(dr, wr);
            __m512d bc = _mm512_mul_pd(di, wr);
            _mm512_storeu_pd(out_re + off, _mm512_fnmadd_pd(di, wi, ac));
            _mm512_storeu_pd(out_im + off, _mm512_fmadd_pd(dr, wi, bc));
        }
    }

    for (; i + 8 <= count; i += 8)
    {
        __m512d dr = _mm512_loadu_pd(data_re + i);
        __m512d di = _mm512_loadu_pd(data_im + i);
        __m512d wr = _mm512_loadu_pd(tw_re + i);
        __m512d wi = _mm512_loadu_pd(tw_im + i);

        __m512d ac = _mm512_mul_pd(dr, wr);
        __m512d bc = _mm512_mul_pd(di, wr);
        _mm512_storeu_pd(out_re + i, _mm512_fnmadd_pd(di, wi, ac));
        _mm512_storeu_pd(out_im + i, _mm512_fmadd_pd(dr, wi, bc));
    }

    if (i < count)
    {
        const __mmask8 mask = (__mmask8)((1u << (count - i)) - 1u);
        __m512d dr = _mm512_maskz_loadu_pd(mask, data_re + i);
        __m512d di = _mm512_maskz_loadu_pd(mask, data_im + i);
        __m512d wr = _mm512_maskz_loadu_pd(mask, tw_re + i);
        __m512d wi = _mm512_maskz_loadu_pd(mask, tw_im + i);

        __m512d ac = _mm512_mul_pd(dr, wr);
        __m512d bc = _mm512_mul_pd(di, wr);
        _mm512_mask_storeu_pd(out_re + i, mask, _mm512_fnmadd_pd(di, wi, ac));
        _mm512_mask_storeu_pd(out_im + i, mask, _mm512_fmadd_pd(dr, wi, bc));
    }
}

/* ============================================================================
 * CONVENIENCE: SKIP ROW 0
 *
 * For radix-R butterfly output with K columns per row:
 *   Row 0 always has W_N^0 = 1 (identity twiddle), so skip it.
 *   Apply twiddles to rows 1..R-1, i.e. (R-1)·K elements starting at offset K.
 *
 * Usage for radix-16:
 *   fft_twiddle_apply_avx512_skip_row0(16, K, data_re, data_im, tw_re, tw_im);
 * ========================================================================= */

TWDZ_INLINE void fft_twiddle_apply_avx512_skip_row0(
    size_t radix,
    size_t K,
    double *TWDZ_RESTRICT data_re,
    double *TWDZ_RESTRICT data_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im)
{
    assert(radix >= 2);
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_avx512(count,
        data_re + offset, data_im + offset,
        tw_re + offset,   tw_im + offset);
}

/* Out-of-place variant: copies row 0 unchanged, twiddles rows 1..R-1 */
TWDZ_INLINE void fft_twiddle_apply_avx512_skip_row0_oop(
    size_t radix,
    size_t K,
    const double *TWDZ_RESTRICT data_re,
    const double *TWDZ_RESTRICT data_im,
    double *TWDZ_RESTRICT out_re,
    double *TWDZ_RESTRICT out_im,
    const double *TWDZ_RESTRICT tw_re,
    const double *TWDZ_RESTRICT tw_im)
{
    assert(radix >= 2);

    /* Copy row 0 unchanged */
    for (size_t i = 0; i < K; i += 8)
    {
        if (i + 8 <= K)
        {
            _mm512_storeu_pd(out_re + i, _mm512_loadu_pd(data_re + i));
            _mm512_storeu_pd(out_im + i, _mm512_loadu_pd(data_im + i));
        }
        else
        {
            const __mmask8 m = (__mmask8)((1u << (K - i)) - 1u);
            _mm512_mask_storeu_pd(out_re + i, m, _mm512_maskz_loadu_pd(m, data_re + i));
            _mm512_mask_storeu_pd(out_im + i, m, _mm512_maskz_loadu_pd(m, data_im + i));
        }
    }

    /* Twiddle rows 1..R-1 */
    const size_t offset = K;
    const size_t count  = (radix - 1) * K;
    fft_twiddle_apply_avx512_oop(count,
        data_re + offset, data_im + offset,
        out_re + offset,  out_im + offset,
        tw_re + offset,   tw_im + offset);
}

#endif /* FFT_TWIDDLE_AVX512_H */
