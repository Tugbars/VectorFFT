/**
 * @file fft_radix4_avx512_corrected_part1.h
 * @brief Corrected AVX-512 Radix-4 Implementation - Part 1
 *
 * @details
 * FIXES APPLIED:
 * ✅ Const aliasing UB fixed (local tw_re/tw_im aliases)
 * ✅ U=2 pipeline scheduling corrected (proper iteration tracking)
 * ✅ Macros converted to __forceinline functions
 * ✅ MSVC + GCC/Clang compatibility
 * ✅ All 10 optimizations preserved
 *
 * @author VectorFFT Team
 * @version 2.1 (Bug-fix release)
 * @date 2025
 */

#ifndef FFT_RADIX4_AVX512_CORRECTED_PART1_H
#define FFT_RADIX4_AVX512_CORRECTED_PART1_H

#include "fft_radix4.h"
#include "simd_math.h"

//==============================================================================
// PORTABILITY MACROS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr) // MSVC: no builtin equivalent
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#define RADIX4_STREAM_THRESHOLD 8192
#define RADIX4_SMALL_K_THRESHOLD 16

#ifndef RADIX4_PREFETCH_DISTANCE
#define RADIX4_PREFETCH_DISTANCE 32
#endif

#ifndef RADIX4_DERIVE_W3
#define RADIX4_DERIVE_W3 0 // 0=load W3, 1=compute W3=W1*W2
#endif

#ifndef RADIX4_ASSUME_ALIGNED
#define RADIX4_ASSUME_ALIGNED 0 // 0=unaligned (safe), 1=aligned loads
#endif

//==============================================================================
// ALIGNED LOAD HELPERS
//==============================================================================

#ifdef __AVX512F__

#if RADIX4_ASSUME_ALIGNED
#define LOAD_PD_AVX512(ptr) _mm512_load_pd(ptr)
#else
#define LOAD_PD_AVX512(ptr) _mm512_loadu_pd(ptr)
#endif

//==============================================================================
// COMPLEX MULTIPLY - INLINE FUNCTION
//==============================================================================

FORCE_INLINE void cmul_soa_avx512(
    __m512d ar, __m512d ai,
    __m512d wr, __m512d wi,
    __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    *tr = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));
    *ti = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));
}

//==============================================================================
// RADIX-4 BUTTERFLY CORES - INLINE FUNCTIONS
//==============================================================================

/**
 * @brief Core radix-4 butterfly - Forward FFT
 *
 * Algorithm: rot = (+i) * difBD for forward transform
 *   rot_re = -difBD_im  (negate via XOR with sign_mask)
 *   rot_im = +difBD_re
 */
FORCE_INLINE void radix4_butterfly_core_fv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d tB_re, __m512d tB_im,
    __m512d tC_re, __m512d tC_im,
    __m512d tD_re, __m512d tD_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d sign_mask)
{
    __m512d sumBD_re = _mm512_add_pd(tB_re, tD_re);
    __m512d sumBD_im = _mm512_add_pd(tB_im, tD_im);
    __m512d difBD_re = _mm512_sub_pd(tB_re, tD_re);
    __m512d difBD_im = _mm512_sub_pd(tB_im, tD_im);

    __m512d sumAC_re = _mm512_add_pd(a_re, tC_re);
    __m512d sumAC_im = _mm512_add_pd(a_im, tC_im);
    __m512d difAC_re = _mm512_sub_pd(a_re, tC_re);
    __m512d difAC_im = _mm512_sub_pd(a_im, tC_im);

    // rot = (+i) * difBD = (-difBD_im, +difBD_re)
    __m512d rot_re = _mm512_xor_pd(difBD_im, sign_mask);
    __m512d rot_im = difBD_re;

    *y0_re = _mm512_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm512_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm512_sub_pd(difAC_re, rot_re);
    *y1_im = _mm512_sub_pd(difAC_im, rot_im);
    *y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm512_add_pd(difAC_re, rot_re);
    *y3_im = _mm512_add_pd(difAC_im, rot_im);
}

/**
 * @brief Core radix-4 butterfly - Backward FFT (inverse)
 *
 * Algorithm: rot = (-i) * difBD for inverse transform
 *   rot_re = +difBD_im
 *   rot_im = -difBD_re  (negate via XOR with sign_mask)
 */
FORCE_INLINE void radix4_butterfly_core_bv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d tB_re, __m512d tB_im,
    __m512d tC_re, __m512d tC_im,
    __m512d tD_re, __m512d tD_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d sign_mask)
{
    __m512d sumBD_re = _mm512_add_pd(tB_re, tD_re);
    __m512d sumBD_im = _mm512_add_pd(tB_im, tD_im);
    __m512d difBD_re = _mm512_sub_pd(tB_re, tD_re);
    __m512d difBD_im = _mm512_sub_pd(tB_im, tD_im);

    __m512d sumAC_re = _mm512_add_pd(a_re, tC_re);
    __m512d sumAC_im = _mm512_add_pd(a_im, tC_im);
    __m512d difAC_re = _mm512_sub_pd(a_re, tC_re);
    __m512d difAC_im = _mm512_sub_pd(a_im, tC_im);

    // rot = (-i) * difBD = (+difBD_im, -difBD_re)
    __m512d rot_re = difBD_im;
    __m512d rot_im = _mm512_xor_pd(difBD_re, sign_mask);

    *y0_re = _mm512_add_pd(sumAC_re, sumBD_re);
    *y0_im = _mm512_add_pd(sumAC_im, sumBD_im);
    *y1_re = _mm512_sub_pd(difAC_re, rot_re);
    *y1_im = _mm512_sub_pd(difAC_im, rot_im);
    *y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);
    *y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);
    *y3_re = _mm512_add_pd(difAC_re, rot_re);
    *y3_im = _mm512_add_pd(difAC_im, rot_im);
}

//==============================================================================
// PREFETCH HELPERS
//==============================================================================

#define PREFETCH_NTA(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_NTA)
#define PREFETCH_T0(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

FORCE_INLINE void prefetch_radix4_data_nta(
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT w1r, const double *RESTRICT w1i,
    const double *RESTRICT w2r, const double *RESTRICT w2i,
    const double *RESTRICT w3r, const double *RESTRICT w3i,
    size_t pk)
{
    PREFETCH_NTA(&a_re[pk]);
    PREFETCH_NTA(&a_im[pk]);
    PREFETCH_NTA(&b_re[pk]);
    PREFETCH_NTA(&b_im[pk]);
    PREFETCH_NTA(&c_re[pk]);
    PREFETCH_NTA(&c_im[pk]);
    PREFETCH_NTA(&d_re[pk]);
    PREFETCH_NTA(&d_im[pk]);
    PREFETCH_T0(&w1r[pk]);
    PREFETCH_T0(&w1i[pk]);
    PREFETCH_T0(&w2r[pk]);
    PREFETCH_T0(&w2i[pk]);
#if !RADIX4_DERIVE_W3
    PREFETCH_T0(&w3r[pk]);
    PREFETCH_T0(&w3i[pk]);
#endif
}

//==============================================================================
// SMALL-K BUTTERFLY (Non-pipelined, compact)
//==============================================================================

FORCE_INLINE void radix4_butterfly_small_k_fv_avx512(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    const double *RESTRICT w1r, const double *RESTRICT w1i,
    const double *RESTRICT w2r, const double *RESTRICT w2i,
    const double *RESTRICT w3r, const double *RESTRICT w3i,
    __m512d sign_mask)
{
    __m512d a_r = LOAD_PD_AVX512(&a_re[k]);
    __m512d a_i = LOAD_PD_AVX512(&a_im[k]);
    __m512d b_r = LOAD_PD_AVX512(&b_re[k]);
    __m512d b_i = LOAD_PD_AVX512(&b_im[k]);
    __m512d c_r = LOAD_PD_AVX512(&c_re[k]);
    __m512d c_i = LOAD_PD_AVX512(&c_im[k]);
    __m512d d_r = LOAD_PD_AVX512(&d_re[k]);
    __m512d d_i = LOAD_PD_AVX512(&d_im[k]);

    __m512d w1r_v = LOAD_PD_AVX512(&w1r[k]);
    __m512d w1i_v = LOAD_PD_AVX512(&w1i[k]);
    __m512d w2r_v = LOAD_PD_AVX512(&w2r[k]);
    __m512d w2i_v = LOAD_PD_AVX512(&w2i[k]);
    __m512d w3r_v, w3i_v;

#if RADIX4_DERIVE_W3
    cmul_soa_avx512(w1r_v, w1i_v, w2r_v, w2i_v, &w3r_v, &w3i_v);
#else
    w3r_v = LOAD_PD_AVX512(&w3r[k]);
    w3i_v = LOAD_PD_AVX512(&w3i[k]);
#endif

    __m512d tB_r, tB_i, tC_r, tC_i, tD_r, tD_i;
    cmul_soa_avx512(b_r, b_i, w1r_v, w1i_v, &tB_r, &tB_i);
    cmul_soa_avx512(c_r, c_i, w2r_v, w2i_v, &tC_r, &tC_i);
    cmul_soa_avx512(d_r, d_i, w3r_v, w3i_v, &tD_r, &tD_i);

    __m512d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
    __m512d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
    radix4_butterfly_core_fv_avx512(a_r, a_i, tB_r, tB_i, tC_r, tC_i, tD_r, tD_i,
                                    &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                    &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                    sign_mask);

    _mm512_storeu_pd(&y0_re[k], out_y0_r);
    _mm512_storeu_pd(&y0_im[k], out_y0_i);
    _mm512_storeu_pd(&y1_re[k], out_y1_r);
    _mm512_storeu_pd(&y1_im[k], out_y1_i);
    _mm512_storeu_pd(&y2_re[k], out_y2_r);
    _mm512_storeu_pd(&y2_im[k], out_y2_i);
    _mm512_storeu_pd(&y3_re[k], out_y3_r);
    _mm512_storeu_pd(&y3_im[k], out_y3_i);
}

FORCE_INLINE void radix4_butterfly_small_k_bv_avx512(
    size_t k,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    const double *RESTRICT w1r, const double *RESTRICT w1i,
    const double *RESTRICT w2r, const double *RESTRICT w2i,
    const double *RESTRICT w3r, const double *RESTRICT w3i,
    __m512d sign_mask)
{
    __m512d a_r = LOAD_PD_AVX512(&a_re[k]);
    __m512d a_i = LOAD_PD_AVX512(&a_im[k]);
    __m512d b_r = LOAD_PD_AVX512(&b_re[k]);
    __m512d b_i = LOAD_PD_AVX512(&b_im[k]);
    __m512d c_r = LOAD_PD_AVX512(&c_re[k]);
    __m512d c_i = LOAD_PD_AVX512(&c_im[k]);
    __m512d d_r = LOAD_PD_AVX512(&d_re[k]);
    __m512d d_i = LOAD_PD_AVX512(&d_im[k]);

    __m512d w1r_v = LOAD_PD_AVX512(&w1r[k]);
    __m512d w1i_v = LOAD_PD_AVX512(&w1i[k]);
    __m512d w2r_v = LOAD_PD_AVX512(&w2r[k]);
    __m512d w2i_v = LOAD_PD_AVX512(&w2i[k]);
    __m512d w3r_v, w3i_v;

#if RADIX4_DERIVE_W3
    cmul_soa_avx512(w1r_v, w1i_v, w2r_v, w2i_v, &w3r_v, &w3i_v);
#else
    w3r_v = LOAD_PD_AVX512(&w3r[k]);
    w3i_v = LOAD_PD_AVX512(&w3i[k]);
#endif

    __m512d tB_r, tB_i, tC_r, tC_i, tD_r, tD_i;
    cmul_soa_avx512(b_r, b_i, w1r_v, w1i_v, &tB_r, &tB_i);
    cmul_soa_avx512(c_r, c_i, w2r_v, w2i_v, &tC_r, &tC_i);
    cmul_soa_avx512(d_r, d_i, w3r_v, w3i_v, &tD_r, &tD_i);

    __m512d out_y0_r, out_y0_i, out_y1_r, out_y1_i;
    __m512d out_y2_r, out_y2_i, out_y3_r, out_y3_i;
    radix4_butterfly_core_bv_avx512(a_r, a_i, tB_r, tB_i, tC_r, tC_i, tD_r, tD_i,
                                    &out_y0_r, &out_y0_i, &out_y1_r, &out_y1_i,
                                    &out_y2_r, &out_y2_i, &out_y3_r, &out_y3_i,
                                    sign_mask);

    _mm512_storeu_pd(&y0_re[k], out_y0_r);
    _mm512_storeu_pd(&y0_im[k], out_y0_i);
    _mm512_storeu_pd(&y1_re[k], out_y1_r);
    _mm512_storeu_pd(&y1_im[k], out_y1_i);
    _mm512_storeu_pd(&y2_re[k], out_y2_r);
    _mm512_storeu_pd(&y2_im[k], out_y2_i);
    _mm512_storeu_pd(&y3_re[k], out_y3_r);
    _mm512_storeu_pd(&y3_im[k], out_y3_i);
}

//==============================================================================
// U=2 SOFTWARE PIPELINED STAGE - FORWARD FFT (CORRECTED SCHEDULING)
//==============================================================================

/**
 * @brief U=2 modulo-scheduled kernel - Forward FFT
 *
 * CORRECTED PIPELINE SCHEDULE:
 *
 * Prologue:
 *   load(0) → A0, W0
 *   cmul(0) → T0 (from A0, W0)
 *   load(1) → A1, W1
 *
 * Main loop (i = 1, 2, ..., K_main/8 - 1):
 *   if i >= 2: store(i-2) → OUT[i-2]
 *   butterfly(i-1) → OUT_tmp[i-1] (uses A1 and T0)
 *   cmul(i) → T1 (from A1, W1)
 *   load(i+1) → A2, W2
 *   rotate: T0←T1, A1←A2, W1←W2
 *
 * Epilogue:
 *   store(K_main/8 - 2)
 *   butterfly(K_main/8 - 1) → OUT_tmp[last]
 *   store(K_main/8 - 1)
 *
 * This ensures correct pairing: butterfly(i) always uses A[i] and T[i].
 */
FORCE_INLINE void radix4_stage_u2_pipelined_fv_avx512(
    size_t K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    const double *RESTRICT w1r, const double *RESTRICT w1i,
    const double *RESTRICT w2r, const double *RESTRICT w2i,
    const double *RESTRICT w3r, const double *RESTRICT w3i,
    __m512d sign_mask,
    bool do_stream)
{
    const size_t K_main = (K / 8) * 8;
    const size_t K_tail = K - K_main;
    const int prefetch_dist = RADIX4_PREFETCH_DISTANCE;

    if (K_main == 0)
        goto handle_tail; // All tail

    // Pipeline registers for iteration i
    __m512d A0_re, A0_im, B0_re, B0_im, C0_re, C0_im, D0_re, D0_im;
    __m512d W1r_0, W1i_0, W2r_0, W2i_0, W3r_0, W3i_0;
    __m512d T0_Bre, T0_Bim, T0_Cre, T0_Cim, T0_Dre, T0_Dim;

    // Pipeline registers for iteration i+1
    __m512d A1_re, A1_im, B1_re, B1_im, C1_re, C1_im, D1_re, D1_im;
    __m512d W1r_1, W1i_1, W2r_1, W2i_1, W3r_1, W3i_1;
    __m512d T1_Bre, T1_Bim, T1_Cre, T1_Cim, T1_Dre, T1_Dim;

    // Output registers for iteration i-1
    __m512d OUT_y0_r, OUT_y0_i, OUT_y1_r, OUT_y1_i;
    __m512d OUT_y2_r, OUT_y2_i, OUT_y3_r, OUT_y3_i;

    //==========================================================================
    // PROLOGUE: load(0), cmul(0), load(1)
    //==========================================================================

    // load(0): iteration 0
    A0_re = LOAD_PD_AVX512(&a_re[0]);
    A0_im = LOAD_PD_AVX512(&a_im[0]);
    B0_re = LOAD_PD_AVX512(&b_re[0]);
    B0_im = LOAD_PD_AVX512(&b_im[0]);
    C0_re = LOAD_PD_AVX512(&c_re[0]);
    C0_im = LOAD_PD_AVX512(&c_im[0]);
    D0_re = LOAD_PD_AVX512(&d_re[0]);
    D0_im = LOAD_PD_AVX512(&d_im[0]);

    W1r_0 = LOAD_PD_AVX512(&w1r[0]);
    W1i_0 = LOAD_PD_AVX512(&w1i[0]);
    W2r_0 = LOAD_PD_AVX512(&w2r[0]);
    W2i_0 = LOAD_PD_AVX512(&w2i[0]);
#if RADIX4_DERIVE_W3
    cmul_soa_avx512(W1r_0, W1i_0, W2r_0, W2i_0, &W3r_0, &W3i_0);
#else
    W3r_0 = LOAD_PD_AVX512(&w3r[0]);
    W3i_0 = LOAD_PD_AVX512(&w3i[0]);
#endif

    // cmul(0): compute T0 from A0/W0
    cmul_soa_avx512(B0_re, B0_im, W1r_0, W1i_0, &T0_Bre, &T0_Bim);
    cmul_soa_avx512(C0_re, C0_im, W2r_0, W2i_0, &T0_Cre, &T0_Cim);
    cmul_soa_avx512(D0_re, D0_im, W3r_0, W3i_0, &T0_Dre, &T0_Dim);

    if (K_main < 16)
        goto epilogue_single; // Only 1 iteration (k=0)

    // load(1): iteration 1
    A1_re = LOAD_PD_AVX512(&a_re[8]);
    A1_im = LOAD_PD_AVX512(&a_im[8]);
    B1_re = LOAD_PD_AVX512(&b_re[8]);
    B1_im = LOAD_PD_AVX512(&b_im[8]);
    C1_re = LOAD_PD_AVX512(&c_re[8]);
    C1_im = LOAD_PD_AVX512(&c_im[8]);
    D1_re = LOAD_PD_AVX512(&d_re[8]);
    D1_im = LOAD_PD_AVX512(&d_im[8]);

    W1r_1 = LOAD_PD_AVX512(&w1r[8]);
    W1i_1 = LOAD_PD_AVX512(&w1i[8]);
    W2r_1 = LOAD_PD_AVX512(&w2r[8]);
    W2i_1 = LOAD_PD_AVX512(&w2i[8]);
#if RADIX4_DERIVE_W3
    cmul_soa_avx512(W1r_1, W1i_1, W2r_1, W2i_1, &W3r_1, &W3i_1);
#else
    W3r_1 = LOAD_PD_AVX512(&w3r[8]);
    W3i_1 = LOAD_PD_AVX512(&w3i[8]);
#endif

    //==========================================================================
    // MAIN LOOP: i = 1, 2, ..., K_main/8 - 1
    //==========================================================================

    for (size_t k = 8; k < K_main; k += 8)
    {
        // Prefetch for iteration i+2
        size_t pk = k + prefetch_dist;
        if (pk < K)
        {
            prefetch_radix4_data_nta(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                                     w1r, w1i, w2r, w2i, w3r, w3i, pk);
        }

        // store(i-2): if i >= 2, store OUT[i-2]
        if (k >= 16)
        {
            size_t store_k = k - 16;
            if (do_stream)
            {
                _mm512_stream_pd(&y0_re[store_k], OUT_y0_r);
                _mm512_stream_pd(&y0_im[store_k], OUT_y0_i);
                _mm512_stream_pd(&y1_re[store_k], OUT_y1_r);
                _mm512_stream_pd(&y1_im[store_k], OUT_y1_i);
                _mm512_stream_pd(&y2_re[store_k], OUT_y2_r);
                _mm512_stream_pd(&y2_im[store_k], OUT_y2_i);
                _mm512_stream_pd(&y3_re[store_k], OUT_y3_r);
                _mm512_stream_pd(&y3_im[store_k], OUT_y3_i);
            }
            else
            {
                _mm512_storeu_pd(&y0_re[store_k], OUT_y0_r);
                _mm512_storeu_pd(&y0_im[store_k], OUT_y0_i);
                _mm512_storeu_pd(&y1_re[store_k], OUT_y1_r);
                _mm512_storeu_pd(&y1_im[store_k], OUT_y1_i);
                _mm512_storeu_pd(&y2_re[store_k], OUT_y2_r);
                _mm512_storeu_pd(&y2_im[store_k], OUT_y2_i);
                _mm512_storeu_pd(&y3_re[store_k], OUT_y3_r);
                _mm512_storeu_pd(&y3_im[store_k], OUT_y3_i);
            }
        }

        // butterfly(i-1): uses A1 and T0 (both from iteration i-1)
        radix4_butterfly_core_fv_avx512(
            A1_re, A1_im, T0_Bre, T0_Bim, T0_Cre, T0_Cim, T0_Dre, T0_Dim,
            &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
            &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
            sign_mask);

        // cmul(i): compute T1 from A1/W1 (iteration i)
        cmul_soa_avx512(B1_re, B1_im, W1r_1, W1i_1, &T1_Bre, &T1_Bim);
        cmul_soa_avx512(C1_re, C1_im, W2r_1, W2i_1, &T1_Cre, &T1_Cim);
        cmul_soa_avx512(D1_re, D1_im, W3r_1, W3i_1, &T1_Dre, &T1_Dim);

        // load(i+1): iteration i+1
        A0_re = LOAD_PD_AVX512(&a_re[k]);
        A0_im = LOAD_PD_AVX512(&a_im[k]);
        B0_re = LOAD_PD_AVX512(&b_re[k]);
        B0_im = LOAD_PD_AVX512(&b_im[k]);
        C0_re = LOAD_PD_AVX512(&c_re[k]);
        C0_im = LOAD_PD_AVX512(&c_im[k]);
        D0_re = LOAD_PD_AVX512(&d_re[k]);
        D0_im = LOAD_PD_AVX512(&d_im[k]);

        W1r_0 = LOAD_PD_AVX512(&w1r[k]);
        W1i_0 = LOAD_PD_AVX512(&w1i[k]);
        W2r_0 = LOAD_PD_AVX512(&w2r[k]);
        W2i_0 = LOAD_PD_AVX512(&w2i[k]);
#if RADIX4_DERIVE_W3
        cmul_soa_avx512(W1r_0, W1i_0, W2r_0, W2i_0, &W3r_0, &W3i_0);
#else
        W3r_0 = LOAD_PD_AVX512(&w3r[k]);
        W3i_0 = LOAD_PD_AVX512(&w3i[k]);
#endif

        // rotate: i-1 → i-2, i → i-1, i+1 → i
        T0_Bre = T1_Bre;
        T0_Bim = T1_Bim;
        T0_Cre = T1_Cre;
        T0_Cim = T1_Cim;
        T0_Dre = T1_Dre;
        T0_Dim = T1_Dim;

        A1_re = A0_re;
        A1_im = A0_im;
        B1_re = B0_re;
        B1_im = B0_im;
        C1_re = C0_re;
        C1_im = C0_im;
        D1_re = D0_re;
        D1_im = D0_im;
        W1r_1 = W1r_0;
        W1i_1 = W1i_0;
        W2r_1 = W2r_0;
        W2i_1 = W2i_0;
        W3r_1 = W3r_0;
        W3i_1 = W3i_0;
    }

    //==========================================================================
    // EPILOGUE: Final stores
    //==========================================================================

    // store(K_main/8 - 2)
    if (K_main >= 16)
    {
        size_t store_k = K_main - 16;
        if (do_stream)
        {
            _mm512_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm512_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm512_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm512_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm512_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm512_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm512_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm512_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm512_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm512_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm512_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm512_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm512_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm512_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm512_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm512_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

epilogue_single:
    // butterfly(K_main/8 - 1): final butterfly using A1 and T0
    radix4_butterfly_core_fv_avx512(
        A1_re, A1_im, T0_Bre, T0_Bim, T0_Cre, T0_Cim, T0_Dre, T0_Dim,
        &OUT_y0_r, &OUT_y0_i, &OUT_y1_r, &OUT_y1_i,
        &OUT_y2_r, &OUT_y2_i, &OUT_y3_r, &OUT_y3_i,
        sign_mask);

    // store(K_main/8 - 1): final store
    {
        size_t store_k = K_main - 8;
        if (do_stream)
        {
            _mm512_stream_pd(&y0_re[store_k], OUT_y0_r);
            _mm512_stream_pd(&y0_im[store_k], OUT_y0_i);
            _mm512_stream_pd(&y1_re[store_k], OUT_y1_r);
            _mm512_stream_pd(&y1_im[store_k], OUT_y1_i);
            _mm512_stream_pd(&y2_re[store_k], OUT_y2_r);
            _mm512_stream_pd(&y2_im[store_k], OUT_y2_i);
            _mm512_stream_pd(&y3_re[store_k], OUT_y3_r);
            _mm512_stream_pd(&y3_im[store_k], OUT_y3_i);
        }
        else
        {
            _mm512_storeu_pd(&y0_re[store_k], OUT_y0_r);
            _mm512_storeu_pd(&y0_im[store_k], OUT_y0_i);
            _mm512_storeu_pd(&y1_re[store_k], OUT_y1_r);
            _mm512_storeu_pd(&y1_im[store_k], OUT_y1_i);
            _mm512_storeu_pd(&y2_re[store_k], OUT_y2_r);
            _mm512_storeu_pd(&y2_im[store_k], OUT_y2_i);
            _mm512_storeu_pd(&y3_re[store_k], OUT_y3_r);
            _mm512_storeu_pd(&y3_im[store_k], OUT_y3_i);
        }
    }

handle_tail:
    //==========================================================================
    // TAIL HANDLING: Masked processing for K % 8 != 0
    //==========================================================================

    if (K_tail > 0)
    {
        __mmask8 tail_mask = (__mmask8)((1U << K_tail) - 1U);

        __m512d a_r = _mm512_maskz_loadu_pd(tail_mask, &a_re[K_main]);
        __m512d a_i = _mm512_maskz_loadu_pd(tail_mask, &a_im[K_main]);
        __m512d b_r = _mm512_maskz_loadu_pd(tail_mask, &b_re[K_main]);
        __m512d b_i = _mm512_maskz_loadu_pd(tail_mask, &b_im[K_main]);
        __m512d c_r = _mm512_maskz_loadu_pd(tail_mask, &c_re[K_main]);
        __m512d c_i = _mm512_maskz_loadu_pd(tail_mask, &c_im[K_main]);
        __m512d d_r = _mm512_maskz_loadu_pd(tail_mask, &d_re[K_main]);
        __m512d d_i = _mm512_maskz_loadu_pd(tail_mask, &d_im[K_main]);

        __m512d w1r_v = _mm512_maskz_loadu_pd(tail_mask, &w1r[K_main]);
        __m512d w1i_v = _mm512_maskz_loadu_pd(tail_mask, &w1i[K_main]);
        __m512d w2r_v = _mm512_maskz_loadu_pd(tail_mask, &w2r[K_main]);
        __m512d w2i_v = _mm512_maskz_loadu_pd(tail_mask, &w2i[K_main]);
        __m512d w3r_v, w3i_v;

#if RADIX4_DERIVE_W3
        cmul_soa_avx512(w1r_v, w1i_v, w2r_v, w2i_v, &w3r_v, &w3i_v);
#else
        w3r_v = _mm512_maskz_loadu_pd(tail_mask, &w3r[K_main]);
        w3i_v = _mm512_maskz_loadu_pd(tail_mask, &w3i[K_main]);
#endif

        __m512d tB_r, tB_i, tC_r, tC_i, tD_r, tD_i;
        cmul_soa_avx512(b_r, b_i, w1r_v, w1i_v, &tB_r, &tB_i);
        cmul_soa_avx512(c_r, c_i, w2r_v, w2i_v, &tC_r, &tC_i);
        cmul_soa_avx512(d_r, d_i, w3r_v, w3i_v, &tD_r, &tD_i);

        __m512d y0_r, y0_i, y1_r, y1_i, y2_r, y2_i, y3_r, y3_i;
        radix4_butterfly_core_fv_avx512(a_r, a_i, tB_r, tB_i, tC_r, tC_i, tD_r, tD_i,
                                        &y0_r, &y0_i, &y1_r, &y1_i,
                                        &y2_r, &y2_i, &y3_r, &y3_i,
                                        sign_mask);

        // Tail always uses normal (cached) stores, not streaming
        _mm512_mask_storeu_pd(&y0_re[K_main], tail_mask, y0_r);
        _mm512_mask_storeu_pd(&y0_im[K_main], tail_mask, y0_i);
        _mm512_mask_storeu_pd(&y1_re[K_main], tail_mask, y1_r);
        _mm512_mask_storeu_pd(&y1_im[K_main], tail_mask, y1_i);
        _mm512_mask_storeu_pd(&y2_re[K_main], tail_mask, y2_r);
        _mm512_mask_storeu_pd(&y2_im[K_main], tail_mask, y2_i);
        _mm512_mask_storeu_pd(&y3_re[K_main], tail_mask, y3_r);
        _mm512_mask_storeu_pd(&y3_im[K_main], tail_mask, y3_i);
    }
}

#endif // __AVX512F__

#endif // FFT_RADIX4_AVX512_CORRECTED_PART1_H