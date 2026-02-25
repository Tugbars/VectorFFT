/**
 * @file fft_radix16_scalar_butterfly.h
 * @brief Radix-16 Scalar Butterfly — Pure C, 4×4 Cooley-Tukey DIT
 *
 * @details
 * 4×4 decomposition: DFT-16 via Cooley-Tukey with R1=R2=4
 *
 * Three stages per column:
 *   Stage 1: 4× DFT-4 on stride-4 groups → T[n2][k1]
 *   Twiddle: T[n2][k1] *= W₁₆^{n2·k1}  (9 non-trivial of 16 entries)
 *   Stage 2: 4× DFT-4 across groups → Y[k1 + 4·k2]
 *
 * The 8 DFT-4 butterflies are multiply-free (add/sub only). The W₁₆
 * intermediate twiddles have 7 trivial entries (identity or ×(-j)) and 9
 * non-trivial entries requiring ~20 FMAs total — still ~50% fewer FMAs
 * than the 8×2 decomposition (~40 FMAs for W₁₆ intermediates there).
 *
 * W₁₆ twiddle map (row=n2, col=k1, exponent=n2·k1 mod 16):
 *       k1=0   k1=1   k1=2   k1=3
 *  n2=0:  1      1      1      1     ← all trivial (row skip)
 *  n2=1:  1     W¹     W²     W³     ← 3 non-trivial
 *  n2=2:  1     W²     -j     W⁶     ← 2 non-trivial (W⁴=-j is trivial)
 *  n2=3:  1     W³     W⁶     W⁹     ← 3 non-trivial
 *
 * Unrolling: U=2 sequential — two back-to-back columns per iteration.
 * Peak 10 FP registers per radix-4 call, intermediate buffer on stack.
 *
 * TODO(benchmark): Test U=3 sequential. Three back-to-back columns would
 * give the OoO scheduler ~360 µops of lookahead (near ROB capacity on
 * ICX/SPR). May win 5-10% from better memory overlap on large K with
 * stride-K access patterns. Risk: I-cache pressure from larger loop body.
 * Requires measurement on target hardware (ICX, SPR, Zen4, Alder Lake).
 * DO NOT ASSUME U=3 > U=2 without benchmarks.
 *
 * SoA memory layout:
 *   in_re[r * K + k], in_im[r * K + k]   for r=0..15, k=0..K-1
 *
 * @version 2.0
 * @date 2025
 */

#ifndef FFT_RADIX16_SCALAR_BUTTERFLY_H
#define FFT_RADIX16_SCALAR_BUTTERFLY_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
#define R16S_INLINE static __forceinline
#define R16S_RESTRICT __restrict
#define R16S_NOINLINE static __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define R16S_INLINE static inline __attribute__((always_inline))
#define R16S_RESTRICT __restrict__
#define R16S_NOINLINE static __attribute__((noinline))
#else
#define R16S_INLINE static inline
#define R16S_RESTRICT
#define R16S_NOINLINE static
#endif

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define R16S_IVDEP _Pragma("ivdep")
#elif defined(__GNUC__)
#define R16S_IVDEP _Pragma("GCC ivdep")
#else
#define R16S_IVDEP
#endif

/* ============================================================================
 * W₁₆ TWIDDLE CONSTANTS
 * ========================================================================= */

#define R16S_COS_PI_8 0.92387953251128675613  /* cos(π/8)  */
#define R16S_SIN_PI_8 0.38268343236508977173  /* sin(π/8)  */
#define R16S_COS_3PI_8 0.38268343236508977173 /* cos(3π/8) = sin(π/8) */
#define R16S_SIN_3PI_8 0.92387953251128675613 /* sin(3π/8) = cos(π/8) */
#define R16S_SQRT2_2 0.70710678118654752440   /* √2/2 */

/* ============================================================================
 * CORE: SCALAR RADIX-4 BUTTERFLY
 *
 * Forward (sign = -1):                    Backward (sign = +1):
 *   y0 = (a+c) + (b+d)                     y0 = (a+c) + (b+d)
 *   y1 = (a-c) - j(b-d)                    y1 = (a-c) + j(b-d)
 *   y2 = (a+c) - (b+d)                     y2 = (a+c) - (b+d)
 *   y3 = (a-c) + j(b-d)                    y3 = (a-c) - j(b-d)
 *
 * 12 add/sub per call, 0 multiplies. Peak live: 10 registers.
 * ========================================================================= */

R16S_INLINE void r16s_radix4_fwd(
    double a_re, double a_im, double b_re, double b_im,
    double c_re, double c_im, double d_re, double d_im,
    double *R16S_RESTRICT y0_re, double *R16S_RESTRICT y0_im,
    double *R16S_RESTRICT y1_re, double *R16S_RESTRICT y1_im,
    double *R16S_RESTRICT y2_re, double *R16S_RESTRICT y2_im,
    double *R16S_RESTRICT y3_re, double *R16S_RESTRICT y3_im)
{
    double sAC_re = a_re + c_re, dAC_re = a_re - c_re;
    double sAC_im = a_im + c_im, dAC_im = a_im - c_im;
    double sBD_re = b_re + d_re, dBD_re = b_re - d_re;
    double sBD_im = b_im + d_im, dBD_im = b_im - d_im;

    *y0_re = sAC_re + sBD_re;
    *y0_im = sAC_im + sBD_im;
    *y2_re = sAC_re - sBD_re;
    *y2_im = sAC_im - sBD_im;
    /* Forward: -j*(b-d) → re += dBD_im, im -= dBD_re */
    *y1_re = dAC_re + dBD_im;
    *y1_im = dAC_im - dBD_re;
    *y3_re = dAC_re - dBD_im;
    *y3_im = dAC_im + dBD_re;
}

R16S_INLINE void r16s_radix4_bwd(
    double a_re, double a_im, double b_re, double b_im,
    double c_re, double c_im, double d_re, double d_im,
    double *R16S_RESTRICT y0_re, double *R16S_RESTRICT y0_im,
    double *R16S_RESTRICT y1_re, double *R16S_RESTRICT y1_im,
    double *R16S_RESTRICT y2_re, double *R16S_RESTRICT y2_im,
    double *R16S_RESTRICT y3_re, double *R16S_RESTRICT y3_im)
{
    double sAC_re = a_re + c_re, dAC_re = a_re - c_re;
    double sAC_im = a_im + c_im, dAC_im = a_im - c_im;
    double sBD_re = b_re + d_re, dBD_re = b_re - d_re;
    double sBD_im = b_im + d_im, dBD_im = b_im - d_im;

    *y0_re = sAC_re + sBD_re;
    *y0_im = sAC_im + sBD_im;
    *y2_re = sAC_re - sBD_re;
    *y2_im = sAC_im - sBD_im;
    /* Backward: +j*(b-d) → re -= dBD_im, im += dBD_re */
    *y1_re = dAC_re - dBD_im;
    *y1_im = dAC_im + dBD_re;
    *y3_re = dAC_re + dBD_im;
    *y3_im = dAC_im - dBD_re;
}

/* ============================================================================
 * W₁₆ TWIDDLE APPLICATION — INLINE FOR 4×4 INTERMEDIATE
 *
 * Forward: T[n2][k1] *= W₁₆^{n2·k1} where W₁₆ = e^{-2πi/16}
 * Backward: T[n2][k1] *= conj(W₁₆^{n2·k1})
 *
 * Complex multiply by W = wr - j·wi (forward):
 *   re' = a·wr + b·wi,  im' = b·wr - a·wi
 *
 * Complex multiply by conj(W) = wr + j·wi (backward):
 *   re' = a·wr - b·wi,  im' = b·wr + a·wi
 *
 * Special optimizations:
 *   W² = √2/2·(1-j): re' = √2/2·(a+b), im' = √2/2·(b-a)    [1 mul, 2 add]
 *   W⁴ = -j:         re' = b, im' = -a                        [0 mul]
 *   W⁶ = -√2/2·(1+j): re' = √2/2·(-a+b), im' = √2/2·(-a-b)  [1 mul, 2 add]
 * ========================================================================= */

R16S_INLINE void r16s_twiddle_fwd(double T_re[4][4], double T_im[4][4])
{
    double a, b;

    /* n2=0: all identity — skip */

    /* ---- n2=1: W^1, W^2, W^3 ---- */

    /* T[1][1] *= W^1 = cos(π/8) - j·sin(π/8) */
    a = T_re[1][1];
    b = T_im[1][1];
    T_re[1][1] = a * R16S_COS_PI_8 + b * R16S_SIN_PI_8;
    T_im[1][1] = b * R16S_COS_PI_8 - a * R16S_SIN_PI_8;

    /* T[1][2] *= W^2 = √2/2·(1 - j) */
    a = T_re[1][2];
    b = T_im[1][2];
    T_re[1][2] = R16S_SQRT2_2 * (a + b);
    T_im[1][2] = R16S_SQRT2_2 * (b - a);

    /* T[1][3] *= W^3 = cos(3π/8) - j·sin(3π/8) */
    a = T_re[1][3];
    b = T_im[1][3];
    T_re[1][3] = a * R16S_COS_3PI_8 + b * R16S_SIN_3PI_8;
    T_im[1][3] = b * R16S_COS_3PI_8 - a * R16S_SIN_3PI_8;

    /* ---- n2=2: W^2, W^4=-j, W^6 ---- */

    /* T[2][1] *= W^2 */
    a = T_re[2][1];
    b = T_im[2][1];
    T_re[2][1] = R16S_SQRT2_2 * (a + b);
    T_im[2][1] = R16S_SQRT2_2 * (b - a);

    /* T[2][2] *= W^4 = -j: (re,im) → (im, -re) */
    a = T_re[2][2];
    b = T_im[2][2];
    T_re[2][2] = b;
    T_im[2][2] = -a;

    /* T[2][3] *= W^6 = -√2/2·(1 + j) */
    a = T_re[2][3];
    b = T_im[2][3];
    T_re[2][3] = R16S_SQRT2_2 * (-a + b);
    T_im[2][3] = R16S_SQRT2_2 * (-a - b);

    /* ---- n2=3: W^3, W^6, W^9 ---- */

    /* T[3][1] *= W^3 */
    a = T_re[3][1];
    b = T_im[3][1];
    T_re[3][1] = a * R16S_COS_3PI_8 + b * R16S_SIN_3PI_8;
    T_im[3][1] = b * R16S_COS_3PI_8 - a * R16S_SIN_3PI_8;

    /* T[3][2] *= W^6 */
    a = T_re[3][2];
    b = T_im[3][2];
    T_re[3][2] = R16S_SQRT2_2 * (-a + b);
    T_im[3][2] = R16S_SQRT2_2 * (-a - b);

    /* T[3][3] *= W^9: W₁₆^9 = e^{-2πi·9/16} = cos(9π/8) - j·sin(9π/8)
       cos(9π/8) = -cos(π/8), sin(9π/8) = -sin(π/8)
       So W^9 = -cos(π/8) + j·sin(π/8)   ...as a complex: wr=-cos(π/8), wi=-sin(π/8)
       Wait: W = e^{-iθ} = cosθ - j·sinθ where θ=2π·9/16 = 9π/8
       cos(9π/8) = -cos(π/8), sin(9π/8) = -sin(π/8)
       So W^9 = -cos(π/8) - j·(-sin(π/8)) = -cos(π/8) + j·sin(π/8)
       Complex mul by (wr + j·wi) where wr=-cos(π/8), wi=sin(π/8):
         re' = a·wr - b·wi = -a·cos(π/8) - b·sin(π/8)
         im' = a·wi + b·wr =  a·sin(π/8) - b·cos(π/8)  */
    a = T_re[3][3];
    b = T_im[3][3];
    T_re[3][3] = -a * R16S_COS_PI_8 - b * R16S_SIN_PI_8;
    T_im[3][3] = a * R16S_SIN_PI_8 - b * R16S_COS_PI_8;
}

R16S_INLINE void r16s_twiddle_bwd(double T_re[4][4], double T_im[4][4])
{
    double a, b;

    /* Backward: multiply by conj(W₁₆^{n2·k1})
     * conj(wr + j·wi) = wr - j·wi
     * Complex mul by (wr - j·wi): re' = a·wr + b·wi, im' = b·wr - a·wi
     * This is the same formula as forward twiddle with negated exponent.
     */

    /* ---- n2=1: conj(W^1), conj(W^2), conj(W^3) ---- */

    /* T[1][1] *= conj(W^1) = cos(π/8) + j·sin(π/8) */
    a = T_re[1][1];
    b = T_im[1][1];
    T_re[1][1] = a * R16S_COS_PI_8 - b * R16S_SIN_PI_8;
    T_im[1][1] = b * R16S_COS_PI_8 + a * R16S_SIN_PI_8;

    /* T[1][2] *= conj(W^2) = √2/2·(1 + j): re'=√2/2·(a-b), im'=√2/2·(a+b) */
    a = T_re[1][2];
    b = T_im[1][2];
    T_re[1][2] = R16S_SQRT2_2 * (a - b);
    T_im[1][2] = R16S_SQRT2_2 * (a + b);

    /* T[1][3] *= conj(W^3) = cos(3π/8) + j·sin(3π/8) */
    a = T_re[1][3];
    b = T_im[1][3];
    T_re[1][3] = a * R16S_COS_3PI_8 - b * R16S_SIN_3PI_8;
    T_im[1][3] = b * R16S_COS_3PI_8 + a * R16S_SIN_3PI_8;

    /* ---- n2=2: conj(W^2), conj(W^4)=j, conj(W^6) ---- */

    /* T[2][1] *= conj(W^2) */
    a = T_re[2][1];
    b = T_im[2][1];
    T_re[2][1] = R16S_SQRT2_2 * (a - b);
    T_im[2][1] = R16S_SQRT2_2 * (a + b);

    /* T[2][2] *= conj(W^4) = conj(-j) = j: (re,im) → (-im, re) */
    a = T_re[2][2];
    b = T_im[2][2];
    T_re[2][2] = -b;
    T_im[2][2] = a;

    /* T[2][3] *= conj(W^6) = conj(-√2/2·(1+j)) = -√2/2·(1-j)
       = -√2/2 + j·√2/2
       re' = a·(-√2/2) - b·(√2/2) = -√2/2·(a+b)
       im' = a·(√2/2) + b·(-√2/2) = √2/2·(a-b) */
    a = T_re[2][3];
    b = T_im[2][3];
    T_re[2][3] = R16S_SQRT2_2 * (-a - b);
    T_im[2][3] = R16S_SQRT2_2 * (a - b);

    /* ---- n2=3: conj(W^3), conj(W^6), conj(W^9) ---- */

    /* T[3][1] *= conj(W^3) */
    a = T_re[3][1];
    b = T_im[3][1];
    T_re[3][1] = a * R16S_COS_3PI_8 - b * R16S_SIN_3PI_8;
    T_im[3][1] = b * R16S_COS_3PI_8 + a * R16S_SIN_3PI_8;

    /* T[3][2] *= conj(W^6) */
    a = T_re[3][2];
    b = T_im[3][2];
    T_re[3][2] = R16S_SQRT2_2 * (-a - b);
    T_im[3][2] = R16S_SQRT2_2 * (a - b);

    /* T[3][3] *= conj(W^9):
       W^9 = -cos(π/8) + j·sin(π/8), so conj(W^9) = -cos(π/8) - j·sin(π/8)
       re' = a·(-cos(π/8)) - b·(-sin(π/8)) = -a·cos(π/8) + b·sin(π/8)
       im' = a·(-sin(π/8)) + b·(-cos(π/8)) = -a·sin(π/8) - b·cos(π/8) */
    a = T_re[3][3];
    b = T_im[3][3];
    T_re[3][3] = -a * R16S_COS_PI_8 + b * R16S_SIN_PI_8;
    T_im[3][3] = -a * R16S_SIN_PI_8 - b * R16S_COS_PI_8;
}

/* ============================================================================
 * COMPLETE RADIX-16 BUTTERFLY — ONE COLUMN
 *
 * Stage 1: 4 DFT-4s on stride-4 groups → T[n2][k1]  (4 groups, n2=0..3)
 * Twiddle: T[n2][k1] *= W₁₆^{n2·k1}
 * Stage 2: 4 DFT-4s across groups → Y[k1 + 4·k2]    (4 output groups, k1=0..3)
 *
 * The 4×4 intermediate buffer T lives on the stack (32 doubles). Each
 * radix-4 call peaks at 10 FP registers. The buffer provides the cross-group
 * data flow that AVX2 handles via SIMD lane shuffles (implicit transpose).
 * ========================================================================= */

R16S_INLINE void r16s_butterfly_fwd_1col(
    const double *R16S_RESTRICT const *R16S_RESTRICT row_re,
    const double *R16S_RESTRICT const *R16S_RESTRICT row_im,
    double *R16S_RESTRICT const *R16S_RESTRICT dst_re,
    double *R16S_RESTRICT const *R16S_RESTRICT dst_im,
    size_t k)
{
    double T_re[4][4], T_im[4][4];

    /* ---- Stage 1: 4× DFT-4 on stride-4 groups ---- */
    /* Group n2: inputs at rows {n2, n2+4, n2+8, n2+12} */

    r16s_radix4_fwd(
        row_re[0][k], row_im[0][k], row_re[4][k], row_im[4][k],
        row_re[8][k], row_im[8][k], row_re[12][k], row_im[12][k],
        &T_re[0][0], &T_im[0][0], &T_re[0][1], &T_im[0][1],
        &T_re[0][2], &T_im[0][2], &T_re[0][3], &T_im[0][3]);

    r16s_radix4_fwd(
        row_re[1][k], row_im[1][k], row_re[5][k], row_im[5][k],
        row_re[9][k], row_im[9][k], row_re[13][k], row_im[13][k],
        &T_re[1][0], &T_im[1][0], &T_re[1][1], &T_im[1][1],
        &T_re[1][2], &T_im[1][2], &T_re[1][3], &T_im[1][3]);

    r16s_radix4_fwd(
        row_re[2][k], row_im[2][k], row_re[6][k], row_im[6][k],
        row_re[10][k], row_im[10][k], row_re[14][k], row_im[14][k],
        &T_re[2][0], &T_im[2][0], &T_re[2][1], &T_im[2][1],
        &T_re[2][2], &T_im[2][2], &T_re[2][3], &T_im[2][3]);

    r16s_radix4_fwd(
        row_re[3][k], row_im[3][k], row_re[7][k], row_im[7][k],
        row_re[11][k], row_im[11][k], row_re[15][k], row_im[15][k],
        &T_re[3][0], &T_im[3][0], &T_re[3][1], &T_im[3][1],
        &T_re[3][2], &T_im[3][2], &T_re[3][3], &T_im[3][3]);

    /* ---- Twiddle: T[n2][k1] *= W₁₆^{n2·k1} ---- */
    r16s_twiddle_fwd(T_re, T_im);

    /* ---- Stage 2: 4× DFT-4 across groups ---- */
    /* For each k1: DFT-4 of {T[0][k1], T[1][k1], T[2][k1], T[3][k1]} */
    /* Output: Y[k1 + 4·k2] for k2=0..3 */
    {
        double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

        /* k1=0 → outputs at m = {0, 4, 8, 12} */
        r16s_radix4_fwd(
            T_re[0][0], T_im[0][0], T_re[1][0], T_im[1][0],
            T_re[2][0], T_im[2][0], T_re[3][0], T_im[3][0],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[0][k] = y0r;
        dst_im[0][k] = y0i;
        dst_re[4][k] = y1r;
        dst_im[4][k] = y1i;
        dst_re[8][k] = y2r;
        dst_im[8][k] = y2i;
        dst_re[12][k] = y3r;
        dst_im[12][k] = y3i;

        /* k1=1 → outputs at m = {1, 5, 9, 13} */
        r16s_radix4_fwd(
            T_re[0][1], T_im[0][1], T_re[1][1], T_im[1][1],
            T_re[2][1], T_im[2][1], T_re[3][1], T_im[3][1],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[1][k] = y0r;
        dst_im[1][k] = y0i;
        dst_re[5][k] = y1r;
        dst_im[5][k] = y1i;
        dst_re[9][k] = y2r;
        dst_im[9][k] = y2i;
        dst_re[13][k] = y3r;
        dst_im[13][k] = y3i;

        /* k1=2 → outputs at m = {2, 6, 10, 14} */
        r16s_radix4_fwd(
            T_re[0][2], T_im[0][2], T_re[1][2], T_im[1][2],
            T_re[2][2], T_im[2][2], T_re[3][2], T_im[3][2],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[2][k] = y0r;
        dst_im[2][k] = y0i;
        dst_re[6][k] = y1r;
        dst_im[6][k] = y1i;
        dst_re[10][k] = y2r;
        dst_im[10][k] = y2i;
        dst_re[14][k] = y3r;
        dst_im[14][k] = y3i;

        /* k1=3 → outputs at m = {3, 7, 11, 15} */
        r16s_radix4_fwd(
            T_re[0][3], T_im[0][3], T_re[1][3], T_im[1][3],
            T_re[2][3], T_im[2][3], T_re[3][3], T_im[3][3],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[3][k] = y0r;
        dst_im[3][k] = y0i;
        dst_re[7][k] = y1r;
        dst_im[7][k] = y1i;
        dst_re[11][k] = y2r;
        dst_im[11][k] = y2i;
        dst_re[15][k] = y3r;
        dst_im[15][k] = y3i;
    }
}

R16S_INLINE void r16s_butterfly_bwd_1col(
    const double *R16S_RESTRICT const *R16S_RESTRICT row_re,
    const double *R16S_RESTRICT const *R16S_RESTRICT row_im,
    double *R16S_RESTRICT const *R16S_RESTRICT dst_re,
    double *R16S_RESTRICT const *R16S_RESTRICT dst_im,
    size_t k)
{
    double T_re[4][4], T_im[4][4];

    /* ---- Stage 1: 4× DFT-4 (backward) on stride-4 groups ---- */

    r16s_radix4_bwd(
        row_re[0][k], row_im[0][k], row_re[4][k], row_im[4][k],
        row_re[8][k], row_im[8][k], row_re[12][k], row_im[12][k],
        &T_re[0][0], &T_im[0][0], &T_re[0][1], &T_im[0][1],
        &T_re[0][2], &T_im[0][2], &T_re[0][3], &T_im[0][3]);

    r16s_radix4_bwd(
        row_re[1][k], row_im[1][k], row_re[5][k], row_im[5][k],
        row_re[9][k], row_im[9][k], row_re[13][k], row_im[13][k],
        &T_re[1][0], &T_im[1][0], &T_re[1][1], &T_im[1][1],
        &T_re[1][2], &T_im[1][2], &T_re[1][3], &T_im[1][3]);

    r16s_radix4_bwd(
        row_re[2][k], row_im[2][k], row_re[6][k], row_im[6][k],
        row_re[10][k], row_im[10][k], row_re[14][k], row_im[14][k],
        &T_re[2][0], &T_im[2][0], &T_re[2][1], &T_im[2][1],
        &T_re[2][2], &T_im[2][2], &T_re[2][3], &T_im[2][3]);

    r16s_radix4_bwd(
        row_re[3][k], row_im[3][k], row_re[7][k], row_im[7][k],
        row_re[11][k], row_im[11][k], row_re[15][k], row_im[15][k],
        &T_re[3][0], &T_im[3][0], &T_re[3][1], &T_im[3][1],
        &T_re[3][2], &T_im[3][2], &T_re[3][3], &T_im[3][3]);

    /* ---- Twiddle: conjugate W₁₆ factors ---- */
    r16s_twiddle_bwd(T_re, T_im);

    /* ---- Stage 2: 4× DFT-4 (backward) across groups ---- */
    {
        double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

        r16s_radix4_bwd(
            T_re[0][0], T_im[0][0], T_re[1][0], T_im[1][0],
            T_re[2][0], T_im[2][0], T_re[3][0], T_im[3][0],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[0][k] = y0r;
        dst_im[0][k] = y0i;
        dst_re[4][k] = y1r;
        dst_im[4][k] = y1i;
        dst_re[8][k] = y2r;
        dst_im[8][k] = y2i;
        dst_re[12][k] = y3r;
        dst_im[12][k] = y3i;

        r16s_radix4_bwd(
            T_re[0][1], T_im[0][1], T_re[1][1], T_im[1][1],
            T_re[2][1], T_im[2][1], T_re[3][1], T_im[3][1],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[1][k] = y0r;
        dst_im[1][k] = y0i;
        dst_re[5][k] = y1r;
        dst_im[5][k] = y1i;
        dst_re[9][k] = y2r;
        dst_im[9][k] = y2i;
        dst_re[13][k] = y3r;
        dst_im[13][k] = y3i;

        r16s_radix4_bwd(
            T_re[0][2], T_im[0][2], T_re[1][2], T_im[1][2],
            T_re[2][2], T_im[2][2], T_re[3][2], T_im[3][2],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[2][k] = y0r;
        dst_im[2][k] = y0i;
        dst_re[6][k] = y1r;
        dst_im[6][k] = y1i;
        dst_re[10][k] = y2r;
        dst_im[10][k] = y2i;
        dst_re[14][k] = y3r;
        dst_im[14][k] = y3i;

        r16s_radix4_bwd(
            T_re[0][3], T_im[0][3], T_re[1][3], T_im[1][3],
            T_re[2][3], T_im[2][3], T_re[3][3], T_im[3][3],
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        dst_re[3][k] = y0r;
        dst_im[3][k] = y0i;
        dst_re[7][k] = y1r;
        dst_im[7][k] = y1i;
        dst_re[11][k] = y2r;
        dst_im[11][k] = y2i;
        dst_re[15][k] = y3r;
        dst_im[15][k] = y3i;
    }
}

/* ============================================================================
 * PUBLIC API
 *
 * Row pointer hoisting: row_re[r] = &in_re[r * K] once outside k-loop.
 * U=2 sequential: two back-to-back butterfly_1col calls per k-iteration.
 * The OoO engine overlaps column B's loads with column A's computation.
 * ========================================================================= */

R16S_NOINLINE void radix16_butterfly_forward_scalar(
    size_t K,
    const double *R16S_RESTRICT in_re,
    const double *R16S_RESTRICT in_im,
    double *R16S_RESTRICT out_re,
    double *R16S_RESTRICT out_im)
{
    assert(K >= 1);

    const double *row_re[16];
    const double *row_im[16];
    double *dst_re[16];
    double *dst_im[16];

    for (int r = 0; r < 16; r++)
    {
        row_re[r] = &in_re[r * K];
        row_im[r] = &in_im[r * K];
        dst_re[r] = &out_re[r * K];
        dst_im[r] = &out_im[r * K];
    }

    size_t k = 0;
    R16S_IVDEP
    for (; k + 2 <= K; k += 2)
    {
        r16s_butterfly_fwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k);

        r16s_butterfly_fwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k + 1);
    }
    for (; k < K; k++)
    {
        r16s_butterfly_fwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k);
    }
}

R16S_NOINLINE void radix16_butterfly_backward_scalar(
    size_t K,
    const double *R16S_RESTRICT in_re,
    const double *R16S_RESTRICT in_im,
    double *R16S_RESTRICT out_re,
    double *R16S_RESTRICT out_im)
{
    assert(K >= 1);

    const double *row_re[16];
    const double *row_im[16];
    double *dst_re[16];
    double *dst_im[16];

    for (int r = 0; r < 16; r++)
    {
        row_re[r] = &in_re[r * K];
        row_im[r] = &in_im[r * K];
        dst_re[r] = &out_re[r * K];
        dst_im[r] = &out_im[r * K];
    }

    size_t k = 0;
    R16S_IVDEP
    for (; k + 2 <= K; k += 2)
    {
        r16s_butterfly_bwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k);

        r16s_butterfly_bwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k + 1);
    }
    for (; k < K; k++)
    {
        r16s_butterfly_bwd_1col(
            (const double *R16S_RESTRICT const *)row_re,
            (const double *R16S_RESTRICT const *)row_im,
            (double *R16S_RESTRICT const *)dst_re,
            (double *R16S_RESTRICT const *)dst_im, k);
    }
}

#endif /* FFT_RADIX16_SCALAR_BUTTERFLY_H */

/*
 * ============================================================================
 * DESIGN NOTES (v2.0)
 * ============================================================================
 *
 * Decomposition: 4×4 Cooley-Tukey DIT
 *   n = 4·n1 + n2,  m = k1 + 4·k2
 *   Stage 1: DFT-4 over n1 for each n2 (4 independent butterflies)
 *   Twiddle: T[n2][k1] *= W₁₆^{n2·k1}
 *   Stage 2: DFT-4 over n2 for each k1 (4 independent butterflies)
 *
 * Intermediate twiddle cost:
 *   7 trivial entries (identity, ×(-j)): 0 multiplies
 *   4 entries using √2/2 optimization: ~8 FMAs (W², W⁶)
 *   5 entries using full complex multiply: ~15 FMAs (W¹, W³, W⁹)
 *   Total: ~20 FMAs for intermediates
 *   vs 8×2: ~40 FMAs. 4×4 saves ~20 FMAs.
 *
 * Register budget:
 *   Each radix-4: peak 10 FP registers, 6 free.
 *   Intermediate T[4][4] (32 doubles): stack-allocated, not in registers.
 *
 * Errata (v1.0 → v2.0):
 *   v1.0 applied stage-2 DFT-4 within groups, when Cooley-Tukey requires
 *   stage-2 across groups. v1.0 also used W₄ twiddles instead of W₁₆.
 *   These errors stemmed from conflating the AVX2 lane-shuffle approach
 *   (which handles the cross-group transpose implicitly via SIMD) with
 *   the scalar data flow (which requires an explicit intermediate buffer).
 *
 * ============================================================================
 */