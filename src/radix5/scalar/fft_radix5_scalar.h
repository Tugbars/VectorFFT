/**
 * @file fft_radix5_scalar_optimized.h
 * @brief Radix-5 FFT - Pure Scalar C Implementation (OPTIMIZED)
 *
 * @details
 * PURE SCALAR VERSION - ALL ALGORITHMIC OPTIMIZATIONS PRESERVED:
 * ==============================================================
 * ✅ Blocked twiddle layout (contiguous W1, W2, [W3], [W4])
 * ✅ Compile-time W3/W4 derivation option (saves 50% twiddle bandwidth)
 * ✅ Const-qualified twiddle pointers for compiler optimization
 * ✅ Optimized W4 = W2² computation (saves 1 multiply)
 * ✅ Force-inline butterfly core
 * ✅ Exact arithmetic from vectorized versions
 * ✅ No streaming stores (pure scalar writes)
 * ✅ Prefetching disabled (compiler decides on scalar code)
 *
 * SCALAR SPECIFIC PROPERTIES:
 * ===========================
 * • No SIMD, no alignment requirements
 * • Pure C99 compatible
 * • Loop over K elements sequentially
 * • Serves as reference implementation
 * • Useful for remainder loops and validation
 * • Good starting point for auto-vectorization experiments
 *
 * EXPECTED PERFORMANCE: ~1/8 of AVX-512, but portable to any platform
 *
 * @author FFT Optimization Team + Scalar Port (2025)
 * @version 7.0 (Scalar reference implementation)
 * @date 2025
 */

#ifndef FFT_RADIX5_SCALAR_OPTIMIZED_H
#define FFT_RADIX5_SCALAR_OPTIMIZED_H

#include <stdint.h>
#include <stdbool.h>

//==============================================================================
// COMPILER ABSTRACTIONS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#else
#define FORCE_INLINE static inline
#define RESTRICT
#endif

//==============================================================================
// CONFIGURATION KNOBS
//==============================================================================

/**
 * @def RADIX5_DERIVE_W3W4
 * @brief Compute w3/w4 from w1/w2 instead of loading (saves bandwidth)
 */
#ifndef RADIX5_DERIVE_W3W4
#define RADIX5_DERIVE_W3W4 1
#endif

//==============================================================================
// RADIX-5 GEOMETRIC CONSTANTS
//==============================================================================

#define C5_1 0.30901699437494742410    // cos(2π/5)
#define C5_2 (-0.80901699437494742410) // cos(4π/5)
#define S5_1 0.95105651629515357212    // sin(2π/5)
#define S5_2 0.58778525229247312917    // sin(4π/5)

//==============================================================================
// TWIDDLE STRUCTURE (Blocked Layout)
//==============================================================================

typedef struct
{
    const double *RESTRICT re; // Real parts: [w1_re | w2_re | w3_re | w4_re]
    const double *RESTRICT im; // Imag parts: [w1_im | w2_im | w3_im | w4_im]
} radix5_twiddles_t;

//==============================================================================
// FORCE-INLINE HELPER: Complex Multiply
//==============================================================================

/**
 * @brief Complex multiply: (ar + i*ai) * (wr + i*wi) → (tr + i*ti)
 */
FORCE_INLINE void cmul_scalar(
    double ar, double ai,
    double wr, double wi,
    double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = ar * wr - ai * wi;
    *ti = ar * wi + ai * wr;
}

/**
 * @brief Optimized W*W (square a complex number): (wr + i*wi)² → (tr + i*ti)
 * @details Uses W4_re = wr²-wi², W4_im = 2*wr*wi (one fewer multiply)
 */
FORCE_INLINE void csquare_scalar(
    double wr, double wi,
    double *RESTRICT tr, double *RESTRICT ti)
{
    double wr2 = wr * wr;
    double wi2 = wi * wi;
    *tr = wr2 - wi2;
    *ti = 2.0 * wr * wi;
}

//==============================================================================
// FORCE-INLINE CORE: Radix-5 Butterfly (Forward)
//==============================================================================

/**
 * @brief Radix-5 butterfly core - Forward transform (Scalar)
 * @details Takes TWIDDLED inputs (a, tb, tc, td, te) and produces 5 outputs
 *          a is UN-twiddled (first element), b-e are ALREADY multiplied by twiddles
 *
 * ✅ PRESERVED: Exact arithmetic from vectorized versions
 */
FORCE_INLINE void radix5_butterfly_core_fv_scalar(
    double a_re, double a_im,
    double tb_re, double tb_im,
    double tc_re, double tc_im,
    double td_re, double td_im,
    double te_re, double te_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    // Stage 1: Sum and difference butterflies
    double s1_re = tb_re + te_re;
    double s1_im = tb_im + te_im;
    double d1_re = tb_re - te_re;
    double d1_im = tb_im - te_im;

    double s2_re = tc_re + td_re;
    double s2_im = tc_im + td_im;
    double d2_re = tc_re - td_re;
    double d2_im = tc_im - td_im;

    // Y[0] = A + s1 + s2
    *y0_re = a_re + s1_re + s2_re;
    *y0_im = a_im + s1_im + s2_im;

    // Stage 2: Weighted sums for real parts
    double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;
    double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;

    double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;
    double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;

    // Stage 3: Weighted differences for imaginary rotation
    double base1_re = S5_1 * d1_re + S5_2 * d2_re;
    double base1_im = S5_1 * d1_im + S5_2 * d2_im;

    double base2_re = S5_2 * d1_re - S5_1 * d2_re;
    double base2_im = S5_2 * d1_im - S5_1 * d2_im;

    // Stage 4: Multiply by -i (rotate by -90°)
    // -i * z = -i*(x + iy) = y - ix, so: Re(-i*z) = Im(z), Im(-i*z) = -Re(z)
    double u1_re = base1_im;
    double u1_im = -base1_re;
    double u2_re = base2_im;
    double u2_im = -base2_re;

    // Stage 5: Final outputs
    *y1_re = t1_re + u1_re;
    *y1_im = t1_im + u1_im;
    *y4_re = t1_re - u1_re;
    *y4_im = t1_im - u1_im;
    *y2_re = t2_re + u2_re;
    *y2_im = t2_im + u2_im;
    *y3_re = t2_re - u2_re;
    *y3_im = t2_im - u2_im;
}

//==============================================================================
// FORCE-INLINE CORE: Radix-5 Butterfly (Backward)
//==============================================================================

/**
 * @brief Radix-5 butterfly core - Backward transform (Scalar)
 * @details Identical to forward but with negated twiddles (conjugate)
 */
FORCE_INLINE void radix5_butterfly_core_bv_scalar(
    double a_re, double a_im,
    double tb_re, double tb_im,
    double tc_re, double tc_im,
    double td_re, double td_im,
    double te_re, double te_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    // Stage 1: Sum and difference butterflies (same as forward)
    double s1_re = tb_re + te_re;
    double s1_im = tb_im + te_im;
    double d1_re = tb_re - te_re;
    double d1_im = tb_im - te_im;

    double s2_re = tc_re + td_re;
    double s2_im = tc_im + td_im;
    double d2_re = tc_re - td_re;
    double d2_im = tc_im - td_im;

    *y0_re = a_re + s1_re + s2_re;
    *y0_im = a_im + s1_im + s2_im;

    // Stage 2: Weighted sums (same as forward)
    double t1_re = a_re + C5_1 * s1_re + C5_2 * s2_re;
    double t1_im = a_im + C5_1 * s1_im + C5_2 * s2_im;

    double t2_re = a_re + C5_2 * s1_re + C5_1 * s2_re;
    double t2_im = a_im + C5_2 * s1_im + C5_1 * s2_im;

    // Stage 3: Weighted differences (NEGATED for backward)
    double base1_re = -(S5_1 * d1_re + S5_2 * d2_re);
    double base1_im = -(S5_1 * d1_im + S5_2 * d2_im);

    double base2_re = S5_1 * d2_re - S5_2 * d1_re;
    double base2_im = S5_1 * d2_im - S5_2 * d1_im;

    // Stage 4: Multiply by -i (same rotation)
    double u1_re = base1_im;
    double u1_im = -base1_re;
    double u2_re = base2_im;
    double u2_im = -base2_re;

    // Stage 5: Final outputs (same as forward)
    *y1_re = t1_re + u1_re;
    *y1_im = t1_im + u1_im;
    *y4_re = t1_re - u1_re;
    *y4_im = t1_im - u1_im;
    *y2_re = t2_re + u2_re;
    *y2_im = t2_im + u2_im;
    *y3_re = t2_re - u2_re;
    *y3_im = t2_im - u2_im;
}

//==============================================================================
// MAIN KERNEL: Forward Transform (Scalar)
//==============================================================================

/**
 * @brief Forward FFT - Pure scalar implementation
 * @details Simple loop over K elements, no vectorization or pipelining
 *
 * ✅ PRESERVED: Exact algorithm and twiddle layout from SIMD versions
 * ✅ OPTIMIZED: W4 = W2² using csquare (saves 1 multiply per element)
 */
void fft_radix5_fv_scalar(
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    // Twiddle layout: blocked and SoA
    const double *w1_re = tw->re;
    const double *w1_im = tw->im;
    const double *w2_re = tw->re + K;
    const double *w2_im = tw->im + K;
#if !RADIX5_DERIVE_W3W4
    const double *w3_re = tw->re + 2 * K;
    const double *w3_im = tw->im + 2 * K;
    const double *w4_re = tw->re + 3 * K;
    const double *w4_im = tw->im + 3 * K;
#endif

    // Loop over all K elements
    for (int k = 0; k < K; k++)
    {
        // Load inputs
        double a_re_k = a_re[k];
        double a_im_k = a_im[k];
        double b_re_k = b_re[k];
        double b_im_k = b_im[k];
        double c_re_k = c_re[k];
        double c_im_k = c_im[k];
        double d_re_k = d_re[k];
        double d_im_k = d_im[k];
        double e_re_k = e_re[k];
        double e_im_k = e_im[k];

        // Load twiddles
        double w1_re_k = w1_re[k];
        double w1_im_k = w1_im[k];
        double w2_re_k = w2_re[k];
        double w2_im_k = w2_im[k];

#if RADIX5_DERIVE_W3W4
        // ✅ OPTIMIZED: Derive W3 and W4
        double w3_re_k, w3_im_k, w4_re_k, w4_im_k;
        cmul_scalar(w1_re_k, w1_im_k, w2_re_k, w2_im_k, &w3_re_k, &w3_im_k); // W3 = W1 * W2
        csquare_scalar(w2_re_k, w2_im_k, &w4_re_k, &w4_im_k);                 // W4 = W2²
#else
        double w3_re_k = w3_re[k];
        double w3_im_k = w3_im[k];
        double w4_re_k = w4_re[k];
        double w4_im_k = w4_im[k];
#endif

        // Apply twiddles: TB = B * W1, etc.
        double tb_re_k, tb_im_k, tc_re_k, tc_im_k, td_re_k, td_im_k, te_re_k, te_im_k;
        cmul_scalar(b_re_k, b_im_k, w1_re_k, w1_im_k, &tb_re_k, &tb_im_k);
        cmul_scalar(c_re_k, c_im_k, w2_re_k, w2_im_k, &tc_re_k, &tc_im_k);
        cmul_scalar(d_re_k, d_im_k, w3_re_k, w3_im_k, &td_re_k, &td_im_k);
        cmul_scalar(e_re_k, e_im_k, w4_re_k, w4_im_k, &te_re_k, &te_im_k);

        // Radix-5 butterfly
        double y0_re_k, y0_im_k, y1_re_k, y1_im_k, y2_re_k, y2_im_k;
        double y3_re_k, y3_im_k, y4_re_k, y4_im_k;

        radix5_butterfly_core_fv_scalar(
            a_re_k, a_im_k,
            tb_re_k, tb_im_k,
            tc_re_k, tc_im_k,
            td_re_k, td_im_k,
            te_re_k, te_im_k,
            &y0_re_k, &y0_im_k,
            &y1_re_k, &y1_im_k,
            &y2_re_k, &y2_im_k,
            &y3_re_k, &y3_im_k,
            &y4_re_k, &y4_im_k);

        // Store outputs
        y0_re[k] = y0_re_k;
        y0_im[k] = y0_im_k;
        y1_re[k] = y1_re_k;
        y1_im[k] = y1_im_k;
        y2_re[k] = y2_re_k;
        y2_im[k] = y2_im_k;
        y3_re[k] = y3_re_k;
        y3_im[k] = y3_im_k;
        y4_re[k] = y4_re_k;
        y4_im[k] = y4_im_k;
    }
}

//==============================================================================
// MAIN KERNEL: Backward Transform (Scalar)
//==============================================================================

/**
 * @brief Backward FFT - Pure scalar implementation
 * @details Identical to forward but uses backward butterfly
 */
void fft_radix5_bv_scalar(
    int K,
    const double *RESTRICT a_re, const double *RESTRICT a_im,
    const double *RESTRICT b_re, const double *RESTRICT b_im,
    const double *RESTRICT c_re, const double *RESTRICT c_im,
    const double *RESTRICT d_re, const double *RESTRICT d_im,
    const double *RESTRICT e_re, const double *RESTRICT e_im,
    const radix5_twiddles_t *RESTRICT tw,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double *RESTRICT y4_re, double *RESTRICT y4_im)
{
    const double *w1_re = tw->re;
    const double *w1_im = tw->im;
    const double *w2_re = tw->re + K;
    const double *w2_im = tw->im + K;
#if !RADIX5_DERIVE_W3W4
    const double *w3_re = tw->re + 2 * K;
    const double *w3_im = tw->im + 2 * K;
    const double *w4_re = tw->re + 3 * K;
    const double *w4_im = tw->im + 3 * K;
#endif

    for (int k = 0; k < K; k++)
    {
        double a_re_k = a_re[k];
        double a_im_k = a_im[k];
        double b_re_k = b_re[k];
        double b_im_k = b_im[k];
        double c_re_k = c_re[k];
        double c_im_k = c_im[k];
        double d_re_k = d_re[k];
        double d_im_k = d_im[k];
        double e_re_k = e_re[k];
        double e_im_k = e_im[k];

        double w1_re_k = w1_re[k];
        double w1_im_k = w1_im[k];
        double w2_re_k = w2_re[k];
        double w2_im_k = w2_im[k];

#if RADIX5_DERIVE_W3W4
        double w3_re_k, w3_im_k, w4_re_k, w4_im_k;
        cmul_scalar(w1_re_k, w1_im_k, w2_re_k, w2_im_k, &w3_re_k, &w3_im_k);
        csquare_scalar(w2_re_k, w2_im_k, &w4_re_k, &w4_im_k);
#else
        double w3_re_k = w3_re[k];
        double w3_im_k = w3_im[k];
        double w4_re_k = w4_re[k];
        double w4_im_k = w4_im[k];
#endif

        double tb_re_k, tb_im_k, tc_re_k, tc_im_k, td_re_k, td_im_k, te_re_k, te_im_k;
        cmul_scalar(b_re_k, b_im_k, w1_re_k, w1_im_k, &tb_re_k, &tb_im_k);
        cmul_scalar(c_re_k, c_im_k, w2_re_k, w2_im_k, &tc_re_k, &tc_im_k);
        cmul_scalar(d_re_k, d_im_k, w3_re_k, w3_im_k, &td_re_k, &td_im_k);
        cmul_scalar(e_re_k, e_im_k, w4_re_k, w4_im_k, &te_re_k, &te_im_k);

        double y0_re_k, y0_im_k, y1_re_k, y1_im_k, y2_re_k, y2_im_k;
        double y3_re_k, y3_im_k, y4_re_k, y4_im_k;

        radix5_butterfly_core_bv_scalar(
            a_re_k, a_im_k,
            tb_re_k, tb_im_k,
            tc_re_k, tc_im_k,
            td_re_k, td_im_k,
            te_re_k, te_im_k,
            &y0_re_k, &y0_im_k,
            &y1_re_k, &y1_im_k,
            &y2_re_k, &y2_im_k,
            &y3_re_k, &y3_im_k,
            &y4_re_k, &y4_im_k);

        y0_re[k] = y0_re_k;
        y0_im[k] = y0_im_k;
        y1_re[k] = y1_re_k;
        y1_im[k] = y1_im_k;
        y2_re[k] = y2_re_k;
        y2_im[k] = y2_im_k;
        y3_re[k] = y3_re_k;
        y3_im[k] = y3_im_k;
        y4_re[k] = y4_re_k;
        y4_im[k] = y4_im_k;
    }
}

#endif // FFT_RADIX5_SCALAR_OPTIMIZED_H