/**
 * @file fft_radix8_scalar_blocked_hybrid_fixed.h
 * @brief Production Radix-8 SCALAR with Hybrid Blocked Twiddles - ALL OPTIMIZATIONS
 *
 * @details
 * HYBRID TWIDDLE SYSTEM:
 * ======================
 * - BLOCKED4: K ≤ 256 (twiddles fit in L1D)
 *   * Load W1..W4 (4 blocks)
 *   * W5=-W1, W6=-W2, W7=-W3 (sign flips only)
 *   * Zero derivation overhead, 43% bandwidth savings
 *
 * - BLOCKED2: K > 256 (twiddles stream from L2/L3)
 *   * Load W1, W2 (2 blocks)
 *   * Derive W3=W1×W2, W4=W2² (scalar operations)
 *   * W5=-W1, W6=-W2, W7=-W3 (sign flips)
 *   * Maximum bandwidth savings (71%)
 *
 * OPTIMIZATIONS INCLUDED:
 * =======================
 * ✅ Hybrid twiddle system (43-71% bandwidth savings)
 * ✅ Hoisted constants (W8 geometric values)
 * ✅ Optimized complex multiply (4 muls + 2 add/sub)
 * ✅ Prefetch hints (where supported by compiler)
 * ✅ Restrict pointers for aliasing optimization
 * ✅ Pure C - no SIMD, maximum portability
 *
 * PORTABILITY:
 * ============
 * - Runs on ANY architecture (x86, ARM, RISC-V, POWER, etc.)
 * - No SIMD instructions
 * - Standard C99
 * - Compiler-agnostic (GCC, Clang, MSVC, ICC, etc.)
 *
 * @author FFT Optimization Team
 * @version 3.1-SCALAR (Converted from AVX-512, All Algorithmic Optimizations Preserved)
 * @date 2025
 */

#ifndef FFT_RADIX8_SCALAR_BLOCKED_HYBRID_FIXED_H
#define FFT_RADIX8_SCALAR_BLOCKED_HYBRID_FIXED_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

//==============================================================================
// COMPILER PORTABILITY
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

// Prefetch hints (best effort - may be no-op on some compilers)
#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#else
#define PREFETCH(addr) ((void)0)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX8_BLOCKED4_THRESHOLD
 * @brief K threshold for BLOCKED4 vs BLOCKED2
 */
#ifndef RADIX8_BLOCKED4_THRESHOLD
#define RADIX8_BLOCKED4_THRESHOLD 256
#endif

/**
 * @def RADIX8_PREFETCH_DISTANCE_SCALAR
 * @brief Prefetch distance for scalar (8 complex numbers ahead)
 */
#ifndef RADIX8_PREFETCH_DISTANCE_SCALAR
#define RADIX8_PREFETCH_DISTANCE_SCALAR 8
#endif

//==============================================================================
// BLOCKED TWIDDLE STRUCTURES
//==============================================================================

typedef struct
{
    const double *RESTRICT re;
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked4_t;

typedef struct
{
    const double *RESTRICT re;
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked2_t;

typedef enum
{
    RADIX8_TW_BLOCKED4,
    RADIX8_TW_BLOCKED2
} radix8_twiddle_mode_t;

//==============================================================================
// W_8 GEOMETRIC CONSTANTS
//==============================================================================

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

#define W8_FV_1_RE C8_CONSTANT
#define W8_FV_1_IM (-C8_CONSTANT)
#define W8_FV_3_RE (-C8_CONSTANT)
#define W8_FV_3_IM (-C8_CONSTANT)

#define W8_BV_1_RE C8_CONSTANT
#define W8_BV_1_IM C8_CONSTANT
#define W8_BV_3_RE (-C8_CONSTANT)
#define W8_BV_3_IM C8_CONSTANT

//==============================================================================
// PLANNING HELPER
//==============================================================================

FORCE_INLINE radix8_twiddle_mode_t
radix8_choose_twiddle_mode(size_t K)
{
    return (K <= RADIX8_BLOCKED4_THRESHOLD) ? RADIX8_TW_BLOCKED4 : RADIX8_TW_BLOCKED2;
}

//==============================================================================
// CORE PRIMITIVES (SCALAR)
//==============================================================================

FORCE_INLINE void
cmul_scalar(double ar, double ai, double br, double bi,
            double *RESTRICT tr, double *RESTRICT ti)
{
    // (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
    *tr = ar * br - ai * bi;
    *ti = ar * bi + ai * br;
}

FORCE_INLINE void
csquare_scalar(double wr, double wi,
               double *RESTRICT tr, double *RESTRICT ti)
{
    // (wr + i*wi)^2 = (wr^2 - wi^2) + i*(2*wr*wi)
    *tr = wr * wr - wi * wi;
    *ti = 2.0 * wr * wi;
}

FORCE_INLINE void
radix4_core_scalar(
    double x0_re, double x0_im, double x1_re, double x1_im,
    double x2_re, double x2_im, double x3_re, double x3_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    int forward)
{
    double t0_re = x0_re + x2_re;
    double t0_im = x0_im + x2_im;
    double t1_re = x0_re - x2_re;
    double t1_im = x0_im - x2_im;
    double t2_re = x1_re + x3_re;
    double t2_im = x1_im + x3_im;
    double t3_re = x1_re - x3_re;
    double t3_im = x1_im - x3_im;

    *y0_re = t0_re + t2_re;
    *y0_im = t0_im + t2_im;
    *y2_re = t0_re - t2_re;
    *y2_im = t0_im - t2_im;

    if (forward)
    {
        *y1_re = t1_re + t3_im;
        *y1_im = t1_im - t3_re;
        *y3_re = t1_re - t3_im;
        *y3_im = t1_im + t3_re;
    }
    else
    {
        *y1_re = t1_re - t3_im;
        *y1_im = t1_im + t3_re;
        *y3_re = t1_re + t3_im;
        *y3_im = t1_im - t3_re;
    }
}

FORCE_INLINE void
apply_w8_twiddles_forward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    // W_8^1 multiplication
    double r1 = *o1_re, i1 = *o1_im;
    *o1_re = r1 * W8_1_re - i1 * W8_1_im;
    *o1_im = r1 * W8_1_im + i1 * W8_1_re;

    // W_8^2 = (0, -1) - optimized as swap + negate
    double r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = -r2;

    // W_8^3 multiplication
    double r3 = *o3_re, i3 = *o3_im;
    *o3_re = r3 * W8_3_re - i3 * W8_3_im;
    *o3_im = r3 * W8_3_im + i3 * W8_3_re;
}

FORCE_INLINE void
apply_w8_twiddles_backward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    // W_8^(-1) multiplication
    double r1 = *o1_re, i1 = *o1_im;
    *o1_re = r1 * W8_1_re - i1 * W8_1_im;
    *o1_im = r1 * W8_1_im + i1 * W8_1_re;

    // W_8^(-2) = (0, 1) - optimized as negate + swap
    double r2 = *o2_re;
    *o2_re = -(*o2_im);
    *o2_im = r2;

    // W_8^(-3) multiplication
    double r3 = *o3_re, i3 = *o3_im;
    *o3_re = r3 * W8_3_re - i3 * W8_3_im;
    *o3_im = r3 * W8_3_im + i3 * W8_3_re;
}

//==============================================================================
// BLOCKED4: APPLY STAGE TWIDDLES
//==============================================================================

FORCE_INLINE void
apply_stage_twiddles_blocked4_scalar(
    size_t k, size_t K,
    double *RESTRICT x1_re, double *RESTRICT x1_im,
    double *RESTRICT x2_re, double *RESTRICT x2_im,
    double *RESTRICT x3_re, double *RESTRICT x3_im,
    double *RESTRICT x4_re, double *RESTRICT x4_im,
    double *RESTRICT x5_re, double *RESTRICT x5_im,
    double *RESTRICT x6_re, double *RESTRICT x6_im,
    double *RESTRICT x7_re, double *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const double *re_base = stage_tw->re;
    const double *im_base = stage_tw->im;

    double W1r = re_base[0 * K + k];
    double W1i = im_base[0 * K + k];
    double W2r = re_base[1 * K + k];
    double W2i = im_base[1 * K + k];
    double W3r = re_base[2 * K + k];
    double W3i = im_base[2 * K + k];
    double W4r = re_base[3 * K + k];
    double W4i = im_base[3 * K + k];

    // Derive W5..W7 via sign flips (essentially free)
    double W5r = -W1r;
    double W5i = -W1i;
    double W6r = -W2r;
    double W6i = -W2i;
    double W7r = -W3r;
    double W7i = -W3i;

    // Apply all 7 twiddles
    double t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_scalar(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_scalar(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_scalar(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_scalar(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_scalar(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_scalar(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_scalar(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// BLOCKED2: APPLY STAGE TWIDDLES
//==============================================================================

FORCE_INLINE void
apply_stage_twiddles_blocked2_scalar(
    size_t k, size_t K,
    double *RESTRICT x1_re, double *RESTRICT x1_im,
    double *RESTRICT x2_re, double *RESTRICT x2_im,
    double *RESTRICT x3_re, double *RESTRICT x3_im,
    double *RESTRICT x4_re, double *RESTRICT x4_im,
    double *RESTRICT x5_re, double *RESTRICT x5_im,
    double *RESTRICT x6_re, double *RESTRICT x6_im,
    double *RESTRICT x7_re, double *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const double *re_base = stage_tw->re;
    const double *im_base = stage_tw->im;

    // Load W1, W2 back-to-back
    double W1r = re_base[0 * K + k];
    double W1i = im_base[0 * K + k];
    double W2r = re_base[1 * K + k];
    double W2i = im_base[1 * K + k];

    // Derive W3 = W1 × W2
    double W3r, W3i;
    cmul_scalar(W1r, W1i, W2r, W2i, &W3r, &W3i);

    // Derive W4 = W2²
    double W4r, W4i;
    csquare_scalar(W2r, W2i, &W4r, &W4i);

    // Derive W5..W7 via sign flips
    double W5r = -W1r;
    double W5i = -W1i;
    double W6r = -W2r;
    double W6i = -W2i;
    double W7r = -W3r;
    double W7i = -W3i;

    // Apply all 7 twiddles
    double t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_scalar(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_scalar(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_scalar(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_scalar(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_scalar(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_scalar(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_scalar(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED4 - FORWARD
//==============================================================================

FORCE_INLINE void
radix8_butterfly_blocked4_forward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    double x0_re = in_re[k + 0 * K];
    double x0_im = in_im[k + 0 * K];
    double x1_re = in_re[k + 1 * K];
    double x1_im = in_im[k + 1 * K];
    double x2_re = in_re[k + 2 * K];
    double x2_im = in_im[k + 2 * K];
    double x3_re = in_re[k + 3 * K];
    double x3_im = in_im[k + 3 * K];
    double x4_re = in_re[k + 4 * K];
    double x4_im = in_im[k + 4 * K];
    double x5_re = in_re[k + 5 * K];
    double x5_im = in_im[k + 5 * K];
    double x6_re = in_re[k + 6 * K];
    double x6_im = in_im[k + 6 * K];
    double x7_re = in_re[k + 7 * K];
    double x7_im = in_im[k + 7 * K];

    apply_stage_twiddles_blocked4_scalar(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw);

    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 1);

    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 1);

    apply_w8_twiddles_forward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    out_re[k + 0 * K] = e0_re + o0_re;
    out_im[k + 0 * K] = e0_im + o0_im;
    out_re[k + 1 * K] = e1_re + o1_re;
    out_im[k + 1 * K] = e1_im + o1_im;
    out_re[k + 2 * K] = e2_re + o2_re;
    out_im[k + 2 * K] = e2_im + o2_im;
    out_re[k + 3 * K] = e3_re + o3_re;
    out_im[k + 3 * K] = e3_im + o3_im;
    out_re[k + 4 * K] = e0_re - o0_re;
    out_im[k + 4 * K] = e0_im - o0_im;
    out_re[k + 5 * K] = e1_re - o1_re;
    out_im[k + 5 * K] = e1_im - o1_im;
    out_re[k + 6 * K] = e2_re - o2_re;
    out_im[k + 6 * K] = e2_im - o2_im;
    out_re[k + 7 * K] = e3_re - o3_re;
    out_im[k + 7 * K] = e3_im - o3_im;
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED2 - FORWARD
//==============================================================================

FORCE_INLINE void
radix8_butterfly_blocked2_forward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    double x0_re = in_re[k + 0 * K];
    double x0_im = in_im[k + 0 * K];
    double x1_re = in_re[k + 1 * K];
    double x1_im = in_im[k + 1 * K];
    double x2_re = in_re[k + 2 * K];
    double x2_im = in_im[k + 2 * K];
    double x3_re = in_re[k + 3 * K];
    double x3_im = in_im[k + 3 * K];
    double x4_re = in_re[k + 4 * K];
    double x4_im = in_im[k + 4 * K];
    double x5_re = in_re[k + 5 * K];
    double x5_im = in_im[k + 5 * K];
    double x6_re = in_re[k + 6 * K];
    double x6_im = in_im[k + 6 * K];
    double x7_re = in_re[k + 7 * K];
    double x7_im = in_im[k + 7 * K];

    apply_stage_twiddles_blocked2_scalar(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw);

    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 1);

    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 1);

    apply_w8_twiddles_forward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    out_re[k + 0 * K] = e0_re + o0_re;
    out_im[k + 0 * K] = e0_im + o0_im;
    out_re[k + 1 * K] = e1_re + o1_re;
    out_im[k + 1 * K] = e1_im + o1_im;
    out_re[k + 2 * K] = e2_re + o2_re;
    out_im[k + 2 * K] = e2_im + o2_im;
    out_re[k + 3 * K] = e3_re + o3_re;
    out_im[k + 3 * K] = e3_im + o3_im;
    out_re[k + 4 * K] = e0_re - o0_re;
    out_im[k + 4 * K] = e0_im - o0_im;
    out_re[k + 5 * K] = e1_re - o1_re;
    out_im[k + 5 * K] = e1_im - o1_im;
    out_re[k + 6 * K] = e2_re - o2_re;
    out_im[k + 6 * K] = e2_im - o2_im;
    out_re[k + 7 * K] = e3_re - o3_re;
    out_im[k + 7 * K] = e3_im - o3_im;
}

//==============================================================================
// BACKWARD BUTTERFLIES
//==============================================================================

FORCE_INLINE void
radix8_butterfly_blocked4_backward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    double x0_re = in_re[k + 0 * K];
    double x0_im = in_im[k + 0 * K];
    double x1_re = in_re[k + 1 * K];
    double x1_im = in_im[k + 1 * K];
    double x2_re = in_re[k + 2 * K];
    double x2_im = in_im[k + 2 * K];
    double x3_re = in_re[k + 3 * K];
    double x3_im = in_im[k + 3 * K];
    double x4_re = in_re[k + 4 * K];
    double x4_im = in_im[k + 4 * K];
    double x5_re = in_re[k + 5 * K];
    double x5_im = in_im[k + 5 * K];
    double x6_re = in_re[k + 6 * K];
    double x6_im = in_im[k + 6 * K];
    double x7_re = in_re[k + 7 * K];
    double x7_im = in_im[k + 7 * K];

    apply_stage_twiddles_blocked4_scalar(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw);

    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 0);

    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 0);

    apply_w8_twiddles_backward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    out_re[k + 0 * K] = e0_re + o0_re;
    out_im[k + 0 * K] = e0_im + o0_im;
    out_re[k + 1 * K] = e1_re + o1_re;
    out_im[k + 1 * K] = e1_im + o1_im;
    out_re[k + 2 * K] = e2_re + o2_re;
    out_im[k + 2 * K] = e2_im + o2_im;
    out_re[k + 3 * K] = e3_re + o3_re;
    out_im[k + 3 * K] = e3_im + o3_im;
    out_re[k + 4 * K] = e0_re - o0_re;
    out_im[k + 4 * K] = e0_im - o0_im;
    out_re[k + 5 * K] = e1_re - o1_re;
    out_im[k + 5 * K] = e1_im - o1_im;
    out_re[k + 6 * K] = e2_re - o2_re;
    out_im[k + 6 * K] = e2_im - o2_im;
    out_re[k + 7 * K] = e3_re - o3_re;
    out_im[k + 7 * K] = e3_im - o3_im;
}

FORCE_INLINE void
radix8_butterfly_blocked2_backward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    double x0_re = in_re[k + 0 * K];
    double x0_im = in_im[k + 0 * K];
    double x1_re = in_re[k + 1 * K];
    double x1_im = in_im[k + 1 * K];
    double x2_re = in_re[k + 2 * K];
    double x2_im = in_im[k + 2 * K];
    double x3_re = in_re[k + 3 * K];
    double x3_im = in_im[k + 3 * K];
    double x4_re = in_re[k + 4 * K];
    double x4_im = in_im[k + 4 * K];
    double x5_re = in_re[k + 5 * K];
    double x5_im = in_im[k + 5 * K];
    double x6_re = in_re[k + 6 * K];
    double x6_im = in_im[k + 6 * K];
    double x7_re = in_re[k + 7 * K];
    double x7_im = in_im[k + 7 * K];

    apply_stage_twiddles_blocked2_scalar(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw);

    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 0);

    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 0);

    apply_w8_twiddles_backward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    out_re[k + 0 * K] = e0_re + o0_re;
    out_im[k + 0 * K] = e0_im + o0_im;
    out_re[k + 1 * K] = e1_re + o1_re;
    out_im[k + 1 * K] = e1_im + o1_im;
    out_re[k + 2 * K] = e2_re + o2_re;
    out_im[k + 2 * K] = e2_im + o2_im;
    out_re[k + 3 * K] = e3_re + o3_re;
    out_im[k + 3 * K] = e3_im + o3_im;
    out_re[k + 4 * K] = e0_re - o0_re;
    out_im[k + 4 * K] = e0_im - o0_im;
    out_re[k + 5 * K] = e1_re - o1_re;
    out_im[k + 5 * K] = e1_im - o1_im;
    out_re[k + 6 * K] = e2_re - o2_re;
    out_im[k + 6 * K] = e2_im - o2_im;
    out_re[k + 7 * K] = e3_re - o3_re;
    out_im[k + 7 * K] = e3_im - o3_im;
}

//==============================================================================
// STAGE DRIVERS WITH PREFETCH
//==============================================================================

/**
 * @brief BLOCKED4 Forward - SCALAR WITH ALL ALGORITHMIC OPTIMIZATIONS
 */
FORCE_INLINE void
radix8_stage_blocked4_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    // Hoist constants ONCE per stage
    const double W8_1_re = W8_FV_1_RE;
    const double W8_1_im = W8_FV_1_IM;
    const double W8_3_re = W8_FV_3_RE;
    const double W8_3_im = W8_FV_3_IM;

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_SCALAR;

    for (size_t k = 0; k < K; k++)
    {
        // Prefetch next iteration
        if (k + prefetch_dist < K)
        {
            PREFETCH(&in_re[k + prefetch_dist]);
            PREFETCH(&in_im[k + prefetch_dist]);
            PREFETCH(&stage_tw->re[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[2 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[2 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[3 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[3 * K + (k + prefetch_dist)]);
        }

        radix8_butterfly_blocked4_forward_scalar(k, K, in_re, in_im, out_re, out_im,
                                                 stage_tw, W8_1_re, W8_1_im,
                                                 W8_3_re, W8_3_im);
    }
}

/**
 * @brief BLOCKED2 Forward - SCALAR WITH ALL ALGORITHMIC OPTIMIZATIONS
 */
FORCE_INLINE void
radix8_stage_blocked2_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const double W8_1_re = W8_FV_1_RE;
    const double W8_1_im = W8_FV_1_IM;
    const double W8_3_re = W8_FV_3_RE;
    const double W8_3_im = W8_FV_3_IM;

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_SCALAR;

    for (size_t k = 0; k < K; k++)
    {
        if (k + prefetch_dist < K)
        {
            PREFETCH(&in_re[k + prefetch_dist]);
            PREFETCH(&in_im[k + prefetch_dist]);
            PREFETCH(&stage_tw->re[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[1 * K + (k + prefetch_dist)]);
        }

        radix8_butterfly_blocked2_forward_scalar(k, K, in_re, in_im, out_re, out_im,
                                                 stage_tw, W8_1_re, W8_1_im,
                                                 W8_3_re, W8_3_im);
    }
}

/**
 * @brief BLOCKED4 Backward - SCALAR WITH ALL ALGORITHMIC OPTIMIZATIONS
 */
FORCE_INLINE void
radix8_stage_blocked4_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const double W8_1_re = W8_BV_1_RE;
    const double W8_1_im = W8_BV_1_IM;
    const double W8_3_re = W8_BV_3_RE;
    const double W8_3_im = W8_BV_3_IM;

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_SCALAR;

    for (size_t k = 0; k < K; k++)
    {
        if (k + prefetch_dist < K)
        {
            PREFETCH(&in_re[k + prefetch_dist]);
            PREFETCH(&in_im[k + prefetch_dist]);
            PREFETCH(&stage_tw->re[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[2 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[2 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[3 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[3 * K + (k + prefetch_dist)]);
        }

        radix8_butterfly_blocked4_backward_scalar(k, K, in_re, in_im, out_re, out_im,
                                                  stage_tw, W8_1_re, W8_1_im,
                                                  W8_3_re, W8_3_im);
    }
}

/**
 * @brief BLOCKED2 Backward - SCALAR WITH ALL ALGORITHMIC OPTIMIZATIONS
 */
FORCE_INLINE void
radix8_stage_blocked2_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const double W8_1_re = W8_BV_1_RE;
    const double W8_1_im = W8_BV_1_IM;
    const double W8_3_re = W8_BV_3_RE;
    const double W8_3_im = W8_BV_3_IM;

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_SCALAR;

    for (size_t k = 0; k < K; k++)
    {
        if (k + prefetch_dist < K)
        {
            PREFETCH(&in_re[k + prefetch_dist]);
            PREFETCH(&in_im[k + prefetch_dist]);
            PREFETCH(&stage_tw->re[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[0 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->re[1 * K + (k + prefetch_dist)]);
            PREFETCH(&stage_tw->im[1 * K + (k + prefetch_dist)]);
        }

        radix8_butterfly_blocked2_backward_scalar(k, K, in_re, in_im, out_re, out_im,
                                                  stage_tw, W8_1_re, W8_1_im,
                                                  W8_3_re, W8_3_im);
    }
}

#endif // FFT_RADIX8_SCALAR_BLOCKED_HYBRID_FIXED_H