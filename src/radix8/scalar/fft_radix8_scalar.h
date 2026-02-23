/**
 * @file fft_radix8_scalar_blocked_hybrid_xe_optimized.h
 * @brief Radix-8 SCALAR - Optimized for Xeon Sapphire Rapids / Core i9-14900K
 *
 * SIGN-FLIP BUG FIXED: W5=W1·W4, W6=W2·W4, W7=W3·W4 (not -W1,-W2,-W3)
 *
 * TARGET ARCHITECTURE: Intel Golden Cove / Raptor Cove
 * ====================================================
 * - Xeon Sapphire Rapids / Emerald Rapids
 * - Core i9-13900K / i9-14900K (Raptor Lake / Raptor Lake Refresh)
 *
 * OPTIMIZATIONS:
 * ==============
 * ✅ FMA-based complex arithmetic (exploits 2× FMA512 units)
 * ✅ Branch-free radix-4 cores (separate fwd/bwd, zero conditional overhead)
 * ✅ Hoisted address arithmetic (reduces AGU pressure)
 * ✅ Hybrid twiddle system (BLOCKED4/BLOCKED2)
 * ✅ Fast W8 micro-kernels (optimized operation count)
 * ✅ Prefetch hints
 *
 * @version 5.1-XEON (Sign-flip fix)
 * @date 2025
 */

#ifndef FFT_RADIX8_SCALAR_BLOCKED_HYBRID_XE_OPTIMIZED_H
#define FFT_RADIX8_SCALAR_BLOCKED_HYBRID_XE_OPTIMIZED_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

/*============================================================================
 * COMPILER PORTABILITY
 *============================================================================*/

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

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#else
#define PREFETCH(addr) ((void)0)
#endif

#if defined(__FMA__) || defined(__AVX2__) || \
    (defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__)))
#define SCALAR_HAS_FMA 1
#else
#define SCALAR_HAS_FMA 0
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#ifndef RADIX8_BLOCKED4_THRESHOLD
#define RADIX8_BLOCKED4_THRESHOLD 256
#endif

#ifndef RADIX8_PREFETCH_DISTANCE_SCALAR
#define RADIX8_PREFETCH_DISTANCE_SCALAR 8
#endif

/*============================================================================
 * BLOCKED TWIDDLE STRUCTURES
 *============================================================================*/

#ifndef RADIX8_TWIDDLE_TYPES_DEFINED
#define RADIX8_TWIDDLE_TYPES_DEFINED
typedef struct {
    const double *RESTRICT re;  /* [4*K] = W1[0..K-1], W2, W3, W4 */
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked4_t;

typedef struct {
    const double *RESTRICT re;  /* [2*K] = W1[0..K-1], W2 */
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked2_t;

typedef enum {
    RADIX8_TW_BLOCKED4,
    RADIX8_TW_BLOCKED2
} radix8_twiddle_mode_t;
#endif

/*============================================================================
 * CONSTANTS
 *============================================================================*/

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

#ifndef RADIX8_CHOOSE_MODE_DEFINED
#define RADIX8_CHOOSE_MODE_DEFINED
FORCE_INLINE radix8_twiddle_mode_t
radix8_choose_twiddle_mode(size_t K)
{
    return (K <= RADIX8_BLOCKED4_THRESHOLD) ? RADIX8_TW_BLOCKED4 : RADIX8_TW_BLOCKED2;
}
#endif /* RADIX8_CHOOSE_MODE_DEFINED */

/*============================================================================
 * CORE PRIMITIVES - FMA OPTIMIZED
 *============================================================================*/

#if SCALAR_HAS_FMA

FORCE_INLINE void
cmul_scalar(double ar, double ai, double br, double bi,
            double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = fma(ar, br, -ai * bi);
    *ti = fma(ar, bi, ai * br);
}

FORCE_INLINE void
csquare_scalar(double wr, double wi,
               double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = fma(wr, wr, -wi * wi);
    *ti = 2.0 * wr * wi;
}

#else

FORCE_INLINE void
cmul_scalar(double ar, double ai, double br, double bi,
            double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = ar * br - ai * bi;
    *ti = ar * bi + ai * br;
}

FORCE_INLINE void
csquare_scalar(double wr, double wi,
               double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = wr * wr - wi * wi;
    *ti = 2.0 * wr * wi;
}

#endif

/*============================================================================
 * BRANCH-FREE RADIX-4 CORES
 *============================================================================*/

FORCE_INLINE void
radix4_core_fwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i)
{
    double t0r = x0r + x2r, t0i = x0i + x2i;
    double t1r = x0r - x2r, t1i = x0i - x2i;
    double t2r = x1r + x3r, t2i = x1i + x3i;
    double t3r = x1r - x3r, t3i = x1i - x3i;

    *y0r = t0r + t2r; *y0i = t0i + t2i;
    *y2r = t0r - t2r; *y2i = t0i - t2i;
    /* Forward: -j rotation */
    *y1r = t1r + t3i; *y1i = t1i - t3r;
    *y3r = t1r - t3i; *y3i = t1i + t3r;
}

FORCE_INLINE void
radix4_core_bwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i)
{
    double t0r = x0r + x2r, t0i = x0i + x2i;
    double t1r = x0r - x2r, t1i = x0i - x2i;
    double t2r = x1r + x3r, t2i = x1i + x3i;
    double t3r = x1r - x3r, t3i = x1i - x3i;

    *y0r = t0r + t2r; *y0i = t0i + t2i;
    *y2r = t0r - t2r; *y2i = t0i - t2i;
    /* Backward: +j rotation */
    *y1r = t1r - t3i; *y1i = t1i + t3r;
    *y3r = t1r + t3i; *y3i = t1i - t3r;
}

/*============================================================================
 * FAST W8 MICRO-KERNELS
 *============================================================================*/

FORCE_INLINE void
w8_apply_fast_forward_scalar(
    double *RESTRICT o1r, double *RESTRICT o1i,
    double *RESTRICT o2r, double *RESTRICT o2i,
    double *RESTRICT o3r, double *RESTRICT o3i)
{
    const double c = C8_CONSTANT;
    /* o1 *= (1-j)/√2 */
    { double r = *o1r, i = *o1i;
      *o1r = c * (r + i); *o1i = c * (i - r); }
    /* o2 *= -j */
    { double r = *o2r; *o2r = *o2i; *o2i = -r; }
    /* o3 *= (-1-j)/√2 */
    { double r = *o3r, i = *o3i;
      *o3r = -c * (r - i); *o3i = -c * (r + i); }
}

FORCE_INLINE void
w8_apply_fast_backward_scalar(
    double *RESTRICT o1r, double *RESTRICT o1i,
    double *RESTRICT o2r, double *RESTRICT o2i,
    double *RESTRICT o3r, double *RESTRICT o3i)
{
    const double c = C8_CONSTANT;
    /* o1 *= (1+j)/√2 */
    { double r = *o1r, i = *o1i;
      *o1r = c * (r - i); *o1i = c * (r + i); }
    /* o2 *= +j */
    { double r = *o2r; *o2r = -(*o2i); *o2i = r; }
    /* o3 *= (-1+j)/√2: real=-(r+i)/√2, imag=(r-i)/√2 */
    { double r = *o3r, i = *o3i;
      *o3r = -c * (r + i); *o3i = c * (r - i); }
}

/*============================================================================
 * BLOCKED4: APPLY STAGE TWIDDLES - SIGN-FLIP BUG FIXED
 *
 * Original: W5=-W1, W6=-W2, W7=-W3  (WRONG: -W ≠ W·W4 in general)
 * Fixed:    W5=W1·W4, W6=W2·W4, W7=W3·W4
 *============================================================================*/

FORCE_INLINE void
apply_stage_twiddles_blocked4_scalar(
    size_t k, size_t K,
    double *RESTRICT x1r, double *RESTRICT x1i,
    double *RESTRICT x2r, double *RESTRICT x2i,
    double *RESTRICT x3r, double *RESTRICT x3i,
    double *RESTRICT x4r, double *RESTRICT x4i,
    double *RESTRICT x5r, double *RESTRICT x5i,
    double *RESTRICT x6r, double *RESTRICT x6i,
    double *RESTRICT x7r, double *RESTRICT x7i,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    /* Hoist base pointers (reduces AGU pressure) */
    const double *RESTRICT W1r_base = stage_tw->re + 0 * K;
    const double *RESTRICT W1i_base = stage_tw->im + 0 * K;
    const double *RESTRICT W2r_base = stage_tw->re + 1 * K;
    const double *RESTRICT W2i_base = stage_tw->im + 1 * K;
    const double *RESTRICT W3r_base = stage_tw->re + 2 * K;
    const double *RESTRICT W3i_base = stage_tw->im + 2 * K;
    const double *RESTRICT W4r_base = stage_tw->re + 3 * K;
    const double *RESTRICT W4i_base = stage_tw->im + 3 * K;

    double W1r = W1r_base[k], W1i = W1i_base[k];
    double W2r = W2r_base[k], W2i = W2i_base[k];
    double W3r = W3r_base[k], W3i = W3i_base[k];
    double W4r = W4r_base[k], W4i = W4i_base[k];

    /* Apply W1..W4 directly */
    { double tr, ti;
      cmul_scalar(*x1r,*x1i, W1r,W1i, &tr,&ti); *x1r=tr; *x1i=ti;
      cmul_scalar(*x2r,*x2i, W2r,W2i, &tr,&ti); *x2r=tr; *x2i=ti;
      cmul_scalar(*x3r,*x3i, W3r,W3i, &tr,&ti); *x3r=tr; *x3i=ti;
      cmul_scalar(*x4r,*x4i, W4r,W4i, &tr,&ti); *x4r=tr; *x4i=ti; }

    /* Derive W5=W1·W4, W6=W2·W4, W7=W3·W4 and apply */
    { double W5r,W5i, W6r,W6i, W7r,W7i, tr,ti;
      cmul_scalar(W1r,W1i, W4r,W4i, &W5r,&W5i);
      cmul_scalar(W2r,W2i, W4r,W4i, &W6r,&W6i);
      cmul_scalar(W3r,W3i, W4r,W4i, &W7r,&W7i);
      cmul_scalar(*x5r,*x5i, W5r,W5i, &tr,&ti); *x5r=tr; *x5i=ti;
      cmul_scalar(*x6r,*x6i, W6r,W6i, &tr,&ti); *x6r=tr; *x6i=ti;
      cmul_scalar(*x7r,*x7i, W7r,W7i, &tr,&ti); *x7r=tr; *x7i=ti; }
}

/*============================================================================
 * BLOCKED2: APPLY STAGE TWIDDLES - SIGN-FLIP BUG FIXED
 *
 * Load W1,W2. Derive W3=W1·W2, W4=W2².
 * Then W5=W1·W4, W6=W2·W4, W7=W3·W4.
 *============================================================================*/

FORCE_INLINE void
apply_stage_twiddles_blocked2_scalar(
    size_t k, size_t K,
    double *RESTRICT x1r, double *RESTRICT x1i,
    double *RESTRICT x2r, double *RESTRICT x2i,
    double *RESTRICT x3r, double *RESTRICT x3i,
    double *RESTRICT x4r, double *RESTRICT x4i,
    double *RESTRICT x5r, double *RESTRICT x5i,
    double *RESTRICT x6r, double *RESTRICT x6i,
    double *RESTRICT x7r, double *RESTRICT x7i,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    /* Hoist base pointers */
    const double *RESTRICT W1r_base = stage_tw->re + 0 * K;
    const double *RESTRICT W1i_base = stage_tw->im + 0 * K;
    const double *RESTRICT W2r_base = stage_tw->re + 1 * K;
    const double *RESTRICT W2i_base = stage_tw->im + 1 * K;

    double W1r = W1r_base[k], W1i = W1i_base[k];
    double W2r = W2r_base[k], W2i = W2i_base[k];

    /* Apply W1, W2 */
    { double tr, ti;
      cmul_scalar(*x1r,*x1i, W1r,W1i, &tr,&ti); *x1r=tr; *x1i=ti;
      cmul_scalar(*x2r,*x2i, W2r,W2i, &tr,&ti); *x2r=tr; *x2i=ti; }

    /* Derive W3=W1·W2, W4=W2² */
    double W3r,W3i, W4r,W4i;
    cmul_scalar(W1r,W1i, W2r,W2i, &W3r,&W3i);
    csquare_scalar(W2r,W2i, &W4r,&W4i);

    /* Apply W3, W4 */
    { double tr, ti;
      cmul_scalar(*x3r,*x3i, W3r,W3i, &tr,&ti); *x3r=tr; *x3i=ti;
      cmul_scalar(*x4r,*x4i, W4r,W4i, &tr,&ti); *x4r=tr; *x4i=ti; }

    /* Derive W5=W1·W4, W6=W2·W4, W7=W3·W4 and apply */
    { double W5r,W5i, W6r,W6i, W7r,W7i, tr,ti;
      cmul_scalar(W1r,W1i, W4r,W4i, &W5r,&W5i);
      cmul_scalar(W2r,W2i, W4r,W4i, &W6r,&W6i);
      cmul_scalar(W3r,W3i, W4r,W4i, &W7r,&W7i);
      cmul_scalar(*x5r,*x5i, W5r,W5i, &tr,&ti); *x5r=tr; *x5i=ti;
      cmul_scalar(*x6r,*x6i, W6r,W6i, &tr,&ti); *x6r=tr; *x6i=ti;
      cmul_scalar(*x7r,*x7i, W7r,W7i, &tr,&ti); *x7r=tr; *x7i=ti; }
}

/*============================================================================
 * SINGLE BUTTERFLY - BLOCKED4 FORWARD (HOISTED POINTERS)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_blocked4_forward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    apply_stage_twiddles_blocked4_scalar(k,K,
        &x1r,&x1i, &x2r,&x2i, &x3r,&x3i, &x4r,&x4i,
        &x5r,&x5i, &x6r,&x6i, &x7r,&x7i, stage_tw);

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_fwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_fwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_forward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * SINGLE BUTTERFLY - BLOCKED4 BACKWARD (HOISTED POINTERS)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_blocked4_backward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    apply_stage_twiddles_blocked4_scalar(k,K,
        &x1r,&x1i, &x2r,&x2i, &x3r,&x3i, &x4r,&x4i,
        &x5r,&x5i, &x6r,&x6i, &x7r,&x7i, stage_tw);

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_bwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_bwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_backward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * SINGLE BUTTERFLY - BLOCKED2 FORWARD (HOISTED POINTERS)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_blocked2_forward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    apply_stage_twiddles_blocked2_scalar(k,K,
        &x1r,&x1i, &x2r,&x2i, &x3r,&x3i, &x4r,&x4i,
        &x5r,&x5i, &x6r,&x6i, &x7r,&x7i, stage_tw);

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_fwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_fwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_forward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * SINGLE BUTTERFLY - BLOCKED2 BACKWARD (HOISTED POINTERS)
 *============================================================================*/

FORCE_INLINE void
radix8_butterfly_blocked2_backward_scalar(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const double *RESTRICT r0=in_re+0*K, *RESTRICT r1=in_re+1*K;
    const double *RESTRICT r2=in_re+2*K, *RESTRICT r3=in_re+3*K;
    const double *RESTRICT r4=in_re+4*K, *RESTRICT r5=in_re+5*K;
    const double *RESTRICT r6=in_re+6*K, *RESTRICT r7=in_re+7*K;
    const double *RESTRICT i0=in_im+0*K, *RESTRICT i1=in_im+1*K;
    const double *RESTRICT i2=in_im+2*K, *RESTRICT i3=in_im+3*K;
    const double *RESTRICT i4=in_im+4*K, *RESTRICT i5=in_im+5*K;
    const double *RESTRICT i6=in_im+6*K, *RESTRICT i7=in_im+7*K;

    double x0r=r0[k],x0i=i0[k], x1r=r1[k],x1i=i1[k];
    double x2r=r2[k],x2i=i2[k], x3r=r3[k],x3i=i3[k];
    double x4r=r4[k],x4i=i4[k], x5r=r5[k],x5i=i5[k];
    double x6r=r6[k],x6i=i6[k], x7r=r7[k],x7i=i7[k];

    apply_stage_twiddles_blocked2_scalar(k,K,
        &x1r,&x1i, &x2r,&x2i, &x3r,&x3i, &x4r,&x4i,
        &x5r,&x5i, &x6r,&x6i, &x7r,&x7i, stage_tw);

    double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
    radix4_core_bwd_scalar(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i);
    double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
    radix4_core_bwd_scalar(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i);
    w8_apply_fast_backward_scalar(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

    out_re[0*K+k]=e0r+o0r; out_im[0*K+k]=e0i+o0i;
    out_re[1*K+k]=e1r+o1r; out_im[1*K+k]=e1i+o1i;
    out_re[2*K+k]=e2r+o2r; out_im[2*K+k]=e2i+o2i;
    out_re[3*K+k]=e3r+o3r; out_im[3*K+k]=e3i+o3i;
    out_re[4*K+k]=e0r-o0r; out_im[4*K+k]=e0i-o0i;
    out_re[5*K+k]=e1r-o1r; out_im[5*K+k]=e1i-o1i;
    out_re[6*K+k]=e2r-o2r; out_im[6*K+k]=e2i-o2i;
    out_re[7*K+k]=e3r-o3r; out_im[7*K+k]=e3i-o3i;
}

/*============================================================================
 * STAGE DRIVERS WITH PREFETCH
 *============================================================================*/

FORCE_INLINE void
radix8_stage_blocked4_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[k+pf]); PREFETCH(&in_im[k+pf]);
            PREFETCH(&stage_tw->re[0*K+k+pf]); PREFETCH(&stage_tw->im[0*K+k+pf]);
            PREFETCH(&stage_tw->re[1*K+k+pf]); PREFETCH(&stage_tw->im[1*K+k+pf]);
            PREFETCH(&stage_tw->re[2*K+k+pf]); PREFETCH(&stage_tw->im[2*K+k+pf]);
            PREFETCH(&stage_tw->re[3*K+k+pf]); PREFETCH(&stage_tw->im[3*K+k+pf]);
        }
        radix8_butterfly_blocked4_forward_scalar(k,K, in_re,in_im, out_re,out_im, stage_tw);
    }
}

FORCE_INLINE void
radix8_stage_blocked4_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[k+pf]); PREFETCH(&in_im[k+pf]);
            PREFETCH(&stage_tw->re[0*K+k+pf]); PREFETCH(&stage_tw->im[0*K+k+pf]);
            PREFETCH(&stage_tw->re[1*K+k+pf]); PREFETCH(&stage_tw->im[1*K+k+pf]);
            PREFETCH(&stage_tw->re[2*K+k+pf]); PREFETCH(&stage_tw->im[2*K+k+pf]);
            PREFETCH(&stage_tw->re[3*K+k+pf]); PREFETCH(&stage_tw->im[3*K+k+pf]);
        }
        radix8_butterfly_blocked4_backward_scalar(k,K, in_re,in_im, out_re,out_im, stage_tw);
    }
}

FORCE_INLINE void
radix8_stage_blocked2_forward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[k+pf]); PREFETCH(&in_im[k+pf]);
            PREFETCH(&stage_tw->re[0*K+k+pf]); PREFETCH(&stage_tw->im[0*K+k+pf]);
            PREFETCH(&stage_tw->re[1*K+k+pf]); PREFETCH(&stage_tw->im[1*K+k+pf]);
        }
        radix8_butterfly_blocked2_forward_scalar(k,K, in_re,in_im, out_re,out_im, stage_tw);
    }
}

FORCE_INLINE void
radix8_stage_blocked2_backward_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    const size_t pf = RADIX8_PREFETCH_DISTANCE_SCALAR;
    for (size_t k = 0; k < K; k++) {
        if (k + pf < K) {
            PREFETCH(&in_re[k+pf]); PREFETCH(&in_im[k+pf]);
            PREFETCH(&stage_tw->re[0*K+k+pf]); PREFETCH(&stage_tw->im[0*K+k+pf]);
            PREFETCH(&stage_tw->re[1*K+k+pf]); PREFETCH(&stage_tw->im[1*K+k+pf]);
        }
        radix8_butterfly_blocked2_backward_scalar(k,K, in_re,in_im, out_re,out_im, stage_tw);
    }
}

#endif /* FFT_RADIX8_SCALAR_BLOCKED_HYBRID_XE_OPTIMIZED_H */