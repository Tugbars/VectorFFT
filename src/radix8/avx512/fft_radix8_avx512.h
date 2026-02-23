/**
 * @file fft_radix8_avx512_blocked_hybrid_fixed.h
 * @brief Production Radix-8 AVX-512 with Hybrid Blocked Twiddles - SIGN-FLIP FIXED
 *
 * HYBRID TWIDDLE SYSTEM:
 * ======================
 * - BLOCKED4: K ≤ 256 (twiddles fit in L1D)
 *   * Load W1..W4 (4 blocks), derive W5=W1·W4, W6=W2·W4, W7=W3·W4
 *
 * - BLOCKED2: K > 256 (twiddles stream from L2/L3)
 *   * Load W1, W2 (2 blocks), derive W3=W1·W2, W4=W2²
 *   * Then W5=W1·W4, W6=W2·W4, W7=W3·W4
 *
 * NOTE: Original used W5=-W1 sign-flip which is MATHEMATICALLY WRONG.
 * W_N^(5k) = W_N^(k) · W_N^(4k), and W_N^(4k) ≠ -1 in general.
 * Fixed to derive via multiplication by W4.
 *
 * OPTIMIZATIONS:
 * ==============
 * ✅ U=2 software pipelining (8-wide vectors, stride 8)
 * ✅ Two-wave stores (register pressure control)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Adaptive prefetch NTA/T0 (56 doubles distance for 512-bit)
 * ✅ No-unroll pragmas (preserve instruction scheduling)
 * ✅ Fast W8 micro-kernels (add/sub, no cmul)
 * ✅ sfence after NT stores
 * ✅ ASSUME_ALIGNED(64) for AVX-512
 *
 * @note Requires AVX-512F + AVX-512DQ + FMA
 * @note K must be multiple of 8
 */

#ifndef FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H
#define FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/* Compiler portability */
#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, a) (ptr)
#define TARGET_AVX512
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, a) __builtin_assume_aligned(ptr, a)
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, a) (ptr)
#define TARGET_AVX512
#endif

/* Configuration */
#ifndef RADIX8_BLOCKED4_THRESHOLD_512
#define RADIX8_BLOCKED4_THRESHOLD_512 256
#endif
#ifndef RADIX8_STREAM_THRESHOLD_KB
#define RADIX8_STREAM_THRESHOLD_KB 256
#endif
#ifndef RADIX8_PREFETCH_DISTANCE_AVX512_B4
#define RADIX8_PREFETCH_DISTANCE_AVX512_B4 56
#endif
#ifndef RADIX8_PREFETCH_DISTANCE_AVX512_B2
#define RADIX8_PREFETCH_DISTANCE_AVX512_B2 64
#endif

/* Adaptive prefetch macro. use_nt must be in scope. */
#define RADIX8_PF512(addr) \
    do { if (use_nt) __builtin_prefetch((addr), 0, 0); \
         else        __builtin_prefetch((addr), 0, 3); } while(0)

/* Twiddle structures (same layout, 64-byte aligned) */
typedef struct {
    const double *RESTRICT re;  /* [4*K] = W1[0..K-1], W2[0..K-1], W3[0..K-1], W4[0..K-1] */
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked4_512_t;

typedef struct {
    const double *RESTRICT re;  /* [2*K] = W1[0..K-1], W2[0..K-1] */
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked2_512_t;

/*============================================================================
 * AVX-512 COMPLEX ARITHMETIC PRIMITIVES
 *============================================================================*/

/** Complex multiply: (cr,ci) = (ar,ai) * (br,bi)  [2 MUL + 2 FMA] */
FORCE_INLINE TARGET_AVX512 void
cmul_v512(__m512d ar, __m512d ai, __m512d br, __m512d bi,
          __m512d *cr, __m512d *ci)
{
    __m512d t = _mm512_mul_pd(ar, bi);
    *cr = _mm512_fmsub_pd(ar, br, _mm512_mul_pd(ai, bi));
    *ci = _mm512_fmadd_pd(ai, br, t);
}

/** Complex square: (cr,ci) = (ar,ai)^2  [1 MUL + 2 FMA, better than cmul] */
FORCE_INLINE TARGET_AVX512 void
csquare_v512(__m512d ar, __m512d ai, __m512d *cr, __m512d *ci)
{
    __m512d two_ar = _mm512_add_pd(ar, ar);
    *cr = _mm512_fmsub_pd(ar, ar, _mm512_mul_pd(ai, ai));
    *ci = _mm512_mul_pd(two_ar, ai);
}

/*============================================================================
 * RADIX-4 CORE (AVX-512)
 * Standard split-radix radix-4 butterfly.
 * sign_flip: -0.0 for forward (DFT), 0.0 for backward (IDFT).
 *============================================================================*/
FORCE_INLINE TARGET_AVX512 void
radix4_core_avx512(
    __m512d x0r, __m512d x0i, __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i, __m512d x3r, __m512d x3i,
    __m512d *y0r, __m512d *y0i, __m512d *y1r, __m512d *y1i,
    __m512d *y2r, __m512d *y2i, __m512d *y3r, __m512d *y3i,
    __m512d sign_flip)
{
    /* Level 1 */
    __m512d t0r = _mm512_add_pd(x0r, x2r), t0i = _mm512_add_pd(x0i, x2i);
    __m512d t1r = _mm512_sub_pd(x0r, x2r), t1i = _mm512_sub_pd(x0i, x2i);
    __m512d t2r = _mm512_add_pd(x1r, x3r), t2i = _mm512_add_pd(x1i, x3i);
    __m512d t3r = _mm512_sub_pd(x1r, x3r), t3i = _mm512_sub_pd(x1i, x3i);

    /* Level 2: match AVX2 sub/add structure exactly.
       sign_flip = -0.0 for forward, 0.0 for backward.
       Forward: y1 = t1 - j·t3, y3 = t1 + j·t3
       Backward: y1 = t1 + j·t3, y3 = t1 - j·t3 */
    __m512d sf3i = _mm512_xor_pd(t3i, sign_flip);  /* fwd: -t3i, bwd: t3i */
    __m512d sf3r = _mm512_xor_pd(t3r, sign_flip);  /* fwd: -t3r, bwd: t3r */

    *y0r = _mm512_add_pd(t0r, t2r); *y0i = _mm512_add_pd(t0i, t2i);
    *y1r = _mm512_sub_pd(t1r, sf3i); *y1i = _mm512_add_pd(t1i, sf3r);
    *y2r = _mm512_sub_pd(t0r, t2r); *y2i = _mm512_sub_pd(t0i, t2i);
    *y3r = _mm512_add_pd(t1r, sf3i); *y3i = _mm512_sub_pd(t1i, sf3r);
}

/*============================================================================
 * W8 TWIDDLE MICRO-KERNELS (AVX-512)
 * Apply W8^1, W8^2=-j, W8^3 to odd radix-4 outputs.
 *============================================================================*/

/** Forward: W8^1 = (1-j)/√2, W8^2 = -j, W8^3 = (-1-j)/√2 */
FORCE_INLINE TARGET_AVX512 void
w8_apply_fast_forward_avx512(__m512d *o1r, __m512d *o1i,
                              __m512d *o2r, __m512d *o2i,
                              __m512d *o3r, __m512d *o3i)
{
    const __m512d INV_SQRT2 = _mm512_set1_pd(0.7071067811865475244);
    /* o1 *= W8^1 = (1-j)/√2: new_r = (r+i)/√2, new_i = (i-r)/√2 */
    { __m512d sr = _mm512_add_pd(*o1r, *o1i);
      __m512d si = _mm512_sub_pd(*o1i, *o1r);
      *o1r = _mm512_mul_pd(sr, INV_SQRT2);
      *o1i = _mm512_mul_pd(si, INV_SQRT2); }
    /* o2 *= W8^2 = -j: new_r = i, new_i = -r */
    { __m512d tr = *o2i;
      *o2i = _mm512_xor_pd(*o2r, _mm512_set1_pd(-0.0));
      *o2r = tr; }
    /* o3 *= W8^3 = (-1-j)/√2: new_r = (i-r)/√2, new_i = -(r+i)/√2 */
    { __m512d sr = _mm512_sub_pd(*o3i, *o3r);
      __m512d si = _mm512_add_pd(*o3r, *o3i);
      *o3r = _mm512_mul_pd(sr, INV_SQRT2);
      *o3i = _mm512_mul_pd(_mm512_xor_pd(si, _mm512_set1_pd(-0.0)), INV_SQRT2); }
}

/** Backward: W8^{-1} = (1+j)/√2, W8^{-2} = j, W8^{-3} = (-1+j)/√2 */
FORCE_INLINE TARGET_AVX512 void
w8_apply_fast_backward_avx512(__m512d *o1r, __m512d *o1i,
                               __m512d *o2r, __m512d *o2i,
                               __m512d *o3r, __m512d *o3i)
{
    const __m512d INV_SQRT2 = _mm512_set1_pd(0.7071067811865475244);
    /* o1 *= (1+j)/√2: new_r = (r-i)/√2, new_i = (r+i)/√2 */
    { __m512d sr = _mm512_sub_pd(*o1r, *o1i);
      __m512d si = _mm512_add_pd(*o1r, *o1i);
      *o1r = _mm512_mul_pd(sr, INV_SQRT2);
      *o1i = _mm512_mul_pd(si, INV_SQRT2); }
    /* o2 *= j: new_r = -i, new_i = r */
    { __m512d tr = _mm512_xor_pd(*o2i, _mm512_set1_pd(-0.0));
      *o2i = *o2r;
      *o2r = tr; }
    /* o3 *= (-1+j)/√2: new_r = -(r+i)/√2, new_i = (r-i)/√2 */
    { __m512d sr = _mm512_add_pd(*o3r, *o3i);
      __m512d si = _mm512_sub_pd(*o3r, *o3i);
      *o3r = _mm512_mul_pd(_mm512_xor_pd(sr, _mm512_set1_pd(-0.0)), INV_SQRT2);
      *o3i = _mm512_mul_pd(si, INV_SQRT2); }
}

/*============================================================================
 * BLOCKED4 FORWARD (AVX-512)
 *
 * Load W1..W4 per k-step, derive W5=W1·W4, W6=W2·W4, W7=W3·W4.
 * U=2 software pipelining, two-wave stores.
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_blocked4_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_512_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 64);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 64);

    /* PROLOGUE: load first inputs + twiddles */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

    __m512d nW1r=_mm512_load_pd(&re_base[0*K]), nW1i=_mm512_load_pd(&im_base[0*K]);
    __m512d nW2r=_mm512_load_pd(&re_base[1*K]), nW2i=_mm512_load_pd(&im_base[1*K]);
    __m512d nW3r=_mm512_load_pd(&re_base[2*K]), nW3i=_mm512_load_pd(&im_base[2*K]);
    __m512d nW4r=_mm512_load_pd(&re_base[3*K]), nW4i=_mm512_load_pd(&im_base[3*K]);

    /* STEADY-STATE U=2 LOOP */
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        __m512d W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;
        const size_t kn = k + 8;

        /* Apply twiddles: W1..W4 loaded, W5=W1·W4, W6=W2·W4, W7=W3·W4 */
        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        /* Load next even inputs */
        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        /* Even radix-4 */
        __m512d e0r,e0i, e1r,e1i, e2r,e2i, e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                             &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, SF); }

        /* Load next half-odd + twiddles */
        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        /* Odd radix-4 */
        __m512d o0r,o0i, o1r,o1i, o2r,o2i, o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                             &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, SF); }

        /* W8 twiddles */
        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        /* Store Wave A */
        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r));
        STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r));
        STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));

        /* Load remaining next odd + next twiddles */
        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);
        nW1r=_mm512_load_pd(&re_base[0*K+kn]); nW1i=_mm512_load_pd(&im_base[0*K+kn]);
        nW2r=_mm512_load_pd(&re_base[1*K+kn]); nW2i=_mm512_load_pd(&im_base[1*K+kn]);

        /* Store Wave B */
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r));
        STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r));
        STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r));
        STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r));
        STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r));
        STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r));
        STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        /* Load remaining next twiddles */
        nW3r=_mm512_load_pd(&re_base[2*K+kn]); nW3i=_mm512_load_pd(&im_base[2*K+kn]);
        nW4r=_mm512_load_pd(&re_base[3*K+kn]); nW4i=_mm512_load_pd(&im_base[3*K+kn]);

        /* Prefetch */
        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[2*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[2*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[3*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[3*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        __m512d W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;

        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                             &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                             &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}

/*============================================================================
 * BLOCKED4 BACKWARD (AVX-512)
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_blocked4_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_512_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    const __m512d ZERO = _mm512_setzero_pd();
    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 64);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 64);

    /* PROLOGUE */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

    __m512d nW1r=_mm512_load_pd(&re_base[0*K]), nW1i=_mm512_load_pd(&im_base[0*K]);
    __m512d nW2r=_mm512_load_pd(&re_base[1*K]), nW2i=_mm512_load_pd(&im_base[1*K]);
    __m512d nW3r=_mm512_load_pd(&re_base[2*K]), nW3i=_mm512_load_pd(&im_base[2*K]);
    __m512d nW4r=_mm512_load_pd(&re_base[3*K]), nW4i=_mm512_load_pd(&im_base[3*K]);

    /* STEADY-STATE */
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        __m512d W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;
        const size_t kn = k + 8;

        /* Twiddles: W5=W1·W4, W6=W2·W4, W7=W3·W4 */
        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, ZERO);

        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));

        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);
        nW1r=_mm512_load_pd(&re_base[0*K+kn]); nW1i=_mm512_load_pd(&im_base[0*K+kn]);
        nW2r=_mm512_load_pd(&re_base[1*K+kn]); nW2i=_mm512_load_pd(&im_base[1*K+kn]);

        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        nW3r=_mm512_load_pd(&re_base[2*K+kn]); nW3i=_mm512_load_pd(&im_base[2*K+kn]);
        nW4r=_mm512_load_pd(&re_base[3*K+kn]); nW4i=_mm512_load_pd(&im_base[3*K+kn]);

        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[2*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[2*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[3*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[3*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        __m512d W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;

        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}

/*============================================================================
 * BLOCKED2 FORWARD (AVX-512)
 *
 * Load W1, W2; derive W3=W1·W2, W4=W2², W5=W1·W4, W6=W2·W4, W7=W3·W4
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_blocked2_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_512_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 64);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 64);

    /* PROLOGUE */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

    __m512d nW1r=_mm512_load_pd(&re_base[0*K]), nW1i=_mm512_load_pd(&im_base[0*K]);
    __m512d nW2r=_mm512_load_pd(&re_base[1*K]), nW2i=_mm512_load_pd(&im_base[1*K]);

    /* STEADY-STATE */
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        const size_t kn = k + 8;

        /* Derive W3..W7 from W1,W2 and apply */
        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m512d W3r,W3i; cmul_v512(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m512d W4r,W4i; csquare_v512(W2r,W2i, &W4r,&W4i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                             &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                             &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));

        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);
        nW1r=_mm512_load_pd(&re_base[0*K+kn]); nW1i=_mm512_load_pd(&im_base[0*K+kn]);
        nW2r=_mm512_load_pd(&re_base[1*K+kn]); nW2i=_mm512_load_pd(&im_base[1*K+kn]);

        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[1*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;

        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m512d W3r,W3i; cmul_v512(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m512d W4r,W4i; csquare_v512(W2r,W2i, &W4r,&W4i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                             &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m512d SF = _mm512_set1_pd(-0.0);
          radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                             &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}

/*============================================================================
 * BLOCKED2 BACKWARD (AVX-512)
 *============================================================================*/
TARGET_AVX512
__attribute__((optimize("no-unroll-loops")))
static void
radix8_stage_blocked2_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_512_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define ST(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define STS(p, v) (use_nt ? _mm512_stream_pd(p, v) : ST(p, v))

    const __m512d ZERO = _mm512_setzero_pd();
    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 64);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 64);

    /* PROLOGUE */
    __m512d nx0r=LD(&in_re[0*K]), nx0i=LD(&in_im[0*K]);
    __m512d nx1r=LD(&in_re[1*K]), nx1i=LD(&in_im[1*K]);
    __m512d nx2r=LD(&in_re[2*K]), nx2i=LD(&in_im[2*K]);
    __m512d nx3r=LD(&in_re[3*K]), nx3i=LD(&in_im[3*K]);
    __m512d nx4r=LD(&in_re[4*K]), nx4i=LD(&in_im[4*K]);
    __m512d nx5r=LD(&in_re[5*K]), nx5i=LD(&in_im[5*K]);
    __m512d nx6r=LD(&in_re[6*K]), nx6i=LD(&in_im[6*K]);
    __m512d nx7r=LD(&in_re[7*K]), nx7i=LD(&in_im[7*K]);

    __m512d nW1r=_mm512_load_pd(&re_base[0*K]), nW1i=_mm512_load_pd(&im_base[0*K]);
    __m512d nW2r=_mm512_load_pd(&re_base[1*K]), nW2i=_mm512_load_pd(&im_base[1*K]);

    /* STEADY-STATE */
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 8 < K; k += 8) {
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        const size_t kn = k + 8;

        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m512d W3r,W3i; cmul_v512(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m512d W4r,W4i; csquare_v512(W2r,W2i, &W4r,&W4i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        nx0r=LD(&in_re[0*K+kn]); nx0i=LD(&in_im[0*K+kn]);
        nx2r=LD(&in_re[2*K+kn]); nx2i=LD(&in_im[2*K+kn]);
        nx4r=LD(&in_re[4*K+kn]); nx4i=LD(&in_im[4*K+kn]);
        nx6r=LD(&in_re[6*K+kn]); nx6i=LD(&in_im[6*K+kn]);

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, ZERO);

        nx1r=LD(&in_re[1*K+kn]); nx1i=LD(&in_im[1*K+kn]);
        nx3r=LD(&in_re[3*K+kn]); nx3i=LD(&in_im[3*K+kn]);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));

        nx5r=LD(&in_re[5*K+kn]); nx5i=LD(&in_im[5*K+kn]);
        nx7r=LD(&in_re[7*K+kn]); nx7i=LD(&in_im[7*K+kn]);
        nW1r=_mm512_load_pd(&re_base[0*K+kn]); nW1i=_mm512_load_pd(&im_base[0*K+kn]);
        nW2r=_mm512_load_pd(&re_base[1*K+kn]); nW2i=_mm512_load_pd(&im_base[1*K+kn]);

        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));

        if (kn + prefetch_dist < K) {
            RADIX8_PF512((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF512((const char *)&im_base[1*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 8;
        __m512d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i;
        __m512d x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m512d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i;
        __m512d x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m512d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;

        { cmul_v512(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v512(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m512d W3r,W3i; cmul_v512(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m512d W4r,W4i; csquare_v512(W2r,W2i, &W4r,&W4i);
          cmul_v512(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v512(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m512d W5r,W5i; cmul_v512(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m512d W6r,W6i; cmul_v512(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m512d W7r,W7i; cmul_v512(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v512(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v512(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v512(x7r,x7i, W7r,W7i, &x7r,&x7i); }

        __m512d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx512(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);

        __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx512(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx512(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        STS(&out_re[0*K+k], _mm512_add_pd(e0r,o0r)); STS(&out_im[0*K+k], _mm512_add_pd(e0i,o0i));
        STS(&out_re[1*K+k], _mm512_add_pd(e1r,o1r)); STS(&out_im[1*K+k], _mm512_add_pd(e1i,o1i));
        STS(&out_re[2*K+k], _mm512_add_pd(e2r,o2r)); STS(&out_im[2*K+k], _mm512_add_pd(e2i,o2i));
        STS(&out_re[3*K+k], _mm512_add_pd(e3r,o3r)); STS(&out_im[3*K+k], _mm512_add_pd(e3i,o3i));
        STS(&out_re[4*K+k], _mm512_sub_pd(e0r,o0r)); STS(&out_im[4*K+k], _mm512_sub_pd(e0i,o0i));
        STS(&out_re[5*K+k], _mm512_sub_pd(e1r,o1r)); STS(&out_im[5*K+k], _mm512_sub_pd(e1i,o1i));
        STS(&out_re[6*K+k], _mm512_sub_pd(e2r,o2r)); STS(&out_im[6*K+k], _mm512_sub_pd(e2i,o2i));
        STS(&out_re[7*K+k], _mm512_sub_pd(e3r,o3r)); STS(&out_im[7*K+k], _mm512_sub_pd(e3i,o3i));
    }

    if (use_nt) _mm_sfence();

#undef LD
#undef ST
#undef STS
}

#endif /* FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H */