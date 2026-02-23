/**
 * @file fft_radix8_avx2_blocked_hybrid_fixed.h
 * @brief Production Radix-8 AVX2 with Hybrid Blocked Twiddles
 */

#ifndef FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H
#define FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/* Compiler portability */
#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2_FMA
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2_FMA
#endif

/* Configuration */
#ifndef RADIX8_BLOCKED4_THRESHOLD
#define RADIX8_BLOCKED4_THRESHOLD 256
#endif
#ifndef RADIX8_STREAM_THRESHOLD_KB
#define RADIX8_STREAM_THRESHOLD_KB 256
#endif
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2
#define RADIX8_PREFETCH_DISTANCE_AVX2 28
#endif
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2_B4
#define RADIX8_PREFETCH_DISTANCE_AVX2_B4 24
#endif
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2_B2
#define RADIX8_PREFETCH_DISTANCE_AVX2_B2 32
#endif

/* Adaptive prefetch: NTA for streaming, T0 for cache reuse.
   use_nt must be in scope. Both branches use compile-time constants. */
#define RADIX8_PF(addr) \
    do { if (use_nt) __builtin_prefetch((addr), 0, 0); \
         else        __builtin_prefetch((addr), 0, 3); } while(0)

/* Twiddle structures */
#ifndef RADIX8_TWIDDLE_TYPES_DEFINED
#define RADIX8_TWIDDLE_TYPES_DEFINED
typedef struct {
    const double *RESTRICT re;
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked4_t;

typedef struct {
    const double *RESTRICT re;
    const double *RESTRICT im;
} radix8_stage_twiddles_blocked2_t;

typedef enum {
    RADIX8_TW_BLOCKED4,
    RADIX8_TW_BLOCKED2
} radix8_twiddle_mode_t;
#endif

/* W8 constants */
#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

#ifndef RADIX8_CHOOSE_MODE_DEFINED
#define RADIX8_CHOOSE_MODE_DEFINED
FORCE_INLINE radix8_twiddle_mode_t
radix8_choose_twiddle_mode(size_t K) {
    return (K <= RADIX8_BLOCKED4_THRESHOLD) ? RADIX8_TW_BLOCKED4 : RADIX8_TW_BLOCKED2;
}
#endif /* RADIX8_CHOOSE_MODE_DEFINED */

/* Core primitives */
TARGET_AVX2_FMA FORCE_INLINE void
cmul_v256(__m256d ar, __m256d ai, __m256d br, __m256d bi,
          __m256d *RESTRICT tr, __m256d *RESTRICT ti) {
    *tr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
    *ti = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
}

TARGET_AVX2_FMA FORCE_INLINE void
csquare_v256(__m256d wr, __m256d wi,
             __m256d *RESTRICT tr, __m256d *RESTRICT ti) {
    __m256d wr2 = _mm256_mul_pd(wr, wr);
    __m256d wi2 = _mm256_mul_pd(wi, wi);
    __m256d t = _mm256_mul_pd(wr, wi);
    *tr = _mm256_sub_pd(wr2, wi2);
    *ti = _mm256_add_pd(t, t);
}

TARGET_AVX2_FMA FORCE_INLINE void
radix4_core_avx2(
    __m256d x0_re, __m256d x0_im, __m256d x1_re, __m256d x1_im,
    __m256d x2_re, __m256d x2_im, __m256d x3_re, __m256d x3_im,
    __m256d *RESTRICT y0_re, __m256d *RESTRICT y0_im,
    __m256d *RESTRICT y1_re, __m256d *RESTRICT y1_im,
    __m256d *RESTRICT y2_re, __m256d *RESTRICT y2_im,
    __m256d *RESTRICT y3_re, __m256d *RESTRICT y3_im,
    __m256d sign_mask)
{
    __m256d t0_re = _mm256_add_pd(x0_re, x2_re);
    __m256d t0_im = _mm256_add_pd(x0_im, x2_im);
    __m256d t1_re = _mm256_sub_pd(x0_re, x2_re);
    __m256d t1_im = _mm256_sub_pd(x0_im, x2_im);
    __m256d t2_re = _mm256_add_pd(x1_re, x3_re);
    __m256d t2_im = _mm256_add_pd(x1_im, x3_im);
    __m256d t3_re = _mm256_sub_pd(x1_re, x3_re);
    __m256d t3_im = _mm256_sub_pd(x1_im, x3_im);

    *y0_re = _mm256_add_pd(t0_re, t2_re);
    *y0_im = _mm256_add_pd(t0_im, t2_im);
    *y1_re = _mm256_sub_pd(t1_re, _mm256_xor_pd(t3_im, sign_mask));
    *y1_im = _mm256_add_pd(t1_im, _mm256_xor_pd(t3_re, sign_mask));
    *y2_re = _mm256_sub_pd(t0_re, t2_re);
    *y2_im = _mm256_sub_pd(t0_im, t2_im);
    *y3_re = _mm256_add_pd(t1_re, _mm256_xor_pd(t3_im, sign_mask));
    *y3_im = _mm256_sub_pd(t1_im, _mm256_xor_pd(t3_re, sign_mask));
}

/* W8 fast micro-kernels */
TARGET_AVX2_FMA FORCE_INLINE void
w8_apply_fast_forward_avx2(
    __m256d *RESTRICT o1r, __m256d *RESTRICT o1i,
    __m256d *RESTRICT o2r, __m256d *RESTRICT o2i,
    __m256d *RESTRICT o3r, __m256d *RESTRICT o3i)
{
    const __m256d c = _mm256_set1_pd(C8_CONSTANT);
    const __m256d neg0 = _mm256_set1_pd(-0.0);

    __m256d s1 = _mm256_add_pd(*o1r, *o1i);
    __m256d d1 = _mm256_sub_pd(*o1i, *o1r);
    *o1r = _mm256_mul_pd(c, s1);
    *o1i = _mm256_mul_pd(c, d1);

    __m256d r2 = *o2r;
    *o2r = *o2i;
    *o2i = _mm256_xor_pd(r2, neg0);

    __m256d s3 = _mm256_sub_pd(*o3r, *o3i);
    __m256d d3 = _mm256_add_pd(*o3r, *o3i);
    *o3r = _mm256_xor_pd(_mm256_mul_pd(c, s3), neg0);
    *o3i = _mm256_xor_pd(_mm256_mul_pd(c, d3), neg0);
}

TARGET_AVX2_FMA FORCE_INLINE void
w8_apply_fast_backward_avx2(
    __m256d *RESTRICT o1r, __m256d *RESTRICT o1i,
    __m256d *RESTRICT o2r, __m256d *RESTRICT o2i,
    __m256d *RESTRICT o3r, __m256d *RESTRICT o3i)
{
    const __m256d c = _mm256_set1_pd(C8_CONSTANT);
    const __m256d neg0 = _mm256_set1_pd(-0.0);

    __m256d s1 = _mm256_sub_pd(*o1r, *o1i);
    __m256d d1 = _mm256_add_pd(*o1r, *o1i);
    *o1r = _mm256_mul_pd(c, s1);
    *o1i = _mm256_mul_pd(c, d1);

    __m256d r2 = *o2r;
    *o2r = _mm256_xor_pd(*o2i, neg0);
    *o2i = r2;

    __m256d s3 = _mm256_add_pd(*o3r, *o3i);
    __m256d d3 = _mm256_sub_pd(*o3i, *o3r);
    *o3r = _mm256_xor_pd(_mm256_mul_pd(c, s3), neg0);
    *o3i = _mm256_xor_pd(_mm256_mul_pd(c, d3), neg0);
}

/*============================================================================
 * BLOCKED4 FORWARD STAGE DRIVER (U=2 pipelining)
 *============================================================================*/
TARGET_AVX2_FMA
/* no-unroll: GCC attribute, Clang/ICX use pragma inside loop */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
static void
radix8_stage_blocked4_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    /* PROLOGUE */
    __m256d nx0r = LDPD(&in_re[0*K]); __m256d nx0i = LDPD(&in_im[0*K]);
    __m256d nx1r = LDPD(&in_re[1*K]); __m256d nx1i = LDPD(&in_im[1*K]);
    __m256d nx2r = LDPD(&in_re[2*K]); __m256d nx2i = LDPD(&in_im[2*K]);
    __m256d nx3r = LDPD(&in_re[3*K]); __m256d nx3i = LDPD(&in_im[3*K]);
    __m256d nx4r = LDPD(&in_re[4*K]); __m256d nx4i = LDPD(&in_im[4*K]);
    __m256d nx5r = LDPD(&in_re[5*K]); __m256d nx5i = LDPD(&in_im[5*K]);
    __m256d nx6r = LDPD(&in_re[6*K]); __m256d nx6i = LDPD(&in_im[6*K]);
    __m256d nx7r = LDPD(&in_re[7*K]); __m256d nx7i = LDPD(&in_im[7*K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0*K]); __m256d nW1i = _mm256_load_pd(&im_base[0*K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1*K]); __m256d nW2i = _mm256_load_pd(&im_base[1*K]);
    __m256d nW3r = _mm256_load_pd(&re_base[2*K]); __m256d nW3i = _mm256_load_pd(&im_base[2*K]);
    __m256d nW4r = _mm256_load_pd(&re_base[3*K]); __m256d nW4i = _mm256_load_pd(&im_base[3*K]);

    /* STEADY-STATE LOOP */
/* Prevent unroll: GCC uses pragma GCC, Clang/ICX uses pragma clang */
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i, W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;
        const size_t kn = k + 4;

        /* Apply stage twiddles: W1..W4 loaded, W5=W1·W4, W6=W2·W4, W7=W3·W4 */
        {
            cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
            cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
            cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
            cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
            __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
            __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
            __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
            cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
            cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
            cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        /* Load next even inputs */
        nx0r=LDPD(&in_re[0*K+kn]); nx0i=LDPD(&in_im[0*K+kn]);
        nx2r=LDPD(&in_re[2*K+kn]); nx2i=LDPD(&in_im[2*K+kn]);
        nx4r=LDPD(&in_re[4*K+kn]); nx4i=LDPD(&in_im[4*K+kn]);
        nx6r=LDPD(&in_re[6*K+kn]); nx6i=LDPD(&in_im[6*K+kn]);

        /* Even radix-4 */
        __m256d e0r,e0i, e1r,e1i, e2r,e2i, e3r,e3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x0r,x0i, x2r,x2i, x4r,x4i, x6r,x6i,
                           &e0r,&e0i, &e1r,&e1i, &e2r,&e2i, &e3r,&e3i, SF); }

        /* Load next half-odd */
        nx1r=LDPD(&in_re[1*K+kn]); nx1i=LDPD(&in_im[1*K+kn]);
        nx3r=LDPD(&in_re[3*K+kn]); nx3i=LDPD(&in_im[3*K+kn]);

        /* Odd radix-4 */
        __m256d o0r,o0i, o1r,o1i, o2r,o2i, o3r,o3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x1r,x1i, x3r,x3i, x5r,x5i, x7r,x7i,
                           &o0r,&o0i, &o1r,&o1i, &o2r,&o2i, &o3r,&o3i, SF); }

        /* W8 twiddles */
        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        /* Store wave A */
        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));

        nx5r=LDPD(&in_re[5*K+kn]); nx5i=LDPD(&in_im[5*K+kn]);
        nx7r=LDPD(&in_re[7*K+kn]); nx7i=LDPD(&in_im[7*K+kn]);

        /* Store wave B */
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));

        /* Load next twiddles */
        nW1r=_mm256_load_pd(&re_base[0*K+kn]); nW1i=_mm256_load_pd(&im_base[0*K+kn]);
        nW2r=_mm256_load_pd(&re_base[1*K+kn]); nW2i=_mm256_load_pd(&im_base[1*K+kn]);
        nW3r=_mm256_load_pd(&re_base[2*K+kn]); nW3i=_mm256_load_pd(&im_base[2*K+kn]);
        nW4r=_mm256_load_pd(&re_base[3*K+kn]); nW4i=_mm256_load_pd(&im_base[3*K+kn]);

        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[3*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[3*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i, W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;

        { /* W5=W1*W4, W6=W2*W4, W7=W3*W4 (NOT sign-flip) */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/*============================================================================
 * BLOCKED4 BACKWARD STAGE DRIVER
 *============================================================================*/
TARGET_AVX2_FMA
/* no-unroll: GCC attribute, Clang/ICX use pragma inside loop */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
static void
radix8_stage_blocked4_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d nx0r=LDPD(&in_re[0*K]), nx0i=LDPD(&in_im[0*K]);
    __m256d nx1r=LDPD(&in_re[1*K]), nx1i=LDPD(&in_im[1*K]);
    __m256d nx2r=LDPD(&in_re[2*K]), nx2i=LDPD(&in_im[2*K]);
    __m256d nx3r=LDPD(&in_re[3*K]), nx3i=LDPD(&in_im[3*K]);
    __m256d nx4r=LDPD(&in_re[4*K]), nx4i=LDPD(&in_im[4*K]);
    __m256d nx5r=LDPD(&in_re[5*K]), nx5i=LDPD(&in_im[5*K]);
    __m256d nx6r=LDPD(&in_re[6*K]), nx6i=LDPD(&in_im[6*K]);
    __m256d nx7r=LDPD(&in_re[7*K]), nx7i=LDPD(&in_im[7*K]);

    __m256d nW1r=_mm256_load_pd(&re_base[0*K]), nW1i=_mm256_load_pd(&im_base[0*K]);
    __m256d nW2r=_mm256_load_pd(&re_base[1*K]), nW2i=_mm256_load_pd(&im_base[1*K]);
    __m256d nW3r=_mm256_load_pd(&re_base[2*K]), nW3i=_mm256_load_pd(&im_base[2*K]);
    __m256d nW4r=_mm256_load_pd(&re_base[3*K]), nW4i=_mm256_load_pd(&im_base[3*K]);

/* Prevent unroll: GCC uses pragma GCC, Clang/ICX uses pragma clang */
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i, W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;
        const size_t kn = k + 4;

        { /* W5=W1*W4, W6=W2*W4, W7=W3*W4 (NOT sign-flip) */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        nx0r=LDPD(&in_re[0*K+kn]); nx0i=LDPD(&in_im[0*K+kn]);
        nx2r=LDPD(&in_re[2*K+kn]); nx2i=LDPD(&in_im[2*K+kn]);
        nx4r=LDPD(&in_re[4*K+kn]); nx4i=LDPD(&in_im[4*K+kn]);
        nx6r=LDPD(&in_re[6*K+kn]); nx6i=LDPD(&in_im[6*K+kn]);

        /* Backward: pass zero sign_mask (XOR(-0.0,-0.0)=0.0) to radix4 */
        const __m256d ZERO = _mm256_setzero_pd();

        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                         &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);

        nx1r=LDPD(&in_re[1*K+kn]); nx1i=LDPD(&in_im[1*K+kn]);
        nx3r=LDPD(&in_re[3*K+kn]); nx3i=LDPD(&in_im[3*K+kn]);

        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                         &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));

        nx5r=LDPD(&in_re[5*K+kn]); nx5i=LDPD(&in_im[5*K+kn]);
        nx7r=LDPD(&in_re[7*K+kn]); nx7i=LDPD(&in_im[7*K+kn]);

        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));

        nW1r=_mm256_load_pd(&re_base[0*K+kn]); nW1i=_mm256_load_pd(&im_base[0*K+kn]);
        nW2r=_mm256_load_pd(&re_base[1*K+kn]); nW2i=_mm256_load_pd(&im_base[1*K+kn]);
        nW3r=_mm256_load_pd(&re_base[2*K+kn]); nW3i=_mm256_load_pd(&im_base[2*K+kn]);
        nW4r=_mm256_load_pd(&re_base[3*K+kn]); nW4i=_mm256_load_pd(&im_base[3*K+kn]);

        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[2*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[3*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[3*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i, W3r=nW3r,W3i=nW3i, W4r=nW4r,W4i=nW4i;

        { /* W5=W1*W4, W6=W2*W4, W7=W3*W4 (NOT sign-flip) */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        const __m256d ZERO = _mm256_setzero_pd();
        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                         &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);
        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                         &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);
        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/*============================================================================
 * BLOCKED2 FORWARD STAGE DRIVER
 *============================================================================*/
TARGET_AVX2_FMA
/* no-unroll: GCC attribute, Clang/ICX use pragma inside loop */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
static void
radix8_stage_blocked2_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d nx0r=LDPD(&in_re[0*K]), nx0i=LDPD(&in_im[0*K]);
    __m256d nx1r=LDPD(&in_re[1*K]), nx1i=LDPD(&in_im[1*K]);
    __m256d nx2r=LDPD(&in_re[2*K]), nx2i=LDPD(&in_im[2*K]);
    __m256d nx3r=LDPD(&in_re[3*K]), nx3i=LDPD(&in_im[3*K]);
    __m256d nx4r=LDPD(&in_re[4*K]), nx4i=LDPD(&in_im[4*K]);
    __m256d nx5r=LDPD(&in_re[5*K]), nx5i=LDPD(&in_im[5*K]);
    __m256d nx6r=LDPD(&in_re[6*K]), nx6i=LDPD(&in_im[6*K]);
    __m256d nx7r=LDPD(&in_re[7*K]), nx7i=LDPD(&in_im[7*K]);

    __m256d nW1r=_mm256_load_pd(&re_base[0*K]), nW1i=_mm256_load_pd(&im_base[0*K]);
    __m256d nW2r=_mm256_load_pd(&re_base[1*K]), nW2i=_mm256_load_pd(&im_base[1*K]);

/* Prevent unroll: GCC uses pragma GCC, Clang/ICX uses pragma clang */
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        const size_t kn = k + 4;

        { /* Derive W3=W1*W2, W4=W2^2, W5=W1*W4, W6=W2*W4, W7=W3*W4 */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m256d W3r,W3i; cmul_v256(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m256d W4r,W4i; csquare_v256(W2r,W2i, &W4r,&W4i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        nx0r=LDPD(&in_re[0*K+kn]); nx0i=LDPD(&in_im[0*K+kn]);
        nx2r=LDPD(&in_re[2*K+kn]); nx2i=LDPD(&in_im[2*K+kn]);
        nx4r=LDPD(&in_re[4*K+kn]); nx4i=LDPD(&in_im[4*K+kn]);
        nx6r=LDPD(&in_re[6*K+kn]); nx6i=LDPD(&in_im[6*K+kn]);

        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }

        nx1r=LDPD(&in_re[1*K+kn]); nx1i=LDPD(&in_im[1*K+kn]);
        nx3r=LDPD(&in_re[3*K+kn]); nx3i=LDPD(&in_im[3*K+kn]);

        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }

        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));

        nx5r=LDPD(&in_re[5*K+kn]); nx5i=LDPD(&in_im[5*K+kn]);
        nx7r=LDPD(&in_re[7*K+kn]); nx7i=LDPD(&in_im[7*K+kn]);

        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));

        nW1r=_mm256_load_pd(&re_base[0*K+kn]); nW1i=_mm256_load_pd(&im_base[0*K+kn]);
        nW2r=_mm256_load_pd(&re_base[1*K+kn]); nW2i=_mm256_load_pd(&im_base[1*K+kn]);

        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[1*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;

        { /* Derive W3=W1*W2, W4=W2^2, W5=W1*W4, W6=W2*W4, W7=W3*W4 */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m256d W3r,W3i; cmul_v256(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m256d W4r,W4i; csquare_v256(W2r,W2i, &W4r,&W4i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                           &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, SF); }
        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { const __m256d SF = _mm256_set1_pd(-0.0);
          radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                           &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, SF); }
        w8_apply_fast_forward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/*============================================================================
 * BLOCKED2 BACKWARD STAGE DRIVER
 *============================================================================*/
TARGET_AVX2_FMA
/* no-unroll: GCC attribute, Clang/ICX use pragma inside loop */
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
__attribute__((optimize("no-unroll-loops")))
#endif
static void
radix8_stage_blocked2_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned  = (((uintptr_t)in_re  | (uintptr_t)in_im)  & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d nx0r=LDPD(&in_re[0*K]), nx0i=LDPD(&in_im[0*K]);
    __m256d nx1r=LDPD(&in_re[1*K]), nx1i=LDPD(&in_im[1*K]);
    __m256d nx2r=LDPD(&in_re[2*K]), nx2i=LDPD(&in_im[2*K]);
    __m256d nx3r=LDPD(&in_re[3*K]), nx3i=LDPD(&in_im[3*K]);
    __m256d nx4r=LDPD(&in_re[4*K]), nx4i=LDPD(&in_im[4*K]);
    __m256d nx5r=LDPD(&in_re[5*K]), nx5i=LDPD(&in_im[5*K]);
    __m256d nx6r=LDPD(&in_re[6*K]), nx6i=LDPD(&in_im[6*K]);
    __m256d nx7r=LDPD(&in_re[7*K]), nx7i=LDPD(&in_im[7*K]);

    __m256d nW1r=_mm256_load_pd(&re_base[0*K]), nW1i=_mm256_load_pd(&im_base[0*K]);
    __m256d nW2r=_mm256_load_pd(&re_base[1*K]), nW2i=_mm256_load_pd(&im_base[1*K]);

/* Prevent unroll: GCC uses pragma GCC, Clang/ICX uses pragma clang */
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
    for (size_t k = 0; k + 4 < K; k += 4) {
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;
        const size_t kn = k + 4;

        { /* Derive W3=W1*W2, W4=W2^2, W5=W1*W4, W6=W2*W4, W7=W3*W4 */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m256d W3r,W3i; cmul_v256(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m256d W4r,W4i; csquare_v256(W2r,W2i, &W4r,&W4i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        nx0r=LDPD(&in_re[0*K+kn]); nx0i=LDPD(&in_im[0*K+kn]);
        nx2r=LDPD(&in_re[2*K+kn]); nx2i=LDPD(&in_im[2*K+kn]);
        nx4r=LDPD(&in_re[4*K+kn]); nx4i=LDPD(&in_im[4*K+kn]);
        nx6r=LDPD(&in_re[6*K+kn]); nx6i=LDPD(&in_im[6*K+kn]);

        const __m256d ZERO = _mm256_setzero_pd();
        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                         &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);

        nx1r=LDPD(&in_re[1*K+kn]); nx1i=LDPD(&in_im[1*K+kn]);
        nx3r=LDPD(&in_re[3*K+kn]); nx3i=LDPD(&in_im[3*K+kn]);

        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                         &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);

        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));

        nx5r=LDPD(&in_re[5*K+kn]); nx5i=LDPD(&in_im[5*K+kn]);
        nx7r=LDPD(&in_re[7*K+kn]); nx7i=LDPD(&in_im[7*K+kn]);

        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));

        nW1r=_mm256_load_pd(&re_base[0*K+kn]); nW1i=_mm256_load_pd(&im_base[0*K+kn]);
        nW2r=_mm256_load_pd(&re_base[1*K+kn]); nW2i=_mm256_load_pd(&im_base[1*K+kn]);

        if (kn + prefetch_dist < K) {
            RADIX8_PF((const char *)&in_re[kn+prefetch_dist]);
            RADIX8_PF((const char *)&in_im[kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[0*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&re_base[1*K+kn+prefetch_dist]);
            RADIX8_PF((const char *)&im_base[1*K+kn+prefetch_dist]);
        }
    }

    /* EPILOGUE */
    {
        size_t k = K - 4;
        __m256d x0r=nx0r,x0i=nx0i, x1r=nx1r,x1i=nx1i, x2r=nx2r,x2i=nx2i, x3r=nx3r,x3i=nx3i;
        __m256d x4r=nx4r,x4i=nx4i, x5r=nx5r,x5i=nx5i, x6r=nx6r,x6i=nx6i, x7r=nx7r,x7i=nx7i;
        __m256d W1r=nW1r,W1i=nW1i, W2r=nW2r,W2i=nW2i;

        { /* Derive W3=W1*W2, W4=W2^2, W5=W1*W4, W6=W2*W4, W7=W3*W4 */
          cmul_v256(x1r,x1i, W1r,W1i, &x1r,&x1i);
          cmul_v256(x2r,x2i, W2r,W2i, &x2r,&x2i);
          __m256d W3r,W3i; cmul_v256(W1r,W1i, W2r,W2i, &W3r,&W3i);
          __m256d W4r,W4i; csquare_v256(W2r,W2i, &W4r,&W4i);
          cmul_v256(x3r,x3i, W3r,W3i, &x3r,&x3i);
          cmul_v256(x4r,x4i, W4r,W4i, &x4r,&x4i);
          __m256d W5r,W5i; cmul_v256(W1r,W1i, W4r,W4i, &W5r,&W5i);
          __m256d W6r,W6i; cmul_v256(W2r,W2i, W4r,W4i, &W6r,&W6i);
          __m256d W7r,W7i; cmul_v256(W3r,W3i, W4r,W4i, &W7r,&W7i);
          cmul_v256(x5r,x5i, W5r,W5i, &x5r,&x5i);
          cmul_v256(x6r,x6i, W6r,W6i, &x6r,&x6i);
          cmul_v256(x7r,x7i, W7r,W7i, &x7r,&x7i);
        }

        const __m256d ZERO = _mm256_setzero_pd();
        __m256d e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        radix4_core_avx2(x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i,
                         &e0r,&e0i,&e1r,&e1i,&e2r,&e2i,&e3r,&e3i, ZERO);
        __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        radix4_core_avx2(x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i,
                         &o0r,&o0i,&o1r,&o1i,&o2r,&o2i,&o3r,&o3i, ZERO);
        w8_apply_fast_backward_avx2(&o1r,&o1i, &o2r,&o2i, &o3r,&o3i);

        ST_STREAM(&out_re[0*K+k], _mm256_add_pd(e0r,o0r));
        ST_STREAM(&out_im[0*K+k], _mm256_add_pd(e0i,o0i));
        ST_STREAM(&out_re[1*K+k], _mm256_add_pd(e1r,o1r));
        ST_STREAM(&out_im[1*K+k], _mm256_add_pd(e1i,o1i));
        ST_STREAM(&out_re[2*K+k], _mm256_add_pd(e2r,o2r));
        ST_STREAM(&out_im[2*K+k], _mm256_add_pd(e2i,o2i));
        ST_STREAM(&out_re[3*K+k], _mm256_add_pd(e3r,o3r));
        ST_STREAM(&out_im[3*K+k], _mm256_add_pd(e3i,o3i));
        ST_STREAM(&out_re[4*K+k], _mm256_sub_pd(e0r,o0r));
        ST_STREAM(&out_im[4*K+k], _mm256_sub_pd(e0i,o0i));
        ST_STREAM(&out_re[5*K+k], _mm256_sub_pd(e1r,o1r));
        ST_STREAM(&out_im[5*K+k], _mm256_sub_pd(e1i,o1i));
        ST_STREAM(&out_re[6*K+k], _mm256_sub_pd(e2r,o2r));
        ST_STREAM(&out_im[6*K+k], _mm256_sub_pd(e2i,o2i));
        ST_STREAM(&out_re[7*K+k], _mm256_sub_pd(e3r,o3r));
        ST_STREAM(&out_im[7*K+k], _mm256_sub_pd(e3i,o3i));
    }

    if (use_nt) { _mm_sfence(); _mm256_zeroupper(); }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

#endif /* FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H */