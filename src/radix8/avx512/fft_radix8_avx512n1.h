/**
 * @file fft_radix8_avx512_n1.h
 * @brief Radix-8 N=1 (Twiddle-less) AVX-512 with U=2 Software Pipelining
 *
 * @details
 * N=1 CODELET ARCHITECTURE (FFTW Terminology):
 * ============================================
 * - "N=1" = No stage twiddles (W1...W7 all equal to 1+0i)
 * - Only internal W_8 geometric twiddles remain
 * - Used as base case in larger mixed-radix factorizations
 * - Processes K independent radix-8 butterflies in parallel
 *
 * NEW OPTIMIZATIONS (2025):
 * ========================
 * ✅ U=2 software pipelining (32 ZMM peak)
 * ✅ Split nx_odd loads (controls register pressure)
 * ✅ Transient W8 constants (scoped broadcasts)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Optimized prefetch distance (56 doubles)
 * ✅ Two-wave store pattern (minimizes live ranges)
 *
 * @author FFT Optimization Team
 * @version 2.0-N1-AVX512-U2
 * @date 2025
 */

#ifndef FFT_RADIX8_AVX512_N1_H
#define FFT_RADIX8_AVX512_N1_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

//==============================================================================
// COMPILER PORTABILITY
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define TARGET_AVX512_FMA
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define TARGET_AVX512_FMA __attribute__((target("avx512f,avx512dq,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define TARGET_AVX512_FMA
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#ifndef RADIX8_N1_STREAM_THRESHOLD_KB_AVX512
#define RADIX8_N1_STREAM_THRESHOLD_KB_AVX512 256
#endif

#ifndef RADIX8_N1_PREFETCH_DISTANCE_AVX512
#define RADIX8_N1_PREFETCH_DISTANCE_AVX512 56
#endif

//==============================================================================
// W_8 GEOMETRIC CONSTANTS
//==============================================================================

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

// Forward transform twiddles: W_8^k = exp(-2πik/8)
#define W8_FV_1_RE C8_CONSTANT
#define W8_FV_1_IM (-C8_CONSTANT)
#define W8_FV_3_RE (-C8_CONSTANT)
#define W8_FV_3_IM (-C8_CONSTANT)

// Backward transform twiddles: W_8^(-k) = exp(+2πik/8)
#define W8_BV_1_RE C8_CONSTANT
#define W8_BV_1_IM C8_CONSTANT
#define W8_BV_3_RE (-C8_CONSTANT)
#define W8_BV_3_IM C8_CONSTANT

//==============================================================================
// CORE PRIMITIVES (from twiddle version)
//==============================================================================

/**
 * @brief Complex multiplication (SoA layout, optimal for AVX-512)
 */
TARGET_AVX512_FMA
FORCE_INLINE void
cmul_v512(__m512d ar, __m512d ai, __m512d br, __m512d bi,
          __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    *tr = _mm512_fmsub_pd(ar, br, _mm512_mul_pd(ai, bi));
    *ti = _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br));
}

/**
 * @brief Complex squaring (for W^2 derivation)
 */
TARGET_AVX512_FMA
FORCE_INLINE void
csquare_v512(__m512d ar, __m512d ai, __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    *tr = _mm512_fmsub_pd(ar, ar, _mm512_mul_pd(ai, ai));              // ar*ar - ai*ai
    *ti = _mm512_add_pd(_mm512_mul_pd(ar, ai), _mm512_mul_pd(ar, ai)); // 2*ar*ai
}

/**
 * @brief Radix-4 core butterfly (from twiddle version)
 */
TARGET_AVX512_FMA
FORCE_INLINE void
radix4_core_avx512(
    __m512d x0_re, __m512d x0_im, __m512d x1_re, __m512d x1_im,
    __m512d x2_re, __m512d x2_im, __m512d x3_re, __m512d x3_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d sign_mask)
{
    __m512d t0_re = _mm512_add_pd(x0_re, x2_re);
    __m512d t0_im = _mm512_add_pd(x0_im, x2_im);
    __m512d t1_re = _mm512_sub_pd(x0_re, x2_re);
    __m512d t1_im = _mm512_sub_pd(x0_im, x2_im);
    __m512d t2_re = _mm512_add_pd(x1_re, x3_re);
    __m512d t2_im = _mm512_add_pd(x1_im, x3_im);
    __m512d t3_re = _mm512_sub_pd(x1_re, x3_re);
    __m512d t3_im = _mm512_sub_pd(x1_im, x3_im);

    *y0_re = _mm512_add_pd(t0_re, t2_re);
    *y0_im = _mm512_add_pd(t0_im, t2_im);
    *y1_re = _mm512_sub_pd(t1_re, _mm512_xor_pd(t3_im, sign_mask));
    *y1_im = _mm512_add_pd(t1_im, _mm512_xor_pd(t3_re, sign_mask));
    *y2_re = _mm512_sub_pd(t0_re, t2_re);
    *y2_im = _mm512_sub_pd(t0_im, t2_im);
    *y3_re = _mm512_add_pd(t1_re, _mm512_xor_pd(t3_im, sign_mask));
    *y3_im = _mm512_sub_pd(t1_im, _mm512_xor_pd(t3_re, sign_mask));
}

//==============================================================================
// OPTIMIZED W₈ TWIDDLES (from twiddle version)
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
apply_w8_twiddles_forward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    const __m512d C8 = _mm512_set1_pd(C8_CONSTANT);
    const __m512d NEG_C8 = _mm512_set1_pd(-C8_CONSTANT);
    const __m512d neg_zero = _mm512_set1_pd(-0.0);

    // W₈^1 = (C8, -C8)
    {
        __m512d r1 = *o1_re;
        __m512d i1 = *o1_im;
        __m512d sum = _mm512_add_pd(r1, i1);
        __m512d diff = _mm512_sub_pd(i1, r1);
        *o1_re = _mm512_mul_pd(sum, C8);
        *o1_im = _mm512_mul_pd(diff, C8);
    }

    // W₈^2 = (0, -1)
    {
        __m512d r2 = *o2_re;
        *o2_re = *o2_im;
        *o2_im = _mm512_xor_pd(r2, neg_zero);
    }

    // W₈^3 = (-C8, -C8)
    {
        __m512d r3 = *o3_re;
        __m512d i3 = *o3_im;
        __m512d diff = _mm512_sub_pd(r3, i3);
        __m512d sum = _mm512_add_pd(r3, i3);
        *o3_re = _mm512_mul_pd(diff, NEG_C8);
        *o3_im = _mm512_mul_pd(sum, NEG_C8);
    }
}

TARGET_AVX512_FMA
FORCE_INLINE void
apply_w8_twiddles_backward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    const __m512d C8 = _mm512_set1_pd(C8_CONSTANT);
    const __m512d NEG_C8 = _mm512_set1_pd(-C8_CONSTANT);
    const __m512d neg_zero = _mm512_set1_pd(-0.0);

    // W₈^(-1) = (C8, +C8)
    {
        __m512d r1 = *o1_re;
        __m512d i1 = *o1_im;
        __m512d diff = _mm512_sub_pd(r1, i1);
        __m512d sum = _mm512_add_pd(r1, i1);
        *o1_re = _mm512_mul_pd(diff, C8);
        *o1_im = _mm512_mul_pd(sum, C8);
    }

    // W₈^(-2) = (0, +1)
    {
        __m512d r2 = *o2_re;
        *o2_re = _mm512_xor_pd(*o2_im, neg_zero);
        *o2_im = r2;
    }

    // W₈^(-3) = (-C8, +C8)
    {
        __m512d r3 = *o3_re;
        __m512d i3 = *o3_im;
        __m512d sum = _mm512_add_pd(r3, i3);
        __m512d diff = _mm512_sub_pd(i3, r3);
        *o3_re = _mm512_mul_pd(sum, NEG_C8);
        *o3_im = _mm512_mul_pd(diff, C8);
    }
}

//==============================================================================
// N=1 STAGE DRIVERS WITH U=2 PIPELINING
//==============================================================================

/**
 * @brief N=1 radix-8 stage driver - FORWARD transform with U=2 pipelining
 *
 * @details
 * Peak register usage: 32 ZMM
 * - Split nx_odd loads around stores
 * - Transient W8 constants
 * - Two-wave store pattern
 *
 * @param K Number of parallel butterflies (must be multiple of 8)
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_n1_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_N1_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_N1_STREAM_THRESHOLD_KB_AVX512 * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm512_stream_pd(p, v) : STPD(p, v))

    //==========================================================================
    // PROLOGUE: Load iteration k=0
    //==========================================================================
    __m512d nx0r = LDPD(&in_re[0 * K]);
    __m512d nx0i = LDPD(&in_im[0 * K]);
    __m512d nx1r = LDPD(&in_re[1 * K]);
    __m512d nx1i = LDPD(&in_im[1 * K]);
    __m512d nx2r = LDPD(&in_re[2 * K]);
    __m512d nx2i = LDPD(&in_im[2 * K]);
    __m512d nx3r = LDPD(&in_re[3 * K]);
    __m512d nx3i = LDPD(&in_im[3 * K]);
    __m512d nx4r = LDPD(&in_re[4 * K]);
    __m512d nx4i = LDPD(&in_im[4 * K]);
    __m512d nx5r = LDPD(&in_re[5 * K]);
    __m512d nx5i = LDPD(&in_im[5 * K]);
    __m512d nx6r = LDPD(&in_re[6 * K]);
    __m512d nx6i = LDPD(&in_im[6 * K]);
    __m512d nx7r = LDPD(&in_re[7 * K]);
    __m512d nx7i = LDPD(&in_im[7 * K]);

//==========================================================================
// STEADY-STATE U=2 LOOP
//==========================================================================
#pragma clang loop unroll(disable)
    for (size_t k = 0; k + 8 < K; k += 8)
    {
        // Move "next" → "current"
        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        const size_t kn = k + 8;

        // N=1: NO stage twiddle application (that's the defining characteristic!)

        //======================================================================
        // Load Next EVEN Inputs
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // Even Radix-4
        //======================================================================
        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // Odd Radix-4
        //======================================================================
        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // Apply W8 Twiddles (transient constants)
        //======================================================================
        {
            const __m512d W8_1_re = _mm512_set1_pd(W8_FV_1_RE);
            const __m512d W8_1_im = _mm512_set1_pd(W8_FV_1_IM);
            const __m512d W8_3_re = _mm512_set1_pd(W8_FV_3_RE);
            const __m512d W8_3_im = _mm512_set1_pd(W8_FV_3_IM);

            apply_w8_twiddles_forward_avx512(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                                             W8_1_re, W8_1_im, W8_3_re, W8_3_im);
        }

        //======================================================================
        // Combine & Store in Two Waves
        //======================================================================
        // Wave A: Store y0, y1
        ST_STREAM(&out_re[0 * K + k], _mm512_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm512_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm512_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm512_add_pd(e1i, o1i));

        // Load remaining NEXT ODD
        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        // Wave B: Store y2, y3 and all differences
        ST_STREAM(&out_re[2 * K + k], _mm512_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm512_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm512_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm512_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm512_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm512_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm512_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm512_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm512_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm512_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm512_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm512_sub_pd(e3i, o3i));

        //======================================================================
        // Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[4 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[4 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[5 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[5 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[6 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[6 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[7 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[7 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE: Compute Final Iteration
    //==========================================================================
    {
        size_t k = K - 8;

        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               SIGN_FLIP);
        }

        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               SIGN_FLIP);
        }

        {
            const __m512d W8_1_re = _mm512_set1_pd(W8_FV_1_RE);
            const __m512d W8_1_im = _mm512_set1_pd(W8_FV_1_IM);
            const __m512d W8_3_re = _mm512_set1_pd(W8_FV_3_RE);
            const __m512d W8_3_im = _mm512_set1_pd(W8_FV_3_IM);

            apply_w8_twiddles_forward_avx512(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                                             W8_1_re, W8_1_im, W8_3_re, W8_3_im);
        }

        ST_STREAM(&out_re[0 * K + k], _mm512_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm512_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm512_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm512_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm512_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm512_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm512_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm512_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm512_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm512_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm512_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm512_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm512_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm512_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm512_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm512_sub_pd(e3i, o3i));
    }

    if (use_nt)
        _mm_sfence();

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief N=1 radix-8 stage driver - BACKWARD transform with U=2 pipelining
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_n1_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_N1_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_N1_STREAM_THRESHOLD_KB_AVX512 * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm512_stream_pd(p, v) : STPD(p, v))

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m512d nx0r = LDPD(&in_re[0 * K]);
    __m512d nx0i = LDPD(&in_im[0 * K]);
    __m512d nx1r = LDPD(&in_re[1 * K]);
    __m512d nx1i = LDPD(&in_im[1 * K]);
    __m512d nx2r = LDPD(&in_re[2 * K]);
    __m512d nx2i = LDPD(&in_im[2 * K]);
    __m512d nx3r = LDPD(&in_re[3 * K]);
    __m512d nx3i = LDPD(&in_im[3 * K]);
    __m512d nx4r = LDPD(&in_re[4 * K]);
    __m512d nx4i = LDPD(&in_im[4 * K]);
    __m512d nx5r = LDPD(&in_re[5 * K]);
    __m512d nx5i = LDPD(&in_im[5 * K]);
    __m512d nx6r = LDPD(&in_re[6 * K]);
    __m512d nx6i = LDPD(&in_im[6 * K]);
    __m512d nx7r = LDPD(&in_re[7 * K]);
    __m512d nx7i = LDPD(&in_im[7 * K]);

//==========================================================================
// STEADY-STATE U=2 LOOP
//==========================================================================
#pragma clang loop unroll(disable)
    for (size_t k = 0; k + 8 < K; k += 8)
    {
        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        const size_t kn = k + 8;

        //======================================================================
        // Load Next EVEN Inputs
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // Even Radix-4 (BACKWARD: negated sign mask)
        //======================================================================
        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            const __m512d neg_zero = _mm512_set1_pd(-0.0);
            const __m512d neg_sign = _mm512_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               neg_sign);
        }

        //======================================================================
        // Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // Odd Radix-4 (BACKWARD: negated sign mask)
        //======================================================================
        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            const __m512d neg_zero = _mm512_set1_pd(-0.0);
            const __m512d neg_sign = _mm512_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               neg_sign);
        }

        //======================================================================
        // Apply W8 Twiddles (BACKWARD version)
        //======================================================================
        {
            const __m512d W8_1_re = _mm512_set1_pd(W8_BV_1_RE);
            const __m512d W8_1_im = _mm512_set1_pd(W8_BV_1_IM);
            const __m512d W8_3_re = _mm512_set1_pd(W8_BV_3_RE);
            const __m512d W8_3_im = _mm512_set1_pd(W8_BV_3_IM);

            apply_w8_twiddles_backward_avx512(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                                              W8_1_re, W8_1_im, W8_3_re, W8_3_im);
        }

        //======================================================================
        // Combine & Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], _mm512_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm512_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm512_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm512_add_pd(e1i, o1i));

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[2 * K + k], _mm512_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm512_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm512_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm512_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm512_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm512_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm512_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm512_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm512_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm512_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm512_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm512_sub_pd(e3i, o3i));

        //======================================================================
        // Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[4 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[4 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[5 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[5 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[6 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[6 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_re[7 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[7 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 8;

        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            const __m512d neg_zero = _mm512_set1_pd(-0.0);
            const __m512d neg_sign = _mm512_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               neg_sign);
        }

        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            const __m512d neg_zero = _mm512_set1_pd(-0.0);
            const __m512d neg_sign = _mm512_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               neg_sign);
        }

        {
            const __m512d W8_1_re = _mm512_set1_pd(W8_BV_1_RE);
            const __m512d W8_1_im = _mm512_set1_pd(W8_BV_1_IM);
            const __m512d W8_3_re = _mm512_set1_pd(W8_BV_3_RE);
            const __m512d W8_3_im = _mm512_set1_pd(W8_BV_3_IM);

            apply_w8_twiddles_backward_avx512(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                                              W8_1_re, W8_1_im, W8_3_re, W8_3_im);
        }

        ST_STREAM(&out_re[0 * K + k], _mm512_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm512_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm512_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm512_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm512_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm512_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm512_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm512_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm512_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm512_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm512_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm512_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm512_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm512_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm512_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm512_sub_pd(e3i, o3i));
    }

    if (use_nt)
        _mm_sfence();

#undef LDPD
#undef STPD
#undef ST_STREAM
}

#endif // FFT_RADIX8_AVX512_N1_H