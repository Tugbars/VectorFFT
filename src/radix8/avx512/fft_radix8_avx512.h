/**
 * @file fft_radix8_avx512_blocked_hybrid_fixed.h
 * @brief Production Radix-8 AVX-512 with Hybrid Blocked Twiddles - ALL OPTIMIZATIONS
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
 *   * Derive W3=W1×W2, W4=W2² (FMA operations)
 *   * W5=-W1, W6=-W2, W7=-W3 (sign flips)
 *   * Maximum bandwidth savings (71%)
 *
 * OPTIMIZATIONS INCLUDED:
 * =======================
 * ✅ U=2 software pipelining (load next while computing current)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (56 doubles for AVX-512)
 * ✅ Hoisted constants (W8, sign masks)
 * ✅ Alignment hints (ASSUME_ALIGNED)
 * ✅ Target attributes (explicit AVX-512 FMA)
 * ✅ Tail handling (assert K % 8 == 0 at plan time)
 *
 * @author FFT Optimization Team
 * @version 3.1 (All Regressions Fixed)
 * @date 2025
 */

#ifndef FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H
#define FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H

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
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX512_FMA
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_AVX512_FMA __attribute__((target("avx512f,avx512dq,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX512_FMA
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
 * @def RADIX8_STREAM_THRESHOLD_KB
 * @brief NT store threshold (in KB)
 */
#ifndef RADIX8_STREAM_THRESHOLD_KB
#define RADIX8_STREAM_THRESHOLD_KB 256
#endif

/**
 * @def RADIX8_PREFETCH_DISTANCE_AVX512
 * @brief Prefetch distance for AVX-512 (56 doubles)
 */
#ifndef RADIX8_PREFETCH_DISTANCE_AVX512
#define RADIX8_PREFETCH_DISTANCE_AVX512 56
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
// CORE PRIMITIVES
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
cmul_v512(__m512d ar, __m512d ai, __m512d br, __m512d bi,
          __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    *tr = _mm512_fmsub_pd(ar, br, _mm512_mul_pd(ai, bi));
    *ti = _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br));
}

TARGET_AVX512_FMA
FORCE_INLINE void
csquare_v512(__m512d wr, __m512d wi,
             __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    __m512d wr2 = _mm512_mul_pd(wr, wr);
    __m512d wi2 = _mm512_mul_pd(wi, wi);
    __m512d t = _mm512_mul_pd(wr, wi);
    *tr = _mm512_sub_pd(wr2, wi2);
    *ti = _mm512_add_pd(t, t);
}

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

TARGET_AVX512_FMA
FORCE_INLINE void
apply_w8_twiddles_forward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    // W_8^1 multiplication
    __m512d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm512_fmsub_pd(r1, W8_1_re, _mm512_mul_pd(i1, W8_1_im));
    *o1_im = _mm512_fmadd_pd(r1, W8_1_im, _mm512_mul_pd(i1, W8_1_re));

    // W_8^2 = (0, -1) - optimized as swap + negate (use constant inline)
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = _mm512_xor_pd(r2, neg_zero);

    // W_8^3 multiplication
    __m512d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm512_fmsub_pd(r3, W8_3_re, _mm512_mul_pd(i3, W8_3_im));
    *o3_im = _mm512_fmadd_pd(r3, W8_3_im, _mm512_mul_pd(i3, W8_3_re));
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
    // W_8^(-1) multiplication
    __m512d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm512_fmsub_pd(r1, W8_1_re, _mm512_mul_pd(i1, W8_1_im));
    *o1_im = _mm512_fmadd_pd(r1, W8_1_im, _mm512_mul_pd(i1, W8_1_re));

    // W_8^(-2) = (0, 1) - optimized as negate + swap (use constant inline)
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d r2 = *o2_re;
    *o2_re = _mm512_xor_pd(*o2_im, neg_zero);
    *o2_im = r2;

    // W_8^(-3) multiplication
    __m512d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm512_fmsub_pd(r3, W8_3_re, _mm512_mul_pd(i3, W8_3_im));
    *o3_im = _mm512_fmadd_pd(r3, W8_3_im, _mm512_mul_pd(i3, W8_3_re));
}

//==============================================================================
// BLOCKED4: APPLY STAGE TWIDDLES
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
apply_stage_twiddles_blocked4_avx512(
    size_t k, size_t K,
    __m512d *RESTRICT x1_re, __m512d *RESTRICT x1_im,
    __m512d *RESTRICT x2_re, __m512d *RESTRICT x2_im,
    __m512d *RESTRICT x3_re, __m512d *RESTRICT x3_im,
    __m512d *RESTRICT x4_re, __m512d *RESTRICT x4_im,
    __m512d *RESTRICT x5_re, __m512d *RESTRICT x5_im,
    __m512d *RESTRICT x6_re, __m512d *RESTRICT x6_im,
    __m512d *RESTRICT x7_re, __m512d *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m512d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 64);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 64);

    __m512d W1r = _mm512_load_pd(&re_base[0 * K + k]);
    __m512d W1i = _mm512_load_pd(&im_base[0 * K + k]);
    __m512d W2r = _mm512_load_pd(&re_base[1 * K + k]);
    __m512d W2i = _mm512_load_pd(&im_base[1 * K + k]);
    __m512d W3r = _mm512_load_pd(&re_base[2 * K + k]);
    __m512d W3i = _mm512_load_pd(&im_base[2 * K + k]);
    __m512d W4r = _mm512_load_pd(&re_base[3 * K + k]);
    __m512d W4i = _mm512_load_pd(&im_base[3 * K + k]);

    // Derive W5..W7 via sign flips (essentially free)
    __m512d W5r = _mm512_xor_pd(W1r, sign_mask);
    __m512d W5i = _mm512_xor_pd(W1i, sign_mask);
    __m512d W6r = _mm512_xor_pd(W2r, sign_mask);
    __m512d W6i = _mm512_xor_pd(W2i, sign_mask);
    __m512d W7r = _mm512_xor_pd(W3r, sign_mask);
    __m512d W7i = _mm512_xor_pd(W3i, sign_mask);

    // Apply all 7 twiddles (7 cmuls still happen)
    __m512d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_v512(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_v512(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_v512(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_v512(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_v512(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_v512(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_v512(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// BLOCKED2: APPLY STAGE TWIDDLES
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
apply_stage_twiddles_blocked2_avx512(
    size_t k, size_t K,
    __m512d *RESTRICT x1_re, __m512d *RESTRICT x1_im,
    __m512d *RESTRICT x2_re, __m512d *RESTRICT x2_im,
    __m512d *RESTRICT x3_re, __m512d *RESTRICT x3_im,
    __m512d *RESTRICT x4_re, __m512d *RESTRICT x4_im,
    __m512d *RESTRICT x5_re, __m512d *RESTRICT x5_im,
    __m512d *RESTRICT x6_re, __m512d *RESTRICT x6_im,
    __m512d *RESTRICT x7_re, __m512d *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m512d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 64);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 64);

    // Load W1, W2 back-to-back (good for LD unit queues)
    __m512d W1r = _mm512_load_pd(&re_base[0 * K + k]);
    __m512d W1i = _mm512_load_pd(&im_base[0 * K + k]);
    __m512d W2r = _mm512_load_pd(&re_base[1 * K + k]);
    __m512d W2i = _mm512_load_pd(&im_base[1 * K + k]);

    // Derive W3 = W1 × W2
    __m512d W3r, W3i;
    cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);

    // Derive W4 = W2²
    __m512d W4r, W4i;
    csquare_v512(W2r, W2i, &W4r, &W4i);

    // Derive W5..W7 via sign flips
    __m512d W5r = _mm512_xor_pd(W1r, sign_mask);
    __m512d W5i = _mm512_xor_pd(W1i, sign_mask);
    __m512d W6r = _mm512_xor_pd(W2r, sign_mask);
    __m512d W6i = _mm512_xor_pd(W2i, sign_mask);
    __m512d W7r = _mm512_xor_pd(W3r, sign_mask);
    __m512d W7i = _mm512_xor_pd(W3i, sign_mask);

    // Apply all 7 twiddles
    __m512d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_v512(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_v512(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_v512(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_v512(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_v512(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_v512(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_v512(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED4 - FORWARD
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_forward_avx512(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm512_store_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_store_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED4 - FORWARD - NT STORES
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_forward_avx512_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm512_stream_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_stream_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED2 - FORWARD
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_forward_avx512(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm512_store_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_store_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED2 - FORWARD - NT STORES
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_forward_avx512_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm512_stream_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_stream_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

//==============================================================================
// BACKWARD BUTTERFLIES (BLOCKED4 + BLOCKED2)
//==============================================================================

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_backward_avx512(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d neg_sign = _mm512_xor_pd(sign_mask, neg_zero);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       neg_sign);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       neg_sign);

    apply_w8_twiddles_backward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm512_store_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_store_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_backward_avx512_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d neg_sign = _mm512_xor_pd(sign_mask, neg_zero);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       neg_sign);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       neg_sign);

    apply_w8_twiddles_backward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm512_stream_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_stream_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_backward_avx512_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d neg_sign = _mm512_xor_pd(sign_mask, neg_zero);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       neg_sign);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       neg_sign);

    apply_w8_twiddles_backward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm512_stream_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_stream_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_stream_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_stream_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_stream_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_stream_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_stream_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_stream_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_stream_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

TARGET_AVX512_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_backward_avx512(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    __m512d x0_re = _mm512_load_pd(&in_re_aligned[k + 0 * K]);
    __m512d x0_im = _mm512_load_pd(&in_im_aligned[k + 0 * K]);
    __m512d x1_re = _mm512_load_pd(&in_re_aligned[k + 1 * K]);
    __m512d x1_im = _mm512_load_pd(&in_im_aligned[k + 1 * K]);
    __m512d x2_re = _mm512_load_pd(&in_re_aligned[k + 2 * K]);
    __m512d x2_im = _mm512_load_pd(&in_im_aligned[k + 2 * K]);
    __m512d x3_re = _mm512_load_pd(&in_re_aligned[k + 3 * K]);
    __m512d x3_im = _mm512_load_pd(&in_im_aligned[k + 3 * K]);
    __m512d x4_re = _mm512_load_pd(&in_re_aligned[k + 4 * K]);
    __m512d x4_im = _mm512_load_pd(&in_im_aligned[k + 4 * K]);
    __m512d x5_re = _mm512_load_pd(&in_re_aligned[k + 5 * K]);
    __m512d x5_im = _mm512_load_pd(&in_im_aligned[k + 5 * K]);
    __m512d x6_re = _mm512_load_pd(&in_re_aligned[k + 6 * K]);
    __m512d x6_im = _mm512_load_pd(&in_im_aligned[k + 6 * K]);
    __m512d x7_re = _mm512_load_pd(&in_re_aligned[k + 7 * K]);
    __m512d x7_im = _mm512_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx512(k, K,
                                         &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                         &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d neg_sign = _mm512_xor_pd(sign_mask, neg_zero);

    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       neg_sign);

    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       neg_sign);

    apply_w8_twiddles_backward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm512_store_pd(&out_re_aligned[k + 0 * K], _mm512_add_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 0 * K], _mm512_add_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 1 * K], _mm512_add_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 1 * K], _mm512_add_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 2 * K], _mm512_add_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 2 * K], _mm512_add_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 3 * K], _mm512_add_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 3 * K], _mm512_add_pd(e3_im, o3_im));
    _mm512_store_pd(&out_re_aligned[k + 4 * K], _mm512_sub_pd(e0_re, o0_re));
    _mm512_store_pd(&out_im_aligned[k + 4 * K], _mm512_sub_pd(e0_im, o0_im));
    _mm512_store_pd(&out_re_aligned[k + 5 * K], _mm512_sub_pd(e1_re, o1_re));
    _mm512_store_pd(&out_im_aligned[k + 5 * K], _mm512_sub_pd(e1_im, o1_im));
    _mm512_store_pd(&out_re_aligned[k + 6 * K], _mm512_sub_pd(e2_re, o2_re));
    _mm512_store_pd(&out_im_aligned[k + 6 * K], _mm512_sub_pd(e2_im, o2_im));
    _mm512_store_pd(&out_re_aligned[k + 7 * K], _mm512_sub_pd(e3_re, o3_re));
    _mm512_store_pd(&out_im_aligned[k + 7 * K], _mm512_sub_pd(e3_im, o3_im));
}

//==============================================================================
// STAGE DRIVERS WITH U=2 PIPELINING + NT STORES + PREFETCH
//==============================================================================

/**
 * @brief BLOCKED4 Forward - WITH ALL OPTIMIZATIONS
 *
 * ✅ U=2 software pipelining (overlapped loads/compute)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (56 doubles ahead)
 * ✅ Hoisted constants
 */
/**
 * @brief BLOCKED4 Forward with U=2 Software Pipelining
 * Peak: 32 ZMM (split nx_odd loads + transient W8)
 *
 * BLOCKED4: Loads all 4 twiddles directly (no derivation)
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_stage_blocked4_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
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

    __m512d nW1r = _mm512_load_pd(&stage_tw->re[0 * K]);
    __m512d nW1i = _mm512_load_pd(&stage_tw->im[0 * K]);
    __m512d nW2r = _mm512_load_pd(&stage_tw->re[1 * K]);
    __m512d nW2i = _mm512_load_pd(&stage_tw->im[1 * K]);
    __m512d nW3r = _mm512_load_pd(&stage_tw->re[2 * K]);
    __m512d nW3i = _mm512_load_pd(&stage_tw->im[2 * K]);
    __m512d nW4r = _mm512_load_pd(&stage_tw->re[3 * K]);
    __m512d nW4i = _mm512_load_pd(&stage_tw->im[3 * K]);

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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m512d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 8;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED4: No derivation)
        //======================================================================
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i); // Direct use (no derivation)
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i); // Direct use

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
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
        // STAGE 3: Even Radix-4
        //======================================================================
        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Odd Radix-4
        //======================================================================
        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (transient constants)
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
        // STAGE 7: Combine & Store in Two Waves
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
        // STAGE 8: Load Next Twiddles (all 4 blocks)
        //======================================================================
        nW1r = _mm512_load_pd(&stage_tw->re[0 * K + kn]);
        nW1i = _mm512_load_pd(&stage_tw->im[0 * K + kn]);
        nW2r = _mm512_load_pd(&stage_tw->re[1 * K + kn]);
        nW2i = _mm512_load_pd(&stage_tw->im[1 * K + kn]);
        nW3r = _mm512_load_pd(&stage_tw->re[2 * K + kn]);
        nW3i = _mm512_load_pd(&stage_tw->im[2 * K + kn]);
        nW4r = _mm512_load_pd(&stage_tw->re[3 * K + kn]);
        nW4i = _mm512_load_pd(&stage_tw->im[3 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch (4 twiddle blocks)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[3 * K + kn + prefetch_dist], pf_hint);
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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m512d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

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
 * @brief BLOCKED2 Forward with True U=2 Software Pipelining
 *
 * Register budget:
 * - Peak: 36 ZMM (brief window during odd radix-4 + combine)
 * - Typical: 24-28 ZMM
 * - Margin: 4 ZMM for allocator flexibility
 *
 * Optimizations:
 * ✅ In-place twiddle application (saves 14 ZMM)
 * ✅ JIT W3/W4 derivation (hides latency)
 * ✅ Staggered next-iteration loads (controls peak)
 * ✅ Transient SIGN_FLIP (saves 1 ZMM during critical path)
 * ✅ W8 constants kept live (avoids reload overhead)
 * ✅ NTA prefetch for streaming workloads
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_stage_blocked2_forward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
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

    __m512d nW1r = _mm512_load_pd(&stage_tw->re[0 * K]);
    __m512d nW1i = _mm512_load_pd(&stage_tw->im[0 * K]);
    __m512d nW2r = _mm512_load_pd(&stage_tw->re[1 * K]);
    __m512d nW2i = _mm512_load_pd(&stage_tw->im[1 * K]);

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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 8;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (In-Place + Interleaved Derivation)
        //======================================================================
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m512d W3r, W3i;
            cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m512d W4r, W4i;
            csquare_v512(W2r, W2i, &W4r, &W4i);

            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
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
        // STAGE 3: Even Radix-4
        //======================================================================
        __m512d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                               &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs (nx1, nx3 only)
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);
        // Peak here: e*(8) + x_odd(8) + nx_even(8) + nx_odd_half(4) = 28 ZMM

        //======================================================================
        // STAGE 5: Odd Radix-4
        //======================================================================
        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);
            radix4_core_avx512(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                               &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                               SIGN_FLIP);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (transient constants)
        //======================================================================
        {
            const __m512d W8_1_re = _mm512_set1_pd(W8_FV_1_RE);
            const __m512d W8_1_im = _mm512_set1_pd(W8_FV_1_IM);
            const __m512d W8_3_re = _mm512_set1_pd(W8_FV_3_RE);
            const __m512d W8_3_im = _mm512_set1_pd(W8_FV_3_IM);

            apply_w8_twiddles_forward_avx512(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                                             W8_1_re, W8_1_im, W8_3_re, W8_3_im);
        }
        // Peak here: e*(8) + o*(8) + nx_even(8) + nx_odd_half(4) + W8(4) = 32 ZMM

        //======================================================================
        // STAGE 7: Combine & Store in Two Waves
        //======================================================================
        // Wave A: Store y0, y1 → frees e0,e1,o0,o1
        ST_STREAM(&out_re[0 * K + k], _mm512_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm512_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm512_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm512_add_pd(e1i, o1i));
        // e0,e1,o0,o1 dead → 20 ZMM

        // Load remaining NEXT ODD (nx5, nx7)
        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);
        // Now: e2,e3,o2,o3(8) + nx_all(16) = 24 ZMM

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
        // All e*,o* dead → 16 ZMM

        //======================================================================
        // STAGE 8: Load Next Twiddles
        //======================================================================
        nW1r = _mm512_load_pd(&stage_tw->re[0 * K + kn]);
        nW1i = _mm512_load_pd(&stage_tw->im[0 * K + kn]);
        nW2r = _mm512_load_pd(&stage_tw->re[1 * K + kn]);
        nW2i = _mm512_load_pd(&stage_tw->im[1 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[1 * K + kn + prefetch_dist], pf_hint);
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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m512d W3r, W3i;
            cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m512d W4r, W4i;
            csquare_v512(W2r, W2i, &W4r, &W4i);

            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

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

/*
## **Register Budget Summary:**
```
Prologue:             20 ZMM (nx* + nW* + W8)
Twiddle Application:  ~30 ZMM (x* + W* + mW* + W8, transient peaks)
After Twiddles:       20 ZMM (x* + W8)
Load nx_even:         28 ZMM (x* + nx_even + W8)
Even Radix-4:         28 ZMM (x_odd + e* + nx_even + W8)
Load nx_odd:          36 ZMM ⚠️ PEAK (x_odd + e* + nx_all + W8)
Odd Radix-4:          36 ZMM (e* + o* + nx_all + W8)
Apply W8:             36 ZMM (e* + o* + nx_all + W8)
After Stores:         20 ZMM (nx_all + W8)
Load Twiddles:        24 ZMM (nx_all + nW* + W8)
Epilogue:             ~24 ZMM
*/

/**
 * @brief BLOCKED4 Backward with U=2 Software Pipelining
 * Peak: 32 ZMM (split nx_odd loads + transient W8)
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_stage_blocked4_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
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

    __m512d nW1r = _mm512_load_pd(&stage_tw->re[0 * K]);
    __m512d nW1i = _mm512_load_pd(&stage_tw->im[0 * K]);
    __m512d nW2r = _mm512_load_pd(&stage_tw->re[1 * K]);
    __m512d nW2i = _mm512_load_pd(&stage_tw->im[1 * K]);
    __m512d nW3r = _mm512_load_pd(&stage_tw->re[2 * K]);
    __m512d nW3i = _mm512_load_pd(&stage_tw->im[2 * K]);
    __m512d nW4r = _mm512_load_pd(&stage_tw->re[3 * K]);
    __m512d nW4i = _mm512_load_pd(&stage_tw->im[3 * K]);

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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m512d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 8;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED4: No derivation)
        //======================================================================
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
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
        // STAGE 3: Even Radix-4 (BACKWARD: negated sign mask)
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
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Odd Radix-4 (BACKWARD: negated sign mask)
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
        // STAGE 6: Apply W8 Twiddles (BACKWARD version)
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
        // STAGE 7: Combine & Store in Two Waves
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
        // STAGE 8: Load Next Twiddles (all 4 blocks)
        //======================================================================
        nW1r = _mm512_load_pd(&stage_tw->re[0 * K + kn]);
        nW1i = _mm512_load_pd(&stage_tw->im[0 * K + kn]);
        nW2r = _mm512_load_pd(&stage_tw->re[1 * K + kn]);
        nW2i = _mm512_load_pd(&stage_tw->im[1 * K + kn]);
        nW3r = _mm512_load_pd(&stage_tw->re[2 * K + kn]);
        nW3i = _mm512_load_pd(&stage_tw->im[2 * K + kn]);
        nW4r = _mm512_load_pd(&stage_tw->re[3 * K + kn]);
        nW4i = _mm512_load_pd(&stage_tw->im[3 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch (4 twiddle blocks)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[3 * K + kn + prefetch_dist], pf_hint);
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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m512d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

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

/**
 * @brief BLOCKED2 Backward with True U=2 Software Pipelining
 *
 * Changes from forward:
 * ✅ W8 constants: Use W8_BV_* (backward) instead of W8_FV_*
 * ✅ Radix-4 sign mask: Negated (+0.0 instead of -0.0) for IDFT
 * ✅ W8 application: Call apply_w8_twiddles_backward_avx512
 */
TARGET_AVX512_FMA
__attribute__((optimize("no-unroll-loops"))) static void
radix8_stage_blocked2_backward_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 7) == 0 && "K must be multiple of 8");
    assert(K >= 16 && "K must be >= 16 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 63) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 63) == 0;

#define LDPD(p) (in_aligned ? _mm512_load_pd(p) : _mm512_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm512_store_pd(p, v) : _mm512_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX512;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
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

    __m512d nW1r = _mm512_load_pd(&stage_tw->re[0 * K]);
    __m512d nW1i = _mm512_load_pd(&stage_tw->im[0 * K]);
    __m512d nW2r = _mm512_load_pd(&stage_tw->re[1 * K]);
    __m512d nW2i = _mm512_load_pd(&stage_tw->im[1 * K]);

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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 8;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (Identical to forward)
        //======================================================================
        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m512d W3r, W3i;
            cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m512d W4r, W4i;
            csquare_v512(W2r, W2i, &W4r, &W4i);

            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
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
        // STAGE 3: Even Radix-4 (BACKWARD: negated sign mask)
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
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Odd Radix-4 (BACKWARD: negated sign mask)
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
        // STAGE 6: Apply W8 Twiddles (BACKWARD version)
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
        // STAGE 7: Combine & Store in Two Waves
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
        // STAGE 8: Load Next Twiddles
        //======================================================================
        nW1r = _mm512_load_pd(&stage_tw->re[0 * K + kn]);
        nW1i = _mm512_load_pd(&stage_tw->im[0 * K + kn]);
        nW2r = _mm512_load_pd(&stage_tw->re[1 * K + kn]);
        nW2i = _mm512_load_pd(&stage_tw->im[1 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->re[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&stage_tw->im[1 * K + kn + prefetch_dist], pf_hint);
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
        __m512d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m512d SIGN_FLIP = _mm512_set1_pd(-0.0);

            cmul_v512(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v512(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m512d W3r, W3i;
            cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m512d W4r, W4i;
            csquare_v512(W2r, W2i, &W4r, &W4i);

            cmul_v512(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v512(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m512d mW1r = _mm512_xor_pd(W1r, SIGN_FLIP);
            __m512d mW1i = _mm512_xor_pd(W1i, SIGN_FLIP);
            __m512d mW2r = _mm512_xor_pd(W2r, SIGN_FLIP);
            __m512d mW2i = _mm512_xor_pd(W2i, SIGN_FLIP);
            __m512d mW3r = _mm512_xor_pd(W3r, SIGN_FLIP);
            __m512d mW3i = _mm512_xor_pd(W3i, SIGN_FLIP);

            cmul_v512(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v512(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v512(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

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

#endif // FFT_RADIX8_AVX512_BLOCKED_HYBRID_FIXED_H