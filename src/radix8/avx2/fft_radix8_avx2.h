/**
 * @file fft_radix8_avx2_blocked_hybrid_fixed.h
 * @brief Production Radix-8 AVX2 with Hybrid Blocked Twiddles - ALL OPTIMIZATIONS
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
 * ✅ Prefetch tuning (28 doubles for AVX2)
 * ✅ Hoisted constants (W8, sign masks)
 * ✅ Alignment hints (ASSUME_ALIGNED)
 * ✅ Target attributes (explicit AVX2 FMA)
 * ✅ Tail handling (assert K % 4 == 0 at plan time)
 *
 * @author FFT Optimization Team
 * @version 3.1-AVX2 (Converted from AVX-512, All Optimizations Preserved)
 * @date 2025
 */

#ifndef FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H
#define FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H

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
 * @def RADIX8_PREFETCH_DISTANCE_AVX2
 * @brief Prefetch distance for AVX2 (28 doubles - half of AVX-512's 56)
 */
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2
#define RADIX8_PREFETCH_DISTANCE_AVX2 28
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

TARGET_AVX2_FMA
FORCE_INLINE void
cmul_v256(__m256d ar, __m256d ai, __m256d br, __m256d bi,
          __m256d *RESTRICT tr, __m256d *RESTRICT ti)
{
    *tr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
    *ti = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
}

TARGET_AVX2_FMA
FORCE_INLINE void
csquare_v256(__m256d wr, __m256d wi,
             __m256d *RESTRICT tr, __m256d *RESTRICT ti)
{
    __m256d wr2 = _mm256_mul_pd(wr, wr);
    __m256d wi2 = _mm256_mul_pd(wi, wi);
    __m256d t = _mm256_mul_pd(wr, wi);
    *tr = _mm256_sub_pd(wr2, wi2);
    *ti = _mm256_add_pd(t, t);
}

TARGET_AVX2_FMA
FORCE_INLINE void
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

TARGET_AVX2_FMA
FORCE_INLINE void
apply_w8_twiddles_forward_avx2(
    __m256d *RESTRICT o1_re, __m256d *RESTRICT o1_im,
    __m256d *RESTRICT o2_re, __m256d *RESTRICT o2_im,
    __m256d *RESTRICT o3_re, __m256d *RESTRICT o3_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im)
{
    // W_8^1 multiplication
    __m256d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm256_fmsub_pd(r1, W8_1_re, _mm256_mul_pd(i1, W8_1_im));
    *o1_im = _mm256_fmadd_pd(r1, W8_1_im, _mm256_mul_pd(i1, W8_1_re));

    // W_8^2 = (0, -1) - optimized as swap + negate (use constant inline)
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = _mm256_xor_pd(r2, neg_zero);

    // W_8^3 multiplication
    __m256d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm256_fmsub_pd(r3, W8_3_re, _mm256_mul_pd(i3, W8_3_im));
    *o3_im = _mm256_fmadd_pd(r3, W8_3_im, _mm256_mul_pd(i3, W8_3_re));
}

TARGET_AVX2_FMA
FORCE_INLINE void
apply_w8_twiddles_backward_avx2(
    __m256d *RESTRICT o1_re, __m256d *RESTRICT o1_im,
    __m256d *RESTRICT o2_re, __m256d *RESTRICT o2_im,
    __m256d *RESTRICT o3_re, __m256d *RESTRICT o3_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im)
{
    // W_8^(-1) multiplication
    __m256d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm256_fmsub_pd(r1, W8_1_re, _mm256_mul_pd(i1, W8_1_im));
    *o1_im = _mm256_fmadd_pd(r1, W8_1_im, _mm256_mul_pd(i1, W8_1_re));

    // W_8^(-2) = (0, 1) - optimized as negate + swap (use constant inline)
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d r2 = *o2_re;
    *o2_re = _mm256_xor_pd(*o2_im, neg_zero);
    *o2_im = r2;

    // W_8^(-3) multiplication
    __m256d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm256_fmsub_pd(r3, W8_3_re, _mm256_mul_pd(i3, W8_3_im));
    *o3_im = _mm256_fmadd_pd(r3, W8_3_im, _mm256_mul_pd(i3, W8_3_re));
}

//==============================================================================
// BLOCKED4: APPLY STAGE TWIDDLES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
apply_stage_twiddles_blocked4_avx2(
    size_t k, size_t K,
    __m256d *RESTRICT x1_re, __m256d *RESTRICT x1_im,
    __m256d *RESTRICT x2_re, __m256d *RESTRICT x2_im,
    __m256d *RESTRICT x3_re, __m256d *RESTRICT x3_im,
    __m256d *RESTRICT x4_re, __m256d *RESTRICT x4_im,
    __m256d *RESTRICT x5_re, __m256d *RESTRICT x5_im,
    __m256d *RESTRICT x6_re, __m256d *RESTRICT x6_im,
    __m256d *RESTRICT x7_re, __m256d *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 32);

    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);
    __m256d W3r = _mm256_load_pd(&re_base[2 * K + k]);
    __m256d W3i = _mm256_load_pd(&im_base[2 * K + k]);
    __m256d W4r = _mm256_load_pd(&re_base[3 * K + k]);
    __m256d W4i = _mm256_load_pd(&im_base[3 * K + k]);

    // Derive W5..W7 via sign flips (essentially free)
    __m256d W5r = _mm256_xor_pd(W1r, sign_mask);
    __m256d W5i = _mm256_xor_pd(W1i, sign_mask);
    __m256d W6r = _mm256_xor_pd(W2r, sign_mask);
    __m256d W6i = _mm256_xor_pd(W2i, sign_mask);
    __m256d W7r = _mm256_xor_pd(W3r, sign_mask);
    __m256d W7i = _mm256_xor_pd(W3i, sign_mask);

    // Apply all 7 twiddles (7 cmuls still happen)
    __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_v256(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_v256(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_v256(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_v256(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_v256(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_v256(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_v256(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// BLOCKED2: APPLY STAGE TWIDDLES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
apply_stage_twiddles_blocked2_avx2(
    size_t k, size_t K,
    __m256d *RESTRICT x1_re, __m256d *RESTRICT x1_im,
    __m256d *RESTRICT x2_re, __m256d *RESTRICT x2_im,
    __m256d *RESTRICT x3_re, __m256d *RESTRICT x3_im,
    __m256d *RESTRICT x4_re, __m256d *RESTRICT x4_im,
    __m256d *RESTRICT x5_re, __m256d *RESTRICT x5_im,
    __m256d *RESTRICT x6_re, __m256d *RESTRICT x6_im,
    __m256d *RESTRICT x7_re, __m256d *RESTRICT x7_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m256d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(stage_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(stage_tw->im, 32);

    // Load W1, W2 back-to-back (good for LD unit queues)
    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);

    // Derive W3 = W1 × W2
    __m256d W3r, W3i;
    cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

    // Derive W4 = W2²
    __m256d W4r, W4i;
    csquare_v256(W2r, W2i, &W4r, &W4i);

    // Derive W5..W7 via sign flips
    __m256d W5r = _mm256_xor_pd(W1r, sign_mask);
    __m256d W5i = _mm256_xor_pd(W1i, sign_mask);
    __m256d W6r = _mm256_xor_pd(W2r, sign_mask);
    __m256d W6i = _mm256_xor_pd(W2i, sign_mask);
    __m256d W7r = _mm256_xor_pd(W3r, sign_mask);
    __m256d W7i = _mm256_xor_pd(W3i, sign_mask);

    // Apply all 7 twiddles
    __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i;
    cmul_v256(*x1_re, *x1_im, W1r, W1i, &t1r, &t1i);
    *x1_re = t1r;
    *x1_im = t1i;
    cmul_v256(*x2_re, *x2_im, W2r, W2i, &t2r, &t2i);
    *x2_re = t2r;
    *x2_im = t2i;
    cmul_v256(*x3_re, *x3_im, W3r, W3i, &t3r, &t3i);
    *x3_re = t3r;
    *x3_im = t3i;
    cmul_v256(*x4_re, *x4_im, W4r, W4i, &t4r, &t4i);
    *x4_re = t4r;
    *x4_im = t4i;
    cmul_v256(*x5_re, *x5_im, W5r, W5i, &t5r, &t5i);
    *x5_re = t5r;
    *x5_im = t5i;
    cmul_v256(*x6_re, *x6_im, W6r, W6i, &t6r, &t6i);
    *x6_re = t6r;
    *x6_im = t6i;
    cmul_v256(*x7_re, *x7_im, W7r, W7i, &t7r, &t7i);
    *x7_re = t7r;
    *x7_im = t7i;
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED4 - FORWARD - REGULAR STORES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_forward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm256_store_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_store_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED4 - FORWARD - NT STORES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_forward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm256_stream_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_stream_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED2 - FORWARD - REGULAR STORES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_forward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm256_store_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_store_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

//==============================================================================
// SINGLE BUTTERFLY - BLOCKED2 - FORWARD - NT STORES
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_forward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm256_stream_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_stream_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

//==============================================================================
// BACKWARD BUTTERFLIES (BLOCKED4 + BLOCKED2)
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_backward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm256_store_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_store_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked4_backward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked4_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm256_stream_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_stream_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_backward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Non-temporal stores
    _mm256_stream_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_stream_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_stream_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_stream_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_stream_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_stream_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_stream_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_stream_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_stream_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

TARGET_AVX2_FMA
FORCE_INLINE void
radix8_butterfly_blocked2_backward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    __m256d x0_re = _mm256_load_pd(&in_re_aligned[k + 0 * K]);
    __m256d x0_im = _mm256_load_pd(&in_im_aligned[k + 0 * K]);
    __m256d x1_re = _mm256_load_pd(&in_re_aligned[k + 1 * K]);
    __m256d x1_im = _mm256_load_pd(&in_im_aligned[k + 1 * K]);
    __m256d x2_re = _mm256_load_pd(&in_re_aligned[k + 2 * K]);
    __m256d x2_im = _mm256_load_pd(&in_im_aligned[k + 2 * K]);
    __m256d x3_re = _mm256_load_pd(&in_re_aligned[k + 3 * K]);
    __m256d x3_im = _mm256_load_pd(&in_im_aligned[k + 3 * K]);
    __m256d x4_re = _mm256_load_pd(&in_re_aligned[k + 4 * K]);
    __m256d x4_im = _mm256_load_pd(&in_im_aligned[k + 4 * K]);
    __m256d x5_re = _mm256_load_pd(&in_re_aligned[k + 5 * K]);
    __m256d x5_im = _mm256_load_pd(&in_im_aligned[k + 5 * K]);
    __m256d x6_re = _mm256_load_pd(&in_re_aligned[k + 6 * K]);
    __m256d x6_im = _mm256_load_pd(&in_im_aligned[k + 6 * K]);
    __m256d x7_re = _mm256_load_pd(&in_re_aligned[k + 7 * K]);
    __m256d x7_im = _mm256_load_pd(&in_im_aligned[k + 7 * K]);

    apply_stage_twiddles_blocked2_avx2(k, K,
                                       &x1_re, &x1_im, &x2_re, &x2_im, &x3_re, &x3_im, &x4_re, &x4_im,
                                       &x5_re, &x5_im, &x6_re, &x6_im, &x7_re, &x7_im, stage_tw, sign_mask);

    // Negate sign_mask for backward transform
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    _mm256_store_pd(&out_re_aligned[k + 0 * K], _mm256_add_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 0 * K], _mm256_add_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 1 * K], _mm256_add_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 1 * K], _mm256_add_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 2 * K], _mm256_add_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 2 * K], _mm256_add_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 3 * K], _mm256_add_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 3 * K], _mm256_add_pd(e3_im, o3_im));
    _mm256_store_pd(&out_re_aligned[k + 4 * K], _mm256_sub_pd(e0_re, o0_re));
    _mm256_store_pd(&out_im_aligned[k + 4 * K], _mm256_sub_pd(e0_im, o0_im));
    _mm256_store_pd(&out_re_aligned[k + 5 * K], _mm256_sub_pd(e1_re, o1_re));
    _mm256_store_pd(&out_im_aligned[k + 5 * K], _mm256_sub_pd(e1_im, o1_im));
    _mm256_store_pd(&out_re_aligned[k + 6 * K], _mm256_sub_pd(e2_re, o2_re));
    _mm256_store_pd(&out_im_aligned[k + 6 * K], _mm256_sub_pd(e2_im, o2_im));
    _mm256_store_pd(&out_re_aligned[k + 7 * K], _mm256_sub_pd(e3_re, o3_re));
    _mm256_store_pd(&out_im_aligned[k + 7 * K], _mm256_sub_pd(e3_im, o3_im));
}

//==============================================================================
// W8 FAST ADD/SUB MICRO-KERNELS (NO FMA - PURE ADD/SUB)
//==============================================================================

/**
 * @brief Fast W8 twiddle application using add/sub instead of cmul
 *
 * Replaces 2 complex FMAs (8 ops) with 4 adds + 4 muls + 2 XORs (10 ops)
 * BUT: Shorter critical path (add = 3 cycles vs FMA = 4 cycles)
 * AND: Better port distribution (adds run on more ports)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
w8_apply_fast_forward_avx2(
    __m256d *RESTRICT o1r, __m256d *RESTRICT o1i,
    __m256d *RESTRICT o2r, __m256d *RESTRICT o2i,
    __m256d *RESTRICT o3r, __m256d *RESTRICT o3i)
{
    const __m256d c = _mm256_set1_pd(C8_CONSTANT);
    const __m256d neg0 = _mm256_set1_pd(-0.0);

    // o1 *= c(1 - i) = c·(re + im) + i·c·(im - re)
    __m256d s1 = _mm256_add_pd(*o1r, *o1i);
    __m256d d1 = _mm256_sub_pd(*o1i, *o1r);
    *o1r = _mm256_mul_pd(c, s1);
    *o1i = _mm256_mul_pd(c, d1);

    // o2 *= -i (swap + negate)
    __m256d r2 = *o2r;
    *o2r = *o2i;
    *o2i = _mm256_xor_pd(r2, neg0);

    // o3 *= -c(1 + i) = -c·(re - im) + i·(-c)·(re + im)
    __m256d s3 = _mm256_sub_pd(*o3r, *o3i);
    __m256d d3 = _mm256_add_pd(*o3r, *o3i);
    *o3r = _mm256_xor_pd(_mm256_mul_pd(c, s3), neg0);
    *o3i = _mm256_xor_pd(_mm256_mul_pd(c, d3), neg0);
}

TARGET_AVX2_FMA
FORCE_INLINE void
w8_apply_fast_backward_avx2(
    __m256d *RESTRICT o1r, __m256d *RESTRICT o1i,
    __m256d *RESTRICT o2r, __m256d *RESTRICT o2i,
    __m256d *RESTRICT o3r, __m256d *RESTRICT o3i)
{
    const __m256d c = _mm256_set1_pd(C8_CONSTANT);
    const __m256d neg0 = _mm256_set1_pd(-0.0);

    // o1 *= c(1 + i) = c·(re - im) + i·c·(re + im)
    __m256d s1 = _mm256_sub_pd(*o1r, *o1i);
    __m256d d1 = _mm256_add_pd(*o1r, *o1i);
    *o1r = _mm256_mul_pd(c, s1);
    *o1i = _mm256_mul_pd(c, d1);

    // o2 *= +i (negate + swap)
    __m256d r2 = *o2r;
    *o2r = _mm256_xor_pd(*o2i, neg0);
    *o2i = r2;

    // o3 *= -c(1 - i) = -c·(re + im) + i·(-c)·(im - re)
    __m256d s3 = _mm256_add_pd(*o3r, *o3i);
    __m256d d3 = _mm256_sub_pd(*o3i, *o3r);
    *o3r = _mm256_xor_pd(_mm256_mul_pd(c, s3), neg0);
    *o3i = _mm256_xor_pd(_mm256_mul_pd(c, d3), neg0);
}

//==============================================================================
// PREFETCH DISTANCE TUNING (SEPARATE FOR BLOCKED4/BLOCKED2)
//==============================================================================

/**
 * @def RADIX8_PREFETCH_DISTANCE_AVX2_B4
 * @brief Prefetch distance for BLOCKED4 (denser memory streams)
 */
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2_B4
#define RADIX8_PREFETCH_DISTANCE_AVX2_B4 24
#endif

/**
 * @def RADIX8_PREFETCH_DISTANCE_AVX2_B2
 * @brief Prefetch distance for BLOCKED2 (sparser memory streams)
 */
#ifndef RADIX8_PREFETCH_DISTANCE_AVX2_B2
#define RADIX8_PREFETCH_DISTANCE_AVX2_B2 32
#endif

//==============================================================================
// STAGE DRIVERS WITH TRUE U=2 PIPELINING + ALL OPTIMIZATIONS
//==============================================================================

/**
 * @brief BLOCKED4 Forward with True U=2 Software Pipelining (AVX2)
 *
 * Peak: 16 YMM (split nx_odd loads + transient W8)
 *
 * Optimizations:
 * ✅ U=2 software pipelining (overlapped loads/compute)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ NTA prefetch for streaming workloads
 * ✅ Fast W8 micro-kernels (add/sub instead of cmul)
 * ✅ Prefetch tuning (24 doubles for BLOCKED4)
 * ✅ Hoisted constants
 * ✅ Two-wave stores (control register pressure)
 * ✅ Transient constants (SIGN_FLIP, W8)
 * ✅ Unroll disable (preserve instruction scheduling)
 * ✅ zeroupper after NT stores (avoid AVX→SSE penalty)
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void
radix8_stage_blocked4_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    // Alignment checks
    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

    // Adaptive load/store
#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    // NT store decision + adaptive prefetch hint
    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    // Pointer setup for AGU optimization
    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE: Load first iteration
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);
    __m256d nW3r = _mm256_load_pd(&re_base[2 * K]);
    __m256d nW3i = _mm256_load_pd(&im_base[2 * K]);
    __m256d nW4r = _mm256_load_pd(&re_base[3 * K]);
    __m256d nW4i = _mm256_load_pd(&im_base[3 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        // Current iteration: consume nx* from previous iteration
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED4: Direct loads, no derivation)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            // Apply W1..W4 (loaded from memory)
            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            // Derive W5=-W1, W6=-W2, W7=-W3 (compute at use)
            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs (while twiddles compute)
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
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                             &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                             SIGN_FLIP);
        }

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs (smooth LD port usage)
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Odd Radix-4
        //======================================================================
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             SIGN_FLIP);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (FAST micro-kernel - no cmul)
        //======================================================================
        w8_apply_fast_forward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        //======================================================================
        // STAGE 7: Combine & Store in Two Waves (control register pressure)
        //======================================================================
        // Wave A: Store y0, y1 → frees e0, e1, o0, o1 (8 YMM)
        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));

        // Load remaining NEXT ODD (nx5, nx7) - now we have room
        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        // Wave B: Store remaining outputs
        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));

        //======================================================================
        // STAGE 8: Load Next Twiddles (all 4 blocks for BLOCKED4)
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);
        nW3r = _mm256_load_pd(&re_base[2 * K + kn]);
        nW3i = _mm256_load_pd(&im_base[2 * K + kn]);
        nW4r = _mm256_load_pd(&re_base[3 * K + kn]);
        nW4i = _mm256_load_pd(&im_base[3 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch (4 twiddle blocks for BLOCKED4)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[3 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE: Final iteration (no next loads needed)
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                             &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                             SIGN_FLIP);
        }

        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             SIGN_FLIP);
        }

        w8_apply_fast_forward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));
    }

    // Cleanup: sfence + zeroupper for NT stores
    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief BLOCKED2 Forward with True U=2 Software Pipelining (AVX2)
 *
 * Peak: ~14 YMM (controlled via staged loads)
 *
 * BLOCKED2: Loads only W1, W2; derives W3=W1×W2, W4=W2²
 * Bandwidth savings: 71% (load 2 blocks instead of 7)
 *
 * Optimizations:
 * ✅ U=2 software pipelining (overlapped loads/compute)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ NTA prefetch for streaming workloads
 * ✅ Fast W8 micro-kernels (add/sub instead of cmul)
 * ✅ Prefetch tuning (32 doubles for BLOCKED2 - sparser streams)
 * ✅ In-place twiddle derivation (saves registers)
 * ✅ Two-wave stores (control register pressure)
 * ✅ Transient constants
 * ✅ Unroll disable
 * ✅ zeroupper after NT stores
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void
radix8_stage_blocked2_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED2: Derive W3, W4)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            // Apply W1, W2 (loaded from memory)
            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            // Derive W3 = W1 × W2
            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            // Derive W4 = W2²
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            // Apply W3, W4
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            // Derive W5=-W1, W6=-W2, W7=-W3 (compute at use)
            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
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
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
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
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             SIGN_FLIP);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (FAST micro-kernel)
        //======================================================================
        w8_apply_fast_forward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        //======================================================================
        // STAGE 7: Combine & Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));

        //======================================================================
        // STAGE 8: Load Next Twiddles (only 2 blocks for BLOCKED2)
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch (2 twiddle blocks for BLOCKED2)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                             &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                             SIGN_FLIP);
        }

        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             SIGN_FLIP);
        }

        w8_apply_fast_forward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief BLOCKED4 Backward with True U=2 Software Pipelining (AVX2)
 *
 * Changes from forward:
 * ✅ Radix-4 sign mask: Negated for IDFT (backward transform)
 * ✅ W8 application: Call w8_apply_fast_backward_avx2
 *
 * All other optimizations identical to forward version.
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void
radix8_stage_blocked4_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);
    __m256d nW3r = _mm256_load_pd(&re_base[2 * K]);
    __m256d nW3i = _mm256_load_pd(&im_base[2 * K]);
    __m256d nW4r = _mm256_load_pd(&re_base[3 * K]);
    __m256d nW4i = _mm256_load_pd(&im_base[3 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (identical to forward)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
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
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
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
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             neg_sign);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (BACKWARD version)
        //======================================================================
        w8_apply_fast_backward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        //======================================================================
        // STAGE 7: Combine & Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));

        //======================================================================
        // STAGE 8: Load Next Twiddles
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);
        nW3r = _mm256_load_pd(&re_base[2 * K + kn]);
        nW3i = _mm256_load_pd(&im_base[2 * K + kn]);
        nW4r = _mm256_load_pd(&re_base[3 * K + kn]);
        nW4i = _mm256_load_pd(&im_base[3 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[3 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                             &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                             neg_sign);
        }

        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             neg_sign);
        }

        w8_apply_fast_backward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief BLOCKED2 Backward with True U=2 Software Pipelining (AVX2)
 *
 * Changes from forward:
 * ✅ Radix-4 sign mask: Negated for IDFT (backward transform)
 * ✅ W8 application: Call w8_apply_fast_backward_avx2
 *
 * All other optimizations identical to BLOCKED2 forward version.
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void
radix8_stage_blocked2_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (identical to forward)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
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
        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
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
        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             neg_sign);
        }

        //======================================================================
        // STAGE 6: Apply W8 Twiddles (BACKWARD version)
        //======================================================================
        w8_apply_fast_backward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        //======================================================================
        // STAGE 7: Combine & Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));

        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));

        //======================================================================
        // STAGE 8: Load Next Twiddles
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        //======================================================================
        // STAGE 9: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i,
                             &e0r, &e0i, &e1r, &e1i, &e2r, &e2i, &e3r, &e3i,
                             neg_sign);
        }

        __m256d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
            const __m256d neg_zero = _mm256_set1_pd(-0.0);
            const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

            radix4_core_avx2(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i,
                             &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                             neg_sign);
        }

        w8_apply_fast_backward_avx2(&o1r, &o1i, &o2r, &o2i, &o3r, &o3i);

        ST_STREAM(&out_re[0 * K + k], _mm256_add_pd(e0r, o0r));
        ST_STREAM(&out_im[0 * K + k], _mm256_add_pd(e0i, o0i));
        ST_STREAM(&out_re[1 * K + k], _mm256_add_pd(e1r, o1r));
        ST_STREAM(&out_im[1 * K + k], _mm256_add_pd(e1i, o1i));
        ST_STREAM(&out_re[2 * K + k], _mm256_add_pd(e2r, o2r));
        ST_STREAM(&out_im[2 * K + k], _mm256_add_pd(e2i, o2i));
        ST_STREAM(&out_re[3 * K + k], _mm256_add_pd(e3r, o3r));
        ST_STREAM(&out_im[3 * K + k], _mm256_add_pd(e3i, o3i));
        ST_STREAM(&out_re[4 * K + k], _mm256_sub_pd(e0r, o0r));
        ST_STREAM(&out_im[4 * K + k], _mm256_sub_pd(e0i, o0i));
        ST_STREAM(&out_re[5 * K + k], _mm256_sub_pd(e1r, o1r));
        ST_STREAM(&out_im[5 * K + k], _mm256_sub_pd(e1i, o1i));
        ST_STREAM(&out_re[6 * K + k], _mm256_sub_pd(e2r, o2r));
        ST_STREAM(&out_im[6 * K + k], _mm256_sub_pd(e2i, o2i));
        ST_STREAM(&out_re[7 * K + k], _mm256_sub_pd(e3r, o3r));
        ST_STREAM(&out_im[7 * K + k], _mm256_sub_pd(e3i, o3i));
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}


//==============================================================================
// OPTIMIZATION SUMMARY
//==============================================================================

/*
 * COMPLETE AVX2 RADIX-8 IMPLEMENTATION - ALL OPTIMIZATIONS APPLIED
 * =================================================================
 *
 * ✅ TRUE U=2 SOFTWARE PIPELINING
 *    - Load k+4 while computing k
 *    - Staged loads: even inputs → half odd → remaining odd
 *    - Two-wave stores to control register pressure
 *    - Twiddles loaded at end of iteration
 *
 * ✅ FAST W8 MICRO-KERNELS
 *    - Replaced complex FMAs with add/sub operations
 *    - 2 FMAs saved per butterfly
 *    - Shorter critical path (add = 3 cycles vs FMA = 4 cycles)
 *
 * ✅ ADAPTIVE NT STORES
 *    - Enabled for transforms >256KB
 *    - Bypasses cache for write-only output path
 *    - Includes _mm_sfence() + _mm256_zeroupper()
 *
 * ✅ NTA PREFETCH
 *    - Non-temporal prefetch when NT storing
 *    - Cuts L1D pollution on streaming workloads
 *    - Adaptive hint: _MM_HINT_NTA vs _MM_HINT_T0
 *
 * ✅ TUNED PREFETCH DISTANCES
 *    - BLOCKED4: 24 doubles (denser memory streams)
 *    - BLOCKED2: 32 doubles (sparser memory streams)
 *    - Separate macros for independent tuning
 *
 * ✅ POINTER BUMPING (AGU OPTIMIZATION)
 *    - Precomputed aligned base pointers
 *    - Reduces 64-bit LEA operations in loop body
 *    - Better AGU utilization (only 2 AGUs on pre-Sunny Cove)
 *
 * ✅ UNROLL DISABLE
 *    - Pragmas to prevent compiler over-unrolling
 *    - Preserves carefully staged instruction scheduling
 *    - Prevents register pressure spills
 *
 * ✅ TRANSIENT CONSTANTS
 *    - SIGN_FLIP, W8 constants in local scopes
 *    - Compiler can CSE but semantic clarity maintained
 *    - Reduces live ranges in critical sections
 *
 * ✅ HYBRID TWIDDLE SYSTEM PRESERVED
 *    - BLOCKED4: Load 4 blocks, derive W5-W7 via sign flips (43% savings)
 *    - BLOCKED2: Load 2 blocks, derive W3-W7 via cmul/square (71% savings)
 *    - Automatic mode selection based on K threshold
 *
 * ✅ ALIGNMENT EXPLOITATION
 *    - ASSUME_ALIGNED on twiddle pointers
 *    - Adaptive aligned/unaligned loads for input
 *    - Aligned stores when output pointer is 32-byte aligned
 *
 * REGISTER BUDGET (AVX2 - 16 YMM):
 * ================================
 * BLOCKED4 Peak: ~14-16 YMM (controlled via staged loads)
 * BLOCKED2 Peak: ~14 YMM (W3/W4 derived on-the-fly)
 *
 * Both modes stay comfortably within 16 YMM register limit through:
 * - Two-wave stores (free registers between waves)
 * - Split odd loads (nx1/nx3 then nx5/nx7)
 * - Transient constants (short live ranges)
 *
 * PERFORMANCE CHARACTERISTICS:
 * ============================
 * - Memory bandwidth: 43-71% savings vs naive 7-block twiddles
 * - Computational cost: ~48 FLOPs per radix-8 butterfly
 * - L1D hits: ~4-5 cycles latency hidden by U=2 pipelining
 * - Cache pollution: Minimized via NTA prefetch + NT stores
 *
 * ARCHITECTURAL TARGETS:
 * =====================
 * - Primary: Haswell, Broadwell, Skylake, Cascade Lake
 * - Compatible: Any x86-64 CPU with AVX2 + FMA
 * - Optimal: CPUs with 2 load ports, 1 store port, 2 FMA units
 */

#endif // FFT_RADIX8_AVX2_BLOCKED_HYBRID_FIXED_H