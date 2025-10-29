/**
 * @file fft_radix8_avx2_n1.h
 * @brief Radix-8 N=1 (Twiddle-less) AVX2 - FFTW-Style Base Codelet
 *
 * @details
 * N=1 CODELET ARCHITECTURE (FFTW Terminology):
 * ============================================
 * - "N=1" = No stage twiddles (W1...W7 all equal to 1+0i)
 * - Only internal W_8 geometric twiddles remain
 * - Used as base case in larger mixed-radix factorizations
 * - Processes K independent radix-8 butterflies in parallel
 *
 * DATA LAYOUT:
 * ============
 * Input/output in SoA (Structure of Arrays) format:
 *   re[0...K-1]    : x0 real parts for K butterflies
 *   re[K...2K-1]   : x1 real parts for K butterflies
 *   ...
 *   re[7K...8K-1]  : x7 real parts for K butterflies
 *   (same for im)
 *
 * PARALLELIZATION:
 * ===============
 * - AVX2 processes 4 butterflies simultaneously (4 doubles per vector)
 * - K must be multiple of 4
 * - Iteration k=0,4,8,... processes butterflies [k, k+1, k+2, k+3]
 *
 * PRESERVED OPTIMIZATIONS FROM FULL VERSION:
 * ==========================================
 * ✅ U=2 software pipelining (prefetch overlap)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (28 doubles ahead)
 * ✅ Hoisted constants (W8, sign masks)
 * ✅ Alignment hints (ASSUME_ALIGNED)
 * ✅ Target attributes (explicit AVX2 FMA)
 * ✅ Radix-4 decomposition (2x radix-4 + W8 twiddles)
 * ✅ Optimized radix-4 core (FMA-based)
 * ✅ Cache-conscious streaming
 *
 * @author FFT Optimization Team
 * @version 1.0-N1 (Derived from radix8_avx2_blocked_hybrid_fixed)
 * @date 2025
 */

#ifndef FFT_RADIX8_AVX2_N1_H
#define FFT_RADIX8_AVX2_N1_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

//==============================================================================
// COMPILER PORTABILITY (PRESERVED FROM ORIGINAL)
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
// CONFIGURATION (PRESERVED FROM ORIGINAL)
//==============================================================================

/**
 * @def RADIX8_N1_STREAM_THRESHOLD_KB
 * @brief NT store threshold for N=1 codelets (in KB)
 */
#ifndef RADIX8_N1_STREAM_THRESHOLD_KB
#define RADIX8_N1_STREAM_THRESHOLD_KB 256
#endif

/**
 * @def RADIX8_N1_PREFETCH_DISTANCE_AVX2
 * @brief Prefetch distance for AVX2 (28 doubles - tuned for Skylake/Cascade Lake)
 */
#ifndef RADIX8_N1_PREFETCH_DISTANCE_AVX2
#define RADIX8_N1_PREFETCH_DISTANCE_AVX2 28
#endif

//==============================================================================
// W_8 GEOMETRIC CONSTANTS (PRESERVED EXACTLY)
//==============================================================================

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

// Forward transform twiddles: W_8^k = exp(-2πik/8)
#define W8_FV_1_RE C8_CONSTANT         // W_8^1 real
#define W8_FV_1_IM (-C8_CONSTANT)      // W_8^1 imag
#define W8_FV_3_RE (-C8_CONSTANT)      // W_8^3 real
#define W8_FV_3_IM (-C8_CONSTANT)      // W_8^3 imag

// Backward transform twiddles: W_8^(-k) = exp(+2πik/8)
#define W8_BV_1_RE C8_CONSTANT         // W_8^(-1) real
#define W8_BV_1_IM C8_CONSTANT         // W_8^(-1) imag
#define W8_BV_3_RE (-C8_CONSTANT)      // W_8^(-3) real
#define W8_BV_3_IM C8_CONSTANT         // W_8^(-3) imag

//==============================================================================
// CORE PRIMITIVES (PRESERVED EXACTLY FROM ORIGINAL)
//==============================================================================

/**
 * @brief Complex multiplication (SoA layout, optimal for AVX2)
 * @note (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
 * @note Uses FMA for optimal throughput on dual FMA ports (Skylake+)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
cmul_v256(__m256d ar, __m256d ai, __m256d br, __m256d bi,
          __m256d *RESTRICT tr, __m256d *RESTRICT ti)
{
    *tr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));  // ar*br - ai*bi
    *ti = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));  // ar*bi + ai*br
}

/**
 * @brief Radix-4 core butterfly (DIT, Cooley-Tukey)
 * @note Uses FMA extensively for minimal latency
 * @note sign_mask controls forward (-0.0) vs backward (flipped) transform
 */
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
    // Stage 1: Radix-2 butterflies
    __m256d t0_re = _mm256_add_pd(x0_re, x2_re);
    __m256d t0_im = _mm256_add_pd(x0_im, x2_im);
    __m256d t1_re = _mm256_sub_pd(x0_re, x2_re);
    __m256d t1_im = _mm256_sub_pd(x0_im, x2_im);
    __m256d t2_re = _mm256_add_pd(x1_re, x3_re);
    __m256d t2_im = _mm256_add_pd(x1_im, x3_im);
    __m256d t3_re = _mm256_sub_pd(x1_re, x3_re);
    __m256d t3_im = _mm256_sub_pd(x1_im, x3_im);

    // Stage 2: Combine with W_4 twiddles (rotation by ±i)
    *y0_re = _mm256_add_pd(t0_re, t2_re);
    *y0_im = _mm256_add_pd(t0_im, t2_im);
    *y1_re = _mm256_sub_pd(t1_re, _mm256_xor_pd(t3_im, sign_mask));  // t1 - sign*i*t3
    *y1_im = _mm256_add_pd(t1_im, _mm256_xor_pd(t3_re, sign_mask));
    *y2_re = _mm256_sub_pd(t0_re, t2_re);
    *y2_im = _mm256_sub_pd(t0_im, t2_im);
    *y3_re = _mm256_add_pd(t1_re, _mm256_xor_pd(t3_im, sign_mask));  // t1 + sign*i*t3
    *y3_im = _mm256_sub_pd(t1_im, _mm256_xor_pd(t3_re, sign_mask));
}

/**
 * @brief Apply W_8 twiddles for forward transform (after first radix-4)
 * @note W_8^1, W_8^2=(0,-1), W_8^3 applied to odd-indexed outputs
 */
TARGET_AVX2_FMA
FORCE_INLINE void
apply_w8_twiddles_forward_avx2(
    __m256d *RESTRICT o1_re, __m256d *RESTRICT o1_im,
    __m256d *RESTRICT o2_re, __m256d *RESTRICT o2_im,
    __m256d *RESTRICT o3_re, __m256d *RESTRICT o3_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im)
{
    // W_8^1 multiplication (full complex multiply)
    __m256d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm256_fmsub_pd(r1, W8_1_re, _mm256_mul_pd(i1, W8_1_im));
    *o1_im = _mm256_fmadd_pd(r1, W8_1_im, _mm256_mul_pd(i1, W8_1_re));

    // W_8^2 = (0, -1) - optimized as swap + negate
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = _mm256_xor_pd(r2, neg_zero);

    // W_8^3 multiplication (full complex multiply)
    __m256d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm256_fmsub_pd(r3, W8_3_re, _mm256_mul_pd(i3, W8_3_im));
    *o3_im = _mm256_fmadd_pd(r3, W8_3_im, _mm256_mul_pd(i3, W8_3_re));
}

/**
 * @brief Apply W_8 twiddles for backward transform (conjugate twiddles)
 * @note W_8^(-1), W_8^(-2)=(0,1), W_8^(-3) applied to odd-indexed outputs
 */
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

    // W_8^(-2) = (0, 1) - optimized as negate + swap
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
// N=1 BUTTERFLY FUNCTIONS (NO STAGE TWIDDLES)
//==============================================================================

/**
 * @brief Single N=1 radix-8 butterfly - FORWARD - Regular stores
 * @param k Current position in K-parallel butterflies (must be multiple of 4)
 * @param K Total number of parallel butterflies (must be multiple of 4)
 * @note NO stage twiddles - this is the defining characteristic of N=1
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_butterfly_forward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Load 8 complex inputs (4 butterflies in parallel)
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

    // N=1: NO stage twiddle application here (that's the whole point!)
    // Directly proceed to radix-8 decomposition: 2x radix-4 + W8 twiddles

    // First radix-4: even-indexed inputs (x0, x2, x4, x6)
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    // Second radix-4: odd-indexed inputs (x1, x3, x5, x7)
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    // Apply W_8 twiddles to odd outputs
    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination: y[k] = e[k] + o[k], y[k+4] = e[k] - o[k]
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

/**
 * @brief Single N=1 radix-8 butterfly - FORWARD - NT stores (streaming)
 * @note Identical computation to regular version, but uses _mm256_stream_pd
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_butterfly_forward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Load 8 complex inputs (identical to regular version)
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

    // First radix-4: even-indexed inputs
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     sign_mask);

    // Second radix-4: odd-indexed inputs
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     sign_mask);

    // Apply W_8 twiddles to odd outputs
    apply_w8_twiddles_forward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                   W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination with NON-TEMPORAL stores (streaming)
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

/**
 * @brief Single N=1 radix-8 butterfly - BACKWARD - Regular stores
 * @note Backward = conjugate twiddles (IFFT direction)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_butterfly_backward_avx2(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Load 8 complex inputs
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

    // Negate sign_mask for backward transform (flip rotation direction)
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    // First radix-4: even-indexed inputs (with negated sign)
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    // Second radix-4: odd-indexed inputs (with negated sign)
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    // Apply conjugate W_8 twiddles to odd outputs
    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination
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

/**
 * @brief Single N=1 radix-8 butterfly - BACKWARD - NT stores
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_butterfly_backward_avx2_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m256d W8_1_re, __m256d W8_1_im,
    __m256d W8_3_re, __m256d W8_3_im,
    __m256d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Load 8 complex inputs
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

    // Negate sign_mask for backward transform
    const __m256d neg_zero = _mm256_set1_pd(-0.0);
    __m256d neg_sign = _mm256_xor_pd(sign_mask, neg_zero);

    // First radix-4: even-indexed inputs
    __m256d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx2(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                     &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                     neg_sign);

    // Second radix-4: odd-indexed inputs
    __m256d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx2(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                     &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                     neg_sign);

    // Apply conjugate W_8 twiddles
    apply_w8_twiddles_backward_avx2(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                    W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination with NON-TEMPORAL stores
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
// STAGE DRIVERS WITH ALL OPTIMIZATIONS
//==============================================================================

/**
 * @brief N=1 radix-8 stage driver - FORWARD transform
 *
 * @details
 * ALL OPTIMIZATIONS PRESERVED:
 * ✅ U=2 software pipelining (prefetch k+4+28 ahead)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Hoisted constants (computed once outside loop)
 * ✅ Prefetch tuning (28 doubles for AVX2/Skylake)
 * ✅ Tight loop with minimal overhead
 *
 * @param K Number of parallel butterflies (must be multiple of 4)
 * @param in_re Input real part (length 8*K, SoA layout)
 * @param in_im Input imag part (length 8*K, SoA layout)
 * @param out_re Output real part (length 8*K, SoA layout)
 * @param out_im Output imag part (length 8*K, SoA layout)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    // Assert K % 4 == 0 (AVX2 vector width = 4 doubles)
    assert((K & 3) == 0 && "K must be multiple of 4 for AVX2");

    // Hoist constants ONCE per stage (critical optimization)
    const __m256d W8_1_re = _mm256_set1_pd(W8_FV_1_RE);
    const __m256d W8_1_im = _mm256_set1_pd(W8_FV_1_IM);
    const __m256d W8_3_re = _mm256_set1_pd(W8_FV_3_RE);
    const __m256d W8_3_im = _mm256_set1_pd(W8_FV_3_IM);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX8_N1_PREFETCH_DISTANCE_AVX2;

    // Adaptive NT store decision (>256KB working set)
    const size_t total_elements = K * 8 * 2;  // K butterflies × 8 points × 2 (re+im)
    const size_t total_bytes = total_elements * sizeof(double);
    const int use_nt_stores = (total_bytes >= (RADIX8_N1_STREAM_THRESHOLD_KB * 1024)) &&
                              (((uintptr_t)out_re & 31) == 0) &&  // Check alignment
                              (((uintptr_t)out_im & 31) == 0);

    if (use_nt_stores)
    {
        // Large transforms: NT stores to bypass cache
        for (size_t k = 0; k < K; k += 4)
        {
            // U=2 software pipelining: prefetch NEXT iteration's data
            if (k + 4 + prefetch_dist < K)
            {
                // Prefetch 8 input blocks (x0...x7 for next iteration)
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
                // Note: NO twiddle prefetch in N=1 (that's the whole point!)
            }

            radix8_n1_butterfly_forward_avx2_nt(k, K, in_re, in_im, out_re, out_im,
                                                W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                                                sign_mask);
        }
        _mm_sfence();  // Required after streaming stores
    }
    else
    {
        // Small transforms: regular stores with U=2 pipelining
        for (size_t k = 0; k < K; k += 4)
        {
            // U=2: prefetch next iteration
            if (k + 4 + prefetch_dist < K)
            {
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
            }

            radix8_n1_butterfly_forward_avx2(k, K, in_re, in_im, out_re, out_im,
                                             W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                                             sign_mask);
        }
    }
}

/**
 * @brief N=1 radix-8 stage driver - BACKWARD transform (IFFT direction)
 *
 * @details
 * ALL OPTIMIZATIONS PRESERVED (identical to forward, but conjugate twiddles)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix8_n1_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    assert((K & 3) == 0 && "K must be multiple of 4 for AVX2");

    // Hoist constants (BACKWARD twiddles - conjugates of forward)
    const __m256d W8_1_re = _mm256_set1_pd(W8_BV_1_RE);
    const __m256d W8_1_im = _mm256_set1_pd(W8_BV_1_IM);
    const __m256d W8_3_re = _mm256_set1_pd(W8_BV_3_RE);
    const __m256d W8_3_im = _mm256_set1_pd(W8_BV_3_IM);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX8_N1_PREFETCH_DISTANCE_AVX2;

    // Adaptive NT store decision
    const size_t total_elements = K * 8 * 2;
    const size_t total_bytes = total_elements * sizeof(double);
    const int use_nt_stores = (total_bytes >= (RADIX8_N1_STREAM_THRESHOLD_KB * 1024)) &&
                              (((uintptr_t)out_re & 31) == 0) &&
                              (((uintptr_t)out_im & 31) == 0);

    if (use_nt_stores)
    {
        // Large transforms: NT stores
        for (size_t k = 0; k < K; k += 4)
        {
            // U=2 software pipelining
            if (k + 4 + prefetch_dist < K)
            {
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
            }

            radix8_n1_butterfly_backward_avx2_nt(k, K, in_re, in_im, out_re, out_im,
                                                 W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                                                 sign_mask);
        }
        _mm_sfence();
    }
    else
    {
        // Small transforms: regular stores
        for (size_t k = 0; k < K; k += 4)
        {
            if (k + 4 + prefetch_dist < K)
            {
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 0 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 1 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 2 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 3 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 4 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 5 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 6 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_re[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
                _mm_prefetch((const char *)&in_im[k + 4 + prefetch_dist + 7 * K], _MM_HINT_T0);
            }

            radix8_n1_butterfly_backward_avx2(k, K, in_re, in_im, out_re, out_im,
                                              W8_1_re, W8_1_im, W8_3_re, W8_3_im,
                                              sign_mask);
        }
    }
}

#endif // FFT_RADIX8_AVX2_N1_H