/**
 * @file fft_radix8_avx512_n1.h
 * @brief Radix-8 N=1 (Twiddle-less) AVX-512 - FFTW-Style Base Codelet
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
 * - AVX-512 processes 8 butterflies simultaneously (8 doubles per vector)
 * - K must be multiple of 8
 * - Iteration k=0,8,16,... processes butterflies [k, k+1, ..., k+7]
 *
 * PRESERVED OPTIMIZATIONS FROM FULL VERSION:
 * ==========================================
 * ✅ U=2 software pipelining (prefetch overlap)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (56 doubles ahead - double AVX2's 28)
 * ✅ Hoisted constants (W8, sign masks)
 * ✅ Alignment hints (ASSUME_ALIGNED, 64-byte)
 * ✅ Target attributes (explicit AVX-512 FMA)
 * ✅ Radix-4 decomposition (2x radix-4 + W8 twiddles)
 * ✅ Optimized radix-4 core (FMA-based)
 * ✅ Cache-conscious streaming
 *
 * @author FFT Optimization Team
 * @version 1.0-N1-AVX512 (Derived from radix8_avx512_blocked_hybrid_fixed)
 * @date 2025
 */

#ifndef FFT_RADIX8_AVX512_N1_H
#define FFT_RADIX8_AVX512_N1_H

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
// CONFIGURATION (PRESERVED FROM ORIGINAL)
//==============================================================================

/**
 * @def RADIX8_N1_STREAM_THRESHOLD_KB_AVX512
 * @brief NT store threshold for N=1 codelets (in KB)
 */
#ifndef RADIX8_N1_STREAM_THRESHOLD_KB_AVX512
#define RADIX8_N1_STREAM_THRESHOLD_KB_AVX512 256
#endif

/**
 * @def RADIX8_N1_PREFETCH_DISTANCE_AVX512
 * @brief Prefetch distance for AVX-512 (56 doubles - tuned for Skylake-SP/Cascade Lake)
 */
#ifndef RADIX8_N1_PREFETCH_DISTANCE_AVX512
#define RADIX8_N1_PREFETCH_DISTANCE_AVX512 56
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
 * @brief Complex multiplication (SoA layout, optimal for AVX-512)
 * @note (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
 * @note Uses FMA for optimal throughput on dual FMA ports (Skylake+)
 */
TARGET_AVX512_FMA
FORCE_INLINE void
cmul_v512(__m512d ar, __m512d ai, __m512d br, __m512d bi,
          __m512d *RESTRICT tr, __m512d *RESTRICT ti)
{
    *tr = _mm512_fmsub_pd(ar, br, _mm512_mul_pd(ai, bi));  // ar*br - ai*bi
    *ti = _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br));  // ar*bi + ai*br
}

/**
 * @brief Radix-4 core butterfly (DIT, Cooley-Tukey)
 * @note Uses FMA extensively for minimal latency
 * @note sign_mask controls forward (-0.0) vs backward (flipped) transform
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
    // Stage 1: Radix-2 butterflies
    __m512d t0_re = _mm512_add_pd(x0_re, x2_re);
    __m512d t0_im = _mm512_add_pd(x0_im, x2_im);
    __m512d t1_re = _mm512_sub_pd(x0_re, x2_re);
    __m512d t1_im = _mm512_sub_pd(x0_im, x2_im);
    __m512d t2_re = _mm512_add_pd(x1_re, x3_re);
    __m512d t2_im = _mm512_add_pd(x1_im, x3_im);
    __m512d t3_re = _mm512_sub_pd(x1_re, x3_re);
    __m512d t3_im = _mm512_sub_pd(x1_im, x3_im);

    // Stage 2: Combine with W_4 twiddles (rotation by ±i)
    *y0_re = _mm512_add_pd(t0_re, t2_re);
    *y0_im = _mm512_add_pd(t0_im, t2_im);
    *y1_re = _mm512_sub_pd(t1_re, _mm512_xor_pd(t3_im, sign_mask));  // t1 - sign*i*t3
    *y1_im = _mm512_add_pd(t1_im, _mm512_xor_pd(t3_re, sign_mask));
    *y2_re = _mm512_sub_pd(t0_re, t2_re);
    *y2_im = _mm512_sub_pd(t0_im, t2_im);
    *y3_re = _mm512_add_pd(t1_re, _mm512_xor_pd(t3_im, sign_mask));  // t1 + sign*i*t3
    *y3_im = _mm512_sub_pd(t1_im, _mm512_xor_pd(t3_re, sign_mask));
}

/**
 * @brief Apply W_8 twiddles for forward transform (after first radix-4)
 * @note W_8^1, W_8^2=(0,-1), W_8^3 applied to odd-indexed outputs
 */
TARGET_AVX512_FMA
FORCE_INLINE void
apply_w8_twiddles_forward_avx512(
    __m512d *RESTRICT o1_re, __m512d *RESTRICT o1_im,
    __m512d *RESTRICT o2_re, __m512d *RESTRICT o2_im,
    __m512d *RESTRICT o3_re, __m512d *RESTRICT o3_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im)
{
    // W_8^1 multiplication (full complex multiply)
    __m512d r1 = *o1_re, i1 = *o1_im;
    *o1_re = _mm512_fmsub_pd(r1, W8_1_re, _mm512_mul_pd(i1, W8_1_im));
    *o1_im = _mm512_fmadd_pd(r1, W8_1_im, _mm512_mul_pd(i1, W8_1_re));

    // W_8^2 = (0, -1) - optimized as swap + negate
    const __m512d neg_zero = _mm512_set1_pd(-0.0);
    __m512d r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = _mm512_xor_pd(r2, neg_zero);

    // W_8^3 multiplication (full complex multiply)
    __m512d r3 = *o3_re, i3 = *o3_im;
    *o3_re = _mm512_fmsub_pd(r3, W8_3_re, _mm512_mul_pd(i3, W8_3_im));
    *o3_im = _mm512_fmadd_pd(r3, W8_3_im, _mm512_mul_pd(i3, W8_3_re));
}

/**
 * @brief Apply W_8 twiddles for backward transform (conjugate twiddles)
 * @note W_8^(-1), W_8^(-2)=(0,1), W_8^(-3) applied to odd-indexed outputs
 */
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

    // W_8^(-2) = (0, 1) - optimized as negate + swap
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
// N=1 BUTTERFLY FUNCTIONS (NO STAGE TWIDDLES)
//==============================================================================

/**
 * @brief Single N=1 radix-8 butterfly - FORWARD - Regular stores
 * @param k Current position in K-parallel butterflies (must be multiple of 8)
 * @param K Total number of parallel butterflies (must be multiple of 8)
 * @note NO stage twiddles - this is the defining characteristic of N=1
 */
TARGET_AVX512_FMA
FORCE_INLINE void
radix8_n1_butterfly_forward_avx512(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    // Load 8 complex inputs (8 butterflies in parallel - full AVX-512 width)
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

    // N=1: NO stage twiddle application here (that's the whole point!)
    // Directly proceed to radix-8 decomposition: 2x radix-4 + W8 twiddles

    // First radix-4: even-indexed inputs (x0, x2, x4, x6)
    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    // Second radix-4: odd-indexed inputs (x1, x3, x5, x7)
    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    // Apply W_8 twiddles to odd outputs
    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination: y[k] = e[k] + o[k], y[k+4] = e[k] - o[k]
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

/**
 * @brief Single N=1 radix-8 butterfly - FORWARD - NT stores (streaming)
 * @note Identical computation to regular version, but uses _mm512_stream_pd
 */
TARGET_AVX512_FMA
FORCE_INLINE void
radix8_n1_butterfly_forward_avx512_nt(
    size_t k, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    __m512d W8_1_re, __m512d W8_1_im,
    __m512d W8_3_re, __m512d W8_3_im,
    __m512d sign_mask)
{
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    // Load 8 complex inputs (identical to regular version)
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

    // First radix-4: even-indexed inputs
    __m512d e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_avx512(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im,
                       sign_mask);

    // Second radix-4: odd-indexed inputs
    __m512d o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_avx512(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                       sign_mask);

    // Apply W_8 twiddles to odd outputs
    apply_w8_twiddles_forward_avx512(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_1_re, W8_1_im, W8_3_re, W8_3_im);

    // Final combination with NON-TEMPORAL stores (streaming)
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