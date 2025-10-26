/**
 * @file radix3_sse2_optimized.h
 * @brief Complete SSE2 Radix-3 Implementation (Macros + Stages + Tail Handling)
 *
 * COMPREHENSIVE FALLBACK IMPLEMENTATION:
 * ======================================
 * This file contains everything needed for SSE2 fallback:
 * - Butterfly macros (with and without constant parameters)
 * - Stage-level functions with AGU optimization
 * - Tail handling for odd-sized transforms
 * - Forward and backward transforms
 *
 * NO U=2 PIPELINING: SSE2 is for compatibility/fallback, not peak performance
 *
 * CRITICAL DIFFERENCE FROM AVX2/AVX-512:
 * ======================================
 * SSE2 does NOT have FMA instructions!
 * - Must use separate MUL + ADD/SUB operations
 * - Negative multiply becomes: SUB(0, MUL) instead of FNMADD
 * - Slightly higher instruction count, but same optimization principles
 *
 * OPTIMIZATIONS APPLIED:
 * ======================
 * 1. Constant elimination: SUB(0, MUL) instead of separate negative constant
 * 2. Micro-scheduling: Start rot_re BEFORE common_* to reduce critical path
 * 3. Base pointer AGU optimization: Precompute &in[k+K] once
 * 4. Blocked twiddle layout: All twiddles for vector in one cache line
 * 5. Hoisted constants: Load VZERO, VHALF, VSQ3 once per stage
 *
 * SSE2 CHARACTERISTICS:
 * =====================
 * - Vector width: 128-bit (2 doubles)
 * - Register count: 16 XMM registers
 * - No FMA: Separate MUL + ADD/SUB
 * - Alignment: 16-byte
 * - Loop stride: k += 2 (2 doubles per vector)
 * - Twiddle offset: (k/2)*8
 * - Target: Pentium 4, Core 2 Duo, and SSE2-only modern CPUs
 *
 * @author Tugbars
 * @version 3.1-SSE2 (Complete implementation)
 * @date 2025
 */

#ifndef RADIX3_SSE2_OPTIMIZED_H
#define RADIX3_SSE2_OPTIMIZED_H

#ifdef __SSE2__

#include <emmintrin.h>
#include <stddef.h>

//==============================================================================
// COMPILER ATTRIBUTES
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
// GEOMETRIC CONSTANTS
//==============================================================================

#define C_HALF_SSE2 (-0.5)
#define S_SQRT3_2_SSE2 0.8660254037844386467618

//==============================================================================
// VECTOR CONSTANTS - REDUCED SET
//==============================================================================

static inline __m128d V128_HALF(void) { 
    return _mm_set1_pd(C_HALF_SSE2); 
}

static inline __m128d V128_SQRT3_2(void) { 
    return _mm_set1_pd(S_SQRT3_2_SSE2); 
}

//==============================================================================
// TWIDDLE LAYOUT
//==============================================================================

#define TWIDDLE_BLOCK_OFFSET_R3_SSE2(k) (((k) >> 1) << 3) // (k/2)*8
#define TW_W1_RE_OFFSET 0                                  // W^1 real
#define TW_W1_IM_OFFSET 2                                  // W^1 imag
#define TW_W2_RE_OFFSET 4                                  // W^2 real
#define TW_W2_IM_OFFSET 6                                  // W^2 imag

//==============================================================================
// LOAD/STORE MACROS
//==============================================================================

#define LOAD_RE_SSE2(ptr) _mm_load_pd(ptr)
#define LOAD_IM_SSE2(ptr) _mm_load_pd(ptr)
#define STORE_RE_SSE2(ptr, v) _mm_store_pd(ptr, v)
#define STORE_IM_SSE2(ptr, v) _mm_store_pd(ptr, v)

//==============================================================================
// COMPLEX MULTIPLICATION (NO FMA)
//==============================================================================

/**
 * @brief Complex multiplication: (a + ib) * (c + id) without FMA
 * Result: re = ac - bd, im = ad + bc
 */
#define CMUL_SSE2(a_re, a_im, b_re, b_im, out_re, out_im)  \
    do                                                       \
    {                                                        \
        __m128d ac = _mm_mul_pd(a_re, b_re);                 \
        __m128d bd = _mm_mul_pd(a_im, b_im);                 \
        out_re = _mm_sub_pd(ac, bd);                         \
        __m128d ad = _mm_mul_pd(a_re, b_im);                 \
        __m128d bc = _mm_mul_pd(a_im, b_re);                 \
        out_im = _mm_add_pd(ad, bc);                         \
    } while (0)

//==============================================================================
// RADIX-3 BUTTERFLY MACROS
//==============================================================================

/**
 * @brief Radix-3 butterfly - FORWARD - OPTIMIZED (no FMA)
 *
 * Optimizations:
 * 1. SUB(0, MUL) for negative: Saves one constant register
 * 2. Micro-scheduling: rot_re computed before common_*
 * 3. Manual MUL+ADD ordering for better ILP
 */
#define RADIX3_BUTTERFLY_FV_SSE2_OPT(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                     y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                            \
    {                                                                             \
        __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                \
        __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                \
        __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                \
        __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                \
        __m128d rot_re = _mm_mul_pd(V128_SQRT3_2(), dif_im);                      \
        __m128d half_sum_re = _mm_mul_pd(V128_HALF(), sum_re);                    \
        __m128d half_sum_im = _mm_mul_pd(V128_HALF(), sum_im);                    \
        __m128d common_re = _mm_add_pd(half_sum_re, a_re);                        \
        __m128d common_im = _mm_add_pd(half_sum_im, a_im);                        \
        __m128d rot_im = _mm_sub_pd(_mm_setzero_pd(), _mm_mul_pd(V128_SQRT3_2(), dif_re)); \
        y0_re = _mm_add_pd(a_re, sum_re);                                         \
        y0_im = _mm_add_pd(a_im, sum_im);                                         \
        y1_re = _mm_add_pd(common_re, rot_re);                                    \
        y1_im = _mm_add_pd(common_im, rot_im);                                    \
        y2_re = _mm_sub_pd(common_re, rot_re);                                    \
        y2_im = _mm_sub_pd(common_im, rot_im);                                    \
    } while (0)

/**
 * @brief Radix-3 butterfly - BACKWARD - OPTIMIZED (no FMA)
 */
#define RADIX3_BUTTERFLY_BV_SSE2_OPT(a_re, a_im, tB_re, tB_im, tC_re, tC_im,    \
                                     y0_re, y0_im, y1_re, y1_im, y2_re, y2_im)  \
    do                                                                             \
    {                                                                              \
        __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                 \
        __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                 \
        __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                 \
        __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                 \
        __m128d rot_re = _mm_sub_pd(_mm_setzero_pd(), _mm_mul_pd(V128_SQRT3_2(), dif_im)); \
        __m128d half_sum_re = _mm_mul_pd(V128_HALF(), sum_re);                     \
        __m128d half_sum_im = _mm_mul_pd(V128_HALF(), sum_im);                     \
        __m128d common_re = _mm_add_pd(half_sum_re, a_re);                         \
        __m128d common_im = _mm_add_pd(half_sum_im, a_im);                         \
        __m128d rot_im = _mm_mul_pd(V128_SQRT3_2(), dif_re);                       \
        y0_re = _mm_add_pd(a_re, sum_re);                                          \
        y0_im = _mm_add_pd(a_im, sum_im);                                          \
        y1_re = _mm_add_pd(common_re, rot_re);                                     \
        y1_im = _mm_add_pd(common_im, rot_im);                                     \
        y2_re = _mm_sub_pd(common_re, rot_re);                                     \
        y2_im = _mm_sub_pd(common_im, rot_im);                                     \
    } while (0)

/**
 * @brief Radix-3 butterfly - FORWARD - With hoisted constants
 */
#define RADIX3_BUTTERFLY_FV_SSE2_OPT_C(a_re,a_im,tB_re,tB_im,tC_re,tC_im, \
    y0_re,y0_im,y1_re,y1_im,y2_re,y2_im, VHALF,VSQ3,VZERO)                   \
do {                                                                          \
    __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                \
    __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                \
    __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                \
    __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                \
    __m128d rot_re = _mm_mul_pd(VSQ3, dif_im);                                \
    __m128d half_sum_re = _mm_mul_pd(VHALF, sum_re);                          \
    __m128d half_sum_im = _mm_mul_pd(VHALF, sum_im);                          \
    __m128d common_re = _mm_add_pd(half_sum_re, a_re);                        \
    __m128d common_im = _mm_add_pd(half_sum_im, a_im);                        \
    __m128d rot_im = _mm_sub_pd(VZERO, _mm_mul_pd(VSQ3, dif_re));            \
    y0_re = _mm_add_pd(a_re, sum_re);                                         \
    y0_im = _mm_add_pd(a_im, sum_im);                                         \
    y1_re = _mm_add_pd(common_re, rot_re);                                    \
    y1_im = _mm_add_pd(common_im, rot_im);                                    \
    y2_re = _mm_sub_pd(common_re, rot_re);                                    \
    y2_im = _mm_sub_pd(common_im, rot_im);                                    \
} while (0)

/**
 * @brief Radix-3 butterfly - BACKWARD - With hoisted constants
 */
#define RADIX3_BUTTERFLY_BV_SSE2_OPT_C(a_re,a_im,tB_re,tB_im,tC_re,tC_im, \
    y0_re,y0_im,y1_re,y1_im,y2_re,y2_im, VHALF,VSQ3,VZERO)                   \
do {                                                                          \
    __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                \
    __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                \
    __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                \
    __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                \
    __m128d rot_re = _mm_sub_pd(VZERO, _mm_mul_pd(VSQ3, dif_im));            \
    __m128d half_sum_re = _mm_mul_pd(VHALF, sum_re);                          \
    __m128d half_sum_im = _mm_mul_pd(VHALF, sum_im);                          \
    __m128d common_re = _mm_add_pd(half_sum_re, a_re);                        \
    __m128d common_im = _mm_add_pd(half_sum_im, a_im);                        \
    __m128d rot_im = _mm_mul_pd(VSQ3, dif_re);                                \
    y0_re = _mm_add_pd(a_re, sum_re);                                         \
    y0_im = _mm_add_pd(a_im, sum_im);                                         \
    y1_re = _mm_add_pd(common_re, rot_re);                                    \
    y1_im = _mm_add_pd(common_im, rot_im);                                    \
    y2_re = _mm_sub_pd(common_re, rot_re);                                    \
    y2_im = _mm_sub_pd(common_im, rot_im);                                    \
} while (0)

//==============================================================================
// INLINE BUTTERFLY FUNCTIONS (With Base Pointers)
//==============================================================================

/**
 * @brief Single butterfly - FORWARD - With base pointers
 */
FORCE_INLINE void radix3_butterfly_sse2_fv_baseptr(
    const size_t k,
    const double *RESTRICT in0r, const double *RESTRICT in0i,
    const double *RESTRICT in1r, const double *RESTRICT in1i,
    const double *RESTRICT in2r, const double *RESTRICT in2i,
    double *RESTRICT out0r, double *RESTRICT out0i,
    double *RESTRICT out1r, double *RESTRICT out1i,
    double *RESTRICT out2r, double *RESTRICT out2i,
    const double *RESTRICT tw,
    const size_t pf_dist,
    const size_t K)
{
    // Hoist constants
    const __m128d VZERO = _mm_setzero_pd();
    const __m128d VHALF = _mm_set1_pd(C_HALF_SSE2);
    const __m128d VSQ3 = _mm_set1_pd(S_SQRT3_2_SSE2);

    // Load input data
    __m128d a_re = LOAD_RE_SSE2(&in0r[k]);
    __m128d a_im = LOAD_IM_SSE2(&in0i[k]);
    __m128d b_re = LOAD_RE_SSE2(&in1r[k]);
    __m128d b_im = LOAD_IM_SSE2(&in1i[k]);
    __m128d c_re = LOAD_RE_SSE2(&in2r[k]);
    __m128d c_im = LOAD_IM_SSE2(&in2i[k]);

    // Prefetch future data
    if (k + pf_dist < K)
    {
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_SSE2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 6], _MM_HINT_T0);
    }

    // Load twiddles - BLOCKED LAYOUT - ALIGNED (16-byte)
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_SSE2(k);
    __m128d w1_re = _mm_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m128d w1_im = _mm_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m128d w2_re = _mm_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m128d w2_im = _mm_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m128d tB_re, tB_im, tC_re, tC_im;
    CMUL_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD
    __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_SSE2_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store results
    STORE_RE_SSE2(&out0r[k], y0_re);
    STORE_IM_SSE2(&out0i[k], y0_im);
    STORE_RE_SSE2(&out1r[k], y1_re);
    STORE_IM_SSE2(&out1i[k], y1_im);
    STORE_RE_SSE2(&out2r[k], y2_re);
    STORE_IM_SSE2(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - BACKWARD - With base pointers
 */
FORCE_INLINE void radix3_butterfly_sse2_bv_baseptr(
    const size_t k,
    const double *RESTRICT in0r, const double *RESTRICT in0i,
    const double *RESTRICT in1r, const double *RESTRICT in1i,
    const double *RESTRICT in2r, const double *RESTRICT in2i,
    double *RESTRICT out0r, double *RESTRICT out0i,
    double *RESTRICT out1r, double *RESTRICT out1i,
    double *RESTRICT out2r, double *RESTRICT out2i,
    const double *RESTRICT tw,
    const size_t pf_dist,
    const size_t K)
{
    // Hoist constants
    const __m128d VZERO = _mm_setzero_pd();
    const __m128d VHALF = _mm_set1_pd(C_HALF_SSE2);
    const __m128d VSQ3 = _mm_set1_pd(S_SQRT3_2_SSE2);

    // Load input data
    __m128d a_re = LOAD_RE_SSE2(&in0r[k]);
    __m128d a_im = LOAD_IM_SSE2(&in0i[k]);
    __m128d b_re = LOAD_RE_SSE2(&in1r[k]);
    __m128d b_im = LOAD_IM_SSE2(&in1i[k]);
    __m128d c_re = LOAD_RE_SSE2(&in2r[k]);
    __m128d c_im = LOAD_IM_SSE2(&in2i[k]);

    // Prefetch future data
    if (k + pf_dist < K)
    {
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_SSE2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 2], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 6], _MM_HINT_T0);
    }

    // Load twiddles
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_SSE2(k);
    __m128d w1_re = _mm_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m128d w1_im = _mm_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m128d w2_re = _mm_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m128d w2_im = _mm_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m128d tB_re, tB_im, tC_re, tC_im;
    CMUL_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD
    __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_SSE2_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store results
    STORE_RE_SSE2(&out0r[k], y0_re);
    STORE_IM_SSE2(&out0i[k], y0_im);
    STORE_RE_SSE2(&out1r[k], y1_re);
    STORE_IM_SSE2(&out1i[k], y1_im);
    STORE_RE_SSE2(&out2r[k], y2_re);
    STORE_IM_SSE2(&out2i[k], y2_im);
}

//==============================================================================
// TAIL HANDLING (1 element - scalar fallback)
//==============================================================================

/**
 * @brief Handle tail element - FORWARD (scalar fallback for 1 element)
 */
FORCE_INLINE void radix3_butterfly_sse2_fv_tail(
    const size_t k_start,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const double c_half = C_HALF_SSE2;
    const double s_sqrt3_2 = S_SQRT3_2_SSE2;

    for (size_t i = 0; i < count; ++i)
    {
        size_t k = k_start + i;
        
        // BLOCKED LAYOUT indexing: (k/2)*8 + lane_offset
        size_t block = ((k >> 1) << 3);  // (k/2)*8
        size_t lane = (k & 1);            // 0..1 within the block

        // Load inputs
        double a_re = in_re[k];
        double a_im = in_im[k];
        double b_re = in_re[k + K];
        double b_im = in_im[k + K];
        double c_re = in_re[k + 2 * K];
        double c_im = in_im[k + 2 * K];

        // Load twiddles from blocked layout
        double w1_re = tw[block + 0 + lane];   // W1_re block
        double w1_im = tw[block + 2 + lane];   // W1_im block
        double w2_re = tw[block + 4 + lane];   // W2_re block
        double w2_im = tw[block + 6 + lane];   // W2_im block

        // Complex multiply
        double tB_re = b_re * w1_re - b_im * w1_im;
        double tB_im = b_re * w1_im + b_im * w1_re;
        double tC_re = c_re * w2_re - c_im * w2_im;
        double tC_im = c_re * w2_im + c_im * w2_re;

        // Radix-3 butterfly
        double sum_re = tB_re + tC_re;
        double sum_im = tB_im + tC_im;
        double dif_re = tB_re - tC_re;
        double dif_im = tB_im - tC_im;

        double rot_re = s_sqrt3_2 * dif_im;
        double rot_im = -s_sqrt3_2 * dif_re;
        double common_re = a_re + c_half * sum_re;
        double common_im = a_im + c_half * sum_im;

        out_re[k] = a_re + sum_re;
        out_im[k] = a_im + sum_im;
        out_re[k + K] = common_re + rot_re;
        out_im[k + K] = common_im + rot_im;
        out_re[k + 2 * K] = common_re - rot_re;
        out_im[k + 2 * K] = common_im - rot_im;
    }
}

/**
 * @brief Handle tail element - BACKWARD (scalar fallback for 1 element)
 */
FORCE_INLINE void radix3_butterfly_sse2_bv_tail(
    const size_t k_start,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const double c_half = C_HALF_SSE2;
    const double s_sqrt3_2 = S_SQRT3_2_SSE2;

    for (size_t i = 0; i < count; ++i)
    {
        size_t k = k_start + i;
        
        // BLOCKED LAYOUT indexing
        size_t block = ((k >> 1) << 3);
        size_t lane = (k & 1);

        // Load inputs
        double a_re = in_re[k];
        double a_im = in_im[k];
        double b_re = in_re[k + K];
        double b_im = in_im[k + K];
        double c_re = in_re[k + 2 * K];
        double c_im = in_im[k + 2 * K];

        // Load twiddles from blocked layout
        double w1_re = tw[block + 0 + lane];
        double w1_im = tw[block + 2 + lane];
        double w2_re = tw[block + 4 + lane];
        double w2_im = tw[block + 6 + lane];

        // Complex multiply
        double tB_re = b_re * w1_re - b_im * w1_im;
        double tB_im = b_re * w1_im + b_im * w1_re;
        double tC_re = c_re * w2_re - c_im * w2_im;
        double tC_im = c_re * w2_im + c_im * w2_re;

        // Radix-3 butterfly (backward - sign flip)
        double sum_re = tB_re + tC_re;
        double sum_im = tB_im + tC_im;
        double dif_re = tB_re - tC_re;
        double dif_im = tB_im - tC_im;

        double rot_re = -s_sqrt3_2 * dif_im;
        double rot_im = s_sqrt3_2 * dif_re;
        double common_re = a_re + c_half * sum_re;
        double common_im = a_im + c_half * sum_im;

        out_re[k] = a_re + sum_re;
        out_im[k] = a_im + sum_im;
        out_re[k + K] = common_re + rot_re;
        out_im[k + K] = common_im + rot_im;
        out_re[k + 2 * K] = common_re - rot_re;
        out_im[k + 2 * K] = common_im - rot_im;
    }
}

//==============================================================================
// STAGE-LEVEL FUNCTIONS
//==============================================================================

/**
 * @brief Execute complete radix-3 stage - FORWARD - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_sse2_fv_opt(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t pf_dist)
{
    // Precompute base pointers (AGU optimization)
    const double *RESTRICT in0r = in_re;
    const double *RESTRICT in1r = in_re + K;
    const double *RESTRICT in2r = in_re + 2 * K;
    const double *RESTRICT in0i = in_im;
    const double *RESTRICT in1i = in_im + K;
    const double *RESTRICT in2i = in_im + 2 * K;

    double *RESTRICT out0r = out_re;
    double *RESTRICT out1r = out_re + K;
    double *RESTRICT out2r = out_re + 2 * K;
    double *RESTRICT out0i = out_im;
    double *RESTRICT out1i = out_im + K;
    double *RESTRICT out2i = out_im + 2 * K;

    // Alignment hints (16-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 16);
    in1r = (const double *)__builtin_assume_aligned(in1r, 16);
    in2r = (const double *)__builtin_assume_aligned(in2r, 16);
    in0i = (const double *)__builtin_assume_aligned(in0i, 16);
    in1i = (const double *)__builtin_assume_aligned(in1i, 16);
    in2i = (const double *)__builtin_assume_aligned(in2i, 16);
    out0r = (double *)__builtin_assume_aligned(out0r, 16);
    out1r = (double *)__builtin_assume_aligned(out1r, 16);
    out2r = (double *)__builtin_assume_aligned(out2r, 16);
    out0i = (double *)__builtin_assume_aligned(out0i, 16);
    out1i = (double *)__builtin_assume_aligned(out1i, 16);
    out2i = (double *)__builtin_assume_aligned(out2i, 16);
    tw = (const double *)__builtin_assume_aligned(tw, 16);
#endif

    const size_t k_end = K & ~1UL; // Round down to multiple of 2 (SSE2 vector width)

    // Main loop: Process 2 butterflies per iteration
    for (size_t k = 0; k < k_end; k += 2)
    {
        radix3_butterfly_sse2_fv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail: process remaining 1 butterfly
    const size_t remainder = K & 1UL;
    if (remainder > 0)
    {
        radix3_butterfly_sse2_fv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_sse2_bv_opt(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t pf_dist)
{
    // Precompute base pointers
    const double *RESTRICT in0r = in_re;
    const double *RESTRICT in1r = in_re + K;
    const double *RESTRICT in2r = in_re + 2 * K;
    const double *RESTRICT in0i = in_im;
    const double *RESTRICT in1i = in_im + K;
    const double *RESTRICT in2i = in_im + 2 * K;

    double *RESTRICT out0r = out_re;
    double *RESTRICT out1r = out_re + K;
    double *RESTRICT out2r = out_re + 2 * K;
    double *RESTRICT out0i = out_im;
    double *RESTRICT out1i = out_im + K;
    double *RESTRICT out2i = out_im + 2 * K;

    // Alignment hints (16-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 16);
    in1r = (const double *)__builtin_assume_aligned(in1r, 16);
    in2r = (const double *)__builtin_assume_aligned(in2r, 16);
    in0i = (const double *)__builtin_assume_aligned(in0i, 16);
    in1i = (const double *)__builtin_assume_aligned(in1i, 16);
    in2i = (const double *)__builtin_assume_aligned(in2i, 16);
    out0r = (double *)__builtin_assume_aligned(out0r, 16);
    out1r = (double *)__builtin_assume_aligned(out1r, 16);
    out2r = (double *)__builtin_assume_aligned(out2r, 16);
    out0i = (double *)__builtin_assume_aligned(out0i, 16);
    out1i = (double *)__builtin_assume_aligned(out1i, 16);
    out2i = (double *)__builtin_assume_aligned(out2i, 16);
    tw = (const double *)__builtin_assume_aligned(tw, 16);
#endif

    const size_t k_end = K & ~1UL;

    // Main loop
    for (size_t k = 0; k < k_end; k += 2)
    {
        radix3_butterfly_sse2_bv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail
    const size_t remainder = K & 1UL;
    if (remainder > 0)
    {
        radix3_butterfly_sse2_bv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================
/*
 * SSE2 IMPLEMENTATION NOTES:
 * ===========================
 *
 * WHY SSE2 IS VALUABLE:
 * ---------------------
 * 1. Backward compatibility: Works on 2001+ hardware (Pentium 4, Core 2 Duo)
 * 2. Fallback for VMs: Some virtual machines only expose SSE2
 * 3. Verification: Reference for testing AVX2/AVX-512 implementations
 * 4. Embedded systems: Many embedded x86 processors support only SSE2
 *
 * INSTRUCTION COUNT COMPARISON (per butterfly):
 * ==============================================
 * SSE2 (no FMA):        ~18 instructions (6 MUL, 12 ADD/SUB)
 * AVX2 (with FMA):      ~11 instructions (2 FMA, 2 MUL, 7 ADD/SUB)
 * AVX-512 (with FMA):   ~11 instructions (same as AVX2, wider vectors)
 *
 * PERFORMANCE EXPECTATIONS:
 * =========================
 * - Core 2 Duo:         Baseline performance (what this targets)
 * - Core i3/i5/i7:      Use AVX2 instead (much faster)
 * - Modern Ryzen/EPYC:  Use AVX2 instead (much faster)
 * - Old Pentium 4:      May be 5-10% faster than scalar
 * - VMs with SSE2 only: 20-40% faster than scalar
 *
 * SSE2 vs SCALAR TRADE-OFF:
 * ==========================
 * Benefits:
 * + 2× parallelism (2 doubles per vector)
 * + Reduced register pressure vs scalar (6 XMM vs 12+ GPR)
 * + Cache-friendly (blocked twiddle layout)
 *
 * Costs:
 * - No FMA (higher instruction count)
 * - 16-byte alignment requirements
 * - Older CPUs have slower SSE2 execution
 *
 * WHEN TO USE SSE2:
 * ==================
 * Use SSE2 when:
 * - Target is 2001-2010 era hardware
 * - Running in VM with only SSE2 support
 * - Verifying AVX implementation correctness
 * - Embedded x86 processor without AVX
 *
 * Do NOT use SSE2 when:
 * - CPU supports AVX2 (use AVX2 instead - 2-3× faster)
 * - CPU supports AVX-512 (use AVX-512 instead - 4-6× faster)
 * - K < 4 (use scalar instead - less overhead)
 */

#endif // __SSE2__

#endif // RADIX3_SSE2_OPTIMIZED_H