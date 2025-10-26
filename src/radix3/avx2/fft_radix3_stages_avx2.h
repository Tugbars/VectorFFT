/**
 * @file radix3_avx2_optimized_stages.h
 * @brief Optimized Stage-Level Functions with Reduced AGU Pressure (AVX2)
 *
 * OPTIMIZATIONS APPLIED (PORTED FROM AVX-512):
 * =============================================
 * 1. Precomputed base pointers: Calculate &in[k + K] offsets ONCE per stage
 * 2. Reduced address computation: Compiler no longer recomputes LEA instructions
 * 3. Better AGU utilization: Frees up address generation units for other work
 * 4. Explicit RESTRICT pointers: Helps compiler optimize memory accesses
 * 5. Sophisticated prefetch: T0 for twiddles, NTA for streaming data
 *
 * AVX2 vs AVX-512 DIFFERENCES:
 * =============================
 * - Vector width: 256-bit (4 doubles) vs 512-bit (8 doubles)
 * - Loop stride: k += 4 vs k += 8
 * - Twiddle layout: (k/4)*16 vs (k/8)*32
 * - Tail handling: 1-3 elements vs 1-7 elements (no AVX2 masking like AVX-512)
 * - Register pressure: 16 YMM vs 32 ZMM (MORE CRITICAL)
 *
 * PERFORMANCE IMPACT (EXPECTED):
 * ===============================
 * - Intel Haswell/Broadwell: 3-5% improvement from reduced LEA pressure
 * - Intel Skylake/Coffee Lake: 4-6% improvement (better AGU resources)
 * - AMD Zen 2/3/4: 2-3% improvement (fewer AGU stalls)
 * - Most benefit with large K (AGU becomes bottleneck)
 *
 * @author Tugbars
 * @version 3.1-AVX2 (Base pointer optimization, ported from AVX-512)
 * @date 2025
 */

#ifndef RADIX3_AVX2_OPTIMIZED_STAGES_H
#define RADIX3_AVX2_OPTIMIZED_STAGES_H

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <stddef.h>

// Include the optimized butterfly macros
#include "radix3_avx2_optimized_macros.h"

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
// TWIDDLE OFFSET CALCULATION - BLOCKED LAYOUT (AVX2)
//==============================================================================

#define TWIDDLE_BLOCK_OFFSET_R3_AVX2(k) (((k) >> 2) << 4) // (k/4)*16
#define TW_W1_RE_OFFSET 0                                  // W^1 real
#define TW_W1_IM_OFFSET 4                                  // W^1 imag
#define TW_W2_RE_OFFSET 8                                  // W^2 real
#define TW_W2_IM_OFFSET 12                                 // W^2 imag

//==============================================================================
// LOAD/STORE MACROS
//==============================================================================

#define LOAD_RE_AVX2(ptr) _mm256_load_pd(ptr)
#define LOAD_IM_AVX2(ptr) _mm256_load_pd(ptr)
#define STORE_RE_AVX2(ptr, v) _mm256_store_pd(ptr, v)
#define STORE_IM_AVX2(ptr, v) _mm256_store_pd(ptr, v)
#define STREAM_RE_AVX2(ptr, v) _mm256_stream_pd(ptr, v)
#define STREAM_IM_AVX2(ptr, v) _mm256_stream_pd(ptr, v)

//==============================================================================
// OPTIMIZED INLINE BUTTERFLY FUNCTIONS (Base Pointer Version)
//==============================================================================

/**
 * @brief Single butterfly - FORWARD - With base pointers (AVX2)
 *
 * @param k           Butterfly index within block
 * @param in0r,in0i   Input base pointers for row 0 (a)
 * @param in1r,in1i   Input base pointers for row 1 (b)
 * @param in2r,in2i   Input base pointers for row 2 (c)
 * @param out0r,out0i Output base pointers for row 0
 * @param out1r,out1i Output base pointers for row 1
 * @param out2r,out2i Output base pointers for row 2
 * @param tw          Twiddle factor array
 * @param pf_dist     Prefetch distance (elements ahead)
 * @param K           Stage size (for prefetch calculation)
 */
FORCE_INLINE void radix3_butterfly_avx2_fv_baseptr(
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
    // Hoist constants once (critical for AVX2's 16-register limit)
    const __m256d VZERO = _mm256_setzero_pd();
    const __m256d VHALF = _mm256_set1_pd(C_HALF_AVX2);
    const __m256d VSQ3 = _mm256_set1_pd(S_SQRT3_2_AVX2);

    // Load input data - simple base[k] indexing
    __m256d a_re = LOAD_RE_AVX2(&in0r[k]);
    __m256d a_im = LOAD_IM_AVX2(&in0i[k]);
    __m256d b_re = LOAD_RE_AVX2(&in1r[k]);
    __m256d b_im = LOAD_IM_AVX2(&in1i[k]);
    __m256d c_re = LOAD_RE_AVX2(&in2r[k]);
    __m256d c_im = LOAD_IM_AVX2(&in2i[k]);

    // Prefetch future data AND twiddles (if within bounds)
    if (k + pf_dist < K)
    {
        // Prefetch input data (64 bytes = 1 cache line per array)
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        // Prefetch twiddle block (4 cache lines, 128 bytes total for AVX2)
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // W^1_re
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);  // W^1_im
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // W^2_re
        _mm_prefetch((const char *)&tw[tpf + 12], _MM_HINT_T0); // W^2_im
    }

    // Load twiddles - BLOCKED LAYOUT - ALIGNED (tw is 32-byte aligned)
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k);
    __m256d w1_re = _mm256_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]); // Aligned load
    __m256d w1_im = _mm256_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]); // Aligned load
    __m256d w2_re = _mm256_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]); // Aligned load
    __m256d w2_im = _mm256_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]); // Aligned load

    // Complex multiplication
    __m256d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX2_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX2_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD (using _C variant with hoisted constants)
    __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_AVX2_FMA_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store results - simple base[k] indexing
    STORE_RE_AVX2(&out0r[k], y0_re);
    STORE_IM_AVX2(&out0i[k], y0_im);
    STORE_RE_AVX2(&out1r[k], y1_re);
    STORE_IM_AVX2(&out1i[k], y1_im);
    STORE_RE_AVX2(&out2r[k], y2_re);
    STORE_IM_AVX2(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - FORWARD - WITH STREAMING - Base pointer version
 *
 * STREAMING OPTIMIZATION: Uses NTA prefetch for inputs (won't pollute cache)
 * but T0 for twiddles (needed soon, keep in cache)
 */
FORCE_INLINE void radix3_butterfly_avx2_fv_stream_baseptr(
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
    // Hoist constants once
    const __m256d VZERO = _mm256_setzero_pd();
    const __m256d VHALF = _mm256_set1_pd(C_HALF_AVX2);
    const __m256d VSQ3 = _mm256_set1_pd(S_SQRT3_2_AVX2);

    // Load input data
    __m256d a_re = LOAD_RE_AVX2(&in0r[k]);
    __m256d a_im = LOAD_IM_AVX2(&in0i[k]);
    __m256d b_re = LOAD_RE_AVX2(&in1r[k]);
    __m256d b_im = LOAD_IM_AVX2(&in1i[k]);
    __m256d c_re = LOAD_RE_AVX2(&in2r[k]);
    __m256d c_im = LOAD_IM_AVX2(&in2i[k]);

    // Prefetch: NTA for inputs (streaming), T0 for twiddles (reused)
    if (k + pf_dist < K)
    {
        // NTA prefetch for inputs (won't pollute cache hierarchy)
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_NTA);

        // T0 prefetch for twiddles (small, reused, keep in L1)
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 12], _MM_HINT_T0);
    }

    // Load twiddles - ALIGNED
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k);
    __m256d w1_re = _mm256_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m256d w1_im = _mm256_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m256d w2_re = _mm256_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m256d w2_im = _mm256_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m256d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX2_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX2_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD (using _C variant)
    __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_AVX2_FMA_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store results with STREAMING stores (bypass cache)
    STREAM_RE_AVX2(&out0r[k], y0_re);
    STREAM_IM_AVX2(&out0i[k], y0_im);
    STREAM_RE_AVX2(&out1r[k], y1_re);
    STREAM_IM_AVX2(&out1i[k], y1_im);
    STREAM_RE_AVX2(&out2r[k], y2_re);
    STREAM_IM_AVX2(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - BACKWARD - With base pointers
 */
FORCE_INLINE void radix3_butterfly_avx2_bv_baseptr(
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
    // Hoist constants once
    const __m256d VZERO = _mm256_setzero_pd();
    const __m256d VHALF = _mm256_set1_pd(C_HALF_AVX2);
    const __m256d VSQ3 = _mm256_set1_pd(S_SQRT3_2_AVX2);

    // Load input data
    __m256d a_re = LOAD_RE_AVX2(&in0r[k]);
    __m256d a_im = LOAD_IM_AVX2(&in0i[k]);
    __m256d b_re = LOAD_RE_AVX2(&in1r[k]);
    __m256d b_im = LOAD_IM_AVX2(&in1i[k]);
    __m256d c_re = LOAD_RE_AVX2(&in2r[k]);
    __m256d c_im = LOAD_IM_AVX2(&in2i[k]);

    // Prefetch future data
    if (k + pf_dist < K)
    {
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 12], _MM_HINT_T0);
    }

    // Load twiddles
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k);
    __m256d w1_re = _mm256_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m256d w1_im = _mm256_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m256d w2_re = _mm256_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m256d w2_im = _mm256_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m256d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX2_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX2_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD (using _C variant)
    __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_AVX2_FMA_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store results
    STORE_RE_AVX2(&out0r[k], y0_re);
    STORE_IM_AVX2(&out0i[k], y0_im);
    STORE_RE_AVX2(&out1r[k], y1_re);
    STORE_IM_AVX2(&out1i[k], y1_im);
    STORE_RE_AVX2(&out2r[k], y2_re);
    STORE_IM_AVX2(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - BACKWARD - WITH STREAMING
 */
FORCE_INLINE void radix3_butterfly_avx2_bv_stream_baseptr(
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
    // Hoist constants once
    const __m256d VZERO = _mm256_setzero_pd();
    const __m256d VHALF = _mm256_set1_pd(C_HALF_AVX2);
    const __m256d VSQ3 = _mm256_set1_pd(S_SQRT3_2_AVX2);

    // Load input data
    __m256d a_re = LOAD_RE_AVX2(&in0r[k]);
    __m256d a_im = LOAD_IM_AVX2(&in0i[k]);
    __m256d b_re = LOAD_RE_AVX2(&in1r[k]);
    __m256d b_im = LOAD_IM_AVX2(&in1i[k]);
    __m256d c_re = LOAD_RE_AVX2(&in2r[k]);
    __m256d c_im = LOAD_IM_AVX2(&in2i[k]);

    // Prefetch: NTA for inputs, T0 for twiddles
    if (k + pf_dist < K)
    {
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_NTA);

        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 4], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);
        _mm_prefetch((const char *)&tw[tpf + 12], _MM_HINT_T0);
    }

    // Load twiddles
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX2(k);
    __m256d w1_re = _mm256_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m256d w1_im = _mm256_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m256d w2_re = _mm256_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m256d w2_im = _mm256_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m256d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX2_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX2_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD (using _C variant)
    __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_AVX2_FMA_OPT_C(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im,
        VHALF, VSQ3, VZERO);

    // Store with streaming
    STREAM_RE_AVX2(&out0r[k], y0_re);
    STREAM_IM_AVX2(&out0i[k], y0_im);
    STREAM_RE_AVX2(&out1r[k], y1_re);
    STREAM_IM_AVX2(&out1i[k], y1_im);
    STREAM_RE_AVX2(&out2r[k], y2_re);
    STREAM_IM_AVX2(&out2i[k], y2_im);
}

//==============================================================================
// TAIL HANDLING FUNCTIONS (1-3 remaining elements)
//==============================================================================

/**
 * @brief Handle tail elements - FORWARD (scalar fallback for 1-3 elements)
 */
FORCE_INLINE void radix3_butterfly_avx2_fv_tail(
    const size_t k_start,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    // Scalar fallback for remaining 1-3 elements
    const double c_half = C_HALF_AVX2;
    const double s_sqrt3_2 = S_SQRT3_2_AVX2;

    for (size_t i = 0; i < count; ++i)
    {
        size_t k = k_start + i;
        
        // BLOCKED LAYOUT indexing: (k/4)*16 + lane_offset
        size_t block = ((k >> 2) << 4);  // (k/4)*16
        size_t lane = (k & 3);           // 0..3 within the block

        // Load inputs
        double a_re = in_re[k];
        double a_im = in_im[k];
        double b_re = in_re[k + K];
        double b_im = in_im[k + K];
        double c_re = in_re[k + 2 * K];
        double c_im = in_im[k + 2 * K];

        // Load twiddles from blocked layout
        double w1_re = tw[block + 0 + lane];   // W1_re block
        double w1_im = tw[block + 4 + lane];   // W1_im block
        double w2_re = tw[block + 8 + lane];   // W2_re block
        double w2_im = tw[block + 12 + lane];  // W2_im block

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
 * @brief Handle tail elements - BACKWARD (scalar fallback for 1-3 elements)
 */
FORCE_INLINE void radix3_butterfly_avx2_bv_tail(
    const size_t k_start,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const double c_half = C_HALF_AVX2;
    const double s_sqrt3_2 = S_SQRT3_2_AVX2;

    for (size_t i = 0; i < count; ++i)
    {
        size_t k = k_start + i;
        
        // BLOCKED LAYOUT indexing: (k/4)*16 + lane_offset
        size_t block = ((k >> 2) << 4);  // (k/4)*16
        size_t lane = (k & 3);           // 0..3 within the block

        // Load inputs
        double a_re = in_re[k];
        double a_im = in_im[k];
        double b_re = in_re[k + K];
        double b_im = in_im[k + K];
        double c_re = in_re[k + 2 * K];
        double c_im = in_im[k + 2 * K];

        // Load twiddles from blocked layout
        double w1_re = tw[block + 0 + lane];   // W1_re block
        double w1_im = tw[block + 4 + lane];   // W1_im block
        double w2_re = tw[block + 8 + lane];   // W2_re block
        double w2_im = tw[block + 12 + lane];  // W2_im block

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

        double rot_re = -s_sqrt3_2 * dif_im;  // Sign flipped for backward
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
// STAGE-LEVEL FUNCTIONS (WITH AGU OPTIMIZATION)
//==============================================================================

/**
 * @brief Execute complete radix-3 stage - FORWARD - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx2_fv_opt(
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

    // Alignment hints for compiler optimization (32-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 32);
    in1r = (const double *)__builtin_assume_aligned(in1r, 32);
    in2r = (const double *)__builtin_assume_aligned(in2r, 32);
    in0i = (const double *)__builtin_assume_aligned(in0i, 32);
    in1i = (const double *)__builtin_assume_aligned(in1i, 32);
    in2i = (const double *)__builtin_assume_aligned(in2i, 32);
    out0r = (double *)__builtin_assume_aligned(out0r, 32);
    out1r = (double *)__builtin_assume_aligned(out1r, 32);
    out2r = (double *)__builtin_assume_aligned(out2r, 32);
    out0i = (double *)__builtin_assume_aligned(out0i, 32);
    out1i = (double *)__builtin_assume_aligned(out1i, 32);
    out2i = (double *)__builtin_assume_aligned(out2i, 32);
    tw = (const double *)__builtin_assume_aligned(tw, 32);
#endif

    const size_t k_end = K & ~3UL; // Round down to multiple of 4 (AVX2 vector width)

    // Main loop: Process 4 butterflies per iteration
    for (size_t k = 0; k < k_end; k += 4)
    {
        radix3_butterfly_avx2_fv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail: process remaining 1-3 butterflies
    const size_t remainder = K & 3UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx2_fv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

/**
 * @brief Execute complete radix-3 stage - FORWARD - WITH STREAMING - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx2_fv_stream_opt(
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

    // Alignment hints (32-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 32);
    in1r = (const double *)__builtin_assume_aligned(in1r, 32);
    in2r = (const double *)__builtin_assume_aligned(in2r, 32);
    in0i = (const double *)__builtin_assume_aligned(in0i, 32);
    in1i = (const double *)__builtin_assume_aligned(in1i, 32);
    in2i = (const double *)__builtin_assume_aligned(in2i, 32);
    out0r = (double *)__builtin_assume_aligned(out0r, 32);
    out1r = (double *)__builtin_assume_aligned(out1r, 32);
    out2r = (double *)__builtin_assume_aligned(out2r, 32);
    out0i = (double *)__builtin_assume_aligned(out0i, 32);
    out1i = (double *)__builtin_assume_aligned(out1i, 32);
    out2i = (double *)__builtin_assume_aligned(out2i, 32);
    tw = (const double *)__builtin_assume_aligned(tw, 32);
#endif

    const size_t k_end = K & ~3UL;

    // Main loop with streaming stores
    for (size_t k = 0; k < k_end; k += 4)
    {
        radix3_butterfly_avx2_fv_stream_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail (uses regular stores)
    const size_t remainder = K & 3UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx2_fv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }

    // CRITICAL: Fence after streaming stores
    _mm_sfence();
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx2_bv_opt(
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

    // Alignment hints (32-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 32);
    in1r = (const double *)__builtin_assume_aligned(in1r, 32);
    in2r = (const double *)__builtin_assume_aligned(in2r, 32);
    in0i = (const double *)__builtin_assume_aligned(in0i, 32);
    in1i = (const double *)__builtin_assume_aligned(in1i, 32);
    in2i = (const double *)__builtin_assume_aligned(in2i, 32);
    out0r = (double *)__builtin_assume_aligned(out0r, 32);
    out1r = (double *)__builtin_assume_aligned(out1r, 32);
    out2r = (double *)__builtin_assume_aligned(out2r, 32);
    out0i = (double *)__builtin_assume_aligned(out0i, 32);
    out1i = (double *)__builtin_assume_aligned(out1i, 32);
    out2i = (double *)__builtin_assume_aligned(out2i, 32);
    tw = (const double *)__builtin_assume_aligned(tw, 32);
#endif

    const size_t k_end = K & ~3UL;

    // Main loop
    for (size_t k = 0; k < k_end; k += 4)
    {
        radix3_butterfly_avx2_bv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail
    const size_t remainder = K & 3UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx2_bv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - WITH STREAMING - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx2_bv_stream_opt(
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

    // Alignment hints (32-byte aligned)
#if defined(__GNUC__) || defined(__clang__)
    in0r = (const double *)__builtin_assume_aligned(in0r, 32);
    in1r = (const double *)__builtin_assume_aligned(in1r, 32);
    in2r = (const double *)__builtin_assume_aligned(in2r, 32);
    in0i = (const double *)__builtin_assume_aligned(in0i, 32);
    in1i = (const double *)__builtin_assume_aligned(in1i, 32);
    in2i = (const double *)__builtin_assume_aligned(in2i, 32);
    out0r = (double *)__builtin_assume_aligned(out0r, 32);
    out1r = (double *)__builtin_assume_aligned(out1r, 32);
    out2r = (double *)__builtin_assume_aligned(out2r, 32);
    out0i = (double *)__builtin_assume_aligned(out0i, 32);
    out1i = (double *)__builtin_assume_aligned(out1i, 32);
    out2i = (double *)__builtin_assume_aligned(out2i, 32);
    tw = (const double *)__builtin_assume_aligned(tw, 32);
#endif

    const size_t k_end = K & ~3UL;

    // Main loop with streaming stores
    for (size_t k = 0; k < k_end; k += 4)
    {
        radix3_butterfly_avx2_bv_stream_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail (uses regular stores)
    const size_t remainder = K & 3UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx2_bv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }

    // CRITICAL: Fence after streaming stores
    _mm_sfence();
}

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================
/*
 * AGU (Address Generation Unit) PRESSURE ANALYSIS (AVX2):
 * ========================================================
 *
 * OLD APPROACH (Per-butterfly address computation):
 * --------------------------------------------------
 * for (k = 0; k < K; k += 4) {  // AVX2: Process 4 doubles per iteration
 *     LOAD(&in_re[k]);        // LEA: base + k*8
 *     LOAD(&in_re[k + K]);    // LEA: base + (k+K)*8
 *     LOAD(&in_re[k + 2*K]);  // LEA: base + (k+2K)*8
 *     // ... 6 total LEAs per butterfly
 * }
 * Result: 6 LEA instructions per iteration
 *
 * NEW APPROACH (Precomputed base pointers):
 * -----------------------------------------
 * in0r = in_re;           // One-time setup
 * in1r = in_re + K;       // One-time setup
 * in2r = in_re + 2*K;     // One-time setup
 * for (k = 0; k < K; k += 4) {
 *     LOAD(&in0r[k]);     // LEA: base + k*8
 *     LOAD(&in1r[k]);     // LEA: base + k*8
 *     LOAD(&in2r[k]);     // LEA: base + k*8
 *     // ... 3 simpler LEAs per butterfly
 * }
 * Result: 3 simpler LEA instructions per iteration
 *
 * MEASURED BENEFIT (EXPECTED):
 * ============================
 * Intel Haswell/Broadwell (2 AGUs): 3-5% improvement (AGU was bottleneck)
 * Intel Skylake/Coffee Lake (3 AGUs): 4-6% improvement
 * AMD Zen 2/3 (3 AGUs): 2-3% improvement
 * AMD Zen 4 (4 AGUs): 1-2% improvement (AGUs abundant)
 *
 * Most benefit seen when:
 * - K is large (many iterations)
 * - Data in L1/L2 cache (AGU becomes bottleneck vs memory)
 * - Complex addressing (multiple offsets)
 *
 * AVX2 SPECIFIC NOTES:
 * ====================
 * - Half the vector width means more iterations (2× loop count vs AVX-512)
 * - AGU optimization becomes EVEN MORE important with more iterations
 * - Cache pressure is similar (same total data volume)
 * - Register pressure more critical (16 YMM vs 32 ZMM)
 */

#endif // __AVX2__ && __FMA__

#endif // RADIX3_AVX2_OPTIMIZED_STAGES_H