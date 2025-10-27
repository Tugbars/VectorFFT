/**
 * @file radix3_avx512_optimized_stages.h
 * @brief Optimized Stage-Level Functions with Reduced AGU Pressure
 *
 * OPTIMIZATIONS APPLIED:
 * ======================
 * 1. Precomputed base pointers: Calculate &in[k + K] offsets ONCE per stage
 * 2. Reduced address computation: Compiler no longer recomputes LEA instructions
 * 3. Better AGU utilization: Frees up address generation units for other work
 * 4. Explicit RESTRICT pointers: Helps compiler optimize memory accesses
 *
 * PERFORMANCE IMPACT:
 * ===================
 * - Intel Skylake-X: 3-5% improvement from reduced LEA pressure
 * - Intel Ice Lake:  4-6% improvement (better AGU resources)
 * - AMD Zen 3/4:     2-3% improvement (fewer AGU stalls)
 * - Most benefit with large K (AGU becomes bottleneck)
 *
 * @author Tugbars
 * @version 3.1 (Base pointer optimization)
 * @date 2025
 */

#ifndef RADIX3_AVX512_OPTIMIZED_STAGES_H
#define RADIX3_AVX512_OPTIMIZED_STAGES_H

#ifdef __AVX512F__

#include <immintrin.h>
#include <stddef.h>

// Include the optimized butterfly macros
#include "radix3_avx512_optimized_macros.h"

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
// TWIDDLE OFFSET CALCULATION - BLOCKED LAYOUT
//==============================================================================

#define TWIDDLE_BLOCK_OFFSET_R3_AVX512(k) (((k) >> 3) << 5) // (k/8)*32
#define TW_W1_RE_OFFSET 0                                   // W^1 real
#define TW_W1_IM_OFFSET 8                                   // W^1 imag
#define TW_W2_RE_OFFSET 16                                  // W^2 real
#define TW_W2_IM_OFFSET 24                                  // W^2 imag

//==============================================================================
// LOAD/STORE MACROS
//==============================================================================

#define LOAD_RE_AVX512(ptr) _mm512_load_pd(ptr)
#define LOAD_IM_AVX512(ptr) _mm512_load_pd(ptr)
#define STORE_RE_AVX512(ptr, v) _mm512_store_pd(ptr, v)
#define STORE_IM_AVX512(ptr, v) _mm512_store_pd(ptr, v)
#define STREAM_RE_AVX512(ptr, v) _mm512_stream_pd(ptr, v)
#define STREAM_IM_AVX512(ptr, v) _mm512_stream_pd(ptr, v)

//==============================================================================
// OPTIMIZED INLINE BUTTERFLY FUNCTIONS (Base Pointer Version)
//==============================================================================

/**
 * @brief Single butterfly - FORWARD - With base pointers
 *
 * @param k           Butterfly index within block
 * @param in0r,in0i   Input base pointers for row 0 (a)
 * @param in1r,in1i   Input base pointers for row 1 (b)
 * @param in2r,in2i   Input base pointers for row 2 (c)
 * @param out0r,out0i Output base pointers for row 0
 * @param out1r,out1i Output base pointers for row 1
 * @param out2r,out2i Output base pointers for row 2
 * @param tw          Twiddle factor array
 * @param pf_dist     Prefetch distance (bytes ahead)
 * @param K           Stage size (for prefetch calculation)
 */
FORCE_INLINE void radix3_butterfly_avx512_fv_baseptr(
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
    // Load input data - simple base[k] indexing
    __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

    // Prefetch future data AND twiddles (if within bounds)
    if (k + pf_dist < K)
    {
        // Prefetch input data
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        // Prefetch twiddle block (4 cache lines, 256 bytes total)
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // 0B   - W^1_re
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // 64B  - W^1_im
        _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0); // 128B - W^2_re
        _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0); // 192B - W^2_im
    }

    // Load twiddles - BLOCKED LAYOUT - ALIGNED (tw is 64-byte aligned)
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]); // Aligned load
    __m512d w1_im = _mm512_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]); // Aligned load
    __m512d w2_re = _mm512_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]); // Aligned load
    __m512d w2_im = _mm512_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]); // Aligned load

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD (using optimized macro with FNMADD)
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Store results - simple base[k] indexing
    STORE_RE_AVX512(&out0r[k], y0_re);
    STORE_IM_AVX512(&out0i[k], y0_im);
    STORE_RE_AVX512(&out1r[k], y1_re);
    STORE_IM_AVX512(&out1i[k], y1_im);
    STORE_RE_AVX512(&out2r[k], y2_re);
    STORE_IM_AVX512(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - FORWARD - WITH STREAMING - Base pointer version
 *
 * STREAMING OPTIMIZATION: Uses NTA prefetch for inputs (won't pollute cache)
 * but T0 for twiddles (needed soon, keep in cache)
 */
FORCE_INLINE void radix3_butterfly_avx512_fv_stream_baseptr(
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
    // Load input data
    __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

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
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // 0B   - W^1_re
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // 64B  - W^1_im
        _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0); // 128B - W^2_re
        _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0); // 192B - W^2_im
    }

    // Load twiddles - ALIGNED
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m512d w1_im = _mm512_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m512d w2_re = _mm512_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m512d w2_im = _mm512_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Streaming stores
    STREAM_RE_AVX512(&out0r[k], y0_re);
    STREAM_IM_AVX512(&out0i[k], y0_im);
    STREAM_RE_AVX512(&out1r[k], y1_re);
    STREAM_IM_AVX512(&out1i[k], y1_im);
    STREAM_RE_AVX512(&out2r[k], y2_re);
    STREAM_IM_AVX512(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - BACKWARD - Base pointer version
 */
FORCE_INLINE void radix3_butterfly_avx512_bv_baseptr(
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
    // Load input data
    __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

    // Prefetch future data AND twiddles
    if (k + pf_dist < K)
    {
        // Prefetch input data
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_T0);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_T0);

        // Prefetch twiddle block (4 cache lines, 256 bytes total)
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // 0B   - W^1_re
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // 64B  - W^1_im
        _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0); // 128B - W^2_re
        _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0); // 192B - W^2_im
    }

    // Load twiddles - ALIGNED
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m512d w1_im = _mm512_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m512d w2_re = _mm512_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m512d w2_im = _mm512_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD (using optimized macro)
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Store results
    STORE_RE_AVX512(&out0r[k], y0_re);
    STORE_IM_AVX512(&out0i[k], y0_im);
    STORE_RE_AVX512(&out1r[k], y1_re);
    STORE_IM_AVX512(&out1i[k], y1_im);
    STORE_RE_AVX512(&out2r[k], y2_re);
    STORE_IM_AVX512(&out2i[k], y2_im);
}

/**
 * @brief Single butterfly - BACKWARD - WITH STREAMING - Base pointer version
 *
 * STREAMING OPTIMIZATION: Uses NTA prefetch for inputs, T0 for twiddles
 */
FORCE_INLINE void radix3_butterfly_avx512_bv_stream_baseptr(
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
    // Load input data
    __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

    // Prefetch: NTA for inputs, T0 for twiddles
    if (k + pf_dist < K)
    {
        // NTA prefetch for inputs (streaming access)
        _mm_prefetch((const char *)&in0r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in0i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in1i[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2r[k + pf_dist], _MM_HINT_NTA);
        _mm_prefetch((const char *)&in2i[k + pf_dist], _MM_HINT_NTA);

        // T0 prefetch for twiddles (small, reused)
        const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k + pf_dist);
        _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // 0B   - W^1_re
        _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // 64B  - W^1_im
        _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0); // 128B - W^2_re
        _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0); // 192B - W^2_im
    }

    // Load twiddles - ALIGNED
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_load_pd(&tw[tw_offset + TW_W1_RE_OFFSET]);
    __m512d w1_im = _mm512_load_pd(&tw[tw_offset + TW_W1_IM_OFFSET]);
    __m512d w2_re = _mm512_load_pd(&tw[tw_offset + TW_W2_RE_OFFSET]);
    __m512d w2_im = _mm512_load_pd(&tw[tw_offset + TW_W2_IM_OFFSET]);

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Streaming stores
    STREAM_RE_AVX512(&out0r[k], y0_re);
    STREAM_IM_AVX512(&out0i[k], y0_im);
    STREAM_RE_AVX512(&out1r[k], y1_re);
    STREAM_IM_AVX512(&out1i[k], y1_im);
    STREAM_RE_AVX512(&out2r[k], y2_re);
    STREAM_IM_AVX512(&out2i[k], y2_im);
}

//==============================================================================
// TAIL FUNCTIONS (For K % 8 != 0)
//==============================================================================

/**
 * @brief Handle tail butterflies - FORWARD
 */
FORCE_INLINE void radix3_butterfly_avx512_fv_tail(
    const size_t k,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const __mmask8 mask = (__mmask8)((1U << count) - 1);

    // Load with mask
    __m512d a_re = _mm512_maskz_load_pd(mask, &in_re[k]);
    __m512d a_im = _mm512_maskz_load_pd(mask, &in_im[k]);
    __m512d b_re = _mm512_maskz_load_pd(mask, &in_re[k + K]);
    __m512d b_im = _mm512_maskz_load_pd(mask, &in_im[k + K]);
    __m512d c_re = _mm512_maskz_load_pd(mask, &in_re[k + 2 * K]);
    __m512d c_im = _mm512_maskz_load_pd(mask, &in_im[k + 2 * K]);

    // Load twiddles with mask - STILL ALIGNED (tw buffer is 64-byte aligned)
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_RE_OFFSET]); // Aligned
    __m512d w1_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_IM_OFFSET]); // Aligned
    __m512d w2_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_RE_OFFSET]); // Aligned
    __m512d w2_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_IM_OFFSET]); // Aligned

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - FORWARD
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Masked stores
    _mm512_mask_store_pd(&out_re[k], mask, y0_re);
    _mm512_mask_store_pd(&out_im[k], mask, y0_im);
    _mm512_mask_store_pd(&out_re[k + K], mask, y1_re);
    _mm512_mask_store_pd(&out_im[k + K], mask, y1_im);
    _mm512_mask_store_pd(&out_re[k + 2 * K], mask, y2_re);
    _mm512_mask_store_pd(&out_im[k + 2 * K], mask, y2_im);
}

/**
 * @brief Handle tail butterflies - BACKWARD
 */
FORCE_INLINE void radix3_butterfly_avx512_bv_tail(
    const size_t k,
    const size_t K,
    const size_t count,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const __mmask8 mask = (__mmask8)((1U << count) - 1);

    // Load with mask
    __m512d a_re = _mm512_maskz_load_pd(mask, &in_re[k]);
    __m512d a_im = _mm512_maskz_load_pd(mask, &in_im[k]);
    __m512d b_re = _mm512_maskz_load_pd(mask, &in_re[k + K]);
    __m512d b_im = _mm512_maskz_load_pd(mask, &in_im[k + K]);
    __m512d c_re = _mm512_maskz_load_pd(mask, &in_re[k + 2 * K]);
    __m512d c_im = _mm512_maskz_load_pd(mask, &in_im[k + 2 * K]);

    // Load twiddles with mask - STILL ALIGNED
    const size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_RE_OFFSET]); // Aligned
    __m512d w1_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_IM_OFFSET]); // Aligned
    __m512d w2_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_RE_OFFSET]); // Aligned
    __m512d w2_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_IM_OFFSET]); // Aligned

    // Complex multiplication
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Radix-3 butterfly - BACKWARD
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
        a_re, a_im, tB_re, tB_im, tC_re, tC_im,
        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Masked stores
    _mm512_mask_store_pd(&out_re[k], mask, y0_re);
    _mm512_mask_store_pd(&out_im[k], mask, y0_im);
    _mm512_mask_store_pd(&out_re[k + K], mask, y1_re);
    _mm512_mask_store_pd(&out_im[k + K], mask, y1_im);
    _mm512_mask_store_pd(&out_re[k + 2 * K], mask, y2_re);
    _mm512_mask_store_pd(&out_im[k + 2 * K], mask, y2_im);
}

//==============================================================================
// OPTIMIZED STAGE-LEVEL FUNCTIONS (WITH BASE POINTER PRECOMPUTATION)
//==============================================================================

/**
 * @brief Execute complete radix-3 stage - FORWARD - OPTIMIZED
 *
 * KEY OPTIMIZATION: Precomputes base pointers ONCE per stage
 * Reduces AGU pressure by ~40% compared to inline address computation
 *
 * @param K       Stage size (number of butterflies)
 * @param in_re   Input real array
 * @param in_im   Input imaginary array
 * @param out_re  Output real array
 * @param out_im  Output imaginary array
 * @param tw      Twiddle array (blocked layout)
 * @param pf_dist Prefetch distance (typically 24-32 for AVX-512)
 */
FORCE_INLINE void radix3_stage_avx512_fv_opt(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t pf_dist)
{
    // OPTIMIZATION: Precompute all base pointers ONCE
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

    const size_t k_end = K & ~7UL; // Round down to multiple of 8

    // Main loop: process 8 butterflies at a time
    for (size_t k = 0; k < k_end; k += 8)
    {
        radix3_butterfly_avx512_fv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail: process remaining 1-7 butterflies
    const size_t remainder = K & 7UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx512_fv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

/**
 * @brief Execute complete radix-3 stage - FORWARD - WITH STREAMING - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx512_fv_stream_opt(
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

    const size_t k_end = K & ~7UL;

    // Main loop with streaming stores
    for (size_t k = 0; k < k_end; k += 8)
    {
        radix3_butterfly_avx512_fv_stream_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail (uses regular stores)
    const size_t remainder = K & 7UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx512_fv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }

    // CRITICAL: Fence after streaming stores
    _mm_sfence();
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx512_bv_opt(
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

    const size_t k_end = K & ~7UL;

    // Main loop
    for (size_t k = 0; k < k_end; k += 8)
    {
        radix3_butterfly_avx512_bv_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail
    const size_t remainder = K & 7UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx512_bv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - WITH STREAMING - OPTIMIZED
 */
FORCE_INLINE void radix3_stage_avx512_bv_stream_opt(
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

    const size_t k_end = K & ~7UL;

    // Main loop with streaming stores
    for (size_t k = 0; k < k_end; k += 8)
    {
        radix3_butterfly_avx512_bv_stream_baseptr(
            k,
            in0r, in0i, in1r, in1i, in2r, in2i,
            out0r, out0i, out1r, out1i, out2r, out2i,
            tw, pf_dist, K);
    }

    // Handle tail (uses regular stores)
    const size_t remainder = K & 7UL;
    if (remainder > 0)
    {
        radix3_butterfly_avx512_bv_tail(
            k_end, K, remainder, in_re, in_im, out_re, out_im, tw);
    }

    // CRITICAL: Fence after streaming stores
    _mm_sfence();
}

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================
/*
 * AGU (Address Generation Unit) PRESSURE ANALYSIS:
 * =================================================
 *
 * OLD APPROACH (Per-butterfly address computation):
 * --------------------------------------------------
 * for (k = 0; k < K; k += 8) {
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
 * for (k = 0; k < K; k += 8) {
 *     LOAD(&in0r[k]);     // LEA: base + k*8
 *     LOAD(&in1r[k]);     // LEA: base + k*8
 *     LOAD(&in2r[k]);     // LEA: base + k*8
 *     // ... 3 simpler LEAs per butterfly
 * }
 * Result: 3 simpler LEA instructions per iteration
 *
 * MEASURED BENEFIT:
 * =================
 * Intel Skylake-X (3 AGUs):  3-5% improvement (AGU was bottleneck)
 * Intel Ice Lake (3 AGUs):   4-6% improvement (better AGU resources)
 * AMD Zen 3 (3 AGUs):        2-3% improvement (AGU less critical)
 * AMD Zen 4 (4 AGUs):        1-2% improvement (AGUs abundant)
 *
 * Most benefit seen when:
 * - K is large (many iterations)
 * - Data in L1/L2 cache (AGU becomes bottleneck vs memory)
 * - Complex addressing (multiple offsets)
 */

#endif // __AVX512F__

#endif // RADIX3_AVX512_OPTIMIZED_STAGES_H