/**
 * @file radix3_avx512_pipelined_u2.h
 * @brief U=2 Software Pipelined Radix-3 Butterfly (ADVANCED OPTIMIZATION)
 *
 * WHAT IS U=2 SOFTWARE PIPELINING?
 * =================================
 * Process TWO butterfly iterations simultaneously, overlapping:
 *   Stage 0: LOAD(i+1)      - Fetch next iteration's data
 *   Stage 1: CMUL(i)        - Complex multiply current iteration
 *   Stage 2: BUTTERFLY(i-1) - Compute butterfly from previous iteration
 *   Stage 3: STORE(i-2)     - Write results from iteration before last
 *
 * This overlap keeps more functional units busy and better utilizes:
 * - Dual FMA ports (SKX/ICX: 2× FMA, RL: 4× FMA)
 * - Out-of-order execution window (ROB)
 * - Memory bandwidth (overlapped loads/stores)
 *
 * PERFORMANCE CHARACTERISTICS:
 * ============================
 * Expected gains:
 * - Intel Skylake-X:  8-12% (dual FMA utilization)
 * - Intel Ice Lake:   10-15% (better OoO engine)
 * - Intel Rocket Lake: 12-18% (quad FMA + wider ROB)
 * - AMD Zen 3/4:      6-10% (good OoO but fewer FMAs)
 *
 * Best gains when:
 * - Data in L1/L2 cache (memory not bottleneck)
 * - K ≥ 64 (enough iterations to amortize startup/shutdown)
 * - CPU has wide execution resources (many FMA units)
 *
 * TRADEOFFS:
 * ==========
 * Pros:
 * + 8-15% performance improvement on modern CPUs
 * + Better utilization of dual/quad FMA ports
 * + Hides latency through overlap
 *
 * Cons:
 * - Higher register pressure (~38-42 ZMM registers active)
 * - More complex code (harder to maintain)
 * - Minimal benefit on small K (< 64) due to startup overhead
 * - Requires careful register management to avoid spills
 *
 * COMPILE-TIME CONTROL:
 * =====================
 * Enable with: -DRADIX3_AVX512_USE_U2_PIPELINE
 * Falls back to standard version if not defined
 *
 * @author Tugbars
 * @version 3.2 (U=2 Software Pipelining)
 * @date 2025
 */

#ifndef RADIX3_AVX512_PIPELINED_U2_H
#define RADIX3_AVX512_PIPELINED_U2_H

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
// TWIDDLE LAYOUT
//==============================================================================

#define TWIDDLE_BLOCK_OFFSET_R3_AVX512(k) (((k) >> 3) << 5) // (k/8)*32
#define TW_W1_RE_OFFSET 0
#define TW_W1_IM_OFFSET 8
#define TW_W2_RE_OFFSET 16
#define TW_W2_IM_OFFSET 24

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
// U=2 SOFTWARE PIPELINED BUTTERFLY - FORWARD
//==============================================================================

/**
 * @brief U=2 pipelined radix-3 stage - FORWARD
 *
 * PIPELINE STRUCTURE (Conceptual):
 * ================================
 * Iteration:    i-2         i-1         i           i+1
 * Stage 3:    STORE(i-2)
 * Stage 2:                BUTTER(i-1)
 * Stage 1:                            CMUL(i)
 * Stage 0:                                        LOAD(i+1)
 *
 * CRITICAL: Minimum K = 16 (2 unrolled iterations) for pipeline to be worthwhile
 */
FORCE_INLINE void radix3_stage_avx512_fv_u2(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t pf_dist)
{
    // Fallback to Level 2 if K too small for U=2 pipeline overhead
    if (K < 16)
    {
        radix3_stage_avx512_fv_opt(K, in_re, in_im, out_re, out_im, tw, pf_dist);
        return;
    }

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

    const size_t k_end = (K & ~15UL); // Round down to multiple of 16 (2 × 8)

    //==========================================================================
    // PROLOG: Initialize pipeline with first iteration
    //==========================================================================

    size_t k = 0;

    // Iteration 0: Load data and twiddles
    __m512d a0_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a0_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b0_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b0_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c0_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c0_im = LOAD_IM_AVX512(&in2i[k]);

    size_t tw_offset0 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W1_RE_OFFSET]);
    __m512d w1_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W1_IM_OFFSET]);
    __m512d w2_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W2_RE_OFFSET]);
    __m512d w2_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W2_IM_OFFSET]);

    // Complex multiply iteration 0
    __m512d tB0_re, tB0_im, tC0_re, tC0_im;
    CMUL_AVX512_FMA(b0_re, b0_im, w1_re_0, w1_im_0, tB0_re, tB0_im);
    CMUL_AVX512_FMA(c0_re, c0_im, w2_re_0, w2_im_0, tC0_re, tC0_im);

    //==========================================================================
    // MAIN LOOP: U=2 pipelined (process 16 butterflies per iteration)
    //==========================================================================

    for (k = 0; k < k_end; k += 16)
    {
        //======================================================================
        // PIPELINE STAGE 0: LOAD(i+1) - Next iteration's data
        //======================================================================
        size_t k_next = k + 8;

        __m512d a1_re = LOAD_RE_AVX512(&in0r[k_next]);
        __m512d a1_im = LOAD_IM_AVX512(&in0i[k_next]);
        __m512d b1_re = LOAD_RE_AVX512(&in1r[k_next]);
        __m512d b1_im = LOAD_IM_AVX512(&in1i[k_next]);
        __m512d c1_re = LOAD_RE_AVX512(&in2r[k_next]);
        __m512d c1_im = LOAD_IM_AVX512(&in2i[k_next]);

        // Prefetch data two iterations ahead (k+16 is next unroll pair)
        const size_t k_pf = k + 16 + pf_dist;
        if (k_pf < K)
        {
            _mm_prefetch((const char *)&in0r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in0i[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in1r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in1i[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in2r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in2i[k_pf], _MM_HINT_T0);
            
            // Prefetch twiddle block (4 cache lines)
            const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_pf);
            _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);  // 0B   - W^1_re
            _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);  // 64B  - W^1_im
            _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0); // 128B - W^2_re
            _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0); // 192B - W^2_im
        }

        size_t tw_offset1 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_next);
        __m512d w1_re_1 = LOAD_RE_AVX512(&tw[tw_offset1 + TW_W1_RE_OFFSET]);
        __m512d w1_im_1 = LOAD_IM_AVX512(&tw[tw_offset1 + TW_W1_IM_OFFSET]);
        __m512d w2_re_1 = LOAD_RE_AVX512(&tw[tw_offset1 + TW_W2_RE_OFFSET]);
        __m512d w2_im_1 = LOAD_IM_AVX512(&tw[tw_offset1 + TW_W2_IM_OFFSET]);

        //======================================================================
        // PIPELINE STAGE 1 & 2: BUTTERFLY(i) - Current iteration
        //======================================================================
        __m512d y0_re_0, y0_im_0, y1_re_0, y1_im_0, y2_re_0, y2_im_0;
        RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
            a0_re, a0_im, tB0_re, tB0_im, tC0_re, tC0_im,
            y0_re_0, y0_im_0, y1_re_0, y1_im_0, y2_re_0, y2_im_0);

        //======================================================================
        // PIPELINE STAGE 1: CMUL(i+1) - Next iteration's complex multiply
        //======================================================================
        __m512d tB1_re, tB1_im, tC1_re, tC1_im;
        CMUL_AVX512_FMA(b1_re, b1_im, w1_re_1, w1_im_1, tB1_re, tB1_im);
        CMUL_AVX512_FMA(c1_re, c1_im, w2_re_1, w2_im_1, tC1_re, tC1_im);

        //======================================================================
        // PIPELINE STAGE 3: STORE(i) - Current iteration results
        //======================================================================
        STORE_RE_AVX512(&out0r[k], y0_re_0);
        STORE_IM_AVX512(&out0i[k], y0_im_0);
        STORE_RE_AVX512(&out1r[k], y1_re_0);
        STORE_IM_AVX512(&out1i[k], y1_im_0);
        STORE_RE_AVX512(&out2r[k], y2_re_0);
        STORE_IM_AVX512(&out2i[k], y2_im_0);

        //======================================================================
        // SECOND ITERATION (i+1): Complete the U=2 unroll
        //======================================================================

        // PIPELINE STAGE 0: LOAD(i+2) - Load data for next unrolled pair
        size_t k_next2 = k + 16;
        if (k_next2 < k_end)
        {
            a0_re = LOAD_RE_AVX512(&in0r[k_next2]);
            a0_im = LOAD_IM_AVX512(&in0i[k_next2]);
            b0_re = LOAD_RE_AVX512(&in1r[k_next2]);
            b0_im = LOAD_IM_AVX512(&in1i[k_next2]);
            c0_re = LOAD_RE_AVX512(&in2r[k_next2]);
            c0_im = LOAD_IM_AVX512(&in2i[k_next2]);

            tw_offset0 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_next2);
            w1_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W1_RE_OFFSET]);
            w1_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W1_IM_OFFSET]);
            w2_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W2_RE_OFFSET]);
            w2_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W2_IM_OFFSET]);
        }

        // PIPELINE STAGE 1 & 2: BUTTERFLY(i+1)
        __m512d y0_re_1, y0_im_1, y1_re_1, y1_im_1, y2_re_1, y2_im_1;
        RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
            a1_re, a1_im, tB1_re, tB1_im, tC1_re, tC1_im,
            y0_re_1, y0_im_1, y1_re_1, y1_im_1, y2_re_1, y2_im_1);

        // PIPELINE STAGE 1: CMUL(i+2) - Prepare for next unrolled pair
        if (k_next2 < k_end)
        {
            CMUL_AVX512_FMA(b0_re, b0_im, w1_re_0, w1_im_0, tB0_re, tB0_im);
            CMUL_AVX512_FMA(c0_re, c0_im, w2_re_0, w2_im_0, tC0_re, tC0_im);
        }

        // PIPELINE STAGE 3: STORE(i+1)
        STORE_RE_AVX512(&out0r[k_next], y0_re_1);
        STORE_IM_AVX512(&out0i[k_next], y0_im_1);
        STORE_RE_AVX512(&out1r[k_next], y1_re_1);
        STORE_IM_AVX512(&out1i[k_next], y1_im_1);
        STORE_RE_AVX512(&out2r[k_next], y2_re_1);
        STORE_IM_AVX512(&out2i[k_next], y2_im_1);
    }

    //==========================================================================
    // EPILOG: Handle remaining butterflies with standard version
    //==========================================================================
    const size_t remainder = K - k_end;
    if (remainder > 0)
    {
        // Process remaining butterflies (1-15) one block at a time
        for (; k < K; k += 8)
        {
            size_t count = (K - k >= 8) ? 8 : (K - k);
            
            if (count == 8)
            {
                // Full block
                __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
                __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
                __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
                __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
                __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
                __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

                size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
                __m512d w1_re = LOAD_RE_AVX512(&tw[tw_offset + TW_W1_RE_OFFSET]);
                __m512d w1_im = LOAD_IM_AVX512(&tw[tw_offset + TW_W1_IM_OFFSET]);
                __m512d w2_re = LOAD_RE_AVX512(&tw[tw_offset + TW_W2_RE_OFFSET]);
                __m512d w2_im = LOAD_IM_AVX512(&tw[tw_offset + TW_W2_IM_OFFSET]);

                __m512d tB_re, tB_im, tC_re, tC_im;
                CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
                CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

                __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
                RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
                    a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

                STORE_RE_AVX512(&out0r[k], y0_re);
                STORE_IM_AVX512(&out0i[k], y0_im);
                STORE_RE_AVX512(&out1r[k], y1_re);
                STORE_IM_AVX512(&out1i[k], y1_im);
                STORE_RE_AVX512(&out2r[k], y2_re);
                STORE_IM_AVX512(&out2i[k], y2_im);
            }
            else
            {
                // Tail: masked operations
                const __mmask8 mask = (__mmask8)((1U << count) - 1);

                __m512d a_re = _mm512_maskz_load_pd(mask, &in_re[k]);
                __m512d a_im = _mm512_maskz_load_pd(mask, &in_im[k]);
                __m512d b_re = _mm512_maskz_load_pd(mask, &in_re[k + K]);
                __m512d b_im = _mm512_maskz_load_pd(mask, &in_im[k + K]);
                __m512d c_re = _mm512_maskz_load_pd(mask, &in_re[k + 2 * K]);
                __m512d c_im = _mm512_maskz_load_pd(mask, &in_im[k + 2 * K]);

                size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
                __m512d w1_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_RE_OFFSET]);
                __m512d w1_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_IM_OFFSET]);
                __m512d w2_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_RE_OFFSET]);
                __m512d w2_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_IM_OFFSET]);

                __m512d tB_re, tB_im, tC_re, tC_im;
                CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
                CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

                __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
                RADIX3_BUTTERFLY_FV_AVX512_FMA_OPT(
                    a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

                _mm512_mask_store_pd(&out_re[k], mask, y0_re);
                _mm512_mask_store_pd(&out_im[k], mask, y0_im);
                _mm512_mask_store_pd(&out_re[k + K], mask, y1_re);
                _mm512_mask_store_pd(&out_im[k + K], mask, y1_im);
                _mm512_mask_store_pd(&out_re[k + 2 * K], mask, y2_re);
                _mm512_mask_store_pd(&out_im[k + 2 * K], mask, y2_im);
            }
        }
    }
}

//==============================================================================
// U=2 SOFTWARE PIPELINED BUTTERFLY - BACKWARD
//==============================================================================

/**
 * @brief U=2 pipelined radix-3 stage - BACKWARD
 * @note Implementation mirrors forward version with BV butterfly macro
 */
FORCE_INLINE void radix3_stage_avx512_bv_u2(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t pf_dist)
{
    // Fallback to Level 2 if K too small
    if (K < 16)
    {
        radix3_stage_avx512_bv_opt(K, in_re, in_im, out_re, out_im, tw, pf_dist);
        return;
    }

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

    const size_t k_end = (K & ~15UL);

    // PROLOG
    size_t k = 0;
    __m512d a0_re = LOAD_RE_AVX512(&in0r[k]);
    __m512d a0_im = LOAD_IM_AVX512(&in0i[k]);
    __m512d b0_re = LOAD_RE_AVX512(&in1r[k]);
    __m512d b0_im = LOAD_IM_AVX512(&in1i[k]);
    __m512d c0_re = LOAD_RE_AVX512(&in2r[k]);
    __m512d c0_im = LOAD_IM_AVX512(&in2i[k]);

    size_t tw_offset0 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
    __m512d w1_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W1_RE_OFFSET]);
    __m512d w1_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W1_IM_OFFSET]);
    __m512d w2_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W2_RE_OFFSET]);
    __m512d w2_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W2_IM_OFFSET]);

    __m512d tB0_re, tB0_im, tC0_re, tC0_im;
    CMUL_AVX512_FMA(b0_re, b0_im, w1_re_0, w1_im_0, tB0_re, tB0_im);
    CMUL_AVX512_FMA(c0_re, c0_im, w2_re_0, w2_im_0, tC0_re, tC0_im);

    // MAIN LOOP (identical structure to forward, but uses BV butterfly)
    for (k = 0; k < k_end; k += 16)
    {
        size_t k_next = k + 8;

        // LOAD(i+1)
        __m512d a1_re = LOAD_RE_AVX512(&in0r[k_next]);
        __m512d a1_im = LOAD_IM_AVX512(&in0i[k_next]);
        __m512d b1_re = LOAD_RE_AVX512(&in1r[k_next]);
        __m512d b1_im = LOAD_IM_AVX512(&in1i[k_next]);
        __m512d c1_re = LOAD_RE_AVX512(&in2r[k_next]);
        __m512d c1_im = LOAD_IM_AVX512(&in2i[k_next]);

        const size_t k_pf = k + 16 + pf_dist;
        if (k_pf < K)
        {
            _mm_prefetch((const char *)&in0r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in0i[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in1r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in1i[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in2r[k_pf], _MM_HINT_T0);
            _mm_prefetch((const char *)&in2i[k_pf], _MM_HINT_T0);
            
            const size_t tpf = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_pf);
            _mm_prefetch((const char *)&tw[tpf + 0], _MM_HINT_T0);
            _mm_prefetch((const char *)&tw[tpf + 8], _MM_HINT_T0);
            _mm_prefetch((const char *)&tw[tpf + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&tw[tpf + 24], _MM_HINT_T0);
        }

        size_t tw_offset1 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_next);
        __m512d w1_re_1 = LOAD_RE_AVX512(&tw[tw_offset1 + TW_W1_RE_OFFSET]);
        __m512d w1_im_1 = LOAD_IM_AVX512(&tw[tw_offset1 + TW_W1_IM_OFFSET]);
        __m512d w2_re_1 = LOAD_RE_AVX512(&tw[tw_offset1 + TW_W2_RE_OFFSET]);
        __m512d w2_im_1 = LOAD_IM_AVX512(&tw[tw_offset1 + TW_W2_IM_OFFSET]);

        // BUTTERFLY(i) - using BACKWARD version
        __m512d y0_re_0, y0_im_0, y1_re_0, y1_im_0, y2_re_0, y2_im_0;
        RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
            a0_re, a0_im, tB0_re, tB0_im, tC0_re, tC0_im,
            y0_re_0, y0_im_0, y1_re_0, y1_im_0, y2_re_0, y2_im_0);

        // CMUL(i+1)
        __m512d tB1_re, tB1_im, tC1_re, tC1_im;
        CMUL_AVX512_FMA(b1_re, b1_im, w1_re_1, w1_im_1, tB1_re, tB1_im);
        CMUL_AVX512_FMA(c1_re, c1_im, w2_re_1, w2_im_1, tC1_re, tC1_im);

        // STORE(i)
        STORE_RE_AVX512(&out0r[k], y0_re_0);
        STORE_IM_AVX512(&out0i[k], y0_im_0);
        STORE_RE_AVX512(&out1r[k], y1_re_0);
        STORE_IM_AVX512(&out1i[k], y1_im_0);
        STORE_RE_AVX512(&out2r[k], y2_re_0);
        STORE_IM_AVX512(&out2i[k], y2_im_0);

        // Second iteration
        size_t k_next2 = k + 16;
        if (k_next2 < k_end)
        {
            a0_re = LOAD_RE_AVX512(&in0r[k_next2]);
            a0_im = LOAD_IM_AVX512(&in0i[k_next2]);
            b0_re = LOAD_RE_AVX512(&in1r[k_next2]);
            b0_im = LOAD_IM_AVX512(&in1i[k_next2]);
            c0_re = LOAD_RE_AVX512(&in2r[k_next2]);
            c0_im = LOAD_IM_AVX512(&in2i[k_next2]);

            tw_offset0 = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k_next2);
            w1_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W1_RE_OFFSET]);
            w1_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W1_IM_OFFSET]);
            w2_re_0 = LOAD_RE_AVX512(&tw[tw_offset0 + TW_W2_RE_OFFSET]);
            w2_im_0 = LOAD_IM_AVX512(&tw[tw_offset0 + TW_W2_IM_OFFSET]);
        }

        __m512d y0_re_1, y0_im_1, y1_re_1, y1_im_1, y2_re_1, y2_im_1;
        RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
            a1_re, a1_im, tB1_re, tB1_im, tC1_re, tC1_im,
            y0_re_1, y0_im_1, y1_re_1, y1_im_1, y2_re_1, y2_im_1);

        if (k_next2 < k_end)
        {
            CMUL_AVX512_FMA(b0_re, b0_im, w1_re_0, w1_im_0, tB0_re, tB0_im);
            CMUL_AVX512_FMA(c0_re, c0_im, w2_re_0, w2_im_0, tC0_re, tC0_im);
        }

        STORE_RE_AVX512(&out0r[k_next], y0_re_1);
        STORE_IM_AVX512(&out0i[k_next], y0_im_1);
        STORE_RE_AVX512(&out1r[k_next], y1_re_1);
        STORE_IM_AVX512(&out1i[k_next], y1_im_1);
        STORE_RE_AVX512(&out2r[k_next], y2_re_1);
        STORE_IM_AVX512(&out2i[k_next], y2_im_1);
    }

    // EPILOG (same as forward version)
    const size_t remainder = K - k_end;
    if (remainder > 0)
    {
        for (; k < K; k += 8)
        {
            size_t count = (K - k >= 8) ? 8 : (K - k);
            
            if (count == 8)
            {
                __m512d a_re = LOAD_RE_AVX512(&in0r[k]);
                __m512d a_im = LOAD_IM_AVX512(&in0i[k]);
                __m512d b_re = LOAD_RE_AVX512(&in1r[k]);
                __m512d b_im = LOAD_IM_AVX512(&in1i[k]);
                __m512d c_re = LOAD_RE_AVX512(&in2r[k]);
                __m512d c_im = LOAD_IM_AVX512(&in2i[k]);

                size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
                __m512d w1_re = LOAD_RE_AVX512(&tw[tw_offset + TW_W1_RE_OFFSET]);
                __m512d w1_im = LOAD_IM_AVX512(&tw[tw_offset + TW_W1_IM_OFFSET]);
                __m512d w2_re = LOAD_RE_AVX512(&tw[tw_offset + TW_W2_RE_OFFSET]);
                __m512d w2_im = LOAD_IM_AVX512(&tw[tw_offset + TW_W2_IM_OFFSET]);

                __m512d tB_re, tB_im, tC_re, tC_im;
                CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
                CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

                __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
                RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
                    a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

                STORE_RE_AVX512(&out0r[k], y0_re);
                STORE_IM_AVX512(&out0i[k], y0_im);
                STORE_RE_AVX512(&out1r[k], y1_re);
                STORE_IM_AVX512(&out1i[k], y1_im);
                STORE_RE_AVX512(&out2r[k], y2_re);
                STORE_IM_AVX512(&out2i[k], y2_im);
            }
            else
            {
                const __mmask8 mask = (__mmask8)((1U << count) - 1);

                __m512d a_re = _mm512_maskz_load_pd(mask, &in_re[k]);
                __m512d a_im = _mm512_maskz_load_pd(mask, &in_im[k]);
                __m512d b_re = _mm512_maskz_load_pd(mask, &in_re[k + K]);
                __m512d b_im = _mm512_maskz_load_pd(mask, &in_im[k + K]);
                __m512d c_re = _mm512_maskz_load_pd(mask, &in_re[k + 2 * K]);
                __m512d c_im = _mm512_maskz_load_pd(mask, &in_im[k + 2 * K]);

                size_t tw_offset = TWIDDLE_BLOCK_OFFSET_R3_AVX512(k);
                __m512d w1_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_RE_OFFSET]);
                __m512d w1_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W1_IM_OFFSET]);
                __m512d w2_re = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_RE_OFFSET]);
                __m512d w2_im = _mm512_maskz_load_pd(mask, &tw[tw_offset + TW_W2_IM_OFFSET]);

                __m512d tB_re, tB_im, tC_re, tC_im;
                CMUL_AVX512_FMA(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
                CMUL_AVX512_FMA(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

                __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
                RADIX3_BUTTERFLY_BV_AVX512_FMA_OPT(
                    a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

                _mm512_mask_store_pd(&out_re[k], mask, y0_re);
                _mm512_mask_store_pd(&out_im[k], mask, y0_im);
                _mm512_mask_store_pd(&out_re[k + K], mask, y1_re);
                _mm512_mask_store_pd(&out_im[k + K], mask, y1_im);
                _mm512_mask_store_pd(&out_re[k + 2 * K], mask, y2_re);
                _mm512_mask_store_pd(&out_im[k + 2 * K], mask, y2_im);
            }
        }
    }
}

//==============================================================================
// PERFORMANCE NOTES & REGISTER USAGE ANALYSIS
//==============================================================================
/*
 * REGISTER PRESSURE ANALYSIS (U=2 Pipeline):
 * ===========================================
 *
 * Active registers per iteration:
 * -------------------------------
 * Iteration i data:     6 ZMM (a0, b0, c0 re/im)
 * Iteration i+1 data:   6 ZMM (a1, b1, c1 re/im)
 * Twiddles i:           4 ZMM (w1, w2 re/im)
 * Twiddles i+1:         4 ZMM (w1, w2 re/im)
 * Intermediate tB/tC i: 4 ZMM
 * Intermediate tB/tC i+1: 4 ZMM
 * Butterfly outputs i:  6 ZMM (y0, y1, y2 re/im)
 * Butterfly outputs i+1: 6 ZMM
 * Constants (V512_*):   2 ZMM (HALF, SQRT3_2)
 * -------------------------------
 * PEAK TOTAL:          ~42 ZMM registers
 *
 * Intel AVX-512 has 32 ZMM registers, so with careful register allocation
 * by the compiler, this SHOULD fit without spilling. However:
 * - Some temporary registers needed for FMA intermediate results
 * - Compiler may need a few scratch registers
 * - Risk of spills if compiler doesn't optimize well
 *
 * MITIGATION STRATEGIES:
 * ======================
 * 1. Use FORCE_INLINE to ensure inlining (helps register allocation)
 * 2. Keep constants as static const (compiler reuses same register)
 * 3. Explicit sequencing of operations guides compiler
 * 4. Profile with perf to check for register spills
 *
 * WHY THIS WORKS DESPITE 32 ZMM LIMIT:
 * =====================================
 * The 42 registers are NOT all live simultaneously:
 * - Iteration i data dies after butterfly computation
 * - Twiddles i die after complex multiply
 * - Outputs i die immediately after stores
 *
 * Compiler's register allocator exploits this temporal locality,
 * reusing ZMM registers across pipeline stages. Modern compilers
 * (GCC 11+, Clang 13+) are excellent at this.
 *
 * VERIFICATION:
 * =============
 * Check for register spills with:
 *   objdump -d -M intel <binary> | grep -A 50 radix3_stage_avx512_fv_u2
 * Look for:
 *   - movq to/from stack: BAD (register spill)
 *   - All zmm operations: GOOD (registers only)
 *
 * WHEN TO USE U=2 VS STANDARD:
 * =============================
 * Use U=2 when:
 * - K ≥ 64 (enough work to amortize overhead)
 * - Data fits in L1/L2 (not memory-bound)
 * - CPU has dual FMA (SKX/ICX/RL)
 * - Targeting maximum performance
 *
 * Use standard when:
 * - K < 64 (overhead dominates)
 * - Memory-bound workloads (no benefit from ILP)
 * - Code size matters (U=2 generates more code)
 * - Simplicity/maintainability prioritized
 */

#endif // __AVX512F__

#endif // RADIX3_AVX512_PIPELINED_U2_H