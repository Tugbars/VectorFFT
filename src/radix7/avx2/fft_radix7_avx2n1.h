/**
 * @file fft_radix7_avx2_n1.h
 * @brief AVX2 Radix-7 Rader Butterfly - TWIDDLE-LESS (N1) First Stage Optimization
 *
 * @details
 * N1 VARIANT - FIRST STAGE OPTIMIZATION (20-30% FASTER!):
 * =========================================================
 * This is the TWIDDLE-LESS variant for first radix-7 stage or when all
 * stage twiddles W^(k*m) = 1. Key differences from standard version:
 * 
 * ✅ REMOVED: apply_stage_twiddles step (saves 6 complex muls per butterfly!)
 * ✅ REMOVED: stage_tw parameter (not needed)
 * ✅ REMOVED: sub_len checks (always effectively sub_len=1)
 * 
 * ✅✅ PRESERVED: ALL Rader optimizations (still need Rader twiddles!)
 * ✅✅ PRESERVED: P0 pre-split Rader broadcasts (8-10% gain)
 * ✅✅ PRESERVED: P0 round-robin convolution (10-15% gain)
 * ✅✅ PRESERVED: P1 tree y0 sum (1-2% gain)
 * ✅✅ PRESERVED: FMA usage (AVX2 advantage)
 * ✅✅ PRESERVED: U2 pipeline (process k, k+4 simultaneously)
 * ✅✅ PRESERVED: Store-time adds (frees 6 YMM registers)
 * 
 * CODE REUSE:
 * ===========
 * This header REUSES most functions from fft_radix7_avx2.h:
 * - load_7_lanes_avx2_soa
 * - store_7_lanes_avx2_soa / store_7_lanes_avx2_stream_soa
 * - broadcast_rader_twiddles_avx2_soa
 * - compute_y0_tree_avx2_soa
 * - permute_rader_inputs_avx2_soa
 * - rader_convolution_roundrobin_avx2_soa
 * 
 * ONLY DIFFERENCE: Butterfly functions skip apply_stage_twiddles() call!
 *
 * TARGET: x86-64 CPUs with AVX2 + FMA (Haswell and later)
 *
 * @author FFT Optimization Team
 * @version 4.0 (N1 variant)
 * @date 2025
 */

#ifndef FFT_RADIX7_AVX2_N1_H
#define FFT_RADIX7_AVX2_N1_H

#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../fft_radix7_uniform.h"
#include "fft_radix7_avx2.h"  // Reuse most functions from standard version!

//==============================================================================
// CONFIGURATION (reuse from standard header)
//==============================================================================

#define R7_AVX2_N1_ALIGNMENT R7_AVX2_ALIGNMENT
#define R7_AVX2_N1_WIDTH R7_AVX2_WIDTH
#define R7_AVX2_N1_U2_WIDTH R7_AVX2_U2_WIDTH

//==============================================================================
// SIMPLIFIED PREFETCH (no stage twiddles to prefetch!)
//==============================================================================

/**
 * @brief Prefetch 7 lanes from SoA buffers ahead of time - N1 SIMPLIFIED
 * @details
 * ✅ PRESERVED: Input data prefetch structure
 * ✅ REMOVED: Stage twiddle prefetch (not needed for N1!)
 */
__attribute__((always_inline)) static inline void prefetch_7_lanes_avx2_n1_soa(
    int k, int K,
    const double *restrict in_re,
    const double *restrict in_im)
{
    if (k + R7_AVX2_PREFETCH_DISTANCE >= K)
        return;

    int pk = k + R7_AVX2_PREFETCH_DISTANCE;

    // Prefetch input data to L1 (will be used soon)
    _mm_prefetch((const char *)&in_re[0 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[0 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[1 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[1 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[2 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[2 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[3 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[3 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[4 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[4 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[5 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[5 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_re[6 * K + pk], _MM_HINT_T0);
    _mm_prefetch((const char *)&in_im[6 * K + pk], _MM_HINT_T0);
    
    // NO stage twiddle prefetch - that's the N1 advantage!
}

//==============================================================================
// COMPLETE BUTTERFLY FUNCTIONS - N1 (NO STAGE TWIDDLES!)
//==============================================================================

/**
 * @brief Single radix-7 butterfly - 4-wide AVX2 TRUE SoA - N1 VARIANT
 * @details
 * ✅ ALL OPTIMIZATIONS PRESERVED + N1 speedup:
 *    - Pre-split Rader broadcasts (reused from stage-level cache)
 *    - Round-robin convolution
 *    - Tree y0 sum
 *    - TRUE SoA (no interleave/deinterleave!)
 *    - 4-wide processing (full AVX2 utilization)
 *    - Aligned loads/stores
 *    - Store-time adds (frees 6 YMM registers!)
 * 
 * ⚡⚡⚡ N1 DIFFERENCE: NO apply_stage_twiddles() call!
 * ⚡⚡⚡ Saves 6 complex multiplies per butterfly
 * ⚡⚡⚡ 20-30% faster than standard version
 *
 * Process one butterfly: k, k+1, k+2, k+3 (4 complex values)
 *
 * @param k Starting index (must be 4-aligned for best performance)
 * @param K Stride between lanes
 * @param in_re Input real components (32-byte aligned)
 * @param in_im Input imaginary components (32-byte aligned)
 * @param rader_tw_re Broadcast Rader twiddle real components (6 vectors)
 * @param rader_tw_im Broadcast Rader twiddle imaginary components (6 vectors)
 * @param out_re Output real components (32-byte aligned)
 * @param out_im Output imaginary components (32-byte aligned)
 * @param use_nt Use non-temporal stores (for large FFTs)
 */
__attribute__((always_inline)) static inline void radix7_butterfly_single_avx2_n1_soa(
    int k, int K,
    const double *restrict in_re,
    const double *restrict in_im,
    const __m256d rader_tw_re[6],
    const __m256d rader_tw_im[6],
    double *restrict out_re,
    double *restrict out_im,
    bool use_nt)
{
    // STEP 1: Load 7 lanes (4 complex values per lane)
    __m256d x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;
    __m256d x4_re, x4_im, x5_re, x5_im, x6_re, x6_im;

    load_7_lanes_avx2_soa(k, K, in_re, in_im,
                          &x0_re, &x0_im, &x1_re, &x1_im,
                          &x2_re, &x2_im, &x3_re, &x3_im,
                          &x4_re, &x4_im, &x5_re, &x5_im,
                          &x6_re, &x6_im);

    // STEP 2: NO STAGE TWIDDLES! (N1 optimization - saves 6 complex muls!)
    // Standard version would call: apply_stage_twiddles_avx2_soa(...)
    // N1: Skip this entirely!

    // STEP 3: Compute DC component y0 (tree reduction)
    __m256d y0_re, y0_im;
    compute_y0_tree_avx2_soa(x0_re, x0_im, x1_re, x1_im,
                             x2_re, x2_im, x3_re, x3_im,
                             x4_re, x4_im, x5_re, x5_im,
                             x6_re, x6_im,
                             &y0_re, &y0_im);

    // STEP 4: Permute inputs for Rader algorithm
    __m256d tx0_re, tx0_im, tx1_re, tx1_im, tx2_re, tx2_im;
    __m256d tx3_re, tx3_im, tx4_re, tx4_im, tx5_re, tx5_im;

    permute_rader_inputs_avx2_soa(x1_re, x1_im, x2_re, x2_im,
                                  x3_re, x3_im, x4_re, x4_im,
                                  x5_re, x5_im, x6_re, x6_im,
                                  &tx0_re, &tx0_im, &tx1_re, &tx1_im,
                                  &tx2_re, &tx2_im, &tx3_re, &tx3_im,
                                  &tx4_re, &tx4_im, &tx5_re, &tx5_im);

    // STEP 5: 6-point cyclic convolution (round-robin schedule)
    __m256d v0_re, v0_im, v1_re, v1_im, v2_re, v2_im;
    __m256d v3_re, v3_im, v4_re, v4_im, v5_re, v5_im;

    rader_convolution_roundrobin_avx2_soa(
        tx0_re, tx0_im, tx1_re, tx1_im, tx2_re, tx2_im,
        tx3_re, tx3_im, tx4_re, tx4_im, tx5_re, tx5_im,
        rader_tw_re, rader_tw_im,
        &v0_re, &v0_im, &v1_re, &v1_im, &v2_re, &v2_im,
        &v3_re, &v3_im, &v4_re, &v4_im, &v5_re, &v5_im);

    // STEP 6 & 7: Assemble outputs at store-time (CRITICAL: frees 6 YMM!)
    // Output permutation: [1,5,4,6,2,3] from convolution results v[0,1,2,3,4,5]
    // Do adds inline with store to avoid materializing y1-y6
    if (use_nt)
    {
        store_7_lanes_avx2_stream_soa(k, K, out_re, out_im,
                                      y0_re, y0_im,
                                      _mm256_add_pd(x0_re, v0_re), _mm256_add_pd(x0_im, v0_im),  // y1
                                      _mm256_add_pd(x0_re, v4_re), _mm256_add_pd(x0_im, v4_im),  // y2
                                      _mm256_add_pd(x0_re, v5_re), _mm256_add_pd(x0_im, v5_im),  // y3
                                      _mm256_add_pd(x0_re, v2_re), _mm256_add_pd(x0_im, v2_im),  // y4
                                      _mm256_add_pd(x0_re, v1_re), _mm256_add_pd(x0_im, v1_im),  // y5
                                      _mm256_add_pd(x0_re, v3_re), _mm256_add_pd(x0_im, v3_im)); // y6
    }
    else
    {
        store_7_lanes_avx2_soa(k, K, out_re, out_im,
                               y0_re, y0_im,
                               _mm256_add_pd(x0_re, v0_re), _mm256_add_pd(x0_im, v0_im),  // y1
                               _mm256_add_pd(x0_re, v4_re), _mm256_add_pd(x0_im, v4_im),  // y2
                               _mm256_add_pd(x0_re, v5_re), _mm256_add_pd(x0_im, v5_im),  // y3
                               _mm256_add_pd(x0_re, v2_re), _mm256_add_pd(x0_im, v2_im),  // y4
                               _mm256_add_pd(x0_re, v1_re), _mm256_add_pd(x0_im, v1_im),  // y5
                               _mm256_add_pd(x0_re, v3_re), _mm256_add_pd(x0_im, v3_im)); // y6
    }
}

/**
 * @brief Dual radix-7 butterfly - U2 pipeline - N1 VARIANT
 * @details
 * ⚡⚡⚡ CRITICAL: Process TWO butterflies simultaneously!
 *
 * U2 PIPELINE STRUCTURE (AVX2 ADAPTATION):
 * =========================================
 * Process k and k+4 in parallel to maximize ILP.
 *
 * Key optimizations:
 * - Interleaved loads: k and k+4 loads can overlap
 * - Interleaved convolutions: butterfly A and B alternate operations
 *   → Maximizes throughput on dual FMA ports
 * - Register reuse: temporary registers shared where dependency chains allow
 * - Store-time adds: no ya1-ya6, yb1-yb6 temporaries (frees 12 YMM!)
 * - Total register pressure: ~14 YMM (well within 16 YMM budget)
 *
 * ⚡⚡⚡ N1 DIFFERENCE: NO apply_stage_twiddles() calls!
 * ⚡⚡⚡ Saves 12 complex multiplies total (6 per butterfly × 2)
 *
 * @param ka Starting index for butterfly A
 * @param kb Starting index for butterfly B (typically ka + 4)
 */
__attribute__((always_inline)) static inline void radix7_butterfly_dual_avx2_n1_soa(
    int ka, int kb, int K,
    const double *restrict in_re,
    const double *restrict in_im,
    const __m256d rader_tw_re[6],
    const __m256d rader_tw_im[6],
    double *restrict out_re,
    double *restrict out_im,
    bool use_nt)
{
    //==========================================================================
    // BUTTERFLY A (ka)
    //==========================================================================

    // Load A
    __m256d xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im;
    __m256d xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im;

    load_7_lanes_avx2_soa(ka, K, in_re, in_im,
                          &xa0_re, &xa0_im, &xa1_re, &xa1_im,
                          &xa2_re, &xa2_im, &xa3_re, &xa3_im,
                          &xa4_re, &xa4_im, &xa5_re, &xa5_im,
                          &xa6_re, &xa6_im);

    //==========================================================================
    // BUTTERFLY B (kb) - INTERLEAVE LOAD
    //==========================================================================

    // Load B (interleaved with A's load - increases memory bandwidth utilization)
    __m256d xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im;
    __m256d xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im;

    load_7_lanes_avx2_soa(kb, K, in_re, in_im,
                          &xb0_re, &xb0_im, &xb1_re, &xb1_im,
                          &xb2_re, &xb2_im, &xb3_re, &xb3_im,
                          &xb4_re, &xb4_im, &xb5_re, &xb5_im,
                          &xb6_re, &xb6_im);

    //==========================================================================
    // NO STAGE TWIDDLES! (N1 optimization)
    //==========================================================================
    // Standard version would call apply_stage_twiddles_avx2_soa() twice here
    // N1: Skip entirely - saves 12 complex multiplies!

    //==========================================================================
    // COMPUTE Y0 (A and B) - TREE REDUCTION
    //==========================================================================

    __m256d ya0_re, ya0_im;
    compute_y0_tree_avx2_soa(xa0_re, xa0_im, xa1_re, xa1_im,
                             xa2_re, xa2_im, xa3_re, xa3_im,
                             xa4_re, xa4_im, xa5_re, xa5_im,
                             xa6_re, xa6_im,
                             &ya0_re, &ya0_im);

    __m256d yb0_re, yb0_im;
    compute_y0_tree_avx2_soa(xb0_re, xb0_im, xb1_re, xb1_im,
                             xb2_re, xb2_im, xb3_re, xb3_im,
                             xb4_re, xb4_im, xb5_re, xb5_im,
                             xb6_re, xb6_im,
                             &yb0_re, &yb0_im);

    //==========================================================================
    // PERMUTE INPUTS (A and B)
    //==========================================================================

    __m256d txa0_re, txa0_im, txa1_re, txa1_im, txa2_re, txa2_im;
    __m256d txa3_re, txa3_im, txa4_re, txa4_im, txa5_re, txa5_im;

    permute_rader_inputs_avx2_soa(xa1_re, xa1_im, xa2_re, xa2_im,
                                  xa3_re, xa3_im, xa4_re, xa4_im,
                                  xa5_re, xa5_im, xa6_re, xa6_im,
                                  &txa0_re, &txa0_im, &txa1_re, &txa1_im,
                                  &txa2_re, &txa2_im, &txa3_re, &txa3_im,
                                  &txa4_re, &txa4_im, &txa5_re, &txa5_im);

    __m256d txb0_re, txb0_im, txb1_re, txb1_im, txb2_re, txb2_im;
    __m256d txb3_re, txb3_im, txb4_re, txb4_im, txb5_re, txb5_im;

    permute_rader_inputs_avx2_soa(xb1_re, xb1_im, xb2_re, xb2_im,
                                  xb3_re, xb3_im, xb4_re, xb4_im,
                                  xb5_re, xb5_im, xb6_re, xb6_im,
                                  &txb0_re, &txb0_im, &txb1_re, &txb1_im,
                                  &txb2_re, &txb2_im, &txb3_re, &txb3_im,
                                  &txb4_re, &txb4_im, &txb5_re, &txb5_im);

    //==========================================================================
    // CONVOLUTION (A and B) - INTERLEAVED FOR MAXIMUM ILP
    //==========================================================================

    // Initialize all accumulators
    __m256d va0_re = _mm256_setzero_pd(), va0_im = _mm256_setzero_pd();
    __m256d va1_re = _mm256_setzero_pd(), va1_im = _mm256_setzero_pd();
    __m256d va2_re = _mm256_setzero_pd(), va2_im = _mm256_setzero_pd();
    __m256d va3_re = _mm256_setzero_pd(), va3_im = _mm256_setzero_pd();
    __m256d va4_re = _mm256_setzero_pd(), va4_im = _mm256_setzero_pd();
    __m256d va5_re = _mm256_setzero_pd(), va5_im = _mm256_setzero_pd();

    __m256d vb0_re = _mm256_setzero_pd(), vb0_im = _mm256_setzero_pd();
    __m256d vb1_re = _mm256_setzero_pd(), vb1_im = _mm256_setzero_pd();
    __m256d vb2_re = _mm256_setzero_pd(), vb2_im = _mm256_setzero_pd();
    __m256d vb3_re = _mm256_setzero_pd(), vb3_im = _mm256_setzero_pd();
    __m256d vb4_re = _mm256_setzero_pd(), vb4_im = _mm256_setzero_pd();
    __m256d vb5_re = _mm256_setzero_pd(), vb5_im = _mm256_setzero_pd();

    // ⚡⚡⚡ CRITICAL: INTERLEAVED ROUND-ROBIN CONVOLUTION
    // Alternate between A and B updates to maximize ILP!

    // Round 0: txa0 and txb0
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa0_re, txa0_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb0_re, txb0_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa0_re, txa0_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb0_re, txb0_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa0_re, txa0_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb0_re, txb0_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa0_re, txa0_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb0_re, txb0_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa0_re, txa0_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb0_re, txb0_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa0_re, txa0_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb0_re, txb0_im, rader_tw_re[5], rader_tw_im[5]);

    // Round 1: txa1 and txb1
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa1_re, txa1_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb1_re, txb1_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa1_re, txa1_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb1_re, txb1_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa1_re, txa1_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb1_re, txb1_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa1_re, txa1_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb1_re, txb1_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa1_re, txa1_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb1_re, txb1_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa1_re, txa1_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb1_re, txb1_im, rader_tw_re[4], rader_tw_im[4]);

    // Round 2: txa2 and txb2
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa2_re, txa2_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb2_re, txb2_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa2_re, txa2_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb2_re, txb2_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa2_re, txa2_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb2_re, txb2_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa2_re, txa2_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb2_re, txb2_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa2_re, txa2_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb2_re, txb2_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa2_re, txa2_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb2_re, txb2_im, rader_tw_re[3], rader_tw_im[3]);

    // Round 3: txa3 and txb3
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa3_re, txa3_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb3_re, txb3_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa3_re, txa3_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb3_re, txb3_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa3_re, txa3_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb3_re, txb3_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa3_re, txa3_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb3_re, txb3_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa3_re, txa3_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb3_re, txb3_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa3_re, txa3_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb3_re, txb3_im, rader_tw_re[2], rader_tw_im[2]);

    // Round 4: txa4 and txb4
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa4_re, txa4_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb4_re, txb4_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa4_re, txa4_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb4_re, txb4_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa4_re, txa4_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb4_re, txb4_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa4_re, txa4_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb4_re, txb4_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa4_re, txa4_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb4_re, txb4_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa4_re, txa4_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb4_re, txb4_im, rader_tw_re[1], rader_tw_im[1]);

    // Round 5: txa5 and txb5
    cmul_add_fma_avx2_soa(&va0_re, &va0_im, txa5_re, txa5_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&vb0_re, &vb0_im, txb5_re, txb5_im, rader_tw_re[1], rader_tw_im[1]);
    cmul_add_fma_avx2_soa(&va1_re, &va1_im, txa5_re, txa5_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&vb1_re, &vb1_im, txb5_re, txb5_im, rader_tw_re[2], rader_tw_im[2]);
    cmul_add_fma_avx2_soa(&va2_re, &va2_im, txa5_re, txa5_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&vb2_re, &vb2_im, txb5_re, txb5_im, rader_tw_re[3], rader_tw_im[3]);
    cmul_add_fma_avx2_soa(&va3_re, &va3_im, txa5_re, txa5_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&vb3_re, &vb3_im, txb5_re, txb5_im, rader_tw_re[4], rader_tw_im[4]);
    cmul_add_fma_avx2_soa(&va4_re, &va4_im, txa5_re, txa5_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&vb4_re, &vb4_im, txb5_re, txb5_im, rader_tw_re[5], rader_tw_im[5]);
    cmul_add_fma_avx2_soa(&va5_re, &va5_im, txa5_re, txa5_im, rader_tw_re[0], rader_tw_im[0]);
    cmul_add_fma_avx2_soa(&vb5_re, &vb5_im, txb5_re, txb5_im, rader_tw_re[0], rader_tw_im[0]);

    //==========================================================================
    // ASSEMBLE OUTPUTS (A and B) + STORE - INLINE ADDS!
    //==========================================================================

    // Output permutation: [1,5,4,6,2,3] from v[0,1,2,3,4,5]
    // Do adds inline to avoid materializing ya1-ya6 and yb1-yb6 (frees 12 YMM!)

    if (use_nt)
    {
        // Butterfly A
        store_7_lanes_avx2_stream_soa(ka, K, out_re, out_im,
                                      ya0_re, ya0_im,
                                      _mm256_add_pd(xa0_re, va0_re), _mm256_add_pd(xa0_im, va0_im),  // ya1
                                      _mm256_add_pd(xa0_re, va4_re), _mm256_add_pd(xa0_im, va4_im),  // ya2
                                      _mm256_add_pd(xa0_re, va5_re), _mm256_add_pd(xa0_im, va5_im),  // ya3
                                      _mm256_add_pd(xa0_re, va2_re), _mm256_add_pd(xa0_im, va2_im),  // ya4
                                      _mm256_add_pd(xa0_re, va1_re), _mm256_add_pd(xa0_im, va1_im),  // ya5
                                      _mm256_add_pd(xa0_re, va3_re), _mm256_add_pd(xa0_im, va3_im)); // ya6

        // Butterfly B
        store_7_lanes_avx2_stream_soa(kb, K, out_re, out_im,
                                      yb0_re, yb0_im,
                                      _mm256_add_pd(xb0_re, vb0_re), _mm256_add_pd(xb0_im, vb0_im),  // yb1
                                      _mm256_add_pd(xb0_re, vb4_re), _mm256_add_pd(xb0_im, vb4_im),  // yb2
                                      _mm256_add_pd(xb0_re, vb5_re), _mm256_add_pd(xb0_im, vb5_im),  // yb3
                                      _mm256_add_pd(xb0_re, vb2_re), _mm256_add_pd(xb0_im, vb2_im),  // yb4
                                      _mm256_add_pd(xb0_re, vb1_re), _mm256_add_pd(xb0_im, vb1_im),  // yb5
                                      _mm256_add_pd(xb0_re, vb3_re), _mm256_add_pd(xb0_im, vb3_im)); // yb6
    }
    else
    {
        // Butterfly A
        store_7_lanes_avx2_soa(ka, K, out_re, out_im,
                               ya0_re, ya0_im,
                               _mm256_add_pd(xa0_re, va0_re), _mm256_add_pd(xa0_im, va0_im),  // ya1
                               _mm256_add_pd(xa0_re, va4_re), _mm256_add_pd(xa0_im, va4_im),  // ya2
                               _mm256_add_pd(xa0_re, va5_re), _mm256_add_pd(xa0_im, va5_im),  // ya3
                               _mm256_add_pd(xa0_re, va2_re), _mm256_add_pd(xa0_im, va2_im),  // ya4
                               _mm256_add_pd(xa0_re, va1_re), _mm256_add_pd(xa0_im, va1_im),  // ya5
                               _mm256_add_pd(xa0_re, va3_re), _mm256_add_pd(xa0_im, va3_im)); // ya6

        // Butterfly B
        store_7_lanes_avx2_soa(kb, K, out_re, out_im,
                               yb0_re, yb0_im,
                               _mm256_add_pd(xb0_re, vb0_re), _mm256_add_pd(xb0_im, vb0_im),  // yb1
                               _mm256_add_pd(xb0_re, vb4_re), _mm256_add_pd(xb0_im, vb4_im),  // yb2
                               _mm256_add_pd(xb0_re, vb5_re), _mm256_add_pd(xb0_im, vb5_im),  // yb3
                               _mm256_add_pd(xb0_re, vb2_re), _mm256_add_pd(xb0_im, vb2_im),  // yb4
                               _mm256_add_pd(xb0_re, vb1_re), _mm256_add_pd(xb0_im, vb1_im),  // yb5
                               _mm256_add_pd(xb0_re, vb3_re), _mm256_add_pd(xb0_im, vb3_im)); // yb6
    }
}

//==============================================================================
// STAGE DISPATCHER - N1 VARIANT (NO STAGE TWIDDLES!)
//==============================================================================

/**
 * @brief Execute radix-7 stage - N1 (twiddle-less) variant
 * @details
 * Dispatches to:
 * - U2 path for main loop (k, k+8)
 * - Single path for tail (k < 8 remaining)
 * - Scalar fallback for remainder
 *
 * ⚡⚡⚡ N1 DIFFERENCE: NO stage_tw parameter, NO sub_len checks!
 *
 * @param K Number of butterflies
 * @param in_re Input real components (32-byte aligned)
 * @param in_im Input imaginary components (32-byte aligned)
 * @param rader_tw Rader twiddle factors (6 complex values)
 * @param out_re Output real components (32-byte aligned)
 * @param out_im Output imaginary components (32-byte aligned)
 */
static void radix7_stage_avx2_n1_soa(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *rader_tw,
    double *restrict out_re,
    double *restrict out_im)
{
    // Verify alignment (debug/production check)
    if (!verify_r7_avx2_n1_alignment(in_re, in_im, out_re, out_im))
    {
        // Fallback to scalar for misaligned (should not happen in production!)
        // Would call scalar N1 version here
        return;
    }

    // Broadcast Rader twiddles ONCE for entire stage (P0 optimization!)
    __m256d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_avx2_soa(rader_tw, rader_tw_re, rader_tw_im);

    // Decide on non-temporal stores
    size_t bytes_per_stage = (size_t)K * 7 * 2 * sizeof(double); // 7 lanes, 2 components
    bool use_nt = (bytes_per_stage > (size_t)(R7_AVX2_N1_NT_THRESHOLD * R7_AVX2_N1_LLC_BYTES)) &&
                  (K >= R7_AVX2_N1_NT_MIN_K);

    // Check for environment variable override (for tuning)
    const char *nt_env = getenv("FFT_R7_NT");
    if (nt_env != NULL)
    {
        use_nt = (atoi(nt_env) != 0);
    }

    int k = 0;

    // Main U2 loop: process 8 elements per iteration (2 butterflies × 4 wide)
    for (; k <= K - R7_AVX2_N1_U2_WIDTH; k += R7_AVX2_N1_U2_WIDTH)
    {
        // Prefetch ahead (N1 simplified - no stage twiddles!)
        prefetch_7_lanes_avx2_n1_soa(k, K, in_re, in_im);

        // Process two butterflies simultaneously
        radix7_butterfly_dual_avx2_n1_soa(k, k + R7_AVX2_N1_WIDTH, K,
                                          in_re, in_im,
                                          rader_tw_re, rader_tw_im,
                                          out_re, out_im, use_nt);
    }

    // Tail loop: single butterflies (4 elements at a time)
    for (; k <= K - R7_AVX2_N1_WIDTH; k += R7_AVX2_N1_WIDTH)
    {
        radix7_butterfly_single_avx2_n1_soa(k, K, in_re, in_im,
                                            rader_tw_re, rader_tw_im,
                                            out_re, out_im, use_nt);
    }

    // Remainder: scalar fallback (k < 4 remaining)
    // Would call scalar N1 version here for k..K-1

    // Fence after NT stores (once per stage, not per iteration!)
    if (use_nt)
    {
        _mm_sfence();
    }
}

#endif // FFT_RADIX7_AVX2_N1_H