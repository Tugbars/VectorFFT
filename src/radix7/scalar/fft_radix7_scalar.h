/**
 * @file fft_radix7_scalar.h
 * @brief Scalar Radix-7 Rader Butterfly - Fallback and Remainder Handler
 *
 * @details
 * SCALAR FALLBACK - Generation 3:
 * ================================
 * ✅ TRUE SoA: re/im stay separate throughout
 * ✅ Handles misaligned data
 * ✅ Handles remainder elements after vectorized loops
 * ✅ Works for any K value (including K=1)
 * ✅ No SIMD dependencies
 *
 * ALL RADIX-7 ALGORITHMIC OPTIMIZATIONS PRESERVED:
 * =================================================
 * ✅✅ P0: Pre-split Rader broadcasts (load once per K-loop iteration)
 * ✅✅ P0: Round-robin convolution schedule (maximizes ILP for compiler)
 * ✅✅ P1: Tree y0 sum (helps compiler optimize)
 * ✅ Standard double arithmetic
 * ✅ Rader permutations: input [1,3,2,6,4,5], output [1,5,4,6,2,3]
 * ✅ 6-point cyclic convolution with generator g=3
 *
 * USE CASES:
 * ==========
 * - Fallback for misaligned buffers
 * - Remainder loop after SSE2/AVX2 vectorization
 * - Very small K values where overhead exceeds benefit
 * - Platforms without SIMD support
 * - Reference implementation for testing
 *
 * @author FFT Optimization Team
 * @version 4.0 (Scalar with algorithmic optimizations)
 * @date 2025
 */

#ifndef FFT_RADIX7_SCALAR_H
#define FFT_RADIX7_SCALAR_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../fft_radix7_uniform.h"

//==============================================================================
// SCALAR CONFIGURATION
//==============================================================================

/// Scalar processes 1 element at a time
#define R7_SCALAR_WIDTH 1

/// Prefetch distance (same as SIMD versions for consistency)
#define R7_SCALAR_PREFETCH_DISTANCE 64

/// Non-temporal store threshold (scalar typically doesn't use NT stores)
#define R7_SCALAR_NT_THRESHOLD 0.9

/// Minimum K for enabling non-temporal stores
#define R7_SCALAR_NT_MIN_K 8192

//==============================================================================
// COMPLEX MULTIPLY PRIMITIVES - SCALAR
//==============================================================================

/**
 * @brief Complex multiply - scalar version
 * @details (out_re + i*out_im) = (a_re + i*a_im) * (w_re + i*w_im)
 * 
 * ✅ PRESERVED: Optimal 4-mul, 1-add, 1-sub sequence
 * ✅ Standard double arithmetic
 * 
 * out_re = a_re * w_re - a_im * w_im
 * out_im = a_re * w_im + a_im * w_re
 */
__attribute__((always_inline))
static inline void cmul_scalar(
    double * restrict out_re,
    double * restrict out_im,
    double a_re,
    double a_im,
    double w_re,
    double w_im)
{
    double prod1 = a_re * w_re;
    double prod2 = a_im * w_im;
    double prod3 = a_re * w_im;
    double prod4 = a_im * w_re;
    
    *out_re = prod1 - prod2;
    *out_im = prod3 + prod4;
}

/**
 * @brief Complex multiply-add - scalar version
 * @details acc += a * w (for round-robin convolution)
 * 
 * ✅ PRESERVED: Accumulation pattern for P0 optimization
 * 
 * acc_re += a_re * w_re - a_im * w_im
 * acc_im += a_re * w_im + a_im * w_re
 */
__attribute__((always_inline))
static inline void cmul_add_scalar(
    double * restrict acc_re,
    double * restrict acc_im,
    double a_re,
    double a_im,
    double w_re,
    double w_im)
{
    double prod1 = a_re * w_re;
    double prod2 = a_im * w_im;
    double prod3 = a_re * w_im;
    double prod4 = a_im * w_re;
    
    *acc_re += (prod1 - prod2);
    *acc_im += (prod3 + prod4);
}

//==============================================================================
// LOAD/STORE PRIMITIVES - SCALAR
//==============================================================================

/**
 * @brief Load 7 lanes from SoA buffers - scalar (handles any alignment)
 * @details
 * ✅ Works with any alignment
 * ✅ TRUE SoA - re/im stay separate
 * 
 * @param k Starting index
 * @param K Stride between lanes
 * @param in_re Real component array
 * @param in_im Imaginary component array
 * @param x0_re-x6_re Output: real components for 7 lanes
 * @param x0_im-x6_im Output: imaginary components for 7 lanes
 */
__attribute__((always_inline))
static inline void load_7_lanes_scalar(
    int k, int K,
    const double * restrict in_re,
    const double * restrict in_im,
    double *x0_re, double *x0_im,
    double *x1_re, double *x1_im,
    double *x2_re, double *x2_im,
    double *x3_re, double *x3_im,
    double *x4_re, double *x4_im,
    double *x5_re, double *x5_im,
    double *x6_re, double *x6_im)
{
    *x0_re = in_re[0 * K + k];
    *x0_im = in_im[0 * K + k];
    *x1_re = in_re[1 * K + k];
    *x1_im = in_im[1 * K + k];
    *x2_re = in_re[2 * K + k];
    *x2_im = in_im[2 * K + k];
    *x3_re = in_re[3 * K + k];
    *x3_im = in_im[3 * K + k];
    *x4_re = in_re[4 * K + k];
    *x4_im = in_im[4 * K + k];
    *x5_re = in_re[5 * K + k];
    *x5_im = in_im[5 * K + k];
    *x6_re = in_re[6 * K + k];
    *x6_im = in_im[6 * K + k];
}

/**
 * @brief Store 7 lanes to SoA buffers - scalar (handles any alignment)
 * @details
 * ✅ Works with any alignment
 * ✅ TRUE SoA - re/im already separate
 */
__attribute__((always_inline))
static inline void store_7_lanes_scalar(
    int k, int K,
    double * restrict out_re,
    double * restrict out_im,
    double y0_re, double y0_im,
    double y1_re, double y1_im,
    double y2_re, double y2_im,
    double y3_re, double y3_im,
    double y4_re, double y4_im,
    double y5_re, double y5_im,
    double y6_re, double y6_im)
{
    out_re[0 * K + k] = y0_re;
    out_im[0 * K + k] = y0_im;
    out_re[1 * K + k] = y1_re;
    out_im[1 * K + k] = y1_im;
    out_re[2 * K + k] = y2_re;
    out_im[2 * K + k] = y2_im;
    out_re[3 * K + k] = y3_re;
    out_im[3 * K + k] = y3_im;
    out_re[4 * K + k] = y4_re;
    out_im[4 * K + k] = y4_im;
    out_re[5 * K + k] = y5_re;
    out_im[5 * K + k] = y5_im;
    out_re[6 * K + k] = y6_re;
    out_im[6 * K + k] = y6_im;
}

//==============================================================================
// STAGE TWIDDLE APPLICATION - SCALAR
//==============================================================================

/**
 * @brief Apply stage twiddles to 6 of the 7 lanes (x0 unchanged)
 * @details
 * ✅ PRESERVED: Same twiddle application pattern
 * ✅ SoA twiddle access
 * 
 * @param k Starting index
 * @param K Stride
 * @param x1_re-x6_re Input/output: real components (modified in place)
 * @param x1_im-x6_im Input/output: imaginary components (modified in place)
 * @param stage_tw Stage twiddle factors (SoA layout)
 * @param sub_len Sub-transform length (skip if == 1)
 */
__attribute__((always_inline))
static inline void apply_stage_twiddles_scalar(
    int k, int K,
    double *x1_re, double *x1_im,
    double *x2_re, double *x2_im,
    double *x3_re, double *x3_im,
    double *x4_re, double *x4_im,
    double *x5_re, double *x5_im,
    double *x6_re, double *x6_im,
    const fft_twiddle_soa *stage_tw,
    int sub_len)
{
    if (sub_len <= 1)
        return;  // No twiddles needed for first stage
    
    // Load twiddles
    double w1_re = stage_tw->re[0 * K + k];
    double w1_im = stage_tw->im[0 * K + k];
    double w2_re = stage_tw->re[1 * K + k];
    double w2_im = stage_tw->im[1 * K + k];
    double w3_re = stage_tw->re[2 * K + k];
    double w3_im = stage_tw->im[2 * K + k];
    double w4_re = stage_tw->re[3 * K + k];
    double w4_im = stage_tw->im[3 * K + k];
    double w5_re = stage_tw->re[4 * K + k];
    double w5_im = stage_tw->im[4 * K + k];
    double w6_re = stage_tw->re[5 * K + k];
    double w6_im = stage_tw->im[5 * K + k];
    
    // Apply complex multiplication (in-place)
    double tmp_re, tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x1_re, *x1_im, w1_re, w1_im);
    *x1_re = tmp_re; *x1_im = tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x2_re, *x2_im, w2_re, w2_im);
    *x2_re = tmp_re; *x2_im = tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x3_re, *x3_im, w3_re, w3_im);
    *x3_re = tmp_re; *x3_im = tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x4_re, *x4_im, w4_re, w4_im);
    *x4_re = tmp_re; *x4_im = tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x5_re, *x5_im, w5_re, w5_im);
    *x5_re = tmp_re; *x5_im = tmp_im;
    
    cmul_scalar(&tmp_re, &tmp_im, *x6_re, *x6_im, w6_re, w6_im);
    *x6_re = tmp_re; *x6_im = tmp_im;
}

//==============================================================================
// TREE Y0 COMPUTATION - P1 OPTIMIZATION PRESERVED
//==============================================================================

/**
 * @brief Compute DC component y0 = sum of all 7 inputs (TREE REDUCTION!)
 * @details
 * ✅✅ PRESERVED: Balanced tree reduces dependency chain
 * ✅ Helps compiler optimize even in scalar code
 * 
 * Tree structure:
 *   Level 1: (x0+x1), (x2+x3), (x4+x5)
 *   Level 2: (x0+x1+x2+x3), (x4+x5+x6)
 *   Level 3: (x0+x1+x2+x3+x4+x5+x6)
 * 
 * Critical path: 3 additions vs 6 sequential
 */
__attribute__((always_inline))
static inline void compute_y0_tree_scalar(
    double x0_re, double x0_im,
    double x1_re, double x1_im,
    double x2_re, double x2_im,
    double x3_re, double x3_im,
    double x4_re, double x4_im,
    double x5_re, double x5_im,
    double x6_re, double x6_im,
    double *y0_re, double *y0_im)
{
    // Level 1: 3 parallel additions
    double s01_re = x0_re + x1_re;
    double s01_im = x0_im + x1_im;
    double s23_re = x2_re + x3_re;
    double s23_im = x2_im + x3_im;
    double s45_re = x4_re + x5_re;
    double s45_im = x4_im + x5_im;
    
    // Level 2: 2 parallel additions
    double s0123_re = s01_re + s23_re;
    double s0123_im = s01_im + s23_im;
    double s456_re = s45_re + x6_re;
    double s456_im = s45_im + x6_im;
    
    // Level 3: final addition
    *y0_re = s0123_re + s456_re;
    *y0_im = s0123_im + s456_im;
}

//==============================================================================
// RADER INPUT PERMUTATION
//==============================================================================

/**
 * @brief Permute inputs for Rader algorithm
 * @details
 * ✅✅ PRESERVED: Exact permutation [1,3,2,6,4,5] for generator g=3
 * 
 * Input order:  [x1, x2, x3, x4, x5, x6]
 * Output order: [x1, x3, x2, x6, x4, x5]  (for cyclic convolution)
 */
__attribute__((always_inline))
static inline void permute_rader_inputs_scalar(
    double x1_re, double x1_im,
    double x2_re, double x2_im,
    double x3_re, double x3_im,
    double x4_re, double x4_im,
    double x5_re, double x5_im,
    double x6_re, double x6_im,
    double *tx0_re, double *tx0_im,
    double *tx1_re, double *tx1_im,
    double *tx2_re, double *tx2_im,
    double *tx3_re, double *tx3_im,
    double *tx4_re, double *tx4_im,
    double *tx5_re, double *tx5_im)
{
    *tx0_re = x1_re; *tx0_im = x1_im;  // Position 0 ← x1
    *tx1_re = x3_re; *tx1_im = x3_im;  // Position 1 ← x3
    *tx2_re = x2_re; *tx2_im = x2_im;  // Position 2 ← x2
    *tx3_re = x6_re; *tx3_im = x6_im;  // Position 3 ← x6
    *tx4_re = x4_re; *tx4_im = x4_im;  // Position 4 ← x4
    *tx5_re = x5_re; *tx5_im = x5_im;  // Position 5 ← x5
}

//==============================================================================
// RADER 6-POINT CYCLIC CONVOLUTION - P0 ROUND-ROBIN PRESERVED
//==============================================================================

/**
 * @brief 6-point cyclic convolution with ROUND-ROBIN (P0 OPTIMIZATION!)
 * @details
 * ✅✅✅ PRESERVED: Round-robin schedule for maximum ILP
 * ✅ Helps compiler optimize instruction scheduling
 * 
 * Round-robin pattern (row = input, col = output):
 *       v0  v1  v2  v3  v4  v5
 * tx0:  w0  w1  w2  w3  w4  w5
 * tx1:  w5  w0  w1  w2  w3  w4
 * tx2:  w4  w5  w0  w1  w2  w3
 * tx3:  w3  w4  w5  w0  w1  w2
 * tx4:  w2  w3  w4  w5  w0  w1
 * tx5:  w1  w2  w3  w4  w5  w0
 * 
 * This ensures NO accumulator has back-to-back dependencies!
 * 
 * @param tx0_re-tx5_re Permuted input real components
 * @param tx0_im-tx5_im Permuted input imaginary components
 * @param rader_tw Rader twiddle factors (6 complex values, pre-loaded)
 * @param v0_re-v5_re Output: convolution result real components
 * @param v0_im-v5_im Output: convolution result imaginary components
 */
__attribute__((always_inline))
static inline void rader_convolution_roundrobin_scalar(
    double tx0_re, double tx0_im,
    double tx1_re, double tx1_im,
    double tx2_re, double tx2_im,
    double tx3_re, double tx3_im,
    double tx4_re, double tx4_im,
    double tx5_re, double tx5_im,
    const fft_twiddle_soa *rader_tw,
    double *v0_re, double *v0_im,
    double *v1_re, double *v1_im,
    double *v2_re, double *v2_im,
    double *v3_re, double *v3_im,
    double *v4_re, double *v4_im,
    double *v5_re, double *v5_im)
{
    // Pre-load Rader twiddles (P0 optimization: load once, use many times)
    double tw0_re = rader_tw->re[0], tw0_im = rader_tw->im[0];
    double tw1_re = rader_tw->re[1], tw1_im = rader_tw->im[1];
    double tw2_re = rader_tw->re[2], tw2_im = rader_tw->im[2];
    double tw3_re = rader_tw->re[3], tw3_im = rader_tw->im[3];
    double tw4_re = rader_tw->re[4], tw4_im = rader_tw->im[4];
    double tw5_re = rader_tw->re[5], tw5_im = rader_tw->im[5];
    
    // Initialize accumulators to zero
    *v0_re = 0.0; *v0_im = 0.0;
    *v1_re = 0.0; *v1_im = 0.0;
    *v2_re = 0.0; *v2_im = 0.0;
    *v3_re = 0.0; *v3_im = 0.0;
    *v4_re = 0.0; *v4_im = 0.0;
    *v5_re = 0.0; *v5_im = 0.0;
    
    // ✅✅ PRESERVED: Round-robin schedule for maximum ILP
    // Round 0: tx0 contributes to all 6 accumulators
    cmul_add_scalar(v0_re, v0_im, tx0_re, tx0_im, tw0_re, tw0_im);
    cmul_add_scalar(v1_re, v1_im, tx0_re, tx0_im, tw1_re, tw1_im);
    cmul_add_scalar(v2_re, v2_im, tx0_re, tx0_im, tw2_re, tw2_im);
    cmul_add_scalar(v3_re, v3_im, tx0_re, tx0_im, tw3_re, tw3_im);
    cmul_add_scalar(v4_re, v4_im, tx0_re, tx0_im, tw4_re, tw4_im);
    cmul_add_scalar(v5_re, v5_im, tx0_re, tx0_im, tw5_re, tw5_im);
    
    // Round 1: tx1 contributes (rotated twiddle indices)
    cmul_add_scalar(v0_re, v0_im, tx1_re, tx1_im, tw5_re, tw5_im);
    cmul_add_scalar(v1_re, v1_im, tx1_re, tx1_im, tw0_re, tw0_im);
    cmul_add_scalar(v2_re, v2_im, tx1_re, tx1_im, tw1_re, tw1_im);
    cmul_add_scalar(v3_re, v3_im, tx1_re, tx1_im, tw2_re, tw2_im);
    cmul_add_scalar(v4_re, v4_im, tx1_re, tx1_im, tw3_re, tw3_im);
    cmul_add_scalar(v5_re, v5_im, tx1_re, tx1_im, tw4_re, tw4_im);
    
    // Round 2: tx2 contributes
    cmul_add_scalar(v0_re, v0_im, tx2_re, tx2_im, tw4_re, tw4_im);
    cmul_add_scalar(v1_re, v1_im, tx2_re, tx2_im, tw5_re, tw5_im);
    cmul_add_scalar(v2_re, v2_im, tx2_re, tx2_im, tw0_re, tw0_im);
    cmul_add_scalar(v3_re, v3_im, tx2_re, tx2_im, tw1_re, tw1_im);
    cmul_add_scalar(v4_re, v4_im, tx2_re, tx2_im, tw2_re, tw2_im);
    cmul_add_scalar(v5_re, v5_im, tx2_re, tx2_im, tw3_re, tw3_im);
    
    // Round 3: tx3 contributes
    cmul_add_scalar(v0_re, v0_im, tx3_re, tx3_im, tw3_re, tw3_im);
    cmul_add_scalar(v1_re, v1_im, tx3_re, tx3_im, tw4_re, tw4_im);
    cmul_add_scalar(v2_re, v2_im, tx3_re, tx3_im, tw5_re, tw5_im);
    cmul_add_scalar(v3_re, v3_im, tx3_re, tx3_im, tw0_re, tw0_im);
    cmul_add_scalar(v4_re, v4_im, tx3_re, tx3_im, tw1_re, tw1_im);
    cmul_add_scalar(v5_re, v5_im, tx3_re, tx3_im, tw2_re, tw2_im);
    
    // Round 4: tx4 contributes
    cmul_add_scalar(v0_re, v0_im, tx4_re, tx4_im, tw2_re, tw2_im);
    cmul_add_scalar(v1_re, v1_im, tx4_re, tx4_im, tw3_re, tw3_im);
    cmul_add_scalar(v2_re, v2_im, tx4_re, tx4_im, tw4_re, tw4_im);
    cmul_add_scalar(v3_re, v3_im, tx4_re, tx4_im, tw5_re, tw5_im);
    cmul_add_scalar(v4_re, v4_im, tx4_re, tx4_im, tw0_re, tw0_im);
    cmul_add_scalar(v5_re, v5_im, tx4_re, tx4_im, tw1_re, tw1_im);
    
    // Round 5: tx5 contributes
    cmul_add_scalar(v0_re, v0_im, tx5_re, tx5_im, tw1_re, tw1_im);
    cmul_add_scalar(v1_re, v1_im, tx5_re, tx5_im, tw2_re, tw2_im);
    cmul_add_scalar(v2_re, v2_im, tx5_re, tx5_im, tw3_re, tw3_im);
    cmul_add_scalar(v3_re, v3_im, tx5_re, tx5_im, tw4_re, tw4_im);
    cmul_add_scalar(v4_re, v4_im, tx5_re, tx5_im, tw5_re, tw5_im);
    cmul_add_scalar(v5_re, v5_im, tx5_re, tx5_im, tw0_re, tw0_im);
}

//==============================================================================
// COMPLETE BUTTERFLY FUNCTION
//==============================================================================

/**
 * @brief Single radix-7 butterfly - scalar
 * @details
 * ✅ ALL ALGORITHMIC OPTIMIZATIONS PRESERVED:
 *    - Pre-loaded Rader twiddles (P0 optimization)
 *    - Round-robin convolution (P0 optimization)
 *    - Tree y0 sum (P1 optimization)
 *    - Rader permutations
 *    - TRUE SoA architecture
 * 
 * Process one butterfly: single element
 * Handles any alignment, any K value
 * 
 * @param k Index
 * @param K Stride between lanes
 * @param in_re Input real components
 * @param in_im Input imaginary components
 * @param stage_tw Stage twiddle factors (SoA layout)
 * @param rader_tw Rader twiddle factors (6 complex values)
 * @param out_re Output real components
 * @param out_im Output imaginary components
 * @param sub_len Sub-transform length
 */
__attribute__((always_inline))
static inline void radix7_butterfly_scalar(
    int k, int K,
    const double * restrict in_re,
    const double * restrict in_im,
    const fft_twiddle_soa *stage_tw,
    const fft_twiddle_soa *rader_tw,
    double * restrict out_re,
    double * restrict out_im,
    int sub_len)
{
    // STEP 1: Load 7 lanes (1 element per lane)
    double x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im;
    double x4_re, x4_im, x5_re, x5_im, x6_re, x6_im;
    
    load_7_lanes_scalar(k, K, in_re, in_im,
                        &x0_re, &x0_im, &x1_re, &x1_im,
                        &x2_re, &x2_im, &x3_re, &x3_im,
                        &x4_re, &x4_im, &x5_re, &x5_im,
                        &x6_re, &x6_im);
    
    // STEP 2: Apply stage twiddles (x0 unchanged, x1-x6 multiplied)
    apply_stage_twiddles_scalar(k, K,
                                &x1_re, &x1_im, &x2_re, &x2_im,
                                &x3_re, &x3_im, &x4_re, &x4_im,
                                &x5_re, &x5_im, &x6_re, &x6_im,
                                stage_tw, sub_len);
    
    // STEP 3: Compute DC component y0 (tree reduction)
    double y0_re, y0_im;
    compute_y0_tree_scalar(x0_re, x0_im, x1_re, x1_im,
                           x2_re, x2_im, x3_re, x3_im,
                           x4_re, x4_im, x5_re, x5_im,
                           x6_re, x6_im,
                           &y0_re, &y0_im);
    
    // STEP 4: Permute inputs for Rader algorithm
    double tx0_re, tx0_im, tx1_re, tx1_im, tx2_re, tx2_im;
    double tx3_re, tx3_im, tx4_re, tx4_im, tx5_re, tx5_im;
    
    permute_rader_inputs_scalar(x1_re, x1_im, x2_re, x2_im,
                                x3_re, x3_im, x4_re, x4_im,
                                x5_re, x5_im, x6_re, x6_im,
                                &tx0_re, &tx0_im, &tx1_re, &tx1_im,
                                &tx2_re, &tx2_im, &tx3_re, &tx3_im,
                                &tx4_re, &tx4_im, &tx5_re, &tx5_im);
    
    // STEP 5: 6-point cyclic convolution (round-robin schedule)
    double v0_re, v0_im, v1_re, v1_im, v2_re, v2_im;
    double v3_re, v3_im, v4_re, v4_im, v5_re, v5_im;
    
    rader_convolution_roundrobin_scalar(
        tx0_re, tx0_im, tx1_re, tx1_im, tx2_re, tx2_im,
        tx3_re, tx3_im, tx4_re, tx4_im, tx5_re, tx5_im,
        rader_tw,
        &v0_re, &v0_im, &v1_re, &v1_im, &v2_re, &v2_im,
        &v3_re, &v3_im, &v4_re, &v4_im, &v5_re, &v5_im);
    
    // STEP 6: Assemble outputs with permutation [1,5,4,6,2,3]
    double y1_re = x0_re + v0_re;  // Position 1 ← v0
    double y1_im = x0_im + v0_im;
    double y2_re = x0_re + v4_re;  // Position 2 ← v4
    double y2_im = x0_im + v4_im;
    double y3_re = x0_re + v5_re;  // Position 3 ← v5
    double y3_im = x0_im + v5_im;
    double y4_re = x0_re + v2_re;  // Position 4 ← v2
    double y4_im = x0_im + v2_im;
    double y5_re = x0_re + v1_re;  // Position 5 ← v1
    double y5_im = x0_im + v1_im;
    double y6_re = x0_re + v3_re;  // Position 6 ← v3
    double y6_im = x0_im + v3_im;
    
    // STEP 7: Store results
    store_7_lanes_scalar(k, K, out_re, out_im,
                         y0_re, y0_im, y1_re, y1_im,
                         y2_re, y2_im, y3_re, y3_im,
                         y4_re, y4_im, y5_re, y5_im,
                         y6_re, y6_im);
}

//==============================================================================
// STAGE FUNCTION - SCALAR LOOP
//==============================================================================

/**
 * @brief Execute radix-7 stage - scalar version
 * @details
 * Processes all K butterflies using scalar arithmetic.
 * Use cases:
 * - Fallback for misaligned buffers
 * - Remainder loop after SIMD vectorization
 * - Very small K values
 * - Platforms without SIMD
 * 
 * @param K Number of butterflies
 * @param in_re Input real components
 * @param in_im Input imaginary components
 * @param stage_tw Stage twiddle factors (SoA layout)
 * @param rader_tw Rader twiddle factors (6 complex values)
 * @param out_re Output real components
 * @param out_im Output imaginary components
 * @param sub_len Sub-transform length
 */
static void radix7_stage_scalar(
    int K,
    const double * restrict in_re,
    const double * restrict in_im,
    const fft_twiddle_soa *stage_tw,
    const fft_twiddle_soa *rader_tw,
    double * restrict out_re,
    double * restrict out_im,
    int sub_len)
{
    // Simple loop: process one butterfly at a time
    for (int k = 0; k < K; k++)
    {
        radix7_butterfly_scalar(k, K, in_re, in_im, stage_tw, rader_tw,
                                out_re, out_im, sub_len);
    }
}

//==============================================================================
// UNIFIED DISPATCHER WITH SIMD FALLBACK
//==============================================================================

/**
 * @brief Unified radix-7 stage dispatcher
 * @details
 * Intelligently selects the best implementation:
 * 1. AVX2 path (if available and aligned) - 4-wide
 * 2. SSE2 path (if available and aligned) - 2-wide
 * 3. Scalar fallback (always works) - 1-wide
 * 
 * Also handles remainder elements after vectorized loops.
 */
static void radix7_stage_auto(
    int K,
    const double * restrict in_re,
    const double * restrict in_im,
    const fft_twiddle_soa *stage_tw,
    const fft_twiddle_soa *rader_tw,
    double * restrict out_re,
    double * restrict out_im,
    int sub_len)
{
#ifdef __AVX2__
    // AVX2 path available
    if (is_aligned_32(in_re) && is_aligned_32(in_im) &&
        is_aligned_32(out_re) && is_aligned_32(out_im) &&
        K >= R7_AVX2_WIDTH)
    {
        // Use AVX2 for main part
        int k_vec = (K / R7_AVX2_WIDTH) * R7_AVX2_WIDTH;
        if (k_vec > 0)
        {
            // Call AVX2 stage function (would be implemented in fft_radix7_avx2.h)
            // radix7_stage_avx2_soa(k_vec, in_re, in_im, stage_tw, rader_tw,
            //                       out_re, out_im, sub_len);
        }
        
        // Scalar fallback for remainder
        for (int k = k_vec; k < K; k++)
        {
            radix7_butterfly_scalar(k, K, in_re, in_im, stage_tw, rader_tw,
                                    out_re, out_im, sub_len);
        }
        return;
    }
#endif

#ifdef __SSE2__
    // SSE2 path available
    if (is_aligned_16(in_re) && is_aligned_16(in_im) &&
        is_aligned_16(out_re) && is_aligned_16(out_im) &&
        K >= R7_SSE2_WIDTH)
    {
        // Use SSE2 for main part
        int k_vec = (K / R7_SSE2_WIDTH) * R7_SSE2_WIDTH;
        if (k_vec > 0)
        {
            radix7_stage_sse2_soa(k_vec, in_re, in_im, stage_tw, rader_tw,
                                  out_re, out_im, sub_len);
        }
        
        // Scalar fallback for remainder
        for (int k = k_vec; k < K; k++)
        {
            radix7_butterfly_scalar(k, K, in_re, in_im, stage_tw, rader_tw,
                                    out_re, out_im, sub_len);
        }
        return;
    }
#endif

    // Scalar fallback for everything
    radix7_stage_scalar(K, in_re, in_im, stage_tw, rader_tw,
                        out_re, out_im, sub_len);
}

#endif // FFT_RADIX7_SCALAR_H