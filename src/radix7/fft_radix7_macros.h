//==============================================================================
// fft_radix7_macros.h - Shared Macros for Radix-7 Rader Butterflies (OPTIMIZED)
//==============================================================================
//
// DESIGN:
// - Rader's algorithm for prime-length DFT (N=7)
// - Generator g=3: perm_in=[1,3,2,6,4,5], out_perm=[1,5,4,6,2,3]
// - Direction in function names (_fv vs _bv)
// - Shared implementation via macros
//
// OPTIMIZATION FEATURES:
// - FMA (Fused Multiply-Add) for AVX2 and AVX-512
// - Fused complex multiply-add operations in convolution
// - Hoisted Rader twiddle broadcasts outside hot loops
// - Streaming stores for large transforms (K >= 8192)
// - Single-level prefetching with tuned distances
// - Separate code paths for streaming vs normal stores (no branches in loops)
//

#ifndef FFT_RADIX7_MACROS_H
#define FFT_RADIX7_MACROS_H

#include "fft_radix7.h"
#include "simd_math.h"

//==============================================================================
// STREAMING THRESHOLD
//==============================================================================

/**
 * @brief Threshold for switching to streaming stores
 * 
 * For K >= STREAM_THRESHOLD, use non-temporal stores to avoid cache pollution.
 * Typical value: 8192 (produces ~230KB of output data with radix-7).
 */
#define STREAM_THRESHOLD 8192

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

/**
 * @brief Complex multiply for AVX-512: out = a * w (4 complex values)
 * 
 * Computes (ar + i*ai) * (wr + i*wi) = (ar*wr - ai*wi) + i*(ar*wi + ai*wr)
 * 
 * Uses FMA (fused multiply-add) for optimal performance:
 * - re = ar*wr - ai*wi  (via fmsub)
 * - im = ar*wi + ai*wr  (via fmadd)
 * 
 * @param out Result vector (4 complex doubles)
 * @param a   First operand (4 complex doubles)
 * @param w   Second operand (4 complex doubles, typically twiddle)
 * 
 * @note Uses shuffle_pd instead of unpack for correct lane handling
 */
#define CMUL_FMA_R7_AVX512(out, a, w)                                     \
    do                                                                    \
    {                                                                     \
        __m512d ar = _mm512_shuffle_pd(a, a, 0x00);                       \
        __m512d ai = _mm512_shuffle_pd(a, a, 0xFF);                       \
        __m512d wr = _mm512_shuffle_pd(w, w, 0x00);                       \
        __m512d wi = _mm512_shuffle_pd(w, w, 0xFF);                       \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));     \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));     \
        (out) = _mm512_unpacklo_pd(re, im);                               \
    } while (0)

//==============================================================================
// FUSED COMPLEX MULTIPLY-ADD - AVX-512
//==============================================================================

/**
 * @brief Fused complex multiply-add: acc += a * w (4 complex values)
 * 
 * Performs complex multiplication and accumulates into acc.
 * More efficient than separate CMUL + ADD due to reduced register pressure.
 * 
 * Critical for Rader convolution performance (36 multiply-adds per butterfly).
 * 
 * @param acc Accumulator (modified in-place)
 * @param a   First operand
 * @param w   Second operand (twiddle)
 */
#define CMUL_ADD_FMA_R7_AVX512(acc, a, w)                                 \
    do                                                                    \
    {                                                                     \
        __m512d ar = _mm512_shuffle_pd(a, a, 0x00);                       \
        __m512d ai = _mm512_shuffle_pd(a, a, 0xFF);                       \
        __m512d wr = _mm512_shuffle_pd(w, w, 0x00);                       \
        __m512d wi = _mm512_shuffle_pd(w, w, 0xFF);                       \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));     \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));     \
        (acc) = _mm512_add_pd(acc, _mm512_unpacklo_pd(re, im));          \
    } while (0)

//==============================================================================
// RADER Y0 COMPUTATION - AVX-512
//==============================================================================

/**
 * @brief Compute DC component y0 = sum of all 7 inputs
 * 
 * First step of Rader's algorithm: compute the DC (zero-frequency) component.
 * This is simply the sum of all input values.
 * 
 * @param x0-x6 Input values (7 lanes)
 * @param y0    Output DC component
 */
#define COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0)              \
    do {                                                                   \
        y0 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(x0, x1),           \
                                         _mm512_add_pd(x2, x3)),           \
                          _mm512_add_pd(_mm512_add_pd(x4, x5), x6));      \
    } while (0)

//==============================================================================
// RADER TWIDDLE BROADCAST - AVX-512
//==============================================================================

/**
 * @brief Broadcast 6 Rader twiddle factors to AVX-512 registers
 * 
 * Each twiddle is replicated 4 times (for 4 parallel butterflies).
 * Layout: [re0, im0, re1, im1, re2, im2, re3, im3] where all copies are identical.
 * 
 * This should be called ONCE before the butterfly loop, not inside it.
 * 
 * @param rader_tw Input twiddle array (6 complex doubles)
 * @param tw_brd   Output broadcast array (6 AVX-512 vectors)
 */
#define BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd)              \
    do {                                                                   \
        tw_brd[0] = _mm512_set_pd(                                         \
            rader_tw[0].im, rader_tw[0].re,                                \
            rader_tw[0].im, rader_tw[0].re,                                \
            rader_tw[0].im, rader_tw[0].re,                                \
            rader_tw[0].im, rader_tw[0].re);                               \
        tw_brd[1] = _mm512_set_pd(                                         \
            rader_tw[1].im, rader_tw[1].re,                                \
            rader_tw[1].im, rader_tw[1].re,                                \
            rader_tw[1].im, rader_tw[1].re,                                \
            rader_tw[1].im, rader_tw[1].re);                               \
        tw_brd[2] = _mm512_set_pd(                                         \
            rader_tw[2].im, rader_tw[2].re,                                \
            rader_tw[2].im, rader_tw[2].re,                                \
            rader_tw[2].im, rader_tw[2].re,                                \
            rader_tw[2].im, rader_tw[2].re);                               \
        tw_brd[3] = _mm512_set_pd(                                         \
            rader_tw[3].im, rader_tw[3].re,                                \
            rader_tw[3].im, rader_tw[3].re,                                \
            rader_tw[3].im, rader_tw[3].re,                                \
            rader_tw[3].im, rader_tw[3].re);                               \
        tw_brd[4] = _mm512_set_pd(                                         \
            rader_tw[4].im, rader_tw[4].re,                                \
            rader_tw[4].im, rader_tw[4].re,                                \
            rader_tw[4].im, rader_tw[4].re,                                \
            rader_tw[4].im, rader_tw[4].re);                               \
        tw_brd[5] = _mm512_set_pd(                                         \
            rader_tw[5].im, rader_tw[5].re,                                \
            rader_tw[5].im, rader_tw[5].re,                                \
            rader_tw[5].im, rader_tw[5].re,                                \
            rader_tw[5].im, rader_tw[5].re);                               \
    } while (0)

//==============================================================================
// RADER CYCLIC CONVOLUTION - AVX-512
//==============================================================================

/**
 * @brief 6-point cyclic convolution for Rader's algorithm (4 butterflies)
 * 
 * Core of Rader's algorithm: convolve permuted inputs with precomputed twiddles.
 * Computes v[q] = sum_{l=0}^{5} tx[l] * tw[(q-l) mod 6] for q=0..5
 * 
 * Uses fused multiply-add operations for optimal performance.
 * This is the most expensive part of radix-7 (36 complex multiplies per butterfly).
 * 
 * @param tx0-tx5 Permuted input values
 * @param tw_brd  Broadcast Rader twiddles (6 vectors)
 * @param v0-v5   Output convolution results
 */
#define RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                    v0, v1, v2, v3, v4, v5)               \
    do {                                                                   \
        /* Initialize accumulators to zero */                              \
        v0 = _mm512_setzero_pd();                                          \
        v1 = _mm512_setzero_pd();                                          \
        v2 = _mm512_setzero_pd();                                          \
        v3 = _mm512_setzero_pd();                                          \
        v4 = _mm512_setzero_pd();                                          \
        v5 = _mm512_setzero_pd();                                          \
        \
        /* q=0: indices [0,5,4,3,2,1] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v0, tx0, tw_brd[0]);                        \
        CMUL_ADD_FMA_R7_AVX512(v0, tx1, tw_brd[5]);                        \
        CMUL_ADD_FMA_R7_AVX512(v0, tx2, tw_brd[4]);                        \
        CMUL_ADD_FMA_R7_AVX512(v0, tx3, tw_brd[3]);                        \
        CMUL_ADD_FMA_R7_AVX512(v0, tx4, tw_brd[2]);                        \
        CMUL_ADD_FMA_R7_AVX512(v0, tx5, tw_brd[1]);                        \
        \
        /* q=1: indices [1,0,5,4,3,2] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v1, tx0, tw_brd[1]);                        \
        CMUL_ADD_FMA_R7_AVX512(v1, tx1, tw_brd[0]);                        \
        CMUL_ADD_FMA_R7_AVX512(v1, tx2, tw_brd[5]);                        \
        CMUL_ADD_FMA_R7_AVX512(v1, tx3, tw_brd[4]);                        \
        CMUL_ADD_FMA_R7_AVX512(v1, tx4, tw_brd[3]);                        \
        CMUL_ADD_FMA_R7_AVX512(v1, tx5, tw_brd[2]);                        \
        \
        /* q=2: indices [2,1,0,5,4,3] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v2, tx0, tw_brd[2]);                        \
        CMUL_ADD_FMA_R7_AVX512(v2, tx1, tw_brd[1]);                        \
        CMUL_ADD_FMA_R7_AVX512(v2, tx2, tw_brd[0]);                        \
        CMUL_ADD_FMA_R7_AVX512(v2, tx3, tw_brd[5]);                        \
        CMUL_ADD_FMA_R7_AVX512(v2, tx4, tw_brd[4]);                        \
        CMUL_ADD_FMA_R7_AVX512(v2, tx5, tw_brd[3]);                        \
        \
        /* q=3: indices [3,2,1,0,5,4] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v3, tx0, tw_brd[3]);                        \
        CMUL_ADD_FMA_R7_AVX512(v3, tx1, tw_brd[2]);                        \
        CMUL_ADD_FMA_R7_AVX512(v3, tx2, tw_brd[1]);                        \
        CMUL_ADD_FMA_R7_AVX512(v3, tx3, tw_brd[0]);                        \
        CMUL_ADD_FMA_R7_AVX512(v3, tx4, tw_brd[5]);                        \
        CMUL_ADD_FMA_R7_AVX512(v3, tx5, tw_brd[4]);                        \
        \
        /* q=4: indices [4,3,2,1,0,5] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v4, tx0, tw_brd[4]);                        \
        CMUL_ADD_FMA_R7_AVX512(v4, tx1, tw_brd[3]);                        \
        CMUL_ADD_FMA_R7_AVX512(v4, tx2, tw_brd[2]);                        \
        CMUL_ADD_FMA_R7_AVX512(v4, tx3, tw_brd[1]);                        \
        CMUL_ADD_FMA_R7_AVX512(v4, tx4, tw_brd[0]);                        \
        CMUL_ADD_FMA_R7_AVX512(v4, tx5, tw_brd[5]);                        \
        \
        /* q=5: indices [5,4,3,2,1,0] */                                   \
        CMUL_ADD_FMA_R7_AVX512(v5, tx0, tw_brd[5]);                        \
        CMUL_ADD_FMA_R7_AVX512(v5, tx1, tw_brd[4]);                        \
        CMUL_ADD_FMA_R7_AVX512(v5, tx2, tw_brd[3]);                        \
        CMUL_ADD_FMA_R7_AVX512(v5, tx3, tw_brd[2]);                        \
        CMUL_ADD_FMA_R7_AVX512(v5, tx4, tw_brd[1]);                        \
        CMUL_ADD_FMA_R7_AVX512(v5, tx5, tw_brd[0]);                        \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512
//==============================================================================

/**
 * @brief Assemble final outputs using inverse permutation
 * 
 * Adds DC component (x0) to each convolution result (v0-v5) and
 * places them in the correct output positions according to out_perm=[1,5,4,6,2,3].
 * 
 * @param x0    DC component (copied to all non-DC outputs)
 * @param v0-v5 Convolution results
 * @param y0-y6 Final output values (7 lanes)
 */
#define ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,           \
                                   y0, y1, y2, y3, y4, y5, y6)           \
    do {                                                                  \
        y1 = _mm512_add_pd(x0, v0);  /* out_perm[0] = 1 */               \
        y5 = _mm512_add_pd(x0, v1);  /* out_perm[1] = 5 */               \
        y4 = _mm512_add_pd(x0, v2);  /* out_perm[2] = 4 */               \
        y6 = _mm512_add_pd(x0, v3);  /* out_perm[3] = 6 */               \
        y2 = _mm512_add_pd(x0, v4);  /* out_perm[4] = 2 */               \
        y3 = _mm512_add_pd(x0, v5);  /* out_perm[5] = 3 */               \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED STAGE TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief Apply inter-stage twiddle factors to inputs x1-x6
 * 
 * In multi-stage FFTs, each butterfly needs to multiply by twiddle factors
 * that depend on its position in the stage. x0 is never multiplied (always 1).
 * 
 * @param kk       Butterfly index (base of 4 parallel butterflies)
 * @param x1-x6    Input/output values (modified in-place)
 * @param stage_tw Stage twiddle array (6 twiddles per butterfly)
 * @param sub_len  Length of sub-transform (skip if sub_len==1)
 */
#define APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len) \
    do {                                                                               \
        if (sub_len > 1) {                                                             \
            __m512d w1 = load4_aos(&stage_tw[(kk)*6 + 0],                             \
                                   &stage_tw[(kk+1)*6 + 0],                            \
                                   &stage_tw[(kk+2)*6 + 0],                            \
                                   &stage_tw[(kk+3)*6 + 0]);                           \
            __m512d w2 = load4_aos(&stage_tw[(kk)*6 + 1],                             \
                                   &stage_tw[(kk+1)*6 + 1],                            \
                                   &stage_tw[(kk+2)*6 + 1],                            \
                                   &stage_tw[(kk+3)*6 + 1]);                           \
            __m512d w3 = load4_aos(&stage_tw[(kk)*6 + 2],                             \
                                   &stage_tw[(kk+1)*6 + 2],                            \
                                   &stage_tw[(kk+2)*6 + 2],                            \
                                   &stage_tw[(kk+3)*6 + 2]);                           \
            __m512d w4 = load4_aos(&stage_tw[(kk)*6 + 3],                             \
                                   &stage_tw[(kk+1)*6 + 3],                            \
                                   &stage_tw[(kk+2)*6 + 3],                            \
                                   &stage_tw[(kk+3)*6 + 3]);                           \
            __m512d w5 = load4_aos(&stage_tw[(kk)*6 + 4],                             \
                                   &stage_tw[(kk+1)*6 + 4],                            \
                                   &stage_tw[(kk+2)*6 + 4],                            \
                                   &stage_tw[(kk+3)*6 + 4]);                           \
            __m512d w6 = load4_aos(&stage_tw[(kk)*6 + 5],                             \
                                   &stage_tw[(kk+1)*6 + 5],                            \
                                   &stage_tw[(kk+2)*6 + 5],                            \
                                   &stage_tw[(kk+3)*6 + 5]);                           \
                                                                                       \
            CMUL_FMA_R7_AVX512(x1, x1, w1);                                            \
            CMUL_FMA_R7_AVX512(x2, x2, w2);                                            \
            CMUL_FMA_R7_AVX512(x3, x3, w3);                                            \
            CMUL_FMA_R7_AVX512(x4, x4, w4);                                            \
            CMUL_FMA_R7_AVX512(x5, x5, w5);                                            \
            CMUL_FMA_R7_AVX512(x6, x6, w6);                                            \
        }                                                                              \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 7 lanes of data for 4 parallel butterflies
 * 
 * Loads data in AoS (Array of Structures) format where each butterfly's
 * 7 inputs are strided by K positions in memory.
 * 
 * @param kk          Base butterfly index
 * @param K           Stride between lanes
 * @param sub_outputs Input data array
 * @param x0-x6       Output registers (7 AVX-512 vectors)
 */
#define LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6) \
    do {                                                                     \
        x0 = load4_aos(&sub_outputs[kk],                                     \
                       &sub_outputs[(kk)+1],                                 \
                       &sub_outputs[(kk)+2],                                 \
                       &sub_outputs[(kk)+3]);                                \
        x1 = load4_aos(&sub_outputs[(kk)+K],                                 \
                       &sub_outputs[(kk)+1+K],                               \
                       &sub_outputs[(kk)+2+K],                               \
                       &sub_outputs[(kk)+3+K]);                              \
        x2 = load4_aos(&sub_outputs[(kk)+2*K],                               \
                       &sub_outputs[(kk)+1+2*K],                             \
                       &sub_outputs[(kk)+2+2*K],                             \
                       &sub_outputs[(kk)+3+2*K]);                            \
        x3 = load4_aos(&sub_outputs[(kk)+3*K],                               \
                       &sub_outputs[(kk)+1+3*K],                             \
                       &sub_outputs[(kk)+2+3*K],                             \
                       &sub_outputs[(kk)+3+3*K]);                            \
        x4 = load4_aos(&sub_outputs[(kk)+4*K],                               \
                       &sub_outputs[(kk)+1+4*K],                             \
                       &sub_outputs[(kk)+2+4*K],                             \
                       &sub_outputs[(kk)+3+4*K]);                            \
        x5 = load4_aos(&sub_outputs[(kk)+5*K],                               \
                       &sub_outputs[(kk)+1+5*K],                             \
                       &sub_outputs[(kk)+2+5*K],                             \
                       &sub_outputs[(kk)+3+5*K]);                            \
        x6 = load4_aos(&sub_outputs[(kk)+6*K],                               \
                       &sub_outputs[(kk)+1+6*K],                             \
                       &sub_outputs[(kk)+2+6*K],                             \
                       &sub_outputs[(kk)+3+6*K]);                            \
    } while (0)

/**
 * @brief Store 7 lanes of data for 4 parallel butterflies (normal stores)
 * 
 * Uses regular unaligned stores. Suitable for small/medium transforms
 * where data fits in cache.
 */
#define STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6) \
    do {                                                                        \
        STOREU_PD512(&output_buffer[kk].re, y0);                                \
        STOREU_PD512(&output_buffer[(kk)+K].re, y1);                            \
        STOREU_PD512(&output_buffer[(kk)+2*K].re, y2);                          \
        STOREU_PD512(&output_buffer[(kk)+3*K].re, y3);                          \
        STOREU_PD512(&output_buffer[(kk)+4*K].re, y4);                          \
        STOREU_PD512(&output_buffer[(kk)+5*K].re, y5);                          \
        STOREU_PD512(&output_buffer[(kk)+6*K].re, y6);                          \
    } while (0)

/**
 * @brief Store 7 lanes of data using streaming (non-temporal) stores
 * 
 * For large transforms (K >= STREAM_THRESHOLD), streaming stores avoid
 * polluting the cache with write-allocate traffic.
 * 
 * @note Requires _mm_sfence() after the loop
 */
#define STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6) \
    do {                                                                               \
        _mm512_stream_pd(&output_buffer[kk].re, y0);                                   \
        _mm512_stream_pd(&output_buffer[(kk)+K].re, y1);                               \
        _mm512_stream_pd(&output_buffer[(kk)+2*K].re, y2);                             \
        _mm512_stream_pd(&output_buffer[(kk)+3*K].re, y3);                             \
        _mm512_stream_pd(&output_buffer[(kk)+4*K].re, y4);                             \
        _mm512_stream_pd(&output_buffer[(kk)+5*K].re, y5);                             \
        _mm512_stream_pd(&output_buffer[(kk)+6*K].re, y6);                             \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512
//==============================================================================

/** @brief Prefetch distance for L1 cache (4 iterations ahead) */
#define PREFETCH_L1_R7_AVX512 16

/** @brief Prefetch distance for twiddle factors */
#define PREFETCH_TWIDDLE_R7_AVX512 16

/**
 * @brief Prefetch all 7 lanes for upcoming butterflies
 * 
 * Issues prefetch hints for data that will be needed in future iterations.
 * Single-level prefetch to L1 cache to avoid pollution.
 * 
 * @param k         Current butterfly index
 * @param K         Stride between lanes
 * @param distance  How many iterations ahead to prefetch
 * @param sub_outputs Data array to prefetch from
 * @param hint      Cache hint (_MM_HINT_T0 for temporal L1)
 */
#define PREFETCH_7_LANES_R7_AVX512(k, K, distance, sub_outputs, hint)             \
    do {                                                                          \
        if ((k) + (distance) < K) {                                               \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], hint);       \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+K], hint);     \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+2*K], hint);   \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+3*K], hint);   \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+4*K], hint);   \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+5*K], hint);   \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+6*K], hint);   \
        }                                                                         \
    } while (0)

//==============================================================================
// RADER PERMUTATIONS
//==============================================================================

/**
 * @brief Permute non-DC inputs according to Rader's algorithm
 * 
 * Generator g=3 for prime 7 produces permutation: [1,3,2,6,4,5]
 * This reorders inputs before the cyclic convolution.
 * 
 * @param x1-x6   Original inputs
 * @param tx0-tx5 Permuted outputs
 */
#define PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5) \
    do { \
        tx0 = x1; \
        tx1 = x3; \
        tx2 = x2; \
        tx3 = x6; \
        tx4 = x4; \
        tx5 = x5; \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINES - AVX-512
//==============================================================================

/**
 * @brief Complete radix-7 butterfly pipeline with internal broadcast (forward)
 * 
 * Processes 4 butterflies in parallel. Broadcasts Rader twiddles inside the macro.
 * Use this version when Rader twiddles cannot be hoisted.
 * 
 * @param kk            Butterfly index (0, 4, 8, ...)
 * @param K             Sub-transform length
 * @param sub_outputs   Input data
 * @param stage_tw      Stage twiddle factors
 * @param rader_tw      Rader twiddle factors (6 complex doubles)
 * @param output_buffer Output data
 * @param sub_len       Length indicator (controls stage twiddle application)
 */
#define RADIX7_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                             \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                          \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                         \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);               \
        __m512d y0;                                                                                  \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                        \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                        \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                    \
        __m512d tw_brd[6];                                                                           \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                        \
        __m512d v0, v1, v2, v3, v4, v5;                                                              \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                            \
                                    v0, v1, v2, v3, v4, v5);                                         \
        __m512d y1, y2, y3, y4, y5, y6;                                                              \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                       \
                                   y0, y1, y2, y3, y4, y5, y6);                                      \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

/**
 * @brief Complete radix-7 butterfly pipeline with hoisted broadcast (forward, normal stores)
 * 
 * More efficient version where Rader twiddles are broadcast outside the loop.
 * Use this when possible for ~1-2% speedup.
 * 
 * @param tw_brd Pre-broadcast Rader twiddles (array of 6 AVX-512 vectors)
 */
#define RADIX7_PIPELINE_4_FV_AVX512_HOISTED(kk, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do {                                                                                                   \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                               \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                     \
        __m512d y0;                                                                                        \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                              \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                              \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                          \
        __m512d v0, v1, v2, v3, v4, v5;                                                                    \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                  \
                                    v0, v1, v2, v3, v4, v5);                                               \
        __m512d y1, y2, y3, y4, y5, y6;                                                                    \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                             \
                                   y0, y1, y2, y3, y4, y5, y6);                                            \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                            \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast and streaming stores (forward)
 */
#define RADIX7_PIPELINE_4_FV_AVX512_STREAM_HOISTED(kk, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do {                                                                                                          \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                       \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                                      \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                            \
        __m512d y0;                                                                                               \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                                     \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                                     \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                                 \
        __m512d v0, v1, v2, v3, v4, v5;                                                                           \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                         \
                                    v0, v1, v2, v3, v4, v5);                                                      \
        __m512d y1, y2, y3, y4, y5, y6;                                                                           \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                                    \
                                   y0, y1, y2, y3, y4, y5, y6);                                                   \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                            \
    } while (0)

/**
 * @brief Complete radix-7 butterfly pipeline (inverse, internal broadcast)
 */
#define RADIX7_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                             \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                          \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                         \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);               \
        __m512d y0;                                                                                  \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                        \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                        \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                    \
        __m512d tw_brd[6];                                                                           \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                        \
        __m512d v0, v1, v2, v3, v4, v5;                                                              \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                            \
                                    v0, v1, v2, v3, v4, v5);                                         \
        __m512d y1, y2, y3, y4, y5, y6;                                                              \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                       \
                                   y0, y1, y2, y3, y4, y5, y6);                                      \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast (inverse, normal stores)
 */
#define RADIX7_PIPELINE_4_BV_AVX512_HOISTED(kk, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do {                                                                                                   \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                               \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                     \
        __m512d y0;                                                                                        \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                              \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                              \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                          \
        __m512d v0, v1, v2, v3, v4, v5;                                                                    \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                  \
                                    v0, v1, v2, v3, v4, v5);                                               \
        __m512d y1, y2, y3, y4, y5, y6;                                                                    \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                             \
                                   y0, y1, y2, y3, y4, y5, y6);                                            \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                            \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast and streaming stores (inverse)
 */
#define RADIX7_PIPELINE_4_BV_AVX512_STREAM_HOISTED(kk, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do {                                                                                                          \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                       \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                                      \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                            \
        __m512d y0;                                                                                               \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                                     \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                                     \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                                 \
        __m512d v0, v1, v2, v3, v4, v5;                                                                           \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                         \
                                    v0, v1, v2, v3, v4, v5);                                                      \
        __m512d y1, y2, y3, y4, y5, y6;                                                                           \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                                    \
                                   y0, y1, y2, y3, y4, y5, y6);                                                   \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                            \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX2
//==============================================================================

/**
 * @brief Complex multiply for AVX2: out = a * w (2 complex values)
 * 
 * Uses FMA if available (Haswell+), falls back to separate mul/add on Ivy Bridge.
 */
#if defined(__FMA__)
#define CMUL_FMA_R7_AVX2(out, a, w)                                       \
    do                                                                    \
    {                                                                     \
        __m256d ar = _mm256_unpacklo_pd(a, a);                            \
        __m256d ai = _mm256_unpackhi_pd(a, a);                            \
        __m256d wr = _mm256_unpacklo_pd(w, w);                            \
        __m256d wi = _mm256_unpackhi_pd(w, w);                            \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi));     \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr));     \
        (out) = _mm256_unpacklo_pd(re, im);                               \
    } while (0)
#else
#define CMUL_FMA_R7_AVX2(out, a, w) (out) = cmul_avx2_aos(a, w)
#endif

//==============================================================================
// FUSED COMPLEX MULTIPLY-ADD - AVX2
//==============================================================================

/**
 * @brief Fused complex multiply-add: acc += a * w (2 complex values)
 * 
 * Critical for Rader convolution performance on AVX2 systems.
 */
#if defined(__FMA__)
#define CMUL_ADD_FMA_R7_AVX2(acc, a, w)                                   \
    do                                                                    \
    {                                                                     \
        __m256d ar = _mm256_unpacklo_pd(a, a);                            \
        __m256d ai = _mm256_unpackhi_pd(a, a);                            \
        __m256d wr = _mm256_unpacklo_pd(w, w);                            \
        __m256d wi = _mm256_unpackhi_pd(w, w);                            \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi));     \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr));     \
        (acc) = _mm256_add_pd(acc, _mm256_unpacklo_pd(re, im));          \
    } while (0)
#else
#define CMUL_ADD_FMA_R7_AVX2(acc, a, w)                                   \
    do                                                                    \
    {                                                                     \
        __m256d tmp = cmul_avx2_aos(a, w);                                \
        (acc) = _mm256_add_pd(acc, tmp);                                  \
    } while (0)
#endif

//==============================================================================
// Y0 COMPUTATION - AVX2
//==============================================================================

/**
 * @brief Compute DC component (sum of all inputs)
 */
#define COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0) \
    do { \
        y0 = _mm256_add_pd( \
            _mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)), \
            _mm256_add_pd(_mm256_add_pd(x4, x5), x6)); \
    } while (0)

//==============================================================================
// APPLY STAGE TWIDDLES - AVX2
//==============================================================================

/**
 * @brief Apply inter-stage twiddle factors (AVX2 version)
 */
#define APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len) \
    do { \
        if (sub_len > 1) { \
            __m256d w1 = load2_aos(&stage_tw[6*(k)+0], &stage_tw[6*(k+1)+0]); \
            __m256d w2 = load2_aos(&stage_tw[6*(k)+1], &stage_tw[6*(k+1)+1]); \
            __m256d w3 = load2_aos(&stage_tw[6*(k)+2], &stage_tw[6*(k+1)+2]); \
            __m256d w4 = load2_aos(&stage_tw[6*(k)+3], &stage_tw[6*(k+1)+3]); \
            __m256d w5 = load2_aos(&stage_tw[6*(k)+4], &stage_tw[6*(k+1)+4]); \
            __m256d w6 = load2_aos(&stage_tw[6*(k)+5], &stage_tw[6*(k+1)+5]); \
            \
            CMUL_FMA_R7_AVX2(x1, x1, w1); \
            CMUL_FMA_R7_AVX2(x2, x2, w2); \
            CMUL_FMA_R7_AVX2(x3, x3, w3); \
            CMUL_FMA_R7_AVX2(x4, x4, w4); \
            CMUL_FMA_R7_AVX2(x5, x5, w5); \
            CMUL_FMA_R7_AVX2(x6, x6, w6); \
        } \
    } while (0)

//==============================================================================
// BROADCAST RADER TWIDDLES - AVX2
//==============================================================================

/**
 * @brief Broadcast 6 Rader twiddles for AVX2 (2 parallel butterflies)
 */
#define BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd) \
    do { \
        tw_brd[0] = _mm256_set_pd(rader_tw[0].im, rader_tw[0].re, \
                                   rader_tw[0].im, rader_tw[0].re); \
        tw_brd[1] = _mm256_set_pd(rader_tw[1].im, rader_tw[1].re, \
                                   rader_tw[1].im, rader_tw[1].re); \
        tw_brd[2] = _mm256_set_pd(rader_tw[2].im, rader_tw[2].re, \
                                   rader_tw[2].im, rader_tw[2].re); \
        tw_brd[3] = _mm256_set_pd(rader_tw[3].im, rader_tw[3].re, \
                                   rader_tw[3].im, rader_tw[3].re); \
        tw_brd[4] = _mm256_set_pd(rader_tw[4].im, rader_tw[4].re, \
                                   rader_tw[4].im, rader_tw[4].re); \
        tw_brd[5] = _mm256_set_pd(rader_tw[5].im, rader_tw[5].re, \
                                   rader_tw[5].im, rader_tw[5].re); \
    } while (0)

//==============================================================================
// RADER CYCLIC CONVOLUTION - AVX2
//==============================================================================

/**
 * @brief 6-point cyclic convolution for AVX2 (2 butterflies)
 * 
 * Uses fused multiply-add operations when FMA is available.
 */
#define RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5) \
    do { \
        v0 = _mm256_setzero_pd(); \
        v1 = _mm256_setzero_pd(); \
        v2 = _mm256_setzero_pd(); \
        v3 = _mm256_setzero_pd(); \
        v4 = _mm256_setzero_pd(); \
        v5 = _mm256_setzero_pd(); \
        \
        CMUL_ADD_FMA_R7_AVX2(v0, tx0, tw_brd[0]); \
        CMUL_ADD_FMA_R7_AVX2(v0, tx1, tw_brd[5]); \
        CMUL_ADD_FMA_R7_AVX2(v0, tx2, tw_brd[4]); \
        CMUL_ADD_FMA_R7_AVX2(v0, tx3, tw_brd[3]); \
        CMUL_ADD_FMA_R7_AVX2(v0, tx4, tw_brd[2]); \
        CMUL_ADD_FMA_R7_AVX2(v0, tx5, tw_brd[1]); \
        \
        CMUL_ADD_FMA_R7_AVX2(v1, tx0, tw_brd[1]); \
        CMUL_ADD_FMA_R7_AVX2(v1, tx1, tw_brd[0]); \
        CMUL_ADD_FMA_R7_AVX2(v1, tx2, tw_brd[5]); \
        CMUL_ADD_FMA_R7_AVX2(v1, tx3, tw_brd[4]); \
        CMUL_ADD_FMA_R7_AVX2(v1, tx4, tw_brd[3]); \
        CMUL_ADD_FMA_R7_AVX2(v1, tx5, tw_brd[2]); \
        \
        CMUL_ADD_FMA_R7_AVX2(v2, tx0, tw_brd[2]); \
        CMUL_ADD_FMA_R7_AVX2(v2, tx1, tw_brd[1]); \
        CMUL_ADD_FMA_R7_AVX2(v2, tx2, tw_brd[0]); \
        CMUL_ADD_FMA_R7_AVX2(v2, tx3, tw_brd[5]); \
        CMUL_ADD_FMA_R7_AVX2(v2, tx4, tw_brd[4]); \
        CMUL_ADD_FMA_R7_AVX2(v2, tx5, tw_brd[3]); \
        \
        CMUL_ADD_FMA_R7_AVX2(v3, tx0, tw_brd[3]); \
        CMUL_ADD_FMA_R7_AVX2(v3, tx1, tw_brd[2]); \
        CMUL_ADD_FMA_R7_AVX2(v3, tx2, tw_brd[1]); \
        CMUL_ADD_FMA_R7_AVX2(v3, tx3, tw_brd[0]); \
        CMUL_ADD_FMA_R7_AVX2(v3, tx4, tw_brd[5]); \
        CMUL_ADD_FMA_R7_AVX2(v3, tx5, tw_brd[4]); \
        \
        CMUL_ADD_FMA_R7_AVX2(v4, tx0, tw_brd[4]); \
        CMUL_ADD_FMA_R7_AVX2(v4, tx1, tw_brd[3]); \
        CMUL_ADD_FMA_R7_AVX2(v4, tx2, tw_brd[2]); \
        CMUL_ADD_FMA_R7_AVX2(v4, tx3, tw_brd[1]); \
        CMUL_ADD_FMA_R7_AVX2(v4, tx4, tw_brd[0]); \
        CMUL_ADD_FMA_R7_AVX2(v4, tx5, tw_brd[5]); \
        \
        CMUL_ADD_FMA_R7_AVX2(v5, tx0, tw_brd[5]); \
        CMUL_ADD_FMA_R7_AVX2(v5, tx1, tw_brd[4]); \
        CMUL_ADD_FMA_R7_AVX2(v5, tx2, tw_brd[3]); \
        CMUL_ADD_FMA_R7_AVX2(v5, tx3, tw_brd[2]); \
        CMUL_ADD_FMA_R7_AVX2(v5, tx4, tw_brd[1]); \
        CMUL_ADD_FMA_R7_AVX2(v5, tx5, tw_brd[0]); \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX2
//==============================================================================

/**
 * @brief Assemble final outputs (AVX2 version)
 */
#define ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6) \
    do { \
        y1 = _mm256_add_pd(x0, v0); \
        y5 = _mm256_add_pd(x0, v1); \
        y4 = _mm256_add_pd(x0, v2); \
        y6 = _mm256_add_pd(x0, v3); \
        y2 = _mm256_add_pd(x0, v4); \
        y3 = _mm256_add_pd(x0, v5); \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX2
//==============================================================================

/**
 * @brief Load 7 lanes for 2 parallel butterflies
 */
#define LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6) \
    do { \
        x0 = load2_aos(&sub_outputs[(k)+0*K], &sub_outputs[(k)+1+0*K]); \
        x1 = load2_aos(&sub_outputs[(k)+1*K], &sub_outputs[(k)+1+1*K]); \
        x2 = load2_aos(&sub_outputs[(k)+2*K], &sub_outputs[(k)+1+2*K]); \
        x3 = load2_aos(&sub_outputs[(k)+3*K], &sub_outputs[(k)+1+3*K]); \
        x4 = load2_aos(&sub_outputs[(k)+4*K], &sub_outputs[(k)+1+4*K]); \
        x5 = load2_aos(&sub_outputs[(k)+5*K], &sub_outputs[(k)+1+5*K]); \
        x6 = load2_aos(&sub_outputs[(k)+6*K], &sub_outputs[(k)+1+6*K]); \
    } while (0)

/**
 * @brief Store 7 lanes using normal stores (AVX2 version)
 */
#define STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6) \
    do { \
        STOREU_PD(&output_buffer[(k)+0*K].re, y0); \
        STOREU_PD(&output_buffer[(k)+1*K].re, y1); \
        STOREU_PD(&output_buffer[(k)+2*K].re, y2); \
        STOREU_PD(&output_buffer[(k)+3*K].re, y3); \
        STOREU_PD(&output_buffer[(k)+4*K].re, y4); \
        STOREU_PD(&output_buffer[(k)+5*K].re, y5); \
        STOREU_PD(&output_buffer[(k)+6*K].re, y6); \
    } while (0)

/**
 * @brief Store 7 lanes using streaming stores (AVX2 version)
 */
#define STORE_7_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6) \
    do { \
        _mm256_stream_pd(&output_buffer[(k)+0*K].re, y0); \
        _mm256_stream_pd(&output_buffer[(k)+1*K].re, y1); \
        _mm256_stream_pd(&output_buffer[(k)+2*K].re, y2); \
        _mm256_stream_pd(&output_buffer[(k)+3*K].re, y3); \
        _mm256_stream_pd(&output_buffer[(k)+4*K].re, y4); \
        _mm256_stream_pd(&output_buffer[(k)+5*K].re, y5); \
        _mm256_stream_pd(&output_buffer[(k)+6*K].re, y6); \
    } while (0)

//==============================================================================
// PREFETCHING - AVX2
//==============================================================================

/** @brief Prefetch distance for AVX2 systems */
#define PREFETCH_L1_R7 8
#define PREFETCH_TWIDDLE_R7 8

/**
 * @brief Prefetch 7 lanes for AVX2
 */
#define PREFETCH_7_LANES_R7_AVX2(k, K, distance, sub_outputs, hint) \
    do { \
        if ((k) + (distance) < K) { \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+2*K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+3*K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+4*K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+5*K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+6*K], hint); \
        } \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINES - AVX2
//==============================================================================

/**
 * @brief Complete radix-7 butterfly pipeline with internal broadcast (forward, AVX2)
 */
#define RADIX7_PIPELINE_2_FV_AVX2(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d tw_brd[6]; \
        BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast (forward, normal stores, AVX2)
 */
#define RADIX7_PIPELINE_2_FV_AVX2_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast and streaming stores (forward, AVX2)
 */
#define RADIX7_PIPELINE_2_FV_AVX2_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

/**
 * @brief Complete radix-7 butterfly pipeline (inverse, internal broadcast, AVX2)
 */
#define RADIX7_PIPELINE_2_BV_AVX2(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d tw_brd[6]; \
        BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast (inverse, normal stores, AVX2)
 */
#define RADIX7_PIPELINE_2_BV_AVX2_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

/**
 * @brief Radix-7 butterfly with hoisted broadcast and streaming stores (inverse, AVX2)
 */
#define RADIX7_PIPELINE_2_BV_AVX2_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

/**
 * @brief Scalar 6-point cyclic convolution for Rader's algorithm
 * 
 * Reference implementation, used for tail cases and systems without SIMD.
 * 
 * @param tx       Permuted input array (6 complex values)
 * @param rader_tw Rader twiddle factors (6 complex values)
 * @param v        Output convolution results (6 complex values)
 */
#define RADER_CONVOLUTION_R7_SCALAR(tx, rader_tw, v) \
    do { \
        for (int _q = 0; _q < 6; ++_q) { \
            v[_q].re = 0.0; \
            v[_q].im = 0.0; \
        } \
        \
        for (int _q = 0; _q < 6; ++_q) { \
            for (int _l = 0; _l < 6; ++_l) { \
                int _idx = (_q - _l); \
                if (_idx < 0) _idx += 6; \
                double _tr = tx[_l].re * rader_tw[_idx].re - tx[_l].im * rader_tw[_idx].im; \
                double _ti = tx[_l].re * rader_tw[_idx].im + tx[_l].im * rader_tw[_idx].re; \
                v[_q].re += _tr; \
                v[_q].im += _ti; \
            } \
        } \
    } while (0)

//==============================================================================
// SCALAR BUTTERFLY MACROS
//==============================================================================

/**
 * @brief Complete scalar radix-7 butterfly (forward version)
 * 
 * Implements full Rader's algorithm in scalar mode for tail cases.
 * Used when K is not a multiple of SIMD width.
 * 
 * @param k             Butterfly index
 * @param K             Stride between lanes
 * @param sub_outputs   Input data
 * @param stage_tw      Stage twiddle factors
 * @param rader_tw      Rader twiddle factors
 * @param output_buffer Output data
 * @param sub_len       Sub-transform length
 */
#define RADIX7_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        fft_data x[7], tx[6], v[6], y[7]; \
        \
        /* Load 7 inputs */ \
        for (int _i = 0; _i < 7; _i++) { \
            x[_i] = sub_outputs[k + _i*K]; \
        } \
        \
        /* Apply stage twiddles (if not base case) */ \
        if (sub_len > 1) { \
            for (int _r = 1; _r < 7; _r++) { \
                const fft_data *w = &stage_tw[6*k + (_r-1)]; \
                double tmp_re = x[_r].re * w->re - x[_r].im * w->im; \
                double tmp_im = x[_r].re * w->im + x[_r].im * w->re; \
                x[_r].re = tmp_re; \
                x[_r].im = tmp_im; \
            } \
        } \
        \
        /* Compute y0 (DC component) */ \
        y[0].re = x[0].re + x[1].re + x[2].re + x[3].re + x[4].re + x[5].re + x[6].re; \
        y[0].im = x[0].im + x[1].im + x[2].im + x[3].im + x[4].im + x[5].im + x[6].im; \
        \
        /* Permute inputs: perm_in = [1,3,2,6,4,5] */ \
        tx[0] = x[1]; \
        tx[1] = x[3]; \
        tx[2] = x[2]; \
        tx[3] = x[6]; \
        tx[4] = x[4]; \
        tx[5] = x[5]; \
        \
        /* 6-point cyclic convolution */ \
        RADER_CONVOLUTION_R7_SCALAR(tx, rader_tw, v); \
        \
        /* Assemble outputs: out_perm = [1,5,4,6,2,3] */ \
        y[1].re = x[0].re + v[0].re; \
        y[1].im = x[0].im + v[0].im; \
        y[5].re = x[0].re + v[1].re; \
        y[5].im = x[0].im + v[1].im; \
        y[4].re = x[0].re + v[2].re; \
        y[4].im = x[0].im + v[2].im; \
        y[6].re = x[0].re + v[3].re; \
        y[6].im = x[0].im + v[3].im; \
        y[2].re = x[0].re + v[4].re; \
        y[2].im = x[0].im + v[4].im; \
        y[3].re = x[0].re + v[5].re; \
        y[3].im = x[0].im + v[5].im; \
        \
        /* Store 7 outputs */ \
        for (int _i = 0; _i < 7; _i++) { \
            output_buffer[k + _i*K] = y[_i]; \
        } \
    } while (0)

/**
 * @brief Complete scalar radix-7 butterfly (inverse version)
 * 
 * Identical to forward version - only rader_tw sign differs (precomputed by manager).
 */
#define RADIX7_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        fft_data x[7], tx[6], v[6], y[7]; \
        \
        /* Load 7 inputs */ \
        for (int _i = 0; _i < 7; _i++) { \
            x[_i] = sub_outputs[k + _i*K]; \
        } \
        \
        /* Apply stage twiddles (if not base case) */ \
        if (sub_len > 1) { \
            for (int _r = 1; _r < 7; _r++) { \
                const fft_data *w = &stage_tw[6*k + (_r-1)]; \
                double tmp_re = x[_r].re * w->re - x[_r].im * w->im; \
                double tmp_im = x[_r].re * w->im + x[_r].im * w->re; \
                x[_r].re = tmp_re; \
                x[_r].im = tmp_im; \
            } \
        } \
        \
        /* Compute y0 (DC component) */ \
        y[0].re = x[0].re + x[1].re + x[2].re + x[3].re + x[4].re + x[5].re + x[6].re; \
        y[0].im = x[0].im + x[1].im + x[2].im + x[3].im + x[4].im + x[5].im + x[6].im; \
        \
        /* Permute inputs: perm_in = [1,3,2,6,4,5] */ \
        tx[0] = x[1]; \
        tx[1] = x[3]; \
        tx[2] = x[2]; \
        tx[3] = x[6]; \
        tx[4] = x[4]; \
        tx[5] = x[5]; \
        \
        /* 6-point cyclic convolution (rader_tw has inverse sign) */ \
        RADER_CONVOLUTION_R7_SCALAR(tx, rader_tw, v); \
        \
        /* Assemble outputs: out_perm = [1,5,4,6,2,3] */ \
        y[1].re = x[0].re + v[0].re; \
        y[1].im = x[0].im + v[0].im; \
        y[5].re = x[0].re + v[1].re; \
        y[5].im = x[0].im + v[1].im; \
        y[4].re = x[0].re + v[2].re; \
        y[4].im = x[0].im + v[2].im; \
        y[6].re = x[0].re + v[3].re; \
        y[6].im = x[0].im + v[3].im; \
        y[2].re = x[0].re + v[4].re; \
        y[2].im = x[0].im + v[4].im; \
        y[3].re = x[0].re + v[5].re; \
        y[3].im = x[0].im + v[5].im; \
        \
        /* Store 7 outputs */ \
        for (int _i = 0; _i < 7; _i++) { \
            output_buffer[k + _i*K] = y[_i]; \
        } \
    } while (0)

#endif // FFT_RADIX7_MACROS_H

//==============================================================================
// OPTIMIZATION SUMMARY
//==============================================================================

/**
 * OPTIMIZATIONS IMPLEMENTED:
 * 
 * 1. ✅ FMA Support
 *    - AVX-512: Always uses FMA (part of AVX-512F)
 *    - AVX2: Conditional FMA (Haswell+), fallback for Ivy Bridge
 *    - Estimated gain: 5-10% on FMA-capable CPUs
 * 
 * 2. ✅ Fused Multiply-Add Operations
 *    - CMUL_ADD_FMA_R7_* macros eliminate temporary variables
 *    - Critical for Rader convolution (36 complex multiplies per butterfly)
 *    - Estimated gain: 2-5% (reduced register pressure)
 * 
 * 3. ✅ Hoisted Rader Twiddle Broadcasts
 *    - Broadcast done ONCE outside loop instead of per-iteration
 *    - Separate _HOISTED macros for all variants
 *    - Estimated gain: 1-2%
 * 
 * 4. ✅ Separate Streaming/Normal Store Loops
 *    - No branches in hot path
 *    - Streaming for K >= 8192 (avoids cache pollution)
 *    - Estimated gain: 1-3%
 * 
 * 5. ✅ Single-Level Prefetching
 *    - Tuned distances: 16 for AVX-512, 8 for AVX2
 *    - Only L1 prefetch (no cache pollution)
 *    - Estimated gain: 2-5%
 * 
 * 6. ✅ Alignment Hints
 *    - __builtin_assume_aligned in calling functions
 *    - Better compiler codegen
 *    - Estimated gain: 2-5%
 * 
 * TOTAL ESTIMATED GAIN: 15-30% over naive implementation
 * 
 * PERFORMANCE TARGETS:
 * - AVX-512: ~2.5 cycles/butterfly (4 butterflies per iteration)
 * - AVX2:    ~5.0 cycles/butterfly (2 butterflies per iteration)
 * - Scalar:  ~25 cycles/butterfly
 * 
 * ARCHITECTURE NOTES:
 * - Radix-7 is more expensive than radix-2/3/4 due to Rader's algorithm
 * - 6-point cyclic convolution is the bottleneck (36 complex multiplies)
 * - Still worth it for mixed-radix FFTs of size 7^k * 2^m
 * - Twiddles come from external manager (single source of truth)
 * - Forward/inverse handled by twiddle sign (no runtime branches)
 */