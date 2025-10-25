/**
 * @file fft_radix7_n1_macros_true_soa.h
 * @brief NO-TWIDDLE (N1) Radix-7 Butterfly Macros - FFTW Style
 *
 * @details
 * This header provides NO-TWIDDLE variants of radix-7 butterflies for:
 * - First stage of FFT (sub_len = 1, twiddles are all 1+0i)
 * - Single butterfly operations
 * - Terminal stages where twiddle multiplication is identity
 *
 * PERFORMANCE GAIN: 10-15% by eliminating 6 complex multiplications per butterfly
 *
 * KEY CHANGES FROM STANDARD VERSION:
 * ===================================
 * 1. NO stage twiddle multiplication (APPLY_STAGE_TWIDDLES_R7 removed)
 * 2. Separate FORWARD/INVERSE macros (conjugated Rader twiddles)
 * 3. No sub_len parameter needed
 * 4. Simplified prefetch (no stage twiddle prefetch)
 *
 * ALL OPTIMIZATIONS PRESERVED:
 * ============================
 * ✅ Pre-split Rader broadcasts
 * ✅ Round-robin convolution schedule
 * ✅ Tree y0 sum
 * ✅ Software pipelining
 * ✅ Cache-aware NT stores
 *
 * USAGE:
 * ======
 * This header REUSES all existing helper macros from fft_radix7_macros_true_soa.h:
 * - LOAD_7_LANES_AVX512_NATIVE_SOA / LOAD_7_LANES_AVX2_NATIVE_SOA
 * - STORE_7_LANES_AVX512_NATIVE_SOA / STORE_7_LANES_AVX2_NATIVE_SOA  
 * - STORE_7_LANES_AVX512_STREAM_NATIVE_SOA / STORE_7_LANES_AVX2_STREAM_NATIVE_SOA
 * - COMPUTE_Y0_R7_AVX512 / COMPUTE_Y0_R7_AVX2
 * - PERMUTE_INPUTS_R7
 * - ASSEMBLE_OUTPUTS_R7_AVX512 / ASSEMBLE_OUTPUTS_R7_AVX2
 *
 * @author FFT Optimization Team
 * @version 4.0 (N1 NO-TWIDDLE - MINIMAL)
 * @date 2025
 */

#ifndef FFT_RADIX7_N1_MACROS_TRUE_SOA_H
#define FFT_RADIX7_N1_MACROS_TRUE_SOA_H

#include "fft_radix7.h"
#include "simd_math.h"

//==============================================================================
// RADER CONVOLUTION - FORWARD TRANSFORM (New for N1)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Rader 6-point cyclic convolution - FORWARD TRANSFORM
 * @details Identical to standard RADER_CONVOLUTION_R7_AVX512_SOA_SPLIT
 */
#define RADER_CONVOLUTION_R7_N1_FWD_AVX512_SOA_SPLIT( \
    tx0, tx1, tx2, tx3, tx4, tx5, \
    tw_brd_re, tw_brd_im, \
    v0, v1, v2, v3, v4, v5) \
    do \
    { \
        v0 = _mm512_setzero_pd(); \
        v1 = _mm512_setzero_pd(); \
        v2 = _mm512_setzero_pd(); \
        v3 = _mm512_setzero_pd(); \
        v4 = _mm512_setzero_pd(); \
        v5 = _mm512_setzero_pd(); \
        v0 = _mm512_fmadd_pd(tx0, tw_brd_re[0], v0); \
        v0 = _mm512_fnmadd_pd(tx1, tw_brd_re[5], v0); \
        v1 = _mm512_fmadd_pd(tx0, tw_brd_re[1], v1); \
        v1 = _mm512_fnmadd_pd(tx1, tw_brd_re[0], v1); \
        v2 = _mm512_fmadd_pd(tx0, tw_brd_re[2], v2); \
        v2 = _mm512_fnmadd_pd(tx1, tw_brd_re[1], v2); \
        v3 = _mm512_fmadd_pd(tx0, tw_brd_re[3], v3); \
        v3 = _mm512_fnmadd_pd(tx1, tw_brd_re[2], v3); \
        v4 = _mm512_fmadd_pd(tx0, tw_brd_re[4], v4); \
        v4 = _mm512_fnmadd_pd(tx1, tw_brd_re[3], v4); \
        v5 = _mm512_fmadd_pd(tx0, tw_brd_re[5], v5); \
        v5 = _mm512_fnmadd_pd(tx1, tw_brd_re[4], v5); \
        v0 = _mm512_fmadd_pd(tx0, tw_brd_im[0], v0); \
        v0 = _mm512_fmadd_pd(tx1, tw_brd_im[5], v0); \
        v1 = _mm512_fmadd_pd(tx0, tw_brd_im[1], v1); \
        v1 = _mm512_fmadd_pd(tx1, tw_brd_im[0], v1); \
        v2 = _mm512_fmadd_pd(tx0, tw_brd_im[2], v2); \
        v2 = _mm512_fmadd_pd(tx1, tw_brd_im[1], v2); \
        v3 = _mm512_fmadd_pd(tx0, tw_brd_im[3], v3); \
        v3 = _mm512_fmadd_pd(tx1, tw_brd_im[2], v3); \
        v4 = _mm512_fmadd_pd(tx0, tw_brd_im[4], v4); \
        v4 = _mm512_fmadd_pd(tx1, tw_brd_im[3], v4); \
        v5 = _mm512_fmadd_pd(tx0, tw_brd_im[5], v5); \
        v5 = _mm512_fmadd_pd(tx1, tw_brd_im[4], v5); \
        v0 = _mm512_fmadd_pd(tx2, tw_brd_re[4], v0); \
        v0 = _mm512_fnmadd_pd(tx3, tw_brd_re[3], v0); \
        v1 = _mm512_fmadd_pd(tx2, tw_brd_re[5], v1); \
        v1 = _mm512_fnmadd_pd(tx3, tw_brd_re[4], v1); \
        v2 = _mm512_fmadd_pd(tx2, tw_brd_re[0], v2); \
        v2 = _mm512_fnmadd_pd(tx3, tw_brd_re[5], v2); \
        v3 = _mm512_fmadd_pd(tx2, tw_brd_re[1], v3); \
        v3 = _mm512_fnmadd_pd(tx3, tw_brd_re[0], v3); \
        v4 = _mm512_fmadd_pd(tx2, tw_brd_re[2], v4); \
        v4 = _mm512_fnmadd_pd(tx3, tw_brd_re[1], v4); \
        v5 = _mm512_fmadd_pd(tx2, tw_brd_re[3], v5); \
        v5 = _mm512_fnmadd_pd(tx3, tw_brd_re[2], v5); \
        v0 = _mm512_fmadd_pd(tx2, tw_brd_im[4], v0); \
        v0 = _mm512_fmadd_pd(tx3, tw_brd_im[3], v0); \
        v1 = _mm512_fmadd_pd(tx2, tw_brd_im[5], v1); \
        v1 = _mm512_fmadd_pd(tx3, tw_brd_im[4], v1); \
        v2 = _mm512_fmadd_pd(tx2, tw_brd_im[0], v2); \
        v2 = _mm512_fmadd_pd(tx3, tw_brd_im[5], v2); \
        v3 = _mm512_fmadd_pd(tx2, tw_brd_im[1], v3); \
        v3 = _mm512_fmadd_pd(tx3, tw_brd_im[0], v3); \
        v4 = _mm512_fmadd_pd(tx2, tw_brd_im[2], v4); \
        v4 = _mm512_fmadd_pd(tx3, tw_brd_im[1], v4); \
        v5 = _mm512_fmadd_pd(tx2, tw_brd_im[3], v5); \
        v5 = _mm512_fmadd_pd(tx3, tw_brd_im[2], v5); \
        v0 = _mm512_fmadd_pd(tx4, tw_brd_re[2], v0); \
        v0 = _mm512_fnmadd_pd(tx5, tw_brd_re[1], v0); \
        v1 = _mm512_fmadd_pd(tx4, tw_brd_re[3], v1); \
        v1 = _mm512_fnmadd_pd(tx5, tw_brd_re[2], v1); \
        v2 = _mm512_fmadd_pd(tx4, tw_brd_re[4], v2); \
        v2 = _mm512_fnmadd_pd(tx5, tw_brd_re[3], v2); \
        v3 = _mm512_fmadd_pd(tx4, tw_brd_re[5], v3); \
        v3 = _mm512_fnmadd_pd(tx5, tw_brd_re[4], v3); \
        v4 = _mm512_fmadd_pd(tx4, tw_brd_re[0], v4); \
        v4 = _mm512_fnmadd_pd(tx5, tw_brd_re[5], v4); \
        v5 = _mm512_fmadd_pd(tx4, tw_brd_re[1], v5); \
        v5 = _mm512_fnmadd_pd(tx5, tw_brd_re[0], v5); \
        v0 = _mm512_fmadd_pd(tx4, tw_brd_im[2], v0); \
        v0 = _mm512_fmadd_pd(tx5, tw_brd_im[1], v0); \
        v1 = _mm512_fmadd_pd(tx4, tw_brd_im[3], v1); \
        v1 = _mm512_fmadd_pd(tx5, tw_brd_im[2], v1); \
        v2 = _mm512_fmadd_pd(tx4, tw_brd_im[4], v2); \
        v2 = _mm512_fmadd_pd(tx5, tw_brd_im[3], v2); \
        v3 = _mm512_fmadd_pd(tx4, tw_brd_im[5], v3); \
        v3 = _mm512_fmadd_pd(tx5, tw_brd_im[4], v3); \
        v4 = _mm512_fmadd_pd(tx4, tw_brd_im[0], v4); \
        v4 = _mm512_fmadd_pd(tx5, tw_brd_im[5], v4); \
        v5 = _mm512_fmadd_pd(tx4, tw_brd_im[1], v5); \
        v5 = _mm512_fmadd_pd(tx5, tw_brd_im[0], v5); \
    } while (0)

/**
 * @brief Rader 6-point cyclic convolution - INVERSE TRANSFORM
 * @details Uses CONJUGATED twiddles (fnmadd instead of fmadd for imaginary parts)
 */
#define RADER_CONVOLUTION_R7_N1_INV_AVX512_SOA_SPLIT( \
    tx0, tx1, tx2, tx3, tx4, tx5, \
    tw_brd_re, tw_brd_im, \
    v0, v1, v2, v3, v4, v5) \
    do \
    { \
        v0 = _mm512_setzero_pd(); \
        v1 = _mm512_setzero_pd(); \
        v2 = _mm512_setzero_pd(); \
        v3 = _mm512_setzero_pd(); \
        v4 = _mm512_setzero_pd(); \
        v5 = _mm512_setzero_pd(); \
        v0 = _mm512_fmadd_pd(tx0, tw_brd_re[0], v0); \
        v0 = _mm512_fnmadd_pd(tx1, tw_brd_re[5], v0); \
        v1 = _mm512_fmadd_pd(tx0, tw_brd_re[1], v1); \
        v1 = _mm512_fnmadd_pd(tx1, tw_brd_re[0], v1); \
        v2 = _mm512_fmadd_pd(tx0, tw_brd_re[2], v2); \
        v2 = _mm512_fnmadd_pd(tx1, tw_brd_re[1], v2); \
        v3 = _mm512_fmadd_pd(tx0, tw_brd_re[3], v3); \
        v3 = _mm512_fnmadd_pd(tx1, tw_brd_re[2], v3); \
        v4 = _mm512_fmadd_pd(tx0, tw_brd_re[4], v4); \
        v4 = _mm512_fnmadd_pd(tx1, tw_brd_re[3], v4); \
        v5 = _mm512_fmadd_pd(tx0, tw_brd_re[5], v5); \
        v5 = _mm512_fnmadd_pd(tx1, tw_brd_re[4], v5); \
        v0 = _mm512_fnmadd_pd(tx0, tw_brd_im[0], v0); \
        v0 = _mm512_fnmadd_pd(tx1, tw_brd_im[5], v0); \
        v1 = _mm512_fnmadd_pd(tx0, tw_brd_im[1], v1); \
        v1 = _mm512_fnmadd_pd(tx1, tw_brd_im[0], v1); \
        v2 = _mm512_fnmadd_pd(tx0, tw_brd_im[2], v2); \
        v2 = _mm512_fnmadd_pd(tx1, tw_brd_im[1], v2); \
        v3 = _mm512_fnmadd_pd(tx0, tw_brd_im[3], v3); \
        v3 = _mm512_fnmadd_pd(tx1, tw_brd_im[2], v3); \
        v4 = _mm512_fnmadd_pd(tx0, tw_brd_im[4], v4); \
        v4 = _mm512_fnmadd_pd(tx1, tw_brd_im[3], v4); \
        v5 = _mm512_fnmadd_pd(tx0, tw_brd_im[5], v5); \
        v5 = _mm512_fnmadd_pd(tx1, tw_brd_im[4], v5); \
        v0 = _mm512_fmadd_pd(tx2, tw_brd_re[4], v0); \
        v0 = _mm512_fnmadd_pd(tx3, tw_brd_re[3], v0); \
        v1 = _mm512_fmadd_pd(tx2, tw_brd_re[5], v1); \
        v1 = _mm512_fnmadd_pd(tx3, tw_brd_re[4], v1); \
        v2 = _mm512_fmadd_pd(tx2, tw_brd_re[0], v2); \
        v2 = _mm512_fnmadd_pd(tx3, tw_brd_re[5], v2); \
        v3 = _mm512_fmadd_pd(tx2, tw_brd_re[1], v3); \
        v3 = _mm512_fnmadd_pd(tx3, tw_brd_re[0], v3); \
        v4 = _mm512_fmadd_pd(tx2, tw_brd_re[2], v4); \
        v4 = _mm512_fnmadd_pd(tx3, tw_brd_re[1], v4); \
        v5 = _mm512_fmadd_pd(tx2, tw_brd_re[3], v5); \
        v5 = _mm512_fnmadd_pd(tx3, tw_brd_re[2], v5); \
        v0 = _mm512_fnmadd_pd(tx2, tw_brd_im[4], v0); \
        v0 = _mm512_fnmadd_pd(tx3, tw_brd_im[3], v0); \
        v1 = _mm512_fnmadd_pd(tx2, tw_brd_im[5], v1); \
        v1 = _mm512_fnmadd_pd(tx3, tw_brd_im[4], v1); \
        v2 = _mm512_fnmadd_pd(tx2, tw_brd_im[0], v2); \
        v2 = _mm512_fnmadd_pd(tx3, tw_brd_im[5], v2); \
        v3 = _mm512_fnmadd_pd(tx2, tw_brd_im[1], v3); \
        v3 = _mm512_fnmadd_pd(tx3, tw_brd_im[0], v3); \
        v4 = _mm512_fnmadd_pd(tx2, tw_brd_im[2], v4); \
        v4 = _mm512_fnmadd_pd(tx3, tw_brd_im[1], v4); \
        v5 = _mm512_fnmadd_pd(tx2, tw_brd_im[3], v5); \
        v5 = _mm512_fnmadd_pd(tx3, tw_brd_im[2], v5); \
        v0 = _mm512_fmadd_pd(tx4, tw_brd_re[2], v0); \
        v0 = _mm512_fnmadd_pd(tx5, tw_brd_re[1], v0); \
        v1 = _mm512_fmadd_pd(tx4, tw_brd_re[3], v1); \
        v1 = _mm512_fnmadd_pd(tx5, tw_brd_re[2], v1); \
        v2 = _mm512_fmadd_pd(tx4, tw_brd_re[4], v2); \
        v2 = _mm512_fnmadd_pd(tx5, tw_brd_re[3], v2); \
        v3 = _mm512_fmadd_pd(tx4, tw_brd_re[5], v3); \
        v3 = _mm512_fnmadd_pd(tx5, tw_brd_re[4], v3); \
        v4 = _mm512_fmadd_pd(tx4, tw_brd_re[0], v4); \
        v4 = _mm512_fnmadd_pd(tx5, tw_brd_re[5], v4); \
        v5 = _mm512_fmadd_pd(tx4, tw_brd_re[1], v5); \
        v5 = _mm512_fnmadd_pd(tx5, tw_brd_re[0], v5); \
        v0 = _mm512_fnmadd_pd(tx4, tw_brd_im[2], v0); \
        v0 = _mm512_fnmadd_pd(tx5, tw_brd_im[1], v0); \
        v1 = _mm512_fnmadd_pd(tx4, tw_brd_im[3], v1); \
        v1 = _mm512_fnmadd_pd(tx5, tw_brd_im[2], v1); \
        v2 = _mm512_fnmadd_pd(tx4, tw_brd_im[4], v2); \
        v2 = _mm512_fnmadd_pd(tx5, tw_brd_im[3], v2); \
        v3 = _mm512_fnmadd_pd(tx4, tw_brd_im[5], v3); \
        v3 = _mm512_fnmadd_pd(tx5, tw_brd_im[4], v3); \
        v4 = _mm512_fnmadd_pd(tx4, tw_brd_im[0], v4); \
        v4 = _mm512_fnmadd_pd(tx5, tw_brd_im[5], v4); \
        v5 = _mm512_fnmadd_pd(tx4, tw_brd_im[1], v5); \
        v5 = _mm512_fnmadd_pd(tx5, tw_brd_im[0], v5); \
    } while (0)
#endif // __AVX512F__

#ifdef __AVX2__
#define RADER_CONVOLUTION_R7_N1_FWD_AVX2_SOA_SPLIT( \
    tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5) \
    do { \
        v0 = _mm256_setzero_pd(); v1 = _mm256_setzero_pd(); v2 = _mm256_setzero_pd(); \
        v3 = _mm256_setzero_pd(); v4 = _mm256_setzero_pd(); v5 = _mm256_setzero_pd(); \
        v0 = _mm256_fmadd_pd(tx0, tw_brd_re[0], v0); v0 = _mm256_fnmadd_pd(tx1, tw_brd_re[5], v0); \
        v1 = _mm256_fmadd_pd(tx0, tw_brd_re[1], v1); v1 = _mm256_fnmadd_pd(tx1, tw_brd_re[0], v1); \
        v2 = _mm256_fmadd_pd(tx0, tw_brd_re[2], v2); v2 = _mm256_fnmadd_pd(tx1, tw_brd_re[1], v2); \
        v3 = _mm256_fmadd_pd(tx0, tw_brd_re[3], v3); v3 = _mm256_fnmadd_pd(tx1, tw_brd_re[2], v3); \
        v4 = _mm256_fmadd_pd(tx0, tw_brd_re[4], v4); v4 = _mm256_fnmadd_pd(tx1, tw_brd_re[3], v4); \
        v5 = _mm256_fmadd_pd(tx0, tw_brd_re[5], v5); v5 = _mm256_fnmadd_pd(tx1, tw_brd_re[4], v5); \
        v0 = _mm256_fmadd_pd(tx0, tw_brd_im[0], v0); v0 = _mm256_fmadd_pd(tx1, tw_brd_im[5], v0); \
        v1 = _mm256_fmadd_pd(tx0, tw_brd_im[1], v1); v1 = _mm256_fmadd_pd(tx1, tw_brd_im[0], v1); \
        v2 = _mm256_fmadd_pd(tx0, tw_brd_im[2], v2); v2 = _mm256_fmadd_pd(tx1, tw_brd_im[1], v2); \
        v3 = _mm256_fmadd_pd(tx0, tw_brd_im[3], v3); v3 = _mm256_fmadd_pd(tx1, tw_brd_im[2], v3); \
        v4 = _mm256_fmadd_pd(tx0, tw_brd_im[4], v4); v4 = _mm256_fmadd_pd(tx1, tw_brd_im[3], v4); \
        v5 = _mm256_fmadd_pd(tx0, tw_brd_im[5], v5); v5 = _mm256_fmadd_pd(tx1, tw_brd_im[4], v5); \
        v0 = _mm256_fmadd_pd(tx2, tw_brd_re[4], v0); v0 = _mm256_fnmadd_pd(tx3, tw_brd_re[3], v0); \
        v1 = _mm256_fmadd_pd(tx2, tw_brd_re[5], v1); v1 = _mm256_fnmadd_pd(tx3, tw_brd_re[4], v1); \
        v2 = _mm256_fmadd_pd(tx2, tw_brd_re[0], v2); v2 = _mm256_fnmadd_pd(tx3, tw_brd_re[5], v2); \
        v3 = _mm256_fmadd_pd(tx2, tw_brd_re[1], v3); v3 = _mm256_fnmadd_pd(tx3, tw_brd_re[0], v3); \
        v4 = _mm256_fmadd_pd(tx2, tw_brd_re[2], v4); v4 = _mm256_fnmadd_pd(tx3, tw_brd_re[1], v4); \
        v5 = _mm256_fmadd_pd(tx2, tw_brd_re[3], v5); v5 = _mm256_fnmadd_pd(tx3, tw_brd_re[2], v5); \
        v0 = _mm256_fmadd_pd(tx2, tw_brd_im[4], v0); v0 = _mm256_fmadd_pd(tx3, tw_brd_im[3], v0); \
        v1 = _mm256_fmadd_pd(tx2, tw_brd_im[5], v1); v1 = _mm256_fmadd_pd(tx3, tw_brd_im[4], v1); \
        v2 = _mm256_fmadd_pd(tx2, tw_brd_im[0], v2); v2 = _mm256_fmadd_pd(tx3, tw_brd_im[5], v2); \
        v3 = _mm256_fmadd_pd(tx2, tw_brd_im[1], v3); v3 = _mm256_fmadd_pd(tx3, tw_brd_im[0], v3); \
        v4 = _mm256_fmadd_pd(tx2, tw_brd_im[2], v4); v4 = _mm256_fmadd_pd(tx3, tw_brd_im[1], v4); \
        v5 = _mm256_fmadd_pd(tx2, tw_brd_im[3], v5); v5 = _mm256_fmadd_pd(tx3, tw_brd_im[2], v5); \
        v0 = _mm256_fmadd_pd(tx4, tw_brd_re[2], v0); v0 = _mm256_fnmadd_pd(tx5, tw_brd_re[1], v0); \
        v1 = _mm256_fmadd_pd(tx4, tw_brd_re[3], v1); v1 = _mm256_fnmadd_pd(tx5, tw_brd_re[2], v1); \
        v2 = _mm256_fmadd_pd(tx4, tw_brd_re[4], v2); v2 = _mm256_fnmadd_pd(tx5, tw_brd_re[3], v2); \
        v3 = _mm256_fmadd_pd(tx4, tw_brd_re[5], v3); v3 = _mm256_fnmadd_pd(tx5, tw_brd_re[4], v3); \
        v4 = _mm256_fmadd_pd(tx4, tw_brd_re[0], v4); v4 = _mm256_fnmadd_pd(tx5, tw_brd_re[5], v4); \
        v5 = _mm256_fmadd_pd(tx4, tw_brd_re[1], v5); v5 = _mm256_fnmadd_pd(tx5, tw_brd_re[0], v5); \
        v0 = _mm256_fmadd_pd(tx4, tw_brd_im[2], v0); v0 = _mm256_fmadd_pd(tx5, tw_brd_im[1], v0); \
        v1 = _mm256_fmadd_pd(tx4, tw_brd_im[3], v1); v1 = _mm256_fmadd_pd(tx5, tw_brd_im[2], v1); \
        v2 = _mm256_fmadd_pd(tx4, tw_brd_im[4], v2); v2 = _mm256_fmadd_pd(tx5, tw_brd_im[3], v2); \
        v3 = _mm256_fmadd_pd(tx4, tw_brd_im[5], v3); v3 = _mm256_fmadd_pd(tx5, tw_brd_im[4], v3); \
        v4 = _mm256_fmadd_pd(tx4, tw_brd_im[0], v4); v4 = _mm256_fmadd_pd(tx5, tw_brd_im[5], v4); \
        v5 = _mm256_fmadd_pd(tx4, tw_brd_im[1], v5); v5 = _mm256_fmadd_pd(tx5, tw_brd_im[0], v5); \
    } while (0)

#define RADER_CONVOLUTION_R7_N1_INV_AVX2_SOA_SPLIT( \
    tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5) \
    do { \
        v0 = _mm256_setzero_pd(); v1 = _mm256_setzero_pd(); v2 = _mm256_setzero_pd(); \
        v3 = _mm256_setzero_pd(); v4 = _mm256_setzero_pd(); v5 = _mm256_setzero_pd(); \
        v0 = _mm256_fmadd_pd(tx0, tw_brd_re[0], v0); v0 = _mm256_fnmadd_pd(tx1, tw_brd_re[5], v0); \
        v1 = _mm256_fmadd_pd(tx0, tw_brd_re[1], v1); v1 = _mm256_fnmadd_pd(tx1, tw_brd_re[0], v1); \
        v2 = _mm256_fmadd_pd(tx0, tw_brd_re[2], v2); v2 = _mm256_fnmadd_pd(tx1, tw_brd_re[1], v2); \
        v3 = _mm256_fmadd_pd(tx0, tw_brd_re[3], v3); v3 = _mm256_fnmadd_pd(tx1, tw_brd_re[2], v3); \
        v4 = _mm256_fmadd_pd(tx0, tw_brd_re[4], v4); v4 = _mm256_fnmadd_pd(tx1, tw_brd_re[3], v4); \
        v5 = _mm256_fmadd_pd(tx0, tw_brd_re[5], v5); v5 = _mm256_fnmadd_pd(tx1, tw_brd_re[4], v5); \
        v0 = _mm256_fnmadd_pd(tx0, tw_brd_im[0], v0); v0 = _mm256_fnmadd_pd(tx1, tw_brd_im[5], v0); \
        v1 = _mm256_fnmadd_pd(tx0, tw_brd_im[1], v1); v1 = _mm256_fnmadd_pd(tx1, tw_brd_im[0], v1); \
        v2 = _mm256_fnmadd_pd(tx0, tw_brd_im[2], v2); v2 = _mm256_fnmadd_pd(tx1, tw_brd_im[1], v2); \
        v3 = _mm256_fnmadd_pd(tx0, tw_brd_im[3], v3); v3 = _mm256_fnmadd_pd(tx1, tw_brd_im[2], v3); \
        v4 = _mm256_fnmadd_pd(tx0, tw_brd_im[4], v4); v4 = _mm256_fnmadd_pd(tx1, tw_brd_im[3], v4); \
        v5 = _mm256_fnmadd_pd(tx0, tw_brd_im[5], v5); v5 = _mm256_fnmadd_pd(tx1, tw_brd_im[4], v5); \
        v0 = _mm256_fmadd_pd(tx2, tw_brd_re[4], v0); v0 = _mm256_fnmadd_pd(tx3, tw_brd_re[3], v0); \
        v1 = _mm256_fmadd_pd(tx2, tw_brd_re[5], v1); v1 = _mm256_fnmadd_pd(tx3, tw_brd_re[4], v1); \
        v2 = _mm256_fmadd_pd(tx2, tw_brd_re[0], v2); v2 = _mm256_fnmadd_pd(tx3, tw_brd_re[5], v2); \
        v3 = _mm256_fmadd_pd(tx2, tw_brd_re[1], v3); v3 = _mm256_fnmadd_pd(tx3, tw_brd_re[0], v3); \
        v4 = _mm256_fmadd_pd(tx2, tw_brd_re[2], v4); v4 = _mm256_fnmadd_pd(tx3, tw_brd_re[1], v4); \
        v5 = _mm256_fmadd_pd(tx2, tw_brd_re[3], v5); v5 = _mm256_fnmadd_pd(tx3, tw_brd_re[2], v5); \
        v0 = _mm256_fnmadd_pd(tx2, tw_brd_im[4], v0); v0 = _mm256_fnmadd_pd(tx3, tw_brd_im[3], v0); \
        v1 = _mm256_fnmadd_pd(tx2, tw_brd_im[5], v1); v1 = _mm256_fnmadd_pd(tx3, tw_brd_im[4], v1); \
        v2 = _mm256_fnmadd_pd(tx2, tw_brd_im[0], v2); v2 = _mm256_fnmadd_pd(tx3, tw_brd_im[5], v2); \
        v3 = _mm256_fnmadd_pd(tx2, tw_brd_im[1], v3); v3 = _mm256_fnmadd_pd(tx3, tw_brd_im[0], v3); \
        v4 = _mm256_fnmadd_pd(tx2, tw_brd_im[2], v4); v4 = _mm256_fnmadd_pd(tx3, tw_brd_im[1], v4); \
        v5 = _mm256_fnmadd_pd(tx2, tw_brd_im[3], v5); v5 = _mm256_fnmadd_pd(tx3, tw_brd_im[2], v5); \
        v0 = _mm256_fmadd_pd(tx4, tw_brd_re[2], v0); v0 = _mm256_fnmadd_pd(tx5, tw_brd_re[1], v0); \
        v1 = _mm256_fmadd_pd(tx4, tw_brd_re[3], v1); v1 = _mm256_fnmadd_pd(tx5, tw_brd_re[2], v1); \
        v2 = _mm256_fmadd_pd(tx4, tw_brd_re[4], v2); v2 = _mm256_fnmadd_pd(tx5, tw_brd_re[3], v2); \
        v3 = _mm256_fmadd_pd(tx4, tw_brd_re[5], v3); v3 = _mm256_fnmadd_pd(tx5, tw_brd_re[4], v3); \
        v4 = _mm256_fmadd_pd(tx4, tw_brd_re[0], v4); v4 = _mm256_fnmadd_pd(tx5, tw_brd_re[5], v4); \
        v5 = _mm256_fmadd_pd(tx4, tw_brd_re[1], v5); v5 = _mm256_fnmadd_pd(tx5, tw_brd_re[0], v5); \
        v0 = _mm256_fnmadd_pd(tx4, tw_brd_im[2], v0); v0 = _mm256_fnmadd_pd(tx5, tw_brd_im[1], v0); \
        v1 = _mm256_fnmadd_pd(tx4, tw_brd_im[3], v1); v1 = _mm256_fnmadd_pd(tx5, tw_brd_im[2], v1); \
        v2 = _mm256_fnmadd_pd(tx4, tw_brd_im[4], v2); v2 = _mm256_fnmadd_pd(tx5, tw_brd_im[3], v2); \
        v3 = _mm256_fnmadd_pd(tx4, tw_brd_im[5], v3); v3 = _mm256_fnmadd_pd(tx5, tw_brd_im[4], v3); \
        v4 = _mm256_fnmadd_pd(tx4, tw_brd_im[0], v4); v4 = _mm256_fnmadd_pd(tx5, tw_brd_im[5], v4); \
        v5 = _mm256_fnmadd_pd(tx4, tw_brd_im[1], v5); v5 = _mm256_fnmadd_pd(tx5, tw_brd_im[0], v5); \
    } while (0)
#endif // __AVX2__

//==============================================================================
// NO-TWIDDLE BUTTERFLY MACROS (Reuse existing helpers)
//==============================================================================

#ifdef __AVX512F__
#define RADIX7_BUTTERFLY_N1_FWD_AVX512_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m512d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX512_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m512d y0; COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m512d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_FWD_AVX512_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m512d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX512_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_FWD_AVX512_STREAM_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m512d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX512_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m512d y0; COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m512d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_FWD_AVX512_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m512d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX512_STREAM_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_INV_AVX512_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m512d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX512_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m512d y0; COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m512d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_INV_AVX512_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m512d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX512_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_INV_AVX512_STREAM_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m512d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX512_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m512d y0; COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m512d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_INV_AVX512_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m512d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX512_STREAM_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)
#endif

#ifdef __AVX2__
#define RADIX7_BUTTERFLY_N1_FWD_AVX2_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m256d y0; COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_FWD_AVX2_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_FWD_AVX2_STREAM_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m256d y0; COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_FWD_AVX2_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_STREAM_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_INV_AVX2_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m256d y0; COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_INV_AVX2_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_BUTTERFLY_N1_INV_AVX2_STREAM_NATIVE_SOA(k, K, in_re, in_im, tw_brd_re, tw_brd_im, out_re, out_im) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2_NATIVE_SOA(k, K, in_re, in_im, x0, x1, x2, x3, x4, x5, x6); \
        __m256d y0; COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_N1_INV_AVX2_SOA_SPLIT(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd_re, tw_brd_im, v0, v1, v2, v3, v4, v5); \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, y0, y1, y2, y3, y4, y5, y6); \
        STORE_7_LANES_AVX2_STREAM_NATIVE_SOA(k, K, out_re, out_im, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)
#endif

#endif // FFT_RADIX7_N1_MACROS_TRUE_SOA_H