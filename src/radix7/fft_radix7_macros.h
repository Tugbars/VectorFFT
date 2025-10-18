//==============================================================================
// fft_radix7_macros.h - Shared Macros for Radix-7 Rader Butterflies
//==============================================================================
//
// USAGE:
//   #include "fft_radix7_macros.h" in both fft_radix7_fv.c and fft_radix7_bv.c
//
// BENEFITS:
//   - 99% code reuse between forward/inverse
//   - Single source of truth for Rader's algorithm
//   - Only difference: convolution twiddle sign (from Rader Manager)
//

#ifndef FFT_RADIX7_MACROS_H
#define FFT_RADIX7_MACROS_H

#include "simd_math.h"

//==============================================================================
// RADER PERMUTATIONS (UNIVERSAL CONSTANTS FOR RADIX-7)
//==============================================================================

// Generator g=3 for prime 7:
//   perm_in  = [1,3,2,6,4,5]  (reorder inputs x1..x6 before convolution)
//   out_perm = [1,5,4,6,2,3]  (where conv[q] lands in output)

// These permutations are IDENTICAL for both forward and inverse
// Only the convolution twiddle signs differ

//==============================================================================
// COMPLEX MULTIPLICATION - IDENTICAL for both directions
//==============================================================================

/**
 * @brief Complex multiply for AoS layout (uses cmul_avx2_aos from simd_math.h)
 *
 * This macro performs a complex multiplication using AVX2 instructions from simd_math.h.
 * It is used for applying twiddle factors during the convolution in Rader's algorithm for both forward and inverse transforms.
 */
#ifdef __AVX2__
#define CMUL_R7(out, a, w) \
    do { \
        out = cmul_avx2_aos(a, w); \
    } while (0)
#endif

//==============================================================================
// APPLY PRECOMPUTED STAGE TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Apply stage twiddles for 2 butterflies (k and k+1)
 * 
 * Stage twiddles: stage_tw[k*6 + (r-1)] = W_N^(r*k) for r=1..6
 *
 * This macro applies precomputed twiddle factors to the six non-DC inputs for two butterflies simultaneously using AVX2.
 * It loads twiddles in AoS format and multiplies them with inputs, skipping if sub_len <= 1 (base case).
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw) \
    do { \
        if (sub_len > 1) { \
            __m256d w1 = load2_aos(&stage_tw[6*(k)+0], &stage_tw[6*(k+1)+0]); \
            __m256d w2 = load2_aos(&stage_tw[6*(k)+1], &stage_tw[6*(k+1)+1]); \
            __m256d w3 = load2_aos(&stage_tw[6*(k)+2], &stage_tw[6*(k+1)+2]); \
            __m256d w4 = load2_aos(&stage_tw[6*(k)+3], &stage_tw[6*(k+1)+3]); \
            __m256d w5 = load2_aos(&stage_tw[6*(k)+4], &stage_tw[6*(k+1)+4]); \
            __m256d w6 = load2_aos(&stage_tw[6*(k)+5], &stage_tw[6*(k+1)+5]); \
            \
            x1 = cmul_avx2_aos(x1, w1); \
            x2 = cmul_avx2_aos(x2, w2); \
            x3 = cmul_avx2_aos(x3, w3); \
            x4 = cmul_avx2_aos(x4, w4); \
            x5 = cmul_avx2_aos(x5, w5); \
            x6 = cmul_avx2_aos(x6, w6); \
        } \
    } while (0)
#endif

//==============================================================================
// RADER Y0 COMPUTATION - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief y0 = sum of all inputs (DC component)
 *
 * This macro computes the DC output (y0) as the sum of all seven inputs using AVX2 additions.
 * It is the first step in Rader's algorithm, isolating the zero-frequency component.
 */
#ifdef __AVX2__
#define COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0) \
    do { \
        y0 = _mm256_add_pd( \
            _mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)), \
            _mm256_add_pd(_mm256_add_pd(x4, x5), x6)); \
    } while (0)
#endif

//==============================================================================
// RADER INPUT PERMUTATION - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Permute inputs according to perm_in = [1,3,2,6,4,5]
 * 
 * tx = [x1, x3, x2, x6, x4, x5]
 *
 * This macro reorders the six non-DC inputs according to Rader's input permutation for the cyclic convolution.
 * It prepares the inputs for the 6-point convolution and is identical for forward and inverse.
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
// RADER CYCLIC CONVOLUTION - IDENTICAL for forward/inverse (uses precomputed tw)
//==============================================================================

/**
 * @brief 6-point cyclic convolution: conv[q] = Σ_l tx[l] * rader_tw[(q-l) mod 6]
 * 
 * @param tx0..tx5 Permuted inputs
 * @param tw_brd Precomputed Rader twiddles (broadcast for AVX2)
 * @param v0..v5 Convolution outputs
 * 
 * NOTE: rader_tw is precomputed with correct sign by Rader Manager
 *
 * This macro computes the 6-point cyclic convolution using AVX2 complex multiplications and additions.
 * It is the core of Rader's algorithm, transforming the DFT into a convolution, with twiddle signs handling forward/inverse.
 */
#ifdef __AVX2__
#define RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5) \
    do { \
        /* q=0: indices [0,5,4,3,2,1] */ \
        v0 = cmul_avx2_aos(tx0, tw_brd[0]); \
        v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx1, tw_brd[5])); \
        v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx2, tw_brd[4])); \
        v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx3, tw_brd[3])); \
        v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx4, tw_brd[2])); \
        v0 = _mm256_add_pd(v0, cmul_avx2_aos(tx5, tw_brd[1])); \
        \
        /* q=1: indices [1,0,5,4,3,2] */ \
        v1 = cmul_avx2_aos(tx0, tw_brd[1]); \
        v1 = _mm256_add_pd(v1, cmul_avx2_aos(tx1, tw_brd[0])); \
        v1 = _mm256_add_pd(v1, cmul_avx2_aos(tx2, tw_brd[5])); \
        v1 = _mm256_add_pd(v1, cmul_avx2_aos(tx3, tw_brd[4])); \
        v1 = _mm256_add_pd(v1, cmul_avx2_aos(tx4, tw_brd[3])); \
        v1 = _mm256_add_pd(v1, cmul_avx2_aos(tx5, tw_brd[2])); \
        \
        /* q=2: indices [2,1,0,5,4,3] */ \
        v2 = cmul_avx2_aos(tx0, tw_brd[2]); \
        v2 = _mm256_add_pd(v2, cmul_avx2_aos(tx1, tw_brd[1])); \
        v2 = _mm256_add_pd(v2, cmul_avx2_aos(tx2, tw_brd[0])); \
        v2 = _mm256_add_pd(v2, cmul_avx2_aos(tx3, tw_brd[5])); \
        v2 = _mm256_add_pd(v2, cmul_avx2_aos(tx4, tw_brd[4])); \
        v2 = _mm256_add_pd(v2, cmul_avx2_aos(tx5, tw_brd[3])); \
        \
        /* q=3: indices [3,2,1,0,5,4] */ \
        v3 = cmul_avx2_aos(tx0, tw_brd[3]); \
        v3 = _mm256_add_pd(v3, cmul_avx2_aos(tx1, tw_brd[2])); \
        v3 = _mm256_add_pd(v3, cmul_avx2_aos(tx2, tw_brd[1])); \
        v3 = _mm256_add_pd(v3, cmul_avx2_aos(tx3, tw_brd[0])); \
        v3 = _mm256_add_pd(v3, cmul_avx2_aos(tx4, tw_brd[5])); \
        v3 = _mm256_add_pd(v3, cmul_avx2_aos(tx5, tw_brd[4])); \
        \
        /* q=4: indices [4,3,2,1,0,5] */ \
        v4 = cmul_avx2_aos(tx0, tw_brd[4]); \
        v4 = _mm256_add_pd(v4, cmul_avx2_aos(tx1, tw_brd[3])); \
        v4 = _mm256_add_pd(v4, cmul_avx2_aos(tx2, tw_brd[2])); \
        v4 = _mm256_add_pd(v4, cmul_avx2_aos(tx3, tw_brd[1])); \
        v4 = _mm256_add_pd(v4, cmul_avx2_aos(tx4, tw_brd[0])); \
        v4 = _mm256_add_pd(v4, cmul_avx2_aos(tx5, tw_brd[5])); \
        \
        /* q=5: indices [5,4,3,2,1,0] */ \
        v5 = cmul_avx2_aos(tx0, tw_brd[5]); \
        v5 = _mm256_add_pd(v5, cmul_avx2_aos(tx1, tw_brd[4])); \
        v5 = _mm256_add_pd(v5, cmul_avx2_aos(tx2, tw_brd[3])); \
        v5 = _mm256_add_pd(v5, cmul_avx2_aos(tx3, tw_brd[2])); \
        v5 = _mm256_add_pd(v5, cmul_avx2_aos(tx4, tw_brd[1])); \
        v5 = _mm256_add_pd(v5, cmul_avx2_aos(tx5, tw_brd[0])); \
    } while (0)
#endif

//==============================================================================
// RADER OUTPUT ASSEMBLY - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Assemble final outputs using out_perm = [1,5,4,6,2,3]
 * 
 * y[out_perm[q]] = x0 + conv[q]
 *
 * This macro adds the non-DC input (x0) to each convolution output and assigns them to final outputs according to Rader's output permutation.
 * It completes the radix-7 butterfly by placing results in y1 to y6 (y0 is the DC sum).
 */
#ifdef __AVX2__
#define ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6) \
    do { \
        /* y0 already computed (DC) */ \
        y1 = _mm256_add_pd(x0, v0);  /* out_perm[0] = 1 */ \
        y5 = _mm256_add_pd(x0, v1);  /* out_perm[1] = 5 */ \
        y4 = _mm256_add_pd(x0, v2);  /* out_perm[2] = 4 */ \
        y6 = _mm256_add_pd(x0, v3);  /* out_perm[3] = 6 */ \
        y2 = _mm256_add_pd(x0, v4);  /* out_perm[4] = 2 */ \
        y3 = _mm256_add_pd(x0, v5);  /* out_perm[5] = 3 */ \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Load 7 lanes for two butterflies (k and k+1) using AVX2.
 *
 * This macro loads seven strided inputs from the sub_outputs buffer into AVX2 vectors.
 * Each vector holds two complex values (for two butterflies), assuming AoS layout.
 */
#ifdef __AVX2__
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
#endif

/**
 * @brief Store 7 lanes for two butterflies (k and k+1) using AVX2.
 *
 * This macro stores seven AVX2 vectors (each with two complex values) back to the output_buffer in strided fashion.
 * It uses unaligned stores for flexibility.
 */
#ifdef __AVX2__
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
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Prefetch distances for L1, L2, and L3 caches in radix-7.
 *
 * These constants define how far ahead to prefetch data in terms of indices for the radix-7 butterfly.
 * They are tuned to optimize memory access by loading data into caches preemptively.
 */
#define PREFETCH_L1_R7 16
#define PREFETCH_L2_R7 32
#define PREFETCH_L3_R7 64

/**
 * @brief Prefetch 7 lanes ahead for AVX2 in radix-7.
 *
 * This macro issues prefetch instructions for future strided data accesses in the sub_outputs buffer.
 * It prefetches all seven lanes, using the specified cache hint to optimize memory hierarchy usage.
 */
#ifdef __AVX2__
#define PREFETCH_7_LANES_R7(k, K, distance, sub_outputs, hint) \
    do { \
        if ((k) + (distance) < K) { \
            for (int _lane = 0; _lane < 7; _lane++) { \
                _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+_lane*K], hint); \
            } \
        } \
    } while (0)
#endif

//==============================================================================
// BROADCAST RADER TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Broadcast Rader convolution twiddles for AVX2
 * 
 * Converts rader_tw[6] into tw_brd[6] for AoS complex multiply
 *
 * This macro prepares the six Rader twiddles by broadcasting each complex value into AVX2 vectors.
 * Each tw_brd[q] holds the same twiddle duplicated for two butterflies, enabling SIMD convolution.
 */
#ifdef __AVX2__
#define BROADCAST_RADER_TWIDDLES_R7(rader_tw, tw_brd) \
    do { \
        for (int _q = 0; _q < 6; ++_q) { \
            tw_brd[_q] = _mm256_set_pd( \
                rader_tw[_q].im, rader_tw[_q].re, \
                rader_tw[_q].im, rader_tw[_q].re); \
        } \
    } while (0)
#endif

//==============================================================================
// SCALAR RADER CONVOLUTION - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Scalar 6-point cyclic convolution
 *
 * This macro computes the 6-point cyclic convolution in scalar mode, accumulating products for each output v[q].
 * It is used for tail cases or non-SIMD environments, with twiddle signs handling forward/inverse via precomputation.
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

#endif // FFT_RADIX7_MACROS_H