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
// AVX-512 SUPPORT - Radix-7 Rader's Algorithm (processes 4 butterflies)
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

/**
 * @brief Optimized complex multiply for AVX-512: out = a * w (4 complex values)
 */
#define CMUL_FMA_R7_AVX512(out, a, w)                                     \
    do                                                                    \
    {                                                                     \
        __m512d ar = _mm512_unpacklo_pd(a, a);                            \
        __m512d ai = _mm512_unpackhi_pd(a, a);                            \
        __m512d wr = _mm512_unpacklo_pd(w, w);                            \
        __m512d wi = _mm512_unpackhi_pd(w, w);                            \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));     \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));     \
        (out) = _mm512_unpacklo_pd(re, im);                               \
    } while (0)

//==============================================================================
// RADER Y0 COMPUTATION - AVX-512 (IDENTICAL for forward/inverse)
//==============================================================================

/**
 * @brief y0 = sum of all inputs (DC component) (AVX-512, 4 butterflies)
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
 * @brief Broadcast Rader convolution twiddles for AVX-512
 * 
 * Converts rader_tw[6] into tw_brd[6] for 4-way AoS complex multiply.
 * Each twiddle is replicated across all 4 butterfly positions.
 */
#define BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd)              \
    do {                                                                   \
        for (int _q = 0; _q < 6; ++_q) {                                   \
            tw_brd[_q] = _mm512_set_pd(                                    \
                rader_tw[_q].im, rader_tw[_q].re,                          \
                rader_tw[_q].im, rader_tw[_q].re,                          \
                rader_tw[_q].im, rader_tw[_q].re,                          \
                rader_tw[_q].im, rader_tw[_q].re);                         \
        }                                                                  \
    } while (0)

//==============================================================================
// RADER CYCLIC CONVOLUTION - AVX-512
//==============================================================================

/**
 * @brief 6-point cyclic convolution for AVX-512 (4 butterflies)
 * 
 * conv[q] = Σ_l tx[l] * rader_tw[(q-l) mod 6] for q=0..5
 * 
 * Processes 4 independent convolutions simultaneously.
 */
#define RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                    v0, v1, v2, v3, v4, v5)               \
    do {                                                                   \
        /* q=0: indices [0,5,4,3,2,1] */                                   \
        v0 = CMUL_FMA_R7_AVX512(v0, tx0, tw_brd[0]);                       \
        v0 = _mm512_add_pd(v0, CMUL_FMA_R7_AVX512(v0, tx1, tw_brd[5]));   \
        v0 = _mm512_add_pd(v0, CMUL_FMA_R7_AVX512(v0, tx2, tw_brd[4]));   \
        v0 = _mm512_add_pd(v0, CMUL_FMA_R7_AVX512(v0, tx3, tw_brd[3]));   \
        v0 = _mm512_add_pd(v0, CMUL_FMA_R7_AVX512(v0, tx4, tw_brd[2]));   \
        v0 = _mm512_add_pd(v0, CMUL_FMA_R7_AVX512(v0, tx5, tw_brd[1]));   \
                                                                           \
        /* q=1: indices [1,0,5,4,3,2] */                                   \
        v1 = CMUL_FMA_R7_AVX512(v1, tx0, tw_brd[1]);                       \
        v1 = _mm512_add_pd(v1, CMUL_FMA_R7_AVX512(v1, tx1, tw_brd[0]));   \
        v1 = _mm512_add_pd(v1, CMUL_FMA_R7_AVX512(v1, tx2, tw_brd[5]));   \
        v1 = _mm512_add_pd(v1, CMUL_FMA_R7_AVX512(v1, tx3, tw_brd[4]));   \
        v1 = _mm512_add_pd(v1, CMUL_FMA_R7_AVX512(v1, tx4, tw_brd[3]));   \
        v1 = _mm512_add_pd(v1, CMUL_FMA_R7_AVX512(v1, tx5, tw_brd[2]));   \
                                                                           \
        /* q=2: indices [2,1,0,5,4,3] */                                   \
        v2 = CMUL_FMA_R7_AVX512(v2, tx0, tw_brd[2]);                       \
        v2 = _mm512_add_pd(v2, CMUL_FMA_R7_AVX512(v2, tx1, tw_brd[1]));   \
        v2 = _mm512_add_pd(v2, CMUL_FMA_R7_AVX512(v2, tx2, tw_brd[0]));   \
        v2 = _mm512_add_pd(v2, CMUL_FMA_R7_AVX512(v2, tx3, tw_brd[5]));   \
        v2 = _mm512_add_pd(v2, CMUL_FMA_R7_AVX512(v2, tx4, tw_brd[4]));   \
        v2 = _mm512_add_pd(v2, CMUL_FMA_R7_AVX512(v2, tx5, tw_brd[3]));   \
                                                                           \
        /* q=3: indices [3,2,1,0,5,4] */                                   \
        v3 = CMUL_FMA_R7_AVX512(v3, tx0, tw_brd[3]);                       \
        v3 = _mm512_add_pd(v3, CMUL_FMA_R7_AVX512(v3, tx1, tw_brd[2]));   \
        v3 = _mm512_add_pd(v3, CMUL_FMA_R7_AVX512(v3, tx2, tw_brd[1]));   \
        v3 = _mm512_add_pd(v3, CMUL_FMA_R7_AVX512(v3, tx3, tw_brd[0]));   \
        v3 = _mm512_add_pd(v3, CMUL_FMA_R7_AVX512(v3, tx4, tw_brd[5]));   \
        v3 = _mm512_add_pd(v3, CMUL_FMA_R7_AVX512(v3, tx5, tw_brd[4]));   \
                                                                           \
        /* q=4: indices [4,3,2,1,0,5] */                                   \
        v4 = CMUL_FMA_R7_AVX512(v4, tx0, tw_brd[4]);                       \
        v4 = _mm512_add_pd(v4, CMUL_FMA_R7_AVX512(v4, tx1, tw_brd[3]));   \
        v4 = _mm512_add_pd(v4, CMUL_FMA_R7_AVX512(v4, tx2, tw_brd[2]));   \
        v4 = _mm512_add_pd(v4, CMUL_FMA_R7_AVX512(v4, tx3, tw_brd[1]));   \
        v4 = _mm512_add_pd(v4, CMUL_FMA_R7_AVX512(v4, tx4, tw_brd[0]));   \
        v4 = _mm512_add_pd(v4, CMUL_FMA_R7_AVX512(v4, tx5, tw_brd[5]));   \
                                                                           \
        /* q=5: indices [5,4,3,2,1,0] */                                   \
        v5 = CMUL_FMA_R7_AVX512(v5, tx0, tw_brd[5]);                       \
        v5 = _mm512_add_pd(v5, CMUL_FMA_R7_AVX512(v5, tx1, tw_brd[4]));   \
        v5 = _mm512_add_pd(v5, CMUL_FMA_R7_AVX512(v5, tx2, tw_brd[3]));   \
        v5 = _mm512_add_pd(v5, CMUL_FMA_R7_AVX512(v5, tx3, tw_brd[2]));   \
        v5 = _mm512_add_pd(v5, CMUL_FMA_R7_AVX512(v5, tx4, tw_brd[1]));   \
        v5 = _mm512_add_pd(v5, CMUL_FMA_R7_AVX512(v5, tx5, tw_brd[0]));   \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512 (IDENTICAL for forward/inverse)
//==============================================================================

/**
 * @brief Assemble final outputs using out_perm = [1,5,4,6,2,3] (AVX-512)
 * 
 * y[out_perm[q]] = x0 + conv[q] for q=0..5
 */
#define ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,           \
                                   y0, y1, y2, y3, y4, y5, y6)           \
    do {                                                                  \
        /* y0 already computed (DC component) */                          \
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
 * @brief AVX-512: Apply stage twiddles for 4 butterflies (kk through kk+3)
 *
 * stage_tw layout: [W^(1*k), W^(2*k), ..., W^(6*k)] for each k
 */
#define APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw) \
    do {                                                                      \
        if (sub_len > 1) {                                                    \
            __m512d w1 = load4_aos(&stage_tw[(kk)*6 + 0],                    \
                                   &stage_tw[(kk+1)*6 + 0],                   \
                                   &stage_tw[(kk+2)*6 + 0],                   \
                                   &stage_tw[(kk+3)*6 + 0]);                  \
            __m512d w2 = load4_aos(&stage_tw[(kk)*6 + 1],                    \
                                   &stage_tw[(kk+1)*6 + 1],                   \
                                   &stage_tw[(kk+2)*6 + 1],                   \
                                   &stage_tw[(kk+3)*6 + 1]);                  \
            __m512d w3 = load4_aos(&stage_tw[(kk)*6 + 2],                    \
                                   &stage_tw[(kk+1)*6 + 2],                   \
                                   &stage_tw[(kk+2)*6 + 2],                   \
                                   &stage_tw[(kk+3)*6 + 2]);                  \
            __m512d w4 = load4_aos(&stage_tw[(kk)*6 + 3],                    \
                                   &stage_tw[(kk+1)*6 + 3],                   \
                                   &stage_tw[(kk+2)*6 + 3],                   \
                                   &stage_tw[(kk+3)*6 + 3]);                  \
            __m512d w5 = load4_aos(&stage_tw[(kk)*6 + 4],                    \
                                   &stage_tw[(kk+1)*6 + 4],                   \
                                   &stage_tw[(kk+2)*6 + 4],                   \
                                   &stage_tw[(kk+3)*6 + 4]);                  \
            __m512d w6 = load4_aos(&stage_tw[(kk)*6 + 5],                    \
                                   &stage_tw[(kk+1)*6 + 5],                   \
                                   &stage_tw[(kk+2)*6 + 5],                   \
                                   &stage_tw[(kk+3)*6 + 5]);                  \
                                                                              \
            x1 = CMUL_FMA_R7_AVX512(x1, x1, w1);                              \
            x2 = CMUL_FMA_R7_AVX512(x2, x2, w2);                              \
            x3 = CMUL_FMA_R7_AVX512(x3, x3, w3);                              \
            x4 = CMUL_FMA_R7_AVX512(x4, x4, w4);                              \
            x5 = CMUL_FMA_R7_AVX512(x5, x5, w5);                              \
            x6 = CMUL_FMA_R7_AVX512(x6, x6, w6);                              \
        }                                                                     \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 7 lanes for 4 butterflies (kk through kk+3)
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
 * @brief Store 7 lanes for 4 butterflies (AVX-512)
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
 * @brief Store with streaming (AVX-512)
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

#define PREFETCH_L1_R7_AVX512 16
#define PREFETCH_L2_R7_AVX512 64
#define PREFETCH_L3_R7_AVX512 128

#define PREFETCH_7_LANES_R7_AVX512(k, K, distance, sub_outputs, stage_tw, hint)     \
    do {                                                                            \
        if ((k) + (distance) < K) {                                                 \
            for (int _lane = 0; _lane < 7; _lane++) {                               \
                _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+_lane*K], hint); \
            }                                                                       \
            _mm_prefetch((const char *)&stage_tw[((k)+(distance))*6], hint);        \
        }                                                                           \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

/**
 * @brief Complete AVX-512 radix-7 Rader butterfly (FORWARD, 4 butterflies)
 *
 * Processes 4 butterflies (28 complex values) in one macro call.
 * 
 * Algorithm (Rader's with generator g=3):
 * 1. Load 7 lanes for 4 butterflies (28 complex values)
 * 2. Apply input twiddles to lanes 1-6
 * 3. Compute y0 (DC component, sum of all inputs)
 * 4. Permute inputs according to perm_in = [1,3,2,6,4,5]
 * 5. Perform 6-point cyclic convolution with Rader twiddles
 * 6. Assemble outputs using out_perm = [1,5,4,6,2,3]
 * 7. Store 28 outputs
 */
#define RADIX7_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer) \
    do {                                                                                   \
        /* Step 1: Load 7 lanes for 4 butterflies */                                      \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);               \
                                                                                           \
        /* Step 2: Apply precomputed stage twiddles */                                    \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw);              \
                                                                                           \
        /* Step 3: Compute y0 (DC component) */                                           \
        __m512d y0;                                                                        \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                              \
                                                                                           \
        /* Step 4: Rader input permutation (perm_in = [1,3,2,6,4,5]) */                   \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                              \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);          \
                                                                                           \
        /* Step 5: Broadcast Rader twiddles */                                            \
        __m512d tw_brd[6];                                                                 \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                              \
                                                                                           \
        /* Step 6: 6-point cyclic convolution */                                          \
        __m512d v0, v1, v2, v3, v4, v5;                                                    \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                  \
                                    v0, v1, v2, v3, v4, v5);                               \
                                                                                           \
        /* Step 7: Assemble outputs (out_perm = [1,5,4,6,2,3]) */                         \
        __m512d y1, y2, y3, y4, y5, y6;                                                    \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                             \
                                   y0, y1, y2, y3, y4, y5, y6);                            \
                                                                                           \
        /* Step 8: Store results */                                                       \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);            \
    } while (0)

/**
 * @brief Complete AVX-512 radix-7 Rader butterfly (INVERSE, 4 butterflies)
 *
 * Identical structure to forward - Rader twiddles have inverse sign.
 */
#define RADIX7_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer) \
    do {                                                                                   \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);               \
                                                                                           \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw);              \
                                                                                           \
        __m512d y0;                                                                        \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                              \
                                                                                           \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                              \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);          \
                                                                                           \
        __m512d tw_brd[6];                                                                 \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);  /* Inverse twiddles */     \
                                                                                           \
        __m512d v0, v1, v2, v3, v4, v5;                                                    \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                  \
                                    v0, v1, v2, v3, v4, v5);                               \
                                                                                           \
        __m512d y1, y2, y3, y4, y5, y6;                                                    \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                             \
                                   y0, y1, y2, y3, y4, y5, y6);                            \
                                                                                           \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);            \
    } while (0)

//==============================================================================
// STREAMING VERSIONS
//==============================================================================

#define RADIX7_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer) \
    do {                                                                                           \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                        \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                       \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw);                      \
        __m512d y0;                                                                                \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                      \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                      \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                  \
        __m512d tw_brd[6];                                                                         \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                      \
        __m512d v0, v1, v2, v3, v4, v5;                                                            \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                          \
                                    v0, v1, v2, v3, v4, v5);                                       \
        __m512d y1, y2, y3, y4, y5, y6;                                                            \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                     \
                                   y0, y1, y2, y3, y4, y5, y6);                                    \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);             \
    } while (0)

#define RADIX7_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer) \
    do {                                                                                           \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                        \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                       \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw);                      \
        __m512d y0;                                                                                \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                      \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                      \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                  \
        __m512d tw_brd[6];                                                                         \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                      \
        __m512d v0, v1, v2, v3, v4, v5;                                                            \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                          \
                                    v0, v1, v2, v3, v4, v5);                                       \
        __m512d y1, y2, y3, y4, y5, y6;                                                            \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                     \
                                   y0, y1, y2, y3, y4, y5, y6);                                    \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);             \
    } while (0)

#endif // __AVX512F__

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