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

#ifndef FFT_RADIX7_MACROS_H
#define FFT_RADIX7_MACROS_H

#include "fft_radix7.h"
#include "simd_math.h"

//==============================================================================
// STREAMING THRESHOLD
//==============================================================================

#define STREAM_THRESHOLD 8192

//==============================================================================
// AVX-512 SUPPORT
//==============================================================================

#ifdef __AVX512F__


#define CMUL_ADD_AVX512(acc, a, w) \
    do { \
        __m512d ar = _mm512_shuffle_pd(a, a, 0x00); \
        __m512d ai = _mm512_shuffle_pd(a, a, 0xFF); \
        __m512d wr = _mm512_shuffle_pd(w, w, 0x00); \
        __m512d wi = _mm512_shuffle_pd(w, w, 0xFF); \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
        (acc) = _mm512_add_pd(acc, _mm512_unpacklo_pd(re, im)); \
    } while (0)

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512 (CORRECTED)
//==============================================================================

/**
 * @brief Optimized complex multiply for AVX-512: out = a * w (4 complex values)
 * 
 * CORRECTED: Uses shuffle_pd instead of unpacklo/hi
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
// RADER Y0 COMPUTATION - AVX-512
//==============================================================================

#define COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0)              \
    do {                                                                   \
        y0 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(x0, x1),           \
                                         _mm512_add_pd(x2, x3)),           \
                          _mm512_add_pd(_mm512_add_pd(x4, x5), x6));      \
    } while (0)

//==============================================================================
// RADER TWIDDLE BROADCAST - AVX-512 (UNROLLED)
//==============================================================================

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
// RADER CYCLIC CONVOLUTION - AVX-512 (FIXED)
//==============================================================================

/**
 * @brief 6-point cyclic convolution for AVX-512 (4 butterflies)
 * 
 * FIXED: Proper initialization and temporary variable usage
 */
#define RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                    v0, v1, v2, v3, v4, v5)               \
    do {                                                                   \
        __m512d tmp;                                                       \
        \
        /* q=0: indices [0,5,4,3,2,1] */                                   \
        CMUL_FMA_R7_AVX512(v0, tx0, tw_brd[0]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[5]);                           \
        v0 = _mm512_add_pd(v0, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[4]);                           \
        v0 = _mm512_add_pd(v0, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[3]);                           \
        v0 = _mm512_add_pd(v0, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[2]);                           \
        v0 = _mm512_add_pd(v0, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[1]);                           \
        v0 = _mm512_add_pd(v0, tmp);                                       \
        \
        /* q=1: indices [1,0,5,4,3,2] */                                   \
        CMUL_FMA_R7_AVX512(v1, tx0, tw_brd[1]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[0]);                           \
        v1 = _mm512_add_pd(v1, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[5]);                           \
        v1 = _mm512_add_pd(v1, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[4]);                           \
        v1 = _mm512_add_pd(v1, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[3]);                           \
        v1 = _mm512_add_pd(v1, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[2]);                           \
        v1 = _mm512_add_pd(v1, tmp);                                       \
        \
        /* q=2: indices [2,1,0,5,4,3] */                                   \
        CMUL_FMA_R7_AVX512(v2, tx0, tw_brd[2]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[1]);                           \
        v2 = _mm512_add_pd(v2, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[0]);                           \
        v2 = _mm512_add_pd(v2, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[5]);                           \
        v2 = _mm512_add_pd(v2, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[4]);                           \
        v2 = _mm512_add_pd(v2, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[3]);                           \
        v2 = _mm512_add_pd(v2, tmp);                                       \
        \
        /* q=3: indices [3,2,1,0,5,4] */                                   \
        CMUL_FMA_R7_AVX512(v3, tx0, tw_brd[3]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[2]);                           \
        v3 = _mm512_add_pd(v3, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[1]);                           \
        v3 = _mm512_add_pd(v3, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[0]);                           \
        v3 = _mm512_add_pd(v3, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[5]);                           \
        v3 = _mm512_add_pd(v3, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[4]);                           \
        v3 = _mm512_add_pd(v3, tmp);                                       \
        \
        /* q=4: indices [4,3,2,1,0,5] */                                   \
        CMUL_FMA_R7_AVX512(v4, tx0, tw_brd[4]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[3]);                           \
        v4 = _mm512_add_pd(v4, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[2]);                           \
        v4 = _mm512_add_pd(v4, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[1]);                           \
        v4 = _mm512_add_pd(v4, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[0]);                           \
        v4 = _mm512_add_pd(v4, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[5]);                           \
        v4 = _mm512_add_pd(v4, tmp);                                       \
        \
        /* q=5: indices [5,4,3,2,1,0] */                                   \
        CMUL_FMA_R7_AVX512(v5, tx0, tw_brd[5]);                            \
        CMUL_FMA_R7_AVX512(tmp, tx1, tw_brd[4]);                           \
        v5 = _mm512_add_pd(v5, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx2, tw_brd[3]);                           \
        v5 = _mm512_add_pd(v5, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx3, tw_brd[2]);                           \
        v5 = _mm512_add_pd(v5, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx4, tw_brd[1]);                           \
        v5 = _mm512_add_pd(v5, tmp);                                       \
        CMUL_FMA_R7_AVX512(tmp, tx5, tw_brd[0]);                           \
        v5 = _mm512_add_pd(v5, tmp);                                       \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512
//==============================================================================

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
// PREFETCHING - AVX-512 (UNROLLED)
//==============================================================================

#define PREFETCH_L1_R7_AVX512 16
#define PREFETCH_TWIDDLE_R7_AVX512 16

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
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

#define RADIX7_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                             \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                          \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                         \
                                                                                                     \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);               \
                                                                                                     \
        __m512d y0;                                                                                  \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                        \
                                                                                                     \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                        \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                    \
                                                                                                     \
        __m512d tw_brd[6];                                                                           \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                        \
                                                                                                     \
        __m512d v0, v1, v2, v3, v4, v5;                                                              \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                            \
                                    v0, v1, v2, v3, v4, v5);                                         \
                                                                                                     \
        __m512d y1, y2, y3, y4, y5, y6;                                                              \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                       \
                                   y0, y1, y2, y3, y4, y5, y6);                                      \
                                                                                                     \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

#define RADIX7_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                             \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                          \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                         \
                                                                                                     \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);               \
                                                                                                     \
        __m512d y0;                                                                                  \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                        \
                                                                                                     \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                        \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                    \
                                                                                                     \
        __m512d tw_brd[6];                                                                           \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                        \
                                                                                                     \
        __m512d v0, v1, v2, v3, v4, v5;                                                              \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                            \
                                    v0, v1, v2, v3, v4, v5);                                         \
                                                                                                     \
        __m512d y1, y2, y3, y4, y5, y6;                                                              \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                       \
                                   y0, y1, y2, y3, y4, y5, y6);                                      \
                                                                                                     \
        STORE_7_LANES_AVX512(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

#define RADIX7_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                                    \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                 \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                                \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                      \
        __m512d y0;                                                                                         \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                               \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                               \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                           \
        __m512d tw_brd[6];                                                                                  \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                               \
        __m512d v0, v1, v2, v3, v4, v5;                                                                     \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                   \
                                    v0, v1, v2, v3, v4, v5);                                                \
        __m512d y1, y2, y3, y4, y5, y6;                                                                     \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                              \
                                   y0, y1, y2, y3, y4, y5, y6);                                             \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

#define RADIX7_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do {                                                                                                    \
        __m512d x0, x1, x2, x3, x4, x5, x6;                                                                 \
        LOAD_7_LANES_AVX512(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);                                \
        APPLY_STAGE_TWIDDLES_R7_AVX512(kk, x1, x2, x3, x4, x5, x6, stage_tw, sub_len);                      \
        __m512d y0;                                                                                         \
        COMPUTE_Y0_R7_AVX512(x0, x1, x2, x3, x4, x5, x6, y0);                                               \
        __m512d tx0, tx1, tx2, tx3, tx4, tx5;                                                               \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);                           \
        __m512d tw_brd[6];                                                                                  \
        BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);                                               \
        __m512d v0, v1, v2, v3, v4, v5;                                                                     \
        RADER_CONVOLUTION_R7_AVX512(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,                                   \
                                    v0, v1, v2, v3, v4, v5);                                                \
        __m512d y1, y2, y3, y4, y5, y6;                                                                     \
        ASSEMBLE_OUTPUTS_R7_AVX512(x0, v0, v1, v2, v3, v4, v5,                                              \
                                   y0, y1, y2, y3, y4, y5, y6);                                             \
        STORE_7_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);                      \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// AVX2 SUPPORT
//==============================================================================

#ifdef __AVX2__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX2 with FMA
//==============================================================================

#if defined(__AVX2__)
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
// Y0 COMPUTATION - AVX2
//==============================================================================

#define COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0) \
    do { \
        y0 = _mm256_add_pd( \
            _mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)), \
            _mm256_add_pd(_mm256_add_pd(x4, x5), x6)); \
    } while (0)

//==============================================================================
// APPLY STAGE TWIDDLES - AVX2
//==============================================================================

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
            x1 = cmul_avx2_aos(x1, w1); \
            x2 = cmul_avx2_aos(x2, w2); \
            x3 = cmul_avx2_aos(x3, w3); \
            x4 = cmul_avx2_aos(x4, w4); \
            x5 = cmul_avx2_aos(x5, w5); \
            x6 = cmul_avx2_aos(x6, w6); \
        } \
    } while (0)

//==============================================================================
// BROADCAST RADER TWIDDLES - AVX2 (UNROLLED)
//==============================================================================

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
// RADER CYCLIC CONVOLUTION - AVX2 (FIXED)
//==============================================================================

#define RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5) \
    do { \
        __m256d tmp; \
        \
        /* q=0 */ \
        v0 = cmul_avx2_aos(tx0, tw_brd[0]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[5]); \
        v0 = _mm256_add_pd(v0, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[4]); \
        v0 = _mm256_add_pd(v0, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[3]); \
        v0 = _mm256_add_pd(v0, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[2]); \
        v0 = _mm256_add_pd(v0, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[1]); \
        v0 = _mm256_add_pd(v0, tmp); \
        \
        /* q=1 */ \
        v1 = cmul_avx2_aos(tx0, tw_brd[1]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[0]); \
        v1 = _mm256_add_pd(v1, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[5]); \
        v1 = _mm256_add_pd(v1, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[4]); \
        v1 = _mm256_add_pd(v1, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[3]); \
        v1 = _mm256_add_pd(v1, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[2]); \
        v1 = _mm256_add_pd(v1, tmp); \
        \
        /* q=2 */ \
        v2 = cmul_avx2_aos(tx0, tw_brd[2]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[1]); \
        v2 = _mm256_add_pd(v2, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[0]); \
        v2 = _mm256_add_pd(v2, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[5]); \
        v2 = _mm256_add_pd(v2, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[4]); \
        v2 = _mm256_add_pd(v2, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[3]); \
        v2 = _mm256_add_pd(v2, tmp); \
        \
        /* q=3 */ \
        v3 = cmul_avx2_aos(tx0, tw_brd[3]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[2]); \
        v3 = _mm256_add_pd(v3, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[1]); \
        v3 = _mm256_add_pd(v3, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[0]); \
        v3 = _mm256_add_pd(v3, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[5]); \
        v3 = _mm256_add_pd(v3, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[4]); \
        v3 = _mm256_add_pd(v3, tmp); \
        \
        /* q=4 */ \
        v4 = cmul_avx2_aos(tx0, tw_brd[4]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[3]); \
        v4 = _mm256_add_pd(v4, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[2]); \
        v4 = _mm256_add_pd(v4, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[1]); \
        v4 = _mm256_add_pd(v4, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[0]); \
        v4 = _mm256_add_pd(v4, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[5]); \
        v4 = _mm256_add_pd(v4, tmp); \
        \
        /* q=5 */ \
        v5 = cmul_avx2_aos(tx0, tw_brd[5]); \
        tmp = cmul_avx2_aos(tx1, tw_brd[4]); \
        v5 = _mm256_add_pd(v5, tmp); \
        tmp = cmul_avx2_aos(tx2, tw_brd[3]); \
        v5 = _mm256_add_pd(v5, tmp); \
        tmp = cmul_avx2_aos(tx3, tw_brd[2]); \
        v5 = _mm256_add_pd(v5, tmp); \
        tmp = cmul_avx2_aos(tx4, tw_brd[1]); \
        v5 = _mm256_add_pd(v5, tmp); \
        tmp = cmul_avx2_aos(tx5, tw_brd[0]); \
        v5 = _mm256_add_pd(v5, tmp); \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX2
//==============================================================================

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
// PREFETCHING - AVX2 (UNROLLED)
//==============================================================================

#define PREFETCH_L1_R7 8
#define PREFETCH_TWIDDLE_R7 8

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
// COMPLETE BUTTERFLY PIPELINE - AVX2
//==============================================================================

#define RADIX7_PIPELINE_2_FV_AVX2(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        \
        __m256d tw_brd[6]; \
        BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd); \
        \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_PIPELINE_2_BV_AVX2(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
    do { \
        __m256d x0, x1, x2, x3, x4, x5, x6; \
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6); \
        \
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw, sub_len); \
        \
        __m256d y0; \
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0); \
        \
        __m256d tx0, tx1, tx2, tx3, tx4, tx5; \
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5); \
        \
        __m256d tw_brd[6]; \
        BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd); \
        \
        __m256d v0, v1, v2, v3, v4, v5; \
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd, \
                                   v0, v1, v2, v3, v4, v5); \
        \
        __m256d y1, y2, y3, y4, y5, y6; \
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5, \
                                  y0, y1, y2, y3, y4, y5, y6); \
        \
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_PIPELINE_2_FV_AVX2_STREAM(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
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
        STORE_7_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#define RADIX7_PIPELINE_2_BV_AVX2_STREAM(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len) \
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
        STORE_7_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6); \
    } while (0)

#endif // __AVX2__

//==============================================================================
// SCALAR SUPPORT
//==============================================================================

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
 * Identical to forward - only rader_tw sign differs (precomputed by manager).
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
