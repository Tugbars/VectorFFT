//==============================================================================
// fft_radix7_bv.c - Inverse Radix-7 Rader Butterfly (OPTIMIZED)
//==============================================================================
//
// ARCHITECTURE:
// - Uses Rader's algorithm for prime-length DFT (N=7)
// - Separate from _fv for single source of truth on Rader twiddles
// - Direction encoded in function name, not runtime parameter
// - Shared implementation via macros in fft_radix7_macros.h
//
// ONLY DIFFERENCE FROM FORWARD: Rader twiddles have inverse sign
//   Forward:  exp(-2πij/N) for generator sequence
//   Inverse:  exp(+2πij/N) for generator sequence
//

#include "fft_radix7_uniform.h"
#include "simd_math.h"
#include "fft_radix7_macros.h"

/**
 * @brief Inverse radix-7 butterfly using Rader's algorithm
 * 
 * Identical algorithm to forward transform, but uses inverse-sign Rader twiddles.
 * The twiddle manager precomputes twiddles with correct sign.
 * 
 * @param output_buffer Output array (7*K complex values, stride K)
 * @param sub_outputs   Input array (7*K complex values, stride K)
 * @param stage_tw      Stage twiddle factors (6*K complex values, inverse sign)
 * @param rader_tw      Rader twiddle factors (6 complex values, INVERSE sign)
 * @param sub_len       Length of sub-transform
 * 
 * @note rader_tw must have inverse sign: exp(+2πij/N) for generator sequence
 * @note All other aspects identical to forward transform
 */
void fft_radix7_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len)
{
    // Alignment hints (data pre-aligned by planner)
    output_buffer = __builtin_assume_aligned(output_buffer, 32);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 32);
    stage_tw = __builtin_assume_aligned(stage_tw, 32);
    
    const int K = sub_len;
    int k = 0;
    
    //==========================================================================
    // STREAMING DECISION (outside loop - zero runtime overhead)
    //==========================================================================
    const int use_streaming = (K >= STREAM_THRESHOLD);
    
    //==========================================================================
    // AVX-512 PATH: 4 butterflies per iteration
    //==========================================================================
#ifdef __AVX512F__
    // Hoist Rader twiddle broadcast outside loop
    __m512d tw_brd[6];
    BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);
    
    if (use_streaming) {
        for (; k + 3 < K; k += 4) {
            PREFETCH_7_LANES_R7_AVX512(k, K, PREFETCH_L1_R7_AVX512, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7_AVX512 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7_AVX512) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_4_BV_AVX512_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
        _mm_sfence();
    } else {
        for (; k + 3 < K; k += 4) {
            PREFETCH_7_LANES_R7_AVX512(k, K, PREFETCH_L1_R7_AVX512, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7_AVX512 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7_AVX512) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_4_BV_AVX512_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
    }
#endif // __AVX512F__
    
    //==========================================================================
    // AVX2 PATH: 2 butterflies per iteration
    //==========================================================================
#ifdef __AVX2__
    // Hoist Rader twiddle broadcast for AVX2
    __m256d tw_brd[6];
    BROADCAST_RADER_TWIDDLES_R7_AVX2(rader_tw, tw_brd);
    
    if (use_streaming) {
        for (; k + 1 < K; k += 2) {
            PREFETCH_7_LANES_R7_AVX2(k, K, PREFETCH_L1_R7, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_2_BV_AVX2_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
        _mm_sfence();
    } else {
        for (; k + 1 < K; k += 2) {
            PREFETCH_7_LANES_R7_AVX2(k, K, PREFETCH_L1_R7, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_2_BV_AVX2_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
    }
#endif // __AVX2__
    
    //==========================================================================
    // SCALAR TAIL: Process remaining butterflies
    //==========================================================================
    for (; k < K; k++) {
        RADIX7_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len);
    }
}

//==============================================================================
// FORWARD vs INVERSE COMPARISON
//==============================================================================

/**
 * CODE COMPARISON:
 * 
 * IDENTICAL (100%):
 * ✅ All loop structures
 * ✅ All prefetching logic
 * ✅ All load/store patterns
 * ✅ Rader's algorithm structure
 * ✅ Convolution computation
 * ✅ Input permutations
 * ✅ Output assembly
 * 
 * DIFFERENT (handled by Rader Manager):
 * ⚠️  rader_tw sign only:
 *     Forward:  W = exp(-2πij/N) where j ∈ generator_sequence
 *     Inverse:  W = exp(+2πij/N) where j ∈ generator_sequence
 * 