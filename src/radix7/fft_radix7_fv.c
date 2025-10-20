//==============================================================================
// fft_radix7_fv.c - Forward Radix-7 Rader Butterfly (OPTIMIZED)
//==============================================================================
//
// ARCHITECTURE:
// - Uses Rader's algorithm for prime-length DFT (N=7)
// - Separate from _bv for single source of truth on Rader twiddles
// - Direction encoded in function name, not runtime parameter
// - Shared implementation via macros in fft_radix7_macros.h
//
// OPTIMIZATIONS:
// - Hoisted Rader twiddle broadcasts outside hot loop
// - FMA (fused multiply-add) for complex arithmetic
// - Streaming stores for large transforms
// - Tuned prefetching distances
// - Separate code paths for streaming vs normal stores
//

#include "fft_radix7.h"
#include "simd_math.h"
#include "fft_radix7_macros.h"

/**
 * @brief Forward radix-7 butterfly using Rader's algorithm
 * 
 * Processes K butterflies, each transforming 7 complex inputs to 7 complex outputs.
 * Uses Rader's algorithm with generator g=3 for prime N=7.
 * 
 * Algorithm:
 * 1. Apply stage twiddles (if sub_len > 1)
 * 2. Compute DC component y0 = sum of all inputs
 * 3. Permute non-DC inputs: [1,3,2,6,4,5]
 * 4. Convolve with precomputed Rader twiddles (6-point cyclic convolution)
 * 5. Assemble outputs with inverse permutation: [1,5,4,6,2,3]
 * 
 * @param output_buffer Output array (7*K complex values, stride K)
 * @param sub_outputs   Input array (7*K complex values, stride K)
 * @param stage_tw      Stage twiddle factors (6*K complex values, or NULL if sub_len==1)
 * @param rader_tw      Rader twiddle factors (6 complex values, forward sign)
 * @param sub_len       Length of sub-transform (controls stage twiddle application)
 * 
 * @note All arrays must be 32-byte aligned for optimal performance
 * @note Rader twiddles have forward sign: exp(-2πij/N) for generator sequence
 */
void fft_radix7_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len)
{
    // Alignment hints for better codegen (data pre-aligned by planner)
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
    // Hoist Rader twiddle broadcast outside loop (~1-2% gain)
    __m512d tw_brd[6];
    BROADCAST_RADER_TWIDDLES_R7_AVX512(rader_tw, tw_brd);
    
    if (use_streaming) {
        // Streaming version for large transforms (K >= 8192)
        for (; k + 3 < K; k += 4) {
            // Prefetch data for upcoming iterations
            PREFETCH_7_LANES_R7_AVX512(k, K, PREFETCH_L1_R7_AVX512, sub_outputs, _MM_HINT_T0);
            
            // Prefetch stage twiddles (6 twiddles * 4 butterflies = 192 bytes)
            if (k + PREFETCH_TWIDDLE_R7_AVX512 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7_AVX512) * 6], _MM_HINT_T0);
            }
            
            // Process 4 butterflies with streaming stores
            RADIX7_PIPELINE_4_FV_AVX512_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
        _mm_sfence();  // Ensure all streaming stores complete
    } else {
        // Normal stores for small/medium transforms
        for (; k + 3 < K; k += 4) {
            PREFETCH_7_LANES_R7_AVX512(k, K, PREFETCH_L1_R7_AVX512, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7_AVX512 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7_AVX512) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_4_FV_AVX512_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
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
        // Streaming version
        for (; k + 1 < K; k += 2) {
            PREFETCH_7_LANES_R7_AVX2(k, K, PREFETCH_L1_R7, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_2_FV_AVX2_STREAM_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
        _mm_sfence();
    } else {
        // Normal stores
        for (; k + 1 < K; k += 2) {
            PREFETCH_7_LANES_R7_AVX2(k, K, PREFETCH_L1_R7, sub_outputs, _MM_HINT_T0);
            
            if (k + PREFETCH_TWIDDLE_R7 < K) {
                _mm_prefetch((const char *)&stage_tw[(k + PREFETCH_TWIDDLE_R7) * 6], _MM_HINT_T0);
            }
            
            RADIX7_PIPELINE_2_FV_AVX2_HOISTED(k, K, sub_outputs, stage_tw, tw_brd, output_buffer, sub_len);
        }
    }
#endif // __AVX2__
    
    //==========================================================================
    // SCALAR TAIL: Process remaining butterflies
    //==========================================================================
    for (; k < K; k++) {
        RADIX7_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, rader_tw, output_buffer, sub_len);
    }
}
