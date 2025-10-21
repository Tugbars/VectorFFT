#ifndef FFT_RADIX2_H
#define FFT_RADIX2_H

#include "highspeedFFT.h"
#include "../fft_plan/fft_planning_types.h"


/**
 * @brief Optimized radix-2 FFT butterfly operation with SIMD acceleration
 * 
 * Performs the core butterfly computation for radix-2 FFT algorithm:
 *   Y[k] = X_even[k] + W^k * X_odd[k]
 *   Y[k + N/2] = X_even[k] - W^k * X_odd[k]
 * 
 * This implementation includes several key optimizations:
 * 1. Special-case handling for k=0 (no twiddle factor needed)
 * 2. Special-case handling for k=N/4 (90° rotation, no complex multiply)
 * 3. Multi-tier SIMD acceleration (AVX-512, AVX2, SSE2)
 * 4. Prefetching for cache optimization
 * 5. Loop unrolling for better instruction-level parallelism
 * 
 * @param output_buffer[out] Destination buffer for butterfly results (size: sub_len * 2)
 * @param sub_outputs[in]    Input array containing even indices [0..sub_len-1] 
 *                            and odd indices [sub_len..2*sub_len-1]
 * @param stage_tw[in]       Precomputed twiddle factors W^k for k=0..sub_len-1
 *                            where W = exp(-2πi/N) for forward transform
 * @param sub_len            Half the transform size (N/2), must be power of 2
 * @param transform_sign     +1 for inverse FFT (IFFT), -1 for forward FFT
 * 
 * @note The function assumes:
 *       - All buffers are properly aligned for SIMD operations
 *       - sub_len is a power of 2
 *       - output_buffer has space for at least 2*sub_len elements
 * 
 * @performance The function automatically selects the best SIMD path available:
 *              - AVX-512: Processes 16 complex pairs per iteration
 *              - AVX2:    Processes 8 complex pairs per iteration  
 *              - SSE2:    Processes 1 complex pair per iteration (fallback)
 */
void fft_radix2_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw,  
    int sub_len);

void fft_radix2_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw,  
    int sub_len);

#endif // FFT_RADIX2_H


// 1000.
