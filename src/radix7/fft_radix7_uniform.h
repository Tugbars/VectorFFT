#ifndef FFT_RADIX7_H
#define FFT_RADIX7_H

#include "highspeedFFT.h"

/**
 * @brief Radix-7 FFT butterfly using Rader's Algorithm for prime-length DFTs
 * 
 * =============================================================================
 * WHY RADIX-7 IS SPECIAL: RADER'S ALGORITHM
 * =============================================================================
 * 
 * Unlike radix-2, radix-4, and radix-8 (which are powers of 2), radix-7 is a 
 * PRIME number. Prime-length FFTs cannot be decomposed using traditional 
 * Cooley-Tukey factorization. Instead, we use **Rader's Algorithm**, which 
 * converts a prime-length DFT into a cyclic convolution.
 * 
 * MATHEMATICAL FOUNDATION:
 * -----------------------
 * For a prime N (here N=7), Rader's algorithm works as follows:
 * 
 * 1. SEPARATE DC COMPONENT: 
 *    Y[0] = X[0] + X[1] + X[2] + ... + X[6]  (simple sum)
 * 
 * 2. FIND PRIMITIVE ROOT (generator):
 *    For N=7, the primitive root g=3. This means powers of 3 modulo 7 
 *    generate all non-zero elements: {3^0, 3^1, 3^2, ...} mod 7 = {1,3,2,6,4,5}
 * 
 * 3. PERMUTE INPUTS using generator powers:
 *    Instead of processing X[1..6] in order, we reorder them according to 
 *    the generator sequence. This converts the DFT into a cyclic convolution.
 *    
 *    Input permutation:  [X[1], X[3], X[2], X[6], X[4], X[5]]
 * 
 * 4. COMPUTE CYCLIC CONVOLUTION:
 *    The 6 non-DC outputs are computed as:
 *    Y[m] = X[0] + Σ(permuted_inputs[l] * twiddle[(q-l) mod 6])
 *    
 *    where the convolution twiddles are precomputed exponentials based on
 *    the OUTPUT permutation.
 * 
 * 5. OUTPUT PERMUTATION:
 *    Results are placed according to output permutation [1,5,4,6,2,3]:
 *    - Convolution result 0 → Y[1]
 *    - Convolution result 1 → Y[5]
 *    - Convolution result 2 → Y[4]
 *    - Convolution result 3 → Y[6]
 *    - Convolution result 4 → Y[2]
 *    - Convolution result 5 → Y[3]
 * 
 * WHY THIS IS COMPLEX:
 * --------------------
 * - **No symmetry exploits**: Unlike radix-2's simple butterfly pattern, 
 *   Rader requires 6 full complex multiplications per output (36 total)
 * 
 * - **Dual permutations**: Both input and output require specific reordering
 *   based on number theory (primitive roots modulo N)
 * 
 * - **Cyclic convolution**: Each output depends on ALL 6 non-DC inputs with
 *   different rotation of twiddle factors - this is computationally intensive
 * 
 * - **Twiddle factor complexity**: We need 6 convolution twiddles (computed
 *   from output permutation) PLUS per-stage DIT twiddles (6 per butterfly)
 *   for multi-stage FFTs
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-7 uses TWO types of twiddle factors:
 * 
 * 1. **CONVOLUTION TWIDDLES** (Rader-specific, constant per transform):
 *    - 6 complex values: exp(-2πi * out_perm[q] / 7) for q=0..5
 *    - Output permutation: [1,5,4,6,2,3]
 *    - These implement the cyclic convolution kernel
 *    - Computed once at function start, reused for all butterflies
 *    - Stored in tw_brd[6] for SIMD broadcasting
 * 
 * 2. **PER-STAGE DIT TWIDDLES** (standard FFT twiddles):
 *    - 6 twiddles per butterfly: W^(k*m) for m=1..6
 *    - Applied BEFORE the Rader convolution
 *    - Layout: stage_tw[6*k + 0..5] contains twiddles for butterfly k
 *    - Only used when sub_len > 1 (multi-stage FFT)
 * 
 * COMPARISON TO RADIX-2:
 * ----------------------
 * Radix-2: stage_tw[k] = single twiddle W^k
 * Radix-7: stage_tw[6*k + m] = six twiddles W^(k*m) for m=1..6
 * 
 * This 6× storage requirement is unavoidable for prime radices.
 * 
 * =============================================================================
 * ALGORITHM OVERVIEW
 * =============================================================================
 * 
 * For each group of 7 inputs (one butterfly):
 * 
 * STEP 1: Load X[0..6] from sub_outputs (stride = sub_len)
 * 
 * STEP 2: Apply per-stage DIT twiddles (if multi-stage):
 *         X[m] *= stage_tw[6*k + (m-1)] for m=1..6
 * 
 * STEP 3: Compute DC output:
 *         Y[0] = X[0] + X[1] + ... + X[6]
 * 
 * STEP 4: Permute inputs for Rader:
 *         TX = [X[1], X[3], X[2], X[6], X[4], X[5]]
 * 
 * STEP 5: Compute 6-point cyclic convolution:
 *         For q=0..5:
 *           V[q] = Σ(l=0..5) TX[l] * convolution_twiddle[(q-l) mod 6]
 * 
 * STEP 6: Apply output permutation:
 *         Y[1] = X[0] + V[0]
 *         Y[5] = X[0] + V[1]
 *         Y[4] = X[0] + V[2]
 *         Y[6] = X[0] + V[3]
 *         Y[2] = X[0] + V[4]
 *         Y[3] = X[0] + V[5]
 * 
 * STEP 7: Store Y[0..6] to output_buffer
 * 
 * =============================================================================
 * OPTIMIZATION STRATEGY
 * =============================================================================
 * 
 * - **AVX2 8× unrolling**: Process 8 butterflies simultaneously (56 outputs)
 *   Each convolution position computed with parallel complex multiplies
 * 
 * - **Convolution twiddle broadcasting**: 6 convolution twiddles stored in
 *   __m256d registers, duplicated for AoS layout, eliminating reload overhead
 * 
 * - **Prefetching**: 7 strided loads per butterfly benefits from aggressive
 *   prefetching (16 butterflies ahead for AVX2)
 * 
 * - **Pure AoS layout**: All operations stay in Array-of-Structures format
 *   matching fft_data layout, avoiding shuffle overhead
 * 
 * PERFORMANCE CHARACTERISTICS:
 * ---------------------------
 * - Computational cost: 36 complex multiplies + 30 additions per butterfly
 *   (vs 1 multiply + 2 additions for radix-2)
 * 
 * - Memory bandwidth: 7 strided loads + 7 strided stores per butterfly
 *   (vs 2 loads + 2 stores for radix-2)
 * 
 * - SIMD efficiency: Lower than radix-2/4/8 due to complex dependency chains
 *   in cyclic convolution, but 8× unrolling provides significant speedup
 * 
 * @param output_buffer[out] Destination buffer for butterfly results (size: sub_len * 7)
 * @param sub_outputs[in]    Input array with 7 strided sub-transforms
 *                            Layout: [X[k+0*sub_len], ..., X[k+6*sub_len]]
 * @param stage_tw[in]       Per-stage DIT twiddle factors
 *                            Layout: stage_tw[6*k + 0..5] = W^(k*1), ..., W^(k*6)
 *                            Used only when sub_len > 1 (multi-stage)
 * @param sub_len            Size of each sub-transform (total size = 7 * sub_len)
 * @param transform_sign     +1 for inverse FFT, -1 for forward FFT
 *                            Affects convolution twiddle sign: forward uses -2π/7
 * 
 * @note CRITICAL DIFFERENCES FROM RADIX-2, 4, 8 etc:
 *       - Requires 6 twiddles per butterfly (not 1)
 *       - Uses Rader's algorithm with dual permutations
 *       - No special cases for k=0 or k=N/4 (all require full convolution)
 *       - Convolution twiddles recomputed based on transform_sign
 * 
 * @warning This is the most computationally expensive radix in the mixed-radix
 *          FFT due to Rader's algorithm. Used only when N contains factor 7.
 */
void fft_radix7_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len);

void fft_radix7_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len);

#endif // FFT_RADIX3_H

// 1400. 
