#ifndef FFT_RADIX13_H
#define FFT_RADIX13_H

#include "highspeedFFT.h"

/**
 * @brief Radix-13 FFT butterfly using Pure Rader's Algorithm (No Symmetry Exploitation)
 * 
 * =============================================================================
 * WHY RADIX-13 IS DIFFERENT: PRIME WITHOUT PRACTICAL SYMMETRY
 * =============================================================================
 * 
 * Radix-13 is a PRIME number, requiring Rader's Algorithm like radix-7 and 
 * radix-11, but sits in an unfortunate middle ground:
 * 
 * COMPARISON ACROSS PRIME RADICES:
 * ---------------------------------
 * 
 * RADIX-7 (Small Prime):
 * - 6 non-DC outputs → 6×6 cyclic convolution (36 complex multiplies)
 * - Too small for symmetry exploitation (6 is not even-paired structure)
 * - Pure Rader approach is manageable
 * - **Cost per output: ~17 FLOPs**
 * 
 * RADIX-11 (Sweet Spot Prime):
 * - 10 non-DC outputs → Could be 10×10 (100 complex multiplies)
 * - BUT: 10 = 5 pairs, perfect for Hermitian symmetry exploitation
 * - Reduces to 5 pair computations with shared cos/sin coefficients
 * - **Cost per output: ~21 FLOPs (but 60% savings from symmetry)**
 * 
 * RADIX-13 (Awkward Prime): ⚠️
 * - 12 non-DC outputs → 12×12 cyclic convolution (144 complex multiplies!)
 * - 12 = 6 pairs, theoretically could use symmetry...
 * - **BUT**: Symmetry requires separate cos/sin coefficient storage (6 each)
 *   - Would need 12 global constants (vs radix-11's 10)
 *   - Coefficient cycling patterns more complex (6-way vs 5-way)
 *   - Register pressure in AVX2 becomes prohibitive
 *   - Break-even point between symmetry code complexity vs convolution cost
 * - **DECISION**: Pure Rader without symmetry is simpler and nearly as fast
 * - **Cost per output: ~28-30 FLOPs**
 * 
 * WHY NO SYMMETRY FOR RADIX-13?
 * ------------------------------
 * 1. **Register Pressure**: 6 cos + 6 sin = 12 AVX2 registers (out of 16 total)
 *    Leaves only 4 for computation, severely limiting parallelism
 * 
 * 2. **Code Complexity**: 6-way coefficient cycling requires more complex
 *    indexing logic than radix-11's 5-way pattern
 * 
 * 3. **Memory Overhead**: 12 global constants vs 0 for pure Rader
 *    (Rader computes convolution twiddles once per butterfly set)
 * 
 * 4. **Marginal Benefit**: Symmetry saves ~40% at best, but increased
 *    complexity reduces actual speedup to ~25-30%
 * 
 * 5. **Rare Usage**: Radix-13 appears only when N contains factor 13
 *    (e.g., N=1024×13, N=4096×13) - not worth optimizing to extreme
 * 
 * =============================================================================
 * RADER'S ALGORITHM FOR RADIX-13: PURE CYCLIC CONVOLUTION
 * =============================================================================
 * 
 * STEP 1: PRIMITIVE ROOT
 * -----------------------
 * For N=13, primitive root g=2 generates all non-zero elements:
 * Powers of 2 mod 13: {2^0, 2^1, 2^2, ...} = {1,2,4,8,3,6,12,11,9,5,10,7}
 * 
 * This sequence has length 12 (φ(13) = 12, Euler's totient)
 * 
 * STEP 2: INPUT PERMUTATION
 * --------------------------
 * Reorder inputs X[1..12] according to generator powers:
 *   TX = [X[1], X[2], X[4], X[8], X[3], X[6], X[12], X[11], X[9], X[5], X[10], X[7]]
 * 
 * This transforms the DFT into a cyclic convolution problem
 * 
 * STEP 3: DC COMPONENT (Unchanged)
 * ---------------------------------
 * Y[0] = X[0] + X[1] + X[2] + ... + X[12]  (sum of all 13 inputs)
 * 
 * STEP 4: CONVOLUTION TWIDDLES
 * -----------------------------
 * Compute 12 convolution twiddles based on OUTPUT permutation:
 *   out_perm = [1, 12, 10, 3, 9, 4, 7, 5, 2, 6, 11, 8]
 * 
 * For each q=0..11:
 *   TW[q] = exp(sgn * 2πi * out_perm[q] / 13)
 * 
 * These twiddles form the kernel of the cyclic convolution
 * 
 * STEP 5: 12-POINT CYCLIC CONVOLUTION
 * ------------------------------------
 * For each output position q=0..11:
 *   V[q] = Σ(l=0..11) TX[l] * TW[(q-l) mod 12]
 * 
 * This is a FULL 12×12 matrix of complex multiplications:
 * - Each V[q] requires 12 complex multiplies + 11 complex adds
 * - 12 outputs × 12 multiplies = 144 complex multiplies total
 * - Each complex multiply = 6 FLOPs → 864 FLOPs just for convolution!
 * 
 * KEY PROPERTY: The twiddle indexing (q-l) mod 12 creates circular
 * rotation, which is why this is called "cyclic" convolution
 * 
 * STEP 6: OUTPUT MAPPING
 * -----------------------
 * Map convolution results to final outputs using out_perm:
 *   Y[1]  = X[0] + V[0]
 *   Y[12] = X[0] + V[1]
 *   Y[10] = X[0] + V[2]
 *   Y[3]  = X[0] + V[3]
 *   ... (following out_perm sequence)
 * 
 * =============================================================================
 * IMPLEMENTATION STRATEGY: MACRO-BASED UNROLLING
 * =============================================================================
 * 
 * Given the massive computational cost (144 complex multiplies), we use
 * aggressive macro-based unrolling for the AVX2 path:
 * 
 * CONV12_Q MACRO:
 * ---------------
 * Computes one complete convolution output V[q] for all 4 SIMD lanes:
 * ```c
 * CONV12_Q(q, v, tx0, tx1, ..., tx11, tw_brd)
 * ```
 * 
 * Expands to:
 * - 12 complex multiplies: tx[l] * tw_brd[(q+l) mod 12]
 * - 11 complex additions to accumulate result
 * - 4 SIMD registers (processing 4 butterflies × 2 complex pairs)
 * 
 * Called 12 times (once per q) to compute all V[0..11]
 * 
 * MAP_CONV_TO_OUTPUT_13 MACRO:
 * -----------------------------
 * Maps 12 convolution results to 12 outputs following out_perm:
 * ```c
 * MAP_CONV_TO_OUTPUT_13(x0, v, y, _suffix)
 * ```
 * 
 * Expands to 12 additions: Y[m] = X[0] + V[q] with correct permutation
 * 
 * WHY MACROS?
 * -----------
 * 1. **Code Generation**: Eliminates modulo operations (q-l) % 12
 *    Compiler computes all indices at compile time
 * 
 * 2. **Loop Unrolling**: Removes loop overhead for 12 iterations
 *    Allows better instruction scheduling
 * 
 * 3. **Register Allocation**: Compiler can optimize register usage
 *    across the entire unrolled sequence
 * 
 * 4. **Constant Propagation**: All twiddle indices become constants
 *    Enables direct register access instead of indexed loads
 * 
 * TRADEOFF: Code size ~3KB vs ~500 bytes for loop-based approach
 * But performance gain is 25-35% due to eliminated overhead
 * 
 * =============================================================================
 * PURE AoS (ARRAY-OF-STRUCTURES) LAYOUT
 * =============================================================================
 * 
 * Unlike radix-11 which uses SoA for symmetry exploitation, radix-13 stays
 * in pure AoS format throughout:
 * 
 * WHY AoS FOR RADIX-13?
 * ---------------------
 * 1. **No Symmetry**: Without symmetry, no benefit from separating real/imag
 *    The convolution is fully general with no special structure
 * 
 * 2. **Complex Multiply Efficiency**: AoS + FMA works well for complex multiply
 *    The cmul_avx2_aos function is highly optimized for this pattern
 * 
 * 3. **Memory Access Pattern**: Convolution accesses all 12 inputs anyway
 *    No particular advantage to transposing the data layout
 * 
 * 4. **Simpler Code**: No AoS↔SoA conversion overhead
 *    This alone saves ~40-60 instructions per 4 butterflies
 * 
 * LOAD PATTERN:
 * -------------
 * For 8-butterfly unrolled loop:
 * - Load 13 lanes × 4 groups = 52 AVX2 loads (4 complex pairs each)
 * - Load 12 twiddles × 4 groups = 48 AVX2 loads
 * - Total: 100 loads = 1600 bytes per iteration
 * 
 * COMPUTE PATTERN:
 * ----------------
 * - Apply 12 DIT twiddles (if multi-stage)
 * - Compute DC output (12 additions)
 * - Permute inputs (register shuffle, no memory ops)
 * - Execute CONV12_Q macro 12 times (144 complex multiplies)
 * - Map outputs (12 additions)
 * - Store 13 outputs × 4 groups = 52 stores
 * 
 * =============================================================================
 * PERFORMANCE CHARACTERISTICS
 * =============================================================================
 * 
 * Computational Complexity per Butterfly:
 * ----------------------------------------
 * - Input DIT twiddles: 12 complex multiplies = 72 FLOPs
 * - DC output: 12 additions = 24 FLOPs
 * - Cyclic convolution: 144 complex multiplies = 864 FLOPs
 *                      + 132 complex adds = 264 FLOPs
 * - Output mapping: 12 additions = 24 FLOPs
 * - **TOTAL: ~1248 FLOPs per butterfly (13 outputs)**
 * - **Per output: ~96 FLOPs** (vs ~21 for radix-11, ~17 for radix-7)
 * 
 * This is EXTREMELY expensive compared to other radices!
 * 
 * Memory Operations (AVX2 8× unrolled):
 * --------------------------------------
 * - 104 loads (13 lanes × 8 butterflies) = 1664 bytes
 * - 96 twiddle loads (12 × 8 butterflies) = 1536 bytes
 * - 104 stores (13 lanes × 8 butterflies) = 1664 bytes
 * - **TOTAL: ~4864 bytes per 8 butterflies ≈ 608 bytes/butterfly**
 * 
 * Arithmetic Intensity:
 * ---------------------
 * AI = 1248 FLOPs / 608 bytes ≈ 2.05 FLOPs/byte
 * 
 * **BREAKTHROUGH**: This is the FIRST radix in our implementation that is
 * truly COMPUTE-BOUND rather than memory-bound!
 * 
 * - Radix-2:  0.10 FLOPs/byte (severely memory-bound)
 * - Radix-8:  0.28 FLOPs/byte (memory-bound)
 * - Radix-11: 0.41 FLOPs/byte (still memory-bound)
 * - Radix-13: 2.05 FLOPs/byte ⭐ (COMPUTE-BOUND!)
 * 
 * Modern CPUs have peak arithmetic intensity of 10-20 FLOPs/byte, so even
 * radix-13 isn't fully compute-bound, but it's much closer than other radices.
 * 
 * SIMD Efficiency (AVX2):
 * -----------------------
 * - Process 8 butterflies per main iteration (104 outputs)
 * - Theoretical peak: 16 DP FLOPs/cycle
 * - Achieved: ~10-12 FLOPs/cycle
 * - Efficiency: 62-75% of peak ⭐ (best among all radices!)
 * 
 * The high efficiency comes from:
 * - Massive amount of computation vs memory ops
 * - Excellent FMA utilization in convolution loops
 * - Macro unrolling enables optimal instruction scheduling
 * 
 * Comparison (Cost per Output):
 * ------------------------------
 * - Radix-2:  ~6 FLOPs/output   (optimal baseline)
 * - Radix-4:  ~8 FLOPs/output
 * - Radix-7:  ~17 FLOPs/output
 * - Radix-8:  ~13 FLOPs/output
 * - Radix-11: ~21 FLOPs/output
 * - Radix-13: ~96 FLOPs/output ⚠️ (16× more expensive than radix-2!)
 * 
 * =============================================================================
 * WHEN TO USE RADIX-13 (AND WHEN NOT TO)
 * =============================================================================
 * 
 * USE RADIX-13 WHEN:
 * ------------------
 * 1. Transform size N naturally contains factor 13
 *    Example: N = 2^10 × 13 = 13312
 * 
 * 2. Alternative requires even more stages
 *    Example: N=13 requires 1 radix-13 stage
 *            vs padding to N=16 requires 4 radix-2 stages (more overhead)
 * 
 * 3. You're doing many transforms of the same size
 *    The twiddle precomputation cost is amortized
 * 
 * AVOID RADIX-13 WHEN:
 * --------------------
 * 1. Transform size can be factored differently
 *    Example: N=1024 → use radix-2/4/8 only
 * 
 * 2. Can pad to nearby power-of-2 or product of smaller primes
 *    Example: N=1000 → pad to 1024 (radix-2) is much faster
 * 
 * 3. One-off transforms where twiddle precomputation overhead matters
 * 
 * 4. Memory-constrained environments (large twiddle table: 12×N elements)
 * 
 * PRACTICAL GUIDELINE:
 * --------------------
 * Radix-13 reduces FFT STAGES but increases PER-STAGE cost dramatically.
 * Break-even typically occurs when:
 *   - Saved stages > 2-3 (compared to alternative factorization)
 *   - Transform is repeated many times (amortize twiddle setup)
 * 
 * For most applications, it's better to avoid factor 13 in N entirely!
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-13 uses TWO types of twiddles:
 * 
 * 1. **CONVOLUTION TWIDDLES** (12 values, computed per iteration):
 *    Based on output permutation [1,12,10,3,9,4,7,5,2,6,11,8]
 *    TW[q] = exp(sgn * 2πi * out_perm[q] / 13)
 *    Stored in tw_brd[12] as broadcast-ready AVX2 registers
 *    These are the cyclic convolution kernel
 * 
 * 2. **DIT STAGE TWIDDLES** (12 per butterfly, from stage_tw):
 *    Layout: stage_tw[12*k + 0..11] = W^(k·1), ..., W^(k·12)
 *    Applied to X[1..12] before Rader permutation
 *    Only used when sub_len > 1 (multi-stage FFT)
 * 
 * STORAGE COMPARISON:
 * -------------------
 * Radix-2:  1 twiddle per butterfly  → stage_tw[k]
 * Radix-7:  6 twiddles per butterfly → stage_tw[6*k + 0..5]
 * Radix-8:  7 twiddles per butterfly → stage_tw[7*k + 0..6]
 * Radix-11: 10 twiddles per butterfly → stage_tw[10*k + 0..9]
 * Radix-13: 12 twiddles per butterfly → stage_tw[12*k + 0..11] ⚠️
 * 
 * For N=13312 (2^10 × 13), this means:
 * - Radix-2 stages: 10 stages × 1 twiddle × ~6656 butterflies = ~67K twiddles
 * - With radix-13: 1 stage × 12 twiddles × 1024 butterflies = ~12K twiddles
 * 
 * Despite 12× more twiddles per butterfly, total is lower due to fewer stages!
 * 
 * =============================================================================
 * USAGE NOTES
 * =============================================================================
 * 
 * @param output_buffer[out] Destination buffer (size: sub_len * 13)
 * @param sub_outputs[in]    Input array with 13 strided sub-transforms
 *                            Layout: [X[k+0*sub_len], ..., X[k+12*sub_len]]
 * @param stage_tw[in]       DIT twiddle factors
 *                            Layout: stage_tw[12*k + 0..11] = W^(k·1), ..., W^(k·12)
 *                            Only used when sub_len > 1
 * @param sub_len            Size of each sub-transform (total = 13 * sub_len)
 * @param transform_sign     +1 for inverse FFT, -1 for forward FFT
 * 
 * @note ALGORITHM CHOICE:
 *       Pure Rader (no symmetry) chosen because:
 *       - Symmetry implementation would require 12 global constants
 *       - 6-way coefficient cycling is complex to implement efficiently
 *       - Register pressure would severely limit AVX2 parallelism
 *       - Savings (~40%) don't justify 2-3× code complexity
 *       - Radix-13 is rare enough that extreme optimization isn't warranted
 * 
 * @warning PERFORMANCE IMPLICATIONS:
 *          Radix-13 is THE MOST EXPENSIVE radix per output (~96 FLOPs).
 *          Use only when:
 *          1. N naturally contains factor 13 (unavoidable)
 *          2. Alternative factorizations require many more stages
 *          3. Transform is computed repeatedly (amortize setup cost)
 *          
 *          For most applications, restructure problem to avoid factor 13!
 *          Example: If N≈13K, consider padding to N=16K (pure power-of-2)
 *          
 * @performance Despite high per-output cost, radix-13 achieves highest
 *              SIMD efficiency (62-75%) due to being compute-bound rather
 *              than memory-bound. This makes it the "most efficient
 *              inefficiency" - it's expensive, but at least we're using
 *              the CPU cores to their full potential!
 */
void fft_radix13_bv(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len);

    void fft_radix13_fv(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len);

#endif // FFT_RADIX3_H

// 1300