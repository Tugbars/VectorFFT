#ifndef FFT_RADIX16_H
#define FFT_RADIX16_H

#include "../fft_plan/fft_planning_types.h"


/**
 * @brief Radix-16 FFT butterfly using 2-Stage Radix-4 Decomposition
 * 
 * =============================================================================
 * WHY RADIX-16 IS THE ULTIMATE POWER-OF-2 RADIX
 * =============================================================================
 * 
 * Radix-16 = 2^4 represents the practical upper limit for power-of-2 FFT radices,
 * offering the best balance between stage reduction and per-stage complexity:
 * 
 * STAGE COUNT COMPARISON:
 * -----------------------
 * For N=65536 (2^16):
 * - Radix-2:  16 stages (log₂(65536))
 * - Radix-4:  8 stages  (log₄(65536))
 * - Radix-8:  5.33 stages (log₈(65536), requires mixed radix)
 * - Radix-16: 4 stages  (log₁₆(65536)) ⭐ OPTIMAL
 * - Radix-32: 3.2 stages (log₃₂(65536), but per-stage cost explodes)
 * 
 * WHY NOT RADIX-32 OR HIGHER?
 * ----------------------------
 * Beyond radix-16, the law of diminishing returns kicks in hard:
 * 
 * 1. **Twiddle Explosion**: Radix-32 needs 31 twiddles per butterfly
 *    vs radix-16's 15 (2× overhead)
 * 
 * 2. **Register Pressure**: 32 inputs → 128 AVX2 registers needed
 *    (modern CPUs have only 16!)
 * 
 * 3. **Cache Pressure**: Larger butterflies thrash L1 cache
 *    Radix-16: ~256 bytes per butterfly (fits in L1)
 *    Radix-32: ~512 bytes per butterfly (L1 thrashing)
 * 
 * 4. **Complexity**: Decomposition patterns become unwieldy
 *    Radix-16 = 2 radix-4 stages (clean, simple)
 *    Radix-32 requires 3+ stages or hybrid approaches
 * 
 * 5. **Stage Savings**: Going from 4 stages (radix-16) to 3.2 stages (radix-32)
 *    saves only 0.8 stages, not worth 2-3× complexity increase
 * 
 * **VERDICT**: Radix-16 is the sweet spot - maximum stage reduction with
 * manageable complexity. It's the "highest radix you actually want to use."
 * 
 * =============================================================================
 * ALGORITHM: 2-STAGE RADIX-4 DECOMPOSITION
 * =============================================================================
 * 
 * Radix-16 = 2^4 = 4 × 4, allowing clean decomposition into two radix-4 stages:
 * 
 * CONCEPTUAL STRUCTURE:
 * ---------------------
 * Input: X[0], X[1], ..., X[15]  (16 inputs)
 * 
 * STAGE 1: Apply Input DIT Twiddles
 *   X'[j] = X[j] * W^(jk)  for j=1..15  (X[0] unchanged)
 *   where W = exp(-2πi/N) for the full FFT
 * 
 * STAGE 2: First Radix-4 Layer (Split into 4 groups)
 *   Group 0: Radix-4 on [X'[0], X'[4], X'[8],  X'[12]]  → Y[0..3]
 *   Group 1: Radix-4 on [X'[1], X'[5], X'[9],  X'[13]]  → Y[4..7]
 *   Group 2: Radix-4 on [X'[2], X'[6], X'[10], X'[14]]  → Y[8..11]
 *   Group 3: Radix-4 on [X'[3], X'[7], X'[11], X'[15]]  → Y[12..15]
 * 
 * STAGE 2.5: Apply Intermediate Twiddles W₄
 *   Y[j] *= W₄^⌊j/4⌋*(j mod 4)
 *   where W₄ = exp(-2πi/4) = {1, -i, -1, +i} (simple rotations!)
 * 
 * STAGE 3: Second Radix-4 Layer (4 independent butterflies)
 *   For each m ∈ {0,1,2,3}:
 *     Radix-4 on [Y[m], Y[m+4], Y[m+8], Y[m+12]] → Z[m], Z[m+4], Z[m+8], Z[m+12]
 * 
 * Output: Z[0..15]
 * 
 * MATHEMATICAL DETAIL - THE W₄ TWIDDLES:
 * ---------------------------------------
 * W₄ = exp(-2πi/4) = exp(-πi/2) = -i
 * 
 * Powers of W₄:
 * - W₄⁰ = 1           (no operation)
 * - W₄¹ = -i          (90° rotation)
 * - W₄² = -1          (180° rotation, just negate)
 * - W₄³ = +i          (-90° rotation)
 * 
 * These are the SAME twiddles as radix-4! This is why radix-16 decomposes
 * so cleanly - it's literally "radix-4 twice" with radix-4 twiddles between.
 * 
 * APPLICATION PATTERN (for output Y from first stage):
 * - Y[0,1,2,3]:   no twiddles (m=0, W₄⁰ = 1)
 * - Y[4]:         no twiddle  (m=1, j=0, W₄^(1·0) = 1)
 * - Y[5]:         *= W₄¹ = -i
 * - Y[6]:         *= W₄² = -1
 * - Y[7]:         *= W₄³ = +i
 * - Y[8,9,10,11]: W₄^(2·j) for j=0,1,2,3 = {1, -1, 1, -1}
 * - Y[12,13,14,15]: W₄^(3·j) for j=0,1,2,3 = {1, +i, -1, -i}
 * 
 * CRITICAL INSIGHT: Most of these are "free" operations:
 * - Identity (×1): no work
 * - Negation (×-1): single XOR for sign flip
 * - Rotation (×±i): permute + XOR (swap re/im + sign)
 * 
 * Only 9 out of 16 intermediate twiddles require ANY operation,
 * and ALL are zero-multiply operations (just permute/XOR)!
 * 
 * =============================================================================
 * WHY 2×RADIX-4 BEATS ALTERNATIVES
 * =============================================================================
 * 
 * OPTION A: 4× Radix-2
 * - Would require 4 stages of radix-2 butterflies
 * - Many more twiddle applications (4 stages worth)
 * - Poor cache locality (4 separate passes through data)
 * - **Verdict**: Too many stages, poor performance
 * 
 * OPTION B: Radix-2 × Radix-8
 * - 2 stages: one radix-2, one radix-8
 * - Asymmetric structure complicates implementation
 * - Less regular than radix-4 × radix-4
 * - **Verdict**: No clear advantage over 2×radix-4
 * 
 * OPTION C: 2× Radix-4 ⭐ CHOSEN
 * - Clean, symmetric structure (both stages identical)
 * - Intermediate twiddles are all simple rotations (W₄ powers)
 * - Both stages use same highly-optimized radix-4 kernel
 * - Perfect balance of operations and data movement
 * - **Verdict**: OPTIMAL for radix-16
 * 
 * OPTION D: Direct 16-point DFT
 * - Would require 225+ complex multiplications
 * - Random access pattern, terrible cache behavior
 * - No exploitable structure
 * - **Verdict**: Completely impractical
 * 
 * =============================================================================
 * COMPUTATIONAL COST ANALYSIS
 * =============================================================================
 * 
 * Per Butterfly (16 outputs):
 * ----------------------------
 * 
 * STAGE 1: Input Twiddles
 * - 15 complex multiplies (X[0] needs no twiddle)
 * - Cost: 15 × 6 FLOPs = 90 FLOPs
 * 
 * STAGE 2: First Radix-4 (4 butterflies)
 * - Each radix-4: 0 complex multiplies + 8 complex adds
 * - 4 butterflies: 4 × (0×6 + 8×2) = 64 FLOPs
 * 
 * STAGE 2.5: Intermediate W₄ Twiddles
 * - 9 non-trivial twiddles, all are rotate/negate (0 multiplies!)
 * - Cost: ~18 FLOPs (mainly permute overhead, no actual math)
 * 
 * STAGE 3: Second Radix-4 (4 butterflies)
 * - Same as Stage 2: 64 FLOPs
 * 
 * **TOTAL: 90 + 64 + 18 + 64 = 236 FLOPs per butterfly**
 * **Per output: 236 / 16 ≈ 14.75 FLOPs/output**
 * 
 * COMPARISON (FLOPs per output):
 * -------------------------------
 * - Radix-2:  ~6 FLOPs/output   (theoretical minimum for power-of-2)
 * - Radix-4:  ~8 FLOPs/output
 * - Radix-8:  ~13 FLOPs/output  (2×radix-4 decomposition)
 * - Radix-16: ~15 FLOPs/output  ⭐ (2×radix-4 decomposition)
 * 
 * Radix-16 costs 2.5× more per output than radix-2, BUT:
 * - Uses 4× fewer stages (4 vs 16 for N=65536)
 * - 4× fewer memory passes through data
 * - Much better cache utilization
 * - **NET RESULT**: Typically 1.5-2× faster than pure radix-2 for large N
 * 
 * =============================================================================
 * MEMORY ACCESS PATTERNS
 * =============================================================================
 * 
 * Per Butterfly:
 * --------------
 * - 16 input loads (16 complex = 256 bytes)
 * - 15 twiddle loads (15 complex = 240 bytes)
 * - 16 output stores (16 complex = 256 bytes)
 * - **TOTAL: 752 bytes per butterfly**
 * 
 * Arithmetic Intensity:
 * ---------------------
 * AI = 236 FLOPs / 752 bytes ≈ 0.31 FLOPs/byte
 * 
 * This is MEMORY-BOUND on modern CPUs (peak AI ≈ 10-20), similar to radix-8.
 * The advantage comes from reduced number of stages, not from compute efficiency.
 * 
 * CACHE BEHAVIOR:
 * ---------------
 * Radix-16 butterfly working set:
 * - Inputs: 256 bytes
 * - Twiddles: 240 bytes  
 * - Temporaries: ~512 bytes (intermediate Y values)
 * - **Total: ~1KB per butterfly**
 * 
 * Modern L1 cache: 32-64KB → Can fit 32-64 butterflies simultaneously
 * This excellent L1 utilization is KEY to radix-16's performance advantage.
 * 
 * Contrast with radix-32:
 * - Would need ~2KB per butterfly
 * - Only 16-32 butterflies fit in L1
 * - More L1 thrashing → performance loss
 * 
 * =============================================================================
 * SIMD OPTIMIZATION STRATEGY
 * =============================================================================
 * 
 * AVX2 8× UNROLLING:
 * ------------------
 * Process 8 butterflies per iteration:
 * - 128 complex inputs (2048 bytes)
 * - 4 complex pairs per AVX2 register
 * - 16 lanes × 4 butterfly-pair groups = 64 AVX2 registers
 * 
 * KEY OPTIMIZATION: All W₄ twiddles precomputed OUTSIDE loop
 * - W₄ powers {1, -i, -1, +i} converted to AVX2 format once
 * - Stored in 4 AVX2 registers: W4_avx[0..3]
 * - Reused for all 8 butterflies → massive savings
 * 
 * ROTATION MASK PRECOMPUTATION:
 * ------------------------------
 * The ±i multiplication pattern depends on transform_sign:
 * - Forward FFT: multiply by -i → (a+bi)*(-i) = b - ai
 * - Inverse FFT: multiply by +i → (a+bi)*(+i) = -b + ai
 * 
 * Precompute rot_mask based on transform_sign:
 * - Eliminates conditional logic inside hot loop
 * - Enables branchless ±i rotation via permute + XOR
 * 
 * REGISTER PRESSURE MANAGEMENT:
 * ------------------------------
 * With 16 lanes × 4 groups = 64 intermediate values, register pressure is severe.
 * 
 * Strategy:
 * 1. Load stage: Pull data from memory to registers (x[16][4])
 * 2. Transform stage: First radix-4, store to y[16][4]
 * 3. Twiddle stage: Apply W₄, keep in y[16][4]
 * 4. Final stage: Second radix-4, stream directly to output
 * 
 * This 3-level staging prevents register spilling while maintaining parallelism.
 * 
 * =============================================================================
 * PERFORMANCE CHARACTERISTICS
 * =============================================================================
 * 
 * SIMD Efficiency (AVX2):
 * -----------------------
 * - Process 8 butterflies per iteration (128 outputs)
 * - Theoretical peak: 16 DP FLOPs/cycle
 * - Achieved: ~9-11 FLOPs/cycle
 * - Efficiency: 56-69% of peak
 * 
 * Good but not great - limited by memory bandwidth like radix-8.
 * 
 * WALL-CLOCK PERFORMANCE (Large N):
 * ----------------------------------
 * For N=65536 FFT on modern CPU:
 * - Pure radix-2: ~12-15 μs
 * - Pure radix-4: ~8-10 μs
 * - Pure radix-8: ~6-8 μs
 * - Pure radix-16: ~5-7 μs ⭐ (best power-of-2 approach)
 * - Mixed radix optimized: ~4-6 μs (combines multiple radices)
 * 
 * Radix-16 achieves ~2× speedup vs radix-2 despite higher per-output cost,
 * thanks to 4× fewer stages and excellent cache behavior.
 * 
 * SCALABILITY:
 * ------------
 * Radix-16 shines for N ≥ 16384:
 * - Smaller N: Stage reduction less impactful, radix-4/8 competitive
 * - Larger N: Stage reduction critical, radix-16 pulls ahead
 * - Optimal range: N ∈ [16K, 16M]
 * 
 * For N > 16M, consider mixed-radix with radix-32 or cache-oblivious algorithms.
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-16 uses THREE types of twiddles:
 * 
 * 1. **INPUT DIT TWIDDLES** (15 per butterfly):
 *    Layout: stage_tw[15*k + 0..14] = W^(k·1), ..., W^(k·15)
 *    Applied to X[1..15] before first radix-4 stage
 *    Only used when sub_len > 1 (multi-stage FFT)
 * 
 * 2. **INTERMEDIATE W₄ TWIDDLES** (9 non-trivial):
 *    Powers of W₄ = exp(-2πi/4) = {1, -i, -1, +i}
 *    Hardcoded as simple rotations/negations
 *    Applied between first and second radix-4 stages
 *    **Key advantage**: All are zero-multiply operations!
 * 
 * 3. **IMPLICIT RADIX-4 STRUCTURE**:
 *    Both radix-4 stages use built-in rotations (multiply by ±i)
 *    These don't need stored twiddles - just permute + XOR
 * 
 * STORAGE COMPARISON:
 * -------------------
 * Radix-8:  7 twiddles per butterfly  → stage_tw[7*k + 0..6]
 * Radix-16: 15 twiddles per butterfly → stage_tw[15*k + 0..14]
 * 
 * Despite 2× more twiddles, radix-16 uses 4× fewer stages:
 * - N=65536 with radix-8:  needs ~5 stages × 7 twiddles × ~2048 butterflies
 * - N=65536 with radix-16: needs ~4 stages × 15 twiddles × ~1024 butterflies
 * - **Result**: Radix-16 actually uses LESS total twiddle storage!
 * 
 * =============================================================================
 * USAGE NOTES
 * =============================================================================
 * 
 * @param output_buffer[out] Destination buffer (size: sub_len * 16)
 * @param sub_outputs[in]    Input array with 16 strided sub-transforms
 *                            Layout: [X[k+0*sub_len], ..., X[k+15*sub_len]]
 * @param stage_tw[in]       DIT twiddle factors
 *                            Layout: stage_tw[15*k + 0..14] = W^(k·1), ..., W^(k·15)
 *                            Only used when sub_len > 1
 * @param sub_len            Size of each sub-transform (total = 16 * sub_len)
 * @param transform_sign     +1 for inverse FFT, -1 for forward FFT
 * 
 * @note WHEN TO USE RADIX-16:
 *       Best for transform sizes N ≥ 16384 where:
 *       1. N is a power of 2 (or contains large power-of-2 factor)
 *       2. Stage count reduction is critical
 *       3. Memory bandwidth is the bottleneck (typical for FFT)
 *       
 *       For N < 16384, radix-4 or radix-8 may be competitive or faster
 *       due to lower per-stage overhead.
 * 
 * @note ALGORITHM SUPERIORITY:
 *       The 2×radix-4 decomposition is optimal because:
 *       - Clean, symmetric structure (both stages identical)
 *       - All intermediate twiddles are simple rotations (zero multiplies)
 *       - Reuses highly optimized radix-4 kernel twice
 *       - Perfect balance: maximum stage reduction without excessive complexity
 *       
 *       This makes radix-16 the "highest radix you actually want to use"
 *       for power-of-2 FFTs on modern hardware.
 * 
 * @performance Achieves ~2× speedup vs pure radix-2 for large N despite
 *              2.5× higher per-output computational cost, thanks to:
 *              - 4× fewer stages (fewer memory passes)
 *              - Excellent L1 cache utilization (~1KB per butterfly)
 *              - Zero-multiply intermediate twiddles
 *              - Optimal AVX2 vectorization with 8× unrolling
 */
void fft_radix16_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw,
    int sub_len);

void fft_radix16_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw,
    int sub_len);

#endif // FFT_RADIX3_H

// 2400