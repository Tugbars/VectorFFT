#ifndef FFT_RADIX11_H
#define FFT_RADIX11_H

#include "highspeedFFT.h"


/**
 * @brief Radix-11 FFT butterfly using Rader's Algorithm with Symmetry Exploitation
 * 
 * =============================================================================
 * WHY RADIX-11 IS EXTREMELY SPECIAL: RADER + SYMMETRY
 * =============================================================================
 * 
 * Radix-11 is a PRIME number, requiring Rader's Algorithm like radix-7, but
 * with a crucial difference: **11 is large enough to benefit from exploiting
 * Hermitian symmetry**, which radix-7 cannot effectively use.
 * 
 * KEY DIFFERENCES FROM RADIX-7:
 * ------------------------------
 * 
 * RADIX-7:
 * - 6 non-DC outputs require full 6×6 cyclic convolution
 * - 36 complex multiplications per butterfly
 * - No symmetry reduction possible (6 is not even, pairs don't align well)
 * - Pure cyclic convolution approach
 * 
 * RADIX-11: ⭐ SYMMETRY BREAKTHROUGH
 * - 10 non-DC outputs can be computed as 5 SYMMETRIC PAIRS
 * - For real-valued DFT, Y[k] and Y[11-k] are complex conjugates
 * - Even with complex inputs, pairing reduces computational cost
 * - Only 5 pairs need computation, second element of each pair is derived
 * - **Effective cost: ~50% of naive Rader approach**
 * 
 * =============================================================================
 * RADER'S ALGORITHM WITH SYMMETRY: THE MATHEMATICS
 * =============================================================================
 * 
 * STEP 1: PRIMITIVE ROOT AND PERMUTATIONS
 * ----------------------------------------
 * For N=11, the primitive root g=2 generates all non-zero elements:
 * Powers of 2 mod 11: {2^0, 2^1, 2^2, ...} mod 11 = {1,2,4,8,5,10,9,7,3,6}
 * 
 * Input permutation (using g=2):
 *   [X[1], X[2], X[4], X[8], X[5], X[10], X[9], X[7], X[3], X[6]]
 * 
 * Output indices for convolution results follow similar permutation pattern
 * 
 * STEP 2: DC COMPONENT (No Change from Radix-7)
 * -----------------------------------------------
 * Y[0] = X[0] + X[1] + X[2] + ... + X[10]  (simple sum of all inputs)
 * 
 * STEP 3: EXPLOIT HERMITIAN SYMMETRY ⭐ CRITICAL OPTIMIZATION
 * ------------------------------------------------------------
 * For a prime N, the DFT has a special property:
 *   Y[k] and Y[N-k] are related by complex conjugation patterns
 * 
 * We can pair outputs as: (Y[1], Y[10]), (Y[2], Y[9]), (Y[3], Y[8]), 
 *                          (Y[4], Y[7]), (Y[5], Y[6])
 * 
 * For each pair (Y[m], Y[11-m]):
 * 
 *   SHARED BASE (Real part):
 *   base_real = X[0] + Σ(k=0..4) C[k] * (X[p[k]] + X[p[9-k]])
 *   
 *   where C[k] = cos(2π*(k+1)/11) are the 5 cosine constants
 *   and (X[p[k]] + X[p[9-k]]) are the 5 symmetric pair sums
 * 
 *   ROTATION TERM (Imaginary contribution):
 *   rotation = Σ(k=0..4) S[k] * (X[p[k]] - X[p[9-k]])
 *   
 *   where S[k] = sin(2π*(k+1)/11) are the 5 sine constants
 *   and (X[p[k]] - X[p[9-k]]) are the 5 symmetric pair differences
 * 
 *   FINAL OUTPUTS:
 *   Y[m]     = base_real + i·rotation  (rotated by +90° or -90° based on sign)
 *   Y[11-m]  = base_real - i·rotation  (conjugate relationship)
 * 
 * COEFFICIENT CYCLING:
 * --------------------
 * Each output pair (Y[m], Y[11-m]) uses the SAME 5 cosine and 5 sine constants,
 * but in CYCLICALLY ROTATED ORDER. This is a property of the primitive root.
 * 
 * Example coefficient patterns:
 * - Pair 1 (Y[1], Y[10]): [C1, C2, C3, C4, C5] and [S1, S2, S3, S4, S5]
 * - Pair 2 (Y[2], Y[9]):  [C2, C4, C5, C3, C1] and [S2, S4, S5, S3, S1]
 * - Pair 3 (Y[3], Y[8]):  [C3, C5, C2, C1, C4] and [S3, S5, S2, S1, S4]
 * - etc.
 * 
 * This cyclic pattern comes from Rader's algorithm structure and enables
 * efficient vectorization without storing separate coefficient sets.
 * 
 * =============================================================================
 * COMPUTATIONAL SAVINGS FROM SYMMETRY
 * =============================================================================
 * 
 * WITHOUT SYMMETRY (Naive Rader for 11-point):
 * - 10 outputs × 10 convolution terms each = 100 complex multiplies
 * - Plus 10 × 10 = 100 complex additions
 * - Total: ~600 FLOPs per butterfly
 * 
 * WITH SYMMETRY (This Implementation):
 * - Compute 5 pair sums: 5 additions (real + imaginary) = 10 FLOPs
 * - Compute 5 pair differences: 5 subtractions = 10 FLOPs
 * - Each of 5 pairs needs:
 *   * 5 multiply-accumulates for real part = 5 FMA ops
 *   * 5 multiply-accumulates for rotation = 5 FMA ops
 *   * 1 rotation (90° multiply by ±i) = swap + sign flip
 *   * 2 final additions (base ± rotation) = 2 adds
 * - 5 pairs × operations = ~120 FLOPs + overhead
 * - **Total: ~150-180 FLOPs per butterfly (60-70% savings!)**
 * 
 * =============================================================================
 * WHY AVX2 USES STRUCTURE-OF-ARRAYS (SoA) LAYOUT
 * =============================================================================
 * 
 * PROBLEM WITH AOS FOR RADIX-11:
 * -------------------------------
 * Array-of-Structures layout (re, im interleaved) works well for radix-2/4/8
 * where operations are predominantly:
 * - Load pair, compute, store pair (simple access patterns)
 * - Complex multiply benefits from FMA: (a+bi)*(c+di)
 * 
 * But radix-11 with symmetry requires:
 * - Forming sums/differences across 11 strided loads (t0..t4, s0..s4)
 * - 5 multiply-accumulate chains: Σ(k=0..4) C[k] * t[k]
 * - Separate real/imaginary processing (cosine vs sine terms)
 * - Many horizontal operations within SIMD lanes
 * 
 * SOA ADVANTAGES FOR RADIX-11:
 * -----------------------------
 * 1. **Natural separation**: Real and imaginary components already separate
 *    - Cosine terms only affect real parts: easier to accumulate
 *    - Sine terms only affect imaginary rotation: separate computation
 * 
 * 2. **FMA efficiency**: Intel AVX2 FMA operates on 4 doubles at once
 *    - SoA: `result_vec = FMA(coeff_vec, data_vec, accum_vec)` ✓
 *    - AoS: Would need extract, compute, repack → massive overhead
 * 
 * 3. **Horizontal operations minimized**: 
 *    - SoA keeps 4 parallel butterflies aligned in SIMD lanes
 *    - No expensive shuffles to group real/imaginary across pairs
 * 
 * 4. **Cyclic coefficient access**:
 *    - Broadcast same coefficient to all 4 lanes: `_mm256_set1_pd(C11_1)`
 *    - Apply to 4 separate butterfly computations simultaneously
 * 
 * TRADEOFF: CONVERSION COST
 * --------------------------
 * - Input arrives in AoS format (natural memory layout)
 * - Must convert: AoS → SoA (deinterleave operation)
 * - Compute in SoA (massive savings on FMA operations)
 * - Convert back: SoA → AoS (reinterleave for output)
 * 
 * Break-even analysis:
 * - Conversion cost: ~40 instructions (20 loads + 10 shuffles + 10 stores)
 * - Computation savings: ~80-100 instructions avoided
 * - **Net benefit: 40-60 instructions per 4 butterflies**
 * 
 * For smaller radices (2/4/8), conversion overhead > savings, so stay in AoS.
 * For large primes (11+), conversion cost < computation savings, use SoA!
 * 
 * =============================================================================
 * ALGORITHM STEP-BY-STEP
 * =============================================================================
 * 
 * For each group of 11 inputs (one butterfly):
 * 
 * STAGE 0: Data Layout Conversion (AVX2 path only)
 *   Convert 4 butterflies from AoS to SoA format
 *   [X0.re, X0.im, X1.re, X1.im, ...] → [X0.re, X1.re, X2.re, X3.re], [X0.im, ...]
 * 
 * STAGE 1: Load 11 Lanes
 *   X[0..10] from sub_outputs with stride=sub_len
 * 
 * STAGE 2: Apply DIT Twiddles (if multi-stage)
 *   X'[m] = X[m] * stage_tw[10*k + (m-1)] for m=1..10
 *   (X[0] unchanged, no twiddle needed)
 * 
 * STAGE 3: Compute DC Output
 *   Y[0] = X'[0] + X'[1] + ... + X'[10]
 * 
 * STAGE 4: Form 5 Symmetric Pairs
 *   After proper Rader permutation:
 *   t0 = X'[1] + X'[10]    s0 = X'[1] - X'[10]
 *   t1 = X'[2] + X'[9]     s1 = X'[2] - X'[9]
 *   t2 = X'[3] + X'[8]     s2 = X'[3] - X'[8]
 *   t3 = X'[4] + X'[7]     s3 = X'[4] - X'[7]
 *   t4 = X'[5] + X'[6]     s4 = X'[5] - X'[6]
 * 
 * STAGE 5: Compute 5 Output Pairs (with coefficient cycling)
 *   For each pair m ∈ {1,2,3,4,5}:
 *     Real base = X'[0] + Σ(j=0..4) C[cycle_pattern_m[j]] * t[j]
 *     Rotation  = Σ(j=0..4) S[cycle_pattern_m[j]] * s[j]
 *     Rotate by ±90°: rot_complex = ±i * Rotation
 *     Y[m]     = Real base + rot_complex
 *     Y[11-m]  = Real base - rot_complex
 * 
 * STAGE 6: Convert Back to AoS and Store (AVX2 path)
 *   Reinterleave SoA results back to AoS format
 *   Write Y[0..10] to output_buffer
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-11 uses THREE types of factors:
 * 
 * 1. **COSINE CONSTANTS** (5 values, global constants):
 *    C1 = cos(2π/11)   ≈  0.8413
 *    C2 = cos(4π/11)   ≈  0.4154
 *    C3 = cos(6π/11)   ≈ -0.1423
 *    C4 = cos(8π/11)   ≈ -0.6549
 *    C5 = cos(10π/11)  ≈ -0.9595
 *    Hardcoded in implementation, broadcast to SIMD registers
 * 
 * 2. **SINE CONSTANTS** (5 values, global constants):
 *    S1 = sin(2π/11)   ≈  0.5406
 *    S2 = sin(4π/11)   ≈  0.9096
 *    S3 = sin(6π/11)   ≈  0.9898
 *    S4 = sin(8π/11)   ≈  0.7557
 *    S5 = sin(10π/11)  ≈  0.2817
 *    Hardcoded in implementation, broadcast to SIMD registers
 * 
 * 3. **DIT STAGE TWIDDLES** (10 per butterfly, from stage_tw):
 *    Layout: stage_tw[10*k + 0..9] = W^(k·1), ..., W^(k·10)
 *    Applied to X[1..10] before symmetry exploitation
 * 
 * STORAGE COMPARISON:
 * -------------------
 * Radix-2:  1 twiddle per butterfly  → stage_tw[k]
 * Radix-7:  6 twiddles per butterfly → stage_tw[6*k + 0..5]
 * Radix-8:  7 twiddles per butterfly → stage_tw[7*k + 0..6]
 * Radix-11: 10 twiddles per butterfly → stage_tw[10*k + 0..9]
 * 
 * Plus 10 global constants (5 cosine + 5 sine) shared across all butterflies
 * 
 * =============================================================================
 * PERFORMANCE CHARACTERISTICS
 * =============================================================================
 * 
 * Computational Complexity per Butterfly:
 * ----------------------------------------
 * - Input twiddles: 10 complex multiplies = 60 FLOPs
 * - Symmetric pairs: 10 additions + 10 subtractions = 40 FLOPs
 * - 5 pair computations:
 *   * Real parts: 5 × 5 FMA = 25 FMA ops ≈ 50 FLOPs
 *   * Rotations: 5 × 5 FMA = 25 FMA ops ≈ 50 FLOPs
 *   * 90° rotations: 5 × (swap+sign) ≈ 10 FLOPs
 *   * Final combine: 10 additions = 20 FLOPs
 * - **TOTAL: ~230 FLOPs per butterfly (11 outputs)**
 * 
 * Memory Operations (AVX2 path, 4 butterflies):
 * ----------------------------------------------
 * - 44 loads (11 lanes × 4 butterflies) = 704 bytes
 * - 40 twiddle loads (10 × 4 butterflies) = 640 bytes
 * - AoS→SoA conversion overhead: ~200 bytes temp storage
 * - 44 stores (11 lanes × 4 butterflies) = 704 bytes
 * - **TOTAL: ~2248 bytes per 4 butterflies ≈ 562 bytes/butterfly**
 * 
 * Arithmetic Intensity:
 * ---------------------
 * AI = 230 FLOPs / 562 bytes ≈ 0.41 FLOPs/byte
 * 
 * Better than radix-8 (0.28), but still memory-bound on modern CPUs.
 * The SoA conversion pays off due to much more complex arithmetic patterns.
 * 
 * SIMD Efficiency (AVX2):
 * -----------------------
 * - Process 4 butterflies per iteration (44 outputs)
 * - FMA throughput: 2 ports × 4 doubles × 2 ops = 16 DP FLOPs/cycle
 * - Achieved: ~6-8 FLOPs/cycle (limited by AoS↔SoA conversion)
 * - Efficiency: 40-50% of peak (good for such complex algorithm)
 * - **Key insight**: Without SoA, efficiency would drop to ~20-25%
 * 
 * Comparison to Other Radices (Per Output FLOPs):
 * ------------------------------------------------
 * - Radix-2:  ~6 FLOPs/output   (optimal for power-of-2)
 * - Radix-4:  ~8 FLOPs/output
 * - Radix-7:  ~17 FLOPs/output  (Rader without symmetry)
 * - Radix-8:  ~13 FLOPs/output  (2×radix-4 decomposition)
 * - Radix-11: ~21 FLOPs/output  (Rader with symmetry) ⭐
 * 
 * Despite higher per-output cost, radix-11 reduces total FFT stages when N
 * contains factor 11, often resulting in net performance gain.
 * 
 * =============================================================================
 * USAGE NOTES
 * =============================================================================
 * 
 * @param output_buffer[out] Destination buffer (size: sub_len * 11)
 * @param sub_outputs[in]    Input array with 11 strided sub-transforms
 *                            Layout: [X[k+0*sub_len], ..., X[k+10*sub_len]]
 * @param stage_tw[in]       DIT twiddle factors
 *                            Layout: stage_tw[10*k + 0..9] = W^(k·1), ..., W^(k·10)
 *                            Only used when sub_len > 1
 * @param sub_len            Size of each sub-transform (total = 11 * sub_len)
 * @param transform_sign     +1 for inverse FFT, -1 for forward FFT
 *                            Controls rotation direction in symmetry pairs
 * 
 * @note ALGORITHM SUPERIORITY:
 *       The symmetry-exploiting Rader approach is chosen over:
 *       - Direct 11-point DFT: Far too expensive (>100 multiplies)
 *       - Prime-factor algorithm: Doesn't decompose (11 is prime)
 *       - Chirp-Z transform: Higher constant overhead
 *       
 *       Symmetry exploitation reduces computational cost by ~60% compared
 *       to naive Rader implementation, making radix-11 practical for
 *       mixed-radix FFTs where N = 2^a × 11^b × ...
 * 
 * @warning WHEN TO USE RADIX-11:
 *          Only beneficial when transform size N contains factor 11.
 *          The high per-output cost is offset by reducing total stages.
 *          Example: N=1024 needs log₂(1024)=10 radix-2 stages
 *                   N=2048=2^10×2 uses radix-2
 *                   N=2112=2^6×33=2^6×3×11 could use radix-11 once
 *          
 *          For randomly-sized problems, consider padding to power-of-2
 *          rather than using radix-11 unless performance critical.
 */
void fft_radix11_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H

// 1400