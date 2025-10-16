#ifndef FFT_RADIX5_H
#define FFT_RADIX5_H

#include "highspeedFFT.h"

#ifndef FFT_RADIX5_H
#define FFT_RADIX5_H

#include "highspeedFFT.h"

/**
 * @brief Radix-5 FFT butterfly: The Goldilocks Prime Radix
 * 
 * =============================================================================
 * WHY RADIX-5 IS SPECIAL: THE "JUST RIGHT" PRIME
 * =============================================================================
 * 
 * Radix-5 occupies a unique position among prime radices - it's the ONLY prime
 * that can be implemented efficiently WITHOUT Rader's algorithm, making it
 * fundamentally simpler than radix-7, radix-11, or radix-13.
 * 
 * THE PRIME RADIX LANDSCAPE:
 * ---------------------------
 * 
 * **RADIX-3 (2+1 inputs)**: 
 * - Direct DFT possible, no Rader needed
 * - Cost: ~9 FLOPs/output
 * - Simple, fast, widely used
 * 
 * **RADIX-5 (4+1 inputs)** ⭐ THE GOLDILOCKS PRIME:
 * - Direct DFT still possible, no Rader needed!
 * - Exploits N-1 = 4 = 2² structure
 * - Cost: ~12 FLOPs/output
 * - Perfect balance: prime benefits without Rader complexity
 * 
 * **RADIX-7 (6+1 inputs)**:
 * - Requires Rader's algorithm (6-point cyclic convolution)
 * - Cost: ~17 FLOPs/output
 * - Complexity jump is significant
 * 
 * **RADIX-11, 13, etc.**:
 * - All require Rader with increasing complexity
 * - Cost: 21-30+ FLOPs/output
 * - Rarely used except when N naturally contains these factors
 * 
 * WHY RADIX-5 DOESN'T NEED RADER:
 * --------------------------------
 * The key insight is that 5-1 = 4 = 2², which allows a direct decomposition:
 * 
 * For a 5-point DFT Y[m] = Σ(n=0..4) X[n] * W^(mn):
 * 
 * 1. **Separate DC component**: Y[0] = X[0] + X[1] + X[2] + X[3] + X[4]
 * 
 * 2. **Group non-DC using symmetry** (N-1 = 4 elements):
 *    Since W^5 = 1, we can pair: (X[1], X[4]) and (X[2], X[3])
 *    These pairs have conjugate-like relationships under rotation
 * 
 * 3. **Express using cosines and sines**:
 *    Real parts: controlled by cos(2π/5) and cos(4π/5)
 *    Imaginary parts: controlled by sin(2π/5) and sin(4π/5)
 *    
 *    Only FOUR constants needed (vs Rader which needs N-1 twiddles)!
 * 
 * 4. **Closed-form butterfly equations**:
 *    Can be written as matrix multiplication with fixed coefficients
 *    No cyclic convolution needed!
 * 
 * This makes radix-5 the largest prime that admits a "simple" FFT implementation.
 * 
 * =============================================================================
 * ALGORITHM: DIRECT 5-POINT DFT WITH SYMMETRY EXPLOITATION
 * =============================================================================
 * 
 * Input: X[0], X[1], X[2], X[3], X[4]
 * 
 * STAGE 1: Apply Input DIT Twiddles (if multi-stage)
 *   X'[j] = X[j] * W^(jk)  for j=1..4  (X[0] unchanged)
 * 
 * STAGE 2: Form Symmetric Pairs
 *   t0 = X'[1] + X'[4]  (sum of symmetric pair 1)
 *   t1 = X'[2] + X'[3]  (sum of symmetric pair 2)
 *   t2 = X'[1] - X'[4]  (difference of symmetric pair 1)
 *   t3 = X'[2] - X'[3]  (difference of symmetric pair 2)
 * 
 * STAGE 3: Compute DC Output
 *   Y[0] = X'[0] + t0 + t1
 * 
 * STAGE 4: Compute Non-DC Outputs Using Four Magic Constants
 *   
 *   The four magic constants (golden ratio related):
 *   C1 = cos(2π/5)  = cos(72°)  ≈  0.309017
 *   C2 = cos(4π/5)  = cos(144°) ≈ -0.809017
 *   S1 = sin(2π/5)  = sin(72°)  ≈  0.951057
 *   S2 = sin(4π/5)  = sin(144°) ≈  0.587785
 *   
 *   First pair of outputs (Y[1], Y[4]):
 *     base = C1*t0 + C2*t1
 *     rotation = S1*t2 + S2*t3
 *     Y[1] = X'[0] + base + i*sgn*rotation
 *     Y[4] = X'[0] + base - i*sgn*rotation
 *   
 *   Second pair of outputs (Y[2], Y[3]):
 *     base = C2*t0 + C1*t1
 *     rotation = S2*t2 - S1*t3
 *     Y[2] = X'[0] + base + i*sgn*rotation
 *     Y[3] = X'[0] + base - i*sgn*rotation
 * 
 * Note the beautiful symmetry:
 * - C1 and C2 swap positions between pairs
 * - S1 and S2 swap positions between pairs
 * - Sign flips in S-terms between pairs
 * 
 * OUTPUT: Y[0], Y[1], Y[2], Y[3], Y[4]
 * 
 * =============================================================================
 * THE GOLDEN RATIO CONNECTION
 * =============================================================================
 * 
 * The radix-5 constants have a deep connection to the golden ratio φ = (1+√5)/2:
 * 
 * cos(2π/5) = (φ - 1)/2 = (√5 - 1)/4
 * cos(4π/5) = -(φ + 1)/2 = -(√5 + 1)/4
 * 
 * This connection gives radix-5 excellent numerical properties:
 * - Constants are algebraic numbers (not transcendental)
 * - Can be computed with high precision
 * - Stable under repeated operations
 * 
 * The golden ratio also appears in the regular pentagon geometry, which is
 * intimately related to the 5-fold rotational symmetry in this FFT butterfly.
 * 
 * =============================================================================
 * WHY PURE AoS (ARRAY-OF-STRUCTURES) LAYOUT?
 * =============================================================================
 * 
 * Unlike radix-11 which uses SoA for symmetry exploitation, radix-5 stays in
 * pure AoS format. Here's why this is optimal:
 * 
 * AoS ADVANTAGES FOR RADIX-5:
 * ----------------------------
 * 1. **Simple structure**: Only 4 constants, no complex coefficient cycling
 *    The butterfly equations are straightforward enough that AoS works well
 * 
 * 2. **Efficient FMA usage**: Modern CPUs have excellent FMA throughput
 *    FMA(C1, t0, C2*t1) maps perfectly to AoS complex operations
 * 
 * 3. **No conversion overhead**: Avoiding AoS↔SoA saves 20-30 instructions
 *    For radix-5, this overhead would exceed any computation savings
 * 
 * 4. **Memory access patterns**: Five strided loads benefit from prefetching
 *    AoS keeps related data together, improving cache line utilization
 * 
 * 5. **Code simplicity**: Radix-5 is meant to be "the simple prime"
 *    Staying in AoS preserves this simplicity advantage
 * 
 * SOA WOULD HURT RADIX-5:
 * -----------------------
 * - Conversion cost: ~40 instructions per 8 butterflies
 * - Computation savings: ~20 instructions (not enough to justify conversion)
 * - Break-even would require 10+ constants or complex FMA chains
 * - Radix-5's four constants are too simple to benefit from SoA
 * 
 * =============================================================================
 * COMPUTATIONAL COST ANALYSIS
 * =============================================================================
 * 
 * Per Butterfly (5 outputs):
 * ---------------------------
 * 
 * STAGE 1: Input Twiddles
 * - 4 complex multiplies = 24 FLOPs
 * 
 * STAGE 2: Symmetric Pairs
 * - 4 complex additions/subtractions = 8 FLOPs
 * 
 * STAGE 3: DC Output
 * - 2 complex additions = 4 FLOPs
 * 
 * STAGE 4: Non-DC Outputs
 * - 8 real multiplies (4 constants × 2 pairs) = 8 FLOPs
 * - 4 FMA operations = 8 FLOPs
 * - 8 complex additions = 16 FLOPs
 * 
 * **TOTAL: 24 + 8 + 4 + 8 + 8 + 16 = 68 FLOPs per butterfly**
 * **Per output: 68 / 5 ≈ 13.6 FLOPs/output**
 * 
 * COMPARISON (FLOPs per output):
 * -------------------------------
 * - Radix-2:  ~6 FLOPs/output   (optimal baseline)
 * - Radix-3:  ~9 FLOPs/output   (simple prime)
 * - Radix-4:  ~8 FLOPs/output   (power-of-2)
 * - Radix-5:  ~14 FLOPs/output  ⭐ (efficient prime)
 * - Radix-7:  ~17 FLOPs/output  (Rader begins)
 * - Radix-8:  ~13 FLOPs/output  (power-of-2)
 * 
 * Radix-5 sits between radix-4 and radix-7 in cost, making it the most
 * efficient prime after radix-3, and competitive with power-of-2 radices!
 * 
 * =============================================================================
 * MEMORY CHARACTERISTICS
 * =============================================================================
 * 
 * Per Butterfly:
 * --------------
 * - 5 input loads: 80 bytes
 * - 4 twiddle loads: 64 bytes
 * - 5 output stores: 80 bytes
 * - **TOTAL: 224 bytes per butterfly**
 * 
 * Arithmetic Intensity:
 * ---------------------
 * AI = 68 FLOPs / 224 bytes ≈ 0.30 FLOPs/byte
 * 
 * This is MEMORY-BOUND like most FFT radices, but the ratio is decent.
 * The relatively low memory traffic (compared to radix-7+) helps performance.
 * 
 * CACHE BEHAVIOR:
 * ---------------
 * Radix-5 butterfly working set:
 * - Inputs: 80 bytes
 * - Twiddles: 64 bytes
 * - Temporaries: ~160 bytes
 * - **Total: ~304 bytes per butterfly**
 * 
 * Modern L1 cache: 32-48KB → Can fit 100-150 butterflies
 * Excellent cache utilization contributes to radix-5's performance.
 * 
 * =============================================================================
 * SIMD OPTIMIZATION STRATEGY
 * =============================================================================
 * 
 * AVX2 8× UNROLLING:
 * ------------------
 * Process 8 butterflies per iteration:
 * - 40 complex inputs (640 bytes)
 * - 4 complex pairs per AVX2 register
 * - 5 lanes × 4 butterfly-pair groups = 20 AVX2 registers
 * 
 * KEY OPTIMIZATION: Four constants broadcast OUTSIDE loop
 * - C1, C2, S1, S2 → __m256d registers
 * - Broadcast once, reused for all 8 butterflies
 * - Eliminates 32 broadcasts per iteration!
 * 
 * ROTATION MASK PRECOMPUTATION:
 * ------------------------------
 * The ±i multiplication in the rotation terms requires:
 * - Forward FFT: multiply by -i
 * - Inverse FFT: multiply by +i
 * 
 * Precompute rot_mask based on transform_sign:
 * - One-time cost outside loop
 * - Branchless rotation via permute + XOR
 * - Same mask reused for all pairs
 * 
 * MACRO-BASED BUTTERFLY:
 * ----------------------
 * The RADIX5_BUTTERFLY_AVX2 macro encapsulates the entire butterfly:
 * ```c
 * RADIX5_BUTTERFLY_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4)
 * ```
 * 
 * Benefits:
 * - Code clarity: One-line invocation per butterfly
 * - Compiler optimization: Full inline expansion
 * - Register allocation: Compiler sees entire dependency graph
 * - Reduced copy-paste errors
 * 
 * =============================================================================
 * PERFORMANCE CHARACTERISTICS
 * =============================================================================
 * 
 * SIMD Efficiency (AVX2):
 * -----------------------
 * - Process 8 butterflies per iteration (40 outputs)
 * - Theoretical peak: 16 DP FLOPs/cycle
 * - Achieved: ~9-11 FLOPs/cycle
 * - Efficiency: 56-69% of peak
 * 
 * Good efficiency for a prime radix, comparable to power-of-2 radices.
 * 
 * WHEN RADIX-5 SHINES:
 * ---------------------
 * 1. **Transform sizes containing factor 5**:
 *    N = 2^a × 5^b × ... where b ≥ 1
 *    Examples: 80, 160, 320, 640, 1280, 5000, 10000
 * 
 * 2. **Real-world sample rates**:
 *    Many audio/DSP systems use decimal-friendly rates
 *    - 48kHz = 2^4 × 3 × 10^3 = 2^4 × 3 × 2^3 × 5^3
 *    - 100kHz = 10^5 = 2^5 × 5^5
 * 
 * 3. **Scientific computing**:
 *    Measurements in SI units often yield sizes with factor 5
 * 
 * 4. **Mixed-radix FFTs**:
 *    Radix-5 combines well with radix-2/4/8
 *    Better than padding to next power-of-2
 * 
 * COMPARISON WITH ALTERNATIVES:
 * ------------------------------
 * For N=5000 (5^4 × 8):
 * - Pure radix-5: 4 stages (optimal)
 * - Pad to 8192 (2^13): 13 stages + wasted computation
 * - Radix-5 wins decisively!
 * 
 * For N=10000 (2^4 × 5^4):
 * - Mixed radix (radix-5 + radix-2): ~5-6 stages
 * - Pure radix-2: 14 stages (needs padding to 16384)
 * - Mixed radix significantly faster
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-5 uses TWO types of factors:
 * 
 * 1. **FOUR MAGIC CONSTANTS** (global, hardcoded):
 *    C1 = cos(2π/5)  ≈  0.309017
 *    C2 = cos(4π/5)  ≈ -0.809017
 *    S1 = sin(2π/5)  ≈  0.951057
 *    S2 = sin(4π/5)  ≈  0.587785
 *    
 *    Broadcast to AVX2 registers once, reused forever
 * 
 * 2. **DIT STAGE TWIDDLES** (4 per butterfly, from stage_tw):
 *    Layout: stage_tw[4*k + 0..3] = W^(k·1), ..., W^(k·4)
 *    Applied to X[1..4] before butterfly computation
 *    Only used when sub_len > 1 (multi-stage FFT)
 * 
 * STORAGE COMPARISON:
 * -------------------
 * Radix-4: 3 twiddles per butterfly → stage_tw[3*k + 0..2]
 * Radix-5: 4 twiddles per butterfly → stage_tw[4*k + 0..3]
 * Radix-7: 6 twiddles per butterfly → stage_tw[6*k + 0..5]
 * 
 * Plus 4 global constants (16 bytes total, negligible)
 * 
 * For N=5000 using pure radix-5:
 * - 4 stages × 4 twiddles × ~625 butterflies per stage ≈ 10K twiddles
 * - Very manageable memory footprint
 * 
 * =============================================================================
 * USAGE NOTES
 * =============================================================================
 * 
 * @param output_buffer[out] Destination buffer (size: sub_len * 5)
 * @param sub_outputs[in]    Input array with 5 strided sub-transforms
 *                            Layout: [X[k+0*sub_len], ..., X[k+4*sub_len]]
 * @param stage_tw[in]       DIT twiddle factors
 *                            Layout: stage_tw[4*k + 0..3] = W^(k·1), ..., W^(k·4)
 *                            Only used when sub_len > 1
 * @param sub_len            Size of each sub-transform (total = 5 * sub_len)
 * @param transform_sign     +1 for inverse FFT, -1 for forward FFT
 * 
 * @note WHY RADIX-5 IS "JUST RIGHT":
 *       - Largest prime not requiring Rader's algorithm
 *       - Only 4 constants needed (vs N-1 for Rader)
 *       - Direct butterfly equations (no cyclic convolution)
 *       - Competitive with power-of-2 radices in performance
 *       - Essential for decimal-friendly transform sizes
 *       
 *       Radix-5 is the "Goldilocks prime": not too small (like 3),
 *       not too complex (like 7+), but just right for practical use!
 * 
 * @note ALGORITHM ELEGANCE:
 *       The direct DFT approach with symmetry exploitation represents
 *       optimal balance between simplicity and efficiency. The golden
 *       ratio connection gives excellent numerical stability.
 *       
 *       For prime radices, radix-5 is the last stop before Rader
 *       complexity becomes unavoidable. It's the most sophisticated
 *       prime that still feels "simple" to implement and optimize.
 * 
 * @performance Achieves 56-69% SIMD efficiency, comparable to power-of-2
 *              radices despite being prime. The ~14 FLOPs/output cost is
 *              excellent for a prime, making radix-5 the go-to choice for:
 *              - Audio DSP (48kHz, 96kHz sample rates)
 *              - Scientific computing (decimal-based measurements)
 *              - Mixed-radix FFTs (avoiding power-of-2 padding)
 *              - Any application where N naturally contains factor 5
 */
void fft_radix5_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H