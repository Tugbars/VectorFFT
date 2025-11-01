/**
 * @file fft_radix7_uniform.h
 * @brief Unified Radix-7 FFT Butterfly Interface - Multi-Architecture with Rader's Algorithm
 * 
 * @details
 * High-performance radix-7 FFT butterfly operations using Rader's algorithm for
 * prime-length DFTs. Features automatic SIMD dispatch and twiddle-less (N1)
 * optimization for first-stage acceleration.
 * 
 * Architecture Support:
 * - AVX-512: 8 doubles/vector, U2 pipeline (16 butterflies/iter), streaming stores
 * - AVX2:    4 doubles/vector, U2 pipeline (8 butterflies/iter), streaming stores
 * - SSE2:    2 doubles/vector, U2 pipeline (4 butterflies/iter), streaming stores
 * - Scalar:  Fallback with full algorithmic optimizations (P0/P1)
 * 
 * =============================================================================
 * WHY RADIX-7 IS SPECIAL: RADER'S ALGORITHM
 * =============================================================================
 * 
 * Unlike radix-2, radix-4, and radix-8 (powers of 2), radix-7 is PRIME.
 * Prime-length FFTs cannot use traditional Cooley-Tukey factorization.
 * Instead, we use **Rader's Algorithm**, which converts a prime-length DFT
 * into a cyclic convolution.
 * 
 * MATHEMATICAL FOUNDATION:
 * -----------------------
 * For prime N=7, Rader's algorithm:
 * 
 * 1. SEPARATE DC COMPONENT:
 *    Y[0] = X[0] + X[1] + X[2] + ... + X[6]  (simple sum, tree reduction)
 * 
 * 2. FIND PRIMITIVE ROOT (generator):
 *    For N=7, primitive root g=3 generates: {3^0, 3^1, 3^2, ...} mod 7 = {1,3,2,6,4,5}
 * 
 * 3. INPUT PERMUTATION:
 *    Reorder X[1..6] according to generator: [X[1], X[3], X[2], X[6], X[4], X[5]]
 * 
 * 4. 6-POINT CYCLIC CONVOLUTION:
 *    Y[m] = X[0] + Σ(permuted_inputs[l] * rader_twiddle[(q-l) mod 6])
 *    Uses round-robin scheduling for maximum ILP
 * 
 * 5. OUTPUT PERMUTATION:
 *    Place results according to [1,5,4,6,2,3]:
 *    - Convolution[0] → Y[1], Convolution[1] → Y[5], etc.
 * 
 * =============================================================================
 * TWIDDLE FACTOR ORGANIZATION
 * =============================================================================
 * 
 * Radix-7 uses TWO types of twiddle factors:
 * 
 * 1. **STAGE TWIDDLES** (standard FFT twiddles):
 *    - 6 twiddles per butterfly: W^(k*m) for m=1..6
 *    - Applied BEFORE Rader convolution
 *    - Layout: stage_tw->re[r*K + k], stage_tw->im[r*K + k] for r=0..5
 *    - Blocked SoA: all k values for lane r are contiguous (unit-stride!)
 *    - Only used when sub_len > 1 (multi-stage FFT)
 * 
 * 2. **RADER CONVOLUTION TWIDDLES** (constant per transform):
 *    - 6 complex values: exp(±2πi * out_perm[q] / 7) for q=0..5
 *    - Output permutation: [1,5,4,6,2,3]
 *    - Computed once during planning, reused for all butterflies
 *    - Broadcast to SIMD registers at stage start (P0 optimization)
 * 
 * =============================================================================
 * KEY OPTIMIZATIONS (ALL PRESERVED)
 * =============================================================================
 * 
 * **P0: Pre-split Rader Broadcasts** (8-10% gain):
 * - Load 6 scalar Rader twiddles ONCE
 * - Broadcast to SIMD vectors (all lanes identical)
 * - Reuse across ALL K butterflies
 * - Saves ~12 shuffles per butterfly × K iterations
 * 
 * **P0: Round-Robin Convolution** (10-15% gain):
 * - 6 independent accumulators updated in rotation
 * - Each accumulator updated every 6 FMAs (> 4-cycle latency)
 * - Perfect latency hiding, maximizes ILP
 * - With U2: interleave butterfly A and B → 2 FMAs/cycle
 * 
 * **P1: Tree Y0 Sum** (1-2% gain):
 * - Balanced tree reduction: 3 levels vs 6 sequential
 * - Reduces critical path for DC component
 * 
 * **Unit-Stride Twiddle Loads** (KILLER optimization):
 * - Blocked SoA layout: stage_tw->re[r*K+k..k+7] contiguous
 * - Direct aligned loads: 3 cycles (vs 10-cycle gathers!)
 * - 7× latency reduction for twiddle access
 * 
 * **U2 Pipeline**:
 * - Process 2 butterflies simultaneously (k and k+W)
 * - Saturates dual FMA ports on modern CPUs
 * - Interleaved loads/convolutions/stores
 * 
 * **Store-Time Adds**:
 * - Compute y1-y6 = x0 + v[permuted] during store
 * - Frees 6 SIMD registers (critical on AVX2!)
 * 
 * =============================================================================
 * PERFORMANCE CHARACTERISTICS
 * =============================================================================
 * 
 * Computational cost per butterfly:
 * - DC component: 6 additions (tree-reduced)
 * - Stage twiddles: 6 complex multiplies (if sub_len > 1)
 * - Convolution: 36 complex multiplies + 30 additions
 * - Total: 42 complex muls + 36 adds (vs radix-2: 1 mul + 2 adds)
 * 
 * Memory bandwidth per butterfly:
 * - Loads: 7 strided reads (7K elements total)
 * - Stores: 7 strided writes (7K elements total)
 * - Twiddles: 6 unit-stride reads (if sub_len > 1)
 * 
 * SIMD speedup (vs scalar):
 * - AVX-512: ~6-7× (8-wide with U2)
 * - AVX2:    ~4-5× (4-wide with U2)
 * - SSE2:    ~2-3× (2-wide with U2)
 * 
 * @author Tugbars
 * @version 4.0 (Generation 3 TRUE SoA + U2)
 * @date 2025
 */

#ifndef FFT_RADIX7_UNIFORM_H
#define FFT_RADIX7_UNIFORM_H

#include <stddef.h>
#include <stdbool.h>

//==============================================================================
// MAIN API: RADIX-7 BUTTERFLY OPERATIONS
//==============================================================================

/**
 * @brief Radix-7 FFT butterfly - BACKWARD (Inverse) - WITH TWIDDLES
 * 
 * @details
 * Standard radix-7 DIT butterfly using Rader's algorithm for inverse FFT.
 * 
 * Algorithm:
 * 1. Load 7 lanes from input (stride K between lanes)
 * 2. Apply stage twiddles (x0 unchanged, x1-x6 multiplied by W^(k*m))
 * 3. Compute DC component (y0 = sum of all inputs, tree reduction)
 * 4. Permute inputs for Rader: [1,3,2,6,4,5]
 * 5. 6-point cyclic convolution (round-robin schedule)
 * 6. Assemble outputs with permutation: [1,5,4,6,2,3]
 * 7. Store 7 lanes (stride K between lanes)
 * 
 * Input/Output Layout (TRUE SoA, blocked):
 * @code
 *   in_re[0*K + k]  in_im[0*K + k]  ← Lane 0 (k=0..K-1)
 *   in_re[1*K + k]  in_im[1*K + k]  ← Lane 1
 *   ...
 *   in_re[6*K + k]  in_im[6*K + k]  ← Lane 6
 * @endcode
 * 
 * @param[out] out_re Output real array (7K elements, TRUE SoA blocked)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements, TRUE SoA blocked)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] stage_tw Stage twiddle factors (6K elements, blocked SoA)
 *                     Layout: stage_tw->re[r*K + k] for r=0..5, k=0..K-1
 *                     Must be aligned to architecture requirement (16/32/64 bytes)
 * @param[in] rader_tw Rader convolution twiddles (6 elements, SoA)
 *                     Precomputed: exp(-2πi * out_perm[q] / 7) for q=0..5
 *                     Output permutation: [1,5,4,6,2,3]
 * @param[in] K Transform size (N/7, number of butterflies)
 * @param[in] sub_len Sub-transform length (use 1 to skip stage twiddles)
 * 
 * @note For first stage where all stage twiddles ≈ 1, use fft_radix7_bv_n1()
 *       (~20-30% faster, skips stage twiddle multiply)
 * 
 * @performance
 *   - AVX-512: ~6-7× faster than scalar (8-wide + U2)
 *   - AVX2:    ~4-5× faster than scalar (4-wide + U2)
 *   - SSE2:    ~2-3× faster than scalar (2-wide + U2)
 * 
 * @see fft_radix7_bv_n1() - Twiddle-less variant for first stage
 */
void fft_radix7_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict stage_tw,
    const fft_twiddle_soa *restrict rader_tw,
    int K,
    int sub_len);

/**
 * @brief Radix-7 FFT butterfly - FORWARD - WITH TWIDDLES
 * 
 * @details
 * Standard radix-7 DIT butterfly using Rader's algorithm for forward FFT.
 * 
 * Identical algorithm to backward variant, but uses forward-FFT twiddles:
 * - Stage twiddles: exp(-2πi*k*m/N) instead of exp(+2πi*k*m/N)
 * - Rader twiddles: exp(-2πi*q/7) instead of exp(+2πi*q/7)
 * 
 * (Twiddle signs are baked into precomputed values during planning)
 * 
 * @param[out] out_re Output real array (7K elements, TRUE SoA blocked)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements, TRUE SoA blocked)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] stage_tw Stage twiddle factors (6K elements, blocked SoA)
 * @param[in] rader_tw Rader convolution twiddles (6 elements, SoA)
 * @param[in] K Transform size (N/7, number of butterflies)
 * @param[in] sub_len Sub-transform length (use 1 to skip stage twiddles)
 * 
 * @note For first stage where all stage twiddles ≈ 1, use fft_radix7_fv_n1()
 * 
 * @see fft_radix7_fv_n1() - Twiddle-less variant for first stage
 */
void fft_radix7_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict stage_tw,
    const fft_twiddle_soa *restrict rader_tw,
    int K,
    int sub_len);

/**
 * @brief Radix-7 FFT butterfly - BACKWARD - NO TWIDDLES (N1)
 * 
 * @details
 * Twiddle-less variant for first radix-7 stage (inverse) or when all stage
 * twiddles ≈ 1. Skips stage twiddle multiply, ~20-30% faster.
 * 
 * Algorithm (simplified):
 * 1. Load 7 lanes (NO stage twiddles applied!)
 * 2. Compute DC component y0
 * 3. Permute + 6-point convolution (with Rader twiddles)
 * 4. Assemble + store
 * 
 * @param[out] out_re Output real array (7K elements)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] rader_tw Rader convolution twiddles (6 elements, SoA)
 * @param[in] K Transform size (N/7)
 * 
 * @note NO stage_tw parameter - assumes all W^(k*m) = 1
 * @note Still requires Rader twiddles for convolution
 * @note ~20-30% faster than fft_radix7_bv() when applicable
 * 
 * @see fft_radix7_bv() - Standard version with stage twiddles
 */
void fft_radix7_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict rader_tw,
    int K);

/**
 * @brief Radix-7 FFT butterfly - FORWARD - NO TWIDDLES (N1)
 * 
 * @details
 * Twiddle-less variant for first radix-7 stage (forward) or when all stage
 * twiddles ≈ 1. Skips stage twiddle multiply, ~20-30% faster.
 * 
 * @param[out] out_re Output real array (7K elements)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] rader_tw Rader convolution twiddles (6 elements, SoA)
 * @param[in] K Transform size (N/7)
 * 
 * @note NO stage_tw parameter - assumes all W^(k*m) = 1
 * @note Still requires Rader twiddles for convolution
 * 
 * @see fft_radix7_fv() - Standard version with stage twiddles
 */
void fft_radix7_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict rader_tw,
    int K);

//==============================================================================
// CAPABILITY QUERY FUNCTIONS
//==============================================================================

/**
 * @brief Query available SIMD capabilities for radix-7
 * 
 * @return String describing current SIMD support level
 * 
 * Example outputs:
 * - "AVX-512 (8×double, U2 pipeline, Rader P0/P1, N1 support)"
 * - "AVX2 (4×double, U2 pipeline, Rader P0/P1, N1 support)"
 * - "SSE2 (2×double, U2 pipeline, Rader P0/P1, N1 support)"
 */
const char *radix7_get_simd_capabilities(void);

/**
 * @brief Get required buffer alignment for optimal performance
 * 
 * @return Alignment in bytes (16 for SSE2, 32 for AVX2, 64 for AVX-512)
 * 
 * @note Use this for allocating buffers with posix_memalign or aligned_alloc
 * @note Streaming stores REQUIRE aligned buffers
 */
size_t radix7_get_alignment_requirement(void);

/**
 * @brief Get SIMD vector width
 * 
 * @return Number of doubles per SIMD vector (2/4/8 for SSE2/AVX2/AVX-512)
 * 
 * @note Useful for sizing work blocks and understanding parallelism level
 */
int radix7_get_vector_width(void);

#endif // FFT_RADIX7_UNIFORM_H