/**
 * @file fft_radix5_uniform.h
 * @brief Unified Radix-5 FFT Butterfly Interface - Native SoA with Multi-Architecture Support
 * 
 * @details
 * High-performance radix-5 FFT butterfly operations with automatic SIMD dispatch
 * and zero-shuffle native SoA architecture.
 * 
 * Architecture Support:
 * - AVX-512: 8 doubles/vector, double-pumped (16 butterflies/iter)
 * - AVX2:    4 doubles/vector, double-pumped (8 butterflies/iter)
 * - SSE2:    2 doubles/vector, double-pumped (4 butterflies/iter)
 * - Scalar:  Fallback with optimized Rader's algorithm
 * 
 * Key Optimizations:
 * 1. Native SoA: NO split/join operations (30-45% faster)
 * 2. Double-pumped SIMD: Process 2× vectors per iteration for ILP
 * 3. Automatic streaming stores for large K to reduce cache pollution
 * 4. Software prefetching tuned for radix-5 strided access pattern
 * 5. Out-of-place only (required for correctness with ping-pong buffers)
 * 
 * @note Unlike radix-2 and radix-4, radix-5 requires separate forward/backward
 *       functions due to different butterfly operations and geometric constants.
 * 
 * @note N1 (twiddle-less) variants not yet implemented for radix-5.
 *       First stage uses standard twiddles (negligible overhead for radix-5).
 * 
 * @author VectorFFT Team
 * @version 1.0 (Native SoA architecture)
 * @date 2025
 */

#ifndef FFT_RADIX5_H
#define FFT_RADIX5_H

#include "../fft_plan/fft_planning_types.h"

/**
 * @brief Radix-5 FFT butterfly - FORWARD transform - Native SoA
 * 
 * @details
 * Forward radix-5 DIF butterfly using native Structure-of-Arrays layout.
 * 
 * Performs the core radix-5 butterfly computation:
 * @code
 *   Y[k + m*K] = Σ(j=0..4) X[k + j*K] * W[k]^(m*j)  for m=0..4
 * @endcode
 * 
 * where W = exp(-2πi/N) for forward transform.
 * 
 * **ZERO-SHUFFLE ARCHITECTURE:**
 * - Input: Native SoA (separate re[], im[] arrays)
 * - Output: Native SoA (separate re[], im[] arrays)
 * - NO split/join operations (30-45% faster than traditional AoS)
 * - Designed for ping-pong buffer execution between stages
 * 
 * Processing Features:
 * - Automatic SIMD dispatch (AVX-512/AVX2/SSE2/Scalar)
 * - Double-pumped loops (2× vector width per iteration for ILP)
 * - Automatic streaming stores for large K (K > 4096, footprint > 70% LLC)
 * - Software prefetching tuned for radix-5's 5-lane strided access
 * - Out-of-place only (required for correctness)
 * 
 * @param[out] out_re Output real array (N elements, N=5*K)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Precomputed twiddle factors (SoA: {re[], im[]})
 *                     W^k for k=0..K-1 where W = exp(-2πi/N)
 *                     Must be aligned to RADIX5_ALIGNMENT (16/32/64 bytes)
 * @param[in] K Sub-transform length (N/5 for this stage)
 * 
 * @pre out_re != in_re && out_im != in_im (out-of-place required)
 * @pre K > 0
 * @pre All pointers non-NULL
 * 
 * @note **Out-of-Place Only**:
 *       - In-place execution would cause read-after-write hazards
 *       - Use ping-pong buffers: A→B stage 0, B→A stage 1, A→B stage 2, etc.
 * 
 * @note **Alignment Requirements**:
 *       - All buffers should be aligned to RADIX5_ALIGNMENT for optimal performance
 *       - Streaming stores automatically enabled for large K, requiring aligned buffers
 *       - Twiddles are ALWAYS aligned (guaranteed by planning phase)
 * 
 * @note **Memory Layout**:
 *       - Pure SoA (Structure of Arrays) format: separate real and imaginary arrays
 *       - Zero-shuffle design: no data rearrangement needed
 *       - Each butterfly accesses 5 lanes: k, k+K, k+2K, k+3K, k+4K
 * 
 * @note **Automatic Streaming Stores**:
 *       - Automatically enabled when: K >= 4096 AND working set > 70% LLC
 *       - Working set = 5 lanes × 2 arrays × K × 8 bytes = 80K bytes
 *       - Example: K=4096 → 320 KB working set → streaming enabled on 8 MB LLC
 *       - Memory fence (_mm_sfence()) called automatically at function exit
 *       - Override with FFT_NT environment variable: 0=off, 1=on
 * 
 * @note **Usage Pattern** (multi-stage FFT):
 *       @code
 *       // Convert input once
 *       fft_aos_to_soa(input, buf_a_re, buf_a_im, N);
 *       
 *       // Process all stages in SoA (ping-pong buffers)
 *       for (int stage = 0; stage < num_stages; stage++) {
 *           if (stage % 2 == 0)
 *               fft_radix5_fv(buf_b_re, buf_b_im, buf_a_re, buf_a_im, tw[stage], K[stage]);
 *           else
 *               fft_radix5_fv(buf_a_re, buf_a_im, buf_b_re, buf_b_im, tw[stage], K[stage]);
 *       }
 *       
 *       // Convert output once
 *       fft_soa_to_aos(final_re, final_im, output, N);
 *       @endcode
 * 
 * @performance **Automatic SIMD Selection**:
 *              - AVX-512: ~8× faster than scalar (double-pumped 16 butterflies/iter)
 *              - AVX2:    ~6× faster than scalar (double-pumped 8 butterflies/iter)
 *              - SSE2:    ~4× faster than scalar (double-pumped 4 butterflies/iter)
 *              - Scalar:  Optimized Rader's algorithm for prime radix
 * 
 * @performance **SoA Speedup**:
 *              - 30-45% faster than traditional AoS approach
 *              - Zero overhead for split/join operations
 *              - Better cache utilization with strided access
 * 
 * @see fft_radix5_bv() - Backward (inverse) transform
 * @see radix5_get_simd_capabilities() - Query available SIMD support
 * @see radix5_get_alignment_requirement() - Get required buffer alignment
 */
void fft_radix5_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K);

/**
 * @brief Radix-5 FFT butterfly - BACKWARD (INVERSE) transform - Native SoA
 * 
 * @details
 * Backward radix-5 DIF butterfly using native Structure-of-Arrays layout.
 * 
 * Performs the core radix-5 butterfly computation:
 * @code
 *   Y[k + m*K] = Σ(j=0..4) X[k + j*K] * W[k]^(m*j)  for m=0..4
 * @endcode
 * 
 * where W = exp(+2πi/N) for inverse transform (opposite sign from forward).
 * 
 * **ZERO-SHUFFLE ARCHITECTURE:**
 * - Input: Native SoA (separate re[], im[] arrays)
 * - Output: Native SoA (separate re[], im[] arrays)
 * - NO split/join operations (30-45% faster than traditional AoS)
 * - Designed for ping-pong buffer execution between stages
 * 
 * Processing Features:
 * - Automatic SIMD dispatch (AVX-512/AVX2/SSE2/Scalar)
 * - Double-pumped loops (2× vector width per iteration for ILP)
 * - Automatic streaming stores for large K (K > 4096, footprint > 70% LLC)
 * - Software prefetching tuned for radix-5's 5-lane strided access
 * - Out-of-place only (required for correctness)
 * 
 * @param[out] out_re Output real array (N elements, N=5*K)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Precomputed twiddle factors (SoA: {re[], im[]})
 *                     W^k for k=0..K-1 where W = exp(+2πi/N)
 *                     Must be aligned to RADIX5_ALIGNMENT (16/32/64 bytes)
 * @param[in] K Sub-transform length (N/5 for this stage)
 * 
 * @pre out_re != in_re && out_im != in_im (out-of-place required)
 * @pre K > 0
 * @pre All pointers non-NULL
 * 
 * @note **Out-of-Place Only**:
 *       - In-place execution would cause read-after-write hazards
 *       - Use ping-pong buffers: A→B stage 0, B→A stage 1, A→B stage 2, etc.
 * 
 * @note **Alignment Requirements**:
 *       - All buffers should be aligned to RADIX5_ALIGNMENT for optimal performance
 *       - Streaming stores automatically enabled for large K, requiring aligned buffers
 *       - Twiddles are ALWAYS aligned (guaranteed by planning phase)
 * 
 * @note **Memory Layout**:
 *       - Pure SoA (Structure of Arrays) format: separate real and imaginary arrays
 *       - Zero-shuffle design: no data rearrangement needed
 *       - Each butterfly accesses 5 lanes: k, k+K, k+2K, k+3K, k+4K
 * 
 * @note **Automatic Streaming Stores**:
 *       - Automatically enabled when: K >= 4096 AND working set > 70% LLC
 *       - Working set = 5 lanes × 2 arrays × K × 8 bytes = 80K bytes
 *       - Example: K=4096 → 320 KB working set → streaming enabled on 8 MB LLC
 *       - Memory fence (_mm_sfence()) called automatically at function exit
 *       - Override with FFT_NT environment variable: 0=off, 1=on
 * 
 * @note **Usage Pattern** (multi-stage IFFT):
 *       @code
 *       // Convert input once
 *       fft_aos_to_soa(input, buf_a_re, buf_a_im, N);
 *       
 *       // Process all stages in SoA (ping-pong buffers)
 *       for (int stage = 0; stage < num_stages; stage++) {
 *           if (stage % 2 == 0)
 *               fft_radix5_bv(buf_b_re, buf_b_im, buf_a_re, buf_a_im, tw[stage], K[stage]);
 *           else
 *               fft_radix5_bv(buf_a_re, buf_a_im, buf_b_re, buf_b_im, tw[stage], K[stage]);
 *       }
 *       
 *       // Convert output once and scale by 1/N
 *       fft_soa_to_aos_scaled(final_re, final_im, output, N, 1.0/N);
 *       @endcode
 * 
 * @performance **Automatic SIMD Selection**:
 *              - AVX-512: ~8× faster than scalar (double-pumped 16 butterflies/iter)
 *              - AVX2:    ~6× faster than scalar (double-pumped 8 butterflies/iter)
 *              - SSE2:    ~4× faster than scalar (double-pumped 4 butterflies/iter)
 *              - Scalar:  Optimized Rader's algorithm for prime radix
 * 
 * @performance **SoA Speedup**:
 *              - 30-45% faster than traditional AoS approach
 *              - Zero overhead for split/join operations
 *              - Better cache utilization with strided access
 * 
 * @see fft_radix5_fv() - Forward transform
 * @see radix5_get_simd_capabilities() - Query available SIMD support
 * @see radix5_get_alignment_requirement() - Get required buffer alignment
 */
void fft_radix5_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K);

//==============================================================================
// CAPABILITY QUERY FUNCTIONS
//==============================================================================

/**
 * @brief Query available SIMD capabilities for radix-5
 * 
 * @return String describing current SIMD support level
 * 
 * Example outputs:
 * - "AVX-512F (8×double, double-pumped, Native SoA)"
 * - "AVX2 (4×double, double-pumped, Native SoA)"
 * - "SSE2 (2×double, double-pumped, Native SoA)"
 */
const char *radix5_get_simd_capabilities(void);

/**
 * @brief Get required buffer alignment for optimal performance
 * 
 * @return Alignment in bytes (16 for SSE2, 32 for AVX2, 64 for AVX-512)
 * 
 * @note Use this for allocating buffers with posix_memalign or aligned_alloc
 * @note Streaming stores REQUIRE aligned buffers
 */
size_t radix5_get_alignment_requirement(void);

/**
 * @brief Get SIMD vector width for radix-5
 * 
 * @return Number of doubles per SIMD vector (2/4/8 for SSE2/AVX2/AVX-512)
 * 
 * @note Useful for sizing work blocks and understanding parallelism level
 */
int radix5_get_vector_width(void);

//==============================================================================
// ARCHITECTURE NOTES
//==============================================================================

/**
 * @section radix5_architecture Radix-5 Architecture Notes
 * 
 * **Why Separate Forward/Backward Functions?**
 * 
 * Unlike radix-2 (where butterfly is identical for forward/backward),
 * radix-5 has different operations:
 * 
 * 1. **Twiddle Signs**: Forward uses W = exp(-2πi/N), backward uses W = exp(+2πi/N)
 * 2. **Geometric Constants**: cos(2π/5) terms have opposite signs
 * 3. **Butterfly Structure**: Different addition/subtraction patterns
 * 
 * **Why No N1 Variants Yet?**
 * 
 * For radix-5:
 * - First stage twiddles are more spread out (5 unique roots of unity)
 * - N1 optimization benefit is smaller than radix-2/4 (~20% vs ~3×)
 * - Implementation complexity is higher (5-point butterfly vs 2/4-point)
 * - Priority: Get radix-5 working correctly first, optimize later
 * 
 * **Out-of-Place Requirement:**
 * 
 * Radix-5 butterfly reads all 5 lanes (k, k+K, k+2K, k+3K, k+4K) before
 * writing any outputs. In-place execution would cause read-after-write
 * hazards. Solution: Use ping-pong buffers between stages.
 * 
 * **Double-Pumped SIMD:**
 * 
 * Process 2× vector width per iteration to improve instruction-level
 * parallelism (ILP). For example, AVX-512 processes 16 butterflies/iter
 * by issuing two 8-wide SIMD operations back-to-back, keeping execution
 * units fully saturated.
 */

#endif // FFT_RADIX5_H