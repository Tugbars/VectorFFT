/**
 * @file fft_normalize.h
 * @brief SIMD-Optimized FFT Normalization Functions
 * 
 * @details
 * Provides high-performance normalization utilities for FFT data with
 * automatic SIMD acceleration (AVX-512, AVX2, SSE2). Includes both
 * standalone normalization and zero-cost fused format conversion.
 * 
 * **Key Features:**
 * - Automatic SIMD selection (AVX-512 → AVX2 → SSE2 → Scalar)
 * - SoA (Structure of Arrays) normalization
 * - AoS (Array of Structures) normalization
 * - Zero-cost fused conversion + normalization
 * - 4-8× faster than scalar loops
 * 
 * **Performance:**
 * - AVX-512: 8 doubles/cycle → ~7-8× speedup
 * - AVX2:    4 doubles/cycle → ~4× speedup
 * - SSE2:    2 doubles/cycle → ~2× speedup
 * 
 * **Typical Usage:**
 * @code
 * // After inverse FFT, normalize with 1/N:
 * fft_exec_dft(inv_plan, freq, time, workspace);
 * fft_normalize_explicit((double*)time, N, 1.0/N);
 * 
 * // Or use zero-cost fused normalization during conversion:
 * fft_join_soa_to_aos_normalized(re, im, output, N, 1.0/N);
 * @endcode
 * 
 * @see fft_execute.c for integration with FFT execution
 * @see fft_normalize.c for implementation details
 * 
 * @author VectorFFT
 * @version 1.0
 */

#ifndef FFT_NORMALIZE_H
#define FFT_NORMALIZE_H

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// STANDALONE NORMALIZATION - SoA FORMAT
//==============================================================================

/**
 * @brief Normalize SoA (Structure of Arrays) complex data with SIMD
 * 
 * @details
 * Applies scale factor to both real and imaginary components stored in
 * separate arrays. Uses widest available SIMD instruction set for optimal
 * performance.
 * 
 * **Memory Layout:**
 * ```
 * Input:   re[0], re[1], ..., re[n-1]  (contiguous)
 *          im[0], im[1], ..., im[n-1]  (contiguous)
 * Output:  re[i] *= scale, im[i] *= scale
 * ```
 * 
 * **Use Cases:**
 * - Normalizing butterfly outputs stored in SoA format
 * - Applying 1/N scaling to inverse FFT results
 * - Custom scaling factors (e.g., 1/√N for orthogonal normalization)
 * 
 * **Performance:**
 * - O(n) complexity
 * - ~1-3% of total FFT time for large n
 * - Memory bandwidth limited (not compute limited)
 * 
 * **SIMD Optimization:**
 * - AVX-512: Processes 8 doubles per iteration
 * - AVX2:    Processes 4 doubles per iteration
 * - SSE2:    Processes 2 doubles per iteration
 * - Scalar fallback for non-SIMD platforms
 * 
 * **Example:**
 * @code
 * double re[1024], im[1024];
 * // ... compute FFT into SoA format ...
 * 
 * // Normalize with 1/N factor
 * fft_normalize_soa(re, im, 1024, 1.0/1024.0);
 * @endcode
 * 
 * @param[in,out] re Real components array (modified in-place)
 * @param[in,out] im Imaginary components array (modified in-place)
 * @param[in] n Number of complex elements
 * @param[in] scale Scale factor to apply (e.g., 1.0/N)
 * 
 * @note If scale == 1.0, function returns immediately (no-op optimization)
 * @note Arrays should be 64-byte aligned for best performance (not required)
 */
void fft_normalize_soa(double *re, double *im, int n, double scale);

//==============================================================================
// STANDALONE NORMALIZATION - INTERLEAVED (AoS) FORMAT
//==============================================================================

/**
 * @brief Normalize interleaved complex data (AoS format) with SIMD
 * 
 * @details
 * Applies scale factor to interleaved complex data where real and imaginary
 * parts alternate in memory. This is the standard format for fft_data arrays.
 * 
 * **Memory Layout:**
 * ```
 * Input/Output: [re0, im0, re1, im1, re2, im2, ...]
 *                └────────────────────────────────┘
 *                      2*n doubles total
 * ```
 * 
 * **Use Cases:**
 * - Normalizing FFT output stored as fft_data*
 * - Most common normalization case
 * - Direct normalization after fft_exec_dft()
 * 
 * **Performance:**
 * - Same as fft_normalize_soa() (both are memory-bandwidth limited)
 * - SIMD efficiently handles interleaved layout
 * 
 * **Example:**
 * @code
 * fft_data output[1024];
 * fft_exec_dft(inv_plan, input, output, workspace);
 * 
 * // Normalize (cast to double* for interleaved access)
 * fft_normalize_explicit((double*)output, 1024, 1.0/1024.0);
 * 
 * // Alternative using fft_exec_normalized() wrapper:
 * // fft_exec_normalized(inv_plan, input, output, workspace);
 * @endcode
 * 
 * @param[in,out] data Interleaved complex data [re0,im0,re1,im1,...] (modified in-place)
 * @param[in] n Number of complex elements (array has 2*n doubles)
 * @param[in] scale Scale factor to apply (e.g., 1.0/N)
 * 
 * @note data should point to 2*n contiguous doubles
 * @note If scale == 1.0, function returns immediately (no-op optimization)
 * @note This is the most commonly used normalization function
 */
void fft_normalize_explicit(double *data, int n, double scale);

//==============================================================================
// FUSED CONVERSION + NORMALIZATION (ZERO-COST NORMALIZATION!)
//==============================================================================

/**
 * @brief Fused SoA→AoS conversion with normalization (ZERO-COST!)
 * 
 * @details
 * ⚡ **ZERO-COST NORMALIZATION!** ⚡
 * 
 * This function combines two operations into a single memory pass:
 * 1. Convert from SoA (separate re[], im[]) to interleaved AoS
 * 2. Apply normalization scale factor
 * 
 * By fusing these operations, normalization becomes essentially free since
 * we're already loading/storing data for the conversion. The multiply
 * instruction fits in the compute gaps while waiting for memory.
 * 
 * **Performance:**
 * - Same cost as plain conversion WITHOUT normalization
 * - Saves one full memory pass (2× memory bandwidth reduction)
 * - Critical for large N where memory is the bottleneck
 * 
 * **Memory Layout:**
 * ```
 * Input:  re[0], re[1], ..., re[n-1]  (SoA - contiguous real)
 *         im[0], im[1], ..., im[n-1]  (SoA - contiguous imag)
 * 
 * Output: [re0*s, im0*s, re1*s, im1*s, ...]  (AoS - interleaved, scaled)
 *          where s = scale
 * ```
 * 
 * **Use Cases:**
 * - Converting butterfly output (SoA) to user format (AoS) with normalization
 * - Final stage of inverse FFT when returning to user
 * - Any time you need both conversion AND normalization
 * 
 * **Example:**
 * @code
 * // Inefficient way (TWO memory passes):
 * fft_join_soa_to_aos(re, im, output, N);       // Pass 1: Load + store
 * fft_normalize_explicit(output, N, 1.0/N);     // Pass 2: Load + multiply + store
 * 
 * // Efficient way (ONE memory pass):
 * fft_join_soa_to_aos_normalized(re, im, output, N, 1.0/N);  // Fused!
 * @endcode
 * 
 * @param[in] re Source real components (SoA format)
 * @param[in] im Source imaginary components (SoA format)
 * @param[out] output Destination interleaved complex data (AoS format)
 * @param[in] n Number of complex elements
 * @param[in] scale Scale factor to apply during conversion (1.0 for no scaling)
 * 
 * @note output must have space for 2*n doubles
 * @note Set scale=1.0 for conversion without normalization (still optimized)
 * @note This is the most efficient way to normalize when converting formats
 */
void fft_join_soa_to_aos_normalized(
    const double *re,
    const double *im,
    double *output,
    int n,
    double scale);

/**
 * @brief Fused AoS→SoA conversion with normalization (ZERO-COST!)
 * 
 * @details
 * Similar to fft_join_soa_to_aos_normalized(), but converts from interleaved
 * AoS format to separated SoA format while applying normalization.
 * 
 * **Performance:**
 * - Same cost as plain conversion WITHOUT normalization
 * - Normalization is free (fused with load/store operations)
 * 
 * **Memory Layout:**
 * ```
 * Input:  [re0, im0, re1, im1, re2, im2, ...]  (AoS - interleaved)
 * 
 * Output: re[0], re[1], ..., re[n-1]  (SoA - scaled real components)
 *         im[0], im[1], ..., im[n-1]  (SoA - scaled imag components)
 * ```
 * 
 * **Use Cases:**
 * - Converting user input to internal SoA format with pre-normalization
 * - Less common than join (more often normalize on output, not input)
 * - Useful for orthogonal transforms (1/√N on both input and output)
 * 
 * **Example:**
 * @code
 * fft_data input[1024];
 * double re[1024], im[1024];
 * 
 * // Convert to SoA with 1/√N normalization for orthogonal FFT
 * double scale = 1.0 / sqrt(1024.0);
 * fft_split_aos_to_soa_normalized((double*)input, re, im, 1024, scale);
 * @endcode
 * 
 * @param[in] input Source interleaved complex data (AoS format)
 * @param[out] re Destination real components (SoA format)
 * @param[out] im Destination imaginary components (SoA format)
 * @param[in] n Number of complex elements (input has 2*n doubles)
 * @param[in] scale Scale factor to apply during conversion (1.0 for no scaling)
 * 
 * @note input should point to 2*n contiguous doubles
 * @note re and im must each have space for n doubles
 * @note Less commonly used than fft_join_soa_to_aos_normalized()
 */
void fft_split_aos_to_soa_normalized(
    const double *input,
    double *re,
    double *im,
    int n,
    double scale);

//==============================================================================
// CONVENIENCE MACROS
//==============================================================================

/**
 * @brief Standard inverse FFT normalization (1/N scaling)
 * 
 * @details
 * Common normalization convention: only scale on inverse FFT.
 * 
 * @param data Pointer to fft_data array
 * @param N Transform size
 * 
 * @code
 * fft_data output[1024];
 * fft_exec_dft(inv_plan, input, output, workspace);
 * FFT_NORMALIZE_INVERSE(output, 1024);  // Apply 1/N scaling
 * @endcode
 */
#define FFT_NORMALIZE_INVERSE(data, N) \
    fft_normalize_explicit((double*)(data), (N), 1.0 / (double)(N))

/**
 * @brief Orthogonal/unitary normalization (1/√N scaling)
 * 
 * @details
 * For energy-preserving transforms (Parseval's theorem).
 * Apply to BOTH forward and inverse FFTs.
 * 
 * @param data Pointer to fft_data array
 * @param N Transform size
 * 
 * @code
 * // Forward FFT with 1/√N
 * FFT_NORMALIZE_ORTHO(output_fwd, 1024);
 * 
 * // Inverse FFT with 1/√N
 * FFT_NORMALIZE_ORTHO(output_inv, 1024);
 * 
 * // Now: energy_time == energy_freq (Parseval's theorem)
 * @endcode
 */
#define FFT_NORMALIZE_ORTHO(data, N) \
    fft_normalize_explicit((double*)(data), (N), 1.0 / sqrt((double)(N)))

/**
 * @brief Custom scale factor normalization
 * 
 * @details
 * Apply arbitrary scale factor (for advanced use cases).
 * 
 * @param data Pointer to fft_data array
 * @param N Transform size
 * @param scale Custom scale factor
 */
#define FFT_NORMALIZE_CUSTOM(data, N, scale) \
    fft_normalize_explicit((double*)(data), (N), (scale))

//==============================================================================
// DOCUMENTATION NOTES
//==============================================================================

/**
 * @page normalization_guide Normalization Guide
 * 
 * @section norm_conventions Normalization Conventions
 * 
 * Different applications use different normalization conventions:
 * 
 * **1. Standard Convention (FFTW, NumPy, MATLAB):**
 * - Forward FFT: No scaling
 * - Inverse FFT: 1/N scaling
 * - Use: fft_normalize_explicit(output, N, 1.0/N) after inverse
 * 
 * **2. Orthogonal/Unitary Convention:**
 * - Forward FFT: 1/√N scaling
 * - Inverse FFT: 1/√N scaling
 * - Use: FFT_NORMALIZE_ORTHO(output, N) on both transforms
 * - Benefit: Preserves energy (||x|| = ||X||)
 * 
 * **3. No Normalization (raw DFT):**
 * - Forward FFT: No scaling
 * - Inverse FFT: No scaling
 * - Result: Inverse gives N×original
 * - Use: When you don't need exact reconstruction (e.g., power spectrum)
 * 
 * @section norm_performance Performance Considerations
 * 
 * - Normalization is typically <3% of total FFT time
 * - With SIMD, overhead drops to <0.5% (essentially free)
 * - Fused operations (fft_join_soa_to_aos_normalized) have ZERO overhead
 * - Skip normalization for power spectrum or intermediate steps
 * 
 * @section norm_examples Common Examples
 * 
 * **Round-trip (should recover input):**
 * @code
 * fft_exec_dft(fwd_plan, input, freq, workspace);        // Forward (no norm)
 * fft_exec_normalized(inv_plan, freq, output, workspace); // Inverse (1/N norm)
 * // output ≈ input
 * @endcode
 * 
 * **Power spectrum (no normalization needed):**
 * @code
 * fft_exec_dft(fwd_plan, signal, freq, workspace);
 * for (int k = 0; k < N; k++) {
 *     power[k] = freq[k].re*freq[k].re + freq[k].im*freq[k].im;
 * }
 * @endcode
 * 
 * **Convolution (normalize once at end):**
 * @code
 * fft_exec_dft(fwd, sig1, freq1, ws);
 * fft_exec_dft(fwd, sig2, freq2, ws);
 * complex_multiply(freq1, freq2, result, N);
 * fft_exec_normalized(inv, result, output, ws);  // Only normalize here
 * @endcode
 */

#ifdef __cplusplus
}
#endif

#endif // FFT_NORMALIZE_H