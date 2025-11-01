/**
 * @file fft_radix2.h
 * @brief Unified Radix-2 FFT Butterfly Interface - Multi-Architecture with N1 Optimization
 * 
 * @details
 * High-performance radix-2 FFT butterfly operations with automatic SIMD dispatch
 * and twiddle-less (N1) optimization for first-stage acceleration.
 * 
 * Architecture Support:
 * - AVX-512: 8 doubles/vector, 4× unroll, masked cleanup, streaming stores
 * - AVX2:    4 doubles/vector, 2× unroll, FMA acceleration, streaming stores
 * - SSE2:    2 doubles/vector, 2× unroll, baseline x86-64, streaming stores
 * - Scalar:  Fallback with special-case optimizations (k=0, N/4, N/8, 3N/8)
 * 
 * Key Optimizations:
 * 1. N1 (Twiddle-Less) Optimization: ~3× faster for W≈1 cases
 * 2. Special-case handling: k=0 (W=1), k=N/4 (W=-i), k=N/8, k=3N/8 (W=±√2/2)
 * 3. Multi-tier SIMD acceleration with optimal unrolling per architecture
 * 4. Software prefetching (T0 for data, T1 for twiddles)
 * 5. Non-temporal (streaming) stores for large N to reduce cache pollution
 * 6. Zero-shuffle SoA layout - pure arithmetic, no data rearrangement
 * 
 * @author FFT Optimization Team
 * @version 3.1 (Unified architecture with N1 integration)
 * @date 2025
 */

#ifndef FFT_RADIX2_H
#define FFT_RADIX2_H

#include "../fft_plan/fft_planning_types.h"

/**
 * @brief Radix-2 FFT butterfly - WITH TWIDDLES - Standard version
 * 
 * @details
 * Standard radix-2 DIT butterfly with twiddle factors for general FFT stages.
 * 
 * Performs the core butterfly computation:
 * @code
 *   Y[k]      = X_even[k] + W^k * X_odd[k]
 *   Y[k+half] = X_even[k] - W^k * X_odd[k]
 * @endcode
 * 
 * where W = exp(±2πi/N) (sign baked into precomputed twiddles during planning).
 * 
 * **Note**: The butterfly operation is identical for forward and inverse FFT.
 * The transform direction is determined by the sign of twiddle factors computed
 * during the planning phase, not by this function.
 * 
 * Processing Features:
 * - Special case optimization for k=0, N/4, N/8, 3N/8
 * - Automatic SIMD dispatch (AVX-512/AVX2/SSE2/Scalar)
 * - Optimal unrolling per architecture (4×/2×/2×/1×)
 * - Automatic streaming stores for large N (> 256K elements)
 * - Software prefetching for cache optimization
 * 
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements, even indices in [0..half-1])
 * @param[in] in_im Input imaginary array (N elements, odd indices in [half..N-1])
 * @param[in] stage_tw Precomputed twiddle factors (SoA: {re[], im[]})
 *                     W^k for k=0..half-1 where W = exp(±2πi/N)
 *                     Must be aligned to RADIX2_ALIGNMENT (16/32/64 bytes)
 * @param[in] half Half the transform size (N/2)
 * 
 * @note For first stage where all twiddles = 1, use fft_radix2_bv_n1() instead
 *       (~3× faster, no twiddle multiplies)
 * 
 * @performance
 *   - AVX-512: ~4.5× faster than scalar
 *   - AVX2:    ~3.5× faster than scalar  
 *   - SSE2:    ~2.5× faster than scalar
 * 
 * @see fft_radix2_bv_n1() - Twiddle-less variant for first stage
 */
void fft_radix2_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half);

/**
 * @brief Radix-2 FFT butterfly - NO TWIDDLES - First stage optimization
 * 
 * @details
 * Twiddle-less variant for first radix-2 stage or Stockham auto-sort where
 * all twiddles W=1. This is ~3× faster than standard version due to:
 * - No twiddle loads (saves 2 memory accesses per butterfly)
 * - No complex multiplies (saves 4 FMAs per butterfly)
 * - Reduced register pressure
 * 
 * Algorithm (simplified):
 * @code
 *   Y[k]      = X_even[k] + X_odd[k]    // No twiddle multiply!
 *   Y[k+half] = X_even[k] - X_odd[k]
 * @endcode
 * 
 * **USAGE:**
 * @code
 * // First radix-2 stage (all twiddles = 1)
 * fft_radix2_bv_n1(out_re, out_im, in_re, in_im, half);
 * 
 * // Subsequent stages (need twiddles)
 * fft_radix2_bv(out_re, out_im, in_re, in_im, stage_tw, half);
 * @endcode
 * 
 * Processing Features:
 * - Automatic SIMD dispatch (AVX-512/AVX2/SSE2/Scalar)
 * - Optimal unrolling per architecture
 * - Automatic streaming stores for large N
 * - Zero twiddle overhead (~3× faster than WITH-twiddles version)
 * 
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] half Half the transform size (N/2)
 * 
 * @note NO stage_tw parameter - this variant assumes all W=1
 * @note ~3× faster than fft_radix2_bv() when applicable
 * @note Use only when ALL butterflies have W=1 (first stage or Stockham)
 * 
 * @performance
 *   - AVX-512: ~13× faster than scalar (3× n1 speedup × 4.5× SIMD)
 *   - AVX2:    ~10× faster than scalar (3× n1 speedup × 3.5× SIMD)
 *   - SSE2:    ~7× faster than scalar (3× n1 speedup × 2.5× SIMD)
 * 
 * @see fft_radix2_bv() - Standard version with twiddles
 */
void fft_radix2_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int half);

//==============================================================================
// CAPABILITY QUERY FUNCTIONS
//==============================================================================