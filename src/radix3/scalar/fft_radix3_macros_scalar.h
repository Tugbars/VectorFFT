/**
 * @file radix3_scalar_optimized.h
 * @brief Scalar Radix-3 Implementation (Ultimate Fallback)
 *
 * PURE SCALAR IMPLEMENTATION:
 * ===========================
 * This is the final fallback when no SIMD is available:
 * - No vector instructions
 * - Works on ANY architecture (x86, ARM, RISC-V, etc.)
 * - Portable C code with manual optimizations
 * - Serves as reference implementation
 *
 * OPTIMIZATIONS APPLIED:
 * ======================
 * 1. Constant elimination: Single sqrt(3)/2 constant (no negative version)
 * 2. Blocked twiddle layout: All twiddles for element in one cache line
 * 3. Manual instruction scheduling hints via careful ordering
 * 4. Compiler-friendly code patterns (enables auto-vectorization on some compilers)
 *
 * WHEN TO USE SCALAR:
 * ===================
 * - K < 4 (SIMD overhead not worth it)
 * - No SIMD support (embedded, non-x86 architectures)
 * - Reference implementation for verification
 * - Debugging (easier to step through)
 *
 * @author Tugbars
 * @version 3.1-SCALAR (Portable implementation)
 * @date 2025
 */

#ifndef RADIX3_SCALAR_OPTIMIZED_H
#define RADIX3_SCALAR_OPTIMIZED_H

#include <stddef.h>

//==============================================================================
// COMPILER ATTRIBUTES
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#else
#define FORCE_INLINE static inline
#define RESTRICT
#endif

//==============================================================================
// GEOMETRIC CONSTANTS
//==============================================================================

#define C_HALF_SCALAR (-0.5)
#define S_SQRT3_2_SCALAR 0.8660254037844386467618

//==============================================================================
// TWIDDLE LAYOUT (Per-element)
//==============================================================================
// For scalar, twiddles are stored per-element: [W1_re, W1_im, W2_re, W2_im]
// This is simpler than blocked layout for scalar access

#define TWIDDLE_OFFSET_R3_SCALAR(k) ((k) * 4)

//==============================================================================
// INLINE BUTTERFLY FUNCTIONS
//==============================================================================

/**
 * @brief Single scalar butterfly - FORWARD
 *
 * Optimizations:
 * 1. Single constant (no negative sqrt(3)/2)
 * 2. Manual instruction ordering for ILP
 * 3. Careful reuse of temporaries to reduce register pressure
 */
FORCE_INLINE void radix3_butterfly_scalar_fv(
    const size_t k,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t K)
{
    const double c_half = C_HALF_SCALAR;
    const double s_sqrt3_2 = S_SQRT3_2_SCALAR;

    // Load inputs
    const double a_re = in_re[k];
    const double a_im = in_im[k];
    const double b_re = in_re[k + K];
    const double b_im = in_im[k + K];
    const double c_re = in_re[k + 2 * K];
    const double c_im = in_im[k + 2 * K];

    // Load twiddles (per-element layout)
    const size_t tw_idx = TWIDDLE_OFFSET_R3_SCALAR(k);
    const double w1_re = tw[tw_idx + 0];
    const double w1_im = tw[tw_idx + 1];
    const double w2_re = tw[tw_idx + 2];
    const double w2_im = tw[tw_idx + 3];

    // Complex multiply: tB = b * W^1
    const double tB_re = b_re * w1_re - b_im * w1_im;
    const double tB_im = b_re * w1_im + b_im * w1_re;

    // Complex multiply: tC = c * W^2
    const double tC_re = c_re * w2_re - c_im * w2_im;
    const double tC_im = c_re * w2_im + c_im * w2_re;

    // Radix-3 butterfly computation
    const double sum_re = tB_re + tC_re;
    const double sum_im = tB_im + tC_im;
    const double dif_re = tB_re - tC_re;
    const double dif_im = tB_im - tC_im;

    // Rotation (manually scheduled for ILP)
    const double rot_re = s_sqrt3_2 * dif_im;
    const double rot_im = -s_sqrt3_2 * dif_re;

    // Common term
    const double common_re = a_re + c_half * sum_re;
    const double common_im = a_im + c_half * sum_im;

    // Output
    out_re[k] = a_re + sum_re;
    out_im[k] = a_im + sum_im;
    out_re[k + K] = common_re + rot_re;
    out_im[k + K] = common_im + rot_im;
    out_re[k + 2 * K] = common_re - rot_re;
    out_im[k + 2 * K] = common_im - rot_im;
}

/**
 * @brief Single scalar butterfly - BACKWARD
 */
FORCE_INLINE void radix3_butterfly_scalar_bv(
    const size_t k,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw,
    const size_t K)
{
    const double c_half = C_HALF_SCALAR;
    const double s_sqrt3_2 = S_SQRT3_2_SCALAR;

    // Load inputs
    const double a_re = in_re[k];
    const double a_im = in_im[k];
    const double b_re = in_re[k + K];
    const double b_im = in_im[k + K];
    const double c_re = in_re[k + 2 * K];
    const double c_im = in_im[k + 2 * K];

    // Load twiddles
    const size_t tw_idx = TWIDDLE_OFFSET_R3_SCALAR(k);
    const double w1_re = tw[tw_idx + 0];
    const double w1_im = tw[tw_idx + 1];
    const double w2_re = tw[tw_idx + 2];
    const double w2_im = tw[tw_idx + 3];

    // Complex multiply
    const double tB_re = b_re * w1_re - b_im * w1_im;
    const double tB_im = b_re * w1_im + b_im * w1_re;
    const double tC_re = c_re * w2_re - c_im * w2_im;
    const double tC_im = c_re * w2_im + c_im * w2_re;

    // Butterfly computation
    const double sum_re = tB_re + tC_re;
    const double sum_im = tB_im + tC_im;
    const double dif_re = tB_re - tC_re;
    const double dif_im = tB_im - tC_im;

    // Rotation (backward - sign flipped)
    const double rot_re = -s_sqrt3_2 * dif_im;
    const double rot_im = s_sqrt3_2 * dif_re;

    // Common term
    const double common_re = a_re + c_half * sum_re;
    const double common_im = a_im + c_half * sum_im;

    // Output
    out_re[k] = a_re + sum_re;
    out_im[k] = a_im + sum_im;
    out_re[k + K] = common_re + rot_re;
    out_im[k + K] = common_im + rot_im;
    out_re[k + 2 * K] = common_re - rot_re;
    out_im[k + 2 * K] = common_im - rot_im;
}

//==============================================================================
// STAGE-LEVEL FUNCTIONS
//==============================================================================

/**
 * @brief Execute complete radix-3 stage - FORWARD - SCALAR
 */
FORCE_INLINE void radix3_stage_scalar_fv(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    // Simple loop - let compiler optimize
    for (size_t k = 0; k < K; ++k)
    {
        radix3_butterfly_scalar_fv(k, in_re, in_im, out_re, out_im, tw, K);
    }
}

/**
 * @brief Execute complete radix-3 stage - BACKWARD - SCALAR
 */
FORCE_INLINE void radix3_stage_scalar_bv(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    for (size_t k = 0; k < K; ++k)
    {
        radix3_butterfly_scalar_bv(k, in_re, in_im, out_re, out_im, tw, K);
    }
}

//==============================================================================
// AUTO-VECTORIZATION FRIENDLY VERSION (Optional)
//==============================================================================

/**
 * @brief Auto-vectorization friendly loop - FORWARD
 *
 * This version is structured to help compilers auto-vectorize.
 * Some modern compilers (GCC 11+, Clang 13+) can vectorize this
 * with -O3 -march=native flags.
 */
FORCE_INLINE void radix3_stage_scalar_fv_autovec(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const double c_half = C_HALF_SCALAR;
    const double s_sqrt3_2 = S_SQRT3_2_SCALAR;

    // Compiler hints for vectorization
    #ifdef __GNUC__
    #pragma GCC ivdep
    #endif
    #ifdef __clang__
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif

    for (size_t k = 0; k < K; ++k)
    {
        // Load inputs (compiler can vectorize loads)
        const double a_re = in_re[k];
        const double a_im = in_im[k];
        const double b_re = in_re[k + K];
        const double b_im = in_im[k + K];
        const double c_re = in_re[k + 2 * K];
        const double c_im = in_im[k + 2 * K];

        // Load twiddles
        const size_t tw_idx = k * 4;
        const double w1_re = tw[tw_idx + 0];
        const double w1_im = tw[tw_idx + 1];
        const double w2_re = tw[tw_idx + 2];
        const double w2_im = tw[tw_idx + 3];

        // Complex multiply
        const double tB_re = b_re * w1_re - b_im * w1_im;
        const double tB_im = b_re * w1_im + b_im * w1_re;
        const double tC_re = c_re * w2_re - c_im * w2_im;
        const double tC_im = c_re * w2_im + c_im * w2_re;

        // Butterfly
        const double sum_re = tB_re + tC_re;
        const double sum_im = tB_im + tC_im;
        const double dif_re = tB_re - tC_re;
        const double dif_im = tB_im - tC_im;

        const double rot_re = s_sqrt3_2 * dif_im;
        const double rot_im = -s_sqrt3_2 * dif_re;

        const double common_re = a_re + c_half * sum_re;
        const double common_im = a_im + c_half * sum_im;

        // Store outputs (compiler can vectorize stores)
        out_re[k] = a_re + sum_re;
        out_im[k] = a_im + sum_im;
        out_re[k + K] = common_re + rot_re;
        out_im[k + K] = common_im + rot_im;
        out_re[k + 2 * K] = common_re - rot_re;
        out_im[k + 2 * K] = common_im - rot_im;
    }
}

/**
 * @brief Auto-vectorization friendly loop - BACKWARD
 */
FORCE_INLINE void radix3_stage_scalar_bv_autovec(
    const size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT tw)
{
    const double c_half = C_HALF_SCALAR;
    const double s_sqrt3_2 = S_SQRT3_2_SCALAR;

    #ifdef __GNUC__
    #pragma GCC ivdep
    #endif
    #ifdef __clang__
    #pragma clang loop vectorize(enable) interleave(enable)
    #endif

    for (size_t k = 0; k < K; ++k)
    {
        const double a_re = in_re[k];
        const double a_im = in_im[k];
        const double b_re = in_re[k + K];
        const double b_im = in_im[k + K];
        const double c_re = in_re[k + 2 * K];
        const double c_im = in_im[k + 2 * K];

        const size_t tw_idx = k * 4;
        const double w1_re = tw[tw_idx + 0];
        const double w1_im = tw[tw_idx + 1];
        const double w2_re = tw[tw_idx + 2];
        const double w2_im = tw[tw_idx + 3];

        const double tB_re = b_re * w1_re - b_im * w1_im;
        const double tB_im = b_re * w1_im + b_im * w1_re;
        const double tC_re = c_re * w2_re - c_im * w2_im;
        const double tC_im = c_re * w2_im + c_im * w2_re;

        const double sum_re = tB_re + tC_re;
        const double sum_im = tB_im + tC_im;
        const double dif_re = tB_re - tC_re;
        const double dif_im = tB_im - tC_im;

        const double rot_re = -s_sqrt3_2 * dif_im;
        const double rot_im = s_sqrt3_2 * dif_re;

        const double common_re = a_re + c_half * sum_re;
        const double common_im = a_im + c_half * sum_im;

        out_re[k] = a_re + sum_re;
        out_im[k] = a_im + sum_im;
        out_re[k + K] = common_re + rot_re;
        out_im[k + K] = common_im + rot_im;
        out_re[k + 2 * K] = common_re - rot_re;
        out_im[k + 2 * K] = common_im - rot_im;
    }
}

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================
/*
 * SCALAR IMPLEMENTATION PHILOSOPHY:
 * ==================================
 *
 * This is NOT designed for maximum performance - it's designed for:
 * 1. PORTABILITY: Works on any architecture
 * 2. SIMPLICITY: Easy to understand and debug
 * 3. CORRECTNESS: Reference for SIMD implementations
 * 4. FALLBACK: When K < 4 or no SIMD available
 *
 * WHEN SCALAR IS ACTUALLY FASTER:
 * ================================
 * - K < 4: SIMD overhead (vector load/store) dominates
 * - Cache-bound: Memory bandwidth saturated, SIMD doesn't help
 * - Old CPUs: Some ancient processors have slow SIMD execution
 * - Non-x86: ARM/RISC-V without SIMD extensions
 *
 * AUTO-VECTORIZATION:
 * ===================
 * Modern compilers can auto-vectorize the *_autovec versions with:
 * - GCC: -O3 -march=native -ftree-vectorize
 * - Clang: -O3 -march=native
 * - MSVC: /O2 /arch:AVX2
 *
 * This can give 1.5-2× speedup over pure scalar, but still slower than
 * hand-written SIMD due to:
 * - Suboptimal instruction selection
 * - Missed optimization opportunities (constant hoisting, etc.)
 * - Conservative alias analysis
 *
 * INSTRUCTION COUNT (per butterfly):
 * ===================================
 * Scalar:              ~20-25 instructions (compiler dependent)
 * SSE2:                ~18 vector instructions (≈ 36 scalar equivalent)
 * AVX2 (FMA):          ~11 vector instructions (≈ 44 scalar equivalent)
 * AVX-512 (FMA):       ~11 vector instructions (≈ 88 scalar equivalent)
 *
 * Note: Instruction count is misleading - SIMD throughput matters more!
 *
 * EXPECTED PERFORMANCE:
 * =====================
 * Scalar baseline:     1.0× (reference)
 * Auto-vectorized:     1.5-2.0× (compiler dependent)
 * SSE2:                1.8-2.2× (2 doubles per vector)
 * AVX2:                3.5-4.2× (4 doubles per vector + FMA)
 * AVX-512:             7.0-8.5× (8 doubles per vector + FMA)
 *
 * TWIDDLE LAYOUT FOR SCALAR:
 * ===========================
 * Unlike SIMD versions which use blocked layout:
 *   SIMD:   [W1_re[0:3], W1_im[0:3], W2_re[0:3], W2_im[0:3]]
 *   Scalar: [W1_re[0], W1_im[0], W2_re[0], W2_im[0], W1_re[1], ...]
 *
 * Per-element layout is simpler for scalar access and doesn't hurt
 * performance since we're loading one element at a time anyway.
 *
 * COMPILER OPTIMIZATION TIPS:
 * ============================
 * - Use RESTRICT pointers (helps alias analysis)
 * - Mark functions FORCE_INLINE (eliminates call overhead)
 * - Use const for constants (enables constant propagation)
 * - Enable auto-vectorization with pragmas
 * - Use -ffast-math if accuracy permits (enables FMA, reordering)
 */

#endif // RADIX3_SCALAR_OPTIMIZED_H