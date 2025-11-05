/**
 * @file fft_radix32_scalar_native_soa.h
 * @brief Production Radix-32 SCALAR Native SoA - 2×16 Cooley-Tukey + Merge
 *
 * @details
 * ARCHITECTURE: 2×16 COOLEY-TUKEY DECOMPOSITION
 * ==============================================
 * Identical to AVX-2 version, but processes one k-index at a time.
 *
 * SCALAR ADAPTATIONS:
 * ===================
 * ✅ Single doubles instead of 256-bit vectors
 * ✅ U=4 software pipelining: k, k+1, k+2, k+3 (was k, k+4 in AVX-2)
 * ✅ Main loop: k += 4 (was k += 8)
 * ✅ Tail loop: k += 1 (was k += 4)
 * ✅ Prefetch distance: 64 doubles (was 16)
 * ✅ No masking needed (scalar processes any remainder)
 * ✅ FMA intrinsics for cmul (compiler generates vfmadd with -mfma)
 *
 * PRESERVED OPTIMIZATIONS FROM AVX-2:
 * ====================================
 * ✅ K-tiling (Tk=64) for L1 cache optimization
 * ✅ U=4 software pipelining (process k, k+1, k+2, k+3 together)
 * ✅ Interleaved cmul order (breaks FMA dependency chains)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (adjusted for scalar)
 * ✅ BLOCKED8/BLOCKED4 twiddle systems
 * ✅ Twiddle recurrence for large K (K > 4096)
 * ✅ All alignment hints and compiler attributes
 * ✅ Separate even/odd walkers for radix-16 sub-FFTs
 *
 * MERGE TWIDDLE SYSTEM:
 * =====================
 * Same as AVX-2 - BLOCKED8 (50% savings), BLOCKED4 (75% savings), recurrence
 *
 * @author Tugbars (Scalar version)
 * @version 1.0 (2×16 Cooley-Tukey, Scalar)
 * @date 2025
 */

#ifndef FFT_RADIX32_SCALAR_NATIVE_SOA_H
#define FFT_RADIX32_SCALAR_NATIVE_SOA_H

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>

// FMA support (optional but recommended)
#ifdef __FMA__
#include <immintrin.h>
#define HAS_FMA 1
#else
#define HAS_FMA 0
#endif

// CRITICAL: Include radix-16 scalar version for reuse
#include "fft_radix16_scalar_native_soa.h"

//==============================================================================
// COMPILER PORTABILITY
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_FMA
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_FMA __attribute__((target("fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_FMA
#endif

//==============================================================================
// CONFIGURATION (TUNED FOR RADIX-32, SCALAR ADJUSTED)
//==============================================================================

#ifndef RADIX32_MERGE_BLOCKED8_THRESHOLD_SCALAR
#define RADIX32_MERGE_BLOCKED8_THRESHOLD_SCALAR 256 // Same as AVX-2
#endif

#ifndef RADIX32_STREAM_THRESHOLD_KB_SCALAR
#define RADIX32_STREAM_THRESHOLD_KB_SCALAR 256 // Same as AVX-2
#endif

#ifndef RADIX32_PREFETCH_DISTANCE_SCALAR
#define RADIX32_PREFETCH_DISTANCE_SCALAR 64 // 64 doubles for scalar (was 16 for AVX-2)
#endif

#ifndef RADIX32_TILE_SIZE_SCALAR
#define RADIX32_TILE_SIZE_SCALAR 64 // Keep 64 - accuracy proven
#endif

#ifndef RADIX32_RECURRENCE_THRESHOLD_SCALAR
#define RADIX32_RECURRENCE_THRESHOLD_SCALAR 4096 // Same as AVX-2
#endif

//==============================================================================
// MERGE TWIDDLE STRUCTURES (SCALAR)
//==============================================================================

/**
 * @brief Merge twiddles for radix-2 combine (BLOCKED8)
 * Stores W₃₂^m for m=1..8, derives W₉..W₁₆ via negation
 */
typedef struct
{
    const double *RESTRICT re; // [8 * K]
    const double *RESTRICT im; // [8 * K]
} radix32_merge_twiddles_blocked8_scalar_t;

/**
 * @brief Merge twiddles for radix-2 combine (BLOCKED4 with recurrence)
 * Stores W₃₂^m for m=1..4, derives W₅..W₁₆ via products + negation
 */
typedef struct
{
    const double *RESTRICT re; // [4 * K]
    const double *RESTRICT im; // [4 * K]
    double delta_w_re[16];     // Phase increments for recurrence (scalar)
    double delta_w_im[16];     // Phase increments for recurrence (scalar)
    size_t K;                  // K value for this stage
    bool recurrence_enabled;   // Whether to use twiddle walking
} radix32_merge_twiddles_blocked4_scalar_t;

typedef enum
{
    RADIX32_MERGE_TW_BLOCKED8_SCALAR,
    RADIX32_MERGE_TW_BLOCKED4_SCALAR
} radix32_merge_twiddle_mode_scalar_t;

/**
 * @brief Complete radix-32 stage twiddles (combines radix-16 + merge)
 */
typedef struct
{
    // Radix-16 sub-FFT twiddles (reuse existing scalar structures)
    void *radix16_tw_opaque;
    radix16_twiddle_mode_scalar_t radix16_mode;

    // Merge twiddles (W₃₂ for combining even/odd halves)
    void *merge_tw_opaque;
    radix32_merge_twiddle_mode_scalar_t merge_mode;
} radix32_stage_twiddles_scalar_t;

//==============================================================================
// PLANNING HELPERS
//==============================================================================

FORCE_INLINE radix32_merge_twiddle_mode_scalar_t
radix32_choose_merge_twiddle_mode_scalar(size_t K)
{
    return (K <= RADIX32_MERGE_BLOCKED8_THRESHOLD_SCALAR)
               ? RADIX32_MERGE_TW_BLOCKED8_SCALAR
               : RADIX32_MERGE_TW_BLOCKED4_SCALAR;
}

FORCE_INLINE bool
radix32_should_use_merge_recurrence_scalar(size_t K)
{
    return (K > RADIX32_RECURRENCE_THRESHOLD_SCALAR);
}

/**
 * @brief NT Store Decision (same as AVX-2, overflow-safe)
 */
FORCE_INLINE bool
radix32_should_use_nt_stores_scalar(
    size_t K,
    const void *out_re,
    const void *out_im)
{
    const size_t bytes_per_k = 32 * 2 * sizeof(double); // 512 bytes (32 complex)
    const size_t threshold_k = (RADIX32_STREAM_THRESHOLD_KB_SCALAR * 1024) / bytes_per_k;

    return (K >= threshold_k) &&
           (((uintptr_t)out_re & 31) == 0) &&
           (((uintptr_t)out_im & 31) == 0);
}

//==============================================================================
// COMPLEX MATH PRIMITIVES (SCALAR WITH FMA)
//==============================================================================

/**
 * @brief Complex multiplication: (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
 * Uses FMA when available for accuracy and performance
 */
TARGET_FMA
FORCE_INLINE void
cmul_fma_scalar(double ar, double ai, double br, double bi,
                double *RESTRICT cr, double *RESTRICT ci)
{
#if HAS_FMA
    // FMA version: cr = ar*br - ai*bi, ci = ar*bi + ai*br
    *cr = _mm_cvtsd_f64(_mm_fmsub_sd(_mm_set_sd(ar), _mm_set_sd(br),
                                     _mm_mul_sd(_mm_set_sd(ai), _mm_set_sd(bi))));
    *ci = _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(ar), _mm_set_sd(bi),
                                     _mm_mul_sd(_mm_set_sd(ai), _mm_set_sd(br))));
#else
    // Standard version (compiler may auto-vectorize with -ffast-math)
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
#endif
}

/**
 * @brief Complex square: (a + bi)² = (a² - b²) + (2ab)i
 */
TARGET_FMA
FORCE_INLINE void
csquare_fma_scalar(double ar, double ai, double *RESTRICT cr, double *RESTRICT ci)
{
#if HAS_FMA
    *cr = _mm_cvtsd_f64(_mm_fmsub_sd(_mm_set_sd(ar), _mm_set_sd(ar),
                                     _mm_mul_sd(_mm_set_sd(ai), _mm_set_sd(ai))));
    *ci = _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(2.0), _mm_mul_sd(_mm_set_sd(ar), _mm_set_sd(ai)),
                                     _mm_setzero_pd()));
#else
    *cr = ar * ar - ai * ai;
    *ci = 2.0 * ar * ai;
#endif
}

//==============================================================================
// RADIX-2 BUTTERFLY (NATIVE SOA, SCALAR)
//==============================================================================

/**
 * @brief Radix-2 butterfly for combining even/odd halves (scalar)
 *
 * @details After radix-16 sub-FFTs and merge twiddle application:
 *   out[m]    = even[m] + odd[m]
 *   out[m+16] = even[m] - odd[m]
 */
FORCE_INLINE void
radix2_butterfly_combine_soa_scalar(
    const double even_re[16], const double even_im[16],
    const double odd_re[16], const double odd_im[16],
    double out_re[32], double out_im[32])
{
    // First half: out[0..15] = even + odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m] = even_re[m] + odd_re[m];
        out_im[m] = even_im[m] + odd_im[m];
    }

    // Second half: out[16..31] = even - odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m + 16] = even_re[m] - odd_re[m];
        out_im[m + 16] = even_im[m] - odd_im[m];
    }
}

//==============================================================================
// MERGE TWIDDLES: BLOCKED8 (LOAD + DERIVE, SCALAR)
//==============================================================================

/**
 * @brief Apply merge twiddles to odd half (BLOCKED8, scalar)
 *
 * @details Loads W₁..W₈, derives W₉..W₁₆ via negation.
 * Applies W₃₂^m twiddles in interleaved order (preserves optimization).
 *
 * CRITICAL: This multiplies the odd half by merge twiddles BEFORE
 * the radix-2 butterfly. The interleaved cmul order breaks FMA
 * dependency chains exactly as in AVX-2 version.
 */
TARGET_FMA
FORCE_INLINE void
apply_merge_twiddles_blocked8_scalar(
    size_t k, size_t K,
    double odd_re[16], double odd_im[16],
    const radix32_merge_twiddles_blocked8_scalar_t *RESTRICT merge_tw)
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₈
    double W_re[8], W_im[8];
    for (int r = 0; r < 8; r++)
    {
        W_re[r] = re_base[r * K + k];
        W_im[r] = im_base[r * K + k];
    }

    // Derive W₉..W₁₆ = -W₁..-W₈
    double NW_re[8], NW_im[8];
    for (int r = 0; r < 8; r++)
    {
        NW_re[r] = -W_re[r];
        NW_im[r] = -W_im[r];
    }

    // Apply twiddles in INTERLEAVED order (critical optimization!)
    // Pattern: preserves FMA dependency chain breaking from radix-16
    double tr, ti;

    // odd[0] stays unchanged (W₃₂^0 = 1)

    // Group 1: indices 1,5,9,13
    cmul_fma_scalar(odd_re[1], odd_im[1], W_re[0], W_im[0], &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_scalar(odd_re[5], odd_im[5], W_re[4], W_im[4], &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_scalar(odd_re[9], odd_im[9], NW_re[0], NW_im[0], &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_scalar(odd_re[13], odd_im[13], NW_re[4], NW_im[4], &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    // Group 2: indices 2,6,10,14
    cmul_fma_scalar(odd_re[2], odd_im[2], W_re[1], W_im[1], &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_scalar(odd_re[6], odd_im[6], W_re[5], W_im[5], &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_scalar(odd_re[10], odd_im[10], NW_re[1], NW_im[1], &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_scalar(odd_re[14], odd_im[14], NW_re[5], NW_im[5], &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    // Group 3: indices 3,7,11,15
    cmul_fma_scalar(odd_re[3], odd_im[3], W_re[2], W_im[2], &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_scalar(odd_re[7], odd_im[7], W_re[6], W_im[6], &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_scalar(odd_re[11], odd_im[11], NW_re[2], NW_im[2], &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_scalar(odd_re[15], odd_im[15], NW_re[6], NW_im[6], &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    // Group 4: indices 4,8,12
    cmul_fma_scalar(odd_re[4], odd_im[4], W_re[3], W_im[3], &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_scalar(odd_re[8], odd_im[8], W_re[7], W_im[7], &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_scalar(odd_re[12], odd_im[12], NW_re[3], NW_im[3], &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;
}

//==============================================================================
// MERGE TWIDDLES: BLOCKED4 (LOAD + DERIVE, SCALAR)
//==============================================================================

/**
 * @brief Apply merge twiddles to odd half (BLOCKED4, scalar)
 *
 * @details Loads W₁..W₄, derives W₅..W₁₆ via products + negation.
 * Same interleaved order as BLOCKED8.
 */
TARGET_FMA
FORCE_INLINE void
apply_merge_twiddles_blocked4_scalar(
    size_t k, size_t K,
    double odd_re[16], double odd_im[16],
    const radix32_merge_twiddles_blocked4_scalar_t *RESTRICT merge_tw)
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₄
    double W1r = re_base[0 * K + k];
    double W1i = im_base[0 * K + k];
    double W2r = re_base[1 * K + k];
    double W2i = im_base[1 * K + k];
    double W3r = re_base[2 * K + k];
    double W3i = im_base[2 * K + k];
    double W4r = re_base[3 * K + k];
    double W4i = im_base[3 * K + k];

    // Derive W₅..W₈ via products
    double W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_scalar(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul_fma_scalar(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_scalar(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_scalar(W4r, W4i, &W8r, &W8i);

    // Derive W₉..W₁₆ via negation
    double NW1r = -W1r, NW1i = -W1i;
    double NW2r = -W2r, NW2i = -W2i;
    double NW3r = -W3r, NW3i = -W3i;
    double NW4r = -W4r, NW4i = -W4i;
    double NW5r = -W5r, NW5i = -W5i;
    double NW6r = -W6r, NW6i = -W6i;
    double NW7r = -W7r, NW7i = -W7i;

    // Apply in interleaved order (same pattern as BLOCKED8)
    double tr, ti;

    // odd[0] unchanged

    cmul_fma_scalar(odd_re[1], odd_im[1], W1r, W1i, &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_scalar(odd_re[5], odd_im[5], W5r, W5i, &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_scalar(odd_re[9], odd_im[9], NW1r, NW1i, &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_scalar(odd_re[13], odd_im[13], NW5r, NW5i, &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    cmul_fma_scalar(odd_re[2], odd_im[2], W2r, W2i, &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_scalar(odd_re[6], odd_im[6], W6r, W6i, &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_scalar(odd_re[10], odd_im[10], NW2r, NW2i, &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_scalar(odd_re[14], odd_im[14], NW6r, NW6i, &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    cmul_fma_scalar(odd_re[3], odd_im[3], W3r, W3i, &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_scalar(odd_re[7], odd_im[7], W7r, W7i, &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_scalar(odd_re[11], odd_im[11], NW3r, NW3i, &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_scalar(odd_re[15], odd_im[15], NW7r, NW7i, &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    cmul_fma_scalar(odd_re[4], odd_im[4], W4r, W4i, &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_scalar(odd_re[8], odd_im[8], W8r, W8i, &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_scalar(odd_re[12], odd_im[12], NW4r, NW4i, &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;
}

//==============================================================================
// MERGE TWIDDLES: RECURRENCE INITIALIZATION (SCALAR)
//==============================================================================

/**
 * @brief Initialize merge twiddle recurrence state at tile boundary (scalar)
 *
 * @details Loads W₁..W₄, derives W₅..W₁₆ via products + negation.
 * Identical structure to AVX-2 recurrence initialization.
 */
TARGET_FMA
FORCE_INLINE void
radix32_init_merge_recurrence_state_scalar(
    size_t k, size_t K,
    const radix32_merge_twiddles_blocked4_scalar_t *RESTRICT merge_tw,
    double w_state_re[16], double w_state_im[16])
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₄
    double W1r = re_base[0 * K + k];
    double W1i = im_base[0 * K + k];
    double W2r = re_base[1 * K + k];
    double W2i = im_base[1 * K + k];
    double W3r = re_base[2 * K + k];
    double W3i = im_base[2 * K + k];
    double W4r = re_base[3 * K + k];
    double W4i = im_base[3 * K + k];

    // Derive W₅..W₈
    double W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_scalar(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul_fma_scalar(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_scalar(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_scalar(W4r, W4i, &W8r, &W8i);

    // W₀ = 1 (identity)
    w_state_re[0] = 1.0;
    w_state_im[0] = 0.0;

    // Store W₁..W₈
    w_state_re[1] = W1r;
    w_state_im[1] = W1i;
    w_state_re[2] = W2r;
    w_state_im[2] = W2i;
    w_state_re[3] = W3r;
    w_state_im[3] = W3i;
    w_state_re[4] = W4r;
    w_state_im[4] = W4i;
    w_state_re[5] = W5r;
    w_state_im[5] = W5i;
    w_state_re[6] = W6r;
    w_state_im[6] = W6i;
    w_state_re[7] = W7r;
    w_state_im[7] = W7i;
    w_state_re[8] = W8r;
    w_state_im[8] = W8i;

    // W₉..W₁₆ = -W₁..-W₈
    for (int r = 1; r <= 8; r++)
    {
        w_state_re[8 + r] = -w_state_re[r];
        w_state_im[8 + r] = -w_state_im[r];
    }
}

//==============================================================================
// MERGE TWIDDLES: RECURRENCE (APPLY + ADVANCE, SCALAR)
//==============================================================================

/**
 * @brief Apply merge twiddles with tile-local recurrence (scalar)
 *
 * @details Identical recurrence strategy to AVX-2:
 * - Refresh accurate twiddles at tile boundaries
 * - Advance via w ← w × δw within tile
 * - Maintains <1e-14 accuracy over 64-step tiles
 *
 * CRITICAL: Interleaved cmul order preserved!
 */
TARGET_FMA
FORCE_INLINE void
apply_merge_twiddles_recur_scalar(
    size_t k, size_t k_tile_start, bool is_tile_start,
    double odd_re[16], double odd_im[16],
    const radix32_merge_twiddles_blocked4_scalar_t *RESTRICT merge_tw,
    double w_state_re[16], double w_state_im[16],
    const double *delta_w_re, const double *delta_w_im)
{
    if (is_tile_start)
    {
        // REFRESH: Load accurate twiddles from memory at tile boundary
        radix32_init_merge_recurrence_state_scalar(k, merge_tw->K, merge_tw,
                                                    w_state_re, w_state_im);
    }

    // Apply current twiddles (interleaved order - preserves optimization!)
    double tr, ti;

    // odd[0] unchanged (W₀ = 1)

    // Group 1: indices 1,5,9,13
    cmul_fma_scalar(odd_re[1], odd_im[1], w_state_re[1], w_state_im[1], &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_scalar(odd_re[5], odd_im[5], w_state_re[5], w_state_im[5], &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_scalar(odd_re[9], odd_im[9], w_state_re[9], w_state_im[9], &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_scalar(odd_re[13], odd_im[13], w_state_re[13], w_state_im[13], &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    // Group 2: indices 2,6,10,14
    cmul_fma_scalar(odd_re[2], odd_im[2], w_state_re[2], w_state_im[2], &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_scalar(odd_re[6], odd_im[6], w_state_re[6], w_state_im[6], &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_scalar(odd_re[10], odd_im[10], w_state_re[10], w_state_im[10], &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_scalar(odd_re[14], odd_im[14], w_state_re[14], w_state_im[14], &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    // Group 3: indices 3,7,11,15
    cmul_fma_scalar(odd_re[3], odd_im[3], w_state_re[3], w_state_im[3], &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_scalar(odd_re[7], odd_im[7], w_state_re[7], w_state_im[7], &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_scalar(odd_re[11], odd_im[11], w_state_re[11], w_state_im[11], &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_scalar(odd_re[15], odd_im[15], w_state_re[15], w_state_im[15], &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    // Group 4: indices 4,8,12
    cmul_fma_scalar(odd_re[4], odd_im[4], w_state_re[4], w_state_im[4], &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_scalar(odd_re[8], odd_im[8], w_state_re[8], w_state_im[8], &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_scalar(odd_re[12], odd_im[12], w_state_re[12], w_state_im[12], &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;

    // ADVANCE: w ← w × δw (for next iteration within tile)
    // CRITICAL: Planner MUST set delta_w[0] = 1+0i to keep W₀ stationary
    for (int r = 0; r < 16; r++)
    {
        double w_new_re, w_new_im;
        cmul_fma_scalar(w_state_re[r], w_state_im[r],
                        delta_w_re[r], delta_w_im[r],
                        &w_new_re, &w_new_im);
        w_state_re[r] = w_new_re;
        w_state_im[r] = w_new_im;
    }
}

//==============================================================================
// PREFETCH HELPERS FOR SCALAR
//==============================================================================

/**
 * @brief Prefetch helper (scalar)
 * Uses compiler intrinsics or hints
 */
FORCE_INLINE void
prefetch_scalar(const void *addr)
{
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3); // Read, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch((const char *)addr, _MM_HINT_T0);
#else
    (void)addr; // No-op if no prefetch support
#endif
}

/**
 * @brief Prefetch inputs + merge twiddles for BLOCKED8
 */
#define RADIX32_PREFETCH_MERGE_BLOCKED8_SCALAR(k_next, k_limit, K, in_re, in_im, merge_tw) \
    do                                                                                      \
    {                                                                                       \
        if ((k_next) < (k_limit))                                                           \
        {                                                                                   \
            for (int _r = 16; _r < 32; _r++)                                                \
            {                                                                               \
                prefetch_scalar(&(in_re)[(k_next) + _r * (K)]);                             \
                prefetch_scalar(&(in_im)[(k_next) + _r * (K)]);                             \
            }                                                                               \
            for (int _b = 0; _b < 8; _b++)                                                  \
            {                                                                               \
                prefetch_scalar(&(merge_tw)->re[_b * (K) + (k_next)]);                      \
                prefetch_scalar(&(merge_tw)->im[_b * (K) + (k_next)]);                      \
            }                                                                               \
        }                                                                                   \
    } while (0)

/**
 * @brief Prefetch inputs + merge twiddles for BLOCKED4
 */
#define RADIX32_PREFETCH_MERGE_BLOCKED4_SCALAR(k_next, k_limit, K, in_re, in_im, merge_tw) \
    do                                                                                      \
    {                                                                                       \
        if ((k_next) < (k_limit))                                                           \
        {                                                                                   \
            for (int _r = 16; _r < 32; _r++)                                                \
            {                                                                               \
                prefetch_scalar(&(in_re)[(k_next) + _r * (K)]);                             \
                prefetch_scalar(&(in_im)[(k_next) + _r * (K)]);                             \
            }                                                                               \
            for (int _b = 0; _b < 4; _b++)                                                  \
            {                                                                               \
                prefetch_scalar(&(merge_tw)->re[_b * (K) + (k_next)]);                      \
                prefetch_scalar(&(merge_tw)->im[_b * (K) + (k_next)]);                      \
            }                                                                               \
        }                                                                                   \
    } while (0)

/**
 * @brief Prefetch inputs only (for recurrence mode - no twiddle loads)
 */
#define RADIX32_PREFETCH_MERGE_RECURRENCE_SCALAR(k_next, k_limit, K, in_re, in_im) \
    do                                                                              \
    {                                                                               \
        if ((k_next) < (k_limit))                                                   \
        {                                                                           \
            for (int _r = 16; _r < 32; _r++)                                        \
            {                                                                       \
                prefetch_scalar(&(in_re)[(k_next) + _r * (K)]);                     \
                prefetch_scalar(&(in_im)[(k_next) + _r * (K)]);                     \
            }                                                                       \
        }                                                                           \
    } while (0)

//==============================================================================
// LOAD/STORE FOR 32 LANES (SCALAR)
//==============================================================================

/**
 * @brief Store 32 complex values (scalar)
 */
FORCE_INLINE void
store_32_lanes_soa_scalar(size_t k, size_t K,
                          double *RESTRICT out_re, double *RESTRICT out_im,
                          const double y_re[32], const double y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 32; r++)
    {
        out_re_aligned[k + r * K] = y_re[r];
        out_im_aligned[k + r * K] = y_im[r];
    }
}

/**
 * @brief Store with non-temporal hint (scalar)
 * Note: Actual NT stores require compiler/platform support
 */
FORCE_INLINE void
store_32_lanes_soa_scalar_stream(size_t k, size_t K,
                                 double *RESTRICT out_re, double *RESTRICT out_im,
                                 const double y_re[32], const double y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

#if defined(__GNUC__) || defined(__clang__)
    // Use non-temporal store hint via builtin
    for (int r = 0; r < 32; r++)
    {
        __builtin_nontemporal_store(y_re[r], &out_re_aligned[k + r * K]);
        __builtin_nontemporal_store(y_im[r], &out_im_aligned[k + r * K]);
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    // MSVC x64: use _mm_stream_si64
    for (int r = 0; r < 32; r++)
    {
        _mm_stream_si64((long long *)&out_re_aligned[k + r * K], *(long long *)&y_re[r]);
        _mm_stream_si64((long long *)&out_im_aligned[k + r * K], *(long long *)&y_im[r]);
    }
#else
    // Fallback to regular stores
    for (int r = 0; r < 32; r++)
    {
        out_re_aligned[k + r * K] = y_re[r];
        out_im_aligned[k + r * K] = y_im[r];
    }
#endif
}

//==============================================================================
// COMPLETE STAGE DRIVERS: FORWARD (BLOCKED8, SCALAR)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - BLOCKED8 Merge (SCALAR)
 *
 * SCALAR ADAPTATIONS:
 * - Main loop: k += 4 (process k, k+1, k+2, k+3 for U=4 ILP)
 * - Tail loop: k += 1 (scalar handles any remainder naturally)
 * - Prefetch distance: 64 doubles (was 16 for AVX-2)
 * - Single doubles instead of vectors
 *
 * PRESERVED OPTIMIZATIONS:
 * - K-tiling (Tk=64)
 * - U=4 software pipelining (4-way ILP via k, k+1, k+2, k+3)
 * - Prefetch (adjusted for scalar)
 * - Adaptive NT stores
 * - All radix-16 optimizations inherited
 */
TARGET_FMA
FORCE_INLINE void
radix32_stage_dit_forward_blocked8_merge_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_scalar_t *RESTRICT stage_tw)
{
    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_SCALAR; // 64 for scalar
    const size_t tile_size = RADIX32_TILE_SIZE_SCALAR;             // 64

    const bool use_nt_stores = radix32_should_use_nt_stores_scalar(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_scalar_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked8_scalar_t *merge_tw =
        (const radix32_merge_twiddles_blocked8_scalar_t *)stage_tw->merge_tw_opaque;

    // Cast to specific types for branching
    const radix16_stage_twiddles_blocked8_scalar_t *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            ? (const radix16_stage_twiddles_blocked8_scalar_t *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_scalar_t *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            ? (const radix16_stage_twiddles_blocked4_scalar_t *)radix16_tw
            : NULL;

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=4 LOOP: k += 4 (process k, k+1, k+2, k+3 together)
        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            // Prefetch next iteration
            size_t k_next = k + 4 + prefetch_dist;
            if (k_next < k_end)
            {
                // Prefetch even half (radix-16 inputs + twiddles)
                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b8);
                }
                else // RADIX16_TW_BLOCKED4_SCALAR
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b4);
                }

                // Prefetch odd half + merge twiddles
                RADIX32_PREFETCH_MERGE_BLOCKED8_SCALAR(k_next, k_end, K,
                                                       in_re_aligned, in_im_aligned, merge_tw);
            }

            // ==================== PROCESS k (1st of 4) ====================
            {
                // Even half (r=0..15): radix-16 butterfly
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im,
                                                         tw16b8);
                }
                else // RADIX16_TW_BLOCKED4_SCALAR
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                // Odd half (r=16..31): radix-16 butterfly
                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                // Apply merge twiddles to odd half
                apply_merge_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                     merge_tw);

                // Radix-2 combine
                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                // Store
                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                     y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+1 (2nd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 1, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 1 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 1 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 1, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 1, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+2 (3rd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 2, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 2 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 2 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 2, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 2, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+3 (4th of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 3, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 3 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 3 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 3, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 3, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }
        }

        // TAIL LOOP: k += 1 (handle any remainder)
        for (; k < k_end; k++)
        {
            double even_re[16], even_im[16];
            load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            {
                apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im, tw16b8);
            }
            else
            {
                apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im, tw16b4);
            }
            radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

            double odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = in_re_aligned[k + (r + 16) * K];
                odd_im[r] = in_im_aligned[k + (r + 16) * K];
            }

            if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            {
                apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im, tw16b8);
            }
            else
            {
                apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im, tw16b4);
            }
            radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

            apply_merge_twiddles_blocked8_scalar(k, K, odd_re, odd_im, merge_tw);

            double y_re[32], y_im[32];
            radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }
    }

    if (use_nt_stores)
    {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("sfence" ::: "memory");
#elif defined(_MSC_VER)
        _mm_sfence();
#endif
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: FORWARD (BLOCKED4 WITH RECURRENCE, SCALAR)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - BLOCKED4 Merge (SCALAR WITH R16 RECURRENCE)
 *
 * CRITICAL FEATURES:
 * - Detects radix-16 recurrence mode
 * - Maintains separate walker state for even/odd halves
 * - Branches on r16_recur in twiddle application
 * 
 * SCALAR ADAPTATIONS:
 * - Main loop: k += 4, U=4 with k, k+1, k+2, k+3
 * - Tail loop: k += 1 (no masking needed)
 * - Prefetch: 64 doubles ahead
 * - Separate walkers: [16] elements each
 */
TARGET_FMA
FORCE_INLINE void
radix32_stage_dit_forward_blocked4_merge_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_scalar_t *RESTRICT stage_tw,
    bool use_merge_recurrence,
    const double *RESTRICT merge_delta_w_re,
    const double *RESTRICT merge_delta_w_im)
{
    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_SCALAR;
    const size_t tile_size = RADIX32_TILE_SIZE_SCALAR;

    const bool use_nt_stores = radix32_should_use_nt_stores_scalar(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_scalar_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked4_scalar_t *merge_tw =
        (const radix32_merge_twiddles_blocked4_scalar_t *)stage_tw->merge_tw_opaque;

    // CRITICAL: Detect radix-16 recurrence mode
    const radix16_stage_twiddles_blocked8_scalar_t *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            ? (const radix16_stage_twiddles_blocked8_scalar_t *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_scalar_t *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            ? (const radix16_stage_twiddles_blocked4_scalar_t *)radix16_tw
            : NULL;

    bool r16_recur = (r16_mode == RADIX16_TW_BLOCKED4_SCALAR) && tw16b4->recurrence_enabled;

    // Merge recurrence state
    double merge_w_state_re[16], merge_w_state_im[16];

    // CRITICAL: Radix-16 recurrence state (separate for even/odd halves)
    // [16] elements - planner MUST set delta_w[15] to safe value
    double r16_w_even_re[16], r16_w_even_im[16]; // For r=0..15
    double r16_w_odd_re[16], r16_w_odd_im[16];   // For r=16..31

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=4 LOOP: k += 4
        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            // Prefetch next iteration
            size_t k_next = k + 4 + prefetch_dist;
            if (k_next < k_end)
            {
                // Prefetch even half (radix-16)
                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b8);
                }
                else if (r16_recur)
                {
                    RADIX16_PREFETCH_NEXT_RECURRENCE_SCALAR(k_next, k_end, K,
                                                            in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b4);
                }

                // Prefetch merge
                if (use_merge_recurrence)
                {
                    RADIX32_PREFETCH_MERGE_RECURRENCE_SCALAR(k_next, k_end, K,
                                                             in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX32_PREFETCH_MERGE_BLOCKED4_SCALAR(k_next, k_end, K,
                                                           in_re_aligned, in_im_aligned, merge_tw);
                }
            }

            bool is_tile_start = (k == k_tile);

            // ==================== PROCESS k (1st of 4) ====================
            {
                // Even half: radix-16 butterfly
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                // CRITICAL: Branch on radix-16 recurrence mode
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im,
                                                         tw16b4);
                }
                else // RADIX16_TW_BLOCKED8_SCALAR
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                // Odd half: radix-16 butterfly
                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + (r + 16) * K];
                }

                // CRITICAL: Odd half uses separate walker
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                // Apply merge twiddles
                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         merge_tw);
                }

                // Radix-2 combine
                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                // Store
                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                     y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+1 (2nd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 1, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                // CRITICAL: is_tile_start = false for k+1, k+2, k+3
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 1 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 1 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 1, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 1, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+2 (3rd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 2, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 2 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 2 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 2, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 2, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+3 (4th of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 3, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 3 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 3 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 3, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 3, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }
        }

        // TAIL LOOP: k += 1 (handle any remainder - scalar handles easily!)
        for (; k < k_end; k++)
        {
            double even_re[16], even_im[16];
            load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_recur)
            {
                apply_stage_twiddles_recur_scalar(k, k_tile, false,
                                                  even_re, even_im, tw16b4,
                                                  r16_w_even_re, r16_w_even_im,
                                                  tw16b4->delta_w_re, tw16b4->delta_w_im);
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            {
                apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im, tw16b4);
            }
            else
            {
                apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im, tw16b8);
            }
            radix16_complete_butterfly_forward_soa_scalar(even_re, even_im);

            double odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = in_re_aligned[k + (r + 16) * K];
                odd_im[r] = in_im_aligned[k + (r + 16) * K];
            }

            if (r16_recur)
            {
                apply_stage_twiddles_recur_scalar(k, k_tile, false,
                                                  odd_re, odd_im, tw16b4,
                                                  r16_w_odd_re, r16_w_odd_im,
                                                  tw16b4->delta_w_re, tw16b4->delta_w_im);
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            {
                apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im, tw16b4);
            }
            else
            {
                apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im, tw16b8);
            }
            radix16_complete_butterfly_forward_soa_scalar(odd_re, odd_im);

            if (use_merge_recurrence)
            {
                apply_merge_twiddles_recur_scalar(k, k_tile, false, odd_re, odd_im,
                                                  merge_tw, merge_w_state_re, merge_w_state_im,
                                                  merge_delta_w_re, merge_delta_w_im);
            }
            else
            {
                apply_merge_twiddles_blocked4_scalar(k, K, odd_re, odd_im, merge_tw);
            }

            double y_re[32], y_im[32];
            radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }
    }

    if (use_nt_stores)
    {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("sfence" ::: "memory");
#elif defined(_MSC_VER)
        _mm_sfence();
#endif
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: BACKWARD (BLOCKED8, SCALAR)
//==============================================================================

/**
 * @brief Radix-32 DIT Backward Stage - BLOCKED8 Merge (SCALAR)
 *
 * SCALAR ADAPTATIONS:
 * - Main loop: k += 4 (process k, k+1, k+2, k+3 for U=4 ILP)
 * - Tail loop: k += 1 (scalar handles any remainder naturally)
 * - Prefetch distance: 64 doubles (was 16 for AVX-2)
 */
TARGET_FMA
FORCE_INLINE void
radix32_stage_dit_backward_blocked8_merge_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_scalar_t *RESTRICT stage_tw)
{
    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_SCALAR;
    const size_t tile_size = RADIX32_TILE_SIZE_SCALAR;

    const bool use_nt_stores = radix32_should_use_nt_stores_scalar(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_scalar_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked8_scalar_t *merge_tw =
        (const radix32_merge_twiddles_blocked8_scalar_t *)stage_tw->merge_tw_opaque;

    const radix16_stage_twiddles_blocked8_scalar_t *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            ? (const radix16_stage_twiddles_blocked8_scalar_t *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_scalar_t *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            ? (const radix16_stage_twiddles_blocked4_scalar_t *)radix16_tw
            : NULL;

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            size_t k_next = k + 4 + prefetch_dist;
            if (k_next < k_end)
            {
                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b8);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b4);
                }

                RADIX32_PREFETCH_MERGE_BLOCKED8_SCALAR(k_next, k_end, K,
                                                       in_re_aligned, in_im_aligned, merge_tw);
            }

            // ==================== PROCESS k (1st of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                     y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+1 (2nd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 1, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 1 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 1 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 1, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 1, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+2 (3rd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 2, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 2 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 2 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 2, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 2, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+3 (4th of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 3, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, even_re, even_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, even_re, even_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 3 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 3 + (r + 16) * K];
                }

                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b8);
                }
                else
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b4);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                apply_merge_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                     merge_tw);

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 3, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 3, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }
        }

        for (; k < k_end; k++)
        {
            double even_re[16], even_im[16];
            load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            {
                apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im, tw16b8);
            }
            else
            {
                apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im, tw16b4);
            }
            radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

            double odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = in_re_aligned[k + (r + 16) * K];
                odd_im[r] = in_im_aligned[k + (r + 16) * K];
            }

            if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            {
                apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im, tw16b8);
            }
            else
            {
                apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im, tw16b4);
            }
            radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

            apply_merge_twiddles_blocked8_scalar(k, K, odd_re, odd_im, merge_tw);

            double y_re[32], y_im[32];
            radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }
    }

    if (use_nt_stores)
    {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("sfence" ::: "memory");
#elif defined(_MSC_VER)
        _mm_sfence();
#endif
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: BACKWARD (BLOCKED4 WITH RECURRENCE, SCALAR)
//==============================================================================

/**
 * @brief Radix-32 DIT Backward Stage - BLOCKED4 Merge (SCALAR WITH R16 RECURRENCE)
 *
 * CRITICAL FEATURES:
 * - Detects radix-16 recurrence mode
 * - Maintains separate walker state for even/odd halves
 * - Branches on r16_recur in twiddle application
 * 
 * SCALAR ADAPTATIONS:
 * - Main loop: k += 4, U=4 with k, k+1, k+2, k+3
 * - Tail loop: k += 1 (no masking needed)
 * - Prefetch: 64 doubles ahead
 */
TARGET_FMA
FORCE_INLINE void
radix32_stage_dit_backward_blocked4_merge_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_scalar_t *RESTRICT stage_tw,
    bool use_merge_recurrence,
    const double *RESTRICT merge_delta_w_re,
    const double *RESTRICT merge_delta_w_im)
{
    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE_SCALAR;
    const size_t tile_size = RADIX32_TILE_SIZE_SCALAR;

    const bool use_nt_stores = radix32_should_use_nt_stores_scalar(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_scalar_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked4_scalar_t *merge_tw =
        (const radix32_merge_twiddles_blocked4_scalar_t *)stage_tw->merge_tw_opaque;

    const radix16_stage_twiddles_blocked8_scalar_t *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
            ? (const radix16_stage_twiddles_blocked8_scalar_t *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_scalar_t *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            ? (const radix16_stage_twiddles_blocked4_scalar_t *)radix16_tw
            : NULL;

    bool r16_recur = (r16_mode == RADIX16_TW_BLOCKED4_SCALAR) && tw16b4->recurrence_enabled;

    double merge_w_state_re[16], merge_w_state_im[16];
    
    // CRITICAL: [16] elements - planner sets delta_w[15] to safe value
    double r16_w_even_re[16], r16_w_even_im[16];
    double r16_w_odd_re[16], r16_w_odd_im[16];

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            size_t k_next = k + 4 + prefetch_dist;
            if (k_next < k_end)
            {
                if (r16_mode == RADIX16_TW_BLOCKED8_SCALAR)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b8);
                }
                else if (r16_recur)
                {
                    RADIX16_PREFETCH_NEXT_RECURRENCE_SCALAR(k_next, k_end, K,
                                                            in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_SCALAR(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned, tw16b4);
                }

                if (use_merge_recurrence)
                {
                    RADIX32_PREFETCH_MERGE_RECURRENCE_SCALAR(k_next, k_end, K,
                                                             in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX32_PREFETCH_MERGE_BLOCKED4_SCALAR(k_next, k_end, K,
                                                           in_re_aligned, in_im_aligned, merge_tw);
                }
            }

            bool is_tile_start = (k == k_tile);

            // ==================== PROCESS k (1st of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k, k_tile, is_tile_start,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                     y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+1 (2nd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 1, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 1 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 1 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 1, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 1, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 1, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 1, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 1, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+2 (3rd of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 2, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 2 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 2 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 2, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 2, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 2, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 2, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 2, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }

            // ==================== PROCESS k+3 (4th of 4) ====================
            {
                double even_re[16], even_im[16];
                load_16_lanes_soa_scalar(k + 3, K, in_re_aligned, in_im_aligned,
                                         even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      even_re, even_im, tw16b4,
                                                      r16_w_even_re, r16_w_even_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, even_re, even_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, even_re, even_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

                double odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = in_re_aligned[k + 3 + (r + 16) * K];
                    odd_im[r] = in_im_aligned[k + 3 + (r + 16) * K];
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      odd_re, odd_im, tw16b4,
                                                      r16_w_odd_re, r16_w_odd_im,
                                                      tw16b4->delta_w_re, tw16b4->delta_w_im);
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
                {
                    apply_stage_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b4);
                }
                else
                {
                    apply_stage_twiddles_blocked8_scalar(k + 3, K, odd_re, odd_im,
                                                         tw16b8);
                }
                radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_scalar(k + 3, k_tile, false,
                                                      odd_re, odd_im, merge_tw,
                                                      merge_w_state_re, merge_w_state_im,
                                                      merge_delta_w_re, merge_delta_w_im);
                }
                else
                {
                    apply_merge_twiddles_blocked4_scalar(k + 3, K, odd_re, odd_im,
                                                         merge_tw);
                }

                double y_re[32], y_im[32];
                radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im,
                                                    y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_scalar_stream(k + 3, K, out_re_aligned,
                                                     out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_scalar(k + 3, K, out_re_aligned, out_im_aligned,
                                              y_re, y_im);
                }
            }
        }

        for (; k < k_end; k++)
        {
            double even_re[16], even_im[16];
            load_16_lanes_soa_scalar(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_recur)
            {
                apply_stage_twiddles_recur_scalar(k, k_tile, false,
                                                  even_re, even_im, tw16b4,
                                                  r16_w_even_re, r16_w_even_im,
                                                  tw16b4->delta_w_re, tw16b4->delta_w_im);
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            {
                apply_stage_twiddles_blocked4_scalar(k, K, even_re, even_im, tw16b4);
            }
            else
            {
                apply_stage_twiddles_blocked8_scalar(k, K, even_re, even_im, tw16b8);
            }
            radix16_complete_butterfly_backward_soa_scalar(even_re, even_im);

            double odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = in_re_aligned[k + (r + 16) * K];
                odd_im[r] = in_im_aligned[k + (r + 16) * K];
            }

            if (r16_recur)
            {
                apply_stage_twiddles_recur_scalar(k, k_tile, false,
                                                  odd_re, odd_im, tw16b4,
                                                  r16_w_odd_re, r16_w_odd_im,
                                                  tw16b4->delta_w_re, tw16b4->delta_w_im);
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4_SCALAR)
            {
                apply_stage_twiddles_blocked4_scalar(k, K, odd_re, odd_im, tw16b4);
            }
            else
            {
                apply_stage_twiddles_blocked8_scalar(k, K, odd_re, odd_im, tw16b8);
            }
            radix16_complete_butterfly_backward_soa_scalar(odd_re, odd_im);

            if (use_merge_recurrence)
            {
                apply_merge_twiddles_recur_scalar(k, k_tile, false, odd_re, odd_im,
                                                  merge_tw, merge_w_state_re, merge_w_state_im,
                                                  merge_delta_w_re, merge_delta_w_im);
            }
            else
            {
                apply_merge_twiddles_blocked4_scalar(k, K, odd_re, odd_im, merge_tw);
            }

            double y_re[32], y_im[32];
            radix2_butterfly_combine_soa_scalar(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_scalar_stream(k, K, out_re_aligned, out_im_aligned,
                                                 y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_scalar(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }
    }

    if (use_nt_stores)
    {
#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__("sfence" ::: "memory");
#elif defined(_MSC_VER)
        _mm_sfence();
#endif
    }
}

//==============================================================================
// PUBLIC API (SCALAR)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - Public API (SCALAR)
 *
 * @param K Number of k-indices per radix lane
 * @param in_re Input real part (SoA: [32][K])
 * @param in_im Input imaginary part (SoA: [32][K])
 * @param out_re Output real part (SoA: [32][K])
 * @param out_im Output imaginary part (SoA: [32][K])
 * @param stage_tw_opaque Opaque pointer to radix32_stage_twiddles_scalar_t
 *
 * @note Twiddle structures should be prepared during planning phase:
 *       - radix16_tw: Stage twiddles for both radix-16 sub-FFTs
 *       - merge_tw: W₃₂^m twiddles for combining even/odd halves
 *       - delta_w: Phase increments for recurrence (if enabled)
 *
 * CRITICAL PLANNER REQUIREMENTS:
 * - Radix-16 delta_w[16]: indices 0..14 for W₁..W₁₅, index 15 = identity
 * - Merge delta_w[16]: index 0 = identity (W₀), indices 1..15 for W₁..W₁₅
 */
TARGET_FMA
void radix32_stage_dit_forward_soa_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque)
{
    const radix32_stage_twiddles_scalar_t *stage_tw =
        (const radix32_stage_twiddles_scalar_t *)stage_tw_opaque;

    if (stage_tw->merge_mode == RADIX32_MERGE_TW_BLOCKED8_SCALAR)
    {
        radix32_stage_dit_forward_blocked8_merge_scalar(
            K, in_re, in_im, out_re, out_im, stage_tw);
    }
    else // RADIX32_MERGE_TW_BLOCKED4_SCALAR
    {
        const radix32_merge_twiddles_blocked4_scalar_t *merge_tw =
            (const radix32_merge_twiddles_blocked4_scalar_t *)stage_tw->merge_tw_opaque;

        if (merge_tw->recurrence_enabled)
        {
            radix32_stage_dit_forward_blocked4_merge_scalar(
                K, in_re, in_im, out_re, out_im,
                stage_tw, true, merge_tw->delta_w_re, merge_tw->delta_w_im);
        }
        else
        {
            radix32_stage_dit_forward_blocked4_merge_scalar(
                K, in_re, in_im, out_re, out_im,
                stage_tw, false, NULL, NULL);
        }
    }
}

/**
 * @brief Radix-32 DIT Backward Stage - Public API (SCALAR)
 *
 * @param K Number of k-indices per radix lane
 * @param in_re Input real part (SoA: [32][K])
 * @param in_im Input imaginary part (SoA: [32][K])
 * @param out_re Output real part (SoA: [32][K])
 * @param out_im Output imaginary part (SoA: [32][K])
 * @param stage_tw_opaque Opaque pointer to radix32_stage_twiddles_scalar_t
 */
TARGET_FMA
void radix32_stage_dit_backward_soa_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque)
{
    const radix32_stage_twiddles_scalar_t *stage_tw =
        (const radix32_stage_twiddles_scalar_t *)stage_tw_opaque;

    if (stage_tw->merge_mode == RADIX32_MERGE_TW_BLOCKED8_SCALAR)
    {
        radix32_stage_dit_backward_blocked8_merge_scalar(
            K, in_re, in_im, out_re, out_im, stage_tw);
    }
    else // RADIX32_MERGE_TW_BLOCKED4_SCALAR
    {
        const radix32_merge_twiddles_blocked4_scalar_t *merge_tw =
            (const radix32_merge_twiddles_blocked4_scalar_t *)stage_tw->merge_tw_opaque;

        if (merge_tw->recurrence_enabled)
        {
            radix32_stage_dit_backward_blocked4_merge_scalar(
                K, in_re, in_im, out_re, out_im,
                stage_tw, true, merge_tw->delta_w_re, merge_tw->delta_w_im);
        }
        else
        {
            radix32_stage_dit_backward_blocked4_merge_scalar(
                K, in_re, in_im, out_re, out_im,
                stage_tw, false, NULL, NULL);
        }
    }
}

#endif // FFT_RADIX32_SCALAR_NATIVE_SOA_H


/*

🚨 Correctness: W9..W16 are not -W1..-W8

For N = 32, the roots satisfy:

W32^(m+8) = W32^m * W32^8 (multiply by e^{-jπ/4})

W32^(m+16) = -W32^m

Your BLOCKED8 path loads W1..W8 then does:

// Derive W9..W16 = -W1..-W8
NW[r] = -W[r];


and then uses those for indices 9, 13, 10, 14, 11, 15, 12 (etc). That gives you W17..W24, not W9..W16. Same issue in BLOCKED4 (you derive W5..W8 correctly via products, but then set W9..W16 by negation).

Minimal fix (BLOCKED8)

Compute W8 once and build W9..W16 = W8 * W1..W8:

// After loading W1..W8:
double W8r, W8i;
// W8 = W4^2 if you have W4, or just load from table as the 8th element.
// In BLOCKED8 you already loaded W[7] as W8 (1-based), so:
W8r = W_re[7]; W8i = W_im[7];

double A_re[8], A_im[8];   // W1..W8
double B_re[8], B_im[8];   // W9..W16 = W8 * W1..W8
for (int r = 0; r < 8; ++r) { A_re[r] = W_re[r]; A_im[r] = W_im[r]; }
for (int r = 0; r < 8; ++r) {
    cmul_fma_scalar(A_re[r], A_im[r], W8r, W8i, &B_re[r], &B_im[r]);
}
// Then use:
//   indices 1,5,9,13 → A[0], A[4], B[0], B[4]
//   indices 2,6,10,14 → A[1], A[5], B[1], B[5]
//   indices 3,7,11,15 → A[2], A[6], B[2], B[6]
//   indices 4,8,12    → A[3], A[7], B[3]

Minimal fix (BLOCKED4)

You already compute W5..W8 via W4 products and W8 = W4². Build W9..W16 via W8 * {W1..W8} (not negation):

// After W1..W4 and W5..W8 and W8 done:
double V_re[8] = { W1r,W2r,W3r,W4r,W5r,W6r,W7r,W8r };
double V_im[8] = { W1i,W2i,W3i,W4i,W5i,W6i,W7i,W8i };
double U_re[8], U_im[8]; // W9..W16
for (int r = 0; r < 8; ++r) {
    cmul_fma_scalar(V_re[r], V_im[r], W8r, W8i, &U_re[r], &U_im[r]);
}
// Use A = V for 1..8, B = U for 9..16 in the same interleaved pattern.

Recurrence path

radix32_init_merge_recurrence_state_scalar() currently sets W9..W16 = -W1..-W8. Replace that with W9..W16 = W8 * W1..W8 (after you’ve computed W8), and keep negation only if you ever need W17..W32.


FMA/Intrinsic nits

cmul_fma_scalar/csquare_fma_scalar build and tear down SSE scalars every call:

That’s a lot of _mm_set_sd traffic. On scalar builds, fma() from <math.h> (or just plain a*b±c with -ffp-contract=fast) usually wins and is cleaner.

Consider two variants:

#if HAS_FMA
  *cr = fma(ar, br, -(ai*bi));   // cr = ar*br - ai*bi
  *ci = fma(ar, bi,  (ai*br));
#else
  *cr = ar*br - ai*bi;
  *ci = ar*bi + ai*br;
#endif


TARGET_FMA on scalar functions is fine, but you don’t need SSE intrinsics to get scalar FMA codegen.
*/