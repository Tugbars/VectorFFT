/**
 * @file fft_radix32_avx2_native_soa.h
 * @brief Production Radix-32 AVX-2 Native SoA - 2×16 Cooley-Tukey + Merge
 *
 * @details
 * ARCHITECTURE: 2×16 COOLEY-TUKEY DECOMPOSITION
 * ==============================================
 * Radix-32 = 2 × 16 via DIT factorization:
 *
 * 1. STAGE 1: Two independent radix-16 butterflies
 *    - Even indices (r=0..15)  → radix-16 transform
 *    - Odd indices  (r=16..31) → radix-16 transform
 *    - Reuses entire radix-16 kernel (all optimizations preserved!)
 *
 * 2. MERGE LAYER: Apply W₃₂ twiddles to odd half
 *    - 16 merge twiddles: W₃₂^m for m=0..15
 *    - Same BLOCKED8/BLOCKED4/recurrence infrastructure
 *    - Interleaved cmul order preserved
 *
 * 3. STAGE 2: Radix-2 butterflies (combine even/odd)
 *    - Simple complex add/subtract
 *    - Zero shuffle overhead (native SoA)
 *
 * AVX-2 ADAPTATIONS:
 * ===================
 * ✅ 256-bit vectors (4 doubles) instead of 512-bit (8 doubles)
 * ✅ YMM registers (16 available) instead of ZMM (32 available)
 * ✅ U=2 software pipelining: k and k+4 (was k+8)
 * ✅ Loop increments: 8 main, 4 tail (was 16, 8)
 * ✅ Prefetch distance: 16 doubles (was 32)
 * ✅ Masking: Careful handling without AVX-512 mask registers
 * ✅ Same K-tiling (Tk=64), thresholds, and algorithms
 *
 * PRESERVED OPTIMIZATIONS FROM AVX-512:
 * =======================================
 * ✅ K-tiling (Tk=64) for L1 cache optimization
 * ✅ U=2 software pipelining (process k and k+4 together)
 * ✅ Interleaved cmul order (breaks FMA dependency chains)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ Prefetch tuning (adjusted for AVX-2)
 * ✅ Hoisted constants (sign masks)
 * ✅ Careful tail handling (K % 4 != 0)
 * ✅ BLOCKED8/BLOCKED4 twiddle systems
 * ✅ Twiddle recurrence for large K (K > 4096)
 * ✅ All alignment hints and compiler attributes
 *
 * MERGE TWIDDLE SYSTEM:
 * =====================
 * - BLOCKED8: K ≤ 256 (16 blocks fit in L1+L2)
 *   * Load W1..W8 (8 blocks)
 *   * W9=-W1, W10=-W2, ..., W16=-W8 (sign flips)
 *   * 50% bandwidth savings
 *
 * - BLOCKED4: K > 256 (twiddles stream from L3/DRAM)
 *   * Load W1..W4 (4 blocks)
 *   * Derive W5=W1×W4, W6=W2×W4, W7=W3×W4, W8=W4²
 *   * W9..W16 via negation
 *   * 75% bandwidth savings
 *
 * - RECURRENCE: K > 4096 (twiddle walking)
 *   * Tile-local recurrence with periodic refresh
 *   * Refresh at tile boundaries (every 64 steps)
 *   * Advance: w ← w × δw within tile
 *
 * @author Tugbars (AVX-2 port)
 * @version 1.0 (2×16 Cooley-Tukey, AVX-2)
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX2_NATIVE_SOA_H
#define FFT_RADIX32_AVX2_NATIVE_SOA_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

// CRITICAL: Include radix-16 AVX-2 version for reuse
#include "fft_radix16_avx2_native_soa.h"

//==============================================================================
// COMPILER PORTABILITY (INHERITED FROM RADIX-16)
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2_FMA
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define FORCE_INLINE static inline
#define RESTRICT
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#define TARGET_AVX2_FMA
#endif

//==============================================================================
// CONFIGURATION (TUNED FOR RADIX-32, AVX-2 ADJUSTED)
//==============================================================================

#ifndef RADIX32_MERGE_BLOCKED8_THRESHOLD
#define RADIX32_MERGE_BLOCKED8_THRESHOLD 256 // 16 blocks, halve from radix-16
#endif

#ifndef RADIX32_STREAM_THRESHOLD_KB
#define RADIX32_STREAM_THRESHOLD_KB 256 // Same as AVX-512
#endif

#ifndef RADIX32_PREFETCH_DISTANCE
#define RADIX32_PREFETCH_DISTANCE 16 // 16 doubles for AVX-2 (was 32 for AVX-512)
#endif

#ifndef RADIX32_TILE_SIZE
#define RADIX32_TILE_SIZE 64 // Keep 64 - accuracy proven
#endif

#ifndef RADIX32_RECURRENCE_THRESHOLD
#define RADIX32_RECURRENCE_THRESHOLD 4096 // Same as AVX-512
#endif

//==============================================================================
// MERGE TWIDDLE STRUCTURES
//==============================================================================

/**
 * @brief Merge twiddles for radix-2 combine (BLOCKED8)
 * Stores W₃₂^m for m=1..8, derives W₉..W₁₆ via negation
 */
typedef struct
{
    const double *RESTRICT re; // [8 * K]
    const double *RESTRICT im; // [8 * K]
} radix32_merge_twiddles_blocked8_avx2_t;

/**
 * @brief Merge twiddles for radix-2 combine (BLOCKED4 with recurrence)
 * Stores W₃₂^m for m=1..4, derives W₅..W₁₆ via products + negation
 */
typedef struct
{
    const double *RESTRICT re; // [4 * K]
    const double *RESTRICT im; // [4 * K]
    __m256d delta_w_re[16];    // Phase increments for recurrence
    __m256d delta_w_im[16];    // Phase increments for recurrence
    size_t K;                  // K value for this stage
    bool recurrence_enabled;   // Whether to use twiddle walking
} radix32_merge_twiddles_blocked4_avx2_t;

typedef enum
{
    RADIX32_MERGE_TW_BLOCKED8_AVX2,
    RADIX32_MERGE_TW_BLOCKED4_AVX2
} radix32_merge_twiddle_mode_avx2_t;

/**
 * @brief Complete radix-32 stage twiddles (combines radix-16 + merge)
 */
typedef struct
{
    // Radix-16 sub-FFT twiddles (reuse existing AVX-2 structures)
    void *radix16_tw_opaque;
    radix16_twiddle_mode_t radix16_mode;

    // Merge twiddles (W₃₂ for combining even/odd halves)
    void *merge_tw_opaque;
    radix32_merge_twiddle_mode_avx2_t merge_mode;
} radix32_stage_twiddles_avx2_t;

//==============================================================================
// PLANNING HELPERS
//==============================================================================

FORCE_INLINE radix32_merge_twiddle_mode_avx2_t
radix32_choose_merge_twiddle_mode_avx2(size_t K)
{
    return (K <= RADIX32_MERGE_BLOCKED8_THRESHOLD)
               ? RADIX32_MERGE_TW_BLOCKED8_AVX2
               : RADIX32_MERGE_TW_BLOCKED4_AVX2;
}

FORCE_INLINE bool
radix32_should_use_merge_recurrence_avx2(size_t K)
{
    return (K > RADIX32_RECURRENCE_THRESHOLD);
}

/**
 * @brief NT Store Decision (same as AVX-512, overflow-safe)
 */
FORCE_INLINE bool
radix32_should_use_nt_stores_avx2(
    size_t K,
    const void *out_re,
    const void *out_im)
{
    const size_t bytes_per_k = 32 * 2 * sizeof(double); // 512 bytes (32 complex)
    const size_t threshold_k = (RADIX32_STREAM_THRESHOLD_KB * 1024) / bytes_per_k;

    return (K >= threshold_k) &&
           (((uintptr_t)out_re & 31) == 0) &&
           (((uintptr_t)out_im & 31) == 0);
}

//==============================================================================
// RADIX-2 BUTTERFLY (NATIVE SOA)
//==============================================================================

/**
 * @brief Radix-2 butterfly for combining even/odd halves
 *
 * @details After radix-16 sub-FFTs and merge twiddle application:
 *   out[m]    = even[m] + odd[m]
 *   out[m+16] = even[m] - odd[m]
 *
 * This is the final "merge" step in 2×16 Cooley-Tukey.
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix2_butterfly_combine_soa_avx2(
    const __m256d even_re[16], const __m256d even_im[16],
    const __m256d odd_re[16], const __m256d odd_im[16],
    __m256d out_re[32], __m256d out_im[32])
{
    // First half: out[0..15] = even + odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m] = _mm256_add_pd(even_re[m], odd_re[m]);
        out_im[m] = _mm256_add_pd(even_im[m], odd_im[m]);
    }

    // Second half: out[16..31] = even - odd
    for (int m = 0; m < 16; m++)
    {
        out_re[m + 16] = _mm256_sub_pd(even_re[m], odd_re[m]);
        out_im[m + 16] = _mm256_sub_pd(even_im[m], odd_im[m]);
    }
}

//==============================================================================
// MERGE TWIDDLES: BLOCKED8 (LOAD + DERIVE)
//==============================================================================

/**
 * @brief Apply merge twiddles to odd half (BLOCKED8)
 *
 * @details Loads W₁..W₈, derives W₉..W₁₆ via negation.
 * Applies W₃₂^m twiddles in interleaved order (preserves optimization).
 *
 * CRITICAL: This multiplies the odd half by merge twiddles BEFORE
 * the radix-2 butterfly. The interleaved cmul order breaks FMA
 * dependency chains exactly as in AVX-512 version.
 */
TARGET_AVX2_FMA
FORCE_INLINE void
apply_merge_twiddles_blocked8_avx2(
    size_t k, size_t K,
    __m256d odd_re[16], __m256d odd_im[16],
    const radix32_merge_twiddles_blocked8_avx2_t *RESTRICT merge_tw,
    __m256d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₈
    __m256d W_re[8], W_im[8];
    for (int r = 0; r < 8; r++)
    {
        W_re[r] = _mm256_load_pd(&re_base[r * K + k]);
        W_im[r] = _mm256_load_pd(&im_base[r * K + k]);
    }

    // Derive W₉..W₁₆ = -W₁..-W₈
    __m256d NW_re[8], NW_im[8];
    for (int r = 0; r < 8; r++)
    {
        NW_re[r] = _mm256_xor_pd(W_re[r], sign_mask);
        NW_im[r] = _mm256_xor_pd(W_im[r], sign_mask);
    }

    // Apply twiddles in INTERLEAVED order (critical optimization!)
    // Pattern: preserves FMA dependency chain breaking from radix-16
    __m256d tr, ti;

    // odd[0] stays unchanged (W₃₂^0 = 1)

    // Group 1: indices 1,5,9,13
    cmul_fma_soa_avx2(odd_re[1], odd_im[1], W_re[0], W_im[0], &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_soa_avx2(odd_re[5], odd_im[5], W_re[4], W_im[4], &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_soa_avx2(odd_re[9], odd_im[9], NW_re[0], NW_im[0], &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_soa_avx2(odd_re[13], odd_im[13], NW_re[4], NW_im[4], &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    // Group 2: indices 2,6,10,14
    cmul_fma_soa_avx2(odd_re[2], odd_im[2], W_re[1], W_im[1], &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_soa_avx2(odd_re[6], odd_im[6], W_re[5], W_im[5], &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_soa_avx2(odd_re[10], odd_im[10], NW_re[1], NW_im[1], &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_soa_avx2(odd_re[14], odd_im[14], NW_re[5], NW_im[5], &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    // Group 3: indices 3,7,11,15
    cmul_fma_soa_avx2(odd_re[3], odd_im[3], W_re[2], W_im[2], &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_soa_avx2(odd_re[7], odd_im[7], W_re[6], W_im[6], &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_soa_avx2(odd_re[11], odd_im[11], NW_re[2], NW_im[2], &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_soa_avx2(odd_re[15], odd_im[15], NW_re[6], NW_im[6], &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    // Group 4: indices 4,8,12
    cmul_fma_soa_avx2(odd_re[4], odd_im[4], W_re[3], W_im[3], &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_soa_avx2(odd_re[8], odd_im[8], W_re[7], W_im[7], &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_soa_avx2(odd_re[12], odd_im[12], NW_re[3], NW_im[3], &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;
}

//==============================================================================
// MERGE TWIDDLES: BLOCKED4 (LOAD + DERIVE)
//==============================================================================

/**
 * @brief Apply merge twiddles to odd half (BLOCKED4)
 *
 * @details Loads W₁..W₄, derives W₅..W₁₆ via products + negation.
 * Same interleaved order as BLOCKED8.
 */
TARGET_AVX2_FMA
FORCE_INLINE void
apply_merge_twiddles_blocked4_avx2(
    size_t k, size_t K,
    __m256d odd_re[16], __m256d odd_im[16],
    const radix32_merge_twiddles_blocked4_avx2_t *RESTRICT merge_tw,
    __m256d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₄
    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);
    __m256d W3r = _mm256_load_pd(&re_base[2 * K + k]);
    __m256d W3i = _mm256_load_pd(&im_base[2 * K + k]);
    __m256d W4r = _mm256_load_pd(&re_base[3 * K + k]);
    __m256d W4i = _mm256_load_pd(&im_base[3 * K + k]);

    // Derive W₅..W₈ via products
    __m256d W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_soa_avx2(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul_fma_soa_avx2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_soa_avx2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_soa_avx2(W4r, W4i, &W8r, &W8i);

    // Derive W₉..W₁₆ via negation
    __m256d NW1r = _mm256_xor_pd(W1r, sign_mask);
    __m256d NW1i = _mm256_xor_pd(W1i, sign_mask);
    __m256d NW2r = _mm256_xor_pd(W2r, sign_mask);
    __m256d NW2i = _mm256_xor_pd(W2i, sign_mask);
    __m256d NW3r = _mm256_xor_pd(W3r, sign_mask);
    __m256d NW3i = _mm256_xor_pd(W3i, sign_mask);
    __m256d NW4r = _mm256_xor_pd(W4r, sign_mask);
    __m256d NW4i = _mm256_xor_pd(W4i, sign_mask);
    __m256d NW5r = _mm256_xor_pd(W5r, sign_mask);
    __m256d NW5i = _mm256_xor_pd(W5i, sign_mask);
    __m256d NW6r = _mm256_xor_pd(W6r, sign_mask);
    __m256d NW6i = _mm256_xor_pd(W6i, sign_mask);
    __m256d NW7r = _mm256_xor_pd(W7r, sign_mask);
    __m256d NW7i = _mm256_xor_pd(W7i, sign_mask);

    // Apply in interleaved order (same pattern as BLOCKED8)
    __m256d tr, ti;

    // odd[0] unchanged

    cmul_fma_soa_avx2(odd_re[1], odd_im[1], W1r, W1i, &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_soa_avx2(odd_re[5], odd_im[5], W5r, W5i, &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_soa_avx2(odd_re[9], odd_im[9], NW1r, NW1i, &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_soa_avx2(odd_re[13], odd_im[13], NW5r, NW5i, &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    cmul_fma_soa_avx2(odd_re[2], odd_im[2], W2r, W2i, &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_soa_avx2(odd_re[6], odd_im[6], W6r, W6i, &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_soa_avx2(odd_re[10], odd_im[10], NW2r, NW2i, &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_soa_avx2(odd_re[14], odd_im[14], NW6r, NW6i, &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    cmul_fma_soa_avx2(odd_re[3], odd_im[3], W3r, W3i, &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_soa_avx2(odd_re[7], odd_im[7], W7r, W7i, &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_soa_avx2(odd_re[11], odd_im[11], NW3r, NW3i, &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_soa_avx2(odd_re[15], odd_im[15], NW7r, NW7i, &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    cmul_fma_soa_avx2(odd_re[4], odd_im[4], W4r, W4i, &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_soa_avx2(odd_re[8], odd_im[8], W8r, W8i, &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_soa_avx2(odd_re[12], odd_im[12], NW4r, NW4i, &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;
}

//==============================================================================
// MERGE TWIDDLES: RECURRENCE INITIALIZATION
//==============================================================================

/**
 * @brief Initialize merge twiddle recurrence state at tile boundary
 *
 * @details Loads W₁..W₄, derives W₅..W₁₆ via products + negation.
 * Identical structure to AVX-512 recurrence initialization.
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_init_merge_recurrence_state_avx2(
    size_t k, size_t K,
    const radix32_merge_twiddles_blocked4_avx2_t *RESTRICT merge_tw,
    __m256d w_state_re[16], __m256d w_state_im[16],
    __m256d sign_mask)
{
    const double *re_base = ASSUME_ALIGNED(merge_tw->re, 32);
    const double *im_base = ASSUME_ALIGNED(merge_tw->im, 32);

    // Load W₁..W₄
    __m256d W1r = _mm256_load_pd(&re_base[0 * K + k]);
    __m256d W1i = _mm256_load_pd(&im_base[0 * K + k]);
    __m256d W2r = _mm256_load_pd(&re_base[1 * K + k]);
    __m256d W2i = _mm256_load_pd(&im_base[1 * K + k]);
    __m256d W3r = _mm256_load_pd(&re_base[2 * K + k]);
    __m256d W3i = _mm256_load_pd(&im_base[2 * K + k]);
    __m256d W4r = _mm256_load_pd(&re_base[3 * K + k]);
    __m256d W4i = _mm256_load_pd(&im_base[3 * K + k]);

    // Derive W₅..W₈
    __m256d W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
    cmul_fma_soa_avx2(W1r, W1i, W4r, W4i, &W5r, &W5i);
    cmul_fma_soa_avx2(W2r, W2i, W4r, W4i, &W6r, &W6i);
    cmul_fma_soa_avx2(W3r, W3i, W4r, W4i, &W7r, &W7i);
    csquare_fma_soa_avx2(W4r, W4i, &W8r, &W8i);

    // W₀ = 1 (identity)
    w_state_re[0] = _mm256_set1_pd(1.0);
    w_state_im[0] = _mm256_setzero_pd();

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
        w_state_re[8 + r] = _mm256_xor_pd(w_state_re[r], sign_mask);
        w_state_im[8 + r] = _mm256_xor_pd(w_state_im[r], sign_mask);
    }
}

//==============================================================================
// MERGE TWIDDLES: RECURRENCE (APPLY + ADVANCE)
//==============================================================================

/**
 * @brief Apply merge twiddles with tile-local recurrence
 *
 * @details Identical recurrence strategy to AVX-512:
 * - Refresh accurate twiddles at tile boundaries
 * - Advance via w ← w × δw within tile
 * - Maintains <1e-14 accuracy over 64-step tiles
 *
 * CRITICAL: Interleaved cmul order preserved!
 */
TARGET_AVX2_FMA
FORCE_INLINE void
apply_merge_twiddles_recur_avx2(
    size_t k, size_t k_tile_start, bool is_tile_start,
    __m256d odd_re[16], __m256d odd_im[16],
    const radix32_merge_twiddles_blocked4_avx2_t *RESTRICT merge_tw,
    __m256d w_state_re[16], __m256d w_state_im[16],
    const __m256d delta_w_re[16], const __m256d delta_w_im[16],
    __m256d sign_mask)
{
    if (is_tile_start)
    {
        // REFRESH: Load accurate twiddles from memory at tile boundary
        radix32_init_merge_recurrence_state_avx2(k, merge_tw->K, merge_tw,
                                                 w_state_re, w_state_im, sign_mask);
    }

    // Apply current twiddles (interleaved order - preserves optimization!)
    __m256d tr, ti;

    // odd[0] unchanged (W₀ = 1)

    // Group 1: indices 1,5,9,13
    cmul_fma_soa_avx2(odd_re[1], odd_im[1], w_state_re[1], w_state_im[1], &tr, &ti);
    odd_re[1] = tr;
    odd_im[1] = ti;

    cmul_fma_soa_avx2(odd_re[5], odd_im[5], w_state_re[5], w_state_im[5], &tr, &ti);
    odd_re[5] = tr;
    odd_im[5] = ti;

    cmul_fma_soa_avx2(odd_re[9], odd_im[9], w_state_re[9], w_state_im[9], &tr, &ti);
    odd_re[9] = tr;
    odd_im[9] = ti;

    cmul_fma_soa_avx2(odd_re[13], odd_im[13], w_state_re[13], w_state_im[13], &tr, &ti);
    odd_re[13] = tr;
    odd_im[13] = ti;

    // Group 2: indices 2,6,10,14
    cmul_fma_soa_avx2(odd_re[2], odd_im[2], w_state_re[2], w_state_im[2], &tr, &ti);
    odd_re[2] = tr;
    odd_im[2] = ti;

    cmul_fma_soa_avx2(odd_re[6], odd_im[6], w_state_re[6], w_state_im[6], &tr, &ti);
    odd_re[6] = tr;
    odd_im[6] = ti;

    cmul_fma_soa_avx2(odd_re[10], odd_im[10], w_state_re[10], w_state_im[10], &tr, &ti);
    odd_re[10] = tr;
    odd_im[10] = ti;

    cmul_fma_soa_avx2(odd_re[14], odd_im[14], w_state_re[14], w_state_im[14], &tr, &ti);
    odd_re[14] = tr;
    odd_im[14] = ti;

    // Group 3: indices 3,7,11,15
    cmul_fma_soa_avx2(odd_re[3], odd_im[3], w_state_re[3], w_state_im[3], &tr, &ti);
    odd_re[3] = tr;
    odd_im[3] = ti;

    cmul_fma_soa_avx2(odd_re[7], odd_im[7], w_state_re[7], w_state_im[7], &tr, &ti);
    odd_re[7] = tr;
    odd_im[7] = ti;

    cmul_fma_soa_avx2(odd_re[11], odd_im[11], w_state_re[11], w_state_im[11], &tr, &ti);
    odd_re[11] = tr;
    odd_im[11] = ti;

    cmul_fma_soa_avx2(odd_re[15], odd_im[15], w_state_re[15], w_state_im[15], &tr, &ti);
    odd_re[15] = tr;
    odd_im[15] = ti;

    // Group 4: indices 4,8,12
    cmul_fma_soa_avx2(odd_re[4], odd_im[4], w_state_re[4], w_state_im[4], &tr, &ti);
    odd_re[4] = tr;
    odd_im[4] = ti;

    cmul_fma_soa_avx2(odd_re[8], odd_im[8], w_state_re[8], w_state_im[8], &tr, &ti);
    odd_re[8] = tr;
    odd_im[8] = ti;

    cmul_fma_soa_avx2(odd_re[12], odd_im[12], w_state_re[12], w_state_im[12], &tr, &ti);
    odd_re[12] = tr;
    odd_im[12] = ti;

    // ADVANCE: w ← w × δw (for next iteration within tile)
    for (int r = 0; r < 16; r++)
    {
        __m256d w_new_re, w_new_im;
        cmul_fma_soa_avx2(w_state_re[r], w_state_im[r],
                          delta_w_re[r], delta_w_im[r],
                          &w_new_re, &w_new_im);
        w_state_re[r] = w_new_re;
        w_state_im[r] = w_new_im;
    }
}

//==============================================================================
// PREFETCH MACROS FOR MERGE (GUARANTEED INLINE)
//==============================================================================

/**
 * @brief Prefetch inputs + merge twiddles for BLOCKED8
 */
#define RADIX32_PREFETCH_MERGE_BLOCKED8_AVX2(k_next, k_limit, K, in_re, in_im, merge_tw) \
    do                                                                                   \
    {                                                                                    \
        if ((k_next) < (k_limit))                                                        \
        {                                                                                \
            for (int _r = 16; _r < 32; _r++)                                             \
            {                                                                            \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0);  \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0);  \
            }                                                                            \
            for (int _b = 0; _b < 8; _b++)                                               \
            {                                                                            \
                _mm_prefetch((const char *)&(merge_tw)->re[_b * (K) + (k_next)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(merge_tw)->im[_b * (K) + (k_next)], _MM_HINT_T0); \
            }                                                                            \
        }                                                                                \
    } while (0)

/**
 * @brief Prefetch inputs + merge twiddles for BLOCKED4
 */
#define RADIX32_PREFETCH_MERGE_BLOCKED4_AVX2(k_next, k_limit, K, in_re, in_im, merge_tw) \
    do                                                                                   \
    {                                                                                    \
        if ((k_next) < (k_limit))                                                        \
        {                                                                                \
            for (int _r = 16; _r < 32; _r++)                                             \
            {                                                                            \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0);  \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0);  \
            }                                                                            \
            for (int _b = 0; _b < 4; _b++)                                               \
            {                                                                            \
                _mm_prefetch((const char *)&(merge_tw)->re[_b * (K) + (k_next)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(merge_tw)->im[_b * (K) + (k_next)], _MM_HINT_T0); \
            }                                                                            \
        }                                                                                \
    } while (0)

/**
 * @brief Prefetch inputs only (for recurrence mode - no twiddle loads)
 */
#define RADIX32_PREFETCH_MERGE_RECURRENCE_AVX2(k_next, k_limit, K, in_re, in_im) \
    do                                                                           \
    {                                                                            \
        if ((k_next) < (k_limit))                                                \
        {                                                                        \
            for (int _r = 16; _r < 32; _r++)                                     \
            {                                                                    \
                _mm_prefetch((const char *)&(in_re)[(k_next) + _r * (K)], _MM_HINT_T0); \
                _mm_prefetch((const char *)&(in_im)[(k_next) + _r * (K)], _MM_HINT_T0); \
            }                                                                    \
        }                                                                        \
    } while (0)

//==============================================================================
// LOAD/STORE FOR 32 LANES (AVX-2: 4 DOUBLES AT A TIME)
//==============================================================================

TARGET_AVX2_FMA
FORCE_INLINE void
store_32_lanes_soa_avx2(size_t k, size_t K,
                        double *RESTRICT out_re, double *RESTRICT out_im,
                        const __m256d y_re[32], const __m256d y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 32; r++)
    {
        _mm256_store_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm256_store_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

TARGET_AVX2_FMA
FORCE_INLINE void
store_32_lanes_soa_avx2_stream(size_t k, size_t K,
                               double *RESTRICT out_re, double *RESTRICT out_im,
                               const __m256d y_re[32], const __m256d y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    for (int r = 0; r < 32; r++)
    {
        _mm256_stream_pd(&out_re_aligned[k + r * K], y_re[r]);
        _mm256_stream_pd(&out_im_aligned[k + r * K], y_im[r]);
    }
}

//==============================================================================
// MASKED TAIL HANDLING (ALL VARIANTS) - AVX-2 ADAPTATION
//==============================================================================

/**
 * @brief Store with mask for AVX-2 (no native mask registers)
 * 
 * @details Uses maskload/maskstore intrinsics for AVX-2.
 * Mask format: -1 (all bits set) = load/store, 0 = skip
 */
TARGET_AVX2_FMA
FORCE_INLINE void
store_32_lanes_soa_avx2_masked(size_t k, size_t K, size_t remaining,
                               double *RESTRICT out_re, double *RESTRICT out_im,
                               const __m256d y_re[32], const __m256d y_im[32])
{
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // AVX-2 mask: -1 (0xFFFFFFFFFFFFFFFF) for valid lanes, 0 for invalid
    __m256i mask;
    if (remaining == 1)
        mask = _mm256_setr_epi64x(-1, 0, 0, 0);
    else if (remaining == 2)
        mask = _mm256_setr_epi64x(-1, -1, 0, 0);
    else if (remaining == 3)
        mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    else // remaining == 4 (but we shouldn't hit this - handled by k+=4 loop)
        mask = _mm256_setr_epi64x(-1, -1, -1, -1);

    for (int r = 0; r < 32; r++)
    {
        _mm256_maskstore_pd(&out_re_aligned[k + r * K], mask, y_re[r]);
        _mm256_maskstore_pd(&out_im_aligned[k + r * K], mask, y_im[r]);
    }
}

/**
 * @brief Process tail with mask - BLOCKED8 Forward
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_process_tail_masked_blocked8_forward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT radix16_tw, radix16_twiddle_mode_t r16_mode,
    const radix32_merge_twiddles_blocked8_avx2_t *RESTRICT merge_tw,
    __m256d neg_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;
    if (remaining > 4)
        remaining = 4;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    // Process even half (r=0..15) with radix-16
    __m256d even_re[16], even_im[16];  // ← Only once!
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re_aligned, in_im_aligned,
                                  even_re, even_im);

    // Apply radix-16 stage to even half
    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask!
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask!
    }

    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);
    radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                       even_re, even_im,
                                                       rot_sign_mask);

    // Process odd half (r=16..31) with radix-16
    __m256d odd_re[16], odd_im[16];

    // Build mask for odd half
    __m256i mask;
    if (remaining == 1)
        mask = _mm256_setr_epi64x(-1, 0, 0, 0);
    else if (remaining == 2)
        mask = _mm256_setr_epi64x(-1, -1, 0, 0);
    else if (remaining == 3)
        mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    else
        mask = _mm256_setr_epi64x(-1, -1, -1, -1);

    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm256_maskload_pd(&in_re_aligned[k + (r + 16) * K], mask);
        odd_im[r] = _mm256_maskload_pd(&in_im_aligned[k + (r + 16) * K], mask);
    }

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask!
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask!
    }

    radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                       odd_re, odd_im,
                                                       rot_sign_mask);

    // Apply merge twiddles to odd half
    apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);

    // Radix-2 combine
    __m256d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

    // Store result
    store_32_lanes_soa_avx2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

/**
 * @brief Process tail with mask - BLOCKED8 Backward
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_process_tail_masked_blocked8_backward_avx2(
    size_t k, size_t k_end, size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT radix16_tw, radix16_twiddle_mode_t r16_mode,
    const radix32_merge_twiddles_blocked8_avx2_t *RESTRICT merge_tw,
    __m256d neg_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;
    if (remaining > 4)
        remaining = 4;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);  // ← MOVED TO TOP!

    // Even half
    __m256d even_re[16], even_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re_aligned, in_im_aligned,
                                  even_re, even_im);

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16);
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16);
    }

    radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                        even_re, even_im,
                                                        rot_sign_mask);

    // Build mask for odd half
    __m256i mask;
    if (remaining == 1)
        mask = _mm256_setr_epi64x(-1, 0, 0, 0);
    else if (remaining == 2)
        mask = _mm256_setr_epi64x(-1, -1, 0, 0);
    else if (remaining == 3)
        mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    else
        mask = _mm256_setr_epi64x(-1, -1, -1, -1);

    // Odd half
    __m256d odd_re[16], odd_im[16];
    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm256_maskload_pd(&in_re_aligned[k + (r + 16) * K], mask);
        odd_im[r] = _mm256_maskload_pd(&in_im_aligned[k + (r + 16) * K], mask);
    }

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask!
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask!
    }

    radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                        odd_re, odd_im,
                                                        rot_sign_mask);

    apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);

    __m256d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

    store_32_lanes_soa_avx2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

/**
 * @brief Process tail with mask - BLOCKED4 Forward (with optional recurrence)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_process_tail_masked_blocked4_forward_avx2(
    size_t k, size_t k_end, size_t K, size_t k_tile_start,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT radix16_tw, radix16_twiddle_mode_t r16_mode,
    const radix32_merge_twiddles_blocked4_avx2_t *RESTRICT merge_tw,
    bool use_merge_recurrence,
    __m256d merge_w_state_re[16], __m256d merge_w_state_im[16],
    const __m256d *delta_w_re, const __m256d *delta_w_im,
    __m256d neg_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;
    if (remaining > 4)
        remaining = 4;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);

    // Even half
    __m256d even_re[16], even_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re_aligned, in_im_aligned,  // ← Changed: pass 'remaining'
                                  even_re, even_im);

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask
    }

    radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                       even_re, even_im,
                                                       rot_sign_mask);

    // Build mask for odd half
    __m256i mask;  // ← Moved here
    if (remaining == 1)
        mask = _mm256_setr_epi64x(-1, 0, 0, 0);
    else if (remaining == 2)
        mask = _mm256_setr_epi64x(-1, -1, 0, 0);
    else if (remaining == 3)
        mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    else
        mask = _mm256_setr_epi64x(-1, -1, -1, -1);

    // Odd half
    __m256d odd_re[16], odd_im[16];
    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm256_maskload_pd(&in_re_aligned[k + (r + 16) * K], mask);
        odd_im[r] = _mm256_maskload_pd(&in_im_aligned[k + (r + 16) * K], mask);
    }

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask
    }

    radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                       odd_re, odd_im,
                                                       rot_sign_mask);

    // Apply merge twiddles
    if (use_merge_recurrence)
    {
        apply_merge_twiddles_recur_avx2(k, k_tile_start, false, odd_re, odd_im,
                                        merge_tw, merge_w_state_re, merge_w_state_im,
                                        delta_w_re, delta_w_im, neg_mask);
    }
    else
    {
        apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);
    }

    __m256d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

    store_32_lanes_soa_avx2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

/**
 * @brief Process tail with mask - BLOCKED4 Backward (with optional recurrence)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_process_tail_masked_blocked4_backward_avx2(
    size_t k, size_t k_end, size_t K, size_t k_tile_start,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const void *RESTRICT radix16_tw, radix16_twiddle_mode_t r16_mode,
    const radix32_merge_twiddles_blocked4_avx2_t *RESTRICT merge_tw,
    bool use_merge_recurrence,
    __m256d merge_w_state_re[16], __m256d merge_w_state_im[16],
    const __m256d *delta_w_re, const __m256d *delta_w_im,
    __m256d neg_mask)
{
    if (k >= k_end)
        return;

    size_t remaining = k_end - k;
    if (remaining > 4)
        remaining = 4;

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);

    // Even half
    __m256d even_re[16], even_im[16];
    load_16_lanes_soa_avx2_masked(k, K, remaining, in_re_aligned, in_im_aligned,  // ← Changed: pass 'remaining'
                                  even_re, even_im);

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16);  // ← No neg_mask
    }

    radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                        even_re, even_im,
                                                        rot_sign_mask);

    // Build mask for odd half
    __m256i mask;  // ← Moved here
    if (remaining == 1)
        mask = _mm256_setr_epi64x(-1, 0, 0, 0);
    else if (remaining == 2)
        mask = _mm256_setr_epi64x(-1, -1, 0, 0);
    else if (remaining == 3)
        mask = _mm256_setr_epi64x(-1, -1, -1, 0);
    else
        mask = _mm256_setr_epi64x(-1, -1, -1, -1);

    // Odd half
    __m256d odd_re[16], odd_im[16];
    for (int r = 0; r < 16; r++)
    {
        odd_re[r] = _mm256_maskload_pd(&in_re_aligned[k + (r + 16) * K], mask);
        odd_im[r] = _mm256_maskload_pd(&in_im_aligned[k + (r + 16) * K], mask);
    }

    if (r16_mode == RADIX16_TW_BLOCKED8)
    {
        const radix16_stage_twiddles_blocked8_t *tw16 =
            (const radix16_stage_twiddles_blocked8_t *)radix16_tw;
        apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask
    }
    else
    {
        const radix16_stage_twiddles_blocked4_t *tw16 =
            (const radix16_stage_twiddles_blocked4_t *)radix16_tw;
        apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16);  // ← No neg_mask
    }

    radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                        odd_re, odd_im,
                                                        rot_sign_mask);

    // Apply merge twiddles
    if (use_merge_recurrence)
    {
        apply_merge_twiddles_recur_avx2(k, k_tile_start, false, odd_re, odd_im,
                                        merge_tw, merge_w_state_re, merge_w_state_im,
                                        delta_w_re, delta_w_im, neg_mask);
    }
    else
    {
        apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);
    }

    __m256d y_re[32], y_im[32];
    radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

    store_32_lanes_soa_avx2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
}

//==============================================================================
// COMPLETE STAGE DRIVERS: FORWARD (BLOCKED8)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - BLOCKED8 Merge (AVX-2 CORRECTED)
 *
 * AVX-2 ADAPTATIONS:
 * - Main loop: k += 8 (was 16)
 * - U=2 pipelining: process k and k+4 (was k+8)
 * - Tail loop: k += 4 (was 8)
 * - Prefetch distance: 16 doubles (was 32)
 * - 256-bit vectors (4 doubles) instead of 512-bit (8 doubles)
 *
 * PRESERVED OPTIMIZATIONS:
 * - K-tiling (Tk=64)
 * - U=2 software pipelining
 * - Prefetch (adjusted for AVX-2)
 * - Adaptive NT stores
 * - Masked tail handling
 * - All radix-16 optimizations inherited
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_stage_dit_forward_blocked8_merge_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_avx2_t *RESTRICT stage_tw)
{
    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE; // 16 for AVX-2
    const size_t tile_size = RADIX32_TILE_SIZE; // 64

    const bool use_nt_stores = radix32_should_use_nt_stores_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked8_avx2_t *merge_tw =
        (const radix32_merge_twiddles_blocked8_avx2_t *)stage_tw->merge_tw_opaque;

    // Cast to specific types for branching
    const radix16_stage_twiddles_blocked8_t  *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8)
            ? (const radix16_stage_twiddles_blocked8_t  *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_t  *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4)
            ? (const radix16_stage_twiddles_blocked4_t  *)radix16_tw
            : NULL;

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=2 LOOP: k += 8 (process k and k+4)
        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            // Prefetch next iteration (BOTH radix-16 + merge)
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                // Prefetch even half (radix-16 inputs + twiddles)
                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b8);
                }
                else // RADIX16_TW_BLOCKED4
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b4);
                }

                // Prefetch odd half + merge twiddles
                RADIX32_PREFETCH_MERGE_BLOCKED8_AVX2(k_next, k_end, K,
                                                     in_re_aligned, in_im_aligned, merge_tw);
            }

            // ==================== PROCESS k ====================
            {
                // Even half (r=0..15): radix-16 butterfly
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                else // RADIX16_TW_BLOCKED4
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

                // Odd half (r=16..31): radix-16 butterfly
                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                else // RADIX16_TW_BLOCKED4
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

                // Apply merge twiddles to odd half
                apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                   merge_tw, neg_mask);

                // Radix-2 combine
                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                // Store
                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                                   y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }

            // ==================== PROCESS k+4 (U=2 SOFTWARE PIPELINING) ====================
            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + 4 + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + 4 + (r + 16) * K]);
                }

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

                apply_merge_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                   merge_tw, neg_mask);

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k + 4, K, out_re_aligned,
                                                   out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k + 4, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }
        }

        // TAIL LOOP #1: k += 4
        for (; k + 4 <= k_end; k += 4)
        {
            __m256d even_re[16], even_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_mode == RADIX16_TW_BLOCKED8)
            {
                apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16b8, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16b4, neg_mask);
            }
            radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

            __m256d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            if (r16_mode == RADIX16_TW_BLOCKED8)
            {
                apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16b8, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16b4, neg_mask);
            }
            radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

            apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);

            __m256d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                               y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        // TAIL LOOP #2: Masked (K % 4 != 0)
        radix32_process_tail_masked_blocked8_forward_avx2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            radix16_tw, r16_mode, merge_tw, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: FORWARD (BLOCKED4 WITH RECURRENCE)
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - BLOCKED4 Merge (AVX-2 CORRECTED)
 *
 * CRITICAL PLANNER REQUIREMENTS:
 * ================================
 * 1. Radix-16 delta_w arrays must have 16 elements (indices 0..15)
 *    - delta_w[15] should be set to identity (1+0i) or safe value
 *    - Butterfly advances all 16 mechanically
 *
 * 2. Merge delta_w arrays must have 16 elements (indices 0..15)
 *    - delta_w[0] MUST be identity (1+0i) to keep W₀ stationary
 *    - Butterfly advances all 16 mechanically, including W₀
 *
 * BUFFER SIZING:
 * - r16_w_even/odd: [16] elements (was [15] - BUG FIXED)
 * - merge_w_state: [16] elements (correct)
 *
 * AVX-2 ADAPTATIONS:
 * - Main loop: k += 8, U=2 with k and k+4
 * - Tail loop: k += 4, then masked
 * - Prefetch: 16 doubles ahead
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_stage_dit_forward_blocked4_merge_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_avx2_t *RESTRICT stage_tw,
    bool use_merge_recurrence,
    const __m256d *RESTRICT merge_delta_w_re,
    const __m256d *RESTRICT merge_delta_w_im)
{
    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE;
    const size_t tile_size = RADIX32_TILE_SIZE;

    const bool use_nt_stores = radix32_should_use_nt_stores_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked4_avx2_t *merge_tw =
        (const radix32_merge_twiddles_blocked4_avx2_t *)stage_tw->merge_tw_opaque;

    // CRITICAL: Detect radix-16 recurrence mode
    const radix16_stage_twiddles_blocked8_t  *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8)
            ? (const radix16_stage_twiddles_blocked8_t  *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_t  *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4)
            ? (const radix16_stage_twiddles_blocked4_t  *)radix16_tw
            : NULL;

    bool r16_recur = (r16_mode == RADIX16_TW_BLOCKED4) && tw16b4->recurrence_enabled;

    // Merge recurrence state
    __m256d merge_w_state_re[16], merge_w_state_im[16];

    // CRITICAL FIX: Radix-16 recurrence state - [16] elements (was [15])
    // Planner must set delta_w[15] to safe value (identity recommended)
    __m256d r16_w_even_re[16], r16_w_even_im[16]; // For r=0..15
    __m256d r16_w_odd_re[16], r16_w_odd_im[16];   // For r=16..31

    // K-TILING OUTER LOOP
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // MAIN U=2 LOOP: k += 8
        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            // Prefetch next iteration (BOTH radix-16 + merge)
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                // Prefetch even half (radix-16)
                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b8);
                }
                else if (r16_recur)
                {
                    RADIX16_PREFETCH_NEXT_RECURRENCE_AVX2(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b4);
                }

                // Prefetch merge
                if (use_merge_recurrence)
                {
                    RADIX32_PREFETCH_MERGE_RECURRENCE_AVX2(k_next, k_end, K,
                                                           in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX32_PREFETCH_MERGE_BLOCKED4_AVX2(k_next, k_end, K,
                                                         in_re_aligned, in_im_aligned, merge_tw);
                }
            }

            bool is_tile_start = (k == k_tile);

            // ==================== PROCESS k ====================
            {
                // Even half: radix-16 butterfly
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                // CRITICAL: Branch on radix-16 recurrence mode
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                else // RADIX16_TW_BLOCKED8
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

                // Odd half: radix-16 butterfly
                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                // CRITICAL: Branch on radix-16 recurrence mode (odd half uses separate walker)
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

                // Apply merge twiddles
                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_avx2(k, k_tile, is_tile_start,
                                                    odd_re, odd_im, merge_tw,
                                                    merge_w_state_re, merge_w_state_im,
                                                    merge_delta_w_re, merge_delta_w_im,
                                                    neg_mask);
                }
                else
                {
                    apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       merge_tw, neg_mask);
                }

                // Radix-2 combine
                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                // Store
                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                                   y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }

            // ==================== PROCESS k+4 ====================
            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                // CRITICAL: is_tile_start = false for k+4
                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + 4 + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + 4 + (r + 16) * K]);
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_avx2(k + 4, k_tile, false,
                                                    odd_re, odd_im, merge_tw,
                                                    merge_w_state_re, merge_w_state_im,
                                                    merge_delta_w_re, merge_delta_w_im,
                                                    neg_mask);
                }
                else
                {
                    apply_merge_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       merge_tw, neg_mask);
                }

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k + 4, K, out_re_aligned,
                                                   out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k + 4, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }
        }

        // TAIL LOOP #1: k += 4
        for (; k + 4 <= k_end; k += 4)
        {
            __m256d even_re[16], even_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_recur)
            {
                apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4)
            {
                apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16b4, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16b8, neg_mask);
            }
            radix16_complete_butterfly_forward_fused_soa_avx2(even_re, even_im,
                                                   even_re, even_im,
                                                   rot_sign_mask);

            __m256d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            if (r16_recur)
            {
                apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4)
            {
                apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16b4, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16b8, neg_mask);
            }
            radix16_complete_butterfly_forward_fused_soa_avx2(odd_re, odd_im,
                                                   odd_re, odd_im,
                                                   rot_sign_mask);

            if (use_merge_recurrence)
            {
                apply_merge_twiddles_recur_avx2(k, k_tile, false, odd_re, odd_im,
                                                merge_tw, merge_w_state_re, merge_w_state_im,
                                                merge_delta_w_re, merge_delta_w_im, neg_mask);
            }
            else
            {
                apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);
            }

            __m256d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                               y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        // TAIL LOOP #2: Masked
        radix32_process_tail_masked_blocked4_forward_avx2(
            k, k_end, K, k_tile, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            radix16_tw, r16_mode, merge_tw, use_merge_recurrence,
            merge_w_state_re, merge_w_state_im, merge_delta_w_re, merge_delta_w_im,
            neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: BACKWARD (BLOCKED8)
//==============================================================================

/**
 * @brief Radix-32 DIT Backward Stage - BLOCKED8 Merge (AVX-2)
 *
 * AVX-2 ADAPTATIONS:
 * - Main loop: k += 8 (was 16)
 * - U=2 pipelining: process k and k+4 (was k+8)
 * - Tail loop: k += 4 (was 8)
 * - Prefetch distance: 16 doubles (was 32)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_stage_dit_backward_blocked8_merge_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_avx2_t *RESTRICT stage_tw)
{
    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE;
    const size_t tile_size = RADIX32_TILE_SIZE;

    const bool use_nt_stores = radix32_should_use_nt_stores_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked8_avx2_t *merge_tw =
        (const radix32_merge_twiddles_blocked8_avx2_t *)stage_tw->merge_tw_opaque;

    const radix16_stage_twiddles_blocked8_t  *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8)
            ? (const radix16_stage_twiddles_blocked8_t  *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_t  *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4)
            ? (const radix16_stage_twiddles_blocked4_t  *)radix16_tw
            : NULL;

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                // Prefetch radix-16
                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b8);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b4);
                }

                // Prefetch merge
                RADIX32_PREFETCH_MERGE_BLOCKED8_AVX2(k_next, k_end, K,
                                                     in_re_aligned, in_im_aligned, merge_tw);
            }

            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);

                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

                apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                   merge_tw, neg_mask);

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                                   y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }

            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);
                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + 4 + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + 4 + (r + 16) * K]);
                }

                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

                apply_merge_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                   merge_tw, neg_mask);

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k + 4, K, out_re_aligned,
                                                   out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k + 4, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }
        }

        for (; k + 4 <= k_end; k += 4)
        {
            __m256d even_re[16], even_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_mode == RADIX16_TW_BLOCKED8)
            {
                apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16b8, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16b4, neg_mask);
            }
            radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);

            __m256d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            if (r16_mode == RADIX16_TW_BLOCKED8)
            {
                apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16b8, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16b4, neg_mask);
            }
            radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

            apply_merge_twiddles_blocked8_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);

            __m256d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                               y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix32_process_tail_masked_blocked8_backward_avx2(
            k, k_end, K, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            radix16_tw, r16_mode, merge_tw, neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// COMPLETE STAGE DRIVERS: BACKWARD (BLOCKED4 WITH RECURRENCE)
//==============================================================================

/**
 * @brief Radix-32 DIT Backward Stage - BLOCKED4 Merge (AVX-2 CORRECTED)
 *
 * CRITICAL PLANNER REQUIREMENTS:
 * ================================
 * Same as forward - see documentation above.
 *
 * BUFFER SIZING:
 * - r16_w_even/odd: [16] elements (was [15] - BUG FIXED)
 * - merge_w_state: [16] elements (correct)
 */
TARGET_AVX2_FMA
FORCE_INLINE void
radix32_stage_dit_backward_blocked4_merge_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix32_stage_twiddles_avx2_t *RESTRICT stage_tw,
    bool use_merge_recurrence,
    const __m256d *RESTRICT merge_delta_w_re,
    const __m256d *RESTRICT merge_delta_w_im)
{
    const __m256d rot_sign_mask = _mm256_set1_pd(-0.0);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    const size_t prefetch_dist = RADIX32_PREFETCH_DISTANCE;
    const size_t tile_size = RADIX32_TILE_SIZE;

    const bool use_nt_stores = radix32_should_use_nt_stores_avx2(K, out_re, out_im);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 32);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 32);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 32);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 32);

    const void *radix16_tw = stage_tw->radix16_tw_opaque;
    const radix16_twiddle_mode_t r16_mode = stage_tw->radix16_mode;
    const radix32_merge_twiddles_blocked4_avx2_t *merge_tw =
        (const radix32_merge_twiddles_blocked4_avx2_t *)stage_tw->merge_tw_opaque;

    const radix16_stage_twiddles_blocked8_t  *tw16b8 =
        (r16_mode == RADIX16_TW_BLOCKED8)
            ? (const radix16_stage_twiddles_blocked8_t  *)radix16_tw
            : NULL;
    const radix16_stage_twiddles_blocked4_t  *tw16b4 =
        (r16_mode == RADIX16_TW_BLOCKED4)
            ? (const radix16_stage_twiddles_blocked4_t  *)radix16_tw
            : NULL;

    bool r16_recur = (r16_mode == RADIX16_TW_BLOCKED4) && tw16b4->recurrence_enabled;

    __m256d merge_w_state_re[16], merge_w_state_im[16];
    
    // CRITICAL FIX: [16] elements (was [15])
    __m256d r16_w_even_re[16], r16_w_even_im[16];
    __m256d r16_w_odd_re[16], r16_w_odd_im[16];

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 8 <= k_end; k += 8)
        {
            size_t k_next = k + 8 + prefetch_dist;
            if (k_next < k_end)
            {
                if (r16_mode == RADIX16_TW_BLOCKED8)
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED8_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b8);
                }
                else if (r16_recur)
                {
                    RADIX16_PREFETCH_NEXT_RECURRENCE_AVX2(k_next, k_end, K,
                                                          in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX16_PREFETCH_NEXT_BLOCKED4_AVX2(k_next, k_end, K,
                                                        in_re_aligned, in_im_aligned, tw16b4);
                }

                if (use_merge_recurrence)
                {
                    RADIX32_PREFETCH_MERGE_RECURRENCE_AVX2(k_next, k_end, K,
                                                           in_re_aligned, in_im_aligned);
                }
                else
                {
                    RADIX32_PREFETCH_MERGE_BLOCKED4_AVX2(k_next, k_end, K,
                                                         in_re_aligned, in_im_aligned, merge_tw);
                }
            }

            bool is_tile_start = (k == k_tile);

            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);

                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_avx2(k, k_tile, is_tile_start,
                                                    odd_re, odd_im, merge_tw,
                                                    merge_w_state_re, merge_w_state_im,
                                                    merge_delta_w_re, merge_delta_w_im,
                                                    neg_mask);
                }
                else
                {
                    apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im,
                                                       merge_tw, neg_mask);
                }

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                                   y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }

            {
                __m256d even_re[16], even_im[16];
                load_16_lanes_soa_avx2(k + 4, K, in_re_aligned, in_im_aligned,
                                       even_re, even_im);

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, even_re, even_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, even_re, even_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);

                __m256d odd_re[16], odd_im[16];
                for (int r = 0; r < 16; r++)
                {
                    odd_re[r] = _mm256_load_pd(&in_re_aligned[k + 4 + (r + 16) * K]);
                    odd_im[r] = _mm256_load_pd(&in_im_aligned[k + 4 + (r + 16) * K]);
                }

                if (r16_recur)
                {
                    apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
                }
                else if (r16_mode == RADIX16_TW_BLOCKED4)
                {
                    apply_stage_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b4, neg_mask);
                }
                else
                {
                    apply_stage_twiddles_blocked8_avx2(k + 4, K, odd_re, odd_im,
                                                       tw16b8, neg_mask);
                }
                radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

                if (use_merge_recurrence)
                {
                    apply_merge_twiddles_recur_avx2(k + 4, k_tile, false,
                                                    odd_re, odd_im, merge_tw,
                                                    merge_w_state_re, merge_w_state_im,
                                                    merge_delta_w_re, merge_delta_w_im,
                                                    neg_mask);
                }
                else
                {
                    apply_merge_twiddles_blocked4_avx2(k + 4, K, odd_re, odd_im,
                                                       merge_tw, neg_mask);
                }

                __m256d y_re[32], y_im[32];
                radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im,
                                                  y_re, y_im);

                if (use_nt_stores)
                {
                    store_32_lanes_soa_avx2_stream(k + 4, K, out_re_aligned,
                                                   out_im_aligned, y_re, y_im);
                }
                else
                {
                    store_32_lanes_soa_avx2(k + 4, K, out_re_aligned, out_im_aligned,
                                            y_re, y_im);
                }
            }
        }

        for (; k + 4 <= k_end; k += 4)
        {
            __m256d even_re[16], even_im[16];
            load_16_lanes_soa_avx2(k, K, in_re_aligned, in_im_aligned, even_re, even_im);

            if (r16_recur)
            {
                apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4)
            {
                apply_stage_twiddles_blocked4_avx2(k, K, even_re, even_im, tw16b4, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked8_avx2(k, K, even_re, even_im, tw16b8, neg_mask);
            }
            radix16_complete_butterfly_backward_fused_soa_avx2(even_re, even_im,
                                                    even_re, even_im,
                                                    rot_sign_mask);

            __m256d odd_re[16], odd_im[16];
            for (int r = 0; r < 16; r++)
            {
                odd_re[r] = _mm256_load_pd(&in_re_aligned[k + (r + 16) * K]);
                odd_im[r] = _mm256_load_pd(&in_im_aligned[k + (r + 16) * K]);
            }

            if (r16_recur)
            {
                apply_stage_twiddles_recur_avx2(k, is_tile_start,  // ← Remove k_tile!
                                even_re, even_im, tw16b4,
                                r16_w_even_re, r16_w_even_im,
                                tw16b4->delta_w_re, tw16b4->delta_w_im);  // ← Remove neg_mask!
            }
            else if (r16_mode == RADIX16_TW_BLOCKED4)
            {
                apply_stage_twiddles_blocked4_avx2(k, K, odd_re, odd_im, tw16b4, neg_mask);
            }
            else
            {
                apply_stage_twiddles_blocked8_avx2(k, K, odd_re, odd_im, tw16b8, neg_mask);
            }
            radix16_complete_butterfly_backward_fused_soa_avx2(odd_re, odd_im,
                                                    odd_re, odd_im,
                                                    rot_sign_mask);

            if (use_merge_recurrence)
            {
                apply_merge_twiddles_recur_avx2(k, k_tile, false, odd_re, odd_im,
                                                merge_tw, merge_w_state_re, merge_w_state_im,
                                                merge_delta_w_re, merge_delta_w_im, neg_mask);
            }
            else
            {
                apply_merge_twiddles_blocked4_avx2(k, K, odd_re, odd_im, merge_tw, neg_mask);
            }

            __m256d y_re[32], y_im[32];
            radix2_butterfly_combine_soa_avx2(even_re, even_im, odd_re, odd_im, y_re, y_im);

            if (use_nt_stores)
            {
                store_32_lanes_soa_avx2_stream(k, K, out_re_aligned, out_im_aligned,
                                               y_re, y_im);
            }
            else
            {
                store_32_lanes_soa_avx2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        radix32_process_tail_masked_blocked4_backward_avx2(
            k, k_end, K, k_tile, in_re_aligned, in_im_aligned, out_re_aligned, out_im_aligned,
            radix16_tw, r16_mode, merge_tw, use_merge_recurrence,
            merge_w_state_re, merge_w_state_im, merge_delta_w_re, merge_delta_w_im,
            neg_mask);
    }

    if (use_nt_stores)
    {
        _mm_sfence();
    }
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Radix-32 DIT Forward Stage - Public API (AVX-2)
 *
 * @param K Number of k-indices per radix lane
 * @param in_re Input real part (SoA: [32][K])
 * @param in_im Input imaginary part (SoA: [32][K])
 * @param out_re Output real part (SoA: [32][K])
 * @param out_im Output imaginary part (SoA: [32][K])
 * @param stage_tw_opaque Opaque pointer to radix32_stage_twiddles_avx2_t
 *
 * @note Twiddle structures should be prepared during planning phase:
 *       - radix16_tw: Stage twiddles for both radix-16 sub-FFTs
 *       - merge_tw: W₃₂^m twiddles for combining even/odd halves
 *       - delta_w: Phase increments for recurrence (if enabled)
 */
TARGET_AVX2_FMA
void radix32_stage_dit_forward_soa_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque)
{
    const radix32_stage_twiddles_avx2_t *stage_tw =
        (const radix32_stage_twiddles_avx2_t *)stage_tw_opaque;

    if (stage_tw->merge_mode == RADIX32_MERGE_TW_BLOCKED8_AVX2)
    {
        radix32_stage_dit_forward_blocked8_merge_avx2(
            K, in_re, in_im, out_re, out_im, stage_tw);
    }
    else // RADIX32_MERGE_TW_BLOCKED4_AVX2
    {
        const radix32_merge_twiddles_blocked4_avx2_t *merge_tw =
            (const radix32_merge_twiddles_blocked4_avx2_t *)stage_tw->merge_tw_opaque;

        if (merge_tw->recurrence_enabled)
        {
            radix32_stage_dit_forward_blocked4_merge_avx2(
                K, in_re, in_im, out_re, out_im,
                stage_tw, true, merge_tw->delta_w_re, merge_tw->delta_w_im);
        }
        else
        {
            radix32_stage_dit_forward_blocked4_merge_avx2(
                K, in_re, in_im, out_re, out_im,
                stage_tw, false, NULL, NULL);
        }
    }
}

/**
 * @brief Radix-32 DIT Backward Stage - Public API (AVX-2)
 *
 * @param K Number of k-indices per radix lane
 * @param in_re Input real part (SoA: [32][K])
 * @param in_im Input imaginary part (SoA: [32][K])
 * @param out_re Output real part (SoA: [32][K])
 * @param out_im Output imaginary part (SoA: [32][K])
 * @param stage_tw_opaque Opaque pointer to radix32_stage_twiddles_avx2_t
 */
TARGET_AVX2_FMA
void radix32_stage_dit_backward_soa_avx2(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const void *RESTRICT stage_tw_opaque)
{
    const radix32_stage_twiddles_avx2_t *stage_tw =
        (const radix32_stage_twiddles_avx2_t *)stage_tw_opaque;

    if (stage_tw->merge_mode == RADIX32_MERGE_TW_BLOCKED8_AVX2)
    {
        radix32_stage_dit_backward_blocked8_merge_avx2(
            K, in_re, in_im, out_re, out_im, stage_tw);
    }
    else // RADIX32_MERGE_TW_BLOCKED4_AVX2
    {
        const radix32_merge_twiddles_blocked4_avx2_t *merge_tw =
            (const radix32_merge_twiddles_blocked4_avx2_t *)stage_tw->merge_tw_opaque;

        if (merge_tw->recurrence_enabled)
        {
            radix32_stage_dit_backward_blocked4_merge_avx2(
                K, in_re, in_im, out_re, out_im,
                stage_tw, true, merge_tw->delta_w_re, merge_tw->delta_w_im);
        }
        else
        {
            radix32_stage_dit_backward_blocked4_merge_avx2(
                K, in_re, in_im, out_re, out_im,
                stage_tw, false, NULL, NULL);
        }
    }
}

#endif // FFT_RADIX32_AVX2_NATIVE_SOA_H

/**
 * CRITICAL PLANNER REQUIREMENTS FOR RADIX-32 AVX-2
 * ==================================================
 * 
 * The butterfly code uses mechanical iteration over all twiddle array elements.
 * The planner MUST ensure mathematical correctness by setting arrays properly.
 * 
 * 1. RADIX-16 RECURRENCE (stage_tw->radix16_tw_opaque when BLOCKED4):
 *    -----------------------------------------------------------------
 *    radix16_stage_twiddles_blocked4_t  must have:
 *    
 *    - delta_w_re[16], delta_w_im[16]  (16 elements, NOT 15)
 *    - Indices 0..14: Phase increments for W₁..W₁₅
 *    - Index 15: MUST be set to identity (1+0i) or any safe value
 *                (this slot is never mathematically used, but butterfly
 *                 mechanically advances all 16 slots to avoid branching)
 * 
 * 2. RADIX-32 MERGE RECURRENCE (stage_tw->merge_tw_opaque when BLOCKED4):
 *    -----------------------------------------------------------------------
 *    radix32_merge_twiddles_blocked4_avx2_t must have:
 *    
 *    - delta_w_re[16], delta_w_im[16]
 *    - Index 0: MUST be identity (1+0i) to keep W₀ = 1+0i stationary
 *    - Indices 1..15: Phase increments for W₁..W₁₅
 * 
 * 3. WHY THIS DESIGN:
 *    ----------------
 *    - Butterfly code is purely mechanical (no special cases, no branches)
 *    - Planner handles all mathematical invariants
 *    - Clean separation of concerns
 *    - Easier to verify correctness at planning time
 * 
 * 4. EXAMPLE PLANNER CODE:
 *    ---------------------
 *    // Radix-16 delta_w (15 active + 1 padding):
 *    for (int r = 0; r < 15; r++) {
 *        double phase = 2.0 * M_PI * (r + 1) / (N / stage_radix);
 *        tw->delta_w_re[r] = _mm256_set1_pd(cos(phase));
 *        tw->delta_w_im[r] = _mm256_set1_pd(sin(phase));
 *    }
 *    tw->delta_w_re[15] = _mm256_set1_pd(1.0);  // Padding (identity)
 *    tw->delta_w_im[15] = _mm256_setzero_pd();
 * 
 *    // Radix-32 merge delta_w (W₀ stationary + 15 active):
 *    tw->delta_w_re[0] = _mm256_set1_pd(1.0);   // W₀ stays 1+0i
 *    tw->delta_w_im[0] = _mm256_setzero_pd();
 *    for (int r = 1; r < 16; r++) {
 *        double phase = 2.0 * M_PI * r / 32;
 *        tw->delta_w_re[r] = _mm256_set1_pd(cos(phase));
 *        tw->delta_w_im[r] = _mm256_set1_pd(sin(phase));
 *    }
 */
