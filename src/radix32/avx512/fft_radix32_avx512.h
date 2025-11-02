/**
 * @file fft_radix32_8x4_fused.h
 * @brief Fully fused radix-32 8×4 decomposition with streaming architecture
 *
 * @details
 * ARCHITECTURE PHILOSOPHY:
 * =======================
 * CLEAN SEPARATION OF CONCERNS:
 *   - Execution phase: Handles stripe layout, tiling, gather/scatter
 *   - Butterfly kernel: Pure SIMD math on contiguous tile-local data
 *   - Planner: Prepares twiddles, geometric constants, dispatch tables
 *
 * FUSION STRATEGY:
 *   Radix-8 (4 groups) → Cross-group twiddles → Radix-4 (8 positions)
 *   All stages execute in registers on tile-local data (~64 samples/tile)
 *   No intermediate temp buffers, no transpose operation
 *
 * MEMORY FLOW:
 *   [32 stripes][K] → gather → [32][tile] → FUSED BUTTERFLY → [32][tile] → scatter → [32 stripes][K]
 *                                            ↑
 *                                     All math happens here
 *                                     (L1-resident, <64KB)
 *
 * OPTIMIZATION FEATURES:
 * =====================
 * ✓ Geometric constants (positions 1,2,4): Broadcast, no per-k loads
 * ✓ Mini-recurrence (positions 3,5,6,7): W,W²,W³ computed once per tile
 * ✓ Trivial twiddle dispatch: Identity, ±i, ±1, W₄ fast paths
 * ✓ Gray-code BLOCKED4: 1 complex mul/step vs 3-4
 * ✓ Per-tile BLOCKED8 precomputation: W9..W16 computed per tile
 * ✓ Tile-local buffers: 64-sample tiles, L1-hot
 * ✓ Conjugate symmetry: Single twiddle storage for forward/backward
 *
 * PERFORMANCE TARGETS:
 * ===================
 * - Bandwidth: 2× reduction (eliminate temp buffer read/write)
 * - Latency: Hide via tile pipelining and prefetch
 * - Cache: L1-resident working set, optimal locality
 * - Overall: 70-85% speedup vs staged implementation
 *
 * @author Tugbars Heptaskin
 * @version 3.0 (Fully fused architecture)
 * @date 2025
 */

#ifndef FFT_RADIX32_8X4_FUSED_H
#define FFT_RADIX32_8X4_FUSED_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <immintrin.h>

//==============================================================================
// COMPILER PORTABILITY
//==============================================================================

#ifdef _MSC_VER
#define RESTRICT __restrict
#define ALIGNAS(x) __declspec(align(x))
#define FORCE_INLINE __forceinline
#define TARGET_AVX512
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#define ALIGNAS(x) __attribute__((aligned(x)))
#define FORCE_INLINE __attribute__((always_inline)) inline
#define TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define RESTRICT
#define ALIGNAS(x)
#define FORCE_INLINE inline
#define TARGET_AVX512
#endif

//==============================================================================
// ARCHITECTURE CONSTANTS
//==============================================================================

#define RADIX32_NUM_STRIPES 32  // Radix-32 produces 32 output stripes
#define RADIX32_NUM_GROUPS 4    // Radix-8 decomposition: 4 groups of 8
#define RADIX32_GROUP_SIZE 8    // Each group processes 8 frequency bins
#define RADIX32_NUM_POSITIONS 8 // Radix-4 decomposition: 8 positions

#define RADIX32_TILE_SIZE 64       // Samples per tile (L1-friendly: 64KB total)
#define RADIX32_SMALL_TILE_SIZE 32 // For very small K (K=32, K=64)

#define RADIX32_NT_THRESHOLD 1024     // Use non-temporal stores for K > 1024
#define RADIX32_PREFETCH_DISTANCE 128 // Cache lines ahead to prefetch

//==============================================================================
// PASS 1 (RADIX-8) TWIDDLE CONFIGURATION
//==============================================================================

#define RADIX32_BLOCKED16_THRESHOLD 128
#define RADIX32_BLOCKED8_THRESHOLD 1024
#define RADIX32_BLOCKED4_THRESHOLD 4096

typedef enum
{
    RADIX32_PASS1_BLOCKED16, ///< K ≤ 128: 16 twiddles stored directly
    RADIX32_PASS1_BLOCKED8,  ///< K ≤ 1024: 8 stored, 8 precomputed per tile
    RADIX32_PASS1_BLOCKED4,  ///< K ≤ 4096: 4 stored, Gray-code derivation
    RADIX32_PASS1_RECURRENCE ///< K > 4096: delta_w walking
} radix32_pass1_twiddle_mode_t;

//==============================================================================
// TRIVIAL TWIDDLE CLASSIFICATION
//==============================================================================

/**
 * @brief Twiddle type for specialized fast paths
 *
 * Enables dispatch to optimized code for common twiddle values:
 * - IDENTITY: W = 1 (no-op)
 * - PLUS_I: W = i (swap + sign: (a+ib)×i = -b+ia)
 * - MINUS_I: W = -i (swap + opposite signs)
 * - MINUS_ONE: W = -1 (negate both components)
 * - W8_CONST: W = exp(-πi/4) = √2/2(1-i) (radix-8 constant)
 * - GENERIC: Full complex multiply required
 */
typedef enum
{
    TWIDDLE_IDENTITY,
    TWIDDLE_PLUS_I,
    TWIDDLE_MINUS_I,
    TWIDDLE_MINUS_ONE,
    TWIDDLE_W8_CONST,
    TWIDDLE_GENERIC
} twiddle_type_t;

/**
 * @brief Radix-8 twiddle dispatch table for 4 groups × 8 stages
 *
 * Maps (group, stage) → twiddle_type for Pass 1 radix-8 butterflies
 *
 * Group g, stage s uses twiddle W_32^(s×g):
 * - Group 0: W^0 = 1 (all IDENTITY)
 * - Group 1, stage 4: W^4 = exp(-πi/4) (W8_CONST)
 * - Group 2, stage 4: W^8 = i (PLUS_I)
 * - Others: GENERIC complex multiply
 *
 * This table eliminates ~30% of complex multiplies in Pass 1.
 */
static const twiddle_type_t radix8_twiddle_dispatch[4][8] = {
    // Group 0: All identity (W^(s×0) = 1)
    {TWIDDLE_IDENTITY, TWIDDLE_IDENTITY, TWIDDLE_IDENTITY, TWIDDLE_IDENTITY,
     TWIDDLE_IDENTITY, TWIDDLE_IDENTITY, TWIDDLE_IDENTITY, TWIDDLE_IDENTITY},

    // Group 1: W^(s×1) = W^s
    // Stage 4: W^4 = exp(-πi/4) = √2/2(1-i)
    {TWIDDLE_IDENTITY, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC,
     TWIDDLE_W8_CONST, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC},

    // Group 2: W^(s×2)
    // Stage 4: W^8 = exp(-πi/2) = i
    {TWIDDLE_IDENTITY, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC,
     TWIDDLE_PLUS_I, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC},

    // Group 3: W^(s×3)
    {TWIDDLE_IDENTITY, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC,
     TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC, TWIDDLE_GENERIC}};

//==============================================================================
// PASS 1 TWIDDLE STRUCTURES
//==============================================================================

typedef struct
{
    const double *RESTRICT re; ///< [16 * K] blocked: W1..W16
    const double *RESTRICT im;
} radix32_pass1_blocked16_t;

/**
 * @brief BLOCKED8 with per-tile precomputation
 *
 * Storage: [8 * K] blocked: W1..W8
 * Runtime: For each tile, precompute W9..W16 = W1..W8 × W8[tile]
 */
typedef struct
{
    const double *RESTRICT re; ///< [8 * K] base twiddles
    const double *RESTRICT im;
} radix32_pass1_blocked8_t;

/**
 * @brief BLOCKED4 with Gray-code derivation
 *
 * Storage: [4 * K] blocked: W1, W2, W4, W8
 * Runtime: Gray-code walk (1 mul/step avg)
 */
typedef struct
{
    const double *RESTRICT re; ///< [4 * K] base twiddles
    const double *RESTRICT im;
} radix32_pass1_blocked4_t;

/**
 * @brief RECURRENCE with walking twiddles
 *
 * Storage: [1 * K] seed: W1 only
 * Runtime: w[k+1] = w[k] × delta_w, refresh periodically
 */
typedef struct
{
    const double *RESTRICT re; ///< [1 * K] seed twiddle W1
    const double *RESTRICT im;

    ALIGNAS(64)
    __m512d delta_w_re[31]; ///< Phase increments for s=1..31
    ALIGNAS(64)
    __m512d delta_w_im[31];

    size_t refresh_interval; ///< Iterations before refresh (64-256)
} radix32_pass1_recurrence_t;

//==============================================================================
// PASS 1 UNIFIED PLAN
//==============================================================================

typedef struct
{
    radix32_pass1_twiddle_mode_t mode;
    size_t K;

    union
    {
        radix32_pass1_blocked16_t blocked16;
        radix32_pass1_blocked8_t blocked8;
        radix32_pass1_blocked4_t blocked4;
        radix32_pass1_recurrence_t recurrence;
    } tw;

    // W8 geometric constant (used in trivial dispatch)
    // W8 = exp(-πi/4) = √2/2 (1 - i)
    ALIGNAS(64)
    __m512d w8_const_re; ///< Broadcast √2/2
    ALIGNAS(64)
    __m512d w8_const_im; ///< Broadcast -√2/2
} radix32_pass1_plan_t;

//==============================================================================
// PASS 2 (CROSS-GROUP) GEOMETRIC CONSTANTS
//==============================================================================

/**
 * @brief Cross-group twiddle constants for 8 positions
 *
 * After radix-8 on groups A,B,C,D, position m combines:
 *   [A_m, B_m × W_32^m, C_m × W_32^(2m), D_m × W_32^(3m)]
 *
 * For power-of-2 positions, these map to geometric constants:
 * - Position 1: W_32, W_32², W_32³ (generic, but constant in k)
 * - Position 2: W_16, W_16², W_16³
 * - Position 4: W_8, W_8², W_8³ (i, -1, W_8*)
 *
 * For positions 3,5,6,7: Compute W, W², W³ via mini-recurrence
 */
typedef struct
{
    // Position 0: Identity (no twiddles needed)

    // Position 1: W_32^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos1_w1_re, pos1_w1_im; ///< W_32
    ALIGNAS(64)
    __m512d pos1_w2_re, pos1_w2_im; ///< W_32²
    ALIGNAS(64)
    __m512d pos1_w3_re, pos1_w3_im; ///< W_32³

    // Position 2: W_16^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos2_w1_re, pos2_w1_im; ///< W_16
    ALIGNAS(64)
    __m512d pos2_w2_re, pos2_w2_im; ///< W_16²
    ALIGNAS(64)
    __m512d pos2_w3_re, pos2_w3_im; ///< W_16³

    // Position 3: W_32^3, W_32^6, W_32^9 (mini-recurrence base)
    ALIGNAS(64)
    __m512d pos3_w_re, pos3_w_im; ///< W_32^3 (seed for W², W³)

    // Position 4: W_8^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos4_w1_re, pos4_w1_im; ///< W_8 = √2/2(1-i)
    ALIGNAS(64)
    __m512d pos4_w2_re, pos4_w2_im; ///< W_8² = i
    ALIGNAS(64)
    __m512d pos4_w3_re, pos4_w3_im; ///< W_8³

    // Position 5: W_32^5 (seed)
    ALIGNAS(64)
    __m512d pos5_w_re, pos5_w_im;

    // Position 6: W_32^6 (seed)
    ALIGNAS(64)
    __m512d pos6_w_re, pos6_w_im;

    // Position 7: W_32^7 (seed)
    ALIGNAS(64)
    __m512d pos7_w_re, pos7_w_im;

    bool is_forward; ///< True: forward FFT, False: inverse (conjugate twiddles)
} radix32_pass2_plan_t;

//==============================================================================
// UNIFIED RADIX-32 PLAN
//==============================================================================

typedef struct
{
    size_t K;         ///< Butterflies per radix-32 stage (K = N/32)
    size_t tile_size; ///< Samples per tile (64 for normal, 32 for small K)

    radix32_pass1_plan_t pass1; ///< Radix-8 stage twiddles
    radix32_pass2_plan_t pass2; ///< Cross-group geometric constants

    // Execution hints
    bool use_nt_stores;   ///< Non-temporal stores for K > 1024
    bool use_prefetch;    ///< Software prefetch for K > 256
    size_t prefetch_dist; ///< Cache lines ahead to prefetch
} radix32_plan_t;

//==============================================================================
// PLANNING FUNCTIONS
//==============================================================================

/**
 * @brief Select Pass 1 twiddle mode based on K
 */
static inline radix32_pass1_twiddle_mode_t
radix32_choose_pass1_mode(size_t K)
{
    if (K <= RADIX32_BLOCKED16_THRESHOLD)
        return RADIX32_PASS1_BLOCKED16;
    if (K <= RADIX32_BLOCKED8_THRESHOLD)
        return RADIX32_PASS1_BLOCKED8;
    if (K <= RADIX32_BLOCKED4_THRESHOLD)
        return RADIX32_PASS1_BLOCKED4;
    return RADIX32_PASS1_RECURRENCE;
}

/**
 * @brief Initialize Pass 1 twiddle plan from reorganization system
 *
 * Handles all twiddle modes, sets up geometric constants
 *
 * @param handle Twiddle handle from reorganization (radix=8 stage)
 * @param plan Output: Pass 1 plan
 * @return 0 on success, -1 on error
 */
static inline int radix32_prepare_pass1_plan(
    const void *handle,
    radix32_pass1_plan_t *plan)
{
    // Type-cast handle (minimal coupling to reorganization system)
    const struct
    {
        int radix;
        size_t butterflies_per_stage;
        const double *materialized_re;
        const double *materialized_im;
        void *layout_specific_data;
    } *h = (const void *)handle;

    if (!h || h->radix != 8 || !h->materialized_re || !h->materialized_im || !plan)
        return -1;

    plan->K = h->butterflies_per_stage;
    plan->mode = radix32_choose_pass1_mode(plan->K);

    // Initialize W8 geometric constant (√2/2 (1-i))
    const double w8_re_scalar = 0.70710678118654752440084436210485;  // √2/2
    const double w8_im_scalar = -0.70710678118654752440084436210485; // -√2/2
    plan->w8_const_re = _mm512_set1_pd(w8_re_scalar);
    plan->w8_const_im = _mm512_set1_pd(w8_im_scalar);

    switch (plan->mode)
    {
    case RADIX32_PASS1_BLOCKED16:
        plan->tw.blocked16.re = h->materialized_re;
        plan->tw.blocked16.im = h->materialized_im;
        break;

    case RADIX32_PASS1_BLOCKED8:
        plan->tw.blocked8.re = h->materialized_re;
        plan->tw.blocked8.im = h->materialized_im;
        break;

    case RADIX32_PASS1_BLOCKED4:
        plan->tw.blocked4.re = h->materialized_re;
        plan->tw.blocked4.im = h->materialized_im;
        break;

    case RADIX32_PASS1_RECURRENCE:
        plan->tw.recurrence.re = h->materialized_re;
        plan->tw.recurrence.im = h->materialized_im;

        if (h->layout_specific_data)
        {
            typedef struct
            {
                __m512d delta_w_re[31];
                __m512d delta_w_im[31];
            } recurrence_data_t;

            const recurrence_data_t *rd = (const recurrence_data_t *)h->layout_specific_data;
            for (int i = 0; i < 31; i++)
            {
                plan->tw.recurrence.delta_w_re[i] = rd->delta_w_re[i];
                plan->tw.recurrence.delta_w_im[i] = rd->delta_w_im[i];
            }
            plan->tw.recurrence.refresh_interval = 128;
        }
        else
        {
            return -1;
        }
        break;
    }

    return 0;
}

/**
 * @brief Initialize Pass 2 geometric constants
 *
 * Computes broadcast-ready constants for positions 1,2,4
 * and seeds for mini-recurrence at positions 3,5,6,7
 *
 * @param plan Output: Pass 2 plan
 * @param is_forward True for FFT, false for IFFT
 * @return 0 on success
 */
static inline int radix32_prepare_pass2_plan(
    radix32_pass2_plan_t *plan,
    bool is_forward)
{
    if (!plan)
        return -1;

    plan->is_forward = is_forward;
    const double sign = is_forward ? 1.0 : -1.0; // Conjugate for IFFT

    // Mathematical constants
    const double PI = 3.14159265358979323846;
    const double SQRT2_2 = 0.70710678118654752440084436210485; // √2/2

    //==========================================================================
    // POSITION 1: W_32, W_32², W_32³
    //==========================================================================
    double angle1 = -sign * 2.0 * PI / 32.0; // -2π/32
    plan->pos1_w1_re = _mm512_set1_pd(cos(angle1));
    plan->pos1_w1_im = _mm512_set1_pd(sin(angle1));

    double angle2 = -sign * 4.0 * PI / 32.0; // -2π/16
    plan->pos1_w2_re = _mm512_set1_pd(cos(angle2));
    plan->pos1_w2_im = _mm512_set1_pd(sin(angle2));

    double angle3 = -sign * 6.0 * PI / 32.0; // -3π/16
    plan->pos1_w3_re = _mm512_set1_pd(cos(angle3));
    plan->pos1_w3_im = _mm512_set1_pd(sin(angle3));

    //==========================================================================
    // POSITION 2: W_16, W_16², W_16³
    //==========================================================================
    double angle_w16 = -sign * 2.0 * PI / 16.0;
    plan->pos2_w1_re = _mm512_set1_pd(cos(angle_w16));
    plan->pos2_w1_im = _mm512_set1_pd(sin(angle_w16));

    double angle_w16_2 = -sign * 4.0 * PI / 16.0;
    plan->pos2_w2_re = _mm512_set1_pd(cos(angle_w16_2));
    plan->pos2_w2_im = _mm512_set1_pd(sin(angle_w16_2));

    double angle_w16_3 = -sign * 6.0 * PI / 16.0;
    plan->pos2_w3_re = _mm512_set1_pd(cos(angle_w16_3));
    plan->pos2_w3_im = _mm512_set1_pd(sin(angle_w16_3));

    //==========================================================================
    // POSITION 3: W_32^3 (seed for mini-recurrence)
    //==========================================================================
    double angle_p3 = -sign * 3.0 * 2.0 * PI / 32.0;
    plan->pos3_w_re = _mm512_set1_pd(cos(angle_p3));
    plan->pos3_w_im = _mm512_set1_pd(sin(angle_p3));

    //==========================================================================
    // POSITION 4: W_32^4, W_32^8, W_32^12 = W_8^1, W_8^2, W_8^3
    // CORRECTED: Use angle computation to match convention of other positions
    //==========================================================================
    {
        // Position 4 corresponds to k = 4, 8, 12 in W_32^k
        // Following your convention: angle = -sign * 2π * k / 32

        double angle_4 = -sign * 2.0 * PI * 4.0 / 32.0;   // -sign * π/4
        double angle_8 = -sign * 2.0 * PI * 8.0 / 32.0;   // -sign * π/2
        double angle_12 = -sign * 2.0 * PI * 12.0 / 32.0; // -sign * 3π/4

        plan->pos4_w1_re = _mm512_set1_pd(cos(angle_4));
        plan->pos4_w1_im = _mm512_set1_pd(sin(angle_4));

        plan->pos4_w2_re = _mm512_set1_pd(cos(angle_8));
        plan->pos4_w2_im = _mm512_set1_pd(sin(angle_8));

        plan->pos4_w3_re = _mm512_set1_pd(cos(angle_12));
        plan->pos4_w3_im = _mm512_set1_pd(sin(angle_12));
    }

    //==========================================================================
    // POSITIONS 5, 6, 7: Seeds for mini-recurrence
    //==========================================================================
    double angle_p5 = -sign * 5.0 * 2.0 * PI / 32.0;
    plan->pos5_w_re = _mm512_set1_pd(cos(angle_p5));
    plan->pos5_w_im = _mm512_set1_pd(sin(angle_p5));

    double angle_p6 = -sign * 6.0 * 2.0 * PI / 32.0;
    plan->pos6_w_re = _mm512_set1_pd(cos(angle_p6));
    plan->pos6_w_im = _mm512_set1_pd(sin(angle_p6));

    double angle_p7 = -sign * 7.0 * 2.0 * PI / 32.0;
    plan->pos7_w_re = _mm512_set1_pd(cos(angle_p7));
    plan->pos7_w_im = _mm512_set1_pd(sin(angle_p7));

    return 0;
}

/**
 * @brief Create unified radix-32 plan
 *
 * @param handle_pass1 Twiddle handle for radix-8 stage
 * @param K Butterflies per stage
 * @param is_forward True for FFT, false for IFFT
 * @param plan Output: Complete radix-32 plan
 * @return 0 on success, -1 on error
 */
static inline int radix32_create_plan(
    const void *handle_pass1,
    size_t K,
    bool is_forward,
    radix32_plan_t *plan)
{
    if (!plan)
        return -1;

    plan->K = K;

    // Determine tile size
    if (K <= RADIX32_SMALL_TILE_SIZE)
    {
        plan->tile_size = K; // Process entire K in one tile
    }
    else
    {
        plan->tile_size = RADIX32_TILE_SIZE;
    }

    // Execution hints
    plan->use_nt_stores = (K > RADIX32_NT_THRESHOLD);
    plan->use_prefetch = (K > 256);
    plan->prefetch_dist = RADIX32_PREFETCH_DISTANCE;

    // Initialize Pass 1 (radix-8 twiddles)
    if (radix32_prepare_pass1_plan(handle_pass1, &plan->pass1) != 0)
        return -1;

    // Initialize Pass 2 (geometric constants)
    if (radix32_prepare_pass2_plan(&plan->pass2, is_forward) != 0)
        return -1;

    return 0;
}

//==============================================================================
// GATHER/SCATTER: STRIPE LAYOUT ↔ TILE-LOCAL LAYOUT
//==============================================================================

/**
 * @brief Gather input data from 32 stripes into contiguous tile
 *
 * Converts from blocked stripe layout to tile-local contiguous layout:
 *   Source: in[stripe][k_offset..k_offset+tile_size-1]  (32 stripes, stride K)
 *   Dest:   tile[stripe][0..tile_size-1]                (contiguous per stripe)
 *
 * Memory pattern:
 *   - 32 separate reads (one per stripe), each contiguous
 *   - Compiler vectorizes inner loop effectively
 *   - Total: tile_size × 32 × 16 bytes (re+im) = 32KB @ tile_size=64
 *
 * @param in_re Source real [32 stripes][K]
 * @param in_im Source imag [32 stripes][K]
 * @param K Total butterflies per stage
 * @param k_offset Starting position in K dimension
 * @param tile_size Number of samples to gather (≤ RADIX32_TILE_SIZE)
 * @param tile_re Destination real [32][tile_size]
 * @param tile_im Destination imag [32][tile_size]
 */
TARGET_AVX512
FORCE_INLINE void radix32_gather_stripes_to_tile(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K,
    size_t k_offset,
    size_t tile_size,
    double *RESTRICT tile_re,
    double *RESTRICT tile_im)
{
    // Gather each of 32 stripes into contiguous tile-local buffer
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &in_re[stripe * K + k_offset];
        const double *src_im = &in_im[stripe * K + k_offset];
        double *dst_re = &tile_re[stripe * tile_size];
        double *dst_im = &tile_im[stripe * tile_size];

        // Vectorized copy (compiler auto-vectorizes, or use explicit SIMD)
        size_t i = 0;

        // Main loop: process 8 doubles at a time
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_loadu_pd(&src_re[i]);
            __m512d vi = _mm512_loadu_pd(&src_im[i]);
            _mm512_store_pd(&dst_re[i], vr);
            _mm512_store_pd(&dst_im[i], vi);
        }

        // Tail: handle remaining samples (scalar fallback)
        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Gather with software prefetch for large K
 *
 * Identical to radix32_gather_stripes_to_tile but adds prefetch hints
 * for the next tile to hide memory latency.
 *
 * @param in_re Source real [32 stripes][K]
 * @param in_im Source imag [32 stripes][K]
 * @param K Total butterflies per stage
 * @param k_offset Starting position in K dimension
 * @param tile_size Number of samples to gather
 * @param prefetch_offset Offset for next tile prefetch (in samples)
 * @param tile_re Destination real [32][tile_size]
 * @param tile_im Destination imag [32][tile_size]
 */
TARGET_AVX512
FORCE_INLINE void radix32_gather_stripes_to_tile_prefetch(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K,
    size_t k_offset,
    size_t tile_size,
    size_t prefetch_offset,
    double *RESTRICT tile_re,
    double *RESTRICT tile_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &in_re[stripe * K + k_offset];
        const double *src_im = &in_im[stripe * K + k_offset];
        double *dst_re = &tile_re[stripe * tile_size];
        double *dst_im = &tile_im[stripe * tile_size];

        // Prefetch next tile if within bounds
        if (k_offset + prefetch_offset < K)
        {
            _mm_prefetch((const char *)&in_re[stripe * K + k_offset + prefetch_offset], _MM_HINT_T0);
            _mm_prefetch((const char *)&in_im[stripe * K + k_offset + prefetch_offset], _MM_HINT_T0);
        }

        // Vectorized copy
        size_t i = 0;
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_loadu_pd(&src_re[i]);
            __m512d vi = _mm512_loadu_pd(&src_im[i]);
            _mm512_store_pd(&dst_re[i], vr);
            _mm512_store_pd(&dst_im[i], vi);
        }

        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Scatter tile-local data back to 32 stripes (normal stores)
 *
 * Converts from tile-local contiguous layout back to blocked stripes:
 *   Source: tile[stripe][0..tile_size-1]                (contiguous per stripe)
 *   Dest:   out[stripe][k_offset..k_offset+tile_size-1] (32 stripes, stride K)
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K]
 * @param out_im Destination imag [32 stripes][K]
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &tile_re[stripe * tile_size];
        const double *src_im = &tile_im[stripe * tile_size];
        double *dst_re = &out_re[stripe * K + k_offset];
        double *dst_im = &out_im[stripe * K + k_offset];

        // Vectorized copy
        size_t i = 0;
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_load_pd(&src_re[i]);
            __m512d vi = _mm512_load_pd(&src_im[i]);
            _mm512_storeu_pd(&dst_re[i], vr);
            _mm512_storeu_pd(&dst_im[i], vi);
        }

        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Scatter with non-temporal stores for large K
 *
 * Uses streaming stores (_mm512_stream_pd) to avoid cache pollution
 * when output data won't be read back soon.
 *
 * CRITICAL: Requires 64-byte alignment of destination addresses.
 * Use only when K > RADIX32_NT_THRESHOLD and outputs are aligned.
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K] (64-byte aligned)
 * @param out_im Destination imag [32 stripes][K] (64-byte aligned)
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes_nt(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &tile_re[stripe * tile_size];
        const double *src_im = &tile_im[stripe * tile_size];
        double *dst_re = &out_re[stripe * K + k_offset];
        double *dst_im = &out_im[stripe * K + k_offset];

        // Non-temporal streaming stores (requires alignment)
        size_t i = 0;

        // Ensure alignment: k_offset must be 8-aligned for NT stores
        // If not aligned, use regular stores for first few elements
        size_t align_offset = ((uintptr_t)dst_re & 63) / 8; // Doubles to alignment
        if (align_offset > 0)
        {
            size_t align_count = (8 - align_offset) & 7;
            for (i = 0; i < align_count && i < tile_size; i++)
            {
                dst_re[i] = src_re[i];
                dst_im[i] = src_im[i];
            }
        }

        // Main loop: NT stores (8 doubles at a time)
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_load_pd(&src_re[i]);
            __m512d vi = _mm512_load_pd(&src_im[i]);
            _mm512_stream_pd(&dst_re[i], vr);
            _mm512_stream_pd(&dst_im[i], vi);
        }

        // Tail: regular stores
        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }

    // Ensure NT stores are visible before function returns
    _mm_sfence();
}

/**
 * @brief Adaptive scatter dispatcher
 *
 * Selects between normal and non-temporal stores based on plan configuration.
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K]
 * @param out_im Destination imag [32 stripes][K]
 * @param use_nt True to use non-temporal stores
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes_auto(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    bool use_nt)
{
    if (use_nt)
    {
        radix32_scatter_tile_to_stripes_nt(tile_re, tile_im, tile_size,
                                           k_offset, K, out_re, out_im);
    }
    else
    {
        radix32_scatter_tile_to_stripes(tile_re, tile_im, tile_size,
                                        k_offset, K, out_re, out_im);
    }
}

//==============================================================================
// TILE MANAGEMENT: ALLOCATION AND LIFETIME
//==============================================================================

/**
 * @brief Tile workspace for radix-32 execution
 *
 * Holds tile-local buffers for input and output.
 * Total size: 4 × tile_size × 32 × 8 bytes = 32 KB @ tile_size=64
 *
 * Allocated once per execution call, lives on stack or in thread-local storage.
 */
typedef struct
{
    ALIGNAS(64)
    double input_re[RADIX32_NUM_STRIPES * RADIX32_TILE_SIZE];
    ALIGNAS(64)
    double input_im[RADIX32_NUM_STRIPES * RADIX32_TILE_SIZE];
    ALIGNAS(64)
    double output_re[RADIX32_NUM_STRIPES * RADIX32_TILE_SIZE];
    ALIGNAS(64)
    double output_im[RADIX32_NUM_STRIPES * RADIX32_TILE_SIZE];
} radix32_tile_workspace_t;

/**
 * @brief Small tile workspace for K ≤ 64
 *
 * Smaller buffers for tiny transforms that fit entirely in L1.
 * Total size: 4 × 32 × 32 × 8 bytes = 32 KB
 */
typedef struct
{
    ALIGNAS(64)
    double input_re[RADIX32_NUM_STRIPES * RADIX32_SMALL_TILE_SIZE];
    ALIGNAS(64)
    double input_im[RADIX32_NUM_STRIPES * RADIX32_SMALL_TILE_SIZE];
    ALIGNAS(64)
    double output_re[RADIX32_NUM_STRIPES * RADIX32_SMALL_TILE_SIZE];
    ALIGNAS(64)
    double output_im[RADIX32_NUM_STRIPES * RADIX32_SMALL_TILE_SIZE];
} radix32_small_tile_workspace_t;

//==============================================================================
// COMPLEX ARITHMETIC PRIMITIVES (Inline helpers for butterfly)
//==============================================================================

/**
 * @brief Complex multiply: (a + ib) × (c + id) → (result_re + i×result_im)
 *
 * Formula:
 *   result_re = a×c - b×d
 *   result_im = a×d + b×c
 *
 * Uses FMA instructions for optimal performance (2 FMA + 2 MUL = 4 uops).
 */
TARGET_AVX512
FORCE_INLINE void cmul_avx512(
    __m512d a_re, __m512d a_im,
    __m512d b_re, __m512d b_im,
    __m512d *RESTRICT result_re,
    __m512d *RESTRICT result_im)
{
    // result_re = a_re * b_re - a_im * b_im
    *result_re = _mm512_fmsub_pd(a_re, b_re, _mm512_mul_pd(a_im, b_im));

    // result_im = a_re * b_im + a_im * b_re
    *result_im = _mm512_fmadd_pd(a_re, b_im, _mm512_mul_pd(a_im, b_re));
}

/**
 * @brief In-place complex multiply: (a + ib) ← (a + ib) × (c + id)
 */
TARGET_AVX512
FORCE_INLINE void cmul_inplace_avx512(
    __m512d *RESTRICT a_re, __m512d *RESTRICT a_im,
    __m512d b_re, __m512d b_im)
{
    __m512d tmp_re = _mm512_fmsub_pd(*a_re, b_re, _mm512_mul_pd(*a_im, b_im));
    __m512d tmp_im = _mm512_fmadd_pd(*a_re, b_im, _mm512_mul_pd(*a_im, b_re));
    *a_re = tmp_re;
    *a_im = tmp_im;
}

/**
 * @brief Multiply by i: (a + ib) × i = -b + ia
 *
 * FAST PATH: Just swap and negate (no FMAs needed).
 */
TARGET_AVX512
FORCE_INLINE void cmul_by_i_avx512(
    __m512d a_re, __m512d a_im,
    __m512d *RESTRICT result_re,
    __m512d *RESTRICT result_im)
{
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    *result_re = _mm512_xor_pd(a_im, sign_mask); // -a_im
    *result_im = a_re;                           // +a_re
}

/**
 * @brief Multiply by -i: (a + ib) × (-i) = b - ia
 */
TARGET_AVX512
FORCE_INLINE void cmul_by_minus_i_avx512(
    __m512d a_re, __m512d a_im,
    __m512d *RESTRICT result_re,
    __m512d *RESTRICT result_im)
{
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    *result_re = a_im;                           // +a_im
    *result_im = _mm512_xor_pd(a_re, sign_mask); // -a_re
}

/**
 * @brief Multiply by -1: (a + ib) × (-1) = -a - ib
 */
TARGET_AVX512
FORCE_INLINE void cmul_by_minus_one_avx512(
    __m512d a_re, __m512d a_im,
    __m512d *RESTRICT result_re,
    __m512d *RESTRICT result_im)
{
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    *result_re = _mm512_xor_pd(a_re, sign_mask);
    *result_im = _mm512_xor_pd(a_im, sign_mask);
}

/**
 * @brief Trivial twiddle dispatcher
 *
 * Applies twiddle based on type classification (fast paths for common values).
 *
 * @param type Twiddle type (IDENTITY, PLUS_I, etc.)
 * @param data_re Input/output real
 * @param data_im Input/output imag
 * @param w_re Twiddle real (used only for GENERIC and W8_CONST)
 * @param w_im Twiddle imag (used only for GENERIC and W8_CONST)
 * @param w8_const_re Broadcast W8 real (√2/2)
 * @param w8_const_im Broadcast W8 imag (-√2/2)
 */
TARGET_AVX512
FORCE_INLINE void apply_twiddle_classified_avx512(
    twiddle_type_t type,
    __m512d *RESTRICT data_re,
    __m512d *RESTRICT data_im,
    __m512d w_re,
    __m512d w_im,
    __m512d w8_const_re,
    __m512d w8_const_im)
{
    switch (type)
    {
    case TWIDDLE_IDENTITY:
        // No-op: data unchanged
        break;

    case TWIDDLE_PLUS_I:
    {
        // Multiply by i: (a+ib) × i = -b + ia
        __m512d tmp_re, tmp_im;
        cmul_by_i_avx512(*data_re, *data_im, &tmp_re, &tmp_im);
        *data_re = tmp_re;
        *data_im = tmp_im;
        break;
    }

    case TWIDDLE_MINUS_I:
    {
        // Multiply by -i: (a+ib) × (-i) = b - ia
        __m512d tmp_re, tmp_im;
        cmul_by_minus_i_avx512(*data_re, *data_im, &tmp_re, &tmp_im);
        *data_re = tmp_re;
        *data_im = tmp_im;
        break;
    }

    case TWIDDLE_MINUS_ONE:
    {
        // Multiply by -1: (a+ib) × (-1) = -a - ib
        __m512d tmp_re, tmp_im;
        cmul_by_minus_one_avx512(*data_re, *data_im, &tmp_re, &tmp_im);
        *data_re = tmp_re;
        *data_im = tmp_im;
        break;
    }

    case TWIDDLE_W8_CONST:
    {
        // Multiply by W8 = √2/2(1-i) using precomputed broadcast
        cmul_inplace_avx512(data_re, data_im, w8_const_re, w8_const_im);
        break;
    }

    case TWIDDLE_GENERIC:
    {
        // Full complex multiply
        cmul_inplace_avx512(data_re, data_im, w_re, w_im);
        break;
    }
    }
}

//==============================================================================
// FIX #2: MINI-RECURRENCE IMPLEMENTATION (COMPLETE)
//==============================================================================

/**
 * @brief Compute w^1, w^2, w^3 from seed twiddle w via mini-recurrence
 *
 * Given: w = exp(-j*2π*m/32) as (w_re, w_im)
 * Compute: w^1 = w, w^2 = w*w, w^3 = w^2*w
 *
 * Uses 2 FMAs for w^2 (complex square) + 2 FMAs for w^3 = 4 FMAs total
 * vs. 6 FMAs for three independent complex muls
 *
 * @param w_re Seed twiddle real
 * @param w_im Seed twiddle imag
 * @param w1_re Output: w^1 real (just copy of w_re)
 * @param w1_im Output: w^1 imag (just copy of w_im)
 * @param w2_re Output: w^2 real
 * @param w2_im Output: w^2 imag
 * @param w3_re Output: w^3 real
 * @param w3_im Output: w^3 imag
 */
TARGET_AVX512
FORCE_INLINE void cross_twiddle_powers3(
    __m512d w_re, __m512d w_im,
    __m512d *RESTRICT w1_re, __m512d *RESTRICT w1_im,
    __m512d *RESTRICT w2_re, __m512d *RESTRICT w2_im,
    __m512d *RESTRICT w3_re, __m512d *RESTRICT w3_im)
{
    // w^1 = w (trivial copy)
    *w1_re = w_re;
    *w1_im = w_im;

    // w^2 = w * w (complex square via FMA)
    // Re(w^2) = w_re^2 - w_im^2
    // Im(w^2) = 2 * w_re * w_im
    __m512d w2r = _mm512_fmsub_pd(w_re, w_re, _mm512_mul_pd(w_im, w_im));
    __m512d w2i = _mm512_fmadd_pd(w_re, w_im, _mm512_mul_pd(w_im, w_re)); // 2*Re*Im

    *w2_re = w2r;
    *w2_im = w2i;

    // w^3 = w^2 * w (complex multiply via FMA)
    // Re(w^3) = w2_re * w_re - w2_im * w_im
    // Im(w^3) = w2_re * w_im + w2_im * w_re
    *w3_re = _mm512_fmsub_pd(w2r, w_re, _mm512_mul_pd(w2i, w_im));
    *w3_im = _mm512_fmadd_pd(w2r, w_im, _mm512_mul_pd(w2i, w_re));
}

//==============================================================================
// FIX #3: TAIL HANDLING HELPERS (COMPLETE)
//==============================================================================

/**
 * @brief Masked load for tail handling
 *
 * Loads up to 7 elements safely with masking
 */
TARGET_AVX512
FORCE_INLINE __m512d masked_load_tail(const double *ptr, size_t tail)
{
    if (tail >= 8)
    {
        return _mm512_loadu_pd(ptr);
    }
    else if (tail > 0)
    {
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);
        return _mm512_maskz_loadu_pd(mask, ptr);
    }
    else
    {
        return _mm512_setzero_pd();
    }
}

/**
 * @brief Masked store for tail handling
 *
 * Stores up to 7 elements safely with masking
 */
TARGET_AVX512
FORCE_INLINE void masked_store_tail(double *ptr, __m512d value, size_t tail)
{
    if (tail >= 8)
    {
        _mm512_storeu_pd(ptr, value);
    }
    else if (tail > 0)
    {
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);
        _mm512_mask_storeu_pd(ptr, mask, value);
    }
}

//==============================================================================
// OPTIMIZATION #2: Helpers for ±j rotations (reduces live ranges)
//==============================================================================

/**
 * @brief Rotation by -j: (a+jb)*(-j) = b - ja
 */
TARGET_AVX512
FORCE_INLINE void rot_neg_j(__m512d re, __m512d im, __m512d sign_mask,
                            __m512d *RESTRICT r2, __m512d *RESTRICT i2)
{
    *r2 = im;                           // New real = old imag
    *i2 = _mm512_xor_pd(re, sign_mask); // New imag = -old real
}

/**
 * @brief Rotation by +j: (a+jb)*(+j) = -b + ja
 */
TARGET_AVX512
FORCE_INLINE void rot_pos_j(__m512d re, __m512d im, __m512d sign_mask,
                            __m512d *RESTRICT r2, __m512d *RESTRICT i2)
{
    *r2 = _mm512_xor_pd(im, sign_mask); // New real = -old imag
    *i2 = re;                           // New imag = old real
}

//==============================================================================
// FIX #2: LOAD/STORE WRAPPERS FOR UNIFORM MACRO INTERFACE
//==============================================================================

/**
 * @brief Aligned load (unmasked)
 */
TARGET_AVX512
FORCE_INLINE __m512d load_aligned(const double *RESTRICT ptr)
{
    return _mm512_load_pd(ptr);
}

/**
 * @brief Masked load (zeros invalid lanes)
 */
TARGET_AVX512
FORCE_INLINE __m512d load_masked(__mmask8 mask, const double *RESTRICT ptr)
{
    return _mm512_maskz_loadu_pd(mask, ptr);
}

/**
 * @brief Aligned store (unmasked)
 */
TARGET_AVX512
FORCE_INLINE void store_aligned(double *RESTRICT ptr, __m512d val)
{
    _mm512_store_pd(ptr, val);
}

/**
 * @brief Masked store (preserves invalid lanes)
 */
TARGET_AVX512
FORCE_INLINE void store_masked(double *RESTRICT ptr, __mmask8 mask, __m512d val)
{
    _mm512_mask_storeu_pd(ptr, mask, val);
}

//==============================================================================
// TWO-PAIR WAVE OPTIMIZATION: Stage-1 Reuse
//==============================================================================
/**
 * @brief Wave architecture: Compute stage-1 once, emit two pairs
 *
 * EVEN WAVE (t0..t3):
 * 1. Compute t0,t1,t2,t3 from x0..x7 (4 butterflies)
 * 2. Call emit_pair_04_from_t0123 → bins (0,4)
 * 3. Call emit_pair_26_from_t0123 → bins (2,6)
 * 4. Free t0..t3
 *
 * ODD WAVE (t4..t7):
 * 1. Compute t4,t5,t6,t7 from x0..x7 (4 butterflies)
 * 2. Call emit_pair_15_from_t4567 → bins (1,5)
 * 3. Call emit_pair_37_from_t4567 → bins (3,7)
 * 4. Free t4..t7
 *
 * BENEFIT: Halves stage-1 work, keeps only 4 t* live at a time
 */

/**
 * @brief Emit pair (0,4) from precomputed t0,t1,t2,t3
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_04_from_t0123_avx512(
    __m512d t0_re, __m512d t0_im,
    __m512d t1_re, __m512d t1_im,
    __m512d t2_re, __m512d t2_im,
    __m512d t3_re, __m512d t3_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y4_re, __m512d *RESTRICT y4_im)
{
    // Stage 2: Only u0, u1
    __m512d u0_re = _mm512_add_pd(t0_re, t2_re);
    __m512d u0_im = _mm512_add_pd(t0_im, t2_im);

    __m512d u1_re = _mm512_add_pd(t1_re, t3_re);
    __m512d u1_im = _mm512_add_pd(t1_im, t3_im);

    // Stage 3: Final butterfly (W8^0 = 1, identity)
    *y0_re = _mm512_add_pd(u0_re, u1_re);
    *y0_im = _mm512_add_pd(u0_im, u1_im);
    *y4_re = _mm512_sub_pd(u0_re, u1_re);
    *y4_im = _mm512_sub_pd(u0_im, u1_im);
}

/**
 * @brief Emit pair (2,6) from precomputed t0,t1,t2,t3
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_26_from_t0123_avx512(
    __m512d t0_re, __m512d t0_im,
    __m512d t1_re, __m512d t1_im,
    __m512d t2_re, __m512d t2_im,
    __m512d t3_re, __m512d t3_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y6_re, __m512d *RESTRICT y6_im,
    __m512d sign_mask)
{
    // Stage 2: Only u2, u3
    __m512d u2_re = _mm512_sub_pd(t0_re, t2_re);
    __m512d u2_im = _mm512_sub_pd(t0_im, t2_im);

    __m512d u3_re = _mm512_sub_pd(t1_re, t3_re);
    __m512d u3_im = _mm512_sub_pd(t1_im, t3_im);

    // Stage 3: Apply W8^2 = -j
    __m512d u3_tw_re, u3_tw_im;
    rot_neg_j(u3_re, u3_im, sign_mask, &u3_tw_re, &u3_tw_im);

    // Final butterfly
    *y2_re = _mm512_add_pd(u2_re, u3_tw_re);
    *y2_im = _mm512_add_pd(u2_im, u3_tw_im);
    *y6_re = _mm512_sub_pd(u2_re, u3_tw_re);
    *y6_im = _mm512_sub_pd(u2_im, u3_tw_im);
}

/**
 * @brief Emit pair (1,5) from precomputed t4,t5,t6,t7
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_15_from_t4567_avx512(
    __m512d t4_re, __m512d t4_im,
    __m512d t5_re, __m512d t5_im,
    __m512d t6_re, __m512d t6_im,
    __m512d t7_re, __m512d t7_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y5_re, __m512d *RESTRICT y5_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    // Stage 2: Apply W4^2 = -j, compute only u4, u5
    __m512d t6_tw_re, t6_tw_im;
    rot_neg_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_neg_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u4_re = _mm512_add_pd(t4_re, t6_tw_re);
    __m512d u4_im = _mm512_add_pd(t4_im, t6_tw_im);

    __m512d u5_re = _mm512_add_pd(t5_re, t7_tw_re);
    __m512d u5_im = _mm512_add_pd(t5_im, t7_tw_im);

    // Stage 3: Apply W8^1 = √2/2(1-i) with XOR+ADD dependency breaking
    __m512d sum5 = _mm512_add_pd(u5_re, u5_im);
    __m512d nre5 = _mm512_xor_pd(u5_re, sign_mask);
    __m512d diff5 = _mm512_add_pd(u5_im, nre5);

    __m512d u5_tw_re = _mm512_mul_pd(sqrt2_2, sum5);
    __m512d u5_tw_im = _mm512_mul_pd(sqrt2_2, diff5);

    // Final butterfly
    *y1_re = _mm512_add_pd(u4_re, u5_tw_re);
    *y1_im = _mm512_add_pd(u4_im, u5_tw_im);
    *y5_re = _mm512_sub_pd(u4_re, u5_tw_re);
    *y5_im = _mm512_sub_pd(u4_im, u5_tw_im);
}

/**
 * @brief Emit pair (3,7) from precomputed t4,t5,t6,t7
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_37_from_t4567_avx512(
    __m512d t4_re, __m512d t4_im,
    __m512d t5_re, __m512d t5_im,
    __m512d t6_re, __m512d t6_im,
    __m512d t7_re, __m512d t7_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d *RESTRICT y7_re, __m512d *RESTRICT y7_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    // Stage 2: Apply W4^2 = -j, compute only u6, u7
    __m512d t6_tw_re, t6_tw_im;
    rot_neg_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_neg_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u6_re = _mm512_sub_pd(t4_re, t6_tw_re);
    __m512d u6_im = _mm512_sub_pd(t4_im, t6_tw_im);

    __m512d u7_re = _mm512_sub_pd(t5_re, t7_tw_re);
    __m512d u7_im = _mm512_sub_pd(t5_im, t7_tw_im);

    // Stage 3: Apply W8^3 = √2/2(-1-i) with XOR+ADD dependency breaking
    __m512d sum7 = _mm512_add_pd(u7_re, u7_im);
    __m512d nre7 = _mm512_xor_pd(u7_re, sign_mask);
    __m512d diff7 = _mm512_add_pd(u7_im, nre7);

    __m512d u7_tw_re_tmp = _mm512_mul_pd(sqrt2_2, sum7);
    __m512d u7_tw_re = _mm512_xor_pd(u7_tw_re_tmp, sign_mask);
    __m512d u7_tw_im = _mm512_mul_pd(sqrt2_2, diff7);

    // Final butterfly
    *y3_re = _mm512_add_pd(u6_re, u7_tw_re);
    *y3_im = _mm512_add_pd(u6_im, u7_tw_im);
    *y7_re = _mm512_sub_pd(u6_re, u7_tw_re);
    *y7_im = _mm512_sub_pd(u6_im, u7_tw_im);
}

/**
 * @brief Helper: Compute t0,t1,t2,t3 from x0..x7 (even wave stage-1)
 */
TARGET_AVX512
FORCE_INLINE void radix8_compute_t0123_avx512(
    __m512d x0_re, __m512d x0_im,
    __m512d x1_re, __m512d x1_im,
    __m512d x2_re, __m512d x2_im,
    __m512d x3_re, __m512d x3_im,
    __m512d x4_re, __m512d x4_im,
    __m512d x5_re, __m512d x5_im,
    __m512d x6_re, __m512d x6_im,
    __m512d x7_re, __m512d x7_im,
    __m512d *RESTRICT t0_re, __m512d *RESTRICT t0_im,
    __m512d *RESTRICT t1_re, __m512d *RESTRICT t1_im,
    __m512d *RESTRICT t2_re, __m512d *RESTRICT t2_im,
    __m512d *RESTRICT t3_re, __m512d *RESTRICT t3_im)
{
    *t0_re = _mm512_add_pd(x0_re, x4_re);
    *t0_im = _mm512_add_pd(x0_im, x4_im);

    *t1_re = _mm512_add_pd(x1_re, x5_re);
    *t1_im = _mm512_add_pd(x1_im, x5_im);

    *t2_re = _mm512_add_pd(x2_re, x6_re);
    *t2_im = _mm512_add_pd(x2_im, x6_im);

    *t3_re = _mm512_add_pd(x3_re, x7_re);
    *t3_im = _mm512_add_pd(x3_im, x7_im);
}

/**
 * @brief Helper: Compute t4,t5,t6,t7 from x0..x7 (odd wave stage-1)
 */
TARGET_AVX512
FORCE_INLINE void radix8_compute_t4567_avx512(
    __m512d x0_re, __m512d x0_im,
    __m512d x1_re, __m512d x1_im,
    __m512d x2_re, __m512d x2_im,
    __m512d x3_re, __m512d x3_im,
    __m512d x4_re, __m512d x4_im,
    __m512d x5_re, __m512d x5_im,
    __m512d x6_re, __m512d x6_im,
    __m512d x7_re, __m512d x7_im,
    __m512d *RESTRICT t4_re, __m512d *RESTRICT t4_im,
    __m512d *RESTRICT t5_re, __m512d *RESTRICT t5_im,
    __m512d *RESTRICT t6_re, __m512d *RESTRICT t6_im,
    __m512d *RESTRICT t7_re, __m512d *RESTRICT t7_im)
{
    *t4_re = _mm512_sub_pd(x0_re, x4_re);
    *t4_im = _mm512_sub_pd(x0_im, x4_im);

    *t5_re = _mm512_sub_pd(x1_re, x5_re);
    *t5_im = _mm512_sub_pd(x1_im, x5_im);

    *t6_re = _mm512_sub_pd(x2_re, x6_re);
    *t6_im = _mm512_sub_pd(x2_im, x6_im);

    *t7_re = _mm512_sub_pd(x3_re, x7_re);
    *t7_im = _mm512_sub_pd(x3_im, x7_im);
}

/**
 * @brief Emit pair (1, 5) - Backward FFT (CORRECTED)
 *
 * W8^1 = √2/2(1+i): z × W8^1 = √2/2[(a-b) + j(a+b)]
 * CRITICAL: Real part is (a-b), NOT (b-a)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_15_backward_avx512(
    __m512d x0_re, __m512d x0_im,
    __m512d x1_re, __m512d x1_im,
    __m512d x2_re, __m512d x2_im,
    __m512d x3_re, __m512d x3_im,
    __m512d x4_re, __m512d x4_im,
    __m512d x5_re, __m512d x5_im,
    __m512d x6_re, __m512d x6_im,
    __m512d x7_re, __m512d x7_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y5_re, __m512d *RESTRICT y5_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    //==========================================================================
    // STAGE 1: Only t4, t5, t6, t7
    //==========================================================================

    __m512d t4_re = _mm512_sub_pd(x0_re, x4_re);
    __m512d t4_im = _mm512_sub_pd(x0_im, x4_im);

    __m512d t5_re = _mm512_sub_pd(x1_re, x5_re);
    __m512d t5_im = _mm512_sub_pd(x1_im, x5_im);

    __m512d t6_re = _mm512_sub_pd(x2_re, x6_re);
    __m512d t6_im = _mm512_sub_pd(x2_im, x6_im);

    __m512d t7_re = _mm512_sub_pd(x3_re, x7_re);
    __m512d t7_im = _mm512_sub_pd(x3_im, x7_im);

    //==========================================================================
    // STAGE 2: Apply W4^2 = +j (BACKWARD), Compute Only u4, u5
    //==========================================================================

    __m512d t6_tw_re, t6_tw_im;
    rot_pos_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_pos_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u4_re = _mm512_add_pd(t4_re, t6_tw_re);
    __m512d u4_im = _mm512_add_pd(t4_im, t6_tw_im);

    __m512d u5_re = _mm512_add_pd(t5_re, t7_tw_re);
    __m512d u5_im = _mm512_add_pd(t5_im, t7_tw_im);

    //==========================================================================
    // STAGE 3: Apply W8^1 = √2/2(1+i) - CORRECTED
    //==========================================================================
    // Formula: √2/2[(a-b) + j(a+b)]
    // Use XOR+ADD to compute (a-b) without SUB dependency

    __m512d nb5 = _mm512_xor_pd(u5_im, sign_mask); // -b
    __m512d diff5 = _mm512_add_pd(u5_re, nb5);     // a - b (CORRECTED)
    __m512d sum5 = _mm512_add_pd(u5_re, u5_im);    // a + b

    __m512d u5_tw_re = _mm512_mul_pd(sqrt2_2, diff5); // √2/2 * (a-b)
    __m512d u5_tw_im = _mm512_mul_pd(sqrt2_2, sum5);  // √2/2 * (a+b)

    // Final butterfly
    *y1_re = _mm512_add_pd(u4_re, u5_tw_re);
    *y1_im = _mm512_add_pd(u4_im, u5_tw_im);
    *y5_re = _mm512_sub_pd(u4_re, u5_tw_re);
    *y5_im = _mm512_sub_pd(u4_im, u5_tw_im);
}

/**
 * @brief Emit pair (3, 7) - Backward FFT (CORRECTED)
 *
 * W8^3 = √2/2(-1+i): z × W8^3 = √2/2[-(a+b) + j(a-b)]
 * CRITICAL: Real part is -(a+b), imag part is (a-b)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_37_backward_avx512(
    __m512d x0_re, __m512d x0_im,
    __m512d x1_re, __m512d x1_im,
    __m512d x2_re, __m512d x2_im,
    __m512d x3_re, __m512d x3_im,
    __m512d x4_re, __m512d x4_im,
    __m512d x5_re, __m512d x5_im,
    __m512d x6_re, __m512d x6_im,
    __m512d x7_re, __m512d x7_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d *RESTRICT y7_re, __m512d *RESTRICT y7_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    //==========================================================================
    // STAGE 1: Only t4, t5, t6, t7
    //==========================================================================

    __m512d t4_re = _mm512_sub_pd(x0_re, x4_re);
    __m512d t4_im = _mm512_sub_pd(x0_im, x4_im);

    __m512d t5_re = _mm512_sub_pd(x1_re, x5_re);
    __m512d t5_im = _mm512_sub_pd(x1_im, x5_im);

    __m512d t6_re = _mm512_sub_pd(x2_re, x6_re);
    __m512d t6_im = _mm512_sub_pd(x2_im, x6_im);

    __m512d t7_re = _mm512_sub_pd(x3_re, x7_re);
    __m512d t7_im = _mm512_sub_pd(x3_im, x7_im);

    //==========================================================================
    // STAGE 2: Apply W4^2 = +j (BACKWARD), Compute Only u6, u7
    //==========================================================================

    __m512d t6_tw_re, t6_tw_im;
    rot_pos_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_pos_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u6_re = _mm512_sub_pd(t4_re, t6_tw_re);
    __m512d u6_im = _mm512_sub_pd(t4_im, t6_tw_im);

    __m512d u7_re = _mm512_sub_pd(t5_re, t7_tw_re);
    __m512d u7_im = _mm512_sub_pd(t5_im, t7_tw_im);

    //==========================================================================
    // STAGE 3: Apply W8^3 = √2/2(-1+i) - CORRECTED
    //==========================================================================
    // Formula: √2/2[-(a+b) + j(a-b)]
    // Real: -(a+b), Imag: (a-b)

    __m512d sum7 = _mm512_add_pd(u7_re, u7_im);    // a + b
    __m512d nb7 = _mm512_xor_pd(u7_im, sign_mask); // -b
    __m512d diff7 = _mm512_add_pd(u7_re, nb7);     // a - b (XOR+ADD style)

    __m512d u7_tw_re_tmp = _mm512_mul_pd(sqrt2_2, sum7);       // √2/2 * (a+b)
    __m512d u7_tw_re = _mm512_xor_pd(u7_tw_re_tmp, sign_mask); // -√2/2 * (a+b)
    __m512d u7_tw_im = _mm512_mul_pd(sqrt2_2, diff7);          // √2/2 * (a-b)

    // Final butterfly
    *y3_re = _mm512_add_pd(u6_re, u7_tw_re);
    *y3_im = _mm512_add_pd(u6_im, u7_tw_im);
    *y7_re = _mm512_sub_pd(u6_re, u7_tw_re);
    *y7_im = _mm512_sub_pd(u6_im, u7_tw_im);
}

/**
 * @brief Emit pair (1,5) from t4..t7 - Backward (CORRECTED)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_15_from_t4567_backward_avx512(
    __m512d t4_re, __m512d t4_im,
    __m512d t5_re, __m512d t5_im,
    __m512d t6_re, __m512d t6_im,
    __m512d t7_re, __m512d t7_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y5_re, __m512d *RESTRICT y5_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    __m512d t6_tw_re, t6_tw_im;
    rot_pos_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_pos_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u4_re = _mm512_add_pd(t4_re, t6_tw_re);
    __m512d u4_im = _mm512_add_pd(t4_im, t6_tw_im);

    __m512d u5_re = _mm512_add_pd(t5_re, t7_tw_re);
    __m512d u5_im = _mm512_add_pd(t5_im, t7_tw_im);

    // Backward W8^1: √2/2[(a-b) + j(a+b)] - CORRECTED
    __m512d nb5 = _mm512_xor_pd(u5_im, sign_mask);
    __m512d diff5 = _mm512_add_pd(u5_re, nb5);  // a - b
    __m512d sum5 = _mm512_add_pd(u5_re, u5_im); // a + b

    __m512d u5_tw_re = _mm512_mul_pd(sqrt2_2, diff5);
    __m512d u5_tw_im = _mm512_mul_pd(sqrt2_2, sum5);

    *y1_re = _mm512_add_pd(u4_re, u5_tw_re);
    *y1_im = _mm512_add_pd(u4_im, u5_tw_im);
    *y5_re = _mm512_sub_pd(u4_re, u5_tw_re);
    *y5_im = _mm512_sub_pd(u4_im, u5_tw_im);
}

/**
 * @brief Emit pair (3,7) from t4..t7 - Backward (CORRECTED)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_37_from_t4567_backward_avx512(
    __m512d t4_re, __m512d t4_im,
    __m512d t5_re, __m512d t5_im,
    __m512d t6_re, __m512d t6_im,
    __m512d t7_re, __m512d t7_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d *RESTRICT y7_re, __m512d *RESTRICT y7_im,
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    __m512d t6_tw_re, t6_tw_im;
    rot_pos_j(t6_re, t6_im, sign_mask, &t6_tw_re, &t6_tw_im);

    __m512d t7_tw_re, t7_tw_im;
    rot_pos_j(t7_re, t7_im, sign_mask, &t7_tw_re, &t7_tw_im);

    __m512d u6_re = _mm512_sub_pd(t4_re, t6_tw_re);
    __m512d u6_im = _mm512_sub_pd(t4_im, t6_tw_im);

    __m512d u7_re = _mm512_sub_pd(t5_re, t7_tw_re);
    __m512d u7_im = _mm512_sub_pd(t5_im, t7_tw_im);

    // Backward W8^3: √2/2[-(a+b) + j(a-b)] - CORRECTED
    __m512d sum7 = _mm512_add_pd(u7_re, u7_im);
    __m512d nb7 = _mm512_xor_pd(u7_im, sign_mask);
    __m512d diff7 = _mm512_add_pd(u7_re, nb7); // a - b

    __m512d u7_tw_re_tmp = _mm512_mul_pd(sqrt2_2, sum7);
    __m512d u7_tw_re = _mm512_xor_pd(u7_tw_re_tmp, sign_mask); // -(a+b)
    __m512d u7_tw_im = _mm512_mul_pd(sqrt2_2, diff7);          // (a-b)

    *y3_re = _mm512_add_pd(u6_re, u7_tw_re);
    *y3_im = _mm512_add_pd(u6_im, u7_tw_im);
    *y7_re = _mm512_sub_pd(u6_re, u7_tw_re);
    *y7_im = _mm512_sub_pd(u6_im, u7_tw_im);
}

//==============================================================================
// OPTIMIZED MINI-RECURRENCE (SKYLAKE-X PORT-5 OPTIMIZATION)
//==============================================================================

/**
 * @brief Compute w^1, w^2, w^3 from seed twiddle w - OPTIMIZED for Skylake-X
 *
 * SKYLAKE-X OPTIMIZATION:
 * - Replace FMA for w2_im with MUL+ADD to ease port-5 pressure
 * - FMA ports (0,1,5) are heavily contended in complex multiply chains
 * - Spreading to ADD ports (0,1,5) + MUL ports (0,1,5) improves throughput
 *
 * FORMULA:
 * - w^1 = w (trivial copy)
 * - w^2 = w * w (complex square)
 *   - Re(w^2) = w_re^2 - w_im^2
 *   - Im(w^2) = 2 * w_re * w_im  ← OPTIMIZED: MUL+ADD instead of FMA
 * - w^3 = w^2 * w (complex multiply)
 *
 * COST:
 * - Original: 4 FMA + 2 MUL = 6 FMA-port ops
 * - Optimized: 3 FMA + 3 MUL + 1 ADD = better port distribution
 *
 * @param w_re Seed twiddle real
 * @param w_im Seed twiddle imag
 * @param w1_re Output: w^1 real
 * @param w1_im Output: w^1 imag
 * @param w2_re Output: w^2 real
 * @param w2_im Output: w^2 imag
 * @param w3_re Output: w^3 real
 * @param w3_im Output: w^3 imag
 */
TARGET_AVX512
FORCE_INLINE void cross_twiddle_powers3_optimized(
    __m512d w_re, __m512d w_im,
    __m512d *RESTRICT w1_re, __m512d *RESTRICT w1_im,
    __m512d *RESTRICT w2_re, __m512d *RESTRICT w2_im,
    __m512d *RESTRICT w3_re, __m512d *RESTRICT w3_im)
{
    //==========================================================================
    // w^1 = w (trivial copy)
    //==========================================================================
    *w1_re = w_re;
    *w1_im = w_im;

    //==========================================================================
    // w^2 = w * w (complex square with PORT-5 OPTIMIZATION)
    //==========================================================================
    // Re(w^2) = w_re^2 - w_im^2 (uses FMA)
    __m512d w2r = _mm512_fmsub_pd(w_re, w_re, _mm512_mul_pd(w_im, w_im));

    // Im(w^2) = 2 * w_re * w_im (MUL+ADD instead of FMA to ease port-5)
    __m512d re_im = _mm512_mul_pd(w_re, w_im); // w_re * w_im (MUL: ports 0,1,5)
    __m512d w2i = _mm512_add_pd(re_im, re_im); // 2*re*im (ADD: ports 0,1,5)

    *w2_re = w2r;
    *w2_im = w2i;

    //==========================================================================
    // w^3 = w^2 * w (complex multiply via FMA)
    //==========================================================================
    // Re(w^3) = w2_re * w_re - w2_im * w_im
    *w3_re = _mm512_fmsub_pd(w2r, w_re, _mm512_mul_pd(w2i, w_im));

    // Im(w^3) = w2_re * w_im + w2_im * w_re
    *w3_im = _mm512_fmadd_pd(w2r, w_im, _mm512_mul_pd(w2i, w_re));
}

//==============================================================================
// POSITION-4 FAST-PATH: W8 FAMILY CROSS-GROUP TWIDDLES
//==============================================================================

/**
 * @brief Position-4 cross-group twiddles using W8 fast-paths
 *
 * POSITION 4 TWIDDLES (W32^4k for k=0,1,2,3):
 * - Group A (k=0): W32^0  = 1         (identity)
 * - Group B (k=1): W32^4  = W8^1  = √2/2(1-i)  (sum/diff)
 * - Group C (k=2): W32^8  = W8^2  = -i         (rotation)
 * - Group D (k=3): W32^12 = W8^3  = √2/2(-1-i) (sum/diff + negate)
 *
 * BENEFIT: Eliminates 3 generic complex multiplies (12 FMA + 3 MUL)
 * Replaces with: 2 sum/diff (8 ADD + 4 MUL + 2 XOR) + 1 rotation (2 MOV + 1 XOR)
 *
 * FORWARD vs BACKWARD:
 * - Forward uses: -i, √2/2(1-i), √2/2(-1-i)
 * - Backward uses: +i, √2/2(1+i), √2/2(-1+i) (conjugates)
 */

/**
 * @brief Radix-4 butterfly core (forward variant)
 *
 * Standard DIT radix-4:
 * y0 = a + b + c + d
 * y1 = a - jb - c + jd
 * y2 = a - b + c - d
 * y3 = a + jb - c - jd
 */
TARGET_AVX512
FORCE_INLINE void radix4_butterfly_core_fv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im,
    __m512d d_re, __m512d d_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d sign_mask)
{
    // Stage 1: Compute sums and differences
    __m512d t0_re = _mm512_add_pd(a_re, c_re); // a + c
    __m512d t0_im = _mm512_add_pd(a_im, c_im);
    __m512d t1_re = _mm512_sub_pd(a_re, c_re); // a - c
    __m512d t1_im = _mm512_sub_pd(a_im, c_im);

    __m512d t2_re = _mm512_add_pd(b_re, d_re); // b + d
    __m512d t2_im = _mm512_add_pd(b_im, d_im);
    __m512d t3_re = _mm512_sub_pd(b_re, d_re); // b - d
    __m512d t3_im = _mm512_sub_pd(b_im, d_im);

    // Stage 2: Apply ±j rotations to t3 (forward: -j)
    __m512d t3_rot_re, t3_rot_im;
    rot_neg_j(t3_re, t3_im, sign_mask, &t3_rot_re, &t3_rot_im); // (b-d)*(-j)

    // Stage 3: Final combinations
    *y0_re = _mm512_add_pd(t0_re, t2_re); // (a+c) + (b+d)
    *y0_im = _mm512_add_pd(t0_im, t2_im);

    *y1_re = _mm512_add_pd(t1_re, t3_rot_re); // (a-c) + (b-d)*(-j)
    *y1_im = _mm512_add_pd(t1_im, t3_rot_im);

    *y2_re = _mm512_sub_pd(t0_re, t2_re); // (a+c) - (b+d)
    *y2_im = _mm512_sub_pd(t0_im, t2_im);

    *y3_re = _mm512_sub_pd(t1_re, t3_rot_re); // (a-c) - (b-d)*(-j)
    *y3_im = _mm512_sub_pd(t1_im, t3_rot_im);
}

/**
 * @brief Radix-4 butterfly core (backward variant)
 *
 * Backward uses +j instead of -j
 */
TARGET_AVX512
FORCE_INLINE void radix4_butterfly_core_bv_avx512(
    __m512d a_re, __m512d a_im,
    __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im,
    __m512d d_re, __m512d d_im,
    __m512d *RESTRICT y0_re, __m512d *RESTRICT y0_im,
    __m512d *RESTRICT y1_re, __m512d *RESTRICT y1_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y3_re, __m512d *RESTRICT y3_im,
    __m512d sign_mask)
{
    __m512d t0_re = _mm512_add_pd(a_re, c_re);
    __m512d t0_im = _mm512_add_pd(a_im, c_im);
    __m512d t1_re = _mm512_sub_pd(a_re, c_re);
    __m512d t1_im = _mm512_sub_pd(a_im, c_im);

    __m512d t2_re = _mm512_add_pd(b_re, d_re);
    __m512d t2_im = _mm512_add_pd(b_im, d_im);
    __m512d t3_re = _mm512_sub_pd(b_re, d_re);
    __m512d t3_im = _mm512_sub_pd(b_im, d_im);

    // Backward: use +j rotation
    __m512d t3_rot_re, t3_rot_im;
    rot_pos_j(t3_re, t3_im, sign_mask, &t3_rot_re, &t3_rot_im);

    *y0_re = _mm512_add_pd(t0_re, t2_re);
    *y0_im = _mm512_add_pd(t0_im, t2_im);

    *y1_re = _mm512_add_pd(t1_re, t3_rot_re);
    *y1_im = _mm512_add_pd(t1_im, t3_rot_im);

    *y2_re = _mm512_sub_pd(t0_re, t2_re);
    *y2_im = _mm512_sub_pd(t0_im, t2_im);

    *y3_re = _mm512_sub_pd(t1_re, t3_rot_re);
    *y3_im = _mm512_sub_pd(t1_im, t3_rot_im);
}

//==============================================================================
// POSITION-4 FAST-PATH MACROS
//==============================================================================

/**
 * @brief Process position-4 using W8 fast-paths - FORWARD, UNMASKED
 *
 * Applies:
 * - B *= W8^1 = √2/2(1-i) using sum/diff
 * - C *= W8^2 = -i using rotation
 * - D *= W8^3 = √2/2(-1-i) using sum/diff + negate
 */
#define RADIX32_POSITION4_FAST_FORWARD_UNMASKED(STRIPE)                                       \
    do                                                                                        \
    {                                                                                         \
        __m512d a_re = A_re[1], a_im = A_im[1]; /* Position 4 is at index 1 (bit-reversed) */ \
        __m512d b_re = B_re[1], b_im = B_im[1];                                               \
        __m512d c_re = C_re[1], c_im = C_im[1];                                               \
        __m512d d_re = D_re[1], d_im = D_im[1];                                               \
                                                                                              \
        /* B *= W8^1 = √2/2(1-i) using sum/diff */                                            \
        __m512d sum_b = _mm512_add_pd(b_re, b_im);                                            \
        __m512d nb = _mm512_xor_pd(b_re, sign_mask);                                          \
        __m512d diff_b = _mm512_add_pd(b_im, nb); /* im - re */                               \
        b_re = _mm512_mul_pd(sqrt2_2, sum_b);                                                 \
        b_im = _mm512_mul_pd(sqrt2_2, diff_b);                                                \
                                                                                              \
        /* C *= W8^2 = -i using rotation */                                                   \
        __m512d c_tmp_re, c_tmp_im;                                                           \
        rot_neg_j(c_re, c_im, sign_mask, &c_tmp_re, &c_tmp_im);                               \
        c_re = c_tmp_re;                                                                      \
        c_im = c_tmp_im;                                                                      \
                                                                                              \
        /* D *= W8^3 = √2/2(-1-i) using sum/diff + negate */                                  \
        __m512d sum_d = _mm512_add_pd(d_re, d_im);                                            \
        __m512d nd = _mm512_xor_pd(d_re, sign_mask);                                          \
        __m512d diff_d = _mm512_add_pd(d_im, nd); /* im - re */                               \
        __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);                                     \
        d_re = _mm512_xor_pd(d_re_tmp, sign_mask); /* Negate */                               \
        d_im = _mm512_mul_pd(sqrt2_2, diff_d);                                                \
                                                                                              \
        /* Radix-4 combine */                                                                 \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;                       \
        radix4_butterfly_core_fv_avx512(                                                      \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                                   \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,                   \
            sign_mask);                                                                       \
                                                                                              \
        /* Store outputs (stripes: 4, 12, 20, 28) */                                          \
        store_aligned(&tile_out_re[(STRIPE + 0) * tile_size + k], y0_re);                     \
        store_aligned(&tile_out_im[(STRIPE + 0) * tile_size + k], y0_im);                     \
        store_aligned(&tile_out_re[(STRIPE + 8) * tile_size + k], y1_re);                     \
        store_aligned(&tile_out_im[(STRIPE + 8) * tile_size + k], y1_im);                     \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);                    \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);                    \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);                    \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);                    \
    } while (0)

/**
 * @brief Process position-4 using W8 fast-paths - FORWARD, MASKED
 */
#define RADIX32_POSITION4_FAST_FORWARD_MASKED(STRIPE, MASK)                     \
    do                                                                          \
    {                                                                           \
        __m512d a_re = A_re[1], a_im = A_im[1];                                 \
        __m512d b_re = B_re[1], b_im = B_im[1];                                 \
        __m512d c_re = C_re[1], c_im = C_im[1];                                 \
        __m512d d_re = D_re[1], d_im = D_im[1];                                 \
                                                                                \
        __m512d sum_b = _mm512_add_pd(b_re, b_im);                              \
        __m512d nb = _mm512_xor_pd(b_re, sign_mask);                            \
        __m512d diff_b = _mm512_add_pd(b_im, nb);                               \
        b_re = _mm512_mul_pd(sqrt2_2, sum_b);                                   \
        b_im = _mm512_mul_pd(sqrt2_2, diff_b);                                  \
                                                                                \
        __m512d c_tmp_re, c_tmp_im;                                             \
        rot_neg_j(c_re, c_im, sign_mask, &c_tmp_re, &c_tmp_im);                 \
        c_re = c_tmp_re;                                                        \
        c_im = c_tmp_im;                                                        \
                                                                                \
        __m512d sum_d = _mm512_add_pd(d_re, d_im);                              \
        __m512d nd = _mm512_xor_pd(d_re, sign_mask);                            \
        __m512d diff_d = _mm512_add_pd(d_im, nd);                               \
        __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);                       \
        d_re = _mm512_xor_pd(d_re_tmp, sign_mask);                              \
        d_im = _mm512_mul_pd(sqrt2_2, diff_d);                                  \
                                                                                \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;         \
        radix4_butterfly_core_fv_avx512(                                        \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                     \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,     \
            sign_mask);                                                         \
                                                                                \
        store_masked(&tile_out_re[(STRIPE + 0) * tile_size + k], MASK, y0_re);  \
        store_masked(&tile_out_im[(STRIPE + 0) * tile_size + k], MASK, y0_im);  \
        store_masked(&tile_out_re[(STRIPE + 8) * tile_size + k], MASK, y1_re);  \
        store_masked(&tile_out_im[(STRIPE + 8) * tile_size + k], MASK, y1_im);  \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re); \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im); \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re); \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im); \
    } while (0)

/**
 * @brief Process position-4 using W8 fast-paths - BACKWARD, UNMASKED
 *
 * Applies:
 * - B *= W8^1 = √2/2(1+i) using sum/diff (conjugated)
 * - C *= W8^2 = +i using rotation (conjugated)
 * - D *= W8^3 = √2/2(-1+i) using sum/diff + negate (conjugated)
 */
#define RADIX32_POSITION4_FAST_BACKWARD_UNMASKED(STRIPE)                    \
    do                                                                      \
    {                                                                       \
        __m512d a_re = A_re[1], a_im = A_im[1];                             \
        __m512d b_re = B_re[1], b_im = B_im[1];                             \
        __m512d c_re = C_re[1], c_im = C_im[1];                             \
        __m512d d_re = D_re[1], d_im = D_im[1];                             \
                                                                            \
        /* B *= W8^1 = √2/2(1+i): [(a-b) + j(a+b)] */                       \
        __m512d sum_b = _mm512_add_pd(b_re, b_im);                          \
        __m512d nb = _mm512_xor_pd(b_im, sign_mask);                        \
        __m512d diff_b = _mm512_add_pd(b_re, nb); /* re - im */             \
        b_re = _mm512_mul_pd(sqrt2_2, diff_b);                              \
        b_im = _mm512_mul_pd(sqrt2_2, sum_b);                               \
                                                                            \
        /* C *= W8^2 = +i using rotation */                                 \
        __m512d c_tmp_re, c_tmp_im;                                         \
        rot_pos_j(c_re, c_im, sign_mask, &c_tmp_re, &c_tmp_im);             \
        c_re = c_tmp_re;                                                    \
        c_im = c_tmp_im;                                                    \
                                                                            \
        /* D *= W8^3 = √2/2(-1+i): [-(a+b) + j(a-b)] */                     \
        __m512d sum_d = _mm512_add_pd(d_re, d_im);                          \
        __m512d nd = _mm512_xor_pd(d_im, sign_mask);                        \
        __m512d diff_d = _mm512_add_pd(d_re, nd); /* re - im */             \
        __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);                   \
        d_re = _mm512_xor_pd(d_re_tmp, sign_mask); /* Negate */             \
        d_im = _mm512_mul_pd(sqrt2_2, diff_d);                              \
                                                                            \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;     \
        radix4_butterfly_core_bv_avx512(                                    \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                 \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im, \
            sign_mask);                                                     \
                                                                            \
        store_aligned(&tile_out_re[(STRIPE + 0) * tile_size + k], y0_re);   \
        store_aligned(&tile_out_im[(STRIPE + 0) * tile_size + k], y0_im);   \
        store_aligned(&tile_out_re[(STRIPE + 8) * tile_size + k], y1_re);   \
        store_aligned(&tile_out_im[(STRIPE + 8) * tile_size + k], y1_im);   \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);  \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);  \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);  \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);  \
    } while (0)

/**
 * @brief Process position-4 using W8 fast-paths - BACKWARD, MASKED
 */
#define RADIX32_POSITION4_FAST_BACKWARD_MASKED(STRIPE, MASK)                    \
    do                                                                          \
    {                                                                           \
        __m512d a_re = A_re[1], a_im = A_im[1];                                 \
        __m512d b_re = B_re[1], b_im = B_im[1];                                 \
        __m512d c_re = C_re[1], c_im = C_im[1];                                 \
        __m512d d_re = D_re[1], d_im = D_im[1];                                 \
                                                                                \
        __m512d sum_b = _mm512_add_pd(b_re, b_im);                              \
        __m512d nb = _mm512_xor_pd(b_im, sign_mask);                            \
        __m512d diff_b = _mm512_add_pd(b_re, nb);                               \
        b_re = _mm512_mul_pd(sqrt2_2, diff_b);                                  \
        b_im = _mm512_mul_pd(sqrt2_2, sum_b);                                   \
                                                                                \
        __m512d c_tmp_re, c_tmp_im;                                             \
        rot_pos_j(c_re, c_im, sign_mask, &c_tmp_re, &c_tmp_im);                 \
        c_re = c_tmp_re;                                                        \
        c_im = c_tmp_im;                                                        \
                                                                                \
        __m512d sum_d = _mm512_add_pd(d_re, d_im);                              \
        __m512d nd = _mm512_xor_pd(d_im, sign_mask);                            \
        __m512d diff_d = _mm512_add_pd(d_re, nd);                               \
        __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);                       \
        d_re = _mm512_xor_pd(d_re_tmp, sign_mask);                              \
        d_im = _mm512_mul_pd(sqrt2_2, diff_d);                                  \
                                                                                \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;         \
        radix4_butterfly_core_bv_avx512(                                        \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                     \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,     \
            sign_mask);                                                         \
                                                                                \
        store_masked(&tile_out_re[(STRIPE + 0) * tile_size + k], MASK, y0_re);  \
        store_masked(&tile_out_im[(STRIPE + 0) * tile_size + k], MASK, y0_im);  \
        store_masked(&tile_out_re[(STRIPE + 8) * tile_size + k], MASK, y1_re);  \
        store_masked(&tile_out_im[(STRIPE + 8) * tile_size + k], MASK, y1_im);  \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re); \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im); \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re); \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im); \
    } while (0)

//==============================================================================
// MACROS FOR RADIX-32 FUSED BUTTERFLY (CORRECTED)
//==============================================================================

/**
 * @brief Process one radix-8 group - UNMASKED (main loop)
 */
#define RADIX32_PROCESS_GROUP_UNMASKED(GROUP, STRIPE_BASE, OUT_ARRAY, RADIX8_FUNC)               \
    do                                                                                           \
    {                                                                                            \
        __m512d g##GROUP##_x0_re = load_aligned(&tile_in_re[(STRIPE_BASE + 0) * tile_size + k]); \
        __m512d g##GROUP##_x0_im = load_aligned(&tile_in_im[(STRIPE_BASE + 0) * tile_size + k]); \
        __m512d g##GROUP##_x1_re = load_aligned(&tile_in_re[(STRIPE_BASE + 1) * tile_size + k]); \
        __m512d g##GROUP##_x1_im = load_aligned(&tile_in_im[(STRIPE_BASE + 1) * tile_size + k]); \
        __m512d g##GROUP##_x2_re = load_aligned(&tile_in_re[(STRIPE_BASE + 2) * tile_size + k]); \
        __m512d g##GROUP##_x2_im = load_aligned(&tile_in_im[(STRIPE_BASE + 2) * tile_size + k]); \
        __m512d g##GROUP##_x3_re = load_aligned(&tile_in_re[(STRIPE_BASE + 3) * tile_size + k]); \
        __m512d g##GROUP##_x3_im = load_aligned(&tile_in_im[(STRIPE_BASE + 3) * tile_size + k]); \
        __m512d g##GROUP##_x4_re = load_aligned(&tile_in_re[(STRIPE_BASE + 4) * tile_size + k]); \
        __m512d g##GROUP##_x4_im = load_aligned(&tile_in_im[(STRIPE_BASE + 4) * tile_size + k]); \
        __m512d g##GROUP##_x5_re = load_aligned(&tile_in_re[(STRIPE_BASE + 5) * tile_size + k]); \
        __m512d g##GROUP##_x5_im = load_aligned(&tile_in_im[(STRIPE_BASE + 5) * tile_size + k]); \
        __m512d g##GROUP##_x6_re = load_aligned(&tile_in_re[(STRIPE_BASE + 6) * tile_size + k]); \
        __m512d g##GROUP##_x6_im = load_aligned(&tile_in_im[(STRIPE_BASE + 6) * tile_size + k]); \
        __m512d g##GROUP##_x7_re = load_aligned(&tile_in_re[(STRIPE_BASE + 7) * tile_size + k]); \
        __m512d g##GROUP##_x7_im = load_aligned(&tile_in_im[(STRIPE_BASE + 7) * tile_size + k]); \
                                                                                                 \
        RADIX8_FUNC(                                                                             \
            g##GROUP##_x0_re, g##GROUP##_x0_im, g##GROUP##_x1_re, g##GROUP##_x1_im,              \
            g##GROUP##_x2_re, g##GROUP##_x2_im, g##GROUP##_x3_re, g##GROUP##_x3_im,              \
            g##GROUP##_x4_re, g##GROUP##_x4_im, g##GROUP##_x5_re, g##GROUP##_x5_im,              \
            g##GROUP##_x6_re, g##GROUP##_x6_im, g##GROUP##_x7_re, g##GROUP##_x7_im,              \
            &OUT_ARRAY##_re[0], &OUT_ARRAY##_im[0], &OUT_ARRAY##_re[1], &OUT_ARRAY##_im[1],      \
            &OUT_ARRAY##_re[2], &OUT_ARRAY##_im[2], &OUT_ARRAY##_re[3], &OUT_ARRAY##_im[3],      \
            &OUT_ARRAY##_re[4], &OUT_ARRAY##_im[4], &OUT_ARRAY##_re[5], &OUT_ARRAY##_im[5],      \
            &OUT_ARRAY##_re[6], &OUT_ARRAY##_im[6], &OUT_ARRAY##_re[7], &OUT_ARRAY##_im[7],      \
            sign_mask, sqrt2_2);                                                                 \
    } while (0)

/**
 * @brief Process one radix-8 group - MASKED (tail handling)
 */
#define RADIX32_PROCESS_GROUP_MASKED(GROUP, STRIPE_BASE, OUT_ARRAY, RADIX8_FUNC, MASK)                \
    do                                                                                                \
    {                                                                                                 \
        __m512d g##GROUP##_x0_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 0) * tile_size + k]); \
        __m512d g##GROUP##_x0_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 0) * tile_size + k]); \
        __m512d g##GROUP##_x1_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 1) * tile_size + k]); \
        __m512d g##GROUP##_x1_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 1) * tile_size + k]); \
        __m512d g##GROUP##_x2_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 2) * tile_size + k]); \
        __m512d g##GROUP##_x2_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 2) * tile_size + k]); \
        __m512d g##GROUP##_x3_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 3) * tile_size + k]); \
        __m512d g##GROUP##_x3_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 3) * tile_size + k]); \
        __m512d g##GROUP##_x4_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 4) * tile_size + k]); \
        __m512d g##GROUP##_x4_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 4) * tile_size + k]); \
        __m512d g##GROUP##_x5_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 5) * tile_size + k]); \
        __m512d g##GROUP##_x5_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 5) * tile_size + k]); \
        __m512d g##GROUP##_x6_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 6) * tile_size + k]); \
        __m512d g##GROUP##_x6_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 6) * tile_size + k]); \
        __m512d g##GROUP##_x7_re = load_masked(MASK, &tile_in_re[(STRIPE_BASE + 7) * tile_size + k]); \
        __m512d g##GROUP##_x7_im = load_masked(MASK, &tile_in_im[(STRIPE_BASE + 7) * tile_size + k]); \
                                                                                                      \
        RADIX8_FUNC(                                                                                  \
            g##GROUP##_x0_re, g##GROUP##_x0_im, g##GROUP##_x1_re, g##GROUP##_x1_im,                   \
            g##GROUP##_x2_re, g##GROUP##_x2_im, g##GROUP##_x3_re, g##GROUP##_x3_im,                   \
            g##GROUP##_x4_re, g##GROUP##_x4_im, g##GROUP##_x5_re, g##GROUP##_x5_im,                   \
            g##GROUP##_x6_re, g##GROUP##_x6_im, g##GROUP##_x7_re, g##GROUP##_x7_im,                   \
            &OUT_ARRAY##_re[0], &OUT_ARRAY##_im[0], &OUT_ARRAY##_re[1], &OUT_ARRAY##_im[1],           \
            &OUT_ARRAY##_re[2], &OUT_ARRAY##_im[2], &OUT_ARRAY##_re[3], &OUT_ARRAY##_im[3],           \
            &OUT_ARRAY##_re[4], &OUT_ARRAY##_im[4], &OUT_ARRAY##_re[5], &OUT_ARRAY##_im[5],           \
            &OUT_ARRAY##_re[6], &OUT_ARRAY##_im[6], &OUT_ARRAY##_re[7], &OUT_ARRAY##_im[7],           \
            sign_mask, sqrt2_2);                                                                      \
    } while (0)

/**
 * @brief Process one radix-4 position - UNMASKED, IDENTITY (no twiddles)
 */
#define RADIX32_POSITION_IDENTITY_UNMASKED(POS, STRIPE)                     \
    do                                                                      \
    {                                                                       \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                         \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                         \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                         \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                         \
                                                                            \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;     \
                                                                            \
        radix4_butterfly_core_fv_avx512(                                    \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                 \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im, \
            sign_mask);                                                     \
                                                                            \
        store_aligned(&tile_out_re[(STRIPE + 0) * tile_size + k], y0_re);   \
        store_aligned(&tile_out_im[(STRIPE + 0) * tile_size + k], y0_im);   \
        store_aligned(&tile_out_re[(STRIPE + 8) * tile_size + k], y1_re);   \
        store_aligned(&tile_out_im[(STRIPE + 8) * tile_size + k], y1_im);   \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);  \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);  \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);  \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);  \
    } while (0)

/**
 * @brief Process one radix-4 position - MASKED, IDENTITY
 */
#define RADIX32_POSITION_IDENTITY_MASKED(POS, STRIPE, MASK)                     \
    do                                                                          \
    {                                                                           \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                             \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                             \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                             \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                             \
                                                                                \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;         \
                                                                                \
        radix4_butterfly_core_fv_avx512(                                        \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                     \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,     \
            sign_mask);                                                         \
                                                                                \
        store_masked(&tile_out_re[(STRIPE + 0) * tile_size + k], MASK, y0_re);  \
        store_masked(&tile_out_im[(STRIPE + 0) * tile_size + k], MASK, y0_im);  \
        store_masked(&tile_out_re[(STRIPE + 8) * tile_size + k], MASK, y1_re);  \
        store_masked(&tile_out_im[(STRIPE + 8) * tile_size + k], MASK, y1_im);  \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re); \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im); \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re); \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im); \
    } while (0)

/**
 * @brief Process one radix-4 position - UNMASKED, TWIDDLED
 */
#define RADIX32_POSITION_TWIDDLED_UNMASKED(POS, STRIPE, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM) \
    do                                                                                            \
    {                                                                                             \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                               \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                               \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                               \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                               \
                                                                                                  \
        cmul_avx512(b_re, b_im, W1_RE, W1_IM, &b_re, &b_im);                                      \
        cmul_avx512(c_re, c_im, W2_RE, W2_IM, &c_re, &c_im);                                      \
        cmul_avx512(d_re, d_im, W3_RE, W3_IM, &d_re, &d_im);                                      \
                                                                                                  \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;                           \
                                                                                                  \
        radix4_butterfly_core_fv_avx512(                                                          \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                                       \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,                       \
            sign_mask);                                                                           \
                                                                                                  \
        store_aligned(&tile_out_re[(STRIPE + 0) * tile_size + k], y0_re);                         \
        store_aligned(&tile_out_im[(STRIPE + 0) * tile_size + k], y0_im);                         \
        store_aligned(&tile_out_re[(STRIPE + 8) * tile_size + k], y1_re);                         \
        store_aligned(&tile_out_im[(STRIPE + 8) * tile_size + k], y1_im);                         \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);                        \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);                        \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);                        \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);                        \
    } while (0)

/**
 * @brief Process one radix-4 position - MASKED, TWIDDLED
 */
#define RADIX32_POSITION_TWIDDLED_MASKED(POS, STRIPE, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM, MASK) \
    do                                                                                                \
    {                                                                                                 \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                                   \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                                   \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                                   \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                                   \
                                                                                                      \
        cmul_avx512(b_re, b_im, W1_RE, W1_IM, &b_re, &b_im);                                          \
        cmul_avx512(c_re, c_im, W2_RE, W2_IM, &c_re, &c_im);                                          \
        cmul_avx512(d_re, d_im, W3_RE, W3_IM, &d_re, &d_im);                                          \
                                                                                                      \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;                               \
                                                                                                      \
        radix4_butterfly_core_fv_avx512(                                                              \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                                           \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,                           \
            sign_mask);                                                                               \
                                                                                                      \
        store_masked(&tile_out_re[(STRIPE + 0) * tile_size + k], MASK, y0_re);                        \
        store_masked(&tile_out_im[(STRIPE + 0) * tile_size + k], MASK, y0_im);                        \
        store_masked(&tile_out_re[(STRIPE + 8) * tile_size + k], MASK, y1_re);                        \
        store_masked(&tile_out_im[(STRIPE + 8) * tile_size + k], MASK, y1_im);                        \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re);                       \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im);                       \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re);                       \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im);                       \
    } while (0)

//==============================================================================
// POSITION-4 GENERIC POSITIONS (BACKWARD VARIANTS)
//==============================================================================

#define RADIX32_POSITION_IDENTITY_UNMASKED_BV(POS, STRIPE)                         \
    do {                                                                           \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                 \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                 \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                 \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                 \
                                                                                    \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;             \
        radix4_butterfly_core_bv_avx512(                                            \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                         \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,         \
            sign_mask);                                                             \
                                                                                    \
        store_aligned(&tile_out_re[(STRIPE + 0 ) * tile_size + k], y0_re);          \
        store_aligned(&tile_out_im[(STRIPE + 0 ) * tile_size + k], y0_im);          \
        store_aligned(&tile_out_re[(STRIPE + 8 ) * tile_size + k], y1_re);          \
        store_aligned(&tile_out_im[(STRIPE + 8 ) * tile_size + k], y1_im);          \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);          \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);          \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);          \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);          \
    } while (0)

#define RADIX32_POSITION_IDENTITY_MASKED_BV(POS, STRIPE, MASK)                      \
    do {                                                                            \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                  \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                  \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                  \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                  \
                                                                                     \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;              \
        radix4_butterfly_core_bv_avx512(                                             \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                          \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,          \
            sign_mask);                                                              \
                                                                                     \
        store_masked(&tile_out_re[(STRIPE + 0 ) * tile_size + k], MASK, y0_re);      \
        store_masked(&tile_out_im[(STRIPE + 0 ) * tile_size + k], MASK, y0_im);      \
        store_masked(&tile_out_re[(STRIPE + 8 ) * tile_size + k], MASK, y1_re);      \
        store_masked(&tile_out_im[(STRIPE + 8 ) * tile_size + k], MASK, y1_im);      \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re);      \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im);      \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re);      \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im);      \
    } while (0)

#define RADIX32_POSITION_TWIDDLED_UNMASKED_BV(POS, STRIPE, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM) \
    do {                                                                                              \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                                   \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                                   \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                                   \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                                   \
                                                                                                      \
        cmul_avx512(b_re, b_im, W1_RE, W1_IM, &b_re, &b_im);                                          \
        cmul_avx512(c_re, c_im, W2_RE, W2_IM, &c_re, &c_im);                                          \
        cmul_avx512(d_re, d_im, W3_RE, W3_IM, &d_re, &d_im);                                          \
                                                                                                      \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;                               \
        radix4_butterfly_core_bv_avx512(                                                              \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                                           \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,                           \
            sign_mask);                                                                               \
                                                                                                      \
        store_aligned(&tile_out_re[(STRIPE + 0 ) * tile_size + k], y0_re);                            \
        store_aligned(&tile_out_im[(STRIPE + 0 ) * tile_size + k], y0_im);                            \
        store_aligned(&tile_out_re[(STRIPE + 8 ) * tile_size + k], y1_re);                            \
        store_aligned(&tile_out_im[(STRIPE + 8 ) * tile_size + k], y1_im);                            \
        store_aligned(&tile_out_re[(STRIPE + 16) * tile_size + k], y2_re);                            \
        store_aligned(&tile_out_im[(STRIPE + 16) * tile_size + k], y2_im);                            \
        store_aligned(&tile_out_re[(STRIPE + 24) * tile_size + k], y3_re);                            \
        store_aligned(&tile_out_im[(STRIPE + 24) * tile_size + k], y3_im);                            \
    } while (0)

#define RADIX32_POSITION_TWIDDLED_MASKED_BV(POS, STRIPE, W1_RE, W1_IM, W2_RE, W2_IM, W3_RE, W3_IM, MASK) \
    do {                                                                                                 \
        __m512d a_re = A_re[POS], a_im = A_im[POS];                                                      \
        __m512d b_re = B_re[POS], b_im = B_im[POS];                                                      \
        __m512d c_re = C_re[POS], c_im = C_im[POS];                                                      \
        __m512d d_re = D_re[POS], d_im = D_im[POS];                                                      \
                                                                                                         \
        cmul_avx512(b_re, b_im, W1_RE, W1_IM, &b_re, &b_im);                                             \
        cmul_avx512(c_re, c_im, W2_RE, W2_IM, &c_re, &c_im);                                             \
        cmul_avx512(d_re, d_im, W3_RE, W3_IM, &d_re, &d_im);                                             \
                                                                                                         \
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;                                  \
        radix4_butterfly_core_bv_avx512(                                                                 \
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,                                              \
            &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,                              \
            sign_mask);                                                                                  \
                                                                                                         \
        store_masked(&tile_out_re[(STRIPE + 0 ) * tile_size + k], MASK, y0_re);                          \
        store_masked(&tile_out_im[(STRIPE + 0 ) * tile_size + k], MASK, y0_im);                          \
        store_masked(&tile_out_re[(STRIPE + 8 ) * tile_size + k], MASK, y1_re);                          \
        store_masked(&tile_out_im[(STRIPE + 8 ) * tile_size + k], MASK, y1_im);                          \
        store_masked(&tile_out_re[(STRIPE + 16) * tile_size + k], MASK, y2_re);                          \
        store_masked(&tile_out_im[(STRIPE + 16) * tile_size + k], MASK, y2_im);                          \
        store_masked(&tile_out_re[(STRIPE + 24) * tile_size + k], MASK, y3_re);                          \
        store_masked(&tile_out_im[(STRIPE + 24) * tile_size + k], MASK, y3_im);                          \
    } while (0)


//==============================================================================
// ENHANCED PREFETCH MACROS
//==============================================================================

/**
 * @brief Prefetch next k-iteration inputs (distance = 8)
 * Fetch data that will be needed 8 doubles later
 */
#define PREFETCH_NEXT_K_ITERATION(STRIPE, K_OFFSET) \
    do { \
        _mm_prefetch((const char *)&tile_in_re[(STRIPE) * tile_size + k + (K_OFFSET)], _MM_HINT_T0); \
        _mm_prefetch((const char *)&tile_in_im[(STRIPE) * tile_size + k + (K_OFFSET)], _MM_HINT_T0); \
    } while(0)

/**
 * @brief Prefetch output cache lines that will be written
 * Use T1 hint (L2 cache) to avoid L1 pollution
 */
#define PREFETCH_OUTPUT_STRIPES(STRIPE_BASE, K_OFFSET) \
    do { \
        _mm_prefetch((const char *)&tile_out_re[(STRIPE_BASE + 0) * tile_size + k + (K_OFFSET)], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tile_out_re[(STRIPE_BASE + 8) * tile_size + k + (K_OFFSET)], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tile_out_re[(STRIPE_BASE + 16) * tile_size + k + (K_OFFSET)], _MM_HINT_T1); \
        _mm_prefetch((const char *)&tile_out_re[(STRIPE_BASE + 24) * tile_size + k + (K_OFFSET)], _MM_HINT_T1); \
    } while(0)

/**
 * @brief Non-temporal prefetch for streaming access patterns
 * Use NTA hint when data won't be reused
 */
#define PREFETCH_STREAMING_INPUT(STRIPE) \
    do { \
        _mm_prefetch((const char *)&tile_in_re[(STRIPE) * tile_size + k + 64], _MM_HINT_NTA); \
        _mm_prefetch((const char *)&tile_in_im[(STRIPE) * tile_size + k + 64], _MM_HINT_NTA); \
    } while(0)

//==============================================================================
// COMPLETE RADIX-32 FUSED BUTTERFLY - FORWARD FFT
//==============================================================================

/**
 * @brief Fully optimized radix-32 8×4 fused butterfly - Forward FFT
 *
 * INTEGRATION OF ALL OPTIMIZATIONS:
 * 1. Pair-emitter architecture (wave-based stage-1 reuse)
 * 2. Position-4 W8 fast-path (eliminates 3 complex muls per transform)
 * 3. Optimized mini-recurrence (MUL+ADD for Skylake-X)
 * 4. Software pipelining (load/compute overlap)
 * 5. Unified masked/unmasked paths
 *
 * ARCHITECTURE:
 * For each k-tile:
 *   For each group (A, B, C, D):
 *     Even wave: compute t0..t3 → emit pairs (0,4) and (2,6)
 *     Odd wave:  compute t4..t7 → emit pairs (1,5) and (3,7)
 *
 *   For each position (0,4,2,6,1,5,3,7):
 *     Apply cross-group twiddles (using fast-path for position 4)
 *     Radix-4 combine
 *     Store outputs
 *
 * @param tile_in_re Input real [32 stripes][tile_size]
 * @param tile_in_im Input imag [32 stripes][tile_size]
 * @param tile_out_re Output real [32 stripes][tile_size]
 * @param tile_out_im Output imag [32 stripes][tile_size]
 * @param tile_size Samples per stripe (must be multiple of 8)
 * @param pass1_plan Pass 1 plan (unused in pair-emitter architecture)
 * @param pass2_plan Pass 2 cross-group twiddle factors
 */
TARGET_AVX512
FORCE_INLINE void radix32_fused_butterfly_forward_avx512(
    const double *RESTRICT tile_in_re,
    const double *RESTRICT tile_in_im,
    double *RESTRICT tile_out_re,
    double *RESTRICT tile_out_im,
    size_t tile_size,
    const radix32_pass1_plan_t *RESTRICT pass1_plan,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    // Hoist constants (single computation per butterfly call)
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

    //==========================================================================
    // PRECOMPUTE CROSS-GROUP TWIDDLES (positions 1, 2, 3, 5, 6, 7)
    // Position 0: Identity (no twiddles)
    // Position 4: W8 fast-path (handled inline)
    //==========================================================================

    // Position 1: W_32^{1,2,3} (broadcast constants)
    const __m512d pos1_w1_re = pass2_plan->pos1_w1_re;
    const __m512d pos1_w1_im = pass2_plan->pos1_w1_im;
    const __m512d pos1_w2_re = pass2_plan->pos1_w2_re;
    const __m512d pos1_w2_im = pass2_plan->pos1_w2_im;
    const __m512d pos1_w3_re = pass2_plan->pos1_w3_re;
    const __m512d pos1_w3_im = pass2_plan->pos1_w3_im;

    // Position 2: W_16^{1,2,3} (broadcast constants)
    const __m512d pos2_w1_re = pass2_plan->pos2_w1_re;
    const __m512d pos2_w1_im = pass2_plan->pos2_w1_im;
    const __m512d pos2_w2_re = pass2_plan->pos2_w2_re;
    const __m512d pos2_w2_im = pass2_plan->pos2_w2_im;
    const __m512d pos2_w3_re = pass2_plan->pos2_w3_re;
    const __m512d pos2_w3_im = pass2_plan->pos2_w3_im;

    // Positions 3, 5, 6, 7: Mini-recurrence (optimized for Skylake-X)
    __m512d pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos3_w_re, pass2_plan->pos3_w_im,
                                    &pos3_w1_re, &pos3_w1_im, &pos3_w2_re, &pos3_w2_im,
                                    &pos3_w3_re, &pos3_w3_im);

    __m512d pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos5_w_re, pass2_plan->pos5_w_im,
                                    &pos5_w1_re, &pos5_w1_im, &pos5_w2_re, &pos5_w2_im,
                                    &pos5_w3_re, &pos5_w3_im);

    __m512d pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos6_w_re, pass2_plan->pos6_w_im,
                                    &pos6_w1_re, &pos6_w1_im, &pos6_w2_re, &pos6_w2_im,
                                    &pos6_w3_re, &pos6_w3_im);

    __m512d pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos7_w_re, pass2_plan->pos7_w_im,
                                    &pos7_w1_re, &pos7_w1_im, &pos7_w2_re, &pos7_w2_im,
                                    &pos7_w3_re, &pos7_w3_im);

    //==========================================================================
    // MAIN K-LOOP: Process full vectors (k += 8)
    //==========================================================================

    size_t k = 0;
    const size_t k_main = (tile_size / 8) * 8;

    for (; k < k_main; k += 8)
    {
        //==========================================================================
        // PREFETCH STRATEGY (before any computation)
        //==========================================================================

        // Prefetch next k-iteration (all groups A, B, C, D)
        if (k + 16 < k_main)
        {
            PREFETCH_NEXT_K_ITERATION(0, 16);  // Group A
            PREFETCH_NEXT_K_ITERATION(8, 16);  // Group B
            PREFETCH_NEXT_K_ITERATION(16, 16); // Group C
            PREFETCH_NEXT_K_ITERATION(24, 16); // Group D
        }

        // Prefetch output buffers for positions 0, 1, 2, 3
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(0, 8); // Position 0
            PREFETCH_OUTPUT_STRIPES(4, 8); // Position 4
            PREFETCH_OUTPUT_STRIPES(2, 8); // Position 2
            PREFETCH_OUTPUT_STRIPES(6, 8); // Position 6
        }

        // Output arrays for 4 groups × 2 pairs = 8 positions
        // Indexed as: [0]=pos0, [1]=pos4, [2]=pos2, [3]=pos6, [4]=pos1, [5]=pos5, [6]=pos3, [7]=pos7
        __m512d A_re[8], A_im[8];
        __m512d B_re[8], B_im[8];
        __m512d C_re[8], C_im[8];
        __m512d D_re[8], D_im[8];

        //======================================================================
        // PHASE 1: RADIX-8 PAIR EMITTERS (TWO-WAVE ARCHITECTURE)
        //======================================================================

        //----------------------------------------------------------------------
        // GROUP A: Load inputs (with software pipelining for next group)
        //----------------------------------------------------------------------
        __m512d xa0_re = load_aligned(&tile_in_re[0 * tile_size + k]);
        __m512d xa0_im = load_aligned(&tile_in_im[0 * tile_size + k]);

        // Prefetch group B stripe 1 while loading A stripe 0
        _mm_prefetch((const char *)&tile_in_re[9 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[9 * tile_size + k], _MM_HINT_T0);

        __m512d xa1_re = load_aligned(&tile_in_re[1 * tile_size + k]);
        __m512d xa1_im = load_aligned(&tile_in_im[1 * tile_size + k]);
        
        // SOFTWARE PIPELINING: Issue first loads for group B while loading A
        __m512d xb0_re = load_aligned(&tile_in_re[8 * tile_size + k]);
        __m512d xb0_im = load_aligned(&tile_in_im[8 * tile_size + k]);

        // Continue loading A with more B prefetches
        _mm_prefetch((const char *)&tile_in_re[10 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[10 * tile_size + k], _MM_HINT_T0);

        __m512d xa2_re = load_aligned(&tile_in_re[2 * tile_size + k]);
        __m512d xa2_im = load_aligned(&tile_in_im[2 * tile_size + k]);
        __m512d xa3_re = load_aligned(&tile_in_re[3 * tile_size + k]);
        __m512d xa3_im = load_aligned(&tile_in_im[3 * tile_size + k]);
        __m512d xa4_re = load_aligned(&tile_in_re[4 * tile_size + k]);
        __m512d xa4_im = load_aligned(&tile_in_im[4 * tile_size + k]);
        __m512d xa5_re = load_aligned(&tile_in_re[5 * tile_size + k]);
        __m512d xa5_im = load_aligned(&tile_in_im[5 * tile_size + k]);
        __m512d xa6_re = load_aligned(&tile_in_re[6 * tile_size + k]);
        __m512d xa6_im = load_aligned(&tile_in_im[6 * tile_size + k]);
        __m512d xa7_re = load_aligned(&tile_in_re[7 * tile_size + k]);
        __m512d xa7_im = load_aligned(&tile_in_im[7 * tile_size + k]);

        // Even wave (pairs 0,4 and 2,6)
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[0], &A_im[0], &A_re[1], &A_im[1]);

            radix8_emit_pair_26_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[2], &A_im[2], &A_re[3], &A_im[3],
                sign_mask);
        }

        // Odd wave (pairs 1,5 and 3,7)
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[4], &A_im[4], &A_re[5], &A_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[6], &A_im[6], &A_re[7], &A_im[7],
                sign_mask, sqrt2_2);
        }

           //----------------------------------------------------------------------
        // GROUP B: Continue loading (already started xb0 during A)
        //----------------------------------------------------------------------
        __m512d xb1_re = load_aligned(&tile_in_re[9 * tile_size + k]);
        __m512d xb1_im = load_aligned(&tile_in_im[9 * tile_size + k]);

        // SOFTWARE PIPELINING: Prefetch group C while loading B
        _mm_prefetch((const char *)&tile_in_re[16 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[16 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_re[17 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[17 * tile_size + k], _MM_HINT_T0);

        __m512d xb2_re = load_aligned(&tile_in_re[10 * tile_size + k]);
        __m512d xb2_im = load_aligned(&tile_in_im[10 * tile_size + k]);
        
        _mm_prefetch((const char *)&tile_in_re[18 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[18 * tile_size + k], _MM_HINT_T0);
        
        __m512d xb3_re = load_aligned(&tile_in_re[11 * tile_size + k]);
        __m512d xb3_im = load_aligned(&tile_in_im[11 * tile_size + k]);
        __m512d xb4_re = load_aligned(&tile_in_re[12 * tile_size + k]);
        __m512d xb4_im = load_aligned(&tile_in_im[12 * tile_size + k]);
        
        _mm_prefetch((const char *)&tile_in_re[19 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[19 * tile_size + k], _MM_HINT_T0);
        
        __m512d xb5_re = load_aligned(&tile_in_re[13 * tile_size + k]);
        __m512d xb5_im = load_aligned(&tile_in_im[13 * tile_size + k]);
        __m512d xb6_re = load_aligned(&tile_in_re[14 * tile_size + k]);
        __m512d xb6_im = load_aligned(&tile_in_im[14 * tile_size + k]);
        __m512d xb7_re = load_aligned(&tile_in_re[15 * tile_size + k]);
        __m512d xb7_im = load_aligned(&tile_in_im[15 * tile_size + k]);

        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[0], &B_im[0], &B_re[1], &B_im[1]);

            radix8_emit_pair_26_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[2], &B_im[2], &B_re[3], &B_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[4], &B_im[4], &B_re[5], &B_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[6], &B_im[6], &B_re[7], &B_im[7],
                sign_mask, sqrt2_2);
        }

         //----------------------------------------------------------------------
        // GROUP C: Load inputs (prefetches already issued during B)
        //----------------------------------------------------------------------
        __m512d xc0_re = load_aligned(&tile_in_re[16 * tile_size + k]);
        __m512d xc0_im = load_aligned(&tile_in_im[16 * tile_size + k]);
        __m512d xc1_re = load_aligned(&tile_in_re[17 * tile_size + k]);
        __m512d xc1_im = load_aligned(&tile_in_im[17 * tile_size + k]);

        // SOFTWARE PIPELINING: Prefetch group D while loading C
        _mm_prefetch((const char *)&tile_in_re[24 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[24 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_re[25 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[25 * tile_size + k], _MM_HINT_T0);

        __m512d xc2_re = load_aligned(&tile_in_re[18 * tile_size + k]);
        __m512d xc2_im = load_aligned(&tile_in_im[18 * tile_size + k]);
        
        _mm_prefetch((const char *)&tile_in_re[26 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[26 * tile_size + k], _MM_HINT_T0);
        
        __m512d xc3_re = load_aligned(&tile_in_re[19 * tile_size + k]);
        __m512d xc3_im = load_aligned(&tile_in_im[19 * tile_size + k]);
        __m512d xc4_re = load_aligned(&tile_in_re[20 * tile_size + k]);
        __m512d xc4_im = load_aligned(&tile_in_im[20 * tile_size + k]);
        
        _mm_prefetch((const char *)&tile_in_re[27 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[27 * tile_size + k], _MM_HINT_T0);
        
        __m512d xc5_re = load_aligned(&tile_in_re[21 * tile_size + k]);
        __m512d xc5_im = load_aligned(&tile_in_im[21 * tile_size + k]);
        __m512d xc6_re = load_aligned(&tile_in_re[22 * tile_size + k]);
        __m512d xc6_im = load_aligned(&tile_in_im[22 * tile_size + k]);
        
        _mm_prefetch((const char *)&tile_in_re[28 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[28 * tile_size + k], _MM_HINT_T0);
        
        __m512d xc7_re = load_aligned(&tile_in_re[23 * tile_size + k]);
        __m512d xc7_im = load_aligned(&tile_in_im[23 * tile_size + k]);

        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[0], &C_im[0], &C_re[1], &C_im[1]);

            radix8_emit_pair_26_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[2], &C_im[2], &C_re[3], &C_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[4], &C_im[4], &C_re[5], &C_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[6], &C_im[6], &C_re[7], &C_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP D: Load inputs (prefetches already issued during C)
        //----------------------------------------------------------------------
        __m512d xd0_re = load_aligned(&tile_in_re[24 * tile_size + k]);
        __m512d xd0_im = load_aligned(&tile_in_im[24 * tile_size + k]);
        __m512d xd1_re = load_aligned(&tile_in_re[25 * tile_size + k]);
        __m512d xd1_im = load_aligned(&tile_in_im[25 * tile_size + k]);
        
        // Prefetch remaining output stripes for positions 4,5,6,7
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(1, 8); // Position 1
            PREFETCH_OUTPUT_STRIPES(5, 8); // Position 5
        }
        
        __m512d xd2_re = load_aligned(&tile_in_re[26 * tile_size + k]);
        __m512d xd2_im = load_aligned(&tile_in_im[26 * tile_size + k]);
        __m512d xd3_re = load_aligned(&tile_in_re[27 * tile_size + k]);
        __m512d xd3_im = load_aligned(&tile_in_im[27 * tile_size + k]);
        
        // Continue output prefetches
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(3, 8); // Position 3
            PREFETCH_OUTPUT_STRIPES(7, 8); // Position 7
        }
        
        __m512d xd4_re = load_aligned(&tile_in_re[28 * tile_size + k]);
        __m512d xd4_im = load_aligned(&tile_in_im[28 * tile_size + k]);
        __m512d xd5_re = load_aligned(&tile_in_re[29 * tile_size + k]);
        __m512d xd5_im = load_aligned(&tile_in_im[29 * tile_size + k]);
        __m512d xd6_re = load_aligned(&tile_in_re[30 * tile_size + k]);
        __m512d xd6_im = load_aligned(&tile_in_im[30 * tile_size + k]);
        __m512d xd7_re = load_aligned(&tile_in_re[31 * tile_size + k]);
        __m512d xd7_im = load_aligned(&tile_in_im[31 * tile_size + k]);

        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[0], &D_im[0], &D_re[1], &D_im[1]);

            radix8_emit_pair_26_from_t0123_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[2], &D_im[2], &D_re[3], &D_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[4], &D_im[4], &D_re[5], &D_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[6], &D_im[6], &D_re[7], &D_im[7],
                sign_mask, sqrt2_2);
        }

        //======================================================================
        // PHASE 2: CROSS-GROUP RADIX-4 COMBINES
        // Process in bit-reversed order: 0, 4, 2, 6, 1, 5, 3, 7
        //======================================================================

        // Position 0 (index 0): Identity (no twiddles)
        RADIX32_POSITION_IDENTITY_UNMASKED(0, 0);

        // Position 4 (index 1): W8 FAST-PATH
        RADIX32_POSITION4_FAST_FORWARD_UNMASKED(4);

        // Position 2 (index 2): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(2, 2, pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im);

        // Position 6 (index 3): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(3, 6, pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im);

        // Position 1 (index 4): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(4, 1, pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im);

        // Position 5 (index 5): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(5, 5, pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im);

        // Position 3 (index 6): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(6, 3, pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im);

        // Position 7 (index 7): Generic twiddles
        RADIX32_POSITION_TWIDDLED_UNMASKED(7, 7, pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im);
    }

    //==========================================================================
    // TAIL HANDLING: Process remaining k < 8 samples with masks
    //==========================================================================

    if (k < tile_size)
    {
        const size_t tail = tile_size - k;
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);

        __m512d A_re[8], A_im[8];
        __m512d B_re[8], B_im[8];
        __m512d C_re[8], C_im[8];
        __m512d D_re[8], D_im[8];

        // Load with masks (zero-extend invalid lanes)
        __m512d xa0_re = load_masked(mask, &tile_in_re[0 * tile_size + k]);
        __m512d xa0_im = load_masked(mask, &tile_in_im[0 * tile_size + k]);
        // ... (load all inputs with masks - similar pattern to main loop)

        // Process groups A, B, C, D (same wave architecture, but with masked loads)
        // ... (similar structure to main loop, using load_masked instead of load_aligned)

        // Cross-group combines with masked stores
        RADIX32_POSITION_IDENTITY_MASKED(0, 0, mask);
        RADIX32_POSITION4_FAST_FORWARD_MASKED(4, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(2, 2, pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(3, 6, pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(4, 1, pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(5, 5, pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(6, 3, pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED(7, 7, pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im, mask);
    }
}

//==============================================================================
// COMPLETE RADIX-32 FUSED BUTTERFLY - BACKWARD/INVERSE FFT
//==============================================================================

/**
 * @brief Fully optimized radix-32 8×4 fused butterfly - Backward/Inverse FFT
 *
 * IDENTICAL ARCHITECTURE TO FORWARD:
 * - Pair-emitter architecture with wave-based stage-1 reuse
 * - Position-4 W8 fast-path (conjugated twiddles)
 * - Optimized mini-recurrence (MUL+ADD for Skylake-X)
 * - Software pipelining (load/compute overlap)
 * - Unified masked/unmasked paths
 *
 * KEY DIFFERENCE: Uses conjugated twiddles throughout
 * - W4^2 = +j (instead of -j)
 * - W8 family conjugated
 * - Cross-group twiddles conjugated via pass2_plan
 *
 * @param tile_in_re Input real [32 stripes][tile_size]
 * @param tile_in_im Input imag [32 stripes][tile_size]
 * @param tile_out_re Output real [32 stripes][tile_size]
 * @param tile_out_im Output imag [32 stripes][tile_size]
 * @param tile_size Samples per stripe (must be multiple of 8)
 * @param pass1_plan Pass 1 plan (unused in pair-emitter architecture)
 * @param pass2_plan Pass 2 cross-group twiddle factors (conjugated for backward)
 */
TARGET_AVX512
FORCE_INLINE void radix32_fused_butterfly_backward_avx512(
    const double *RESTRICT tile_in_re,
    const double *RESTRICT tile_in_im,
    double *RESTRICT tile_out_re,
    double *RESTRICT tile_out_im,
    size_t tile_size,
    const radix32_pass1_plan_t *RESTRICT pass1_plan,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    // Hoist constants
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

    //==========================================================================
    // PRECOMPUTE CROSS-GROUP TWIDDLES (conjugated for backward)
    //==========================================================================

    // Position 1: W_32^{1,2,3} (conjugated)
    const __m512d pos1_w1_re = pass2_plan->pos1_w1_re;
    const __m512d pos1_w1_im = pass2_plan->pos1_w1_im;
    const __m512d pos1_w2_re = pass2_plan->pos1_w2_re;
    const __m512d pos1_w2_im = pass2_plan->pos1_w2_im;
    const __m512d pos1_w3_re = pass2_plan->pos1_w3_re;
    const __m512d pos1_w3_im = pass2_plan->pos1_w3_im;

    // Position 2: W_16^{1,2,3} (conjugated)
    const __m512d pos2_w1_re = pass2_plan->pos2_w1_re;
    const __m512d pos2_w1_im = pass2_plan->pos2_w1_im;
    const __m512d pos2_w2_re = pass2_plan->pos2_w2_re;
    const __m512d pos2_w2_im = pass2_plan->pos2_w2_im;
    const __m512d pos2_w3_re = pass2_plan->pos2_w3_re;
    const __m512d pos2_w3_im = pass2_plan->pos2_w3_im;

    // Positions 3, 5, 6, 7: Mini-recurrence (conjugated seeds)
    __m512d pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos3_w_re, pass2_plan->pos3_w_im,
                                    &pos3_w1_re, &pos3_w1_im, &pos3_w2_re, &pos3_w2_im,
                                    &pos3_w3_re, &pos3_w3_im);

    __m512d pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos5_w_re, pass2_plan->pos5_w_im,
                                    &pos5_w1_re, &pos5_w1_im, &pos5_w2_re, &pos5_w2_im,
                                    &pos5_w3_re, &pos5_w3_im);

    __m512d pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos6_w_re, pass2_plan->pos6_w_im,
                                    &pos6_w1_re, &pos6_w1_im, &pos6_w2_re, &pos6_w2_im,
                                    &pos6_w3_re, &pos6_w3_im);

    __m512d pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im;
    cross_twiddle_powers3_optimized(pass2_plan->pos7_w_re, pass2_plan->pos7_w_im,
                                    &pos7_w1_re, &pos7_w1_im, &pos7_w2_re, &pos7_w2_im,
                                    &pos7_w3_re, &pos7_w3_im);

    //==========================================================================
    // MAIN K-LOOP: Process full vectors (k += 8)
    //==========================================================================

    size_t k = 0;
    const size_t k_main = (tile_size / 8) * 8;

    for (; k < k_main; k += 8)
    {

        //==========================================================================
        // PREFETCH STRATEGY (before any computation)
        //==========================================================================

        // Prefetch next k-iteration (all groups A, B, C, D)
        if (k + 16 < k_main)
        {
            PREFETCH_NEXT_K_ITERATION(0, 16);  // Group A
            PREFETCH_NEXT_K_ITERATION(8, 16);  // Group B
            PREFETCH_NEXT_K_ITERATION(16, 16); // Group C
            PREFETCH_NEXT_K_ITERATION(24, 16); // Group D
        }

        // Prefetch output buffers for positions 0, 1, 2, 3
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(0, 8); // Position 0
            PREFETCH_OUTPUT_STRIPES(4, 8); // Position 4
            PREFETCH_OUTPUT_STRIPES(2, 8); // Position 2
            PREFETCH_OUTPUT_STRIPES(6, 8); // Position 6
        }

        __m512d A_re[8], A_im[8];
        __m512d B_re[8], B_im[8];
        __m512d C_re[8], C_im[8];
        __m512d D_re[8], D_im[8];

        //======================================================================
        // PHASE 1: RADIX-8 PAIR EMITTERS (BACKWARD - TWO-WAVE ARCHITECTURE)
        //======================================================================

         //----------------------------------------------------------------------
        // GROUP A: Load inputs with software pipelining
        //----------------------------------------------------------------------
        __m512d xa0_re = load_aligned(&tile_in_re[0 * tile_size + k]);
        __m512d xa0_im = load_aligned(&tile_in_im[0 * tile_size + k]);

        // Prefetch group B stripe 1 while loading A stripe 0
        _mm_prefetch((const char *)&tile_in_re[9 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[9 * tile_size + k], _MM_HINT_T0);

        __m512d xa1_re = load_aligned(&tile_in_re[1 * tile_size + k]);
        __m512d xa1_im = load_aligned(&tile_in_im[1 * tile_size + k]);

        // SOFTWARE PIPELINING: Start loading group B
        __m512d xb0_re = load_aligned(&tile_in_re[8 * tile_size + k]);
        __m512d xb0_im = load_aligned(&tile_in_im[8 * tile_size + k]);

        // Continue loading A with more B prefetches
        _mm_prefetch((const char *)&tile_in_re[10 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[10 * tile_size + k], _MM_HINT_T0);

        __m512d xa2_re = load_aligned(&tile_in_re[2 * tile_size + k]);
        __m512d xa2_im = load_aligned(&tile_in_im[2 * tile_size + k]);
        __m512d xa3_re = load_aligned(&tile_in_re[3 * tile_size + k]);
        __m512d xa3_im = load_aligned(&tile_in_im[3 * tile_size + k]);
        __m512d xa4_re = load_aligned(&tile_in_re[4 * tile_size + k]);
        __m512d xa4_im = load_aligned(&tile_in_im[4 * tile_size + k]);
        __m512d xa5_re = load_aligned(&tile_in_re[5 * tile_size + k]);
        __m512d xa5_im = load_aligned(&tile_in_im[5 * tile_size + k]);
        __m512d xa6_re = load_aligned(&tile_in_re[6 * tile_size + k]);
        __m512d xa6_im = load_aligned(&tile_in_im[6 * tile_size + k]);
        __m512d xa7_re = load_aligned(&tile_in_re[7 * tile_size + k]);
        __m512d xa7_im = load_aligned(&tile_in_im[7 * tile_size + k]);

        // Even wave (pairs 0,4 and 2,6)
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[0], &A_im[0], &A_re[1], &A_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[2], &A_im[2], &A_re[3], &A_im[3],
                sign_mask);
        }

        // Odd wave (pairs 1,5 and 3,7)
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[4], &A_im[4], &A_re[5], &A_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[6], &A_im[6], &A_re[7], &A_im[7],
                sign_mask, sqrt2_2);
        }

       //----------------------------------------------------------------------
        // GROUP B: Continue loading
        //----------------------------------------------------------------------
        __m512d xb1_re = load_aligned(&tile_in_re[9 * tile_size + k]);
        __m512d xb1_im = load_aligned(&tile_in_im[9 * tile_size + k]);

        // SOFTWARE PIPELINING: Prefetch group C while loading B
        _mm_prefetch((const char *)&tile_in_re[16 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[16 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_re[17 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[17 * tile_size + k], _MM_HINT_T0);

        __m512d xb2_re = load_aligned(&tile_in_re[10 * tile_size + k]);
        __m512d xb2_im = load_aligned(&tile_in_im[10 * tile_size + k]);

        _mm_prefetch((const char *)&tile_in_re[18 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[18 * tile_size + k], _MM_HINT_T0);

        __m512d xb3_re = load_aligned(&tile_in_re[11 * tile_size + k]);
        __m512d xb3_im = load_aligned(&tile_in_im[11 * tile_size + k]);
        __m512d xb4_re = load_aligned(&tile_in_re[12 * tile_size + k]);
        __m512d xb4_im = load_aligned(&tile_in_im[12 * tile_size + k]);

        _mm_prefetch((const char *)&tile_in_re[19 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[19 * tile_size + k], _MM_HINT_T0);

        __m512d xb5_re = load_aligned(&tile_in_re[13 * tile_size + k]);
        __m512d xb5_im = load_aligned(&tile_in_im[13 * tile_size + k]);
        __m512d xb6_re = load_aligned(&tile_in_re[14 * tile_size + k]);
        __m512d xb6_im = load_aligned(&tile_in_im[14 * tile_size + k]);
        __m512d xb7_re = load_aligned(&tile_in_re[15 * tile_size + k]);
        __m512d xb7_im = load_aligned(&tile_in_im[15 * tile_size + k]);

        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[0], &B_im[0], &B_re[1], &B_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[2], &B_im[2], &B_re[3], &B_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[4], &B_im[4], &B_re[5], &B_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[6], &B_im[6], &B_re[7], &B_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP C: Load inputs (prefetches already issued during B)
        //----------------------------------------------------------------------
        __m512d xc0_re = load_aligned(&tile_in_re[16 * tile_size + k]);
        __m512d xc0_im = load_aligned(&tile_in_im[16 * tile_size + k]);
        __m512d xc1_re = load_aligned(&tile_in_re[17 * tile_size + k]);
        __m512d xc1_im = load_aligned(&tile_in_im[17 * tile_size + k]);

        // SOFTWARE PIPELINING: Prefetch group D while loading C
        _mm_prefetch((const char *)&tile_in_re[24 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[24 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_re[25 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[25 * tile_size + k], _MM_HINT_T0);

        __m512d xc2_re = load_aligned(&tile_in_re[18 * tile_size + k]);
        __m512d xc2_im = load_aligned(&tile_in_im[18 * tile_size + k]);

        _mm_prefetch((const char *)&tile_in_re[26 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[26 * tile_size + k], _MM_HINT_T0);

        __m512d xc3_re = load_aligned(&tile_in_re[19 * tile_size + k]);
        __m512d xc3_im = load_aligned(&tile_in_im[19 * tile_size + k]);
        __m512d xc4_re = load_aligned(&tile_in_re[20 * tile_size + k]);
        __m512d xc4_im = load_aligned(&tile_in_im[20 * tile_size + k]);

        _mm_prefetch((const char *)&tile_in_re[27 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[27 * tile_size + k], _MM_HINT_T0);

        __m512d xc5_re = load_aligned(&tile_in_re[21 * tile_size + k]);
        __m512d xc5_im = load_aligned(&tile_in_im[21 * tile_size + k]);
        __m512d xc6_re = load_aligned(&tile_in_re[22 * tile_size + k]);
        __m512d xc6_im = load_aligned(&tile_in_im[22 * tile_size + k]);

        _mm_prefetch((const char *)&tile_in_re[28 * tile_size + k], _MM_HINT_T0);
        _mm_prefetch((const char *)&tile_in_im[28 * tile_size + k], _MM_HINT_T0);

        __m512d xc7_re = load_aligned(&tile_in_re[23 * tile_size + k]);
        __m512d xc7_im = load_aligned(&tile_in_im[23 * tile_size + k]);


        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[0], &C_im[0], &C_re[1], &C_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[2], &C_im[2], &C_re[3], &C_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[4], &C_im[4], &C_re[5], &C_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[6], &C_im[6], &C_re[7], &C_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP D: Load inputs (prefetches already issued during C)
        //----------------------------------------------------------------------
        __m512d xd0_re = load_aligned(&tile_in_re[24 * tile_size + k]);
        __m512d xd0_im = load_aligned(&tile_in_im[24 * tile_size + k]);
        __m512d xd1_re = load_aligned(&tile_in_re[25 * tile_size + k]);
        __m512d xd1_im = load_aligned(&tile_in_im[25 * tile_size + k]);

        // Prefetch remaining output stripes for positions 4,5,6,7
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(1, 8); // Position 1
            PREFETCH_OUTPUT_STRIPES(5, 8); // Position 5
        }

        __m512d xd2_re = load_aligned(&tile_in_re[26 * tile_size + k]);
        __m512d xd2_im = load_aligned(&tile_in_im[26 * tile_size + k]);
        __m512d xd3_re = load_aligned(&tile_in_re[27 * tile_size + k]);
        __m512d xd3_im = load_aligned(&tile_in_im[27 * tile_size + k]);

        // Continue output prefetches
        if (k + 8 < tile_size)
        {
            PREFETCH_OUTPUT_STRIPES(3, 8); // Position 3
            PREFETCH_OUTPUT_STRIPES(7, 8); // Position 7
        }

        // Even wave
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[0], &D_im[0], &D_re[1], &D_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[2], &D_im[2], &D_re[3], &D_im[3],
                sign_mask);
        }

        // Odd wave
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[4], &D_im[4], &D_re[5], &D_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[6], &D_im[6], &D_re[7], &D_im[7],
                sign_mask, sqrt2_2);
        }

        //======================================================================
        // PHASE 2: CROSS-GROUP RADIX-4 COMBINES (BACKWARD)
        // Process in bit-reversed order: 0, 4, 2, 6, 1, 5, 3, 7
        //======================================================================

        // Position 0 (index 0): Identity (same for forward/backward)
        RADIX32_POSITION_IDENTITY_UNMASKED_BV(0, 0);

        // Position 4 (index 1): W8 FAST-PATH (BACKWARD - conjugated)
        RADIX32_POSITION4_FAST_BACKWARD_UNMASKED(4);

        // Position 2 (index 2): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(2, 2, pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im);

        // Position 6 (index 3): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(3, 6, pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im);

        // Position 1 (index 4): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(4, 1, pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im);

        // Position 5 (index 5): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(5, 5, pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im);

        // Position 3 (index 6): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(6, 3, pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im);

        // Position 7 (index 7): Generic twiddles (conjugated)
        RADIX32_POSITION_TWIDDLED_UNMASKED_BV(7, 7, pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im);
    }

    //==========================================================================
    // TAIL HANDLING: Process remaining k < 8 samples with masks
    //==========================================================================

    //==========================================================================
    // TAIL HANDLING: Process remaining k < 8 samples with masks (BACKWARD)
    //==========================================================================

    if (k < tile_size)
    {
        const size_t tail = tile_size - k;
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);

        __m512d A_re[8], A_im[8];
        __m512d B_re[8], B_im[8];
        __m512d C_re[8], C_im[8];
        __m512d D_re[8], D_im[8];

        //======================================================================
        // PHASE 1: RADIX-8 PAIR EMITTERS WITH MASKED LOADS (BACKWARD)
        //======================================================================

        //----------------------------------------------------------------------
        // GROUP A: Load inputs with masks
        //----------------------------------------------------------------------
        __m512d xa0_re = load_masked(mask, &tile_in_re[0 * tile_size + k]);
        __m512d xa0_im = load_masked(mask, &tile_in_im[0 * tile_size + k]);
        __m512d xa1_re = load_masked(mask, &tile_in_re[1 * tile_size + k]);
        __m512d xa1_im = load_masked(mask, &tile_in_im[1 * tile_size + k]);
        __m512d xa2_re = load_masked(mask, &tile_in_re[2 * tile_size + k]);
        __m512d xa2_im = load_masked(mask, &tile_in_im[2 * tile_size + k]);
        __m512d xa3_re = load_masked(mask, &tile_in_re[3 * tile_size + k]);
        __m512d xa3_im = load_masked(mask, &tile_in_im[3 * tile_size + k]);
        __m512d xa4_re = load_masked(mask, &tile_in_re[4 * tile_size + k]);
        __m512d xa4_im = load_masked(mask, &tile_in_im[4 * tile_size + k]);
        __m512d xa5_re = load_masked(mask, &tile_in_re[5 * tile_size + k]);
        __m512d xa5_im = load_masked(mask, &tile_in_im[5 * tile_size + k]);
        __m512d xa6_re = load_masked(mask, &tile_in_re[6 * tile_size + k]);
        __m512d xa6_im = load_masked(mask, &tile_in_im[6 * tile_size + k]);
        __m512d xa7_re = load_masked(mask, &tile_in_re[7 * tile_size + k]);
        __m512d xa7_im = load_masked(mask, &tile_in_im[7 * tile_size + k]);

        // Even wave (pairs 0,4 and 2,6) - BACKWARD
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[0], &A_im[0], &A_re[1], &A_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &A_re[2], &A_im[2], &A_re[3], &A_im[3],
                sign_mask);
        }

        // Odd wave (pairs 1,5 and 3,7) - BACKWARD
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xa0_re, xa0_im, xa1_re, xa1_im, xa2_re, xa2_im, xa3_re, xa3_im,
                xa4_re, xa4_im, xa5_re, xa5_im, xa6_re, xa6_im, xa7_re, xa7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[4], &A_im[4], &A_re[5], &A_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &A_re[6], &A_im[6], &A_re[7], &A_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP B: Load inputs with masks
        //----------------------------------------------------------------------
        __m512d xb0_re = load_masked(mask, &tile_in_re[8 * tile_size + k]);
        __m512d xb0_im = load_masked(mask, &tile_in_im[8 * tile_size + k]);
        __m512d xb1_re = load_masked(mask, &tile_in_re[9 * tile_size + k]);
        __m512d xb1_im = load_masked(mask, &tile_in_im[9 * tile_size + k]);
        __m512d xb2_re = load_masked(mask, &tile_in_re[10 * tile_size + k]);
        __m512d xb2_im = load_masked(mask, &tile_in_im[10 * tile_size + k]);
        __m512d xb3_re = load_masked(mask, &tile_in_re[11 * tile_size + k]);
        __m512d xb3_im = load_masked(mask, &tile_in_im[11 * tile_size + k]);
        __m512d xb4_re = load_masked(mask, &tile_in_re[12 * tile_size + k]);
        __m512d xb4_im = load_masked(mask, &tile_in_im[12 * tile_size + k]);
        __m512d xb5_re = load_masked(mask, &tile_in_re[13 * tile_size + k]);
        __m512d xb5_im = load_masked(mask, &tile_in_im[13 * tile_size + k]);
        __m512d xb6_re = load_masked(mask, &tile_in_re[14 * tile_size + k]);
        __m512d xb6_im = load_masked(mask, &tile_in_im[14 * tile_size + k]);
        __m512d xb7_re = load_masked(mask, &tile_in_re[15 * tile_size + k]);
        __m512d xb7_im = load_masked(mask, &tile_in_im[15 * tile_size + k]);

        // Even wave - BACKWARD
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[0], &B_im[0], &B_re[1], &B_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &B_re[2], &B_im[2], &B_re[3], &B_im[3],
                sign_mask);
        }

        // Odd wave - BACKWARD
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xb0_re, xb0_im, xb1_re, xb1_im, xb2_re, xb2_im, xb3_re, xb3_im,
                xb4_re, xb4_im, xb5_re, xb5_im, xb6_re, xb6_im, xb7_re, xb7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[4], &B_im[4], &B_re[5], &B_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &B_re[6], &B_im[6], &B_re[7], &B_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP C: Load inputs with masks
        //----------------------------------------------------------------------
        __m512d xc0_re = load_masked(mask, &tile_in_re[16 * tile_size + k]);
        __m512d xc0_im = load_masked(mask, &tile_in_im[16 * tile_size + k]);
        __m512d xc1_re = load_masked(mask, &tile_in_re[17 * tile_size + k]);
        __m512d xc1_im = load_masked(mask, &tile_in_im[17 * tile_size + k]);
        __m512d xc2_re = load_masked(mask, &tile_in_re[18 * tile_size + k]);
        __m512d xc2_im = load_masked(mask, &tile_in_im[18 * tile_size + k]);
        __m512d xc3_re = load_masked(mask, &tile_in_re[19 * tile_size + k]);
        __m512d xc3_im = load_masked(mask, &tile_in_im[19 * tile_size + k]);
        __m512d xc4_re = load_masked(mask, &tile_in_re[20 * tile_size + k]);
        __m512d xc4_im = load_masked(mask, &tile_in_im[20 * tile_size + k]);
        __m512d xc5_re = load_masked(mask, &tile_in_re[21 * tile_size + k]);
        __m512d xc5_im = load_masked(mask, &tile_in_im[21 * tile_size + k]);
        __m512d xc6_re = load_masked(mask, &tile_in_re[22 * tile_size + k]);
        __m512d xc6_im = load_masked(mask, &tile_in_im[22 * tile_size + k]);
        __m512d xc7_re = load_masked(mask, &tile_in_re[23 * tile_size + k]);
        __m512d xc7_im = load_masked(mask, &tile_in_im[23 * tile_size + k]);

        // Even wave - BACKWARD
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[0], &C_im[0], &C_re[1], &C_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &C_re[2], &C_im[2], &C_re[3], &C_im[3],
                sign_mask);
        }

        // Odd wave - BACKWARD
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xc0_re, xc0_im, xc1_re, xc1_im, xc2_re, xc2_im, xc3_re, xc3_im,
                xc4_re, xc4_im, xc5_re, xc5_im, xc6_re, xc6_im, xc7_re, xc7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[4], &C_im[4], &C_re[5], &C_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &C_re[6], &C_im[6], &C_re[7], &C_im[7],
                sign_mask, sqrt2_2);
        }

        //----------------------------------------------------------------------
        // GROUP D: Load inputs with masks
        //----------------------------------------------------------------------
        __m512d xd0_re = load_masked(mask, &tile_in_re[24 * tile_size + k]);
        __m512d xd0_im = load_masked(mask, &tile_in_im[24 * tile_size + k]);
        __m512d xd1_re = load_masked(mask, &tile_in_re[25 * tile_size + k]);
        __m512d xd1_im = load_masked(mask, &tile_in_im[25 * tile_size + k]);
        __m512d xd2_re = load_masked(mask, &tile_in_re[26 * tile_size + k]);
        __m512d xd2_im = load_masked(mask, &tile_in_im[26 * tile_size + k]);
        __m512d xd3_re = load_masked(mask, &tile_in_re[27 * tile_size + k]);
        __m512d xd3_im = load_masked(mask, &tile_in_im[27 * tile_size + k]);
        __m512d xd4_re = load_masked(mask, &tile_in_re[28 * tile_size + k]);
        __m512d xd4_im = load_masked(mask, &tile_in_im[28 * tile_size + k]);
        __m512d xd5_re = load_masked(mask, &tile_in_re[29 * tile_size + k]);
        __m512d xd5_im = load_masked(mask, &tile_in_im[29 * tile_size + k]);
        __m512d xd6_re = load_masked(mask, &tile_in_re[30 * tile_size + k]);
        __m512d xd6_im = load_masked(mask, &tile_in_im[30 * tile_size + k]);
        __m512d xd7_re = load_masked(mask, &tile_in_re[31 * tile_size + k]);
        __m512d xd7_im = load_masked(mask, &tile_in_im[31 * tile_size + k]);

        // Even wave - BACKWARD
        {
            __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
            radix8_compute_t0123_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

            radix8_emit_pair_04_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[0], &D_im[0], &D_re[1], &D_im[1]);

            radix8_emit_pair_26_from_t0123_backward_avx512(
                t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
                &D_re[2], &D_im[2], &D_re[3], &D_im[3],
                sign_mask);
        }

        // Odd wave - BACKWARD
        {
            __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
            radix8_compute_t4567_avx512(
                xd0_re, xd0_im, xd1_re, xd1_im, xd2_re, xd2_im, xd3_re, xd3_im,
                xd4_re, xd4_im, xd5_re, xd5_im, xd6_re, xd6_im, xd7_re, xd7_im,
                &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

            radix8_emit_pair_15_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[4], &D_im[4], &D_re[5], &D_im[5],
                sign_mask, sqrt2_2);

            radix8_emit_pair_37_from_t4567_backward_avx512(
                t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
                &D_re[6], &D_im[6], &D_re[7], &D_im[7],
                sign_mask, sqrt2_2);
        }

        //======================================================================
        // PHASE 2: CROSS-GROUP RADIX-4 COMBINES WITH MASKED STORES (BACKWARD)
        //======================================================================

        RADIX32_POSITION_IDENTITY_MASKED_BV(0, 0, mask);
        RADIX32_POSITION4_FAST_BACKWARD_MASKED(4, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(2, 2, pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(3, 6, pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(4, 1, pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(5, 5, pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(6, 3, pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im, mask);
        RADIX32_POSITION_TWIDDLED_MASKED_BV(7, 7, pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im, mask);
    }
}

//==============================================================================
// TOP-LEVEL EXECUTION FUNCTIONS
//==============================================================================

/**
 * @brief Main execution function for radix-32 stage (forward)
 *
 * Orchestrates the complete tiled execution:
 *   1. Loop over K in tiles of RADIX32_TILE_SIZE
 *   2. Gather from 32 stripes → tile-local buffers
 *   3. Execute fused radix-32 butterfly
 *   4. Scatter from tile-local buffers → 32 stripes
 *
 * @param K Total butterflies in stage (must be multiple of 8)
 * @param in_re Input real [32 stripes][K] (SoA)
 * @param in_im Input imag [32 stripes][K] (SoA)
 * @param out_re Output real [32 stripes][K] (SoA)
 * @param out_im Output imag [32 stripes][K] (SoA)
 * @param pass1_plan Pass 1 twiddle plan
 * @param pass2_plan Pass 2 geometric constants
 */
TARGET_AVX512
void radix32_execute_forward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix32_pass1_plan_t *RESTRICT pass1_plan,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    assert((K & 7) == 0 && "K must be multiple of 8");

    // Determine tile size and NT threshold
    const size_t tile_size = (K <= RADIX32_SMALL_TILE_SIZE)
                                 ? RADIX32_SMALL_TILE_SIZE
                                 : RADIX32_TILE_SIZE;

    const bool use_nt = (K > RADIX32_NT_THRESHOLD);

    // Allocate tile workspace
    radix32_tile_workspace_t workspace;

    // Main tiled loop
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        const size_t current_tile_size = (k_tile + tile_size <= K)
                                             ? tile_size
                                             : (K - k_tile);

        //======================================================================
        // GATHER: Stripe layout → Tile-local layout
        //======================================================================
        radix32_gather_stripes_to_tile(
            in_re, in_im, K, k_tile, current_tile_size,
            workspace.input_re, workspace.input_im);

        //======================================================================
        // COMPUTE: Fused radix-32 8×4 butterfly
        //======================================================================
        radix32_fused_butterfly_forward_avx512(
            workspace.input_re, workspace.input_im,
            workspace.output_re, workspace.output_im,
            current_tile_size, pass1_plan, pass2_plan);

        //======================================================================
        // SCATTER: Tile-local layout → Stripe layout
        //======================================================================
        if (use_nt)
        {
            radix32_scatter_tile_to_stripes_nt(
                workspace.output_re, workspace.output_im,
                current_tile_size, k_tile, K, out_re, out_im);
        }
        else
        {
            radix32_scatter_tile_to_stripes(
                workspace.output_re, workspace.output_im,
                current_tile_size, k_tile, K, out_re, out_im);
        }
    }
}

/**
 * @brief Main execution function for radix-32 stage (backward/inverse)
 */
TARGET_AVX512
void radix32_execute_backward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix32_pass1_plan_t *RESTRICT pass1_plan,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    assert((K & 7) == 0 && "K must be multiple of 8");

    const size_t tile_size = (K <= RADIX32_SMALL_TILE_SIZE)
                                 ? RADIX32_SMALL_TILE_SIZE
                                 : RADIX32_TILE_SIZE;

    const bool use_nt = (K > RADIX32_NT_THRESHOLD);

    radix32_tile_workspace_t workspace;

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        const size_t current_tile_size = (k_tile + tile_size <= K)
                                             ? tile_size
                                             : (K - k_tile);

        // GATHER
        radix32_gather_stripes_to_tile(
            in_re, in_im, K, k_tile, current_tile_size,
            workspace.input_re, workspace.input_im);

        // COMPUTE (BACKWARD)
        radix32_fused_butterfly_backward_avx512(
            workspace.input_re, workspace.input_im,
            workspace.output_re, workspace.output_im,
            current_tile_size, pass1_plan, pass2_plan);

        // SCATTER
        if (use_nt)
        {
            radix32_scatter_tile_to_stripes_nt(
                workspace.output_re, workspace.output_im,
                current_tile_size, k_tile, K, out_re, out_im);
        }
        else
        {
            radix32_scatter_tile_to_stripes(
                workspace.output_re, workspace.output_im,
                current_tile_size, k_tile, K, out_re, out_im);
        }
    }
}

#endif


/*

Structural / integration issues

❗ Forward tail path is incomplete:
In radix32_fused_butterfly_forward_avx512, the masked tail block only loads xa0_* and then has // 
... placeholders, yet you call the masked position macros that expect fully populated A_re/B_re/C_re/D_re (and, transitively, the pair-emitters to have run for all four groups). 
That’s a functional bug: those arrays will contain garbage in the tail.
Fix: mirror the backward tail section you wrote (which is complete) — load masked inputs for groups A–D and run the two-wave emitters to fill A/B/C/D before the masked RADIX32_POSITION_* calls.
*/