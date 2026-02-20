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
#include "fft_radix32_avx512_tile_io.h"

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
// PASS 2 (CROSS-GROUP) GEOMETRIC CONSTANTS
//==============================================================================

typedef struct
{
    // Position 0: Identity (no twiddles needed)

    // Position 1: W_32^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos1_w1_re, pos1_w1_im;
    ALIGNAS(64)
    __m512d pos1_w2_re, pos1_w2_im;
    ALIGNAS(64)
    __m512d pos1_w3_re, pos1_w3_im;

    // Position 2: W_16^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos2_w1_re, pos2_w1_im;
    ALIGNAS(64)
    __m512d pos2_w2_re, pos2_w2_im;
    ALIGNAS(64)
    __m512d pos2_w3_re, pos2_w3_im;

    // Position 3: W_32^3, W_32^6, W_32^9 - NOW PRECOMPUTED ✅
    ALIGNAS(64)
    __m512d pos3_w1_re, pos3_w1_im;
    ALIGNAS(64)
    __m512d pos3_w2_re, pos3_w2_im;
    ALIGNAS(64)
    __m512d pos3_w3_re, pos3_w3_im;

    // Position 4: W_8^g for g=1,2,3
    ALIGNAS(64)
    __m512d pos4_w1_re, pos4_w1_im;
    ALIGNAS(64)
    __m512d pos4_w2_re, pos4_w2_im;
    ALIGNAS(64)
    __m512d pos4_w3_re, pos4_w3_im;

    // Position 5: W_32^5, W_32^10, W_32^15 - NOW PRECOMPUTED ✅
    ALIGNAS(64)
    __m512d pos5_w1_re, pos5_w1_im;
    ALIGNAS(64)
    __m512d pos5_w2_re, pos5_w2_im;
    ALIGNAS(64)
    __m512d pos5_w3_re, pos5_w3_im;

    // Position 6: W_32^6, W_32^12, W_32^18 - NOW PRECOMPUTED ✅
    ALIGNAS(64)
    __m512d pos6_w1_re, pos6_w1_im;
    ALIGNAS(64)
    __m512d pos6_w2_re, pos6_w2_im;
    ALIGNAS(64)
    __m512d pos6_w3_re, pos6_w3_im;

    // Position 7: W_32^7, W_32^14, W_32^21 - NOW PRECOMPUTED ✅
    ALIGNAS(64)
    __m512d pos7_w1_re, pos7_w1_im;
    ALIGNAS(64)
    __m512d pos7_w2_re, pos7_w2_im;
    ALIGNAS(64)
    __m512d pos7_w3_re, pos7_w3_im;

    bool is_forward;
} radix32_pass2_plan_t;

//==============================================================================
// UNIFIED RADIX-32 PLAN
//==============================================================================

typedef struct
{
    size_t K;         ///< Butterflies per radix-32 stage (K = N/32)
    size_t tile_size; ///< Samples per tile (64 for normal, 32 for small K)

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
    const double sign = is_forward ? 1.0 : -1.0;

    const double PI = 3.14159265358979323846;

    //==========================================================================
    // POSITION 1: W_32, W_32², W_32³
    //==========================================================================
    double angle1 = -sign * 2.0 * PI / 32.0;
    plan->pos1_w1_re = _mm512_set1_pd(cos(angle1));
    plan->pos1_w1_im = _mm512_set1_pd(sin(angle1));

    double angle2 = -sign * 4.0 * PI / 32.0;
    plan->pos1_w2_re = _mm512_set1_pd(cos(angle2));
    plan->pos1_w2_im = _mm512_set1_pd(sin(angle2));

    double angle3 = -sign * 6.0 * PI / 32.0;
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
    // POSITION 3: W_32^3, W_32^6, W_32^9 - PRECOMPUTE ALL THREE ✅
    //==========================================================================
    {
        double angle_p3 = -sign * 3.0 * 2.0 * PI / 32.0;
        __m512d seed_re = _mm512_set1_pd(cos(angle_p3));
        __m512d seed_im = _mm512_set1_pd(sin(angle_p3));

        cross_twiddle_powers3_optimized(
            seed_re, seed_im,
            &plan->pos3_w1_re, &plan->pos3_w1_im,
            &plan->pos3_w2_re, &plan->pos3_w2_im,
            &plan->pos3_w3_re, &plan->pos3_w3_im);
    }

    //==========================================================================
    // POSITION 4: W_32^4, W_32^8, W_32^12 = W_8^1, W_8^2, W_8^3
    //==========================================================================
    {
        double angle_4 = -sign * 2.0 * PI * 4.0 / 32.0;
        double angle_8 = -sign * 2.0 * PI * 8.0 / 32.0;
        double angle_12 = -sign * 2.0 * PI * 12.0 / 32.0;

        plan->pos4_w1_re = _mm512_set1_pd(cos(angle_4));
        plan->pos4_w1_im = _mm512_set1_pd(sin(angle_4));

        plan->pos4_w2_re = _mm512_set1_pd(cos(angle_8));
        plan->pos4_w2_im = _mm512_set1_pd(sin(angle_8));

        plan->pos4_w3_re = _mm512_set1_pd(cos(angle_12));
        plan->pos4_w3_im = _mm512_set1_pd(sin(angle_12));
    }

    //==========================================================================
    // POSITION 5: W_32^5, W_32^10, W_32^15 - PRECOMPUTE ALL THREE ✅
    //==========================================================================
    {
        double angle_p5 = -sign * 5.0 * 2.0 * PI / 32.0;
        __m512d seed_re = _mm512_set1_pd(cos(angle_p5));
        __m512d seed_im = _mm512_set1_pd(sin(angle_p5));

        cross_twiddle_powers3_optimized(
            seed_re, seed_im,
            &plan->pos5_w1_re, &plan->pos5_w1_im,
            &plan->pos5_w2_re, &plan->pos5_w2_im,
            &plan->pos5_w3_re, &plan->pos5_w3_im);
    }

    //==========================================================================
    // POSITION 6: W_32^6, W_32^12, W_32^18 - PRECOMPUTE ALL THREE ✅
    //==========================================================================
    {
        double angle_p6 = -sign * 6.0 * 2.0 * PI / 32.0;
        __m512d seed_re = _mm512_set1_pd(cos(angle_p6));
        __m512d seed_im = _mm512_set1_pd(sin(angle_p6));

        cross_twiddle_powers3_optimized(
            seed_re, seed_im,
            &plan->pos6_w1_re, &plan->pos6_w1_im,
            &plan->pos6_w2_re, &plan->pos6_w2_im,
            &plan->pos6_w3_re, &plan->pos6_w3_im);
    }

    //==========================================================================
    // POSITION 7: W_32^7, W_32^14, W_32^21 - PRECOMPUTE ALL THREE ✅
    //==========================================================================
    {
        double angle_p7 = -sign * 7.0 * 2.0 * PI / 32.0;
        __m512d seed_re = _mm512_set1_pd(cos(angle_p7));
        __m512d seed_im = _mm512_set1_pd(sin(angle_p7));

        cross_twiddle_powers3_optimized(
            seed_re, seed_im,
            &plan->pos7_w1_re, &plan->pos7_w1_im,
            &plan->pos7_w2_re, &plan->pos7_w2_im,
            &plan->pos7_w3_re, &plan->pos7_w3_im);
    }

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

    // Initialize Pass 2 (geometric constants)
    if (radix32_prepare_pass2_plan(&plan->pass2, is_forward) != 0)
        return -1;

    return 0;
}

//==============================================================================
// STREAMING STORE HELPERS
//==============================================================================

/**
 * @brief Conditional streaming store (full vector)
 */
TARGET_AVX512
FORCE_INLINE void store8_pd(double *dst, __m512d v, bool use_nt)
{
    if (use_nt)
    {
        _mm512_stream_pd(dst, v);
    }
    else
    {
        _mm512_store_pd(dst, v);
    }
}

/**
 * @brief Conditional streaming store (masked)
 * Note: Only stream if mask is full (0xFF), otherwise regular masked store
 */
TARGET_AVX512
FORCE_INLINE void mask_store8_pd(double *dst, __mmask8 mask, __m512d v, bool use_nt)
{
    if (use_nt && mask == 0xFF)
    {
        _mm512_stream_pd(dst, v);
    }
    else
    {
        _mm512_mask_store_pd(dst, mask, v);
    }
}

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
 * @brief Emit pair (0,4) from t0..t3 - Backward (W8^0 = 1, identity)
 * IDENTICAL to forward version (W8^0* = W8^0 = 1)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_04_from_t0123_backward_avx512(
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

    // Stage 3: Final butterfly (W8^0 = 1, identity - same for forward/backward)
    *y0_re = _mm512_add_pd(u0_re, u1_re);
    *y0_im = _mm512_add_pd(u0_im, u1_im);
    *y4_re = _mm512_sub_pd(u0_re, u1_re);
    *y4_im = _mm512_sub_pd(u0_im, u1_im);
}

/**
 * @brief Emit pair (2,6) from t0..t3 - Backward (W8^2 = +j)
 */
TARGET_AVX512
FORCE_INLINE void radix8_emit_pair_26_from_t0123_backward_avx512(
    __m512d t0_re, __m512d t0_im,
    __m512d t1_re, __m512d t1_im,
    __m512d t2_re, __m512d t2_im,
    __m512d t3_re, __m512d t3_im,
    __m512d *RESTRICT y2_re, __m512d *RESTRICT y2_im,
    __m512d *RESTRICT y6_re, __m512d *RESTRICT y6_im,
    __m512d sign_mask)
{
    __m512d u2_re = _mm512_sub_pd(t0_re, t2_re);
    __m512d u2_im = _mm512_sub_pd(t0_im, t2_im);

    __m512d u3_re = _mm512_sub_pd(t1_re, t3_re);
    __m512d u3_im = _mm512_sub_pd(t1_im, t3_im);

    __m512d u3_tw_re, u3_tw_im;
    rot_pos_j(u3_re, u3_im, sign_mask, &u3_tw_re, &u3_tw_im);

    *y2_re = _mm512_add_pd(u2_re, u3_tw_re);
    *y2_im = _mm512_add_pd(u2_im, u3_tw_im);
    *y6_re = _mm512_sub_pd(u2_re, u3_tw_re);
    *y6_im = _mm512_sub_pd(u2_im, u3_tw_im);
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
// HOISTED ADDRESS HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Load 8 stripes (one group) with precomputed base pointers
 *
 * @param bases_re Array of 8 base pointers for real part
 * @param bases_im Array of 8 base pointers for imag part
 * @param k Index into each stripe
 * @param x_re Output: 8 real vectors
 * @param x_im Output: 8 imag vectors
 */
TARGET_AVX512
FORCE_INLINE void load_group_hoisted(
    double *RESTRICT const *bases_re,
    double *RESTRICT const *bases_im,
    size_t k,
    __m512d *RESTRICT x_re,
    __m512d *RESTRICT x_im)
{
    x_re[0] = _mm512_load_pd(&bases_re[0][k]);
    x_im[0] = _mm512_load_pd(&bases_im[0][k]);
    x_re[1] = _mm512_load_pd(&bases_re[1][k]);
    x_im[1] = _mm512_load_pd(&bases_im[1][k]);
    x_re[2] = _mm512_load_pd(&bases_re[2][k]);
    x_im[2] = _mm512_load_pd(&bases_im[2][k]);
    x_re[3] = _mm512_load_pd(&bases_re[3][k]);
    x_im[3] = _mm512_load_pd(&bases_im[3][k]);
    x_re[4] = _mm512_load_pd(&bases_re[4][k]);
    x_im[4] = _mm512_load_pd(&bases_im[4][k]);
    x_re[5] = _mm512_load_pd(&bases_re[5][k]);
    x_im[5] = _mm512_load_pd(&bases_im[5][k]);
    x_re[6] = _mm512_load_pd(&bases_re[6][k]);
    x_im[6] = _mm512_load_pd(&bases_im[6][k]);
    x_re[7] = _mm512_load_pd(&bases_re[7][k]);
    x_im[7] = _mm512_load_pd(&bases_im[7][k]);
}

/**
 * @brief Load group with mask (tail handling) using precomputed bases
 */
TARGET_AVX512
FORCE_INLINE void load_group_hoisted_masked(
    double *RESTRICT const *bases_re,
    double *RESTRICT const *bases_im,
    size_t k,
    __mmask8 mask,
    __m512d *RESTRICT x_re,
    __m512d *RESTRICT x_im)
{
    x_re[0] = _mm512_maskz_load_pd(mask, &bases_re[0][k]);
    x_im[0] = _mm512_maskz_load_pd(mask, &bases_im[0][k]);
    x_re[1] = _mm512_maskz_load_pd(mask, &bases_re[1][k]);
    x_im[1] = _mm512_maskz_load_pd(mask, &bases_im[1][k]);
    x_re[2] = _mm512_maskz_load_pd(mask, &bases_re[2][k]);
    x_im[2] = _mm512_maskz_load_pd(mask, &bases_im[2][k]);
    x_re[3] = _mm512_maskz_load_pd(mask, &bases_re[3][k]);
    x_im[3] = _mm512_maskz_load_pd(mask, &bases_im[3][k]);
    x_re[4] = _mm512_maskz_load_pd(mask, &bases_re[4][k]);
    x_im[4] = _mm512_maskz_load_pd(mask, &bases_im[4][k]);
    x_re[5] = _mm512_maskz_load_pd(mask, &bases_re[5][k]);
    x_im[5] = _mm512_maskz_load_pd(mask, &bases_im[5][k]);
    x_re[6] = _mm512_maskz_load_pd(mask, &bases_re[6][k]);
    x_im[6] = _mm512_maskz_load_pd(mask, &bases_im[6][k]);
    x_re[7] = _mm512_maskz_load_pd(mask, &bases_re[7][k]);
    x_im[7] = _mm512_maskz_load_pd(mask, &bases_im[7][k]);
}

/**
 * @brief Prefetch one group (selective stripes) using precomputed bases
 */
TARGET_AVX512
FORCE_INLINE void prefetch_group_hoisted(
    double *RESTRICT const *bases_re,
    double *RESTRICT const *bases_im,
    size_t k_offset)
{
    _mm_prefetch((const char *)&bases_re[0][k_offset], _MM_HINT_T0);
    _mm_prefetch((const char *)&bases_im[0][k_offset], _MM_HINT_T0);
    _mm_prefetch((const char *)&bases_re[1][k_offset], _MM_HINT_T0);
    _mm_prefetch((const char *)&bases_im[1][k_offset], _MM_HINT_T0);
    _mm_prefetch((const char *)&bases_re[4][k_offset], _MM_HINT_T0);
    _mm_prefetch((const char *)&bases_im[4][k_offset], _MM_HINT_T0);
}

/**
 * @brief Prefetch into L2 cache (far ahead) using precomputed bases
 */
TARGET_AVX512
FORCE_INLINE void prefetch_group_hoisted_L2(
    double *RESTRICT const *bases_re,
    double *RESTRICT const *bases_im,
    size_t k_offset)
{
    _mm_prefetch((const char *)&bases_re[0][k_offset], _MM_HINT_T1);
    _mm_prefetch((const char *)&bases_im[0][k_offset], _MM_HINT_T1);
    _mm_prefetch((const char *)&bases_re[1][k_offset], _MM_HINT_T1);
    _mm_prefetch((const char *)&bases_im[1][k_offset], _MM_HINT_T1);
    _mm_prefetch((const char *)&bases_re[4][k_offset], _MM_HINT_T1);
    _mm_prefetch((const char *)&bases_im[4][k_offset], _MM_HINT_T1);
}

//==============================================================================
// HELPER FUNCTIONS - ALL FORCE-INLINED
//==============================================================================

/**
 * @brief Process one radix-8 group (even + odd waves)
 */
TARGET_AVX512
FORCE_INLINE void process_radix8_group(
    const __m512d x_re[8],
    const __m512d x_im[8],
    __m512d out_re[8],
    __m512d out_im[8],
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    // Even wave (pairs 0,4 and 2,6)
    __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
    radix8_compute_t0123_avx512(
        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],
        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7],
        &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

    radix8_emit_pair_04_from_t0123_avx512(
        t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
        &out_re[0], &out_im[0], &out_re[1], &out_im[1]);

    radix8_emit_pair_26_from_t0123_avx512(
        t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
        &out_re[2], &out_im[2], &out_re[3], &out_im[3],
        sign_mask);

    // Odd wave (pairs 1,5 and 3,7)
    __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
    radix8_compute_t4567_avx512(
        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],
        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7],
        &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

    radix8_emit_pair_15_from_t4567_avx512(
        t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
        &out_re[4], &out_im[4], &out_re[5], &out_im[5],
        sign_mask, sqrt2_2);

    radix8_emit_pair_37_from_t4567_avx512(
        t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
        &out_re[6], &out_im[6], &out_re[7], &out_im[7],
        sign_mask, sqrt2_2);
}

//==============================================================================
// FORWARD CROSS-GROUP POSITION FUNCTIONS
//==============================================================================

TARGET_AVX512
FORCE_INLINE void radix32_position_identity_hoisted_forward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

    radix4_butterfly_core_fv_avx512( // FORWARD
        A_re[pos], A_im[pos], B_re[pos], B_im[pos],
        C_re[pos], C_im[pos], D_re[pos], D_im[pos],
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

TARGET_AVX512
FORCE_INLINE void radix32_position_identity_hoisted_forward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask, __mmask8 mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

    radix4_butterfly_core_fv_avx512( // FORWARD
        A_re[pos], A_im[pos], B_re[pos], B_im[pos],
        C_re[pos], C_im[pos], D_re[pos], D_im[pos],
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

TARGET_AVX512
FORCE_INLINE void radix32_position_twiddled_hoisted_forward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask,
    __m512d w1_re, __m512d w1_im,
    __m512d w2_re, __m512d w2_im,
    __m512d w3_re, __m512d w3_im,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[pos], a_im = A_im[pos];
    __m512d b_re = B_re[pos], b_im = B_im[pos];
    __m512d c_re = C_re[pos], c_im = C_im[pos];
    __m512d d_re = D_re[pos], d_im = D_im[pos];

    cmul_avx512(b_re, b_im, w1_re, w1_im, &b_re, &b_im);
    cmul_avx512(c_re, c_im, w2_re, w2_im, &c_re, &c_im);
    cmul_avx512(d_re, d_im, w3_re, w3_im, &d_re, &d_im);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_fv_avx512( // FORWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

TARGET_AVX512
FORCE_INLINE void radix32_position4_fast_hoisted_forward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int stripe, size_t k,
    __m512d sign_mask, __m512d sqrt2_2,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[4], a_im = A_im[4];
    __m512d b_re = B_re[4], b_im = B_im[4];
    __m512d c_re = C_re[4], c_im = C_im[4];
    __m512d d_re = D_re[4], d_im = D_im[4];

    // W8^1 (forward):  (1 - i)/√2
    // re = s*(b_re + b_im),  im = s*(b_im - b_re)
    __m512d b_sum = _mm512_add_pd(b_re, b_im);
    __m512d b_diff = _mm512_sub_pd(b_im, b_re);
    b_re = _mm512_mul_pd(sqrt2_2, b_sum);
    b_im = _mm512_mul_pd(sqrt2_2, b_diff);

    // W8^2 (forward):  -i  => rot_neg_j
    __m512d c_tw_re, c_tw_im;
    rot_neg_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
    c_re = c_tw_re;
    c_im = c_tw_im;

    // W8^3 (forward):  (-1 - i)/√2
    // re = s*(d_im - d_re),  im = s*(-d_re - d_im)
    __m512d d_im_minus_re = _mm512_sub_pd(d_im, d_re);
    __m512d d_neg_re = _mm512_xor_pd(d_re, sign_mask);         // -d_re
    __m512d d_neg_re_minus_im = _mm512_sub_pd(d_neg_re, d_im); // -d_re - d_im
    __m512d d_re2 = _mm512_mul_pd(sqrt2_2, d_im_minus_re);
    __m512d d_im2 = _mm512_mul_pd(sqrt2_2, d_neg_re_minus_im);
    d_re = d_re2;
    d_im = d_im2;

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_fv_avx512( // FORWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

// twiddled (masked) - FORWARD, hoisted store bases
TARGET_AVX512
FORCE_INLINE void radix32_position_twiddled_hoisted_forward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask, __mmask8 mask,
    __m512d w1_re, __m512d w1_im,
    __m512d w2_re, __m512d w2_im,
    __m512d w3_re, __m512d w3_im,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[pos], a_im = A_im[pos];
    __m512d b_re = B_re[pos], b_im = B_im[pos];
    __m512d c_re = C_re[pos], c_im = C_im[pos];
    __m512d d_re = D_re[pos], d_im = D_im[pos];

    cmul_avx512(b_re, b_im, w1_re, w1_im, &b_re, &b_im);
    cmul_avx512(c_re, c_im, w2_re, w2_im, &c_re, &c_im);
    cmul_avx512(d_re, d_im, w3_re, w3_im, &d_re, &d_im);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_fv_avx512( // FORWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

// fast position-4 (masked) - FORWARD, hoisted store bases
TARGET_AVX512
FORCE_INLINE void radix32_position4_fast_hoisted_forward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int stripe, size_t k,
    __m512d sign_mask, __m512d sqrt2_2, __mmask8 mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[4], a_im = A_im[4];
    __m512d b_re = B_re[4], b_im = B_im[4];
    __m512d c_re = C_re[4], c_im = C_im[4];
    __m512d d_re = D_re[4], d_im = D_im[4];

    // W8^1 (forward): (1 - i)/√2
    __m512d b_sum = _mm512_add_pd(b_re, b_im);
    __m512d b_diff = _mm512_sub_pd(b_im, b_re);
    b_re = _mm512_mul_pd(sqrt2_2, b_sum);
    b_im = _mm512_mul_pd(sqrt2_2, b_diff);

    // W8^2 (forward): -i
    __m512d c_tw_re, c_tw_im;
    rot_neg_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
    c_re = c_tw_re;
    c_im = c_tw_im;

    // W8^3 (forward): (-1 - i)/√2
    __m512d d_im_minus_re = _mm512_sub_pd(d_im, d_re);
    __m512d d_neg_re = _mm512_xor_pd(d_re, sign_mask);         // -d_re
    __m512d d_neg_re_minus_im = _mm512_sub_pd(d_neg_re, d_im); // -d_re - d_im
    __m512d d_re2 = _mm512_mul_pd(sqrt2_2, d_im_minus_re);
    __m512d d_im2 = _mm512_mul_pd(sqrt2_2, d_neg_re_minus_im);
    d_re = d_re2;
    d_im = d_im2;

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_fv_avx512( // FORWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

//==============================================================================
// OPTIMIZED STAGE TWIDDLE APPLICATION - PHASE 1
//==============================================================================

/**
 * @brief Apply stage twiddles with 4-way stripe unrolling and adaptive prefetch
 *
 * OPTIMIZATIONS:
 * - Process 4 stripes per k iteration (better cache reuse)
 * - Two-level prefetching (L2 @ +64, L1 @ +16)
 * - Skip stripe 0 (twiddle = 1)
 */
TARGET_AVX512
void radix32_apply_stage_twiddles_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t count,
    size_t stride,
    const double *RESTRICT stage_tw_re,
    const double *RESTRICT stage_tw_im)
{
    const size_t k_main = (count / 8) * 8;

    // Process in groups of 4 stripes for better cache locality
    for (int s_group = 1; s_group < 32; s_group += 4)
    {
        const int s_end = (s_group + 4 <= 32) ? (s_group + 4) : 32;
        const int s_count = s_end - s_group;

        // Pointers for this stripe group
        double *data_re[4];
        double *data_im[4];
        const double *tw_re[4];
        const double *tw_im[4];

        for (int i = 0; i < s_count; i++)
        {
            int s = s_group + i;
            data_re[i] = &stripe_re[s * stride];
            data_im[i] = &stripe_im[s * stride];
            tw_re[i] = &stage_tw_re[(s - 1) * stride];
            tw_im[i] = &stage_tw_im[(s - 1) * stride];
        }

        size_t k = 0;
        for (; k < k_main; k += 8)
        {
            // ADAPTIVE PREFETCH: Two-level strategy
            if (k + 64 < k_main)
            {
                for (int i = 0; i < s_count; i++)
                {
                    _mm_prefetch((const char *)&data_re[i][k + 64], _MM_HINT_T1);
                    _mm_prefetch((const char *)&data_im[i][k + 64], _MM_HINT_T1);
                    _mm_prefetch((const char *)&tw_re[i][k + 64], _MM_HINT_T1);
                    _mm_prefetch((const char *)&tw_im[i][k + 64], _MM_HINT_T1);
                }
            }

            if (k + 16 < k_main)
            {
                for (int i = 0; i < s_count; i++)
                {
                    _mm_prefetch((const char *)&data_re[i][k + 16], _MM_HINT_T0);
                    _mm_prefetch((const char *)&data_im[i][k + 16], _MM_HINT_T0);
                    _mm_prefetch((const char *)&tw_re[i][k + 16], _MM_HINT_T0);
                    _mm_prefetch((const char *)&tw_im[i][k + 16], _MM_HINT_T0);
                }
            }

            // Process all stripes in this group for current k
            for (int i = 0; i < s_count; i++)
            {
                __m512d x_re = _mm512_load_pd(&data_re[i][k]);
                __m512d x_im = _mm512_load_pd(&data_im[i][k]);

                __m512d tw_re_v = _mm512_load_pd(&tw_re[i][k]);
                __m512d tw_im_v = _mm512_load_pd(&tw_im[i][k]);

                // Complex multiply
                __m512d result_re = _mm512_fmsub_pd(x_re, tw_re_v, _mm512_mul_pd(x_im, tw_im_v));
                __m512d result_im = _mm512_fmadd_pd(x_re, tw_im_v, _mm512_mul_pd(x_im, tw_re_v));

                _mm512_store_pd(&data_re[i][k], result_re);
                _mm512_store_pd(&data_im[i][k], result_im);
            }
        }

        // Tail handling
        if (k < count)
        {
            const size_t tail = count - k;
            const __mmask8 mask = (__mmask8)((1u << tail) - 1u);

            for (int i = 0; i < s_count; i++)
            {
                __m512d x_re = _mm512_maskz_load_pd(mask, &data_re[i][k]);
                __m512d x_im = _mm512_maskz_load_pd(mask, &data_im[i][k]);

                __m512d tw_re_v = _mm512_maskz_load_pd(mask, &tw_re[i][k]);
                __m512d tw_im_v = _mm512_maskz_load_pd(mask, &tw_im[i][k]);

                __m512d result_re = _mm512_fmsub_pd(x_re, tw_re_v, _mm512_mul_pd(x_im, tw_im_v));
                __m512d result_im = _mm512_fmadd_pd(x_re, tw_im_v, _mm512_mul_pd(x_im, tw_re_v));

                _mm512_mask_store_pd(&data_re[i][k], mask, result_re);
                _mm512_mask_store_pd(&data_im[i][k], mask, result_im);
            }
        }
    }
}

/**
 * @brief Apply stage twiddles - BACKWARD (same as forward, twiddles pre-conjugated)
 */
TARGET_AVX512
FORCE_INLINE void radix32_apply_stage_twiddles_backward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t count,
    size_t stride,
    const double *RESTRICT stage_tw_re,
    const double *RESTRICT stage_tw_im)
{
    // Same implementation as forward - twiddles are pre-conjugated in plan
    radix32_apply_stage_twiddles_avx512(
        stripe_re, stripe_im, count, stride,
        stage_tw_re, stage_tw_im);
}

//==============================================================================
// UPDATED BUTTERFLY - SAME NAME, HOISTED ADDRESSES
//==============================================================================

TARGET_AVX512
void radix32_butterfly_strided_forward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t count,
    size_t stride,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    //==========================================================================
    // HOIST BASE ADDRESSES - Eliminate IMUL from hot path
    //==========================================================================

    // Group A (stripes 0-7)
    double *RESTRICT const grpA_re[8] = {
        &stripe_re[0 * stride],
        &stripe_re[1 * stride],
        &stripe_re[2 * stride],
        &stripe_re[3 * stride],
        &stripe_re[4 * stride],
        &stripe_re[5 * stride],
        &stripe_re[6 * stride],
        &stripe_re[7 * stride]};
    double *RESTRICT const grpA_im[8] = {
        &stripe_im[0 * stride],
        &stripe_im[1 * stride],
        &stripe_im[2 * stride],
        &stripe_im[3 * stride],
        &stripe_im[4 * stride],
        &stripe_im[5 * stride],
        &stripe_im[6 * stride],
        &stripe_im[7 * stride]};

    // Group B (stripes 8-15)
    double *RESTRICT const grpB_re[8] = {
        &stripe_re[8 * stride],
        &stripe_re[9 * stride],
        &stripe_re[10 * stride],
        &stripe_re[11 * stride],
        &stripe_re[12 * stride],
        &stripe_re[13 * stride],
        &stripe_re[14 * stride],
        &stripe_re[15 * stride]};
    double *RESTRICT const grpB_im[8] = {
        &stripe_im[8 * stride],
        &stripe_im[9 * stride],
        &stripe_im[10 * stride],
        &stripe_im[11 * stride],
        &stripe_im[12 * stride],
        &stripe_im[13 * stride],
        &stripe_im[14 * stride],
        &stripe_im[15 * stride]};

    // Group C (stripes 16-23)
    double *RESTRICT const grpC_re[8] = {
        &stripe_re[16 * stride],
        &stripe_re[17 * stride],
        &stripe_re[18 * stride],
        &stripe_re[19 * stride],
        &stripe_re[20 * stride],
        &stripe_re[21 * stride],
        &stripe_re[22 * stride],
        &stripe_re[23 * stride]};
    double *RESTRICT const grpC_im[8] = {
        &stripe_im[16 * stride],
        &stripe_im[17 * stride],
        &stripe_im[18 * stride],
        &stripe_im[19 * stride],
        &stripe_im[20 * stride],
        &stripe_im[21 * stride],
        &stripe_im[22 * stride],
        &stripe_im[23 * stride]};

    // Group D (stripes 24-31)
    double *RESTRICT const grpD_re[8] = {
        &stripe_re[24 * stride],
        &stripe_re[25 * stride],
        &stripe_re[26 * stride],
        &stripe_re[27 * stride],
        &stripe_re[28 * stride],
        &stripe_re[29 * stride],
        &stripe_re[30 * stride],
        &stripe_re[31 * stride]};
    double *RESTRICT const grpD_im[8] = {
        &stripe_im[24 * stride],
        &stripe_im[25 * stride],
        &stripe_im[26 * stride],
        &stripe_im[27 * stride],
        &stripe_im[28 * stride],
        &stripe_im[29 * stride],
        &stripe_im[30 * stride],
        &stripe_im[31 * stride]};

    double *RESTRICT const out_re[32] = {
        &stripe_re[0 * stride], &stripe_re[1 * stride], &stripe_re[2 * stride], &stripe_re[3 * stride],
        &stripe_re[4 * stride], &stripe_re[5 * stride], &stripe_re[6 * stride], &stripe_re[7 * stride],
        &stripe_re[8 * stride], &stripe_re[9 * stride], &stripe_re[10 * stride], &stripe_re[11 * stride],
        &stripe_re[12 * stride], &stripe_re[13 * stride], &stripe_re[14 * stride], &stripe_re[15 * stride],
        &stripe_re[16 * stride], &stripe_re[17 * stride], &stripe_re[18 * stride], &stripe_re[19 * stride],
        &stripe_re[20 * stride], &stripe_re[21 * stride], &stripe_re[22 * stride], &stripe_re[23 * stride],
        &stripe_re[24 * stride], &stripe_re[25 * stride], &stripe_re[26 * stride], &stripe_re[27 * stride],
        &stripe_re[28 * stride], &stripe_re[29 * stride], &stripe_re[30 * stride], &stripe_re[31 * stride]};

    double *RESTRICT const out_im[32] = {
        &stripe_im[0 * stride], &stripe_im[1 * stride], &stripe_im[2 * stride], &stripe_im[3 * stride],
        &stripe_im[4 * stride], &stripe_im[5 * stride], &stripe_im[6 * stride], &stripe_im[7 * stride],
        &stripe_im[8 * stride], &stripe_im[9 * stride], &stripe_im[10 * stride], &stripe_im[11 * stride],
        &stripe_im[12 * stride], &stripe_im[13 * stride], &stripe_im[14 * stride], &stripe_im[15 * stride],
        &stripe_im[16 * stride], &stripe_im[17 * stride], &stripe_im[18 * stride], &stripe_im[19 * stride],
        &stripe_im[20 * stride], &stripe_im[21 * stride], &stripe_im[22 * stride], &stripe_im[23 * stride],
        &stripe_im[24 * stride], &stripe_im[25 * stride], &stripe_im[26 * stride], &stripe_im[27 * stride],
        &stripe_im[28 * stride], &stripe_im[29 * stride], &stripe_im[30 * stride], &stripe_im[31 * stride]};

    // Hoist constants
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

    // Extract geometric twiddles
    const __m512d pos1_w1_re = pass2_plan->pos1_w1_re;
    const __m512d pos1_w1_im = pass2_plan->pos1_w1_im;
    const __m512d pos1_w2_re = pass2_plan->pos1_w2_re;
    const __m512d pos1_w2_im = pass2_plan->pos1_w2_im;
    const __m512d pos1_w3_re = pass2_plan->pos1_w3_re;
    const __m512d pos1_w3_im = pass2_plan->pos1_w3_im;

    const __m512d pos2_w1_re = pass2_plan->pos2_w1_re;
    const __m512d pos2_w1_im = pass2_plan->pos2_w1_im;
    const __m512d pos2_w2_re = pass2_plan->pos2_w2_re;
    const __m512d pos2_w2_im = pass2_plan->pos2_w2_im;
    const __m512d pos2_w3_re = pass2_plan->pos2_w3_re;
    const __m512d pos2_w3_im = pass2_plan->pos2_w3_im;

    const __m512d pos3_w1_re = pass2_plan->pos3_w1_re;
    const __m512d pos3_w1_im = pass2_plan->pos3_w1_im;
    const __m512d pos3_w2_re = pass2_plan->pos3_w2_re;
    const __m512d pos3_w2_im = pass2_plan->pos3_w2_im;
    const __m512d pos3_w3_re = pass2_plan->pos3_w3_re;
    const __m512d pos3_w3_im = pass2_plan->pos3_w3_im;

    const __m512d pos5_w1_re = pass2_plan->pos5_w1_re;
    const __m512d pos5_w1_im = pass2_plan->pos5_w1_im;
    const __m512d pos5_w2_re = pass2_plan->pos5_w2_re;
    const __m512d pos5_w2_im = pass2_plan->pos5_w2_im;
    const __m512d pos5_w3_re = pass2_plan->pos5_w3_re;
    const __m512d pos5_w3_im = pass2_plan->pos5_w3_im;

    const __m512d pos6_w1_re = pass2_plan->pos6_w1_re;
    const __m512d pos6_w1_im = pass2_plan->pos6_w1_im;
    const __m512d pos6_w2_re = pass2_plan->pos6_w2_re;
    const __m512d pos6_w2_im = pass2_plan->pos6_w2_im;
    const __m512d pos6_w3_re = pass2_plan->pos6_w3_re;
    const __m512d pos6_w3_im = pass2_plan->pos6_w3_im;

    const __m512d pos7_w1_re = pass2_plan->pos7_w1_re;
    const __m512d pos7_w1_im = pass2_plan->pos7_w1_im;
    const __m512d pos7_w2_re = pass2_plan->pos7_w2_re;
    const __m512d pos7_w2_im = pass2_plan->pos7_w2_im;
    const __m512d pos7_w3_re = pass2_plan->pos7_w3_re;
    const __m512d pos7_w3_im = pass2_plan->pos7_w3_im;

    //==========================================================================
    // MAIN LOOP: Double-buffered (k += 16)
    //==========================================================================

    size_t k = 0;
    const size_t k_main16 = (count / 16) * 16;
    const size_t k_main8 = (count / 8) * 8;

    for (; k < k_main16; k += 16)
    {
        // Adaptive two-level prefetch
        if (k + 64 < k_main16)
        {
            prefetch_group_hoisted_L2(grpA_re, grpA_im, k + 64);
            prefetch_group_hoisted_L2(grpB_re, grpB_im, k + 64);
            prefetch_group_hoisted_L2(grpC_re, grpC_im, k + 64);
            prefetch_group_hoisted_L2(grpD_re, grpD_im, k + 64);
        }

        if (k + 16 < k_main16)
        {
            prefetch_group_hoisted(grpA_re, grpA_im, k + 16);
            prefetch_group_hoisted(grpB_re, grpB_im, k + 16);
            prefetch_group_hoisted(grpC_re, grpC_im, k + 16);
            prefetch_group_hoisted(grpD_re, grpD_im, k + 16);
        }

        // Set 0: k+0..k+7
        __m512d A_re_0[8], A_im_0[8], B_re_0[8], B_im_0[8];
        __m512d C_re_0[8], C_im_0[8], D_re_0[8], D_im_0[8];
        __m512d x_re[8], x_im[8];

        // Set 1: k+8..k+15
        __m512d A_re_1[8], A_im_1[8], B_re_1[8], B_im_1[8];
        __m512d C_re_1[8], C_im_1[8], D_re_1[8], D_im_1[8];

        //======================================================================
        // LOAD AND PROCESS: Set 0 (k+0..k+7)
        //======================================================================

        load_group_hoisted(grpA_re, grpA_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, A_re_0, A_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, B_re_0, B_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, C_re_0, C_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, D_re_0, D_im_0, sign_mask, sqrt2_2);

        //======================================================================
        // LOAD AND PROCESS: Set 1 (k+8..k+15)
        //======================================================================

        load_group_hoisted(grpA_re, grpA_im, k + 8, x_re, x_im);
        process_radix8_group(x_re, x_im, A_re_1, A_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k + 8, x_re, x_im);
        process_radix8_group(x_re, x_im, B_re_1, B_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k + 8, x_re, x_im);
        process_radix8_group(x_re, x_im, C_re_1, C_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k + 8, x_re, x_im);
        process_radix8_group(x_re, x_im, D_re_1, D_im_1, sign_mask, sqrt2_2);

        //======================================================================
        // CROSS-GROUP RADIX-4: Set 0 (k+0..k+7)
        //======================================================================

        radix32_position_identity_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            0, /*pos*/ 0 /*stripe*/, k, sign_mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            1, /*pos*/ 1 /*stripe*/, k, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            2, /*pos*/ 2 /*stripe*/, k, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            3, /*pos*/ 3 /*stripe*/, k, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            /*stripe*/ 4, k, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            5, /*pos*/ 5 /*stripe*/, k, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            6, /*pos*/ 6 /*stripe*/, k, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            7, /*pos*/ 7 /*stripe*/, k, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        //======================================================================
        // CROSS-GROUP RADIX-4: Set 1 (k+8..k+15)
        //======================================================================

        radix32_position_identity_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            0, /*pos*/ 0 /*stripe*/, k + 8, sign_mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            1, /*pos*/ 1 /*stripe*/, k + 8, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            2, /*pos*/ 2 /*stripe*/, k + 8, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            3, /*pos*/ 3 /*stripe*/, k + 8, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            /*stripe*/ 4, k + 8, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            5, /*pos*/ 5 /*stripe*/, k + 8, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            6, /*pos*/ 6 /*stripe*/, k + 8, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            7, /*pos*/ 7 /*stripe*/, k + 8, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
    }
    //==========================================================================
    // CLEANUP: Process remaining 8-butterfly chunks
    //==========================================================================

    for (; k < k_main8; k += 8)
    {
        if (k + 16 < k_main8)
        {
            prefetch_group_hoisted(grpA_re, grpA_im, k + 16);
            prefetch_group_hoisted(grpB_re, grpB_im, k + 16);
            prefetch_group_hoisted(grpC_re, grpC_im, k + 16);
            prefetch_group_hoisted(grpD_re, grpD_im, k + 16);
        }

        __m512d A_re[8], A_im[8], B_re[8], B_im[8];
        __m512d C_re[8], C_im[8], D_re[8], D_im[8];
        __m512d x_re[8], x_im[8];

        load_group_hoisted(grpA_re, grpA_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, A_re, A_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, B_re, B_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, C_re, C_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k, x_re, x_im);
        process_radix8_group(x_re, x_im, D_re, D_im, sign_mask, sqrt2_2);

        // CROSS-GROUP WRITES (hoisted store bases) — FORWARD
        radix32_position_identity_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            0, /*stripe*/ 0, k, sign_mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            1, /*stripe*/ 1, k, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            2, /*stripe*/ 2, k, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            3, /*stripe*/ 3, k, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            /*stripe*/ 4, k, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            5, /*stripe*/ 5, k, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            6, /*stripe*/ 6, k, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            7, /*stripe*/ 7, k, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
    }

    //==========================================================================
    // TAIL: Handle remaining < 8 butterflies
    //==========================================================================

    if (k < count)
    {
        const size_t tail = count - k;
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);

        __m512d A_re[8], A_im[8], B_re[8], B_im[8];
        __m512d C_re[8], C_im[8], D_re[8], D_im[8];
        __m512d x_re[8], x_im[8];

        load_group_hoisted_masked(grpA_re, grpA_im, k, mask, x_re, x_im);
        process_radix8_group(x_re, x_im, A_re, A_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpB_re, grpB_im, k, mask, x_re, x_im);
        process_radix8_group(x_re, x_im, B_re, B_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpC_re, grpC_im, k, mask, x_re, x_im);
        process_radix8_group(x_re, x_im, C_re, C_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpD_re, grpD_im, k, mask, x_re, x_im);
        process_radix8_group(x_re, x_im, D_re, D_im, sign_mask, sqrt2_2);

        // CROSS-GROUP WRITES (masked, hoisted store bases) — FORWARD
        radix32_position_identity_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            0, /*stripe*/ 0, k, sign_mask, mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            1, /*stripe*/ 1, k, sign_mask, mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            2, /*stripe*/ 2, k, sign_mask, mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            3, /*stripe*/ 3, k, sign_mask, mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            /*stripe*/ 4, k, sign_mask, sqrt2_2, mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            5, /*stripe*/ 5, k, sign_mask, mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            6, /*stripe*/ 6, k, sign_mask, mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_forward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            7, /*stripe*/ 7, k, sign_mask, mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
    }
}

//==============================================================================
// BACKWARD-SPECIFIC HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Process one radix-8 group - BACKWARD (conjugated emitters)
 */
TARGET_AVX512
FORCE_INLINE void process_radix8_group_backward(
    const __m512d x_re[8],
    const __m512d x_im[8],
    __m512d out_re[8],
    __m512d out_im[8],
    __m512d sign_mask,
    __m512d sqrt2_2)
{
    // Even wave (pairs 0,4 and 2,6) - BACKWARD
    __m512d t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im;
    radix8_compute_t0123_avx512(
        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],
        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7],
        &t0_re, &t0_im, &t1_re, &t1_im, &t2_re, &t2_im, &t3_re, &t3_im);

    radix8_emit_pair_04_from_t0123_backward_avx512(
        t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
        &out_re[0], &out_im[0], &out_re[1], &out_im[1]);

    radix8_emit_pair_26_from_t0123_backward_avx512(
        t0_re, t0_im, t1_re, t1_im, t2_re, t2_im, t3_re, t3_im,
        &out_re[2], &out_im[2], &out_re[3], &out_im[3],
        sign_mask);

    // Odd wave (pairs 1,5 and 3,7) - BACKWARD
    __m512d t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im;
    radix8_compute_t4567_avx512(
        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],
        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7],
        &t4_re, &t4_im, &t5_re, &t5_im, &t6_re, &t6_im, &t7_re, &t7_im);

    radix8_emit_pair_15_from_t4567_backward_avx512(
        t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
        &out_re[4], &out_im[4], &out_re[5], &out_im[5],
        sign_mask, sqrt2_2);

    radix8_emit_pair_37_from_t4567_backward_avx512(
        t4_re, t4_im, t5_re, t5_im, t6_re, t6_im, t7_re, t7_im,
        &out_re[6], &out_im[6], &out_re[7], &out_im[7],
        sign_mask, sqrt2_2);
}

//==============================================================================
// BACKWARD CROSS-GROUP POSITION FUNCTIONS (hoisted store bases)
//==============================================================================

/**
 * @brief Cross-group position - identity (no twiddles) - BACKWARD
 * Uses radix4_butterfly_core_bv_avx512 and hoisted base arrays.
 */
TARGET_AVX512
FORCE_INLINE void radix32_position_identity_hoisted_backward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

    radix4_butterfly_core_bv_avx512( // BACKWARD
        A_re[pos], A_im[pos], B_re[pos], B_im[pos],
        C_re[pos], C_im[pos], D_re[pos], D_im[pos],
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

/**
 * @brief Cross-group position - identity (masked) - BACKWARD
 */
TARGET_AVX512
FORCE_INLINE void radix32_position_identity_hoisted_backward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask, __mmask8 mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

    radix4_butterfly_core_bv_avx512( // BACKWARD
        A_re[pos], A_im[pos], B_re[pos], B_im[pos],
        C_re[pos], C_im[pos], D_re[pos], D_im[pos],
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

/**
 * @brief Cross-group position - generic twiddles - BACKWARD
 * Twiddle multiplication is on B,C,D then BV core.
 */
TARGET_AVX512
FORCE_INLINE void radix32_position_twiddled_hoisted_backward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask,
    __m512d w1_re, __m512d w1_im,
    __m512d w2_re, __m512d w2_im,
    __m512d w3_re, __m512d w3_im,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[pos], a_im = A_im[pos];
    __m512d b_re = B_re[pos], b_im = B_im[pos];
    __m512d c_re = C_re[pos], c_im = C_im[pos];
    __m512d d_re = D_re[pos], d_im = D_im[pos];

    cmul_avx512(b_re, b_im, w1_re, w1_im, &b_re, &b_im);
    cmul_avx512(c_re, c_im, w2_re, w2_im, &c_re, &c_im);
    cmul_avx512(d_re, d_im, w3_re, w3_im, &d_re, &d_im);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_bv_avx512( // BACKWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

/**
 * @brief Cross-group position - generic twiddles (masked) - BACKWARD
 */
TARGET_AVX512
FORCE_INLINE void radix32_position_twiddled_hoisted_backward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int pos, int stripe, size_t k,
    __m512d sign_mask, __mmask8 mask,
    __m512d w1_re, __m512d w1_im,
    __m512d w2_re, __m512d w2_im,
    __m512d w3_re, __m512d w3_im,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[pos], a_im = A_im[pos];
    __m512d b_re = B_re[pos], b_im = B_im[pos];
    __m512d c_re = C_re[pos], c_im = C_im[pos];
    __m512d d_re = D_re[pos], d_im = D_im[pos];

    cmul_avx512(b_re, b_im, w1_re, w1_im, &b_re, &b_im);
    cmul_avx512(c_re, c_im, w2_re, w2_im, &c_re, &c_im);
    cmul_avx512(d_re, d_im, w3_re, w3_im, &d_re, &d_im);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_bv_avx512( // BACKWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

/**
 * @brief Cross-group position-4 fast path - BACKWARD
 * Implements conjugated W8^1/W8^2/W8^3 and BV core.
 */
TARGET_AVX512
FORCE_INLINE void radix32_position4_fast_hoisted_backward(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int stripe, size_t k,
    __m512d sign_mask, __m512d sqrt2_2,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[4], a_im = A_im[4];
    __m512d b_re = B_re[4], b_im = B_im[4];
    __m512d c_re = C_re[4], c_im = C_im[4];
    __m512d d_re = D_re[4], d_im = D_im[4];

    // W8^1 (conjugated): sqrt(2)/2 * (1 + i)
    __m512d sum_b = _mm512_add_pd(b_re, b_im);
    __m512d nim_b = _mm512_xor_pd(b_im, sign_mask);
    __m512d diff_b = _mm512_add_pd(b_re, nim_b);
    b_re = _mm512_mul_pd(sqrt2_2, sum_b);
    b_im = _mm512_mul_pd(sqrt2_2, diff_b);

    // W8^2 (conjugated): +i
    __m512d c_tw_re, c_tw_im;
    rot_pos_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
    c_re = c_tw_re;
    c_im = c_tw_im;

    // W8^3 (conjugated): sqrt(2)/2 * (-1 + i)
    __m512d sum_d = _mm512_add_pd(d_re, d_im);
    __m512d nim_d = _mm512_xor_pd(d_im, sign_mask);
    __m512d diff_d = _mm512_add_pd(d_re, nim_d);
    __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);
    d_re = _mm512_xor_pd(d_re_tmp, sign_mask);
    d_im = _mm512_mul_pd(sqrt2_2, diff_d);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_bv_avx512( // BACKWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_store_pd(base0_re[stripe] + k, y0_re);
    _mm512_store_pd(base0_im[stripe] + k, y0_im);
    _mm512_store_pd(base8_re[stripe] + k, y1_re);
    _mm512_store_pd(base8_im[stripe] + k, y1_im);
    _mm512_store_pd(base16_re[stripe] + k, y2_re);
    _mm512_store_pd(base16_im[stripe] + k, y2_im);
    _mm512_store_pd(base24_re[stripe] + k, y3_re);
    _mm512_store_pd(base24_im[stripe] + k, y3_im);
}

/**
 * @brief Cross-group position-4 fast path (masked) - BACKWARD
 */
TARGET_AVX512
FORCE_INLINE void radix32_position4_fast_hoisted_backward_masked(
    const __m512d A_re[8], const __m512d A_im[8],
    const __m512d B_re[8], const __m512d B_im[8],
    const __m512d C_re[8], const __m512d C_im[8],
    const __m512d D_re[8], const __m512d D_im[8],
    int stripe, size_t k,
    __m512d sign_mask, __m512d sqrt2_2, __mmask8 mask,
    double *RESTRICT const base0_re[8], double *RESTRICT const base0_im[8],
    double *RESTRICT const base8_re[8], double *RESTRICT const base8_im[8],
    double *RESTRICT const base16_re[8], double *RESTRICT const base16_im[8],
    double *RESTRICT const base24_re[8], double *RESTRICT const base24_im[8])
{
    __m512d a_re = A_re[4], a_im = A_im[4];
    __m512d b_re = B_re[4], b_im = B_im[4];
    __m512d c_re = C_re[4], c_im = C_im[4];
    __m512d d_re = D_re[4], d_im = D_im[4];

    // W8^1 (conjugated)
    __m512d sum_b = _mm512_add_pd(b_re, b_im);
    __m512d nim_b = _mm512_xor_pd(b_im, sign_mask);
    __m512d diff_b = _mm512_add_pd(b_re, nim_b);
    b_re = _mm512_mul_pd(sqrt2_2, sum_b);
    b_im = _mm512_mul_pd(sqrt2_2, diff_b);

    // W8^2 (conjugated): +i
    __m512d c_tw_re, c_tw_im;
    rot_pos_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
    c_re = c_tw_re;
    c_im = c_tw_im;

    // W8^3 (conjugated): sqrt(2)/2 * (-1 + i)
    __m512d sum_d = _mm512_add_pd(d_re, d_im);
    __m512d nim_d = _mm512_xor_pd(d_im, sign_mask);
    __m512d diff_d = _mm512_add_pd(d_re, nim_d);
    __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);
    d_re = _mm512_xor_pd(d_re_tmp, sign_mask);
    d_im = _mm512_mul_pd(sqrt2_2, diff_d);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;
    radix4_butterfly_core_bv_avx512( // BACKWARD
        a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
        sign_mask);

    _mm512_mask_store_pd(base0_re[stripe] + k, mask, y0_re);
    _mm512_mask_store_pd(base0_im[stripe] + k, mask, y0_im);
    _mm512_mask_store_pd(base8_re[stripe] + k, mask, y1_re);
    _mm512_mask_store_pd(base8_im[stripe] + k, mask, y1_im);
    _mm512_mask_store_pd(base16_re[stripe] + k, mask, y2_re);
    _mm512_mask_store_pd(base16_im[stripe] + k, mask, y2_im);
    _mm512_mask_store_pd(base24_re[stripe] + k, mask, y3_re);
    _mm512_mask_store_pd(base24_im[stripe] + k, mask, y3_im);
}

//==============================================================================
// UPDATED BACKWARD BUTTERFLY WITH HOISTED ADDRESSES
//==============================================================================

/**
 * @brief Radix-32 butterfly BACKWARD with hoisted address computation
 *
 * OPTIMIZATIONS:
 * - Hoisted base addresses (eliminates IMUL from hot path)
 * - Double-buffered processing (k += 16)
 * - Adaptive two-level prefetching
 * - Uses backward-specific radix-8 and radix-4 cores
 *
 * @param stripe_re Real part [32 stripes][stride]
 * @param stripe_im Imag part [32 stripes][stride]
 * @param count Number of butterflies (must be multiple of 8)
 * @param stride Stride between samples in same stripe
 * @param pass2_plan Geometric constants (pre-conjugated for backward)
 */
TARGET_AVX512
void radix32_butterfly_strided_backward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t count,
    size_t stride,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    //==========================================================================
    // HOIST BASE ADDRESSES - Eliminate IMUL from hot path
    //==========================================================================

    // Group A (stripes 0-7)
    double *RESTRICT const grpA_re[8] = {
        &stripe_re[0 * stride],
        &stripe_re[1 * stride],
        &stripe_re[2 * stride],
        &stripe_re[3 * stride],
        &stripe_re[4 * stride],
        &stripe_re[5 * stride],
        &stripe_re[6 * stride],
        &stripe_re[7 * stride]};
    double *RESTRICT const grpA_im[8] = {
        &stripe_im[0 * stride],
        &stripe_im[1 * stride],
        &stripe_im[2 * stride],
        &stripe_im[3 * stride],
        &stripe_im[4 * stride],
        &stripe_im[5 * stride],
        &stripe_im[6 * stride],
        &stripe_im[7 * stride]};

    // Group B (stripes 8-15)
    double *RESTRICT const grpB_re[8] = {
        &stripe_re[8 * stride],
        &stripe_re[9 * stride],
        &stripe_re[10 * stride],
        &stripe_re[11 * stride],
        &stripe_re[12 * stride],
        &stripe_re[13 * stride],
        &stripe_re[14 * stride],
        &stripe_re[15 * stride]};
    double *RESTRICT const grpB_im[8] = {
        &stripe_im[8 * stride],
        &stripe_im[9 * stride],
        &stripe_im[10 * stride],
        &stripe_im[11 * stride],
        &stripe_im[12 * stride],
        &stripe_im[13 * stride],
        &stripe_im[14 * stride],
        &stripe_im[15 * stride]};

    // Group C (stripes 16-23)
    double *RESTRICT const grpC_re[8] = {
        &stripe_re[16 * stride],
        &stripe_re[17 * stride],
        &stripe_re[18 * stride],
        &stripe_re[19 * stride],
        &stripe_re[20 * stride],
        &stripe_re[21 * stride],
        &stripe_re[22 * stride],
        &stripe_re[23 * stride]};
    double *RESTRICT const grpC_im[8] = {
        &stripe_im[16 * stride],
        &stripe_im[17 * stride],
        &stripe_im[18 * stride],
        &stripe_im[19 * stride],
        &stripe_im[20 * stride],
        &stripe_im[21 * stride],
        &stripe_im[22 * stride],
        &stripe_im[23 * stride]};

    // Group D (stripes 24-31)
    double *RESTRICT const grpD_re[8] = {
        &stripe_re[24 * stride],
        &stripe_re[25 * stride],
        &stripe_re[26 * stride],
        &stripe_re[27 * stride],
        &stripe_re[28 * stride],
        &stripe_re[29 * stride],
        &stripe_re[30 * stride],
        &stripe_re[31 * stride]};
    double *RESTRICT const grpD_im[8] = {
        &stripe_im[24 * stride],
        &stripe_im[25 * stride],
        &stripe_im[26 * stride],
        &stripe_im[27 * stride],
        &stripe_im[28 * stride],
        &stripe_im[29 * stride],
        &stripe_im[30 * stride],
        &stripe_im[31 * stride]};

    double *RESTRICT const out_re[32] = {
        &stripe_re[0 * stride], &stripe_re[1 * stride], &stripe_re[2 * stride], &stripe_re[3 * stride],
        &stripe_re[4 * stride], &stripe_re[5 * stride], &stripe_re[6 * stride], &stripe_re[7 * stride],
        &stripe_re[8 * stride], &stripe_re[9 * stride], &stripe_re[10 * stride], &stripe_re[11 * stride],
        &stripe_re[12 * stride], &stripe_re[13 * stride], &stripe_re[14 * stride], &stripe_re[15 * stride],
        &stripe_re[16 * stride], &stripe_re[17 * stride], &stripe_re[18 * stride], &stripe_re[19 * stride],
        &stripe_re[20 * stride], &stripe_re[21 * stride], &stripe_re[22 * stride], &stripe_re[23 * stride],
        &stripe_re[24 * stride], &stripe_re[25 * stride], &stripe_re[26 * stride], &stripe_re[27 * stride],
        &stripe_re[28 * stride], &stripe_re[29 * stride], &stripe_re[30 * stride], &stripe_re[31 * stride]};

    double *RESTRICT const out_im[32] = {
        &stripe_im[0 * stride], &stripe_im[1 * stride], &stripe_im[2 * stride], &stripe_im[3 * stride],
        &stripe_im[4 * stride], &stripe_im[5 * stride], &stripe_im[6 * stride], &stripe_im[7 * stride],
        &stripe_im[8 * stride], &stripe_im[9 * stride], &stripe_im[10 * stride], &stripe_im[11 * stride],
        &stripe_im[12 * stride], &stripe_im[13 * stride], &stripe_im[14 * stride], &stripe_im[15 * stride],
        &stripe_im[16 * stride], &stripe_im[17 * stride], &stripe_im[18 * stride], &stripe_im[19 * stride],
        &stripe_im[20 * stride], &stripe_im[21 * stride], &stripe_im[22 * stride], &stripe_im[23 * stride],
        &stripe_im[24 * stride], &stripe_im[25 * stride], &stripe_im[26 * stride], &stripe_im[27 * stride],
        &stripe_im[28 * stride], &stripe_im[29 * stride], &stripe_im[30 * stride], &stripe_im[31 * stride]};

    // Hoist constants
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

    // Extract geometric twiddles (pre-conjugated for backward)
    const __m512d pos1_w1_re = pass2_plan->pos1_w1_re;
    const __m512d pos1_w1_im = pass2_plan->pos1_w1_im;
    const __m512d pos1_w2_re = pass2_plan->pos1_w2_re;
    const __m512d pos1_w2_im = pass2_plan->pos1_w2_im;
    const __m512d pos1_w3_re = pass2_plan->pos1_w3_re;
    const __m512d pos1_w3_im = pass2_plan->pos1_w3_im;

    const __m512d pos2_w1_re = pass2_plan->pos2_w1_re;
    const __m512d pos2_w1_im = pass2_plan->pos2_w1_im;
    const __m512d pos2_w2_re = pass2_plan->pos2_w2_re;
    const __m512d pos2_w2_im = pass2_plan->pos2_w2_im;
    const __m512d pos2_w3_re = pass2_plan->pos2_w3_re;
    const __m512d pos2_w3_im = pass2_plan->pos2_w3_im;

    const __m512d pos3_w1_re = pass2_plan->pos3_w1_re;
    const __m512d pos3_w1_im = pass2_plan->pos3_w1_im;
    const __m512d pos3_w2_re = pass2_plan->pos3_w2_re;
    const __m512d pos3_w2_im = pass2_plan->pos3_w2_im;
    const __m512d pos3_w3_re = pass2_plan->pos3_w3_re;
    const __m512d pos3_w3_im = pass2_plan->pos3_w3_im;

    const __m512d pos5_w1_re = pass2_plan->pos5_w1_re;
    const __m512d pos5_w1_im = pass2_plan->pos5_w1_im;
    const __m512d pos5_w2_re = pass2_plan->pos5_w2_re;
    const __m512d pos5_w2_im = pass2_plan->pos5_w2_im;
    const __m512d pos5_w3_re = pass2_plan->pos5_w3_re;
    const __m512d pos5_w3_im = pass2_plan->pos5_w3_im;

    const __m512d pos6_w1_re = pass2_plan->pos6_w1_re;
    const __m512d pos6_w1_im = pass2_plan->pos6_w1_im;
    const __m512d pos6_w2_re = pass2_plan->pos6_w2_re;
    const __m512d pos6_w2_im = pass2_plan->pos6_w2_im;
    const __m512d pos6_w3_re = pass2_plan->pos6_w3_re;
    const __m512d pos6_w3_im = pass2_plan->pos6_w3_im;

    const __m512d pos7_w1_re = pass2_plan->pos7_w1_re;
    const __m512d pos7_w1_im = pass2_plan->pos7_w1_im;
    const __m512d pos7_w2_re = pass2_plan->pos7_w2_re;
    const __m512d pos7_w2_im = pass2_plan->pos7_w2_im;
    const __m512d pos7_w3_re = pass2_plan->pos7_w3_re;
    const __m512d pos7_w3_im = pass2_plan->pos7_w3_im;

    //==========================================================================
    // MAIN LOOP: Double-buffered (k += 16) - BACKWARD
    //==========================================================================

    size_t k = 0;
    const size_t k_main16 = (count / 16) * 16;
    const size_t k_main8 = (count / 8) * 8;

    for (; k < k_main16; k += 16)
    {
        // Adaptive two-level prefetch
        if (k + 64 < k_main16)
        {
            prefetch_group_hoisted_L2(grpA_re, grpA_im, k + 64);
            prefetch_group_hoisted_L2(grpB_re, grpB_im, k + 64);
            prefetch_group_hoisted_L2(grpC_re, grpC_im, k + 64);
            prefetch_group_hoisted_L2(grpD_re, grpD_im, k + 64);
        }

        if (k + 16 < k_main16)
        {
            prefetch_group_hoisted(grpA_re, grpA_im, k + 16);
            prefetch_group_hoisted(grpB_re, grpB_im, k + 16);
            prefetch_group_hoisted(grpC_re, grpC_im, k + 16);
            prefetch_group_hoisted(grpD_re, grpD_im, k + 16);
        }

        // Set 0: k+0..k+7
        __m512d A_re_0[8], A_im_0[8], B_re_0[8], B_im_0[8];
        __m512d C_re_0[8], C_im_0[8], D_re_0[8], D_im_0[8];
        __m512d x_re[8], x_im[8];

        // Set 1: k+8..k+15
        __m512d A_re_1[8], A_im_1[8], B_re_1[8], B_im_1[8];
        __m512d C_re_1[8], C_im_1[8], D_re_1[8], D_im_1[8];

        //======================================================================
        // LOAD AND PROCESS: Set 0 (k+0..k+7) - BACKWARD
        //======================================================================

        load_group_hoisted(grpA_re, grpA_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, A_re_0, A_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, B_re_0, B_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, C_re_0, C_im_0, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, D_re_0, D_im_0, sign_mask, sqrt2_2);

        //======================================================================
        // LOAD AND PROCESS: Set 1 (k+8..k+15) - BACKWARD
        //======================================================================

        load_group_hoisted(grpA_re, grpA_im, k + 8, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, A_re_1, A_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k + 8, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, B_re_1, B_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k + 8, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, C_re_1, C_im_1, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k + 8, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, D_re_1, D_im_1, sign_mask, sqrt2_2);

        //======================================================================
        // CROSS-GROUP RADIX-4: Set 0 (k+0..k+7) — BACKWARD, hoisted stores
        //======================================================================
        radix32_position_identity_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            0, /*pos*/ 0 /*stripe*/, k, sign_mask,
            /* bases: {0,8,16,24}+stripe */
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            1, /*stripe*/ 1, k, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            2, /*stripe*/ 2, k, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            3, /*stripe*/ 3, k, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            /*stripe*/ 4, k, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            5, /*stripe*/ 5, k, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            6, /*stripe*/ 6, k, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_0, A_im_0, B_re_0, B_im_0, C_re_0, C_im_0, D_re_0, D_im_0,
            7, /*stripe*/ 7, k, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
        //======================================================================
        // CROSS-GROUP RADIX-4: Set 1 (k+8..k+15) — BACKWARD, hoisted stores
        //======================================================================
        radix32_position_identity_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            0, /*stripe*/ 0, k + 8, sign_mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            1, /*stripe*/ 1, k + 8, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            2, /*stripe*/ 2, k + 8, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            3, /*stripe*/ 3, k + 8, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            /*stripe*/ 4, k + 8, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            5, /*stripe*/ 5, k + 8, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            6, /*stripe*/ 6, k + 8, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re_1, A_im_1, B_re_1, B_im_1, C_re_1, C_im_1, D_re_1, D_im_1,
            7, /*stripe*/ 7, k + 8, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
    }

    //==========================================================================
    // CLEANUP: Process remaining 8-butterfly chunks - BACKWARD
    //==========================================================================

    for (; k < k_main8; k += 8)
    {
        if (k + 16 < k_main8)
        {
            prefetch_group_hoisted(grpA_re, grpA_im, k + 16);
            prefetch_group_hoisted(grpB_re, grpB_im, k + 16);
            prefetch_group_hoisted(grpC_re, grpC_im, k + 16);
            prefetch_group_hoisted(grpD_re, grpD_im, k + 16);
        }

        __m512d A_re[8], A_im[8], B_re[8], B_im[8];
        __m512d C_re[8], C_im[8], D_re[8], D_im[8];
        __m512d x_re[8], x_im[8];

        load_group_hoisted(grpA_re, grpA_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, A_re, A_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpB_re, grpB_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, B_re, B_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpC_re, grpC_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, C_re, C_im, sign_mask, sqrt2_2);

        load_group_hoisted(grpD_re, grpD_im, k, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, D_re, D_im, sign_mask, sqrt2_2);

        // CROSS-GROUP WRITES (hoisted store bases) — BACKWARD
        radix32_position_identity_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            0, /*stripe*/ 0, k, sign_mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            1, /*stripe*/ 1, k, sign_mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            2, /*stripe*/ 2, k, sign_mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            3, /*stripe*/ 3, k, sign_mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            /*stripe*/ 4, k, sign_mask, sqrt2_2,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            5, /*stripe*/ 5, k, sign_mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            6, /*stripe*/ 6, k, sign_mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            7, /*stripe*/ 7, k, sign_mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
    }

    //==========================================================================
    // TAIL: Handle remaining < 8 butterflies - BACKWARD
    //==========================================================================

    if (k < count)
    {
        const size_t tail = count - k;
        const __mmask8 mask = (__mmask8)((1u << tail) - 1u);

        __m512d A_re[8], A_im[8], B_re[8], B_im[8];
        __m512d C_re[8], C_im[8], D_re[8], D_im[8];
        __m512d x_re[8], x_im[8];

        load_group_hoisted_masked(grpA_re, grpA_im, k, mask, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, A_re, A_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpB_re, grpB_im, k, mask, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, B_re, B_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpC_re, grpC_im, k, mask, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, C_re, C_im, sign_mask, sqrt2_2);

        load_group_hoisted_masked(grpD_re, grpD_im, k, mask, x_re, x_im);
        process_radix8_group_backward(x_re, x_im, D_re, D_im, sign_mask, sqrt2_2);

        // CROSS-GROUP WRITES (masked, hoisted store bases) — BACKWARD
        radix32_position_identity_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            0, /*stripe*/ 0, k, sign_mask, mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            1, /*stripe*/ 1, k, sign_mask, mask,
            pos1_w1_re, pos1_w1_im, pos1_w2_re, pos1_w2_im, pos1_w3_re, pos1_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            2, /*stripe*/ 2, k, sign_mask, mask,
            pos2_w1_re, pos2_w1_im, pos2_w2_re, pos2_w2_im, pos2_w3_re, pos2_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            3, /*stripe*/ 3, k, sign_mask, mask,
            pos3_w1_re, pos3_w1_im, pos3_w2_re, pos3_w2_im, pos3_w3_re, pos3_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position4_fast_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            /*stripe*/ 4, k, sign_mask, sqrt2_2, mask,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            5, /*stripe*/ 5, k, sign_mask, mask,
            pos5_w1_re, pos5_w1_im, pos5_w2_re, pos5_w2_im, pos5_w3_re, pos5_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            6, /*stripe*/ 6, k, sign_mask, mask,
            pos6_w1_re, pos6_w1_im, pos6_w2_re, pos6_w2_im, pos6_w3_re, pos6_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);

        radix32_position_twiddled_hoisted_backward_masked(
            A_re, A_im, B_re, B_im, C_re, C_im, D_re, D_im,
            7, /*stripe*/ 7, k, sign_mask, mask,
            pos7_w1_re, pos7_w1_im, pos7_w2_re, pos7_w2_im, pos7_w3_re, pos7_w3_im,
            grpA_re, grpA_im, grpB_re, grpB_im, grpC_re, grpC_im, grpD_re, grpD_im);
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
//==============================================================================
// OPTIMIZED TOP-LEVEL WITH ADAPTIVE TILING AND NT STORES
//==============================================================================

/**
 * @brief Complete radix-32 stage - FORWARD
 *
 * PHASE 1 OPTIMIZATIONS:
 * - Adaptive tile size based on K
 * - 4-way stripe unrolling in twiddle application
 * - Two-level prefetching
 *
 * BUG FIXES:
 * - Removed misleading NT store code
 */
TARGET_AVX512
void radix32_stage_forward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t K,
    const double *RESTRICT stage_tw_re,
    const double *RESTRICT stage_tw_im,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    assert((K & 7) == 0 && "K must be multiple of 8");

    // PHASE 1: ADAPTIVE TILE SIZE based on cache hierarchy
    size_t tile_size;
    if (K <= 64)
    {
        // Entire dataset fits in L1
        tile_size = K;
    }
    else if (K <= 512)
    {
        // Target L2 cache
        tile_size = 256;
    }
    else if (K <= 2048)
    {
        // Balance for L3
        tile_size = 128;
    }
    else
    {
        // Large K: smaller tiles
        tile_size = 64;
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        const size_t current_tile = (k_tile + tile_size <= K)
                                        ? tile_size
                                        : (K - k_tile);

        // PHASE 1: Apply stage twiddles (optimized with stripe unrolling)
        if (stage_tw_re != NULL && stage_tw_im != NULL)
        {
            radix32_apply_stage_twiddles_avx512(
                &stripe_re[k_tile],
                &stripe_im[k_tile],
                current_tile,
                K,
                &stage_tw_re[k_tile],
                &stage_tw_im[k_tile]);
        }

        // PHASE 2: Butterfly (with double buffering and corrected position mapping)
        radix32_butterfly_strided_forward_avx512(
            &stripe_re[k_tile],
            &stripe_im[k_tile],
            current_tile,
            K,
            pass2_plan);
    }
}

/**
 * @brief First-stage convenience wrapper (forward)
 */
TARGET_AVX512
FORCE_INLINE void radix32_first_stage_forward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t K,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    radix32_stage_forward_avx512(
        stripe_re, stripe_im, K,
        NULL, NULL,
        pass2_plan);
}

//==============================================================================
// TOP-LEVEL STAGE FUNCTIONS - BACKWARD (CORRECTED)
//==============================================================================

/**
 * @brief Complete radix-32 stage - BACKWARD
 *
 * PHASE 1 OPTIMIZATIONS:
 * - Adaptive tile size based on K
 * - 4-way stripe unrolling in twiddle application
 * - Two-level prefetching
 *
 * BUG FIXES:
 * - Uses backward butterfly and position functions
 * - Removed misleading NT store code
 */
TARGET_AVX512
void radix32_stage_backward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t K,
    const double *RESTRICT stage_tw_re,
    const double *RESTRICT stage_tw_im,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    assert((K & 7) == 0 && "K must be multiple of 8");

    // PHASE 1: ADAPTIVE TILE SIZE
    size_t tile_size;
    if (K <= 64)
    {
        tile_size = K;
    }
    else if (K <= 512)
    {
        tile_size = 256;
    }
    else if (K <= 2048)
    {
        tile_size = 128;
    }
    else
    {
        tile_size = 64;
    }

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        const size_t current_tile = (k_tile + tile_size <= K)
                                        ? tile_size
                                        : (K - k_tile);

        // PHASE 1: Apply stage twiddles (pre-conjugated, uses same function)
        if (stage_tw_re != NULL && stage_tw_im != NULL)
        {
            radix32_apply_stage_twiddles_backward_avx512(
                &stripe_re[k_tile],
                &stripe_im[k_tile],
                current_tile,
                K,
                &stage_tw_re[k_tile],
                &stage_tw_im[k_tile]);
        }

        // PHASE 2: Butterfly (BACKWARD with double buffering and corrected position mapping)
        radix32_butterfly_strided_backward_avx512(
            &stripe_re[k_tile],
            &stripe_im[k_tile],
            current_tile,
            K,
            pass2_plan);
    }
}

/**
 * @brief First-stage convenience wrapper (backward)
 */
TARGET_AVX512
FORCE_INLINE void radix32_first_stage_backward_avx512(
    double *RESTRICT stripe_re,
    double *RESTRICT stripe_im,
    size_t K,
    const radix32_pass2_plan_t *RESTRICT pass2_plan)
{
    radix32_stage_backward_avx512(
        stripe_re, stripe_im, K,
        NULL, NULL,
        pass2_plan);
}
#endif


/*
Critical Math Error: Backward W₈¹ Twiddle
Location: radix32_position4_fast_hoisted_backward() and radix32_position4_fast_hoisted_backward_masked()
The Issue: The real and imaginary parts are swapped when applying the conjugated twiddle W₈¹* = √2/2(1 + i).
Mathematical Proof:
For complex multiplication by W₈¹*:

(b_re + i·b_im) × √2/2(1 + i)
= √2/2 × (b_re + i·b_re + i·b_im - b_im)
= √2/2 × ((b_re - b_im) + i(b_re + b_im))

Expected Result:
b_re' = (b_re - b_im) × √2/2
b_im' = (b_re + b_im) × √2/2
Current (Incorrect) Implementation:

b_re = _mm512_mul_pd(sqrt2_2, sum_b);      // (b_re + b_im) × √2/2  ❌
b_im = _mm512_mul_pd(sqrt2_2, diff_b);     // (b_re - b_im) × √2/2  ❌

The Fix: Swap the assignments:

b_re = _mm512_mul_pd(sqrt2_2, diff_b);     // (b_re - b_im) × √2/2  ✓
b_im = _mm512_mul_pd(sqrt2_2, sum_b);      // (b_re + b_im) × √2/2  ✓
*/