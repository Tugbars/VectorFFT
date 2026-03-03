, /**
   * @file fft_radix32_scalar.h
   * @brief Production Radix-32 SCALAR — Fused Single-Pass 4×8 with Micro-Buffer
   *
   * ARCHITECTURE: FUSED SINGLE-PASS 4×8 DECOMPOSITION
   * ==================================================
   * Identical mathematical decomposition to the AVX2 two-pass 4×8, but fused
   * into a single pass using a 512-byte stack micro-buffer (32 complex doubles).
   *
   * Per k-index:
   *   Pass 1: 8 × radix-4 DIT → micro-buffer[32]   (all L1 stack, 0 spills)
   *   Pass 2: 4 × fused DIF-8 ← micro-buffer[32]    (peak 16 regs, 0-2 spills)
   *
   * KEY ADVANTAGES OVER 2×16 ARCHITECTURE:
   *   ✅ No radix-16 butterfly (would force 32+ values onto stack)
   *   ✅ Both sub-passes fit in 16 scalar registers
   *   ✅ Single-pass: no K-dependent temp buffer (temp is 512 bytes, always L1)
   *   ✅ ~128 L1 stack ops/point vs ~350 for 2×16
   *   ✅ Eliminates merge twiddle machinery entirely
   *
   * KEY ADVANTAGES OVER TWO-PASS 4×8:
   *   ✅ No K-dependent temp buffer (avoids L2/LLC for large K)
   *   ✅ Single-pass: 32K input loads + 32K output stores (half the traffic)
   *   ✅ Clean API: no temp_re/temp_im parameters
   *
   * SCALAR DESIGN DECISIONS:
   *   ✅ U=1 loop: OoO engine hides latency, no software pipelining needed
   *   ✅ Memory-operand cmul: twiddles loaded on-demand, zero register overhead
   *   ✅ W8 algebraic trick: 1 constant vs 2-3, lower latency on Zen
   *   ✅ Prefetch throttle: k & 7 (once per 64-byte cache line)
   *   ✅ #pragma GCC unroll 1: prevents register-pressure-destroying unrolling
   *
   * TWIDDLE COMPATIBILITY:
   *   Shares twiddle structures with AVX2 version:
   *   - radix4_dit_stage_twiddles_blocked2_t (Pass 1)
   *   - tw_stage8_t / tw_blocked8_t / tw_blocked4_t (Pass 2)
   *   RECURRENCE mode needs scalar-specific delta (step-1 vs step-4).
   *
   * @author Tugbars (scalar architecture by Claude)
   * @date 2025
   */

#ifndef FFT_RADIX32_SCALAR_H
#define FFT_RADIX32_SCALAR_H

#include <assert.h>
#include <math.h>
#include <stdint.h>

/* Shared types: twiddle structs, tw_mode_t, pick_tw_mode() */
#include "../avx2/fft_radix32_avx2.h"

/*==========================================================================
 * COMPILER ATTRIBUTES
 *=========================================================================*/

#ifndef TARGET_FMA
#define TARGET_FMA __attribute__((target("fma")))
#endif

/* Reuse FORCE_INLINE, RESTRICT, NO_UNROLL_LOOPS, ASSUME_ALIGNED from AVX2 hdr */

/*==========================================================================
 * CONFIGURATION
 *=========================================================================*/

/** Prefetch distance in doubles for scalar loops (8 doubles = 1 cache line) */
#define RADIX32_SCALAR_PREFETCH_DIST 64

/** Tile length for scalar recurrence (in scalar samples, not 4-wide steps) */
#ifndef RADIX32_SCALAR_REC_TILE_LEN
#define RADIX32_SCALAR_REC_TILE_LEN 256
#endif

    /*==========================================================================
     * SCALAR RECURRENCE TWIDDLE STRUCTURE
     *
     * Scalar steps by 1 (vs AVX2's step-by-4), so needs δj¹ instead of δj⁴.
     * Seeds share the same [8][K] layout as AVX2: seed_re[j * K + k_tile_start]
     * gives the exact Wj value at that position, loaded as a single double.
     *
     * Delta relationship:
     *   δj¹ = exp(-2πi·j / (8·K))        (scalar step-1)
     *   δj⁴ = exp(-2πi·j·4 / (8·K))      (AVX2 step-4)
     *   δj⁴ = (δj¹)⁴
     *=========================================================================*/

    typedef struct
{
    int tile_len;          ///< Tile length in scalar samples (e.g. 256)
    const double *seed_re; ///< [8][K] seeds at tile boundaries (shared w/ AVX2)
    const double *seed_im; ///< [8][K] seeds at tile boundaries
    double delta_re[8];    ///< δj¹ per-j step-1 deltas: Wj(k+1) = Wj(k) × δj¹
    double delta_im[8];    ///< δj¹ per-j step-1 deltas
    size_t K;
} tw_recurrence_scalar_t;

/*==========================================================================
 * SCALAR COMPLEX MULTIPLY PRIMITIVES
 *=========================================================================*/

/**
 * @brief Scalar complex multiply using FMA: (ar+i·ai)×(br+i·bi)
 *
 * Compiles to 2 MUL + 2 FMA with -mfma. Throughput: 2 cycles on 2-wide FMA.
 */
TARGET_FMA
static FORCE_INLINE void cmul_s(
    double ar, double ai, double br, double bi,
    double *RESTRICT cr, double *RESTRICT ci)
{
    double ai_bi = ai * bi;
    double ai_br = ai * br;
    *cr = fma(ar, br, -ai_bi);
    *ci = fma(ar, bi, ai_br);
}

/**
 * @brief Memory-operand scalar cmul: loads twiddle from pointer, short live range
 *
 * Compiler can fold the loads into MUL/FMA memory operands, avoiding dedicated
 * twiddle registers entirely.
 */
#define CMUL_MEM_S(ar, ai, wr_ptr, wi_ptr, cr, ci) \
    do                                             \
    {                                              \
        double _wr = *(wr_ptr);                    \
        double _wi = *(wi_ptr);                    \
        double _ai_wi = (ai) * _wi;                \
        double _ai_wr = (ai) * _wr;                \
        (cr) = fma((ar), _wr, -_ai_wi);            \
        (ci) = fma((ar), _wi, _ai_wr);             \
    } while (0)

/**
 * @brief Two-step twiddle for BLOCKED4 derived twiddles (scalar)
 *
 * Computes x × Wj × W4 = x × W(j+4) without materializing W(j+4):
 *   Step 1: tmp = x × Wj (Wj from memory → memory operands)
 *   Step 2: out = tmp × W4 (W4 in register)
 */
#define CMUL_DERIVED_W4_S(xr, xi, wj_re_ptr, wj_im_ptr,  \
                          W4r, W4i, cr, ci)              \
    do                                                   \
    {                                                    \
        double _tr, _ti;                                 \
        CMUL_MEM_S((xr), (xi), (wj_re_ptr), (wj_im_ptr), \
                   _tr, _ti);                            \
        cmul_s(_tr, _ti, (W4r), (W4i), &(cr), &(ci));    \
    } while (0)

/*==========================================================================
 * RADIX-4 DIT CORE — SCALAR (FORWARD)
 *
 * Peak registers: ~12 (8 inputs/outputs + 4 intermediates)
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void radix4_dit_core_fwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i)
{
    /* Stage 1: even/odd butterflies */
    double t0r = x0r + x2r, t0i = x0i + x2i;
    double t1r = x0r - x2r, t1i = x0i - x2i;
    double t2r = x1r + x3r, t2i = x1i + x3i;
    double t3r = x1r - x3r, t3i = x1i - x3i;

    /* Stage 2: final combination */
    *y0r = t0r + t2r;
    *y0i = t0i + t2i; /* y0 = t0 + t2       */
    *y1r = t1r + t3i;
    *y1i = t1i - t3r; /* y1 = t1 - j·t3     */
    *y2r = t0r - t2r;
    *y2i = t0i - t2i; /* y2 = t0 - t2       */
    *y3r = t1r - t3i;
    *y3i = t1i + t3r; /* y3 = t1 + j·t3     */
}

/*==========================================================================
 * RADIX-4 DIT CORE — SCALAR (BACKWARD / IFFT)
 *
 * Conjugated ±j rotations.
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void radix4_dit_core_bwd_scalar(
    double x0r, double x0i, double x1r, double x1i,
    double x2r, double x2i, double x3r, double x3i,
    double *RESTRICT y0r, double *RESTRICT y0i,
    double *RESTRICT y1r, double *RESTRICT y1i,
    double *RESTRICT y2r, double *RESTRICT y2i,
    double *RESTRICT y3r, double *RESTRICT y3i)
{
    double t0r = x0r + x2r, t0i = x0i + x2i;
    double t1r = x0r - x2r, t1i = x0i - x2i;
    double t2r = x1r + x3r, t2i = x1i + x3i;
    double t3r = x1r - x3r, t3i = x1i - x3i;

    *y0r = t0r + t2r;
    *y0i = t0i + t2i; /* y0 = t0 + t2       */
    *y1r = t1r - t3i;
    *y1i = t1i + t3r; /* y1 = t1 + j·t3     */
    *y2r = t0r - t2r;
    *y2i = t0i - t2i; /* y2 = t0 - t2       */
    *y3r = t1r + t3i;
    *y3i = t1i - t3r; /* y3 = t1 - j·t3     */
}

/*==========================================================================
 * FUSED DIF-8: BLOCKED8, FORWARD (from micro-buffer)
 *
 * Reads 8 complex from micro-buffer (flat), writes 8 complex to strided output.
 * Direct scalar translation of AVX2 dif8_fused_fwd_blocked8().
 *
 * Twiddle mapping: tw->re[j][k] = W_{j+1}
 *   x0: untwidded
 *   x1: × W1 = tw[0]    x5: × W5 = tw[4]
 *   x2: × W2 = tw[1]    x6: × W6 = tw[5]
 *   x3: × W3 = tw[2]    x7: × W7 = tw[6]
 *   x4: × W4 = tw[3]
 *
 * Peak register pressure: 16 scalars (at e/d junction)
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_fwd_blocked8_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const tw_blocked8_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    /*==================================================================
     * Phase 1: Paired twiddle-apply + stage-1 sums/diffs
     * Order: (0,4),(2,6) → early DIF-4 → (1,5),(3,7) → complete DIF-4
     *================================================================*/

    /* --- Pair (0,4): x0 untwidded, x4 × W4 = tw[3] --- */
    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    CMUL_MEM_S(tmp_re[4], tmp_im[4],
               &tw->re[3][k], &tw->im[3][k], t4r, t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;
    /* Live: s0,d0 = 4 regs */

    /* --- Pair (2,6): x2 × W2 = tw[1], x6 × W6 = tw[5] --- */
    double t2r, t2i;
    CMUL_MEM_S(tmp_re[2], tmp_im[2],
               &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    double t6r, t6i;
    CMUL_MEM_S(tmp_re[6], tmp_im[6],
               &tw->re[5][k], &tw->im[5][k], t6r, t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;
    /* Live: s0,d0,s2,d2 = 8 regs */

    /* Early even DIF-4 half: e0 = s0+s2, e1 = s0-s2 (kills s0,s2) */
    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;
    /* Live: e0,e1,d0,d2 = 8 regs */

    /* --- Pair (1,5): x1 × W1 = tw[0], x5 × W5 = tw[4] --- */
    double t1r, t1i;
    CMUL_MEM_S(tmp_re[1], tmp_im[1],
               &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    double t5r, t5i;
    CMUL_MEM_S(tmp_re[5], tmp_im[5],
               &tw->re[4][k], &tw->im[4][k], t5r, t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;
    /* Live: e0,e1,d0,d1,d2,s1 = 12 regs */

    /* --- Pair (3,7): x3 × W3 = tw[2], x7 × W7 = tw[6] --- */
    double t3r, t3i;
    CMUL_MEM_S(tmp_re[3], tmp_im[3],
               &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    double t7r, t7i;
    CMUL_MEM_S(tmp_re[7], tmp_im[7],
               &tw->re[6][k], &tw->im[6][k], t7r, t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;
    /* Live: e0,e1,d0,d1,d2,d3,s1,s3 = 16 regs  ← PEAK */

    /* Complete even DIF-4: e2 = s1+s3, e3 = s1-s3 (kills s1,s3) */
    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;
    /* Live: e0,e1,e2,e3,d0,d1,d2,d3 = 16 regs */

    /*==================================================================
     * Wave A: Even outputs y0,y2,y4,y6 — store immediately
     *================================================================*/
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i; /* y0 */
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i; /* y4 */
    /* e0,e2 dead → 12 regs */

    /* y2 = e1 - j·e3 (forward) */
    out_re[2 * K + k] = e1r + e3i;
    out_im[2 * K + k] = e1i - e3r;
    /* y6 = e1 + j·e3 (forward) */
    out_re[6 * K + k] = e1r - e3i;
    out_im[6 * K + k] = e1i + e3r;
    /* e1,e3 dead → Live: d0,d1,d2,d3 = 8 regs */

    /*==================================================================
     * Wave B: W8 rotations on diffs, odd DIF-4 → store
     *================================================================*/
    const double W8_C = 0.70710678118654752440; /* √2/2 */

    /* d0: no rotation (W8^0 = 1) */

    /* d1 *= W8 = c(1-j): Re = c·(r+i), Im = c·(i-r) */
    {
        double sum = d1r + d1i;
        double diff = d1i - d1r;
        d1r = W8_C * sum;
        d1i = W8_C * diff;
    }

    /* d2 *= -j → (im, -re) */
    {
        double tmp = d2r;
        d2r = d2i;
        d2i = -tmp;
    }

    /* d3 *= W8³ = -c(1+j): Re = c·(i-r), Im = -c·(r+i) */
    {
        double sum = d3r + d3i;
        double diff = d3i - d3r;
        d3r = W8_C * diff;
        d3i = -(W8_C * sum);
    }

    /* Odd DIF-4 on {d0,d1,d2,d3} */
    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i; /* y1 */
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i; /* y5 */

    /* y3 = o1 - j·o3 (forward) */
    out_re[3 * K + k] = o1r + o3i;
    out_im[3 * K + k] = o1i - o3r;
    /* y7 = o1 + j·o3 (forward) */
    out_re[7 * K + k] = o1r - o3i;
    out_im[7 * K + k] = o1i + o3r;
}

/*==========================================================================
 * FUSED DIF-8: BLOCKED8, BACKWARD (from micro-buffer)
 *
 * Conjugated W8 rotations and ±j in DIF-4 sub-butterflies.
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_bwd_blocked8_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const tw_blocked8_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    /* Phase 1: identical twiddle-apply + sums/diffs */

    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    CMUL_MEM_S(tmp_re[4], tmp_im[4],
               &tw->re[3][k], &tw->im[3][k], t4r, t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;

    double t2r, t2i;
    CMUL_MEM_S(tmp_re[2], tmp_im[2],
               &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    double t6r, t6i;
    CMUL_MEM_S(tmp_re[6], tmp_im[6],
               &tw->re[5][k], &tw->im[5][k], t6r, t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;

    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;

    double t1r, t1i;
    CMUL_MEM_S(tmp_re[1], tmp_im[1],
               &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    double t5r, t5i;
    CMUL_MEM_S(tmp_re[5], tmp_im[5],
               &tw->re[4][k], &tw->im[4][k], t5r, t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;

    double t3r, t3i;
    CMUL_MEM_S(tmp_re[3], tmp_im[3],
               &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    double t7r, t7i;
    CMUL_MEM_S(tmp_re[7], tmp_im[7],
               &tw->re[6][k], &tw->im[6][k], t7r, t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;

    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;

    /* Wave A: Even outputs (conjugated ±j) */
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i;
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i;

    /* y2 = e1 + j·e3 (backward) */
    out_re[2 * K + k] = e1r - e3i;
    out_im[2 * K + k] = e1i + e3r;
    /* y6 = e1 - j·e3 (backward) */
    out_re[6 * K + k] = e1r + e3i;
    out_im[6 * K + k] = e1i - e3r;

    /* Wave B: Conjugated W8 rotations + odd DIF-4 */
    const double W8_C = 0.70710678118654752440;

    /* d1 *= W8* = c(1+j): Re = c·(r-i), Im = c·(r+i) */
    {
        double diff = d1r - d1i;
        double sum = d1r + d1i;
        d1r = W8_C * diff;
        d1i = W8_C * sum;
    }

    /* d2 *= +j → (-im, re) (backward) */
    {
        double tmp = d2r;
        d2r = -d2i;
        d2i = tmp;
    }

    /* d3 *= c(-1+j): Re = -c·(r+i), Im = c·(r-i) */
    {
        double sum = d3r + d3i;
        double diff = d3r - d3i;
        d3r = -(W8_C * sum);
        d3i = W8_C * diff;
    }

    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i;
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i;

    /* y3 = o1 + j·o3 (backward) */
    out_re[3 * K + k] = o1r - o3i;
    out_im[3 * K + k] = o1i + o3r;
    /* y7 = o1 - j·o3 (backward) */
    out_re[7 * K + k] = o1r + o3i;
    out_im[7 * K + k] = o1i - o3r;
}

/*==========================================================================
 * FUSED DIF-8: BLOCKED4, FORWARD (from micro-buffer)
 *
 * BLOCKED4: W1..W4 in memory, derive W5=W1×W4, W6=W2×W4, W7=W3×W4.
 * W4 kept in register for derivation → +2 regs → peak ~18, expect ~2 spills.
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_fwd_blocked4_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const tw_blocked4_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    /* Load W4 into register — kept alive for W5,W6,W7 derivation */
    double W4r = tw->re[3][k], W4i = tw->im[3][k];

    /* Pair (0,4): x0 untwidded, x4 × W4 (register) */
    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    cmul_s(tmp_re[4], tmp_im[4], W4r, W4i, &t4r, &t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;

    /* Pair (2,6): x2 × W2 (mem), x6 × W6 = (x6×W2)×W4 */
    double t2r, t2i;
    CMUL_MEM_S(tmp_re[2], tmp_im[2],
               &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    double t6r, t6i;
    CMUL_DERIVED_W4_S(tmp_re[6], tmp_im[6],
                      &tw->re[1][k], &tw->im[1][k],
                      W4r, W4i, t6r, t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;

    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;

    /* Pair (1,5): x1 × W1 (mem), x5 × W5 = (x5×W1)×W4 */
    double t1r, t1i;
    CMUL_MEM_S(tmp_re[1], tmp_im[1],
               &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    double t5r, t5i;
    CMUL_DERIVED_W4_S(tmp_re[5], tmp_im[5],
                      &tw->re[0][k], &tw->im[0][k],
                      W4r, W4i, t5r, t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;

    /* Pair (3,7): x3 × W3 (mem), x7 × W7 = (x7×W3)×W4 */
    double t3r, t3i;
    CMUL_MEM_S(tmp_re[3], tmp_im[3],
               &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    double t7r, t7i;
    CMUL_DERIVED_W4_S(tmp_re[7], tmp_im[7],
                      &tw->re[2][k], &tw->im[2][k],
                      W4r, W4i, t7r, t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;

    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;

    /* Wave A: Even outputs */
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i;
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i;
    out_re[2 * K + k] = e1r + e3i;
    out_im[2 * K + k] = e1i - e3r;
    out_re[6 * K + k] = e1r - e3i;
    out_im[6 * K + k] = e1i + e3r;

    /* Wave B: W8 rotations + odd DIF-4 (forward) */
    const double W8_C = 0.70710678118654752440;

    {
        double sum = d1r + d1i;
        double diff = d1i - d1r;
        d1r = W8_C * sum;
        d1i = W8_C * diff;
    }
    {
        double tmp = d2r;
        d2r = d2i;
        d2i = -tmp;
    }
    {
        double sum = d3r + d3i;
        double diff = d3i - d3r;
        d3r = W8_C * diff;
        d3i = -(W8_C * sum);
    }

    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i;
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i;
    out_re[3 * K + k] = o1r + o3i;
    out_im[3 * K + k] = o1i - o3r;
    out_re[7 * K + k] = o1r - o3i;
    out_im[7 * K + k] = o1i + o3r;
}

/*==========================================================================
 * FUSED DIF-8: BLOCKED4, BACKWARD
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_bwd_blocked4_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const tw_blocked4_t *RESTRICT tw,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    double W4r = tw->re[3][k], W4i = tw->im[3][k];

    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    cmul_s(tmp_re[4], tmp_im[4], W4r, W4i, &t4r, &t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;

    double t2r, t2i;
    CMUL_MEM_S(tmp_re[2], tmp_im[2],
               &tw->re[1][k], &tw->im[1][k], t2r, t2i);
    double t6r, t6i;
    CMUL_DERIVED_W4_S(tmp_re[6], tmp_im[6],
                      &tw->re[1][k], &tw->im[1][k],
                      W4r, W4i, t6r, t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;

    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;

    double t1r, t1i;
    CMUL_MEM_S(tmp_re[1], tmp_im[1],
               &tw->re[0][k], &tw->im[0][k], t1r, t1i);
    double t5r, t5i;
    CMUL_DERIVED_W4_S(tmp_re[5], tmp_im[5],
                      &tw->re[0][k], &tw->im[0][k],
                      W4r, W4i, t5r, t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;

    double t3r, t3i;
    CMUL_MEM_S(tmp_re[3], tmp_im[3],
               &tw->re[2][k], &tw->im[2][k], t3r, t3i);
    double t7r, t7i;
    CMUL_DERIVED_W4_S(tmp_re[7], tmp_im[7],
                      &tw->re[2][k], &tw->im[2][k],
                      W4r, W4i, t7r, t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;

    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;

    /* Wave A: Even outputs (backward conjugated ±j) */
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i;
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i;
    out_re[2 * K + k] = e1r - e3i;
    out_im[2 * K + k] = e1i + e3r;
    out_re[6 * K + k] = e1r + e3i;
    out_im[6 * K + k] = e1i - e3r;

    /* Wave B: Conjugated W8 rotations + backward odd DIF-4 */
    const double W8_C = 0.70710678118654752440;

    {
        double diff = d1r - d1i;
        double sum = d1r + d1i;
        d1r = W8_C * diff;
        d1i = W8_C * sum;
    }
    {
        double tmp = d2r;
        d2r = -d2i;
        d2i = tmp;
    }
    {
        double sum = d3r + d3i;
        double diff = d3r - d3i;
        d3r = -(W8_C * sum);
        d3i = W8_C * diff;
    }

    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i;
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i;
    out_re[3 * K + k] = o1r - o3i;
    out_im[3 * K + k] = o1i + o3r;
    out_re[7 * K + k] = o1r + o3i;
    out_im[7 * K + k] = o1i - o3r;
}

/*==========================================================================
 * SCALAR RECURRENCE STATE + INIT/STEP
 *
 * The recurrence state holds W1..W8 (current twiddles) + δ1..δ8 (step-1
 * deltas). At each k-step: Wj(k+1) = Wj(k) × δj¹.
 *
 * Tile boundaries: every tile_len samples, W values are refreshed from
 * precomputed seeds to prevent cumulative numerical drift.
 *=========================================================================*/

typedef struct
{
    double r[8], i[8];   ///< Current W1..W8 values
    double dr[8], di[8]; ///< Per-j δj¹ step deltas (loop-invariant)
} rec8_scalar_state_t;

/**
 * @brief Initialize recurrence state at tile boundary (scalar)
 *
 * Loads all 8 seed values for position k_tile from precomputed table,
 * and loads the per-j step-1 deltas (constant across all tiles).
 */
TARGET_FMA
static FORCE_INLINE void rec8_scalar_tile_init(
    const tw_recurrence_scalar_t *RESTRICT tw,
    size_t k_tile,
    rec8_scalar_state_t *RESTRICT st)
{
    for (int j = 0; j < 8; j++)
    {
        st->r[j] = tw->seed_re[j * tw->K + k_tile];
        st->i[j] = tw->seed_im[j * tw->K + k_tile];
        st->dr[j] = tw->delta_re[j];
        st->di[j] = tw->delta_im[j];
    }
}

/**
 * @brief Advance recurrence by one scalar step: Wj ← Wj × δj¹
 *
 * 8 complex multiplies. Each Wj advances at its own frequency rate.
 */
TARGET_FMA
static FORCE_INLINE void rec8_scalar_step(rec8_scalar_state_t *RESTRICT st)
{
    for (int j = 0; j < 8; j++)
    {
        double nr, ni;
        cmul_s(st->r[j], st->i[j], st->dr[j], st->di[j], &nr, &ni);
        st->r[j] = nr;
        st->i[j] = ni;
    }
}

/*==========================================================================
 * FUSED DIF-8: RECURRENCE, FORWARD (from micro-buffer)
 *
 * Same two-wave butterfly as BLOCKED8, but twiddles come from the
 * recurrence state (8-element arrays) instead of [j][K] memory blocks.
 *
 * Twiddle access: tw_re[j], tw_im[j] — scalar loads, very short live range.
 * Register pressure: identical to BLOCKED8 (16 peak).
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_fwd_rec_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const double *RESTRICT tw_re,
    const double *RESTRICT tw_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    /* Pair (0,4): x0 untwidded, x4 × W4 = tw[3] */
    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    cmul_s(tmp_re[4], tmp_im[4], tw_re[3], tw_im[3], &t4r, &t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;

    /* Pair (2,6): x2 × W2 = tw[1], x6 × W6 = tw[5] */
    double t2r, t2i;
    cmul_s(tmp_re[2], tmp_im[2], tw_re[1], tw_im[1], &t2r, &t2i);
    double t6r, t6i;
    cmul_s(tmp_re[6], tmp_im[6], tw_re[5], tw_im[5], &t6r, &t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;

    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;

    /* Pair (1,5): x1 × W1 = tw[0], x5 × W5 = tw[4] */
    double t1r, t1i;
    cmul_s(tmp_re[1], tmp_im[1], tw_re[0], tw_im[0], &t1r, &t1i);
    double t5r, t5i;
    cmul_s(tmp_re[5], tmp_im[5], tw_re[4], tw_im[4], &t5r, &t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;

    /* Pair (3,7): x3 × W3 = tw[2], x7 × W7 = tw[6] */
    double t3r, t3i;
    cmul_s(tmp_re[3], tmp_im[3], tw_re[2], tw_im[2], &t3r, &t3i);
    double t7r, t7i;
    cmul_s(tmp_re[7], tmp_im[7], tw_re[6], tw_im[6], &t7r, &t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;

    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;

    /* Wave A: Even outputs */
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i;
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i;
    out_re[2 * K + k] = e1r + e3i;
    out_im[2 * K + k] = e1i - e3r;
    out_re[6 * K + k] = e1r - e3i;
    out_im[6 * K + k] = e1i + e3r;

    /* Wave B: W8 rotations + odd DIF-4 (forward) */
    const double W8_C = 0.70710678118654752440;

    {
        double sum = d1r + d1i, diff = d1i - d1r;
        d1r = W8_C * sum;
        d1i = W8_C * diff;
    }
    {
        double tmp = d2r;
        d2r = d2i;
        d2i = -tmp;
    }
    {
        double sum = d3r + d3i, diff = d3i - d3r;
        d3r = W8_C * diff;
        d3i = -(W8_C * sum);
    }

    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i;
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i;
    out_re[3 * K + k] = o1r + o3i;
    out_im[3 * K + k] = o1i - o3r;
    out_re[7 * K + k] = o1r - o3i;
    out_im[7 * K + k] = o1i + o3r;
}

/*==========================================================================
 * FUSED DIF-8: RECURRENCE, BACKWARD
 *=========================================================================*/

TARGET_FMA
static FORCE_INLINE void dif8_fused_bwd_rec_scalar(
    const double *RESTRICT tmp_re,
    const double *RESTRICT tmp_im,
    size_t K, size_t k,
    const double *RESTRICT tw_re,
    const double *RESTRICT tw_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    double x0r = tmp_re[0], x0i = tmp_im[0];
    double t4r, t4i;
    cmul_s(tmp_re[4], tmp_im[4], tw_re[3], tw_im[3], &t4r, &t4i);
    double s0r = x0r + t4r, s0i = x0i + t4i;
    double d0r = x0r - t4r, d0i = x0i - t4i;

    double t2r, t2i;
    cmul_s(tmp_re[2], tmp_im[2], tw_re[1], tw_im[1], &t2r, &t2i);
    double t6r, t6i;
    cmul_s(tmp_re[6], tmp_im[6], tw_re[5], tw_im[5], &t6r, &t6i);
    double s2r = t2r + t6r, s2i = t2i + t6i;
    double d2r = t2r - t6r, d2i = t2i - t6i;

    double e0r = s0r + s2r, e0i = s0i + s2i;
    double e1r = s0r - s2r, e1i = s0i - s2i;

    double t1r, t1i;
    cmul_s(tmp_re[1], tmp_im[1], tw_re[0], tw_im[0], &t1r, &t1i);
    double t5r, t5i;
    cmul_s(tmp_re[5], tmp_im[5], tw_re[4], tw_im[4], &t5r, &t5i);
    double s1r = t1r + t5r, s1i = t1i + t5i;
    double d1r = t1r - t5r, d1i = t1i - t5i;

    double t3r, t3i;
    cmul_s(tmp_re[3], tmp_im[3], tw_re[2], tw_im[2], &t3r, &t3i);
    double t7r, t7i;
    cmul_s(tmp_re[7], tmp_im[7], tw_re[6], tw_im[6], &t7r, &t7i);
    double s3r = t3r + t7r, s3i = t3i + t7i;
    double d3r = t3r - t7r, d3i = t3i - t7i;

    double e2r = s1r + s3r, e2i = s1i + s3i;
    double e3r = s1r - s3r, e3i = s1i - s3i;

    /* Wave A: backward conjugated ±j */
    out_re[0 * K + k] = e0r + e2r;
    out_im[0 * K + k] = e0i + e2i;
    out_re[4 * K + k] = e0r - e2r;
    out_im[4 * K + k] = e0i - e2i;
    out_re[2 * K + k] = e1r - e3i;
    out_im[2 * K + k] = e1i + e3r;
    out_re[6 * K + k] = e1r + e3i;
    out_im[6 * K + k] = e1i - e3r;

    /* Wave B: Conjugated W8 + backward odd DIF-4 */
    const double W8_C = 0.70710678118654752440;

    {
        double diff = d1r - d1i, sum = d1r + d1i;
        d1r = W8_C * diff;
        d1i = W8_C * sum;
    }
    {
        double tmp = d2r;
        d2r = -d2i;
        d2i = tmp;
    }
    {
        double sum = d3r + d3i, diff = d3r - d3i;
        d3r = -(W8_C * sum);
        d3i = W8_C * diff;
    }

    double o0r = d0r + d2r, o0i = d0i + d2i;
    double o1r = d0r - d2r, o1i = d0i - d2i;
    double o2r = d1r + d3r, o2i = d1i + d3i;
    double o3r = d1r - d3r, o3i = d1i - d3i;

    out_re[1 * K + k] = o0r + o2r;
    out_im[1 * K + k] = o0i + o2i;
    out_re[5 * K + k] = o0r - o2r;
    out_im[5 * K + k] = o0i - o2i;
    out_re[3 * K + k] = o1r - o3i;
    out_im[3 * K + k] = o1i + o3r;
    out_re[7 * K + k] = o1r + o3i;
    out_im[7 * K + k] = o1i - o3r;
}

/*==========================================================================
 * FUSED SINGLE-PASS RADIX-32 DRIVER — FORWARD
 *
 * Per k-index:
 *   1. Compute pass-1 twiddles (W1,W2,W3 — shared across all 8 groups)
 *   2. 8 × radix-4 DIT: input stripes → micro-buffer[32]
 *   3. 4 × fused DIF-8: micro-buffer[32] → output stripes
 *
 * Input layout:  [32 stripes][K]  →  in_re[stripe * K + k]
 * Output layout: [32 stripes][K]  →  out_re[stripe * K + k]
 * Micro-buffer:  [4 bins][8 groups] = tmp[bin*8 + group]
 *
 * Group g processes stripes {g, g+8, g+16, g+24}
 * Bin b reads micro-buffer[b*8 + 0..7], writes stripes {b*8+0 .. b*8+7}
 *=========================================================================*/

TARGET_FMA
NO_UNROLL_LOOPS
static void radix32_stage_forward_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw) /* NULL if not RECURRENCE */
{
    assert(K >= 4 && "K must be >= 4");

    const double *p1_re = pass1_tw->re;
    const double *p1_im = pass1_tw->im;

    const tw_mode_t mode = pass2_tw->mode;

    /* Pointers for pass-2 twiddle access (mode-dependent) */
    const tw_blocked8_t *b8 = (mode == TW_MODE_BLOCKED8) ? &pass2_tw->b8 : NULL;
    const tw_blocked4_t *b4 = (mode == TW_MODE_BLOCKED4) ? &pass2_tw->b4 : NULL;

    /* Recurrence state (initialized at first tile boundary inside loop) */
    rec8_scalar_state_t rec_state;
    int rec_tile_len = 0;
    if (mode == TW_MODE_RECURRENCE)
    {
        assert(rec_tw != NULL && "RECURRENCE mode requires rec_tw");
        rec_tile_len = rec_tw->tile_len;
    }

#pragma GCC unroll 1
    for (size_t k = 0; k < K; k++)
    {
        /*==============================================================
         * RECURRENCE: Tile boundary refresh
         *
         * Every tile_len samples, reload W1..W8 from precomputed seeds
         * to prevent cumulative numerical drift. Between refreshes,
         * step via Wj(k+1) = Wj(k) × δj¹.
         *============================================================*/
        if (mode == TW_MODE_RECURRENCE && (k % (size_t)rec_tile_len) == 0)
        {
            rec8_scalar_tile_init(rec_tw, k, &rec_state);

            /* Prefetch next tile seeds */
            if (k + (size_t)rec_tile_len < K && (k & 7) == 0)
            {
                size_t k_next = k + (size_t)rec_tile_len;
                for (int j = 0; j < 8; j++)
                {
                    __builtin_prefetch(&rec_tw->seed_re[j * K + k_next], 0, 3);
                    __builtin_prefetch(&rec_tw->seed_im[j * K + k_next], 0, 3);
                }
            }
        }

        /*==============================================================
         * MICRO-BUFFER: 32 complex values on stack (512 bytes, 8 lines)
         * Layout: tmp[bin*8 + group] where bin=0..3, group=0..7
         *============================================================*/
        double tmp_re[32], tmp_im[32];

        /*==============================================================
         * PASS 1: 8 × radix-4 DIT → micro-buffer
         *============================================================*/
        for (size_t g = 0; g < 8; g++)
        {
            double x0r = in_re[(g + 0) * K + k];
            double x0i = in_im[(g + 0) * K + k];
            double x1r = in_re[(g + 8) * K + k];
            double x1i = in_im[(g + 8) * K + k];
            double x2r = in_re[(g + 16) * K + k];
            double x2i = in_im[(g + 16) * K + k];
            double x3r = in_re[(g + 24) * K + k];
            double x3i = in_im[(g + 24) * K + k];

            /* Apply pass-1 twiddles: x1 *= W1, x2 *= W2, x3 *= W3=W1×W2 */
            {
                double W1r = p1_re[0 * K + k];
                double W1i = p1_im[0 * K + k];
                double W2r = p1_re[1 * K + k];
                double W2i = p1_im[1 * K + k];

                double t1r, t1i;
                cmul_s(x1r, x1i, W1r, W1i, &t1r, &t1i);
                x1r = t1r;
                x1i = t1i;

                double t2r, t2i;
                cmul_s(x2r, x2i, W2r, W2i, &t2r, &t2i);
                x2r = t2r;
                x2i = t2i;

                double W3r, W3i;
                cmul_s(W1r, W1i, W2r, W2i, &W3r, &W3i);

                double t3r, t3i;
                cmul_s(x3r, x3i, W3r, W3i, &t3r, &t3i);
                x3r = t3r;
                x3i = t3i;
            }

            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_dit_core_fwd_scalar(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

            tmp_re[0 * 8 + g] = y0r;
            tmp_im[0 * 8 + g] = y0i;
            tmp_re[1 * 8 + g] = y1r;
            tmp_im[1 * 8 + g] = y1i;
            tmp_re[2 * 8 + g] = y2r;
            tmp_im[2 * 8 + g] = y2i;
            tmp_re[3 * 8 + g] = y3r;
            tmp_im[3 * 8 + g] = y3i;
        }

        /*==============================================================
         * PASS 2: 4 × fused DIF-8 ← micro-buffer → output
         *
         * Register budget:
         *   BLOCKED8:    16 peak (0 spills)
         *   BLOCKED4:    18 peak (~2 spills)
         *   RECURRENCE:  16 peak (0 spills, twiddles from state arrays)
         *============================================================*/
        for (size_t b = 0; b < 4; b++)
        {
            double *bin_out_re = &out_re[b * 8 * K];
            double *bin_out_im = &out_im[b * 8 * K];

            if (mode == TW_MODE_BLOCKED8)
            {
                dif8_fused_fwd_blocked8_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k, b8, bin_out_re, bin_out_im);
            }
            else if (mode == TW_MODE_BLOCKED4)
            {
                dif8_fused_fwd_blocked4_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k, b4, bin_out_re, bin_out_im);
            }
            else /* TW_MODE_RECURRENCE */
            {
                dif8_fused_fwd_rec_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k,
                    rec_state.r, rec_state.i,
                    bin_out_re, bin_out_im);
            }
        }

        /*==============================================================
         * RECURRENCE: Advance Wj(k) → Wj(k+1)
         * Must happen AFTER pass-2 uses the current twiddle values.
         *============================================================*/
        if (mode == TW_MODE_RECURRENCE)
            rec8_scalar_step(&rec_state);

        /*==============================================================
         * PREFETCH: Once per cache line (every 8th k)
         *============================================================*/
        if ((k & 7) == 0)
        {
            const size_t kpf = k + RADIX32_SCALAR_PREFETCH_DIST;
            if (kpf < K)
            {
                for (size_t s = 0; s < 32; s += 8)
                {
                    __builtin_prefetch(&in_re[s * K + kpf], 0, 3);
                    __builtin_prefetch(&in_im[s * K + kpf], 0, 3);
                }

                __builtin_prefetch(&p1_re[0 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_im[0 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_re[1 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_im[1 * K + kpf], 0, 3);

                if (mode == TW_MODE_BLOCKED8)
                {
                    for (int j = 0; j < 7; j++)
                    {
                        __builtin_prefetch(&b8->re[j][kpf], 0, 3);
                        __builtin_prefetch(&b8->im[j][kpf], 0, 3);
                    }
                }
                else if (mode == TW_MODE_BLOCKED4)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        __builtin_prefetch(&b4->re[j][kpf], 0, 3);
                        __builtin_prefetch(&b4->im[j][kpf], 0, 3);
                    }
                }
                /* RECURRENCE: no twiddle prefetch needed — deltas are
                 * loop-invariant scalars, seeds prefetched at tile boundary */
            }
        }
    }
}

/*==========================================================================
 * FUSED SINGLE-PASS RADIX-32 DRIVER — BACKWARD (IFFT)
 *=========================================================================*/

TARGET_FMA
NO_UNROLL_LOOPS
static void radix32_stage_backward_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw) /* NULL if not RECURRENCE */
{
    assert(K >= 4 && "K must be >= 4");

    const double *p1_re = pass1_tw->re;
    const double *p1_im = pass1_tw->im;

    const tw_mode_t mode = pass2_tw->mode;
    const tw_blocked8_t *b8 = (mode == TW_MODE_BLOCKED8) ? &pass2_tw->b8 : NULL;
    const tw_blocked4_t *b4 = (mode == TW_MODE_BLOCKED4) ? &pass2_tw->b4 : NULL;

    rec8_scalar_state_t rec_state;
    int rec_tile_len = 0;
    if (mode == TW_MODE_RECURRENCE)
    {
        assert(rec_tw != NULL && "RECURRENCE mode requires rec_tw");
        rec_tile_len = rec_tw->tile_len;
    }

#pragma GCC unroll 1
    for (size_t k = 0; k < K; k++)
    {
        /* Recurrence: tile boundary refresh */
        if (mode == TW_MODE_RECURRENCE && (k % (size_t)rec_tile_len) == 0)
        {
            rec8_scalar_tile_init(rec_tw, k, &rec_state);
            if (k + (size_t)rec_tile_len < K && (k & 7) == 0)
            {
                size_t k_next = k + (size_t)rec_tile_len;
                for (int j = 0; j < 8; j++)
                {
                    __builtin_prefetch(&rec_tw->seed_re[j * K + k_next], 0, 3);
                    __builtin_prefetch(&rec_tw->seed_im[j * K + k_next], 0, 3);
                }
            }
        }

        double tmp_re[32], tmp_im[32];

        /* PASS 1: 8 × radix-4 DIT BACKWARD → micro-buffer */
        for (size_t g = 0; g < 8; g++)
        {
            double x0r = in_re[(g + 0) * K + k];
            double x0i = in_im[(g + 0) * K + k];
            double x1r = in_re[(g + 8) * K + k];
            double x1i = in_im[(g + 8) * K + k];
            double x2r = in_re[(g + 16) * K + k];
            double x2i = in_im[(g + 16) * K + k];
            double x3r = in_re[(g + 24) * K + k];
            double x3i = in_im[(g + 24) * K + k];

            {
                double W1r = p1_re[0 * K + k];
                double W1i = p1_im[0 * K + k];
                double W2r = p1_re[1 * K + k];
                double W2i = p1_im[1 * K + k];

                double t1r, t1i;
                cmul_s(x1r, x1i, W1r, W1i, &t1r, &t1i);
                x1r = t1r;
                x1i = t1i;

                double t2r, t2i;
                cmul_s(x2r, x2i, W2r, W2i, &t2r, &t2i);
                x2r = t2r;
                x2i = t2i;

                double W3r, W3i;
                cmul_s(W1r, W1i, W2r, W2i, &W3r, &W3i);

                double t3r, t3i;
                cmul_s(x3r, x3i, W3r, W3i, &t3r, &t3i);
                x3r = t3r;
                x3i = t3i;
            }

            double y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_dit_core_bwd_scalar(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);

            tmp_re[0 * 8 + g] = y0r;
            tmp_im[0 * 8 + g] = y0i;
            tmp_re[1 * 8 + g] = y1r;
            tmp_im[1 * 8 + g] = y1i;
            tmp_re[2 * 8 + g] = y2r;
            tmp_im[2 * 8 + g] = y2i;
            tmp_re[3 * 8 + g] = y3r;
            tmp_im[3 * 8 + g] = y3i;
        }

        /* PASS 2: 4 × fused DIF-8 BACKWARD ← micro-buffer → output */
        for (size_t b = 0; b < 4; b++)
        {
            double *bin_out_re = &out_re[b * 8 * K];
            double *bin_out_im = &out_im[b * 8 * K];

            if (mode == TW_MODE_BLOCKED8)
            {
                dif8_fused_bwd_blocked8_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k, b8, bin_out_re, bin_out_im);
            }
            else if (mode == TW_MODE_BLOCKED4)
            {
                dif8_fused_bwd_blocked4_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k, b4, bin_out_re, bin_out_im);
            }
            else /* TW_MODE_RECURRENCE */
            {
                dif8_fused_bwd_rec_scalar(
                    &tmp_re[b * 8], &tmp_im[b * 8],
                    K, k,
                    rec_state.r, rec_state.i,
                    bin_out_re, bin_out_im);
            }
        }

        /* Recurrence: advance after use */
        if (mode == TW_MODE_RECURRENCE)
            rec8_scalar_step(&rec_state);

        /* Prefetch */
        if ((k & 7) == 0)
        {
            const size_t kpf = k + RADIX32_SCALAR_PREFETCH_DIST;
            if (kpf < K)
            {
                for (size_t s = 0; s < 32; s += 8)
                {
                    __builtin_prefetch(&in_re[s * K + kpf], 0, 3);
                    __builtin_prefetch(&in_im[s * K + kpf], 0, 3);
                }
                __builtin_prefetch(&p1_re[0 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_im[0 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_re[1 * K + kpf], 0, 3);
                __builtin_prefetch(&p1_im[1 * K + kpf], 0, 3);

                if (mode == TW_MODE_BLOCKED8)
                {
                    for (int j = 0; j < 7; j++)
                    {
                        __builtin_prefetch(&b8->re[j][kpf], 0, 3);
                        __builtin_prefetch(&b8->im[j][kpf], 0, 3);
                    }
                }
                else if (mode == TW_MODE_BLOCKED4)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        __builtin_prefetch(&b4->re[j][kpf], 0, 3);
                        __builtin_prefetch(&b4->im[j][kpf], 0, 3);
                    }
                }
            }
        }
    }
}

#endif /* FFT_RADIX32_SCALAR_H */