/**
 * @file fft_radix32_avx512_core.h
 * @brief AVX-512 Butterfly Kernels for Radix-32 FFT
 *
 * FOUNDATION LAYER — complex arithmetic + radix-4 DIT + radix-8 DIF cores.
 * No driver, no twiddle management, no fused pass. Just the compute kernels
 * that everything else builds on.
 *
 * KEY DIFFERENCES FROM AVX2:
 *   - ZMM (8 doubles) vs YMM (4 doubles): 2× throughput per iteration
 *   - 32 architectural vector registers vs 16: enables U=2 pipelining
 *   - k-step = 8 (vs 4): halves loop iteration count
 *   - AVX-512DQ provides _mm512_xor_pd for sign negation
 *   - Embedded rounding available but unused (IEEE default is fine)
 *
 * REGISTER BUDGET (unchanged in count from AVX2):
 *   Radix-4 DIT core:  8 in + 4 intermediates = 12 peak
 *   Radix-8 DIF core:  16 in + ~8 intermediates = 16 peak (reuse after stage 1)
 *   cmul:              4 ops (2 MUL + 2 FMA), 0 extra registers
 *
 * TARGET: ICX (Icelake-SP), SPR (Sapphire Rapids), Zen4
 *   All have 2× FMA-512 ports (0,1), 1× shuffle port (5)
 *   Throughput: 2 FMA-512/cycle, 2 ADD-512/cycle
 *
 * @author Tugbars (AVX-512 kernels by Claude)
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX512_CORE_H
#define FFT_RADIX32_AVX512_CORE_H

#include <immintrin.h>

/* Cross-platform macros: TARGET_AVX512, FORCE_INLINE, RESTRICT, etc. */
#include "../fft_radix32_platform.h"

/*==========================================================================
 * SIGN BIT MASK (AVX-512)
 *
 * Used for XOR-based negation: -x = x ^ 0x8000000000000000
 * On ICX this goes through port 0 (integer logic), not the FP pipeline.
 * Single-cycle latency, no FP domain crossing penalty on ICX/SPR.
 *=========================================================================*/

TARGET_AVX512
static inline __m512d signbit_pd_512(void)
{
    const __m512i sb = _mm512_set1_epi64((long long)0x8000000000000000ULL);
    return _mm512_castsi512_pd(sb);
}

/**
 * @def VNEG_PD_512
 * @brief Negate ZMM vector via XOR with signbit
 */
#define VNEG_PD_512(v, signbit) _mm512_xor_pd((v), (signbit))

/*==========================================================================
 * COMPLEX MULTIPLICATION (AVX-512)
 *
 * (ar + i·ai) × (br + i·bi) = (ar·br - ai·bi) + i·(ar·bi + ai·br)
 *
 * Instruction sequence (4 ops, 2 MUL + 2 FMA):
 *   vmulpd    ai_bi  = ai × bi          ; port 0 or 1
 *   vmulpd    ai_br  = ai × br          ; port 0 or 1 (parallel)
 *   vfmsubpd  cr     = ar × br - ai_bi  ; port 0 or 1 (waits for ai_bi)
 *   vfmaddpd  ci     = ar × bi + ai_br  ; port 0 or 1 (waits for ai_br)
 *
 * Critical path: 2 cycles (MUL ∥ MUL → FMA ∥ FMA)
 * On ICX: MUL-512 latency=4, FMA-512 latency=4, throughput=0.5c each
 *
 * Optimization: Issue pure MULs first to break FMA dependency chains.
 * The two MULs are independent → dual-issue on ports 0,1.
 * The two FMAs depend on one MUL each → dual-issue one cycle later.
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void cmul_v512(
    __m512d ar, __m512d ai,
    __m512d br, __m512d bi,
    __m512d *RESTRICT cr, __m512d *RESTRICT ci)
{
    /* Issue independent MULs first (can dual-dispatch) */
    __m512d ai_bi = _mm512_mul_pd(ai, bi);
    __m512d ai_br = _mm512_mul_pd(ai, br);

    /* FMAs wait for their respective MUL result only */
    *cr = _mm512_fmsub_pd(ar, br, ai_bi);  /* ar*br - ai*bi */
    *ci = _mm512_fmadd_pd(ar, bi, ai_br);  /* ar*bi + ai*br */
}

/**
 * @brief Complex multiply with memory-source twiddle (AVX-512)
 *
 * Loads twiddle re/im from memory directly into FMA operands.
 * On ICX/SPR, memory-source FMA folds the load into the µop —
 * no separate load µop, no register consumed for the twiddle.
 *
 * @param xr,xi   Signal (in registers)
 * @param tw_re   Pointer to twiddle real (8 aligned doubles)
 * @param tw_im   Pointer to twiddle imag (8 aligned doubles)
 * @param cr,ci   Result
 */
TARGET_AVX512
static FORCE_INLINE void cmul_mem_v512(
    __m512d xr, __m512d xi,
    const double *RESTRICT tw_re,
    const double *RESTRICT tw_im,
    __m512d *RESTRICT cr, __m512d *RESTRICT ci)
{
    __m512d br = _mm512_load_pd(tw_re);
    __m512d bi = _mm512_load_pd(tw_im);

    __m512d xi_bi = _mm512_mul_pd(xi, bi);
    __m512d xi_br = _mm512_mul_pd(xi, br);

    *cr = _mm512_fmsub_pd(xr, br, xi_bi);
    *ci = _mm512_fmadd_pd(xr, bi, xi_br);
}

/**
 * @brief Complex square (AVX-512)
 *
 * (ar + i·ai)² = (ar² - ai²) + i·(2·ar·ai)
 * 3 ops: 1 MUL + 1 FMA + 1 ADD (vs 4 for general cmul)
 */
TARGET_AVX512
static FORCE_INLINE void csquare_v512(
    __m512d ar, __m512d ai,
    __m512d *RESTRICT cr, __m512d *RESTRICT ci)
{
    __m512d ar_ai = _mm512_mul_pd(ar, ai);
    *cr = _mm512_fmsub_pd(ar, ar, _mm512_mul_pd(ai, ai));
    *ci = _mm512_add_pd(ar_ai, ar_ai);  /* 2·ar·ai */
}

/*==========================================================================
 * RADIX-4 DIT BUTTERFLY CORE (AVX-512)
 *
 * 4-point DFT, decimation in time.
 * Only ±j rotations — no W8 constant needed.
 *
 * Stage 1: t0 = x0+x2, t1 = x0-x2, t2 = x1+x3, t3 = x1-x3
 * Stage 2: y0 = t0+t2, y1 = t1-j·t3, y2 = t0-t2, y3 = t1+j·t3
 *
 * Op count: 16 add/sub (pure ADD/SUB ports, no FMA needed)
 * Register peak: 12 (8 inputs → 4 intermediates → 4 outputs)
 * Latency: 2 stages × 4c = 8 cycles critical path
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void radix4_dit_core_forward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i)
{
    /* Stage 1: even/odd butterfly */
    __m512d t0r = _mm512_add_pd(x0r, x2r);
    __m512d t0i = _mm512_add_pd(x0i, x2i);
    __m512d t1r = _mm512_sub_pd(x0r, x2r);
    __m512d t1i = _mm512_sub_pd(x0i, x2i);

    __m512d t2r = _mm512_add_pd(x1r, x3r);
    __m512d t2i = _mm512_add_pd(x1i, x3i);
    __m512d t3r = _mm512_sub_pd(x1r, x3r);
    __m512d t3i = _mm512_sub_pd(x1i, x3i);

    /* Stage 2: final combination */
    *y0r = _mm512_add_pd(t0r, t2r);
    *y0i = _mm512_add_pd(t0i, t2i);

    /* y1 = t1 - j·t3 = (t1r + t3i) + i·(t1i - t3r) */
    *y1r = _mm512_add_pd(t1r, t3i);
    *y1i = _mm512_sub_pd(t1i, t3r);

    *y2r = _mm512_sub_pd(t0r, t2r);
    *y2i = _mm512_sub_pd(t0i, t2i);

    /* y3 = t1 + j·t3 = (t1r - t3i) + i·(t1i + t3r) */
    *y3r = _mm512_sub_pd(t1r, t3i);
    *y3i = _mm512_add_pd(t1i, t3r);
}

TARGET_AVX512
static FORCE_INLINE void radix4_dit_core_backward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i)
{
    __m512d t0r = _mm512_add_pd(x0r, x2r);
    __m512d t0i = _mm512_add_pd(x0i, x2i);
    __m512d t1r = _mm512_sub_pd(x0r, x2r);
    __m512d t1i = _mm512_sub_pd(x0i, x2i);

    __m512d t2r = _mm512_add_pd(x1r, x3r);
    __m512d t2i = _mm512_add_pd(x1i, x3i);
    __m512d t3r = _mm512_sub_pd(x1r, x3r);
    __m512d t3i = _mm512_sub_pd(x1i, x3i);

    *y0r = _mm512_add_pd(t0r, t2r);
    *y0i = _mm512_add_pd(t0i, t2i);

    /* y1 = t1 + j·t3 (conjugated: -j → +j) */
    *y1r = _mm512_sub_pd(t1r, t3i);
    *y1i = _mm512_add_pd(t1i, t3r);

    *y2r = _mm512_sub_pd(t0r, t2r);
    *y2i = _mm512_sub_pd(t0i, t2i);

    /* y3 = t1 - j·t3 (conjugated: +j → -j) */
    *y3r = _mm512_add_pd(t1r, t3i);
    *y3i = _mm512_sub_pd(t1i, t3r);
}

/*==========================================================================
 * RADIX-8 DIF BUTTERFLY CORE (AVX-512)
 *
 * 8-point DFT, decimation in frequency.
 *
 * Structure (identical math to AVX2, wider datapath):
 *   Stage 1:  4 radix-2 butterflies → a0..a7 (sums/diffs)
 *   Stage 2:  Geometric rotations on diffs (a4..a7)
 *             a4: identity, a5: ×W8, a6: ×(-j), a7: ×W8³
 *   Stage 3:  Two radix-4 DIF butterflies
 *             Evens (a0,a1,a2,a3) → y0,y2,y4,y6
 *             Odds  (a4,b5,b6,b7) → y1,y3,y5,y7
 *
 * Op count:
 *   Stage 1: 16 ADD/SUB
 *   Stage 2: 4 MUL + 2 XOR (W8 rotations) + 1 XOR (-j rotation)
 *   Stage 3: 32 ADD/SUB
 *   Total:   48 ADD/SUB + 4 MUL + 3 XOR = 55 µops
 *
 * Register peak: 16 ZMM
 *   After stage 1: a0..a7 = 16 values (x0..x7 dead)
 *   Stage 2 overwrites a5..a7 in-place → b5,b6,b7
 *   Stage 3 produces outputs, releasing intermediates as it goes
 *
 * On ICX (2× FMA-512 + 2× ADD-512 per cycle):
 *   Stage 1: 16 ops / 2 ports = 8 cycles
 *   Stage 2: ~5 ops (MULs on FMA ports, XORs on port 5)
 *   Stage 3: 32 ops / 2 ports = 16 cycles
 *   Total:   ~29 cycles throughput (pipeline overlaps reduce this)
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void radix8_dif_core_forward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d x4r, __m512d x4i,
    __m512d x5r, __m512d x5i,
    __m512d x6r, __m512d x6i,
    __m512d x7r, __m512d x7i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i,
    __m512d *RESTRICT y4r, __m512d *RESTRICT y4i,
    __m512d *RESTRICT y5r, __m512d *RESTRICT y5i,
    __m512d *RESTRICT y6r, __m512d *RESTRICT y6i,
    __m512d *RESTRICT y7r, __m512d *RESTRICT y7i)
{
    const __m512d W8_C = _mm512_set1_pd(0.70710678118654752440);

    /*======================================================================
     * STAGE 1: Length-4 butterflies (x0±x4, x1±x5, x2±x6, x3±x7)
     *====================================================================*/
    __m512d a0r = _mm512_add_pd(x0r, x4r);
    __m512d a0i = _mm512_add_pd(x0i, x4i);
    __m512d a4r = _mm512_sub_pd(x0r, x4r);
    __m512d a4i = _mm512_sub_pd(x0i, x4i);

    __m512d a1r = _mm512_add_pd(x1r, x5r);
    __m512d a1i = _mm512_add_pd(x1i, x5i);
    __m512d a5r = _mm512_sub_pd(x1r, x5r);
    __m512d a5i = _mm512_sub_pd(x1i, x5i);

    __m512d a2r = _mm512_add_pd(x2r, x6r);
    __m512d a2i = _mm512_add_pd(x2i, x6i);
    __m512d a6r = _mm512_sub_pd(x2r, x6r);
    __m512d a6i = _mm512_sub_pd(x2i, x6i);

    __m512d a3r = _mm512_add_pd(x3r, x7r);
    __m512d a3i = _mm512_add_pd(x3i, x7i);
    __m512d a7r = _mm512_sub_pd(x3r, x7r);
    __m512d a7i = _mm512_sub_pd(x3i, x7i);

    /*======================================================================
     * STAGE 2: Geometric rotations on differences
     *
     * a4: W⁰ = identity (no-op)
     * a5: W⁸¹ = c(1-j)  → Re = c·(r+i), Im = c·(i-r)
     * a6: W⁸² = -j       → Re = i,       Im = -r
     * a7: W⁸³ = -c(1+j)  → Re = c·(i-r), Im = -c·(r+i)
     *====================================================================*/

    /* a5 *= W8 = c(1-j) */
    __m512d b5r = _mm512_mul_pd(W8_C, _mm512_add_pd(a5r, a5i));
    __m512d b5i = _mm512_mul_pd(W8_C, _mm512_sub_pd(a5i, a5r));

    /* a6 *= -j: swap(re,im) then negate new im */
    __m512d b6r = a6i;
    __m512d b6i = _mm512_xor_pd(a6r, signbit_pd_512());

    /* a7 *= W8³ = -c(1+j) */
    __m512d b7r = _mm512_mul_pd(W8_C, _mm512_sub_pd(a7i, a7r));
    __m512d b7i = _mm512_xor_pd(
        _mm512_mul_pd(W8_C, _mm512_add_pd(a7r, a7i)),
        signbit_pd_512());

    /*======================================================================
     * STAGE 3: Two radix-4 DIF butterflies
     *====================================================================*/

    /* --- Evens (a0, a1, a2, a3) → y0, y2, y4, y6 --- */
    __m512d e0r = _mm512_add_pd(a0r, a2r);
    __m512d e0i = _mm512_add_pd(a0i, a2i);
    __m512d e1r = _mm512_sub_pd(a0r, a2r);
    __m512d e1i = _mm512_sub_pd(a0i, a2i);

    __m512d e2r = _mm512_add_pd(a1r, a3r);
    __m512d e2i = _mm512_add_pd(a1i, a3i);
    __m512d e3r = _mm512_sub_pd(a1r, a3r);
    __m512d e3i = _mm512_sub_pd(a1i, a3i);

    *y0r = _mm512_add_pd(e0r, e2r);
    *y0i = _mm512_add_pd(e0i, e2i);

    /* y2 = e1 - j·e3 */
    *y2r = _mm512_add_pd(e1r, e3i);
    *y2i = _mm512_sub_pd(e1i, e3r);

    *y4r = _mm512_sub_pd(e0r, e2r);
    *y4i = _mm512_sub_pd(e0i, e2i);

    /* y6 = e1 + j·e3 */
    *y6r = _mm512_sub_pd(e1r, e3i);
    *y6i = _mm512_add_pd(e1i, e3r);

    /* --- Odds (a4, b5, b6, b7) → y1, y3, y5, y7 --- */
    __m512d o0r = _mm512_add_pd(a4r, b6r);
    __m512d o0i = _mm512_add_pd(a4i, b6i);
    __m512d o1r = _mm512_sub_pd(a4r, b6r);
    __m512d o1i = _mm512_sub_pd(a4i, b6i);

    __m512d o2r = _mm512_add_pd(b5r, b7r);
    __m512d o2i = _mm512_add_pd(b5i, b7i);
    __m512d o3r = _mm512_sub_pd(b5r, b7r);
    __m512d o3i = _mm512_sub_pd(b5i, b7i);

    *y1r = _mm512_add_pd(o0r, o2r);
    *y1i = _mm512_add_pd(o0i, o2i);

    /* y3 = o1 - j·o3 */
    *y3r = _mm512_add_pd(o1r, o3i);
    *y3i = _mm512_sub_pd(o1i, o3r);

    *y5r = _mm512_sub_pd(o0r, o2r);
    *y5i = _mm512_sub_pd(o0i, o2i);

    /* y7 = o1 + j·o3 */
    *y7r = _mm512_sub_pd(o1r, o3i);
    *y7i = _mm512_add_pd(o1i, o3r);
}

/*==========================================================================
 * RADIX-8 DIF BUTTERFLY CORE - BACKWARD (AVX-512)
 *
 * Conjugated geometric rotations:
 *   a5: W8* = c(1+j), a6: +j, a7: c(-1+j)
 * Conjugated ±j in radix-4 sub-butterflies.
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void radix8_dif_core_backward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d x4r, __m512d x4i,
    __m512d x5r, __m512d x5i,
    __m512d x6r, __m512d x6i,
    __m512d x7r, __m512d x7i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i,
    __m512d *RESTRICT y4r, __m512d *RESTRICT y4i,
    __m512d *RESTRICT y5r, __m512d *RESTRICT y5i,
    __m512d *RESTRICT y6r, __m512d *RESTRICT y6i,
    __m512d *RESTRICT y7r, __m512d *RESTRICT y7i)
{
    const __m512d W8_C = _mm512_set1_pd(0.70710678118654752440);

    /* Stage 1: butterflies (identical to forward) */
    __m512d a0r = _mm512_add_pd(x0r, x4r);
    __m512d a0i = _mm512_add_pd(x0i, x4i);
    __m512d a4r = _mm512_sub_pd(x0r, x4r);
    __m512d a4i = _mm512_sub_pd(x0i, x4i);

    __m512d a1r = _mm512_add_pd(x1r, x5r);
    __m512d a1i = _mm512_add_pd(x1i, x5i);
    __m512d a5r = _mm512_sub_pd(x1r, x5r);
    __m512d a5i = _mm512_sub_pd(x1i, x5i);

    __m512d a2r = _mm512_add_pd(x2r, x6r);
    __m512d a2i = _mm512_add_pd(x2i, x6i);
    __m512d a6r = _mm512_sub_pd(x2r, x6r);
    __m512d a6i = _mm512_sub_pd(x2i, x6i);

    __m512d a3r = _mm512_add_pd(x3r, x7r);
    __m512d a3i = _mm512_add_pd(x3i, x7i);
    __m512d a7r = _mm512_sub_pd(x3r, x7r);
    __m512d a7i = _mm512_sub_pd(x3i, x7i);

    /* Stage 2: conjugated rotations */
    /* a5 *= W8* = c(1+j): Re = c·(r-i), Im = c·(r+i) */
    __m512d b5r = _mm512_mul_pd(W8_C, _mm512_sub_pd(a5r, a5i));
    __m512d b5i = _mm512_mul_pd(W8_C, _mm512_add_pd(a5r, a5i));

    /* a6 *= +j: Re = -im, Im = re */
    __m512d b6r = _mm512_xor_pd(a6i, signbit_pd_512());
    __m512d b6i = a6r;

    /* a7 *= c(-1+j): Re = -c·(r+i), Im = c·(r-i) */
    __m512d b7r = _mm512_xor_pd(
        _mm512_mul_pd(W8_C, _mm512_add_pd(a7r, a7i)),
        signbit_pd_512());
    __m512d b7i = _mm512_mul_pd(W8_C, _mm512_sub_pd(a7r, a7i));

    /* Stage 3: radix-4 on evens (conjugated ±j) */
    __m512d e0r = _mm512_add_pd(a0r, a2r);
    __m512d e0i = _mm512_add_pd(a0i, a2i);
    __m512d e1r = _mm512_sub_pd(a0r, a2r);
    __m512d e1i = _mm512_sub_pd(a0i, a2i);

    __m512d e2r = _mm512_add_pd(a1r, a3r);
    __m512d e2i = _mm512_add_pd(a1i, a3i);
    __m512d e3r = _mm512_sub_pd(a1r, a3r);
    __m512d e3i = _mm512_sub_pd(a1i, a3i);

    *y0r = _mm512_add_pd(e0r, e2r);
    *y0i = _mm512_add_pd(e0i, e2i);

    /* y2 = e1 + j·e3 (conjugated) */
    *y2r = _mm512_sub_pd(e1r, e3i);
    *y2i = _mm512_add_pd(e1i, e3r);

    *y4r = _mm512_sub_pd(e0r, e2r);
    *y4i = _mm512_sub_pd(e0i, e2i);

    /* y6 = e1 - j·e3 (conjugated) */
    *y6r = _mm512_add_pd(e1r, e3i);
    *y6i = _mm512_sub_pd(e1i, e3r);

    /* Stage 3: radix-4 on odds (conjugated ±j) */
    __m512d o0r = _mm512_add_pd(a4r, b6r);
    __m512d o0i = _mm512_add_pd(a4i, b6i);
    __m512d o1r = _mm512_sub_pd(a4r, b6r);
    __m512d o1i = _mm512_sub_pd(a4i, b6i);

    __m512d o2r = _mm512_add_pd(b5r, b7r);
    __m512d o2i = _mm512_add_pd(b5i, b7i);
    __m512d o3r = _mm512_sub_pd(b5r, b7r);
    __m512d o3i = _mm512_sub_pd(b5i, b7i);

    *y1r = _mm512_add_pd(o0r, o2r);
    *y1i = _mm512_add_pd(o0i, o2i);

    /* y3 = o1 + j·o3 (conjugated) */
    *y3r = _mm512_sub_pd(o1r, o3i);
    *y3i = _mm512_add_pd(o1i, o3r);

    *y5r = _mm512_sub_pd(o0r, o2r);
    *y5i = _mm512_sub_pd(o0i, o2i);

    /* y7 = o1 - j·o3 (conjugated) */
    *y7r = _mm512_add_pd(o1r, o3i);
    *y7i = _mm512_sub_pd(o1i, o3r);
}

/*==========================================================================
 * TWIDDLE VECTOR CONTAINERS (AVX-512)
 *=========================================================================*/

/**
 * @brief 8 complex twiddle vectors (W1..W8), ZMM width
 *
 * Same logical structure as AVX2 tw8_vecs_t, wider datapath.
 * Used by BLOCKED8/BLOCKED4 loaders and passed to the fused
 * twiddle-apply + butterfly function.
 *
 * Only W1..W7 (indices 0..6) are applied as stage twiddles —
 * x0 is untwidded. W8 (index 7) is used for Parseval validation
 * but not in the main butterfly.
 */
typedef struct
{
    __m512d r[8]; ///< Real parts of W1..W8
    __m512d i[8]; ///< Imag parts of W1..W8
} tw8_vecs_512_t;

/*==========================================================================
 * BLOCKED8 TWIDDLE LOADER (AVX-512, step=8)
 *
 * Loads all 8 twiddle blocks at offset k. Each load pulls 8 doubles
 * (64 bytes = one cache line) into a ZMM register.
 *
 * Total bandwidth: 16 loads × 64B = 1024 bytes per k-step
 * (shared across all 4 DIF-8 bins)
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void load_tw_blocked8_k8(
    const tw_blocked8_t *RESTRICT tw,
    size_t k,
    tw8_vecs_512_t *RESTRICT out)
{
    out->r[0] = _mm512_load_pd(&tw->re[0][k]);
    out->i[0] = _mm512_load_pd(&tw->im[0][k]);
    out->r[1] = _mm512_load_pd(&tw->re[1][k]);
    out->i[1] = _mm512_load_pd(&tw->im[1][k]);
    out->r[2] = _mm512_load_pd(&tw->re[2][k]);
    out->i[2] = _mm512_load_pd(&tw->im[2][k]);
    out->r[3] = _mm512_load_pd(&tw->re[3][k]);
    out->i[3] = _mm512_load_pd(&tw->im[3][k]);
    out->r[4] = _mm512_load_pd(&tw->re[4][k]);
    out->i[4] = _mm512_load_pd(&tw->im[4][k]);
    out->r[5] = _mm512_load_pd(&tw->re[5][k]);
    out->i[5] = _mm512_load_pd(&tw->im[5][k]);
    out->r[6] = _mm512_load_pd(&tw->re[6][k]);
    out->i[6] = _mm512_load_pd(&tw->im[6][k]);
    out->r[7] = _mm512_load_pd(&tw->re[7][k]);
    out->i[7] = _mm512_load_pd(&tw->im[7][k]);
}

/*==========================================================================
 * BLOCKED4 DERIVATION (AVX-512)
 *
 * Given W1..W4 in registers, derive:
 *   W5 = W1 × W4    (3 cmul + 1 csquare = 15 FMA-port ops)
 *   W6 = W2 × W4
 *   W7 = W3 × W4
 *   W8 = W4²
 *
 * This saves 4 loads × 2 (re+im) × 64B = 512 bytes per k-step (43%),
 * at the cost of 15 compute ops. Break-even is K ≈ 128 on ICX where
 * compute/BW ratio favors derivation over memory traffic.
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void derive_w5_to_w8_512(
    __m512d W1r, __m512d W1i,
    __m512d W2r, __m512d W2i,
    __m512d W3r, __m512d W3i,
    __m512d W4r, __m512d W4i,
    __m512d *RESTRICT W5r, __m512d *RESTRICT W5i,
    __m512d *RESTRICT W6r, __m512d *RESTRICT W6i,
    __m512d *RESTRICT W7r, __m512d *RESTRICT W7i,
    __m512d *RESTRICT W8r, __m512d *RESTRICT W8i)
{
    cmul_v512(W1r, W1i, W4r, W4i, W5r, W5i);  /* W5 = W1 × W4 */
    cmul_v512(W2r, W2i, W4r, W4i, W6r, W6i);  /* W6 = W2 × W4 */
    cmul_v512(W3r, W3i, W4r, W4i, W7r, W7i);  /* W7 = W3 × W4 */
    csquare_v512(W4r, W4i, W8r, W8i);          /* W8 = W4²      */
}

/*==========================================================================
 * BLOCKED4 TWIDDLE LOADER (AVX-512, step=8)
 *
 * Loads W1..W4 from memory, derives W5..W8 on-the-fly.
 * Total bandwidth: 8 loads × 64B = 512 bytes per k-step (vs 1024 for B8)
 * Extra compute: 3 cmul + 1 csquare = 15 ops (hidden behind load latency)
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void load_tw_blocked4_k8(
    const tw_blocked4_t *RESTRICT tw,
    size_t k,
    tw8_vecs_512_t *RESTRICT out)
{
    /* Load W1..W4 from memory */
    out->r[0] = _mm512_load_pd(&tw->re[0][k]);
    out->i[0] = _mm512_load_pd(&tw->im[0][k]);
    out->r[1] = _mm512_load_pd(&tw->re[1][k]);
    out->i[1] = _mm512_load_pd(&tw->im[1][k]);
    out->r[2] = _mm512_load_pd(&tw->re[2][k]);
    out->i[2] = _mm512_load_pd(&tw->im[2][k]);
    out->r[3] = _mm512_load_pd(&tw->re[3][k]);
    out->i[3] = _mm512_load_pd(&tw->im[3][k]);

    /* Derive W5..W8 from W1..W4 */
    derive_w5_to_w8_512(
        out->r[0], out->i[0], out->r[1], out->i[1],
        out->r[2], out->i[2], out->r[3], out->i[3],
        &out->r[4], &out->i[4], &out->r[5], &out->i[5],
        &out->r[6], &out->i[6], &out->r[7], &out->i[7]);
}

/*==========================================================================
 * FUSED TWIDDLE-APPLY + DIF-8 BUTTERFLY (AVX-512)
 *
 * This is the "pipeline A" compute block for U=2 pipelining.
 * Takes pre-loaded twiddles W1..W7 (in registers), applies them to
 * inputs x1..x7, then runs the radix-8 DIF core.
 *
 * REGISTER BUDGET:
 *   Inputs:         16 ZMM  (x0r..x7r, x0i..x7i — passed by value)
 *   Twiddles:        passed via pointer to tw_r[]/tw_i[] arrays
 *   cmul scratch:    2 ZMM  (reused across 7 cmul calls)
 *   butterfly:       ~8 ZMM intermediates (reuse input slots after stage 1)
 *   Peak:           16 ZMM  (same as DIF-8 core — twiddle temps overwrite
 *                            inputs that are consumed)
 *
 * The 7 cmuls produce t1..t7 which replace x1..x7 as butterfly inputs.
 * Each cmul needs 2 scratch registers. But x_j is dead after its cmul
 * produces t_j, so the scratch reuses the freed x_j slot. This keeps
 * peak at 16 rather than 16 + 14.
 *
 * TWIDDLE INDEX MAPPING:
 *   tw_r[0]/tw_i[0] = W1  (applied to x1)
 *   tw_r[1]/tw_i[1] = W2  (applied to x2)
 *   ...
 *   tw_r[6]/tw_i[6] = W7  (applied to x7)
 *   x0 passes through untwidded.
 *=========================================================================*/

TARGET_AVX512
static FORCE_INLINE void dif8_twiddle_and_butterfly_forward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d x4r, __m512d x4i,
    __m512d x5r, __m512d x5i,
    __m512d x6r, __m512d x6i,
    __m512d x7r, __m512d x7i,
    const __m512d *RESTRICT tw_r,
    const __m512d *RESTRICT tw_i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i,
    __m512d *RESTRICT y4r, __m512d *RESTRICT y4i,
    __m512d *RESTRICT y5r, __m512d *RESTRICT y5i,
    __m512d *RESTRICT y6r, __m512d *RESTRICT y6i,
    __m512d *RESTRICT y7r, __m512d *RESTRICT y7i)
{
    /* Apply W1..W7 to x1..x7 */
    __m512d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
    __m512d t5r, t5i, t6r, t6i, t7r, t7i;

    cmul_v512(x1r, x1i, tw_r[0], tw_i[0], &t1r, &t1i);
    cmul_v512(x2r, x2i, tw_r[1], tw_i[1], &t2r, &t2i);
    cmul_v512(x3r, x3i, tw_r[2], tw_i[2], &t3r, &t3i);
    cmul_v512(x4r, x4i, tw_r[3], tw_i[3], &t4r, &t4i);
    cmul_v512(x5r, x5i, tw_r[4], tw_i[4], &t5r, &t5i);
    cmul_v512(x6r, x6i, tw_r[5], tw_i[5], &t6r, &t6i);
    cmul_v512(x7r, x7i, tw_r[6], tw_i[6], &t7r, &t7i);

    /* Radix-8 DIF butterfly on twiddled inputs */
    radix8_dif_core_forward_avx512(
        x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
        t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
}

TARGET_AVX512
static FORCE_INLINE void dif8_twiddle_and_butterfly_backward_avx512(
    __m512d x0r, __m512d x0i,
    __m512d x1r, __m512d x1i,
    __m512d x2r, __m512d x2i,
    __m512d x3r, __m512d x3i,
    __m512d x4r, __m512d x4i,
    __m512d x5r, __m512d x5i,
    __m512d x6r, __m512d x6i,
    __m512d x7r, __m512d x7i,
    const __m512d *RESTRICT tw_r,
    const __m512d *RESTRICT tw_i,
    __m512d *RESTRICT y0r, __m512d *RESTRICT y0i,
    __m512d *RESTRICT y1r, __m512d *RESTRICT y1i,
    __m512d *RESTRICT y2r, __m512d *RESTRICT y2i,
    __m512d *RESTRICT y3r, __m512d *RESTRICT y3i,
    __m512d *RESTRICT y4r, __m512d *RESTRICT y4i,
    __m512d *RESTRICT y5r, __m512d *RESTRICT y5i,
    __m512d *RESTRICT y6r, __m512d *RESTRICT y6i,
    __m512d *RESTRICT y7r, __m512d *RESTRICT y7i)
{
    /* Twiddle-apply is identical for fwd/bwd (conjugation is in the core) */
    __m512d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
    __m512d t5r, t5i, t6r, t6i, t7r, t7i;

    cmul_v512(x1r, x1i, tw_r[0], tw_i[0], &t1r, &t1i);
    cmul_v512(x2r, x2i, tw_r[1], tw_i[1], &t2r, &t2i);
    cmul_v512(x3r, x3i, tw_r[2], tw_i[2], &t3r, &t3i);
    cmul_v512(x4r, x4i, tw_r[3], tw_i[3], &t4r, &t4i);
    cmul_v512(x5r, x5i, tw_r[4], tw_i[4], &t5r, &t5i);
    cmul_v512(x6r, x6i, tw_r[5], tw_i[5], &t6r, &t6i);
    cmul_v512(x7r, x7i, tw_r[6], tw_i[6], &t7r, &t7i);

    radix8_dif_core_backward_avx512(
        x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
        t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
        y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
}

/*==========================================================================
 * DIF-8 LOAD/STORE MACROS (AVX-512, step=8)
 *
 * Input layout:  [8 stripes][K] — in_re[stripe * K + k]
 * Output layout: [8 stripes][K] — out_re[stripe * K + k]
 *
 * Two-wave store splits even/odd outputs to keep register pressure
 * at ≤16 ZMM: wave A stores y0,y2,y4,y6 (freeing 8 ZMM), then
 * wave B stores y1,y3,y5,y7.
 *=========================================================================*/

#define DIF8_LOAD_INPUTS_512(in_re, in_im, K, k,                    \
                              x0r, x0i, x1r, x1i, x2r, x2i,        \
                              x3r, x3i, x4r, x4i, x5r, x5i,        \
                              x6r, x6i, x7r, x7i)                   \
    do {                                                             \
        x0r = _mm512_load_pd(&(in_re)[0 * (K) + (k)]);              \
        x0i = _mm512_load_pd(&(in_im)[0 * (K) + (k)]);              \
        x1r = _mm512_load_pd(&(in_re)[1 * (K) + (k)]);              \
        x1i = _mm512_load_pd(&(in_im)[1 * (K) + (k)]);              \
        x2r = _mm512_load_pd(&(in_re)[2 * (K) + (k)]);              \
        x2i = _mm512_load_pd(&(in_im)[2 * (K) + (k)]);              \
        x3r = _mm512_load_pd(&(in_re)[3 * (K) + (k)]);              \
        x3i = _mm512_load_pd(&(in_im)[3 * (K) + (k)]);              \
        x4r = _mm512_load_pd(&(in_re)[4 * (K) + (k)]);              \
        x4i = _mm512_load_pd(&(in_im)[4 * (K) + (k)]);              \
        x5r = _mm512_load_pd(&(in_re)[5 * (K) + (k)]);              \
        x5i = _mm512_load_pd(&(in_im)[5 * (K) + (k)]);              \
        x6r = _mm512_load_pd(&(in_re)[6 * (K) + (k)]);              \
        x6i = _mm512_load_pd(&(in_im)[6 * (K) + (k)]);              \
        x7r = _mm512_load_pd(&(in_re)[7 * (K) + (k)]);              \
        x7i = _mm512_load_pd(&(in_im)[7 * (K) + (k)]);              \
    } while (0)

#define DIF8_STORE_TWO_WAVE_512(ST_FN, out_re, out_im, K, k,        \
                                 y0r, y0i, y1r, y1i, y2r, y2i,      \
                                 y3r, y3i, y4r, y4i, y5r, y5i,      \
                                 y6r, y6i, y7r, y7i)                 \
    do {                                                             \
        /* Wave A: even outputs (frees 8 ZMM) */                     \
        ST_FN(&(out_re)[0 * (K) + (k)], y0r);                       \
        ST_FN(&(out_im)[0 * (K) + (k)], y0i);                       \
        ST_FN(&(out_re)[2 * (K) + (k)], y2r);                       \
        ST_FN(&(out_im)[2 * (K) + (k)], y2i);                       \
        ST_FN(&(out_re)[4 * (K) + (k)], y4r);                       \
        ST_FN(&(out_im)[4 * (K) + (k)], y4i);                       \
        ST_FN(&(out_re)[6 * (K) + (k)], y6r);                       \
        ST_FN(&(out_im)[6 * (K) + (k)], y6i);                       \
        /* Wave B: odd outputs */                                    \
        ST_FN(&(out_re)[1 * (K) + (k)], y1r);                       \
        ST_FN(&(out_im)[1 * (K) + (k)], y1i);                       \
        ST_FN(&(out_re)[3 * (K) + (k)], y3r);                       \
        ST_FN(&(out_im)[3 * (K) + (k)], y3i);                       \
        ST_FN(&(out_re)[5 * (K) + (k)], y5r);                       \
        ST_FN(&(out_im)[5 * (K) + (k)], y5i);                       \
        ST_FN(&(out_re)[7 * (K) + (k)], y7r);                       \
        ST_FN(&(out_im)[7 * (K) + (k)], y7i);                       \
    } while (0)

#endif /* FFT_RADIX32_AVX512_CORE_H */
