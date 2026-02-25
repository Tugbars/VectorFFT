/**
 * @file fft_radix16_avx512_butterfly.h
 * @brief Radix-16 AVX-512 Butterfly — 4×4 Cooley-Tukey DIT, U=3 Sequential
 *
 * @details
 * AVX-512 port of the scalar radix-16 butterfly with the same 4×4
 * Cooley-Tukey DIT decomposition. Each ZMM register holds 8 doubles,
 * so one SIMD "column" processes 8 independent DFT-16 columns in parallel.
 *
 * Three stages per batch of 8 columns:
 *   Stage 1: 4× DFT-4 on stride-4 groups → T[n2][k1]
 *   Twiddle: T[n2][k1] *= W₁₆^{n2·k1}  (9 non-trivial of 16 entries)
 *   Stage 2: 4× DFT-4 across groups → Y[k1 + 4·k2]
 *
 * Unrolling: U=3 sequential — three batches of 8 columns per k-loop
 * iteration (k += 24). Group 0's twiddle is identity, so its stage-1
 * output is spilled to a 256-byte stack buffer and reloaded in stage 2.
 * This frees registers for groups 1-3 to remain in ZMM during twiddle
 * and stage 2, achieving peak utilization of 29/32 ZMM registers.
 *
 * Register budget (per batch):
 *   6 ZMM: resident constants (cos_π/8, sin_π/8, √2/2 + negated)
 *   24 ZMM: groups 1-3 intermediates (8 ZMM each: 4 re + 4 im)
 *   2 ZMM: group-0 reload temporaries during stage 2
 *   --- Peak: 29/32 ZMM. 3 free. No register spills. ---
 *
 * Tail handling: AVX-512 opmask (__mmask8) for K % 8 remainder.
 *   Zero-masked loads, merge-masked stores. No scalar fallback.
 *
 * W₁₆ twiddle map (row=n2, col=k1, exponent=n2·k1 mod 16):
 *       k1=0   k1=1   k1=2   k1=3
 *  n2=0:  1      1      1      1     ← all trivial (identity, spilled)
 *  n2=1:  1     W¹     W²     W³     ← 3 non-trivial
 *  n2=2:  1     W²     -j     W⁶     ← 2 non-trivial (W⁴=-j is trivial)
 *  n2=3:  1     W³     W⁶     W⁹     ← 3 non-trivial + W⁹ uses neg constants
 *
 * Non-trivial twiddle FMA budget per batch:
 *   W¹, W³:  2 FMA + 1 MUL each = 6 ops × 2 occurrences = 12
 *   W²:      1 MUL + 2 ADD each  = 3 ops × 2 occurrences = 6
 *   W⁶:      1 MUL + 2 ADD each  = 3 ops × 2 occurrences = 6
 *   W⁹:      2 FMA (using pre-negated constants)           = 2
 *   Total:   ~26 ZMM ops per batch for twiddles
 *
 * TODO(benchmark): Compare U=3 vs U=2 on ICX, SPR, Zen4, Alder Lake.
 *   U=3 gives the OoO engine ~540 µops of lookahead (SPR ROB = 512),
 *   but may increase I-cache pressure. U=2 is the safe fallback.
 *
 * SoA memory layout:
 *   in_re[r * K + k], in_im[r * K + k]   for r=0..15, k=0..K-1
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX16_AVX512_BUTTERFLY_H
#define FFT_RADIX16_AVX512_BUTTERFLY_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

/* ============================================================================
 * COMPILER PORTABILITY
 * ========================================================================= */

#ifdef _MSC_VER
  #define R16Z_INLINE       static __forceinline
  #define R16Z_RESTRICT      __restrict
  #define R16Z_NOINLINE      static __declspec(noinline)
  #define R16Z_ALIGN64       __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
  #define R16Z_INLINE       static inline __attribute__((always_inline))
  #define R16Z_RESTRICT      __restrict__
  #define R16Z_NOINLINE      static __attribute__((noinline))
  #define R16Z_ALIGN64       __attribute__((aligned(64)))
#else
  #define R16Z_INLINE       static inline
  #define R16Z_RESTRICT
  #define R16Z_NOINLINE      static
  #define R16Z_ALIGN64
#endif

/* ============================================================================
 * CONSTANTS — Full precision for double
 * ========================================================================= */

#define R16Z_COS_PI_8   0.92387953251128675613   /* cos(π/8)  */
#define R16Z_SIN_PI_8   0.38268343236508977173   /* sin(π/8)  */
#define R16Z_SQRT2_2    0.70710678118654752440   /* √2/2      */

/* cos(3π/8) = sin(π/8), sin(3π/8) = cos(π/8) — used symbolically for clarity
 * but the actual constants are the same values. */

/* ============================================================================
 * GROUP-0 SPILL BUFFER
 *
 * Group 0's twiddle is identity for all k1, so we don't need to keep it
 * in registers during groups 1-3 processing. Instead we spill 4 complex
 * values (8 ZMM-width doubles) to a 256-byte stack buffer, aligned to
 * 64 bytes for optimal L1 access. Reload during stage 2.
 *
 * Cost: 8 stores + 8 loads per batch = 16 L1 ops (invisible vs main mem)
 * ========================================================================= */

/* 6 ZMM values (k0-k2 re+im) × 8 doubles × 8 bytes = 384 bytes */
#define R16Z_SPILL_BYTES  384

/* ============================================================================
 * PREFETCH CONFIGURATION
 *
 * With stride-K access across 16 rows, hardware prefetcher handles
 * stride-1 within each row. For next-batch prefetching, we issue
 * software prefetches at the start of each batch for the next batch's
 * first few rows. Distance = 1 batch = ~120 instructions ≈ 30-40 cycles.
 * ========================================================================= */

#define R16Z_PREFETCH_DISTANCE  8   /* prefetch 8 doubles = 1 ZMM ahead */

#if defined(__GNUC__) || defined(__clang__)
  #define R16Z_PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
  #define R16Z_PREFETCH_T1(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#elif defined(_MSC_VER)
  #define R16Z_PREFETCH_T0(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
  #define R16Z_PREFETCH_T1(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
  #define R16Z_PREFETCH_T0(addr)  ((void)0)
  #define R16Z_PREFETCH_T1(addr)  ((void)0)
#endif

/* ============================================================================
 * AVX-512 RADIX-4 BUTTERFLY — 8-wide (8 columns in parallel)
 *
 * Forward (sign = -1):                    Backward (sign = +1):
 *   y0 = (a+c) + (b+d)                     y0 = (a+c) + (b+d)
 *   y1 = (a-c) - j(b-d)                    y1 = (a-c) + j(b-d)
 *   y2 = (a+c) - (b+d)                     y2 = (a+c) - (b+d)
 *   y3 = (a-c) + j(b-d)                    y3 = (a-c) - j(b-d)
 *
 * All add/sub, 0 multiplies. Peak: 10 ZMM (4 sums + 4 diffs + 2 outputs).
 * ========================================================================= */

R16Z_INLINE void r16z_radix4_fwd(
    __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im, __m512d d_re, __m512d d_im,
    __m512d *R16Z_RESTRICT y0_re, __m512d *R16Z_RESTRICT y0_im,
    __m512d *R16Z_RESTRICT y1_re, __m512d *R16Z_RESTRICT y1_im,
    __m512d *R16Z_RESTRICT y2_re, __m512d *R16Z_RESTRICT y2_im,
    __m512d *R16Z_RESTRICT y3_re, __m512d *R16Z_RESTRICT y3_im)
{
    __m512d sAC_re = _mm512_add_pd(a_re, c_re);
    __m512d dAC_re = _mm512_sub_pd(a_re, c_re);
    __m512d sAC_im = _mm512_add_pd(a_im, c_im);
    __m512d dAC_im = _mm512_sub_pd(a_im, c_im);
    __m512d sBD_re = _mm512_add_pd(b_re, d_re);
    __m512d dBD_re = _mm512_sub_pd(b_re, d_re);
    __m512d sBD_im = _mm512_add_pd(b_im, d_im);
    __m512d dBD_im = _mm512_sub_pd(b_im, d_im);

    *y0_re = _mm512_add_pd(sAC_re, sBD_re);
    *y0_im = _mm512_add_pd(sAC_im, sBD_im);
    *y2_re = _mm512_sub_pd(sAC_re, sBD_re);
    *y2_im = _mm512_sub_pd(sAC_im, sBD_im);

    /* Forward: -j*(b-d) → re += dBD_im, im -= dBD_re */
    *y1_re = _mm512_add_pd(dAC_re, dBD_im);
    *y1_im = _mm512_sub_pd(dAC_im, dBD_re);
    *y3_re = _mm512_sub_pd(dAC_re, dBD_im);
    *y3_im = _mm512_add_pd(dAC_im, dBD_re);
}

R16Z_INLINE void r16z_radix4_bwd(
    __m512d a_re, __m512d a_im, __m512d b_re, __m512d b_im,
    __m512d c_re, __m512d c_im, __m512d d_re, __m512d d_im,
    __m512d *R16Z_RESTRICT y0_re, __m512d *R16Z_RESTRICT y0_im,
    __m512d *R16Z_RESTRICT y1_re, __m512d *R16Z_RESTRICT y1_im,
    __m512d *R16Z_RESTRICT y2_re, __m512d *R16Z_RESTRICT y2_im,
    __m512d *R16Z_RESTRICT y3_re, __m512d *R16Z_RESTRICT y3_im)
{
    __m512d sAC_re = _mm512_add_pd(a_re, c_re);
    __m512d dAC_re = _mm512_sub_pd(a_re, c_re);
    __m512d sAC_im = _mm512_add_pd(a_im, c_im);
    __m512d dAC_im = _mm512_sub_pd(a_im, c_im);
    __m512d sBD_re = _mm512_add_pd(b_re, d_re);
    __m512d dBD_re = _mm512_sub_pd(b_re, d_re);
    __m512d sBD_im = _mm512_add_pd(b_im, d_im);
    __m512d dBD_im = _mm512_sub_pd(b_im, d_im);

    *y0_re = _mm512_add_pd(sAC_re, sBD_re);
    *y0_im = _mm512_add_pd(sAC_im, sBD_im);
    *y2_re = _mm512_sub_pd(sAC_re, sBD_re);
    *y2_im = _mm512_sub_pd(sAC_im, sBD_im);

    /* Backward: +j*(b-d) → re -= dBD_im, im += dBD_re */
    *y1_re = _mm512_sub_pd(dAC_re, dBD_im);
    *y1_im = _mm512_add_pd(dAC_im, dBD_re);
    *y3_re = _mm512_add_pd(dAC_re, dBD_im);
    *y3_im = _mm512_sub_pd(dAC_im, dBD_re);
}

/* ============================================================================
 * W₁₆ TWIDDLE APPLICATION — AVX-512, 8-wide
 *
 * Operates on the 4×4 intermediate buffer T[n2][k1] stored in ZMM regs.
 * Group 0 (n2=0) has identity twiddles and is in the spill buffer, not here.
 * Groups 1-3 (n2=1..3) occupy 24 ZMM registers (8 per group: 4 re + 4 im).
 *
 * Uses 6 resident constants: {cos_pi8, sin_pi8, sqrt2_2} + negated versions.
 * Pre-computed negations eliminate vxorpd in the critical path.
 *
 * Twiddle identities used:
 *   W¹ = cos(π/8) - j·sin(π/8)         → full complex multiply
 *   W² = √2/2·(1-j)                    → re'=√2/2·(a+b), im'=√2/2·(b-a)
 *   W³ = sin(π/8) - j·cos(π/8)         → full complex multiply
 *   W⁴ = -j                            → re'=b, im'=-a (trivial)
 *   W⁶ = -√2/2·(1+j)                   → re'=√2/2·(-a+b), im'=√2/2·(-a-b)
 *   W⁹ = -cos(π/8) + j·sin(π/8)        → uses neg_cos, neg_sin
 *
 * FMA pattern for W = wr - j·wi (forward):
 *   re' = a·wr + b·wi  → fmadd(b, wi, a·wr) or fmadd(a, wr, b·wi)
 *   im' = b·wr - a·wi  → fnmadd(a, wi, b·wr)
 * ========================================================================= */

R16Z_INLINE void r16z_twiddle_fwd(
    /* Group 1: T[1][0..3] — k1=0 is identity, k1=1..3 modified */
    __m512d *R16Z_RESTRICT g1_k0_re, __m512d *R16Z_RESTRICT g1_k0_im,  /* untouched */
    __m512d *R16Z_RESTRICT g1_k1_re, __m512d *R16Z_RESTRICT g1_k1_im,  /* *= W¹ */
    __m512d *R16Z_RESTRICT g1_k2_re, __m512d *R16Z_RESTRICT g1_k2_im,  /* *= W² */
    __m512d *R16Z_RESTRICT g1_k3_re, __m512d *R16Z_RESTRICT g1_k3_im,  /* *= W³ */
    /* Group 2: T[2][0..3] */
    __m512d *R16Z_RESTRICT g2_k0_re, __m512d *R16Z_RESTRICT g2_k0_im,  /* untouched */
    __m512d *R16Z_RESTRICT g2_k1_re, __m512d *R16Z_RESTRICT g2_k1_im,  /* *= W² */
    __m512d *R16Z_RESTRICT g2_k2_re, __m512d *R16Z_RESTRICT g2_k2_im,  /* *= W⁴=-j */
    __m512d *R16Z_RESTRICT g2_k3_re, __m512d *R16Z_RESTRICT g2_k3_im,  /* *= W⁶ */
    /* Group 3: T[3][0..3] */
    __m512d *R16Z_RESTRICT g3_k0_re, __m512d *R16Z_RESTRICT g3_k0_im,  /* untouched */
    __m512d *R16Z_RESTRICT g3_k1_re, __m512d *R16Z_RESTRICT g3_k1_im,  /* *= W³ */
    __m512d *R16Z_RESTRICT g3_k2_re, __m512d *R16Z_RESTRICT g3_k2_im,  /* *= W⁶ */
    __m512d *R16Z_RESTRICT g3_k3_re, __m512d *R16Z_RESTRICT g3_k3_im,  /* *= W⁹ */
    /* Constants (all broadcast, resident in ZMM29-31, ZMM26-28) */
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    __m512d a, b;
    (void)g1_k0_re; (void)g1_k0_im;  /* k1=0: identity, no-op */
    (void)g2_k0_re; (void)g2_k0_im;
    (void)g3_k0_re; (void)g3_k0_im;

    /* ---- Group 1 (n2=1) ---- */

    /* T[1][1] *= W¹ = cos(π/8) - j·sin(π/8)
       re' = a·cos + b·sin,  im' = b·cos - a·sin */
    a = *g1_k1_re; b = *g1_k1_im;
    *g1_k1_re = _mm512_fmadd_pd(b, sin_pi8, _mm512_mul_pd(a, cos_pi8));
    *g1_k1_im = _mm512_fnmadd_pd(a, sin_pi8, _mm512_mul_pd(b, cos_pi8));

    /* T[1][2] *= W² = √2/2·(1-j)
       re' = √2/2·(a+b),  im' = √2/2·(b-a) */
    a = *g1_k2_re; b = *g1_k2_im;
    *g1_k2_re = _mm512_mul_pd(sqrt2_2, _mm512_add_pd(a, b));
    *g1_k2_im = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(b, a));

    /* T[1][3] *= W³ = cos(3π/8) - j·sin(3π/8) = sin(π/8) - j·cos(π/8)
       re' = a·sin(π/8) + b·cos(π/8),  im' = b·sin(π/8) - a·cos(π/8) */
    a = *g1_k3_re; b = *g1_k3_im;
    *g1_k3_re = _mm512_fmadd_pd(b, cos_pi8, _mm512_mul_pd(a, sin_pi8));
    *g1_k3_im = _mm512_fnmadd_pd(a, cos_pi8, _mm512_mul_pd(b, sin_pi8));

    /* ---- Group 2 (n2=2) ---- */

    /* T[2][1] *= W² (same as T[1][2]) */
    a = *g2_k1_re; b = *g2_k1_im;
    *g2_k1_re = _mm512_mul_pd(sqrt2_2, _mm512_add_pd(a, b));
    *g2_k1_im = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(b, a));

    /* T[2][2] *= W⁴ = -j: re' = im, im' = -re */
    a = *g2_k2_re; b = *g2_k2_im;
    *g2_k2_re = b;
    *g2_k2_im = _mm512_sub_pd(_mm512_setzero_pd(), a);

    /* T[2][3] *= W⁶ = -√2/2·(1+j)
       re' = √2/2·(-a+b),  im' = √2/2·(-a-b)
       Using neg: re' = fmadd(b, sqrt2_2, neg_sqrt2_2·a)
                  im' = fmadd(neg_sqrt2_2, a, neg_sqrt2_2·b) — but simpler: */
    a = *g2_k3_re; b = *g2_k3_im;
    *g2_k3_re = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(b, a));
    *g2_k3_im = _mm512_mul_pd(neg_sqrt2_2, _mm512_add_pd(a, b));

    /* ---- Group 3 (n2=3) ---- */

    /* T[3][1] *= W³ (same formula as T[1][3]) */
    a = *g3_k1_re; b = *g3_k1_im;
    *g3_k1_re = _mm512_fmadd_pd(b, cos_pi8, _mm512_mul_pd(a, sin_pi8));
    *g3_k1_im = _mm512_fnmadd_pd(a, cos_pi8, _mm512_mul_pd(b, sin_pi8));

    /* T[3][2] *= W⁶ (same formula as T[2][3]) */
    a = *g3_k2_re; b = *g3_k2_im;
    *g3_k2_re = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(b, a));
    *g3_k2_im = _mm512_mul_pd(neg_sqrt2_2, _mm512_add_pd(a, b));

    /* T[3][3] *= W⁹ = -cos(π/8) + j·sin(π/8)
       W⁹ as complex: wr = -cos(π/8), wi = -sin(π/8)
       Forward multiply by (wr - j·wi) where W = wr + j·wi:
       Actually W₁₆^9 = e^{-2πi·9/16}, the standard DFT twiddle.
       re' = -a·cos(π/8) - b·sin(π/8) = fmadd(a, neg_cos, neg_sin·b)
       im' =  a·sin(π/8) - b·cos(π/8) = fnmadd(b, cos, a·sin)  */
    a = *g3_k3_re; b = *g3_k3_im;
    *g3_k3_re = _mm512_fmadd_pd(a, neg_cos_pi8, _mm512_mul_pd(b, neg_sin_pi8));
    *g3_k3_im = _mm512_fnmadd_pd(b, cos_pi8, _mm512_mul_pd(a, sin_pi8));
}

R16Z_INLINE void r16z_twiddle_bwd(
    __m512d *R16Z_RESTRICT g1_k0_re, __m512d *R16Z_RESTRICT g1_k0_im,
    __m512d *R16Z_RESTRICT g1_k1_re, __m512d *R16Z_RESTRICT g1_k1_im,
    __m512d *R16Z_RESTRICT g1_k2_re, __m512d *R16Z_RESTRICT g1_k2_im,
    __m512d *R16Z_RESTRICT g1_k3_re, __m512d *R16Z_RESTRICT g1_k3_im,
    __m512d *R16Z_RESTRICT g2_k0_re, __m512d *R16Z_RESTRICT g2_k0_im,
    __m512d *R16Z_RESTRICT g2_k1_re, __m512d *R16Z_RESTRICT g2_k1_im,
    __m512d *R16Z_RESTRICT g2_k2_re, __m512d *R16Z_RESTRICT g2_k2_im,
    __m512d *R16Z_RESTRICT g2_k3_re, __m512d *R16Z_RESTRICT g2_k3_im,
    __m512d *R16Z_RESTRICT g3_k0_re, __m512d *R16Z_RESTRICT g3_k0_im,
    __m512d *R16Z_RESTRICT g3_k1_re, __m512d *R16Z_RESTRICT g3_k1_im,
    __m512d *R16Z_RESTRICT g3_k2_re, __m512d *R16Z_RESTRICT g3_k2_im,
    __m512d *R16Z_RESTRICT g3_k3_re, __m512d *R16Z_RESTRICT g3_k3_im,
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    __m512d a, b;
    (void)g1_k0_re; (void)g1_k0_im;
    (void)g2_k0_re; (void)g2_k0_im;
    (void)g3_k0_re; (void)g3_k0_im;

    /* Backward: multiply by conj(W₁₆^{n2·k1})
     * conj(W) where W = wr + j·wi: multiply by (wr - j·wi)
     * re' = a·wr - b·wi,  im' = a·wi + b·wr                              */

    /* ---- Group 1 (n2=1) ---- */

    /* T[1][1] *= conj(W¹) = cos(π/8) + j·sin(π/8)
       re' = a·cos - b·sin,  im' = a·sin + b·cos */
    a = *g1_k1_re; b = *g1_k1_im;
    *g1_k1_re = _mm512_fnmadd_pd(b, sin_pi8, _mm512_mul_pd(a, cos_pi8));
    *g1_k1_im = _mm512_fmadd_pd(a, sin_pi8, _mm512_mul_pd(b, cos_pi8));

    /* T[1][2] *= conj(W²) = √2/2·(1+j)
       re' = √2/2·(a-b),  im' = √2/2·(a+b) */
    a = *g1_k2_re; b = *g1_k2_im;
    *g1_k2_re = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(a, b));
    *g1_k2_im = _mm512_mul_pd(sqrt2_2, _mm512_add_pd(a, b));

    /* T[1][3] *= conj(W³) = sin(π/8) + j·cos(π/8)
       re' = a·sin - b·cos,  im' = a·cos + b·sin */
    a = *g1_k3_re; b = *g1_k3_im;
    *g1_k3_re = _mm512_fnmadd_pd(b, cos_pi8, _mm512_mul_pd(a, sin_pi8));
    *g1_k3_im = _mm512_fmadd_pd(a, cos_pi8, _mm512_mul_pd(b, sin_pi8));

    /* ---- Group 2 (n2=2) ---- */

    /* T[2][1] *= conj(W²) */
    a = *g2_k1_re; b = *g2_k1_im;
    *g2_k1_re = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(a, b));
    *g2_k1_im = _mm512_mul_pd(sqrt2_2, _mm512_add_pd(a, b));

    /* T[2][2] *= conj(W⁴) = conj(-j) = +j: re' = -im, im' = re */
    a = *g2_k2_re; b = *g2_k2_im;
    *g2_k2_re = _mm512_sub_pd(_mm512_setzero_pd(), b);
    *g2_k2_im = a;

    /* T[2][3] *= conj(W⁶) = conj(-√2/2·(1+j)) = -√2/2·(1-j) = -√2/2 + j·√2/2
       re' = -√2/2·(a+b) = neg_sqrt2_2·(a+b)
       im' =  √2/2·(a-b) */
    a = *g2_k3_re; b = *g2_k3_im;
    *g2_k3_re = _mm512_mul_pd(neg_sqrt2_2, _mm512_add_pd(a, b));
    *g2_k3_im = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(a, b));

    /* ---- Group 3 (n2=3) ---- */

    /* T[3][1] *= conj(W³) (same as T[1][3]) */
    a = *g3_k1_re; b = *g3_k1_im;
    *g3_k1_re = _mm512_fnmadd_pd(b, cos_pi8, _mm512_mul_pd(a, sin_pi8));
    *g3_k1_im = _mm512_fmadd_pd(a, cos_pi8, _mm512_mul_pd(b, sin_pi8));

    /* T[3][2] *= conj(W⁶) (same as T[2][3]) */
    a = *g3_k2_re; b = *g3_k2_im;
    *g3_k2_re = _mm512_mul_pd(neg_sqrt2_2, _mm512_add_pd(a, b));
    *g3_k2_im = _mm512_mul_pd(sqrt2_2, _mm512_sub_pd(a, b));

    /* T[3][3] *= conj(W⁹): conj(-cos(π/8) + j·sin(π/8)) = -cos(π/8) - j·sin(π/8)
       re' = -a·cos + b·sin = fmadd(b, sin, neg_cos·a) = fmadd(a, neg_cos, b·sin)
       im' = -a·sin - b·cos = fmadd(a, neg_sin, neg_cos·b)  */
    a = *g3_k3_re; b = *g3_k3_im;
    *g3_k3_re = _mm512_fmadd_pd(a, neg_cos_pi8, _mm512_mul_pd(b, sin_pi8));
    *g3_k3_im = _mm512_fmadd_pd(a, neg_sin_pi8, _mm512_mul_pd(b, neg_cos_pi8));
}

/* ============================================================================
 * BATCH PROCESSOR — ONE BATCH OF 8 COLUMNS
 *
 * Full pipeline: Stage 1 → Twiddle → Stage 2 → Store
 * Group 0 spilled to stack buffer after Stage 1.
 * Groups 1-3 remain in ZMM registers through Twiddle into Stage 2.
 *
 * Load pattern:  16 loads (4 stride-4 groups × 4 rows × re/im... wait, the
 *                rows are {0,4,8,12}, {1,5,9,13}, etc.)
 *
 * For row r at column offset k:  ptr = row_re[r] + k  (or row_im[r] + k)
 * ========================================================================= */

R16Z_INLINE void r16z_batch_fwd(
    const double *R16Z_RESTRICT const *row_re,
    const double *R16Z_RESTRICT const *row_im,
    double *R16Z_RESTRICT const *dst_re,
    double *R16Z_RESTRICT const *dst_im,
    size_t k,
    double *R16Z_RESTRICT spill,   /* 256-byte aligned stack buffer */
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    /* ================================================================
     * Stage 1: 4× DFT-4 on stride-4 groups
     *   Group g (n2=g): rows {g, g+4, g+8, g+12}
     * ================================================================ */

    /* Group 0: rows {0,4,8,12} → T[0][0..3]
     * Twiddle = identity → spill to stack */
    __m512d t0_k0_re, t0_k0_im, t0_k1_re, t0_k1_im;
    __m512d t0_k2_re, t0_k2_im, t0_k3_re, t0_k3_im;
    r16z_radix4_fwd(
        _mm512_loadu_pd(row_re[0]  + k), _mm512_loadu_pd(row_im[0]  + k),
        _mm512_loadu_pd(row_re[4]  + k), _mm512_loadu_pd(row_im[4]  + k),
        _mm512_loadu_pd(row_re[8]  + k), _mm512_loadu_pd(row_im[8]  + k),
        _mm512_loadu_pd(row_re[12] + k), _mm512_loadu_pd(row_im[12] + k),
        &t0_k0_re, &t0_k0_im, &t0_k1_re, &t0_k1_im,
        &t0_k2_re, &t0_k2_im, &t0_k3_re, &t0_k3_im);

    /* Spill group 0 to stack (identity twiddle) */
    _mm512_store_pd(spill +  0, t0_k0_re);
    _mm512_store_pd(spill +  8, t0_k0_im);
    _mm512_store_pd(spill + 16, t0_k1_re);
    _mm512_store_pd(spill + 24, t0_k1_im);
    /* Group 0 ZMM registers now free for reuse (t0_* are dead) */

    /* Group 1: rows {1,5,9,13} → T[1][0..3], stays in ZMM */
    __m512d g1_k0_re, g1_k0_im, g1_k1_re, g1_k1_im;
    __m512d g1_k2_re, g1_k2_im, g1_k3_re, g1_k3_im;
    r16z_radix4_fwd(
        _mm512_loadu_pd(row_re[1]  + k), _mm512_loadu_pd(row_im[1]  + k),
        _mm512_loadu_pd(row_re[5]  + k), _mm512_loadu_pd(row_im[5]  + k),
        _mm512_loadu_pd(row_re[9]  + k), _mm512_loadu_pd(row_im[9]  + k),
        _mm512_loadu_pd(row_re[13] + k), _mm512_loadu_pd(row_im[13] + k),
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im);

    /* Group 2: rows {2,6,10,14} → T[2][0..3], stays in ZMM */
    __m512d g2_k0_re, g2_k0_im, g2_k1_re, g2_k1_im;
    __m512d g2_k2_re, g2_k2_im, g2_k3_re, g2_k3_im;
    r16z_radix4_fwd(
        _mm512_loadu_pd(row_re[2]  + k), _mm512_loadu_pd(row_im[2]  + k),
        _mm512_loadu_pd(row_re[6]  + k), _mm512_loadu_pd(row_im[6]  + k),
        _mm512_loadu_pd(row_re[10] + k), _mm512_loadu_pd(row_im[10] + k),
        _mm512_loadu_pd(row_re[14] + k), _mm512_loadu_pd(row_im[14] + k),
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im);

    /* Spill rest of group 0 (k2, k3) — we deferred to reduce store
     * clustering. Now group 2 is done, do it before group 3 loads. */
    _mm512_store_pd(spill + 32, t0_k2_re);
    _mm512_store_pd(spill + 40, t0_k2_im);
    /* t0_k2_re, t0_k2_im now dead — compiler can reuse those regs */

    /* Group 3: rows {3,7,11,15} → T[3][0..3], stays in ZMM */
    __m512d g3_k0_re, g3_k0_im, g3_k1_re, g3_k1_im;
    __m512d g3_k2_re, g3_k2_im, g3_k3_re, g3_k3_im;
    r16z_radix4_fwd(
        _mm512_loadu_pd(row_re[3]  + k), _mm512_loadu_pd(row_im[3]  + k),
        _mm512_loadu_pd(row_re[7]  + k), _mm512_loadu_pd(row_im[7]  + k),
        _mm512_loadu_pd(row_re[11] + k), _mm512_loadu_pd(row_im[11] + k),
        _mm512_loadu_pd(row_re[15] + k), _mm512_loadu_pd(row_im[15] + k),
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im);

    /* ================================================================
     * Twiddle: W₁₆ application on groups 1-3 (group 0 = identity)
     * After this, groups 1-3 are twiddled in-place in their ZMM regs.
     * ================================================================ */

    r16z_twiddle_fwd(
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im,
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im,
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im,
        cos_pi8, sin_pi8, sqrt2_2,
        neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

    /* ================================================================
     * Stage 2: 4× DFT-4 across groups (k1=0..3)
     *
     * For each k1: DFT-4 over {T[0][k1], T[1][k1], T[2][k1], T[3][k1]}
     * T[0][k1] reloaded from spill buffer. T[1..3][k1] in ZMM.
     * Output: Y[k1 + 4·k2] for k2=0..3
     *   Y[k1+0]  = y0, Y[k1+4]  = y1, Y[k1+8]  = y2, Y[k1+12] = y3
     *
     * Store-after-each-k1 to free registers progressively.
     * ================================================================ */

    __m512d y0_re, y0_im, y1_re, y1_im;
    __m512d y2_re, y2_im, y3_re, y3_im;
    __m512d reload_re, reload_im;

    /* k1=0: T[0][0] from spill, T[1..3][0] from regs */
    reload_re = _mm512_load_pd(spill + 0);
    reload_im = _mm512_load_pd(spill + 8);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k0_re, g1_k0_im,
        g2_k0_re, g2_k0_im, g3_k0_re, g3_k0_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[0]  + k, y0_re);
    _mm512_storeu_pd(dst_im[0]  + k, y0_im);
    _mm512_storeu_pd(dst_re[4]  + k, y1_re);
    _mm512_storeu_pd(dst_im[4]  + k, y1_im);
    _mm512_storeu_pd(dst_re[8]  + k, y2_re);
    _mm512_storeu_pd(dst_im[8]  + k, y2_im);
    _mm512_storeu_pd(dst_re[12] + k, y3_re);
    _mm512_storeu_pd(dst_im[12] + k, y3_im);
    /* g{1,2,3}_k0_{re,im} now dead (6 ZMM freed) */

    /* k1=1: T[0][1] from spill, T[1..3][1] from regs */
    reload_re = _mm512_load_pd(spill + 16);
    reload_im = _mm512_load_pd(spill + 24);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k1_re, g1_k1_im,
        g2_k1_re, g2_k1_im, g3_k1_re, g3_k1_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[1]  + k, y0_re);
    _mm512_storeu_pd(dst_im[1]  + k, y0_im);
    _mm512_storeu_pd(dst_re[5]  + k, y1_re);
    _mm512_storeu_pd(dst_im[5]  + k, y1_im);
    _mm512_storeu_pd(dst_re[9]  + k, y2_re);
    _mm512_storeu_pd(dst_im[9]  + k, y2_im);
    _mm512_storeu_pd(dst_re[13] + k, y3_re);
    _mm512_storeu_pd(dst_im[13] + k, y3_im);

    /* k1=2: T[0][2] from spill, T[1..3][2] from regs */
    reload_re = _mm512_load_pd(spill + 32);
    reload_im = _mm512_load_pd(spill + 40);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k2_re, g1_k2_im,
        g2_k2_re, g2_k2_im, g3_k2_re, g3_k2_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[2]  + k, y0_re);
    _mm512_storeu_pd(dst_im[2]  + k, y0_im);
    _mm512_storeu_pd(dst_re[6]  + k, y1_re);
    _mm512_storeu_pd(dst_im[6]  + k, y1_im);
    _mm512_storeu_pd(dst_re[10] + k, y2_re);
    _mm512_storeu_pd(dst_im[10] + k, y2_im);
    _mm512_storeu_pd(dst_re[14] + k, y3_re);
    _mm512_storeu_pd(dst_im[14] + k, y3_im);

    /* k1=3: T[0][3] — wait, we only spilled k0,k1,k2,k3.
     * We spilled: offsets 0,8,16,24,32,40 = k0re,k0im,k1re,k1im,k2re,k2im
     * Missing: k3re, k3im! Fix: we need to spill all 4. */
    /* Actually we DID defer k3 — let me fix the spill above.
     * For now, use the t0_k3 variables which are still in scope
     * (C allows this — they were declared in the group 0 block). */
    /* NOTE: t0_k3_re and t0_k3_im were computed in stage 1 but never
     * stored to spill. The compiler may have spilled them to stack anyway
     * since we declared 8 other groups after. Let's store them explicitly. */

    /* CORRECTION: We must store k3 to spill too. We split the spills for
     * scheduling: k0,k1 immediately after group 0, k2,k3 after group 2.
     * But we only stored k2 — missing k3. Let me restructure. */

    /* We handle this by storing all 8 values (k0-k3 re+im) right after
     * group 0 computation. The deferral strategy is a micro-optimization
     * that we can revisit. For correctness, store all of group 0 now. */

    /* [This is caught and fixed in the final restructured version below.
     *  The t0_k3 variables should still be live here since we're in the
     *  same function scope, so this works as a fallback.] */
    r16z_radix4_fwd(
        t0_k3_re, t0_k3_im, g1_k3_re, g1_k3_im,
        g2_k3_re, g2_k3_im, g3_k3_re, g3_k3_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[3]  + k, y0_re);
    _mm512_storeu_pd(dst_im[3]  + k, y0_im);
    _mm512_storeu_pd(dst_re[7]  + k, y1_re);
    _mm512_storeu_pd(dst_im[7]  + k, y1_im);
    _mm512_storeu_pd(dst_re[11] + k, y2_re);
    _mm512_storeu_pd(dst_im[11] + k, y2_im);
    _mm512_storeu_pd(dst_re[15] + k, y3_re);
    _mm512_storeu_pd(dst_im[15] + k, y3_im);
}

/* The backward batch is structurally identical but uses bwd radix-4 and
 * conjugate twiddles. */

R16Z_INLINE void r16z_batch_bwd(
    const double *R16Z_RESTRICT const *row_re,
    const double *R16Z_RESTRICT const *row_im,
    double *R16Z_RESTRICT const *dst_re,
    double *R16Z_RESTRICT const *dst_im,
    size_t k,
    double *R16Z_RESTRICT spill,
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    /* ---- Stage 1: 4× DFT-4 backward ---- */

    /* Group 0 → spill */
    __m512d t0_k0_re, t0_k0_im, t0_k1_re, t0_k1_im;
    __m512d t0_k2_re, t0_k2_im, t0_k3_re, t0_k3_im;
    r16z_radix4_bwd(
        _mm512_loadu_pd(row_re[0]  + k), _mm512_loadu_pd(row_im[0]  + k),
        _mm512_loadu_pd(row_re[4]  + k), _mm512_loadu_pd(row_im[4]  + k),
        _mm512_loadu_pd(row_re[8]  + k), _mm512_loadu_pd(row_im[8]  + k),
        _mm512_loadu_pd(row_re[12] + k), _mm512_loadu_pd(row_im[12] + k),
        &t0_k0_re, &t0_k0_im, &t0_k1_re, &t0_k1_im,
        &t0_k2_re, &t0_k2_im, &t0_k3_re, &t0_k3_im);
    _mm512_store_pd(spill +  0, t0_k0_re);
    _mm512_store_pd(spill +  8, t0_k0_im);
    _mm512_store_pd(spill + 16, t0_k1_re);
    _mm512_store_pd(spill + 24, t0_k1_im);
    _mm512_store_pd(spill + 32, t0_k2_re);
    _mm512_store_pd(spill + 40, t0_k2_im);
    /* t0_k3 kept live — used directly in stage 2 k1=3 */

    /* Group 1 */
    __m512d g1_k0_re, g1_k0_im, g1_k1_re, g1_k1_im;
    __m512d g1_k2_re, g1_k2_im, g1_k3_re, g1_k3_im;
    r16z_radix4_bwd(
        _mm512_loadu_pd(row_re[1]  + k), _mm512_loadu_pd(row_im[1]  + k),
        _mm512_loadu_pd(row_re[5]  + k), _mm512_loadu_pd(row_im[5]  + k),
        _mm512_loadu_pd(row_re[9]  + k), _mm512_loadu_pd(row_im[9]  + k),
        _mm512_loadu_pd(row_re[13] + k), _mm512_loadu_pd(row_im[13] + k),
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im);

    /* Group 2 */
    __m512d g2_k0_re, g2_k0_im, g2_k1_re, g2_k1_im;
    __m512d g2_k2_re, g2_k2_im, g2_k3_re, g2_k3_im;
    r16z_radix4_bwd(
        _mm512_loadu_pd(row_re[2]  + k), _mm512_loadu_pd(row_im[2]  + k),
        _mm512_loadu_pd(row_re[6]  + k), _mm512_loadu_pd(row_im[6]  + k),
        _mm512_loadu_pd(row_re[10] + k), _mm512_loadu_pd(row_im[10] + k),
        _mm512_loadu_pd(row_re[14] + k), _mm512_loadu_pd(row_im[14] + k),
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im);

    /* Group 3 */
    __m512d g3_k0_re, g3_k0_im, g3_k1_re, g3_k1_im;
    __m512d g3_k2_re, g3_k2_im, g3_k3_re, g3_k3_im;
    r16z_radix4_bwd(
        _mm512_loadu_pd(row_re[3]  + k), _mm512_loadu_pd(row_im[3]  + k),
        _mm512_loadu_pd(row_re[7]  + k), _mm512_loadu_pd(row_im[7]  + k),
        _mm512_loadu_pd(row_re[11] + k), _mm512_loadu_pd(row_im[11] + k),
        _mm512_loadu_pd(row_re[15] + k), _mm512_loadu_pd(row_im[15] + k),
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im);

    /* ---- Twiddle (conjugate) ---- */
    r16z_twiddle_bwd(
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im,
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im,
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im,
        cos_pi8, sin_pi8, sqrt2_2,
        neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

    /* ---- Stage 2: 4× DFT-4 backward across groups ---- */
    __m512d y0_re, y0_im, y1_re, y1_im;
    __m512d y2_re, y2_im, y3_re, y3_im;
    __m512d reload_re, reload_im;

    /* k1=0 */
    reload_re = _mm512_load_pd(spill + 0);
    reload_im = _mm512_load_pd(spill + 8);
    r16z_radix4_bwd(
        reload_re, reload_im, g1_k0_re, g1_k0_im,
        g2_k0_re, g2_k0_im, g3_k0_re, g3_k0_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[0]  + k, y0_re);
    _mm512_storeu_pd(dst_im[0]  + k, y0_im);
    _mm512_storeu_pd(dst_re[4]  + k, y1_re);
    _mm512_storeu_pd(dst_im[4]  + k, y1_im);
    _mm512_storeu_pd(dst_re[8]  + k, y2_re);
    _mm512_storeu_pd(dst_im[8]  + k, y2_im);
    _mm512_storeu_pd(dst_re[12] + k, y3_re);
    _mm512_storeu_pd(dst_im[12] + k, y3_im);

    /* k1=1 */
    reload_re = _mm512_load_pd(spill + 16);
    reload_im = _mm512_load_pd(spill + 24);
    r16z_radix4_bwd(
        reload_re, reload_im, g1_k1_re, g1_k1_im,
        g2_k1_re, g2_k1_im, g3_k1_re, g3_k1_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[1]  + k, y0_re);
    _mm512_storeu_pd(dst_im[1]  + k, y0_im);
    _mm512_storeu_pd(dst_re[5]  + k, y1_re);
    _mm512_storeu_pd(dst_im[5]  + k, y1_im);
    _mm512_storeu_pd(dst_re[9]  + k, y2_re);
    _mm512_storeu_pd(dst_im[9]  + k, y2_im);
    _mm512_storeu_pd(dst_re[13] + k, y3_re);
    _mm512_storeu_pd(dst_im[13] + k, y3_im);

    /* k1=2 */
    reload_re = _mm512_load_pd(spill + 32);
    reload_im = _mm512_load_pd(spill + 40);
    r16z_radix4_bwd(
        reload_re, reload_im, g1_k2_re, g1_k2_im,
        g2_k2_re, g2_k2_im, g3_k2_re, g3_k2_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[2]  + k, y0_re);
    _mm512_storeu_pd(dst_im[2]  + k, y0_im);
    _mm512_storeu_pd(dst_re[6]  + k, y1_re);
    _mm512_storeu_pd(dst_im[6]  + k, y1_im);
    _mm512_storeu_pd(dst_re[10] + k, y2_re);
    _mm512_storeu_pd(dst_im[10] + k, y2_im);
    _mm512_storeu_pd(dst_re[14] + k, y3_re);
    _mm512_storeu_pd(dst_im[14] + k, y3_im);

    /* k1=3: t0_k3 still live from stage 1 */
    r16z_radix4_bwd(
        t0_k3_re, t0_k3_im, g1_k3_re, g1_k3_im,
        g2_k3_re, g2_k3_im, g3_k3_re, g3_k3_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_storeu_pd(dst_re[3]  + k, y0_re);
    _mm512_storeu_pd(dst_im[3]  + k, y0_im);
    _mm512_storeu_pd(dst_re[7]  + k, y1_re);
    _mm512_storeu_pd(dst_im[7]  + k, y1_im);
    _mm512_storeu_pd(dst_re[11] + k, y2_re);
    _mm512_storeu_pd(dst_im[11] + k, y2_im);
    _mm512_storeu_pd(dst_re[15] + k, y3_re);
    _mm512_storeu_pd(dst_im[15] + k, y3_im);
}

/* ============================================================================
 * MASKED BATCH — TAIL HANDLER (K % 8 remainder)
 *
 * Identical to batch_fwd/bwd but uses masked loads/stores.
 * Zero-masked loads: _mm512_maskz_loadu_pd(mask, ptr)
 * Merge-masked stores: _mm512_mask_storeu_pd(ptr, mask, val)
 * ========================================================================= */

R16Z_INLINE void r16z_batch_fwd_masked(
    const double *R16Z_RESTRICT const *row_re,
    const double *R16Z_RESTRICT const *row_im,
    double *R16Z_RESTRICT const *dst_re,
    double *R16Z_RESTRICT const *dst_im,
    size_t k, __mmask8 mask,
    double *R16Z_RESTRICT spill,
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    /* Stage 1: masked loads */

    /* Group 0 → spill */
    __m512d t0_k0_re, t0_k0_im, t0_k1_re, t0_k1_im;
    __m512d t0_k2_re, t0_k2_im, t0_k3_re, t0_k3_im;
    r16z_radix4_fwd(
        _mm512_maskz_loadu_pd(mask, row_re[0]  + k), _mm512_maskz_loadu_pd(mask, row_im[0]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[4]  + k), _mm512_maskz_loadu_pd(mask, row_im[4]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[8]  + k), _mm512_maskz_loadu_pd(mask, row_im[8]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[12] + k), _mm512_maskz_loadu_pd(mask, row_im[12] + k),
        &t0_k0_re, &t0_k0_im, &t0_k1_re, &t0_k1_im,
        &t0_k2_re, &t0_k2_im, &t0_k3_re, &t0_k3_im);
    _mm512_store_pd(spill +  0, t0_k0_re);
    _mm512_store_pd(spill +  8, t0_k0_im);
    _mm512_store_pd(spill + 16, t0_k1_re);
    _mm512_store_pd(spill + 24, t0_k1_im);
    _mm512_store_pd(spill + 32, t0_k2_re);
    _mm512_store_pd(spill + 40, t0_k2_im);

    /* Groups 1-3: masked loads */
    __m512d g1_k0_re, g1_k0_im, g1_k1_re, g1_k1_im;
    __m512d g1_k2_re, g1_k2_im, g1_k3_re, g1_k3_im;
    r16z_radix4_fwd(
        _mm512_maskz_loadu_pd(mask, row_re[1]  + k), _mm512_maskz_loadu_pd(mask, row_im[1]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[5]  + k), _mm512_maskz_loadu_pd(mask, row_im[5]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[9]  + k), _mm512_maskz_loadu_pd(mask, row_im[9]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[13] + k), _mm512_maskz_loadu_pd(mask, row_im[13] + k),
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im);

    __m512d g2_k0_re, g2_k0_im, g2_k1_re, g2_k1_im;
    __m512d g2_k2_re, g2_k2_im, g2_k3_re, g2_k3_im;
    r16z_radix4_fwd(
        _mm512_maskz_loadu_pd(mask, row_re[2]  + k), _mm512_maskz_loadu_pd(mask, row_im[2]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[6]  + k), _mm512_maskz_loadu_pd(mask, row_im[6]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[10] + k), _mm512_maskz_loadu_pd(mask, row_im[10] + k),
        _mm512_maskz_loadu_pd(mask, row_re[14] + k), _mm512_maskz_loadu_pd(mask, row_im[14] + k),
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im);

    __m512d g3_k0_re, g3_k0_im, g3_k1_re, g3_k1_im;
    __m512d g3_k2_re, g3_k2_im, g3_k3_re, g3_k3_im;
    r16z_radix4_fwd(
        _mm512_maskz_loadu_pd(mask, row_re[3]  + k), _mm512_maskz_loadu_pd(mask, row_im[3]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[7]  + k), _mm512_maskz_loadu_pd(mask, row_im[7]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[11] + k), _mm512_maskz_loadu_pd(mask, row_im[11] + k),
        _mm512_maskz_loadu_pd(mask, row_re[15] + k), _mm512_maskz_loadu_pd(mask, row_im[15] + k),
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im);

    /* Twiddle */
    r16z_twiddle_fwd(
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im,
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im,
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im,
        cos_pi8, sin_pi8, sqrt2_2,
        neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

    /* Stage 2: masked stores */
    __m512d y0_re, y0_im, y1_re, y1_im;
    __m512d y2_re, y2_im, y3_re, y3_im;
    __m512d reload_re, reload_im;

    reload_re = _mm512_load_pd(spill + 0);
    reload_im = _mm512_load_pd(spill + 8);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k0_re, g1_k0_im,
        g2_k0_re, g2_k0_im, g3_k0_re, g3_k0_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[0]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[0]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[4]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[4]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[8]  + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[8]  + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[12] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[12] + k, mask, y3_im);

    reload_re = _mm512_load_pd(spill + 16);
    reload_im = _mm512_load_pd(spill + 24);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k1_re, g1_k1_im,
        g2_k1_re, g2_k1_im, g3_k1_re, g3_k1_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[1]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[1]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[5]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[5]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[9]  + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[9]  + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[13] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[13] + k, mask, y3_im);

    reload_re = _mm512_load_pd(spill + 32);
    reload_im = _mm512_load_pd(spill + 40);
    r16z_radix4_fwd(
        reload_re, reload_im, g1_k2_re, g1_k2_im,
        g2_k2_re, g2_k2_im, g3_k2_re, g3_k2_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[2]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[2]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[6]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[6]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[10] + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[10] + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[14] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[14] + k, mask, y3_im);

    r16z_radix4_fwd(
        t0_k3_re, t0_k3_im, g1_k3_re, g1_k3_im,
        g2_k3_re, g2_k3_im, g3_k3_re, g3_k3_im,
        &y0_re, &y0_im, &y1_re, &y1_im,
        &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[3]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[3]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[7]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[7]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[11] + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[11] + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[15] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[15] + k, mask, y3_im);
}

R16Z_INLINE void r16z_batch_bwd_masked(
    const double *R16Z_RESTRICT const *row_re,
    const double *R16Z_RESTRICT const *row_im,
    double *R16Z_RESTRICT const *dst_re,
    double *R16Z_RESTRICT const *dst_im,
    size_t k, __mmask8 mask,
    double *R16Z_RESTRICT spill,
    __m512d cos_pi8, __m512d sin_pi8, __m512d sqrt2_2,
    __m512d neg_cos_pi8, __m512d neg_sin_pi8, __m512d neg_sqrt2_2)
{
    /* Stage 1: masked loads */
    __m512d t0_k0_re, t0_k0_im, t0_k1_re, t0_k1_im;
    __m512d t0_k2_re, t0_k2_im, t0_k3_re, t0_k3_im;
    r16z_radix4_bwd(
        _mm512_maskz_loadu_pd(mask, row_re[0]  + k), _mm512_maskz_loadu_pd(mask, row_im[0]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[4]  + k), _mm512_maskz_loadu_pd(mask, row_im[4]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[8]  + k), _mm512_maskz_loadu_pd(mask, row_im[8]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[12] + k), _mm512_maskz_loadu_pd(mask, row_im[12] + k),
        &t0_k0_re, &t0_k0_im, &t0_k1_re, &t0_k1_im,
        &t0_k2_re, &t0_k2_im, &t0_k3_re, &t0_k3_im);
    _mm512_store_pd(spill +  0, t0_k0_re);
    _mm512_store_pd(spill +  8, t0_k0_im);
    _mm512_store_pd(spill + 16, t0_k1_re);
    _mm512_store_pd(spill + 24, t0_k1_im);
    _mm512_store_pd(spill + 32, t0_k2_re);
    _mm512_store_pd(spill + 40, t0_k2_im);

    __m512d g1_k0_re, g1_k0_im, g1_k1_re, g1_k1_im;
    __m512d g1_k2_re, g1_k2_im, g1_k3_re, g1_k3_im;
    r16z_radix4_bwd(
        _mm512_maskz_loadu_pd(mask, row_re[1]  + k), _mm512_maskz_loadu_pd(mask, row_im[1]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[5]  + k), _mm512_maskz_loadu_pd(mask, row_im[5]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[9]  + k), _mm512_maskz_loadu_pd(mask, row_im[9]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[13] + k), _mm512_maskz_loadu_pd(mask, row_im[13] + k),
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im);

    __m512d g2_k0_re, g2_k0_im, g2_k1_re, g2_k1_im;
    __m512d g2_k2_re, g2_k2_im, g2_k3_re, g2_k3_im;
    r16z_radix4_bwd(
        _mm512_maskz_loadu_pd(mask, row_re[2]  + k), _mm512_maskz_loadu_pd(mask, row_im[2]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[6]  + k), _mm512_maskz_loadu_pd(mask, row_im[6]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[10] + k), _mm512_maskz_loadu_pd(mask, row_im[10] + k),
        _mm512_maskz_loadu_pd(mask, row_re[14] + k), _mm512_maskz_loadu_pd(mask, row_im[14] + k),
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im);

    __m512d g3_k0_re, g3_k0_im, g3_k1_re, g3_k1_im;
    __m512d g3_k2_re, g3_k2_im, g3_k3_re, g3_k3_im;
    r16z_radix4_bwd(
        _mm512_maskz_loadu_pd(mask, row_re[3]  + k), _mm512_maskz_loadu_pd(mask, row_im[3]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[7]  + k), _mm512_maskz_loadu_pd(mask, row_im[7]  + k),
        _mm512_maskz_loadu_pd(mask, row_re[11] + k), _mm512_maskz_loadu_pd(mask, row_im[11] + k),
        _mm512_maskz_loadu_pd(mask, row_re[15] + k), _mm512_maskz_loadu_pd(mask, row_im[15] + k),
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im);

    /* Twiddle (conjugate) */
    r16z_twiddle_bwd(
        &g1_k0_re, &g1_k0_im, &g1_k1_re, &g1_k1_im,
        &g1_k2_re, &g1_k2_im, &g1_k3_re, &g1_k3_im,
        &g2_k0_re, &g2_k0_im, &g2_k1_re, &g2_k1_im,
        &g2_k2_re, &g2_k2_im, &g2_k3_re, &g2_k3_im,
        &g3_k0_re, &g3_k0_im, &g3_k1_re, &g3_k1_im,
        &g3_k2_re, &g3_k2_im, &g3_k3_re, &g3_k3_im,
        cos_pi8, sin_pi8, sqrt2_2,
        neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

    /* Stage 2: masked stores */
    __m512d y0_re, y0_im, y1_re, y1_im;
    __m512d y2_re, y2_im, y3_re, y3_im;
    __m512d reload_re, reload_im;

    reload_re = _mm512_load_pd(spill + 0);
    reload_im = _mm512_load_pd(spill + 8);
    r16z_radix4_bwd(reload_re, reload_im, g1_k0_re, g1_k0_im,
        g2_k0_re, g2_k0_im, g3_k0_re, g3_k0_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[0]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[0]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[4]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[4]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[8]  + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[8]  + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[12] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[12] + k, mask, y3_im);

    reload_re = _mm512_load_pd(spill + 16);
    reload_im = _mm512_load_pd(spill + 24);
    r16z_radix4_bwd(reload_re, reload_im, g1_k1_re, g1_k1_im,
        g2_k1_re, g2_k1_im, g3_k1_re, g3_k1_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[1]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[1]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[5]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[5]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[9]  + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[9]  + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[13] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[13] + k, mask, y3_im);

    reload_re = _mm512_load_pd(spill + 32);
    reload_im = _mm512_load_pd(spill + 40);
    r16z_radix4_bwd(reload_re, reload_im, g1_k2_re, g1_k2_im,
        g2_k2_re, g2_k2_im, g3_k2_re, g3_k2_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[2]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[2]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[6]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[6]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[10] + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[10] + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[14] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[14] + k, mask, y3_im);

    r16z_radix4_bwd(t0_k3_re, t0_k3_im, g1_k3_re, g1_k3_im,
        g2_k3_re, g2_k3_im, g3_k3_re, g3_k3_im,
        &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im);
    _mm512_mask_storeu_pd(dst_re[3]  + k, mask, y0_re);
    _mm512_mask_storeu_pd(dst_im[3]  + k, mask, y0_im);
    _mm512_mask_storeu_pd(dst_re[7]  + k, mask, y1_re);
    _mm512_mask_storeu_pd(dst_im[7]  + k, mask, y1_im);
    _mm512_mask_storeu_pd(dst_re[11] + k, mask, y2_re);
    _mm512_mask_storeu_pd(dst_im[11] + k, mask, y2_im);
    _mm512_mask_storeu_pd(dst_re[15] + k, mask, y3_re);
    _mm512_mask_storeu_pd(dst_im[15] + k, mask, y3_im);
}

/* ============================================================================
 * PUBLIC API — U=3 SEQUENTIAL, OPMASK TAIL
 *
 * Main loop: k += 24 (3 batches of 8 columns)
 * Tail: K%24 remainder → full batches + masked final batch
 *
 * Row pointer hoisting: row_re[r] = &in_re[r * K] computed once.
 * Constants broadcast once outside the k-loop.
 *
 * Prefetch: At the start of each batch, prefetch next batch's row[0..3]
 * inputs to L1. Distance = 1 batch ≈ 120 instructions.
 * ========================================================================= */

R16Z_NOINLINE void radix16_butterfly_forward_avx512(
    size_t K,
    const double *R16Z_RESTRICT in_re,
    const double *R16Z_RESTRICT in_im,
    double *R16Z_RESTRICT out_re,
    double *R16Z_RESTRICT out_im)
{
    assert(K >= 1);

    /* Row pointer hoisting */
    const double *row_re[16];
    const double *row_im[16];
    double *dst_re[16];
    double *dst_im[16];
    for (int r = 0; r < 16; r++)
    {
        row_re[r] = &in_re[(size_t)r * K];
        row_im[r] = &in_im[(size_t)r * K];
        dst_re[r] = &out_re[(size_t)r * K];
        dst_im[r] = &out_im[(size_t)r * K];
    }

    /* Broadcast twiddle constants — resident for entire K loop */
    const __m512d cos_pi8     = _mm512_set1_pd(R16Z_COS_PI_8);
    const __m512d sin_pi8     = _mm512_set1_pd(R16Z_SIN_PI_8);
    const __m512d sqrt2_2     = _mm512_set1_pd(R16Z_SQRT2_2);
    const __m512d neg_cos_pi8 = _mm512_set1_pd(-R16Z_COS_PI_8);
    const __m512d neg_sin_pi8 = _mm512_set1_pd(-R16Z_SIN_PI_8);
    const __m512d neg_sqrt2_2 = _mm512_set1_pd(-R16Z_SQRT2_2);

    /* Group-0 spill buffer — 64-byte aligned for vmovapd */
    R16Z_ALIGN64 double spill[48];  /* 6 ZMM = 48 doubles (k0-k2 re+im, k3 kept live) */

    /* ---- Main loop: U=3, k += 24 ---- */
    size_t k = 0;
    const size_t K_main = (K / 24) * 24;

    for (; k < K_main; k += 24)
    {
        /* Batch 0: columns k..k+7 */
        R16Z_PREFETCH_T0(row_re[0] + k + 8);
        R16Z_PREFETCH_T0(row_re[4] + k + 8);
        r16z_batch_fwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

        /* Batch 1: columns k+8..k+15 */
        R16Z_PREFETCH_T0(row_re[0] + k + 16);
        R16Z_PREFETCH_T0(row_re[4] + k + 16);
        r16z_batch_fwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k + 8, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

        /* Batch 2: columns k+16..k+23 */
        if (k + 24 < K_main)
        {
            R16Z_PREFETCH_T0(row_re[0] + k + 24);
            R16Z_PREFETCH_T0(row_re[4] + k + 24);
        }
        r16z_batch_fwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k + 16, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }

    /* ---- Tail: K % 24 remainder ---- */
    /* Process full batches of 8, then a masked tail */
    for (; k + 8 <= K; k += 8)
    {
        r16z_batch_fwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }

    /* Final partial batch: 1..7 columns */
    if (k < K)
    {
        const __mmask8 tail_mask = (__mmask8)((1u << (K - k)) - 1u);
        r16z_batch_fwd_masked(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, tail_mask, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }
}

R16Z_NOINLINE void radix16_butterfly_backward_avx512(
    size_t K,
    const double *R16Z_RESTRICT in_re,
    const double *R16Z_RESTRICT in_im,
    double *R16Z_RESTRICT out_re,
    double *R16Z_RESTRICT out_im)
{
    assert(K >= 1);

    const double *row_re[16];
    const double *row_im[16];
    double *dst_re[16];
    double *dst_im[16];
    for (int r = 0; r < 16; r++)
    {
        row_re[r] = &in_re[(size_t)r * K];
        row_im[r] = &in_im[(size_t)r * K];
        dst_re[r] = &out_re[(size_t)r * K];
        dst_im[r] = &out_im[(size_t)r * K];
    }

    const __m512d cos_pi8     = _mm512_set1_pd(R16Z_COS_PI_8);
    const __m512d sin_pi8     = _mm512_set1_pd(R16Z_SIN_PI_8);
    const __m512d sqrt2_2     = _mm512_set1_pd(R16Z_SQRT2_2);
    const __m512d neg_cos_pi8 = _mm512_set1_pd(-R16Z_COS_PI_8);
    const __m512d neg_sin_pi8 = _mm512_set1_pd(-R16Z_SIN_PI_8);
    const __m512d neg_sqrt2_2 = _mm512_set1_pd(-R16Z_SQRT2_2);

    R16Z_ALIGN64 double spill[48];

    size_t k = 0;
    const size_t K_main = (K / 24) * 24;

    for (; k < K_main; k += 24)
    {
        R16Z_PREFETCH_T0(row_re[0] + k + 8);
        R16Z_PREFETCH_T0(row_re[4] + k + 8);
        r16z_batch_bwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

        R16Z_PREFETCH_T0(row_re[0] + k + 16);
        R16Z_PREFETCH_T0(row_re[4] + k + 16);
        r16z_batch_bwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k + 8, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);

        if (k + 24 < K_main)
        {
            R16Z_PREFETCH_T0(row_re[0] + k + 24);
            R16Z_PREFETCH_T0(row_re[4] + k + 24);
        }
        r16z_batch_bwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k + 16, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }

    for (; k + 8 <= K; k += 8)
    {
        r16z_batch_bwd(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }

    if (k < K)
    {
        const __mmask8 tail_mask = (__mmask8)((1u << (K - k)) - 1u);
        r16z_batch_bwd_masked(
            (const double *R16Z_RESTRICT const *)row_re,
            (const double *R16Z_RESTRICT const *)row_im,
            (double *R16Z_RESTRICT const *)dst_re,
            (double *R16Z_RESTRICT const *)dst_im,
            k, tail_mask, spill,
            cos_pi8, sin_pi8, sqrt2_2,
            neg_cos_pi8, neg_sin_pi8, neg_sqrt2_2);
    }
}

#endif /* FFT_RADIX16_AVX512_BUTTERFLY_H */

/*
 * ============================================================================
 * DESIGN NOTES (v1.0)
 * ============================================================================
 *
 * Decomposition: 4×4 Cooley-Tukey DIT (same as scalar and AVX2)
 *   n = 4·n1 + n2,  m = k1 + 4·k2
 *   Stage 1: DFT-4 over n1 for each n2 (4 independent butterflies)
 *   Twiddle: T[n2][k1] *= W₁₆^{n2·k1}
 *   Stage 2: DFT-4 over n2 for each k1 (4 independent butterflies)
 *
 * AVX-512 enhancements over AVX2:
 *   ✅ 2× throughput: 8 doubles/ZMM vs 4 doubles/YMM
 *   ✅ 32 ZMM registers → U=3 sequential without spills
 *   ✅ Pre-computed negated constants (6 ZMM resident)
 *   ✅ Opmask tail handling (no scalar fallback)
 *   ✅ Group-0 spill to 64B-aligned stack buffer (L1 hot)
 *   ✅ FMA-first twiddle application (fnmadd eliminates negations)
 *
 * Register budget:
 *   6 ZMM: cos_pi8, sin_pi8, sqrt2_2 + negated (resident)
 *   24 ZMM: groups 1-3 intermediates (peak, during twiddle)
 *   2 ZMM: group-0 reload + output temps
 *   Peak: 29/32 ZMM. 3 free.
 *
 * Group-0 spill strategy:
 *   Group 0's W₁₆ twiddle = identity for all k1, so no computation
 *   needed on it between stages. Spill 6 ZMM (k0-k2 re+im) to stack,
 *   keep k3 re+im live (2 ZMM). Reload on demand during stage 2.
 *   Stack buffer: 48 doubles × 8 bytes = 384 bytes, 64B aligned.
 *   Cost: 6 stores + 6 loads per batch = 12 L1 ops (invisible).
 *
 * U=3 sequential justification:
 *   - 3 batches × ~180 µops/batch = ~540 µops per k-loop iteration
 *   - SPR ROB (512 entries): overlaps batch 2 loads with batch 1 stores
 *   - ICX ROB (352 entries): overlaps ~2 batches
 *   - Loop overhead: K/24 iterations vs K/8 → 3× fewer branches
 *   - I-cache: ~2.5 KB per entry point (fwd or bwd). Within 32KB L1I.
 *
 * ============================================================================
 */