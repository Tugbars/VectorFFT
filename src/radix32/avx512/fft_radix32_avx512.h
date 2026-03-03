/**
 * @file fft_radix32_avx512.h
 * @brief AVX-512 Radix-32 FFT Stage — U=1 Fused Single-Pass Driver
 *
 * Two-pass decomposition: 4×DIT → temp → 8×DIF
 *   Pass 1: 8 groups of radix-4 DIT, strided input → bin-major temp
 *   Pass 2: 4 bins of radix-8 DIF, temp → output
 *
 * This is the CORRECTNESS BASELINE — simple sequential k-loop at step=8,
 * no software pipelining, no prefetch, no NT stores. BLOCKED8 and BLOCKED4.
 * U=2 pipelining and RECURRENCE mode will layer on top.
 *
 * KEY DIFFERENCES FROM AVX2 DRIVER:
 *   - step = 8 (vs 4): processes 8 complex per k-iteration
 *   - K must be multiple of 8 (vs 4)
 *   - All loads/stores use _mm512_{load,store}_pd (64-byte aligned)
 *   - Uses layer-0/layer-1 kernels from fft_radix32_avx512_core.h
 *
 * MEMORY LAYOUT (identical to AVX2):
 *   Input:  [32 stripes][K] — in_re[stripe * K + k]
 *   Temp:   [32 stripes][K] — bin-major: bin b, group g → stripe (b*8+g)
 *   Output: [32 stripes][K] — out_re[stripe * K + k]
 *
 * @author Tugbars (AVX-512 driver by Claude)
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX512_H
#define FFT_RADIX32_AVX512_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

/* Core kernels (layer 0 + layer 1) */
#include "fft_radix32_avx512_core.h"

/* AVX2 header for shared type definitions (tw_blocked8_t, etc.) */
#include "fft_radix32_avx2.h"

/*==========================================================================
 * PASS 1: WAVE A/B STORE MACROS (AVX-512, radix-4)
 *
 * Split two-wave stores for U=2 interleaving, same pattern as pass 2.
 * Wave A: even bins {y0, y2} → frees 4 ZMM
 * Wave B: odd bins  {y1, y3} → frees 4 ZMM
 *=========================================================================*/

#define DIT4_STORE_WAVE_A_512(temp_re, temp_im, K, k, s0, s2,   \
                               y0r, y0i, y2r, y2i)               \
    do {                                                          \
        _mm512_store_pd(&(temp_re)[(s0) * (K) + (k)], y0r);      \
        _mm512_store_pd(&(temp_im)[(s0) * (K) + (k)], y0i);      \
        _mm512_store_pd(&(temp_re)[(s2) * (K) + (k)], y2r);      \
        _mm512_store_pd(&(temp_im)[(s2) * (K) + (k)], y2i);      \
    } while (0)

#define DIT4_STORE_WAVE_B_512(temp_re, temp_im, K, k, s1, s3,   \
                               y1r, y1i, y3r, y3i)               \
    do {                                                          \
        _mm512_store_pd(&(temp_re)[(s1) * (K) + (k)], y1r);      \
        _mm512_store_pd(&(temp_im)[(s1) * (K) + (k)], y1i);      \
        _mm512_store_pd(&(temp_re)[(s3) * (K) + (k)], y3r);      \
        _mm512_store_pd(&(temp_im)[(s3) * (K) + (k)], y3i);      \
    } while (0)

/*==========================================================================
 * PASS 1 LOAD + TWIDDLE-APPLY MACRO
 *
 * Shared between forward/backward — twiddle application is identical,
 * only the butterfly direction differs.
 *
 * Loads 4 strided inputs, applies BLOCKED2 twiddles (W1, W2 from memory,
 * W3 = W1×W2 derived), produces t0=x0 (untwidded), t1, t2, t3.
 *=========================================================================*/

#define PASS1_LOAD_AND_TWIDDLE_512(                                         \
    in_re_base, in_im_base, in_stride, tw_re, tw_im, K, k,                 \
    t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i)                               \
    do {                                                                    \
        /* Load 4 strided inputs */                                         \
        t0r = _mm512_load_pd(&(in_re_base)[0 * (in_stride) + (k)]);        \
        t0i = _mm512_load_pd(&(in_im_base)[0 * (in_stride) + (k)]);        \
        __m512d _x1r = _mm512_load_pd(&(in_re_base)[1 * (in_stride) + (k)]);\
        __m512d _x1i = _mm512_load_pd(&(in_im_base)[1 * (in_stride) + (k)]);\
        __m512d _x2r = _mm512_load_pd(&(in_re_base)[2 * (in_stride) + (k)]);\
        __m512d _x2i = _mm512_load_pd(&(in_im_base)[2 * (in_stride) + (k)]);\
        __m512d _x3r = _mm512_load_pd(&(in_re_base)[3 * (in_stride) + (k)]);\
        __m512d _x3i = _mm512_load_pd(&(in_im_base)[3 * (in_stride) + (k)]);\
                                                                            \
        /* Load W1, W2 (sequential — HW prefetcher handles these) */        \
        __m512d _W1r = _mm512_load_pd(&(tw_re)[0 * (K) + (k)]);            \
        __m512d _W1i = _mm512_load_pd(&(tw_im)[0 * (K) + (k)]);            \
        __m512d _W2r = _mm512_load_pd(&(tw_re)[1 * (K) + (k)]);            \
        __m512d _W2i = _mm512_load_pd(&(tw_im)[1 * (K) + (k)]);            \
                                                                            \
        /* x1 *= W1 */                                                      \
        cmul_v512(_x1r, _x1i, _W1r, _W1i, &(t1r), &(t1i));                \
        /* x2 *= W2 */                                                      \
        cmul_v512(_x2r, _x2i, _W2r, _W2i, &(t2r), &(t2i));                \
        /* W3 = W1 × W2 (derived), then x3 *= W3 */                        \
        __m512d _W3r, _W3i;                                                 \
        cmul_v512(_W1r, _W1i, _W2r, _W2i, &_W3r, &_W3i);                  \
        cmul_v512(_x3r, _x3i, _W3r, _W3i, &(t3r), &(t3i));                \
    } while (0)

/*==========================================================================
 * PASS 1 INPUT PREFETCH MACRO
 *
 * Prefetch next iteration's 4 strided inputs into L2 (locality=2).
 * Stride = 8*K, so consecutive loads jump by thousands of bytes —
 * hardware prefetcher can't track these. Explicit prefetch is critical.
 *=========================================================================*/

#define PASS1_PREFETCH_INPUTS_512(in_re_base, in_im_base, in_stride, k)    \
    do {                                                                    \
        R32_PREFETCH(&(in_re_base)[0 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_im_base)[0 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_re_base)[1 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_im_base)[1 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_re_base)[2 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_im_base)[2 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_re_base)[3 * (in_stride) + (k)], 0, 2);          \
        R32_PREFETCH(&(in_im_base)[3 * (in_stride) + (k)], 0, 2);          \
    } while (0)

/*==========================================================================
 * PASS 1: GENERATOR MACRO
 *
 * Generates both U=1 (K=8 fallback) and U=2 (K≥16 pipelined) variants
 * for a given direction. The ONLY difference between forward and backward
 * is the butterfly function call — everything else (loads, twiddles,
 * stores, prefetches) is identical.
 *
 * This avoids duplicating ~100 lines of U=2 pipelining logic.
 *
 * U=2 register budget (pass 1):
 *   Steady-state peak at step 5 (wave A stored, next inputs loading):
 *     4 ZMM — y1,y3 re+im (awaiting wave B)
 *     8 ZMM — nx0..nx3 re+im (next inputs)
 *    ────────
 *    12 ZMM peak, 20 spare
 *
 *   Full breakdown per step:
 *     1. CONSUME:  12 ZMM (x0..x3 + W1,W2 → renamed from next)
 *     2. TWIDDLE:  ~14 ZMM (intermediates reuse input/twiddle slots)
 *     3. BUTTERFLY: 8 ZMM (y0..y3 re+im, inputs dead)
 *     4. WAVE A:    4 ZMM (y1,y3 remain)
 *     5. LOAD NX:  12 ZMM (4 + 8)  ◄── PEAK
 *     6. WAVE B:    8 ZMM (nx only)
 *
 *=========================================================================*/

#define DEFINE_PASS1_FUNCTIONS(DIR_SUFFIX, BUTTERFLY_FN)                     \
                                                                            \
/* ── U=1 fallback (K=8, single iteration) ────────────────────────── */    \
TARGET_AVX512                                                               \
static void radix4_dit_pass1_##DIR_SUFFIX##_avx512(                         \
    size_t K,                                                               \
    const double *RESTRICT in_re_base,                                      \
    const double *RESTRICT in_im_base,                                      \
    size_t in_stride,                                                       \
    double *RESTRICT temp_re,                                               \
    double *RESTRICT temp_im,                                               \
    size_t group,                                                           \
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT stage_tw)          \
{                                                                           \
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");          \
    assert(group < 8);                                                      \
                                                                            \
    const size_t step = 8;                                                  \
    const size_t s0 = 0 * 8 + group;                                        \
    const size_t s1 = 1 * 8 + group;                                        \
    const size_t s2 = 2 * 8 + group;                                        \
    const size_t s3 = 3 * 8 + group;                                        \
                                                                            \
    const double *RESTRICT tw_re = stage_tw->re;                            \
    const double *RESTRICT tw_im = stage_tw->im;                            \
                                                                            \
    for (size_t k = 0; k < K; k += step)                                    \
    {                                                                       \
        __m512d t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i;                    \
                                                                            \
        PASS1_LOAD_AND_TWIDDLE_512(                                         \
            in_re_base, in_im_base, in_stride, tw_re, tw_im, K, k,         \
            t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i);                       \
                                                                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                    \
        BUTTERFLY_FN(                                                       \
            t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,                        \
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);              \
                                                                            \
        _mm512_store_pd(&temp_re[s0 * K + k], y0r);                         \
        _mm512_store_pd(&temp_im[s0 * K + k], y0i);                         \
        _mm512_store_pd(&temp_re[s2 * K + k], y2r);                         \
        _mm512_store_pd(&temp_im[s2 * K + k], y2i);                         \
        _mm512_store_pd(&temp_re[s1 * K + k], y1r);                         \
        _mm512_store_pd(&temp_im[s1 * K + k], y1i);                         \
        _mm512_store_pd(&temp_re[s3 * K + k], y3r);                         \
        _mm512_store_pd(&temp_im[s3 * K + k], y3i);                         \
    }                                                                       \
}                                                                           \
                                                                            \
/* ── U=2 pipelined (K≥16, ≥2 iterations) ────────────────────────── */    \
TARGET_AVX512                                                               \
NO_UNROLL_LOOPS                                                             \
static void radix4_dit_pass1_##DIR_SUFFIX##_u2_avx512(                      \
    size_t K,                                                               \
    const double *RESTRICT in_re_base,                                      \
    const double *RESTRICT in_im_base,                                      \
    size_t in_stride,                                                       \
    double *RESTRICT temp_re,                                               \
    double *RESTRICT temp_im,                                               \
    size_t group,                                                           \
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT stage_tw)          \
{                                                                           \
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");          \
    assert(K >= 16 && "K must be >= 16 for U=2 (need 2+ iterations)");      \
    assert(group < 8);                                                      \
                                                                            \
    const size_t step = 8;                                                  \
    const size_t s0 = 0 * 8 + group;                                        \
    const size_t s1 = 1 * 8 + group;                                        \
    const size_t s2 = 2 * 8 + group;                                        \
    const size_t s3 = 3 * 8 + group;                                        \
                                                                            \
    const double *RESTRICT tw_re = stage_tw->re;                            \
    const double *RESTRICT tw_im = stage_tw->im;                            \
                                                                            \
    /*==================================================================    \
     * PROLOGUE: Load first iteration + prefetch next strided inputs        \
     *================================================================*/    \
    __m512d nt0r, nt0i, nt1r, nt1i, nt2r, nt2i, nt3r, nt3i;                \
                                                                            \
    PASS1_LOAD_AND_TWIDDLE_512(                                             \
        in_re_base, in_im_base, in_stride, tw_re, tw_im, K, 0,             \
        nt0r, nt0i, nt1r, nt1i, nt2r, nt2i, nt3r, nt3i);                   \
                                                                            \
    /* Prefetch 2nd iteration's strided inputs (HW can't predict stride) */ \
    PASS1_PREFETCH_INPUTS_512(in_re_base, in_im_base, in_stride, step);     \
                                                                            \
    /*==================================================================    \
     * STEADY-STATE: U=2 pipelined loop                                     \
     *================================================================*/    \
    _Pragma("GCC unroll 1")                                                 \
    for (size_t k = 0; k + step < K; k += step)                             \
    {                                                                       \
        const size_t kn = k + step;                                         \
                                                                            \
        /* 1. CONSUME: next → current (register rename, 0 cost) */          \
        __m512d t0r = nt0r, t0i = nt0i;                                     \
        __m512d t1r = nt1r, t1i = nt1i;                                     \
        __m512d t2r = nt2r, t2i = nt2i;                                     \
        __m512d t3r = nt3r, t3i = nt3i;                                     \
        /* Live: 8 ZMM (t0..t3 re+im) */                                   \
                                                                            \
        /* 2. BUTTERFLY */                                                  \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                    \
        BUTTERFLY_FN(                                                       \
            t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,                        \
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);              \
        /* Live: 8 ZMM (y0..y3 re+im), inputs dead */                      \
                                                                            \
        /* 3. WAVE A: store even bins y0, y2 → frees 4 ZMM */              \
        DIT4_STORE_WAVE_A_512(temp_re, temp_im, K, k, s0, s2,              \
            y0r, y0i, y2r, y2i);                                            \
        /* Live: 4 ZMM (y1,y3 re+im) */                                    \
                                                                            \
        /* 4. ►► U=2 INTERLEAVE: load+twiddle next iteration ◄◄  */        \
        /*    Strided loads issue while odd stores drain store buffer */     \
        PASS1_LOAD_AND_TWIDDLE_512(                                         \
            in_re_base, in_im_base, in_stride, tw_re, tw_im, K, kn,        \
            nt0r, nt0i, nt1r, nt1i, nt2r, nt2i, nt3r, nt3i);               \
        /* Live: 4 (y1,y3) + 8 (nt0..nt3) = 12 ZMM ← PEAK */             \
                                                                            \
        /* 5. WAVE B: store odd bins y1, y3 → frees 4 ZMM */               \
        DIT4_STORE_WAVE_B_512(temp_re, temp_im, K, k, s1, s3,              \
            y1r, y1i, y3r, y3i);                                            \
        /* Live: 8 ZMM (nt0..nt3) */                                       \
                                                                            \
        /* 6. Prefetch strided inputs 2 steps ahead */                      \
        if (kn + step < K) {                                                \
            PASS1_PREFETCH_INPUTS_512(in_re_base, in_im_base,               \
                                      in_stride, kn + step);                \
        }                                                                   \
    }                                                                       \
                                                                            \
    /*==================================================================    \
     * EPILOGUE: final iteration — compute + store, no next loads           \
     *================================================================*/    \
    {                                                                       \
        const size_t k = K - step;                                          \
                                                                            \
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                    \
        BUTTERFLY_FN(                                                       \
            nt0r, nt0i, nt1r, nt1i, nt2r, nt2i, nt3r, nt3i,                \
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);              \
                                                                            \
        _mm512_store_pd(&temp_re[s0 * K + k], y0r);                         \
        _mm512_store_pd(&temp_im[s0 * K + k], y0i);                         \
        _mm512_store_pd(&temp_re[s2 * K + k], y2r);                         \
        _mm512_store_pd(&temp_im[s2 * K + k], y2i);                         \
        _mm512_store_pd(&temp_re[s1 * K + k], y1r);                         \
        _mm512_store_pd(&temp_im[s1 * K + k], y1i);                         \
        _mm512_store_pd(&temp_re[s3 * K + k], y3r);                         \
        _mm512_store_pd(&temp_im[s3 * K + k], y3i);                         \
    }                                                                       \
}

/* ── Instantiate forward and backward variants ─────────────────────── */
DEFINE_PASS1_FUNCTIONS(forward,  radix4_dit_core_forward_avx512)
DEFINE_PASS1_FUNCTIONS(backward, radix4_dit_core_backward_avx512)

/*==========================================================================
 * WAVE A/B STORE MACROS (AVX-512)
 *
 * Split the two-wave store into separate macros for U=2 interleaving.
 * Between wave A and wave B, the caller inserts next-iteration loads.
 *
 * Wave A: even outputs {y0,y2,y4,y6} → frees 8 ZMM
 * Wave B: odd outputs  {y1,y3,y5,y7} → frees 8 ZMM
 *=========================================================================*/

#define DIF8_STORE_WAVE_A_512(out_re, out_im, K, k,                  \
                               y0r, y0i, y2r, y2i,                   \
                               y4r, y4i, y6r, y6i)                   \
    do {                                                             \
        _mm512_store_pd(&(out_re)[0 * (K) + (k)], y0r);              \
        _mm512_store_pd(&(out_im)[0 * (K) + (k)], y0i);              \
        _mm512_store_pd(&(out_re)[2 * (K) + (k)], y2r);              \
        _mm512_store_pd(&(out_im)[2 * (K) + (k)], y2i);              \
        _mm512_store_pd(&(out_re)[4 * (K) + (k)], y4r);              \
        _mm512_store_pd(&(out_im)[4 * (K) + (k)], y4i);              \
        _mm512_store_pd(&(out_re)[6 * (K) + (k)], y6r);              \
        _mm512_store_pd(&(out_im)[6 * (K) + (k)], y6i);              \
    } while (0)

#define DIF8_STORE_WAVE_B_512(out_re, out_im, K, k,                  \
                               y1r, y1i, y3r, y3i,                   \
                               y5r, y5i, y7r, y7i)                   \
    do {                                                             \
        _mm512_store_pd(&(out_re)[1 * (K) + (k)], y1r);              \
        _mm512_store_pd(&(out_im)[1 * (K) + (k)], y1i);              \
        _mm512_store_pd(&(out_re)[3 * (K) + (k)], y3r);              \
        _mm512_store_pd(&(out_im)[3 * (K) + (k)], y3i);              \
        _mm512_store_pd(&(out_re)[5 * (K) + (k)], y5r);              \
        _mm512_store_pd(&(out_im)[5 * (K) + (k)], y5i);              \
        _mm512_store_pd(&(out_re)[7 * (K) + (k)], y7r);              \
        _mm512_store_pd(&(out_im)[7 * (K) + (k)], y7i);              \
    } while (0)

/*==========================================================================
 * PASS 2: RADIX-8 DIF — U=2 SOFTWARE-PIPELINED (AVX-512, BLOCKED8)
 *
 * Overlaps next-iteration loads with current-iteration stores to hide
 * memory latency. On ICX/SPR where pass-2 working sets hit L2/LLC
 * (128K–512K per bin for K=256–4096), this hides 10-30 cycle load
 * latency behind the store pipeline.
 *
 * ┌──────────────────────────────────────────────────────────────┐
 * │  PROLOGUE: Load first inputs + twiddles into "next" vars    │
 * ├──────────────────────────────────────────────────────────────┤
 * │  STEADY-STATE LOOP (k = 0 to K-2·step):                    │
 * │    1. Consume: current = next              [rename, 0 cost] │
 * │    2. Compute: fused twiddle + DIF-8       [16 ZMM peak]    │
 * │    3. Wave A: store evens y0,y2,y4,y6      [frees 8 ZMM]   │
 * │  ► 4. U=2: load next inputs                [16 ZMM new]    │
 * │         ─── 24 ZMM live (8 odd y + 16 nx) ◄ fits in 32 ─── │
 * │    5. Wave B: store odds y1,y3,y5,y7       [frees 8 ZMM]   │
 * │    6. Load next twiddles                    [to stack]      │
 * ├──────────────────────────────────────────────────────────────┤
 * │  EPILOGUE: Last iteration — compute + store, no next loads  │
 * └──────────────────────────────────────────────────────────────┘
 *
 * Register budget at peak (step 4):
 *   8 ZMM  — odd outputs y1,y3,y5,y7 (awaiting wave B)
 *  16 ZMM  — next inputs nx0..nx7 (re+im)
 *   2 ZMM  — signbit mask + W8_C constant (shared, may be reloaded)
 *  ─────────
 *  26 ZMM  — 6 spare out of 32
 *
 * Twiddles: loaded to stack via load_tw_blocked8_k8, accessed through
 * the FORCE_INLINE fused function → compiler folds from stack into
 * cmul operands (sequential access pattern, no bulk register pressure).
 *=========================================================================*/

TARGET_AVX512
NO_UNROLL_LOOPS
static void radix8_dif_pass2_u2_blocked8_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked8_t *RESTRICT tw,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 (need 2+ iterations)");

    const size_t step = 8;

    /*==================================================================
     * PROLOGUE: Load first iteration's inputs + twiddles
     *================================================================*/
    __m512d nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i;
    __m512d nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i;

    DIF8_LOAD_INPUTS_512(in_re, in_im, K, 0,
        nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
        nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

    tw8_vecs_512_t ntw;
    load_tw_blocked8_k8(tw, 0, &ntw);

    /*==================================================================
     * STEADY-STATE: U=2 pipelined loop
     *
     * Each iteration: compute current butterfly while loading next inputs
     *================================================================*/
#pragma GCC unroll 1
    for (size_t k = 0; k + step < K; k += step)
    {
        const size_t kn = k + step;

        /*--------------------------------------------------------------
         * 1. CONSUME: Transfer "next" → "current" (register rename)
         *------------------------------------------------------------*/
        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        /*--------------------------------------------------------------
         * 2. COMPUTE: Fused twiddle-apply + DIF-8 butterfly
         *    Peak: 16 ZMM (butterfly intermediates reuse input slots)
         *------------------------------------------------------------*/
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }
        /* Live: 16 ZMM (y0..y7 re+im) */

        /*--------------------------------------------------------------
         * 3. WAVE A: Store even outputs y0,y2,y4,y6
         *    Frees 8 ZMM → 8 live (odd y), 24 available
         *------------------------------------------------------------*/
        DIF8_STORE_WAVE_A_512(out_re, out_im, K, k,
            y0r, y0i, y2r, y2i, y4r, y4i, y6r, y6i);
        /* Live: 8 ZMM (y1,y3,y5,y7 re+im) */

        /*--------------------------------------------------------------
         * 4. ►► U=2 INTERLEAVE: Load next iteration's inputs ◄◄
         *    16 loads into nx* while odd outputs still live
         *    Live: 8 (odd y) + 16 (nx) = 24 ZMM, 8 spare
         *------------------------------------------------------------*/
        DIF8_LOAD_INPUTS_512(in_re, in_im, K, kn,
            nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
            nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

        /*--------------------------------------------------------------
         * 5. WAVE B: Store odd outputs y1,y3,y5,y7
         *    Frees 8 ZMM → 16 live (nx only)
         *------------------------------------------------------------*/
        DIF8_STORE_WAVE_B_512(out_re, out_im, K, k,
            y1r, y1i, y3r, y3i, y5r, y5i, y7r, y7i);
        /* Live: 16 ZMM (nx0..nx7 re+im) */

        /*--------------------------------------------------------------
         * 6. Load next iteration's twiddles (to stack struct)
         *    Compiler manages spill/reload — sequential cmul access
         *    pattern in the fused function keeps effective pressure low
         *------------------------------------------------------------*/
        load_tw_blocked8_k8(tw, kn, &ntw);
    }

    /*==================================================================
     * EPILOGUE: Final iteration (no next loads needed)
     *================================================================*/
    {
        const size_t k = K - step;

        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        /* Full store (no interleave needed) */
        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, k,
            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
    }
}

/*==========================================================================
 * PASS 2: RADIX-8 DIF — U=1 FALLBACK (AVX-512, BLOCKED8)
 *
 * Simple sequential k-loop for K=8 (single iteration, U=2 not possible).
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_pass2_blocked8_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked8_t *RESTRICT tw,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");

    const size_t step = 8;

    for (size_t k = 0; k < K; k += step)
    {
        __m512d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        __m512d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, k,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

        tw8_vecs_512_t tvec;
        load_tw_blocked8_k8(tw, k, &tvec);

        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                tvec.r, tvec.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                tvec.r, tvec.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, k,
            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
    }
}

/*==========================================================================
 * PASS 2: RADIX-8 DIF — 4 BINS (AVX-512)
 *
 * Dispatches to U=2 (K≥16) or U=1 fallback (K=8).
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_pass2_all_bins_avx512(
    size_t K,
    const double *RESTRICT temp_re,
    const double *RESTRICT temp_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked8_t *RESTRICT tw,
    int direction)
{
    /* Select U=2 (K≥16, ≥2 iterations) or U=1 fallback (K=8) */
    void (*pass2_fn)(size_t, const double*, const double*,
                     double*, double*, const tw_blocked8_t*, int);

    if (K >= 16)
        pass2_fn = radix8_dif_pass2_u2_blocked8_avx512;
    else
        pass2_fn = radix8_dif_pass2_blocked8_avx512;

    for (size_t bin = 0; bin < 4; bin++)
    {
        const size_t off = bin * 8 * K;
        pass2_fn(K,
                 &temp_re[off], &temp_im[off],
                 &out_re[off],  &out_im[off],
                 tw, direction);
    }
}


/*==========================================================================
 * PASS 2: RADIX-8 DIF — U=2 SOFTWARE-PIPELINED (AVX-512, BLOCKED4)
 *
 * Identical butterfly logic to BLOCKED8 U=2, but loads W1..W4 from
 * memory and derives W5..W8 on-the-fly via load_tw_blocked4_k8.
 *
 * Bandwidth:  8 loads × 64B = 512B per k-step  (vs 1024B for B8)
 * Extra ops:  3 cmul + 1 csquare = 15 FMA-port ops per k-step
 *             (hidden behind load latency on ICX/SPR)
 *=========================================================================*/

TARGET_AVX512
NO_UNROLL_LOOPS
static void radix8_dif_pass2_u2_blocked4_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked4_t *RESTRICT tw,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(K >= 16 && "K must be >= 16 for U=2 (need 2+ iterations)");

    const size_t step = 8;

    /* PROLOGUE */
    __m512d nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i;
    __m512d nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i;

    DIF8_LOAD_INPUTS_512(in_re, in_im, K, 0,
        nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
        nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

    tw8_vecs_512_t ntw;
    load_tw_blocked4_k8(tw, 0, &ntw);

    /* STEADY-STATE: U=2 pipelined loop */
#pragma GCC unroll 1
    for (size_t k = 0; k + step < K; k += step)
    {
        const size_t kn = k + step;

        __m512d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m512d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m512d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m512d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;

        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        DIF8_STORE_WAVE_A_512(out_re, out_im, K, k,
            y0r, y0i, y2r, y2i, y4r, y4i, y6r, y6i);

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, kn,
            nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
            nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i);

        DIF8_STORE_WAVE_B_512(out_re, out_im, K, k,
            y1r, y1i, y3r, y3i, y5r, y5i, y7r, y7i);

        load_tw_blocked4_k8(tw, kn, &ntw);
    }

    /* EPILOGUE */
    {
        const size_t k = K - step;

        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                nx0r, nx0i, nx1r, nx1i, nx2r, nx2i, nx3r, nx3i,
                nx4r, nx4i, nx5r, nx5i, nx6r, nx6i, nx7r, nx7i,
                ntw.r, ntw.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, k,
            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
    }
}

/*==========================================================================
 * PASS 2: RADIX-8 DIF — U=1 FALLBACK (AVX-512, BLOCKED4)
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_pass2_blocked4_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked4_t *RESTRICT tw,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");

    const size_t step = 8;

    for (size_t k = 0; k < K; k += step)
    {
        __m512d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        __m512d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, k,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

        tw8_vecs_512_t tvec;
        load_tw_blocked4_k8(tw, k, &tvec);

        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

        if (direction == 0) {
            dif8_twiddle_and_butterfly_forward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                tvec.r, tvec.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        } else {
            dif8_twiddle_and_butterfly_backward_avx512(
                x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                tvec.r, tvec.i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        }

        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, k,
            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
            y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i);
    }
}

/*==========================================================================
 * PASS 2: RADIX-8 DIF — ALL BINS, BLOCKED4 (AVX-512)
 *=========================================================================*/

TARGET_AVX512
static void radix8_dif_pass2_all_bins_blocked4_avx512(
    size_t K,
    const double *RESTRICT temp_re,
    const double *RESTRICT temp_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const tw_blocked4_t *RESTRICT tw,
    int direction)
{
    void (*pass2_fn)(size_t, const double*, const double*,
                     double*, double*, const tw_blocked4_t*, int);

    if (K >= 16)
        pass2_fn = radix8_dif_pass2_u2_blocked4_avx512;
    else
        pass2_fn = radix8_dif_pass2_blocked4_avx512;

    for (size_t bin = 0; bin < 4; bin++)
    {
        const size_t off = bin * 8 * K;
        pass2_fn(K,
                 &temp_re[off], &temp_im[off],
                 &out_re[off],  &out_im[off],
                 tw, direction);
    }
}


/*==========================================================================
 * TOP-LEVEL RADIX-32 DRIVER (AVX-512, BLOCKED8)
 *
 * Complete 4x8 decomposition:
 *   Pass 1: 8 groups x radix-4 DIT (strided -> bin-major temp)
 *           Split forward/backward, U=2 for K>=16, U=1 fallback for K=8
 *   Pass 2: 4 bins x radix-8 DIF (temp -> output)
 *           U=2 for K>=16, U=1 fallback for K=8
 *
 * @param K         Samples per stripe (multiple of 8)
 * @param in_re     Input real [32 stripes][K], 64-byte aligned
 * @param in_im     Input imag [32 stripes][K], 64-byte aligned
 * @param out_re    Output real [32 stripes][K], 64-byte aligned
 * @param out_im    Output imag [32 stripes][K], 64-byte aligned
 * @param pass1_tw  Radix-4 DIT twiddles (BLOCKED2 layout)
 * @param pass2_tw  Radix-8 DIF twiddles (BLOCKED8 layout)
 * @param temp_re   Temp buffer real [32 stripes][K], 64-byte aligned
 * @param temp_im   Temp buffer imag [32 stripes][K], 64-byte aligned
 *=========================================================================*/

TARGET_AVX512
static void radix32_stage_forward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_blocked8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    void (*pass1_fn)(size_t, const double*, const double*, size_t,
                     double*, double*, size_t,
                     const radix4_dit_stage_twiddles_blocked2_t*);

    if (K >= 16)
        pass1_fn = radix4_dit_pass1_forward_u2_avx512;
    else
        pass1_fn = radix4_dit_pass1_forward_avx512;

    for (size_t group = 0; group < 8; group++)
    {
        pass1_fn(K,
                 &in_re[group * K],
                 &in_im[group * K],
                 in_stride,
                 temp_re, temp_im,
                 group,
                 pass1_tw);
    }

    radix8_dif_pass2_all_bins_avx512(
        K, temp_re, temp_im, out_re, out_im, pass2_tw, 0 /* forward */);
}

TARGET_AVX512
static void radix32_stage_backward_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_blocked8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    void (*pass1_fn)(size_t, const double*, const double*, size_t,
                     double*, double*, size_t,
                     const radix4_dit_stage_twiddles_blocked2_t*);

    if (K >= 16)
        pass1_fn = radix4_dit_pass1_backward_u2_avx512;
    else
        pass1_fn = radix4_dit_pass1_backward_avx512;

    for (size_t group = 0; group < 8; group++)
    {
        pass1_fn(K,
                 &in_re[group * K],
                 &in_im[group * K],
                 in_stride,
                 temp_re, temp_im,
                 group,
                 pass1_tw);
    }

    radix8_dif_pass2_all_bins_avx512(
        K, temp_re, temp_im, out_re, out_im, pass2_tw, 1 /* backward */);
}

/*==========================================================================
 * TOP-LEVEL RADIX-32 DRIVER (AVX-512, MULTI-MODE)
 *
 * Accepts tw_stage8_t and dispatches pass 2 to BLOCKED8 or BLOCKED4.
 * Pass 1 (radix-4 DIT) is mode-independent — only pass 2 differs.
 *=========================================================================*/

TARGET_AVX512
static void radix32_stage_forward_avx512_multi(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    /*==================================================================
     * PASS 1: Radix-4 DIT on 8 groups (mode-independent)
     *================================================================*/
    void (*pass1_fn)(size_t, const double*, const double*, size_t,
                     double*, double*, size_t,
                     const radix4_dit_stage_twiddles_blocked2_t*);

    if (K >= 16)
        pass1_fn = radix4_dit_pass1_forward_u2_avx512;
    else
        pass1_fn = radix4_dit_pass1_forward_avx512;

    for (size_t group = 0; group < 8; group++)
    {
        pass1_fn(K,
                 &in_re[group * K],
                 &in_im[group * K],
                 in_stride,
                 temp_re, temp_im,
                 group,
                 pass1_tw);
    }

    /*==================================================================
     * PASS 2: Radix-8 DIF on 4 bins (mode-dependent)
     *================================================================*/
    switch (pass2_tw->mode) {
    case TW_MODE_BLOCKED8:
        radix8_dif_pass2_all_bins_avx512(
            K, temp_re, temp_im, out_re, out_im,
            &pass2_tw->b8, 0);
        break;
    case TW_MODE_BLOCKED4:
        radix8_dif_pass2_all_bins_blocked4_avx512(
            K, temp_re, temp_im, out_re, out_im,
            &pass2_tw->b4, 0);
        break;
    default:
        assert(0 && "AVX-512 path does not support RECURRENCE mode");
        break;
    }
}

TARGET_AVX512
static void radix32_stage_backward_avx512_multi(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im)
{
    const size_t in_stride = 8 * K;

    void (*pass1_fn)(size_t, const double*, const double*, size_t,
                     double*, double*, size_t,
                     const radix4_dit_stage_twiddles_blocked2_t*);

    if (K >= 16)
        pass1_fn = radix4_dit_pass1_backward_u2_avx512;
    else
        pass1_fn = radix4_dit_pass1_backward_avx512;

    for (size_t group = 0; group < 8; group++)
    {
        pass1_fn(K,
                 &in_re[group * K],
                 &in_im[group * K],
                 in_stride,
                 temp_re, temp_im,
                 group,
                 pass1_tw);
    }

    switch (pass2_tw->mode) {
    case TW_MODE_BLOCKED8:
        radix8_dif_pass2_all_bins_avx512(
            K, temp_re, temp_im, out_re, out_im,
            &pass2_tw->b8, 1);
        break;
    case TW_MODE_BLOCKED4:
        radix8_dif_pass2_all_bins_blocked4_avx512(
            K, temp_re, temp_im, out_re, out_im,
            &pass2_tw->b4, 1);
        break;
    default:
        assert(0 && "AVX-512 path does not support RECURRENCE mode");
        break;
    }
}

#endif /* FFT_RADIX32_AVX512_H */