/**
 * @file fft_radix32_avx512.h
 * @brief AVX-512 Radix-32 FFT Stage — U=1 Fused Single-Pass Driver
 *
 * Two-pass decomposition: 4×DIT → temp → 8×DIF
 *   Pass 1: 8 groups of radix-4 DIT, strided input → bin-major temp
 *   Pass 2: 4 bins of radix-8 DIF, temp → output
 *
 * This is the CORRECTNESS BASELINE — simple sequential k-loop at step=8,
 * no software pipelining, no prefetch, no NT stores. BLOCKED8 only.
 * U=2 pipelining and BLOCKED4/RECURRENCE modes will layer on top.
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
 * PASS 1: RADIX-4 DIT — STRIDED INPUT, BIN-MAJOR OUTPUT (AVX-512)
 *
 * Processes one group (4 stripes at stride 8*K) through:
 *   1. Load 4 inputs from strided layout
 *   2. Apply BLOCKED2 twiddles (W1, W2 from memory; W3 = W1×W2 derived)
 *   3. Radix-4 DIT butterfly
 *   4. Store to bin-major temp: bin b → temp stripe (b*8 + group)
 *
 * Simple sequential k-loop at step=8. No pipelining.
 *
 * Register usage: ~14 ZMM peak
 *   4 inputs (x0..x3 re+im) = 8 ZMM
 *   2 twiddles (W1,W2 re+im) = 4 ZMM  (W3 derived in scratch)
 *   4 outputs (y0..y3 re+im) = reuse input slots
 *   scratch for cmul = 2 ZMM
 *=========================================================================*/

TARGET_AVX512
static void radix4_dit_pass1_avx512(
    size_t K,
    const double *RESTRICT in_re_base,
    const double *RESTRICT in_im_base,
    size_t in_stride,
    double *RESTRICT temp_re,
    double *RESTRICT temp_im,
    size_t group,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT stage_tw,
    int direction)
{
    assert((K & 7) == 0 && "K must be multiple of 8 for AVX-512");
    assert(group < 8);

    const size_t step = 8;

    /* Bin-major output stripe indices */
    const size_t s0 = 0 * 8 + group;  /* Bin 0 */
    const size_t s1 = 1 * 8 + group;  /* Bin 1 */
    const size_t s2 = 2 * 8 + group;  /* Bin 2 */
    const size_t s3 = 3 * 8 + group;  /* Bin 3 */

    const double *RESTRICT tw_re = stage_tw->re;
    const double *RESTRICT tw_im = stage_tw->im;

    for (size_t k = 0; k < K; k += step)
    {
        /*==================================================================
         * Load 4 inputs from strided layout
         *================================================================*/
        __m512d x0r = _mm512_load_pd(&in_re_base[0 * in_stride + k]);
        __m512d x0i = _mm512_load_pd(&in_im_base[0 * in_stride + k]);
        __m512d x1r = _mm512_load_pd(&in_re_base[1 * in_stride + k]);
        __m512d x1i = _mm512_load_pd(&in_im_base[1 * in_stride + k]);
        __m512d x2r = _mm512_load_pd(&in_re_base[2 * in_stride + k]);
        __m512d x2i = _mm512_load_pd(&in_im_base[2 * in_stride + k]);
        __m512d x3r = _mm512_load_pd(&in_re_base[3 * in_stride + k]);
        __m512d x3i = _mm512_load_pd(&in_im_base[3 * in_stride + k]);

        /*==================================================================
         * Apply BLOCKED2 twiddles: W1, W2 from memory; W3 = W1×W2
         *================================================================*/
        __m512d W1r = _mm512_load_pd(&tw_re[0 * K + k]);
        __m512d W1i = _mm512_load_pd(&tw_im[0 * K + k]);
        __m512d W2r = _mm512_load_pd(&tw_re[1 * K + k]);
        __m512d W2i = _mm512_load_pd(&tw_im[1 * K + k]);

        /* x1 *= W1 */
        __m512d t1r, t1i;
        cmul_v512(x1r, x1i, W1r, W1i, &t1r, &t1i);

        /* x2 *= W2 */
        __m512d t2r, t2i;
        cmul_v512(x2r, x2i, W2r, W2i, &t2r, &t2i);

        /* W3 = W1 × W2 (derived), then x3 *= W3 */
        __m512d W3r, W3i;
        cmul_v512(W1r, W1i, W2r, W2i, &W3r, &W3i);
        __m512d t3r, t3i;
        cmul_v512(x3r, x3i, W3r, W3i, &t3r, &t3i);

        /*==================================================================
         * Radix-4 DIT butterfly
         *================================================================*/
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;

        if (direction == 0) {
            radix4_dit_core_forward_avx512(
                x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        } else {
            radix4_dit_core_backward_avx512(
                x0r, x0i, t1r, t1i, t2r, t2i, t3r, t3i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i);
        }

        /*==================================================================
         * Store bin-major: two-wave (even bins 0,2 then odd bins 1,3)
         *================================================================*/
        _mm512_store_pd(&temp_re[s0 * K + k], y0r);
        _mm512_store_pd(&temp_im[s0 * K + k], y0i);
        _mm512_store_pd(&temp_re[s2 * K + k], y2r);
        _mm512_store_pd(&temp_im[s2 * K + k], y2i);

        _mm512_store_pd(&temp_re[s1 * K + k], y1r);
        _mm512_store_pd(&temp_im[s1 * K + k], y1i);
        _mm512_store_pd(&temp_re[s3 * K + k], y3r);
        _mm512_store_pd(&temp_im[s3 * K + k], y3i);
    }
}

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
 * TOP-LEVEL RADIX-32 DRIVER (AVX-512, U=1, BLOCKED8)
 *
 * Complete 4×8 decomposition:
 *   Pass 1: 8 groups × radix-4 DIT (strided → bin-major temp)
 *   Pass 2: 4 bins × radix-8 DIF (temp → output)
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

    /*==================================================================
     * PASS 1: Radix-4 DIT on 8 groups
     *================================================================*/
    for (size_t group = 0; group < 8; group++)
    {
        radix4_dit_pass1_avx512(
            K,
            &in_re[group * K],
            &in_im[group * K],
            in_stride,
            temp_re, temp_im,
            group,
            pass1_tw,
            0 /* forward */);
    }

    /*==================================================================
     * PASS 2: Radix-8 DIF on 4 bins
     *================================================================*/
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

    for (size_t group = 0; group < 8; group++)
    {
        radix4_dit_pass1_avx512(
            K,
            &in_re[group * K],
            &in_im[group * K],
            in_stride,
            temp_re, temp_im,
            group,
            pass1_tw,
            1 /* backward */);
    }

    radix8_dif_pass2_all_bins_avx512(
        K, temp_re, temp_im, out_re, out_im, pass2_tw, 1 /* backward */);
}

#endif /* FFT_RADIX32_AVX512_H */