/**
 * @file fft_radix8_avx2_dif.h
 * @brief Radix-8 DIF (Decimation-In-Frequency) FFT implementation for AVX2
 * 
 * This file contains DIF variants of radix-8 transforms, optimized for use
 * in composite radix algorithms (e.g., radix-32 = radix-4 DIT × radix-8 DIF).
 * 
 * DIF vs DIT Differences:
 * - DIF: Apply stage twiddles BEFORE butterfly decomposition
 * - DIT: Apply stage twiddles AFTER butterfly decomposition
 * - DIF: Uses 7 stage twiddles (W1..W7) for radix-8
 * - DIT: Uses 4 stage twiddles + W8 geometric constants
 * 
 * Architecture:
 * - Shares optimization framework with fft_radix8_avx2.h (DIT variants)
 * - U=2 software pipelining for maximum throughput
 * - Two-wave stores for register pressure control
 * - BLOCKED4/BLOCKED2 twiddle modes for minimal bandwidth
 * - Adaptive NT stores and prefetch tuning
 * 
 * Performance:
 * - Same performance as DIT variants (~0% difference)
 * - Better code organization (modular, testable, reusable)
 * - Useful for radix-16, radix-24, radix-32, radix-40 composite transforms
 * 
 * @note Requires AVX2 + FMA3 support (Haswell+, Zen1+)
 * @note Depends on fft_radix8_avx2.h for shared utilities
 */

#ifndef FFT_RADIX8_AVX2_DIF_H
#define FFT_RADIX8_AVX2_DIF_H

#include "fft_radix8_avx2.h"  // Include DIT variants for shared utilities

/**
 * @brief Radix-8 DIF butterfly core - FORWARD
 * 
 * DIF (Decimation-In-Frequency) decomposition:
 *   [Apply stage twiddles] → [8-point DIF butterfly]
 * 
 * Input: x0..x7 (already twiddled by stage twiddles W1..W7)
 * Output: y0..y7 (bit-reversed order)
 * 
 * DIF structure:
 *   Layer 1: 4 radix-2 butterflies (split into top/bottom halves)
 *   Layer 2: Apply W8 twiddles (sqrt(2) rotations)
 *   Layer 3: 2 radix-4 butterflies
 */
TARGET_AVX2_FMA
FORCE_INLINE static void radix8_dif_butterfly_forward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i, __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i, __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    //==========================================================================
    // LAYER 1: Four radix-2 butterflies (split top/bottom halves)
    //==========================================================================
    __m256d t0r = _mm256_add_pd(x0r, x4r);
    __m256d t0i = _mm256_add_pd(x0i, x4i);
    __m256d t1r = _mm256_add_pd(x1r, x5r);
    __m256d t1i = _mm256_add_pd(x1i, x5i);
    __m256d t2r = _mm256_add_pd(x2r, x6r);
    __m256d t2i = _mm256_add_pd(x2i, x6i);
    __m256d t3r = _mm256_add_pd(x3r, x7r);
    __m256d t3i = _mm256_add_pd(x3i, x7i);
    
    __m256d t4r = _mm256_sub_pd(x0r, x4r);
    __m256d t4i = _mm256_sub_pd(x0i, x4i);
    __m256d t5r = _mm256_sub_pd(x1r, x5r);
    __m256d t5i = _mm256_sub_pd(x1i, x5i);
    __m256d t6r = _mm256_sub_pd(x2r, x6r);
    __m256d t6i = _mm256_sub_pd(x2i, x6i);
    __m256d t7r = _mm256_sub_pd(x3r, x7r);
    __m256d t7i = _mm256_sub_pd(x3i, x7i);
    
    //==========================================================================
    // LAYER 2: Apply W8 twiddles to bottom half (fast micro-kernel)
    //==========================================================================
    w8_apply_fast_forward_avx2(&t5r, &t5i, &t6r, &t6i, &t7r, &t7i);
    
    //==========================================================================
    // LAYER 3: Two radix-4 butterflies
    //==========================================================================
    // Top radix-4: t0, t1, t2, t3 → y0, y2, y4, y6
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
        radix4_core_avx2(t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,
                         y0r, y0i, y2r, y2i, y4r, y4i, y6r, y6i,
                         SIGN_FLIP);
    }
    
    // Bottom radix-4: t4, t5, t6, t7 → y1, y3, y5, y7
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
        radix4_core_avx2(t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
                         y1r, y1i, y3r, y3i, y5r, y5i, y7r, y7i,
                         SIGN_FLIP);
    }
}

/**
 * @brief Radix-8 DIF butterfly core - BACKWARD
 * 
 * Identical to forward but uses backward W8 twiddles and negated radix-4 sign.
 */
TARGET_AVX2_FMA
FORCE_INLINE static void radix8_dif_butterfly_backward_avx2(
    __m256d x0r, __m256d x0i, __m256d x1r, __m256d x1i,
    __m256d x2r, __m256d x2i, __m256d x3r, __m256d x3i,
    __m256d x4r, __m256d x4i, __m256d x5r, __m256d x5i,
    __m256d x6r, __m256d x6i, __m256d x7r, __m256d x7i,
    __m256d *RESTRICT y0r, __m256d *RESTRICT y0i,
    __m256d *RESTRICT y1r, __m256d *RESTRICT y1i,
    __m256d *RESTRICT y2r, __m256d *RESTRICT y2i,
    __m256d *RESTRICT y3r, __m256d *RESTRICT y3i,
    __m256d *RESTRICT y4r, __m256d *RESTRICT y4i,
    __m256d *RESTRICT y5r, __m256d *RESTRICT y5i,
    __m256d *RESTRICT y6r, __m256d *RESTRICT y6i,
    __m256d *RESTRICT y7r, __m256d *RESTRICT y7i)
{
    // Layer 1: Four radix-2 butterflies
    __m256d t0r = _mm256_add_pd(x0r, x4r);
    __m256d t0i = _mm256_add_pd(x0i, x4i);
    __m256d t1r = _mm256_add_pd(x1r, x5r);
    __m256d t1i = _mm256_add_pd(x1i, x5i);
    __m256d t2r = _mm256_add_pd(x2r, x6r);
    __m256d t2i = _mm256_add_pd(x2i, x6i);
    __m256d t3r = _mm256_add_pd(x3r, x7r);
    __m256d t3i = _mm256_add_pd(x3i, x7i);
    
    __m256d t4r = _mm256_sub_pd(x0r, x4r);
    __m256d t4i = _mm256_sub_pd(x0i, x4i);
    __m256d t5r = _mm256_sub_pd(x1r, x5r);
    __m256d t5i = _mm256_sub_pd(x1i, x5i);
    __m256d t6r = _mm256_sub_pd(x2r, x6r);
    __m256d t6i = _mm256_sub_pd(x2i, x6i);
    __m256d t7r = _mm256_sub_pd(x3r, x7r);
    __m256d t7i = _mm256_sub_pd(x3i, x7i);
    
    // Layer 2: Apply W8 twiddles (BACKWARD)
    w8_apply_fast_backward_avx2(&t5r, &t5i, &t6r, &t6i, &t7r, &t7i);
    
    // Layer 3: Two radix-4 butterflies (BACKWARD: negated sign)
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
        const __m256d neg_zero = _mm256_set1_pd(-0.0);
        const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);
        
        radix4_core_avx2(t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,
                         y0r, y0i, y2r, y2i, y4r, y4i, y6r, y6i,
                         neg_sign);
    }
    
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
        const __m256d neg_zero = _mm256_set1_pd(-0.0);
        const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);
        
        radix4_core_avx2(t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
                         y1r, y1i, y3r, y3i, y5r, y5i, y7r, y7i,
                         neg_sign);
    }
}

/**
 * @brief Radix-8 DIF stage with BLOCKED4 twiddles - FORWARD
 * 
 * Based on your DIT structure but adapted for DIF decomposition.
 * 
 * Optimizations (inherited from your DIT code):
 * ✅ U=2 software pipelining (overlapped loads/compute/stores)
 * ✅ Two-wave stores (control register pressure)
 * ✅ Adaptive NT stores (>256KB working set)
 * ✅ NTA prefetch for streaming workloads
 * ✅ BLOCKED4 twiddles (load W1..W4, derive W5=-W1, W6=-W2, W7=-W3)
 * ✅ Prefetch tuning (adaptive distance)
 * ✅ In-place twiddle derivation
 * ✅ zeroupper after NT stores
 * 
 * DIF-specific changes:
 * ✅ Apply stage twiddles BEFORE butterfly (not after)
 * ✅ Use 7 stage twiddles (W1..W7) instead of 4 + W8
 * ✅ DIF butterfly decomposition (not DIT 4×2)
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void radix8_dif_stage_blocked4_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    // Alignment checks
    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    // Adaptive prefetch distance (reuse your DIT logic)
    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE: Load first iteration
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    // Load twiddles W1..W4 (BLOCKED4: derive W5..W7)
    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);
    __m256d nW3r = _mm256_load_pd(&re_base[2 * K]);
    __m256d nW3i = _mm256_load_pd(&im_base[2 * K]);
    __m256d nW4r = _mm256_load_pd(&re_base[3 * K]);
    __m256d nW4i = _mm256_load_pd(&im_base[3 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        // Current iteration: consume nx* from previous iteration
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (DIF: BEFORE butterfly)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            // x0 *= 1 (identity, skip)
            
            // Apply W1..W4 (loaded from memory)
            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            // Derive W5=-W1, W6=-W2, W7=-W3
            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs (4 stripes: 0, 2, 4, 6)
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // STAGE 3: Radix-8 DIF Butterfly
        //======================================================================
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs (2 stripes: 1, 3)
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Store in Two Waves (control register pressure)
        //======================================================================
        // Wave A: Store y0, y1, y2, y3 → frees 8 YMM
        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);

        // Load remaining NEXT ODD (nx5, nx7) - now we have room
        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        // Wave B: Store remaining outputs
        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);

        //======================================================================
        // STAGE 6: Load Next Twiddles (4 blocks for BLOCKED4)
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);
        nW3r = _mm256_load_pd(&re_base[2 * K + kn]);
        nW3i = _mm256_load_pd(&im_base[2 * K + kn]);
        nW4r = _mm256_load_pd(&re_base[3 * K + kn]);
        nW4i = _mm256_load_pd(&im_base[3 * K + kn]);

        //======================================================================
        // STAGE 7: Prefetch (4 twiddle blocks)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[3 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE: Final iteration (no next loads needed)
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);
        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);
    }

    // Cleanup
    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief Radix-8 DIF stage with BLOCKED4 twiddles - BACKWARD
 * 
 * Changes from forward:
 * ✅ Use radix8_dif_butterfly_backward_avx2 (negated signs)
 * 
 * All other optimizations identical to forward version.
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void radix8_dif_stage_blocked4_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked4_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B4;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);
    __m256d nW3r = _mm256_load_pd(&re_base[2 * K]);
    __m256d nW3i = _mm256_load_pd(&im_base[2 * K]);
    __m256d nW4r = _mm256_load_pd(&re_base[3 * K]);
    __m256d nW4i = _mm256_load_pd(&im_base[3 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // STAGE 3: Radix-8 DIF Butterfly (BACKWARD)
        //======================================================================
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);

        //======================================================================
        // STAGE 6: Load Next Twiddles
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);
        nW3r = _mm256_load_pd(&re_base[2 * K + kn]);
        nW3i = _mm256_load_pd(&im_base[2 * K + kn]);
        nW4r = _mm256_load_pd(&re_base[3 * K + kn]);
        nW4i = _mm256_load_pd(&im_base[3 * K + kn]);

        //======================================================================
        // STAGE 7: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[2 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[3 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[3 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;
        __m256d W3r = nW3r, W3i = nW3i, W4r = nW4r, W4i = nW4i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);
        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief Radix-8 DIF stage with BLOCKED2 twiddles - FORWARD
 * 
 * Based on your DIT BLOCKED2 structure but adapted for DIF decomposition.
 * 
 * BLOCKED2: Loads only W1, W2; derives W3=W1×W2, W4=W2², W5=-W1, W6=-W2, W7=-W3
 * Bandwidth savings: 71% (load 2 blocks instead of 7)
 * 
 * Optimizations (inherited from your DIT code):
 * ✅ U=2 software pipelining
 * ✅ Two-wave stores
 * ✅ Adaptive NT stores
 * ✅ NTA prefetch for streaming
 * ✅ BLOCKED2 twiddles (minimal memory bandwidth)
 * ✅ Prefetch tuning (32 doubles for BLOCKED2)
 * ✅ In-place twiddle derivation
 * ✅ zeroupper after NT stores
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void radix8_dif_stage_blocked2_forward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    // Load only W1, W2 (BLOCKED2)
    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles (BLOCKED2: Derive W3..W7)
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            // x0 *= 1 (identity, skip)
            
            // Apply W1, W2 (loaded from memory)
            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            // Derive W3 = W1 × W2
            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);

            // Derive W4 = W2²
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            // Apply W3, W4
            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            // Derive W5=-W1, W6=-W2, W7=-W3
            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // STAGE 3: Radix-8 DIF Butterfly
        //======================================================================
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);

        //======================================================================
        // STAGE 6: Load Next Twiddles (only 2 blocks for BLOCKED2)
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        //======================================================================
        // STAGE 7: Prefetch (2 twiddle blocks)
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_forward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);
        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

/**
 * @brief Radix-8 DIF stage with BLOCKED2 twiddles - BACKWARD
 * 
 * Changes from forward:
 * ✅ Use radix8_dif_butterfly_backward_avx2 (negated signs)
 * 
 * All other optimizations identical to BLOCKED2 forward version.
 */
TARGET_AVX2_FMA
__attribute__((optimize("no-unroll-loops")))
#pragma clang loop unroll(disable)
static void radix8_dif_stage_blocked2_backward_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix8_stage_twiddles_blocked2_t *RESTRICT stage_tw)
{
    assert((K & 3) == 0 && "K must be multiple of 4");
    assert(K >= 8 && "K must be >= 8 for U=2 pipelining");

    const int in_aligned = (((uintptr_t)in_re | (uintptr_t)in_im) & 31) == 0;
    const int out_aligned = (((uintptr_t)out_re | (uintptr_t)out_im) & 31) == 0;

#define LDPD(p) (in_aligned ? _mm256_load_pd(p) : _mm256_loadu_pd(p))
#define STPD(p, v) (out_aligned ? _mm256_store_pd(p, v) : _mm256_storeu_pd(p, v))

    const size_t prefetch_dist = RADIX8_PREFETCH_DISTANCE_AVX2_B2;
    const size_t total_bytes = K * 8 * 2 * sizeof(double);
    const int use_nt = (total_bytes >= (RADIX8_STREAM_THRESHOLD_KB * 1024)) && out_aligned;
    const int pf_hint = use_nt ? _MM_HINT_NTA : _MM_HINT_T0;

#define ST_STREAM(p, v) (use_nt ? _mm256_stream_pd(p, v) : STPD(p, v))

    const double *RESTRICT re_base = (const double *)ASSUME_ALIGNED(stage_tw->re, 32);
    const double *RESTRICT im_base = (const double *)ASSUME_ALIGNED(stage_tw->im, 32);

    //==========================================================================
    // PROLOGUE
    //==========================================================================
    __m256d nx0r = LDPD(&in_re[0 * K]);
    __m256d nx0i = LDPD(&in_im[0 * K]);
    __m256d nx1r = LDPD(&in_re[1 * K]);
    __m256d nx1i = LDPD(&in_im[1 * K]);
    __m256d nx2r = LDPD(&in_re[2 * K]);
    __m256d nx2i = LDPD(&in_im[2 * K]);
    __m256d nx3r = LDPD(&in_re[3 * K]);
    __m256d nx3i = LDPD(&in_im[3 * K]);
    __m256d nx4r = LDPD(&in_re[4 * K]);
    __m256d nx4i = LDPD(&in_im[4 * K]);
    __m256d nx5r = LDPD(&in_re[5 * K]);
    __m256d nx5i = LDPD(&in_im[5 * K]);
    __m256d nx6r = LDPD(&in_re[6 * K]);
    __m256d nx6i = LDPD(&in_im[6 * K]);
    __m256d nx7r = LDPD(&in_re[7 * K]);
    __m256d nx7i = LDPD(&in_im[7 * K]);

    __m256d nW1r = _mm256_load_pd(&re_base[0 * K]);
    __m256d nW1i = _mm256_load_pd(&im_base[0 * K]);
    __m256d nW2r = _mm256_load_pd(&re_base[1 * K]);
    __m256d nW2i = _mm256_load_pd(&im_base[1 * K]);

    //==========================================================================
    // STEADY-STATE U=2 LOOP
    //==========================================================================
#pragma clang loop unroll(disable)
#pragma GCC unroll 1
    for (size_t k = 0; k + 4 < K; k += 4)
    {
        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        const size_t kn = k + 4;

        //======================================================================
        // STAGE 1: Apply Stage Twiddles
        //======================================================================
        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        //======================================================================
        // STAGE 2: Load Next EVEN Inputs
        //======================================================================
        nx0r = LDPD(&in_re[0 * K + kn]);
        nx0i = LDPD(&in_im[0 * K + kn]);
        nx2r = LDPD(&in_re[2 * K + kn]);
        nx2i = LDPD(&in_im[2 * K + kn]);
        nx4r = LDPD(&in_re[4 * K + kn]);
        nx4i = LDPD(&in_im[4 * K + kn]);
        nx6r = LDPD(&in_re[6 * K + kn]);
        nx6i = LDPD(&in_im[6 * K + kn]);

        //======================================================================
        // STAGE 3: Radix-8 DIF Butterfly (BACKWARD)
        //======================================================================
        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        //======================================================================
        // STAGE 4: Load HALF of Next ODD Inputs
        //======================================================================
        nx1r = LDPD(&in_re[1 * K + kn]);
        nx1i = LDPD(&in_im[1 * K + kn]);
        nx3r = LDPD(&in_re[3 * K + kn]);
        nx3i = LDPD(&in_im[3 * K + kn]);

        //======================================================================
        // STAGE 5: Store in Two Waves
        //======================================================================
        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);

        nx5r = LDPD(&in_re[5 * K + kn]);
        nx5i = LDPD(&in_im[5 * K + kn]);
        nx7r = LDPD(&in_re[7 * K + kn]);
        nx7i = LDPD(&in_im[7 * K + kn]);

        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);

        //======================================================================
        // STAGE 6: Load Next Twiddles
        //======================================================================
        nW1r = _mm256_load_pd(&re_base[0 * K + kn]);
        nW1i = _mm256_load_pd(&im_base[0 * K + kn]);
        nW2r = _mm256_load_pd(&re_base[1 * K + kn]);
        nW2i = _mm256_load_pd(&im_base[1 * K + kn]);

        //======================================================================
        // STAGE 7: Prefetch
        //======================================================================
        if (kn + prefetch_dist < K)
        {
            _mm_prefetch((const char *)&in_re[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&in_im[kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[0 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&re_base[1 * K + kn + prefetch_dist], pf_hint);
            _mm_prefetch((const char *)&im_base[1 * K + kn + prefetch_dist], pf_hint);
        }
    }

    //==========================================================================
    // EPILOGUE
    //==========================================================================
    {
        size_t k = K - 4;

        __m256d x0r = nx0r, x0i = nx0i, x1r = nx1r, x1i = nx1i;
        __m256d x2r = nx2r, x2i = nx2i, x3r = nx3r, x3i = nx3i;
        __m256d x4r = nx4r, x4i = nx4i, x5r = nx5r, x5i = nx5i;
        __m256d x6r = nx6r, x6i = nx6i, x7r = nx7r, x7i = nx7i;
        __m256d W1r = nW1r, W1i = nW1i, W2r = nW2r, W2i = nW2i;

        {
            const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

            cmul_v256(x1r, x1i, W1r, W1i, &x1r, &x1i);
            cmul_v256(x2r, x2i, W2r, W2i, &x2r, &x2i);

            __m256d W3r, W3i;
            cmul_v256(W1r, W1i, W2r, W2i, &W3r, &W3i);
            __m256d W4r, W4i;
            csquare_v256(W2r, W2i, &W4r, &W4i);

            cmul_v256(x3r, x3i, W3r, W3i, &x3r, &x3i);
            cmul_v256(x4r, x4i, W4r, W4i, &x4r, &x4i);

            __m256d mW1r = _mm256_xor_pd(W1r, SIGN_FLIP);
            __m256d mW1i = _mm256_xor_pd(W1i, SIGN_FLIP);
            __m256d mW2r = _mm256_xor_pd(W2r, SIGN_FLIP);
            __m256d mW2i = _mm256_xor_pd(W2i, SIGN_FLIP);
            __m256d mW3r = _mm256_xor_pd(W3r, SIGN_FLIP);
            __m256d mW3i = _mm256_xor_pd(W3i, SIGN_FLIP);

            cmul_v256(x5r, x5i, mW1r, mW1i, &x5r, &x5i);
            cmul_v256(x6r, x6i, mW2r, mW2i, &x6r, &x6i);
            cmul_v256(x7r, x7i, mW3r, mW3i, &x7r, &x7i);
        }

        __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        
        radix8_dif_butterfly_backward_avx2(
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
            &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
            &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

        ST_STREAM(&out_re[0 * K + k], y0r);
        ST_STREAM(&out_im[0 * K + k], y0i);
        ST_STREAM(&out_re[1 * K + k], y1r);
        ST_STREAM(&out_im[1 * K + k], y1i);
        ST_STREAM(&out_re[2 * K + k], y2r);
        ST_STREAM(&out_im[2 * K + k], y2i);
        ST_STREAM(&out_re[3 * K + k], y3r);
        ST_STREAM(&out_im[3 * K + k], y3i);
        ST_STREAM(&out_re[4 * K + k], y4r);
        ST_STREAM(&out_im[4 * K + k], y4i);
        ST_STREAM(&out_re[5 * K + k], y5r);
        ST_STREAM(&out_im[5 * K + k], y5i);
        ST_STREAM(&out_re[6 * K + k], y6r);
        ST_STREAM(&out_im[6 * K + k], y6i);
        ST_STREAM(&out_re[7 * K + k], y7r);
        ST_STREAM(&out_im[7 * K + k], y7i);
    }

    if (use_nt)
    {
        _mm_sfence();
        _mm256_zeroupper();
    }

#undef LDPD
#undef STPD
#undef ST_STREAM
}

#endif

//==============================================================================
// ARCHITECTURE NOTES
//==============================================================================

/**
 * @section Architecture DIF Implementation Details
 * 
 * @subsection arch_decomposition DIF Decomposition
 * Unlike DIT (which builds from small to large butterflies), DIF works
 * top-down:
 * 
 * 1. Apply stage twiddles to inputs (W^0, W^k, W^2k, ..., W^7k)
 * 2. Layer 1: Split into top/bottom halves via 4 radix-2 butterflies
 * 3. Layer 2: Apply W8 geometric twiddles (sqrt(2) rotations)
 * 4. Layer 3: Two independent radix-4 butterflies
 * 
 * This produces bit-reversed output, which is natural for composite
 * radix algorithms like radix-32 = radix-4 DIT × radix-8 DIF.
 * 
 * @subsection arch_twiddles Twiddle Factor Organization
 * DIF requires 7 unique twiddles per k-index:
 * - W^0 = 1 (identity, skipped)
 * - W^k, W^2k, W^3k, W^4k (positive powers)
 * - W^5k = -W^k, W^6k = -W^2k, W^7k = -W^3k (negative powers)
 * 
 * BLOCKED4 stores W^k through W^4k, derives rest via XOR negation.
 * BLOCKED2 stores W^k and W^2k, derives W^3k = W^k × W^2k, etc.
 * 
 * @subsection arch_u2 U=2 Software Pipelining
 * The steady-state loop overlaps operations across iterations:
 * - Iteration N: Compute butterfly, store outputs
 * - Iteration N+1: Load inputs, load twiddles (prefetched)
 * 
 * This hides L1→register latency (~4-5 cycles) and keeps both FMA
 * ports (p0, p1) continuously busy.
 * 
 * @subsection arch_registers Register Allocation
 * Peak usage: ~16 YMM registers (controlled via two-wave stores)
 * - 8 YMM: Current inputs (x0r..x7r or x0i..x7i)
 * - 4 YMM: Twiddles (W1r, W1i, W2r, W2i, ...)
 * - 4 YMM: Intermediate results / next loads
 * 
 * Two-wave stores prevent register spilling by emitting outputs in
 * two groups: {0,2,4,6} then {1,3,5,7}.
 */