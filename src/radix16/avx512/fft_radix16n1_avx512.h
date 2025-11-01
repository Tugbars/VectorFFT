/**
 * @file fft_radix16_avx512_native_soa_n1.h
 * @brief Radix-16 AVX512 Native SoA - N=1 (No Twiddles) Optimized Version
 *
 * @details
 * FIRST STAGE OPTIMIZATION: When all twiddle factors = 1+0i
 * - No twiddle loads or multiplications
 * - Pure butterfly operations only
 * - Maximum throughput: ~2-3× faster than general case
 * - Reuses all infrastructure from twiddle version
 * 
 * WHEN TO USE:
 * - First FFT stage (K = N/16, all W^0 = 1)
 * - Small transforms where planning overhead matters
 * - Any radix-16 stage with unity twiddles
 *
 * @version 7.0-AVX512-N1
 * @date 2025
 */

#ifndef FFT_RADIX16_AVX512_NATIVE_SOA_N1_H
#define FFT_RADIX16_AVX512_NATIVE_SOA_N1_H

// Include base optimized version for infrastructure reuse
#include "fft_radix16_avx512_native_soa_optimized.h"

//==============================================================================
// N=1 STAGE DRIVERS (NO TWIDDLES - MAXIMUM THROUGHPUT)
//==============================================================================

/**
 * @brief Radix-16 N=1 Forward - AVX512 with U=4 unrolling
 * 
 * PERFORMANCE NOTES:
 * - 2-3× faster than twiddle case due to no cmuls
 * - U=4 unrolling (k+=32) preserved for maximum throughput
 * - Software pipelining still beneficial for load/store overlap
 * - Memory bandwidth often becomes bottleneck here
 * 
 * OPTIMIZATIONS PRESERVED:
 * - OPT #3: U=4 main loop
 * - OPT #2/#18: Prefetch with distance tuning
 * - OPT #6/#25: NT stores + cache flush
 * - OPT #13: 2× unrolled loads/stores
 * - OPT #14: Planner hints
 * - OPT #17: Butterfly register fusion
 * - OPT #20: Cache-aware tiling
 */
TARGET_AVX512
FORCE_INLINE void radix16_stage_dit_forward_n1_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_planner_hints_avx512_t *hints)
{
    const __m512d rot_sign_mask = kRotSignFwd_avx512;
    const size_t prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t tile_size = radix16_choose_tile_size_avx512(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx512(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        // =====================================================================
        // U=4 MAIN LOOP: Process 4 butterflies per iteration (32 doubles)
        // NO TWIDDLES = Pure load → butterfly → store pipeline
        // =====================================================================
        size_t k;
        for (k = k_tile; k + 32 <= k_end; k += 32)
        {
            size_t k_next = k + 32 + prefetch_dist;

            // Prefetch next iteration (input only - no twiddles!)
            if (k_next < k_end)
            {
                for (int _r = 0; _r < 16; _r++)
                {
                    _mm_prefetch((const char *)&in_re_aligned[k_next + _r * K], _MM_HINT_T0);
                    _mm_prefetch((const char *)&in_im_aligned[k_next + _r * K], _MM_HINT_T0);
                }

                // OPT #18 - Gate output prefetch
                if (!use_nt_stores && !is_inplace)
                {
                    for (int _r = 0; _r < 8; _r++)
                    {
                        _mm_prefetch((const char *)&out_re_aligned[k_next + _r * K], _MM_HINT_T0);
                        _mm_prefetch((const char *)&out_im_aligned[k_next + _r * K], _MM_HINT_T0);
                    }
                }
            }

            // SOFTWARE PIPELINE: Load all 4 butterflies early
            __m512d x0_re[16], x0_im[16];
            __m512d x1_re[16], x1_im[16];
            __m512d x2_re[16], x2_im[16];
            __m512d x3_re[16], x3_im[16];

            load_16_lanes_soa_avx512(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_avx512(k + 8, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
            load_16_lanes_soa_avx512(k + 16, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
            load_16_lanes_soa_avx512(k + 24, K, in_re_aligned, in_im_aligned, x3_re, x3_im);

            // NO TWIDDLES - Direct to butterfly!
            // This is the key simplification - saves 15 cmuls per butterfly
            __m512d y0_re[16], y0_im[16];
            __m512d y1_re[16], y1_im[16];
            __m512d y2_re[16], y2_im[16];
            __m512d y3_re[16], y3_im[16];

            radix16_complete_butterfly_forward_fused_soa_avx512(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
            radix16_complete_butterfly_forward_fused_soa_avx512(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
            radix16_complete_butterfly_forward_fused_soa_avx512(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
            radix16_complete_butterfly_forward_fused_soa_avx512(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);

            // Store all 4 results
            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512_stream(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
                store_16_lanes_soa_avx512_stream(k + 16, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
                store_16_lanes_soa_avx512_stream(k + 24, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
                store_16_lanes_soa_avx512(k + 16, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
                store_16_lanes_soa_avx512(k + 24, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
            }
        }

        // TAIL LOOP #1: k+16 (U=2)
        for (; k + 16 <= k_end; k += 16)
        {
            __m512d x0_re[16], x0_im[16];
            __m512d x1_re[16], x1_im[16];

            load_16_lanes_soa_avx512(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_avx512(k + 8, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            __m512d y0_re[16], y0_im[16];
            __m512d y1_re[16], y1_im[16];

            radix16_complete_butterfly_forward_fused_soa_avx512(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
            radix16_complete_butterfly_forward_fused_soa_avx512(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512_stream(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        // TAIL LOOP #2: k+8 (U=1)
        for (; k + 8 <= k_end; k += 8)
        {
            __m512d x_re[16], x_im[16];
            load_16_lanes_soa_avx512(k, K, in_re_aligned, in_im_aligned, x_re, x_im);

            __m512d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx512(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        // TAIL LOOP #3: Masked tail (1-7 remaining elements)
        if (k < k_end)
        {
            size_t remaining = k_end - k;
            __mmask8 mask = radix16_get_tail_mask_avx512(remaining);

            __m512d x_re[16], x_im[16];
            for (int r = 0; r < 16; r++)
            {
                x_re[r] = _mm512_maskz_load_pd(mask, &in_re_aligned[k + r * K]);
                x_im[r] = _mm512_maskz_load_pd(mask, &in_im_aligned[k + r * K]);
            }

            __m512d y_re[16], y_im[16];
            radix16_complete_butterfly_forward_fused_soa_avx512(x_re, x_im, y_re, y_im, rot_sign_mask);

            for (int r = 0; r < 16; r++)
            {
                _mm512_mask_store_pd(&out_re_aligned[k + r * K], mask, y_re[r]);
                _mm512_mask_store_pd(&out_im_aligned[k + r * K], mask, y_im[r]);
            }
        }
    }

    // OPT #25 - Cache flush after NT stores
    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
        {
            radix16_flush_output_cache_lines_avx512(K, out_re, out_im, true);
        }
    }
}

/**
 * @brief Radix-16 N=1 Backward - AVX512 with U=4 unrolling
 */
TARGET_AVX512
FORCE_INLINE void radix16_stage_dit_backward_n1_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix16_planner_hints_avx512_t *hints)
{
    const __m512d rot_sign_mask = kRotSignBwd_avx512;
    const size_t prefetch_dist = RADIX16_PREFETCH_DISTANCE;
    const size_t tile_size = radix16_choose_tile_size_avx512(K);

    const bool use_nt_stores = radix16_should_use_nt_stores_avx512(
        K, in_re, in_im, out_re, out_im, hints);
    const bool is_inplace = (hints != NULL && hints->in_place);

    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 64);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 64);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 64);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 64);

    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        size_t k;
        for (k = k_tile; k + 32 <= k_end; k += 32)
        {
            size_t k_next = k + 32 + prefetch_dist;

            if (k_next < k_end)
            {
                for (int _r = 0; _r < 16; _r++)
                {
                    _mm_prefetch((const char *)&in_re_aligned[k_next + _r * K], _MM_HINT_T0);
                    _mm_prefetch((const char *)&in_im_aligned[k_next + _r * K], _MM_HINT_T0);
                }

                if (!use_nt_stores && !is_inplace)
                {
                    for (int _r = 0; _r < 8; _r++)
                    {
                        _mm_prefetch((const char *)&out_re_aligned[k_next + _r * K], _MM_HINT_T0);
                        _mm_prefetch((const char *)&out_im_aligned[k_next + _r * K], _MM_HINT_T0);
                    }
                }
            }

            __m512d x0_re[16], x0_im[16];
            __m512d x1_re[16], x1_im[16];
            __m512d x2_re[16], x2_im[16];
            __m512d x3_re[16], x3_im[16];

            load_16_lanes_soa_avx512(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_avx512(k + 8, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
            load_16_lanes_soa_avx512(k + 16, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
            load_16_lanes_soa_avx512(k + 24, K, in_re_aligned, in_im_aligned, x3_re, x3_im);

            __m512d y0_re[16], y0_im[16];
            __m512d y1_re[16], y1_im[16];
            __m512d y2_re[16], y2_im[16];
            __m512d y3_re[16], y3_im[16];

            radix16_complete_butterfly_backward_fused_soa_avx512(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
            radix16_complete_butterfly_backward_fused_soa_avx512(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
            radix16_complete_butterfly_backward_fused_soa_avx512(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
            radix16_complete_butterfly_backward_fused_soa_avx512(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512_stream(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
                store_16_lanes_soa_avx512_stream(k + 16, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
                store_16_lanes_soa_avx512_stream(k + 24, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
                store_16_lanes_soa_avx512(k + 16, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
                store_16_lanes_soa_avx512(k + 24, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
            }
        }

        for (; k + 16 <= k_end; k += 16)
        {
            __m512d x0_re[16], x0_im[16];
            __m512d x1_re[16], x1_im[16];

            load_16_lanes_soa_avx512(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
            load_16_lanes_soa_avx512(k + 8, K, in_re_aligned, in_im_aligned, x1_re, x1_im);

            __m512d y0_re[16], y0_im[16];
            __m512d y1_re[16], y1_im[16];

            radix16_complete_butterfly_backward_fused_soa_avx512(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
            radix16_complete_butterfly_backward_fused_soa_avx512(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512_stream(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
                store_16_lanes_soa_avx512(k + 8, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
            }
        }

        for (; k + 8 <= k_end; k += 8)
        {
            __m512d x_re[16], x_im[16];
            load_16_lanes_soa_avx512(k, K, in_re_aligned, in_im_aligned, x_re, x_im);

            __m512d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx512(x_re, x_im, y_re, y_im, rot_sign_mask);

            if (use_nt_stores)
            {
                store_16_lanes_soa_avx512_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
            else
            {
                store_16_lanes_soa_avx512(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
            }
        }

        if (k < k_end)
        {
            size_t remaining = k_end - k;
            __mmask8 mask = radix16_get_tail_mask_avx512(remaining);

            __m512d x_re[16], x_im[16];
            for (int r = 0; r < 16; r++)
            {
                x_re[r] = _mm512_maskz_load_pd(mask, &in_re_aligned[k + r * K]);
                x_im[r] = _mm512_maskz_load_pd(mask, &in_im_aligned[k + r * K]);
            }

            __m512d y_re[16], y_im[16];
            radix16_complete_butterfly_backward_fused_soa_avx512(x_re, x_im, y_re, y_im, rot_sign_mask);

            for (int r = 0; r < 16; r++)
            {
                _mm512_mask_store_pd(&out_re_aligned[k + r * K], mask, y_re[r]);
                _mm512_mask_store_pd(&out_im_aligned[k + r * K], mask, y_im[r]);
            }
        }
    }

    if (use_nt_stores)
    {
        _mm_sfence();
        if (hints != NULL && hints->is_last_stage)
        {
            radix16_flush_output_cache_lines_avx512(K, out_re, out_im, true);
        }
    }
}

//==============================================================================
// PUBLIC API - N=1 VERSION
//==============================================================================

/**
 * @brief Radix-16 N=1 Forward Stage - Public API
 * 
 * WHEN TO USE:
 * - First FFT stage where K = N/16 and all twiddles are W^0 = 1+0i
 * - Any stage with unity twiddle factors
 * 
 * PERFORMANCE:
 * - 2-3× faster than general twiddle case
 * - Memory bandwidth often becomes bottleneck
 * - Excellent for small-to-medium K where arithmetic dominates
 */
TARGET_AVX512
void radix16_stage_dit_forward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_planner_hints_avx512_t *hints)
{
    radix16_set_ftz_daz_avx512();
    radix16_stage_dit_forward_n1_avx512(K, in_re, in_im, out_re, out_im, hints);
}

/**
 * @brief Radix-16 N=1 Backward Stage - Public API
 */
TARGET_AVX512
void radix16_stage_dit_backward_n1_soa_avx512(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const radix16_planner_hints_avx512_t *hints)
{
    radix16_set_ftz_daz_avx512();
    radix16_stage_dit_backward_n1_avx512(K, in_re, in_im, out_re, out_im, hints);
}

#endif // FFT_RADIX16_AVX512_NATIVE_SOA_N1_H

/*
 * ============================================================================
 * N=1 PERFORMANCE CHARACTERISTICS
 * ============================================================================
 *
 * SPEEDUP VS TWIDDLE VERSION: 2-3×
 * - No twiddle loads (saves 15×8 = 120 bytes/butterfly)
 * - No cmuls (saves 15 FMA pairs/butterfly = ~30 cycles)
 * - Pure butterfly arithmetic (2 radix-4 + W_4 rotations)
 * 
 * ARITHMETIC COST PER BUTTERFLY (N=1):
 * - Radix-4 #1: 8 adds + 8 subs = 16 FP ops
 * - W_4 twiddles: 12 XORs (sign flips) = negligible
 * - Radix-4 #2: 8 adds + 8 subs = 16 FP ops
 * - Total: ~32 FP ops vs ~90 FP ops with twiddles
 * 
 * MEMORY BANDWIDTH:
 * - Loads: 16 rows × 8 doubles × 8 bytes = 1024 bytes
 * - Stores: 16 rows × 8 doubles × 8 bytes = 1024 bytes
 * - Total: 2048 bytes per butterfly (128 complex numbers)
 * - Arithmetic intensity: 32 FP ops / 2048 bytes = 0.016 ops/byte
 * - VERY memory-bound! (compare to ~0.044 ops/byte with twiddles)
 * 
 * WHEN N=1 MATTERS MOST:
 * - First stage of FFT (always N=1)
 * - Small K where arithmetic overhead dominates
 * - Systems with high memory bandwidth (DDR5, HBM)
 * 
 * REUSED INFRASTRUCTURE:
 * ✅ Butterfly functions (radix16_complete_butterfly_*_fused_soa_avx512)
 * ✅ Load/store functions (load_16_lanes_soa_avx512, etc.)
 * ✅ Mask generation (radix16_get_tail_mask_avx512)
 * ✅ Planner hints (radix16_planner_hints_avx512_t)
 * ✅ NT store decision logic (radix16_should_use_nt_stores_avx512)
 * ✅ Cache flush (radix16_flush_output_cache_lines_avx512)
 * ✅ FTZ/DAZ initialization (radix16_set_ftz_daz_avx512)
 * ✅ Tile size selection (radix16_choose_tile_size_avx512)
 * ✅ All constants and macros
 * 
 * ============================================================================
 */