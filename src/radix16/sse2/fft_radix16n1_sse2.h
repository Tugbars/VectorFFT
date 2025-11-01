/**
 * @file fft_radix16_sse2_native_soa_n1.h
 * @brief Radix-16 N=1 Stage (No Twiddles) - SSE2 Implementation
 *
 * Provides N=1 (first stage, no twiddles) specialized implementations
 * by reusing core butterfly functions from fft_radix16_sse2_native_soa_optimized.h
 *
 * Public API:
 *  - radix16_stage_n1_forward_sse2(...)
 *  - radix16_stage_n1_backward_sse2(...)
 *
 * © 2025 MIT-style
 */

#ifndef FFT_RADIX16_SSE2_NATIVE_SOA_N1_H
#define FFT_RADIX16_SSE2_NATIVE_SOA_N1_H

#include "fft_radix16_sse2_native_soa_optimized.h"

//==============================================================================
// N=1 CONFIGURATION (SSE2-tuned)
//==============================================================================

#ifndef RADIX16_SSE2_N1_UNROLL
#define RADIX16_SSE2_N1_UNROLL 4  // 4 or 8 (SSE2 benefits from higher unroll)
#endif

#ifndef RADIX16_SSE2_N1_STREAM_THRESHOLD_KB
#define RADIX16_SSE2_N1_STREAM_THRESHOLD_KB 128  // Lower for N1
#endif

//==============================================================================
// N1 STREAMING DECISION (reuse from main header but with different threshold)
//==============================================================================

TARGET_SSE2
FORCE_INLINE bool radix16_sse2_should_use_nt_stores_n1(
    size_t K,
    const void *out_re, const void *out_im)
{
    const size_t bytes_per_k = 16 * 2 * sizeof(double);
    size_t threshold_k = (RADIX16_SSE2_N1_STREAM_THRESHOLD_KB * 1024) / bytes_per_k;
    
    if (K < threshold_k) return false;
    
    // Check alignment
    if ((((uintptr_t)out_re & 15) != 0) || (((uintptr_t)out_im & 15) != 0)) {
        return false;
    }
    
    return true;
}

//==============================================================================
// N=1 FORWARD TRANSFORM (SSE2)
//==============================================================================

/**
 * @brief N=1 forward stage - no twiddle multiplication needed
 * 
 * This is the first stage of DIT FFT where all twiddles = 1.
 * Optimized with U=4 or U=8 unrolling for pure butterfly computation.
 * 
 * @param K Number of butterfly columns
 * @param in_re Input real part [16 × K], SoA layout
 * @param in_im Input imaginary part [16 × K]
 * @param out_re Output real part [16 × K]
 * @param out_im Output imaginary part [16 × K]
 */
TARGET_SSE2
void radix16_stage_n1_forward_sse2(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im)
{
    radix16_set_ftz_daz_sse2();
    
    const __m128d rot_sign_mask = kRotSignFwd_sse2;
    
    const bool use_nt_stores = radix16_sse2_should_use_nt_stores_n1(
        K, out_re, out_im);
    
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);
    
    size_t k = 0;
    
    //==========================================================================
    // U=8 UNROLL (optional, for maximum throughput)
    //==========================================================================
#if RADIX16_SSE2_N1_UNROLL == 8
    for (; k + 16 <= (size_t)K; k += 16) {
        // Process 8 SSE2 vectors (16 doubles = 8 butterflies)
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        __m128d x2_re[16], x2_im[16];
        __m128d x3_re[16], x3_im[16];
        
        // Load batch 0-1
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        
        // Butterfly batch 0
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        // Load batch 2-3
        load_16_lanes_soa_sse2(k + 4, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
        load_16_lanes_soa_sse2(k + 6, K, in_re_aligned, in_im_aligned, x3_re, x3_im);
        
        // Store batch 0
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        }
        
        // Butterfly batch 1
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        // Load batch 4-5
        __m128d x4_re[16], x4_im[16];
        __m128d x5_re[16], x5_im[16];
        load_16_lanes_soa_sse2(k + 8, K, in_re_aligned, in_im_aligned, x4_re, x4_im);
        load_16_lanes_soa_sse2(k + 10, K, in_re_aligned, in_im_aligned, x5_re, x5_im);
        
        // Store batch 1
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
        
        // Butterfly batch 2
        __m128d y2_re[16], y2_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
        
        // Load batch 6-7
        __m128d x6_re[16], x6_im[16];
        __m128d x7_re[16], x7_im[16];
        load_16_lanes_soa_sse2(k + 12, K, in_re_aligned, in_im_aligned, x6_re, x6_im);
        load_16_lanes_soa_sse2(k + 14, K, in_re_aligned, in_im_aligned, x7_re, x7_im);
        
        // Store batch 2
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        } else {
            store_16_lanes_soa_sse2(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        }
        
        // Butterfly batch 3
        __m128d y3_re[16], y3_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);
        
        // Store batch 3
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        } else {
            store_16_lanes_soa_sse2(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        }
        
        // Butterfly batch 4
        __m128d y4_re[16], y4_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x4_re, x4_im, y4_re, y4_im, rot_sign_mask);
        
        // Store batch 4
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 8, K, out_re_aligned, out_im_aligned, y4_re, y4_im);
        } else {
            store_16_lanes_soa_sse2(k + 8, K, out_re_aligned, out_im_aligned, y4_re, y4_im);
        }
        
        // Butterfly batch 5
        __m128d y5_re[16], y5_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x5_re, x5_im, y5_re, y5_im, rot_sign_mask);
        
        // Store batch 5
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 10, K, out_re_aligned, out_im_aligned, y5_re, y5_im);
        } else {
            store_16_lanes_soa_sse2(k + 10, K, out_re_aligned, out_im_aligned, y5_re, y5_im);
        }
        
        // Butterfly batch 6
        __m128d y6_re[16], y6_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x6_re, x6_im, y6_re, y6_im, rot_sign_mask);
        
        // Store batch 6
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 12, K, out_re_aligned, out_im_aligned, y6_re, y6_im);
        } else {
            store_16_lanes_soa_sse2(k + 12, K, out_re_aligned, out_im_aligned, y6_re, y6_im);
        }
        
        // Butterfly batch 7
        __m128d y7_re[16], y7_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x7_re, x7_im, y7_re, y7_im, rot_sign_mask);
        
        // Store batch 7
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 14, K, out_re_aligned, out_im_aligned, y7_re, y7_im);
        } else {
            store_16_lanes_soa_sse2(k + 14, K, out_re_aligned, out_im_aligned, y7_re, y7_im);
        }
    }
#endif
    
    //==========================================================================
    // U=4 UNROLL
    //==========================================================================
#if RADIX16_SSE2_N1_UNROLL >= 4
    for (; k + 8 <= (size_t)K; k += 8) {
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        __m128d x2_re[16], x2_im[16];
        __m128d x3_re[16], x3_im[16];
        
        // Load all 4
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        load_16_lanes_soa_sse2(k + 4, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
        load_16_lanes_soa_sse2(k + 6, K, in_re_aligned, in_im_aligned, x3_re, x3_im);
        
        // Butterfly 0
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        // Butterfly 1
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        // Store 0
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        }
        
        // Butterfly 2
        __m128d y2_re[16], y2_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
        
        // Store 1
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
        
        // Butterfly 3
        __m128d y3_re[16], y3_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);
        
        // Store 2
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        } else {
            store_16_lanes_soa_sse2(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        }
        
        // Store 3
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        } else {
            store_16_lanes_soa_sse2(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        }
    }
#endif
    
    //==========================================================================
    // U=2 LOOP
    //==========================================================================
    for (; k + 4 <= (size_t)K; k += 4) {
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
    }
    
    //==========================================================================
    // TAIL LOOP (k+2)
    //==========================================================================
    for (; k + 2 <= (size_t)K; k += 2) {
        __m128d x_re[16], x_im[16];
        load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
        
        __m128d y_re[16], y_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
        } else {
            store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
        }
    }
    
    //==========================================================================
    // FINAL TAIL (masked)
    //==========================================================================
    if (k < (size_t)K) {
        size_t remaining = K - k;
        __m128d x_re[16], x_im[16];
        load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
        
        __m128d y_re[16], y_im[16];
        radix16_complete_butterfly_forward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);
        
        store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
    }
    
    if (use_nt_stores) {
        _mm_sfence();
    }
}

//==============================================================================
// N=1 BACKWARD TRANSFORM (SSE2)
//==============================================================================

/**
 * @brief N=1 backward stage - no twiddle multiplication needed
 * 
 * @param K Number of butterfly columns
 * @param in_re Input real part [16 × K], SoA layout
 * @param in_im Input imaginary part [16 × K]
 * @param out_re Output real part [16 × K]
 * @param out_im Output imaginary part [16 × K]
 */
TARGET_SSE2
void radix16_stage_n1_backward_sse2(
    int K,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im)
{
    radix16_set_ftz_daz_sse2();
    
    const __m128d rot_sign_mask = kRotSignBwd_sse2;
    
    const bool use_nt_stores = radix16_sse2_should_use_nt_stores_n1(
        K, out_re, out_im);
    
    const double *in_re_aligned = ASSUME_ALIGNED(in_re, 16);
    const double *in_im_aligned = ASSUME_ALIGNED(in_im, 16);
    double *out_re_aligned = ASSUME_ALIGNED(out_re, 16);
    double *out_im_aligned = ASSUME_ALIGNED(out_im, 16);
    
    size_t k = 0;
    
#if RADIX16_SSE2_N1_UNROLL == 8
    for (; k + 16 <= (size_t)K; k += 16) {
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        __m128d x2_re[16], x2_im[16];
        __m128d x3_re[16], x3_im[16];
        
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        load_16_lanes_soa_sse2(k + 4, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
        load_16_lanes_soa_sse2(k + 6, K, in_re_aligned, in_im_aligned, x3_re, x3_im);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        }
        
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        __m128d x4_re[16], x4_im[16];
        __m128d x5_re[16], x5_im[16];
        load_16_lanes_soa_sse2(k + 8, K, in_re_aligned, in_im_aligned, x4_re, x4_im);
        load_16_lanes_soa_sse2(k + 10, K, in_re_aligned, in_im_aligned, x5_re, x5_im);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
        
        __m128d y2_re[16], y2_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
        
        __m128d x6_re[16], x6_im[16];
        __m128d x7_re[16], x7_im[16];
        load_16_lanes_soa_sse2(k + 12, K, in_re_aligned, in_im_aligned, x6_re, x6_im);
        load_16_lanes_soa_sse2(k + 14, K, in_re_aligned, in_im_aligned, x7_re, x7_im);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        } else {
            store_16_lanes_soa_sse2(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
        }
        
        __m128d y3_re[16], y3_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        } else {
            store_16_lanes_soa_sse2(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        }
        
        __m128d y4_re[16], y4_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x4_re, x4_im, y4_re, y4_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 8, K, out_re_aligned, out_im_aligned, y4_re, y4_im);
        } else {
            store_16_lanes_soa_sse2(k + 8, K, out_re_aligned, out_im_aligned, y4_re, y4_im);
        }
        
        __m128d y5_re[16], y5_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x5_re, x5_im, y5_re, y5_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 10, K, out_re_aligned, out_im_aligned, y5_re, y5_im);
        } else {
            store_16_lanes_soa_sse2(k + 10, K, out_re_aligned, out_im_aligned, y5_re, y5_im);
        }
        
        __m128d y6_re[16], y6_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x6_re, x6_im, y6_re, y6_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 12, K, out_re_aligned, out_im_aligned, y6_re, y6_im);
        } else {
            store_16_lanes_soa_sse2(k + 12, K, out_re_aligned, out_im_aligned, y6_re, y6_im);
        }
        
        __m128d y7_re[16], y7_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x7_re, x7_im, y7_re, y7_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 14, K, out_re_aligned, out_im_aligned, y7_re, y7_im);
        } else {
            store_16_lanes_soa_sse2(k + 14, K, out_re_aligned, out_im_aligned, y7_re, y7_im);
        }
    }
#endif
    
#if RADIX16_SSE2_N1_UNROLL >= 4
    for (; k + 8 <= (size_t)K; k += 8) {
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        __m128d x2_re[16], x2_im[16];
        __m128d x3_re[16], x3_im[16];
        
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        load_16_lanes_soa_sse2(k + 4, K, in_re_aligned, in_im_aligned, x2_re, x2_im);
        load_16_lanes_soa_sse2(k + 6, K, in_re_aligned, in_im_aligned, x3_re, x3_im);
        
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
        }
        
        __m128d y2_re[16], y2_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x2_re, x2_im, y2_re, y2_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
        
        __m128d y3_re[16], y3_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x3_re, x3_im, y3_re, y3_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
            store_16_lanes_soa_sse2_stream(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        } else {
            store_16_lanes_soa_sse2(k + 4, K, out_re_aligned, out_im_aligned, y2_re, y2_im);
            store_16_lanes_soa_sse2(k + 6, K, out_re_aligned, out_im_aligned, y3_re, y3_im);
        }
    }
#endif
    
    for (; k + 4 <= (size_t)K; k += 4) {
        __m128d x0_re[16], x0_im[16];
        __m128d x1_re[16], x1_im[16];
        
        load_16_lanes_soa_sse2(k + 0, K, in_re_aligned, in_im_aligned, x0_re, x0_im);
        load_16_lanes_soa_sse2(k + 2, K, in_re_aligned, in_im_aligned, x1_re, x1_im);
        
        __m128d y0_re[16], y0_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x0_re, x0_im, y0_re, y0_im, rot_sign_mask);
        
        __m128d y1_re[16], y1_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x1_re, x1_im, y1_re, y1_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            store_16_lanes_soa_sse2_stream(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        } else {
            store_16_lanes_soa_sse2(k + 0, K, out_re_aligned, out_im_aligned, y0_re, y0_im);
            store_16_lanes_soa_sse2(k + 2, K, out_re_aligned, out_im_aligned, y1_re, y1_im);
        }
    }
    
    for (; k + 2 <= (size_t)K; k += 2) {
        __m128d x_re[16], x_im[16];
        load_16_lanes_soa_sse2(k, K, in_re_aligned, in_im_aligned, x_re, x_im);
        
        __m128d y_re[16], y_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);
        
        if (use_nt_stores) {
            store_16_lanes_soa_sse2_stream(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
        } else {
            store_16_lanes_soa_sse2(k, K, out_re_aligned, out_im_aligned, y_re, y_im);
        }
    }
    
    if (k < (size_t)K) {
        size_t remaining = K - k;
        __m128d x_re[16], x_im[16];
        load_16_lanes_soa_sse2_masked(k, K, remaining, in_re_aligned, in_im_aligned, x_re, x_im);
        
        __m128d y_re[16], y_im[16];
        radix16_complete_butterfly_backward_fused_soa_sse2(x_re, x_im, y_re, y_im, rot_sign_mask);
        
        store_16_lanes_soa_sse2_masked(k, K, remaining, out_re_aligned, out_im_aligned, y_re, y_im);
    }
    
    if (use_nt_stores) {
        _mm_sfence();
    }
}

#endif /* FFT_RADIX16_SSE2_NATIVE_SOA_N1_H */