/**
 * @file fft_radix3_macros_notwiddle_minimal.h
 * @brief MINIMAL No-Twiddle Radix-3 Macros - Reuses existing butterflies
 *
 * @details
 * This is the SMART way to do no-twiddle: just skip the CMUL and reuse
 * all your existing butterfly macros!
 *
 * REQUIRES: fft_radix3_macros_true_soa.h (your original file)
 *
 * USAGE:
 *   #include "fft_radix3_macros_true_soa.h"
 *   #include "fft_radix3_macros_notwiddle_minimal.h"
 *
 * Total lines: ~200 vs 1713 in the "full" version!
 */

#ifndef FFT_RADIX3_MACROS_NOTWIDDLE_MINIMAL_H
#define FFT_RADIX3_MACROS_NOTWIDDLE_MINIMAL_H

// This header REQUIRES your original butterfly macros
#ifndef FFT_RADIX3_MACROS_TRUE_SOA_H
#error "Must include fft_radix3_macros_true_soa.h before this file!"
#endif

//==============================================================================
// AVX-512 NO-TWIDDLE MACROS
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief AVX-512 Forward No-Twiddle - Depth 8
 * 
 * Just loads a,b,c and calls your existing butterfly - no CMUL!
 */
#define RADIX3_NOTWIDDLE_PIPELINE_8_FV_AVX512(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m512d a_re0 = LOAD_RE_AVX512(&in_re[k+0]);    __m512d a_im0 = LOAD_IM_AVX512(&in_im[k+0]); \
        __m512d b_re0 = LOAD_RE_AVX512(&in_re[k+0+K]);  __m512d b_im0 = LOAD_IM_AVX512(&in_im[k+0+K]); \
        __m512d c_re0 = LOAD_RE_AVX512(&in_re[k+0+2*K]); __m512d c_im0 = LOAD_IM_AVX512(&in_im[k+0+2*K]); \
        __m512d a_re1 = LOAD_RE_AVX512(&in_re[k+8]);    __m512d a_im1 = LOAD_IM_AVX512(&in_im[k+8]); \
        __m512d b_re1 = LOAD_RE_AVX512(&in_re[k+8+K]);  __m512d b_im1 = LOAD_IM_AVX512(&in_im[k+8+K]); \
        __m512d c_re1 = LOAD_RE_AVX512(&in_re[k+8+2*K]); __m512d c_im1 = LOAD_IM_AVX512(&in_im[k+8+2*K]); \
        __m512d a_re2 = LOAD_RE_AVX512(&in_re[k+16]);   __m512d a_im2 = LOAD_IM_AVX512(&in_im[k+16]); \
        __m512d b_re2 = LOAD_RE_AVX512(&in_re[k+16+K]); __m512d b_im2 = LOAD_IM_AVX512(&in_im[k+16+K]); \
        __m512d c_re2 = LOAD_RE_AVX512(&in_re[k+16+2*K]); __m512d c_im2 = LOAD_IM_AVX512(&in_im[k+16+2*K]); \
        __m512d a_re3 = LOAD_RE_AVX512(&in_re[k+24]);   __m512d a_im3 = LOAD_IM_AVX512(&in_im[k+24]); \
        __m512d b_re3 = LOAD_RE_AVX512(&in_re[k+24+K]); __m512d b_im3 = LOAD_IM_AVX512(&in_im[k+24+K]); \
        __m512d c_re3 = LOAD_RE_AVX512(&in_re[k+24+2*K]); __m512d c_im3 = LOAD_IM_AVX512(&in_im[k+24+2*K]); \
        __m512d a_re4 = LOAD_RE_AVX512(&in_re[k+32]);   __m512d a_im4 = LOAD_IM_AVX512(&in_im[k+32]); \
        __m512d b_re4 = LOAD_RE_AVX512(&in_re[k+32+K]); __m512d b_im4 = LOAD_IM_AVX512(&in_im[k+32+K]); \
        __m512d c_re4 = LOAD_RE_AVX512(&in_re[k+32+2*K]); __m512d c_im4 = LOAD_IM_AVX512(&in_im[k+32+2*K]); \
        __m512d a_re5 = LOAD_RE_AVX512(&in_re[k+40]);   __m512d a_im5 = LOAD_IM_AVX512(&in_im[k+40]); \
        __m512d b_re5 = LOAD_RE_AVX512(&in_re[k+40+K]); __m512d b_im5 = LOAD_IM_AVX512(&in_im[k+40+K]); \
        __m512d c_re5 = LOAD_RE_AVX512(&in_re[k+40+2*K]); __m512d c_im5 = LOAD_IM_AVX512(&in_im[k+40+2*K]); \
        __m512d a_re6 = LOAD_RE_AVX512(&in_re[k+48]);   __m512d a_im6 = LOAD_IM_AVX512(&in_im[k+48]); \
        __m512d b_re6 = LOAD_RE_AVX512(&in_re[k+48+K]); __m512d b_im6 = LOAD_IM_AVX512(&in_im[k+48+K]); \
        __m512d c_re6 = LOAD_RE_AVX512(&in_re[k+48+2*K]); __m512d c_im6 = LOAD_IM_AVX512(&in_im[k+48+2*K]); \
        __m512d a_re7 = LOAD_RE_AVX512(&in_re[k+56]);   __m512d a_im7 = LOAD_IM_AVX512(&in_im[k+56]); \
        __m512d b_re7 = LOAD_RE_AVX512(&in_re[k+56+K]); __m512d b_im7 = LOAD_IM_AVX512(&in_im[k+56+K]); \
        __m512d c_re7 = LOAD_RE_AVX512(&in_re[k+56+2*K]); __m512d c_im7 = LOAD_IM_AVX512(&in_im[k+56+2*K]); \
        __m512d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m512d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        __m512d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2; \
        __m512d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3; \
        __m512d y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4; \
        __m512d y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5; \
        __m512d y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6; \
        __m512d y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7; \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re4, a_im4, b_re4, b_im4, c_re4, c_im4, y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re5, a_im5, b_re5, b_im5, c_re5, c_im5, y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re6, a_im6, b_re6, b_im6, c_re6, c_im6, y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re7, a_im7, b_re7, b_im7, c_re7, c_im7, y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7); \
        STORE_RE_AVX512(&out_re[k+0],    y0_re0); STORE_IM_AVX512(&out_im[k+0],    y0_im0); \
        STORE_RE_AVX512(&out_re[k+0+K],  y1_re0); STORE_IM_AVX512(&out_im[k+0+K],  y1_im0); \
        STORE_RE_AVX512(&out_re[k+0+2*K], y2_re0); STORE_IM_AVX512(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_AVX512(&out_re[k+8],    y0_re1); STORE_IM_AVX512(&out_im[k+8],    y0_im1); \
        STORE_RE_AVX512(&out_re[k+8+K],  y1_re1); STORE_IM_AVX512(&out_im[k+8+K],  y1_im1); \
        STORE_RE_AVX512(&out_re[k+8+2*K], y2_re1); STORE_IM_AVX512(&out_im[k+8+2*K], y2_im1); \
        STORE_RE_AVX512(&out_re[k+16],   y0_re2); STORE_IM_AVX512(&out_im[k+16],   y0_im2); \
        STORE_RE_AVX512(&out_re[k+16+K], y1_re2); STORE_IM_AVX512(&out_im[k+16+K], y1_im2); \
        STORE_RE_AVX512(&out_re[k+16+2*K], y2_re2); STORE_IM_AVX512(&out_im[k+16+2*K], y2_im2); \
        STORE_RE_AVX512(&out_re[k+24],   y0_re3); STORE_IM_AVX512(&out_im[k+24],   y0_im3); \
        STORE_RE_AVX512(&out_re[k+24+K], y1_re3); STORE_IM_AVX512(&out_im[k+24+K], y1_im3); \
        STORE_RE_AVX512(&out_re[k+24+2*K], y2_re3); STORE_IM_AVX512(&out_im[k+24+2*K], y2_im3); \
        STORE_RE_AVX512(&out_re[k+32],   y0_re4); STORE_IM_AVX512(&out_im[k+32],   y0_im4); \
        STORE_RE_AVX512(&out_re[k+32+K], y1_re4); STORE_IM_AVX512(&out_im[k+32+K], y1_im4); \
        STORE_RE_AVX512(&out_re[k+32+2*K], y2_re4); STORE_IM_AVX512(&out_im[k+32+2*K], y2_im4); \
        STORE_RE_AVX512(&out_re[k+40],   y0_re5); STORE_IM_AVX512(&out_im[k+40],   y0_im5); \
        STORE_RE_AVX512(&out_re[k+40+K], y1_re5); STORE_IM_AVX512(&out_im[k+40+K], y1_im5); \
        STORE_RE_AVX512(&out_re[k+40+2*K], y2_re5); STORE_IM_AVX512(&out_im[k+40+2*K], y2_im5); \
        STORE_RE_AVX512(&out_re[k+48],   y0_re6); STORE_IM_AVX512(&out_im[k+48],   y0_im6); \
        STORE_RE_AVX512(&out_re[k+48+K], y1_re6); STORE_IM_AVX512(&out_im[k+48+K], y1_im6); \
        STORE_RE_AVX512(&out_re[k+48+2*K], y2_re6); STORE_IM_AVX512(&out_im[k+48+2*K], y2_im6); \
        STORE_RE_AVX512(&out_re[k+56],   y0_re7); STORE_IM_AVX512(&out_im[k+56],   y0_im7); \
        STORE_RE_AVX512(&out_re[k+56+K], y1_re7); STORE_IM_AVX512(&out_im[k+56+K], y1_im7); \
        STORE_RE_AVX512(&out_re[k+56+2*K], y2_re7); STORE_IM_AVX512(&out_im[k+56+2*K], y2_im7); \
    } while(0)

// Backward version - just change FV to BV!
#define RADIX3_NOTWIDDLE_PIPELINE_8_BV_AVX512(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m512d a_re0 = LOAD_RE_AVX512(&in_re[k+0]);    __m512d a_im0 = LOAD_IM_AVX512(&in_im[k+0]); \
        __m512d b_re0 = LOAD_RE_AVX512(&in_re[k+0+K]);  __m512d b_im0 = LOAD_IM_AVX512(&in_im[k+0+K]); \
        __m512d c_re0 = LOAD_RE_AVX512(&in_re[k+0+2*K]); __m512d c_im0 = LOAD_IM_AVX512(&in_im[k+0+2*K]); \
        __m512d a_re1 = LOAD_RE_AVX512(&in_re[k+8]);    __m512d a_im1 = LOAD_IM_AVX512(&in_im[k+8]); \
        __m512d b_re1 = LOAD_RE_AVX512(&in_re[k+8+K]);  __m512d b_im1 = LOAD_IM_AVX512(&in_im[k+8+K]); \
        __m512d c_re1 = LOAD_RE_AVX512(&in_re[k+8+2*K]); __m512d c_im1 = LOAD_IM_AVX512(&in_im[k+8+2*K]); \
        __m512d a_re2 = LOAD_RE_AVX512(&in_re[k+16]);   __m512d a_im2 = LOAD_IM_AVX512(&in_im[k+16]); \
        __m512d b_re2 = LOAD_RE_AVX512(&in_re[k+16+K]); __m512d b_im2 = LOAD_IM_AVX512(&in_im[k+16+K]); \
        __m512d c_re2 = LOAD_RE_AVX512(&in_re[k+16+2*K]); __m512d c_im2 = LOAD_IM_AVX512(&in_im[k+16+2*K]); \
        __m512d a_re3 = LOAD_RE_AVX512(&in_re[k+24]);   __m512d a_im3 = LOAD_IM_AVX512(&in_im[k+24]); \
        __m512d b_re3 = LOAD_RE_AVX512(&in_re[k+24+K]); __m512d b_im3 = LOAD_IM_AVX512(&in_im[k+24+K]); \
        __m512d c_re3 = LOAD_RE_AVX512(&in_re[k+24+2*K]); __m512d c_im3 = LOAD_IM_AVX512(&in_im[k+24+2*K]); \
        __m512d a_re4 = LOAD_RE_AVX512(&in_re[k+32]);   __m512d a_im4 = LOAD_IM_AVX512(&in_im[k+32]); \
        __m512d b_re4 = LOAD_RE_AVX512(&in_re[k+32+K]); __m512d b_im4 = LOAD_IM_AVX512(&in_im[k+32+K]); \
        __m512d c_re4 = LOAD_RE_AVX512(&in_re[k+32+2*K]); __m512d c_im4 = LOAD_IM_AVX512(&in_im[k+32+2*K]); \
        __m512d a_re5 = LOAD_RE_AVX512(&in_re[k+40]);   __m512d a_im5 = LOAD_IM_AVX512(&in_im[k+40]); \
        __m512d b_re5 = LOAD_RE_AVX512(&in_re[k+40+K]); __m512d b_im5 = LOAD_IM_AVX512(&in_im[k+40+K]); \
        __m512d c_re5 = LOAD_RE_AVX512(&in_re[k+40+2*K]); __m512d c_im5 = LOAD_IM_AVX512(&in_im[k+40+2*K]); \
        __m512d a_re6 = LOAD_RE_AVX512(&in_re[k+48]);   __m512d a_im6 = LOAD_IM_AVX512(&in_im[k+48]); \
        __m512d b_re6 = LOAD_RE_AVX512(&in_re[k+48+K]); __m512d b_im6 = LOAD_IM_AVX512(&in_im[k+48+K]); \
        __m512d c_re6 = LOAD_RE_AVX512(&in_re[k+48+2*K]); __m512d c_im6 = LOAD_IM_AVX512(&in_im[k+48+2*K]); \
        __m512d a_re7 = LOAD_RE_AVX512(&in_re[k+56]);   __m512d a_im7 = LOAD_IM_AVX512(&in_im[k+56]); \
        __m512d b_re7 = LOAD_RE_AVX512(&in_re[k+56+K]); __m512d b_im7 = LOAD_IM_AVX512(&in_im[k+56+K]); \
        __m512d c_re7 = LOAD_RE_AVX512(&in_re[k+56+2*K]); __m512d c_im7 = LOAD_IM_AVX512(&in_im[k+56+2*K]); \
        __m512d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m512d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        __m512d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2; \
        __m512d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3; \
        __m512d y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4; \
        __m512d y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5; \
        __m512d y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6; \
        __m512d y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7; \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re4, a_im4, b_re4, b_im4, c_re4, c_im4, y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re5, a_im5, b_re5, b_im5, c_re5, c_im5, y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re6, a_im6, b_re6, b_im6, c_re6, c_im6, y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re7, a_im7, b_re7, b_im7, c_re7, c_im7, y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7); \
        STORE_RE_AVX512(&out_re[k+0],    y0_re0); STORE_IM_AVX512(&out_im[k+0],    y0_im0); \
        STORE_RE_AVX512(&out_re[k+0+K],  y1_re0); STORE_IM_AVX512(&out_im[k+0+K],  y1_im0); \
        STORE_RE_AVX512(&out_re[k+0+2*K], y2_re0); STORE_IM_AVX512(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_AVX512(&out_re[k+8],    y0_re1); STORE_IM_AVX512(&out_im[k+8],    y0_im1); \
        STORE_RE_AVX512(&out_re[k+8+K],  y1_re1); STORE_IM_AVX512(&out_im[k+8+K],  y1_im1); \
        STORE_RE_AVX512(&out_re[k+8+2*K], y2_re1); STORE_IM_AVX512(&out_im[k+8+2*K], y2_im1); \
        STORE_RE_AVX512(&out_re[k+16],   y0_re2); STORE_IM_AVX512(&out_im[k+16],   y0_im2); \
        STORE_RE_AVX512(&out_re[k+16+K], y1_re2); STORE_IM_AVX512(&out_im[k+16+K], y1_im2); \
        STORE_RE_AVX512(&out_re[k+16+2*K], y2_re2); STORE_IM_AVX512(&out_im[k+16+2*K], y2_im2); \
        STORE_RE_AVX512(&out_re[k+24],   y0_re3); STORE_IM_AVX512(&out_im[k+24],   y0_im3); \
        STORE_RE_AVX512(&out_re[k+24+K], y1_re3); STORE_IM_AVX512(&out_im[k+24+K], y1_im3); \
        STORE_RE_AVX512(&out_re[k+24+2*K], y2_re3); STORE_IM_AVX512(&out_im[k+24+2*K], y2_im3); \
        STORE_RE_AVX512(&out_re[k+32],   y0_re4); STORE_IM_AVX512(&out_im[k+32],   y0_im4); \
        STORE_RE_AVX512(&out_re[k+32+K], y1_re4); STORE_IM_AVX512(&out_im[k+32+K], y1_im4); \
        STORE_RE_AVX512(&out_re[k+32+2*K], y2_re4); STORE_IM_AVX512(&out_im[k+32+2*K], y2_im4); \
        STORE_RE_AVX512(&out_re[k+40],   y0_re5); STORE_IM_AVX512(&out_im[k+40],   y0_im5); \
        STORE_RE_AVX512(&out_re[k+40+K], y1_re5); STORE_IM_AVX512(&out_im[k+40+K], y1_im5); \
        STORE_RE_AVX512(&out_re[k+40+2*K], y2_re5); STORE_IM_AVX512(&out_im[k+40+2*K], y2_im5); \
        STORE_RE_AVX512(&out_re[k+48],   y0_re6); STORE_IM_AVX512(&out_im[k+48],   y0_im6); \
        STORE_RE_AVX512(&out_re[k+48+K], y1_re6); STORE_IM_AVX512(&out_im[k+48+K], y1_im6); \
        STORE_RE_AVX512(&out_re[k+48+2*K], y2_re6); STORE_IM_AVX512(&out_im[k+48+2*K], y2_im6); \
        STORE_RE_AVX512(&out_re[k+56],   y0_re7); STORE_IM_AVX512(&out_im[k+56],   y0_im7); \
        STORE_RE_AVX512(&out_re[k+56+K], y1_re7); STORE_IM_AVX512(&out_im[k+56+K], y1_im7); \
        STORE_RE_AVX512(&out_re[k+56+2*K], y2_re7); STORE_IM_AVX512(&out_im[k+56+2*K], y2_im7); \
    } while(0)

#endif // __AVX512F__

//==============================================================================
// AVX2, SSE2, SCALAR - Same pattern, just change the SIMD types
//==============================================================================

#ifdef __AVX2__
// AVX2 depth-4: process 16 elements (4 lanes × 4 depth)
#define RADIX3_NOTWIDDLE_PIPELINE_4_FV_AVX2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m256d a_re0 = LOAD_RE_AVX2(&in_re[k+0]);    __m256d a_im0 = LOAD_IM_AVX2(&in_im[k+0]); \
        __m256d b_re0 = LOAD_RE_AVX2(&in_re[k+0+K]);  __m256d b_im0 = LOAD_IM_AVX2(&in_im[k+0+K]); \
        __m256d c_re0 = LOAD_RE_AVX2(&in_re[k+0+2*K]); __m256d c_im0 = LOAD_IM_AVX2(&in_im[k+0+2*K]); \
        __m256d a_re1 = LOAD_RE_AVX2(&in_re[k+4]);    __m256d a_im1 = LOAD_IM_AVX2(&in_im[k+4]); \
        __m256d b_re1 = LOAD_RE_AVX2(&in_re[k+4+K]);  __m256d b_im1 = LOAD_IM_AVX2(&in_im[k+4+K]); \
        __m256d c_re1 = LOAD_RE_AVX2(&in_re[k+4+2*K]); __m256d c_im1 = LOAD_IM_AVX2(&in_im[k+4+2*K]); \
        __m256d a_re2 = LOAD_RE_AVX2(&in_re[k+8]);    __m256d a_im2 = LOAD_IM_AVX2(&in_im[k+8]); \
        __m256d b_re2 = LOAD_RE_AVX2(&in_re[k+8+K]);  __m256d b_im2 = LOAD_IM_AVX2(&in_im[k+8+K]); \
        __m256d c_re2 = LOAD_RE_AVX2(&in_re[k+8+2*K]); __m256d c_im2 = LOAD_IM_AVX2(&in_im[k+8+2*K]); \
        __m256d a_re3 = LOAD_RE_AVX2(&in_re[k+12]);   __m256d a_im3 = LOAD_IM_AVX2(&in_im[k+12]); \
        __m256d b_re3 = LOAD_RE_AVX2(&in_re[k+12+K]); __m256d b_im3 = LOAD_IM_AVX2(&in_im[k+12+K]); \
        __m256d c_re3 = LOAD_RE_AVX2(&in_re[k+12+2*K]); __m256d c_im3 = LOAD_IM_AVX2(&in_im[k+12+2*K]); \
        __m256d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m256d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        __m256d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2; \
        __m256d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3; \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3); \
        STORE_RE_AVX2(&out_re[k+0],    y0_re0); STORE_IM_AVX2(&out_im[k+0],    y0_im0); \
        STORE_RE_AVX2(&out_re[k+0+K],  y1_re0); STORE_IM_AVX2(&out_im[k+0+K],  y1_im0); \
        STORE_RE_AVX2(&out_re[k+0+2*K], y2_re0); STORE_IM_AVX2(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_AVX2(&out_re[k+4],    y0_re1); STORE_IM_AVX2(&out_im[k+4],    y0_im1); \
        STORE_RE_AVX2(&out_re[k+4+K],  y1_re1); STORE_IM_AVX2(&out_im[k+4+K],  y1_im1); \
        STORE_RE_AVX2(&out_re[k+4+2*K], y2_re1); STORE_IM_AVX2(&out_im[k+4+2*K], y2_im1); \
        STORE_RE_AVX2(&out_re[k+8],    y0_re2); STORE_IM_AVX2(&out_im[k+8],    y0_im2); \
        STORE_RE_AVX2(&out_re[k+8+K],  y1_re2); STORE_IM_AVX2(&out_im[k+8+K],  y1_im2); \
        STORE_RE_AVX2(&out_re[k+8+2*K], y2_re2); STORE_IM_AVX2(&out_im[k+8+2*K], y2_im2); \
        STORE_RE_AVX2(&out_re[k+12],   y0_re3); STORE_IM_AVX2(&out_im[k+12],   y0_im3); \
        STORE_RE_AVX2(&out_re[k+12+K], y1_re3); STORE_IM_AVX2(&out_im[k+12+K], y1_im3); \
        STORE_RE_AVX2(&out_re[k+12+2*K], y2_re3); STORE_IM_AVX2(&out_im[k+12+2*K], y2_im3); \
    } while(0)

#define RADIX3_NOTWIDDLE_PIPELINE_4_BV_AVX2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m256d a_re0 = LOAD_RE_AVX2(&in_re[k+0]);    __m256d a_im0 = LOAD_IM_AVX2(&in_im[k+0]); \
        __m256d b_re0 = LOAD_RE_AVX2(&in_re[k+0+K]);  __m256d b_im0 = LOAD_IM_AVX2(&in_im[k+0+K]); \
        __m256d c_re0 = LOAD_RE_AVX2(&in_re[k+0+2*K]); __m256d c_im0 = LOAD_IM_AVX2(&in_im[k+0+2*K]); \
        __m256d a_re1 = LOAD_RE_AVX2(&in_re[k+4]);    __m256d a_im1 = LOAD_IM_AVX2(&in_im[k+4]); \
        __m256d b_re1 = LOAD_RE_AVX2(&in_re[k+4+K]);  __m256d b_im1 = LOAD_IM_AVX2(&in_im[k+4+K]); \
        __m256d c_re1 = LOAD_RE_AVX2(&in_re[k+4+2*K]); __m256d c_im1 = LOAD_IM_AVX2(&in_im[k+4+2*K]); \
        __m256d a_re2 = LOAD_RE_AVX2(&in_re[k+8]);    __m256d a_im2 = LOAD_IM_AVX2(&in_im[k+8]); \
        __m256d b_re2 = LOAD_RE_AVX2(&in_re[k+8+K]);  __m256d b_im2 = LOAD_IM_AVX2(&in_im[k+8+K]); \
        __m256d c_re2 = LOAD_RE_AVX2(&in_re[k+8+2*K]); __m256d c_im2 = LOAD_IM_AVX2(&in_im[k+8+2*K]); \
        __m256d a_re3 = LOAD_RE_AVX2(&in_re[k+12]);   __m256d a_im3 = LOAD_IM_AVX2(&in_im[k+12]); \
        __m256d b_re3 = LOAD_RE_AVX2(&in_re[k+12+K]); __m256d b_im3 = LOAD_IM_AVX2(&in_im[k+12+K]); \
        __m256d c_re3 = LOAD_RE_AVX2(&in_re[k+12+2*K]); __m256d c_im3 = LOAD_IM_AVX2(&in_im[k+12+2*K]); \
        __m256d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m256d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        __m256d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2; \
        __m256d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3; \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3); \
        STORE_RE_AVX2(&out_re[k+0],    y0_re0); STORE_IM_AVX2(&out_im[k+0],    y0_im0); \
        STORE_RE_AVX2(&out_re[k+0+K],  y1_re0); STORE_IM_AVX2(&out_im[k+0+K],  y1_im0); \
        STORE_RE_AVX2(&out_re[k+0+2*K], y2_re0); STORE_IM_AVX2(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_AVX2(&out_re[k+4],    y0_re1); STORE_IM_AVX2(&out_im[k+4],    y0_im1); \
        STORE_RE_AVX2(&out_re[k+4+K],  y1_re1); STORE_IM_AVX2(&out_im[k+4+K],  y1_im1); \
        STORE_RE_AVX2(&out_re[k+4+2*K], y2_re1); STORE_IM_AVX2(&out_im[k+4+2*K], y2_im1); \
        STORE_RE_AVX2(&out_re[k+8],    y0_re2); STORE_IM_AVX2(&out_im[k+8],    y0_im2); \
        STORE_RE_AVX2(&out_re[k+8+K],  y1_re2); STORE_IM_AVX2(&out_im[k+8+K],  y1_im2); \
        STORE_RE_AVX2(&out_re[k+8+2*K], y2_re2); STORE_IM_AVX2(&out_im[k+8+2*K], y2_im2); \
        STORE_RE_AVX2(&out_re[k+12],   y0_re3); STORE_IM_AVX2(&out_im[k+12],   y0_im3); \
        STORE_RE_AVX2(&out_re[k+12+K], y1_re3); STORE_IM_AVX2(&out_im[k+12+K], y1_im3); \
        STORE_RE_AVX2(&out_re[k+12+2*K], y2_re3); STORE_IM_AVX2(&out_im[k+12+2*K], y2_im3); \
    } while(0)
#endif // __AVX2__

#ifdef __SSE2__
// SSE2 depth-2: process 4 elements (2 lanes × 2 depth)
#define RADIX3_NOTWIDDLE_PIPELINE_2_FV_SSE2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m128d a_re0 = LOAD_RE_SSE2(&in_re[k+0]);    __m128d a_im0 = LOAD_IM_SSE2(&in_im[k+0]); \
        __m128d b_re0 = LOAD_RE_SSE2(&in_re[k+0+K]);  __m128d b_im0 = LOAD_IM_SSE2(&in_im[k+0+K]); \
        __m128d c_re0 = LOAD_RE_SSE2(&in_re[k+0+2*K]); __m128d c_im0 = LOAD_IM_SSE2(&in_im[k+0+2*K]); \
        __m128d a_re1 = LOAD_RE_SSE2(&in_re[k+2]);    __m128d a_im1 = LOAD_IM_SSE2(&in_im[k+2]); \
        __m128d b_re1 = LOAD_RE_SSE2(&in_re[k+2+K]);  __m128d b_im1 = LOAD_IM_SSE2(&in_im[k+2+K]); \
        __m128d c_re1 = LOAD_RE_SSE2(&in_re[k+2+2*K]); __m128d c_im1 = LOAD_IM_SSE2(&in_im[k+2+2*K]); \
        __m128d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m128d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        STORE_RE_SSE2(&out_re[k+0],    y0_re0); STORE_IM_SSE2(&out_im[k+0],    y0_im0); \
        STORE_RE_SSE2(&out_re[k+0+K],  y1_re0); STORE_IM_SSE2(&out_im[k+0+K],  y1_im0); \
        STORE_RE_SSE2(&out_re[k+0+2*K], y2_re0); STORE_IM_SSE2(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_SSE2(&out_re[k+2],    y0_re1); STORE_IM_SSE2(&out_im[k+2],    y0_im1); \
        STORE_RE_SSE2(&out_re[k+2+K],  y1_re1); STORE_IM_SSE2(&out_im[k+2+K],  y1_im1); \
        STORE_RE_SSE2(&out_re[k+2+2*K], y2_re1); STORE_IM_SSE2(&out_im[k+2+2*K], y2_im1); \
    } while(0)

#define RADIX3_NOTWIDDLE_PIPELINE_2_BV_SSE2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m128d a_re0 = LOAD_RE_SSE2(&in_re[k+0]);    __m128d a_im0 = LOAD_IM_SSE2(&in_im[k+0]); \
        __m128d b_re0 = LOAD_RE_SSE2(&in_re[k+0+K]);  __m128d b_im0 = LOAD_IM_SSE2(&in_im[k+0+K]); \
        __m128d c_re0 = LOAD_RE_SSE2(&in_re[k+0+2*K]); __m128d c_im0 = LOAD_IM_SSE2(&in_im[k+0+2*K]); \
        __m128d a_re1 = LOAD_RE_SSE2(&in_re[k+2]);    __m128d a_im1 = LOAD_IM_SSE2(&in_im[k+2]); \
        __m128d b_re1 = LOAD_RE_SSE2(&in_re[k+2+K]);  __m128d b_im1 = LOAD_IM_SSE2(&in_im[k+2+K]); \
        __m128d c_re1 = LOAD_RE_SSE2(&in_re[k+2+2*K]); __m128d c_im1 = LOAD_IM_SSE2(&in_im[k+2+2*K]); \
        __m128d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0; \
        __m128d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1; \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0); \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1); \
        STORE_RE_SSE2(&out_re[k+0],    y0_re0); STORE_IM_SSE2(&out_im[k+0],    y0_im0); \
        STORE_RE_SSE2(&out_re[k+0+K],  y1_re0); STORE_IM_SSE2(&out_im[k+0+K],  y1_im0); \
        STORE_RE_SSE2(&out_re[k+0+2*K], y2_re0); STORE_IM_SSE2(&out_im[k+0+2*K], y2_im0); \
        STORE_RE_SSE2(&out_re[k+2],    y0_re1); STORE_IM_SSE2(&out_im[k+2],    y0_im1); \
        STORE_RE_SSE2(&out_re[k+2+K],  y1_re1); STORE_IM_SSE2(&out_im[k+2+K],  y1_im1); \
        STORE_RE_SSE2(&out_re[k+2+2*K], y2_re1); STORE_IM_SSE2(&out_im[k+2+2*K], y2_im1); \
    } while(0)
#endif // __SSE2__

// Scalar fallback: just call the original scalar macro
#define RADIX3_NOTWIDDLE_PIPELINE_1_FV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do { \
        double a_re = in_re[k], a_im = in_im[k]; \
        double b_re = in_re[k+K], b_im = in_im[k+K]; \
        double c_re = in_re[k+2*K], c_im = in_im[k+2*K]; \
        double sum_re = b_re + c_re, sum_im = b_im + c_im; \
        double dif_re = b_re - c_re, dif_im = b_im - c_im; \
        double common_re = a_re + C_HALF * sum_re, common_im = a_im + C_HALF * sum_im; \
        double rot_re = S_SQRT3_2 * dif_im, rot_im = -S_SQRT3_2 * dif_re; \
        out_re[k] = a_re + sum_re; out_im[k] = a_im + sum_im; \
        out_re[k+K] = common_re + rot_re; out_im[k+K] = common_im + rot_im; \
        out_re[k+2*K] = common_re - rot_re; out_im[k+2*K] = common_im - rot_im; \
    } while(0)

#define RADIX3_NOTWIDDLE_PIPELINE_1_BV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do { \
        double a_re = in_re[k], a_im = in_im[k]; \
        double b_re = in_re[k+K], b_im = in_im[k+K]; \
        double c_re = in_re[k+2*K], c_im = in_im[k+2*K]; \
        double sum_re = b_re + c_re, sum_im = b_im + c_im; \
        double dif_re = b_re - c_re, dif_im = b_im - c_im; \
        double common_re = a_re + C_HALF * sum_re, common_im = a_im + C_HALF * sum_im; \
        double rot_re = -S_SQRT3_2 * dif_im, rot_im = S_SQRT3_2 * dif_re; \
        out_re[k] = a_re + sum_re; out_im[k] = a_im + sum_im; \
        out_re[k+K] = common_re + rot_re; out_im[k+K] = common_im + rot_im; \
        out_re[k+2*K] = common_re - rot_re; out_im[k+2*K] = common_im - rot_im; \
    } while(0)

#endif // FFT_RADIX3_MACROS_NOTWIDDLE_MINIMAL_H