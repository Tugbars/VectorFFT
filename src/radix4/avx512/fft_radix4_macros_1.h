/**
 * @file fft_radix4_macros_notwiddle_minimal.h
 * @brief MINIMAL No-Twiddle Radix-4 Macros - Reuses existing butterflies
 *
 * @details
 * No-twiddle version for radix-4 first stage (k=0, all twiddles = 1).
 * Just skips the 3 CMUL operations and passes b,c,d directly to butterfly.
 *
 * REQUIRES: fft_radix4_macros_true_soa.h (your original file)
 *
 * USAGE:
 *   #include "fft_radix4_macros_true_soa.h"
 *   #include "fft_radix4_macros_notwiddle_minimal.h"
 *
 * PERFORMANCE: ~2x faster first stage (eliminates 3 CMULs per butterfly)
 */

#ifndef FFT_RADIX4_MACROS_NOTWIDDLE_MINIMAL_H
#define FFT_RADIX4_MACROS_NOTWIDDLE_MINIMAL_H

// This header REQUIRES your original butterfly macros
#ifndef FFT_RADIX4_MACROS_TRUE_SOA_H
#error "Must include fft_radix4_macros_true_soa.h before this file!"
#endif

//==============================================================================
// AVX-512 NO-TWIDDLE MACROS
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief AVX-512 Forward No-Twiddle - Depth 8
 * 
 * Radix-4: processes 4 inputs (a,b,c,d), produces 4 outputs
 * No-twiddle: b,c,d multiplied by W^0 = 1, so skip CMUL entirely
 */
#define RADIX4_NOTWIDDLE_PIPELINE_8_FV_AVX512(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m512d a_re0 = _mm512_loadu_pd(&in_re[k+0]);     __m512d a_im0 = _mm512_loadu_pd(&in_im[k+0]); \
        __m512d b_re0 = _mm512_loadu_pd(&in_re[k+0+K]);   __m512d b_im0 = _mm512_loadu_pd(&in_im[k+0+K]); \
        __m512d c_re0 = _mm512_loadu_pd(&in_re[k+0+2*K]); __m512d c_im0 = _mm512_loadu_pd(&in_im[k+0+2*K]); \
        __m512d d_re0 = _mm512_loadu_pd(&in_re[k+0+3*K]); __m512d d_im0 = _mm512_loadu_pd(&in_im[k+0+3*K]); \
        __m512d a_re1 = _mm512_loadu_pd(&in_re[k+8]);     __m512d a_im1 = _mm512_loadu_pd(&in_im[k+8]); \
        __m512d b_re1 = _mm512_loadu_pd(&in_re[k+8+K]);   __m512d b_im1 = _mm512_loadu_pd(&in_im[k+8+K]); \
        __m512d c_re1 = _mm512_loadu_pd(&in_re[k+8+2*K]); __m512d c_im1 = _mm512_loadu_pd(&in_im[k+8+2*K]); \
        __m512d d_re1 = _mm512_loadu_pd(&in_re[k+8+3*K]); __m512d d_im1 = _mm512_loadu_pd(&in_im[k+8+3*K]); \
        __m512d a_re2 = _mm512_loadu_pd(&in_re[k+16]);    __m512d a_im2 = _mm512_loadu_pd(&in_im[k+16]); \
        __m512d b_re2 = _mm512_loadu_pd(&in_re[k+16+K]);  __m512d b_im2 = _mm512_loadu_pd(&in_im[k+16+K]); \
        __m512d c_re2 = _mm512_loadu_pd(&in_re[k+16+2*K]); __m512d c_im2 = _mm512_loadu_pd(&in_im[k+16+2*K]); \
        __m512d d_re2 = _mm512_loadu_pd(&in_re[k+16+3*K]); __m512d d_im2 = _mm512_loadu_pd(&in_im[k+16+3*K]); \
        __m512d a_re3 = _mm512_loadu_pd(&in_re[k+24]);    __m512d a_im3 = _mm512_loadu_pd(&in_im[k+24]); \
        __m512d b_re3 = _mm512_loadu_pd(&in_re[k+24+K]);  __m512d b_im3 = _mm512_loadu_pd(&in_im[k+24+K]); \
        __m512d c_re3 = _mm512_loadu_pd(&in_re[k+24+2*K]); __m512d c_im3 = _mm512_loadu_pd(&in_im[k+24+2*K]); \
        __m512d d_re3 = _mm512_loadu_pd(&in_re[k+24+3*K]); __m512d d_im3 = _mm512_loadu_pd(&in_im[k+24+3*K]); \
        __m512d a_re4 = _mm512_loadu_pd(&in_re[k+32]);    __m512d a_im4 = _mm512_loadu_pd(&in_im[k+32]); \
        __m512d b_re4 = _mm512_loadu_pd(&in_re[k+32+K]);  __m512d b_im4 = _mm512_loadu_pd(&in_im[k+32+K]); \
        __m512d c_re4 = _mm512_loadu_pd(&in_re[k+32+2*K]); __m512d c_im4 = _mm512_loadu_pd(&in_im[k+32+2*K]); \
        __m512d d_re4 = _mm512_loadu_pd(&in_re[k+32+3*K]); __m512d d_im4 = _mm512_loadu_pd(&in_im[k+32+3*K]); \
        __m512d a_re5 = _mm512_loadu_pd(&in_re[k+40]);    __m512d a_im5 = _mm512_loadu_pd(&in_im[k+40]); \
        __m512d b_re5 = _mm512_loadu_pd(&in_re[k+40+K]);  __m512d b_im5 = _mm512_loadu_pd(&in_im[k+40+K]); \
        __m512d c_re5 = _mm512_loadu_pd(&in_re[k+40+2*K]); __m512d c_im5 = _mm512_loadu_pd(&in_im[k+40+2*K]); \
        __m512d d_re5 = _mm512_loadu_pd(&in_re[k+40+3*K]); __m512d d_im5 = _mm512_loadu_pd(&in_im[k+40+3*K]); \
        __m512d a_re6 = _mm512_loadu_pd(&in_re[k+48]);    __m512d a_im6 = _mm512_loadu_pd(&in_im[k+48]); \
        __m512d b_re6 = _mm512_loadu_pd(&in_re[k+48+K]);  __m512d b_im6 = _mm512_loadu_pd(&in_im[k+48+K]); \
        __m512d c_re6 = _mm512_loadu_pd(&in_re[k+48+2*K]); __m512d c_im6 = _mm512_loadu_pd(&in_im[k+48+2*K]); \
        __m512d d_re6 = _mm512_loadu_pd(&in_re[k+48+3*K]); __m512d d_im6 = _mm512_loadu_pd(&in_im[k+48+3*K]); \
        __m512d a_re7 = _mm512_loadu_pd(&in_re[k+56]);    __m512d a_im7 = _mm512_loadu_pd(&in_im[k+56]); \
        __m512d b_re7 = _mm512_loadu_pd(&in_re[k+56+K]);  __m512d b_im7 = _mm512_loadu_pd(&in_im[k+56+K]); \
        __m512d c_re7 = _mm512_loadu_pd(&in_re[k+56+2*K]); __m512d c_im7 = _mm512_loadu_pd(&in_im[k+56+2*K]); \
        __m512d d_re7 = _mm512_loadu_pd(&in_re[k+56+3*K]); __m512d d_im7 = _mm512_loadu_pd(&in_im[k+56+3*K]); \
        __m512d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m512d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        __m512d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2; \
        __m512d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3; \
        __m512d y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4, y3_re4, y3_im4; \
        __m512d y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5, y3_re5, y3_im5; \
        __m512d y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6, y3_re6, y3_im6; \
        __m512d y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7, y3_re7, y3_im7; \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                              y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                              y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, d_re2, d_im2, \
                                              y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, d_re3, d_im3, \
                                              y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re4, a_im4, b_re4, b_im4, c_re4, c_im4, d_re4, d_im4, \
                                              y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4, y3_re4, y3_im4); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re5, a_im5, b_re5, b_im5, c_re5, c_im5, d_re5, d_im5, \
                                              y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5, y3_re5, y3_im5); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re6, a_im6, b_re6, b_im6, c_re6, c_im6, d_re6, d_im6, \
                                              y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6, y3_re6, y3_im6); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re7, a_im7, b_re7, b_im7, c_re7, c_im7, d_re7, d_im7, \
                                              y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7, y3_re7, y3_im7); \
        _mm512_storeu_pd(&out_re[k+0],     y0_re0); _mm512_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm512_storeu_pd(&out_re[k+0+K],   y1_re0); _mm512_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm512_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm512_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm512_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm512_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm512_storeu_pd(&out_re[k+8],     y0_re1); _mm512_storeu_pd(&out_im[k+8],     y0_im1); \
        _mm512_storeu_pd(&out_re[k+8+K],   y1_re1); _mm512_storeu_pd(&out_im[k+8+K],   y1_im1); \
        _mm512_storeu_pd(&out_re[k+8+2*K], y2_re1); _mm512_storeu_pd(&out_im[k+8+2*K], y2_im1); \
        _mm512_storeu_pd(&out_re[k+8+3*K], y3_re1); _mm512_storeu_pd(&out_im[k+8+3*K], y3_im1); \
        _mm512_storeu_pd(&out_re[k+16],    y0_re2); _mm512_storeu_pd(&out_im[k+16],    y0_im2); \
        _mm512_storeu_pd(&out_re[k+16+K],  y1_re2); _mm512_storeu_pd(&out_im[k+16+K],  y1_im2); \
        _mm512_storeu_pd(&out_re[k+16+2*K], y2_re2); _mm512_storeu_pd(&out_im[k+16+2*K], y2_im2); \
        _mm512_storeu_pd(&out_re[k+16+3*K], y3_re2); _mm512_storeu_pd(&out_im[k+16+3*K], y3_im2); \
        _mm512_storeu_pd(&out_re[k+24],    y0_re3); _mm512_storeu_pd(&out_im[k+24],    y0_im3); \
        _mm512_storeu_pd(&out_re[k+24+K],  y1_re3); _mm512_storeu_pd(&out_im[k+24+K],  y1_im3); \
        _mm512_storeu_pd(&out_re[k+24+2*K], y2_re3); _mm512_storeu_pd(&out_im[k+24+2*K], y2_im3); \
        _mm512_storeu_pd(&out_re[k+24+3*K], y3_re3); _mm512_storeu_pd(&out_im[k+24+3*K], y3_im3); \
        _mm512_storeu_pd(&out_re[k+32],    y0_re4); _mm512_storeu_pd(&out_im[k+32],    y0_im4); \
        _mm512_storeu_pd(&out_re[k+32+K],  y1_re4); _mm512_storeu_pd(&out_im[k+32+K],  y1_im4); \
        _mm512_storeu_pd(&out_re[k+32+2*K], y2_re4); _mm512_storeu_pd(&out_im[k+32+2*K], y2_im4); \
        _mm512_storeu_pd(&out_re[k+32+3*K], y3_re4); _mm512_storeu_pd(&out_im[k+32+3*K], y3_im4); \
        _mm512_storeu_pd(&out_re[k+40],    y0_re5); _mm512_storeu_pd(&out_im[k+40],    y0_im5); \
        _mm512_storeu_pd(&out_re[k+40+K],  y1_re5); _mm512_storeu_pd(&out_im[k+40+K],  y1_im5); \
        _mm512_storeu_pd(&out_re[k+40+2*K], y2_re5); _mm512_storeu_pd(&out_im[k+40+2*K], y2_im5); \
        _mm512_storeu_pd(&out_re[k+40+3*K], y3_re5); _mm512_storeu_pd(&out_im[k+40+3*K], y3_im5); \
        _mm512_storeu_pd(&out_re[k+48],    y0_re6); _mm512_storeu_pd(&out_im[k+48],    y0_im6); \
        _mm512_storeu_pd(&out_re[k+48+K],  y1_re6); _mm512_storeu_pd(&out_im[k+48+K],  y1_im6); \
        _mm512_storeu_pd(&out_re[k+48+2*K], y2_re6); _mm512_storeu_pd(&out_im[k+48+2*K], y2_im6); \
        _mm512_storeu_pd(&out_re[k+48+3*K], y3_re6); _mm512_storeu_pd(&out_im[k+48+3*K], y3_im6); \
        _mm512_storeu_pd(&out_re[k+56],    y0_re7); _mm512_storeu_pd(&out_im[k+56],    y0_im7); \
        _mm512_storeu_pd(&out_re[k+56+K],  y1_re7); _mm512_storeu_pd(&out_im[k+56+K],  y1_im7); \
        _mm512_storeu_pd(&out_re[k+56+2*K], y2_re7); _mm512_storeu_pd(&out_im[k+56+2*K], y2_im7); \
        _mm512_storeu_pd(&out_re[k+56+3*K], y3_re7); _mm512_storeu_pd(&out_im[k+56+3*K], y3_im7); \
    } while(0)

// Backward version - just change FV to BV!
#define RADIX4_NOTWIDDLE_PIPELINE_8_BV_AVX512(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m512d a_re0 = _mm512_loadu_pd(&in_re[k+0]);     __m512d a_im0 = _mm512_loadu_pd(&in_im[k+0]); \
        __m512d b_re0 = _mm512_loadu_pd(&in_re[k+0+K]);   __m512d b_im0 = _mm512_loadu_pd(&in_im[k+0+K]); \
        __m512d c_re0 = _mm512_loadu_pd(&in_re[k+0+2*K]); __m512d c_im0 = _mm512_loadu_pd(&in_im[k+0+2*K]); \
        __m512d d_re0 = _mm512_loadu_pd(&in_re[k+0+3*K]); __m512d d_im0 = _mm512_loadu_pd(&in_im[k+0+3*K]); \
        __m512d a_re1 = _mm512_loadu_pd(&in_re[k+8]);     __m512d a_im1 = _mm512_loadu_pd(&in_im[k+8]); \
        __m512d b_re1 = _mm512_loadu_pd(&in_re[k+8+K]);   __m512d b_im1 = _mm512_loadu_pd(&in_im[k+8+K]); \
        __m512d c_re1 = _mm512_loadu_pd(&in_re[k+8+2*K]); __m512d c_im1 = _mm512_loadu_pd(&in_im[k+8+2*K]); \
        __m512d d_re1 = _mm512_loadu_pd(&in_re[k+8+3*K]); __m512d d_im1 = _mm512_loadu_pd(&in_im[k+8+3*K]); \
        __m512d a_re2 = _mm512_loadu_pd(&in_re[k+16]);    __m512d a_im2 = _mm512_loadu_pd(&in_im[k+16]); \
        __m512d b_re2 = _mm512_loadu_pd(&in_re[k+16+K]);  __m512d b_im2 = _mm512_loadu_pd(&in_im[k+16+K]); \
        __m512d c_re2 = _mm512_loadu_pd(&in_re[k+16+2*K]); __m512d c_im2 = _mm512_loadu_pd(&in_im[k+16+2*K]); \
        __m512d d_re2 = _mm512_loadu_pd(&in_re[k+16+3*K]); __m512d d_im2 = _mm512_loadu_pd(&in_im[k+16+3*K]); \
        __m512d a_re3 = _mm512_loadu_pd(&in_re[k+24]);    __m512d a_im3 = _mm512_loadu_pd(&in_im[k+24]); \
        __m512d b_re3 = _mm512_loadu_pd(&in_re[k+24+K]);  __m512d b_im3 = _mm512_loadu_pd(&in_im[k+24+K]); \
        __m512d c_re3 = _mm512_loadu_pd(&in_re[k+24+2*K]); __m512d c_im3 = _mm512_loadu_pd(&in_im[k+24+2*K]); \
        __m512d d_re3 = _mm512_loadu_pd(&in_re[k+24+3*K]); __m512d d_im3 = _mm512_loadu_pd(&in_im[k+24+3*K]); \
        __m512d a_re4 = _mm512_loadu_pd(&in_re[k+32]);    __m512d a_im4 = _mm512_loadu_pd(&in_im[k+32]); \
        __m512d b_re4 = _mm512_loadu_pd(&in_re[k+32+K]);  __m512d b_im4 = _mm512_loadu_pd(&in_im[k+32+K]); \
        __m512d c_re4 = _mm512_loadu_pd(&in_re[k+32+2*K]); __m512d c_im4 = _mm512_loadu_pd(&in_im[k+32+2*K]); \
        __m512d d_re4 = _mm512_loadu_pd(&in_re[k+32+3*K]); __m512d d_im4 = _mm512_loadu_pd(&in_im[k+32+3*K]); \
        __m512d a_re5 = _mm512_loadu_pd(&in_re[k+40]);    __m512d a_im5 = _mm512_loadu_pd(&in_im[k+40]); \
        __m512d b_re5 = _mm512_loadu_pd(&in_re[k+40+K]);  __m512d b_im5 = _mm512_loadu_pd(&in_im[k+40+K]); \
        __m512d c_re5 = _mm512_loadu_pd(&in_re[k+40+2*K]); __m512d c_im5 = _mm512_loadu_pd(&in_im[k+40+2*K]); \
        __m512d d_re5 = _mm512_loadu_pd(&in_re[k+40+3*K]); __m512d d_im5 = _mm512_loadu_pd(&in_im[k+40+3*K]); \
        __m512d a_re6 = _mm512_loadu_pd(&in_re[k+48]);    __m512d a_im6 = _mm512_loadu_pd(&in_im[k+48]); \
        __m512d b_re6 = _mm512_loadu_pd(&in_re[k+48+K]);  __m512d b_im6 = _mm512_loadu_pd(&in_im[k+48+K]); \
        __m512d c_re6 = _mm512_loadu_pd(&in_re[k+48+2*K]); __m512d c_im6 = _mm512_loadu_pd(&in_im[k+48+2*K]); \
        __m512d d_re6 = _mm512_loadu_pd(&in_re[k+48+3*K]); __m512d d_im6 = _mm512_loadu_pd(&in_im[k+48+3*K]); \
        __m512d a_re7 = _mm512_loadu_pd(&in_re[k+56]);    __m512d a_im7 = _mm512_loadu_pd(&in_im[k+56]); \
        __m512d b_re7 = _mm512_loadu_pd(&in_re[k+56+K]);  __m512d b_im7 = _mm512_loadu_pd(&in_im[k+56+K]); \
        __m512d c_re7 = _mm512_loadu_pd(&in_re[k+56+2*K]); __m512d c_im7 = _mm512_loadu_pd(&in_im[k+56+2*K]); \
        __m512d d_re7 = _mm512_loadu_pd(&in_re[k+56+3*K]); __m512d d_im7 = _mm512_loadu_pd(&in_im[k+56+3*K]); \
        __m512d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m512d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        __m512d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2; \
        __m512d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3; \
        __m512d y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4, y3_re4, y3_im4; \
        __m512d y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5, y3_re5, y3_im5; \
        __m512d y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6, y3_re6, y3_im6; \
        __m512d y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7, y3_re7, y3_im7; \
        __m512d sign_mask = _mm512_set1_pd(-0.0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                              y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                              y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, d_re2, d_im2, \
                                              y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, d_re3, d_im3, \
                                              y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re4, a_im4, b_re4, b_im4, c_re4, c_im4, d_re4, d_im4, \
                                              y0_re4, y0_im4, y1_re4, y1_im4, y2_re4, y2_im4, y3_re4, y3_im4, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re5, a_im5, b_re5, b_im5, c_re5, c_im5, d_re5, d_im5, \
                                              y0_re5, y0_im5, y1_re5, y1_im5, y2_re5, y2_im5, y3_re5, y3_im5, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re6, a_im6, b_re6, b_im6, c_re6, c_im6, d_re6, d_im6, \
                                              y0_re6, y0_im6, y1_re6, y1_im6, y2_re6, y2_im6, y3_re6, y3_im6, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re7, a_im7, b_re7, b_im7, c_re7, c_im7, d_re7, d_im7, \
                                              y0_re7, y0_im7, y1_re7, y1_im7, y2_re7, y2_im7, y3_re7, y3_im7, sign_mask); \
        _mm512_storeu_pd(&out_re[k+0],     y0_re0); _mm512_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm512_storeu_pd(&out_re[k+0+K],   y1_re0); _mm512_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm512_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm512_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm512_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm512_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm512_storeu_pd(&out_re[k+8],     y0_re1); _mm512_storeu_pd(&out_im[k+8],     y0_im1); \
        _mm512_storeu_pd(&out_re[k+8+K],   y1_re1); _mm512_storeu_pd(&out_im[k+8+K],   y1_im1); \
        _mm512_storeu_pd(&out_re[k+8+2*K], y2_re1); _mm512_storeu_pd(&out_im[k+8+2*K], y2_im1); \
        _mm512_storeu_pd(&out_re[k+8+3*K], y3_re1); _mm512_storeu_pd(&out_im[k+8+3*K], y3_im1); \
        _mm512_storeu_pd(&out_re[k+16],    y0_re2); _mm512_storeu_pd(&out_im[k+16],    y0_im2); \
        _mm512_storeu_pd(&out_re[k+16+K],  y1_re2); _mm512_storeu_pd(&out_im[k+16+K],  y1_im2); \
        _mm512_storeu_pd(&out_re[k+16+2*K], y2_re2); _mm512_storeu_pd(&out_im[k+16+2*K], y2_im2); \
        _mm512_storeu_pd(&out_re[k+16+3*K], y3_re2); _mm512_storeu_pd(&out_im[k+16+3*K], y3_im2); \
        _mm512_storeu_pd(&out_re[k+24],    y0_re3); _mm512_storeu_pd(&out_im[k+24],    y0_im3); \
        _mm512_storeu_pd(&out_re[k+24+K],  y1_re3); _mm512_storeu_pd(&out_im[k+24+K],  y1_im3); \
        _mm512_storeu_pd(&out_re[k+24+2*K], y2_re3); _mm512_storeu_pd(&out_im[k+24+2*K], y2_im3); \
        _mm512_storeu_pd(&out_re[k+24+3*K], y3_re3); _mm512_storeu_pd(&out_im[k+24+3*K], y3_im3); \
        _mm512_storeu_pd(&out_re[k+32],    y0_re4); _mm512_storeu_pd(&out_im[k+32],    y0_im4); \
        _mm512_storeu_pd(&out_re[k+32+K],  y1_re4); _mm512_storeu_pd(&out_im[k+32+K],  y1_im4); \
        _mm512_storeu_pd(&out_re[k+32+2*K], y2_re4); _mm512_storeu_pd(&out_im[k+32+2*K], y2_im4); \
        _mm512_storeu_pd(&out_re[k+32+3*K], y3_re4); _mm512_storeu_pd(&out_im[k+32+3*K], y3_im4); \
        _mm512_storeu_pd(&out_re[k+40],    y0_re5); _mm512_storeu_pd(&out_im[k+40],    y0_im5); \
        _mm512_storeu_pd(&out_re[k+40+K],  y1_re5); _mm512_storeu_pd(&out_im[k+40+K],  y1_im5); \
        _mm512_storeu_pd(&out_re[k+40+2*K], y2_re5); _mm512_storeu_pd(&out_im[k+40+2*K], y2_im5); \
        _mm512_storeu_pd(&out_re[k+40+3*K], y3_re5); _mm512_storeu_pd(&out_im[k+40+3*K], y3_im5); \
        _mm512_storeu_pd(&out_re[k+48],    y0_re6); _mm512_storeu_pd(&out_im[k+48],    y0_im6); \
        _mm512_storeu_pd(&out_re[k+48+K],  y1_re6); _mm512_storeu_pd(&out_im[k+48+K],  y1_im6); \
        _mm512_storeu_pd(&out_re[k+48+2*K], y2_re6); _mm512_storeu_pd(&out_im[k+48+2*K], y2_im6); \
        _mm512_storeu_pd(&out_re[k+48+3*K], y3_re6); _mm512_storeu_pd(&out_im[k+48+3*K], y3_im6); \
        _mm512_storeu_pd(&out_re[k+56],    y0_re7); _mm512_storeu_pd(&out_im[k+56],    y0_im7); \
        _mm512_storeu_pd(&out_re[k+56+K],  y1_re7); _mm512_storeu_pd(&out_im[k+56+K],  y1_im7); \
        _mm512_storeu_pd(&out_re[k+56+2*K], y2_re7); _mm512_storeu_pd(&out_im[k+56+2*K], y2_im7); \
        _mm512_storeu_pd(&out_re[k+56+3*K], y3_re7); _mm512_storeu_pd(&out_im[k+56+3*K], y3_im7); \
    } while(0)

#endif // __AVX512F__

//==============================================================================
// AVX2 NO-TWIDDLE MACROS
//==============================================================================

#ifdef __AVX2__

// AVX2 depth-4: process 16 elements (4 lanes × 4 depth)
#define RADIX4_NOTWIDDLE_PIPELINE_4_FV_AVX2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m256d a_re0 = _mm256_loadu_pd(&in_re[k+0]);     __m256d a_im0 = _mm256_loadu_pd(&in_im[k+0]); \
        __m256d b_re0 = _mm256_loadu_pd(&in_re[k+0+K]);   __m256d b_im0 = _mm256_loadu_pd(&in_im[k+0+K]); \
        __m256d c_re0 = _mm256_loadu_pd(&in_re[k+0+2*K]); __m256d c_im0 = _mm256_loadu_pd(&in_im[k+0+2*K]); \
        __m256d d_re0 = _mm256_loadu_pd(&in_re[k+0+3*K]); __m256d d_im0 = _mm256_loadu_pd(&in_im[k+0+3*K]); \
        __m256d a_re1 = _mm256_loadu_pd(&in_re[k+4]);     __m256d a_im1 = _mm256_loadu_pd(&in_im[k+4]); \
        __m256d b_re1 = _mm256_loadu_pd(&in_re[k+4+K]);   __m256d b_im1 = _mm256_loadu_pd(&in_im[k+4+K]); \
        __m256d c_re1 = _mm256_loadu_pd(&in_re[k+4+2*K]); __m256d c_im1 = _mm256_loadu_pd(&in_im[k+4+2*K]); \
        __m256d d_re1 = _mm256_loadu_pd(&in_re[k+4+3*K]); __m256d d_im1 = _mm256_loadu_pd(&in_im[k+4+3*K]); \
        __m256d a_re2 = _mm256_loadu_pd(&in_re[k+8]);     __m256d a_im2 = _mm256_loadu_pd(&in_im[k+8]); \
        __m256d b_re2 = _mm256_loadu_pd(&in_re[k+8+K]);   __m256d b_im2 = _mm256_loadu_pd(&in_im[k+8+K]); \
        __m256d c_re2 = _mm256_loadu_pd(&in_re[k+8+2*K]); __m256d c_im2 = _mm256_loadu_pd(&in_im[k+8+2*K]); \
        __m256d d_re2 = _mm256_loadu_pd(&in_re[k+8+3*K]); __m256d d_im2 = _mm256_loadu_pd(&in_im[k+8+3*K]); \
        __m256d a_re3 = _mm256_loadu_pd(&in_re[k+12]);    __m256d a_im3 = _mm256_loadu_pd(&in_im[k+12]); \
        __m256d b_re3 = _mm256_loadu_pd(&in_re[k+12+K]);  __m256d b_im3 = _mm256_loadu_pd(&in_im[k+12+K]); \
        __m256d c_re3 = _mm256_loadu_pd(&in_re[k+12+2*K]); __m256d c_im3 = _mm256_loadu_pd(&in_im[k+12+2*K]); \
        __m256d d_re3 = _mm256_loadu_pd(&in_re[k+12+3*K]); __m256d d_im3 = _mm256_loadu_pd(&in_im[k+12+3*K]); \
        __m256d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m256d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        __m256d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2; \
        __m256d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3; \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                            y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                            y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, d_re2, d_im2, \
                                            y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, d_re3, d_im3, \
                                            y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3); \
        _mm256_storeu_pd(&out_re[k+0],     y0_re0); _mm256_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm256_storeu_pd(&out_re[k+0+K],   y1_re0); _mm256_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm256_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm256_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm256_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm256_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm256_storeu_pd(&out_re[k+4],     y0_re1); _mm256_storeu_pd(&out_im[k+4],     y0_im1); \
        _mm256_storeu_pd(&out_re[k+4+K],   y1_re1); _mm256_storeu_pd(&out_im[k+4+K],   y1_im1); \
        _mm256_storeu_pd(&out_re[k+4+2*K], y2_re1); _mm256_storeu_pd(&out_im[k+4+2*K], y2_im1); \
        _mm256_storeu_pd(&out_re[k+4+3*K], y3_re1); _mm256_storeu_pd(&out_im[k+4+3*K], y3_im1); \
        _mm256_storeu_pd(&out_re[k+8],     y0_re2); _mm256_storeu_pd(&out_im[k+8],     y0_im2); \
        _mm256_storeu_pd(&out_re[k+8+K],   y1_re2); _mm256_storeu_pd(&out_im[k+8+K],   y1_im2); \
        _mm256_storeu_pd(&out_re[k+8+2*K], y2_re2); _mm256_storeu_pd(&out_im[k+8+2*K], y2_im2); \
        _mm256_storeu_pd(&out_re[k+8+3*K], y3_re2); _mm256_storeu_pd(&out_im[k+8+3*K], y3_im2); \
        _mm256_storeu_pd(&out_re[k+12],    y0_re3); _mm256_storeu_pd(&out_im[k+12],    y0_im3); \
        _mm256_storeu_pd(&out_re[k+12+K],  y1_re3); _mm256_storeu_pd(&out_im[k+12+K],  y1_im3); \
        _mm256_storeu_pd(&out_re[k+12+2*K], y2_re3); _mm256_storeu_pd(&out_im[k+12+2*K], y2_im3); \
        _mm256_storeu_pd(&out_re[k+12+3*K], y3_re3); _mm256_storeu_pd(&out_im[k+12+3*K], y3_im3); \
    } while(0)

#define RADIX4_NOTWIDDLE_PIPELINE_4_BV_AVX2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m256d a_re0 = _mm256_loadu_pd(&in_re[k+0]);     __m256d a_im0 = _mm256_loadu_pd(&in_im[k+0]); \
        __m256d b_re0 = _mm256_loadu_pd(&in_re[k+0+K]);   __m256d b_im0 = _mm256_loadu_pd(&in_im[k+0+K]); \
        __m256d c_re0 = _mm256_loadu_pd(&in_re[k+0+2*K]); __m256d c_im0 = _mm256_loadu_pd(&in_im[k+0+2*K]); \
        __m256d d_re0 = _mm256_loadu_pd(&in_re[k+0+3*K]); __m256d d_im0 = _mm256_loadu_pd(&in_im[k+0+3*K]); \
        __m256d a_re1 = _mm256_loadu_pd(&in_re[k+4]);     __m256d a_im1 = _mm256_loadu_pd(&in_im[k+4]); \
        __m256d b_re1 = _mm256_loadu_pd(&in_re[k+4+K]);   __m256d b_im1 = _mm256_loadu_pd(&in_im[k+4+K]); \
        __m256d c_re1 = _mm256_loadu_pd(&in_re[k+4+2*K]); __m256d c_im1 = _mm256_loadu_pd(&in_im[k+4+2*K]); \
        __m256d d_re1 = _mm256_loadu_pd(&in_re[k+4+3*K]); __m256d d_im1 = _mm256_loadu_pd(&in_im[k+4+3*K]); \
        __m256d a_re2 = _mm256_loadu_pd(&in_re[k+8]);     __m256d a_im2 = _mm256_loadu_pd(&in_im[k+8]); \
        __m256d b_re2 = _mm256_loadu_pd(&in_re[k+8+K]);   __m256d b_im2 = _mm256_loadu_pd(&in_im[k+8+K]); \
        __m256d c_re2 = _mm256_loadu_pd(&in_re[k+8+2*K]); __m256d c_im2 = _mm256_loadu_pd(&in_im[k+8+2*K]); \
        __m256d d_re2 = _mm256_loadu_pd(&in_re[k+8+3*K]); __m256d d_im2 = _mm256_loadu_pd(&in_im[k+8+3*K]); \
        __m256d a_re3 = _mm256_loadu_pd(&in_re[k+12]);    __m256d a_im3 = _mm256_loadu_pd(&in_im[k+12]); \
        __m256d b_re3 = _mm256_loadu_pd(&in_re[k+12+K]);  __m256d b_im3 = _mm256_loadu_pd(&in_im[k+12+K]); \
        __m256d c_re3 = _mm256_loadu_pd(&in_re[k+12+2*K]); __m256d c_im3 = _mm256_loadu_pd(&in_im[k+12+2*K]); \
        __m256d d_re3 = _mm256_loadu_pd(&in_re[k+12+3*K]); __m256d d_im3 = _mm256_loadu_pd(&in_im[k+12+3*K]); \
        __m256d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m256d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        __m256d y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2; \
        __m256d y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3; \
        __m256d sign_mask = _mm256_set1_pd(-0.0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                            y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                            y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re2, a_im2, b_re2, b_im2, c_re2, c_im2, d_re2, d_im2, \
                                            y0_re2, y0_im2, y1_re2, y1_im2, y2_re2, y2_im2, y3_re2, y3_im2, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re3, a_im3, b_re3, b_im3, c_re3, c_im3, d_re3, d_im3, \
                                            y0_re3, y0_im3, y1_re3, y1_im3, y2_re3, y2_im3, y3_re3, y3_im3, sign_mask); \
        _mm256_storeu_pd(&out_re[k+0],     y0_re0); _mm256_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm256_storeu_pd(&out_re[k+0+K],   y1_re0); _mm256_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm256_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm256_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm256_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm256_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm256_storeu_pd(&out_re[k+4],     y0_re1); _mm256_storeu_pd(&out_im[k+4],     y0_im1); \
        _mm256_storeu_pd(&out_re[k+4+K],   y1_re1); _mm256_storeu_pd(&out_im[k+4+K],   y1_im1); \
        _mm256_storeu_pd(&out_re[k+4+2*K], y2_re1); _mm256_storeu_pd(&out_im[k+4+2*K], y2_im1); \
        _mm256_storeu_pd(&out_re[k+4+3*K], y3_re1); _mm256_storeu_pd(&out_im[k+4+3*K], y3_im1); \
        _mm256_storeu_pd(&out_re[k+8],     y0_re2); _mm256_storeu_pd(&out_im[k+8],     y0_im2); \
        _mm256_storeu_pd(&out_re[k+8+K],   y1_re2); _mm256_storeu_pd(&out_im[k+8+K],   y1_im2); \
        _mm256_storeu_pd(&out_re[k+8+2*K], y2_re2); _mm256_storeu_pd(&out_im[k+8+2*K], y2_im2); \
        _mm256_storeu_pd(&out_re[k+8+3*K], y3_re2); _mm256_storeu_pd(&out_im[k+8+3*K], y3_im2); \
        _mm256_storeu_pd(&out_re[k+12],    y0_re3); _mm256_storeu_pd(&out_im[k+12],    y0_im3); \
        _mm256_storeu_pd(&out_re[k+12+K],  y1_re3); _mm256_storeu_pd(&out_im[k+12+K],  y1_im3); \
        _mm256_storeu_pd(&out_re[k+12+2*K], y2_re3); _mm256_storeu_pd(&out_im[k+12+2*K], y2_im3); \
        _mm256_storeu_pd(&out_re[k+12+3*K], y3_re3); _mm256_storeu_pd(&out_im[k+12+3*K], y3_im3); \
    } while(0)

#endif // __AVX2__

//==============================================================================
// SSE2 & SCALAR - Following same pattern
//==============================================================================

#ifdef __SSE2__

#define RADIX4_NOTWIDDLE_PIPELINE_2_FV_SSE2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m128d a_re0 = _mm_loadu_pd(&in_re[k+0]);     __m128d a_im0 = _mm_loadu_pd(&in_im[k+0]); \
        __m128d b_re0 = _mm_loadu_pd(&in_re[k+0+K]);   __m128d b_im0 = _mm_loadu_pd(&in_im[k+0+K]); \
        __m128d c_re0 = _mm_loadu_pd(&in_re[k+0+2*K]); __m128d c_im0 = _mm_loadu_pd(&in_im[k+0+2*K]); \
        __m128d d_re0 = _mm_loadu_pd(&in_re[k+0+3*K]); __m128d d_im0 = _mm_loadu_pd(&in_im[k+0+3*K]); \
        __m128d a_re1 = _mm_loadu_pd(&in_re[k+2]);     __m128d a_im1 = _mm_loadu_pd(&in_im[k+2]); \
        __m128d b_re1 = _mm_loadu_pd(&in_re[k+2+K]);   __m128d b_im1 = _mm_loadu_pd(&in_im[k+2+K]); \
        __m128d c_re1 = _mm_loadu_pd(&in_re[k+2+2*K]); __m128d c_im1 = _mm_loadu_pd(&in_im[k+2+2*K]); \
        __m128d d_re1 = _mm_loadu_pd(&in_re[k+2+3*K]); __m128d d_im1 = _mm_loadu_pd(&in_im[k+2+3*K]); \
        __m128d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m128d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                            y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                            y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1); \
        _mm_storeu_pd(&out_re[k+0],     y0_re0); _mm_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm_storeu_pd(&out_re[k+0+K],   y1_re0); _mm_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm_storeu_pd(&out_re[k+2],     y0_re1); _mm_storeu_pd(&out_im[k+2],     y0_im1); \
        _mm_storeu_pd(&out_re[k+2+K],   y1_re1); _mm_storeu_pd(&out_im[k+2+K],   y1_im1); \
        _mm_storeu_pd(&out_re[k+2+2*K], y2_re1); _mm_storeu_pd(&out_im[k+2+2*K], y2_im1); \
        _mm_storeu_pd(&out_re[k+2+3*K], y3_re1); _mm_storeu_pd(&out_im[k+2+3*K], y3_im1); \
    } while(0)

#define RADIX4_NOTWIDDLE_PIPELINE_2_BV_SSE2(k, K, in_re, in_im, out_re, out_im) \
    do { \
        __m128d a_re0 = _mm_loadu_pd(&in_re[k+0]);     __m128d a_im0 = _mm_loadu_pd(&in_im[k+0]); \
        __m128d b_re0 = _mm_loadu_pd(&in_re[k+0+K]);   __m128d b_im0 = _mm_loadu_pd(&in_im[k+0+K]); \
        __m128d c_re0 = _mm_loadu_pd(&in_re[k+0+2*K]); __m128d c_im0 = _mm_loadu_pd(&in_im[k+0+2*K]); \
        __m128d d_re0 = _mm_loadu_pd(&in_re[k+0+3*K]); __m128d d_im0 = _mm_loadu_pd(&in_im[k+0+3*K]); \
        __m128d a_re1 = _mm_loadu_pd(&in_re[k+2]);     __m128d a_im1 = _mm_loadu_pd(&in_im[k+2]); \
        __m128d b_re1 = _mm_loadu_pd(&in_re[k+2+K]);   __m128d b_im1 = _mm_loadu_pd(&in_im[k+2+K]); \
        __m128d c_re1 = _mm_loadu_pd(&in_re[k+2+2*K]); __m128d c_im1 = _mm_loadu_pd(&in_im[k+2+2*K]); \
        __m128d d_re1 = _mm_loadu_pd(&in_re[k+2+3*K]); __m128d d_im1 = _mm_loadu_pd(&in_im[k+2+3*K]); \
        __m128d y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0; \
        __m128d y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1; \
        __m128d sign_mask = _mm_set1_pd(-0.0); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re0, a_im0, b_re0, b_im0, c_re0, c_im0, d_re0, d_im0, \
                                            y0_re0, y0_im0, y1_re0, y1_im0, y2_re0, y2_im0, y3_re0, y3_im0, sign_mask); \
        RADIX4_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re1, a_im1, b_re1, b_im1, c_re1, c_im1, d_re1, d_im1, \
                                            y0_re1, y0_im1, y1_re1, y1_im1, y2_re1, y2_im1, y3_re1, y3_im1, sign_mask); \
        _mm_storeu_pd(&out_re[k+0],     y0_re0); _mm_storeu_pd(&out_im[k+0],     y0_im0); \
        _mm_storeu_pd(&out_re[k+0+K],   y1_re0); _mm_storeu_pd(&out_im[k+0+K],   y1_im0); \
        _mm_storeu_pd(&out_re[k+0+2*K], y2_re0); _mm_storeu_pd(&out_im[k+0+2*K], y2_im0); \
        _mm_storeu_pd(&out_re[k+0+3*K], y3_re0); _mm_storeu_pd(&out_im[k+0+3*K], y3_im0); \
        _mm_storeu_pd(&out_re[k+2],     y0_re1); _mm_storeu_pd(&out_im[k+2],     y0_im1); \
        _mm_storeu_pd(&out_re[k+2+K],   y1_re1); _mm_storeu_pd(&out_im[k+2+K],   y1_im1); \
        _mm_storeu_pd(&out_re[k+2+2*K], y2_re1); _mm_storeu_pd(&out_im[k+2+2*K], y2_im1); \
        _mm_storeu_pd(&out_re[k+2+3*K], y3_re1); _mm_storeu_pd(&out_im[k+2+3*K], y3_im1); \
    } while(0)

#endif // __SSE2__

// Scalar fallback
#define RADIX4_NOTWIDDLE_PIPELINE_1_FV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do { \
        double a_re = in_re[k],     a_im = in_im[k]; \
        double b_re = in_re[k+K],   b_im = in_im[k+K]; \
        double c_re = in_re[k+2*K], c_im = in_im[k+2*K]; \
        double d_re = in_re[k+3*K], d_im = in_im[k+3*K]; \
        double sumBD_re = b_re + d_re, sumBD_im = b_im + d_im; \
        double difBD_re = b_re - d_re, difBD_im = b_im - d_im; \
        double sumAC_re = a_re + c_re, sumAC_im = a_im + c_im; \
        double difAC_re = a_re - c_re, difAC_im = a_im - c_im; \
        double rot_re = difBD_im, rot_im = -difBD_re; \
        out_re[k]     = sumAC_re + sumBD_re; out_im[k]     = sumAC_im + sumBD_im; \
        out_re[k+K]   = difAC_re - rot_re;   out_im[k+K]   = difAC_im - rot_im; \
        out_re[k+2*K] = sumAC_re - sumBD_re; out_im[k+2*K] = sumAC_im - sumBD_im; \
        out_re[k+3*K] = difAC_re + rot_re;   out_im[k+3*K] = difAC_im + rot_im; \
    } while(0)

#define RADIX4_NOTWIDDLE_PIPELINE_1_BV_SCALAR(k, K, in_re, in_im, out_re, out_im) \
    do { \
        double a_re = in_re[k],     a_im = in_im[k]; \
        double b_re = in_re[k+K],   b_im = in_im[k+K]; \
        double c_re = in_re[k+2*K], c_im = in_im[k+2*K]; \
        double d_re = in_re[k+3*K], d_im = in_im[k+3*K]; \
        double sumBD_re = b_re + d_re, sumBD_im = b_im + d_im; \
        double difBD_re = b_re - d_re, difBD_im = b_im - d_im; \
        double sumAC_re = a_re + c_re, sumAC_im = a_im + c_im; \
        double difAC_re = a_re - c_re, difAC_im = a_im - c_im; \
        double rot_re = -difBD_im, rot_im = difBD_re; \
        out_re[k]     = sumAC_re + sumBD_re; out_im[k]     = sumAC_im + sumBD_im; \
        out_re[k+K]   = difAC_re - rot_re;   out_im[k+K]   = difAC_im - rot_im; \
        out_re[k+2*K] = sumAC_re - sumBD_re; out_im[k+2*K] = sumAC_im - sumBD_im; \
        out_re[k+3*K] = difAC_re + rot_re;   out_im[k+3*K] = difAC_im + rot_im; \
    } while(0)

#endif // FFT_RADIX4_MACROS_NOTWIDDLE_MINIMAL_H