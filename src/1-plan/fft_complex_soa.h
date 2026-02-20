/**
 * @file fft_complex_soa.h
 * @brief AoS ↔ SoA complex format conversion — public API
 *
 * Converts between interleaved (AoS) and split (SoA) complex representations
 * using SIMD-optimized kernels (AVX-512, AVX2, SSE2, scalar fallback).
 *
 *   AoS (interleaved): [re0, im0, re1, im1, re2, im2, ...]
 *   SoA (split):        re[] = [re0, re1, re2, ...]
 *                        im[] = [im0, im1, im2, ...]
 *
 * @author Tugbars
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_COMPLEX_SOA_H
#define FFT_COMPLEX_SOA_H

#include <stddef.h>

/**
 * @brief Deinterleave AoS → SoA (interleaved → split)
 *
 * @param[in]  interleaved  Input: [re0,im0,re1,im1,...] (2*n doubles)
 * @param[out] re           Output: real parts (n doubles)
 * @param[out] im           Output: imaginary parts (n doubles)
 * @param[in]  n            Number of complex elements
 */
void fft_deinterleave(const double *restrict interleaved,
                       double *restrict re,
                       double *restrict im,
                       size_t n);

/**
 * @brief Reinterleave SoA → AoS (split → interleaved)
 *
 * @param[in]  re           Input: real parts (n doubles)
 * @param[in]  im           Input: imaginary parts (n doubles)
 * @param[out] interleaved  Output: [re0,im0,re1,im1,...] (2*n doubles)
 * @param[in]  n            Number of complex elements
 */
void fft_reinterleave(const double *restrict re,
                       const double *restrict im,
                       double *restrict interleaved,
                       size_t n);

/**
 * @brief Query SIMD capabilities used by conversion routines
 */
const char *fft_soa_get_simd_capabilities(void);

#endif /* FFT_COMPLEX_SOA_H */
