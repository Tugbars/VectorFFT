/**
 * @file fft_radix7_tw_packed.h
 * @brief DFT-7 twiddled packed contiguous layout — eliminates stride-K wall
 *
 * ═══════════════════════════════════════════════════════════════════
 * LAYOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Packed data:    data_re[block*7*T + n*T + j]   n=0..6, j=0..T-1
 * Packed twiddle: tw_re[block*6*T + (n-1)*T + j] n=1..6, j=0..T-1
 *
 * Each block = 7*T contiguous doubles for data, 6*T for twiddles.
 * Every load is sequential — zero strides.
 *
 * T = block size = SIMD width:
 *   AVX-512: T=8  (7×8 = 56 doubles = 448 bytes per block)
 *   AVX2:    T=4  (7×4 = 28 doubles = 224 bytes per block)
 *   Scalar:  T=1
 *
 * ═══════════════════════════════════════════════════════════════════
 * REQUIRES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Include the kernel header(s) BEFORE this header:
 *   #include "fft_radix7_avx512.h"
 *   #include "fft_radix7_avx2.h"
 *   #include "fft_radix7_scalar.h"
 */

#ifndef FFT_RADIX7_TW_PACKED_H
#define FFT_RADIX7_TW_PACKED_H

#include <stddef.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * REPACK: strided → packed (scalar fallback)
 * ═══════════════════════════════════════════════════════════════ */

static inline void r7_pack_input(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 7; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[b * 7 * T + n * T + j] = src_re[n * K + b * T + j];
                dst_im[b * 7 * T + n * T + j] = src_im[n * K + b * T + j];
            }
}

static inline void r7_unpack_output(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 7; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[n * K + b * T + j] = src_re[b * 7 * T + n * T + j];
                dst_im[n * K + b * T + j] = src_im[b * 7 * T + n * T + j];
            }
}

static inline void r7_pack_twiddles(
    size_t K, size_t T,
    const double * __restrict__ flat_tw_re, const double * __restrict__ flat_tw_im,
    double * __restrict__ packed_tw_re, double * __restrict__ packed_tw_im)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 6; n++)
            for (size_t j = 0; j < T; j++) {
                packed_tw_re[b * 6 * T + n * T + j] = flat_tw_re[n * K + b * T + j];
                packed_tw_im[b * 6 * T + n * T + j] = flat_tw_im[n * K + b * T + j];
            }
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX512F__
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void r7_pack_input_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 56;  /* 7*8 */
        for (int n = 0; n < 7; n++) {
            _mm512_storeu_pd(&dr[dk + n*8], _mm512_loadu_pd(&sr[n*K + sk]));
            _mm512_storeu_pd(&di[dk + n*8], _mm512_loadu_pd(&si[n*K + sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void r7_unpack_output_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 56, dk = b * 8;
        for (int n = 0; n < 7; n++) {
            _mm512_storeu_pd(&dr[n*K + dk], _mm512_loadu_pd(&sr[sk + n*8]));
            _mm512_storeu_pd(&di[n*K + dk], _mm512_loadu_pd(&si[sk + n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void r7_pack_tw_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 48;  /* 6*8 */
        for (int n = 0; n < 6; n++) {
            _mm512_storeu_pd(&dr[dk + n*8], _mm512_loadu_pd(&sr[n*K + sk]));
            _mm512_storeu_pd(&di[dk + n*8], _mm512_loadu_pd(&si[n*K + sk]));
        }
    }
}

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * FLAT TWIDDLE TABLE BUILDER
 *
 * tw_re[(n-1)*K + k] = Re(W_{7K}^{n*k}), n=1..6
 * dir = -1 for forward, +1 for backward
 * ═══════════════════════════════════════════════════════════════ */

static inline void r7_build_flat_twiddles(
    size_t K, int dir,
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    const size_t NN = 7 * K;
    for (int n = 1; n < 7; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0 * M_PI * (double)n * (double)k / (double)NN;
            tw_re[(n - 1) * K + k] = cos(a);
            tw_im[(n - 1) * K + k] = dir * sin(a);
        }
}

/* ═══════════════════════════════════════════════════════════════
 * OPTIMAL BLOCK SIZE
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r7_packed_optimal_T(size_t K) {
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * BUFFER SIZES (doubles per re/im component)
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r7_data_buf_size(size_t K) { return 7 * K; }
static inline size_t r7_tw_buf_size(size_t K)   { return 6 * K; }

#endif /* FFT_RADIX7_TW_PACKED_H */
