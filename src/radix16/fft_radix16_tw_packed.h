/**
 * @file fft_radix16_tw_packed.h
 * @brief DFT-16 twiddled packed contiguous layout — eliminates stride-K wall
 *
 * ═══════════════════════════════════════════════════════════════════
 * LAYOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Packed data:    data_re[block*16*T + n*T + j]   n=0..15, j=0..T-1
 * Packed twiddle: tw_re[block*15*T + (n-1)*T + j] n=1..15, j=0..T-1
 *
 * Each block = 16*T contiguous doubles for data, 15*T for twiddles.
 * Every load is sequential — zero strides.
 *
 * T = block size = SIMD width:
 *   AVX-512: T=8  (16×8 = 128 doubles = 1KB per block)
 *   AVX2:    T=4  (16×4 = 64 doubles = 512 bytes per block)
 *   Scalar:  T=1
 *
 * ═══════════════════════════════════════════════════════════════════
 * REQUIRES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Include the kernel header(s) BEFORE this header:
 *   #include "fft_radix16_scalar_n1_gen.h"  (for N1 scalar drivers)
 *   #include "fft_radix16_avx2_n1_gen.h"    (for N1 AVX2 drivers)
 *   #include "fft_radix16_avx2_tw.h"        (for twiddled AVX2 drivers)
 *   #include "fft_radix16_avx512_tw.h"      (for twiddled AVX-512 drivers)
 */

#ifndef FFT_RADIX16_TW_PACKED_H
#define FFT_RADIX16_TW_PACKED_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * REPACK: strided → packed
 *
 * Data:    src_re[n*K + k] → dst_re[b*16*T + n*T + j]
 * Twiddle: src_tw_re[(n-1)*K + k] → dst_tw_re[b*15*T + (n-1)*T + j]
 *   where b = k/T, j = k%T
 * ═══════════════════════════════════════════════════════════════ */

static inline void r16_pack_input(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 16; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[b * 16 * T + n * T + j] = src_re[n * K + b * T + j];
                dst_im[b * 16 * T + n * T + j] = src_im[n * K + b * T + j];
            }
}

static inline void r16_unpack_output(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 16; n++)
            for (size_t j = 0; j < T; j++) {
                dst_re[n * K + b * T + j] = src_re[b * 16 * T + n * T + j];
                dst_im[n * K + b * T + j] = src_im[b * 16 * T + n * T + j];
            }
}

static inline void r16_pack_twiddles(
    size_t K, size_t T,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    double * __restrict__ packed_tw_re,
    double * __restrict__ packed_tw_im)
{
    const size_t nb = K / T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 15; n++)
            for (size_t j = 0; j < T; j++) {
                packed_tw_re[b * 15 * T + n * T + j] = flat_tw_re[n * K + b * T + j];
                packed_tw_im[b * 15 * T + n * T + j] = flat_tw_im[n * K + b * T + j];
            }
}

/* ═══════════════════════════════════════════════════════════════
 * FLAT TWIDDLE TABLE BUILDER
 *
 * tw_re[(n-1)*K + k] = Re(W_{16K}^{n*k}), n=1..15
 * dir = -1 for forward, +1 for backward
 * ═══════════════════════════════════════════════════════════════ */

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void r16_build_flat_twiddles(
    size_t K, int dir,
    double * __restrict__ tw_re,
    double * __restrict__ tw_im)
{
    const size_t NN = 16 * K;
    for (int n = 1; n < 16; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0 * M_PI * (double)n * (double)k / (double)NN;
            tw_re[(n - 1) * K + k] = cos(a);
            tw_im[(n - 1) * K + k] = dir * sin(a);
        }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED SUPER-BLOCK DRIVERS
 *
 * Process K/T blocks, each calling twiddled kernel at K=T.
 * Data AND twiddles must be in packed layout.
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX2__

__attribute__((target("avx2,fma")))
static inline void r16_tw_packed_fwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 16 * T;
    const size_t tw_bs = 15 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_fwd_avx2(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

__attribute__((target("avx2,fma")))
static inline void r16_tw_packed_bwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 16 * T;
    const size_t tw_bs = 15 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_bwd_avx2(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

#endif /* __AVX2__ */

#ifdef __AVX512F__

__attribute__((target("avx512f,avx512dq,fma")))
static inline void r16_tw_packed_fwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 16 * T;
    const size_t tw_bs = 15 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_fwd_avx512(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void r16_tw_packed_bwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 16 * T;
    const size_t tw_bs = 15 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_bwd_avx512(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * N1 PACKED SUPER-BLOCK DRIVERS
 *
 * Same repack, no twiddle tables. Calls N1 kernel at K=T per block.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r16_n1_packed_fwd_scalar(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_n1_dit_kernel_fwd_scalar(
            in_re + b*bs, in_im + b*bs, out_re + b*bs, out_im + b*bs, T);
}

static inline void r16_n1_packed_bwd_scalar(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_n1_dit_kernel_bwd_scalar(
            in_re + b*bs, in_im + b*bs, out_re + b*bs, out_im + b*bs, T);
}

#ifdef __AVX2__

__attribute__((target("avx2,fma")))
static inline void r16_n1_packed_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_n1_dit_kernel_fwd_avx2(
            in_re + b*bs, in_im + b*bs, out_re + b*bs, out_im + b*bs, T);
}

__attribute__((target("avx2,fma")))
static inline void r16_n1_packed_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    for (size_t b = 0; b < nb; b++)
        radix16_n1_dit_kernel_bwd_avx2(
            in_re + b*bs, in_im + b*bs, out_re + b*bs, out_im + b*bs, T);
}

#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * OPTIMAL BLOCK SIZE
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r16_packed_optimal_T(size_t K) {
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

static inline size_t r16_data_buf_size(size_t K) { return 16 * K; }
static inline size_t r16_tw_buf_size(size_t K) { return 15 * K; }

#endif /* FFT_RADIX16_TW_PACKED_H */