/**
 * @file fft_radix128_n1_packed.h
 * @brief DFT-128 N1 packed contiguous layout — eliminates stride-K wall
 *
 * ═══════════════════════════════════════════════════════════════════
 * LAYOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Packed: data_re[block*128*T + n*T + j]  n=0..127, j=0..T-1
 *   Each block = 128*T contiguous doubles.
 *   Every load is sequential — zero strides.
 *
 * Strided: data_re[n*K + k]  n=0..127, k=0..K-1
 *   Standard Cooley-Tukey layout — stride-K between rows.
 *
 * T = block size = SIMD width (AVX2: 4, AVX-512: 8, scalar: 1)
 * K/T blocks, each calls the N1 kernel at K=T.
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHY PACKED WINS
 * ═══════════════════════════════════════════════════════════════════
 *
 * At K=64, stride-K = 64 doubles = 512 bytes between rows.
 * 128 rows × 512B = 64KB working set → L1 misses.
 *
 * Packed at T=4: each block = 128×4 = 512 doubles = 4KB.
 * Entire block fits in L1. N1 kernel runs at peak throughput.
 *
 * ═══════════════════════════════════════════════════════════════════
 * REQUIRES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Include the appropriate N1 kernel header BEFORE this header:
 *   #include "fft_radix128_scalar_n1_gen.h"  (scalar)
 *   #include "fft_radix128_avx2_n1_gen.h"    (AVX2)
 *   #include "fft_radix128_avx512_n1_gen.h"  (AVX-512)
 */

#ifndef FFT_RADIX128_N1_PACKED_H
#define FFT_RADIX128_N1_PACKED_H

#include <stddef.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * REPACK: stride-K ↔ packed-block layout
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Repack from stride-K to packed blocks of size T.
 * src_re[n*K + k] → dst_re[block*128*T + n*T + j]
 * where block = k/T, j = k%T.
 */
static inline void r128_repack_strided_to_packed(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t block_stride = 128 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 128; n++) {
            for (size_t j = 0; j < T; j++) {
                dst_re[b * block_stride + n * T + j] = src_re[n * K + b * T + j];
                dst_im[b * block_stride + n * T + j] = src_im[n * K + b * T + j];
            }
        }
    }
}

/**
 * Repack from packed blocks back to stride-K.
 */
static inline void r128_repack_packed_to_strided(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t block_stride = 128 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 128; n++) {
            for (size_t j = 0; j < T; j++) {
                dst_re[n * K + b * T + j] = src_re[b * block_stride + n * T + j];
                dst_im[n * K + b * T + j] = src_im[b * block_stride + n * T + j];
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * VECTORIZED REPACK (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
#include <immintrin.h>

__attribute__((target("avx2")))
static inline void r128_repack_strided_to_packed_avx2(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t block_stride = 128 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 128; n++) {
            for (size_t j = 0; j < T; j += 4) {
                __m256d vr = _mm256_loadu_pd(&src_re[n * K + b * T + j]);
                __m256d vi = _mm256_loadu_pd(&src_im[n * K + b * T + j]);
                _mm256_store_pd(&dst_re[b * block_stride + n * T + j], vr);
                _mm256_store_pd(&dst_im[b * block_stride + n * T + j], vi);
            }
        }
    }
}

__attribute__((target("avx2")))
static inline void r128_repack_packed_to_strided_avx2(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t block_stride = 128 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 128; n++) {
            for (size_t j = 0; j < T; j += 4) {
                __m256d vr = _mm256_load_pd(&src_re[b * block_stride + n * T + j]);
                __m256d vi = _mm256_load_pd(&src_im[b * block_stride + n * T + j]);
                _mm256_storeu_pd(&dst_re[n * K + b * T + j], vr);
                _mm256_storeu_pd(&dst_im[n * K + b * T + j], vi);
            }
        }
    }
}
#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════
 * PACKED N1 SUPER-BLOCK DRIVERS
 *
 * Process K/T blocks, each calling N1 at K=T.
 * Data must already be in packed layout.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r128_n1_packed_fwd_scalar(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_fwd_scalar(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}

static inline void r128_n1_packed_bwd_scalar(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_bwd_scalar(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}

#ifdef __AVX2__
__attribute__((target("avx2,fma")))
static inline void r128_n1_packed_fwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_fwd_avx2(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}

__attribute__((target("avx2,fma")))
static inline void r128_n1_packed_bwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_bwd_avx2(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}
#endif /* __AVX2__ */

#ifdef __AVX512F__
__attribute__((target("avx512f,avx512dq,fma")))
static inline void r128_n1_packed_fwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_fwd_avx512(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void r128_n1_packed_bwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 128 * T;
    for (size_t b = 0; b < nb; b++)
        radix128_n1_dit_kernel_bwd_avx512(
            in_re + b * bs, in_im + b * bs,
            out_re + b * bs, out_im + b * bs, T);
}
#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * OPTIMAL BLOCK SIZE
 *
 * AVX-512: T=8  (128×8 = 1024 doubles = 8KB per block, fits L1)
 * AVX2:    T=4  (128×4 = 512 doubles = 4KB per block, fits L1)
 * Scalar:  T=1  (no packing benefit, but still works)
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r128_packed_optimal_T(size_t K) {
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE: full pipeline (repack → N1 → unpack)
 *
 * For use when caller has strided data and wants strided output.
 * The planner should keep data packed between stages instead.
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r128_packed_buf_size(size_t K) {
    return 128 * K;  /* doubles per re/im component */
}

#endif /* FFT_RADIX128_N1_PACKED_H */
