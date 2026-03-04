/**
 * @file fft_radix32_avx2_tw_packed.h
 * @brief Packed-block twiddled DFT-32 AVX2 — contiguous layout
 *
 * Layout (T=4 for AVX2):
 *   packed_re[block*128 + n*4 + lane]   n=0..31, lane=0..3
 *
 * Each block of 4 DFT-32 instances: 128 contiguous doubles = 1KB.
 * Every load is aligned 32-byte. Zero strides.
 *
 * Twiddle layout:
 *   tw_packed_re[block*124 + (n-1)*4 + lane]  n=1..31
 */

#ifndef FFT_RADIX32_AVX2_TW_PACKED_H
#define FFT_RADIX32_AVX2_TW_PACKED_H

#include <immintrin.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * REPACK: stride-K ↔ packed-block layout (T=4 for AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2")))
static void r32a_repack_strided_to_packed(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 32; n++) {
            __m256d vr = _mm256_loadu_pd(&src_re[n*K + b*4]);
            __m256d vi = _mm256_loadu_pd(&src_im[n*K + b*4]);
            _mm256_store_pd(&dst_re[b*128 + n*4], vr);
            _mm256_store_pd(&dst_im[b*128 + n*4], vi);
        }
    }
}

__attribute__((target("avx2")))
static void r32a_repack_packed_to_strided(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 32; n++) {
            __m256d vr = _mm256_load_pd(&src_re[b*128 + n*4]);
            __m256d vi = _mm256_load_pd(&src_im[b*128 + n*4]);
            _mm256_storeu_pd(&dst_re[n*K + b*4], vr);
            _mm256_storeu_pd(&dst_im[n*K + b*4], vi);
        }
    }
}

__attribute__((target("avx2")))
static void r32a_repack_tw_to_packed(
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 31; n++) {
            __m256d vr = _mm256_loadu_pd(&tw_re[n*K + b*4]);
            __m256d vi = _mm256_loadu_pd(&tw_im[n*K + b*4]);
            _mm256_store_pd(&ptw_re[b*124 + n*4], vr);
            _mm256_store_pd(&ptw_im[b*124 + n*4], vi);
        }
    }
}

/* Generic super-block repack for arbitrary T (must be multiple of 4) */
__attribute__((target("avx2")))
static void r32a_repack_strided_to_super(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t ds = 32 * T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 32; n++)
            for (size_t j = 0; j < T; j += 4) {
                __m256d vr = _mm256_loadu_pd(&src_re[n*K + b*T + j]);
                __m256d vi = _mm256_loadu_pd(&src_im[n*K + b*T + j]);
                _mm256_store_pd(&dst_re[b*ds + n*T + j], vr);
                _mm256_store_pd(&dst_im[b*ds + n*T + j], vi);
            }
}

__attribute__((target("avx2")))
static void r32a_repack_super_to_strided(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t ds = 32 * T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 32; n++)
            for (size_t j = 0; j < T; j += 4) {
                __m256d vr = _mm256_load_pd(&src_re[b*ds + n*T + j]);
                __m256d vi = _mm256_load_pd(&src_im[b*ds + n*T + j]);
                _mm256_storeu_pd(&dst_re[n*K + b*T + j], vr);
                _mm256_storeu_pd(&dst_im[n*K + b*T + j], vi);
            }
}

__attribute__((target("avx2")))
static void r32a_repack_tw_to_super(
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t ts = 31 * T;
    for (size_t b = 0; b < nb; b++)
        for (int n = 0; n < 31; n++)
            for (size_t j = 0; j < T; j += 4) {
                __m256d vr = _mm256_loadu_pd(&tw_re[n*K + b*T + j]);
                __m256d vi = _mm256_loadu_pd(&tw_im[n*K + b*T + j]);
                _mm256_store_pd(&ptw_re[b*ts + n*T + j], vr);
                _mm256_store_pd(&ptw_im[b*ts + n*T + j], vi);
            }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DFT-32 DRIVERS
 * ═══════════════════════════════════════════════════════════════ */

/* T=4 blocks (minimum for AVX2) */
__attribute__((target("avx2,fma")))
static void radix32_tw_packed_fwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks)
{
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_fwd_avx2(
            in_re  + b*128, in_im  + b*128,
            out_re + b*128, out_im + b*128,
            tw_re  + b*124, tw_im  + b*124,
            4);
    }
}

__attribute__((target("avx2,fma")))
static void radix32_tw_packed_bwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks)
{
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_bwd_avx2(
            in_re  + b*128, in_im  + b*128,
            out_re + b*128, out_im + b*128,
            tw_re  + b*124, tw_im  + b*124,
            4);
    }
}

/* Super-block: T doubles per block */
__attribute__((target("avx2,fma")))
static void radix32_tw_packed_super_fwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks, size_t T)
{
    const size_t ds = 32 * T, ts = 31 * T;
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_fwd_avx2(
            in_re  + b*ds, in_im  + b*ds,
            out_re + b*ds, out_im + b*ds,
            tw_re  + b*ts, tw_im  + b*ts,
            T);
    }
}

__attribute__((target("avx2,fma")))
static void radix32_tw_packed_super_bwd_avx2(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks, size_t T)
{
    const size_t ds = 32 * T, ts = 31 * T;
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_bwd_avx2(
            in_re  + b*ds, in_im  + b*ds,
            out_re + b*ds, out_im + b*ds,
            tw_re  + b*ts, tw_im  + b*ts,
            T);
    }
}

#endif /* FFT_RADIX32_AVX2_TW_PACKED_H */
