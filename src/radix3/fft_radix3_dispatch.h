/**
 * @file fft_radix3_dispatch.h
 * @brief Radix-3 cross-ISA dispatch — strided AND packed paths
 *
 * ═══════════════════════════════════════════════════════════════════
 * TWO DISPATCH MODES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. STRIDED (default): radix3_tw_fwd / radix3_tw_bwd
 *    Data in stride-K layout: x[n*K + k].
 *    Best for standalone single-stage calls.
 *
 * 2. PACKED (planner): radix3_tw_packed_fwd / radix3_tw_packed_bwd
 *    Data in packed contiguous blocks: x[b*3*T + n*T + j].
 *    Best for multi-stage pipeline where data stays packed.
 *    Kernel alone is 1.2-2× faster than strided.
 *    Pack/unpack cost makes single-stage calls slower.
 *
 * The planner should:
 *   - Pack once at FFT entry  (radix3_pack_input_avx512 / r3_pack_input)
 *   - Use packed dispatch for all intermediate stages
 *   - Unpack once at FFT exit (radix3_unpack_output_avx512 / r3_unpack_output)
 *   - Pre-pack twiddles at plan time (one-time cost)
 *
 * ═══════════════════════════════════════════════════════════════════
 * ISA ROUTING
 * ═══════════════════════════════════════════════════════════════════
 *
 * AVX-512: K % 8 == 0 && K >= 8
 * AVX2:    K % 4 == 0 && K >= 4
 * Scalar:  any K
 *
 * ═══════════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════════
 *
 * radix3_flat_tw_size(K)     = 2*K  (doubles per re/im component)
 * radix3_data_size(K)        = 3*K
 * radix3_packed_optimal_T(K) = 8 (avx512) / 4 (avx2) / 1 (scalar)
 */

#ifndef FFT_RADIX3_DISPATCH_H
#define FFT_RADIX3_DISPATCH_H

#include <stddef.h>

/* ── Include kernel headers ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix3_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix3_avx2.h"
#endif
#include "scalar/fft_radix3_scalar.h"

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void radix3_pack_input_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 24;
        for (int n = 0; n < 3; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix3_unpack_output_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 24, dk = b * 8;
        for (int n = 0; n < 3; n++) {
            _mm512_storeu_pd(&dst_re[n*K + dk], _mm512_loadu_pd(&src_re[sk + n*8]));
            _mm512_storeu_pd(&dst_im[n*K + dk], _mm512_loadu_pd(&src_im[sk + n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix3_pack_twiddles_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 16;
        for (int n = 0; n < 2; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}
#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH (default — single-stage calls)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix3_tw_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix3_tw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix3_tw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix3_tw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

static inline void radix3_tw_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix3_tw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix3_tw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix3_tw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

/* ── Notw strided dispatch ── */

static inline void radix3_notw_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix3_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix3_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix3_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void radix3_notw_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix3_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix3_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix3_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH (planner — multi-stage pipeline)
 *
 * Data must already be in packed layout. T = SIMD width.
 * Loops over K/T blocks, calling kernel(K=T) per block.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix3_tw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 3 * T;
    const size_t tw_bs = 2 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix3_tw_dit_kernel_fwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix3_tw_dit_kernel_fwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix3_tw_dit_kernel_fwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

static inline void radix3_tw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 3 * T;
    const size_t tw_bs = 2 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix3_tw_dit_kernel_bwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix3_tw_dit_kernel_bwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix3_tw_dit_kernel_bwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix3_flat_tw_size(size_t K) { return 2 * K; }
static inline size_t radix3_data_size(size_t K)    { return 3 * K; }

static inline size_t radix3_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX3_DISPATCH_H */
