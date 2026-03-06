/**
 * @file fft_radix7_dispatch.h
 * @brief Radix-7 cross-ISA dispatch — strided AND packed paths
 *
 * ═══════════════════════════════════════════════════════════════════
 * TWO DISPATCH MODES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. STRIDED (default):
 *      radix7_tw_forward  / radix7_tw_backward
 *      radix7_notw_forward / radix7_notw_backward
 *    Data in stride-K layout: x[n*K + k].
 *
 * 2. PACKED (planner):
 *      radix7_tw_packed_fwd / radix7_tw_packed_bwd
 *      radix7_notw_packed_fwd / radix7_notw_packed_bwd
 *    Data in packed contiguous blocks: x[b*7*T + n*T + j].
 *    Kernel alone 1.1-2.3× faster than strided (eliminates stride wall).
 *    Amortize pack/unpack across stages in multi-stage FFT.
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
 * radix7_flat_tw_size(K)     = 6*K  (doubles per re/im component)
 * radix7_data_size(K)        = 7*K
 * radix7_packed_optimal_T(K) = 8 (avx512) / 4 (avx2) / 1 (scalar)
 */

#ifndef FFT_RADIX7_DISPATCH_H
#define FFT_RADIX7_DISPATCH_H

#include <stddef.h>

/* ── Include kernel headers ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix7_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix7_avx2.h"
#endif
#include "scalar/fft_radix7_scalar.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION & NAMING
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t vfft_detect_isa(void) {
#if defined(__AVX512F__) || defined(__AVX512F)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

static inline vfft_isa_level_t radix7_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix7_isa_name(vfft_isa_level_t isa) {
    switch (isa) {
        case VFFT_ISA_AVX512: return "AVX512";
        case VFFT_ISA_AVX2:   return "AVX2";
        default:               return "scalar";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void radix7_pack_input_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 56;  /* 7*8 */
        for (int n = 0; n < 7; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix7_unpack_output_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 56, dk = b * 8;
        for (int n = 0; n < 7; n++) {
            _mm512_storeu_pd(&dst_re[n*K + dk], _mm512_loadu_pd(&src_re[sk + n*8]));
            _mm512_storeu_pd(&dst_im[n*K + dk], _mm512_loadu_pd(&src_im[sk + n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix7_pack_twiddles_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 48;  /* 6*8 */
        for (int n = 0; n < 6; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}
#endif /* __AVX512F__ */

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — twiddled
 *
 * Signature: (K, in_re, in_im, out_re, out_im,
 *             flat_tw_re, flat_tw_im)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix7_tw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix7_tw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix7_tw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix7_tw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

static inline void radix7_tw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix7_tw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix7_tw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix7_tw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix7_notw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix7_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix7_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix7_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void radix7_notw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix7_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix7_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix7_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — twiddled
 *
 * Loops over K/T blocks, calling flat kernel at K=T per block.
 * Data AND twiddles must be in packed layout.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix7_tw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 7 * T;
    const size_t tw_bs = 6 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix7_tw_dit_kernel_fwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix7_tw_dit_kernel_fwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix7_tw_dit_kernel_fwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

static inline void radix7_tw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 7 * T;
    const size_t tw_bs = 6 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix7_tw_dit_kernel_bwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix7_tw_dit_kernel_bwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix7_tw_dit_kernel_bwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix7_notw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 7 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix7_notw_dit_kernel_fwd_avx512(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix7_notw_dit_kernel_fwd_avx2(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix7_notw_dit_kernel_fwd_scalar(
            in_re + b*bs, in_im + b*bs,
            out_re + b*bs, out_im + b*bs, T);
}

static inline void radix7_notw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 7 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix7_notw_dit_kernel_bwd_avx512(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix7_notw_dit_kernel_bwd_avx2(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix7_notw_dit_kernel_bwd_scalar(
            in_re + b*bs, in_im + b*bs,
            out_re + b*bs, out_im + b*bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix7_flat_tw_size(size_t K)  { return 6 * K; }
static inline size_t radix7_data_size(size_t K)     { return 7 * K; }

static inline size_t radix7_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX7_DISPATCH_H */
