/**
 * @file fft_radix32_dif_dispatch.h
 * @brief Radix-32 DIF cross-ISA dispatch — strided + packed paths
 *
 * Include AFTER fft_radix32_dispatch.h (reuses ISA detection, pack/unpack).
 *
 * DIF codelets: twiddle applied AFTER butterfly (output side).
 * Same twiddle table format and packed layout as DIT.
 *
 * ═══════════════════════════════════════════════════════════════════
 * ENTRY POINTS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Strided:
 *   radix32_tw_dif_forward  / radix32_tw_dif_backward
 *
 * Packed:
 *   radix32_tw_packed_dif_fwd / radix32_tw_packed_dif_bwd
 *
 * Twiddle table, pack/unpack, buffer sizes: reused from DIT dispatch.
 */

#ifndef FFT_RADIX32_DIF_DISPATCH_H
#define FFT_RADIX32_DIF_DISPATCH_H

/* ═══════════════════════════════════════════════════════════════
 * DIF CODELET INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

#include "scalar/fft_radix32_scalar_dif_tw.h"

#ifdef __AVX2__
#include "avx2/fft_radix32_avx2_dif_tw.h"
#endif

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix32_avx512_dif_tw.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * DIF STRIDED DISPATCH
 *
 * Same ISA routing as DIT strided. Flat twiddles only (no ladder
 * for DIF — ladder requires DIT-specific twiddle interleaving).
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_dif_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ flat_tw_re, const double *__restrict__ flat_tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix32_tw_flat_dif_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix32_tw_flat_dif_kernel_fwd_avx2(in_re, in_im, out_re, out_im,
                                            flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
    radix32_tw_flat_dif_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                          flat_tw_re, flat_tw_im, K);
}

static inline void radix32_tw_dif_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ flat_tw_re, const double *__restrict__ flat_tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix32_tw_flat_dif_kernel_bwd_avx512(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix32_tw_flat_dif_kernel_bwd_avx2(in_re, in_im, out_re, out_im,
                                            flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
    radix32_tw_flat_dif_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                          flat_tw_re, flat_tw_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * DIF PACKED DISPATCH
 *
 * Same packed layout as DIT. Reuses pack/unpack from DIT dispatch.
 * Flat kernel at K=T per block — no ladder needed.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_packed_dif_fwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 32 * T;
    const size_t tw_bs = 31 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dif_kernel_fwd_avx512(
                in_re + b * data_bs, in_im + b * data_bs,
                out_re + b * data_bs, out_im + b * data_bs,
                tw_re + b * tw_bs, tw_im + b * tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dif_kernel_fwd_avx2(
                in_re + b * data_bs, in_im + b * data_bs,
                out_re + b * data_bs, out_im + b * data_bs,
                tw_re + b * tw_bs, tw_im + b * tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_tw_flat_dif_kernel_fwd_scalar(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

static inline void radix32_tw_packed_dif_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 32 * T;
    const size_t tw_bs = 31 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dif_kernel_bwd_avx512(
                in_re + b * data_bs, in_im + b * data_bs,
                out_re + b * data_bs, out_im + b * data_bs,
                tw_re + b * tw_bs, tw_im + b * tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dif_kernel_bwd_avx2(
                in_re + b * data_bs, in_im + b * data_bs,
                out_re + b * data_bs, out_im + b * data_bs,
                tw_re + b * tw_bs, tw_im + b * tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_tw_flat_dif_kernel_bwd_scalar(
            in_re + b * data_bs, in_im + b * data_bs,
            out_re + b * data_bs, out_im + b * data_bs,
            tw_re + b * tw_bs, tw_im + b * tw_bs, T);
}

#endif /* FFT_RADIX32_DIF_DISPATCH_H */
