/**
 * @file fft_radix25_dispatch.h
 * @brief Radix-25 cross-ISA dispatch — DIT twiddled (5x5 CT)
 *
 * No N1/notw codelet for R=25 — always used as twiddled outer stage.
 * Entry points: radix25_tw_forward / radix25_tw_backward
 */
#ifndef FFT_RADIX25_DISPATCH_H
#define FFT_RADIX25_DISPATCH_H

#include <stddef.h>

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

/* N1 (notw) */
#include "scalar/fft_radix25_scalar_n1.h"
#ifdef __AVX2__
#include "avx2/fft_radix25_avx2_n1.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix25_avx512_n1.h"
#endif

/* Twiddled (DIT) */
#include "scalar/fft_radix25_scalar_tw.h"
#ifdef __AVX2__
#include "avx2/fft_radix25_avx2_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix25_avx512_tw.h"
#endif

static inline vfft_isa_level_t radix25_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline void radix25_tw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix25_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix25_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix25_tw_flat_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
#endif
    default:
        radix25_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
}

static inline void radix25_tw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix25_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix25_tw_flat_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix25_tw_flat_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
#endif
    default:
        radix25_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
}

static inline size_t radix25_flat_tw_size(size_t K) { return 24 * K; }
static inline size_t radix25_data_size(size_t K)    { return 25 * K; }

/* ── N1 (notw) dispatch ── */

static inline void radix25_n1_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
    switch (radix25_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix25_n1_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix25_n1_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix25_n1_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

static inline void radix25_n1_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
    switch (radix25_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix25_n1_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix25_n1_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix25_n1_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

/* ── Twiddled (DIT) dispatch ── */

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED (IL) DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix25_avx512_il_tw.h"
#include "avx512/fft_radix25_avx512_il_dif_tw.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix25_avx2_il_tw.h"
#include "avx2/fft_radix25_avx2_il_dif_tw.h"
#endif
#include "scalar/fft_radix25_scalar_il_tw.h"
#include "scalar/fft_radix25_scalar_il_dif_tw.h"

static inline void radix25_tw_forward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix25_tw_flat_dit_kernel_fwd_il_avx512(in, out, tw_re, tw_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix25_tw_flat_dit_kernel_fwd_il_avx2(in, out, tw_re, tw_im, K); return; }
#endif
    radix25_tw_flat_dit_kernel_fwd_il_scalar(in, out, tw_re, tw_im, K);
}

static inline void radix25_tw_dif_backward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix25_tw_flat_dif_kernel_bwd_il_avx512(in, out, tw_re, tw_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix25_tw_flat_dif_kernel_bwd_il_avx2(in, out, tw_re, tw_im, K); return; }
#endif
    radix25_tw_flat_dif_kernel_bwd_il_scalar(in, out, tw_re, tw_im, K);
}

/* ── Monolithic N1 native IL (translated from FFTW genfft DAG) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix25_avx512_n1_mono_il.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix25_avx2_n1_mono_il.h"
#endif

static inline void radix25_n1_forward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) { radix25_n1_dit_kernel_fwd_il_avx512(in, out, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) { radix25_n1_dit_kernel_fwd_il_avx2(in, out, K); return; }
#endif
    (void)in; (void)out; (void)K;
}

static inline void radix25_n1_backward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) { radix25_n1_dit_kernel_bwd_il_avx512(in, out, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) { radix25_n1_dit_kernel_bwd_il_avx2(in, out, K); return; }
#endif
    (void)in; (void)out; (void)K;
}

#endif /* FFT_RADIX25_DISPATCH_H */
