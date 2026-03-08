/**
 * @file fft_radix7_dispatch.h
 * @brief Radix-7 cross-ISA dispatch
 *
 * DIT tw: fwd only (genfft). DIT bwd tw not available — planner uses DIF bwd.
 * N1 (notw): fwd + bwd available.
 */
#ifndef FFT_RADIX7_DISPATCH_H
#define FFT_RADIX7_DISPATCH_H

#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix7_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix7_avx2.h"
#endif
#include "scalar/fft_radix7_scalar.h"

/* DIF tw (for backward path) */
#include "scalar/fft_radix7_scalar_dif_tw.h"
#ifdef __AVX2__
#include "avx2/fft_radix7_avx2_dif_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix7_avx512_dif_tw.h"
#endif

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix7_effective_isa(size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
        return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
        return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline vfft_isa_level_t vfft_detect_isa(void)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

static inline const char *radix7_isa_name(vfft_isa_level_t isa)
{
    switch (isa)
    {
    case VFFT_ISA_AVX512:
        return "AVX512";
    case VFFT_ISA_AVX2:
        return "AVX2";
    default:
        return "scalar";
    }
}

/* ── DIT tw: fwd only ── */

static inline void radix7_tw_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix7_tw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix7_tw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix7_tw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

/* DIT bwd not available from genfft — backward uses DIF bwd path */
static inline void radix7_tw_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix7_tw_dif_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix7_tw_dif_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix7_tw_dif_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

/* ── N1 (notw): fwd + bwd ── */

static inline void radix7_notw_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix7_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix7_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix7_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void radix7_notw_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix7_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix7_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix7_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ── Planner helpers ── */

static inline size_t radix7_flat_tw_size(size_t K) { return 6 * K; }
static inline size_t radix7_data_size(size_t K) { return 7 * K; }

#endif /* FFT_RADIX7_DISPATCH_H */