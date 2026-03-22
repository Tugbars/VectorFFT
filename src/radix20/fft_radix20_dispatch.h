/**
 * @file fft_radix20_dispatch.h
 * @brief Radix-20 cross-ISA dispatch — N1, DIT tw, DIF tw, IL
 *
 * DIT tw: DAG codelet (genfft, Frigo/Johnson) at low K, 4×5 CT at high K.
 * DIF tw: 4×5 CT (all K).
 * N1: 4×5 CT (split) + 4×5 CT (mono IL).
 *
 * AVX2: sequential passes (peak 12 YMM).
 * AVX-512: ILP-paired sub-FFTs (peak 24 ZMM, 8 free).
 */
#ifndef FFT_RADIX20_DISPATCH_H
#define FFT_RADIX20_DISPATCH_H
#include <stddef.h>

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

/* ── N1 (notw) codelets ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix20_avx512_n1_gen.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix20_avx2_n1_gen.h"
#endif
#include "scalar/fft_radix20_scalar_n1_gen.h"

/* ── Split tw codelets (DIT + DIF, 4×5 CT) ── */
#ifdef __AVX2__
#include "avx2/fft_radix20_avx2_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix20_avx512_tw.h"
#endif

/* ── DAG tw codelets (DIT only, genfft DAG — optimal at low K) ── */
#ifdef __AVX2__
#include "avx2/fft_radix20_avx2_tw_dag.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix20_avx512_tw_dag.h"
#endif
#include "scalar/fft_radix20_scalar_tw.h"

/* K threshold: DAG wins at K≤64, CT wins at K≥128 (benched March 2026) */
#define RADIX20_DAG_K_THRESHOLD 96

/* ── IL tw codelets (DIT + DIF) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix20_avx512_il_tw.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix20_avx2_il_tw.h"
#endif

/* ── ISA selection ── */
static inline vfft_isa_level_t radix20_effective_isa(size_t K)
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

/* ── N1 dispatch ── */
static inline void radix20_n1_forward(size_t K,
                                      const double *__restrict__ ir, const double *__restrict__ ii,
                                      double *__restrict__ or_, double *__restrict__ oi)
{
    switch (radix20_effective_isa(K))
    {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix20_ct_n1_kernel_fwd_avx512(ir, ii, or_, oi, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix20_ct_n1_fwd_avx2(ir, ii, or_, oi, K);
        return;
#endif
    default:
        radix20_ct_n1_kernel_fwd_scalar(ir, ii, or_, oi, K);
        return;
    }
}
static inline void radix20_n1_backward(size_t K,
                                       const double *__restrict__ ir, const double *__restrict__ ii,
                                       double *__restrict__ or_, double *__restrict__ oi)
{
    switch (radix20_effective_isa(K))
    {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix20_ct_n1_kernel_bwd_avx512(ir, ii, or_, oi, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix20_ct_n1_bwd_avx2(ir, ii, or_, oi, K);
        return;
#endif
    default:
        radix20_ct_n1_kernel_bwd_scalar(ir, ii, or_, oi, K);
        return;
    }
}

/* ── DIT tw dispatch (DAG at low K, 4×5CT at high K, scalar fallback) ── */
static inline void radix20_tw_forward(size_t K,
                                      const double *__restrict__ ir, const double *__restrict__ ii,
                                      double *__restrict__ or_, double *__restrict__ oi,
                                      const double *__restrict__ twr, const double *__restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        if (K <= RADIX20_DAG_K_THRESHOLD)
            radix20_tw_dag_dit_contig_fwd_avx512(ir, ii, or_, oi, twr, twi, K);
        else
            radix20_ct_tw_dit_kernel_fwd_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        if (K <= RADIX20_DAG_K_THRESHOLD)
            radix20_tw_dag_dit_contig_fwd_avx2(ir, ii, or_, oi, twr, twi, K);
        else
            radix20_ct_tw_strided_fwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix20_ct_tw_dit_kernel_fwd_scalar(ir, ii, or_, oi, twr, twi, K);
}
static inline void radix20_tw_backward(size_t K,
                                       const double *__restrict__ ir, const double *__restrict__ ii,
                                       double *__restrict__ or_, double *__restrict__ oi,
                                       const double *__restrict__ twr, const double *__restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        if (K <= RADIX20_DAG_K_THRESHOLD)
            radix20_tw_dag_dit_contig_bwd_avx512(ir, ii, or_, oi, twr, twi, K);
        else
            radix20_ct_tw_dit_kernel_bwd_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        if (K <= RADIX20_DAG_K_THRESHOLD)
            radix20_tw_dag_dit_contig_bwd_avx2(ir, ii, or_, oi, twr, twi, K);
        else
            radix20_ct_tw_strided_bwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix20_ct_tw_dit_kernel_bwd_scalar(ir, ii, or_, oi, twr, twi, K);
}

/* ── Scalar DIF fallback: N1 + post-multiply (for odd K) ── */
static inline void radix20_tw_dif_scalar_fwd(size_t K,
                                             const double *__restrict__ ir, const double *__restrict__ ii,
                                             double *__restrict__ or_, double *__restrict__ oi,
                                             const double *__restrict__ twr, const double *__restrict__ twi)
{
    radix20_ct_n1_kernel_fwd_scalar(ir, ii, or_, oi, K);
    for (size_t n = 1; n < 20; n++)
        for (size_t k = 0; k < K; k++)
        {
            double xr = or_[n * K + k], xi = oi[n * K + k];
            double wr = twr[(n - 1) * K + k], wi = twi[(n - 1) * K + k];
            or_[n * K + k] = xr * wr - xi * wi;
            oi[n * K + k] = xr * wi + xi * wr;
        }
}
static inline void radix20_tw_dif_scalar_bwd(size_t K,
                                             const double *__restrict__ ir, const double *__restrict__ ii,
                                             double *__restrict__ or_, double *__restrict__ oi,
                                             const double *__restrict__ twr, const double *__restrict__ twi)
{
    radix20_ct_n1_kernel_bwd_scalar(ir, ii, or_, oi, K);
    for (size_t n = 1; n < 20; n++)
        for (size_t k = 0; k < K; k++)
        {
            double xr = or_[n * K + k], xi = oi[n * K + k];
            double wr = twr[(n - 1) * K + k], wi = twi[(n - 1) * K + k];
            or_[n * K + k] = xr * wr + xi * wi;
            oi[n * K + k] = xi * wr - xr * wi;
        }
}

/* ── DIF tw dispatch (4×5CT SIMD, scalar N1+tw fallback) ── */
static inline void radix20_tw_dif_forward(size_t K,
                                          const double *__restrict__ ir, const double *__restrict__ ii,
                                          double *__restrict__ or_, double *__restrict__ oi,
                                          const double *__restrict__ twr, const double *__restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix20_ct_tw_dif_kernel_fwd_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_dif_strided_fwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix20_tw_dif_scalar_fwd(K, ir, ii, or_, oi, twr, twi);
}
static inline void radix20_tw_dif_backward(size_t K,
                                           const double *__restrict__ ir, const double *__restrict__ ii,
                                           double *__restrict__ or_, double *__restrict__ oi,
                                           const double *__restrict__ twr, const double *__restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix20_ct_tw_dif_kernel_bwd_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_dif_strided_bwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix20_tw_dif_scalar_bwd(K, ir, ii, or_, oi, twr, twi);
}

/* ── IL tw dispatch ── */
static inline void radix20_tw_forward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_tw_dit_kernel_fwd_il_avx512(in, out, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix20_ct_tw_dit_kernel_fwd_il_avx2(in, out, twr, twi, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)twr;
    (void)twi;
    (void)K;
}
static inline void radix20_tw_dif_backward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_tw_dif_kernel_bwd_il_avx512(in, out, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix20_ct_tw_dif_kernel_bwd_il_avx2(in, out, twr, twi, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)twr;
    (void)twi;
    (void)K;
}

/* ── N1 native IL (4×5 CT mono IL) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix20_avx512_n1_mono_il.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix20_avx2_n1_mono_il.h"
#endif

static inline void radix20_n1_forward_il(
    size_t K,
    const double *__restrict__ in, double *__restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_n1_kernel_fwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix20_ct_n1_kernel_fwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)K;
}

static inline void radix20_n1_backward_il(
    size_t K,
    const double *__restrict__ in, double *__restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0)
    {
        radix20_ct_n1_kernel_bwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix20_ct_n1_kernel_bwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)K;
}

static inline size_t radix20_flat_tw_size(size_t K) { return 19 * K; }
static inline size_t radix20_data_size(size_t K) { return 20 * K; }

#endif /* FFT_RADIX20_DISPATCH_H */
