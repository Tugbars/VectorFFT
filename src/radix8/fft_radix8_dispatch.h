/**
 * @file fft_radix8_dispatch.h
 * @brief Radix-8 cross-ISA dispatch — strided, packed, and pack+walk paths
 *
 * DFT-8 = 2×DFT-4 + W8 combine. Only 1 constant: √2/2.
 * 52 adds + 4 muls per DFT-8 — nearly zero-multiply butterfly.
 *
 * Three-tier packed dispatch (same pattern as radix-32):
 *   K <= RADIX8_WALK_THRESHOLD: packed table (pre-packed twiddles)
 *   K >  RADIX8_WALK_THRESHOLD: pack+walk (2 walked bases, derive 5)
 *
 * Twiddle table: 7*K doubles per component (n=1..7).
 * radix8_flat_tw_size(K) = 7*K
 * radix8_data_size(K)    = 8*K
 */

#ifndef FFT_RADIX8_DISPATCH_H
#define FFT_RADIX8_DISPATCH_H

#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix8_avx512.h"
#include "avx512/fft_radix8_avx512_tw_pack_walk.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix8_avx2.h"
#include "avx2/fft_radix8_avx2_tw_pack_walk.h"
#endif
#include "scalar/fft_radix8_scalar.h"

/* Walk threshold: use pack+walk when K > this */
#ifndef RADIX8_WALK_THRESHOLD
#define RADIX8_WALK_THRESHOLD 1024
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix8_effective_isa(size_t K)
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

static inline const char *radix8_isa_name(vfft_isa_level_t isa)
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

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f"))) static inline void radix8_pack_input_avx512(
    const double *__restrict__ sr, const double *__restrict__ si,
    double *__restrict__ dr, double *__restrict__ di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
    {
        const size_t sk = b * 8, dk = b * 64;
        for (int n = 0; n < 8; n++)
        {
            _mm512_storeu_pd(&dr[dk + n * 8], _mm512_loadu_pd(&sr[n * K + sk]));
            _mm512_storeu_pd(&di[dk + n * 8], _mm512_loadu_pd(&si[n * K + sk]));
        }
    }
}

__attribute__((target("avx512f"))) static inline void radix8_unpack_output_avx512(
    const double *__restrict__ sr, const double *__restrict__ si,
    double *__restrict__ dr, double *__restrict__ di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
    {
        const size_t sk = b * 64, dk = b * 8;
        for (int n = 0; n < 8; n++)
        {
            _mm512_storeu_pd(&dr[n * K + dk], _mm512_loadu_pd(&sr[sk + n * 8]));
            _mm512_storeu_pd(&di[n * K + dk], _mm512_loadu_pd(&si[sk + n * 8]));
        }
    }
}

__attribute__((target("avx512f"))) static inline void radix8_pack_twiddles_avx512(
    const double *__restrict__ sr, const double *__restrict__ si,
    double *__restrict__ dr, double *__restrict__ di, size_t K)
{
    for (size_t b = 0; b < K / 8; b++)
    {
        const size_t sk = b * 8, dk = b * 56;
        for (int n = 0; n < 7; n++)
        {
            _mm512_storeu_pd(&dr[dk + n * 8], _mm512_loadu_pd(&sr[n * K + sk]));
            _mm512_storeu_pd(&di[dk + n * 8], _mm512_loadu_pd(&si[n * K + sk]));
        }
    }
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — twiddled
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix8_tw_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_tw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_tw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

static inline void radix8_tw_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_tw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_tw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix8_notw_forward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix8_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void radix8_notw_backward(
    size_t K,
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix8_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — twiddled
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix8_tw_packed_fwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T, dbs = 8 * T, tbs = 7 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_tw_dit_kernel_fwd_avx512(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_tw_dit_kernel_fwd_avx2(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix8_tw_dit_kernel_fwd_scalar(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
}

static inline void radix8_tw_packed_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T, dbs = 8 * T, tbs = 7 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_tw_dit_kernel_bwd_avx512(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_tw_dit_kernel_bwd_avx2(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix8_tw_dit_kernel_bwd_scalar(in_re + b * dbs, in_im + b * dbs, out_re + b * dbs, out_im + b * dbs, tw_re + b * tbs, tw_im + b * tbs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix8_notw_packed_fwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T, bs = 8 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_notw_dit_kernel_fwd_avx512(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_notw_dit_kernel_fwd_avx2(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix8_notw_dit_kernel_fwd_scalar(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
}

static inline void radix8_notw_packed_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T, bs = 8 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_notw_dit_kernel_bwd_avx512(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4)
    {
        for (size_t b = 0; b < nb; b++)
            radix8_notw_dit_kernel_bwd_avx2(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix8_notw_dit_kernel_bwd_scalar(in_re + b * bs, in_im + b * bs, out_re + b * bs, out_im + b * bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACK+WALK AUTO-DISPATCH
 *
 * K <= RADIX8_WALK_THRESHOLD → packed table
 * K >  RADIX8_WALK_THRESHOLD → pack+walk (zero tw table)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix8_tw_packed_auto_fwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    const void *__restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX8_WALK_THRESHOLD && walk_plan)
    {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8)
        {
            radix8_tw_pack_walk_fwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix8_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4)
        {
            radix8_tw_pack_walk_fwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix8_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    radix8_tw_packed_fwd(in_re, in_im, out_re, out_im,
                         tw_re, tw_im, K, T);
}

static inline void radix8_tw_packed_auto_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im,
    const void *__restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX8_WALK_THRESHOLD && walk_plan)
    {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8)
        {
            radix8_tw_pack_walk_bwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix8_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4)
        {
            radix8_tw_pack_walk_bwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix8_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    radix8_tw_packed_bwd(in_re, in_im, out_re, out_im,
                         tw_re, tw_im, K, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix8_flat_tw_size(size_t K) { return 7 * K; }
static inline size_t radix8_data_size(size_t K) { return 8 * K; }

/** Returns 1 if pack+walk should be used instead of packed table */
static inline int radix8_should_walk(size_t K)
{
    return (K > RADIX8_WALK_THRESHOLD) ? 1 : 0;
}

/** Packed twiddle table size. Returns 0 if walk mode (no table needed). */
static inline size_t radix8_packed_tw_size(size_t K)
{
    return radix8_should_walk(K) ? 0 : 7 * K;
}

static inline size_t radix8_packed_optimal_T(size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
        return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
        return 4;
#endif
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * INTERLEAVED (IL) DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix8_avx512_il.h"
#include "avx512/fft_radix8_avx512_il_dif_tw.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix8_avx2_il.h"
#include "avx2/fft_radix8_avx2_il_dif_tw.h"
#endif
#include "scalar/fft_radix8_scalar_il.h"
#include "scalar/fft_radix8_scalar_il_dif_tw.h"

static inline void radix8_tw_forward_il(
    size_t K,
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_tw_dit_kernel_fwd_il_avx512(in, out, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_tw_dit_kernel_fwd_il_avx2(in, out, tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dit_kernel_fwd_il_scalar(in, out, tw_re, tw_im, K);
}

static inline void radix8_tw_dif_backward_il(
    size_t K,
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ tw_re, const double *__restrict__ tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0)
    {
        radix8_tw_dif_kernel_bwd_il_avx512(in, out, tw_re, tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix8_tw_dif_kernel_bwd_il_avx2(in, out, tw_re, tw_im, K);
        return;
    }
#endif
    radix8_tw_dif_kernel_bwd_il_scalar(in, out, tw_re, tw_im, K);
}

#endif /* FFT_RADIX8_DISPATCH_H */