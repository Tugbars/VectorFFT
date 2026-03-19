/**
 * @file fft_radix32_dispatch.h
 * @brief Radix-32 cross-ISA dispatch — strided, packed, and pack+walk paths
 *
 * ═══════════════════════════════════════════════════════════════════
 * THREE DISPATCH MODES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. STRIDED (default):
 *      radix32_tw_forward  / radix32_tw_backward
 *      radix32_notw_forward / radix32_notw_backward
 *    Data in stride-K layout. Routes: flat (K<=64), ladder (K>=128).
 *
 * 2. PACKED (planner, K <= RADIX32_WALK_THRESHOLD):
 *      radix32_tw_packed_fwd / radix32_tw_packed_bwd
 *    Data in packed contiguous blocks + pre-packed twiddle table.
 *    Kernel alone 1.3-1.8× faster than strided.
 *
 * 3. PACK+WALK (planner, K > RADIX32_WALK_THRESHOLD):
 *      radix32_tw_pack_walk_fwd / radix32_tw_pack_walk_bwd
 *    Data in packed contiguous blocks, twiddles generated on-the-fly.
 *    Walks 5 base accumulators, derives 31 via binary tree per block.
 *    Eliminates twiddle table entirely. 1.2-1.3× faster than packed
 *    at K>=1024 where the 31*K tw table exceeds L2.
 *
 * The planner should call radix32_tw_packed_auto_fwd/_bwd which
 * automatically selects packed-table vs pack+walk based on K.
 *
 * ═══════════════════════════════════════════════════════════════════
 * STRIDED ISA ROUTING (tw)
 * ═══════════════════════════════════════════════════════════════════
 *
 * AVX-512: K%8==0, K>=8
 *   K<=64:  flat twiddles
 *   128<=K<256: ladder U1 (5 base loads, derive 31)
 *   K>=256: ladder U2 (2 k-strips per iteration)
 * AVX2: K%4==0, K>=4 → flat only
 * Scalar: any K → flat only
 *
 * ═══════════════════════════════════════════════════════════════════
 * PACKED ROUTING
 * ═══════════════════════════════════════════════════════════════════
 *
 * K <= RADIX32_WALK_THRESHOLD:
 *   Flat kernel at K=T per block, twiddles from pre-packed table.
 * K > RADIX32_WALK_THRESHOLD:
 *   Flat kernel at K=T per block, twiddles derived on-the-fly from
 *   5 walking base accumulators (zero twiddle table allocation).
 *
 * ═══════════════════════════════════════════════════════════════════
 * PLANNER USAGE EXAMPLE
 * ═══════════════════════════════════════════════════════════════════
 *
 *   size_t T = radix32_packed_optimal_T(K);
 *   radix32_walk_plan_t walk_plan, *wp = NULL;
 *   double *pk_twr = NULL, *pk_twi = NULL;
 *
 *   if (radix32_should_walk(K)) {
 *       radix32_walk_plan_init(&walk_plan, K);  // 24 doubles, no table
 *       wp = &walk_plan;
 *   } else {
 *       pk_twr = alloc(31*K); pk_twi = alloc(31*K);
 *       radix32_pack_twiddles_avx512(flat_twr, flat_twi, pk_twr, pk_twi, K);
 *   }
 *
 *   radix32_tw_packed_auto_fwd(pk_in_re, pk_in_im, pk_out_re, pk_out_im,
 *                              pk_twr, pk_twi, wp, K, T);
 */

#ifndef FFT_RADIX32_DISPATCH_H
#define FFT_RADIX32_DISPATCH_H

#include <stddef.h>

/* ── Include kernel headers ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix32_avx512_tw_ladder_v2.h"
#include "avx512/fft_radix32_avx512_notw.h"
#include "avx512/fft_radix32_avx512_tw_pack_walk.h"
#include "avx512/fft_radix32_avx512_il_tw.h"
#include "avx512/fft_radix32_avx512_il_dif_tw.h"
#include "avx512/fft_radix32_avx512_n1_mono_il.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix32_avx2_tw_v2.h"
#include "avx2/fft_radix32_avx2_notw.h"
#include "avx2/fft_radix32_avx2_tw_pack_walk.h"
#include "avx2/fft_radix32_avx2_il_tw.h"
#include "avx2/fft_radix32_avx2_il_dif_tw.h"
#include "avx2/fft_radix32_avx2_n1_mono_il.h"
#endif
#include "scalar/fft_radix32_scalar_tw.h"
#include "scalar/fft_radix32_scalar_notw.h"

/* Walk threshold: use pack+walk when K > this (tw table exceeds L2) */
#ifndef RADIX32_WALK_THRESHOLD
#define RADIX32_WALK_THRESHOLD 512
#endif

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

static inline vfft_isa_level_t radix32_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix32_isa_name(vfft_isa_level_t isa) {
    switch (isa) {
        case VFFT_ISA_AVX512: return "AVX512";
        case VFFT_ISA_AVX2:   return "AVX2";
        default:               return "scalar";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLE STRATEGY QUERIES
 * ═══════════════════════════════════════════════════════════════ */

static inline int radix32_needs_flat(size_t K) {
    vfft_isa_level_t isa = radix32_effective_isa(K);
    if (isa == VFFT_ISA_AVX512) return (K <= 64);
    return 1;  /* AVX2 and scalar always use flat */
}

static inline int radix32_needs_ladder(size_t K) {
    vfft_isa_level_t isa = radix32_effective_isa(K);
    if (isa == VFFT_ISA_AVX512) return (K > 64);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void radix32_pack_input_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 256;
        for (int n = 0; n < 32; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix32_unpack_output_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 256, dk = b * 8;
        for (int n = 0; n < 32; n++) {
            _mm512_storeu_pd(&dst_re[n*K + dk], _mm512_loadu_pd(&src_re[sk + n*8]));
            _mm512_storeu_pd(&dst_im[n*K + dk], _mm512_loadu_pd(&src_im[sk + n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix32_pack_twiddles_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 248;  /* 31*8 */
        for (int n = 0; n < 31; n++) {
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
 *             flat_tw_re, flat_tw_im, ladder_tw_re, ladder_tw_im)
 *
 * Caller provides both flat and ladder twiddle tables.
 * Dispatch picks flat vs ladder internally.
 * ladder_tw_re/im may be NULL if radix32_needs_ladder(K) == 0.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re, const double * __restrict__ flat_tw_im,
    const double * __restrict__ ladder_tw_re, const double * __restrict__ ladder_tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        if (K <= 64)
            radix32_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                                   flat_tw_re, flat_tw_im, K);
        else if (K < 256)
            radix32_tw_ladder_dit_kernel_fwd_avx512_u1(in_re, in_im, out_re, out_im,
                                                        ladder_tw_re, ladder_tw_im, K);
        else
            radix32_tw_ladder_dit_kernel_fwd_avx512_u2(in_re, in_im, out_re, out_im,
                                                        ladder_tw_re, ladder_tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix32_tw_flat_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im,
                                             flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
    radix32_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                           flat_tw_re, flat_tw_im, K);
}

static inline void radix32_tw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re, const double * __restrict__ flat_tw_im,
    const double * __restrict__ ladder_tw_re, const double * __restrict__ ladder_tw_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        if (K <= 64)
            radix32_tw_flat_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im,
                                                   flat_tw_re, flat_tw_im, K);
        else if (K < 256)
            radix32_tw_ladder_dit_kernel_bwd_avx512_u1(in_re, in_im, out_re, out_im,
                                                        ladder_tw_re, ladder_tw_im, K);
        else
            radix32_tw_ladder_dit_kernel_bwd_avx512_u2(in_re, in_im, out_re, out_im,
                                                        ladder_tw_re, ladder_tw_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix32_tw_flat_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im,
                                             flat_tw_re, flat_tw_im, K);
        return;
    }
#endif
    radix32_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                           flat_tw_re, flat_tw_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_notw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix32_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix32_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix32_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void radix32_notw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix32_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix32_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K); return; }
#endif
    radix32_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH (planner — multi-stage pipeline)
 *
 * Always uses FLAT kernel at K=T — no ladder needed because
 * packed twiddles are per-block (31*T doubles fits L1 for any T).
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 32 * T;
    const size_t tw_bs = 31 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dit_kernel_fwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dit_kernel_fwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_tw_flat_dit_kernel_fwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

static inline void radix32_tw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_bs = 32 * T;
    const size_t tw_bs = 31 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dit_kernel_bwd_avx512(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_tw_flat_dit_kernel_bwd_avx2(
                in_re + b*data_bs, in_im + b*data_bs,
                out_re + b*data_bs, out_im + b*data_bs,
                tw_re + b*tw_bs, tw_im + b*tw_bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_tw_flat_dit_kernel_bwd_scalar(
            in_re + b*data_bs, in_im + b*data_bs,
            out_re + b*data_bs, out_im + b*data_bs,
            tw_re + b*tw_bs, tw_im + b*tw_bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACK+WALK AUTO-DISPATCH (planner — three-tier)
 *
 * Automatically selects:
 *   K <= RADIX32_WALK_THRESHOLD → packed table (pre-packed twiddles)
 *   K > RADIX32_WALK_THRESHOLD  → pack+walk (twiddles on-the-fly)
 *
 * Data must be in packed layout for both paths.
 * For pack+walk, caller provides a walk_plan (void*) initialized by
 * radix32_walk_plan_init (AVX-512) or radix32_walk_plan_avx2_init (AVX2).
 * Pass NULL when using packed table path.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_packed_auto_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX32_WALK_THRESHOLD && walk_plan) {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8) {
            radix32_tw_pack_walk_fwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix32_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4) {
            radix32_tw_pack_walk_fwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix32_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    radix32_tw_packed_fwd(in_re, in_im, out_re, out_im,
                          tw_re, tw_im, K, T);
}

static inline void radix32_tw_packed_auto_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX32_WALK_THRESHOLD && walk_plan) {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8) {
            radix32_tw_pack_walk_bwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix32_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4) {
            radix32_tw_pack_walk_bwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix32_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    radix32_tw_packed_bwd(in_re, in_im, out_re, out_im,
                          tw_re, tw_im, K, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_notw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 32 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_notw_dit_kernel_fwd_avx512(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_notw_dit_kernel_fwd_avx2(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_notw_dit_kernel_fwd_scalar(
            in_re + b*bs, in_im + b*bs,
            out_re + b*bs, out_im + b*bs, T);
}

static inline void radix32_notw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 32 * T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T == 8) {
        for (size_t b = 0; b < nb; b++)
            radix32_notw_dit_kernel_bwd_avx512(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
#ifdef __AVX2__
    if (T == 4) {
        for (size_t b = 0; b < nb; b++)
            radix32_notw_dit_kernel_bwd_avx2(
                in_re + b*bs, in_im + b*bs,
                out_re + b*bs, out_im + b*bs, T);
        return;
    }
#endif
    for (size_t b = 0; b < nb; b++)
        radix32_notw_dit_kernel_bwd_scalar(
            in_re + b*bs, in_im + b*bs,
            out_re + b*bs, out_im + b*bs, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix32_flat_tw_size(size_t K)  { return 31 * K; }
static inline size_t radix32_ladder_tw_size(size_t K) { return 5 * K; }
static inline size_t radix32_data_size(size_t K)     { return 32 * K; }

/** Returns 1 if pack+walk should be used instead of packed table */
static inline int radix32_should_walk(size_t K) {
    return (K > RADIX32_WALK_THRESHOLD) ? 1 : 0;
}

/**
 * Packed twiddle table size for planner allocation.
 * Returns 0 if walk mode will be used (no table needed).
 */
static inline size_t radix32_packed_tw_size(size_t K) {
    return radix32_should_walk(K) ? 0 : 31 * K;
}

static inline size_t radix32_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX32_DISPATCH_H */
