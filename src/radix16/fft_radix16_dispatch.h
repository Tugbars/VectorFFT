/**
 * @file fft_radix16_dispatch.h
 * @brief Radix-16 cross-ISA dispatch — strided, packed, and pack+walk paths
 *
 * Three codelet families:
 *   N1 (notw):       radix16_n1_forward / radix16_n1_backward
 *   Tw strided:      radix16_tw_strided_forward / _backward
 *   Tw packed:       radix16_tw_packed_forward / _backward
 *   Pack+walk auto:  radix16_tw_packed_auto_fwd / _bwd
 */

#ifndef FFT_RADIX16_DISPATCH_H
#define FFT_RADIX16_DISPATCH_H

#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * BACKEND INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar: notw + DIT tw + DIF tw (always available) */
#include "scalar/fft_radix16_scalar_n1_gen.h"
#include "scalar/fft_radix16_scalar_tw.h"

/* AVX2 */
#ifdef __AVX2__
#include "avx2/fft_radix16_avx2_n1_gen.h"
#include "avx2/fft_radix16_avx2_tw.h"
#include "avx2/fft_radix16_avx2_tw_pack_walk.h"
#endif

/* AVX-512 */
#if defined(__AVX512F__) || defined(__AVX512F)
#ifndef TARGET_AVX512
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#endif
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_64
#define ALIGNAS_64 __attribute__((aligned(64)))
#endif
#include "avx512/fft_radix16_avx512_n1_gen.h"
#include "avx512/fft_radix16_avx512_tw.h"
#include "avx512/fft_radix16_avx512_tw_pack_walk.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL (shared enum — guarded, matches other dispatches)
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix16_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix16_isa_name(vfft_isa_level_t isa) {
    switch (isa) {
        case VFFT_ISA_AVX512: return "AVX-512";
        case VFFT_ISA_AVX2:   return "AVX2";
        default:               return "Scalar";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * N1 DISPATCH (twiddle-less DFT-16)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_n1_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_n1_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_n1_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix16_n1_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

static inline void radix16_n1_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_n1_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_n1_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix16_n1_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

#define radix16_notw_forward  radix16_n1_forward
#define radix16_notw_backward radix16_n1_backward

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — STRIDED LAYOUT
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_strided_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_tw_flat_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im,
                                            tw_re, tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

static inline void radix16_tw_strided_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_tw_flat_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_tw_flat_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im,
                                            tw_re, tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

#define radix16_tw_forward  radix16_tw_strided_forward
#define radix16_tw_backward radix16_tw_strided_backward

/* ═══════════════════════════════════════════════════════════════
 * PACKED TABLE DRIVERS
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_packed_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        r16_tw_packed_fwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        r16_tw_packed_fwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

static inline void radix16_tw_packed_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        r16_tw_packed_bwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        r16_tw_packed_bwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACK+WALK AUTO-DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

#ifndef RADIX16_WALK_THRESHOLD
#define RADIX16_WALK_THRESHOLD 512
#endif

static inline void radix16_tw_packed_auto_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && (K & 7) == 0) {
        radix16_tw_pack_walk_fwd_avx512(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_t *)walk_plan, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && (K & 3) == 0) {
        radix16_tw_pack_walk_fwd_avx2(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_avx2_t *)walk_plan, K);
        return;
    }
#endif
    (void)walk_plan;
    radix16_tw_packed_forward(K, in_re, in_im, out_re, out_im, tw_re, tw_im);
}

static inline void radix16_tw_packed_auto_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && (K & 7) == 0) {
        radix16_tw_pack_walk_bwd_avx512(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_t *)walk_plan, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && (K & 3) == 0) {
        radix16_tw_pack_walk_bwd_avx2(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_avx2_t *)walk_plan, K);
        return;
    }
#endif
    (void)walk_plan;
    radix16_tw_packed_backward(K, in_re, in_im, out_re, out_im, tw_re, tw_im);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix16_flat_tw_size(size_t K) { return 15 * K; }
static inline size_t radix16_data_size(size_t K)    { return 16 * K; }

static inline size_t radix16_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

static inline int radix16_should_walk(size_t K) {
    return (K > RADIX16_WALK_THRESHOLD) ? 1 : 0;
}

static inline size_t radix16_packed_tw_size(size_t K) {
    return radix16_should_walk(K) ? 0 : 15 * K;
}

#endif /* FFT_RADIX16_DISPATCH_H */