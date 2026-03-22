/**
 * @file fft_radix16_dispatch.h
 * @brief Radix-16 cross-ISA dispatch — N1, DIT tw, DIF tw, IL
 *
 * 4×4 Cooley-Tukey decomposition.
 * N1: DAG (split-radix, tan trick, 72 ops) at all K on AVX2.
 *     4×4 CT register-only on AVX-512.
 * DIT tw: generated CT codelet (AVX2), 4×4 CT (AVX-512).
 * DIF tw: DAG + post-twiddle (AVX2), 4×4 CT (AVX-512).
 * IL native: DAG (AVX2 k-step=2), 4×4 CT (AVX-512 k-step=4).
 *
 * AVX2: DAG butterfly, compiler-managed spills (peak ~16 YMM).
 * AVX-512: 4×4 CT register-only (peak 32 ZMM, zero spill).
 */
#ifndef FFT_RADIX16_DISPATCH_H
#define FFT_RADIX16_DISPATCH_H
#include <stddef.h>

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

/* ── N1 (notw) codelets ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix16_avx512_n1.h"      /* fwd split: radix16_ct_n1_fwd_avx512_regonly */
#include "avx512/fft_radix16_avx512_bwd.h"      /* bwd split: radix16_ct_n1_bwd_avx512 */
#endif
#ifdef __AVX2__
#include "avx2/fft_radix16_avx2_n1.h"           /* fwd/bwd split CT: radix16_ct_n1_{fwd,bwd}_avx2 */
#include "avx2/fft_radix16_avx2_n1_dag.h"       /* fwd split DAG: radix16_dag_n1_fwd_avx2 */
#endif
#include "scalar/fft_radix16_scalar.h"           /* fwd/bwd scalar: radix16_ct_n1_{fwd,bwd}_scalar */

/* ── N1 IL codelets ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix16_avx512_n1_il.h"    /* fwd IL: radix16_ct_n1_fwd_il_avx512 */
/* bwd IL already in avx512/bwd.h: radix16_ct_n1_bwd_il_avx512 */
#endif
#ifdef __AVX2__
#include "avx2/fft_radix16_avx2_n1_il.h"        /* fwd IL DAG: radix16_dag_n1_fwd_il_avx2 */
#include "avx2/fft_radix16_avx2_bwd_il.h"       /* bwd IL DAG: radix16_dag_n1_bwd_il_avx2 */
#endif

/* ── Split tw codelets (DIT + DIF) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
/* DIT split fwd/bwd already in avx512/bwd.h */
#include "avx512/fft_radix16_avx512_dif.h"      /* DIF split fwd/bwd */
#endif
#ifdef __AVX2__
#include "avx2/fft_radix16_avx2_tw.h"           /* DIT split fwd/bwd (generated CT) */
#include "avx2/fft_radix16_avx2_tw_new.h"       /* DIF split + all IL native */
#endif
/* Scalar DIT/DIF already in scalar/fft_radix16_scalar.h */

/* ── IL tw codelets (DIT + DIF, native pre-interleaved) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix16_avx512_il_tw.h"    /* DIT fwd IL native */
/* DIT bwd IL native already in avx512/bwd.h */
/* DIF IL native already in avx512/dif.h */
#endif
/* AVX2 IL native already included via avx2/tw_new.h */

/* ── ISA selection ── */
static inline vfft_isa_level_t radix16_effective_isa(size_t K)
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

/* ═══════════════════════════════════════════════════════════════
 * N1 dispatch
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_n1_forward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_ct_n1_fwd_avx512_regonly(ir, ii, or_, oi, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_dag_n1_fwd_avx2(ir, ii, or_, oi, K);
        return;
#endif
    default:
        radix16_ct_n1_fwd_scalar(ir, ii, or_, oi, K);
        return;
    }
}

static inline void radix16_n1_backward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi)
{
    switch (radix16_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix16_ct_n1_bwd_avx512(ir, ii, or_, oi, K);
        return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix16_ct_n1_bwd_avx2(ir, ii, or_, oi, K);
        return;
#endif
    default:
        radix16_ct_n1_bwd_scalar(ir, ii, or_, oi, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * DIT tw dispatch
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_forward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix16_ct_tw_dit_fwd_split_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix16_tw_flat_dit_kernel_fwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix16_ct_tw_dit_fwd_scalar(ir, ii, or_, oi, twr, twi, K);
}

static inline void radix16_tw_backward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix16_ct_tw_dit_bwd_split_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix16_tw_flat_dit_kernel_bwd_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix16_ct_tw_dit_bwd_scalar(ir, ii, or_, oi, twr, twi, K);
}

/* ═══════════════════════════════════════════════════════════════
 * DIF tw dispatch
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar DIF fallback: N1 + post-multiply */
static inline void radix16_tw_dif_scalar_fwd(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
    radix16_ct_tw_dif_fwd_scalar(ir, ii, or_, oi, twr, twi, K);
}

static inline void radix16_tw_dif_scalar_bwd(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
    radix16_ct_tw_dif_bwd_scalar(ir, ii, or_, oi, twr, twi, K);
}

static inline void radix16_tw_dif_forward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix16_ct_tw_dif_fwd_split_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix16_tw_dif_fwd_split_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix16_tw_dif_scalar_fwd(K, ir, ii, or_, oi, twr, twi);
}

static inline void radix16_tw_dif_backward(size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix16_ct_tw_dif_bwd_split_avx512(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix16_tw_dif_bwd_split_avx2(ir, ii, or_, oi, twr, twi, K);
        return;
    }
#endif
    radix16_tw_dif_scalar_bwd(K, ir, ii, or_, oi, twr, twi);
}

/* ═══════════════════════════════════════════════════════════════
 * IL tw dispatch (native pre-interleaved twiddles)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_forward_il_native(
    const double * __restrict__ in, double * __restrict__ out,
    const double * __restrict__ tw_il, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) {
        radix16_ct_tw_dit_fwd_il_avx512(in, out, tw_il, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) {
        radix16_tw_dit_fwd_il_native_avx2(in, out, tw_il, K);
        return;
    }
#endif
    (void)in; (void)out; (void)tw_il; (void)K;
}

static inline void radix16_tw_dif_backward_il_native(
    const double * __restrict__ in, double * __restrict__ out,
    const double * __restrict__ tw_il, size_t K)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) {
        radix16_ct_tw_dif_bwd_il_avx512(in, out, tw_il, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) {
        radix16_tw_dif_bwd_il_native_avx2(in, out, tw_il, K);
        return;
    }
#endif
    (void)in; (void)out; (void)tw_il; (void)K;
}

/* ═══════════════════════════════════════════════════════════════
 * N1 IL dispatch
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_n1_forward_il(size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) {
        radix16_ct_n1_fwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) {
        radix16_dag_n1_fwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in; (void)out; (void)K;
}

static inline void radix16_n1_backward_il(size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) {
        radix16_ct_n1_bwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) {
        radix16_dag_n1_bwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in; (void)out; (void)K;
}

/* ── Sizes ── */
static inline size_t radix16_flat_tw_size(size_t K) { return 15 * K; }
static inline size_t radix16_data_size(size_t K) { return 16 * K; }

#endif /* FFT_RADIX16_DISPATCH_H */
