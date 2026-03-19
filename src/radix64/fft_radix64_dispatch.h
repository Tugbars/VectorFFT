/**
 * @file fft_radix64_dispatch.h
 * @brief Radix-64 cross-ISA dispatch — N1, DIT tw, DIF tw, IL
 * 8×8 CT + log3 twiddles. No scalar tw fallback (SIMD-only).
 */
#ifndef FFT_RADIX64_DISPATCH_H
#define FFT_RADIX64_DISPATCH_H
#include <stddef.h>

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

/* ── N1 (notw) codelets ── */
#ifdef __AVX2__
#include "avx2/fft_radix64_avx2_n1_gen.h"
#endif

/* ── Split tw codelets (DIT + DIF in same file) ── */
#ifdef __AVX2__
#include "avx2/fft_radix64_avx2_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix64_avx512_tw.h"
#endif

/* ── IL tw codelets (DIT + DIF in same file) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix64_avx512_il_tw.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix64_avx2_il_tw.h"
#endif

/* ── ISA selection ── */
static inline vfft_isa_level_t radix64_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

/* ── N1 dispatch ── */
static inline void radix64_n1_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi) {
#ifdef __AVX2__
    radix64_n1_dit_kernel_fwd_avx2(ir, ii, or_, oi, K);
#else
    (void)ir;(void)ii;(void)or_;(void)oi;(void)K;
#endif
}
static inline void radix64_n1_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi) {
#ifdef __AVX2__
    radix64_n1_dit_kernel_bwd_avx2(ir, ii, or_, oi, K);
#else
    (void)ir;(void)ii;(void)or_;(void)oi;(void)K;
#endif
}

/* ── DIT tw dispatch ── */
static inline void radix64_tw_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix64_tw_flat_dit_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix64_tw_flat_dit_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    (void)ir;(void)ii;(void)or_;(void)oi;(void)twr;(void)twi;(void)K;
}
static inline void radix64_tw_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix64_tw_flat_dit_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix64_tw_flat_dit_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    (void)ir;(void)ii;(void)or_;(void)oi;(void)twr;(void)twi;(void)K;
}

/* ── DIF tw dispatch ── */
static inline void radix64_tw_dif_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix64_tw_flat_dif_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix64_tw_flat_dif_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    (void)ir;(void)ii;(void)or_;(void)oi;(void)twr;(void)twi;(void)K;
}
static inline void radix64_tw_dif_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) {
        radix64_tw_flat_dif_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) {
        radix64_tw_flat_dif_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    (void)ir;(void)ii;(void)or_;(void)oi;(void)twr;(void)twi;(void)K;
}

/* ── IL tw dispatch ── */
static inline void radix64_tw_forward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix64_tw_flat_dit_kernel_fwd_il_avx512(in,out,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix64_tw_flat_dit_kernel_fwd_il_avx2(in,out,twr,twi,K); return; }
#endif
    (void)in;(void)out;(void)twr;(void)twi;(void)K;
}
static inline void radix64_tw_dif_backward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix64_tw_flat_dif_kernel_bwd_il_avx512(in,out,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix64_tw_flat_dif_kernel_bwd_il_avx2(in,out,twr,twi,K); return; }
#endif
    (void)in;(void)out;(void)twr;(void)twi;(void)K;
}

/* ── Monolithic N1 native IL (translated from FFTW genfft DAG) ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix64_avx512_n1_mono_il.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix64_avx2_n1_mono_il.h"
#endif

static inline void radix64_n1_forward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) { radix64_n1_dit_kernel_fwd_il_avx512(in, out, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) { radix64_n1_dit_kernel_fwd_il_avx2(in, out, K); return; }
#endif
    (void)in; (void)out; (void)K;
}

static inline void radix64_n1_backward_il(
    size_t K,
    const double * __restrict__ in, double * __restrict__ out)
{
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 4 && (K & 3) == 0) { radix64_n1_dit_kernel_bwd_il_avx512(in, out, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0) { radix64_n1_dit_kernel_bwd_il_avx2(in, out, K); return; }
#endif
    (void)in; (void)out; (void)K;
}

static inline size_t radix64_flat_tw_size(size_t K) { return 63*K; }
static inline size_t radix64_data_size(size_t K) { return 64*K; }

#endif /* FFT_RADIX64_DISPATCH_H */
