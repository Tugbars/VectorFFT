/**
 * @file fft_radix10_dispatch.h
 * @brief Radix-10 cross-ISA dispatch — N1 + DIT tw (5×2 CT)
 */
#ifndef FFT_RADIX10_DISPATCH_H
#define FFT_RADIX10_DISPATCH_H
#include <stddef.h>

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

#include "scalar/fft_radix10_scalar_n1.h"
#include "scalar/fft_radix10_scalar_tw.h"
#ifdef __AVX2__
#include "avx2/fft_radix10_avx2_n1.h"
#include "avx2/fft_radix10_avx2_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix10_avx512_n1.h"
#include "avx512/fft_radix10_avx512_tw.h"
#endif

static inline vfft_isa_level_t radix10_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline void radix10_n1_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi) {
    switch (radix10_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512: radix10_n1_dit_kernel_fwd_avx512(ir,ii,or_,oi,K); return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2: radix10_n1_dit_kernel_fwd_avx2(ir,ii,or_,oi,K); return;
#endif
    default: radix10_n1_dit_kernel_fwd_scalar(ir,ii,or_,oi,K); return;
    }
}
static inline void radix10_n1_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi) {
    switch (radix10_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512: radix10_n1_dit_kernel_bwd_avx512(ir,ii,or_,oi,K); return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2: radix10_n1_dit_kernel_bwd_avx2(ir,ii,or_,oi,K); return;
#endif
    default: radix10_n1_dit_kernel_bwd_scalar(ir,ii,or_,oi,K); return;
    }
}

static inline void radix10_tw_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
    switch (radix10_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix10_tw_flat_dit_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K); return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix10_tw_flat_dit_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K); return;
#endif
    default:
        radix10_tw_flat_dit_kernel_fwd_scalar(ir,ii,or_,oi,twr,twi,K); return;
    }
}
static inline void radix10_tw_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
    switch (radix10_effective_isa(K)) {
#if defined(__AVX512F__) || defined(__AVX512F)
    case VFFT_ISA_AVX512:
        radix10_tw_flat_dit_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K); return;
#endif
#ifdef __AVX2__
    case VFFT_ISA_AVX2:
        radix10_tw_flat_dit_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K); return;
#endif
    default:
        radix10_tw_flat_dit_kernel_bwd_scalar(ir,ii,or_,oi,twr,twi,K); return;
    }
}

static inline size_t radix10_flat_tw_size(size_t K) { return 9*K; }
static inline size_t radix10_data_size(size_t K) { return 10*K; }

/* ── Interleaved (IL) codelet includes ── */
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix10_avx512_il.h"
#include "avx512/fft_radix10_avx512_il_dif_tw.h"
#endif
#ifdef __AVX2__
#include "avx2/fft_radix10_avx2_il.h"
#include "avx2/fft_radix10_avx2_il_dif_tw.h"
#endif

static inline void radix10_tw_forward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix10_tw_dit_kernel_fwd_il_avx512(in, out, twr, twi, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix10_tw_dit_kernel_fwd_il_avx2(in, out, twr, twi, K); return; }
#endif
    (void)in; (void)out; (void)twr; (void)twi; (void)K; /* no scalar IL fallback */
}

static inline void radix10_tw_dif_backward_il(
    const double *__restrict__ in, double *__restrict__ out,
    const double *__restrict__ twr, const double *__restrict__ twi, size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) { radix10_tw_dif_kernel_bwd_il_avx512(in, out, twr, twi, K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) { radix10_tw_dif_kernel_bwd_il_avx2(in, out, twr, twi, K); return; }
#endif
    (void)in; (void)out; (void)twr; (void)twi; (void)K;
}

#endif /* FFT_RADIX10_DISPATCH_H */