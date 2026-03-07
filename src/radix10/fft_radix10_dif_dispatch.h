/**
 * @file fft_radix10_dif_dispatch.h
 * @brief Radix-10 DIF dispatch — twiddle AFTER butterfly
 */
#ifndef FFT_RADIX10_DIF_DISPATCH_H
#define FFT_RADIX10_DIF_DISPATCH_H

/* Scalar DIF is in fft_radix10_scalar_tw.h (same file as DIT) */
#ifdef __AVX2__
#include "avx2/fft_radix10_avx2_tw.h"
#endif
#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix10_avx512_tw.h"
#endif

static inline void radix10_tw_dif_forward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0){radix10_tw_flat_dif_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K);return;}
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0){radix10_tw_flat_dif_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K);return;}
#endif
    radix10_tw_flat_dif_kernel_fwd_scalar(ir,ii,or_,oi,twr,twi,K);
}
static inline void radix10_tw_dif_backward(size_t K,
    const double *__restrict__ ir, const double *__restrict__ ii,
    double *__restrict__ or_, double *__restrict__ oi,
    const double *__restrict__ twr, const double *__restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0){radix10_tw_flat_dif_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K);return;}
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0){radix10_tw_flat_dif_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K);return;}
#endif
    radix10_tw_flat_dif_kernel_bwd_scalar(ir,ii,or_,oi,twr,twi,K);
}

#endif /* FFT_RADIX10_DIF_DISPATCH_H */
