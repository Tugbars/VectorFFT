/**
 * @file fft_radix5_dispatch.h
 * @brief Radix-5 cross-ISA dispatch — strided and packed paths
 *
 * DFT-5: 2 conjugate pairs, 4 constants, ~32 ops.
 * Twiddle table: 4*K doubles per component (W^1..W^4).
 * radix5_flat_tw_size(K) = 4*K
 * radix5_data_size(K)    = 5*K
 */

#ifndef FFT_RADIX5_DISPATCH_H
#define FFT_RADIX5_DISPATCH_H

#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix5_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix5_avx2.h"
#endif
#include "scalar/fft_radix5_scalar.h"

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix5_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix5_isa_name(vfft_isa_level_t isa) {
    switch (isa) {
        case VFFT_ISA_AVX512: return "AVX512";
        case VFFT_ISA_AVX2:   return "AVX2";
        default:               return "scalar";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void radix5_pack_input_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 5; n++) {
            _mm512_storeu_pd(&dr[b*40+n*8], _mm512_loadu_pd(&sr[n*K+b*8]));
            _mm512_storeu_pd(&di[b*40+n*8], _mm512_loadu_pd(&si[n*K+b*8]));
        }
}

__attribute__((target("avx512f")))
static inline void radix5_unpack_output_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 5; n++) {
            _mm512_storeu_pd(&dr[n*K+b*8], _mm512_loadu_pd(&sr[b*40+n*8]));
            _mm512_storeu_pd(&di[n*K+b*8], _mm512_loadu_pd(&si[b*40+n*8]));
        }
}

__attribute__((target("avx512f")))
static inline void radix5_pack_twiddles_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 4; n++) {
            _mm512_storeu_pd(&dr[b*32+n*8], _mm512_loadu_pd(&sr[n*K+b*8]));
            _mm512_storeu_pd(&di[b*32+n*8], _mm512_loadu_pd(&si[n*K+b*8]));
        }
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix5_tw_forward(size_t K,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *twr, const double *twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if(K>=8&&(K&7)==0){radix5_tw_dit_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K);return;}
#endif
#ifdef __AVX2__
    if(K>=4&&(K&3)==0){radix5_tw_dit_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K);return;}
#endif
    radix5_tw_dit_kernel_fwd_scalar(ir,ii,or_,oi,twr,twi,K);
}

static inline void radix5_tw_backward(size_t K,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *twr, const double *twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if(K>=8&&(K&7)==0){radix5_tw_dit_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K);return;}
#endif
#ifdef __AVX2__
    if(K>=4&&(K&3)==0){radix5_tw_dit_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K);return;}
#endif
    radix5_tw_dit_kernel_bwd_scalar(ir,ii,or_,oi,twr,twi,K);
}

static inline void radix5_notw_forward(size_t K,
    const double *ir, const double *ii, double *or_, double *oi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if(K>=8&&(K&7)==0){radix5_notw_dit_kernel_fwd_avx512(ir,ii,or_,oi,K);return;}
#endif
#ifdef __AVX2__
    if(K>=4&&(K&3)==0){radix5_notw_dit_kernel_fwd_avx2(ir,ii,or_,oi,K);return;}
#endif
    radix5_notw_dit_kernel_fwd_scalar(ir,ii,or_,oi,K);
}

static inline void radix5_notw_backward(size_t K,
    const double *ir, const double *ii, double *or_, double *oi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if(K>=8&&(K&7)==0){radix5_notw_dit_kernel_bwd_avx512(ir,ii,or_,oi,K);return;}
#endif
#ifdef __AVX2__
    if(K>=4&&(K&3)==0){radix5_notw_dit_kernel_bwd_avx2(ir,ii,or_,oi,K);return;}
#endif
    radix5_notw_dit_kernel_bwd_scalar(ir,ii,or_,oi,K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix5_tw_packed_fwd(
    const double *ir,const double *ii,double *or_,double *oi,
    const double *twr,const double *twi,size_t K,size_t T){
    const size_t nb=K/T,dbs=5*T,tbs=4*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_fwd_avx512(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_fwd_avx2(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_fwd_scalar(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);
}

static inline void radix5_tw_packed_bwd(
    const double *ir,const double *ii,double *or_,double *oi,
    const double *twr,const double *twi,size_t K,size_t T){
    const size_t nb=K/T,dbs=5*T,tbs=4*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_bwd_avx512(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_bwd_avx2(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix5_tw_dit_kernel_bwd_scalar(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);
}

static inline void radix5_notw_packed_fwd(
    const double *ir,const double *ii,double *or_,double *oi,size_t K,size_t T){
    const size_t nb=K/T,bs=5*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_fwd_avx512(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_fwd_avx2(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_fwd_scalar(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);
}

static inline void radix5_notw_packed_bwd(
    const double *ir,const double *ii,double *or_,double *oi,size_t K,size_t T){
    const size_t nb=K/T,bs=5*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_bwd_avx512(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_bwd_avx2(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix5_notw_dit_kernel_bwd_scalar(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix5_flat_tw_size(size_t K) { return 4 * K; }
static inline size_t radix5_data_size(size_t K)    { return 5 * K; }

static inline size_t radix5_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX5_DISPATCH_H */