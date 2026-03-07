/**
 * @file fft_radix4_dispatch.h
 * @brief Radix-4 cross-ISA dispatch — strided and packed paths
 *
 * DFT-4 = 2×DFT-2 + ×(-j) combine. 16 adds, 0 muls, 0 constants.
 * Forward/backward differ only in the ×j sign.
 *
 * Twiddle table: 3*K doubles per component (W^1, W^2, W^3).
 * radix4_flat_tw_size(K) = 3*K
 * radix4_data_size(K)    = 4*K
 */

#ifndef FFT_RADIX4_DISPATCH_H
#define FFT_RADIX4_DISPATCH_H

#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix4_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix4_avx2.h"
#endif
#include "scalar/fft_radix4_scalar.h"

/* ISA detection (guarded) */
#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix4_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix4_isa_name(vfft_isa_level_t isa) {
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
static inline void radix4_pack_input_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 4; n++) {
            _mm512_storeu_pd(&dr[b*32+n*8], _mm512_loadu_pd(&sr[n*K+b*8]));
            _mm512_storeu_pd(&di[b*32+n*8], _mm512_loadu_pd(&si[n*K+b*8]));
        }
}

__attribute__((target("avx512f")))
static inline void radix4_unpack_output_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 4; n++) {
            _mm512_storeu_pd(&dr[n*K+b*8], _mm512_loadu_pd(&sr[b*32+n*8]));
            _mm512_storeu_pd(&di[n*K+b*8], _mm512_loadu_pd(&si[b*32+n*8]));
        }
}

__attribute__((target("avx512f")))
static inline void radix4_pack_twiddles_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++)
        for (size_t n = 0; n < 3; n++) {
            _mm512_storeu_pd(&dr[b*24+n*8], _mm512_loadu_pd(&sr[n*K+b*8]));
            _mm512_storeu_pd(&di[b*24+n*8], _mm512_loadu_pd(&si[n*K+b*8]));
        }
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix4_tw_forward(
    size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0) { radix4_tw_dit_kernel_fwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0) { radix4_tw_dit_kernel_fwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    radix4_tw_dit_kernel_fwd_scalar(ir,ii,or_,oi,twr,twi,K);
}

static inline void radix4_tw_backward(
    size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi,
    const double * __restrict__ twr, const double * __restrict__ twi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0) { radix4_tw_dit_kernel_bwd_avx512(ir,ii,or_,oi,twr,twi,K); return; }
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0) { radix4_tw_dit_kernel_bwd_avx2(ir,ii,or_,oi,twr,twi,K); return; }
#endif
    radix4_tw_dit_kernel_bwd_scalar(ir,ii,or_,oi,twr,twi,K);
}

static inline void radix4_notw_forward(
    size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0) { radix4_notw_dit_kernel_fwd_avx512(ir,ii,or_,oi,K); return; }
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0) { radix4_notw_dit_kernel_fwd_avx2(ir,ii,or_,oi,K); return; }
#endif
    radix4_notw_dit_kernel_fwd_scalar(ir,ii,or_,oi,K);
}

static inline void radix4_notw_backward(
    size_t K,
    const double * __restrict__ ir, const double * __restrict__ ii,
    double * __restrict__ or_, double * __restrict__ oi) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K>=8&&(K&7)==0) { radix4_notw_dit_kernel_bwd_avx512(ir,ii,or_,oi,K); return; }
#endif
#ifdef __AVX2__
    if (K>=4&&(K&3)==0) { radix4_notw_dit_kernel_bwd_avx2(ir,ii,or_,oi,K); return; }
#endif
    radix4_notw_dit_kernel_bwd_scalar(ir,ii,or_,oi,K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix4_tw_packed_fwd(
    const double *ir, const double *ii, double *or_, double *oi,
    const double *twr, const double *twi, size_t K, size_t T) {
    const size_t nb=K/T, dbs=4*T, tbs=3*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_fwd_avx512(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_fwd_avx2(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_fwd_scalar(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);
}

static inline void radix4_tw_packed_bwd(
    const double *ir, const double *ii, double *or_, double *oi,
    const double *twr, const double *twi, size_t K, size_t T) {
    const size_t nb=K/T, dbs=4*T, tbs=3*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_bwd_avx512(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_bwd_avx2(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix4_tw_dit_kernel_bwd_scalar(ir+b*dbs,ii+b*dbs,or_+b*dbs,oi+b*dbs,twr+b*tbs,twi+b*tbs,T);
}

static inline void radix4_notw_packed_fwd(
    const double *ir, const double *ii, double *or_, double *oi,
    size_t K, size_t T) {
    const size_t nb=K/T, bs=4*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_fwd_avx512(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_fwd_avx2(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_fwd_scalar(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);
}

static inline void radix4_notw_packed_bwd(
    const double *ir, const double *ii, double *or_, double *oi,
    size_t K, size_t T) {
    const size_t nb=K/T, bs=4*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if(T==8){for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_bwd_avx512(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
#ifdef __AVX2__
    if(T==4){for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_bwd_avx2(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);return;}
#endif
    for(size_t b=0;b<nb;b++)radix4_notw_dit_kernel_bwd_scalar(ir+b*bs,ii+b*bs,or_+b*bs,oi+b*bs,T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix4_flat_tw_size(size_t K) { return 3 * K; }
static inline size_t radix4_data_size(size_t K)    { return 4 * K; }

static inline size_t radix4_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX4_DISPATCH_H */
