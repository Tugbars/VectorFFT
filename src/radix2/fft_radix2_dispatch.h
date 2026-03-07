/**
 * @file fft_radix2_dispatch.h
 * @brief Radix-2 cross-ISA dispatch — strided and packed paths
 *
 * DFT-2: the simplest butterfly. Zero constants.
 * out[0] = x0 + tw*x1,  out[1] = x0 - tw*x1
 * Forward notw = backward notw (self-inverse).
 *
 * Twiddle table: 1*K doubles per component (only W^1).
 * radix2_flat_tw_size(K) = K
 * radix2_data_size(K)    = 2*K
 */

#ifndef FFT_RADIX2_DISPATCH_H
#define FFT_RADIX2_DISPATCH_H

#include <stddef.h>

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix2_avx512.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix2_avx2.h"
#endif
#include "scalar/fft_radix2_scalar.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum { VFFT_ISA_SCALAR=0, VFFT_ISA_AVX2=1, VFFT_ISA_AVX512=2 } vfft_isa_level_t;
#endif

static inline vfft_isa_level_t radix2_effective_isa(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return VFFT_ISA_AVX512;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return VFFT_ISA_AVX2;
#endif
    return VFFT_ISA_SCALAR;
}

static inline const char *radix2_isa_name(vfft_isa_level_t isa) {
    switch (isa) {
        case VFFT_ISA_AVX512: return "AVX512";
        case VFFT_ISA_AVX2:   return "AVX2";
        default:               return "scalar";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * SIMD PACK/UNPACK (AVX-512)
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>

__attribute__((target("avx512f")))
static inline void radix2_pack_input_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++) {
        const size_t sk = b*8, dk = b*16;
        for (size_t n = 0; n < 2; n++) {
            _mm512_storeu_pd(&dr[dk+n*8], _mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8], _mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix2_unpack_output_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    for (size_t b = 0; b < K/8; b++) {
        const size_t sk = b*16, dk = b*8;
        for (size_t n = 0; n < 2; n++) {
            _mm512_storeu_pd(&dr[n*K+dk], _mm512_loadu_pd(&sr[sk+n*8]));
            _mm512_storeu_pd(&di[n*K+dk], _mm512_loadu_pd(&si[sk+n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix2_pack_twiddles_avx512(
    const double * __restrict__ sr, const double * __restrict__ si,
    double * __restrict__ dr, double * __restrict__ di, size_t K) {
    /* Only 1 twiddle index — same as a straight copy in blocks of 8 */
    for (size_t b = 0; b < K/8; b++) {
        _mm512_storeu_pd(&dr[b*8], _mm512_loadu_pd(&sr[b*8]));
        _mm512_storeu_pd(&di[b*8], _mm512_loadu_pd(&si[b*8]));
    }
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — twiddled
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix2_tw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K&7)==0) { radix2_tw_dit_kernel_fwd_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K&3)==0) { radix2_tw_dit_kernel_fwd_avx2(in_re,in_im,out_re,out_im,tw_re,tw_im,K); return; }
#endif
    radix2_tw_dit_kernel_fwd_scalar(in_re,in_im,out_re,out_im,tw_re,tw_im,K);
}

static inline void radix2_tw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K&7)==0) { radix2_tw_dit_kernel_bwd_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K&3)==0) { radix2_tw_dit_kernel_bwd_avx2(in_re,in_im,out_re,out_im,tw_re,tw_im,K); return; }
#endif
    radix2_tw_dit_kernel_bwd_scalar(in_re,in_im,out_re,out_im,tw_re,tw_im,K);
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix2_notw_forward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K&7)==0) { radix2_notw_dit_kernel_fwd_avx512(in_re,in_im,out_re,out_im,K); return; }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K&3)==0) { radix2_notw_dit_kernel_fwd_avx2(in_re,in_im,out_re,out_im,K); return; }
#endif
    radix2_notw_dit_kernel_fwd_scalar(in_re,in_im,out_re,out_im,K);
}

static inline void radix2_notw_backward(
    size_t K,
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im) {
    /* DFT-2 notw is self-inverse — forward = backward */
    radix2_notw_forward(K, in_re, in_im, out_re, out_im);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — twiddled
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix2_tw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T) {
    const size_t nb = K/T, dbs = 2*T, tbs = T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T==8) { for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_fwd_avx512(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T); return; }
#endif
#ifdef __AVX2__
    if (T==4) { for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_fwd_avx2(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T); return; }
#endif
    for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_fwd_scalar(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T);
}

static inline void radix2_tw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K, size_t T) {
    const size_t nb = K/T, dbs = 2*T, tbs = T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T==8) { for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_bwd_avx512(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T); return; }
#endif
#ifdef __AVX2__
    if (T==4) { for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_bwd_avx2(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T); return; }
#endif
    for(size_t b=0;b<nb;b++) radix2_tw_dit_kernel_bwd_scalar(in_re+b*dbs,in_im+b*dbs,out_re+b*dbs,out_im+b*dbs,tw_re+b*tbs,tw_im+b*tbs,T);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — notw
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix2_notw_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T) {
    const size_t nb = K/T, bs = 2*T;
#if defined(__AVX512F__) || defined(__AVX512F)
    if (T==8) { for(size_t b=0;b<nb;b++) radix2_notw_dit_kernel_fwd_avx512(in_re+b*bs,in_im+b*bs,out_re+b*bs,out_im+b*bs,T); return; }
#endif
#ifdef __AVX2__
    if (T==4) { for(size_t b=0;b<nb;b++) radix2_notw_dit_kernel_fwd_avx2(in_re+b*bs,in_im+b*bs,out_re+b*bs,out_im+b*bs,T); return; }
#endif
    for(size_t b=0;b<nb;b++) radix2_notw_dit_kernel_fwd_scalar(in_re+b*bs,in_im+b*bs,out_re+b*bs,out_im+b*bs,T);
}

static inline void radix2_notw_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K, size_t T) {
    radix2_notw_packed_fwd(in_re, in_im, out_re, out_im, K, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix2_flat_tw_size(size_t K) { return K; }
static inline size_t radix2_data_size(size_t K)    { return 2 * K; }

static inline size_t radix2_packed_optimal_T(size_t K) {
#if defined(__AVX512F__) || defined(__AVX512F)
    if (K >= 8 && (K & 7) == 0) return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0) return 4;
#endif
    return 1;
}

#endif /* FFT_RADIX2_DISPATCH_H */
