/**
 * @file fft_radix19_dispatch.h
 * @brief Runtime ISA dispatch for DFT-19 — auto-selects best kernel
 */
#ifndef FFT_RADIX19_DISPATCH_H
#define FFT_RADIX19_DISPATCH_H

#include "fft_radix19_genfft.h"

#if !defined(FFT_RADIX11_DISPATCH_H) && !defined(FFT_RADIX13_DISPATCH_H)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static inline int r19_has_avx512f(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>16)&1 : 0; }
static inline int r19_has_avx2(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>5)&1 : 0; }
static inline int r19_has_fma(void) {
    unsigned int a,b,c,d;
    return __get_cpuid(1,&a,&b,&c,&d) ? (c>>12)&1 : 0; }
#elif defined(_MSC_VER)
#include <intrin.h>
static inline int r19_has_avx512f(void) { int i[4]; __cpuidex(i,7,0); return (i[1]>>16)&1; }
static inline int r19_has_avx2(void)    { int i[4]; __cpuidex(i,7,0); return (i[1]>>5)&1; }
static inline int r19_has_fma(void)     { int i[4]; __cpuid(i,1); return (i[2]>>12)&1; }
#else
static inline int r19_has_avx512f(void) { return 0; }
static inline int r19_has_avx2(void)    { return 0; }
static inline int r19_has_fma(void)     { return 0; }
#endif

typedef enum { R19_ISA_SCALAR=0, R19_ISA_AVX2=1, R19_ISA_AVX512=2 } r19_isa_t;

static inline r19_isa_t r19_detect_isa(void) {
    static r19_isa_t c = (r19_isa_t)-1;
    if (c != (r19_isa_t)-1) return c;
    if (r19_has_avx512f() && r19_has_fma()) c = R19_ISA_AVX512;
    else if (r19_has_avx2() && r19_has_fma()) c = R19_ISA_AVX2;
    else c = R19_ISA_SCALAR;
    return c;
}

static inline size_t r19_simd_width(void) {
    switch(r19_detect_isa()){
        case R19_ISA_AVX512: return 8;
        case R19_ISA_AVX2:   return 4;
        default:             return 1; }
}
#else
/* Reuse detection from whichever radix dispatcher is already included */
#ifdef FFT_RADIX11_DISPATCH_H
#define r19_detect_isa() ((r19_isa_t)r11_detect_isa())
#define r19_simd_width() r11_simd_width()
typedef r11_isa_t r19_isa_t;
#define R19_ISA_SCALAR R11_ISA_SCALAR
#define R19_ISA_AVX2   R11_ISA_AVX2
#define R19_ISA_AVX512 R11_ISA_AVX512
#else
#define r19_detect_isa() ((r19_isa_t)r13_detect_isa())
#define r19_simd_width() r13_simd_width()
typedef r13_isa_t r19_isa_t;
#define R19_ISA_SCALAR R13_ISA_SCALAR
#define R19_ISA_AVX2   R13_ISA_AVX2
#define R19_ISA_AVX512 R13_ISA_AVX512
#endif
#endif

/* ═══ Strided dispatch ═══ */

static inline void r19_dispatch_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K) {
    r19_isa_t isa = r19_detect_isa();
#ifdef __AVX512F__
    if (isa==R19_ISA_AVX512 && K>=8 && (K&7)==0)
        { radix19_genfft_fwd_avx512(ri,ii,ro,io,K); return; }
#endif
#ifdef __AVX2__
    if (isa>=R19_ISA_AVX2 && K>=4 && (K&3)==0)
        { radix19_genfft_fwd_avx2(ri,ii,ro,io,K); return; }
#endif
    radix19_genfft_fwd_scalar(ri,ii,ro,io,K);
}

static inline void r19_dispatch_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{ r19_dispatch_fwd(ii,ri,io,ro,K); }

/* ═══ Packed dispatch ═══ */

static inline void r19_dispatch_packed_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K) {
    r19_isa_t isa = r19_detect_isa();
#ifdef __AVX512F__
    if (isa==R19_ISA_AVX512 && K>=8 && (K&7)==0)
        { r19_genfft_packed_fwd_avx512(ri,ii,ro,io,K); return; }
#endif
#ifdef __AVX2__
    if (isa>=R19_ISA_AVX2 && K>=4 && (K&3)==0)
        { r19_genfft_packed_fwd_avx2(ri,ii,ro,io,K); return; }
#endif
    r19_genfft_packed_fwd_scalar(ri,ii,ro,io,K);
}

static inline void r19_dispatch_packed_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{ /* Backward packed = forward packed with swapped re/im */
    r19_isa_t isa = r19_detect_isa();
#ifdef __AVX512F__
    if (isa==R19_ISA_AVX512 && K>=8 && (K&7)==0) {
        const size_t T=8, bs=19*T, nb=K/T;
        for (size_t b=0; b<nb; b++)
            radix19_genfft_bwd_avx512(ri+b*bs,ii+b*bs,ro+b*bs,io+b*bs,T);
        return; }
#endif
#ifdef __AVX2__
    if (isa>=R19_ISA_AVX2 && K>=4 && (K&3)==0) {
        const size_t T=4, bs=19*T, nb=K/T;
        for (size_t b=0; b<nb; b++)
            radix19_genfft_bwd_avx2(ri+b*bs,ii+b*bs,ro+b*bs,io+b*bs,T);
        return; }
#endif
    const size_t nb=K;
    for (size_t b=0; b<nb; b++)
        radix19_genfft_bwd_scalar(ri+b*19,ii+b*19,ro+b*19,io+b*19,1);
}

#endif /* FFT_RADIX19_DISPATCH_H */
