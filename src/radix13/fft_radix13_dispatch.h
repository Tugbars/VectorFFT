/**
 * @file fft_radix13_dispatch.h
 * @brief Runtime ISA dispatch for DFT-13 — auto-selects best kernel
 */

#ifndef FFT_RADIX13_DISPATCH_H
#define FFT_RADIX13_DISPATCH_H

#include "fft_radix13_genfft.h"

/* Reuse ISA detection from radix-11 dispatcher if available,
 * otherwise inline a local copy. */
#ifndef FFT_RADIX11_DISPATCH_H

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static inline int r13_has_avx512f(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>16)&1 : 0; }
static inline int r13_has_avx2(void) {
    unsigned int a,b,c,d;
    return __get_cpuid_count(7,0,&a,&b,&c,&d) ? (b>>5)&1 : 0; }
static inline int r13_has_fma(void) {
    unsigned int a,b,c,d;
    return __get_cpuid(1,&a,&b,&c,&d) ? (c>>12)&1 : 0; }
#elif defined(_MSC_VER)
#include <intrin.h>
static inline int r13_has_avx512f(void) { int i[4]; __cpuidex(i,7,0); return (i[1]>>16)&1; }
static inline int r13_has_avx2(void)    { int i[4]; __cpuidex(i,7,0); return (i[1]>>5)&1; }
static inline int r13_has_fma(void)     { int i[4]; __cpuid(i,1); return (i[2]>>12)&1; }
#else
static inline int r13_has_avx512f(void) { return 0; }
static inline int r13_has_avx2(void)    { return 0; }
static inline int r13_has_fma(void)     { return 0; }
#endif

typedef enum { R13_ISA_SCALAR=0, R13_ISA_AVX2=1, R13_ISA_AVX512=2 } r13_isa_t;

static inline r13_isa_t r13_detect_isa(void) {
    static r13_isa_t c = (r13_isa_t)-1;
    if (c != (r13_isa_t)-1) return c;
    if (r13_has_avx512f() && r13_has_fma()) c = R13_ISA_AVX512;
    else if (r13_has_avx2() && r13_has_fma()) c = R13_ISA_AVX2;
    else c = R13_ISA_SCALAR;
    return c;
}

static inline size_t r13_simd_width(void) {
    switch(r13_detect_isa()){
        case R13_ISA_AVX512: return 8;
        case R13_ISA_AVX2:   return 4;
        default:             return 1; }
}

#else
/* If radix-11 dispatch is included, reuse its detection */
#define r13_detect_isa() ((r13_isa_t)r11_detect_isa())
#define r13_simd_width() r11_simd_width()
typedef r11_isa_t r13_isa_t;
#define R13_ISA_SCALAR R11_ISA_SCALAR
#define R13_ISA_AVX2   R11_ISA_AVX2
#define R13_ISA_AVX512 R11_ISA_AVX512
#endif

static inline void r13_dispatch_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    r13_isa_t isa = r13_detect_isa();
#ifdef __AVX512F__
    if (isa==R13_ISA_AVX512 && K>=8 && (K&7)==0)
        { radix13_genfft_fwd_avx512(ri,ii,ro,io,K); return; }
#endif
#ifdef __AVX2__
    if (isa>=R13_ISA_AVX2 && K>=4 && (K&3)==0)
        { radix13_genfft_fwd_avx2(ri,ii,ro,io,K); return; }
#endif
    radix13_genfft_fwd_scalar(ri,ii,ro,io,K);
}

static inline void r13_dispatch_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{ r13_dispatch_fwd(ii,ri,io,ro,K); }

static inline void r13_dispatch_packed_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    r13_isa_t isa = r13_detect_isa();
#ifdef __AVX512F__
    if (isa==R13_ISA_AVX512 && K>=8 && (K&7)==0)
        { r13_genfft_packed_fwd_avx512(ri,ii,ro,io,K); return; }
#endif
#ifdef __AVX2__
    if (isa>=R13_ISA_AVX2 && K>=4 && (K&3)==0)
        { r13_genfft_packed_fwd_avx2(ri,ii,ro,io,K); return; }
#endif
    r13_genfft_packed_fwd_scalar(ri,ii,ro,io,K);
}

static inline void r13_dispatch_packed_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    r13_isa_t isa = r13_detect_isa();
#ifdef __AVX512F__
    if (isa==R13_ISA_AVX512 && K>=8 && (K&7)==0)
        { r13_genfft_packed_bwd_avx512(ri,ii,ro,io,K); return; }
#endif
#ifdef __AVX2__
    if (isa>=R13_ISA_AVX2 && K>=4 && (K&3)==0)
        { r13_genfft_packed_bwd_avx2(ri,ii,ro,io,K); return; }
#endif
    r13_genfft_packed_bwd_scalar(ri,ii,ro,io,K);
}

#endif /* FFT_RADIX13_DISPATCH_H */
