/**
 * @file fft_radix11_dispatch.h
 * @brief Runtime ISA dispatch for DFT-11 — auto-selects best kernel
 *
 * Detects AVX-512, AVX2, or scalar at runtime via CPUID.
 * All three ISA paths compile into one binary.
 *
 * Usage:
 *   r11_dispatch_fwd(in_re, in_im, out_re, out_im, K);
 *   r11_dispatch_packed_fwd(packed_re, packed_im, out_re, out_im, K);
 */

#ifndef FFT_RADIX11_DISPATCH_H
#define FFT_RADIX11_DISPATCH_H

#include "fft_radix11_genfft.h"

/* ═══════════════════════════════════════════════════════════════
 * CPUID-BASED ISA DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>

static inline int r11_has_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx >> 16) & 1;  /* AVX-512F = bit 16 of EBX */
}

static inline int r11_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx >> 5) & 1;  /* AVX2 = bit 5 of EBX */
}

static inline int r11_has_fma(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
    return (ecx >> 12) & 1;  /* FMA = bit 12 of ECX */
}

#elif defined(_MSC_VER)
#include <intrin.h>

static inline int r11_has_avx512f(void) {
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] >> 16) & 1;
}

static inline int r11_has_avx2(void) {
    int info[4];
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
}

static inline int r11_has_fma(void) {
    int info[4];
    __cpuid(info, 1);
    return (info[2] >> 12) & 1;
}

#else
static inline int r11_has_avx512f(void) { return 0; }
static inline int r11_has_avx2(void)    { return 0; }
static inline int r11_has_fma(void)     { return 0; }
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL ENUM
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    R11_ISA_SCALAR = 0,
    R11_ISA_AVX2   = 1,
    R11_ISA_AVX512 = 2
} r11_isa_t;

/** Detect best available ISA (cached after first call). */
static inline r11_isa_t r11_detect_isa(void) {
    static r11_isa_t cached = (r11_isa_t)-1;
    if (cached != (r11_isa_t)-1) return cached;

    if (r11_has_avx512f() && r11_has_fma())
        cached = R11_ISA_AVX512;
    else if (r11_has_avx2() && r11_has_fma())
        cached = R11_ISA_AVX2;
    else
        cached = R11_ISA_SCALAR;
    return cached;
}

/** SIMD width (k-step) for the detected ISA. */
static inline size_t r11_simd_width(void) {
    switch (r11_detect_isa()) {
        case R11_ISA_AVX512: return 8;
        case R11_ISA_AVX2:   return 4;
        default:             return 1;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH
 *
 * K must be a multiple of the SIMD width (8 for AVX-512, 4 for AVX2).
 * Falls back to scalar for any K.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_dispatch_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    r11_isa_t isa = r11_detect_isa();
#ifdef __AVX512F__
    if (isa == R11_ISA_AVX512 && K >= 8 && (K & 7) == 0) {
        radix11_genfft_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (isa >= R11_ISA_AVX2 && K >= 4 && (K & 3) == 0) {
        radix11_genfft_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    radix11_genfft_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void r11_dispatch_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    /* Backward = forward with swapped re↔im */
    r11_dispatch_fwd(in_im, in_re, out_im, out_re, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH
 *
 * Data must be in packed contiguous layout (use r11_pack).
 * K must be multiple of SIMD width.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_dispatch_packed_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    r11_isa_t isa = r11_detect_isa();
#ifdef __AVX512F__
    if (isa == R11_ISA_AVX512 && K >= 8 && (K & 7) == 0) {
        r11_genfft_packed_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (isa >= R11_ISA_AVX2 && K >= 4 && (K & 3) == 0) {
        r11_genfft_packed_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    r11_genfft_packed_fwd_scalar(in_re, in_im, out_re, out_im, K);
}

static inline void r11_dispatch_packed_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    r11_isa_t isa = r11_detect_isa();
#ifdef __AVX512F__
    if (isa == R11_ISA_AVX512 && K >= 8 && (K & 7) == 0) {
        r11_genfft_packed_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (isa >= R11_ISA_AVX2 && K >= 4 && (K & 3) == 0) {
        r11_genfft_packed_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
    }
#endif
    /* Scalar backward = forward with swap */
    r11_genfft_packed_fwd_scalar(in_im, in_re, out_im, out_re, K);
}

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE: AUTO PACK + DISPATCH + UNPACK
 *
 * For callers who have strided data but want packed performance.
 * Allocates scratch buffers on the stack for small K, heap for large.
 * ═══════════════════════════════════════════════════════════════ */

#include <stdlib.h>
#include <string.h>

static inline void r11_auto_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t K)
{
    size_t T = r11_simd_width();
    if (T <= 1 || K < T || (K % T) != 0) {
        radix11_genfft_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }

    size_t N = 11 * K;
    /* For K <= 128, use strided (still L1-resident) */
    if (K <= 128) {
        r11_dispatch_fwd(in_re, in_im, out_re, out_im, K);
        return;
    }

    /* Large K: pack → dispatch → unpack */
    double *pir = (double*)aligned_alloc(64, N * sizeof(double));
    double *pii = (double*)aligned_alloc(64, N * sizeof(double));
    double *por = (double*)aligned_alloc(64, N * sizeof(double));
    double *poi = (double*)aligned_alloc(64, N * sizeof(double));

    r11_pack(in_re, in_im, pir, pii, K, T);
    r11_dispatch_packed_fwd(pir, pii, por, poi, K);
    r11_unpack(por, poi, out_re, out_im, K, T);

    free(pir); free(pii); free(por); free(poi);
}

#endif /* FFT_RADIX11_DISPATCH_H */
