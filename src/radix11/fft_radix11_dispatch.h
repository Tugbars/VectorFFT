/**
 * @file fft_radix11_dispatch.h
 * @brief Runtime ISA dispatch for DFT-11 — strided, packed, and pack+walk
 *
 * Three-tier packed dispatch (consistent with radix-8/32):
 *   K <= RADIX11_WALK_THRESHOLD: packed table (pre-packed twiddles)
 *   K >  RADIX11_WALK_THRESHOLD: pack+walk (2 walked bases, derive 8)
 *
 * Usage:
 *   r11_dispatch_fwd(in_re, in_im, out_re, out_im, K);
 *   r11_dispatch_packed_fwd(packed_re, packed_im, out_re, out_im, K);
 *   r11_tw_packed_auto_fwd(..., walk_plan, K, T);  // auto table vs walk
 */

#ifndef FFT_RADIX11_DISPATCH_H
#define FFT_RADIX11_DISPATCH_H

#include "fft_radix11_genfft.h"

#if defined(__AVX512F__) || defined(__AVX512F)
#include "avx512/fft_radix11_avx512_tw_pack_walk.h"
#endif
#if defined(__AVX2__)
#include "avx2/fft_radix11_avx2_tw_pack_walk.h"
#endif

/* Walk threshold: use pack+walk when K > this */
#ifndef RADIX11_WALK_THRESHOLD
#define RADIX11_WALK_THRESHOLD 512
#endif

/* ═══════════════════════════════════════════════════════════════
 * CPUID-BASED ISA DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>

static inline int r11_has_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx >> 16) & 1;
}

static inline int r11_has_avx2(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx >> 5) & 1;
}

static inline int r11_has_fma(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
    return (ecx >> 12) & 1;
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
 * ISA LEVEL
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    R11_ISA_SCALAR = 0,
    R11_ISA_AVX2   = 1,
    R11_ISA_AVX512 = 2
} r11_isa_t;

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

static inline size_t r11_simd_width(void) {
    switch (r11_detect_isa()) {
        case R11_ISA_AVX512: return 8;
        case R11_ISA_AVX2:   return 4;
        default:             return 1;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * STRIDED DISPATCH
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
    r11_dispatch_fwd(in_im, in_re, out_im, out_re, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH (table-based)
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
    r11_genfft_packed_fwd_scalar(in_im, in_re, out_im, out_re, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PACK+WALK AUTO-DISPATCH
 *
 * K <= RADIX11_WALK_THRESHOLD → packed table
 * K >  RADIX11_WALK_THRESHOLD → pack+walk (zero tw table)
 *
 * walk_plan: void* to radix11_walk_plan_t (AVX-512) or
 *            radix11_walk_plan_avx2_t (AVX2). NULL for table path.
 * ═══════════════════════════════════════════════════════════════ */

static inline void r11_tw_packed_auto_fwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX11_WALK_THRESHOLD && walk_plan) {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8) {
            radix11_tw_pack_walk_fwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix11_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4) {
            radix11_tw_pack_walk_fwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix11_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    /* Fall back to existing walk driver or packed table */
    (void)tw_re; (void)tw_im;
    r11_dispatch_packed_fwd(in_re, in_im, out_re, out_im, K);
}

static inline void r11_tw_packed_auto_bwd(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    const void * __restrict__ walk_plan,
    size_t K, size_t T)
{
    if (K > RADIX11_WALK_THRESHOLD && walk_plan) {
#if defined(__AVX512F__) || defined(__AVX512F)
        if (T == 8) {
            radix11_tw_pack_walk_bwd_avx512(
                in_re, in_im, out_re, out_im,
                (const radix11_walk_plan_t *)walk_plan, K);
            return;
        }
#endif
#ifdef __AVX2__
        if (T == 4) {
            radix11_tw_pack_walk_bwd_avx2(
                in_re, in_im, out_re, out_im,
                (const radix11_walk_plan_avx2_t *)walk_plan, K);
            return;
        }
#endif
    }
    (void)tw_re; (void)tw_im;
    r11_dispatch_packed_bwd(in_re, in_im, out_re, out_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r11_flat_tw_size(size_t K)  { return 10 * K; }
static inline size_t r11_data_size(size_t K)     { return 11 * K; }

/** Returns 1 if pack+walk should be used instead of packed table */
static inline int r11_should_walk(size_t K) {
    return (K > RADIX11_WALK_THRESHOLD) ? 1 : 0;
}

/** Packed twiddle table size. Returns 0 if walk mode (no table needed). */
static inline size_t r11_packed_tw_size(size_t K) {
    return r11_should_walk(K) ? 0 : 10 * K;
}

static inline size_t r11_packed_optimal_T(size_t K) {
    r11_isa_t isa = r11_detect_isa();
    if (isa == R11_ISA_AVX512 && K >= 8 && (K & 7) == 0) return 8;
    if (isa >= R11_ISA_AVX2   && K >= 4 && (K & 3) == 0) return 4;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE: AUTO PACK + DISPATCH + UNPACK
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

    /* For K <= 128, use strided (still L1-resident) */
    if (K <= 128) {
        r11_dispatch_fwd(in_re, in_im, out_re, out_im, K);
        return;
    }

    /* Large K: pack → dispatch → unpack */
    size_t N = 11 * K;
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