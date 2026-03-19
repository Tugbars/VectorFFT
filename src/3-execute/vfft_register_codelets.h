/**
 * @file vfft_register_codelets.h
 * @brief Wire all optimized VectorFFT codelets into the planner registry
 *
 * Include this AFTER vfft_planner.h and all codelet/dispatch headers.
 * Creates dispatch wrappers that match vfft_codelet_fn signature
 * and registers the best ISA for each radix.
 *
 * Usage:
 *   #include "vfft_planner.h"
 *
 *   // New-style standalone modules (each has dispatch.h)
 *   #include "fft_radix2_dispatch.h"
 *   #include "fft_radix3_dispatch.h"   // when available
 *   #include "fft_radix4_dispatch.h"
 *   #include "fft_radix5_dispatch.h"
 *   #include "fft_radix7_dispatch.h"   // when available
 *   #include "fft_radix8_dispatch.h"
 *   #include "fft_radix32_dispatch.h"  // when available
 *
 *   // Prime genfft modules
 *   #include "fft_radix11_genfft.h"
 *   #include "fft_radix13_genfft.h"
 *   #include "fft_radix17_genfft.h"
 *   #include "fft_radix19_genfft.h"
 *   #include "fft_radix23_genfft.h"
 *
 *   #include "vfft_register_codelets.h"
 *
 *   vfft_codelet_registry reg;
 *   vfft_register_all(&reg);
 *   vfft_plan *plan = vfft_plan_create(N, &reg);
 */

#ifndef VFFT_REGISTER_CODELETS_H
#define VFFT_REGISTER_CODELETS_H

#include "vfft_planner.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static inline int vfft_has_avx512f(void)
{
    unsigned int a, b, c, d;
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) ? (b >> 16) & 1 : 0;
}
static inline int vfft_has_avx2(void)
{
    unsigned int a, b, c, d;
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) ? (b >> 5) & 1 : 0;
}
static inline int vfft_has_fma(void)
{
    unsigned int a, b, c, d;
    return __get_cpuid(1, &a, &b, &c, &d) ? (c >> 12) & 1 : 0;
}
#elif defined(_MSC_VER)
#include <intrin.h>
static inline int vfft_has_avx512f(void)
{
    int i[4];
    __cpuidex(i, 7, 0);
    return (i[1] >> 16) & 1;
}
static inline int vfft_has_avx2(void)
{
    int i[4];
    __cpuidex(i, 7, 0);
    return (i[1] >> 5) & 1;
}
static inline int vfft_has_fma(void)
{
    int i[4];
    __cpuid(i, 1);
    return (i[2] >> 12) & 1;
}
#else
static inline int vfft_has_avx512f(void) { return 0; }
static inline int vfft_has_avx2(void) { return 0; }
static inline int vfft_has_fma(void) { return 0; }
#endif

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t vfft_detect_isa(void)
{
    static vfft_isa_level_t cached = (vfft_isa_level_t)-1;
    if (cached != (vfft_isa_level_t)-1)
        return cached;
    if (vfft_has_avx512f() && vfft_has_fma())
        cached = VFFT_ISA_AVX512;
    else if (vfft_has_avx2() && vfft_has_fma())
        cached = VFFT_ISA_AVX2;
    else
        cached = VFFT_ISA_SCALAR;
    return cached;
}

/* ═══════════════════════════════════════════════════════════════
 * DISPATCH WRAPPER MACRO
 *
 * Creates a pair of functions matching vfft_codelet_fn signature:
 *   vfft_dispatch_rR_fwd(in_re, in_im, out_re, out_im, K)
 *   vfft_dispatch_rR_bwd(in_re, in_im, out_re, out_im, K)
 *
 * Dispatches to prefix_fwd_avx512 / prefix_fwd_avx2 / prefix_fwd_scalar
 * based on runtime ISA and K alignment.
 * ═══════════════════════════════════════════════════════════════ */

/* Conditional ISA macros */
#if defined(__AVX512F__) || defined(__AVX512F)
#define IF_AVX512(x) x
#else
#define IF_AVX512(x)
#endif
#ifdef __AVX2__
#define IF_AVX2(x) x
#else
#define IF_AVX2(x)
#endif

#define VFFT_DISPATCH_WRAPPER(R, prefix)                                      \
    static void vfft_dispatch_r##R##_fwd(                                     \
        const double *ri, const double *ii, double *ro, double *io, size_t K) \
    {                                                                         \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,K); return; })  \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,K); return; })      \
        prefix##_fwd_scalar(ri, ii, ro, io, K);                               \
    }                                                                         \
    static void vfft_dispatch_r##R##_bwd(                                     \
        const double *ri, const double *ii, double *ro, double *io, size_t K) \
    {                                                                         \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_bwd_avx512(ri,ii,ro,io,K); return; })  \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_bwd_avx2(ri,ii,ro,io,K); return; })      \
        prefix##_bwd_scalar(ri, ii, ro, io, K);                               \
    }

/* ═══════════════════════════════════════════════════════════════
 * INSTANTIATE DISPATCH WRAPPERS
 *
 * New standalone modules: radixN_notw_dit_kernel_{fwd,bwd}_{isa}
 * Genfft primes:          radixN_genfft_{fwd,bwd}_{isa}
 *
 * Guard each on the codelet header's include guard.
 * ═══════════════════════════════════════════════════════════════ */

/* ── Twiddled dispatch wrapper ──
 *
 * Creates vfft_tw_dispatch_rR_fwd / _bwd matching vfft_tw_codelet_fn:
 *   (in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 *
 * For radices with scalar tw codelets, uses VFFT_TW_DISPATCH_WRAPPER.
 * Radix-16 lacks scalar tw — handled with a special wrapper below.
 */

#define VFFT_TW_DISPATCH_WRAPPER(R, prefix)                                  \
    static void vfft_tw_dispatch_r##R##_fwd(                                 \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_fwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }                                                                        \
    static void vfft_tw_dispatch_r##R##_bwd(                                 \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_bwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_bwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_bwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }

/* ── Twiddled wrappers for radices with full ISA coverage ── */

#ifdef FFT_RADIX2_DISPATCH_H
#ifdef FFT_RADIX2_DISPATCH_H
/* R=2 DIT tw: delegate to dispatch.h */
static void vfft_tw_dispatch_r2_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix2_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r2_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix2_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX3_DISPATCH_H
#ifdef FFT_RADIX3_DISPATCH_H
/* R=3 DIT tw: delegate to dispatch.h (K-last signature) */
static void vfft_tw_dispatch_r3_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix3_tw_fwd(ri, ii, ro, io, twr, twi, K);
}
static void vfft_tw_dispatch_r3_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix3_tw_bwd(ri, ii, ro, io, twr, twi, K);
}
#endif
#endif

#ifdef FFT_RADIX4_DISPATCH_H
#ifdef FFT_RADIX4_DISPATCH_H
/* R=4 DIT tw: delegate to dispatch.h */
static void vfft_tw_dispatch_r4_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix4_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r4_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix4_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX5_DISPATCH_H
#ifdef FFT_RADIX5_DISPATCH_H
/* R=5 DIT tw: delegate to dispatch.h */
static void vfft_tw_dispatch_r5_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r5_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

/* R=7: genfft provides DIT fwd only (no DIT bwd) */
#ifdef FFT_RADIX7_DISPATCH_H
static void vfft_tw_dispatch_r7_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix7_tw_forward(K, ri, ii, ro, io, twr, twi);
}
#endif

#ifdef FFT_RADIX8_DISPATCH_H
/* R=8 DIT tw: delegate to dispatch.h (log₃ + scalar fallback for any K) */
static void vfft_tw_dispatch_r8_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r8_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

#ifdef FFT_RADIX32_DISPATCH_H
/*
 * R=32 DIT tw: delegate to dispatch.h (flat + ladder + scalar fallback).
 * Ladder uses same tw table layout as flat — pass flat pointers for both.
 */
static void vfft_tw_dispatch_r32_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix32_tw_forward(K, ri, ii, ro, io, twr, twi, twr, twi);
}
static void vfft_tw_dispatch_r32_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix32_tw_backward(K, ri, ii, ro, io, twr, twi, twr, twi);
}
#endif

#ifdef FFT_RADIX25_DISPATCH_H
/* R=25 DIT tw: delegate to dispatch.h (5×5CT + scalar fallback) */
static void vfft_tw_dispatch_r25_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r25_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

/* ── R=25 Interleaved tw dispatch ── */
#ifdef FFT_RADIX25_DISPATCH_H
static void vfft_tw_il_dispatch_r25_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_forward_il(K, in, out, twr, twi);
}
#endif

#ifdef FFT_RADIX25_DISPATCH_H
static void vfft_tw_il_dif_dispatch_r25_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_dif_backward_il(K, in, out, twr, twi);
}
#endif

/* ── R=5 Interleaved tw dispatch ── */
#ifdef FFT_RADIX5_DISPATCH_H
static void vfft_tw_il_dispatch_r5_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_forward_il(K, in, out, twr, twi);
}
#endif

#ifdef FFT_RADIX5_DISPATCH_H
static void vfft_tw_il_dif_dispatch_r5_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_dif_backward_il(K, in, out, twr, twi);
}
#endif

/* ── R=7 Interleaved tw dispatch ── */
#ifdef FFT_RADIX7_DISPATCH_H
static void vfft_tw_il_dispatch_r7_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix7_tw_forward_il(K, in, out, twr, twi);
}
#endif

#ifdef FFT_RADIX7_DISPATCH_H
static void vfft_tw_il_dif_dispatch_r7_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix7_tw_dif_backward_il(K, in, out, twr, twi);
}
#endif

/* ── R=8 Interleaved tw dispatch ── */
#ifdef FFT_RADIX8_DISPATCH_H
static void vfft_tw_il_dispatch_r8_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_forward_il(K, in, out, twr, twi);
}
#endif

#ifdef FFT_RADIX8_DISPATCH_H
static void vfft_tw_il_dif_dispatch_r8_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_dif_backward_il(K, in, out, twr, twi);
}
#endif

/* ── R=10 Interleaved tw dispatch ── */
#ifdef FFT_RADIX10_DISPATCH_H
static void vfft_tw_il_dispatch_r10_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_forward_il(in, out, twr, twi, K);
}
#endif

#ifdef FFT_RADIX10_DISPATCH_H
static void vfft_tw_il_dif_dispatch_r10_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_dif_backward_il(in, out, twr, twi, K);
}
#endif

#ifdef FFT_RADIX10_DISPATCH_H
#ifdef FFT_RADIX10_DISPATCH_H
/* R=10 DIT tw: delegate to dispatch.h */
static void vfft_tw_dispatch_r10_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r10_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

/* ── Radix-16 tw: no scalar tw kernel, SIMD-only ──
 * Only dispatches for K aligned to SIMD width. The planner will
 * fall back to notw+twiddle for unaligned K (rare in practice —
 * K at a radix-16 stage is always a product of inner radices). */

#ifdef FFT_RADIX16_DISPATCH_H
/* R=16 DIT tw: delegate to dispatch.h */
static void vfft_tw_dispatch_r16_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix16_tw_strided_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r16_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix16_tw_strided_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * DIF TWIDDLED DISPATCH WRAPPERS
 *
 * Same pattern as DIT tw wrappers, but calls _dif_ kernels.
 * DIF codelets apply twiddle AFTER butterfly (output side).
 *
 * Guarded on FFT_RADIXN_DIF_DISPATCH_H — only available when
 * the DIF dispatch header is included.
 * ═══════════════════════════════════════════════════════════════ */

#define VFFT_TW_DIF_DISPATCH_WRAPPER(R, prefix)                              \
    static void vfft_tw_dif_dispatch_r##R##_fwd(                             \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_fwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }                                                                        \
    static void vfft_tw_dif_dispatch_r##R##_bwd(                             \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_bwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_bwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_bwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }

/* ── Small radix DIF codelets ── */

#ifdef FFT_RADIX2_DIF_DISPATCH_H
#ifdef FFT_RADIX2_DISPATCH_H
/* R=2 DIF tw: delegate to dispatch.h */
static void vfft_tw_dif_dispatch_r2_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix2_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r2_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix2_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX3_DIF_DISPATCH_H
#ifdef FFT_RADIX3_DISPATCH_H
/* R=3 DIF tw: delegate to dispatch.h */
static void vfft_tw_dif_dispatch_r3_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix3_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r3_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix3_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX4_DIF_DISPATCH_H
#ifdef FFT_RADIX4_DISPATCH_H
/* R=4 DIF tw: delegate to dispatch.h */
static void vfft_tw_dif_dispatch_r4_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix4_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r4_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix4_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX5_DIF_DISPATCH_H
#ifdef FFT_RADIX5_DISPATCH_H
/* R=5 DIF tw: delegate to dispatch.h */
static void vfft_tw_dif_dispatch_r5_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r5_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix5_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

/* R=7: genfft provides DIF bwd only (no DIF fwd) */
#ifdef FFT_RADIX7_DIF_DISPATCH_H
static void vfft_tw_dif_dispatch_r7_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix7_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

#ifdef FFT_RADIX8_DIF_DISPATCH_H
/* R=8 DIF tw: delegate to dif_dispatch.h */
static void vfft_tw_dif_dispatch_r8_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r8_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix8_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

/* ── Radix-32 DIF (fused single-pass, different kernel prefix) ── */

#ifdef FFT_RADIX32_DIF_DISPATCH_H
/* R=32 DIF tw: delegate to dif_dispatch.h (flat + scalar fallback) */
static void vfft_tw_dif_dispatch_r32_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix32_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r32_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix32_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

/* ── Radix-16 DIF (fused single-pass, flat kernel prefix) ── */

#ifdef FFT_RADIX16_DIF_DISPATCH_H
#ifdef FFT_RADIX16_DIF_DISPATCH_H
/* R=16 DIF tw: delegate to dif_dispatch.h */
static void vfft_tw_dif_dispatch_r16_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix16_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r16_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix16_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif
#endif

#ifdef FFT_RADIX25_DIF_DISPATCH_H
/* R=25 DIF tw: delegate to dif_dispatch.h (flat + scalar fallback) */
static void vfft_tw_dif_dispatch_r25_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r25_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix25_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

#ifdef FFT_RADIX10_DIF_DISPATCH_H
/* R=10 DIF tw: delegate to dif_dispatch.h */
static void vfft_tw_dif_dispatch_r10_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r10_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix10_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}
#endif

/* ── N1 (notw) dispatch wrappers ── */

/* ── New standalone modules ── */

#ifdef FFT_RADIX2_DISPATCH_H
#ifdef FFT_RADIX2_DISPATCH_H
static void vfft_dispatch_r2_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix2_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r2_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix2_notw_backward(K, ri, ii, ro, io);
}
#endif
#endif

#ifdef FFT_RADIX3_DISPATCH_H
#ifdef FFT_RADIX3_DISPATCH_H
static void vfft_dispatch_r3_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix3_notw_fwd(ri, ii, ro, io, K);
}
static void vfft_dispatch_r3_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix3_notw_bwd(ri, ii, ro, io, K);
}
#endif
#endif

#ifdef FFT_RADIX4_DISPATCH_H
#ifdef FFT_RADIX4_DISPATCH_H
static void vfft_dispatch_r4_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix4_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r4_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix4_notw_backward(K, ri, ii, ro, io);
}
#endif
#endif

#ifdef FFT_RADIX5_DISPATCH_H
#ifdef FFT_RADIX5_DISPATCH_H
static void vfft_dispatch_r5_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix5_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r5_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix5_notw_backward(K, ri, ii, ro, io);
}
#endif
#endif

#ifdef FFT_RADIX7_DISPATCH_H
#ifdef FFT_RADIX7_DISPATCH_H
static void vfft_dispatch_r7_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix7_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r7_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix7_notw_backward(K, ri, ii, ro, io);
}
#endif
#endif

#ifdef FFT_RADIX8_DISPATCH_H
/* R=8 N1: delegate to dispatch.h */
static void vfft_dispatch_r8_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix8_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r8_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix8_notw_backward(K, ri, ii, ro, io);
}
#endif

#ifdef FFT_RADIX16_DISPATCH_H
#ifdef FFT_RADIX16_DISPATCH_H
static void vfft_dispatch_r16_fwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix16_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r16_bwd(const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix16_n1_backward(K, ri, ii, ro, io);
}
#endif
#endif

#ifdef FFT_RADIX32_DISPATCH_H
/* R=32 N1: delegate to dispatch.h (all ISAs + scalar fallback) */
static void vfft_dispatch_r32_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix32_notw_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r32_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix32_notw_backward(K, ri, ii, ro, io);
}
#endif

#ifdef FFT_RADIX25_DISPATCH_H
/* R=25 N1: delegate to dispatch.h (all ISAs + scalar fallback) */
static void vfft_dispatch_r25_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix25_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r25_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix25_n1_backward(K, ri, ii, ro, io);
}
#endif

#ifdef FFT_RADIX10_DISPATCH_H
/* R=10 N1: delegate to dispatch.h */
static void vfft_dispatch_r10_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix10_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r10_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix10_n1_backward(K, ri, ii, ro, io);
}
#endif

/* ── Genfft prime modules ── */

#ifdef FFT_RADIX11_GENFFT_H
static void vfft_dispatch_r11_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix11_genfft_fwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix11_genfft_fwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix11_genfft_fwd_scalar(ri, ii, ro, io, K);
}
static void vfft_dispatch_r11_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix11_genfft_bwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix11_genfft_bwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix11_genfft_bwd_scalar(ri, ii, ro, io, K);
}
static void vfft_tw_dispatch_r11_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix11_genfft_tw_fwd_avx512(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix11_genfft_tw_fwd_avx2(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
    radix11_genfft_tw_fwd_scalar(ri, ii, ro, io, twr, twi, K);
}
static void vfft_tw_dif_dispatch_r11_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix11_genfft_tw_dif_bwd_avx512(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix11_genfft_tw_dif_bwd_avx2(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
    radix11_genfft_tw_dif_bwd_scalar(ri, ii, ro, io, twr, twi, K);
}
#endif
#ifdef FFT_RADIX13_GENFFT_H
static void vfft_dispatch_r13_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix13_genfft_fwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix13_genfft_fwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix13_genfft_fwd_scalar(ri, ii, ro, io, K);
}
static void vfft_dispatch_r13_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix13_genfft_bwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix13_genfft_bwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix13_genfft_bwd_scalar(ri, ii, ro, io, K);
}
static void vfft_tw_dispatch_r13_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix13_genfft_tw_fwd_avx512(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix13_genfft_tw_fwd_avx2(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
    radix13_genfft_tw_fwd_scalar(ri, ii, ro, io, twr, twi, K);
}
static void vfft_tw_dif_dispatch_r13_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix13_genfft_tw_dif_bwd_avx512(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix13_genfft_tw_dif_bwd_avx2(ri, ii, ro, io, twr, twi, K);
        return;
    }
#endif
    radix13_genfft_tw_dif_bwd_scalar(ri, ii, ro, io, twr, twi, K);
}
#endif
#ifdef FFT_RADIX17_GENFFT_H
/* R=17 genfft N1: compile-time ISA selection (no dispatch.h) */
static void vfft_dispatch_r17_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix17_genfft_fwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix17_genfft_fwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix17_genfft_fwd_scalar(ri, ii, ro, io, K);
}
static void vfft_dispatch_r17_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix17_genfft_bwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix17_genfft_bwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix17_genfft_bwd_scalar(ri, ii, ro, io, K);
}
#endif
#ifdef FFT_RADIX19_GENFFT_H
static void vfft_dispatch_r19_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix19_genfft_fwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix19_genfft_fwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix19_genfft_fwd_scalar(ri, ii, ro, io, K);
}
static void vfft_dispatch_r19_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix19_genfft_bwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix19_genfft_bwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix19_genfft_bwd_scalar(ri, ii, ro, io, K);
}
#endif
#ifdef FFT_RADIX23_GENFFT_H
static void vfft_dispatch_r23_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix23_genfft_fwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix23_genfft_fwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix23_genfft_fwd_scalar(ri, ii, ro, io, K);
}
static void vfft_dispatch_r23_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 8 && (K & 7) == 0)
    {
        radix23_genfft_bwd_avx512(ri, ii, ro, io, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
    {
        radix23_genfft_bwd_avx2(ri, ii, ro, io, K);
        return;
    }
#endif
    radix23_genfft_bwd_scalar(ri, ii, ro, io, K);
}
#endif

/* ── N1-64 / N1-128 adapters ──
 *
 * These have K-first signature: func(K, in_re, in_im, out_re, out_im)
 * The planner needs K-last:     func(in_re, in_im, out_re, out_im, K)
 * Thin wrappers swap the argument order.
 */

#ifdef FFT_RADIX64_DISPATCH_H
static void vfft_dispatch_r64_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix64_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r64_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    radix64_n1_backward(K, ri, ii, ro, io);
}
#endif

/* ── R=64 tw dispatch (delegates to dispatch.h: DAG + 8×8CT + scalar) ── */
#ifdef FFT_RADIX64_DISPATCH_H
/* R=64 DIT tw: delegate to dispatch.h (DAG K<=12, 8×8CT K>12, scalar fallback) */
static void vfft_tw_dispatch_r64_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dispatch_r64_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_backward(K, ri, ii, ro, io, twr, twi);
}

/* ── R=64 DIF tw dispatch ── */
static void vfft_tw_dif_dispatch_r64_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_dif_forward(K, ri, ii, ro, io, twr, twi);
}
static void vfft_tw_dif_dispatch_r64_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_dif_backward(K, ri, ii, ro, io, twr, twi);
}

/* ── R=64 IL tw dispatch ── */
static void vfft_tw_il_dispatch_r64_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_forward_il(in, out, twr, twi, K);
}
static void vfft_tw_il_dif_dispatch_r64_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
    radix64_tw_dif_backward_il(in, out, twr, twi, K);
}
#endif /* FFT_RADIX64_DISPATCH_H */

/* ── R=32 IL tw dispatch ── */
#ifdef FFT_RADIX32_DISPATCH_H
static void vfft_tw_il_dispatch_r32_fwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__) && defined(FFT_RADIX32_AVX512_TW_LADDER_IL_H)
    if (K >= 8 && (K & 7) == 0)
    {
        radix32_tw_flat_dit_kernel_il_fwd_avx512(in, out, twr, twi, K);
        return;
    }
#endif
#if defined(__AVX2__) && defined(FFT_RADIX32_AVX2_TW_IL_H)
    if (K >= 4 && (K & 3) == 0)
    {
        radix32_tw_flat_dit_kernel_il_fwd_avx2(in, out, twr, twi, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)twr;
    (void)twi;
    (void)K;
}
static void vfft_tw_il_dif_dispatch_r32_bwd(
    const double *in, double *out,
    const double *twr, const double *twi, size_t K)
{
#if defined(__AVX512F__) && defined(FFT_RADIX32_AVX512_DIF_TW_IL_H)
    if (K >= 8 && (K & 7) == 0)
    {
        radix32_tw_flat_dif_kernel_il_bwd_avx512(in, out, twr, twi, K);
        return;
    }
#endif
#if defined(__AVX2__) && defined(FFT_RADIX32_AVX2_DIF_TW_IL_H)
    if (K >= 4 && (K & 3) == 0)
    {
        radix32_tw_flat_dif_kernel_il_bwd_avx2(in, out, twr, twi, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)twr;
    (void)twi;
    (void)K;
}
#endif /* FFT_RADIX32_DISPATCH_H */

/* ── N1 IL dispatch wrappers (monolithic notw, native interleaved) ── */

#ifdef FFT_RADIX25_DISPATCH_H
static void vfft_n1_il_dispatch_r25_fwd(
    const double *in, double *out, size_t K)
{
    radix25_n1_forward_il(K, in, out);
}
static void vfft_n1_il_dispatch_r25_bwd(
    const double *in, double *out, size_t K)
{
    radix25_n1_backward_il(K, in, out);
}
#endif

#ifdef FFT_RADIX64_DISPATCH_H
static void vfft_n1_il_dispatch_r64_fwd(
    const double *in, double *out, size_t K)
{
    radix64_n1_forward_il(K, in, out);
}
static void vfft_n1_il_dispatch_r64_bwd(
    const double *in, double *out, size_t K)
{
    radix64_n1_backward_il(K, in, out);
}
#endif

/* R=32 N1 IL: mono interleaved notw */
#ifdef FFT_RADIX32_AVX512_N1_MONO_IL_H
static void vfft_n1_il_dispatch_r32_fwd(
    const double *in, double *out, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 4 && (K & 3) == 0)
    {
        radix32_n1_dit_kernel_fwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix32_n1_dit_kernel_fwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)K;
}
static void vfft_n1_il_dispatch_r32_bwd(
    const double *in, double *out, size_t K)
{
#if defined(__AVX512F__)
    if (K >= 4 && (K & 3) == 0)
    {
        radix32_n1_dit_kernel_bwd_il_avx512(in, out, K);
        return;
    }
#endif
#ifdef __AVX2__
    if (K >= 2 && (K & 1) == 0)
    {
        radix32_n1_dit_kernel_bwd_il_avx2(in, out, K);
        return;
    }
#endif
    (void)in;
    (void)out;
    (void)K;
}
#endif

#ifdef FFT_RADIX128_N1_H
static void vfft_dispatch_r128_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    fft_radix128_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r128_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    fft_radix128_n1_backward(K, ri, ii, ro, io);
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * REGISTER ALL CODELETS
 *
 * Starts with naive fallbacks, then overrides with optimized
 * versions for every radix that has a compiled codelet.
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_register_all(vfft_codelet_registry *reg)
{
    /* Start with naive fallbacks for all radices */
    vfft_registry_init_naive(reg);

    /* Override with optimized versions */

    /* ── New standalone modules ── */
#ifdef FFT_RADIX2_DISPATCH_H
    vfft_registry_set(reg, 2, vfft_dispatch_r2_fwd, vfft_dispatch_r2_bwd);
#endif
#ifdef FFT_RADIX3_DISPATCH_H
    vfft_registry_set(reg, 3, vfft_dispatch_r3_fwd, vfft_dispatch_r3_bwd);
#endif
#ifdef FFT_RADIX4_DISPATCH_H
    vfft_registry_set(reg, 4, vfft_dispatch_r4_fwd, vfft_dispatch_r4_bwd);
#endif
#ifdef FFT_RADIX5_DISPATCH_H
    vfft_registry_set(reg, 5, vfft_dispatch_r5_fwd, vfft_dispatch_r5_bwd);
#endif
#ifdef FFT_RADIX7_DISPATCH_H
    vfft_registry_set(reg, 7, vfft_dispatch_r7_fwd, vfft_dispatch_r7_bwd);
#endif
#ifdef FFT_RADIX8_DISPATCH_H
    vfft_registry_set(reg, 8, vfft_dispatch_r8_fwd, vfft_dispatch_r8_bwd);
#endif
#ifdef FFT_RADIX16_DISPATCH_H
    vfft_registry_set(reg, 16, vfft_dispatch_r16_fwd, vfft_dispatch_r16_bwd);
#endif
#ifdef FFT_RADIX32_DISPATCH_H
    vfft_registry_set(reg, 32, vfft_dispatch_r32_fwd, vfft_dispatch_r32_bwd);
#endif
#ifdef FFT_RADIX25_DISPATCH_H
    vfft_registry_set(reg, 25, vfft_dispatch_r25_fwd, vfft_dispatch_r25_bwd);
#endif
#ifdef FFT_RADIX10_DISPATCH_H
    vfft_registry_set(reg, 10, vfft_dispatch_r10_fwd, vfft_dispatch_r10_bwd);
#endif

    /* ── Fused tw codelets (single-pass twiddle+butterfly) ── */
#ifdef FFT_RADIX2_DISPATCH_H
    vfft_registry_set_tw(reg, 2, vfft_tw_dispatch_r2_fwd, vfft_tw_dispatch_r2_bwd);
#endif
#ifdef FFT_RADIX3_DISPATCH_H
    vfft_registry_set_tw(reg, 3, vfft_tw_dispatch_r3_fwd, vfft_tw_dispatch_r3_bwd);
#endif
#ifdef FFT_RADIX4_DISPATCH_H
    vfft_registry_set_tw(reg, 4, vfft_tw_dispatch_r4_fwd, vfft_tw_dispatch_r4_bwd);
#endif
#ifdef FFT_RADIX5_DISPATCH_H
    vfft_registry_set_tw(reg, 5, vfft_tw_dispatch_r5_fwd, vfft_tw_dispatch_r5_bwd);
#endif
#ifdef FFT_RADIX7_DISPATCH_H
    vfft_registry_set_tw(reg, 7, vfft_tw_dispatch_r7_fwd, NULL);
#endif
#ifdef FFT_RADIX8_DISPATCH_H
    vfft_registry_set_tw(reg, 8, vfft_tw_dispatch_r8_fwd, vfft_tw_dispatch_r8_bwd);
#endif
    /* radix-16 tw: now has scalar fallback */
#ifdef FFT_RADIX16_DISPATCH_H
    vfft_registry_set_tw(reg, 16, vfft_tw_dispatch_r16_fwd, vfft_tw_dispatch_r16_bwd);
#endif
#ifdef FFT_RADIX32_DISPATCH_H
    vfft_registry_set_tw(reg, 32, vfft_tw_dispatch_r32_fwd, vfft_tw_dispatch_r32_bwd);
#endif
#ifdef FFT_RADIX25_DISPATCH_H
    vfft_registry_set_tw(reg, 25, vfft_tw_dispatch_r25_fwd, vfft_tw_dispatch_r25_bwd);
#endif
#ifdef FFT_RADIX10_DISPATCH_H
    vfft_registry_set_tw(reg, 10, vfft_tw_dispatch_r10_fwd, vfft_tw_dispatch_r10_bwd);
#endif

    /* ── DIF tw codelets (twiddle after butterfly) ── */
#ifdef FFT_RADIX2_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 2, vfft_tw_dif_dispatch_r2_fwd, vfft_tw_dif_dispatch_r2_bwd);
#endif
#ifdef FFT_RADIX3_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 3, vfft_tw_dif_dispatch_r3_fwd, vfft_tw_dif_dispatch_r3_bwd);
#endif
#ifdef FFT_RADIX4_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 4, vfft_tw_dif_dispatch_r4_fwd, vfft_tw_dif_dispatch_r4_bwd);
#endif
#ifdef FFT_RADIX5_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 5, vfft_tw_dif_dispatch_r5_fwd, vfft_tw_dif_dispatch_r5_bwd);
#endif
#ifdef FFT_RADIX7_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 7, NULL, vfft_tw_dif_dispatch_r7_bwd);
#endif
#ifdef FFT_RADIX8_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 8, vfft_tw_dif_dispatch_r8_fwd, vfft_tw_dif_dispatch_r8_bwd);
#endif
#ifdef FFT_RADIX32_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 32, vfft_tw_dif_dispatch_r32_fwd, vfft_tw_dif_dispatch_r32_bwd);
#endif
#ifdef FFT_RADIX16_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 16, vfft_tw_dif_dispatch_r16_fwd, vfft_tw_dif_dispatch_r16_bwd);
#endif
#ifdef FFT_RADIX25_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 25, vfft_tw_dif_dispatch_r25_fwd, vfft_tw_dif_dispatch_r25_bwd);
#endif

    /* ── Interleaved (IL) tw codelets ── */
    /* Crossover K: use IL when K >= this value.
     * Defaults from container benchmarks; calibrator will tune per-platform. */
#ifdef FFT_RADIX25_AVX2_IL_TW_H
    vfft_registry_set_tw_il(reg, 25,
                            vfft_tw_il_dispatch_r25_fwd,
#ifdef FFT_RADIX25_AVX2_IL_DIF_TW_H
                            vfft_tw_il_dif_dispatch_r25_bwd,
#else
                            NULL,
#endif
                            256); /* crossover K=256: IL wins at K>=256 for R=25 */
#endif
#ifdef FFT_RADIX25_DISPATCH_H
    vfft_registry_set_n1_il(reg, 25, vfft_n1_il_dispatch_r25_fwd, vfft_n1_il_dispatch_r25_bwd);
#endif

#ifdef FFT_RADIX32_AVX2_TW_IL_H
    vfft_registry_set_tw_il(reg, 32,
                            vfft_tw_il_dispatch_r32_fwd,
#ifdef FFT_RADIX32_AVX2_DIF_TW_IL_H
                            vfft_tw_il_dif_dispatch_r32_bwd,
#else
                            NULL,
#endif
                            256); /* crossover K=256: R=32 IL */
#endif
#ifdef FFT_RADIX32_AVX512_N1_MONO_IL_H
    vfft_registry_set_n1_il(reg, 32, vfft_n1_il_dispatch_r32_fwd, vfft_n1_il_dispatch_r32_bwd);
#endif

#ifdef FFT_RADIX5_DISPATCH_H
    vfft_registry_set_tw_il(reg, 5,
                            vfft_tw_il_dispatch_r5_fwd,
                            vfft_tw_il_dif_dispatch_r5_bwd,
                            256); /* crossover K=256: R=5 has 10 split streams */
#endif

#ifdef FFT_RADIX7_DISPATCH_H
    vfft_registry_set_tw_il(reg, 7,
                            vfft_tw_il_dispatch_r7_fwd,
                            vfft_tw_il_dif_dispatch_r7_bwd,
                            256); /* crossover K=256: R=7 has 14 split streams */
#endif

#ifdef FFT_RADIX8_DISPATCH_H
    vfft_registry_set_tw_il(reg, 8,
                            vfft_tw_il_dispatch_r8_fwd,
                            vfft_tw_il_dif_dispatch_r8_bwd,
                            512); /* crossover K=512: R=8 has 16 split streams */
#endif

#ifdef FFT_RADIX10_DISPATCH_H
    vfft_registry_set_tw_il(reg, 10,
                            vfft_tw_il_dispatch_r10_fwd,
                            vfft_tw_il_dif_dispatch_r10_bwd,
                            512); /* crossover K=512: R=10 has 20 split streams */
#endif

#ifdef FFT_RADIX10_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 10, vfft_tw_dif_dispatch_r10_fwd, vfft_tw_dif_dispatch_r10_bwd);
#endif

    /* ── Genfft prime modules ── */
#ifdef FFT_RADIX11_GENFFT_H
    vfft_registry_set(reg, 11, vfft_dispatch_r11_fwd, vfft_dispatch_r11_bwd);
    vfft_registry_set_tw(reg, 11, vfft_tw_dispatch_r11_fwd, NULL);
    vfft_registry_set_tw_dif(reg, 11, NULL, vfft_tw_dif_dispatch_r11_bwd);
#endif
#ifdef FFT_RADIX13_GENFFT_H
    vfft_registry_set(reg, 13, vfft_dispatch_r13_fwd, vfft_dispatch_r13_bwd);
    vfft_registry_set_tw(reg, 13, vfft_tw_dispatch_r13_fwd, NULL);
    vfft_registry_set_tw_dif(reg, 13, NULL, vfft_tw_dif_dispatch_r13_bwd);
#endif
#ifdef FFT_RADIX17_GENFFT_H
    vfft_registry_set(reg, 17, vfft_dispatch_r17_fwd, vfft_dispatch_r17_bwd);
#endif
#ifdef FFT_RADIX19_GENFFT_H
    vfft_registry_set(reg, 19, vfft_dispatch_r19_fwd, vfft_dispatch_r19_bwd);
#endif
#ifdef FFT_RADIX23_GENFFT_H
    vfft_registry_set(reg, 23, vfft_dispatch_r23_fwd, vfft_dispatch_r23_bwd);
#endif

    /* ── N1 large codelets (K-first signature → adapted) ── */
#ifdef FFT_RADIX64_DISPATCH_H
    vfft_registry_set(reg, 64, vfft_dispatch_r64_fwd, vfft_dispatch_r64_bwd);
    vfft_registry_set_tw(reg, 64, vfft_tw_dispatch_r64_fwd, vfft_tw_dispatch_r64_bwd);
    vfft_registry_set_tw_dif(reg, 64, vfft_tw_dif_dispatch_r64_fwd, vfft_tw_dif_dispatch_r64_bwd);
    vfft_registry_set_tw_il(reg, 64,
                            vfft_tw_il_dispatch_r64_fwd,
                            vfft_tw_il_dif_dispatch_r64_bwd,
                            512); /* crossover K=512: R=64 has 128 split streams */
    vfft_registry_set_n1_il(reg, 64, vfft_n1_il_dispatch_r64_fwd, vfft_n1_il_dispatch_r64_bwd);
#endif
#ifdef FFT_RADIX128_N1_H
    vfft_registry_set(reg, 128, vfft_dispatch_r128_fwd, vfft_dispatch_r128_bwd);
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * ISA REPORT (debug)
 * ═══════════════════════════════════════════════════════════════ */

static void vfft_print_registry(const vfft_codelet_registry *reg)
{
    const char *isa_name[] = {"scalar", "AVX2", "AVX-512"};
    printf("  ISA: %s\n", isa_name[vfft_detect_isa()]);
    printf("  Registered codelets:\n");

    size_t radixes[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 19, 23, 25, 32, 64, 128};
    for (size_t i = 0; i < sizeof(radixes) / sizeof(radixes[0]); i++)
    {
        size_t r = radixes[i];
        if (reg->fwd[r])
        {
            const char *kind = "naive";

            /* Detect optimized codelets by checking if function pointer
             * differs from the naive fallback */
            switch (r)
            {
#ifdef FFT_RADIX2_DISPATCH_H
            case 2:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX3_DISPATCH_H
            case 3:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX4_DISPATCH_H
            case 4:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX5_DISPATCH_H
            case 5:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX7_DISPATCH_H
            case 7:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX8_DISPATCH_H
            case 8:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX16_DISPATCH_H
            case 16:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX11_GENFFT_H
            case 11:
                kind = "genfft";
                break;
#endif
#ifdef FFT_RADIX13_GENFFT_H
            case 13:
                kind = "genfft";
                break;
#endif
#ifdef FFT_RADIX17_GENFFT_H
            case 17:
                kind = "genfft";
                break;
#endif
#ifdef FFT_RADIX19_GENFFT_H
            case 19:
                kind = "genfft";
                break;
#endif
#ifdef FFT_RADIX23_GENFFT_H
            case 23:
                kind = "genfft";
                break;
#endif
#ifdef FFT_RADIX32_DISPATCH_H
            case 32:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX25_DISPATCH_H
            case 25:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX10_DISPATCH_H
            case 10:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX64_DISPATCH_H
            case 64:
                kind = "optimized";
                break;
#endif
#ifdef FFT_RADIX128_N1_H
            case 128:
                kind = "N1-gen";
                break;
#endif
            default:
                break;
            }

            printf("    R=%-3zu  fwd=%s  bwd=%s  tw=%s  dif=%s  il=%s  %s%s\n", r,
                   reg->fwd[r] ? "yes" : "no ",
                   reg->bwd[r] ? "yes" : "no ",
                   reg->tw_fwd[r] ? "yes" : "no ",
                   reg->tw_dif_bwd[r] ? "yes" : "no ",
                   reg->tw_fwd_il[r] ? "yes" : "no ",
                   kind,
                   reg->tw_fwd_il[r] ? "  [IL]" : "");
            if (reg->il_crossover_K[r] > 0)
            {
                printf("           IL crossover K=%zu\n", reg->il_crossover_K[r]);
            }
        }
    }
}

#undef IF_AVX512
#undef IF_AVX2

#endif /* VFFT_REGISTER_CODELETS_H */