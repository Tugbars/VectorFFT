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
        vfft_isa_level_t isa = vfft_detect_isa();                             \
        (void)isa;                                                            \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,K); return; })  \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,K); return; })      \
        prefix##_fwd_scalar(ri, ii, ro, io, K);                               \
    }                                                                         \
    static void vfft_dispatch_r##R##_bwd(                                     \
        const double *ri, const double *ii, double *ro, double *io, size_t K) \
    {                                                                         \
        vfft_isa_level_t isa = vfft_detect_isa();                             \
        (void)isa;                                                            \
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
        vfft_isa_level_t isa = vfft_detect_isa();                            \
        (void)isa;                                                           \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_fwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }                                                                        \
    static void vfft_tw_dispatch_r##R##_bwd(                                 \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        vfft_isa_level_t isa = vfft_detect_isa();                            \
        (void)isa;                                                           \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_bwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_bwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_bwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }

/* ── Twiddled wrappers for radices with full ISA coverage ── */

#ifdef FFT_RADIX2_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(2, radix2_tw_dit_kernel)
#endif

#ifdef FFT_RADIX3_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(3, radix3_tw_dit_kernel)
#endif

#ifdef FFT_RADIX4_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(4, radix4_tw_dit_kernel)
#endif

#ifdef FFT_RADIX5_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(5, radix5_tw_dit_kernel)
#endif

#ifdef FFT_RADIX7_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(7, radix7_tw_dit_kernel)
#endif

#ifdef FFT_RADIX8_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(8, radix8_tw_dit_kernel)
#endif

#ifdef FFT_RADIX32_DISPATCH_H
VFFT_TW_DISPATCH_WRAPPER(32, radix32_tw_flat_dit_kernel)
#endif

/* ── Radix-16 tw: no scalar tw kernel, SIMD-only ──
 * Only dispatches for K aligned to SIMD width. The planner will
 * fall back to notw+twiddle for unaligned K (rare in practice —
 * K at a radix-16 stage is always a product of inner radices). */

#ifdef FFT_RADIX16_DISPATCH_H
static void vfft_tw_dispatch_r16_fwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    vfft_isa_level_t isa = vfft_detect_isa();
    (void)isa;
    IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { radix16_tw_flat_dit_kernel_fwd_avx512(ri,ii,ro,io,twr,twi,K); return; })
    IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { radix16_tw_flat_dit_kernel_fwd_avx2(ri,ii,ro,io,twr,twi,K); return; })
    /* No scalar tw for r16 — planner's separate twiddle fallback handles this */
}
static void vfft_tw_dispatch_r16_bwd(
    const double *ri, const double *ii, double *ro, double *io,
    const double *twr, const double *twi, size_t K)
{
    vfft_isa_level_t isa = vfft_detect_isa();
    (void)isa;
    IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { radix16_tw_flat_dit_kernel_bwd_avx512(ri,ii,ro,io,twr,twi,K); return; })
    IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { radix16_tw_flat_dit_kernel_bwd_avx2(ri,ii,ro,io,twr,twi,K); return; })
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
        vfft_isa_level_t isa = vfft_detect_isa();                            \
        (void)isa;                                                           \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_fwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_fwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_fwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }                                                                        \
    static void vfft_tw_dif_dispatch_r##R##_bwd(                             \
        const double *ri, const double *ii, double *ro, double *io,          \
        const double *twr, const double *twi, size_t K)                      \
    {                                                                        \
        vfft_isa_level_t isa = vfft_detect_isa();                            \
        (void)isa;                                                           \
        IF_AVX512(if (isa == VFFT_ISA_AVX512 && K >= 8 && (K & 7) == 0) { prefix##_bwd_avx512(ri,ii,ro,io,twr,twi,K); return; }) \
        IF_AVX2(if (isa >= VFFT_ISA_AVX2 && K >= 4 && (K & 3) == 0) { prefix##_bwd_avx2(ri,ii,ro,io,twr,twi,K); return; })     \
        prefix##_bwd_scalar(ri, ii, ro, io, twr, twi, K);                    \
    }

/* ── Small radix DIF codelets ── */

#ifdef FFT_RADIX2_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(2, radix2_tw_dif_kernel)
#endif

#ifdef FFT_RADIX3_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(3, radix3_tw_dif_kernel)
#endif

#ifdef FFT_RADIX4_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(4, radix4_tw_dif_kernel)
#endif

#ifdef FFT_RADIX5_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(5, radix5_tw_dif_kernel)
#endif

#ifdef FFT_RADIX7_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(7, radix7_tw_dif_kernel)
#endif

#ifdef FFT_RADIX8_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(8, radix8_tw_dif_kernel)
#endif

/* ── Radix-32 DIF (fused single-pass, different kernel prefix) ── */

#ifdef FFT_RADIX32_DIF_DISPATCH_H
VFFT_TW_DIF_DISPATCH_WRAPPER(32, radix32_tw_flat_dif_kernel)
#endif

/* ── N1 (notw) dispatch wrappers ── */

/* ── New standalone modules ── */

#ifdef FFT_RADIX2_DISPATCH_H
VFFT_DISPATCH_WRAPPER(2, radix2_notw_dit_kernel)
#endif

#ifdef FFT_RADIX3_DISPATCH_H
VFFT_DISPATCH_WRAPPER(3, radix3_notw_dit_kernel)
#endif

#ifdef FFT_RADIX4_DISPATCH_H
VFFT_DISPATCH_WRAPPER(4, radix4_notw_dit_kernel)
#endif

#ifdef FFT_RADIX5_DISPATCH_H
VFFT_DISPATCH_WRAPPER(5, radix5_notw_dit_kernel)
#endif

#ifdef FFT_RADIX7_DISPATCH_H
VFFT_DISPATCH_WRAPPER(7, radix7_notw_dit_kernel)
#endif

#ifdef FFT_RADIX8_DISPATCH_H
VFFT_DISPATCH_WRAPPER(8, radix8_notw_dit_kernel)
#endif

#ifdef FFT_RADIX16_DISPATCH_H
VFFT_DISPATCH_WRAPPER(16, radix16_n1_dit_kernel)
#endif

#ifdef FFT_RADIX32_DISPATCH_H
VFFT_DISPATCH_WRAPPER(32, radix32_notw_dit_kernel)
#endif

/* ── Genfft prime modules ── */

#ifdef FFT_RADIX11_GENFFT_H
VFFT_DISPATCH_WRAPPER(11, radix11_genfft)
#endif
#ifdef FFT_RADIX13_GENFFT_H
VFFT_DISPATCH_WRAPPER(13, radix13_genfft)
#endif
#ifdef FFT_RADIX17_GENFFT_H
VFFT_DISPATCH_WRAPPER(17, radix17_genfft)
#endif
#ifdef FFT_RADIX19_GENFFT_H
VFFT_DISPATCH_WRAPPER(19, radix19_genfft)
#endif
#ifdef FFT_RADIX23_GENFFT_H
VFFT_DISPATCH_WRAPPER(23, radix23_genfft)
#endif

/* ── N1-64 / N1-128 adapters ──
 *
 * These have K-first signature: func(K, in_re, in_im, out_re, out_im)
 * The planner needs K-last:     func(in_re, in_im, out_re, out_im, K)
 * Thin wrappers swap the argument order.
 */

#ifdef FFT_RADIX64_N1_H
static void vfft_dispatch_r64_fwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    fft_radix64_n1_forward(K, ri, ii, ro, io);
}
static void vfft_dispatch_r64_bwd(
    const double *ri, const double *ii, double *ro, double *io, size_t K)
{
    fft_radix64_n1_backward(K, ri, ii, ro, io);
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
    vfft_registry_set_tw(reg, 7, vfft_tw_dispatch_r7_fwd, vfft_tw_dispatch_r7_bwd);
#endif
#ifdef FFT_RADIX8_DISPATCH_H
    vfft_registry_set_tw(reg, 8, vfft_tw_dispatch_r8_fwd, vfft_tw_dispatch_r8_bwd);
#endif
    /* radix-16 tw: SIMD-only (no scalar tw kernel).
     * Safe: K at a r16 outer stage = product of inner radices,
     * always SIMD-aligned in practice. If K is unaligned, the
     * wrapper is a no-op and planner already wrote output via
     * the notw+twiddle fallback... BUT that fallback is never
     * reached if tw_fwd is non-NULL. So leave unregistered. */
    /* #ifdef FFT_RADIX16_DISPATCH_H
    vfft_registry_set_tw(reg, 16, vfft_tw_dispatch_r16_fwd, vfft_tw_dispatch_r16_bwd);
    #endif */
#ifdef FFT_RADIX32_DISPATCH_H
    vfft_registry_set_tw(reg, 32, vfft_tw_dispatch_r32_fwd, vfft_tw_dispatch_r32_bwd);
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
    vfft_registry_set_tw_dif(reg, 7, vfft_tw_dif_dispatch_r7_fwd, vfft_tw_dif_dispatch_r7_bwd);
#endif
#ifdef FFT_RADIX8_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 8, vfft_tw_dif_dispatch_r8_fwd, vfft_tw_dif_dispatch_r8_bwd);
#endif
#ifdef FFT_RADIX32_DIF_DISPATCH_H
    vfft_registry_set_tw_dif(reg, 32, vfft_tw_dif_dispatch_r32_fwd, vfft_tw_dif_dispatch_r32_bwd);
#endif

    /* ── Genfft prime modules ── */
#ifdef FFT_RADIX11_GENFFT_H
    vfft_registry_set(reg, 11, vfft_dispatch_r11_fwd, vfft_dispatch_r11_bwd);
#endif
#ifdef FFT_RADIX13_GENFFT_H
    vfft_registry_set(reg, 13, vfft_dispatch_r13_fwd, vfft_dispatch_r13_bwd);
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
#ifdef FFT_RADIX64_N1_H
    vfft_registry_set(reg, 64, vfft_dispatch_r64_fwd, vfft_dispatch_r64_bwd);
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

    size_t radixes[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 19, 23, 32, 64, 128};
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
#ifdef FFT_RADIX64_N1_H
            case 64:
                kind = "N1-gen";
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

            printf("    R=%-3zu  fwd=%s  bwd=%s  tw=%s  dif=%s  %s\n", r,
                   reg->fwd[r] ? "yes" : "no ",
                   reg->bwd[r] ? "yes" : "no ",
                   reg->tw_fwd[r] ? "yes" : "no ",
                   reg->tw_dif_bwd[r] ? "yes" : "no ",
                   kind);
        }
    }
}

#undef IF_AVX512
#undef IF_AVX2

#endif /* VFFT_REGISTER_CODELETS_H */