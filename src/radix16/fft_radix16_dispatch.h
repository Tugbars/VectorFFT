/**
 * @file fft_radix16_dispatch.h
 * @brief Radix-16 cross-ISA dispatch — strided, packed, and pack+walk paths
 *
 * ═══════════════════════════════════════════════════════════════════
 * THREE DISPATCH MODES
 * ═══════════════════════════════════════════════════════════════════
 *
 * 1. STRIDED (default):
 *      radix16_tw_strided_forward / radix16_tw_strided_backward
 *      radix16_notw_forward / radix16_notw_backward  (alias: radix16_n1_*)
 *    Data in stride-K layout. Flat twiddles only.
 *
 * 2. PACKED (planner, K <= RADIX16_WALK_THRESHOLD):
 *      radix16_tw_packed_forward / radix16_tw_packed_backward
 *    Data in packed contiguous blocks + pre-packed twiddle table.
 *
 * 3. PACK+WALK (planner, K > RADIX16_WALK_THRESHOLD):
 *      radix16_tw_pack_walk_fwd_avx512 / _bwd_avx512
 *    Data in packed blocks, twiddles derived on-the-fly from
 *    4 walking base accumulators (W^1,W^2,W^4,W^8).
 *    10 cmuls per block to derive 15 twiddles. Zero twiddle table.
 *
 * The planner should call radix16_tw_packed_auto_fwd/_bwd which
 * automatically selects packed-table vs pack+walk based on K.
 *
 * ═══════════════════════════════════════════════════════════════════
 * ISA ROUTING
 * ═══════════════════════════════════════════════════════════════════
 *
 * AVX-512: K%8==0, K>=8  (flat tw + pack+walk)
 * AVX2:    K%4==0, K>=4  (flat tw, packed table)
 * Scalar:  any K>=1      (flat tw)
 *
 * ═══════════════════════════════════════════════════════════════════
 * PLANNER USAGE
 * ═══════════════════════════════════════════════════════════════════
 *
 *   size_t T = radix16_packed_optimal_T(K);
 *   radix16_walk_plan_t walk_plan, *wp = NULL;
 *   double *pk_twr = NULL, *pk_twi = NULL;
 *
 *   if (radix16_should_walk(K)) {
 *       radix16_walk_plan_init(&walk_plan, K);
 *       wp = &walk_plan;
 *   } else {
 *       pk_twr = alloc(15*K); pk_twi = alloc(15*K);
 *       radix16_pack_twiddles_avx512(flat_twr, flat_twi, pk_twr, pk_twi, K);
 *   }
 *
 *   radix16_tw_packed_auto_fwd(pk_in_re, pk_in_im, pk_out_re, pk_out_im,
 *                              pk_twr, pk_twi, wp, K, T);
 */

#ifndef FFT_RADIX16_DISPATCH_H
#define FFT_RADIX16_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════
 * BACKEND INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar: notw + DIT tw + DIF tw (always available) */
#include "scalar/fft_radix16_scalar_n1_gen.h"
#include "scalar/fft_radix16_scalar_tw.h"

/* AVX2 */
#ifdef __AVX2__
#include "avx2/fft_radix16_avx2_n1_gen.h"
#include "avx2/fft_radix16_avx2_tw.h"
#endif

/* AVX-512 */
#ifdef __AVX512F__
#ifndef TARGET_AVX512
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#endif
#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_64
#define ALIGNAS_64 __attribute__((aligned(64)))
#endif
#include "avx512/fft_radix16_avx512_n1_gen.h"
#include "avx512/fft_radix16_avx512_tw.h"
#include "avx512/fft_radix16_avx512_tw_pack_walk.h"
#endif

/* Packed layout drivers (must come after kernel headers) */
#include "fft_radix16_tw_packed.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL + DETECTION
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    ISA_SCALAR = 0,
    ISA_AVX2 = 1,
    ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

#ifndef VFFT_ISA_DETECT_DEFINED
#define VFFT_ISA_DETECT_DEFINED

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#ifdef _MSC_VER
#include <intrin.h>
static inline void vfft_cpuid(int leaf, int sub,
                              uint32_t *a, uint32_t *b,
                              uint32_t *c, uint32_t *d)
{
    int regs[4];
    __cpuidex(regs, leaf, sub);
    *a = (uint32_t)regs[0];
    *b = (uint32_t)regs[1];
    *c = (uint32_t)regs[2];
    *d = (uint32_t)regs[3];
}
static inline uint64_t vfft_xgetbv(uint32_t idx) { return _xgetbv(idx); }
#else
#include <cpuid.h>
static inline void vfft_cpuid(int leaf, int sub,
                              uint32_t *a, uint32_t *b,
                              uint32_t *c, uint32_t *d)
{
    __cpuid_count(leaf, sub, *a, *b, *c, *d);
}
static inline uint64_t vfft_xgetbv(uint32_t idx)
{
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(idx));
    return ((uint64_t)hi << 32) | lo;
}
#endif

static vfft_isa_level_t vfft_detect_isa(void)
{
    static vfft_isa_level_t cached = (vfft_isa_level_t)-1;
    if (cached != (vfft_isa_level_t)-1)
        return cached;

    uint32_t a, b, c, d;
    vfft_cpuid(1, 0, &a, &b, &c, &d);
    if (!((c >> 27) & 1) || !((c >> 12) & 1) || !((c >> 28) & 1))
        return (cached = ISA_SCALAR);

    uint64_t xcr0 = vfft_xgetbv(0);
    if ((xcr0 & 0x06) != 0x06)
        return (cached = ISA_SCALAR);

    vfft_cpuid(7, 0, &a, &b, &c, &d);
    if (!((b >> 5) & 1))
        return (cached = ISA_SCALAR);

    if (((b >> 16) & 1) && ((b >> 17) & 1) && ((b >> 31) & 1) &&
        ((xcr0 & 0xE0) == 0xE0))
        return (cached = ISA_AVX512);

    return (cached = ISA_AVX2);
}

#else /* Non-x86 */
static vfft_isa_level_t vfft_detect_isa(void) { return ISA_SCALAR; }
#endif

#endif /* VFFT_ISA_DETECT_DEFINED */

/* ═══════════════════════════════════════════════════════════════
 * EFFECTIVE ISA FOR RADIX-16
 * ═══════════════════════════════════════════════════════════════ */

static inline vfft_isa_level_t radix16_effective_isa(size_t K)
{
    vfft_isa_level_t hw = vfft_detect_isa();
    if (hw >= ISA_AVX512 && (K & 7) == 0 && K >= 8)
        return ISA_AVX512;
    if (hw >= ISA_AVX2 && (K & 3) == 0 && K >= 4)
        return ISA_AVX2;
    return ISA_SCALAR;
}

static inline const char *radix16_isa_name(vfft_isa_level_t level)
{
    switch (level)
    {
    case ISA_AVX512:
        return "AVX-512";
    case ISA_AVX2:
        return "AVX2";
    case ISA_SCALAR:
        return "Scalar";
    default:
        return "Unknown";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * N1 DISPATCH (twiddle-less DFT-16)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_n1_forward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix16_n1_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix16_n1_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix16_n1_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

static inline void radix16_n1_backward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix16_n1_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix16_n1_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix16_n1_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

/* Planner-compatible aliases */
#define radix16_notw_forward radix16_n1_forward
#define radix16_notw_backward radix16_n1_backward

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — STRIDED LAYOUT
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_strided_forward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ flat_tw_re,
    const double *__restrict__ flat_tw_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix16_tw_flat_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix16_tw_flat_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im,
                                            flat_tw_re, flat_tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
}

static inline void radix16_tw_strided_backward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ flat_tw_re,
    const double *__restrict__ flat_tw_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix16_tw_flat_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix16_tw_flat_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im,
                                            flat_tw_re, flat_tw_im, K);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — TABLE PATH
 *
 * Always flat kernel at K=T — 15*T doubles fits L1 for any T.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_packed_forward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        r16_tw_packed_fwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        r16_tw_packed_fwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K, 4);
        return;
#endif
    default:
        /* Scalar fallback: flat strided at full K */
        radix16_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

static inline void radix16_tw_packed_backward(
    size_t K,
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im)
{
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        r16_tw_packed_bwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        r16_tw_packed_bwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K, 4);
        return;
#endif
    default:
        radix16_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACK+WALK AUTO-DISPATCH (planner — three-tier)
 *
 * K <= RADIX16_WALK_THRESHOLD → packed table
 * K > RADIX16_WALK_THRESHOLD  → pack+walk (zero twiddle table)
 *
 * Data must be in packed layout for both paths.
 * walk_plan: initialized by radix16_walk_plan_init(). NULL for table path.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_packed_auto_fwd(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const void *__restrict__ walk_plan,
    size_t K, size_t T)
{
#ifdef __AVX512F__
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && T == 8)
    {
        radix16_tw_pack_walk_fwd_avx512(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_t *)walk_plan, K);
        return;
    }
#endif
    (void)walk_plan;
    (void)T;
    radix16_tw_packed_forward(K, in_re, in_im, out_re, out_im,
                              tw_re, tw_im);
}

static inline void radix16_tw_packed_auto_bwd(
    const double *__restrict__ in_re,
    const double *__restrict__ in_im,
    double *__restrict__ out_re,
    double *__restrict__ out_im,
    const double *__restrict__ tw_re,
    const double *__restrict__ tw_im,
    const void *__restrict__ walk_plan,
    size_t K, size_t T)
{
#ifdef __AVX512F__
    if (K > RADIX16_WALK_THRESHOLD && walk_plan && T == 8)
    {
        radix16_tw_pack_walk_bwd_avx512(
            in_re, in_im, out_re, out_im,
            (const radix16_walk_plan_t *)walk_plan, K);
        return;
    }
#endif
    (void)walk_plan;
    (void)T;
    radix16_tw_packed_backward(K, in_re, in_im, out_re, out_im,
                               tw_re, tw_im);
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED DISPATCH — NOTW
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_notw_packed_fwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_fwd_avx512(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_fwd_avx2(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
#endif
    default:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_fwd_scalar(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
    }
}

static inline void radix16_notw_packed_bwd(
    const double *__restrict__ in_re, const double *__restrict__ in_im,
    double *__restrict__ out_re, double *__restrict__ out_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t bs = 16 * T;
    switch (radix16_effective_isa(K))
    {
#ifdef __AVX512F__
    case ISA_AVX512:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_bwd_avx512(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_bwd_avx2(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
#endif
    default:
        for (size_t b = 0; b < nb; b++)
            radix16_n1_dit_kernel_bwd_scalar(
                in_re + b * bs, in_im + b * bs,
                out_re + b * bs, out_im + b * bs, T);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix16_flat_tw_size(size_t K) { return 15 * K; }
static inline size_t radix16_data_size(size_t K) { return 16 * K; }

static inline size_t radix16_packed_optimal_T(size_t K)
{
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0)
        return 8;
#endif
#ifdef __AVX2__
    if (K >= 4 && (K & 3) == 0)
        return 4;
#endif
    return 1;
}

/** Returns 1 if pack+walk should be used instead of packed table */
#ifndef RADIX16_WALK_THRESHOLD
#define RADIX16_WALK_THRESHOLD 512
#endif

static inline int radix16_should_walk(size_t K)
{
    return (K > RADIX16_WALK_THRESHOLD) ? 1 : 0;
}

static inline size_t radix16_packed_tw_size(size_t K)
{
    return radix16_should_walk(K) ? 0 : 15 * K;
}

#endif /* FFT_RADIX16_DISPATCH_H */