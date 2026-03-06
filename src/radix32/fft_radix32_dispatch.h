/**
 * @file fft_radix32_dispatch.h
 * @brief Radix-32 cross-ISA dispatch — runtime CPU detection
 *
 * Top-level entry points for radix-32 FFT codelets. Detects CPU ISA
 * at runtime (cached on first call) and routes to the best backend.
 *
 *   AVX-512  →  K % 8 == 0, K ≥ 8   (8-wide, 32 ZMM, flat+ladder)
 *   AVX2     →  K % 4 == 0, K ≥ 4   (4-wide, 16 YMM, flat only)
 *   Scalar   →  any K ≥ 1            (portable fallback)
 *
 * Two codelet families:
 *
 *   notw: Twiddle-less DFT-32 (bottom-of-recursion / standalone batch).
 *     radix32_notw_forward(K, in_re, in_im, out_re, out_im)
 *     radix32_notw_backward(K, in_re, in_im, out_re, out_im)
 *
 *   tw: Twiddled DFT-32 (fused inter-stage twiddle + butterfly).
 *     radix32_tw_forward(K, in_re, in_im, out_re, out_im,
 *                        flat_tw_re, flat_tw_im,
 *                        base_tw_re, base_tw_im)
 *     radix32_tw_backward(...)
 *
 * Data layout: split-real stride-K — in_re[n*K + k], n=0..31.
 *
 * Twiddle tables (twiddled codelet only):
 *   Flat:   flat_tw_re[(n-1)*K + k] = cos(2π·n·k / (32·K))   [31*K doubles]
 *   Ladder: base_tw_re[i*K + k] = cos(2π·(2^i)·k / (32·K))   [5*K doubles]
 *     (ladder tables may be NULL for K ≤ 64 where flat is always used)
 *
 * Generated codelet headers (v2):
 *   fft_radix32_avx512_tw_ladder_v2.h  — flat + ladder U1 + ladder U2
 *   fft_radix32_avx512_notw.h          — NFUSE=8
 *   fft_radix32_avx2_tw_v2.h           — flat only, NFUSE=2
 *   fft_radix32_avx2_notw.h            — NFUSE=2
 *   fft_radix32_scalar_tw.h            — flat only, NFUSE=4
 *   fft_radix32_scalar_notw.h          — NFUSE=4
 */

#ifndef FFT_RADIX32_DISPATCH_H
#define FFT_RADIX32_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════
 * BACKEND INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar: always available (no SIMD headers needed) */
#include "fft_radix32_scalar_tw.h"
#include "fft_radix32_scalar_notw.h"

/* AVX2 */
#ifdef __AVX2__
#define R32A_LD(p)    _mm256_load_pd(p)
#define R32A_ST(p,v)  _mm256_store_pd((p),(v))
#define R32NA_LD(p)   _mm256_load_pd(p)
#define R32NA_ST(p,v) _mm256_store_pd((p),(v))
#include "fft_radix32_avx2_tw_v2.h"
#include "fft_radix32_avx2_notw.h"
#endif

/* AVX-512 */
#ifdef __AVX512F__
#define R32L_LD(p)    _mm512_load_pd(p)
#define R32L_ST(p,v)  _mm512_store_pd((p),(v))
#define R32N5_LD(p)   _mm512_load_pd(p)
#define R32N5_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_tw_ladder_v2.h"
#include "fft_radix32_avx512_notw.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL + DETECTION
 *
 * Shared across radix sizes via include guards.
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    ISA_SCALAR = 0,
    ISA_AVX2   = 1,
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
    *a = (uint32_t)regs[0]; *b = (uint32_t)regs[1];
    *c = (uint32_t)regs[2]; *d = (uint32_t)regs[3];
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

    /* Check OSXSAVE + FMA + AVX (leaf 1, ECX bits 27/12/28) */
    vfft_cpuid(1, 0, &a, &b, &c, &d);
    if (!((c >> 27) & 1) || !((c >> 12) & 1) || !((c >> 28) & 1))
        return (cached = ISA_SCALAR);

    /* Check XCR0: SSE state (bit 1) + AVX state (bit 2) */
    uint64_t xcr0 = vfft_xgetbv(0);
    if ((xcr0 & 0x06) != 0x06)
        return (cached = ISA_SCALAR);

    /* Check AVX2 (leaf 7, EBX bit 5) */
    vfft_cpuid(7, 0, &a, &b, &c, &d);
    if (!((b >> 5) & 1))
        return (cached = ISA_SCALAR);

    /* Check AVX-512F (bit 16) + AVX-512DQ (bit 17) + AVX-512VL (bit 31)
     * + XCR0 opmask/ZMM state (bits 5-7) */
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
 * EFFECTIVE ISA FOR A GIVEN K
 *
 *   AVX-512:  K % 8 == 0, K ≥ 8
 *   AVX2:     K % 4 == 0, K ≥ 4
 *   Scalar:   any K ≥ 1
 * ═══════════════════════════════════════════════════════════════ */

static inline vfft_isa_level_t radix32_effective_isa(size_t K)
{
    vfft_isa_level_t hw = vfft_detect_isa();

    if (hw >= ISA_AVX512 && (K & 7) == 0 && K >= 8)
        return ISA_AVX512;
    if (hw >= ISA_AVX2 && (K & 3) == 0 && K >= 4)
        return ISA_AVX2;
    return ISA_SCALAR;
}

static inline const char *radix32_isa_name(vfft_isa_level_t level)
{
    switch (level) {
        case ISA_AVX512: return "AVX-512";
        case ISA_AVX2:   return "AVX2";
        case ISA_SCALAR: return "Scalar";
        default:         return "Unknown";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * NOTW DISPATCH (twiddle-less DFT-32)
 *
 * Pure DFT-32 butterfly with no inter-stage twiddles.
 * Use as first/last stage or standalone batch.
 *
 * Data layout: stride-K — in_re[n*K + k], n=0..31
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_notw_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix32_notw_dit_kernel_fwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_notw_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix32_notw_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

static inline void radix32_notw_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix32_notw_dit_kernel_bwd_avx512(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_notw_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix32_notw_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH (fused twiddle + DFT-32)
 *
 * Inter-stage twiddle multiply followed by DFT-32 butterfly.
 * Data layout: stride-K — in_re[n*K + k], n=0..31
 *
 * AVX-512 strategy:
 *   K ≤ 64:  flat twiddles (31 loads/k-step, table fits L1)
 *   K ≥ 128: binary ladder (5 base loads + derivation chains)
 *   Ladder variant: U1 for K < 16 or K ≥ 128, U2 for 16 ≤ K ≤ 64
 *     (U2 uses shared 32-slot spill buffer, A-complete then B-complete)
 *
 * AVX2:    flat only (ladder needs 28+ regs, only 16 available)
 * Scalar:  flat only
 *
 * base_tw_re/im may be NULL when K ≤ 64 (flat always used).
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    const double * __restrict__ base_tw_re,
    const double * __restrict__ base_tw_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        if (K <= 64) {
            /* Flat: table is 31*K*8 bytes, fits L1 for K ≤ 64 (15.5 KB) */
            radix32_tw_flat_dit_kernel_fwd_avx512(
                in_re, in_im, out_re, out_im,
                flat_tw_re, flat_tw_im, K);
        } else if (K >= 128 && K < 256) {
            /* Ladder U1: 5 base loads, single k-strip per iteration */
            radix32_tw_ladder_dit_kernel_fwd_avx512_u1(
                in_re, in_im, out_re, out_im,
                base_tw_re, base_tw_im, K);
        } else {
            /* Ladder U2: two k-strips, shared spill buffer.
             * Also used for K ≥ 256 where U2's ILP amortizes
             * the larger working set. Falls back to U1 if K < 16. */
            if (K >= 16)
                radix32_tw_ladder_dit_kernel_fwd_avx512_u2(
                    in_re, in_im, out_re, out_im,
                    base_tw_re, base_tw_im, K);
            else
                radix32_tw_ladder_dit_kernel_fwd_avx512_u1(
                    in_re, in_im, out_re, out_im,
                    base_tw_re, base_tw_im, K);
        }
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_flat_dit_kernel_fwd_avx2(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
        return;
#endif
    default:
        radix32_tw_flat_dit_kernel_fwd_scalar(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
        return;
    }
}

static inline void radix32_tw_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    const double * __restrict__ base_tw_re,
    const double * __restrict__ base_tw_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        if (K <= 64) {
            radix32_tw_flat_dit_kernel_bwd_avx512(
                in_re, in_im, out_re, out_im,
                flat_tw_re, flat_tw_im, K);
        } else if (K >= 128 && K < 256) {
            radix32_tw_ladder_dit_kernel_bwd_avx512_u1(
                in_re, in_im, out_re, out_im,
                base_tw_re, base_tw_im, K);
        } else {
            if (K >= 16)
                radix32_tw_ladder_dit_kernel_bwd_avx512_u2(
                    in_re, in_im, out_re, out_im,
                    base_tw_re, base_tw_im, K);
            else
                radix32_tw_ladder_dit_kernel_bwd_avx512_u1(
                    in_re, in_im, out_re, out_im,
                    base_tw_re, base_tw_im, K);
        }
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_flat_dit_kernel_bwd_avx2(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
        return;
#endif
    default:
        radix32_tw_flat_dit_kernel_bwd_scalar(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

/** Flat twiddle table size in doubles (per re/im component). */
static inline size_t radix32_flat_tw_size(size_t K) { return 31 * K; }

/** Ladder base twiddle table size in doubles (per re/im component). */
static inline size_t radix32_ladder_tw_size(size_t K) { return 5 * K; }

/** Data buffer size in doubles (per re/im component). */
static inline size_t radix32_data_size(size_t K) { return 32 * K; }

/** Whether ladder twiddles are needed for this K on current hardware. */
static inline int radix32_needs_ladder(size_t K)
{
    return (radix32_effective_isa(K) == ISA_AVX512) && (K > 64);
}

/** Whether flat twiddles are needed for this K on current hardware. */
static inline int radix32_needs_flat(size_t K)
{
    vfft_isa_level_t isa = radix32_effective_isa(K);
    /* AVX-512 with K > 64 uses ladder exclusively */
    if (isa == ISA_AVX512 && K > 64) return 0;
    /* Everything else uses flat */
    return 1;
}

#endif /* FFT_RADIX32_DISPATCH_H */
