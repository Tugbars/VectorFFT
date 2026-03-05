/**
 * @file fft_radix16_dispatch.h
 * @brief Radix-16 cross-ISA dispatch — runtime CPU detection
 *
 * ═══════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Top-level entry points for radix-16 FFT codelets. Detects CPU ISA
 * at runtime (cached on first call) and routes to the best backend:
 *
 *   AVX-512  →  K % 8 == 0, K ≥ 8   (8-wide doubles, 32 ZMM regs)
 *   AVX2     →  K % 4 == 0, K ≥ 4   (4-wide doubles, 16 YMM regs)
 *   Scalar   →  any K ≥ 1            (portable fallback)
 *
 * Two codelet families:
 *
 *   N1 (twiddle-less):
 *     radix16_n1_forward / radix16_n1_backward
 *
 *   Twiddled packed (production):
 *     radix16_tw_packed_forward / radix16_tw_packed_backward
 *
 *   Twiddled strided (fallback):
 *     radix16_tw_strided_forward / radix16_tw_strided_backward
 *
 * ═══════════════════════════════════════════════════════════════════
 * INCLUDE CHAIN
 * ═══════════════════════════════════════════════════════════════════
 *
 * scalar/fft_radix16_scalar_n1_gen.h       (N1 scalar, always)
 * avx2/fft_radix16_avx2_n1_gen.h          (N1 AVX2, if __AVX2__)
 * avx2/fft_radix16_avx2_tw.h              (tw AVX2, if __AVX2__)
 * avx512/fft_radix16_avx512_n1_gen.h      (N1 AVX-512, if __AVX512F__)
 * avx512/fft_radix16_avx512_tw.h          (tw AVX-512, if __AVX512F__)
 * fft_radix16_tw_packed.h                  (packed drivers, all ISAs)
 */

#ifndef FFT_RADIX16_DISPATCH_H
#define FFT_RADIX16_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════
 * BACKEND INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar: always available */
#include "fft_radix16_scalar_n1_gen.h"

/* AVX2 */
#ifdef __AVX2__
#include "fft_radix16_avx2_n1_gen.h"
#include "fft_radix16_avx2_tw.h"
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
#include "fft_radix16_avx512_n1_gen.h"
#include "fft_radix16_avx512_tw.h"
#endif

/* Packed layout drivers (must come after kernel headers) */
#include "fft_radix16_tw_packed.h"

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL + DETECTION
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
    switch (level) {
        case ISA_AVX512: return "AVX-512";
        case ISA_AVX2:   return "AVX2";
        case ISA_SCALAR: return "Scalar";
        default:         return "Unknown";
    }
}

/* ═══════════════════════════════════════════════════════════════
 * N1 DISPATCH (twiddle-less DFT-16)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_n1_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix16_effective_isa(K)) {
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
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix16_effective_isa(K)) {
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

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — PACKED LAYOUT (production path)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_packed_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        r16_tw_packed_fwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K, 8);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        r16_tw_packed_fwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K, 4);
        return;
#endif
    default:
        /* Scalar fallback: treat as flat strided */
        /* TODO: scalar twiddled kernel not yet generated */
        return;
    }
}

static inline void radix16_tw_packed_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    switch (radix16_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        r16_tw_packed_bwd_avx512(in_re, in_im, out_re, out_im,
                                 tw_re, tw_im, K, 8);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        r16_tw_packed_bwd_avx2(in_re, in_im, out_re, out_im,
                               tw_re, tw_im, K, 4);
        return;
#endif
    default:
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — STRIDED LAYOUT (fallback)
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix16_tw_strided_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im)
{
    switch (radix16_effective_isa(K)) {
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
        return;
    }
}

static inline void radix16_tw_strided_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im)
{
    switch (radix16_effective_isa(K)) {
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
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix16_packed_optimal_T(size_t K)
{
    return r16_packed_optimal_T(K);
}

static inline size_t radix16_flat_tw_size(size_t K) { return 15 * K; }
static inline size_t radix16_data_size(size_t K)    { return 16 * K; }

#endif /* FFT_RADIX16_DISPATCH_H */