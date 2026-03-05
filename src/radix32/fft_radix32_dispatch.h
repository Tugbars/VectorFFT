/**
 * @file fft_radix32_dispatch.h
 * @brief Radix-32 cross-ISA dispatch — runtime CPU detection
 *
 * ═══════════════════════════════════════════════════════════════════
 * OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Top-level entry points for radix-32 FFT codelets. Detects CPU ISA
 * at runtime (cached on first call) and routes to the best backend:
 *
 *   AVX-512  →  K % 8 == 0, K ≥ 8   (8-wide doubles, 32 ZMM regs)
 *   AVX2     →  K % 4 == 0, K ≥ 4   (4-wide doubles, 16 YMM regs)
 *   Scalar   →  any K ≥ 1            (portable fallback)
 *
 * Two codelet families:
 *
 *   N1 (twiddle-less): Bottom-of-recursion DFT-32, zero memory twiddles.
 *     radix32_n1_forward(K, in_re, in_im, out_re, out_im)
 *     radix32_n1_backward(K, in_re, in_im, out_re, out_im)
 *
 *   Twiddled (packed): Fused twiddle + DFT-32 on contiguous packed data.
 *     radix32_tw_packed_forward(K, in_re, in_im, out_re, out_im, tw_re, tw_im)
 *     radix32_tw_packed_backward(K, in_re, in_im, out_re, out_im, tw_re, tw_im)
 *
 *   Twiddled (strided): Fallback for data arriving in stride-K layout.
 *     radix32_tw_strided_forward(K, in_re, in_im, out_re, out_im, tw_re, tw_im)
 *     radix32_tw_strided_backward(K, in_re, in_im, out_re, out_im, tw_re, tw_im)
 *
 * ═══════════════════════════════════════════════════════════════════
 * INCLUDE CHAIN
 * ═══════════════════════════════════════════════════════════════════
 *
 * scalar/fft_radix32_scalar_gen.h        (N1 + tw, always included)
 * avx2/fft_radix32_avx2_n1.h            (N1, if __AVX2__)
 * avx2/fft_radix32_avx2_tw_unified.h    (tw dispatch, if __AVX2__)
 * avx512/fft_radix32_avx512_n1_u2.h     (N1 U1+U2, if __AVX512F__)
 * avx512/fft_radix32_avx512_tw_unified.h (tw dispatch, if __AVX512F__)
 */

#ifndef FFT_RADIX32_DISPATCH_H
#define FFT_RADIX32_DISPATCH_H

#include <stddef.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════
 * BACKEND INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

/* Scalar: always available */
#include "fft_radix32_scalar_gen.h"

/* AVX2 */
#ifdef __AVX2__
#undef  R32AN_LD
#undef  R32AN_ST
#define R32AN_LD(p)   _mm256_load_pd(p)
#define R32AN_ST(p,v) _mm256_store_pd((p),(v))
#include "fft_radix32_avx2_n1.h"
#include "fft_radix32_avx2_tw_unified.h"
#endif

/* AVX-512 */
#ifdef __AVX512F__
#include "fft_radix32_avx512_n1_u2.h"
#include "fft_radix32_avx512_tw_unified.h"
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA LEVEL + DETECTION
 *
 * Shared with radix-64/128 via include guards.
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
 * EFFECTIVE ISA FOR RADIX-32
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
 * N1 DISPATCH (twiddle-less DFT-32)
 *
 * Signature: (K, in_re, in_im, out_re, out_im)
 * Data layout: stride-K — in_re[n*K + k], n=0..31
 *
 * AVX-512: Uses U=2 for K≥16, U=1 otherwise.
 * AVX2:    U=1 only (16 YMM regs).
 * Scalar:  k-step=1 loop.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_n1_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        if (K >= 16)
            radix32_n1_dit_kernel_fwd_avx512_u2(in_re, in_im, out_re, out_im, K);
        else
            radix32_n1_dit_kernel_fwd_avx512_u1(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_n1_dit_kernel_fwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix32_n1_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

static inline void radix32_n1_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        if (K >= 16)
            radix32_n1_dit_kernel_bwd_avx512_u2(in_re, in_im, out_re, out_im, K);
        else
            radix32_n1_dit_kernel_bwd_avx512_u1(in_re, in_im, out_re, out_im, K);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_n1_dit_kernel_bwd_avx2(in_re, in_im, out_re, out_im, K);
        return;
#endif
    default:
        radix32_n1_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — PACKED LAYOUT (production path)
 *
 * Data and twiddles in contiguous packed blocks.
 * The planner keeps data packed between stages; this is the
 * hot path in the full FFT pipeline.
 *
 * Signature: (K, in_re, in_im, out_re, out_im, tw_re, tw_im)
 *
 * Data layout:   in_re[block*32*T + n*T + j]
 * Twiddle layout: tw_re[block*31*T + (n-1)*T + j]
 * Block size T chosen per-ISA (AVX-512: 32, AVX2: 16, Scalar: 1).
 *
 * The per-ISA unified headers handle T selection internally.
 * Scalar falls back to flat strided (k-step=1, no packing benefit).
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_packed_forward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix32_tw_packed_dispatch_fwd(K, in_re, in_im, out_re, out_im,
                                       tw_re, tw_im);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_packed_dispatch_fwd_avx2(K, in_re, in_im, out_re, out_im,
                                            tw_re, tw_im);
        return;
#endif
    default:
        /* Scalar: no packing benefit, treat packed data as flat with K=T per block.
         * The packed twiddle layout at T=1 is identical to flat layout. */
        radix32_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

static inline void radix32_tw_packed_backward(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        radix32_tw_packed_dispatch_bwd(K, in_re, in_im, out_re, out_im,
                                       tw_re, tw_im);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_packed_dispatch_bwd_avx2(K, in_re, in_im, out_re, out_im,
                                            tw_re, tw_im);
        return;
#endif
    default:
        radix32_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              tw_re, tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * TWIDDLED DISPATCH — STRIDED LAYOUT (fallback)
 *
 * Data in stride-K layout: in_re[n*K + k].
 * Used when data arrives from user API before packing, or for
 * K values not aligned to the SIMD width.
 *
 * AVX-512: flat/ladder/NT dispatch (via tw_unified.h)
 * AVX2:    flat only
 * Scalar:  flat
 *
 * For AVX-512, caller must also provide ladder twiddle tables
 * (base_tw_re[5*K], base_tw_im[5*K]) for K ≥ 128.
 * For simplicity, this dispatch uses only flat twiddles — the
 * per-ISA unified headers handle the internal ladder switch.
 * ═══════════════════════════════════════════════════════════════ */

static inline void radix32_tw_strided_forward(
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
        radix32_tw_strided_dispatch_fwd(K, in_re, in_im, out_re, out_im,
                                        flat_tw_re, flat_tw_im,
                                        base_tw_re, base_tw_im);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_strided_dispatch_fwd_avx2(K, in_re, in_im, out_re, out_im,
                                             flat_tw_re, flat_tw_im);
        return;
#endif
    default:
        radix32_tw_flat_dit_kernel_fwd_scalar(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
}

static inline void radix32_tw_strided_backward(
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
        radix32_tw_strided_dispatch_bwd(K, in_re, in_im, out_re, out_im,
                                        flat_tw_re, flat_tw_im,
                                        base_tw_re, base_tw_im);
        return;
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        radix32_tw_strided_dispatch_bwd_avx2(K, in_re, in_im, out_re, out_im,
                                             flat_tw_re, flat_tw_im);
        return;
#endif
    default:
        radix32_tw_flat_dit_kernel_bwd_scalar(in_re, in_im, out_re, out_im,
                                              flat_tw_re, flat_tw_im, K);
        return;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 *
 * Optimal packed block size T depends on ISA:
 *   AVX-512: T = min(K, 32), clamped to {8, 16, 32}
 *   AVX2:    T = min(K, 16), clamped to {4, 8, 16}
 *   Scalar:  T = K (no packing, flat layout)
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t radix32_packed_optimal_T(size_t K)
{
    switch (radix32_effective_isa(K)) {
#ifdef __AVX512F__
    case ISA_AVX512:
        return r32_packed_optimal_T(K);   /* from avx512 tw_unified */
#endif
#ifdef __AVX2__
    case ISA_AVX2:
        return r32a_packed_optimal_T(K);  /* from avx2 tw_unified */
#endif
    default:
        return K;  /* scalar: no packing */
    }
}

/** Flat twiddle table size in doubles (per re/im component). */
static inline size_t radix32_flat_tw_size(size_t K) { return 31 * K; }

/** Data buffer size in doubles (per re/im component). */
static inline size_t radix32_data_size(size_t K) { return 32 * K; }

#endif /* FFT_RADIX32_DISPATCH_H */
