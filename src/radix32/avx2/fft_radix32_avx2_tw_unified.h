/**
 * @file fft_radix32_avx2_tw_unified.h
 * @brief Unified twiddled DFT-32 AVX2 dispatch
 *
 * ═══════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Two data layout paths:
 *
 * 1. PACKED (production — planner keeps data in packed layout)
 *
 *    Layout:  data_re[block*32*T + n*T + j]   n=0..31, j=0..T-1
 *    Twiddle: tw_re[block*31*T + (n-1)*T + j] n=1..31
 *    Every load is a contiguous aligned 32-byte fetch. Zero strides.
 *
 *    Dispatch: T = min(K, 16), clamped to {4, 8, 16}
 *      K=4       → T=4   (1 block)
 *      K=8       → T=8   (1 block)
 *      K≥16      → T=16  (K/16 blocks, sweet spot for AVX2)
 *
 * 2. STRIDED (fallback — data arrives in stride-K layout)
 *
 *    Layout: data_re[n*K + k]   n=0..31, k=0..K-1
 *    Uses flat twiddles only (no ladder — 16 YMM regs insufficient).
 *
 * ═══════════════════════════════════════════════════════════════════
 * TWIDDLE TABLE FORMAT
 * ═══════════════════════════════════════════════════════════════════
 *
 * Flat: tw_re[(n-1)*K + k] = Re(W_{32K}^{n*k}),  n=1..31, k=0..K-1
 *       Size: 31*K doubles per component
 *
 * Packed: ptw_re[block*31*T + (n-1)*T + j]
 *         Derived from flat via r32a_repack_tw_to_super()
 *         Same total size, different layout for contiguous access.
 *
 * ═══════════════════════════════════════════════════════════════════
 * FILES INCLUDED
 * ═══════════════════════════════════════════════════════════════════
 *
 * fft_radix32_avx2_tw.h        — flat twiddled kernels (fwd + bwd)
 * fft_radix32_avx2_tw_packed.h — repack utilities + super-block drivers
 */

#ifndef FFT_RADIX32_AVX2_TW_UNIFIED_H
#define FFT_RADIX32_AVX2_TW_UNIFIED_H

#include <immintrin.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * KERNEL INCLUDES
 * ═══════════════════════════════════════════════════════════════ */

#undef  R32A_LD
#undef  R32A_ST
#define R32A_LD(p)   _mm256_load_pd(p)
#define R32A_ST(p,v) _mm256_store_pd((p),(v))

#include "fft_radix32_avx2_tw.h"
#include "fft_radix32_avx2_tw_packed.h"

/* ═══════════════════════════════════════════════════════════════
 * TUNABLES
 *
 * AVX2 sweet spot is T=16 (benchmarked on Zen3/Alderlake).
 * T=16: 32*16 = 512 doubles = 4KB data per block, fits L1.
 *       31*16 = 496 doubles = ~4KB twiddles per block, fits L1.
 *       Total working set per block: ~8KB + 1KB spill = ~9KB.
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R32A_PACKED_BLOCK_T
#define R32A_PACKED_BLOCK_T 16
#endif

/* ═══════════════════════════════════════════════════════════════
 * PATH 1: PACKED DISPATCH (production)
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r32a_packed_optimal_T(size_t K) {
    if (K < 8)  return 4;
    if (K < 16) return 8;
    return (size_t)R32A_PACKED_BLOCK_T;
}

__attribute__((target("avx2,fma")))
static inline void radix32_tw_packed_dispatch_fwd_avx2(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    const size_t T = r32a_packed_optimal_T(K);
    const size_t nb = K / T;
    radix32_tw_packed_super_fwd_avx2(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, nb, T);
}

__attribute__((target("avx2,fma")))
static inline void radix32_tw_packed_dispatch_bwd_avx2(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    const size_t T = r32a_packed_optimal_T(K);
    const size_t nb = K / T;
    radix32_tw_packed_super_bwd_avx2(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, nb, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PATH 2: STRIDED DISPATCH (fallback)
 *
 * Flat twiddles only — no ladder (16 YMM regs insufficient).
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void radix32_tw_strided_dispatch_fwd_avx2(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    radix32_tw_flat_dit_kernel_fwd_avx2(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, K);
}

__attribute__((target("avx2,fma")))
static inline void radix32_tw_strided_dispatch_bwd_avx2(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    radix32_tw_flat_dit_kernel_bwd_avx2(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, K);
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 * ═══════════════════════════════════════════════════════════════ */

static inline void r32a_build_flat_twiddles(
    size_t K, int dir,
    double * __restrict__ tw_re,
    double * __restrict__ tw_im)
{
    const size_t N = 32 * K;
    const double two_pi = 6.28318530717958647692;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double angle = two_pi * (double)n * (double)k / (double)N;
            tw_re[(n-1)*K + k] = __builtin_cos(angle);
            tw_im[(n-1)*K + k] = (double)dir * __builtin_sin(angle);
        }
}

static inline void r32a_build_packed_twiddles(
    size_t K, size_t T,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im)
{
    r32a_repack_tw_to_super(flat_tw_re, flat_tw_im, ptw_re, ptw_im, K, T);
}

static inline void r32a_pack_input(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    r32a_repack_strided_to_super(src_re, src_im, dst_re, dst_im, K, T);
}

static inline void r32a_unpack_output(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    r32a_repack_super_to_strided(src_re, src_im, dst_re, dst_im, K, T);
}

/* ═══════════════════════════════════════════════════════════════
 * MEMORY SIZING
 * ═══════════════════════════════════════════════════════════════ */

static inline size_t r32a_flat_tw_size(size_t K) { return 31 * K; }
static inline size_t r32a_packed_tw_size(size_t K) { return 31 * K; }
static inline size_t r32a_data_size(size_t K) { return 32 * K; }

#endif /* FFT_RADIX32_AVX2_TW_UNIFIED_H */
