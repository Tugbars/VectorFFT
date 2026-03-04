/**
 * @file fft_radix32_avx512_tw_unified.h
 * @brief Unified twiddled DFT-32 AVX-512 dispatch
 *
 * ═══════════════════════════════════════════════════════════════════
 * ARCHITECTURE OVERVIEW
 * ═══════════════════════════════════════════════════════════════════
 *
 * Two data layout paths:
 *
 * 1. PACKED (production — planner keeps data in packed layout)
 *
 *    Layout:   data_re[block*32*T + n*T + j]   n=0..31, j=0..T-1
 *    Twiddle:  tw_re[block*31*T + (n-1)*T + j] n=1..31
 *
 *    Every load is a contiguous aligned 64-byte fetch. Zero strides.
 *    DFT runs at peak FMA throughput — memory never bottlenecks.
 *
 *    Benchmark (DFT-only, packed, vs FFTW batched DFT-32):
 *      K=8:   1.03×    K=128:  1.14×
 *      K=16:  1.24×    K=256:  1.08×
 *      K=32:  1.28×    K=512:  1.07×
 *      K=64:  1.10×    K=1024: 0.94×
 *
 *    Dispatch: T = min(K, 32), clamped to {8, 16, 32}
 *      K=8       → T=8    (1 block)
 *      K=16      → T=16   (1 block)
 *      K≥32      → T=32   (K/32 blocks, sweet spot)
 *
 * 2. STRIDED (fallback — data arrives in stride-K layout)
 *
 *    Layout:   data_re[n*K + k]   n=0..31, k=0..K-1
 *
 *    Uses binary-ladder twiddle compression for K≥128 to keep
 *    twiddle table in L1. NT output stores for K≥2048.
 *
 *    Benchmark (fused twiddle+DFT-32 vs FFTW batched DFT-32):
 *      K=8:   1.08×    K=128:  1.03×
 *      K=16:  1.17×    K=256:  0.95×
 *      K=32:  1.21×    K=512:  0.82×
 *      K=64:  1.14×    K=1024: 0.83×
 *
 *    Dispatch:
 *      K < 128   → flat U=1 (tw table fits L1, fewest cmuls)
 *      128 ≤ K < 2048 → ladder U=1 (5 loads vs 31, table in L1)
 *      K ≥ 2048  → ladder U=1 + NT stores + sfence
 *
 * ═══════════════════════════════════════════════════════════════════
 * TWIDDLE TABLE FORMATS
 * ═══════════════════════════════════════════════════════════════════
 *
 * Flat (strided, K < 128):
 *   tw_re[(n-1)*K + k] = Re(W_{32K}^{n*k}),  n=1..31, k=0..K-1
 *   Size: 31*K doubles per component (62*K total)
 *
 * Ladder (strided, K ≥ 128):
 *   base_tw_re[i*K + k] = Re(W_{32K}^{p_i*k}),  p={1,2,4,8,16}
 *   Size: 5*K doubles per component (10*K total)
 *
 * Packed (T-block):
 *   ptw_re[block*31*T + (n-1)*T + j],  n=1..31, j=0..T-1
 *   Derived from flat: ptw[block*31*T + (n-1)*T + j] = tw[(n-1)*K + block*T + j]
 *   Size: 31*K doubles per component (same total, different layout)
 *
 * ═══════════════════════════════════════════════════════════════════
 * FILES INCLUDED
 * ═══════════════════════════════════════════════════════════════════
 *
 * fft_radix32_avx512_tw_ladder.h  — 6 codegen kernels:
 *   flat {fwd,bwd}           (U=1, flat twiddle table)
 *   ladder {fwd,bwd} × {U1}  (binary-ladder twiddle compression)
 *   ladder {fwd,bwd} × {U2}  (dual pipeline, unused in dispatch)
 *
 * Plus NT variants of all 6 via multi-include trick.
 *
 * fft_radix32_avx512_tw_packed.h  — packed-block drivers + repack:
 *   radix32_tw_packed_{fwd,bwd}_avx512        (T=8 blocks)
 *   radix32_tw_packed_super_{fwd,bwd}_avx512  (arbitrary T blocks)
 *   r32_repack_{strided_to,to_strided}_{packed,super}
 *   r32_repack_tw_to_{packed,super}
 */

#ifndef FFT_RADIX32_AVX512_TW_UNIFIED_H
#define FFT_RADIX32_AVX512_TW_UNIFIED_H

#include <immintrin.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * KERNEL INCLUDES — temporal stores
 * ═══════════════════════════════════════════════════════════════ */

#undef  R32L_LD
#undef  R32L_ST
#define R32L_LD(p)   _mm512_load_pd(p)
#define R32L_ST(p,v) _mm512_store_pd((p),(v))
#include "fft_radix32_avx512_tw_ladder.h"

/* ═══════════════════════════════════════════════════════════════
 * KERNEL INCLUDES — NT stores (multi-include trick)
 * ═══════════════════════════════════════════════════════════════ */

#undef FFT_RADIX32_AVX512_TW_LADDER_H
#undef  R32L_LD
#undef  R32L_ST
#define R32L_LD(p)   _mm512_load_pd(p)
#define R32L_ST(p,v) _mm512_stream_pd((p),(v))

#define radix32_tw_flat_dit_kernel_fwd_avx512       radix32_tw_flat_dit_kernel_fwd_avx512_nt
#define radix32_tw_flat_dit_kernel_bwd_avx512       radix32_tw_flat_dit_kernel_bwd_avx512_nt
#define radix32_tw_ladder_dit_kernel_fwd_avx512_u1  radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt
#define radix32_tw_ladder_dit_kernel_bwd_avx512_u1  radix32_tw_ladder_dit_kernel_bwd_avx512_u1_nt
#define radix32_tw_ladder_dit_kernel_fwd_avx512_u2  radix32_tw_ladder_dit_kernel_fwd_avx512_u2_nt
#define radix32_tw_ladder_dit_kernel_bwd_avx512_u2  radix32_tw_ladder_dit_kernel_bwd_avx512_u2_nt

#include "fft_radix32_avx512_tw_ladder.h"

#undef radix32_tw_flat_dit_kernel_fwd_avx512
#undef radix32_tw_flat_dit_kernel_bwd_avx512
#undef radix32_tw_ladder_dit_kernel_fwd_avx512_u1
#undef radix32_tw_ladder_dit_kernel_bwd_avx512_u1
#undef radix32_tw_ladder_dit_kernel_fwd_avx512_u2
#undef radix32_tw_ladder_dit_kernel_bwd_avx512_u2

/* ═══════════════════════════════════════════════════════════════
 * PACKED CODELET + REPACK UTILITIES
 * ═══════════════════════════════════════════════════════════════ */

#include "fft_radix32_avx512_tw_packed.h"

/* ═══════════════════════════════════════════════════════════════
 * TUNABLE THRESHOLDS
 *
 * LADDER_THRESH:  K above which ladder replaces flat (strided path)
 *                 128 → flat tw table at K=64 is 4KB (fits L1)
 *                        ladder tw table at K=128 is 1.3KB vs flat 8KB
 *
 * NT_THRESH:      K above which NT output stores are used
 *                 2048 → output working set 32*2048*8*2 = 1MB (>> L2)
 *                         NT bypasses cache, saves L2 for input reads
 *
 * PACKED_DIRECT:  max K for single-block packed (T=K, one kernel call)
 *                 32 → T=32 is sweet spot, single call = no loop overhead
 *
 * These are compile-time tunables for the target microarchitecture.
 * Defaults are benchmarked on ICX (Ice Lake Server, 48KB L1d, 1.25MB L2).
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R32_LADDER_THRESH
#define R32_LADDER_THRESH 128
#endif

#ifndef R32_NT_THRESH
#define R32_NT_THRESH 2048
#endif

#ifndef R32_PACKED_BLOCK_T
#define R32_PACKED_BLOCK_T 32
#endif

/* ═══════════════════════════════════════════════════════════════
 * PATH 1: PACKED DISPATCH (production)
 *
 * Data and twiddles in packed-block layout. Zero strides.
 *
 * The planner pre-packs twiddles once at plan creation time.
 * Data flows between stages in packed layout — no repacking
 * between stages.
 *
 * Interface:
 *   K = total number of DFT-32 instances (must be multiple of 8)
 *   T = block size in k-values (8, 16, or 32; K must be multiple of T)
 *   num_blocks = K / T
 *
 * Data layout:
 *   in_re[block * 32*T + n*T + j]     n=0..31, j=0..T-1
 *   tw_re[block * 31*T + (n-1)*T + j] n=1..31, j=0..T-1
 *
 * Auto-dispatch picks T from K:
 *   K = 8      → T = 8,  1 block
 *   K = 16     → T = 16, 1 block
 *   K ≥ 32     → T = 32, K/32 blocks (sweet spot)
 *   K % T == 0 required
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Compute optimal block size T for a given K.
 * Returns 8, 16, or 32.
 */
static inline size_t r32_packed_optimal_T(size_t K) {
    if (K < 16)  return 8;
    if (K < 32)  return 16;
    return (size_t)R32_PACKED_BLOCK_T;  /* default 32 */
}

/**
 * Packed forward DFT-32 with auto T selection.
 *
 * @param K          Total DFT-32 instances (must be multiple of 8)
 * @param in_re/im   Input in packed layout [K * 32 doubles]
 * @param out_re/im  Output in packed layout [K * 32 doubles]
 * @param tw_re/im   Twiddles in packed layout [K * 31 doubles]
 *
 * Call r32_packed_optimal_T(K) to get the T used, then
 * r32_repack_tw_to_super() to build the twiddle table.
 */
__attribute__((target("avx512f,avx512dq,fma")))
static inline void radix32_tw_packed_dispatch_fwd(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    const size_t T = r32_packed_optimal_T(K);
    const size_t nb = K / T;

    radix32_tw_packed_super_fwd_avx512(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, nb, T);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void radix32_tw_packed_dispatch_bwd(
    size_t K,
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im)
{
    const size_t T = r32_packed_optimal_T(K);
    const size_t nb = K / T;

    radix32_tw_packed_super_bwd_avx512(
        in_re, in_im, out_re, out_im,
        tw_re, tw_im, nb, T);
}

/* ═══════════════════════════════════════════════════════════════
 * PATH 2: STRIDED DISPATCH (fallback / legacy)
 *
 * Data in stride-K layout: data_re[n*K + k], n=0..31, k=0..K-1
 *
 * Uses flat twiddles for K < LADDER_THRESH, binary-ladder for
 * LADDER_THRESH ≤ K < NT_THRESH, and ladder+NT for K ≥ NT_THRESH.
 *
 * The caller provides BOTH flat and ladder twiddle tables.
 * (In practice, the planner only allocates the one needed.)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static inline void radix32_tw_strided_dispatch_fwd(
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
    if (K < (size_t)R32_LADDER_THRESH) {
        /* Flat: 31 tw loads/k-step, table ≤ 4KB @ K=64 → fits L1 */
        radix32_tw_flat_dit_kernel_fwd_avx512(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
    } else if (K < (size_t)R32_NT_THRESH) {
        /* Ladder: 5 tw loads/k-step, table ≤ 16KB @ K=1024 → fits L1 */
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1(
            in_re, in_im, out_re, out_im,
            base_tw_re, base_tw_im, K);
    } else {
        /* Ladder + NT: bypass output cache for K ≥ 2048 */
        radix32_tw_ladder_dit_kernel_fwd_avx512_u1_nt(
            in_re, in_im, out_re, out_im,
            base_tw_re, base_tw_im, K);
        _mm_sfence();
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void radix32_tw_strided_dispatch_bwd(
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
    if (K < (size_t)R32_LADDER_THRESH) {
        radix32_tw_flat_dit_kernel_bwd_avx512(
            in_re, in_im, out_re, out_im,
            flat_tw_re, flat_tw_im, K);
    } else if (K < (size_t)R32_NT_THRESH) {
        radix32_tw_ladder_dit_kernel_bwd_avx512_u1(
            in_re, in_im, out_re, out_im,
            base_tw_re, base_tw_im, K);
    } else {
        radix32_tw_ladder_dit_kernel_bwd_avx512_u1_nt(
            in_re, in_im, out_re, out_im,
            base_tw_re, base_tw_im, K);
        _mm_sfence();
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLANNER HELPERS
 *
 * Utilities for the planner to build twiddle tables and repack
 * data at plan creation / API boundary.
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Build flat twiddle table for stride-K layout.
 *
 * tw_re[(n-1)*K + k] = cos(2π·n·k / (32·K))
 * tw_im[(n-1)*K + k] = dir · sin(2π·n·k / (32·K))
 *
 * dir = -1 for forward, +1 for backward.
 * Caller allocates 31*K doubles for each of tw_re, tw_im.
 */
static inline void r32_build_flat_twiddles(
    size_t K, int dir,
    double * __restrict__ tw_re,
    double * __restrict__ tw_im)
{
    const size_t N = 32 * K;
    const double two_pi = 6.28318530717958647692;
    for (int n = 1; n < 32; n++) {
        for (size_t k = 0; k < K; k++) {
            double angle = two_pi * (double)n * (double)k / (double)N;
            tw_re[(n-1)*K + k] = __builtin_cos(angle);
            tw_im[(n-1)*K + k] = (double)dir * __builtin_sin(angle);
        }
    }
}

/**
 * Build ladder (compressed) twiddle table for stride-K layout.
 *
 * base_tw_re[i*K + k] = cos(2π·p_i·k / (32·K))
 * base_tw_im[i*K + k] = dir · sin(2π·p_i·k / (32·K))
 *
 * where p = {1, 2, 4, 8, 16} (i=0..4).
 * Caller allocates 5*K doubles for each of base_tw_re, base_tw_im.
 */
static inline void r32_build_ladder_twiddles(
    size_t K, int dir,
    double * __restrict__ base_tw_re,
    double * __restrict__ base_tw_im)
{
    const size_t N = 32 * K;
    const double two_pi = 6.28318530717958647692;
    static const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        for (size_t k = 0; k < K; k++) {
            double angle = two_pi * (double)pows[i] * (double)k / (double)N;
            base_tw_re[i*K + k] = __builtin_cos(angle);
            base_tw_im[i*K + k] = (double)dir * __builtin_sin(angle);
        }
    }
}

/**
 * Build packed twiddle table from flat twiddles.
 *
 * T = block size (call r32_packed_optimal_T(K) to get it).
 * Caller allocates 31*K doubles for each of ptw_re, ptw_im.
 *
 * Requires flat twiddle table as input — call r32_build_flat_twiddles first.
 */
static inline void r32_build_packed_twiddles(
    size_t K, size_t T,
    const double * __restrict__ flat_tw_re,
    const double * __restrict__ flat_tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im)
{
    r32_repack_tw_to_super(flat_tw_re, flat_tw_im, ptw_re, ptw_im, K, T);
}

/**
 * Repack input data: stride-K → packed (at API input boundary).
 */
static inline void r32_pack_input(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    r32_repack_strided_to_super(src_re, src_im, dst_re, dst_im, K, T);
}

/**
 * Repack output data: packed → stride-K (at API output boundary).
 */
static inline void r32_unpack_output(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    r32_repack_super_to_strided(src_re, src_im, dst_re, dst_im, K, T);
}

/* ═══════════════════════════════════════════════════════════════
 * MEMORY SIZING HELPERS
 * ═══════════════════════════════════════════════════════════════ */

/** Flat twiddle table size in doubles (per re/im component). */
static inline size_t r32_flat_tw_size(size_t K) { return 31 * K; }

/** Ladder twiddle table size in doubles (per re/im component). */
static inline size_t r32_ladder_tw_size(size_t K) { return 5 * K; }

/** Packed twiddle table size in doubles (per re/im component). */
static inline size_t r32_packed_tw_size(size_t K) { return 31 * K; }

/** Data buffer size in doubles (per re/im component). */
static inline size_t r32_data_size(size_t K) { return 32 * K; }

#endif /* FFT_RADIX32_AVX512_TW_UNIFIED_H */
