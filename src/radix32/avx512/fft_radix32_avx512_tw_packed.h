/**
 * @file fft_radix32_avx512_tw_packed.h
 * @brief Packed-block twiddled DFT-32 AVX-512 — contiguous layout
 *
 * LAYOUT:
 *   packed_re[block*256 + n*8 + lane]
 *   block = k/8,  n = 0..31,  lane = k%8
 *
 * Each block of 8 DFT-32 instances occupies 256 contiguous doubles.
 * Every AVX-512 load is a single aligned 64-byte fetch — zero strides.
 *
 * The codelet runs K/8 blocks, each calling the flat twiddled kernel at K=8.
 * K=8 is the compute-bound sweet spot: 1.08× FFTW with all data in L1.
 *
 * Twiddle layout (packed):
 *   tw_packed_re[block*248 + (n-1)*8 + lane]  for n=1..31
 *   where tw value = W_{32*K_total}^{n * (block*8 + lane)}
 *
 * Requires: fft_radix32_avx512_tw_ladder.h (for flat twiddled kernel)
 */

#ifndef FFT_RADIX32_AVX512_TW_PACKED_H
#define FFT_RADIX32_AVX512_TW_PACKED_H

#include <immintrin.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════
 * REPACK: stride-K ↔ packed-block layout
 *
 * stride-K:   data_re[n*K + k]
 * packed:     data_re[(k/8)*256 + n*8 + (k%8)]
 *
 * These are O(N) sequential passes — streaming, cache-friendly.
 * In a production planner, data would never be in stride-K layout;
 * the planner arranges stages to flow naturally through packed.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f")))
static void r32_repack_strided_to_packed(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K)
{
    const size_t num_blocks = K / 8;
    for (size_t b = 0; b < num_blocks; b++) {
        for (int n = 0; n < 32; n++) {
            /* 8 consecutive k-values from stride-K layout */
            __m512d vr = _mm512_loadu_pd(&src_re[n*K + b*8]);
            __m512d vi = _mm512_loadu_pd(&src_im[n*K + b*8]);
            /* Store to packed: contiguous within block */
            _mm512_store_pd(&dst_re[b*256 + n*8], vr);
            _mm512_store_pd(&dst_im[b*256 + n*8], vi);
        }
    }
}

__attribute__((target("avx512f")))
static void r32_repack_packed_to_strided(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K)
{
    const size_t num_blocks = K / 8;
    for (size_t b = 0; b < num_blocks; b++) {
        for (int n = 0; n < 32; n++) {
            __m512d vr = _mm512_load_pd(&src_re[b*256 + n*8]);
            __m512d vi = _mm512_load_pd(&src_im[b*256 + n*8]);
            _mm512_storeu_pd(&dst_re[n*K + b*8], vr);
            _mm512_storeu_pd(&dst_im[n*K + b*8], vi);
        }
    }
}

/* Repack flat twiddles: stride-K → packed-block (T=8) */
__attribute__((target("avx512f")))
static void r32_repack_tw_to_packed(
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im,
    size_t K)
{
    const size_t num_blocks = K / 8;
    for (size_t b = 0; b < num_blocks; b++) {
        for (int n = 0; n < 31; n++) {
            __m512d vr = _mm512_loadu_pd(&tw_re[n*K + b*8]);
            __m512d vi = _mm512_loadu_pd(&tw_im[n*K + b*8]);
            _mm512_store_pd(&ptw_re[b*248 + n*8], vr);
            _mm512_store_pd(&ptw_im[b*248 + n*8], vi);
        }
    }
}

/* Generic repack: stride-K → packed super-blocks of size T */
__attribute__((target("avx512f")))
static void r32_repack_strided_to_super(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_stride = 32 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 32; n++) {
            for (size_t j = 0; j < T; j += 8) {
                __m512d vr = _mm512_loadu_pd(&src_re[n*K + b*T + j]);
                __m512d vi = _mm512_loadu_pd(&src_im[n*K + b*T + j]);
                _mm512_store_pd(&dst_re[b*data_stride + n*T + j], vr);
                _mm512_store_pd(&dst_im[b*data_stride + n*T + j], vi);
            }
        }
    }
}

__attribute__((target("avx512f")))
static void r32_repack_super_to_strided(
    const double * __restrict__ src_re,
    const double * __restrict__ src_im,
    double * __restrict__ dst_re,
    double * __restrict__ dst_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t data_stride = 32 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 32; n++) {
            for (size_t j = 0; j < T; j += 8) {
                __m512d vr = _mm512_load_pd(&src_re[b*data_stride + n*T + j]);
                __m512d vi = _mm512_load_pd(&src_im[b*data_stride + n*T + j]);
                _mm512_storeu_pd(&dst_re[n*K + b*T + j], vr);
                _mm512_storeu_pd(&dst_im[n*K + b*T + j], vi);
            }
        }
    }
}

/* Repack flat twiddles: stride-K → packed super-blocks of size T */
__attribute__((target("avx512f")))
static void r32_repack_tw_to_super(
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    double * __restrict__ ptw_re,
    double * __restrict__ ptw_im,
    size_t K, size_t T)
{
    const size_t nb = K / T;
    const size_t tw_stride = 31 * T;
    for (size_t b = 0; b < nb; b++) {
        for (int n = 0; n < 31; n++) {
            for (size_t j = 0; j < T; j += 8) {
                __m512d vr = _mm512_loadu_pd(&tw_re[n*K + b*T + j]);
                __m512d vi = _mm512_loadu_pd(&tw_im[n*K + b*T + j]);
                _mm512_store_pd(&ptw_re[b*tw_stride + n*T + j], vr);
                _mm512_store_pd(&ptw_im[b*tw_stride + n*T + j], vi);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED TWIDDLED DFT-32 — forward
 *
 * Processes K/8 blocks, each block = flat twiddled DFT-32 at K=8.
 * All memory accesses are contiguous aligned loads/stores.
 *
 * Input/output/twiddles must be in packed-block layout.
 * ═══════════════════════════════════════════════════════════════ */

/* Forward reference to flat kernel from tw_ladder.h */
/* radix32_tw_flat_dit_kernel_{fwd,bwd}_avx512(in_re,in_im,out_re,out_im,tw_re,tw_im,K) */

/*
 * Packed codelet with configurable block size T.
 *
 * T=8:  256 doubles/block, twiddle 248/block. Minimum.
 * T=32: 1024 doubles/block, twiddle 992/block. Sweet spot.
 *       Working set: 16KB data + 8KB tw + 2KB spill = 26KB ⊂ L1
 * T=64: 2048 doubles/block, twiddle 1984/block.
 *       Working set: 32KB data + 16KB tw + 2KB spill = 50KB ≈ L1 limit
 *
 * Layout for block size T:
 *   data_re[block*(32*T) + n*T + j]     n=0..31, j=0..T-1
 *   tw_re[block*(31*T) + (n-1)*T + j]   n=1..31, j=0..T-1
 *
 * For T≥16 and T%16==0, uses U=2 flat twiddled kernel.
 * For T=8, uses U=1.
 */

/* T=8 blocks (original) */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_packed_fwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks)
{
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_fwd_avx512(
            in_re  + b*256, in_im  + b*256,
            out_re + b*256, out_im + b*256,
            tw_re  + b*248, tw_im  + b*248,
            8);
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_packed_bwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_blocks)
{
    for (size_t b = 0; b < num_blocks; b++) {
        radix32_tw_flat_dit_kernel_bwd_avx512(
            in_re  + b*256, in_im  + b*256,
            out_re + b*256, out_im + b*256,
            tw_re  + b*248, tw_im  + b*248,
            8);
    }
}

/* Super-block: T doubles per block, each call uses flat kernel at K=T */
__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_packed_super_fwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_super_blocks,
    size_t T)
{
    const size_t data_stride = 32 * T;   /* doubles per super-block data */
    const size_t tw_stride   = 31 * T;   /* doubles per super-block twiddle */
    for (size_t b = 0; b < num_super_blocks; b++) {
        radix32_tw_flat_dit_kernel_fwd_avx512(
            in_re  + b*data_stride, in_im  + b*data_stride,
            out_re + b*data_stride, out_im + b*data_stride,
            tw_re  + b*tw_stride,   tw_im  + b*tw_stride,
            T);
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_packed_super_bwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t num_super_blocks,
    size_t T)
{
    const size_t data_stride = 32 * T;
    const size_t tw_stride   = 31 * T;
    for (size_t b = 0; b < num_super_blocks; b++) {
        radix32_tw_flat_dit_kernel_bwd_avx512(
            in_re  + b*data_stride, in_im  + b*data_stride,
            out_re + b*data_stride, out_im + b*data_stride,
            tw_re  + b*tw_stride,   tw_im  + b*tw_stride,
            T);
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FULL PIPELINE: repack + packed DFT + repack
 *
 * For benchmarking fairness: includes repack overhead.
 * In production, data stays in packed layout between stages.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static void radix32_tw_packed_full_fwd_avx512(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double * __restrict__ out_re,
    double * __restrict__ out_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    const double * __restrict__ ptw_re,
    const double * __restrict__ ptw_im,
    double * __restrict__ scratch_in_re,
    double * __restrict__ scratch_in_im,
    double * __restrict__ scratch_out_re,
    double * __restrict__ scratch_out_im,
    size_t K)
{
    const size_t num_blocks = K / 8;

    /* Phase 1: repack input stride-K → packed */
    r32_repack_strided_to_packed(in_re, in_im,
                                  scratch_in_re, scratch_in_im, K);

    /* Phase 2: packed DFT-32 (all contiguous, peak speed) */
    radix32_tw_packed_fwd_avx512(scratch_in_re, scratch_in_im,
                                  scratch_out_re, scratch_out_im,
                                  ptw_re, ptw_im, num_blocks);

    /* Phase 3: repack output packed → stride-K */
    r32_repack_packed_to_strided(scratch_out_re, scratch_out_im,
                                  out_re, out_im, K);
}

#endif /* FFT_RADIX32_AVX512_TW_PACKED_H */
