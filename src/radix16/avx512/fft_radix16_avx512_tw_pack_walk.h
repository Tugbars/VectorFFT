/**
 * @file fft_radix16_avx512_tw_pack_walk.h
 * @brief DFT-16 AVX-512 packed data + walked twiddles (zero tw table)
 *
 * ═══════════════════════════════════════════════════════════════════
 * ALGORITHM
 * ═══════════════════════════════════════════════════════════════════
 *
 * Data in packed layout: block*16*8 contiguous doubles.
 * Twiddles generated on-the-fly from 4 walking base accumulators:
 *   b[0]=W^1, b[1]=W^2, b[2]=W^4, b[3]=W^8
 *
 * Per block:
 *   1. Derive all 15 twiddles from 4 bases via binary tree:
 *      Direct:  W^1, W^2, W^4, W^8
 *      Level 1: W^3=W^1*W^2
 *      Level 2: W^5=W^1*W^4, W^6=W^2*W^4, W^7=W^3*W^4
 *      Level 3: W^9..W^15 = W^{n-8}*W^8
 *      → 10 cmuls total
 *   2. Call flat DFT-16 kernel on packed block with derived twiddles
 *   3. Walk: b[i] *= step[i]  → 4 cmuls
 *
 * Total overhead: 14 cmuls/block vs 15 loads/block for table approach.
 * Wins when tw table (15*K*16 bytes) exceeds L2 (K > ~512).
 *
 * ═══════════════════════════════════════════════════════════════════
 * REQUIRES
 * ═══════════════════════════════════════════════════════════════════
 *
 * Include the flat tw kernel BEFORE this header:
 *   #include "fft_radix16_avx512_tw.h"
 *   — provides:
 *     radix16_tw_flat_dit_kernel_fwd_avx512(...)
 *     radix16_tw_flat_dit_kernel_bwd_avx512(...)
 */

#ifndef FFT_RADIX16_AVX512_TW_PACK_WALK_H
#define FFT_RADIX16_AVX512_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WALK PLAN
 *
 * step[i] = W_{16K}^{pow[i]*8}   (advance 8 lanes = 1 ZMM vector)
 * init[i][j] = W_{16K}^{pow[i]*j}  j=0..7  (initial twiddle vector)
 *
 * pow = {1, 2, 4, 8}
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double step_re[4];
    double step_im[4];
    double init_re[4][8];
    double init_im[4][8];
} radix16_walk_plan_t;

static inline void radix16_walk_plan_init(
    radix16_walk_plan_t *plan, size_t K)
{
    const size_t NN = 16 * K;
    const int pows[4] = {1, 2, 4, 8};
    for (int i = 0; i < 4; i++) {
        double a_step = -2.0 * M_PI * (double)(pows[i] * 8) / (double)NN;
        plan->step_re[i] = cos(a_step);
        plan->step_im[i] = sin(a_step);
        for (int j = 0; j < 8; j++) {
            double a = -2.0 * M_PI * (double)(pows[i] * j) / (double)NN;
            plan->init_re[i][j] = cos(a);
            plan->init_im[i][j] = sin(a);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * COMPLEX MULTIPLY HELPERS
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R16W_CMUL_DEFINED
#define R16W_CMUL_DEFINED
#define R16W_CMUL_RE(ar,ai,br,bi) \
    _mm512_fnmadd_pd(ai, bi, _mm512_mul_pd(ar, br))
#define R16W_CMUL_IM(ar,ai,br,bi) \
    _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br))
#endif

/* ═══════════════════════════════════════════════════════════════
 * DERIVE 15 TWIDDLES FROM 4 BASES
 *
 * Binary tree:
 *   Direct: b1(slot 0), b2(slot 1), b4(slot 3), b8(slot 7)
 *   L1: W^3 = W^1*W^2 (slot 2)
 *   L2: W^5=W^1*W^4(slot 4), W^6=W^2*W^4(slot 5), W^7=W^3*W^4(slot 6)
 *   L3: W^9..W^15 = W^{n-8}*W^8 (slots 8..14)
 *   Total: 10 cmuls
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void r16w_derive_twiddles(
    const __m512d b_re[4], const __m512d b_im[4],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    /* Direct bases: W^1→slot0, W^2→slot1, W^4→slot3, W^8→slot7 */
    _mm512_store_pd(&tw_re[0*8], b_re[0]);  _mm512_store_pd(&tw_im[0*8], b_im[0]);
    _mm512_store_pd(&tw_re[1*8], b_re[1]);  _mm512_store_pd(&tw_im[1*8], b_im[1]);
    _mm512_store_pd(&tw_re[3*8], b_re[2]);  _mm512_store_pd(&tw_im[3*8], b_im[2]);
    _mm512_store_pd(&tw_re[7*8], b_re[3]);  _mm512_store_pd(&tw_im[7*8], b_im[3]);

    /* W^3 = W^1 * W^2 */
    __m512d w3r = R16W_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m512d w3i = R16W_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm512_store_pd(&tw_re[2*8], w3r);  _mm512_store_pd(&tw_im[2*8], w3i);

    /* W^5 = W^1 * W^4 */
    __m512d w5r = R16W_CMUL_RE(b_re[0],b_im[0],b_re[2],b_im[2]);
    __m512d w5i = R16W_CMUL_IM(b_re[0],b_im[0],b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[4*8], w5r);  _mm512_store_pd(&tw_im[4*8], w5i);

    /* W^6 = W^2 * W^4 */
    __m512d w6r = R16W_CMUL_RE(b_re[1],b_im[1],b_re[2],b_im[2]);
    __m512d w6i = R16W_CMUL_IM(b_re[1],b_im[1],b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[5*8], w6r);  _mm512_store_pd(&tw_im[5*8], w6i);

    /* W^7 = W^3 * W^4 */
    __m512d w7r = R16W_CMUL_RE(w3r,w3i,b_re[2],b_im[2]);
    __m512d w7i = R16W_CMUL_IM(w3r,w3i,b_re[2],b_im[2]);
    _mm512_store_pd(&tw_re[6*8], w7r);  _mm512_store_pd(&tw_im[6*8], w7i);

    /* W^9..W^15 = W^{n-8} * W^8 */
    for (int n = 9; n <= 15; n++) {
        __m512d sr = _mm512_load_pd(&tw_re[(n-9)*8]);
        __m512d si = _mm512_load_pd(&tw_im[(n-9)*8]);
        _mm512_store_pd(&tw_re[(n-1)*8], R16W_CMUL_RE(sr,si,b_re[3],b_im[3]));
        _mm512_store_pd(&tw_im[(n-1)*8], R16W_CMUL_IM(sr,si,b_re[3],b_im[3]));
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static inline void
radix16_tw_pack_walk_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix16_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    __m512d stp_re[4], stp_im[4];
    for (int i = 0; i < 4; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[4], b_im[4];
    for (int i = 0; i < 4; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    /* Scratch for derived twiddles: 15*8 = 120 doubles × 2 = 1920 bytes */
    double __attribute__((aligned(64))) tw_re[15*8];
    double __attribute__((aligned(64))) tw_im[15*8];

    for (size_t blk = 0; blk < nb; blk++) {
        r16w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix16_tw_flat_dit_kernel_fwd_avx512(
            in_re + blk*128, in_im + blk*128,
            out_re + blk*128, out_im + blk*128,
            tw_re, tw_im, 8);

        /* Walk: advance 4 bases by step */
        for (int i = 0; i < 4; i++) {
            __m512d nr = R16W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R16W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static inline void
radix16_tw_pack_walk_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix16_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    __m512d stp_re[4], stp_im[4];
    for (int i = 0; i < 4; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[4], b_im[4];
    for (int i = 0; i < 4; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(64))) tw_re[15*8];
    double __attribute__((aligned(64))) tw_im[15*8];

    for (size_t blk = 0; blk < nb; blk++) {
        r16w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix16_tw_flat_dit_kernel_bwd_avx512(
            in_re + blk*128, in_im + blk*128,
            out_re + blk*128, out_im + blk*128,
            tw_re, tw_im, 8);

        for (int i = 0; i < 4; i++) {
            __m512d nr = R16W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R16W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED LAYOUT HELPERS
 *
 * Repack strided→packed and reverse.
 * Data:    src_re[n*K + k] → dst_re[b*16*T + n*T + j]
 * Twiddle: src_tw[(n-1)*K + k] → dst_tw[b*15*T + (n-1)*T + j]
 *   where b = k/T, j = k%T, T=8 for AVX-512
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f")))
static inline void radix16_pack_input_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 128; /* 16*8 */
        for (int n = 0; n < 16; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix16_unpack_output_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 128, dk = b * 8;
        for (int n = 0; n < 16; n++) {
            _mm512_storeu_pd(&dst_re[n*K + dk], _mm512_loadu_pd(&src_re[sk + n*8]));
            _mm512_storeu_pd(&dst_im[n*K + dk], _mm512_loadu_pd(&src_im[sk + n*8]));
        }
    }
}

__attribute__((target("avx512f")))
static inline void radix16_pack_twiddles_avx512(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K >> 3;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 8, dk = b * 120; /* 15*8 */
        for (int n = 0; n < 15; n++) {
            _mm512_storeu_pd(&dst_re[dk + n*8], _mm512_loadu_pd(&src_re[n*K + sk]));
            _mm512_storeu_pd(&dst_im[dk + n*8], _mm512_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PACKED TABLE DRIVERS (for K ≤ walk threshold)
 *
 * Process K/8 blocks, each calling flat kernel at K=8.
 * Data AND twiddles must be in packed layout.
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,avx512dq,fma")))
static inline void r16_tw_packed_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const size_t nb = K / 8;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_fwd_avx512(
            in_re + b*128, in_im + b*128,
            out_re + b*128, out_im + b*128,
            tw_re + b*120, tw_im + b*120, 8);
}

__attribute__((target("avx512f,avx512dq,fma")))
static inline void r16_tw_packed_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const size_t nb = K / 8;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_bwd_avx512(
            in_re + b*128, in_im + b*128,
            out_re + b*128, out_im + b*128,
            tw_re + b*120, tw_im + b*120, 8);
}

/* ═══════════════════════════════════════════════════════════════
 * AUTO-SELECT: packed table vs pack+walk
 * ═══════════════════════════════════════════════════════════════ */

#ifndef RADIX16_WALK_THRESHOLD
#define RADIX16_WALK_THRESHOLD 512
#endif

static inline int radix16_should_walk(size_t K) { return K > RADIX16_WALK_THRESHOLD; }
static inline size_t radix16_packed_tw_size(size_t K) { return radix16_should_walk(K) ? 0 : 15*K; }

#undef R16W_CMUL_RE
#undef R16W_CMUL_IM

#endif /* FFT_RADIX16_AVX512_TW_PACK_WALK_H */
