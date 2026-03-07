/**
 * @file fft_radix32_avx2_tw_pack_walk.h
 * @brief DFT-32 AVX2 packed data + walked twiddles (zero tw table)
 *
 * AVX2 variant of the pack+walk approach. Identical algorithm to the
 * AVX-512 version but with T=4 (__m256d) instead of T=8 (__m512d).
 *
 * Per block (4 complex elements):
 *   1. Derive 31 twiddles from 5 walked bases via binary tree (26 cmuls)
 *   2. Call flat DFT-32 kernel on packed block
 *   3. Walk: b[i] *= step[i] (5 cmuls)
 *
 * Requires: fft_radix32_avx2_tw_v2.h included first (for flat kernel)
 */

#ifndef FFT_RADIX32_AVX2_TW_PACK_WALK_H
#define FFT_RADIX32_AVX2_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WALK PLAN (AVX2, T=4)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double step_re[5];
    double step_im[5];
    double init_re[5][4];  /* 4 lanes */
    double init_im[5][4];
} radix32_walk_plan_avx2_t;

static inline void radix32_walk_plan_avx2_init(
    radix32_walk_plan_avx2_t *plan, size_t K)
{
    const size_t NN = 32 * K;
    const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++) {
        double a_step = -2.0 * M_PI * (double)(pows[i] * 4) / (double)NN;
        plan->step_re[i] = cos(a_step);
        plan->step_im[i] = sin(a_step);
        for (int j = 0; j < 4; j++) {
            double a = -2.0 * M_PI * (double)(pows[i] * j) / (double)NN;
            plan->init_re[i][j] = cos(a);
            plan->init_im[i][j] = sin(a);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * COMPLEX MULTIPLY HELPERS (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R32W2_CMUL_DEFINED
#define R32W2_CMUL_DEFINED
#define R32W2_CMUL_RE(ar,ai,br,bi) \
    _mm256_fnmadd_pd(ai, bi, _mm256_mul_pd(ar, br))
#define R32W2_CMUL_IM(ar,ai,br,bi) \
    _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br))
#endif

/* ═══════════════════════════════════════════════════════════════
 * DERIVE 31 TWIDDLES FROM 5 BASES (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void r32w2_derive_twiddles(
    const __m256d b_re[5], const __m256d b_im[5],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    /* Direct bases: W^1→slot0, W^2→slot1, W^4→slot3, W^8→slot7, W^16→slot15 */
    _mm256_store_pd(&tw_re[0*4], b_re[0]);  _mm256_store_pd(&tw_im[0*4], b_im[0]);
    _mm256_store_pd(&tw_re[1*4], b_re[1]);  _mm256_store_pd(&tw_im[1*4], b_im[1]);
    _mm256_store_pd(&tw_re[3*4], b_re[2]);  _mm256_store_pd(&tw_im[3*4], b_im[2]);
    _mm256_store_pd(&tw_re[7*4], b_re[3]);  _mm256_store_pd(&tw_im[7*4], b_im[3]);
    _mm256_store_pd(&tw_re[15*4], b_re[4]); _mm256_store_pd(&tw_im[15*4], b_im[4]);

    /* W^3 = W^1 * W^2 */
    __m256d w3r = R32W2_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m256d w3i = R32W2_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm256_store_pd(&tw_re[2*4], w3r);  _mm256_store_pd(&tw_im[2*4], w3i);

    /* W^5 = W^1 * W^4 */
    __m256d w5r = R32W2_CMUL_RE(b_re[0],b_im[0],b_re[2],b_im[2]);
    __m256d w5i = R32W2_CMUL_IM(b_re[0],b_im[0],b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[4*4], w5r);  _mm256_store_pd(&tw_im[4*4], w5i);

    /* W^6 = W^2 * W^4 */
    __m256d w6r = R32W2_CMUL_RE(b_re[1],b_im[1],b_re[2],b_im[2]);
    __m256d w6i = R32W2_CMUL_IM(b_re[1],b_im[1],b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[5*4], w6r);  _mm256_store_pd(&tw_im[5*4], w6i);

    /* W^7 = W^3 * W^4 */
    __m256d w7r = R32W2_CMUL_RE(w3r,w3i,b_re[2],b_im[2]);
    __m256d w7i = R32W2_CMUL_IM(w3r,w3i,b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[6*4], w7r);  _mm256_store_pd(&tw_im[6*4], w7i);

    /* W^9..W^15 = W^{n-8} * W^8 */
    for (int n = 9; n <= 15; n++) {
        __m256d sr = _mm256_load_pd(&tw_re[(n-9)*4]);
        __m256d si = _mm256_load_pd(&tw_im[(n-9)*4]);
        _mm256_store_pd(&tw_re[(n-1)*4], R32W2_CMUL_RE(sr,si,b_re[3],b_im[3]));
        _mm256_store_pd(&tw_im[(n-1)*4], R32W2_CMUL_IM(sr,si,b_re[3],b_im[3]));
    }

    /* W^17..W^31 = W^{n-16} * W^16 */
    for (int n = 17; n <= 31; n++) {
        __m256d sr = _mm256_load_pd(&tw_re[(n-17)*4]);
        __m256d si = _mm256_load_pd(&tw_im[(n-17)*4]);
        _mm256_store_pd(&tw_re[(n-1)*4], R32W2_CMUL_RE(sr,si,b_re[4],b_im[4]));
        _mm256_store_pd(&tw_im[(n-1)*4], R32W2_CMUL_IM(sr,si,b_re[4],b_im[4]));
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK KERNEL (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void
radix32_tw_pack_walk_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix32_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

    __m256d stp_re[5], stp_im[5];
    for (int i = 0; i < 5; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[5], b_im[5];
    for (int i = 0; i < 5; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    /* 31*4 = 124 doubles × 2 = 1984 bytes — fits L1 easily */
    double __attribute__((aligned(32))) tw_re[31*4];
    double __attribute__((aligned(32))) tw_im[31*4];

    for (size_t blk = 0; blk < nb; blk++) {
        r32w2_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix32_tw_flat_dit_kernel_fwd_avx2(
            in_re + blk*128, in_im + blk*128,   /* 32*4=128 */
            out_re + blk*128, out_im + blk*128,
            tw_re, tw_im, 4);

        for (int i = 0; i < 5; i++) {
            __m256d nr = R32W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R32W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK KERNEL (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void
radix32_tw_pack_walk_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix32_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

    __m256d stp_re[5], stp_im[5];
    for (int i = 0; i < 5; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[5], b_im[5];
    for (int i = 0; i < 5; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(32))) tw_re[31*4];
    double __attribute__((aligned(32))) tw_im[31*4];

    for (size_t blk = 0; blk < nb; blk++) {
        r32w2_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix32_tw_flat_dit_kernel_bwd_avx2(
            in_re + blk*128, in_im + blk*128,
            out_re + blk*128, out_im + blk*128,
            tw_re, tw_im, 4);

        for (int i = 0; i < 5; i++) {
            __m256d nr = R32W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R32W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R32W2_CMUL_RE
#undef R32W2_CMUL_IM

#endif /* FFT_RADIX32_AVX2_TW_PACK_WALK_H */
