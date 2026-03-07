/**
 * @file fft_radix8_avx512_tw_pack_walk.h
 * @brief DFT-8 AVX-512 packed data + walked twiddles (zero tw table)
 *
 * Walks 2 base accumulators (W^1, W^2), derives 5 more via tree:
 *   W^3 = W^1 * W^2
 *   W^4 = (W^2)^2
 *   W^5 = W^1 * W^4
 *   W^6 = W^2 * W^4
 *   W^7 = W^3 * W^4
 *
 * Per block: 5 cmuls + 1 csquare (derivation) + 2 cmuls (walk step)
 * Total overhead: 8 cmul-equivalents per block.
 *
 * Requires: a header providing radix8_tw_dit_kernel_fwd/bwd_avx512
 */

#ifndef FFT_RADIX8_AVX512_TW_PACK_WALK_H
#define FFT_RADIX8_AVX512_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WALK PLAN (AVX-512, T=8)
 *
 * step[i] = W_{8K}^{pow[i]*8}
 * init[i][j] = W_{8K}^{pow[i]*j}  j=0..7
 * pow = {1, 2}
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double step_re[2];
    double step_im[2];
    double init_re[2][8];
    double init_im[2][8];
} radix8_walk_plan_t;

static inline void radix8_walk_plan_init(
    radix8_walk_plan_t *plan, size_t K)
{
    const size_t NN = 8 * K;
    const int pows[2] = {1, 2};
    for (int i = 0; i < 2; i++) {
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

#ifndef R8W_CMUL_DEFINED
#define R8W_CMUL_DEFINED
#define R8W_CMUL_RE(ar,ai,br,bi) \
    _mm512_fnmadd_pd(ai, bi, _mm512_mul_pd(ar, br))
#define R8W_CMUL_IM(ar,ai,br,bi) \
    _mm512_fmadd_pd(ar, bi, _mm512_mul_pd(ai, br))
/* csquare: (a+bi)^2 = (a^2-b^2) + 2ab*i */
#define R8W_CSQ_RE(ar,ai) \
    _mm512_fmsub_pd(ar, ar, _mm512_mul_pd(ai, ai))
#define R8W_CSQ_IM(ar,ai) \
    _mm512_mul_pd(_mm512_add_pd(ar, ar), ai)
#endif

/* ═══════════════════════════════════════════════════════════════
 * DERIVE 7 TWIDDLES FROM 2 BASES
 *
 * W^1 → slot 0       (direct)
 * W^2 → slot 1       (direct)
 * W^3 = W^1 * W^2    → slot 2
 * W^4 = (W^2)^2      → slot 3
 * W^5 = W^1 * W^4    → slot 4
 * W^6 = W^2 * W^4    → slot 5
 * W^7 = W^3 * W^4    → slot 6
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void r8w_derive_twiddles(
    const __m512d b_re[2], const __m512d b_im[2],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    /* Store direct bases */
    _mm512_store_pd(&tw_re[0*8], b_re[0]); _mm512_store_pd(&tw_im[0*8], b_im[0]);
    _mm512_store_pd(&tw_re[1*8], b_re[1]); _mm512_store_pd(&tw_im[1*8], b_im[1]);

    /* W^3 = W^1 * W^2 */
    __m512d w3r = R8W_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m512d w3i = R8W_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm512_store_pd(&tw_re[2*8], w3r); _mm512_store_pd(&tw_im[2*8], w3i);

    /* W^4 = (W^2)^2 */
    __m512d w4r = R8W_CSQ_RE(b_re[1],b_im[1]);
    __m512d w4i = R8W_CSQ_IM(b_re[1],b_im[1]);
    _mm512_store_pd(&tw_re[3*8], w4r); _mm512_store_pd(&tw_im[3*8], w4i);

    /* W^5 = W^1 * W^4 */
    _mm512_store_pd(&tw_re[4*8], R8W_CMUL_RE(b_re[0],b_im[0],w4r,w4i));
    _mm512_store_pd(&tw_im[4*8], R8W_CMUL_IM(b_re[0],b_im[0],w4r,w4i));

    /* W^6 = W^2 * W^4 */
    _mm512_store_pd(&tw_re[5*8], R8W_CMUL_RE(b_re[1],b_im[1],w4r,w4i));
    _mm512_store_pd(&tw_im[5*8], R8W_CMUL_IM(b_re[1],b_im[1],w4r,w4i));

    /* W^7 = W^3 * W^4 */
    _mm512_store_pd(&tw_re[6*8], R8W_CMUL_RE(w3r,w3i,w4r,w4i));
    _mm512_store_pd(&tw_im[6*8], R8W_CMUL_IM(w3r,w3i,w4r,w4i));
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix8_tw_pack_walk_fwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix8_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    __m512d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    /* 7*8 = 56 doubles × 2 = 896 bytes — fits L1 trivially */
    double __attribute__((aligned(64))) tw_re[7*8];
    double __attribute__((aligned(64))) tw_im[7*8];

    for (size_t blk = 0; blk < nb; blk++) {
        r8w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix8_tw_dit_kernel_fwd_avx512(
            in_re + blk*64, in_im + blk*64,
            out_re + blk*64, out_im + blk*64,
            tw_re, tw_im, 8);

        for (int i = 0; i < 2; i++) {
            __m512d nr = R8W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R8W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx512f,fma")))
static inline void
radix8_tw_pack_walk_bwd_avx512(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix8_walk_plan_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 8;

    __m512d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm512_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm512_set1_pd(plan->step_im[i]);
    }

    __m512d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm512_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm512_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(64))) tw_re[7*8];
    double __attribute__((aligned(64))) tw_im[7*8];

    for (size_t blk = 0; blk < nb; blk++) {
        r8w_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix8_tw_dit_kernel_bwd_avx512(
            in_re + blk*64, in_im + blk*64,
            out_re + blk*64, out_im + blk*64,
            tw_re, tw_im, 8);

        for (int i = 0; i < 2; i++) {
            __m512d nr = R8W_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m512d ni = R8W_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R8W_CMUL_RE
#undef R8W_CMUL_IM
#undef R8W_CSQ_RE
#undef R8W_CSQ_IM

#endif /* FFT_RADIX8_AVX512_TW_PACK_WALK_H */
