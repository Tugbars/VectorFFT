/**
 * @file fft_radix11_avx2_tw_pack_walk.h
 * @brief DFT-11 AVX2 packed data + walked twiddles (zero tw table)
 *
 * AVX2 variant: T=4 (__m256d). Same 2-base derivation tree as AVX-512.
 *
 * Requires: fft_radix11_genfft.h included first (for notw kernel)
 */

#ifndef FFT_RADIX11_AVX2_TW_PACK_WALK_H
#define FFT_RADIX11_AVX2_TW_PACK_WALK_H

#include <stddef.h>
#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double step_re[2];
    double step_im[2];
    double init_re[2][4];
    double init_im[2][4];
} radix11_walk_plan_avx2_t;

static inline void radix11_walk_plan_avx2_init(
    radix11_walk_plan_avx2_t *plan, size_t K)
{
    const size_t NN = 11 * K;
    const int pows[2] = {1, 2};
    for (int i = 0; i < 2; i++) {
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

#ifndef R11W2_CMUL_DEFINED
#define R11W2_CMUL_DEFINED
#define R11W2_CMUL_RE(ar,ai,br,bi) \
    _mm256_fnmadd_pd(ai, bi, _mm256_mul_pd(ar, br))
#define R11W2_CMUL_IM(ar,ai,br,bi) \
    _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br))
#define R11W2_CSQ_RE(ar,ai) \
    _mm256_fmsub_pd(ar, ar, _mm256_mul_pd(ai, ai))
#define R11W2_CSQ_IM(ar,ai) \
    _mm256_mul_pd(_mm256_add_pd(ar, ar), ai)
#endif

__attribute__((target("avx2,fma")))
static inline void
radix11_tw_pack_walk_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix11_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t T = 4, bs = 11 * T, nb = K / T;

    __m256d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(32))) tw_blk_re[11 * 4];
    double __attribute__((aligned(32))) tw_blk_im[11 * 4];

    for (size_t blk = 0; blk < nb; blk++) {
        const double *blk_ir = in_re + blk * bs;
        const double *blk_ii = in_im + blk * bs;

        /* Derive 10 twiddles from 2 bases */
        __m256d tw_r[10], tw_i[10];
        tw_r[0] = b_re[0]; tw_i[0] = b_im[0];
        tw_r[1] = b_re[1]; tw_i[1] = b_im[1];

        tw_r[2] = R11W2_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
        tw_i[2] = R11W2_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);

        __m256d w4r = R11W2_CSQ_RE(b_re[1],b_im[1]);
        __m256d w4i = R11W2_CSQ_IM(b_re[1],b_im[1]);
        tw_r[3] = w4r; tw_i[3] = w4i;

        tw_r[4] = R11W2_CMUL_RE(b_re[0],b_im[0],w4r,w4i);
        tw_i[4] = R11W2_CMUL_IM(b_re[0],b_im[0],w4r,w4i);

        tw_r[5] = R11W2_CMUL_RE(b_re[1],b_im[1],w4r,w4i);
        tw_i[5] = R11W2_CMUL_IM(b_re[1],b_im[1],w4r,w4i);

        tw_r[6] = R11W2_CMUL_RE(tw_r[2],tw_i[2],w4r,w4i);
        tw_i[6] = R11W2_CMUL_IM(tw_r[2],tw_i[2],w4r,w4i);

        __m256d w8r = R11W2_CSQ_RE(w4r,w4i);
        __m256d w8i = R11W2_CSQ_IM(w4r,w4i);
        tw_r[7] = w8r; tw_i[7] = w8i;

        tw_r[8] = R11W2_CMUL_RE(b_re[0],b_im[0],w8r,w8i);
        tw_i[8] = R11W2_CMUL_IM(b_re[0],b_im[0],w8r,w8i);

        tw_r[9] = R11W2_CMUL_RE(b_re[1],b_im[1],w8r,w8i);
        tw_i[9] = R11W2_CMUL_IM(b_re[1],b_im[1],w8r,w8i);

        /* Apply twiddles */
        _mm256_store_pd(&tw_blk_re[0], _mm256_load_pd(&blk_ir[0]));
        _mm256_store_pd(&tw_blk_im[0], _mm256_load_pd(&blk_ii[0]));

        for (int n = 0; n < 10; n++) {
            __m256d xr = _mm256_load_pd(&blk_ir[(n+1) * T]);
            __m256d xi = _mm256_load_pd(&blk_ii[(n+1) * T]);
            _mm256_store_pd(&tw_blk_re[(n+1) * T],
                R11W2_CMUL_RE(xr, xi, tw_r[n], tw_i[n]));
            _mm256_store_pd(&tw_blk_im[(n+1) * T],
                R11W2_CMUL_IM(xr, xi, tw_r[n], tw_i[n]));
        }

        radix11_genfft_fwd_avx2(tw_blk_re, tw_blk_im,
                                out_re + blk * bs, out_im + blk * bs, T);

        for (int i = 0; i < 2; i++) {
            __m256d nr = R11W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R11W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

__attribute__((target("avx2,fma")))
static inline void
radix11_tw_pack_walk_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix11_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t T = 4, bs = 11 * T, nb = K / T;
    const __m256d neg = _mm256_set1_pd(-0.0);

    __m256d stp_re[2], stp_im[2];
    for (int i = 0; i < 2; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[2], b_im[2];
    for (int i = 0; i < 2; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(32))) tw_blk_re[11 * 4];
    double __attribute__((aligned(32))) tw_blk_im[11 * 4];

    for (size_t blk = 0; blk < nb; blk++) {
        const double *blk_ir = in_re + blk * bs;
        const double *blk_ii = in_im + blk * bs;

        /* Conjugate bases for backward */
        __m256d cb_re0 = b_re[0], cb_im0 = _mm256_xor_pd(b_im[0], neg);
        __m256d cb_re1 = b_re[1], cb_im1 = _mm256_xor_pd(b_im[1], neg);

        __m256d tw_r[10], tw_i[10];
        tw_r[0] = cb_re0; tw_i[0] = cb_im0;
        tw_r[1] = cb_re1; tw_i[1] = cb_im1;

        tw_r[2] = R11W2_CMUL_RE(cb_re0,cb_im0,cb_re1,cb_im1);
        tw_i[2] = R11W2_CMUL_IM(cb_re0,cb_im0,cb_re1,cb_im1);

        __m256d w4r = R11W2_CSQ_RE(cb_re1,cb_im1);
        __m256d w4i = R11W2_CSQ_IM(cb_re1,cb_im1);
        tw_r[3] = w4r; tw_i[3] = w4i;

        tw_r[4] = R11W2_CMUL_RE(cb_re0,cb_im0,w4r,w4i);
        tw_i[4] = R11W2_CMUL_IM(cb_re0,cb_im0,w4r,w4i);

        tw_r[5] = R11W2_CMUL_RE(cb_re1,cb_im1,w4r,w4i);
        tw_i[5] = R11W2_CMUL_IM(cb_re1,cb_im1,w4r,w4i);

        tw_r[6] = R11W2_CMUL_RE(tw_r[2],tw_i[2],w4r,w4i);
        tw_i[6] = R11W2_CMUL_IM(tw_r[2],tw_i[2],w4r,w4i);

        __m256d w8r = R11W2_CSQ_RE(w4r,w4i);
        __m256d w8i = R11W2_CSQ_IM(w4r,w4i);
        tw_r[7] = w8r; tw_i[7] = w8i;

        tw_r[8] = R11W2_CMUL_RE(cb_re0,cb_im0,w8r,w8i);
        tw_i[8] = R11W2_CMUL_IM(cb_re0,cb_im0,w8r,w8i);

        tw_r[9] = R11W2_CMUL_RE(cb_re1,cb_im1,w8r,w8i);
        tw_i[9] = R11W2_CMUL_IM(cb_re1,cb_im1,w8r,w8i);

        _mm256_store_pd(&tw_blk_re[0], _mm256_load_pd(&blk_ir[0]));
        _mm256_store_pd(&tw_blk_im[0], _mm256_load_pd(&blk_ii[0]));

        for (int n = 0; n < 10; n++) {
            __m256d xr = _mm256_load_pd(&blk_ir[(n+1) * T]);
            __m256d xi = _mm256_load_pd(&blk_ii[(n+1) * T]);
            _mm256_store_pd(&tw_blk_re[(n+1) * T],
                R11W2_CMUL_RE(xr, xi, tw_r[n], tw_i[n]));
            _mm256_store_pd(&tw_blk_im[(n+1) * T],
                R11W2_CMUL_IM(xr, xi, tw_r[n], tw_i[n]));
        }

        radix11_genfft_bwd_avx2(tw_blk_re, tw_blk_im,
                                out_re + blk * bs, out_im + blk * bs, T);

        for (int i = 0; i < 2; i++) {
            __m256d nr = R11W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R11W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R11W2_CMUL_RE
#undef R11W2_CMUL_IM
#undef R11W2_CSQ_RE
#undef R11W2_CSQ_IM

#endif /* FFT_RADIX11_AVX2_TW_PACK_WALK_H */