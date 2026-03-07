/**
 * @file fft_radix8_avx2_tw_pack_walk.h
 * @brief DFT-8 AVX2 packed data + walked twiddles (zero tw table)
 *
 * AVX2 variant: T=4 (__m256d). Same derivation tree as AVX-512.
 *
 * Requires: a header providing radix8_tw_dit_kernel_fwd/bwd_avx2
 */

#ifndef FFT_RADIX8_AVX2_TW_PACK_WALK_H
#define FFT_RADIX8_AVX2_TW_PACK_WALK_H

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
} radix8_walk_plan_avx2_t;

static inline void radix8_walk_plan_avx2_init(
    radix8_walk_plan_avx2_t *plan, size_t K)
{
    const size_t NN = 8 * K;
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

#ifndef R8W2_CMUL_DEFINED
#define R8W2_CMUL_DEFINED
#define R8W2_CMUL_RE(ar,ai,br,bi) \
    _mm256_fnmadd_pd(ai, bi, _mm256_mul_pd(ar, br))
#define R8W2_CMUL_IM(ar,ai,br,bi) \
    _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br))
#define R8W2_CSQ_RE(ar,ai) \
    _mm256_fmsub_pd(ar, ar, _mm256_mul_pd(ai, ai))
#define R8W2_CSQ_IM(ar,ai) \
    _mm256_mul_pd(_mm256_add_pd(ar, ar), ai)
#endif

__attribute__((target("avx2,fma")))
static inline void r8w2_derive_twiddles(
    const __m256d b_re[2], const __m256d b_im[2],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    _mm256_store_pd(&tw_re[0*4], b_re[0]); _mm256_store_pd(&tw_im[0*4], b_im[0]);
    _mm256_store_pd(&tw_re[1*4], b_re[1]); _mm256_store_pd(&tw_im[1*4], b_im[1]);

    __m256d w3r = R8W2_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m256d w3i = R8W2_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm256_store_pd(&tw_re[2*4], w3r); _mm256_store_pd(&tw_im[2*4], w3i);

    __m256d w4r = R8W2_CSQ_RE(b_re[1],b_im[1]);
    __m256d w4i = R8W2_CSQ_IM(b_re[1],b_im[1]);
    _mm256_store_pd(&tw_re[3*4], w4r); _mm256_store_pd(&tw_im[3*4], w4i);

    _mm256_store_pd(&tw_re[4*4], R8W2_CMUL_RE(b_re[0],b_im[0],w4r,w4i));
    _mm256_store_pd(&tw_im[4*4], R8W2_CMUL_IM(b_re[0],b_im[0],w4r,w4i));

    _mm256_store_pd(&tw_re[5*4], R8W2_CMUL_RE(b_re[1],b_im[1],w4r,w4i));
    _mm256_store_pd(&tw_im[5*4], R8W2_CMUL_IM(b_re[1],b_im[1],w4r,w4i));

    _mm256_store_pd(&tw_re[6*4], R8W2_CMUL_RE(w3r,w3i,w4r,w4i));
    _mm256_store_pd(&tw_im[6*4], R8W2_CMUL_IM(w3r,w3i,w4r,w4i));
}

__attribute__((target("avx2,fma")))
static inline void
radix8_tw_pack_walk_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix8_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

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

    double __attribute__((aligned(32))) tw_re[7*4];
    double __attribute__((aligned(32))) tw_im[7*4];

    for (size_t blk = 0; blk < nb; blk++) {
        r8w2_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix8_tw_dit_kernel_fwd_avx2(
            in_re + blk*32, in_im + blk*32,
            out_re + blk*32, out_im + blk*32,
            tw_re, tw_im, 4);

        for (int i = 0; i < 2; i++) {
            __m256d nr = R8W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R8W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

__attribute__((target("avx2,fma")))
static inline void
radix8_tw_pack_walk_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix8_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

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

    double __attribute__((aligned(32))) tw_re[7*4];
    double __attribute__((aligned(32))) tw_im[7*4];

    for (size_t blk = 0; blk < nb; blk++) {
        r8w2_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix8_tw_dit_kernel_bwd_avx2(
            in_re + blk*32, in_im + blk*32,
            out_re + blk*32, out_im + blk*32,
            tw_re, tw_im, 4);

        for (int i = 0; i < 2; i++) {
            __m256d nr = R8W2_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R8W2_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

#undef R8W2_CMUL_RE
#undef R8W2_CMUL_IM
#undef R8W2_CSQ_RE
#undef R8W2_CSQ_IM

#endif /* FFT_RADIX8_AVX2_TW_PACK_WALK_H */
