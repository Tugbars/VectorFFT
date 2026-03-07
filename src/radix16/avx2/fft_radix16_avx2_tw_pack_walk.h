/**
 * @file fft_radix16_avx2_tw_pack_walk.h
 * @brief DFT-16 AVX2 packed data + walked twiddles (zero tw table)
 *
 * Same algorithm as AVX-512 version but T=4 (__m256d, k-step=4).
 * 4 walking bases (W^1,W^2,W^4,W^8), 10 cmuls to derive 15 twiddles.
 *
 * REQUIRES: include the flat AVX2 tw kernel before this header:
 *   #include "fft_radix16_avx2_tw.h"
 *   — provides radix16_tw_flat_dit_kernel_{fwd,bwd}_avx2(...)
 */

#ifndef FFT_RADIX16_AVX2_TW_PACK_WALK_H
#define FFT_RADIX16_AVX2_TW_PACK_WALK_H

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
    double step_re[4];
    double step_im[4];
    double init_re[4][4];   /* 4 doubles per lane */
    double init_im[4][4];
} radix16_walk_plan_avx2_t;

__attribute__((target("avx2,fma")))
static inline void radix16_walk_plan_avx2_init(
    radix16_walk_plan_avx2_t *plan, size_t K)
{
    const size_t NN = 16 * K;
    const int pows[4] = {1, 2, 4, 8};
    for (int i = 0; i < 4; i++) {
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
 * COMPLEX MULTIPLY HELPERS
 * ═══════════════════════════════════════════════════════════════ */

#ifndef R16WA_CMUL_DEFINED
#define R16WA_CMUL_DEFINED
#define R16WA_CMUL_RE(ar,ai,br,bi) \
    _mm256_fnmadd_pd(ai, bi, _mm256_mul_pd(ar, br))
#define R16WA_CMUL_IM(ar,ai,br,bi) \
    _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br))
#endif

/* ═══════════════════════════════════════════════════════════════
 * DERIVE 15 TWIDDLES FROM 4 BASES (AVX2, T=4)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void r16wa_derive_twiddles(
    const __m256d b_re[4], const __m256d b_im[4],
    double * __restrict__ tw_re, double * __restrict__ tw_im)
{
    _mm256_store_pd(&tw_re[0*4], b_re[0]);  _mm256_store_pd(&tw_im[0*4], b_im[0]);
    _mm256_store_pd(&tw_re[1*4], b_re[1]);  _mm256_store_pd(&tw_im[1*4], b_im[1]);
    _mm256_store_pd(&tw_re[3*4], b_re[2]);  _mm256_store_pd(&tw_im[3*4], b_im[2]);
    _mm256_store_pd(&tw_re[7*4], b_re[3]);  _mm256_store_pd(&tw_im[7*4], b_im[3]);

    __m256d w3r = R16WA_CMUL_RE(b_re[0],b_im[0],b_re[1],b_im[1]);
    __m256d w3i = R16WA_CMUL_IM(b_re[0],b_im[0],b_re[1],b_im[1]);
    _mm256_store_pd(&tw_re[2*4], w3r);  _mm256_store_pd(&tw_im[2*4], w3i);

    __m256d w5r = R16WA_CMUL_RE(b_re[0],b_im[0],b_re[2],b_im[2]);
    __m256d w5i = R16WA_CMUL_IM(b_re[0],b_im[0],b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[4*4], w5r);  _mm256_store_pd(&tw_im[4*4], w5i);

    __m256d w6r = R16WA_CMUL_RE(b_re[1],b_im[1],b_re[2],b_im[2]);
    __m256d w6i = R16WA_CMUL_IM(b_re[1],b_im[1],b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[5*4], w6r);  _mm256_store_pd(&tw_im[5*4], w6i);

    __m256d w7r = R16WA_CMUL_RE(w3r,w3i,b_re[2],b_im[2]);
    __m256d w7i = R16WA_CMUL_IM(w3r,w3i,b_re[2],b_im[2]);
    _mm256_store_pd(&tw_re[6*4], w7r);  _mm256_store_pd(&tw_im[6*4], w7i);

    for (int n = 9; n <= 15; n++) {
        __m256d sr = _mm256_load_pd(&tw_re[(n-9)*4]);
        __m256d si = _mm256_load_pd(&tw_im[(n-9)*4]);
        _mm256_store_pd(&tw_re[(n-1)*4], R16WA_CMUL_RE(sr,si,b_re[3],b_im[3]));
        _mm256_store_pd(&tw_im[(n-1)*4], R16WA_CMUL_IM(sr,si,b_re[3],b_im[3]));
    }
}

/* ═══════════════════════════════════════════════════════════════
 * FORWARD PACK+WALK (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void
radix16_tw_pack_walk_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix16_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

    __m256d stp_re[4], stp_im[4];
    for (int i = 0; i < 4; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[4], b_im[4];
    for (int i = 0; i < 4; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    /* 15*4 = 60 doubles × 2 = 480 bytes — fits L1 */
    double __attribute__((aligned(32))) tw_re[60];
    double __attribute__((aligned(32))) tw_im[60];

    for (size_t blk = 0; blk < nb; blk++) {
        r16wa_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix16_tw_flat_dit_kernel_fwd_avx2(
            in_re + blk*64, in_im + blk*64,   /* 16*4 = 64 */
            out_re + blk*64, out_im + blk*64,
            tw_re, tw_im, 4);

        for (int i = 0; i < 4; i++) {
            __m256d nr = R16WA_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R16WA_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BACKWARD PACK+WALK (AVX2)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void
radix16_tw_pack_walk_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const radix16_walk_plan_avx2_t * __restrict__ plan,
    size_t K)
{
    const size_t nb = K / 4;

    __m256d stp_re[4], stp_im[4];
    for (int i = 0; i < 4; i++) {
        stp_re[i] = _mm256_set1_pd(plan->step_re[i]);
        stp_im[i] = _mm256_set1_pd(plan->step_im[i]);
    }

    __m256d b_re[4], b_im[4];
    for (int i = 0; i < 4; i++) {
        b_re[i] = _mm256_loadu_pd(plan->init_re[i]);
        b_im[i] = _mm256_loadu_pd(plan->init_im[i]);
    }

    double __attribute__((aligned(32))) tw_re[60];
    double __attribute__((aligned(32))) tw_im[60];

    for (size_t blk = 0; blk < nb; blk++) {
        r16wa_derive_twiddles(b_re, b_im, tw_re, tw_im);

        radix16_tw_flat_dit_kernel_bwd_avx2(
            in_re + blk*64, in_im + blk*64,
            out_re + blk*64, out_im + blk*64,
            tw_re, tw_im, 4);

        for (int i = 0; i < 4; i++) {
            __m256d nr = R16WA_CMUL_RE(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            __m256d ni = R16WA_CMUL_IM(b_re[i],b_im[i],stp_re[i],stp_im[i]);
            b_re[i] = nr;
            b_im[i] = ni;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * AVX2 PACK/UNPACK (T=4)
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2")))
static inline void radix16_pack_input_avx2(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 4, dk = b * 64;
        for (int n = 0; n < 16; n++) {
            _mm256_storeu_pd(&dst_re[dk + n*4], _mm256_loadu_pd(&src_re[n*K + sk]));
            _mm256_storeu_pd(&dst_im[dk + n*4], _mm256_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

__attribute__((target("avx2")))
static inline void radix16_unpack_output_avx2(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 64, dk = b * 4;
        for (int n = 0; n < 16; n++) {
            _mm256_storeu_pd(&dst_re[n*K + dk], _mm256_loadu_pd(&src_re[sk + n*4]));
            _mm256_storeu_pd(&dst_im[n*K + dk], _mm256_loadu_pd(&src_im[sk + n*4]));
        }
    }
}

__attribute__((target("avx2")))
static inline void radix16_pack_twiddles_avx2(
    const double * __restrict__ src_re, const double * __restrict__ src_im,
    double * __restrict__ dst_re, double * __restrict__ dst_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++) {
        const size_t sk = b * 4, dk = b * 60; /* 15*4 */
        for (int n = 0; n < 15; n++) {
            _mm256_storeu_pd(&dst_re[dk + n*4], _mm256_loadu_pd(&src_re[n*K + sk]));
            _mm256_storeu_pd(&dst_im[dk + n*4], _mm256_loadu_pd(&src_im[n*K + sk]));
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * AVX2 PACKED TABLE DRIVERS
 * ═══════════════════════════════════════════════════════════════ */

__attribute__((target("avx2,fma")))
static inline void r16_tw_packed_fwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_fwd_avx2(
            in_re + b*64, in_im + b*64,
            out_re + b*64, out_im + b*64,
            tw_re + b*60, tw_im + b*60, 4);
}

__attribute__((target("avx2,fma")))
static inline void r16_tw_packed_bwd_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ tw_re, const double * __restrict__ tw_im,
    size_t K)
{
    const size_t nb = K / 4;
    for (size_t b = 0; b < nb; b++)
        radix16_tw_flat_dit_kernel_bwd_avx2(
            in_re + b*64, in_im + b*64,
            out_re + b*64, out_im + b*64,
            tw_re + b*60, tw_im + b*60, 4);
}

#undef R16WA_CMUL_RE
#undef R16WA_CMUL_IM

#endif /* FFT_RADIX16_AVX2_TW_PACK_WALK_H */
