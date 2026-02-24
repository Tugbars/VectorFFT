/**
 * @file  fft_radix7_avx2.h
 * @brief Radix-7 FFT butterfly using Rader's algorithm — AVX2 (4-wide)
 *
 * OPTIMIZATIONS:
 *   P0: Pre-split Rader broadcasts — kernel constants as aligned double[4]
 *       quads.  _mm256_load_pd (ports 2/3) replaces broadcast_sd (port 5).
 *       Frees port 5 for DFT6 shuffle work.  (~8-10% gain)
 *
 *   P0: Round-robin convolution — interleaves mul phases of independent
 *       complex multiplies across 3-slot waves.  Saturates both FMA ports,
 *       eliminates 2-cycle stalls between dependent pairs.  (~10-15% gain)
 *
 *   P1: Tree y0 sum — binary tree DC accumulation.  6-deep add chain → 3
 *       levels.  (~1-2% gain)
 *
 * Register budget (16 YMM on AVX2):
 *   12 DFT6 working set + 2 x0 + 2 temps = 16 at peak.
 *   No U=2 pipeline (needs ~24 YMM → AVX-512 only).
 */

#ifndef FFT_RADIX7_AVX2_H
#define FFT_RADIX7_AVX2_H

#ifdef __AVX2__
#include <immintrin.h>

#define R7_AVX2_VW 4

/* ================================================================== */
/*  Pre-split constant quads  (P0: load via p2/3 not broadcast p5)     */
/* ================================================================== */

#define R7Q(name, v) \
    static const double __attribute__((aligned(32))) name[4] = {v, v, v, v}

/* DFT3 */
R7Q(R7_C3_Q, -0.5);
R7Q(R7_S3_Q, 0.86602540378443864676);

/* W6 forward twiddles */
R7Q(R7_W61R_Q, 0.5);
R7Q(R7_W61I_Q, -0.86602540378443864676);
R7Q(R7_W62R_Q, -0.5);
R7Q(R7_W62I_Q, -0.86602540378443864676);

/* W6 backward (conjugate) twiddles */
R7Q(R7_W61I_NEG_Q, 0.86602540378443864676);
R7Q(R7_W62I_NEG_Q, 0.86602540378443864676);

/* Forward Rader kernel: DFT6(time_reverse(b_fwd))/6 — 12 quads */
R7Q(RK7_FR0, -1.66666666666666657415e-01);
R7Q(RK7_FR1, +4.06688893057589595514e-01);
R7Q(RK7_FR2, +3.95078234262700001000e-01);
R7Q(RK7_FR3, +1.85037170770859413132e-17);
R7Q(RK7_FR4, +3.95078234262699945489e-01);
R7Q(RK7_FR5, -4.06688893057589151425e-01);
R7Q(RK7_FI0, -9.25185853854297065662e-18);
R7Q(RK7_FI1, -1.70436465311965518188e-01);
R7Q(RK7_FI2, -1.95851048647464526242e-01);
R7Q(RK7_FI3, -4.40958551844098212147e-01);
R7Q(RK7_FI4, +1.95851048647464942576e-01);
R7Q(RK7_FI5, -1.70436465311965684721e-01);

/* Backward Rader kernel: DFT6(time_reverse(b_bwd))/6 — 12 quads */
R7Q(RK7_BR0, -1.66666666666666657415e-01);
R7Q(RK7_BR1, -4.06688893057589595514e-01);
R7Q(RK7_BR2, +3.95078234262700167534e-01);
R7Q(RK7_BR3, +9.25185853854297158106e-17);
R7Q(RK7_BR4, +3.95078234262700223045e-01);
R7Q(RK7_BR5, +4.06688893057589540003e-01);
R7Q(RK7_BI0, +9.25185853854297065662e-18);
R7Q(RK7_BI1, +1.70436465311965656966e-01);
R7Q(RK7_BI2, -1.95851048647464304198e-01);
R7Q(RK7_BI3, +4.40958551844098767258e-01);
R7Q(RK7_BI4, +1.95851048647464359709e-01);
R7Q(RK7_BI5, +1.70436465311965323899e-01);

#undef R7Q

/* Scalar kernel arrays — for scalar tail fallbacks */
static const double RK7A_FWD_RE[6] = {
    -1.66666666666666657415e-01, +4.06688893057589595514e-01,
    +3.95078234262700001000e-01, +1.85037170770859413132e-17,
    +3.95078234262699945489e-01, -4.06688893057589151425e-01};
static const double RK7A_FWD_IM[6] = {
    -9.25185853854297065662e-18, -1.70436465311965518188e-01,
    -1.95851048647464526242e-01, -4.40958551844098212147e-01,
    +1.95851048647464942576e-01, -1.70436465311965684721e-01};
static const double RK7A_BWD_RE[6] = {
    -1.66666666666666657415e-01, -4.06688893057589595514e-01,
    +3.95078234262700167534e-01, +9.25185853854297158106e-17,
    +3.95078234262700223045e-01, +4.06688893057589540003e-01};
static const double RK7A_BWD_IM[6] = {
    +9.25185853854297065662e-18, +1.70436465311965656966e-01,
    -1.95851048647464304198e-01, +4.40958551844098767258e-01,
    +1.95851048647464359709e-01, +1.70436465311965323899e-01};

/* ================================================================== */
/*  AVX2 complex multiply (FMA)                                        */
/* ================================================================== */

#if defined(__FMA__)
static inline __attribute__((always_inline)) void cmul_avx2_r7(__m256d ar, __m256d ai, __m256d wr, __m256d wi,
                                                               __m256d *cr, __m256d *ci)
{
    __m256d t0 = _mm256_mul_pd(ai, wi);
    *cr = _mm256_fmsub_pd(ar, wr, t0);
    __m256d t1 = _mm256_mul_pd(ai, wr);
    *ci = _mm256_fmadd_pd(ar, wi, t1);
}
#else
static inline __attribute__((always_inline)) void cmul_avx2_r7(__m256d ar, __m256d ai, __m256d wr, __m256d wi,
                                                               __m256d *cr, __m256d *ci)
{
    *cr = _mm256_sub_pd(_mm256_mul_pd(ar, wr), _mm256_mul_pd(ai, wi));
    *ci = _mm256_add_pd(_mm256_mul_pd(ar, wi), _mm256_mul_pd(ai, wr));
}
#endif

/* ================================================================== */
/*  DFT6 forward — pre-split constant loads                            */
/* ================================================================== */

#define DFT6_FWD_AVX2(x0r, x0i, x1r, x1i, x2r, x2i,         \
                      x3r, x3i, x4r, x4i, x5r, x5i)         \
    do                                                      \
    {                                                       \
        __m256d vC3 = _mm256_load_pd(R7_C3_Q);              \
        __m256d vS3 = _mm256_load_pd(R7_S3_Q);              \
        __m256d se_r = _mm256_add_pd(x2r, x4r);             \
        __m256d se_i = _mm256_add_pd(x2i, x4i);             \
        __m256d de_r = _mm256_sub_pd(x2r, x4r);             \
        __m256d de_i = _mm256_sub_pd(x2i, x4i);             \
        __m256d E0r = _mm256_add_pd(x0r, se_r);             \
        __m256d E0i = _mm256_add_pd(x0i, se_i);             \
        __m256d be_r = _mm256_fmadd_pd(vC3, se_r, x0r);     \
        __m256d be_i = _mm256_fmadd_pd(vC3, se_i, x0i);     \
        __m256d E1r = _mm256_fmadd_pd(vS3, de_i, be_r);     \
        __m256d E1i = _mm256_fnmadd_pd(vS3, de_r, be_i);    \
        __m256d E2r = _mm256_fnmadd_pd(vS3, de_i, be_r);    \
        __m256d E2i = _mm256_fmadd_pd(vS3, de_r, be_i);     \
        __m256d so_r = _mm256_add_pd(x3r, x5r);             \
        __m256d so_i = _mm256_add_pd(x3i, x5i);             \
        __m256d do_r = _mm256_sub_pd(x3r, x5r);             \
        __m256d do_i = _mm256_sub_pd(x3i, x5i);             \
        __m256d O0r = _mm256_add_pd(x1r, so_r);             \
        __m256d O0i = _mm256_add_pd(x1i, so_i);             \
        __m256d bo_r = _mm256_fmadd_pd(vC3, so_r, x1r);     \
        __m256d bo_i = _mm256_fmadd_pd(vC3, so_i, x1i);     \
        __m256d O1r = _mm256_fmadd_pd(vS3, do_i, bo_r);     \
        __m256d O1i = _mm256_fnmadd_pd(vS3, do_r, bo_i);    \
        __m256d O2r = _mm256_fnmadd_pd(vS3, do_i, bo_r);    \
        __m256d O2i = _mm256_fmadd_pd(vS3, do_r, bo_i);     \
        x0r = _mm256_add_pd(E0r, O0r);                      \
        x0i = _mm256_add_pd(E0i, O0i);                      \
        x3r = _mm256_sub_pd(E0r, O0r);                      \
        x3i = _mm256_sub_pd(E0i, O0i);                      \
        {                                                   \
            __m256d vW1r = _mm256_load_pd(R7_W61R_Q);       \
            __m256d vW1i = _mm256_load_pd(R7_W61I_Q);       \
            __m256d T1r, T1i;                               \
            cmul_avx2_r7(O1r, O1i, vW1r, vW1i, &T1r, &T1i); \
            x1r = _mm256_add_pd(E1r, T1r);                  \
            x1i = _mm256_add_pd(E1i, T1i);                  \
            x4r = _mm256_sub_pd(E1r, T1r);                  \
            x4i = _mm256_sub_pd(E1i, T1i);                  \
        }                                                   \
        {                                                   \
            __m256d vW2r = _mm256_load_pd(R7_W62R_Q);       \
            __m256d vW2i = _mm256_load_pd(R7_W62I_Q);       \
            __m256d T2r, T2i;                               \
            cmul_avx2_r7(O2r, O2i, vW2r, vW2i, &T2r, &T2i); \
            x2r = _mm256_add_pd(E2r, T2r);                  \
            x2i = _mm256_add_pd(E2i, T2i);                  \
            x5r = _mm256_sub_pd(E2r, T2r);                  \
            x5i = _mm256_sub_pd(E2i, T2i);                  \
        }                                                   \
    } while (0)

/* ================================================================== */
/*  DFT6 backward — conjugate twiddles, pre-split loads                */
/* ================================================================== */

#define DFT6_BWD_AVX2(x0r, x0i, x1r, x1i, x2r, x2i,         \
                      x3r, x3i, x4r, x4i, x5r, x5i)         \
    do                                                      \
    {                                                       \
        __m256d vC3 = _mm256_load_pd(R7_C3_Q);              \
        __m256d vS3 = _mm256_load_pd(R7_S3_Q);              \
        __m256d se_r = _mm256_add_pd(x2r, x4r);             \
        __m256d se_i = _mm256_add_pd(x2i, x4i);             \
        __m256d de_r = _mm256_sub_pd(x2r, x4r);             \
        __m256d de_i = _mm256_sub_pd(x2i, x4i);             \
        __m256d E0r = _mm256_add_pd(x0r, se_r);             \
        __m256d E0i = _mm256_add_pd(x0i, se_i);             \
        __m256d be_r = _mm256_fmadd_pd(vC3, se_r, x0r);     \
        __m256d be_i = _mm256_fmadd_pd(vC3, se_i, x0i);     \
        __m256d E1r = _mm256_fnmadd_pd(vS3, de_i, be_r);    \
        __m256d E1i = _mm256_fmadd_pd(vS3, de_r, be_i);     \
        __m256d E2r = _mm256_fmadd_pd(vS3, de_i, be_r);     \
        __m256d E2i = _mm256_fnmadd_pd(vS3, de_r, be_i);    \
        __m256d so_r = _mm256_add_pd(x3r, x5r);             \
        __m256d so_i = _mm256_add_pd(x3i, x5i);             \
        __m256d do_r = _mm256_sub_pd(x3r, x5r);             \
        __m256d do_i = _mm256_sub_pd(x3i, x5i);             \
        __m256d O0r = _mm256_add_pd(x1r, so_r);             \
        __m256d O0i = _mm256_add_pd(x1i, so_i);             \
        __m256d bo_r = _mm256_fmadd_pd(vC3, so_r, x1r);     \
        __m256d bo_i = _mm256_fmadd_pd(vC3, so_i, x1i);     \
        __m256d O1r = _mm256_fnmadd_pd(vS3, do_i, bo_r);    \
        __m256d O1i = _mm256_fmadd_pd(vS3, do_r, bo_i);     \
        __m256d O2r = _mm256_fmadd_pd(vS3, do_i, bo_r);     \
        __m256d O2i = _mm256_fnmadd_pd(vS3, do_r, bo_i);    \
        x0r = _mm256_add_pd(E0r, O0r);                      \
        x0i = _mm256_add_pd(E0i, O0i);                      \
        x3r = _mm256_sub_pd(E0r, O0r);                      \
        x3i = _mm256_sub_pd(E0i, O0i);                      \
        {                                                   \
            __m256d vW1r = _mm256_load_pd(R7_W61R_Q);       \
            __m256d vW1i = _mm256_load_pd(R7_W61I_NEG_Q);   \
            __m256d T1r, T1i;                               \
            cmul_avx2_r7(O1r, O1i, vW1r, vW1i, &T1r, &T1i); \
            x1r = _mm256_add_pd(E1r, T1r);                  \
            x1i = _mm256_add_pd(E1i, T1i);                  \
            x4r = _mm256_sub_pd(E1r, T1r);                  \
            x4i = _mm256_sub_pd(E1i, T1i);                  \
        }                                                   \
        {                                                   \
            __m256d vW2r = _mm256_load_pd(R7_W62R_Q);       \
            __m256d vW2i = _mm256_load_pd(R7_W62I_NEG_Q);   \
            __m256d T2r, T2i;                               \
            cmul_avx2_r7(O2r, O2i, vW2r, vW2i, &T2r, &T2i); \
            x2r = _mm256_add_pd(E2r, T2r);                  \
            x2i = _mm256_add_pd(E2i, T2i);                  \
            x5r = _mm256_sub_pd(E2r, T2r);                  \
            x5i = _mm256_sub_pd(E2i, T2i);                  \
        }                                                   \
    } while (0)

/* ================================================================== */
/*  Round-robin pointwise multiply  (P0: interleaved FMA schedule)     */
/*                                                                     */
/*  Wave 1 (slots 0,1,2): all mul_pd first, then all fmsub/fmadd      */
/*  Wave 2 (slots 3,4,5): same                                        */
/*  Within each wave, independent ops fill FMA pipeline bubbles.       */
/* ================================================================== */

#define POINTWISE_RR6(s0r, s0i, s1r, s1i, s2r, s2i,                           \
                      s3r, s3i, s4r, s4i, s5r, s5i,                           \
                      KR0, KI0, KR1, KI1, KR2, KI2,                           \
                      KR3, KI3, KR4, KI4, KR5, KI5)                           \
    do                                                                        \
    {                                                                         \
        /* Wave 1: load kernels */                                            \
        __m256d k0r = _mm256_load_pd(KR0), k0i = _mm256_load_pd(KI0);         \
        __m256d k1r = _mm256_load_pd(KR1), k1i = _mm256_load_pd(KI1);         \
        __m256d k2r = _mm256_load_pd(KR2), k2i = _mm256_load_pd(KI2);         \
        /* Mul phase: ai*wi and ai*wr for slots 0,1,2 */                      \
        __m256d p0a = _mm256_mul_pd(s0i, k0i), p0b = _mm256_mul_pd(s0i, k0r); \
        __m256d p1a = _mm256_mul_pd(s1i, k1i), p1b = _mm256_mul_pd(s1i, k1r); \
        __m256d p2a = _mm256_mul_pd(s2i, k2i), p2b = _mm256_mul_pd(s2i, k2r); \
        /* FMA phase: cr=ar*wr-ai*wi, ci=ar*wi+ai*wr */                       \
        __m256d r0r = _mm256_fmsub_pd(s0r, k0r, p0a);                         \
        __m256d r0i = _mm256_fmadd_pd(s0r, k0i, p0b);                         \
        __m256d r1r = _mm256_fmsub_pd(s1r, k1r, p1a);                         \
        __m256d r1i = _mm256_fmadd_pd(s1r, k1i, p1b);                         \
        __m256d r2r = _mm256_fmsub_pd(s2r, k2r, p2a);                         \
        __m256d r2i = _mm256_fmadd_pd(s2r, k2i, p2b);                         \
        s0r = r0r;                                                            \
        s0i = r0i;                                                            \
        s1r = r1r;                                                            \
        s1i = r1i;                                                            \
        s2r = r2r;                                                            \
        s2i = r2i;                                                            \
        /* Wave 2: load kernels */                                            \
        __m256d k3r = _mm256_load_pd(KR3), k3i = _mm256_load_pd(KI3);         \
        __m256d k4r = _mm256_load_pd(KR4), k4i = _mm256_load_pd(KI4);         \
        __m256d k5r = _mm256_load_pd(KR5), k5i = _mm256_load_pd(KI5);         \
        __m256d p3a = _mm256_mul_pd(s3i, k3i), p3b = _mm256_mul_pd(s3i, k3r); \
        __m256d p4a = _mm256_mul_pd(s4i, k4i), p4b = _mm256_mul_pd(s4i, k4r); \
        __m256d p5a = _mm256_mul_pd(s5i, k5i), p5b = _mm256_mul_pd(s5i, k5r); \
        __m256d r3r = _mm256_fmsub_pd(s3r, k3r, p3a);                         \
        __m256d r3i = _mm256_fmadd_pd(s3r, k3i, p3b);                         \
        __m256d r4r = _mm256_fmsub_pd(s4r, k4r, p4a);                         \
        __m256d r4i = _mm256_fmadd_pd(s4r, k4i, p4b);                         \
        __m256d r5r = _mm256_fmsub_pd(s5r, k5r, p5a);                         \
        __m256d r5i = _mm256_fmadd_pd(s5r, k5i, p5b);                         \
        s3r = r3r;                                                            \
        s3i = r3i;                                                            \
        s4r = r4r;                                                            \
        s4i = r4i;                                                            \
        s5r = r5r;                                                            \
        s5i = r5i;                                                            \
    } while (0)

/* ================================================================== */
/*  Tree y0 sum  (P1: 6-deep chain → 3 levels)                        */
/* ================================================================== */

#define TREE_Y0(x0, t1, t2, t3, t4, t5, t6, out) \
    do                                           \
    {                                            \
        __m256d _a = _mm256_add_pd(x0, t1);      \
        __m256d _b = _mm256_add_pd(t2, t3);      \
        __m256d _c = _mm256_add_pd(t4, t5);      \
        __m256d _d = _mm256_add_pd(_a, _b);      \
        __m256d _e = _mm256_add_pd(_c, t6);      \
        out = _mm256_add_pd(_d, _e);             \
    } while (0)

/* ================================================================== */
/*  Forward butterfly — AVX2, BLOCKED3 twiddles, all optimizations     */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx2,fma"))) void radix7_rader_fwd_avx2(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    const double *restrict tw3_re, const double *restrict tw3_im,
    int K)
{
    int k = 0;
    for (; k + R7_AVX2_VW <= K; k += R7_AVX2_VW)
    {
        __m256d x0r = _mm256_loadu_pd(&a_re[k]);
        __m256d x0i = _mm256_loadu_pd(&a_im[k]);
        __m256d xb_r = _mm256_loadu_pd(&b_re[k]);
        __m256d xb_i = _mm256_loadu_pd(&b_im[k]);
        __m256d xc_r = _mm256_loadu_pd(&c_re[k]);
        __m256d xc_i = _mm256_loadu_pd(&c_im[k]);
        __m256d xd_r = _mm256_loadu_pd(&d_re[k]);
        __m256d xd_i = _mm256_loadu_pd(&d_im[k]);
        __m256d xe_r = _mm256_loadu_pd(&e_re[k]);
        __m256d xe_i = _mm256_loadu_pd(&e_im[k]);
        __m256d xf_r = _mm256_loadu_pd(&f_re[k]);
        __m256d xf_i = _mm256_loadu_pd(&f_im[k]);
        __m256d xg_r = _mm256_loadu_pd(&g_re[k]);
        __m256d xg_i = _mm256_loadu_pd(&g_im[k]);

        __m256d w1r = _mm256_loadu_pd(&tw1_re[k]);
        __m256d w1i = _mm256_loadu_pd(&tw1_im[k]);
        __m256d w2r = _mm256_loadu_pd(&tw2_re[k]);
        __m256d w2i = _mm256_loadu_pd(&tw2_im[k]);
        __m256d w3r = _mm256_loadu_pd(&tw3_re[k]);
        __m256d w3i = _mm256_loadu_pd(&tw3_im[k]);

        __m256d t1r, t1i, t2r, t2i, t3r, t3i;
        cmul_avx2_r7(xb_r, xb_i, w1r, w1i, &t1r, &t1i);
        cmul_avx2_r7(xc_r, xc_i, w2r, w2i, &t2r, &t2i);
        cmul_avx2_r7(xd_r, xd_i, w3r, w3i, &t3r, &t3i);

        __m256d w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_avx2_r7(w1r, w1i, w3r, w3i, &w4r, &w4i);
        cmul_avx2_r7(w2r, w2i, w3r, w3i, &w5r, &w5i);
        cmul_avx2_r7(w3r, w3i, w3r, w3i, &w6r, &w6i);

        __m256d t4r, t4i, t5r, t5i, t6r, t6i;
        cmul_avx2_r7(xe_r, xe_i, w4r, w4i, &t4r, &t4i);
        cmul_avx2_r7(xf_r, xf_i, w5r, w5i, &t5r, &t5i);
        cmul_avx2_r7(xg_r, xg_i, w6r, w6i, &t6r, &t6i);

        /* DC: tree sum (P1) */
        __m256d dc_r, dc_i;
        TREE_Y0(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dc_r);
        TREE_Y0(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dc_i);
        _mm256_storeu_pd(&y0_re[k], dc_r);
        _mm256_storeu_pd(&y0_im[k], dc_i);

        /* Rader input permute: {1,3,2,6,4,5} */
        __m256d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m256d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m256d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;

        DFT6_FWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i);

        POINTWISE_RR6(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i,
                      RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1,
                      RK7_FR2, RK7_FI2, RK7_FR3, RK7_FI3,
                      RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);

        DFT6_BWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i);

        /* inv_perm = {1,5,4,6,2,3} */
        _mm256_storeu_pd(&y1_re[k], _mm256_add_pd(x0r, s0r));
        _mm256_storeu_pd(&y1_im[k], _mm256_add_pd(x0i, s0i));
        _mm256_storeu_pd(&y5_re[k], _mm256_add_pd(x0r, s1r));
        _mm256_storeu_pd(&y5_im[k], _mm256_add_pd(x0i, s1i));
        _mm256_storeu_pd(&y4_re[k], _mm256_add_pd(x0r, s2r));
        _mm256_storeu_pd(&y4_im[k], _mm256_add_pd(x0i, s2i));
        _mm256_storeu_pd(&y6_re[k], _mm256_add_pd(x0r, s3r));
        _mm256_storeu_pd(&y6_im[k], _mm256_add_pd(x0i, s3i));
        _mm256_storeu_pd(&y2_re[k], _mm256_add_pd(x0r, s4r));
        _mm256_storeu_pd(&y2_im[k], _mm256_add_pd(x0i, s4i));
        _mm256_storeu_pd(&y3_re[k], _mm256_add_pd(x0r, s5r));
        _mm256_storeu_pd(&y3_im[k], _mm256_add_pd(x0i, s5i));
    }
    /* Scalar tail */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _w1r = tw1_re[k], _w1i = tw1_im[k];
        double _w2r = tw2_re[k], _w2i = tw2_im[k];
        double _w3r = tw3_re[k], _w3i = tw3_im[k];
        double _t1r = b_re[k] * _w1r - b_im[k] * _w1i, _t1i = b_re[k] * _w1i + b_im[k] * _w1r;
        double _t2r = c_re[k] * _w2r - c_im[k] * _w2i, _t2i = c_re[k] * _w2i + c_im[k] * _w2r;
        double _t3r = d_re[k] * _w3r - d_im[k] * _w3i, _t3i = d_re[k] * _w3i + d_im[k] * _w3r;
        double _w4r = _w1r * _w3r - _w1i * _w3i, _w4i = _w1r * _w3i + _w1i * _w3r;
        double _w5r = _w2r * _w3r - _w2i * _w3i, _w5i = _w2r * _w3i + _w2i * _w3r;
        double _w6r = _w3r * _w3r - _w3i * _w3i, _w6i = 2.0 * _w3r * _w3i;
        double _t4r = e_re[k] * _w4r - e_im[k] * _w4i, _t4i = e_re[k] * _w4i + e_im[k] * _w4r;
        double _t5r = f_re[k] * _w5r - f_im[k] * _w5i, _t5i = f_re[k] * _w5i + f_im[k] * _w5r;
        double _t6r = g_re[k] * _w6r - g_im[k] * _w6i, _t6i = g_re[k] * _w6i + g_im[k] * _w6r;
        y0_re[k] = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        y0_im[k] = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        /* Fwd DFT6 scalar */
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7A_FWD_RE[j] - Ai[j] * RK7A_FWD_IM[j];
            Ci[j] = Ar[j] * RK7A_FWD_IM[j] + Ai[j] * RK7A_FWD_RE[j];
        }
        /* Bwd DFT6 scalar */
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        y1_re[k] = x0r_ + cr[0];
        y1_im[k] = x0i_ + ci[0];
        y5_re[k] = x0r_ + cr[1];
        y5_im[k] = x0i_ + ci[1];
        y4_re[k] = x0r_ + cr[2];
        y4_im[k] = x0i_ + ci[2];
        y6_re[k] = x0r_ + cr[3];
        y6_im[k] = x0i_ + ci[3];
        y2_re[k] = x0r_ + cr[4];
        y2_im[k] = x0i_ + ci[4];
        y3_re[k] = x0r_ + cr[5];
        y3_im[k] = x0i_ + ci[5];
    }
}

/* ================================================================== */
/*  Backward butterfly — Rader IDFT then conj twiddle outputs          */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx2,fma"))) void radix7_rader_bwd_avx2(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    const double *restrict tw3_re, const double *restrict tw3_im,
    int K)
{
    int k = 0;
    for (; k + R7_AVX2_VW <= K; k += R7_AVX2_VW)
    {
        __m256d x0r = _mm256_loadu_pd(&a_re[k]);
        __m256d x0i = _mm256_loadu_pd(&a_im[k]);
        __m256d t1r = _mm256_loadu_pd(&b_re[k]);
        __m256d t1i = _mm256_loadu_pd(&b_im[k]);
        __m256d t2r = _mm256_loadu_pd(&c_re[k]);
        __m256d t2i = _mm256_loadu_pd(&c_im[k]);
        __m256d t3r = _mm256_loadu_pd(&d_re[k]);
        __m256d t3i = _mm256_loadu_pd(&d_im[k]);
        __m256d t4r = _mm256_loadu_pd(&e_re[k]);
        __m256d t4i = _mm256_loadu_pd(&e_im[k]);
        __m256d t5r = _mm256_loadu_pd(&f_re[k]);
        __m256d t5i = _mm256_loadu_pd(&f_im[k]);
        __m256d t6r = _mm256_loadu_pd(&g_re[k]);
        __m256d t6i = _mm256_loadu_pd(&g_im[k]);

        __m256d dc_r, dc_i;
        TREE_Y0(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dc_r);
        TREE_Y0(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dc_i);

        __m256d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m256d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m256d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;

        DFT6_FWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i);

        POINTWISE_RR6(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i,
                      RK7_BR0, RK7_BI0, RK7_BR1, RK7_BI1,
                      RK7_BR2, RK7_BI2, RK7_BR3, RK7_BI3,
                      RK7_BR4, RK7_BI4, RK7_BR5, RK7_BI5);

        DFT6_BWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i,
                      s3r, s3i, s4r, s4i, s5r, s5i);

        /* Raw IDFT7 outputs (before twiddle) */
        __m256d r1r = _mm256_add_pd(x0r, s0r), r1i = _mm256_add_pd(x0i, s0i);
        __m256d r5r = _mm256_add_pd(x0r, s1r), r5i = _mm256_add_pd(x0i, s1i);
        __m256d r4r = _mm256_add_pd(x0r, s2r), r4i = _mm256_add_pd(x0i, s2i);
        __m256d r6r = _mm256_add_pd(x0r, s3r), r6i = _mm256_add_pd(x0i, s3i);
        __m256d r2r = _mm256_add_pd(x0r, s4r), r2i = _mm256_add_pd(x0i, s4i);
        __m256d r3r = _mm256_add_pd(x0r, s5r), r3i = _mm256_add_pd(x0i, s5i);

        /* Apply conj twiddles AFTER IDFT */
        __m256d w1r = _mm256_loadu_pd(&tw1_re[k]);
        __m256d w1i = _mm256_loadu_pd(&tw1_im[k]);
        __m256d w2r = _mm256_loadu_pd(&tw2_re[k]);
        __m256d w2i = _mm256_loadu_pd(&tw2_im[k]);
        __m256d w3r = _mm256_loadu_pd(&tw3_re[k]);
        __m256d w3i = _mm256_loadu_pd(&tw3_im[k]);
        __m256d sb = _mm256_set1_pd(-0.0);
        __m256d w1n = _mm256_xor_pd(w1i, sb);
        __m256d w2n = _mm256_xor_pd(w2i, sb);
        __m256d w3n = _mm256_xor_pd(w3i, sb);
        __m256d w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_avx2_r7(w1r, w1n, w3r, w3n, &w4r, &w4i);
        cmul_avx2_r7(w2r, w2n, w3r, w3n, &w5r, &w5i);
        cmul_avx2_r7(w3r, w3n, w3r, w3n, &w6r, &w6i);

        _mm256_storeu_pd(&y0_re[k], dc_r);
        _mm256_storeu_pd(&y0_im[k], dc_i);
        __m256d o1r, o1i;
        cmul_avx2_r7(r1r, r1i, w1r, w1n, &o1r, &o1i);
        _mm256_storeu_pd(&y1_re[k], o1r);
        _mm256_storeu_pd(&y1_im[k], o1i);
        __m256d o2r, o2i;
        cmul_avx2_r7(r2r, r2i, w2r, w2n, &o2r, &o2i);
        _mm256_storeu_pd(&y2_re[k], o2r);
        _mm256_storeu_pd(&y2_im[k], o2i);
        __m256d o3r, o3i;
        cmul_avx2_r7(r3r, r3i, w3r, w3n, &o3r, &o3i);
        _mm256_storeu_pd(&y3_re[k], o3r);
        _mm256_storeu_pd(&y3_im[k], o3i);
        __m256d o4r, o4i;
        cmul_avx2_r7(r4r, r4i, w4r, w4i, &o4r, &o4i);
        _mm256_storeu_pd(&y4_re[k], o4r);
        _mm256_storeu_pd(&y4_im[k], o4i);
        __m256d o5r, o5i;
        cmul_avx2_r7(r5r, r5i, w5r, w5i, &o5r, &o5i);
        _mm256_storeu_pd(&y5_re[k], o5r);
        _mm256_storeu_pd(&y5_im[k], o5i);
        __m256d o6r, o6i;
        cmul_avx2_r7(r6r, r6i, w6r, w6i, &o6r, &o6i);
        _mm256_storeu_pd(&y6_re[k], o6r);
        _mm256_storeu_pd(&y6_im[k], o6i);
    }
    /* Scalar tail */
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _t1r = b_re[k], _t1i = b_im[k], _t2r = c_re[k], _t2i = c_im[k];
        double _t3r = d_re[k], _t3i = d_im[k], _t4r = e_re[k], _t4i = e_im[k];
        double _t5r = f_re[k], _t5i = f_im[k], _t6r = g_re[k], _t6i = g_im[k];
        double dcr_ = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        double dci_ = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Ar[6], Ai[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Ar[0] = E0r + O0r;
            Ai[0] = E0i + O0i;
            Ar[3] = E0r - O0r;
            Ai[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Ar[1] = E1r + T1r_;
            Ai[1] = E1i + T1i_;
            Ar[4] = E1r - T1r_;
            Ai[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Ar[2] = E2r + T2r_;
            Ai[2] = E2i + T2i_;
            Ar[5] = E2r - T2r_;
            Ai[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Ar[j] * RK7A_BWD_RE[j] - Ai[j] * RK7A_BWD_IM[j];
            Ci[j] = Ar[j] * RK7A_BWD_IM[j] + Ai[j] * RK7A_BWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        double _r1r = x0r_ + cr[0], _r1i = x0i_ + ci[0], _r5r = x0r_ + cr[1], _r5i = x0i_ + ci[1];
        double _r4r = x0r_ + cr[2], _r4i = x0i_ + ci[2], _r6r = x0r_ + cr[3], _r6i = x0i_ + ci[3];
        double _r2r = x0r_ + cr[4], _r2i = x0i_ + ci[4], _r3r = x0r_ + cr[5], _r3i = x0i_ + ci[5];
        double _w1r = tw1_re[k], _w1i = tw1_im[k];
        double _w2r = tw2_re[k], _w2i = tw2_im[k];
        double _w3r = tw3_re[k], _w3i = tw3_im[k];
        double _w4r = _w1r * _w3r - _w1i * _w3i, _w4i = _w1r * _w3i + _w1i * _w3r;
        double _w5r = _w2r * _w3r - _w2i * _w3i, _w5i = _w2r * _w3i + _w2i * _w3r;
        double _w6r = _w3r * _w3r - _w3i * _w3i, _w6i = 2.0 * _w3r * _w3i;
        y0_re[k] = dcr_;
        y0_im[k] = dci_;
        y1_re[k] = _r1r * _w1r + _r1i * _w1i;
        y1_im[k] = -_r1r * _w1i + _r1i * _w1r;
        y2_re[k] = _r2r * _w2r + _r2i * _w2i;
        y2_im[k] = -_r2r * _w2i + _r2i * _w2r;
        y3_re[k] = _r3r * _w3r + _r3i * _w3i;
        y3_im[k] = -_r3r * _w3i + _r3i * _w3r;
        y4_re[k] = _r4r * _w4r + _r4i * _w4i;
        y4_im[k] = -_r4r * _w4i + _r4i * _w4r;
        y5_re[k] = _r5r * _w5r + _r5i * _w5i;
        y5_im[k] = -_r5r * _w5i + _r5i * _w5r;
        y6_re[k] = _r6r * _w6r + _r6i * _w6i;
        y6_im[k] = -_r6r * _w6i + _r6i * _w6r;
    }
}

/* ================================================================== */
/*  N1 forward — no twiddles                                           */
/* ================================================================== */

static inline __attribute__((always_inline)) __attribute__((target("avx2,fma"))) void radix7_rader_fwd_avx2_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    const double *restrict f_re, const double *restrict f_im,
    const double *restrict g_re, const double *restrict g_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    double *restrict y5_re, double *restrict y5_im,
    double *restrict y6_re, double *restrict y6_im,
    int K)
{
    int k = 0;
    for (; k + R7_AVX2_VW <= K; k += R7_AVX2_VW)
    {
        __m256d x0r = _mm256_loadu_pd(&a_re[k]), x0i = _mm256_loadu_pd(&a_im[k]);
        __m256d t1r = _mm256_loadu_pd(&b_re[k]), t1i = _mm256_loadu_pd(&b_im[k]);
        __m256d t2r = _mm256_loadu_pd(&c_re[k]), t2i = _mm256_loadu_pd(&c_im[k]);
        __m256d t3r = _mm256_loadu_pd(&d_re[k]), t3i = _mm256_loadu_pd(&d_im[k]);
        __m256d t4r = _mm256_loadu_pd(&e_re[k]), t4i = _mm256_loadu_pd(&e_im[k]);
        __m256d t5r = _mm256_loadu_pd(&f_re[k]), t5i = _mm256_loadu_pd(&f_im[k]);
        __m256d t6r = _mm256_loadu_pd(&g_re[k]), t6i = _mm256_loadu_pd(&g_im[k]);

        __m256d dc_r, dc_i;
        TREE_Y0(x0r, t1r, t2r, t3r, t4r, t5r, t6r, dc_r);
        TREE_Y0(x0i, t1i, t2i, t3i, t4i, t5i, t6i, dc_i);
        _mm256_storeu_pd(&y0_re[k], dc_r);
        _mm256_storeu_pd(&y0_im[k], dc_i);

        __m256d s0r = t1r, s0i = t1i, s1r = t3r, s1i = t3i;
        __m256d s2r = t2r, s2i = t2i, s3r = t6r, s3i = t6i;
        __m256d s4r = t4r, s4i = t4i, s5r = t5r, s5i = t5i;

        DFT6_FWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);
        POINTWISE_RR6(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i,
                      RK7_FR0, RK7_FI0, RK7_FR1, RK7_FI1, RK7_FR2, RK7_FI2,
                      RK7_FR3, RK7_FI3, RK7_FR4, RK7_FI4, RK7_FR5, RK7_FI5);
        DFT6_BWD_AVX2(s0r, s0i, s1r, s1i, s2r, s2i, s3r, s3i, s4r, s4i, s5r, s5i);

        _mm256_storeu_pd(&y1_re[k], _mm256_add_pd(x0r, s0r));
        _mm256_storeu_pd(&y1_im[k], _mm256_add_pd(x0i, s0i));
        _mm256_storeu_pd(&y5_re[k], _mm256_add_pd(x0r, s1r));
        _mm256_storeu_pd(&y5_im[k], _mm256_add_pd(x0i, s1i));
        _mm256_storeu_pd(&y4_re[k], _mm256_add_pd(x0r, s2r));
        _mm256_storeu_pd(&y4_im[k], _mm256_add_pd(x0i, s2i));
        _mm256_storeu_pd(&y6_re[k], _mm256_add_pd(x0r, s3r));
        _mm256_storeu_pd(&y6_im[k], _mm256_add_pd(x0i, s3i));
        _mm256_storeu_pd(&y2_re[k], _mm256_add_pd(x0r, s4r));
        _mm256_storeu_pd(&y2_im[k], _mm256_add_pd(x0i, s4i));
        _mm256_storeu_pd(&y3_re[k], _mm256_add_pd(x0r, s5r));
        _mm256_storeu_pd(&y3_im[k], _mm256_add_pd(x0i, s5i));
    }
    for (; k < K; k++)
    {
        double x0r_ = a_re[k], x0i_ = a_im[k];
        double _t1r = b_re[k], _t1i = b_im[k], _t2r = c_re[k], _t2i = c_im[k];
        double _t3r = d_re[k], _t3i = d_im[k], _t4r = e_re[k], _t4i = e_im[k];
        double _t5r = f_re[k], _t5i = f_im[k], _t6r = g_re[k], _t6i = g_im[k];
        y0_re[k] = x0r_ + _t1r + _t2r + _t3r + _t4r + _t5r + _t6r;
        y0_im[k] = x0i_ + _t1i + _t2i + _t3i + _t4i + _t5i + _t6i;
        double ar[6] = {_t1r, _t3r, _t2r, _t6r, _t4r, _t5r};
        double ai[6] = {_t1i, _t3i, _t2i, _t6i, _t4i, _t5i};
        double Xr[6], Xi[6], Cr[6], Ci[6], cr[6], ci[6];
        {
            double s1 = ar[2] + ar[4], s2 = ai[2] + ai[4], d1 = ar[2] - ar[4], d2 = ai[2] - ai[4];
            double E0r = ar[0] + s1, E0i = ai[0] + s2, br = ar[0] - 0.5 * s1, bi = ai[0] - 0.5 * s2;
            double E1r = br + 0.86602540378443864676 * d2, E1i = bi - 0.86602540378443864676 * d1;
            double E2r = br - 0.86602540378443864676 * d2, E2i = bi + 0.86602540378443864676 * d1;
            double s3 = ar[3] + ar[5], s4 = ai[3] + ai[5], d3 = ar[3] - ar[5], d4 = ai[3] - ai[5];
            double O0r = ar[1] + s3, O0i = ai[1] + s4, br2 = ar[1] - 0.5 * s3, bi2 = ai[1] - 0.5 * s4;
            double O1r = br2 + 0.86602540378443864676 * d4, O1i = bi2 - 0.86602540378443864676 * d3;
            double O2r = br2 - 0.86602540378443864676 * d4, O2i = bi2 + 0.86602540378443864676 * d3;
            Xr[0] = E0r + O0r;
            Xi[0] = E0i + O0i;
            Xr[3] = E0r - O0r;
            Xi[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 + O1i * 0.86602540378443864676;
            double T1i_ = O1r * (-0.86602540378443864676) + O1i * 0.5;
            Xr[1] = E1r + T1r_;
            Xi[1] = E1i + T1i_;
            Xr[4] = E1r - T1r_;
            Xi[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) + O2i * 0.86602540378443864676;
            double T2i_ = O2r * (-0.86602540378443864676) + O2i * (-0.5);
            Xr[2] = E2r + T2r_;
            Xi[2] = E2i + T2i_;
            Xr[5] = E2r - T2r_;
            Xi[5] = E2i - T2i_;
        }
        for (int j = 0; j < 6; j++)
        {
            Cr[j] = Xr[j] * RK7A_FWD_RE[j] - Xi[j] * RK7A_FWD_IM[j];
            Ci[j] = Xr[j] * RK7A_FWD_IM[j] + Xi[j] * RK7A_FWD_RE[j];
        }
        {
            double s1 = Cr[2] + Cr[4], s2 = Ci[2] + Ci[4], d1 = Cr[2] - Cr[4], d2 = Ci[2] - Ci[4];
            double E0r = Cr[0] + s1, E0i = Ci[0] + s2, br = Cr[0] - 0.5 * s1, bi = Ci[0] - 0.5 * s2;
            double E1r = br - 0.86602540378443864676 * d2, E1i = bi + 0.86602540378443864676 * d1;
            double E2r = br + 0.86602540378443864676 * d2, E2i = bi - 0.86602540378443864676 * d1;
            double s3 = Cr[3] + Cr[5], s4 = Ci[3] + Ci[5], d3 = Cr[3] - Cr[5], d4 = Ci[3] - Ci[5];
            double O0r = Cr[1] + s3, O0i = Ci[1] + s4, br2 = Cr[1] - 0.5 * s3, bi2 = Ci[1] - 0.5 * s4;
            double O1r = br2 - 0.86602540378443864676 * d4, O1i = bi2 + 0.86602540378443864676 * d3;
            double O2r = br2 + 0.86602540378443864676 * d4, O2i = bi2 - 0.86602540378443864676 * d3;
            cr[0] = E0r + O0r;
            ci[0] = E0i + O0i;
            cr[3] = E0r - O0r;
            ci[3] = E0i - O0i;
            double T1r_ = O1r * 0.5 - O1i * 0.86602540378443864676;
            double T1i_ = O1r * 0.86602540378443864676 + O1i * 0.5;
            cr[1] = E1r + T1r_;
            ci[1] = E1i + T1i_;
            cr[4] = E1r - T1r_;
            ci[4] = E1i - T1i_;
            double T2r_ = O2r * (-0.5) - O2i * 0.86602540378443864676;
            double T2i_ = O2r * 0.86602540378443864676 + O2i * (-0.5);
            cr[2] = E2r + T2r_;
            ci[2] = E2i + T2i_;
            cr[5] = E2r - T2r_;
            ci[5] = E2i - T2i_;
        }
        y1_re[k] = x0r_ + cr[0];
        y1_im[k] = x0i_ + ci[0];
        y5_re[k] = x0r_ + cr[1];
        y5_im[k] = x0i_ + ci[1];
        y4_re[k] = x0r_ + cr[2];
        y4_im[k] = x0i_ + ci[2];
        y6_re[k] = x0r_ + cr[3];
        y6_im[k] = x0i_ + ci[3];
        y2_re[k] = x0r_ + cr[4];
        y2_im[k] = x0i_ + ci[4];
        y3_re[k] = x0r_ + cr[5];
        y3_im[k] = x0i_ + ci[5];
    }
}

#endif /* __AVX2__ */
#endif /* FFT_RADIX7_AVX2_H */