/**
 * @file  fft_radix7_scalar.h
 * @brief Radix-7 FFT butterfly using Rader's algorithm — scalar fallback
 *
 * Rader converts the prime-7 DFT into a length-6 cyclic convolution,
 * computed via DFT6→pointwise multiply→IDFT6.  The length-6 DFT is
 * decomposed as DFT3 × DFT2 (Cooley–Tukey) with compile-time constants.
 *
 * OPTIMIZATIONS:
 *   P0: Round-robin convolution — interleaves mul-phase products across
 *       3-slot waves.  Helps OoO execution fill pipeline bubbles between
 *       dependent multiply→subtract/add pairs.
 *   P1: Tree y0 sum — binary tree DC accumulation.  6-deep add chain → 3
 *       levels, reducing latency from 24 to 12 cycles (4-cycle FP add).
 *
 * Kernel constants:  DFT6(time_reverse(b_fwd)) / 6   (forward)
 *                    DFT6(time_reverse(b_bwd)) / 6   (backward)
 * where b_fwd[q] = W7^{g^q}, g = 3 (primitive root mod 7).
 * The /6 normalization is absorbed into the kernel so the inverse DFT6
 * in the convolution needs no separate scaling.
 *
 * Twiddle layout: BLOCKED3 — stores W1,W2,W3; derives W4=W1·W3,
 *                 W5=W2·W3, W6=W3².  N1 variant for K=1 (no twiddles).
 */

#ifndef FFT_RADIX7_SCALAR_H
#define FFT_RADIX7_SCALAR_H

#include <math.h>

/* ================================================================== */
/*  Compile-time constants                                             */
/* ================================================================== */

/* Rader permutation:  g^q mod 7, g = 3 */
/*   q: 0  1  2  3  4  5                */
/*   →  1  3  2  6  4  5                */
/* Inverse: g^{-s} mod 7, g^{-1} = 5    */
/*   s: 0  1  2  3  4  5                */
/*   →  1  5  4  6  2  3                */

/* DFT3 constants */
#define R7_C3 (-0.5)                   /* cos(2π/3)   */
#define R7_S3 (0.86602540378443864676) /* sin(2π/3)   */

/* DFT6 inter-stage twiddles (W6^k) */
/* W6^1 = ( 0.5, -√3/2)  W6^2 = (-0.5, -√3/2) */
#define R7_W6_1_RE (0.5)
#define R7_W6_1_IM (-0.86602540378443864676)
#define R7_W6_2_RE (-0.5)
#define R7_W6_2_IM (-0.86602540378443864676)

/* Forward Rader kernel: DFT6(time_reverse(b_fwd)) / 6 */
static const double RK7_FWD_RE[6] = {
    -1.66666666666666657415e-01,
    +4.06688893057589595514e-01,
    +3.95078234262700001000e-01,
    +1.85037170770859413132e-17,
    +3.95078234262699945489e-01,
    -4.06688893057589151425e-01};
static const double RK7_FWD_IM[6] = {
    -9.25185853854297065662e-18,
    -1.70436465311965518188e-01,
    -1.95851048647464526242e-01,
    -4.40958551844098212147e-01,
    +1.95851048647464942576e-01,
    -1.70436465311965684721e-01};

/* Backward Rader kernel: DFT6(time_reverse(b_bwd)) / 6 */
static const double RK7_BWD_RE[6] = {
    -1.66666666666666657415e-01,
    -4.06688893057589595514e-01,
    +3.95078234262700167534e-01,
    +9.25185853854297158106e-17,
    +3.95078234262700223045e-01,
    +4.06688893057589540003e-01};
static const double RK7_BWD_IM[6] = {
    +9.25185853854297065662e-18,
    +1.70436465311965656966e-01,
    -1.95851048647464304198e-01,
    +4.40958551844098767258e-01,
    +1.95851048647464359709e-01,
    +1.70436465311965323899e-01};

/* ================================================================== */
/*  Scalar complex multiply                                            */
/* ================================================================== */

static inline __attribute__((always_inline)) void cmul_scalar(double ar, double ai, double br, double bi,
                                                              double *cr, double *ci)
{
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
}

/* ================================================================== */
/*  Round-robin pointwise multiply  (P0 optimization)                  */
/*                                                                     */
/*  Interleaves mul-phase of independent complex multiplies across     */
/*  3-slot waves so OoO execution can fill pipeline bubbles between    */
/*  dependent multiply→subtract/add pairs.                             */
/* ================================================================== */

static inline __attribute__((always_inline)) void pointwise_rr6_scalar(double *Ar, double *Ai,
                                                                       const double *Kr, const double *Ki,
                                                                       double *Cr, double *Ci)
{
    /* Wave 1: slots 0,1,2 — all products first, then combine */
    double p0a = Ai[0] * Ki[0], p0b = Ai[0] * Kr[0];
    double p1a = Ai[1] * Ki[1], p1b = Ai[1] * Kr[1];
    double p2a = Ai[2] * Ki[2], p2b = Ai[2] * Kr[2];

    Cr[0] = Ar[0] * Kr[0] - p0a;
    Ci[0] = Ar[0] * Ki[0] + p0b;
    Cr[1] = Ar[1] * Kr[1] - p1a;
    Ci[1] = Ar[1] * Ki[1] + p1b;
    Cr[2] = Ar[2] * Kr[2] - p2a;
    Ci[2] = Ar[2] * Ki[2] + p2b;

    /* Wave 2: slots 3,4,5 */
    double p3a = Ai[3] * Ki[3], p3b = Ai[3] * Kr[3];
    double p4a = Ai[4] * Ki[4], p4b = Ai[4] * Kr[4];
    double p5a = Ai[5] * Ki[5], p5b = Ai[5] * Kr[5];

    Cr[3] = Ar[3] * Kr[3] - p3a;
    Ci[3] = Ar[3] * Ki[3] + p3b;
    Cr[4] = Ar[4] * Kr[4] - p4a;
    Ci[4] = Ar[4] * Ki[4] + p4b;
    Cr[5] = Ar[5] * Kr[5] - p5a;
    Ci[5] = Ar[5] * Ki[5] + p5b;
}

/* ================================================================== */
/*  Tree y0 sum  (P1 optimization)                                     */
/*                                                                     */
/*  Binary tree reduction: 6 sequential adds → 3 levels                */
/*    L1: a=x0+t1  b=t2+t3  c=t4+t5                                   */
/*    L2: d=a+b    e=c+t6                                              */
/*    L3: dc=d+e                                                       */
/* ================================================================== */

static inline __attribute__((always_inline)) double tree_y0_scalar(double x0, double t1, double t2, double t3,
                                                                   double t4, double t5, double t6)
{
    double a = x0 + t1;
    double b = t2 + t3;
    double c = t4 + t5;
    double d = a + b;
    double e = c + t6;
    return d + e;
}

/* ================================================================== */
/*  Inline DFT6 (forward):  DFT3 × DFT2  Cooley–Tukey DIT            */
/*                                                                     */
/*  Stage 1:  E[k] = DFT3(x[0], x[2], x[4])   k = 0,1,2             */
/*            O[k] = DFT3(x[1], x[3], x[5])   k = 0,1,2             */
/*  Stage 2:  X[k]   = E[k] + W6^k · O[k]                            */
/*            X[k+3] = E[k] - W6^k · O[k]                            */
/* ================================================================== */

static inline __attribute__((always_inline)) void dft6_forward_scalar(const double xr[6], const double xi[6],
                                                                      double Xr[6], double Xi[6])
{
    /* --- DFT3 on even indices (0,2,4) → E[0..2] --- */
    double se_r = xr[2] + xr[4], se_i = xi[2] + xi[4];
    double de_r = xr[2] - xr[4], de_i = xi[2] - xi[4];

    double E0r = xr[0] + se_r;
    double E0i = xi[0] + se_i;
    double base_er = xr[0] + R7_C3 * se_r;
    double base_ei = xi[0] + R7_C3 * se_i;
    /* forward: -i * S3 * d → re += S3*d_im, im -= S3*d_re */
    double E1r = base_er + R7_S3 * de_i;
    double E1i = base_ei - R7_S3 * de_r;
    double E2r = base_er - R7_S3 * de_i;
    double E2i = base_ei + R7_S3 * de_r;

    /* --- DFT3 on odd indices (1,3,5) → O[0..2] --- */
    double so_r = xr[3] + xr[5], so_i = xi[3] + xi[5];
    double do_r = xr[3] - xr[5], do_i = xi[3] - xi[5];

    double O0r = xr[1] + so_r;
    double O0i = xi[1] + so_i;
    double base_or = xr[1] + R7_C3 * so_r;
    double base_oi = xi[1] + R7_C3 * so_i;
    double O1r = base_or + R7_S3 * do_i;
    double O1i = base_oi - R7_S3 * do_r;
    double O2r = base_or - R7_S3 * do_i;
    double O2i = base_oi + R7_S3 * do_r;

    /* --- DFT2 recombination with W6 twiddles --- */
    /* k=0: W6^0 = 1 */
    Xr[0] = E0r + O0r;
    Xi[0] = E0i + O0i;
    Xr[3] = E0r - O0r;
    Xi[3] = E0i - O0i;

    /* k=1: W6^1 = (0.5, -S3) */
    double T1r, T1i;
    cmul_scalar(O1r, O1i, R7_W6_1_RE, R7_W6_1_IM, &T1r, &T1i);
    Xr[1] = E1r + T1r;
    Xi[1] = E1i + T1i;
    Xr[4] = E1r - T1r;
    Xi[4] = E1i - T1i;

    /* k=2: W6^2 = (-0.5, -S3) */
    double T2r, T2i;
    cmul_scalar(O2r, O2i, R7_W6_2_RE, R7_W6_2_IM, &T2r, &T2i);
    Xr[2] = E2r + T2r;
    Xi[2] = E2i + T2i;
    Xr[5] = E2r - T2r;
    Xi[5] = E2i - T2i;
}

/* ================================================================== */
/*  Inline DFT6 (backward):  same structure, conjugate twiddles        */
/* ================================================================== */

static inline __attribute__((always_inline)) void dft6_backward_scalar(const double xr[6], const double xi[6],
                                                                       double Xr[6], double Xi[6])
{
    /* --- DFT3 backward on even indices (sign flip on cross-term) --- */
    double se_r = xr[2] + xr[4], se_i = xi[2] + xi[4];
    double de_r = xr[2] - xr[4], de_i = xi[2] - xi[4];

    double E0r = xr[0] + se_r;
    double E0i = xi[0] + se_i;
    double base_er = xr[0] + R7_C3 * se_r;
    double base_ei = xi[0] + R7_C3 * se_i;
    /* backward: +i * S3 * d → re -= S3*d_im, im += S3*d_re */
    double E1r = base_er - R7_S3 * de_i;
    double E1i = base_ei + R7_S3 * de_r;
    double E2r = base_er + R7_S3 * de_i;
    double E2i = base_ei - R7_S3 * de_r;

    /* --- DFT3 backward on odd indices --- */
    double so_r = xr[3] + xr[5], so_i = xi[3] + xi[5];
    double do_r = xr[3] - xr[5], do_i = xi[3] - xi[5];

    double O0r = xr[1] + so_r;
    double O0i = xi[1] + so_i;
    double base_or = xr[1] + R7_C3 * so_r;
    double base_oi = xi[1] + R7_C3 * so_i;
    double O1r = base_or - R7_S3 * do_i;
    double O1i = base_oi + R7_S3 * do_r;
    double O2r = base_or + R7_S3 * do_i;
    double O2i = base_oi - R7_S3 * do_r;

    /* --- DFT2 with conjugate W6 twiddles --- */
    Xr[0] = E0r + O0r;
    Xi[0] = E0i + O0i;
    Xr[3] = E0r - O0r;
    Xi[3] = E0i - O0i;

    /* W6_bwd^1 = (0.5, +S3) */
    double T1r, T1i;
    cmul_scalar(O1r, O1i, R7_W6_1_RE, -R7_W6_1_IM, &T1r, &T1i);
    Xr[1] = E1r + T1r;
    Xi[1] = E1i + T1i;
    Xr[4] = E1r - T1r;
    Xi[4] = E1i - T1i;

    /* W6_bwd^2 = (-0.5, +S3) */
    double T2r, T2i;
    cmul_scalar(O2r, O2i, R7_W6_2_RE, -R7_W6_2_IM, &T2r, &T2i);
    Xr[2] = E2r + T2r;
    Xi[2] = E2i + T2i;
    Xr[5] = E2r - T2r;
    Xi[5] = E2i - T2i;
}

/* ================================================================== */
/*  Rader radix-7 butterfly — scalar, forward, with twiddles           */
/*                                                                     */
/*  BLOCKED3 twiddles: W1,W2,W3 stored; W4=W1·W3, W5=W2·W3, W6=W3²  */
/*  Input/output: base-pointer + k indexing, stride implicit.          */
/* ================================================================== */

static inline __attribute__((always_inline)) void radix7_rader_fwd_scalar_1(
    const double *a_re, const double *a_im, /* x0 */
    const double *b_re, const double *b_im, /* x1 */
    const double *c_re, const double *c_im, /* x2 */
    const double *d_re, const double *d_im, /* x3 */
    const double *e_re, const double *e_im, /* x4 */
    const double *f_re, const double *f_im, /* x5 */
    const double *g_re, const double *g_im, /* x6 */
    double *y0_re, double *y0_im,
    double *y1_re, double *y1_im,
    double *y2_re, double *y2_im,
    double *y3_re, double *y3_im,
    double *y4_re, double *y4_im,
    double *y5_re, double *y5_im,
    double *y6_re, double *y6_im,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im,
    const double *tw3_re, const double *tw3_im,
    int K)
{
    for (int k = 0; k < K; k++)
    {
        /* ---- Load x[0] ---- */
        double x0r = a_re[k], x0i = a_im[k];

        /* ---- Load & twiddle x[1..6] ---- */
        double t1r, t1i, t2r, t2i, t3r, t3i;
        double t4r, t4i, t5r, t5i, t6r, t6i;

        double w1r = tw1_re[k], w1i = tw1_im[k];
        double w2r = tw2_re[k], w2i = tw2_im[k];
        double w3r = tw3_re[k], w3i = tw3_im[k];

        cmul_scalar(b_re[k], b_im[k], w1r, w1i, &t1r, &t1i);
        cmul_scalar(c_re[k], c_im[k], w2r, w2i, &t2r, &t2i);
        cmul_scalar(d_re[k], d_im[k], w3r, w3i, &t3r, &t3i);

        /* Derive W4=W1·W3, W5=W2·W3, W6=W3² */
        double w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_scalar(w1r, w1i, w3r, w3i, &w4r, &w4i);
        cmul_scalar(w2r, w2i, w3r, w3i, &w5r, &w5i);
        cmul_scalar(w3r, w3i, w3r, w3i, &w6r, &w6i);

        cmul_scalar(e_re[k], e_im[k], w4r, w4i, &t4r, &t4i);
        cmul_scalar(f_re[k], f_im[k], w5r, w5i, &t5r, &t5i);
        cmul_scalar(g_re[k], g_im[k], w6r, w6i, &t6r, &t6i);

        /* ---- DC output: tree sum (P1) ---- */
        double dcr = tree_y0_scalar(x0r, t1r, t2r, t3r, t4r, t5r, t6r);
        double dci = tree_y0_scalar(x0i, t1i, t2i, t3i, t4i, t5i, t6i);
        y0_re[k] = dcr;
        y0_im[k] = dci;

        /* ---- Rader input permutation: a[q] = t[perm[q]] ---- */
        /* perm = {1, 3, 2, 6, 4, 5} */
        double ar[6], ai[6];
        ar[0] = t1r;
        ai[0] = t1i; /* t[1] */
        ar[1] = t3r;
        ai[1] = t3i; /* t[3] */
        ar[2] = t2r;
        ai[2] = t2i; /* t[2] */
        ar[3] = t6r;
        ai[3] = t6i; /* t[6] */
        ar[4] = t4r;
        ai[4] = t4i; /* t[4] */
        ar[5] = t5r;
        ai[5] = t5i; /* t[5] */

        /* ---- Forward DFT6 ---- */
        double Ar[6], Ai[6];
        dft6_forward_scalar(ar, ai, Ar, Ai);

        /* ---- Round-robin pointwise multiply (P0) ---- */
        double Cr[6], Ci[6];
        pointwise_rr6_scalar(Ar, Ai, RK7_FWD_RE, RK7_FWD_IM, Cr, Ci);

        /* ---- Backward DFT6 (no /6, absorbed in kernel) ---- */
        double cr[6], ci[6];
        dft6_backward_scalar(Cr, Ci, cr, ci);

        /* ---- Output un-permute: Y[inv_perm[s]] = x0 + c[s] ---- */
        /* inv_perm = {1, 5, 4, 6, 2, 3} */
        y1_re[k] = x0r + cr[0];
        y1_im[k] = x0i + ci[0]; /* s=0 → m=1 */
        y5_re[k] = x0r + cr[1];
        y5_im[k] = x0i + ci[1]; /* s=1 → m=5 */
        y4_re[k] = x0r + cr[2];
        y4_im[k] = x0i + ci[2]; /* s=2 → m=4 */
        y6_re[k] = x0r + cr[3];
        y6_im[k] = x0i + ci[3]; /* s=3 → m=6 */
        y2_re[k] = x0r + cr[4];
        y2_im[k] = x0i + ci[4]; /* s=4 → m=2 */
        y3_re[k] = x0r + cr[5];
        y3_im[k] = x0i + ci[5]; /* s=5 → m=3 */
    }
}

/* ================================================================== */
/*  Rader radix-7 butterfly — scalar, backward, with twiddles          */
/* ================================================================== */

static inline __attribute__((always_inline)) void radix7_rader_bwd_scalar_1(
    const double *a_re, const double *a_im,
    const double *b_re, const double *b_im,
    const double *c_re, const double *c_im,
    const double *d_re, const double *d_im,
    const double *e_re, const double *e_im,
    const double *f_re, const double *f_im,
    const double *g_re, const double *g_im,
    double *y0_re, double *y0_im,
    double *y1_re, double *y1_im,
    double *y2_re, double *y2_im,
    double *y3_re, double *y3_im,
    double *y4_re, double *y4_im,
    double *y5_re, double *y5_im,
    double *y6_re, double *y6_im,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im,
    const double *tw3_re, const double *tw3_im,
    int K)
{
    for (int k = 0; k < K; k++)
    {
        /* ---- Step 1: Backward Rader on raw inputs (no twiddle) ---- */
        double x0r = a_re[k], x0i = a_im[k];
        double t1r = b_re[k], t1i = b_im[k];
        double t2r = c_re[k], t2i = c_im[k];
        double t3r = d_re[k], t3i = d_im[k];
        double t4r = e_re[k], t4i = e_im[k];
        double t5r = f_re[k], t5i = f_im[k];
        double t6r = g_re[k], t6i = g_im[k];

        /* DC of backward DFT (tree sum, before twiddle) */
        double dcr = tree_y0_scalar(x0r, t1r, t2r, t3r, t4r, t5r, t6r);
        double dci = tree_y0_scalar(x0i, t1i, t2i, t3i, t4i, t5i, t6i);

        /* Rader permute + backward convolution */
        double ar[6], ai[6];
        ar[0] = t1r;
        ai[0] = t1i;
        ar[1] = t3r;
        ai[1] = t3i;
        ar[2] = t2r;
        ai[2] = t2i;
        ar[3] = t6r;
        ai[3] = t6i;
        ar[4] = t4r;
        ai[4] = t4i;
        ar[5] = t5r;
        ai[5] = t5i;

        double Ar[6], Ai[6];
        dft6_forward_scalar(ar, ai, Ar, Ai);

        double Cr[6], Ci[6];
        pointwise_rr6_scalar(Ar, Ai, RK7_BWD_RE, RK7_BWD_IM, Cr, Ci);

        double cr[6], ci[6];
        dft6_backward_scalar(Cr, Ci, cr, ci);

        /* Raw IDFT7 outputs (before twiddle) */
        /* inv_perm = {1,5,4,6,2,3} */
        double r0r = dcr, r0i = dci;
        double r1r = x0r + cr[0], r1i = x0i + ci[0]; /* m=1 */
        double r5r = x0r + cr[1], r5i = x0i + ci[1]; /* m=5 */
        double r4r = x0r + cr[2], r4i = x0i + ci[2]; /* m=4 */
        double r6r = x0r + cr[3], r6i = x0i + ci[3]; /* m=6 */
        double r2r = x0r + cr[4], r2i = x0i + ci[4]; /* m=2 */
        double r3r = x0r + cr[5], r3i = x0i + ci[5]; /* m=3 */

        /* ---- Step 2: Apply conj twiddles AFTER IDFT ---- */
        /* W0=1, W1, W2, W3, W4=W1·W3, W5=W2·W3, W6=W3² */
        double w1r = tw1_re[k], w1i = tw1_im[k];
        double w2r = tw2_re[k], w2i = tw2_im[k];
        double w3r = tw3_re[k], w3i = tw3_im[k];

        double w4r, w4i, w5r, w5i, w6r, w6i;
        cmul_scalar(w1r, w1i, w3r, w3i, &w4r, &w4i);
        cmul_scalar(w2r, w2i, w3r, w3i, &w5r, &w5i);
        cmul_scalar(w3r, w3i, w3r, w3i, &w6r, &w6i);

        y0_re[k] = r0r;
        y0_im[k] = r0i; /* W0=1, no multiply */
        cmul_scalar(r1r, r1i, w1r, -w1i, &y1_re[k], &y1_im[k]);
        cmul_scalar(r2r, r2i, w2r, -w2i, &y2_re[k], &y2_im[k]);
        cmul_scalar(r3r, r3i, w3r, -w3i, &y3_re[k], &y3_im[k]);
        cmul_scalar(r4r, r4i, w4r, -w4i, &y4_re[k], &y4_im[k]);
        cmul_scalar(r5r, r5i, w5r, -w5i, &y5_re[k], &y5_im[k]);
        cmul_scalar(r6r, r6i, w6r, -w6i, &y6_re[k], &y6_im[k]);
    }
}

/* ================================================================== */
/*  N1 variants — no twiddles (K=1 stage, all twiddles = 1)            */
/* ================================================================== */

static inline __attribute__((always_inline)) void radix7_rader_fwd_scalar_N1(
    const double *a_re, const double *a_im,
    const double *b_re, const double *b_im,
    const double *c_re, const double *c_im,
    const double *d_re, const double *d_im,
    const double *e_re, const double *e_im,
    const double *f_re, const double *f_im,
    const double *g_re, const double *g_im,
    double *y0_re, double *y0_im,
    double *y1_re, double *y1_im,
    double *y2_re, double *y2_im,
    double *y3_re, double *y3_im,
    double *y4_re, double *y4_im,
    double *y5_re, double *y5_im,
    double *y6_re, double *y6_im,
    int K)
{
    for (int k = 0; k < K; k++)
    {
        double x0r = a_re[k], x0i = a_im[k];
        double t1r = b_re[k], t1i = b_im[k];
        double t2r = c_re[k], t2i = c_im[k];
        double t3r = d_re[k], t3i = d_im[k];
        double t4r = e_re[k], t4i = e_im[k];
        double t5r = f_re[k], t5i = f_im[k];
        double t6r = g_re[k], t6i = g_im[k];

        y0_re[k] = tree_y0_scalar(x0r, t1r, t2r, t3r, t4r, t5r, t6r);
        y0_im[k] = tree_y0_scalar(x0i, t1i, t2i, t3i, t4i, t5i, t6i);

        double ar[6], ai[6];
        ar[0] = t1r;
        ai[0] = t1i;
        ar[1] = t3r;
        ai[1] = t3i;
        ar[2] = t2r;
        ai[2] = t2i;
        ar[3] = t6r;
        ai[3] = t6i;
        ar[4] = t4r;
        ai[4] = t4i;
        ar[5] = t5r;
        ai[5] = t5i;

        double Ar[6], Ai[6];
        dft6_forward_scalar(ar, ai, Ar, Ai);

        double Cr[6], Ci[6];
        pointwise_rr6_scalar(Ar, Ai, RK7_FWD_RE, RK7_FWD_IM, Cr, Ci);

        double cr[6], ci[6];
        dft6_backward_scalar(Cr, Ci, cr, ci);

        y1_re[k] = x0r + cr[0];
        y1_im[k] = x0i + ci[0];
        y5_re[k] = x0r + cr[1];
        y5_im[k] = x0i + ci[1];
        y4_re[k] = x0r + cr[2];
        y4_im[k] = x0i + ci[2];
        y6_re[k] = x0r + cr[3];
        y6_im[k] = x0i + ci[3];
        y2_re[k] = x0r + cr[4];
        y2_im[k] = x0i + ci[4];
        y3_re[k] = x0r + cr[5];
        y3_im[k] = x0i + ci[5];
    }
}

static inline __attribute__((always_inline)) void radix7_rader_bwd_scalar_N1(
    const double *a_re, const double *a_im,
    const double *b_re, const double *b_im,
    const double *c_re, const double *c_im,
    const double *d_re, const double *d_im,
    const double *e_re, const double *e_im,
    const double *f_re, const double *f_im,
    const double *g_re, const double *g_im,
    double *y0_re, double *y0_im,
    double *y1_re, double *y1_im,
    double *y2_re, double *y2_im,
    double *y3_re, double *y3_im,
    double *y4_re, double *y4_im,
    double *y5_re, double *y5_im,
    double *y6_re, double *y6_im,
    int K)
{
    for (int k = 0; k < K; k++)
    {
        double x0r = a_re[k], x0i = a_im[k];
        double t1r = b_re[k], t1i = b_im[k];
        double t2r = c_re[k], t2i = c_im[k];
        double t3r = d_re[k], t3i = d_im[k];
        double t4r = e_re[k], t4i = e_im[k];
        double t5r = f_re[k], t5i = f_im[k];
        double t6r = g_re[k], t6i = g_im[k];

        y0_re[k] = tree_y0_scalar(x0r, t1r, t2r, t3r, t4r, t5r, t6r);
        y0_im[k] = tree_y0_scalar(x0i, t1i, t2i, t3i, t4i, t5i, t6i);

        double ar[6], ai[6];
        ar[0] = t1r;
        ai[0] = t1i;
        ar[1] = t3r;
        ai[1] = t3i;
        ar[2] = t2r;
        ai[2] = t2i;
        ar[3] = t6r;
        ai[3] = t6i;
        ar[4] = t4r;
        ai[4] = t4i;
        ar[5] = t5r;
        ai[5] = t5i;

        double Ar[6], Ai[6];
        dft6_forward_scalar(ar, ai, Ar, Ai);

        double Cr[6], Ci[6];
        pointwise_rr6_scalar(Ar, Ai, RK7_BWD_RE, RK7_BWD_IM, Cr, Ci);

        double cr[6], ci[6];
        dft6_backward_scalar(Cr, Ci, cr, ci);

        y1_re[k] = x0r + cr[0];
        y1_im[k] = x0i + ci[0];
        y5_re[k] = x0r + cr[1];
        y5_im[k] = x0i + ci[1];
        y4_re[k] = x0r + cr[2];
        y4_im[k] = x0i + ci[2];
        y6_re[k] = x0r + cr[3];
        y6_im[k] = x0i + ci[3];
        y2_re[k] = x0r + cr[4];
        y2_im[k] = x0i + ci[4];
        y3_re[k] = x0r + cr[5];
        y3_im[k] = x0i + ci[5];
    }
}

#endif /* FFT_RADIX7_SCALAR_H */