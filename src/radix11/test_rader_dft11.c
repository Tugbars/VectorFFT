/*
 * test_rader_dft11.c — Build up from Winograd DFT-5 → DFT-10 → Rader DFT-11
 *
 * DFT-10 = 2×5 Cooley-Tukey:
 *   Pass 1: 5 radix-2 butterflies
 *   4 W₁₀ twiddles (W₁₀^0 trivial)
 *   Pass 2: 2 × Winograd DFT-5
 *
 * DFT-11 = Rader:
 *   DC accumulation
 *   Permute inputs via g^q mod 11 (g=2)
 *   DFT-10 of permuted input
 *   Pointwise multiply by precomputed kernel B[k] = DFT-10(W₁₁^{g^q})
 *   IDFT-10 of product
 *   Unpermute outputs via g^{-q} mod 11
 *   DC fixup
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * WINOGRAD DFT-5 (verified, 10 muls + 34 adds)
 * ═══════════════════════════════════════════════════════════════ */

static const double W5_ALPHA = -0.25;
static const double W5_BETA  =  0.55901699437494745;
static const double W5_S2    =  0.58778525229247325;
static const double W5_S1mS2 =  0.36327126400268028;
static const double W5_S1pS2 =  1.53884176858762678;

static void winograd_dft5_fwd(double *r, double *i) {
    /* In-place on r[0..4], i[0..4] */
    double a1r=r[1]+r[4], a1i=i[1]+i[4];
    double a2r=r[2]+r[3], a2i=i[2]+i[3];
    double b1r=r[1]-r[4], b1i=i[1]-i[4];
    double b2r=r[2]-r[3], b2i=i[2]-i[3];
    double t1r=a1r+a2r, t1i=a1i+a2i;
    double t2r=a1r-a2r, t2i=a1i-a2i;
    double x0s_r=r[0], x0s_i=i[0];
    r[0]=x0s_r+t1r; i[0]=x0s_i+t1i;
    double m1r=t1r*W5_ALPHA, m1i=t1i*W5_ALPHA;
    double m2r=t2r*W5_BETA,  m2i=t2i*W5_BETA;
    double p1r=W5_S2*(b1r+b2r),  p1i=W5_S2*(b1i+b2i);
    double p2r=W5_S1mS2*b1r,     p2i=W5_S1mS2*b1i;
    double p3r=W5_S1pS2*b2r,     p3i=W5_S1pS2*b2i;
    double q1r=p1r+p2r, q1i=p1i+p2i;
    double q2r=p1r-p3r, q2i=p1i-p3i;
    double w1r=x0s_r+m1r, w1i=x0s_i+m1i;
    double w2r=w1r+m2r, w2i=w1i+m2i;
    double w3r=w1r-m2r, w3i=w1i-m2i;
    r[1]=w2r+q1i; i[1]=w2i-q1r;
    r[4]=w2r-q1i; i[4]=w2i+q1r;
    r[2]=w3r+q2i; i[2]=w3i-q2r;
    r[3]=w3r-q2i; i[3]=w3i+q2r;
}

static void winograd_dft5_bwd(double *r, double *i) {
    double a1r=r[1]+r[4], a1i=i[1]+i[4];
    double a2r=r[2]+r[3], a2i=i[2]+i[3];
    double b1r=r[1]-r[4], b1i=i[1]-i[4];
    double b2r=r[2]-r[3], b2i=i[2]-i[3];
    double t1r=a1r+a2r, t1i=a1i+a2i;
    double t2r=a1r-a2r, t2i=a1i-a2i;
    double x0s_r=r[0], x0s_i=i[0];
    r[0]=x0s_r+t1r; i[0]=x0s_i+t1i;
    double m1r=t1r*W5_ALPHA, m1i=t1i*W5_ALPHA;
    double m2r=t2r*W5_BETA,  m2i=t2i*W5_BETA;
    double p1r=W5_S2*(b1r+b2r),  p1i=W5_S2*(b1i+b2i);
    double p2r=W5_S1mS2*b1r,     p2i=W5_S1mS2*b1i;
    double p3r=W5_S1pS2*b2r,     p3i=W5_S1pS2*b2i;
    double q1r=p1r+p2r, q1i=p1i+p2i;
    double q2r=p1r-p3r, q2i=p1i-p3i;
    double w1r=x0s_r+m1r, w1i=x0s_i+m1i;
    double w2r=w1r+m2r, w2i=w1i+m2i;
    double w3r=w1r-m2r, w3i=w1i-m2i;
    r[1]=w2r-q1i; i[1]=w2i+q1r;
    r[4]=w2r+q1i; i[4]=w2i-q1r;
    r[2]=w3r-q2i; i[2]=w3i+q2r;
    r[3]=w3r+q2i; i[3]=w3i-q2r;
}

/* ═══════════════════════════════════════════════════════════════
 * DFT-10 via 2×5 Cooley-Tukey (N1=2, N2=5)
 *
 * Pass 1: 5 radix-2 on (x[n], x[n+5])
 * 4 W₁₀ twiddles on b[1..4] (b[0] is W₁₀⁰=1, trivial)
 * Pass 2: 2 × Winograd DFT-5
 *   a[0..4] → even outputs X[0,2,4,6,8]
 *   b[0..4] → odd outputs  X[1,3,5,7,9]
 *
 * W₁₀ twiddle constants reuse DFT-5 values:
 *   cos(36°) = cos(π/5) ≈ 0.80902  sin(36°) = sin(π/5) ≈ 0.58779
 *   cos(72°) = cos(2π/5) ≈ 0.30902  sin(72°) = sin(2π/5) ≈ 0.95106
 *   W₁₀¹: ( cos36, -sin36)
 *   W₁₀²: ( cos72, -sin72)
 *   W₁₀³: (-cos72, -sin72)
 *   W₁₀⁴: (-cos36, -sin36)
 * ═══════════════════════════════════════════════════════════════ */

static const double W10_C36 = 0.80901699437494742;    /* cos(π/5)  */
static const double W10_S36 = 0.58778525229247313;    /* sin(π/5)  */
static const double W10_C72 = 0.30901699437494742;    /* cos(2π/5) */
static const double W10_S72 = 0.95105651629515357;    /* sin(2π/5) */

/* Complex multiply: (ar,ai) × (wr,wi) → (ar*wr - ai*wi, ar*wi + ai*wr) */
#define CMUL(ar, ai, wr, wi, dr, di) do { \
    double _tr = (ar); \
    (dr) = _tr*(wr) - (ai)*(wi); \
    (di) = _tr*(wi) + (ai)*(wr); \
} while(0)

static void dft10_fwd(double *r, double *im) {
    /* r[0..9], im[0..9] in, out */
    double ar[5], ai[5], br[5], bi[5];

    /* Pass 1: 5 radix-2 */
    for (int n = 0; n < 5; n++) {
        ar[n] = r[n] + r[n+5];  ai[n] = im[n] + im[n+5];
        br[n] = r[n] - r[n+5];  bi[n] = im[n] - im[n+5];
    }

    /* W₁₀ twiddles on b[1..4] (b[0] = W₁₀⁰ = 1, skip) */
    { double tr, ti;
      CMUL(br[1], bi[1],  W10_C36, -W10_S36, tr, ti); br[1]=tr; bi[1]=ti;
      CMUL(br[2], bi[2],  W10_C72, -W10_S72, tr, ti); br[2]=tr; bi[2]=ti;
      CMUL(br[3], bi[3], -W10_C72, -W10_S72, tr, ti); br[3]=tr; bi[3]=ti;
      CMUL(br[4], bi[4], -W10_C36, -W10_S36, tr, ti); br[4]=tr; bi[4]=ti;
    }

    /* Pass 2: 2 × Winograd DFT-5 */
    winograd_dft5_fwd(ar, ai);
    winograd_dft5_fwd(br, bi);

    /* Interleave: X[2k] = a[k], X[2k+1] = b[k] */
    for (int k = 0; k < 5; k++) {
        r[2*k]   = ar[k];  im[2*k]   = ai[k];
        r[2*k+1] = br[k];  im[2*k+1] = bi[k];
    }
}

static void dft10_bwd(double *r, double *im) {
    double ar[5], ai[5], br[5], bi[5];

    for (int n = 0; n < 5; n++) {
        ar[n] = r[n] + r[n+5];  ai[n] = im[n] + im[n+5];
        br[n] = r[n] - r[n+5];  bi[n] = im[n] - im[n+5];
    }

    /* Backward W₁₀ twiddles: conjugate = (wr, -wi) → (wr, +sin) */
    { double tr, ti;
      CMUL(br[1], bi[1],  W10_C36, +W10_S36, tr, ti); br[1]=tr; bi[1]=ti;
      CMUL(br[2], bi[2],  W10_C72, +W10_S72, tr, ti); br[2]=tr; bi[2]=ti;
      CMUL(br[3], bi[3], -W10_C72, +W10_S72, tr, ti); br[3]=tr; bi[3]=ti;
      CMUL(br[4], bi[4], -W10_C36, +W10_S36, tr, ti); br[4]=tr; bi[4]=ti;
    }

    winograd_dft5_bwd(ar, ai);
    winograd_dft5_bwd(br, bi);

    for (int k = 0; k < 5; k++) {
        r[2*k]   = ar[k];  im[2*k]   = ai[k];
        r[2*k+1] = br[k];  im[2*k+1] = bi[k];
    }
}

/* ═══════════════════════════════════════════════════════════════
 * RADER DFT-11
 *
 * g = 2 (primitive root of 11)
 * Permutation:      g^q mod 11 = [1,2,4,8,5,10,9,7,3,6]
 * Inv permutation:  g^{-q} mod 11 = [1,6,3,7,9,10,5,8,4,2]
 *
 * B[k] = DFT-10(W₁₁^{g^q}) — precomputed kernel in freq domain
 *
 * Algorithm:
 *   1. DC = sum(x[0..10])
 *   2. a[q] = x[g^q mod 11] for q=0..9   (Rader permutation)
 *   3. A = DFT-10(a)
 *   4. C[k] = A[k] * B[k]                (pointwise)
 *   5. c = IDFT-10(C) / 10
 *   6. X[g^{-q} mod 11] = x[0] + c[q]   (unpermute + DC fixup)
 *   7. X[0] = DC
 * ═══════════════════════════════════════════════════════════════ */

/* Precomputed B[k] = DFT-10(kernel), kernel[q] = W₁₁^{g^q mod 11} */
static const int RADER_PERM[10]  = {1, 2, 4, 8, 5, 10, 9, 7, 3, 6};
static const int RADER_IPERM[10] = {1, 6, 3, 7, 9, 10, 5, 8, 4, 2};

/* B_rev = DFT-10 of reversed Rader kernel (correlation, not convolution) */
/* b_rev[q] = W_11^{g^{-q mod 10}}, B_rev = DFT-10(b_rev)               */
static const double B_re[10] = {
    -1.00000000000000044409e+00,
    +9.55301877984370717556e-01,
    +2.63610556432483411626e+00,
    +2.54127802471550001684e+00,
    +2.07016209983106902470e+00,
    +0.00000000000000000000e+00,  /* pure imaginary */
    +2.07016209983107213333e+00,
    -2.54127802471550712227e+00,
    +2.63610556432483722489e+00,
    -9.55301877984364278262e-01,
};
static const double B_im[10] = {
    +0.00000000000000000000e+00,  /* real */
    -3.17606648575238903476e+00,
    +2.01269656275744779350e+00,
    +2.13117479365210327202e+00,
    +2.59122150354287761331e+00,
    -3.31662479035540158634e+00,
    -2.59122150354287672513e+00,
    +2.13117479365209971931e+00,
    -2.01269656275744646123e+00,
    -3.17606648575239614019e+00,
};

/* Backward kernel: B_bwd[k] = conj(B_fwd[(10-k) mod 10]) */
static const double Bb_re[10] = {
    -1.00000000000000044409e+00,
    -9.55301877984364278262e-01,
    +2.63610556432483722489e+00,
    -2.54127802471550712227e+00,
    +2.07016209983107213333e+00,
    +0.00000000000000000000e+00,
    +2.07016209983106902470e+00,
    +2.54127802471550001684e+00,
    +2.63610556432483411626e+00,
    +9.55301877984370717556e-01,
};
static const double Bb_im[10] = {
    +0.00000000000000000000e+00,
    +3.17606648575239614019e+00,
    +2.01269656275744646123e+00,
    -2.13117479365209971931e+00,
    +2.59122150354287672513e+00,
    +3.31662479035540158634e+00,
    -2.59122150354287761331e+00,
    -2.13117479365210327202e+00,
    -2.01269656275744779350e+00,
    +3.17606648575238903476e+00,
};

static void rader_dft11_fwd(const double *xr, const double *xi,
                            double *Xr, double *Xi) {
    /* 1. DC */
    double dc_r = 0, dc_i = 0;
    for (int n = 0; n < 11; n++) { dc_r += xr[n]; dc_i += xi[n]; }

    /* 2. Rader permutation */
    double ar[10], ai[10];
    for (int q = 0; q < 10; q++) {
        ar[q] = xr[RADER_PERM[q]];
        ai[q] = xi[RADER_PERM[q]];
    }

    /* 3. DFT-10 */
    dft10_fwd(ar, ai);

    /* 4. Pointwise multiply A[k] × B[k] */
    double cr[10], ci[10];
    for (int k = 0; k < 10; k++) {
        cr[k] = ar[k]*B_re[k] - ai[k]*B_im[k];
        ci[k] = ar[k]*B_im[k] + ai[k]*B_re[k];
    }

    /* 5. IDFT-10 (backward DFT, then divide by 10) */
    dft10_bwd(cr, ci);
    for (int q = 0; q < 10; q++) { cr[q] /= 10.0; ci[q] /= 10.0; }

    /* 6. Unpermute + fixup: X[g^{-q}] = x[0] + c[q] */
    for (int q = 0; q < 10; q++) {
        int idx = RADER_IPERM[q];
        Xr[idx] = xr[0] + cr[q];
        Xi[idx] = xi[0] + ci[q];
    }

    /* 7. DC output */
    Xr[0] = dc_r;
    Xi[0] = dc_i;
}

static void rader_dft11_bwd(const double *xr, const double *xi,
                            double *Xr, double *Xi) {
    /* Same Rader structure, backward kernel Bb = conj(B[(10-k)%10]) */
    double dc_r = 0, dc_i = 0;
    for (int n = 0; n < 11; n++) { dc_r += xr[n]; dc_i += xi[n]; }

    double ar[10], ai[10];
    for (int q = 0; q < 10; q++) {
        ar[q] = xr[RADER_PERM[q]];
        ai[q] = xi[RADER_PERM[q]];
    }

    dft10_fwd(ar, ai);

    /* Pointwise multiply by backward kernel Bb */
    double cr[10], ci[10];
    for (int k = 0; k < 10; k++) {
        cr[k] = ar[k]*Bb_re[k] - ai[k]*Bb_im[k];
        ci[k] = ar[k]*Bb_im[k] + ai[k]*Bb_re[k];
    }

    dft10_bwd(cr, ci);
    for (int q = 0; q < 10; q++) { cr[q] /= 10.0; ci[q] /= 10.0; }

    for (int q = 0; q < 10; q++) {
        int idx = RADER_IPERM[q];
        Xr[idx] = xr[0] + cr[q];
        Xi[idx] = xi[0] + ci[q];
    }
    Xr[0] = dc_r;
    Xi[0] = dc_i;
}

/* ═══════════════════════════════════════════════════════════════
 * NAIVE DFT for verification
 * ═══════════════════════════════════════════════════════════════ */

static void naive_dft(int N, int dir,
    const double *xr, const double *xi, double *Xr, double *Xi) {
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = dir * 2.0 * M_PI * k * n / (double)N;
            sr += xr[n]*cos(a) - xi[n]*sin(a);
            si += xr[n]*sin(a) + xi[n]*cos(a);
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

static double max_rel_err(int N, const double *ar, const double *ai,
                          const double *br, const double *bi) {
    double err = 0, mag = 0;
    for (int i = 0; i < N; i++) {
        double e = fmax(fabs(ar[i]-br[i]), fabs(ai[i]-bi[i]));
        if (e > err) err = e;
        double m = fmax(fabs(br[i]), fabs(bi[i]));
        if (m > mag) mag = m;
    }
    return mag > 0 ? err/mag : err;
}

int main(void) {
    int passed = 0, total = 0;

    /* ── DFT-10 verification ── */
    printf("=== DFT-10 (2×Winograd-5) ===\n");
    for (int trial = 0; trial < 5; trial++) {
        double xr[10], xi[10], Nr[10], Ni[10], Wr[10], Wi[10];
        srand(200 + trial);
        for (int i = 0; i < 10; i++) {
            xr[i] = (double)rand()/RAND_MAX*2-1;
            xi[i] = (double)rand()/RAND_MAX*2-1;
        }
        naive_dft(10, -1, xr, xi, Nr, Ni);
        for (int i = 0; i < 10; i++) { Wr[i]=xr[i]; Wi[i]=xi[i]; }
        dft10_fwd(Wr, Wi);
        double rel = max_rel_err(10, Wr, Wi, Nr, Ni);
        int pass = rel < 1e-14;
        printf("  fwd %d: rel=%.2e %s\n", trial, rel, pass?"PASS":"FAIL");
        total++; passed += pass;
    }

    /* DFT-10 roundtrip */
    for (int trial = 0; trial < 3; trial++) {
        double xr[10], xi[10], fr[10], fi[10];
        srand(300 + trial);
        for (int i = 0; i < 10; i++) {
            xr[i] = (double)rand()/RAND_MAX*2-1;
            xi[i] = (double)rand()/RAND_MAX*2-1;
            fr[i] = xr[i]; fi[i] = xi[i];
        }
        dft10_fwd(fr, fi);
        dft10_bwd(fr, fi);
        for (int i = 0; i < 10; i++) { fr[i] /= 10; fi[i] /= 10; }
        double rel = max_rel_err(10, fr, fi, xr, xi);
        int pass = rel < 1e-14;
        printf("  rt  %d: rel=%.2e %s\n", trial, rel, pass?"PASS":"FAIL");
        total++; passed += pass;
    }

    /* ── Rader DFT-11 verification ── */
    printf("\n=== Rader DFT-11 ===\n");
    for (int trial = 0; trial < 10; trial++) {
        double xr[11], xi[11], Nr[11], Ni[11], Rr[11], Ri[11];
        srand(400 + trial);
        for (int i = 0; i < 11; i++) {
            xr[i] = (double)rand()/RAND_MAX*2-1;
            xi[i] = (double)rand()/RAND_MAX*2-1;
        }
        naive_dft(11, -1, xr, xi, Nr, Ni);
        rader_dft11_fwd(xr, xi, Rr, Ri);
        double rel = max_rel_err(11, Rr, Ri, Nr, Ni);
        int pass = rel < 5e-14;
        printf("  fwd %2d: rel=%.2e %s\n", trial, rel, pass?"PASS":"FAIL");
        total++; passed += pass;
    }

    /* DFT-11 backward */
    printf("\n-- DFT-11 backward --\n");
    for (int trial = 0; trial < 5; trial++) {
        double xr[11], xi[11], Nr[11], Ni[11], Rr[11], Ri[11];
        srand(500 + trial);
        for (int i = 0; i < 11; i++) {
            xr[i] = (double)rand()/RAND_MAX*2-1;
            xi[i] = (double)rand()/RAND_MAX*2-1;
        }
        naive_dft(11, +1, xr, xi, Nr, Ni);
        rader_dft11_bwd(xr, xi, Rr, Ri);
        double rel = max_rel_err(11, Rr, Ri, Nr, Ni);
        int pass = rel < 5e-14;
        printf("  bwd %2d: rel=%.2e %s\n", trial, rel, pass?"PASS":"FAIL");
        total++; passed += pass;
    }

    /* DFT-11 roundtrip */
    printf("\n-- DFT-11 roundtrip --\n");
    for (int trial = 0; trial < 5; trial++) {
        double xr[11], xi[11], fr[11], fi[11], br[11], bi[11];
        srand(600 + trial);
        for (int i = 0; i < 11; i++) {
            xr[i] = (double)rand()/RAND_MAX*2-1;
            xi[i] = (double)rand()/RAND_MAX*2-1;
        }
        rader_dft11_fwd(xr, xi, fr, fi);
        rader_dft11_bwd(fr, fi, br, bi);
        for (int i = 0; i < 11; i++) { br[i] /= 11; bi[i] /= 11; }
        double rel = max_rel_err(11, br, bi, xr, xi);
        int pass = rel < 1e-14;
        printf("  rt  %2d: rel=%.2e %s\n", trial, rel, pass?"PASS":"FAIL");
        total++; passed += pass;
    }

    printf("\n==============================\n  %d/%d %s\n==============================\n",
           passed, total, passed==total ? "ALL PASSED" : "FAILURES");
    return passed != total;
}
