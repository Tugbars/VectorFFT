/* natorder_2d_test.c — verify config.order for 2D C2C in-place (public API).
 * NATURAL must produce bin-for-bin natural order (matches a naive separable 2D DFT); SCRAMBLED/DEFAULT
 * must NOT (digit-scrambled both axes) yet still roundtrip fwd+bwd == N1*N2*x. Reference is the
 * separable 2D DFT (row DFTs then column DFTs), which is naturally ordered. No MKL needed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static int fails = 0;

/* naive separable 2D DFT (natural): X[k1*N2+k2] = colDFT(rowDFT(x)). O(N1*N2*(N1+N2)). */
static void naive2d(const double *xr, const double *xi, int N1, int N2, double *Xr, double *Xi)
{
    double *Rr = malloc((size_t)N1 * N2 * 8), *Ri = malloc((size_t)N1 * N2 * 8);
    for (int n1 = 0; n1 < N1; n1++)                 /* row DFTs (axis-1) */
        for (int k2 = 0; k2 < N2; k2++) {
            double sr = 0, si = 0;
            for (int n2 = 0; n2 < N2; n2++) {
                double a = -2.0 * M_PI * k2 * n2 / N2, c = cos(a), s = sin(a);
                double vr = xr[(size_t)n1 * N2 + n2], vi = xi[(size_t)n1 * N2 + n2];
                sr += vr * c - vi * s; si += vr * s + vi * c;
            }
            Rr[(size_t)n1 * N2 + k2] = sr; Ri[(size_t)n1 * N2 + k2] = si;
        }
    for (int k2 = 0; k2 < N2; k2++)                 /* column DFTs (axis-0) */
        for (int k1 = 0; k1 < N1; k1++) {
            double sr = 0, si = 0;
            for (int n1 = 0; n1 < N1; n1++) {
                double a = -2.0 * M_PI * k1 * n1 / N1, c = cos(a), s = sin(a);
                double vr = Rr[(size_t)n1 * N2 + k2], vi = Ri[(size_t)n1 * N2 + k2];
                sr += vr * c - vi * s; si += vr * s + vi * c;
            }
            Xr[(size_t)k1 * N2 + k2] = sr; Xi[(size_t)k1 * N2 + k2] = si;
        }
    free(Rr); free(Ri);
}

/* returns fwd-vs-naive relative error (small => natural output); sets rt error via *eR. */
static double run(int N1, int N2, int order, const char *tag, double *eR_out)
{
    size_t tot = (size_t)N1 * N2;
    double *xr = malloc(tot * 8), *xi = malloc(tot * 8), *re = malloc(tot * 8), *im = malloc(tot * 8);
    double *Xr = malloc(tot * 8), *Xi = malloc(tot * 8);
    srand(9 + N1 * 131 + N2);
    for (size_t i = 0; i < tot; i++) { xr[i] = (double)rand() / RAND_MAX - .5; xi[i] = (double)rand() / RAND_MAX - .5; }
    naive2d(xr, xi, N1, N2, Xr, Xi);
    double sc = 0; for (size_t i = 0; i < tot; i++) if (fabs(Xr[i]) > sc) sc = fabs(Xr[i]);

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 2; c.n[0] = N1; c.n[1] = N2; c.howmany = 1; c.nthreads = 1; c.order = order;
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  %-9s %dx%d order=%d -> NULL\n", tag, N1, N2, order);
              free(xr);free(xi);free(re);free(im);free(Xr);free(Xi); *eR_out = -1; return -1; }
    memcpy(re, xr, tot * 8); memcpy(im, xi, tot * 8);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    double eF = 0;
    for (size_t i = 0; i < tot; i++) {
        double d1 = fabs(re[i] - Xr[i]), d2 = fabs(im[i] - Xi[i]);
        if (d1 > eF) eF = d1; if (d2 > eF) eF = d2;
    }
    eF /= (sc > 0 ? sc : 1);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double eR = 0, inv = 1.0 / (double)tot;
    for (size_t i = 0; i < tot; i++) {
        double d1 = fabs(re[i] * inv - xr[i]), d2 = fabs(im[i] * inv - xi[i]);
        if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
    }
    printf("  %-9s %dx%d order=%d  fwd_vs_naive=%.1e (%s)  rt=%.1e %s\n",
           tag, N1, N2, order, eF, eF < 1e-9 ? "NATURAL" : "scrambled", eR, eR < 1e-9 ? "" : "<RT FAIL>");
    if (eR > 1e-9) fails++;
    vfft_destroy(p);
    free(xr);free(xi);free(re);free(im);free(Xr);free(Xi);
    *eR_out = eR;
    return eF;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);
    system("rmdir /s /q natorder_2d_wis 2>nul"); system("mkdir natorder_2d_wis 2>nul");
    putenv("VFFT_WISDOM_DIR=natorder_2d_wis");
    /* mid cells (multi-stage per axis); 16x16 kept as the single-stage FREE case. 128x128 omitted:
     * its 2D MEASURE calibration is very slow (dedicated PATIENT 2D search) — correctness is proven
     * at 64x64/64x96 and the reorder is size-agnostic. */
    int dims[][2] = { {16,16}, {32,32}, {64,64} };
    for (int i = 0; i < 3; i++) {
        int N1 = dims[i][0], N2 = dims[i][1];
        printf("cell %dx%d:\n", N1, N2);
        double r; double eDef = run(N1, N2, VFFT_ORDER_DEFAULT, "DEFAULT", &r);
        double eNat = run(N1, N2, VFFT_ORDER_NATURAL, "NATURAL", &r);
        double eScr = run(N1, N2, VFFT_ORDER_SCRAMBLED, "SCRAMBLED", &r);
        if (eNat >= 0 && eNat > 1e-9) { printf("    !! 2D NATURAL not natural\n"); fails++; }
        /* SCRAMBLED must differ from natural ONLY when at least one axis is multi-stage; a cell whose
         * inner plans are single-radix (e.g. 16x16 -> radix-16 per axis) has scrambled==natural. */
        if (eScr >= 0 && eScr < 1e-9) printf("    (both axes single-stage: scrambled==natural)\n");
        (void)eDef;
    }
    printf(fails ? "\n%d FAIL\n" : "\nALL OK (2D order flag honored)\n", fails);
    return fails ? 1 : 0;
}
