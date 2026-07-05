/* natorder_oop_order_test.c — verify config.order is honored for OOP c2c.
 * NATURAL must produce bin-for-bin natural output (matches a naive DFT); SCRAMBLED must NOT (it
 * rides MODEB) yet still roundtrip; DEFAULT is whatever wins (reported). Fresh wisdom dir => every
 * create hits the order-constrained chooser (vfft_oop_plan_create_order). No MKL needed. */
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
static void naive(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * k * n / N, c = cos(a), s = sin(a);
            sr += re[(size_t)n * K] * c - im[(size_t)n * K] * s;
            si += re[(size_t)n * K] * s + im[(size_t)n * K] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}
/* one (N,K,order) case: returns fwd-vs-naive relative error (small => natural output). */
static double run(int N, size_t K, int order, const char *tag)
{
    size_t tot = (size_t)N * K;
    double *x = malloc(tot * 8), *xi = malloc(tot * 8);
    double *sr = malloc(tot * 8), *si = malloc(tot * 8), *dr = malloc(tot * 8), *di = malloc(tot * 8);
    double *Xr = malloc((size_t)N * 8), *Xi = malloc((size_t)N * 8);
    srand(11 + N + (int)K);
    for (size_t i = 0; i < tot; i++) { x[i] = (double)rand() / RAND_MAX - .5; xi[i] = (double)rand() / RAND_MAX - .5; }
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = order;
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  %-9s N=%-5d K=%-4zu  order=%d -> NULL\n", tag, N, (size_t)K, order);
              free(x);free(xi);free(sr);free(si);free(dr);free(di);free(Xr);free(Xi); return -1; }
    memcpy(sr, x, tot * 8); memcpy(si, xi, tot * 8);
    vfft_execute(p, VFFT_FORWARD, sr, si, dr, di);
    double eF = 0;
    for (int k = 0; k < N; k++) {
        double d1 = fabs(dr[(size_t)k * K] - Xr[k]), d2 = fabs(di[(size_t)k * K] - Xi[k]);
        if (d1 > eF) eF = d1; if (d2 > eF) eF = d2;
    }
    eF /= (sc > 0 ? sc : 1);
    /* roundtrip: bwd(dr,di) -> recover N*x (OOP writes back into sr/si) */
    vfft_execute(p, VFFT_BACKWARD, dr, di, sr, si);
    double eR = 0, inv = 1.0 / N;
    for (size_t i = 0; i < tot; i++) {
        double d1 = fabs(sr[i] * inv - x[i]), d2 = fabs(si[i] * inv - xi[i]);
        if (d1 > eR) eR = d1; if (d2 > eR) eR = d2;
    }
    const char *ord = (eF < 1e-9) ? "NATURAL" : "scrambled";
    printf("  %-9s N=%-5d K=%-4zu  order=%d  fwd_vs_naive=%.1e (%s)  rt=%.1e %s\n",
           tag, N, (size_t)K, order, eF, ord, eR, eR < 1e-9 ? "" : "<RT FAIL>");
    if (eR > 1e-9) fails++;
    vfft_destroy(p);
    free(x);free(xi);free(sr);free(si);free(dr);free(di);free(Xr);free(Xi);
    return eF;
}
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);
    system("rmdir /s /q natorder_oopord_wis 2>nul"); system("mkdir natorder_oopord_wis 2>nul");
    putenv("VFFT_WISDOM_DIR=natorder_oopord_wis");
    /* cells where DEFAULT picks MODEB (scrambled) per v1_0 -> NATURAL must OVERRIDE to natural. */
    int Ns[] = {64, 256, 1024}; size_t Ks[] = {256, 32, 256};
    for (int i = 0; i < 3; i++) {
        int N = Ns[i]; size_t K = Ks[i];
        printf("cell N=%d K=%zu:\n", N, K);
        double eDef = run(N, K, VFFT_ORDER_DEFAULT, "DEFAULT");
        double eNat = run(N, K, VFFT_ORDER_NATURAL, "NATURAL");
        double eScr = run(N, K, VFFT_ORDER_SCRAMBLED, "SCRAMBLED");
        /* contract: NATURAL must be natural (small fwd err); SCRAMBLED must be scrambled (large). */
        if (eNat >= 0 && eNat > 1e-9) { printf("    !! NATURAL did NOT produce natural order\n"); fails++; }
        if (eScr >= 0 && eScr < 1e-9) { printf("    !! SCRAMBLED produced natural order (expected scrambled)\n"); fails++; }
        (void)eDef;
    }
    printf(fails ? "\n%d FAIL\n" : "\nALL OK (order flag honored for OOP)\n", fails);
    return fails ? 1 : 0;
}
