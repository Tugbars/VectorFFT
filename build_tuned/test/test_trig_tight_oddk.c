/* test_trig_tight_oddk.c — TIGHT (non-padded) odd-K trig through vfft.h. Was a CRASH (the stride
 * r2c fused first/last stage did an OOP unmasked VW load/store, over-running the remainder lanes).
 * FIX: route a non-VW-aligned B through the explicit-pack fallback (rem-aware inner tail + scalar
 * (un)pack) — see _r2c_worker_fwd/_bwd in r2c.h. This probe checks tight odd-K trig now WORKS:
 *   (A) DCT-II forward vs naive FFTW REDFT10 (the fused-first-stage fix).
 *   (B) DCT-II -> DCT-III roundtrip recovers scale*x (the fused-last-stage / bwd fix).
 * Build: python build.py --src test/test_trig_tight_oddk.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;
static const double PI = 3.14159265358979323846;
static void naive_dct2(const double *x, double *X, int N)
{
    for (int k = 0; k < N; k++)
    { double s = 0; for (int n = 0; n < N; n++) s += x[n] * cos(PI * (2 * n + 1) * k / (2.0 * N)); X[k] = 2.0 * s; }
}

static void probe(int N, int K)
{
    size_t nk = (size_t)N * K;
    double *x = (double *)malloc(nk * sizeof(double));
    double *X = (double *)calloc(nk, sizeof(double));   /* fwd output */
    double *y = (double *)calloc(nk, sizeof(double));   /* roundtrip output */
    srand(3 + N + K);
    for (size_t i = 0; i < nk; i++) x[i] = (double)rand() / RAND_MAX - 0.5;

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_DCT2; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = (size_t)K;   /* NO batch = tight */
    printf("  DCT2 N=%-4d K=%-3d rem%d  creating...\n", N, K, K % 4); fflush(stdout);
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("    -> create NULL (gated)\n"); free(x); free(X); free(y); return; }
    printf("    created; fwd...\n"); fflush(stdout);
    vfft_execute(p, VFFT_FORWARD,  x, NULL, X, NULL);   /* DCT-II */
    printf("    fwd ok; bwd...\n"); fflush(stdout);
    vfft_execute(p, VFFT_BACKWARD, X, NULL, y, NULL);   /* DCT-III (inverse) */
    printf("    bwd ok\n"); fflush(stdout);

    /* (A) forward vs naive */
    double *Xr = (double *)malloc(N * sizeof(double)), *xl = (double *)malloc(N * sizeof(double));
    double fe = 0;
    for (int k = 0; k < K; k++)
    {
        for (int n = 0; n < N; n++) xl[n] = x[n * K + k];
        naive_dct2(xl, Xr, N);
        for (int kk = 0; kk < N; kk++) { double d = fabs(X[(size_t)kk * K + k] - Xr[kk]); if (d > fe) fe = d; }
    }
    /* (B) roundtrip: least-squares scale, residual */
    double sxy = 0, sxx = 0;
    for (size_t i = 0; i < nk; i++) { sxy += x[i] * y[i]; sxx += x[i] * x[i]; }
    double sc = sxx > 0 ? sxy / sxx : 0, re = 0, denom = 0;
    for (size_t i = 0; i < nk; i++) { double a = x[i] * sc, d = fabs(y[i] - a); if (d > re) re = d; if (fabs(a) > denom) denom = fabs(a); }
    if (denom > 0) re /= denom;

    int bad = (fe > 1e-10) || (re > 1e-9);
    if (bad) fails++;
    printf("  DCT2 N=%-4d K=%-3d rem%d  fwd|vs-naive|=%9.1e  roundtrip=%9.1e  %s\n",
           N, K, K % 4, fe, re, bad ? "*** FAIL ***" : "ok");
    free(Xr); free(xl); vfft_destroy(p); free(x); free(X); free(y);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=trig_tight_probe");
    system("mkdir trig_tight_probe 2>nul");
    printf("# TIGHT (non-padded) odd-K trig through vfft.h — should now WORK via the stride explicit-pack fallback\n");
    probe(64, 8);    /* even/aligned control */
    probe(64, 7);    /* odd (rem3) */
    probe(64, 11);
    probe(128, 7);
    probe(256, 7);
    probe(256, 15);
    probe(256, 13);  /* rem1 */
    probe(512, 23);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: tight odd-K trig works (fwd vs naive + roundtrip)\n", fails);
    return fails ? 1 : 0;
}
