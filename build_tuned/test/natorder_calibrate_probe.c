/* natorder_calibrate_probe.c — run the REAL natural-order calibrator race (no forced wisdom) on a
 * few cells and report what it picks (FREE/PURE/PSWAP/SCR) + correctness. Fresh wisdom dir => the
 * create path calibrates the c2c chain (dp_best) then races the natural modes and stamps nat_mode.
 * Build: python build.py --src test/natorder_calibrate_probe.c --vfft */
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
            double a = -2.0 * M_PI * (double)k * n / N, c = cos(a), s = sin(a);
            sr += re[(size_t)n * K] * c - im[(size_t)n * K] * s;
            si += re[(size_t)n * K] * s + im[(size_t)n * K] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

static void cell(int N, size_t K)
{
    size_t n = (size_t)N * K;
    double *re = malloc(n*8), *im = malloc(n*8), *x = malloc(n*8), *xi = malloc(n*8);
    double *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    srand(5 + N + (int)K);
    for (size_t i = 0; i < n; i++) { x[i] = (double)rand()/RAND_MAX - .5; xi[i] = (double)rand()/RAND_MAX - .5; }
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = VFFT_ORDER_NATURAL;
    printf("  N=%-5d K=%-3zu racing...", N, K); fflush(stdout);
    vfft_plan p = vfft_create(&c);              /* calibrates c2c + races natural + stamps */
    if (!p) { printf(" plan NULL <FAIL>\n"); fails++; goto done; }

    memcpy(re, x, n*8); memcpy(im, xi, n*8);
    vfft_execute(p, VFFT_FORWARD, re, im, re, im);
    double eF = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(re[(size_t)k*K]-Xr[k]); if (d>eF) eF = d;
        d = fabs(im[(size_t)k*K]-Xi[k]); if (d>eF) eF = d;
    }
    eF /= (sc > 0 ? sc : 1);
    vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
    double eR = 0, inv = 1.0/N;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(re[i]*inv-x[i]); if (d>eR) eR = d;
        d = fabs(im[i]*inv-xi[i]); if (d>eR) eR = d;
    }
    int bad = (eF > 1e-9) || (eR > 1e-9); if (bad) fails++;
    printf(" done  natural-fwd=%.1e roundtrip=%.1e %s\n", eF, eR, bad ? "<FAIL>" : "ok");
    vfft_destroy(p);
done:
    free(re); free(im); free(x); free(xi); free(Xr); free(Xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);
    system("rmdir /s /q natorder_cal_probe 2>nul"); system("mkdir natorder_cal_probe 2>nul");
    putenv("VFFT_WISDOM_DIR=natorder_cal_probe");
    printf("# natural-order calibrator race (fresh wisdom) — what does each cell pick?\n");
    cell(64, 4);      /* small — FREE (nf=1) likely */
    cell(128, 4);     /* K=4 multi-stage — SCR candidate */
    cell(256, 4);
    cell(1024, 4);
    cell(4096, 4);
    cell(1024, 32);   /* PURE band */
    cell(4096, 32);
    cell(256, 256);
    cell(128, 64);    /* PSWAP won historically */
    printf(fails ? "\n%d FAILURE(S)\n" : "\nALL CORRECT — see nat_mode stamps in the wisdom dump\n", fails);
    return fails ? 1 : 0;
}
