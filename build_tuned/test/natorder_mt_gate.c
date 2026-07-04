/* natorder_mt_gate.c — P1b step 2: ORDER_NATURAL reorder pass under MT (nthreads>1).
 * The pass splits by CYCLE/PAIR ranges (full K-wide rows), never K. Gates per cell:
 *   A. MT natural forward (lane 0) == naive DFT IN ORDER
 *   B. MT natural forward == ST natural forward, BIT-IDENTICAL (the split is exact)
 *   C. MT natural roundtrip == N*input
 * Cells: PURE band (1024/32, 4096/4, 256/256) + PSWAP (128/64) + FREE (64/4, ST-guarded).
 * Caller pins core 0 (workers pin 1..T-1) — same contract as c2c/OOP MT.
 * Build: python build.py --src test/natorder_mt_gate.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define T 4

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
static vfft_plan mk(int N, size_t K, int nth)
{
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = nth; c.order = VFFT_ORDER_NATURAL;
    return vfft_create(&c);
}

static void cell(int N, size_t K)
{
    size_t n = (size_t)N * K;
    double *m = malloc(n*8), *mi = malloc(n*8), *s = malloc(n*8), *si = malloc(n*8);
    double *x = malloc(n*8), *xi = malloc(n*8), *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    srand(19 + N + (int)K);
    for (size_t i = 0; i < n; i++) { x[i] = (double)rand()/RAND_MAX - .5; xi[i] = (double)rand()/RAND_MAX - .5; }
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    /* seed the verdict with an ST create so MT and ST use the SAME nat_mode */
    { vfft_plan p0 = mk(N, K, 1); if (p0) vfft_destroy(p0); }
    vfft_plan pm = mk(N, K, T), ps = mk(N, K, 1);
    if (!pm || !ps) { printf("  N=%-5d K=%-3zu plan NULL <FAIL>\n", N, K); fails++; goto done; }

    memcpy(m, x, n*8); memcpy(mi, xi, n*8);
    vfft_execute(pm, VFFT_FORWARD, m, mi, m, mi);       /* MT fwd */
    memcpy(s, x, n*8); memcpy(si, xi, n*8);
    vfft_execute(ps, VFFT_FORWARD, s, si, s, si);       /* ST fwd */

    double eA = 0, eB = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(m[(size_t)k*K]-Xr[k]); if (d>eA) eA = d;
        d = fabs(mi[(size_t)k*K]-Xi[k]); if (d>eA) eA = d;
    }
    eA /= (sc > 0 ? sc : 1);
    for (size_t i = 0; i < n; i++) {
        double d = fabs(m[i]-s[i]); if (d>eB) eB = d;
        d = fabs(mi[i]-si[i]); if (d>eB) eB = d;
    }
    vfft_execute(pm, VFFT_BACKWARD, m, mi, m, mi);      /* MT roundtrip */
    double eC = 0, inv = 1.0/N;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(m[i]*inv-x[i]); if (d>eC) eC = d;
        d = fabs(mi[i]*inv-xi[i]); if (d>eC) eC = d;
    }
    int bad = (eA > 1e-9) || (eB != 0.0) || (eC > 1e-9);
    if (bad) fails++;
    printf("  N=%-5d K=%-3zu  MT-fwd-vs-naive=%.1e  MT-vs-ST=%.1e  MT-rt=%.1e %s\n",
           N, K, eA, eB, eC, bad ? "<FAIL>" : "ok");
    vfft_destroy(pm); vfft_destroy(ps);
done:
    free(m); free(mi); free(s); free(si); free(x); free(xi); free(Xr); free(Xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);   /* caller pins core 0 */
    putenv("VFFT_WISDOM_DIR=natorder_wis_p0");
    printf("# P1b MT gate (nthreads=%d): natural reorder cycle/pair-split == ST, bit-identical\n", T);
    cell(64, 4);      /* FREE */
    cell(128, 64);    /* PSWAP pair-split */
    cell(1024, 32);   /* PURE cycle-split */
    cell(256, 256);   /* PURE, big rows */
    cell(4096, 4);    /* PURE, many small-row cycles */
    cell(4096, 32);   /* PURE */
    printf(fails ? "\nMT GATE: %d FAILURE(S)\n" : "\nMT GATE PASS: natural reorder MT correct + bit-identical to ST\n", fails);
    return fails ? 1 : 0;
}
