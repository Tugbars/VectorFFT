/* natorder_scr_gate.c — SCR through the PUBLIC API, deterministically. Pre-writes a v7 wisdom file
 * with nat_mode=3 (SCR) so vfft_create takes the stored-verdict SCR-rebuild path (no race lottery),
 * then checks natural forward == naive DFT IN ORDER + natural roundtrip. Includes a REJECTION case
 * (nat_mode=3 stamped on a DIF plan -> natorder_scr_build refuses -> honorable PURE fallback, still
 * correct). Build: python build.py --src test/natorder_scr_gate.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

/* write a single-entry v7 wisdom file forcing this cell's nat_mode */
static void write_wisdom(int N, int K, int nf, const int *f, int dif, const int *v, int nat_mode)
{
    system("mkdir natorder_scr_wis 2>nul");
    FILE *fp = fopen("natorder_scr_wis/spike_wisdom.txt", "w");
    fprintf(fp, "@version 7\n# forced SCR gate\n");
    fprintf(fp, "%d %d %d", N, K, nf);
    for (int i = 0; i < nf; i++) fprintf(fp, " %d", f[i]);
    fprintf(fp, " 1000.00 0 0 0 %d", dif);       /* best_ns ub ss bg dif */
    for (int i = 0; i < nf; i++) fprintf(fp, " %d", v[i]);
    fprintf(fp, " 0 %d 900.00\n", nat_mode);     /* exec_me nat_mode nat_ns */
    fclose(fp);
}

static void cell(int N, size_t K, int nf, const int *f, int dif, const int *v, const char *tag)
{
    write_wisdom(N, (int)K, nf, f, dif, v, 3 /*SCR*/);
    size_t n = (size_t)N * K;
    double *re = malloc(n*8), *im = malloc(n*8), *x = malloc(n*8), *xi = malloc(n*8);
    double *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    srand(41 + N + (int)K);
    for (size_t i = 0; i < n; i++) { x[i] = (double)rand()/RAND_MAX - .5; xi[i] = (double)rand()/RAND_MAX - .5; }
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = 1; c.order = VFFT_ORDER_NATURAL;
    vfft_plan p = vfft_create(&c);
    if (!p) { printf("  %-10s N=%-4d K=%-3zu plan NULL <FAIL>\n", tag, N, K); fails++; goto done; }

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
    printf("  %-10s N=%-4d K=%-3zu  natural-fwd=%.1e  roundtrip=%.1e %s\n",
           tag, N, K, eF, eR, bad ? "<FAIL>" : "ok");
    vfft_destroy(p);
done:
    free(re); free(im); free(x); free(xi); free(Xr); free(Xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=natorder_scr_wis");
    printf("# SCR via public API (forced nat_mode=3 in wisdom): natural fwd==naive + roundtrip\n");
    int v2[]={0,2}, v3[]={0,2,2};
    { int f[]={16,16};  cell(256, 4, 2, f, 0, v2, "SCR"); }
    { int f[]={8,16};   cell(128, 4, 2, f, 0, v2, "SCR"); }
    { int f[]={32,32};  cell(1024,4, 2, f, 0, v2, "SCR"); }
    { int f[]={4,4,16}; cell(256, 4, 3, f, 0, v3, "SCR nf3"); }
    { int f[]={8,16};   cell(128, 7, 2, f, 0, v2, "SCR oddK"); }
    { int f[]={16,16};  cell(256, 64,2, f, 0, v2, "SCR fatK"); }
    { int f[]={8,8};    cell(64,  4, 2, f, 1, v2, "DIF->PURE"); }  /* SCR rejects DIF -> PURE fallback */
    printf(fails ? "\nSCR GATE: %d FAILURE(S)\n" : "\nSCR GATE PASS: scatter terminator correct via public API (fwd+bwd) + honorable DIF fallback\n", fails);
    return fails ? 1 : 0;
}
