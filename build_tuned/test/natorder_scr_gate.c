/* natorder_scr_gate.c — SCR through the PUBLIC API, deterministically. Pre-writes a v7 wisdom file
 * with nat_mode=3 (SCR) so vfft_create takes the stored-verdict SCR-rebuild path (no race lottery),
 * then checks natural forward == naive DFT IN ORDER + natural roundtrip. Includes a REJECTION case
 * (nat_mode=3 stamped on a DIF plan -> natorder_scr_build refuses -> honorable PURE fallback, still
 * correct). Build: python build.py --src test/natorder_scr_gate.c --vfft */
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

static vfft_plan mk(int N, size_t K, int nth)
{
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_INPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = K; c.nthreads = nth; c.order = VFFT_ORDER_NATURAL;
    return vfft_create(&c);
}

static void cell(int N, size_t K, int nf, const int *f, int dif, const int *v, const char *tag)
{
    write_wisdom(N, (int)K, nf, f, dif, v, 3 /*SCR*/);
    size_t n = (size_t)N * K;
    double *re = malloc(n*8), *im = malloc(n*8), *sr = malloc(n*8), *si = malloc(n*8);
    double *x = malloc(n*8), *xi = malloc(n*8), *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    srand(41 + N + (int)K);
    for (size_t i = 0; i < n; i++) { x[i] = (double)rand()/RAND_MAX - .5; xi[i] = (double)rand()/RAND_MAX - .5; }
    naive(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    /* ST fully first (create/exec/destroy), THEN MT — no concurrent plans, so ps's nthreads=1
     * cannot resize the pool under pm's execute (that thrash was the suspected artifact). */
    vfft_plan ps = mk(N, K, 1);
    if (!ps) { printf("  %-10s N=%-4d K=%-3zu ST plan NULL <FAIL>\n", tag, N, K); fails++; goto done; }
    memcpy(sr, x, n*8); memcpy(si, xi, n*8);
    vfft_execute(ps, VFFT_FORWARD, sr, si, sr, si);     /* ST fwd */
    vfft_destroy(ps);
    vfft_plan pm = mk(N, K, 4);
    if (!pm) { printf("  %-10s N=%-4d K=%-3zu MT plan NULL <FAIL>\n", tag, N, K); fails++; goto done; }
    memcpy(re, x, n*8); memcpy(im, xi, n*8);
    vfft_execute(pm, VFFT_FORWARD, re, im, re, im);     /* MT fwd */
    double eF = 0, eMT = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(re[(size_t)k*K]-Xr[k]); if (d>eF) eF = d;
        d = fabs(im[(size_t)k*K]-Xi[k]); if (d>eF) eF = d;
    }
    eF /= (sc > 0 ? sc : 1);
    for (size_t i = 0; i < n; i++) {                    /* MT == ST bit-identical? */
        double d = fabs(re[i]-sr[i]); if (d>eMT) eMT = d;
        d = fabs(im[i]-si[i]); if (d>eMT) eMT = d;
    }
    vfft_execute(pm, VFFT_BACKWARD, re, im, re, im);    /* MT roundtrip */
    double eR = 0, inv = 1.0/N;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(re[i]*inv-x[i]); if (d>eR) eR = d;
        d = fabs(im[i]*inv-xi[i]); if (d>eR) eR = d;
    }
    int bad = (eF > 1e-9) || (eR > 1e-9) || (eMT != 0.0); if (bad) fails++;
    printf("  %-10s N=%-4d K=%-3zu  natural-fwd=%.1e  MT-vs-ST=%.1e  roundtrip=%.1e %s\n",
           tag, N, K, eF, eMT, eR, bad ? "<FAIL>" : "ok");
    vfft_destroy(pm);
done:
    free(re); free(im); free(sr); free(si); free(x); free(xi); free(Xr); free(Xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1);   /* caller pins core 0 for MT */
    putenv("VFFT_WISDOM_DIR=natorder_scr_wis");
    printf("# SCR via public API (forced nat_mode=3): natural fwd==naive + MT==ST bit-identical + roundtrip\n");
    int v2[]={0,2}, v3[]={0,2,2};
    { int f[]={16,16};  cell(256, 4, 2, f, 0, v2, "SCR"); }
    { int f[]={8,16};   cell(128, 4, 2, f, 0, v2, "SCR"); }
    { int f[]={32,32};  cell(1024,4, 2, f, 0, v2, "SCR"); }
    { int f[]={4,4,16}; cell(256, 4, 3, f, 0, v3, "SCR nf3"); }
    { int f[]={8,16};   cell(128, 7, 2, f, 0, v2, "SCR oddK"); }   /* odd tail thru MODEB K-split */
    { int f[]={16,16};  cell(256, 64,2, f, 0, v2, "SCR fatK"); }
    { int f[]={64,64};  cell(4096,64,2, f, 0, v2, "SCR big"); }    /* enough lanes+groups to MT both phases */
    { int f[]={8,8};    cell(64,  4, 2, f, 1, v2, "DIF->PURE"); }  /* SCR rejects DIF -> PURE fallback */
    printf(fails ? "\nSCR GATE: %d FAILURE(S)\n" : "\nSCR GATE PASS: SCR correct + MT==ST bit-identical + honorable DIF fallback\n", fails);
    return fails ? 1 : 0;
}
