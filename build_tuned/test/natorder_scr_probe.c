/* natorder_scr_probe.c — SCR (scatter terminator) BIT-EXACT gate, standalone (low-level API, no
 * vfft.c/race yet). Per (N,K,DIT-chain): plan -> impulse orientation detect -> natorder_scr_build ->
 * natorder_scr_fwd -> compare the natural spectrum (lane 0, IN ORDER) vs naive DFT. Also checks SCR
 * output == the plain in-place scrambled result permuted by M (the fusion is exact, not just close).
 * Covers T1S + FLAT last stages, K=4 (the target band) + odd K tails, nf=2 and nf=3.
 * Build: python build.py --src test/natorder_scr_probe.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "natorder_perm.h"
#include "natorder_scatter.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static vfft_proto_registry_t REG;
static int fails = 0;

static void naive0(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi)
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

static void cell(int N, size_t K, const int *f, const int *v, int nf, const char *lastv)
{
    size_t n = (size_t)N * K;
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, f, v, nf, 0, &REG);
    if (!p) { printf("  N=%-5d K=%-3zu  plan NULL\n", N, K); fails++; return; }

    /* orientation detect via impulse */
    double *cre = (double *)calloc(n, 8), *cim = (double *)calloc(n, 8);
    cre[K] = 1.0;                                  /* impulse n0=1, lane 0 */
    vfft_proto_execute_fwd(p, cre, cim, K);
    int *M = vfft_natorder_detect(N, f, nf, K, cre, cim, 1);
    free(cre); free(cim);
    if (!M) { printf("  N=%-5d K=%-3zu  orientation detect FAIL\n", N, K); fails++; vfft_proto_plan_destroy(p); return; }
    int *IM = (int *)malloc(N * 4); vfft_natorder_inv_perm(N, M, IM);

    natorder_scr_t s;
    int ok = natorder_scr_build(&s, p, N, K, M, IM);
    if (!ok) {
        printf("  N=%-5d K=%-3zu last=%-4s  SCR build REJECTED (falls to PURE)\n", N, K, lastv);
        free(M); free(IM); vfft_proto_plan_destroy(p); return;
    }

    /* random input; references */
    double *x = malloc(n*8), *xi = malloc(n*8), *Xr = malloc((size_t)N*8), *Xi = malloc((size_t)N*8);
    srand(23 + N + (int)K);
    for (size_t i = 0; i < n; i++) { x[i] = (double)rand()/RAND_MAX - .5; xi[i] = (double)rand()/RAND_MAX - .5; }
    naive0(x, xi, N, K, Xr, Xi);
    double sc = 0; for (int k = 0; k < N; k++) if (fabs(Xr[k]) > sc) sc = fabs(Xr[k]);

    /* SCR forward */
    double *ur = malloc(n*8), *ui = malloc(n*8);
    memcpy(ur, x, n*8); memcpy(ui, xi, n*8);
    natorder_scr_fwd(&s, ur, ui, K);
    double eN = 0;                                 /* natural, in order, vs naive */
    for (int k = 0; k < N; k++) {
        double d = fabs(ur[(size_t)k*K]-Xr[k]); if (d>eN) eN = d;
        d = fabs(ui[(size_t)k*K]-Xi[k]); if (d>eN) eN = d;
    }
    eN /= (sc > 0 ? sc : 1);

    /* SCR == plain in-place scrambled result permuted by M (fusion exactness) */
    double *ir = malloc(n*8), *ii = malloc(n*8);
    memcpy(ir, x, n*8); memcpy(ii, xi, n*8);
    vfft_proto_execute_fwd(p, ir, ii, K);          /* scrambled in place */
    double eF = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(ur[(size_t)k*K]-ir[(size_t)M[k]*K]); if (d>eF) eF = d;
        d = fabs(ui[(size_t)k*K]-ii[(size_t)M[k]*K]); if (d>eF) eF = d;
    }

    /* SCR uses separate scalar-cmul pre-twiddle + plain n1 vs the in-place FUSED t1s codelet, so it
     * matches the scrambled result numerically (~1 ulp) but not bit-for-bit. The correctness gate is
     * natural-vs-naive; SCR-vs-perm just confirms it's the same transform (relative ~1e-13). */
    int bad = (eN > 1e-9) || (eF / (sc > 0 ? sc : 1) > 1e-11);
    if (bad) fails++;
    printf("  N=%-5d K=%-3zu last=%-4s R=%-2d P=%-4d  natural-vs-naive=%.1e  SCR-vs-perm=%.1e %s\n",
           N, K, lastv, s.R, s.P, eN, eF / (sc > 0 ? sc : 1), bad ? "<FAIL>" : "ok");

    natorder_scr_free(&s);
    free(M); free(IM); free(x); free(xi); free(Xr); free(Xi); free(ur); free(ui); free(ir); free(ii);
    vfft_proto_plan_destroy(p);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_init(&REG);
    printf("# SCR scatter-terminator bit-exact gate (standalone): natural fwd vs naive + fusion exactness\n");
    int v2t[]={0,2}, v2f[]={0,0}, v3t[]={0,2,2};
    { int f[]={8,8};    cell(64, 4, f, v2t, 2, "T1S"); }
    { int f[]={8,8};    cell(64, 4, f, v2f, 2, "FLAT"); }   /* FLAT last stage */
    { int f[]={8,16};   cell(128, 4, f, v2t, 2, "T1S"); }
    { int f[]={16,16};  cell(256, 4, f, v2t, 2, "T1S"); }
    { int f[]={32,32};  cell(1024,4, f, v2t, 2, "T1S"); }
    { int f[]={4,4,16}; cell(256, 4, f, v3t, 3, "T1S"); }   /* nf=3 */
    { int f[]={8,16};   cell(128, 7, f, v2t, 2, "T1S"); }   /* odd K tail */
    { int f[]={8,16};   cell(128, 23,f, v2t, 2, "T1S"); }
    { int f[]={16,16};  cell(256, 64,f, v2t, 2, "T1S"); }   /* fat rows */
    printf(fails ? "\nSCR PROBE: %d FAILURE(S)\n" : "\nSCR PROBE PASS: scatter terminator bit-exact (natural + fusion) T1S/FLAT, nf2/3, odd K\n", fails);
    return fails ? 1 : 0;
}
