/* test_oop_bailey_natural.c — Phase B: natural-order odd-K BAILEY2 (N>128).
 *
 * Forces BAILEY2 pairs (vfft_oop_plan_create_pair_v) at odd K and compares the
 * forward output to a brute-force O(N^2) DFT in NATURAL order (X[k] at dr[k*K+b]).
 * This isolates the per-lane t1_oop codelet + the per-group twiddle table at odd K
 * (the per-block t1p would straddle k2 boundaries and fail). Even K (8,16) runs the
 * fast per-block t1p path as a control.
 *
 * Build: python build.py --src test/test_oop_bailey_natural.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

static double bailey_vs_naive(int N, int R1, int R2, size_t K)
{
    vfft_oop_plan_t *op = vfft_oop_plan_create_pair_v(N, K, R1, R2, /*flat*/0);
    if (!op) return -2.0;                 /* pair rejected */
    if (op->kind != VFFT_OOP_KIND_BAILEY2) { vfft_oop_plan_destroy(op); return -3.0; }
    size_t tot = (size_t)N * K;
    double *sre = ad(tot), *sim = ad(tot), *dre = ad(tot), *dim = ad(tot);
    srand(17 + N + R1 + (int)K);
    for (size_t t = 0; t < tot; t++) {
        sre[t] = (double)rand()/RAND_MAX - 0.5;
        sim[t] = (double)rand()/RAND_MAX - 0.5;
    }
    vfft_oop_execute_fwd(op, sre, sim, dre, dim);
    double md = 0.0;
    for (size_t b = 0; b < K; b++)
        for (int k = 0; k < N; k++) {
            double xr = 0, xi = 0;
            for (int n = 0; n < N; n++) {
                double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
                double c = cos(ang), s = sin(ang);
                double ar = sre[(size_t)n*K + b], ai = sim[(size_t)n*K + b];
                xr += ar*c - ai*s; xi += ar*s + ai*c;
            }
            double er = fabs(dre[(size_t)k*K + b] - xr);
            double ei = fabs(dim[(size_t)k*K + b] - xi);
            double e = er > ei ? er : ei;
            if (e > md) md = e;
        }
    afree(sre); afree(sim); afree(dre); afree(dim);
    vfft_oop_plan_destroy(op);
    return md;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    (void)reg;

    struct { int N, R1, R2; } pairs[] = {
        {256, 16, 16}, {256, 4, 64}, {256, 8, 32}, {256, 32, 8}, {512, 8, 64}, {1024, 16, 64},
    };
    int nP = (int)(sizeof(pairs)/sizeof(pairs[0]));
    size_t Ks[] = {1, 3, 5, 7, 9, 15, 17, 31, 33, 8, 16};
    int nK = (int)(sizeof(Ks)/sizeof(Ks[0]));

    int fails = 0;
    for (int p = 0; p < nP; p++) {
        printf("\n=== BAILEY2  N=%d  %dx%d ===\n", pairs[p].N, pairs[p].R1, pairs[p].R2);
        printf("  %-5s %-5s %-12s\n", "K", "rem", "vs_naiveDFT");
        for (int i = 0; i < nK; i++) {
            size_t K = Ks[i];
            double md = bailey_vs_naive(pairs[p].N, pairs[p].R1, pairs[p].R2, K);
            const char *flag = "";
            if (md == -2.0) { printf("  %-5zu %-5zu pair-rejected\n", K, K%4); continue; }
            if (md == -3.0) { printf("  %-5zu %-5zu not-bailey\n", K, K%4); continue; }
            double tol = 1e-9 * (double)pairs[p].N;
            if (md > tol || md != md) { flag = " <FAIL>"; fails++; }
            printf("  %-5zu %-5zu %-12.2e%s\n", K, K%4, md, flag);
        }
    }
    printf("\n%s  (%d fail)\n", fails==0 ? "BAILEY2: NATURAL-ORDER bit-correct at odd K" : "FAILURES", fails);
    return fails ? 1 : 0;
}
