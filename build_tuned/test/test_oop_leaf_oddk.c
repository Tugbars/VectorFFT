/* test_oop_leaf_oddk.c — LEAF OOP kind (N<=128, single n1_oop codelet) at odd K vs naive DFT.
 * Forces LEAF via vfft_oop_plan_create at small N; the LEAF codelet runs me=K with the rem-aware
 * tail, so it should serve ANY K. Natural order: X[k] at dr[k*K+b]. (Companion to the BAILEY2 test.)
 * Build: python build.py --src test/test_oop_leaf_oddk.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static vfft_proto_registry_t REG;
static double *ad(size_t n) { double *p=NULL; if (vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0) exit(1); return p; }
static void afree(double *p) { vfft_proto_aligned_free(p); }

static double leaf_vs_naive(int N, size_t K)
{
    vfft_oop_plan_t *op = vfft_oop_plan_create(N, K, NULL, 0, &REG);
    if (!op) return -2.0;
    if (op->kind != VFFT_OOP_KIND_LEAF) { vfft_oop_plan_destroy(op); return -3.0; }
    size_t tot = (size_t)N * K;
    double *sre=ad(tot), *sim=ad(tot), *dre=ad(tot), *dim=ad(tot);
    srand(23 + N + (int)K);
    for (size_t t = 0; t < tot; t++) { sre[t]=(double)rand()/RAND_MAX-0.5; sim[t]=(double)rand()/RAND_MAX-0.5; }
    vfft_oop_execute_fwd(op, sre, sim, dre, dim);
    double md = 0.0;
    for (size_t b = 0; b < K; b++)
        for (int k = 0; k < N; k++) {
            double xr=0, xi=0;
            for (int n = 0; n < N; n++) {
                double ang = -2.0*M_PI*(double)((long)n*k % N)/(double)N, c=cos(ang), s=sin(ang);
                double ar=sre[(size_t)n*K+b], ai=sim[(size_t)n*K+b];
                xr += ar*c - ai*s; xi += ar*s + ai*c;
            }
            double er=fabs(dre[(size_t)k*K+b]-xr), ei=fabs(dim[(size_t)k*K+b]-xi), e=er>ei?er:ei;
            if (e > md) md = e;
        }
    /* roundtrip too (fwd then bwd == N*x) */
    double *bre=ad(tot), *bim=ad(tot);
    vfft_oop_execute_bwd(op, dre, dim, bre, bim);
    double rt = 0.0, inv = 1.0/(double)N;
    for (size_t t = 0; t < tot; t++) { double dr=fabs(bre[t]*inv-sre[t]), di=fabs(bim[t]*inv-sim[t]); if(dr>rt)rt=dr; if(di>rt)rt=di; }
    afree(sre);afree(sim);afree(dre);afree(dim);afree(bre);afree(bim);
    vfft_oop_plan_destroy(op);
    return md > rt ? md : rt;   /* worst of naive-match and roundtrip */
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_init(&REG);
    int Ns[] = {16, 32, 64, 128};
    size_t Ks[] = {1, 3, 5, 7, 9, 15, 17, 31, 33, 8, 16};
    int fails = 0;
    printf("# LEAF OOP (N<=128, single n1_oop codelet) at odd K — vs naive natural DFT + roundtrip\n");
    for (int p = 0; p < (int)(sizeof Ns/sizeof Ns[0]); p++) {
        printf("=== LEAF N=%d ===\n", Ns[p]);
        for (int i = 0; i < (int)(sizeof Ks/sizeof Ks[0]); i++) {
            size_t K = Ks[i];
            double md = leaf_vs_naive(Ns[p], K);
            if (md == -2.0) { printf("  K=%-3zu rem%zu  create NULL\n", K, K%4); continue; }
            if (md == -3.0) { printf("  K=%-3zu rem%zu  not-LEAF (no leaf fn?)\n", K, K%4); fails++; continue; }
            double tol = 1e-9 * (double)Ns[p];
            int bad = (md > tol || md != md); if (bad) fails++;
            printf("  K=%-3zu rem%zu  err=%.2e %s\n", K, K%4, md, bad?"<FAIL>":"ok");
        }
    }
    printf("\n%s  (%d fail)\n", fails==0?"LEAF: correct at odd K (naive + roundtrip)":"FAILURES", fails);
    return fails ? 1 : 0;
}
