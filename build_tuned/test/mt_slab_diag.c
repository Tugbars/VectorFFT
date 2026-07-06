/* mt_slab_diag.c — does the 16·8 (128/32) codelet honor the partial-lane count `me`?
 * SEQUENTIAL (no threads) 2-call split (me=16 + me=16 at +16) vs full batch (me=32). If they differ, the
 * codelet processes the full baked K and OVERRUNS the slab (structural) — that's the _c2c_mt bug root cause.
 * If they match, the bug is concurrency (a race). Tests both the JIT fn and the generic executor.
 * Build: python build.py --src test/mt_slab_diag.c --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "jit/jit_runtime.h"
#include "generator/generated/registry.h"

static double maxd(const double *a, const double *b, const double *c, const double *d, size_t n) {
    double m = 0; for (size_t i = 0; i < n; i++) { double e = fabs(a[i]-c[i]) + fabs(b[i]-d[i]); if (e > m) m = e; } return m;
}

static void probe(vfft_proto_registry_t *reg, int N, size_t K, const int *fac, const int *var, int nf, int dif) {
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, fac, var, nf, dif, reg);
    if (!p) { printf("  N=%d K=%zu chain: plan NULL\n", N, (size_t)K); return; }
    vfft_proto_exec_fn fn = vfft_proto_plan_jit_fwd(p);
    size_t tot = (size_t)N * K, h = K / 2;
    double *xr = malloc(tot*8), *xi = malloc(tot*8);
    double *ar = malloc(tot*8), *ai = malloc(tot*8), *br = malloc(tot*8), *bi = malloc(tot*8);
    for (size_t i = 0; i < tot; i++) { xr[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }

    /* JIT: full vs sequential split */
    double djit = -1;
    if (fn) {
        memcpy(ar, xr, tot*8); memcpy(ai, xi, tot*8); fn(p, ar, ai, K, p->K, 0);
        memcpy(br, xr, tot*8); memcpy(bi, xi, tot*8);
        fn(p, br, bi, h, p->K, 0); fn(p, br + h, bi + h, K - h, p->K, 0);
        djit = maxd(ar, ai, br, bi, tot);
    }
    /* GENERIC: full vs sequential split */
    memcpy(ar, xr, tot*8); memcpy(ai, xi, tot*8); vfft_proto_execute_fwd(p, ar, ai, K);
    memcpy(br, xr, tot*8); memcpy(bi, xi, tot*8);
    vfft_proto_execute_fwd(p, br, bi, h); vfft_proto_execute_fwd(p, br + h, bi + h, K - h);
    double dgen = maxd(ar, ai, br, bi, tot);

    char chain[64]; chain[0]=0; for (int s=0;s<nf;s++){ char t[16]; snprintf(t,sizeof t, s?"·%d":"%d", fac[s]); strcat(chain,t); }
    const char *verdict = (djit > 1e-12 || dgen > 1e-12) ? "*** SPLIT != FULL (codelet ignores `me` -> slab overrun, STRUCTURAL)" : "split==full ok";
    printf("  N=%-4d K=%-3zu %-10s  JIT split-vs-full=%.1e  GEN split-vs-full=%.1e   %s\n",
           N, (size_t)K, chain, djit, dgen, verdict);
    free(xr); free(xi); free(ar); free(ai); free(br); free(bi);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("# sequential half-batch split vs full batch (no threads) — does the codelet honor `me`?\n");
    int f_168[] = {16,8};    int v_168[] = {0,1};   probe(&reg, 128, 32, f_168, v_168, 2, 0);   /* the FAILING plan */
    int f_488[] = {4,8,8};   int v_488[] = {0,2,2}; probe(&reg, 256, 32, f_488, v_488, 3, 0);   /* passing */
    int f_4432[]= {4,4,32};  int v_4432[]={1,1,0};  probe(&reg, 512, 32, f_4432,v_4432,3, 1);   /* passing */
    int f_88[]  = {8,8};     int v_88[]  = {0,2};   probe(&reg, 64,  32, f_88,  v_88,  2, 0);
    int f_168b[]= {16,8};    int v_168b[]= {0,0};   probe(&reg, 128, 32, f_168b,v_168b,2, 0);   /* 16·8 both FLAT (variant probe) */
    return 0;
}
