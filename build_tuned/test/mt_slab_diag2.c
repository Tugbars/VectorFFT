/* mt_slab_diag2.c — the 4·32 DIF (natural 128/32) chain fails MT on RAND but passes on DET. Replicate
 * _c2c_mt's EXACT 4-slab split (S=8: [0,8)[8,16)[16,24)[24,32), me=8 each) SEQUENTIALLY (no threads) on
 * BOTH det and rand input, vs the full batch. If sequential split != full => structural me/offset bug
 * (deterministic, catchable by a rand-input self-check). If sequential split == full but threaded differs
 * => genuine concurrency. Tests DIF and DIT variants.
 * Build: python build.py --src test/mt_slab_diag2.c --jit
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
static void probe(vfft_proto_registry_t *reg, int N, size_t K, const int *fac, const int *var, int nf, int dif, int det) {
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, fac, var, nf, dif, reg);
    if (!p) { printf("  N=%d K=%zu plan NULL\n", N, (size_t)K); return; }
    vfft_proto_exec_fn fn = vfft_proto_plan_jit_fwd(p);
    size_t tot = (size_t)N * K;
    double *xr = malloc(tot*8), *xi = malloc(tot*8);
    double *ar = malloc(tot*8), *ai = malloc(tot*8), *br = malloc(tot*8), *bi = malloc(tot*8);
    if (det) for (size_t i=0;i<tot;i++){ xr[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    else { srand(7+N+(int)K); for (size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; } }

    /* full batch */
    memcpy(ar, xr, tot*8); memcpy(ai, xi, tot*8);
    if (fn) fn(p, ar, ai, K, p->K, 0); else vfft_proto_execute_fwd(p, ar, ai, K);
    /* _c2c_mt EXACT slab split: S = CEIL(K/8) rounded up to 8 */
    int T = 8; size_t S = (((K + (size_t)T - 1)/(size_t)T) + 7) & ~(size_t)7;
    memcpy(br, xr, tot*8); memcpy(bi, xi, tot*8);
    for (size_t k0 = 0; k0 < K; k0 += S) {
        size_t me = (k0 + S > K) ? K - k0 : S;
        if (fn) fn(p, br + k0, bi + k0, me, p->K, 0); else vfft_proto_execute_fwd(p, br + k0, bi + k0, me);
    }
    double d = maxd(ar, ai, br, bi, tot);
    char chain[64]; chain[0]=0; for (int s=0;s<nf;s++){ char t[16]; snprintf(t,sizeof t, s?"·%d":"%d", fac[s]); strcat(chain,t); }
    printf("  N=%-4d K=%-3zu %-8s dif=%d %s  S=%zu  split-vs-full=%.1e   %s\n",
           N,(size_t)K,chain,dif,det?"DET":"RND",S,d, d>1e-12?"*** SPLIT != FULL (structural me/offset bug)":"split==full ok");
    free(xr);free(xi);free(ar);free(ai);free(br);free(bi);
}
int main(void){
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("# _c2c_mt exact 4-slab split (SEQUENTIAL) vs full, DIF/DIT, det+rand:\n");
    int f_432[] = {4,32};  int v_432[] = {1,0};             /* natural 128/32: LOG3,FLAT, DIF */
    probe(&reg,128,32,f_432,v_432,2,1,1); probe(&reg,128,32,f_432,v_432,2,1,0);
    int f_4432[]= {4,4,32}; int v_4432[]={1,1,0};           /* natural 512/32: DIF */
    probe(&reg,512,32,f_4432,v_4432,3,1,1); probe(&reg,512,32,f_4432,v_4432,3,1,0);
    /* same chains as DIT (dif=0) for contrast */
    probe(&reg,128,32,f_432,v_432,2,0,0);
    return 0;
}
