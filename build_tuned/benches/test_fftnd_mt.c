/* test_fftnd_mt.c — MT correctness for fftnd: fwd output must be
 * bit-identical across T for EVERY parallel mode (per-element op order is
 * invariant under all work partitions by construction).
 *
 * Modes exercised:
 *   block-parallel fused        NB >= T   (32x32x32 s=1, T<=8)
 *   STARVED fused (seq blocks,  NB <  T   (4x8x8x8 s=1 -> NB=4;
 *     parallel within, per-pass            3x5x7x8 s=1 -> NB=3, primes;
 *     pool joins)                          2x6x16  s=1 -> NB=2)
 *   single-block fused          NB == 1 case avoided (degenerate axis)
 *   window clamp in tiled       Rb=6 < B=8 (2x6x16: tiles clipped inside
 *                               each block window)
 *   hierarchical axis MT        outer<T with lane splits (unfused axes of
 *                               4x8x8x8 at s=3)
 *   T > pool                    T=8 requested on the pool as configured
 *
 * Plus roundtrip at the highest T per cell. Build like the other tests.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftnd.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static int g_fail = 0;

static stride_plan_t *mk(int rank, const int *N, int split,
                         const vfft_proto_registry_t *reg) {
    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = rank;
    for (int m = 0; m < rank; m++) tmp.N[m] = N[m];
    _fftnd_fill_ok(&tmp);
    size_t B = _fftnd_choose_tile(N[rank-1], tmp.O[rank-1]);
    stride_plan_t *pl[FFTND_MAX_RANK] = {0};
    for (int m = 0; m < rank; m++) {
        size_t Kp = (m == rank-1) ? B : tmp.K[m];
        pl[m] = vfft_proto_auto_plan_dispatch(N[m], Kp, reg, NULL);
        if (!pl[m]) return NULL;
    }
    return stride_plan_nd_from(rank, N, B, split, NULL, pl);
}

static void cell(int rank, const int *N, int split,
                 const vfft_proto_registry_t *reg) {
    size_t n = 1; for (int m=0;m<rank;m++) n *= (size_t)N[m];
    double *xr=AALLOC(n*8),*xi=AALLOC(n*8);
    double *rr=AALLOC(n*8),*ri=AALLOC(n*8);
    double *cr=AALLOC(n*8),*ci=AALLOC(n*8);
    srand(31 + rank + N[0]);
    for (size_t i=0;i<n;i++){ xr[i]=2.0*rand()/RAND_MAX-1;
                              xi[i]=2.0*rand()/RAND_MAX-1; }

    /* T=1 reference (plan created at max T so scratch covers all runs) */
    stride_set_num_threads(8);
    stride_plan_t *p = mk(rank, N, split, reg);
    if (!p) { printf("  plan FAIL\n"); g_fail++; return; }
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)p->override_data;
    size_t NB = d->O[d->split];

    stride_set_num_threads(1);
    memcpy(rr,xr,n*8); memcpy(ri,xi,n*8);
    stride_execute_fwd(p, rr, ri);

    int Ts[3] = {2, 4, 8};
    int all_eq = 1;
    for (int ti = 0; ti < 3; ti++) {
        stride_set_num_threads(Ts[ti]);
        memcpy(cr,xr,n*8); memcpy(ci,xi,n*8);
        stride_execute_fwd(p, cr, ci);
        int eq = !memcmp(rr,cr,n*8) && !memcmp(ri,ci,n*8);
        if (!eq) all_eq = 0;
    }

    /* roundtrip at T=8 */
    stride_set_num_threads(8);
    memcpy(cr,xr,n*8); memcpy(ci,xi,n*8);
    stride_execute_fwd(p, cr, ci);
    stride_execute_bwd(p, cr, ci);
    double sc=(double)n, rt=0;
    for (size_t i=0;i<n;i++){
        double rel=(fabs(cr[i]-sc*xr[i])+fabs(ci[i]-sc*xi[i]))
                  /(fabs(sc*xr[i])+fabs(sc*xi[i])+1e-300);
        if (rel>rt) rt=rel;
    }
    stride_set_num_threads(1);

    const char *mode = NB >= 8 ? "block-par " : (NB > 1 ? "STARVED   " : "single-blk");
    int ok = all_eq && rt < 1e-11;
    if (!ok) g_fail++;
    printf("  r%d ", rank);
    for (int m=0;m<rank;m++) printf("%d%s", N[m], m+1<rank?"x":"");
    printf("  s=%d NB=%zu [%s]  T={1,2,4,8} bit=%s  rt(T8)=%.1e  %s\n",
           d->split, NB, mode, all_eq?"EXACT":"**MISMATCH**", rt,
           ok?"OK":"**FAIL**");

    AFREE(xr);AFREE(xi);AFREE(rr);AFREE(ri);AFREE(cr);AFREE(ci);
    stride_plan_destroy(p);
}

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("fftnd MT matrix (bit-identity across T per mode)\n");

    int a[3]={32,32,32};   cell(3,a,1,&reg);   /* block-parallel      */
    int b[4]={4,8,8,8};    cell(4,b,1,&reg);   /* starved NB=4        */
    int c[4]={3,5,7,8};    cell(4,c,1,&reg);   /* starved NB=3 primes */
    int e[3]={2,6,16};     cell(3,e,1,&reg);   /* starved + Rb<B clamp*/
    int f[4]={4,8,8,8};    cell(4,f,3,&reg);   /* unfused hier axis MT*/
    int g[4]={16,16,16,16};cell(4,g,2,&reg);   /* mixed               */

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
