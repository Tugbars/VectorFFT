/* test_oop_mt.c — OOP c2c MULTITHREADING via the vfft.h public API (now wired: _oop_mt K-split).
 * For each (N,K): build an MT plan (nthreads=4) and an ST plan (nthreads=1); check MT fwd output
 * == ST fwd output, and MT fwd->bwd roundtrip == N*x. N selects the kind: N<=128 LEAF (K-splits),
 * larger N BAILEY2 (K-split unsafe -> ST fallback) or MODEB (K-splits). Both odd + even K; K<8 -> ST.
 * Build: python build.py --src test/test_oop_mt.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;
#define T 4

static vfft_plan mk(int N, int K, int nth)
{
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = N; c.howmany = (size_t)K; c.nthreads = nth;
    return vfft_create(&c);
}

static void cell(int N, int K)
{
    size_t n = (size_t)N * K;
    double *ir=malloc(n*8),*ii=malloc(n*8),*xr=malloc(n*8),*xi=malloc(n*8);
    double *mo_r=malloc(n*8),*mo_i=malloc(n*8),*so_r=malloc(n*8),*so_i=malloc(n*8),*bk_r=malloc(n*8),*bk_i=malloc(n*8);
    srand(31+N+K);
    for(size_t i=0;i<n;i++){double a=(double)rand()/RAND_MAX-.5,b=(double)rand()/RAND_MAX-.5; ir[i]=xr[i]=a; ii[i]=xi[i]=b;}
    /* Seed the wisdom deterministically so the MT and ST plans below are the SAME plan
     * (kind + factorization) — else two independent MEASURE calibrations can pick different
     * kinds/orders, making the MT-vs-ST elementwise compare meaningless. */
    { vfft_plan seed = mk(N,K,1); if(seed) vfft_destroy(seed); }
    vfft_plan pm=mk(N,K,T), ps=mk(N,K,1);
    if(!pm||!ps){printf("  N=%-4d K=%-3d  create NULL\n",N,K);fails++;goto d;}
    vfft_execute(ps, VFFT_FORWARD, ir, ii, so_r, so_i);        /* ST fwd */
    vfft_execute(pm, VFFT_FORWARD, ir, ii, mo_r, mo_i);        /* MT fwd */
    double mt=0; for(size_t i=0;i<n;i++){double dr=fabs(mo_r[i]-so_r[i]),di=fabs(mo_i[i]-so_i[i]);if(dr>mt)mt=dr;if(di>mt)mt=di;}
    vfft_execute(pm, VFFT_BACKWARD, mo_r, mo_i, bk_r, bk_i);   /* MT bwd (roundtrip) */
    double rt=0,inv=1.0/N; for(size_t i=0;i<n;i++){double dr=fabs(bk_r[i]*inv-xr[i]),di=fabs(bk_i[i]*inv-xi[i]);if(dr>rt)rt=dr;if(di>rt)rt=di;}
    int thr = (K>=8);   /* K<8 -> ST fallback */
    int bad=(mt>1e-12)||(rt>1e-9); if(bad)fails++;
    printf("  N=%-4d K=%-3d rem%d %-8s MT-vs-ST=%8.1e roundtrip=%8.1e %s\n",
           N,K,K&3, thr?"(K>=8)":"(K<8 ST)", mt, rt, bad?"<FAIL>":"ok");
    vfft_destroy(pm); vfft_destroy(ps);
d: free(ir);free(ii);free(xr);free(xi);free(mo_r);free(mo_i);free(so_r);free(so_i);free(bk_r);free(bk_i);
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    putenv("VFFT_WISDOM_DIR=oop_mt_test"); system("mkdir oop_mt_test 2>nul");
    printf("# OOP c2c MT via public vfft.h (nthreads=%d): MT fwd==ST fwd + MT roundtrip. N picks kind.\n",T);
    int Ns[]={64,128,256,512,1024};
    int Ks[]={7,8,16,23,31};
    for(int ni=0;ni<5;ni++){ printf("== N=%d (%s) ==\n", Ns[ni], Ns[ni]<=128?"LEAF":"BAILEY2/MODEB");
        for(int ki=0;ki<5;ki++) cell(Ns[ni],Ks[ki]); }
    printf(fails?"\nRESULT: %d FAILURE(S)\n":"\nRESULT: OOP MT correct (MT==ST + roundtrip) across kinds + odd K\n",fails);
    return fails?1:0;
}
