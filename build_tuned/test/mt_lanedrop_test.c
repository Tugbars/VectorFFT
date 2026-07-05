/* mt_lanedrop_test.c — repro/gate for the K-split lane-drop bug in _c2c_mt (vfft.c).
 *
 * The MT slab was S = ((K/T)+7)&~7 (FLOOR then round-to-8). When floor(K/T) is already a multiple of 8
 * and K%T!=0, T slabs cover only T*floor(K/T) = K-(K%T) < K lanes — the last K%T lanes are never
 * executed => their in-place data stays = INPUT (stale), not the FFT result. Triggers at T=8: K=65..71,
 * 129..135. Fix = CEIL: S=(((K+T-1)/T)+7)&~7.
 *
 * Gate: c2c in-place, order=DEFAULT, MT (nthreads=8) vs ST (nthreads=1), same input. Must be bit-
 * identical (same K-split math). A dropped lane shows up as a huge diff (input vs FFT). Caller pins
 * core 0 (MT contract). Build: python build.py --src test/mt_lanedrop_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static vfft_plan mk(int N, size_t K, int nthreads){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=nthreads; c.order=VFFT_ORDER_DEFAULT;
    return vfft_create(&c);
}

static int cell(int N, size_t K){
    size_t tot=(size_t)N*K;
    double *x=malloc(tot*8),*xi=malloc(tot*8);
    double *rs=malloc(tot*8),*is=malloc(tot*8),*rm=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    vfft_plan ps=mk(N,K,1);           /* ST reference (sequence ST fully before MT) */
    if(!ps){ printf("  N=%d K=%-4zu ST plan NULL\n",N,K); return 0; }
    memcpy(rs,x,tot*8); memcpy(is,xi,tot*8);
    vfft_execute(ps,VFFT_FORWARD,rs,is,rs,is);
    vfft_destroy(ps);
    vfft_plan pm=mk(N,K,8);           /* MT (nthreads=8) */
    if(!pm){ printf("  N=%d K=%-4zu MT plan NULL\n",N,K); return 0; }
    memcpy(rm,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(pm,VFFT_FORWARD,rm,im,rm,im);
    vfft_destroy(pm);
    /* compare + locate the first differing lane (a dropped lane = the whole lane stale) */
    double maxd=0; int bad_lane=-1;
    for(size_t i=0;i<tot;i++){ double d=fabs(rm[i]-rs[i])+fabs(im[i]-is[i]);
        if(d>maxd){ maxd=d; if(d>1e-9 && bad_lane<0) bad_lane=(int)(i%K); } }
    int ok=(maxd<1e-9);
    printf("  N=%-3d K=%-4zu  MT-vs-ST maxdiff=%.2e  %s%s\n",N,K,maxd,ok?"PASS":"*** FAIL",
           ok?"":(bad_lane>=0?"":""));
    if(!ok) printf("      first stale lane ~%d (of %zu)\n",bad_lane,K);
    free(x);free(xi);free(rs);free(is);free(rm);free(im);
    return ok;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);   /* core 0 — MT worker-pin contract */
    printf("# _c2c_mt K-split lane-drop gate (MT nthreads=8 vs ST). T=8 triggers: K=65..71,129..135\n");
    int all=1;
    size_t Ks[]={64, 65,66,67,71, 72, 128,129,135, 256};  /* 64/72/128/256=controls; rest=triggers */
    for(int i=0;i<10;i++) all &= cell(16,Ks[i]);
    printf("\n%s\n", all?"ALL PASS (lane-drop fixed)":"*** LANE DROP PRESENT ***");
    return all?0:1;
}
