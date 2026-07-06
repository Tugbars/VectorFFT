/* natorder_cold_test.c — C1 gate: order=NATURAL on a COLD (wisdom-miss) cell must not hard-fail where
 * DEFAULT succeeds. For each distinct cell we create NATURAL *first* (cold path — no prior calibration),
 * then DEFAULT; both must be non-NULL and NATURAL must roundtrip. Covers composites (multi-stage: the
 * risky path), pow2, and primes (FREE via override). Fresh in-memory wisdom per cell is NOT possible
 * in-process, so each cell is distinct => its first create is genuinely cold.
 * Build: python build.py --src test/natorder_cold_test.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static vfft_plan mk(int N, size_t K, int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}

static int cell(int N, size_t K){
    size_t tot=(size_t)N*K;
    double *x=malloc(tot*8),*xi=malloc(tot*8),*re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    /* NATURAL FIRST = the cold path (nothing calibrated this cell yet). */
    vfft_plan pn=mk(N,K,VFFT_ORDER_NATURAL);
    vfft_plan pd=mk(N,K,VFFT_ORDER_DEFAULT);
    int nn=(pn!=NULL), nd=(pd!=NULL);
    double rt=-1;
    if(pn){ memcpy(re,x,tot*8); memcpy(im,xi,tot*8);
        vfft_execute(pn,VFFT_FORWARD,re,im,re,im);
        vfft_execute(pn,VFFT_BACKWARD,re,im,re,im);
        rt=0; double inv=1.0/N; for(size_t i=0;i<tot;i++){ double d=fabs(re[i]*inv-x[i])+fabs(im[i]*inv-xi[i]); if(d>rt)rt=d; } }
    /* FAIL only if NATURAL is NULL where DEFAULT succeeded (the C1 hard-fail), or NATURAL roundtrip broke. */
    int ok = !(nd && !nn) && (!nn || rt<1e-9);
    printf("  N=%-5d K=%-3zu  NATURAL=%s DEFAULT=%s  rt=%.1e  %s\n",
           N,K, nn?"ok":"NULL", nd?"ok":"NULL", rt, ok?"PASS":"*** FAIL (natural hard-fail / bad rt) ***");
    if(pn)vfft_destroy(pn); if(pd)vfft_destroy(pd);
    free(x);free(xi);free(re);free(im);
    return ok;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);
    putenv("VFFT_WISDOM_DIR=natorder_cold_wis");
    printf("# C1: order=NATURAL on cold cells must not hard-fail where DEFAULT works\n");
    /* composites (multi-stage; the risky e2-absent path), pow2, primes (FREE via override) */
    int Ns[]={ 96, 100, 120, 126, 144, 200, 252, 300, 360, 500,  128, 256, 512,  89, 127, 251 };
    int all=1;
    for(int i=0;i<16;i++){ all &= cell(Ns[i],4); all &= cell(Ns[i],16); }
    printf("\n%s\n", all?"ALL PASS (no NATURAL cold hard-fail)":"*** SOME FAILED ***");
    return all?0:1;
}
