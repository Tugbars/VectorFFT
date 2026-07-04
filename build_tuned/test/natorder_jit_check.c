/* natorder_jit_check.c — does the PSWAP verdict SURVIVE the JIT/Tier-1 path?
 * T11 flagged: 128/64's generic baseline runs ~60% above the wisdom-recorded Tier-1 number, and
 * the natorder race times candidates on the GENERIC executor (per the sweep-measures-generic
 * decision). Risk: a verdict won generic-vs-generic flips when the default path runs JIT/baked
 * while the injected chain JITs differently. This probe times the PUBLIC API with a --jit build:
 *   default-order plan vs natural-order plan (stored verdicts: 128/64=PSWAP 4·8·4, 64/64=PURE),
 * warm-up + 5 paced rounds averaged (T8 methodology). natural/default ratio ~= the deployed tax.
 * Build: python build.py --src test/natorder_jit_check.c --vfft --jit */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static double qpc_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static void refill(double *re,double *im,size_t n){ for(size_t i=0;i<n;i++){
    re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; im[i]=(double)((i*40503u)&1023)/1024.0-0.5; } }
static void rescale(double *re,double *im,size_t n){ double mx=0;
    for(size_t i=0;i<n;i+=13){ double a=fabs(re[i]); if(a>mx)mx=a; }
    if(mx>1e80||mx<1e-80){ double s=mx>0?1.0/mx:1.0; for(size_t i=0;i<n;i++){re[i]*=s;im[i]*=s;} } }

static vfft_plan mk(int N,size_t K,int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1; c.order=order;
    return vfft_create(&c); }

static double tfwd(vfft_plan p,double *re,double *im,size_t n){
    refill(re,im,n);
    for(int r=0;r<8;r++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);   /* warm-up */
    double sum=0; int inner=1024;
    for(int o=0;o<5;o++){ Sleep(120); refill(re,im,n); double acc=0; int done=0;
        while(done<inner){ double t0=qpc_ns();
            for(int r=0;r<8;r++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
            acc+=qpc_ns()-t0; done+=8; rescale(re,im,n); }
        sum+=acc/done; }
    return sum/5.0; }

static void cell(int N,size_t K){
    size_t n=(size_t)N*K;
    double *re=malloc(n*8),*im=malloc(n*8);
    vfft_plan pd=mk(N,K,VFFT_ORDER_DEFAULT), pn=mk(N,K,VFFT_ORDER_NATURAL);
    if(!pd||!pn){ printf("N=%d K=%zu plan NULL\n",N,K); return; }
    double td=tfwd(pd,re,im,n), tn=tfwd(pn,re,im,n);
    printf("N=%-4d K=%-3zu  default(JIT) %8.0f ns   natural %8.0f ns   ratio %.3f\n",
           N,K,td,tn,tn/td);
    vfft_destroy(pd); vfft_destroy(pn); free(re); free(im); }

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_wis_p0");
    printf("# JIT recheck: stored natural verdicts vs the JIT/Tier-1 default path (public API)\n");
    cell(128,64);   /* stored PSWAP 4·8·4 — the risky cell */
    cell(64,64);    /* stored PURE — context */
    cell(1024,32);  /* PURE band cell */
    return 0; }
