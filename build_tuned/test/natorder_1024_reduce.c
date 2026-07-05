/* natorder_1024_reduce.c — reduce the 1024/4 natural-order tax (1.56x, PSWAP-inj 4·64·4).
 *
 * 1024/4's calibrated scrambled chain is 64·16 DIT (2-stage, best ~3774ns) — NOT palindromic, so
 * opportunistic PSWAP can't fire. The race injects 4·64·4 (3-stage T1S) to get a palindrome for
 * pair-swap, paying BOTH a slower FFT and the reorder. This probe forces each candidate mode via a
 * pre-written v7 wisdom line and measures the resulting tax (natural/scrambled), so we can see which
 * realization is cheapest:
 *   0 PURE          : fast calibrated 64·16 plan + cycle-following reorder (no FFT penalty, dearer reorder)
 *   1 PSWAP 4·64·4  : the current race pick (3-stage injected + pair-swap)
 *   2 PSWAP 32·32   : 2-stage palindrome injection (close to 64·16 speed + cheap pair-swap) — best hope
 *   3 PSWAP 8·16·8  : another 3-stage palindrome
 * One variant per process (wisdom is cached after first create). argv[1] = variant index.
 *
 * Build: python build.py --src test/natorder_1024_reduce.c --vfft --jit
 * Run:   for v in 0 1 2 3; do ./test/natorder_1024_reduce.exe $v; done
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#define WISDIR "natorder_1024_wis"
#define N 1024
#define K 4

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

/* base calibrated line for 1024/4 (chain 64·16 DIT, variants 0 2) + the forced nat tail per variant.
 * ALL palindromic factorizations of 1024 over radixes {2,4,8,16,32,64} + PURE, to find the cheapest
 * natural realization. */
static const char *NAT_TAIL[7] = {
    "4 0.00",              /* PURE on calibrated 64·16                */
    "5 7383.33 3 4 64 4 2",/* PSWAP injected 4·64·4 (current pick)    */
    "5 0.00 2 32 32 2",    /* PSWAP injected 32·32  (2-stage)         */
    "5 0.00 3 8 16 8 2",   /* PSWAP injected 8·16·8 (3-stage)         */
    "5 0.00 3 16 4 16 2",  /* PSWAP injected 16·4·16 (3-stage)        */
    "5 0.00 4 4 8 8 4 2",  /* PSWAP injected 4·8·8·4 (4-stage)        */
    "5 0.00 5 4 4 4 4 4 2",/* PSWAP injected 4·4·4·4·4 (5-stage unif) */
};
static const char *NAME[7] = { "PURE(64·16)", "PSWAP 4·64·4", "PSWAP 32·32", "PSWAP 8·16·8",
                               "PSWAP 16·4·16", "PSWAP 4·8·8·4", "PSWAP 4·4·4·4·4" };

static void write_wisdom(int v){
    char path[700]; snprintf(path,sizeof path,"%s/spike_wisdom.txt",WISDIR);
    FILE *f=fopen(path,"w"); if(!f){ printf("cannot write %s\n",path); exit(2); }
    fprintf(f,"@version 7\n");
    fprintf(f,"1024 4 2 64 16 3774.02 0 0 0 0 0 2 0 %s\n", NAT_TAIL[v]);
    fclose(f);
}

static vfft_plan mk(int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}
static double burst(vfft_plan p, double *re, double *im, int reps){
    double t0=now_ns(); for(int i=0;i<reps;i++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    return (now_ns()-t0)/reps;
}
/* naive natural-order DFT of lane 0 for correctness */
static void naive(const double*x,const double*xi,double*Xr,double*Xi){
    for(int k=0;k<N;k++){ double ar=0,ai=0; for(int n=0;n<N;n++){ double a=-2.0*M_PI*k*n/N,c=cos(a),s=sin(a);
        double xr=x[(size_t)n*K],xii=xi[(size_t)n*K]; ar+=xr*c-xii*s; ai+=xr*s+xii*c; } Xr[k]=ar; Xi[k]=ai; } }

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1<<2);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    int v = argc>1 ? atoi(argv[1]) : 0; if(v<0||v>3) v=0;
    CreateDirectoryA(WISDIR,NULL);
    write_wisdom(v);                       /* pre-write BEFORE any create so it loads the forced verdict */
    putenv("VFFT_WISDOM_DIR=" WISDIR);

    size_t tot=(size_t)N*K;
    double *x=malloc(tot*8),*xi=malloc(tot*8),*re=malloc(tot*8),*im=malloc(tot*8);
    double *Xr=malloc(N*8),*Xi=malloc(N*8);
    for(size_t i=0;i<tot;i++){ x[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    naive(x,xi,Xr,Xi); double sc=0; for(int k=0;k<N;k++) if(fabs(Xr[k])>sc)sc=fabs(Xr[k]);

    vfft_plan pn=mk(VFFT_ORDER_NATURAL);   /* NATURAL first: reads forced verdict before DEFAULT re-saves */
    vfft_plan pd=mk(VFFT_ORDER_DEFAULT);
    if(!pn||!pd){ printf("v%d %-13s  plan NULL\n",v,NAME[v]); return 1; }

    memcpy(re,x,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(pn,VFFT_FORWARD,re,im,re,im);
    double eF=0; for(int k=0;k<N;k++){ double d1=fabs(re[(size_t)k*K]-Xr[k]),d2=fabs(im[(size_t)k*K]-Xi[k]);
        if(d1>eF)eF=d1; if(d2>eF)eF=d2; } eF/=(sc>0?sc:1);

    int reps=(int)(4e6/(tot+1)); if(reps<20)reps=20;
    for(int w=0;w<6;w++){ burst(pd,re,im,reps); burst(pn,re,im,reps); }
    double bd=1e18,bn=1e18;
    for(int r=0;r<6;r++){ double d=burst(pd,re,im,reps); if(d<bd)bd=d; Sleep(10);
                          double n=burst(pn,re,im,reps); if(n<bn)bn=n; Sleep(10); }
    printf("v%d %-13s  scrambled=%.0f  natural=%.0f  tax=%.2fx  fwd_err=%.1e %s\n",
           v,NAME[v],bd,bn,bn/bd,eF, eF<1e-9?"ok":"<FAIL>");
    vfft_destroy(pn); vfft_destroy(pd);
    return 0;
}
