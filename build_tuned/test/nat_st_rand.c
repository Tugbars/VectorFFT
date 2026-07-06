/* nat_st_rand.c — is natural 128/32 wrong on RAND input SINGLE-THREADED too (i.e. NOT an MT bug at all,
 * but a wrong wisdom entry / reorder permutation masked by structured det input)?
 * For each cell: create NATURAL nthreads=1 (ST) and nthreads=8 (MT); test det AND rand input; also roundtrip.
 * Build: python build.py --src test/nat_st_rand.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static void naive(const double *re, const double *im, int N, size_t K, double *Xr, double *Xi) {
    for (int k = 0; k < N; k++) { double sr=0, si=0;
        for (int n = 0; n < N; n++) { double a=-2.0*3.14159265358979323846*k*n/N, c=cos(a), s=sin(a);
            sr += re[(size_t)n*K]*c - im[(size_t)n*K]*s; si += re[(size_t)n*K]*s + im[(size_t)n*K]*c; }
        Xr[k]=sr; Xi[k]=si; }
}
static double fwd_err(vfft_plan h, int N, size_t K, int det) {
    size_t tot=(size_t)N*K;
    double *xr=malloc(tot*8),*xi=malloc(tot*8),*Xr=malloc((size_t)N*8),*Xi=malloc((size_t)N*8);
    if (det) for(size_t i=0;i<tot;i++){ xr[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    else { srand(7+N+(int)K); for(size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; } }
    naive(xr,xi,N,K,Xr,Xi);
    double sc=0; for(int k=0;k<N;k++) if(fabs(Xr[k])>sc) sc=fabs(Xr[k]);
    double *re=malloc(tot*8),*im=malloc(tot*8); memcpy(re,xr,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(h,VFFT_FORWARD,re,im,re,im);
    double e=0; for(int k=0;k<N;k++){ double d=fabs(re[(size_t)k*K]-Xr[k])+fabs(im[(size_t)k*K]-Xi[k]); if(d>e)e=d; }
    free(xr);free(xi);free(Xr);free(Xi);free(re);free(im);
    return e/(sc>0?sc:1);
}
static double roundtrip_err(vfft_plan h, int N, size_t K) {
    size_t tot=(size_t)N*K;
    double *xr=malloc(tot*8),*xi=malloc(tot*8),*re=malloc(tot*8),*im=malloc(tot*8);
    srand(99+N+(int)K); for(size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; }
    memcpy(re,xr,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(h,VFFT_FORWARD,re,im,re,im);
    vfft_execute(h,VFFT_BACKWARD,re,im,re,im);
    double e=0,sc=0; for(size_t i=0;i<tot;i++){ if(fabs(xr[i])>sc)sc=fabs(xr[i]); double d=fabs(re[i]/N-xr[i])+fabs(im[i]/N-xi[i]); if(d>e)e=d; }
    free(xr);free(xi);free(re);free(im);
    return e/(sc>0?sc:1);
}
static void cell(int N, size_t K, int T) {
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=T; c.order=VFFT_ORDER_NATURAL;
    vfft_plan h = vfft_create(&c);
    if (!h) { printf("  N=%-4d K=%-3zu T=%d create NULL\n", N, K, T); return; }
    double det=fwd_err(h,N,K,1), rnd=fwd_err(h,N,K,0), rt=roundtrip_err(h,N,K);
    printf("  N=%-4d K=%-3zu T=%d  fwd_det=%.1e  fwd_rand=%.1e  roundtrip=%.1e  %s\n",
           N,K,T,det,rnd,rt,(det<1e-9&&rnd<1e-9&&rt<1e-9)?"ok":"*** FAIL");
    vfft_destroy(h);
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natmt_wis");
    printf("# NATURAL, ST (T=1) vs MT (T=8), det+rand fwd + roundtrip:\n");
    cell(128,32,1); cell(128,32,8);
    cell(512,32,1); cell(512,32,8);
    cell(256,32,1); cell(256,32,8);
    return 0;
}
