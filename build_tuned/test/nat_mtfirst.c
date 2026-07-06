/* nat_mtfirst.c — minimal repro of natorder_vs_mkl's flow: create NATURAL nthreads=8 as the FIRST plan
 * (no prior DEFAULT create), execute once, compare vs naive DFT. Tests deterministic AND rand() input to
 * separate an input-dependent overrun from an MT-first flow race. 128/32 (natural chain 4·32) is the cell
 * natorder reports wrong but natmt (DEFAULT-first) reports correct.
 * Build: python build.py --src test/nat_mtfirst.c --vfft --jit
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
static double chk(int N, size_t K, int det) {
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=8; c.order=VFFT_ORDER_NATURAL;
    vfft_plan h = vfft_create(&c);
    if (!h) { printf("  create NULL\n"); return -1; }
    size_t tot=(size_t)N*K;
    double *xr=malloc(tot*8),*xi=malloc(tot*8),*Xr=malloc((size_t)N*8),*Xi=malloc((size_t)N*8);
    if (det) { for(size_t i=0;i<tot;i++){ xr[i]=(double)((i*2654435761u)&1023)/1024.0-0.5; xi[i]=(double)((i*40503u)&1023)/1024.0-0.5; } }
    else { srand(7+N+(int)K); for(size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; } }
    naive(xr,xi,N,K,Xr,Xi);
    double *re=malloc(tot*8),*im=malloc(tot*8); memcpy(re,xr,tot*8); memcpy(im,xi,tot*8);
    vfft_execute(h,VFFT_FORWARD,re,im,re,im);
    double sc=0; for(int k=0;k<N;k++) if(fabs(Xr[k])>sc) sc=fabs(Xr[k]);
    double e=0; for(int k=0;k<N;k++){ double d=fabs(re[(size_t)k*K]-Xr[k])+fabs(im[(size_t)k*K]-Xi[k]); if(d>e)e=d; }
    e/=(sc>0?sc:1);
    free(xr);free(xi);free(Xr);free(Xi);free(re);free(im); vfft_destroy(h);
    return e;
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1);   /* core 0 (MT caller) */
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natmt_wis");
    printf("# NATURAL T=8 as FIRST create (natorder flow), vs naive:\n");
    int cells[][2]={{128,32},{512,32},{256,32}};
    for(int i=0;i<3;i++){ int N=cells[i][0]; size_t K=cells[i][1];
        double det=chk(N,K,1), rnd=chk(N,K,0);
        printf("  N=%-5d K=%-3zu  det=%.1e  rand=%.1e  %s\n", N,K,det,rnd,
               (det<1e-9&&rnd<1e-9)?"ok":"*** FAIL"); }
    return 0;
}
