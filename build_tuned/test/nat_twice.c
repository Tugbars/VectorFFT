/* nat_twice.c — is the natural-MT-first failure a COLD-pool first-dispatch issue or a per-execute race?
 * Create NATURAL 128/32 nthreads=8 as the FIRST plan, then execute the SAME plan TWICE on the SAME rand
 * input, comparing each run to naive. Also a 3rd/4th run. If run1 wrong but run2+ right => first-dispatch
 * (cold) — a create-time warm-up fixes it. If ALL runs wrong (and vary) => a genuine per-execute race.
 * Build: python build.py --src test/nat_twice.c --vfft --jit
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
static void run(int N, size_t K) {
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=8; c.order=VFFT_ORDER_NATURAL;
    vfft_plan h = vfft_create(&c);
    if (!h) { printf("  N=%d K=%zu create NULL\n", N, K); return; }
    size_t tot=(size_t)N*K;
    double *xr=malloc(tot*8),*xi=malloc(tot*8),*Xr=malloc((size_t)N*8),*Xi=malloc((size_t)N*8);
    srand(7+N+(int)K); for(size_t i=0;i<tot;i++){ xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5; }
    naive(xr,xi,N,K,Xr,Xi);
    double sc=0; for(int k=0;k<N;k++) if(fabs(Xr[k])>sc) sc=fabs(Xr[k]);
    double *re=malloc(tot*8),*im=malloc(tot*8);
    printf("  N=%-4d K=%-3zu chain(nat):", N, K);
    for (int rep=0; rep<4; rep++) {
        memcpy(re,xr,tot*8); memcpy(im,xi,tot*8);       /* SAME rand input each run */
        vfft_execute(h,VFFT_FORWARD,re,im,re,im);
        double e=0; for(int k=0;k<N;k++){ double d=fabs(re[(size_t)k*K]-Xr[k])+fabs(im[(size_t)k*K]-Xi[k]); if(d>e)e=d; }
        printf("  run%d=%.1e", rep, e/(sc>0?sc:1));
    }
    printf("\n");
    free(xr);free(xi);free(Xr);free(Xi);free(re);free(im); vfft_destroy(h);
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1);   /* core 0 (MT caller) */
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natmt_wis");
    printf("# NATURAL T=8 FIRST create, SAME plan executed 4x on SAME rand input (vs naive):\n");
    run(128,32);
    run(512,32);
    run(256,32);
    return 0;
}
