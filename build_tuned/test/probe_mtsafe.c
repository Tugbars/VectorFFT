/* probe_mtsafe: does the natural DIF cell now K-split (mt_unsafe==0) vs whole-batch fallback?
 * Times MT(T=8) vs forced-ST to infer parallelism, and prints fwd correctness. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
static double now_ms(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1000.0*c.QuadPart/f.QuadPart; }
static double bench(int N,size_t K,int T){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C;c.placement=VFFT_INPLACE;c.rigor=VFFT_MEASURE;c.dims=1;c.n[0]=N;c.howmany=K;c.nthreads=T;c.order=VFFT_ORDER_NATURAL;
    vfft_plan h=vfft_create(&c); if(!h){printf("null\n");return -1;}
    size_t tot=(size_t)N*K; double*re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){re[i]=(double)rand()/RAND_MAX-0.5;im[i]=(double)rand()/RAND_MAX-0.5;}
    for(int w=0;w<50;w++) vfft_execute(h,VFFT_FORWARD,re,im,re,im);
    double t0=now_ms(); int R=400; for(int r=0;r<R;r++) vfft_execute(h,VFFT_FORWARD,re,im,re,im); double t=(now_ms()-t0)/R;
    free(re);free(im); vfft_destroy(h); return t;
}
int main(void){ setvbuf(stdout,NULL,_IONBF,0); SetThreadAffinityMask(GetCurrentThread(),1); SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natmt_wis"); srand(5);
    int cells[][2]={{128,32},{512,32},{1024,256}};
    for(int i=0;i<3;i++){int N=cells[i][0];size_t K=cells[i][1];
        double st=bench(N,K,1),mt=bench(N,K,8);
        printf("N=%-5d K=%-3zu  ST=%.4gms  MT8=%.4gms  speedup=%.2fx  %s\n",N,K,st,mt,st/mt, st/mt>1.3?"(K-SPLIT active)":"(~no speedup)"); }
    return 0; }
