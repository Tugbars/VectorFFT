/* natorder_2d_tax.c — 2D c2c reorder tax: NATURAL forward vs the scrambled (DEFAULT) forward, same
 * calibrated 2D plan. DEFAULT create calibrates+banks the 2D wisdom; NATURAL/SCRAMBLED are lookups, so
 * only the reorder differs. QPC, core-pinned, best-of-5 + warmup. No MKL. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static double bench(vfft_plan p, double *re, double *im, size_t tot){
    for(int w=0;w<10;w++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    double best=1e18; int reps=(int)(4e6/(tot+1)); if(reps<5)reps=5; if(reps>2000)reps=2000;
    for(int t=0;t<5;t++){ double t0=now_ns(); for(int i=0;i<reps;i++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
        double ns=(now_ns()-t0)/reps; if(ns<best)best=ns; }
    return best;
}
static double one(int N1,int N2,int order){
    size_t tot=(size_t)N1*N2; double *re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){re[i]=(double)(i%97)*0.01-0.5;im[i]=(double)(i%89)*0.011-0.4;}
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C;c.placement=VFFT_INPLACE;c.rigor=VFFT_MEASURE;c.dims=2;c.n[0]=N1;c.n[1]=N2;
    c.howmany=1;c.nthreads=1;c.order=order;
    vfft_plan p=vfft_create(&c); if(!p){free(re);free(im);return -1;}
    double ns=bench(p,re,im,tot); vfft_destroy(p); free(re);free(im); return ns;
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0); SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    system("rmdir /s /q natorder_2dtax_wis 2>nul"); system("mkdir natorder_2dtax_wis 2>nul");
    putenv("VFFT_WISDOM_DIR=natorder_2dtax_wis");
    /* 16x* / *x16 isolate the axes: N=16 inner plan is single-radix => that axis is FREE, so
     * 16x64 = dim2-only reorder and 64x16 = dim1-only reorder. */
    /* dim1-only (dim2 FREE via single-radix N2=16) at matched 1024-elem size vs the 16x64 dim2-only
     * (0.99x) already measured — and a bigger dim1-only (256x16) to see if dim1 tax is cache-driven. */
    int dims[][2]={{64,16},{256,16}};
    printf("# 2D reorder tax (NATURAL fwd / scrambled fwd), same calibrated plan\n");
    printf("# 64x16 + 256x16 = dim1-only (dim2 FREE); compare vs 16x64 dim2-only=0.99x\n");
    printf("%-9s %-12s %-12s %-8s\n","cell","scrambled_ns","natural_ns","tax");
    for(int i=0;i<2;i++){
        int N1=dims[i][0],N2=dims[i][1];
        double s=one(N1,N2,VFFT_ORDER_DEFAULT);   /* calibrates+banks, times scrambled */
        double n=one(N1,N2,VFFT_ORDER_NATURAL);   /* wisdom lookup + reorder, times natural */
        if(s>0&&n>0) printf("%dx%-6d %-12.0f %-12.0f %.2fx\n",N1,N2,s,n,n/s);
        else printf("%dx%d  (NULL)\n",N1,N2);
    }
    return 0;
}
