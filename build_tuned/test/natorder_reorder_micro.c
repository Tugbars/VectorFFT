/* natorder_reorder_micro.c — isolate the dim1 reorder MECHANISM: cycle_pass vs pair_pass on the SAME
 * involution permutation (a palindromic a*a digit reversal), same K-wide rows. No FFT, no calibration,
 * so it directly answers "does PSWAP (independent swaps) beat cycle-following for the whole-row reorder"
 * — the question the confounded macro tax (calibration picks different chains each run) can't. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include "natorder_perm.h"
#include "natorder_exec.h"
static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static void one(int N, int K){
    int a=(int)(sqrt((double)N)+0.5);
    if(a*a!=N){ printf("  N=%d not a perfect square, skip\n",N); return; }
    int f[2]={a,a}, nf=2;
    int *M=(int*)malloc((size_t)N*4);
    vfft_natorder_mk_perm(N,f,nf,M);                 /* a*a digit reversal = swap the two base-a digits */
    int *cyc=vfft_natorder_mk_cycles(N,M);
    int *pairs=vfft_natorder_mk_pairs(N,M);          /* non-NULL iff M is an involution (it is, for a*a) */
    size_t tot=(size_t)N*K;
    double *re=(double*)_aligned_malloc(tot*8,64),*im=(double*)_aligned_malloc(tot*8,64);
    double *tmp=(double*)_aligned_malloc((size_t)2*K*8,64);
    for(size_t i=0;i<tot;i++){re[i]=(double)(i%101)*0.01;im[i]=(double)(i%97)*0.011;}
    int reps=200000000/((int)tot+1); if(reps<200)reps=200; if(reps>200000)reps=200000;
    /* cycle_pass */
    for(int w=0;w<50;w++) vfft_natorder_cycle_pass(re,im,K,cyc,tmp);
    double bc=1e18;
    for(int t=0;t<7;t++){ double t0=now_ns(); for(int i=0;i<reps;i++) vfft_natorder_cycle_pass(re,im,K,cyc,tmp);
        double ns=(now_ns()-t0)/reps; if(ns<bc)bc=ns; }
    /* pair_pass */
    double bp=1e18;
    if(pairs){
        for(int w=0;w<50;w++) vfft_natorder_pair_pass(re,im,K,pairs);
        for(int t=0;t<7;t++){ double t0=now_ns(); for(int i=0;i<reps;i++) vfft_natorder_pair_pass(re,im,K,pairs);
            double ns=(now_ns()-t0)/reps; if(ns<bp)bp=ns; }
    }
    printf("  N=%-4d K=%-3d (chain %dx%d, %d rows x %dB)  cycle=%.0f ns  pair=%.0f ns  speedup=%.2fx\n",
           N,K,a,a,N,K*8, bc, bp, bp>0?bc/bp:0.0);
    free(M);free(cyc);free(pairs);_aligned_free(re);_aligned_free(im);_aligned_free(tmp);
}
int main(void){
    setvbuf(stdout,NULL,_IONBF,0); SetThreadAffinityMask(GetCurrentThread(),1);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    printf("# dim1 whole-row reorder: cycle_pass vs pair_pass (PSWAP), same involution perm\n");
    one(64,16);    /* like 64x16 dim1 */
    one(64,64);    /* like 64x64 dim1 */
    one(256,16);   /* like 256x16 dim1 */
    one(256,64);
    one(1024,16);
    return 0;
}
