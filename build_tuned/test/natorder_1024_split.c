/* natorder_1024_split.c — decompose 1024/4's natural cost into FFT vs REORDER, to bound what
 * scatter-fusion could claw back. The natural path = 4·64·4 in-place FFT + pair-swap reorder.
 * Scatter-fusion can only remove the REORDER portion, so its best-case (ceiling) natural time is
 * (natural - reorder). This probe times the reorder pass ALONE (pair_pass and, for reference,
 * cycle_pass) on the exact 4·64·4 involution at K=4 — no FFT, no registry.
 *
 * Interpretation vs the measured natural~6400ns / scrambled(64·16)~4000ns:
 *   reorder LARGE  -> gap is the reorder -> scatter-fusion has real headroom (worth building).
 *   reorder SMALL  -> gap is the SLOWER 4·64·4 FFT vs 64·16 -> fusion can't help; the FFT is the cost.
 *
 * Build: python build.py --src test/natorder_1024_split.c --vfft --jit   (vfft.c not needed but harmless)
 * (only needs natorder_perm.h + natorder_exec.h — header-only)
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "natorder_perm.h"
#include "natorder_exec.h"

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static void split(const char *label, int N, int K, int *f, int nf){
    int *M=(int*)malloc((size_t)N*4);
    vfft_natorder_mk_perm(N,f,nf,M);
    int *cyc=vfft_natorder_mk_cycles(N,M);
    int *pairs=vfft_natorder_mk_pairs(N,M);        /* non-NULL iff involution (palindrome) */
    size_t tot=(size_t)N*K;
    double *re=(double*)_aligned_malloc(tot*8,64),*im=(double*)_aligned_malloc(tot*8,64);
    double *tmp=(double*)_aligned_malloc((size_t)2*K*8,64);
    for(size_t i=0;i<tot;i++){re[i]=(double)(i%101)*0.01;im[i]=(double)(i%97)*0.011;}
    int reps=200000000/((int)tot+1); if(reps<500)reps=500; if(reps>400000)reps=400000;
    for(int w=0;w<50;w++) vfft_natorder_cycle_pass(re,im,K,cyc,tmp);
    double bc=1e18;
    for(int t=0;t<9;t++){ double t0=now_ns(); for(int i=0;i<reps;i++) vfft_natorder_cycle_pass(re,im,K,cyc,tmp);
        double ns=(now_ns()-t0)/reps; if(ns<bc)bc=ns; }
    double bp=1e18;
    if(pairs){ for(int w=0;w<50;w++) vfft_natorder_pair_pass(re,im,K,pairs);
        for(int t=0;t<9;t++){ double t0=now_ns(); for(int i=0;i<reps;i++) vfft_natorder_pair_pass(re,im,K,pairs);
            double ns=(now_ns()-t0)/reps; if(ns<bp)bp=ns; } }
    int npairs=0; if(pairs){ for(int *p=pairs; *p!=-2; p+=2) npairs++; }
    printf("  %-14s N=%-5d K=%-3d  cycle_reorder=%.0f ns  pair_reorder=%.0f ns  (%d swaps)  pair/cycle=%.2f\n",
           label,N,K,bc,bp,npairs, bc>0?bp/bc:0.0);
    free(M);free(cyc);if(pairs)free(pairs);_aligned_free(re);_aligned_free(im);_aligned_free(tmp);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0); SetThreadAffinityMask(GetCurrentThread(),1<<2);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    printf("# reorder-alone cost for 1024/4's natural mode (4·64·4 involution) + neighbors\n");
    printf("# natural~6400ns, scrambled(64·16)~4000ns => fusion ceiling = natural - pair_reorder\n");
    { int f[3]={4,64,4}; split("1024/4 4·64·4", 1024, 4, f, 3); }   /* THE cell */
    { int f[2]={16,16};  split("256/4 16·16",   256,  4, f, 2); }   /* opp cell, for scale */
    { int f[3]={8,8,8};  split("512/4 8·8·8",   512,  4, f, 3); }
    { int f[3]={4,64,4}; split("1024/32 (K=32)",1024,32, f, 3); }   /* wider rows */
    return 0;
}
