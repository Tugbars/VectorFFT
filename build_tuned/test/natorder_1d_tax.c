/* natorder_1d_tax.c — 1D c2c NATURAL-order reorder tax: natural fwd vs the DEFAULT (scrambled) fwd,
 * SAME calibrated FFT plan (DEFAULT create calibrates+banks the base plan; NATURAL create is a lookup
 * + the reorder verdict), so only the reorder pass differs. Order-neutralized (interleaved rounds with
 * cooldown, min-of-N each), QPC, core-pinned, HIGH priority. No MKL — the tax is MKL-independent.
 *
 * Purpose: quantify what the opportunistic-PSWAP fix changed. 256/4 (palindromic 16·16) now
 * deterministically picks PSWAP (cheap pair-swap reorder) instead of the unpaced race sometimes
 * landing on PURE (expensive 16·16 cycle). Grep the wisdom dir after the run for each cell's marker
 * (`5 0.00` = opportunistic PSWAP, `4 …` = PURE).
 *
 * Build: python build.py --src test/natorder_1d_tax.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void){ LARGE_INTEGER c,f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static vfft_plan mk(int N, size_t K, int order){
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=1; c.order=order;
    return vfft_create(&c);
}

/* time one plan: reps forward executes, return per-exec ns (min over the caller's rounds). */
static double burst(vfft_plan p, double *re, double *im, int reps){
    double t0=now_ns();
    for(int i=0;i<reps;i++) vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    return (now_ns()-t0)/reps;
}

static void cell(int N, size_t K){
    size_t tot=(size_t)N*K;
    double *re=malloc(tot*8),*im=malloc(tot*8);
    for(size_t i=0;i<tot;i++){ re[i]=(double)((i*2654435761u)&1023)/1024.0-0.5;
                               im[i]=(double)((i*40503u)&1023)/1024.0-0.5; }
    vfft_plan pd=mk(N,K,VFFT_ORDER_DEFAULT);   /* scrambled: calibrates+banks the base plan */
    vfft_plan pn=mk(N,K,VFFT_ORDER_NATURAL);   /* natural: lookup + reorder verdict          */
    if(!pd||!pn){ printf("N=%-5d K=%-4zu  (NULL plan)\n",N,K); free(re);free(im); return; }
    int reps=(int)(4e6/(tot+1)); if(reps<20)reps=20; if(reps>4000)reps=4000;
    for(int w=0;w<8;w++){ burst(pd,re,im,reps); burst(pn,re,im,reps); }   /* warm-up both */
    double bd=1e18,bn=1e18;
    for(int r=0;r<7;r++){                       /* interleaved rounds, min-of-7, cooldown */
        double d=burst(pd,re,im,reps); if(d<bd)bd=d;
        Sleep(15);
        double n=burst(pn,re,im,reps); if(n<bn)bn=n;
        Sleep(15);
    }
    printf("N=%-5d K=%-4zu  scrambled=%8.0f ns  natural=%8.0f ns  tax=%.2fx\n",
           N,K,bd,bn,bn/bd);
    vfft_destroy(pd); vfft_destroy(pn); free(re);free(im);
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),1<<2);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    putenv("VFFT_WISDOM_DIR=natorder_1dtax_wis");
    printf("# 1D natural-order reorder tax (natural fwd / scrambled fwd, same base plan)\n");
    printf("# FIX beneficiaries (palindromic chain -> opportunistic PSWAP):\n");
    cell(256,4);      /* 16·16 palindrome — THE fixed flip cell */
    cell(512,4);      /* 8·8·8 palindrome */
    cell(1024,4);     /* check marker */
    printf("# UNAFFECTED by the fix (non-palindromic -> PURE, for contrast):\n");
    cell(256,32);
    cell(1024,32);
    cell(64,64);
    return 0;
}
