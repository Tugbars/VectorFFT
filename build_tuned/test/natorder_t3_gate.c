/* natorder_t3_gate.c — T3: ALL 22 n1_oop leaves compiled WITHOUT restrict (Option-A simulation),
 * swept for alias-safety. AUTO-GENERATED include block (same generated sources, symbols renamed
 * nr_radixN, __restrict__ neutralized). Gates per (N,K):
 *   A. aliased fwd (dst==src) BIT-EXACT vs same-function separate-dst
 *   B. natural-order math sane vs naive DFT (1e-8 rel, lane 0)
 *   C. swap-identity roundtrip ALIASED: fwd aliased, bwd=fn(im,re swapped) aliased, /N == input (1e-9)
 * K sweep adds {1,2,3} (scalar/SSE2 tail lanes G1 never tested). This file IS the per-build gate.
 * Build: python build.py --src test/natorder_t3_gate.c --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <stddef.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define __restrict__
#define __restrict
#define radix2_n1_oop_fwd_avx2_UG_UG nr_radix2
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix2_n1_oop_avx2.c"
#undef radix2_n1_oop_fwd_avx2_UG_UG
#define radix3_n1_oop_fwd_avx2_UG_UG nr_radix3
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix3_n1_oop_avx2.c"
#undef radix3_n1_oop_fwd_avx2_UG_UG
#define radix4_n1_oop_fwd_avx2_UG_UG nr_radix4
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix4_n1_oop_avx2.c"
#undef radix4_n1_oop_fwd_avx2_UG_UG
#define radix5_n1_oop_fwd_avx2_UG_UG nr_radix5
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix5_n1_oop_avx2.c"
#undef radix5_n1_oop_fwd_avx2_UG_UG
#define radix6_n1_oop_fwd_avx2_UG_UG nr_radix6
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix6_n1_oop_avx2.c"
#undef radix6_n1_oop_fwd_avx2_UG_UG
#define radix7_n1_oop_fwd_avx2_UG_UG nr_radix7
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix7_n1_oop_avx2.c"
#undef radix7_n1_oop_fwd_avx2_UG_UG
#define radix8_n1_oop_fwd_avx2_UG_UG nr_radix8
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix8_n1_oop_avx2.c"
#undef radix8_n1_oop_fwd_avx2_UG_UG
#define radix9_n1_oop_fwd_avx2_UG_UG nr_radix9
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix9_n1_oop_avx2.c"
#undef radix9_n1_oop_fwd_avx2_UG_UG
#define radix10_n1_oop_fwd_avx2_UG_UG nr_radix10
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix10_n1_oop_avx2.c"
#undef radix10_n1_oop_fwd_avx2_UG_UG
#define radix11_n1_oop_fwd_avx2_UG_UG nr_radix11
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix11_n1_oop_avx2.c"
#undef radix11_n1_oop_fwd_avx2_UG_UG
#define radix12_n1_oop_fwd_avx2_UG_UG nr_radix12
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix12_n1_oop_avx2.c"
#undef radix12_n1_oop_fwd_avx2_UG_UG
#define radix13_n1_oop_fwd_avx2_UG_UG nr_radix13
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix13_n1_oop_avx2.c"
#undef radix13_n1_oop_fwd_avx2_UG_UG
#define radix14_n1_oop_fwd_avx2_UG_UG nr_radix14
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix14_n1_oop_avx2.c"
#undef radix14_n1_oop_fwd_avx2_UG_UG
#define radix15_n1_oop_fwd_avx2_UG_UG nr_radix15
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix15_n1_oop_avx2.c"
#undef radix15_n1_oop_fwd_avx2_UG_UG
#define radix16_n1_oop_fwd_avx2_UG_UG nr_radix16
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix16_n1_oop_avx2.c"
#undef radix16_n1_oop_fwd_avx2_UG_UG
#define radix17_n1_oop_fwd_avx2_UG_UG nr_radix17
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix17_n1_oop_avx2.c"
#undef radix17_n1_oop_fwd_avx2_UG_UG
#define radix19_n1_oop_fwd_avx2_UG_UG nr_radix19
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix19_n1_oop_avx2.c"
#undef radix19_n1_oop_fwd_avx2_UG_UG
#define radix20_n1_oop_fwd_avx2_UG_UG nr_radix20
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix20_n1_oop_avx2.c"
#undef radix20_n1_oop_fwd_avx2_UG_UG
#define radix25_n1_oop_fwd_avx2_UG_UG nr_radix25
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix25_n1_oop_avx2.c"
#undef radix25_n1_oop_fwd_avx2_UG_UG
#define radix32_n1_oop_fwd_avx2_UG_UG nr_radix32
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix32_n1_oop_avx2.c"
#undef radix32_n1_oop_fwd_avx2_UG_UG
#define radix64_n1_oop_fwd_avx2_UG_UG nr_radix64
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix64_n1_oop_avx2.c"
#undef radix64_n1_oop_fwd_avx2_UG_UG
#define radix128_n1_oop_fwd_avx2_UG_UG nr_radix128
#include "../../src/dag-fft-compiler/codelets/oop/avx2/radix128_n1_oop_avx2.c"
#undef radix128_n1_oop_fwd_avx2_UG_UG
typedef struct { int N; vfft_oop11_fn f; } ent_t;
static const ent_t TAB[] = {
    {2, nr_radix2},
    {3, nr_radix3},
    {4, nr_radix4},
    {5, nr_radix5},
    {6, nr_radix6},
    {7, nr_radix7},
    {8, nr_radix8},
    {9, nr_radix9},
    {10, nr_radix10},
    {11, nr_radix11},
    {12, nr_radix12},
    {13, nr_radix13},
    {14, nr_radix14},
    {15, nr_radix15},
    {16, nr_radix16},
    {17, nr_radix17},
    {19, nr_radix19},
    {20, nr_radix20},
    {25, nr_radix25},
    {32, nr_radix32},
    {64, nr_radix64},
    {128, nr_radix128},
};
#define NTAB (sizeof TAB / sizeof TAB[0])

static double mdiff(const double*a,const double*b,size_t n){double m=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>m)m=d;}return m;}
static int fails=0;

static void cell(int N, vfft_oop11_fn f, size_t K, char *note)
{
    size_t n=(size_t)N*K;
    double *x[2],*r[2],*a[2];
    for(int p=0;p<2;p++){x[p]=_aligned_malloc(n*8,64);r[p]=_aligned_malloc(n*8,64);a[p]=_aligned_malloc(n*8,64);}
    srand(31*N+(int)K);
    for(size_t i=0;i<n;i++){x[0][i]=(double)rand()/RAND_MAX-0.5;x[1][i]=(double)rand()/RAND_MAX-0.5;}
    /* A: aliased vs separate, same function -> bit-exact */
    f(x[0],x[1],r[0],r[1],NULL,NULL,K,1,K,1,K);
    memcpy(a[0],x[0],n*8);memcpy(a[1],x[1],n*8);
    f(a[0],a[1],a[0],a[1],NULL,NULL,K,1,K,1,K);
    double eA = mdiff(a[0],r[0],n)>mdiff(a[1],r[1],n)?mdiff(a[0],r[0],n):mdiff(a[1],r[1],n);
    /* B: naive DFT lane 0 (relative) */
    double eB=0,scale=0;
    for(int k=0;k<N;k++){double sr=0,si=0;
        for(int m=0;m<N;m++){double ang=-2.0*M_PI*k*m/N,c=cos(ang),s=sin(ang);
            sr+=x[0][(size_t)m*K]*c-x[1][(size_t)m*K]*s; si+=x[0][(size_t)m*K]*s+x[1][(size_t)m*K]*c;}
        double d1=fabs(r[0][(size_t)k*K]-sr),d2=fabs(r[1][(size_t)k*K]-si);
        if(d1>eB)eB=d1;if(d2>eB)eB=d2;
        if(fabs(sr)>scale)scale=fabs(sr);if(fabs(si)>scale)scale=fabs(si);}
    if(scale>0)eB/=scale;
    /* C: aliased roundtrip via swap identity (bwd = fwd with re/im swapped, aliased) */
    memcpy(a[0],x[0],n*8);memcpy(a[1],x[1],n*8);
    f(a[0],a[1],a[0],a[1],NULL,NULL,K,1,K,1,K);          /* fwd aliased */
    f(a[1],a[0],a[1],a[0],NULL,NULL,K,1,K,1,K);          /* bwd = swap  */
    double eC=0,inv=1.0/N;
    for(size_t i=0;i<n;i++){double d1=fabs(a[0][i]*inv-x[0][i]),d2=fabs(a[1][i]*inv-x[1][i]);
        if(d1>eC)eC=d1;if(d2>eC)eC=d2;}
    int bad=(eA!=0.0)||(eB>1e-8)||(eC>1e-9);
    if(bad){fails++;sprintf(note+strlen(note)," K=%zu[A=%.1e B=%.1e C=%.1e]",K,eA,eB,eC);}
    for(int p=0;p<2;p++){_aligned_free(x[p]);_aligned_free(r[p]);_aligned_free(a[p]);}
}

int main(void)
{
    setvbuf(stdout,NULL,_IONBF,0);
    printf("# T3 gate: %zu no-restrict leaves x K={1,2,3,4,5,8,12,23,64}; A=aliased-bitexact B=DFT C=aliased-swap-roundtrip\n",NTAB);
    size_t Ks[]={1,2,3,4,5,8,12,23,64};
    int npass=0;
    for(size_t t=0;t<NTAB;t++){
        char note[512]="";
        for(int i=0;i<9;i++) cell(TAB[t].N,TAB[t].f,Ks[i],note);
        if(note[0]) printf("N=%-4d FAIL%s\n",TAB[t].N,note); else npass++;
    }
    printf("\n%d/%zu leaves pass all gates all K\n",npass,NTAB);
    printf(fails?"T3 FAIL\n":"T3 PASS: no-restrict build is alias-safe BY CONTRACT across every leaf, every K incl tails, fwd+bwd\n");
    return fails?1:0;
}
